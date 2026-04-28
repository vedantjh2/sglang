"""
Application-side custom logit processor for prefix-trie-constrained beam
search on top of SGLang.

This is application code — not part of sglang itself. It lives in
custom-scripts/ because it depends on a project-specific prefix-trie data
structure (PR #36's PrefixIndex). The upstream-shippable change is just the
generic per-beam CustomLogitProcessor dispatch in
scheduler_beam_search_processor_mixin.py; that hook works with any user-
written CustomLogitProcessor subclass.

Usage:
    from prefix_index_processor import PrefixIndexCustomLogitProcessor

    out = engine.generate(
        prompt,
        sampling_params={
            "n": 500,
            "max_new_tokens": 3,
            "custom_params": {
                # Module path that provides a `PrefixIndex` factory class with
                # a query(*tokens) -> list[int] method (PR #36 style).
                "prefix_index_module": "prefix_index",
                "prefix_index_class": "PrefixIndex",

                # Kwargs forwarded verbatim to the factory class. The
                # processor does not introspect them.
                "prefix_index_kwargs": {
                    "index_path": "/abs/path/to/prefix_index-v2.npz",
                    "codebook_size": 8192,
                    "num_codebook": 3,
                },

                # Required for the GPU mask buffer.
                "vocab_size": 176245,

                # Optional: extra sys.path entries needed in the worker
                # process to import prefix_index_module. Defaults to none.
                "module_search_paths": ["/path/to/my/app/code"],
            },
        },
        custom_logit_processor=PrefixIndexCustomLogitProcessor.to_str(),
    )

The processor is dill-serialized via .to_str() and shipped to worker
processes through SGLang's existing custom_logit_processor request field.
Each worker lazily builds the PrefixIndex (and its companion GPU lookup
tables) on first call and caches them keyed by the kwargs tuple, so an
index file is loaded at most once per worker even with many concurrent
requests.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


# ---------------------------------------------------------------------------
# GPU helper: per-beam scatter mask
# ---------------------------------------------------------------------------


class PrefixMaskProcessor:
    """
    Per-beam logits mask applier for a 3-level prefix trie.

    Holds GPU-resident lookup tensors keyed by parent-token state. Hot path
    is a single ``torch.scatter_`` per step — no Python loops over vocab and
    no per-call tensor reallocations.

    Generic over the trie source: takes any object with a ``query(*tokens)``
    method that returns a sequence of allowed-next-token IDs (the PR #36
    PrefixIndex API).
    """

    def __init__(
        self,
        prefix_index,
        vocab_size: int,
        device: torch.device,
    ):
        self.prefix_index = prefix_index
        self.vocab_size = vocab_size
        self.device = device

        # Level 0: dense mask, broadcast to all beams at the prompt boundary.
        l0_valid = list(prefix_index.query())
        if not l0_valid:
            raise ValueError("Prefix index has no level-0 entries")
        self._level0_mask = torch.full(
            (vocab_size,), float("-inf"),
            dtype=torch.float32, device=device,
        )
        self._level0_mask[
            torch.tensor(l0_valid, dtype=torch.long, device=device)
        ] = 0.0

        # Level 1: per-parent-token tensor cache (eager).
        self._l1: Dict[int, torch.Tensor] = {}
        for a in l0_valid:
            children = prefix_index.query(a)
            if len(children) > 0:
                self._l1[a] = torch.tensor(
                    list(children), dtype=torch.long, device=device,
                )

        # Levels 2+: lazy cache.
        self._cache: Dict[Tuple[int, ...], torch.Tensor] = {}

    @torch.no_grad()
    def apply(
        self,
        beam_states: List[Tuple[int, ...]],
        logprobs: torch.Tensor,
    ) -> None:
        """In-place mask application on logprobs of shape [n_beams, vocab]."""
        n_beams = logprobs.shape[0]
        if n_beams == 0:
            return
        step = len(beam_states[0])

        if step == 0:
            logprobs.add_(self._level0_mask.unsqueeze(0))
            return

        beam_idxs: List[torch.Tensor] = []
        token_idxs: List[torch.Tensor] = []
        for i, state in enumerate(beam_states):
            allowed = self._lookup(state)
            if allowed is None or allowed.numel() == 0:
                continue
            beam_idxs.append(torch.full(
                (allowed.numel(),), i, dtype=torch.long, device=self.device,
            ))
            token_idxs.append(allowed)

        mask = torch.full_like(logprobs, float("-inf"))
        if beam_idxs:
            mask[torch.cat(beam_idxs), torch.cat(token_idxs)] = 0.0
        logprobs.add_(mask)

    def _lookup(self, state: Tuple[int, ...]) -> Optional[torch.Tensor]:
        if len(state) == 1:
            return self._l1.get(state[0])
        cached = self._cache.get(state)
        if cached is not None:
            return cached
        children = self.prefix_index.query(*state)
        if len(children) > 0:
            tensor = torch.tensor(
                list(children), dtype=torch.long, device=self.device,
            )
        else:
            tensor = torch.empty(0, dtype=torch.long, device=self.device)
        self._cache[state] = tensor
        return tensor


# ---------------------------------------------------------------------------
# CustomLogitProcessor subclass: framework-integrated dispatch
# ---------------------------------------------------------------------------


class PrefixIndexCustomLogitProcessor(CustomLogitProcessor):
    """
    Framework-integrated processor: gets dill-serialized via ``to_str()`` and
    shipped to worker processes through ``SamplingParams.custom_logit_processor``.

    Per-worker state lives in a class-level cache keyed by the prefix index
    kwargs tuple, so a given index file is loaded at most once per worker
    even when many concurrent requests reference it.

    Required ``custom_params`` keys (per request):
        prefix_index_module: str   — import path of the module providing the
                                     PrefixIndex factory class.
        prefix_index_class:  str   — class name within that module. Must accept
                                     **kwargs and provide a query(*tokens)
                                     method returning allowed token IDs.
        prefix_index_kwargs: dict  — forwarded verbatim to the factory. Values
                                     must be dill-friendly primitives (paths,
                                     ints, strings); not Tensor / model objects.
        vocab_size:          int   — model vocabulary size for the GPU mask.

    Optional:
        tokenizer_path: str  — if given, the worker constructs an
                                AutoTokenizer from this path and passes it as
                                a ``tokenizer`` kwarg to the factory. Use this
                                when the prefix-index implementation needs the
                                tokenizer to resolve special-token offsets
                                (e.g. PR #36's PrefixIndex sets
                                ``sid_token_offset`` from
                                ``tokenizer.encode("<a_0>")[0]``).
        module_search_paths: list[str] — extra sys.path entries to ensure the
                                          factory module is importable in the
                                          worker process.
    """

    # Per-worker cache. Reconstructed fresh on each side of the spawn boundary
    # because dill-serialized classes lose attribute state in transit.
    _cache: Dict[Tuple, PrefixMaskProcessor] = {}

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            req = params.get("__req__")
            proc = self._get_or_load(params, logits.device)
            if proc is None:
                continue
            if req is not None and getattr(req, "is_beam_search", False):
                # Per-beam dispatch: read the request's alive beams' token
                # histories and apply the mask in-place on the logits slice
                # the scheduler hook gave us (already sized to this request's
                # n_beams).
                beams = req.beam_list.incomplete
                if not beams:
                    continue
                beam_states = [tuple(b.tokens) for b in beams]
                proc.apply(beam_states, logits)
            else:
                # Non-beam-search call site (the standard sampler): apply the
                # level-0 mask to this row. Without per-beam state we cannot
                # apply deeper-level constraints.
                logits[batch_idx].add_(proc._level0_mask)
        return logits

    @classmethod
    def _get_or_load(
        cls,
        params: Dict[str, Any],
        device: torch.device,
    ) -> Optional[PrefixMaskProcessor]:
        kwargs = params.get("prefix_index_kwargs")
        module_name = params.get("prefix_index_module")
        class_name = params.get("prefix_index_class")
        vocab_size = params.get("vocab_size")
        tokenizer_path = params.get("tokenizer_path")
        if not (kwargs and module_name and class_name and vocab_size):
            return None

        # Cache key uses only dill-friendly primitives (the tokenizer object
        # itself is built lazily and isn't part of the key).
        cache_key = (module_name, class_name,
                     tuple(sorted(kwargs.items())),
                     int(vocab_size), tokenizer_path)
        cached = cls._cache.get(cache_key)
        if cached is not None:
            return cached

        # Optional: extend sys.path if the user's app module isn't otherwise
        # on the worker's import path.
        import sys as _sys
        for p in params.get("module_search_paths") or []:
            if p and p not in _sys.path:
                _sys.path.insert(0, p)

        # Optional: build the tokenizer in the worker and forward it to the
        # factory. Tokenizer instances are not dill-friendly so we cannot
        # accept them through prefix_index_kwargs.
        factory_kwargs = dict(kwargs)
        if tokenizer_path:
            from transformers import AutoTokenizer
            factory_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True,
            )

        import importlib
        module = importlib.import_module(module_name)
        factory = getattr(module, class_name)
        prefix_index = factory(**factory_kwargs)

        proc = PrefixMaskProcessor(
            prefix_index, int(vocab_size), device,
        )
        cls._cache[cache_key] = proc
        return proc
