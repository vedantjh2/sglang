"""
Constrained beam search via the standard SGLang CustomLogitProcessor framework.

Usage (single tenant or multi-tenant):

    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.managers.beam_constraint import PrefixIndexCustomLogitProcessor

    sp = SamplingParams(
        n=500, max_new_tokens=3,
        custom_logit_processor=PrefixIndexCustomLogitProcessor.to_str(),
        custom_params={
            "prefix_index_path": "/.../prefix_index-v2.npz",
            "tokenizer_path": "/.../sft_checkpoint",
            "codebook_size": 8192,
            "num_codebook": 3,
            "vocab_size": 176245,           # required: model vocab
        },
    )

The processor class is dill-serialized and shipped to worker processes via the
existing CustomLogitProcessor machinery. Each worker lazily loads the prefix
index on first call and caches a GPU PrefixMaskProcessor keyed by index path,
so concurrent requests sharing an index pay the load cost once per worker.

Per-beam state (the tokens generated so far for each beam) is read directly
from `req.beam_list.incomplete` via the `__req__` reference passed in
custom_param_list, matching the existing pattern used by
ThinkingBudgetLogitProcessor.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class PrefixMaskProcessor:
    """GPU-resident per-beam mask applier.

    Holds the prefix-trie lookup tables. Hot path is one scatter_ per step.

    This class is NOT the CustomLogitProcessor — it is the underlying GPU
    helper. The user-facing class is PrefixIndexCustomLogitProcessor below,
    which dispatches to one of these via a per-worker cache.
    """

    def __init__(self, prefix_index, vocab_size: int, device: torch.device):
        self.prefix_index = prefix_index
        self.vocab_size = vocab_size
        self.device = device

        # Level 0: dense mask, broadcast to all beams.
        l0_valid = list(prefix_index.query())
        self._level0_mask = torch.full(
            (vocab_size,), float("-inf"), dtype=torch.float32, device=device,
        )
        self._level0_mask[
            torch.tensor(l0_valid, dtype=torch.long, device=device)
        ] = 0.0

        # Level 1 lookup: a -> Tensor[allowed_b]. Eager build (~7K queries,
        # sub-second).
        self._l1: Dict[int, torch.Tensor] = {}
        for a in l0_valid:
            ch = prefix_index.query(a)
            if ch:
                self._l1[a] = torch.tensor(ch, dtype=torch.long, device=device)

        # Level 2+: lazy cache.
        self._cache: Dict[Tuple[int, ...], torch.Tensor] = {}

    @torch.no_grad()
    def apply(self, beam_states: List[Tuple[int, ...]],
              logprobs: torch.Tensor) -> None:
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
            beam_idxs.append(torch.full((allowed.numel(),), i,
                                        dtype=torch.long, device=self.device))
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
        ch = self.prefix_index.query(*state)
        t = (torch.tensor(ch, dtype=torch.long, device=self.device)
             if ch else torch.empty(0, dtype=torch.long, device=self.device))
        self._cache[state] = t
        return t


class PrefixIndexCustomLogitProcessor(CustomLogitProcessor):
    """
    Framework-integrated constrained-beam-search processor.

    Subclasses CustomLogitProcessor — gets dill-serialized via .to_str() and
    shipped to worker processes through SamplingParams.custom_logit_processor.
    The class itself is stateless; per-worker state lives in a class-level
    cache keyed by the index path so an index file is loaded at most once per
    worker, even if many requests reference it.

    Expected per-request custom_params (passed via SamplingParams.custom_params):
        prefix_index_path: str    — required, path to .npz prefix index
        vocab_size:        int    — required, model vocabulary size
        tokenizer_path:    str    — optional, only needed if PrefixIndex requires
                                    a tokenizer for sid_token_offset lookup
        codebook_size:     int    — default 8192
        num_codebook:      int    — default 3
        prefix_index_pythonpath: str — optional, dir to add to sys.path for
                                       the PrefixIndex import (default
                                       "/home/jobuser")
    """

    # Class-level per-worker cache (path -> PrefixMaskProcessor).
    # This survives across requests in the same worker process, but each
    # worker loads independently because dill-serialized classes are
    # reconstructed fresh on each side of the spawn boundary.
    _cache: Dict[str, PrefixMaskProcessor] = {}

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Apply per-beam mask to a per-request slice of logits.

        For beam search, this is invoked separately per request from the
        beam search scheduler hook. logits has shape [n_beams_in_req, vocab],
        and custom_param_list has length 1 (containing the request's params
        plus a __req__ reference so we can pull beam states).

        For non-beam-search use cases (where the framework calls this from
        the standard sampler path), logits is [n_reqs, vocab] and
        custom_param_list has one entry per request — but in that case there
        is no per-beam state and this constraint isn't really applicable.
        We fall back to applying level 0 to all rows.
        """
        if not custom_param_list:
            return logits

        # The beam-search hook always passes one (req, params) pair per call
        # because the iteration over multiple requests in a batch happens at
        # the scheduler level (each req can have a different prefix index).
        for batch_idx, params in enumerate(custom_param_list):
            req = params.get("__req__")
            proc = self._get_or_load(params, logits.device, logits.shape[-1])
            if proc is None:
                continue
            if req is not None and getattr(req, "is_beam_search", False):
                # Beam-search dispatch: iterate this request's alive beams.
                beams = req.beam_list.incomplete
                if not beams:
                    continue
                beam_states = [tuple(b.tokens) for b in beams]
                # Slice of logits for this request's beams. The scheduler hook
                # is expected to call us with logits already sliced to one
                # request — see _apply_beam_search_logit_processor.
                proc.apply(beam_states, logits)
            else:
                # Non-beam-search: fall back to broadcasting level 0 only.
                # (This constraint is fundamentally per-beam-state, so the
                # standard sampler path has no per-beam history. Apply the
                # level-0 mask which is the strictest valid restriction
                # given no history.)
                logits[batch_idx].add_(proc._level0_mask)
        return logits

    @classmethod
    def _get_or_load(
        cls, params: Dict[str, Any], device: torch.device, vocab_size: int,
    ) -> Optional[PrefixMaskProcessor]:
        path = params.get("prefix_index_path")
        if not path:
            return None
        cached = cls._cache.get(path)
        if cached is not None:
            return cached

        import sys as _sys
        extra = params.get("prefix_index_pythonpath", "/home/jobuser")
        if extra and extra not in _sys.path:
            _sys.path.insert(0, extra)
        from prefix_index import PrefixIndex  # type: ignore

        tokenizer = None
        tok_path = params.get("tokenizer_path")
        if tok_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tok_path, trust_remote_code=True
            )
        idx = PrefixIndex(
            index_path=path,
            codebook_size=params.get("codebook_size", 8192),
            num_codebook=params.get("num_codebook", 3),
            tokenizer=tokenizer,
        )
        cls._cache[path] = PrefixMaskProcessor(idx, vocab_size, device)
        return cls._cache[path]
