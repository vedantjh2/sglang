"""
Per-beam logits-mask hook for SGLang beam search (PR #15645).

Activated by setting env var SGL_BEAM_CONSTRAINT_INDEX_PATH to a PR #36
prefix-index .npz file BEFORE constructing the sgl.Engine. SGLang's worker
processes inherit env vars but not Python module state, so the worker lazily
loads the index on first call.

Hot path:
  step 0:  one tensor add (broadcast level-0 mask).
  step >=1: build flat (beam_idx, token_idx) pair tensors, one scatter_,
            one in-place add. Per-(a,b) child tensors are cached on GPU.

Init cost:
  - Level 0:    one query + tensor build (tiny).
  - Level 1:    one query per level-0 token (~7K Python calls, sub-second).
  - Level 2+:   lazy — populated on first sight of each (a, b) prefix.
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple

import torch


# Module-level state. Populated lazily via _maybe_load() on first get().
_PROC: Optional["PrefixMaskProcessor"] = None
_LOAD_ATTEMPTED = False


def get(rid: str) -> Optional["PrefixMaskProcessor"]:
    """Return the active processor (lazy-loaded from env vars)."""
    _maybe_load()
    return _PROC


def _maybe_load() -> None:
    global _PROC, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return
    _LOAD_ATTEMPTED = True
    path = os.environ.get("SGL_BEAM_CONSTRAINT_INDEX_PATH")
    if not path:
        return  # constraint disabled

    # Ensure prefix_index.py is on sys.path (PR #36's class lives there).
    import sys as _sys
    extra = os.environ.get("SGL_BEAM_CONSTRAINT_PYTHONPATH", "/home/jobuser")
    for p in extra.split(":"):
        if p and p not in _sys.path:
            _sys.path.insert(0, p)
    from prefix_index import PrefixIndex  # type: ignore

    tok = None
    tok_path = os.environ.get("SGL_BEAM_CONSTRAINT_TOK_PATH")
    if tok_path:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    idx = PrefixIndex(
        index_path=path,
        codebook_size=int(os.environ.get("SGL_BEAM_CONSTRAINT_CODEBOOK_SIZE", "8192")),
        num_codebook=int(os.environ.get("SGL_BEAM_CONSTRAINT_NUM_CODEBOOK", "3")),
        tokenizer=tok,
    )
    vocab_size = int(os.environ["SGL_BEAM_CONSTRAINT_VOCAB_SIZE"])
    _PROC = PrefixMaskProcessor(idx, vocab_size, torch.device("cuda"))


class PrefixMaskProcessor:
    def __init__(self, prefix_index, vocab_size: int, device: torch.device):
        self.prefix_index = prefix_index
        self.vocab_size = vocab_size
        self.device = device

        # Level 0: precomputed dense additive mask (broadcast to all beams).
        l0_valid = list(prefix_index.query())
        l0_t = torch.tensor(l0_valid, dtype=torch.long, device=device)
        self._level0_mask = torch.full(
            (vocab_size,), float("-inf"), dtype=torch.float32, device=device,
        )
        self._level0_mask[l0_t] = 0.0

        # Level 1 lookup: a -> Tensor[allowed_b]. Built eagerly because it's
        # cheap (one query per a, ~thousands of calls).
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
        """Apply per-beam mask to [n_beams, vocab] logprobs in place."""
        n_beams = logprobs.shape[0]
        if n_beams == 0:
            return
        step = len(beam_states[0])

        if step == 0:
            logprobs.add_(self._level0_mask.unsqueeze(0))
            return

        # Look up allowed children per beam.
        beam_idxs: List[torch.Tensor] = []
        token_idxs: List[torch.Tensor] = []
        for i, state in enumerate(beam_states):
            allowed = self._lookup(state)
            if allowed is None or allowed.numel() == 0:
                continue
            beam_idxs.append(torch.full((allowed.numel(),), i,
                                        dtype=torch.long, device=self.device))
            token_idxs.append(allowed)

        # Mask: -inf everywhere; write 0 at allowed positions.
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
        # Lazy: query the prefix index for this prefix, cache as GPU tensor.
        ch = self.prefix_index.query(*state)
        t = (torch.tensor(ch, dtype=torch.long, device=self.device)
             if ch else torch.empty(0, dtype=torch.long, device=self.device))
        self._cache[state] = t
        return t
