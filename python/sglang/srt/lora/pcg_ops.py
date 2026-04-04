"""Piecewise CUDA Graph (PCG) custom ops for LoRA operations.

These custom ops wrap LoRA backend calls so they are opaque to torch.compile
(via @register_custom_op). They are NOT split ops — they get captured INSIDE
the CUDA graph. The LoRA kernels use pre-allocated GPU tensors (adapter_mask,
scaling_gpu) that are updated via copy_() before each graph replay.
"""

import torch

from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.utils.custom_op import register_custom_op


def _get_lora_backend():
    ctx = get_forward_context()
    assert ctx is not None and ctx.lora_backend is not None
    return ctx.lora_backend


def _offsets_to_list(offset_tensor: torch.Tensor) -> list:
    """Convert offset tensor to Python list. Must be called before CUDA graph capture
    (e.g., cached during layer init). For tensors that are model constants, this is safe."""
    if offset_tensor.is_cuda:
        return offset_tensor.cpu().tolist()
    return offset_tensor.tolist()


# Cache for offset lists — avoids GPU sync during capture
_offset_cache: dict = {}


def _get_cached_offsets(offset_tensor: torch.Tensor) -> list:
    """Get or cache the Python list of offsets for a constant tensor."""
    tid = offset_tensor.data_ptr()
    if tid not in _offset_cache:
        _offset_cache[tid] = _offsets_to_list(offset_tensor)
    return _offset_cache[tid]


# ──────────────────────────────────────────────────────────────────────
# ColumnParallelLinearWithLoRA: combined A + B, mutates base_output
# ──────────────────────────────────────────────────────────────────────
@register_custom_op(mutates_args=["base_output"])
def lora_apply_column_pcg(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    base_output: torch.Tensor,
    output_offset: torch.Tensor,
) -> None:
    backend = _get_lora_backend()
    offsets = _get_cached_offsets(output_offset)
    lora_a_out = backend.run_lora_a_sgemm_pcg(x, lora_a)
    backend.run_lora_b_sgemm_pcg(
        lora_a_out, lora_b, slice_offsets_list=offsets, base_output=base_output,
    )


# ──────────────────────────────────────────────────────────────────────
# QKVParallelLinearWithLoRA: QKV combined, mutates base_output
# ──────────────────────────────────────────────────────────────────────
@register_custom_op(mutates_args=["base_output"])
def lora_apply_qkv_pcg(
    x: torch.Tensor,
    qkv_lora_a: torch.Tensor,
    qkv_lora_b: torch.Tensor,
    base_output: torch.Tensor,
    output_offset: torch.Tensor,
    output_offset_cpu: torch.Tensor,
    max_qkv_out_dim: int,
) -> None:
    backend = _get_lora_backend()
    offsets = _get_cached_offsets(output_offset)
    lora_a_out = backend.run_lora_a_sgemm_pcg(x, qkv_lora_a, num_slices=3)
    backend.run_lora_b_sgemm_pcg(
        lora_a_out, qkv_lora_b, slice_offsets_list=offsets, base_output=base_output,
    )


# ──────────────────────────────────────────────────────────────────────
# MergedColumnParallelLinearWithLoRA (gate_up): mutates base_output
# ──────────────────────────────────────────────────────────────────────
@register_custom_op(mutates_args=["base_output"])
def lora_apply_gate_up_pcg(
    x: torch.Tensor,
    gate_up_lora_a: torch.Tensor,
    gate_up_lora_b: torch.Tensor,
    base_output: torch.Tensor,
    output_offset: torch.Tensor,
) -> None:
    backend = _get_lora_backend()
    offsets = _get_cached_offsets(output_offset)
    lora_a_out = backend.run_lora_a_sgemm_pcg(x, gate_up_lora_a, num_slices=2)
    backend.run_lora_b_sgemm_pcg(
        lora_a_out, gate_up_lora_b, slice_offsets_list=offsets, base_output=base_output,
    )


# ──────────────────────────────────────────────────────────────────────
# RowParallelLinearWithLoRA (TP>1): LoRA A only (before all-reduce)
# ──────────────────────────────────────────────────────────────────────
def _fake_lora_shrink(x: torch.Tensor, lora_a: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], lora_a.shape[1]), dtype=x.dtype, device=x.device)


@register_custom_op(fake_impl=_fake_lora_shrink)
def lora_shrink_pcg(
    x: torch.Tensor,
    lora_a: torch.Tensor,
) -> torch.Tensor:
    backend = _get_lora_backend()
    return backend.run_lora_a_sgemm_pcg(x, lora_a)


# ──────────────────────────────────────────────────────────────────────
# RowParallelLinearWithLoRA (TP>1): LoRA B only (after all-reduce)
# ──────────────────────────────────────────────────────────────────────
@register_custom_op(mutates_args=["base_output"])
def lora_expand_pcg(
    lora_a_out: torch.Tensor,
    lora_b: torch.Tensor,
    base_output: torch.Tensor,
    output_offset: torch.Tensor,
) -> None:
    backend = _get_lora_backend()
    offsets = _get_cached_offsets(output_offset)
    backend.run_lora_b_sgemm_pcg(
        lora_a_out, lora_b, slice_offsets_list=offsets, base_output=base_output,
    )


# ──────────────────────────────────────────────────────────────────────
# RowParallelLinearWithLoRA (TP=1): combined A+B, mutates base_output
# ──────────────────────────────────────────────────────────────────────
@register_custom_op(mutates_args=["base_output"])
def lora_apply_row_pcg(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    base_output: torch.Tensor,
    output_offset: torch.Tensor,
) -> None:
    backend = _get_lora_backend()
    offsets = _get_cached_offsets(output_offset)
    lora_a_out = backend.run_lora_a_sgemm_pcg(x, lora_a)
    backend.run_lora_b_sgemm_pcg(
        lora_a_out, lora_b, slice_offsets_list=offsets, base_output=base_output,
    )


# ──────────────────────────────────────────────────────────────────────
# VocabParallelEmbeddingWithLoRA: embedding A + B, mutates base_output
# ──────────────────────────────────────────────────────────────────────
@register_custom_op(mutates_args=["base_output"])
def lora_apply_embedding_pcg(
    input_ids: torch.Tensor,
    embedding_A: torch.Tensor,
    embedding_B: torch.Tensor,
    base_output: torch.Tensor,
    output_offset: torch.Tensor,
    vocab_size: int,
) -> None:
    backend = _get_lora_backend()
    offsets = _get_cached_offsets(output_offset)
    lora_a_out = backend.run_lora_a_embedding(
        input_ids=input_ids, weights=embedding_A, vocab_size=vocab_size,
    )
    backend.run_lora_b_sgemm_pcg(
        lora_a_out, embedding_B, slice_offsets_list=offsets, base_output=base_output,
    )


# ──────────────────────────────────────────────────────────────────────
# ParallelLMHeadWithLoRA: A + B, mutates base_output
# ──────────────────────────────────────────────────────────────────────
@register_custom_op(mutates_args=["base_output"])
def lora_apply_lm_head_pcg(
    hidden_states: torch.Tensor,
    lm_head_A: torch.Tensor,
    lm_head_B: torch.Tensor,
    base_output: torch.Tensor,
    output_offset: torch.Tensor,
) -> None:
    backend = _get_lora_backend()
    offsets = _get_cached_offsets(output_offset)
    lora_a_out = backend.run_lora_a_sgemm_pcg(hidden_states, lm_head_A)
    backend.run_lora_b_sgemm_pcg(
        lora_a_out, lm_head_B, slice_offsets_list=offsets, base_output=base_output,
    )
