"""Piecewise CUDA Graph (PCG) custom ops for LoRA operations.

These custom ops wrap LoRA backend calls so they are opaque to torch.compile
(via @register_custom_op). They are NOT split ops — they get captured INSIDE
the CUDA graph. The backend's batch_info is pre-allocated at fixed GPU addresses
and updated via copy_() before each graph replay.
"""

import torch

from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.utils.custom_op import register_custom_op


def _get_lora_backend():
    ctx = get_forward_context()
    assert ctx is not None and ctx.lora_backend is not None
    return ctx.lora_backend


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
    lora_a_out = backend.run_lora_a_sgemm(x, lora_a)
    backend.run_lora_b_sgemm(
        lora_a_out, lora_b, output_offset=output_offset, base_output=base_output,
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
    backend.run_qkv_lora(
        x=x, qkv_lora_a=qkv_lora_a, qkv_lora_b=qkv_lora_b,
        base_output=base_output, output_offset=output_offset,
        output_offset_cpu=output_offset_cpu, max_qkv_out_dim=max_qkv_out_dim,
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
    backend.run_gate_up_lora(
        x=x, gate_up_lora_a=gate_up_lora_a, gate_up_lora_b=gate_up_lora_b,
        base_output=base_output, output_offset=output_offset,
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
    return backend.run_lora_a_sgemm(x, lora_a)


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
    backend.run_lora_b_sgemm(
        lora_a_out, lora_b, output_offset=output_offset, base_output=base_output,
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
    lora_a_out = backend.run_lora_a_sgemm(x, lora_a)
    backend.run_lora_b_sgemm(
        lora_a_out, lora_b, output_offset=output_offset, base_output=base_output,
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
    lora_a_out = backend.run_lora_a_embedding(
        input_ids=input_ids, weights=embedding_A, vocab_size=vocab_size,
    )
    backend.run_lora_b_sgemm(
        lora_a_out, embedding_B, output_offset=output_offset, base_output=base_output,
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
    lora_a_out = backend.run_lora_a_sgemm(hidden_states, lm_head_A)
    backend.run_lora_b_sgemm(
        lora_a_out, lm_head_B, output_offset=output_offset, base_output=base_output,
    )
