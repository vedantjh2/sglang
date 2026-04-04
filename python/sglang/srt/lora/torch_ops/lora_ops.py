from typing import Optional

import torch


# ──────────────────────────────────────────────────────────────────────
# CUDA-graph-compatible LoRA kernels for PCG
#
# These replace the Python-loop-based kernels with fixed-iteration GPU ops.
# For each adapter slot (0..max_loras-1), we do a FULL matmul over ALL tokens,
# then mask out tokens that don't belong to that adapter. The loop count is
# always max_loras (constant), all indexing uses GPU tensors, and there are
# no .item() calls — making these safe for CUDA graph capture and replay.
# ──────────────────────────────────────────────────────────────────────


def sgemm_lora_a_fwd_pcg(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    adapter_mask: torch.Tensor,
    scaling_gpu: torch.Tensor,
    max_loras: int,
    num_slices: int = 1,
) -> torch.Tensor:
    """CUDA-graph-compatible LoRA A forward (shrink).

    Args:
        inputs: (total_tokens, input_dim)
        weights: (num_loras, num_slices * max_rank, input_dim)
        adapter_mask: (max_loras, total_tokens) float — 1.0 for tokens using adapter i, else 0.0
        scaling_gpu: (max_loras,) float — scaling factor per adapter
        max_loras: fixed iteration count
        num_slices: number of output slices (1 for column, 2 for gate_up, 3 for QKV)
    """
    total_tokens, input_dim = inputs.shape
    _, weight_out_dim, _ = weights.shape

    output = torch.zeros(
        total_tokens, weight_out_dim, dtype=inputs.dtype, device=inputs.device
    )

    for i in range(max_loras):
        # Full matmul: all tokens × adapter i's weight
        temp = torch.mm(inputs, weights[i].T)  # (total_tokens, weight_out_dim)
        # Mask: zero out tokens not using adapter i, and apply scaling
        # Slice adapter_mask to actual token count (mask is pre-allocated at max size)
        mask_i = adapter_mask[i, :total_tokens].unsqueeze(1)  # (total_tokens, 1)
        temp = temp * (mask_i * scaling_gpu[i])
        # Accumulate
        output = output + temp

    return output


def sgemm_lora_b_fwd_pcg(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    adapter_mask: torch.Tensor,
    slice_offsets: torch.Tensor,
    max_loras: int,
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CUDA-graph-compatible LoRA B forward (expand).

    Args:
        inputs: (total_tokens, num_slices * max_rank) — output from LoRA A
        weights: (num_loras, output_dim, max_rank)
        adapter_mask: (max_loras, total_tokens) float
        slice_offsets: (num_slices + 1,) int — cumulative output slice boundaries
        max_loras: fixed iteration count
        base_output: (total_tokens, total_output_dim) — mutated in-place if provided
    """
    total_tokens, _ = inputs.shape
    num_loras, weight_out_dim, max_rank = weights.shape
    num_slices = len(slice_offsets) - 1
    total_output_dim = slice_offsets[-1].item()

    if base_output is not None:
        output = base_output
    else:
        output = torch.zeros(
            total_tokens, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )

    for i in range(max_loras):
        mask_col = adapter_mask[i, :total_tokens].unsqueeze(1)  # (total_tokens, 1)
        for s in range(num_slices):
            s_start_in = s * max_rank
            s_end_in = (s + 1) * max_rank
            s_start_out = slice_offsets[s]
            s_end_out = slice_offsets[s + 1]

            x_slice = inputs[:, s_start_in:s_end_in]  # (total_tokens, max_rank)
            w_slice = weights[i, s_start_out:s_end_out, :]  # (slice_dim, max_rank)

            temp = torch.mm(x_slice, w_slice.T)  # (total_tokens, slice_dim)
            temp = temp * mask_col
            output[:, s_start_out:s_end_out] = (
                output[:, s_start_out:s_end_out] + temp
            )

    return output


# ──────────────────────────────────────────────────────────────────────
# Original (non-PCG) kernels below
# ──────────────────────────────────────────────────────────────────────


def sgemm_lora_a_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    scaling_tensor: torch.Tensor,
    num_slices: int = 1,
):
    total_seq_len, input_dim = inputs.shape
    if weights.numel() == 0:
        return torch.zeros(total_seq_len, 0, dtype=inputs.dtype, device=inputs.device)

    num_loras, weight_out_dim, _ = weights.shape
    max_rank = weight_out_dim // num_slices

    output = torch.zeros(
        total_seq_len, num_slices * max_rank, dtype=inputs.dtype, device=inputs.device
    )

    token_offset = 0
    for lora_idx, seq_len, rank in zip(
        weight_indices, seg_len_tensor, lora_ranks[weight_indices]
    ):
        if seq_len == 0:
            continue

        if rank > 0:
            x_seq = inputs[token_offset : token_offset + seq_len, :]
            w_seq = weights[lora_idx, : num_slices * rank, :]

            out_slice = output[
                token_offset : token_offset + seq_len, : num_slices * rank
            ]
            torch.addmm(
                out_slice,
                x_seq,
                w_seq.T,
                beta=0,
                alpha=scaling_tensor[lora_idx].item(),
                out=out_slice,
            )

        token_offset += seq_len

    return output


def sgemm_lora_b_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    slice_offsets: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
):
    total_seq_len, _ = inputs.shape
    num_loras, weight_out_dim, _ = weights.shape
    total_output_dim = slice_offsets[-1].item() if len(slice_offsets) > 0 else 0

    if weights.numel() == 0:
        return torch.zeros(
            total_seq_len, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )

    num_slices = len(slice_offsets) - 1

    if base_output is not None:
        output = base_output
    else:
        output = torch.zeros(
            total_seq_len, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )

    token_offset = 0
    for lora_idx, seq_len, rank in zip(
        weight_indices, seg_len_tensor, lora_ranks[weight_indices]
    ):
        if seq_len == 0:
            continue

        if rank == 0:
            token_offset += seq_len
            continue

        for slice_idx in range(num_slices):
            slice_start_input = slice_idx * rank
            slice_end_input = (slice_idx + 1) * rank

            slice_start_output = slice_offsets[slice_idx]
            slice_end_output = slice_offsets[slice_idx + 1]

            x_slice = inputs[
                token_offset : token_offset + seq_len :,
                slice_start_input:slice_end_input,
            ]  # (seq_len, rank)
            w_slice = weights[
                lora_idx, slice_start_output:slice_end_output, :rank
            ]  # (slice_dim, rank)

            out_slice = output[
                token_offset : token_offset + seq_len,
                slice_start_output:slice_end_output,
            ]
            torch.addmm(out_slice, x_slice, w_slice.T, beta=1, alpha=1, out=out_slice)

        token_offset += seq_len

    return output
