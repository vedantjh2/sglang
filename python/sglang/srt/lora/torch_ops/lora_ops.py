from typing import Optional

import torch


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

    output = torch.empty(
        total_seq_len, weight_out_dim, dtype=inputs.dtype, device=inputs.device
    )

    # Pre-convert to Python types to avoid tensor iteration overhead
    w_indices = weight_indices.tolist()
    seg_lens = seg_len_tensor.tolist()
    ranks = lora_ranks.tolist()
    scales = scaling_tensor.tolist()

    token_offset = 0
    for i, seg_len in enumerate(seg_lens):
        if seg_len == 0:
            continue
        w_idx = w_indices[i]
        rank = ranks[w_idx]
        if rank > 0:
            ns_rank = num_slices * rank
            out_slice = output[token_offset : token_offset + seg_len, :ns_rank]
            torch.addmm(
                out_slice,
                inputs[token_offset : token_offset + seg_len],
                weights[w_idx, :ns_rank].T,
                beta=0,
                alpha=scales[w_idx],
                out=out_slice,
            )
            # Zero remaining columns if rank < max
            if ns_rank < weight_out_dim:
                output[
                    token_offset : token_offset + seg_len, ns_rank:
                ].zero_()
        else:
            output[token_offset : token_offset + seg_len].zero_()
        token_offset += seg_len

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

    # Pre-convert to Python types
    w_indices = weight_indices.tolist()
    seg_lens = seg_len_tensor.tolist()
    ranks = lora_ranks.tolist()
    slice_off = slice_offsets.tolist()

    token_offset = 0
    for i, seg_len in enumerate(seg_lens):
        if seg_len == 0:
            continue
        w_idx = w_indices[i]
        rank = ranks[w_idx]
        if rank == 0:
            token_offset += seg_len
            continue

        for slice_idx in range(num_slices):
            s_in = slice_idx * rank
            s_out = slice_off[slice_idx]
            e_out = slice_off[slice_idx + 1]

            out_slice = output[
                token_offset : token_offset + seg_len, s_out:e_out
            ]
            torch.addmm(
                out_slice,
                inputs[token_offset : token_offset + seg_len, s_in : s_in + rank],
                weights[w_idx, s_out:e_out, :rank].T,
                beta=1,
                alpha=1,
                out=out_slice,
            )

        token_offset += seg_len

    return output
