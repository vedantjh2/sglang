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
    seg_lens_list = seg_len_tensor.tolist()

    # Filter active (non-zero-length) segments
    active_indices = [i for i, sl in enumerate(seg_lens_list) if sl > 0]
    if not active_indices:
        return torch.zeros(
            total_seq_len, weight_out_dim, dtype=inputs.dtype, device=inputs.device
        )

    n_active = len(active_indices)
    active_lens = [seg_lens_list[i] for i in active_indices]
    active_w_indices = [int(weight_indices[i]) for i in active_indices]

    # Fast path: all active segments equal length → zero-copy view + bmm
    if len(set(active_lens)) == 1 and sum(active_lens) == total_seq_len:
        seg_len = active_lens[0]
        gathered_w = weights[active_w_indices]
        scales = scaling_tensor[active_w_indices].float().view(-1, 1, 1)
        gathered_w = gathered_w * scales.to(
            device=gathered_w.device, dtype=gathered_w.dtype
        )
        x3 = inputs.view(n_active, seg_len, input_dim)
        return torch.bmm(x3, gathered_w.transpose(-1, -2)).reshape(
            total_seq_len, weight_out_dim
        )

    # Fallback: per-segment addmm loop (zero-copy views, no padding overhead)
    output = torch.zeros(
        total_seq_len, weight_out_dim, dtype=inputs.dtype, device=inputs.device
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
    num_loras, weight_out_dim, max_rank = weights.shape
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

    seg_lens_list = seg_len_tensor.tolist()

    # Filter active segments
    active_indices = [i for i, sl in enumerate(seg_lens_list) if sl > 0]
    if not active_indices:
        return output

    n_active = len(active_indices)
    active_lens = [seg_lens_list[i] for i in active_indices]
    active_w_indices = [int(weight_indices[i]) for i in active_indices]

    # Fast path: all active segments equal length → zero-copy view + bmm per slice
    if len(set(active_lens)) == 1 and sum(active_lens) == total_seq_len:
        seg_len = active_lens[0]
        slice_offsets_list = slice_offsets.tolist()

        for slice_idx in range(num_slices):
            slice_start_in = slice_idx * max_rank
            slice_end_in = (slice_idx + 1) * max_rank
            slice_start_out = int(slice_offsets_list[slice_idx])
            slice_end_out = int(slice_offsets_list[slice_idx + 1])
            slice_dim = slice_end_out - slice_start_out

            x3 = inputs[:, slice_start_in:slice_end_in].view(
                n_active, seg_len, max_rank
            )
            gathered_w = weights[active_w_indices, slice_start_out:slice_end_out, :]
            bmm_result = torch.bmm(x3, gathered_w.transpose(-1, -2)).reshape(
                total_seq_len, slice_dim
            )
            output[:, slice_start_out:slice_end_out] += bmm_result

        return output

    # Fallback: per-segment × per-slice addmm loop
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
                token_offset : token_offset + seq_len,
                slice_start_input:slice_end_input,
            ]
            w_slice = weights[lora_idx, slice_start_output:slice_end_output, :rank]
            out_slice = output[
                token_offset : token_offset + seq_len,
                slice_start_output:slice_end_output,
            ]
            torch.addmm(
                out_slice, x_slice, w_slice.T, beta=1, alpha=1, out=out_slice
            )

        token_offset += seq_len

    return output
