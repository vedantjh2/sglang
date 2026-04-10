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

    # Build expanded weight matrix: fuse all adapters into one large matmul
    # weights[active]: (n_active, weight_out_dim, input_dim)
    gathered_w = weights[active_w_indices]
    # Apply per-adapter scaling into weights
    scales = scaling_tensor[active_w_indices].float().view(-1, 1, 1)
    gathered_w = gathered_w * scales.to(
        device=gathered_w.device, dtype=gathered_w.dtype
    )
    # Reshape to (n_active * weight_out_dim, input_dim) for single mm
    all_w = gathered_w.reshape(-1, input_dim)

    # Single large mm: (total_tokens, input_dim) @ (input_dim, n_active * weight_out_dim)
    all_outputs = torch.mm(inputs, all_w.T)
    # Shape: (total_tokens, n_active * weight_out_dim)

    # Build gather index: for each token, select the correct adapter's columns
    # Each token in segment i needs columns [i*weight_out_dim : (i+1)*weight_out_dim]
    seg_lens_active_t = torch.tensor(active_lens, device=inputs.device)
    col_offsets = torch.repeat_interleave(
        torch.arange(n_active, device=inputs.device) * weight_out_dim,
        seg_lens_active_t,
    )
    gather_idx = col_offsets.unsqueeze(1) + torch.arange(
        weight_out_dim, device=inputs.device
    )
    # Shape: (total_active_tokens, weight_out_dim)

    # If there are inactive segments, we need to map active tokens to output positions
    if len(active_indices) == len(seg_lens_list) and sum(active_lens) == total_seq_len:
        # All segments active, no gaps — gather directly
        return all_outputs.gather(1, gather_idx)
    else:
        # Some segments inactive — need to place results at correct positions
        output = torch.zeros(
            total_seq_len, weight_out_dim, dtype=inputs.dtype, device=inputs.device
        )
        # Compute active token positions in the full output
        active_token_offset = 0
        full_token_offset = 0
        for i, seg_len in enumerate(seg_lens_list):
            if seg_len > 0 and i in active_indices:
                gathered_result = all_outputs[
                    active_token_offset : active_token_offset + seg_len
                ].gather(
                    1,
                    gather_idx[active_token_offset : active_token_offset + seg_len],
                )
                output[full_token_offset : full_token_offset + seg_len] = (
                    gathered_result
                )
                active_token_offset += seg_len
            full_token_offset += seg_len
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
    all_active = (
        len(active_indices) == len(seg_lens_list)
        and sum(active_lens) == total_seq_len
    )

    slice_offsets_list = slice_offsets.tolist()
    seg_lens_active_t = torch.tensor(active_lens, device=inputs.device)

    for slice_idx in range(num_slices):
        slice_start_in = slice_idx * max_rank
        slice_end_in = (slice_idx + 1) * max_rank
        slice_start_out = int(slice_offsets_list[slice_idx])
        slice_end_out = int(slice_offsets_list[slice_idx + 1])
        slice_dim = slice_end_out - slice_start_out

        # Input for this slice: (total_tokens, max_rank)
        input_slice = inputs[:, slice_start_in:slice_end_in]

        # Build expanded weight for this slice: all adapters concatenated
        # weights[active, slice_start_out:slice_end_out, :] → (n_active, slice_dim, max_rank)
        gathered_w = weights[active_w_indices, slice_start_out:slice_end_out, :]
        all_w = gathered_w.reshape(-1, max_rank)  # (n_active * slice_dim, max_rank)

        # Single mm: (total_tokens, max_rank) @ (max_rank, n_active * slice_dim)
        all_outputs = torch.mm(input_slice, all_w.T)
        # Shape: (total_tokens, n_active * slice_dim)

        # Gather: select correct adapter's columns per token
        col_offsets = torch.repeat_interleave(
            torch.arange(n_active, device=inputs.device) * slice_dim,
            seg_lens_active_t,
        )
        gather_idx = col_offsets.unsqueeze(1) + torch.arange(
            slice_dim, device=inputs.device
        )

        if all_active:
            output[:, slice_start_out:slice_end_out] += all_outputs.gather(
                1, gather_idx
            )
        else:
            active_token_offset = 0
            full_token_offset = 0
            for i, seg_len in enumerate(seg_lens_list):
                if seg_len > 0 and i in active_indices:
                    gathered = all_outputs[
                        active_token_offset : active_token_offset + seg_len
                    ].gather(
                        1,
                        gather_idx[
                            active_token_offset : active_token_offset + seg_len
                        ],
                    )
                    output[
                        full_token_offset : full_token_offset + seg_len,
                        slice_start_out:slice_end_out,
                    ] += gathered
                    active_token_offset += seg_len
                full_token_offset += seg_len

    return output
