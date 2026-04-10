from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence


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

    # Split input into segments, keep only active ones, pad to uniform length
    all_segments = torch.split(inputs, seg_lens_list)
    active_segments = [all_segments[i] for i in active_indices]
    padded_x = pad_sequence(active_segments, batch_first=True)
    # Shape: (n_active, max_seg_len, input_dim)

    # Gather weights per segment and fuse scaling into weights
    active_w_indices = [weight_indices[i] for i in active_indices]
    gathered_w = weights[active_w_indices]  # (n_active, weight_out_dim, input_dim)
    scales = scaling_tensor[active_w_indices].float().view(-1, 1, 1)
    gathered_w = gathered_w * scales.to(gathered_w.dtype)

    # Single BMM: (n_active, max_seg_len, input_dim) @ (n_active, input_dim, weight_out_dim)
    padded_output = torch.bmm(padded_x, gathered_w.transpose(-1, -2))
    # Shape: (n_active, max_seg_len, weight_out_dim)

    # Unpad back to flat output
    output = torch.zeros(
        total_seq_len, weight_out_dim, dtype=inputs.dtype, device=inputs.device
    )
    token_offset = 0
    active_idx = 0
    for seg_len in seg_lens_list:
        if seg_len > 0:
            output[token_offset : token_offset + seg_len] = padded_output[
                active_idx, :seg_len
            ]
            active_idx += 1
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
    active_w_indices = [weight_indices[i] for i in active_indices]

    # Convert slice_offsets to list for indexing
    slice_offsets_list = slice_offsets.tolist()

    # Process each slice with one bmm
    for slice_idx in range(num_slices):
        slice_start_input = slice_idx * max_rank
        slice_end_input = (slice_idx + 1) * max_rank
        slice_start_out = int(slice_offsets_list[slice_idx])
        slice_end_out = int(slice_offsets_list[slice_idx + 1])

        # Split the input slice column range into segments, pad
        input_slice = inputs[:, slice_start_input:slice_end_input]
        all_segments = torch.split(input_slice, seg_lens_list)
        active_segments = [all_segments[i] for i in active_indices]
        padded_x = pad_sequence(active_segments, batch_first=True)
        # Shape: (n_active, max_seg_len, max_rank)

        # Gather B weights for this output slice
        gathered_w = weights[active_w_indices, slice_start_out:slice_end_out, :]
        # Shape: (n_active, slice_dim, max_rank)

        # BMM: (n_active, max_seg_len, max_rank) @ (n_active, max_rank, slice_dim)
        padded_result = torch.bmm(padded_x, gathered_w.transpose(-1, -2))
        # Shape: (n_active, max_seg_len, slice_dim)

        # Accumulate into output
        token_offset = 0
        active_idx = 0
        for seg_len in seg_lens_list:
            if seg_len > 0:
                output[
                    token_offset : token_offset + seg_len,
                    slice_start_out:slice_end_out,
                ] += padded_result[active_idx, :seg_len]
                active_idx += 1
            token_offset += seg_len

    return output
