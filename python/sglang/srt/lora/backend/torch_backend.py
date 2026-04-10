from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.torch_ops import sgemm_lora_a_fwd, sgemm_lora_b_fwd
from sglang.srt.lora.utils import LoRABatchInfo, generate_sequence_lengths
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class TorchNativeLoRABatchInfo(LoRABatchInfo):
    # ranks of each lora adapter, in shape (lora_num,) placed on cpu device
    lora_ranks_cpu: Optional[torch.Tensor] = None

    # Indice pointers of each segment in shape (num_segments + 1, ) placed on cpu device
    seg_indptr_cpu: Optional[torch.Tensor] = None

    # Lengths of each segments in shape (num_segments,) placed on cpu device
    seg_lens_cpu: Optional[torch.Tensor] = None

    # The index of lora adapter used by each segment, in shape (num_segments,) placed on cpu device
    weight_indices_cpu: Optional[torch.Tensor] = None

    # Scaling factors for each lora adapter, in shape (lora_num,) placed on cpu device
    scalings_cpu: Optional[torch.Tensor] = None


class TorchNativeLoRABackend(BaseLoRABackend):
    name = "torch_native"

    def __init__(
        self,
        max_loras_per_batch: int,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(max_loras_per_batch, device)
        # Pre-computed per-batch: adapter index per token and gather index cache
        self._adapter_idx: Optional[torch.Tensor] = None
        self._w_idx_list: Optional[list] = None
        self._gather_cache: dict = {}

    def _get_gather_idx(self, out_dim: int) -> torch.Tensor:
        """Get or compute cached gather index for a given output dimension."""
        if out_dim not in self._gather_cache:
            self._gather_cache[out_dim] = (
                self._adapter_idx.unsqueeze(1) * out_dim
                + torch.arange(out_dim, device=self.device)
            )
        return self._gather_cache[out_dim]

    def _expanded_mm_a(
        self, x: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """LoRA A via single expanded mm + cached gather."""
        _, weight_out_dim, input_dim = weights.shape
        gathered_w = weights[self._w_idx_list]
        scales = self.batch_info.scalings_cpu[self._w_idx_list].float().view(-1, 1, 1)
        gathered_w = gathered_w * scales.to(
            device=gathered_w.device, dtype=gathered_w.dtype
        )
        all_w = gathered_w.reshape(-1, input_dim)
        all_out = torch.mm(x, all_w.T)
        return all_out.gather(1, self._get_gather_idx(weight_out_dim))

    def _expanded_mm_b_slice(
        self,
        x_slice: torch.Tensor,
        weights: torch.Tensor,
        slice_start_out: int,
        slice_end_out: int,
    ) -> torch.Tensor:
        """LoRA B for one slice via single expanded mm + cached gather."""
        slice_dim = slice_end_out - slice_start_out
        gathered_w = weights[self._w_idx_list, slice_start_out:slice_end_out, :]
        all_w = gathered_w.reshape(-1, gathered_w.shape[-1])
        all_out = torch.mm(x_slice, all_w.T)
        return all_out.gather(1, self._get_gather_idx(slice_dim))

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        if self._adapter_idx is not None:
            return self._expanded_mm_a(x, weights)
        return sgemm_lora_a_fwd(
            inputs=x,
            weights=weights,
            weight_indices=self.batch_info.weight_indices_cpu,
            seg_len_tensor=self.batch_info.seg_lens_cpu,
            lora_ranks=self.batch_info.lora_ranks_cpu,
            scaling_tensor=self.batch_info.scalings_cpu,
            num_slices=1,
        )

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        base_output: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self._adapter_idx is not None:
            _, weight_out_dim, max_rank = weights.shape
            if base_output is not None:
                output = base_output
            else:
                output = torch.zeros(
                    x.shape[0], weight_out_dim, dtype=x.dtype, device=x.device
                )
            output += self._expanded_mm_b_slice(x, weights, 0, weight_out_dim)
            return output

        _, weight_out_dim, _ = weights.shape
        output_offset = torch.tensor(
            [0, weight_out_dim], dtype=torch.int32, device="cpu"
        )
        return sgemm_lora_b_fwd(
            inputs=x,
            weights=weights,
            weight_indices=self.batch_info.weight_indices_cpu,
            seg_len_tensor=self.batch_info.seg_lens_cpu,
            lora_ranks=self.batch_info.lora_ranks_cpu,
            slice_offsets=output_offset,
            base_output=base_output,
        )

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        output_offset_cpu: torch.Tensor,
        max_qkv_out_dim: int,
        base_output: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self._adapter_idx is not None:
            lora_a_output = self._expanded_mm_a(x, qkv_lora_a)

            _, _, max_rank = qkv_lora_b.shape
            slice_offsets = output_offset_cpu.tolist()
            total_out_dim = int(slice_offsets[-1])
            output = base_output if base_output is not None else torch.zeros(
                x.shape[0], total_out_dim, dtype=x.dtype, device=x.device
            )
            for slice_idx in range(3):
                s_in = slice_idx * max_rank
                e_in = (slice_idx + 1) * max_rank
                s_out = int(slice_offsets[slice_idx])
                e_out = int(slice_offsets[slice_idx + 1])
                output[:, s_out:e_out] += self._expanded_mm_b_slice(
                    lora_a_output[:, s_in:e_in], qkv_lora_b, s_out, e_out
                )
            return output

        lora_a_output = sgemm_lora_a_fwd(
            inputs=x,
            weights=qkv_lora_a,
            weight_indices=self.batch_info.weight_indices_cpu,
            seg_len_tensor=self.batch_info.seg_lens_cpu,
            lora_ranks=self.batch_info.lora_ranks_cpu,
            scaling_tensor=self.batch_info.scalings_cpu,
            num_slices=3,
        )
        return sgemm_lora_b_fwd(
            inputs=lora_a_output,
            weights=qkv_lora_b,
            weight_indices=self.batch_info.weight_indices_cpu,
            seg_len_tensor=self.batch_info.seg_lens_cpu,
            lora_ranks=self.batch_info.lora_ranks_cpu,
            slice_offsets=output_offset_cpu,
            base_output=base_output,
        )

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        base_output: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self._adapter_idx is not None:
            lora_a_output = self._expanded_mm_a(x, gate_up_lora_a)

            _, weight_out_dim_b, max_rank = gate_up_lora_b.shape
            slice_size = weight_out_dim_b // 2
            output = base_output if base_output is not None else torch.zeros(
                x.shape[0], weight_out_dim_b, dtype=x.dtype, device=x.device
            )
            for slice_idx in range(2):
                s_in = slice_idx * max_rank
                e_in = (slice_idx + 1) * max_rank
                s_out = slice_idx * slice_size
                e_out = (slice_idx + 1) * slice_size
                output[:, s_out:e_out] += self._expanded_mm_b_slice(
                    lora_a_output[:, s_in:e_in], gate_up_lora_b, s_out, e_out
                )
            return output

        num_slices = 2
        _, weight_out_dim, _ = gate_up_lora_b.shape
        slice_size = weight_out_dim // num_slices
        output_offset = torch.tensor(
            [0, slice_size, weight_out_dim], dtype=torch.int32, device="cpu"
        )
        lora_a_output = sgemm_lora_a_fwd(
            inputs=x,
            weights=gate_up_lora_a,
            weight_indices=self.batch_info.weight_indices_cpu,
            seg_len_tensor=self.batch_info.seg_lens_cpu,
            lora_ranks=self.batch_info.lora_ranks_cpu,
            scaling_tensor=self.batch_info.scalings_cpu,
            num_slices=num_slices,
        )
        return sgemm_lora_b_fwd(
            inputs=lora_a_output,
            weights=gate_up_lora_b,
            weight_indices=self.batch_info.weight_indices_cpu,
            seg_len_tensor=self.batch_info.seg_lens_cpu,
            lora_ranks=self.batch_info.lora_ranks_cpu,
            slice_offsets=output_offset,
            base_output=base_output,
        )

    def init_cuda_graph_batch_info(
        self,
        max_bs_in_cuda_graph: int,
        num_tokens_per_bs: int,
    ):
        with torch.device("cuda"):
            self.cuda_graph_batch_info = TorchNativeLoRABatchInfo(
                use_cuda_graph=True,
                bs=max_bs_in_cuda_graph,
                num_segments=self.max_loras_per_batch,
                seg_lens=torch.full(
                    (max_bs_in_cuda_graph,), num_tokens_per_bs, dtype=torch.int32
                ),
                seg_indptr=torch.zeros(max_bs_in_cuda_graph + 1, dtype=torch.int32),
                weight_indices=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32),
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32),
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float),
                permutation=None,
                max_len=num_tokens_per_bs,
            )

            # Initialize seg_indptr for CUDA graph as they remain constant
            # across batches.
            torch.cumsum(
                self.cuda_graph_batch_info.seg_lens[:max_bs_in_cuda_graph],
                dim=0,
                out=self.cuda_graph_batch_info.seg_indptr[1 : max_bs_in_cuda_graph + 1],
            )

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        use_cuda_graph: bool,
    ):
        original_seq_lens_cpu = generate_sequence_lengths(forward_batch, device="cpu")
        original_weight_indices_tensor = torch.tensor(
            weight_indices, dtype=torch.int32, device="cpu"
        )

        unique_weight_indices_tensor, inverse_weight_indices_tensor = (
            torch.unique_consecutive(
                original_weight_indices_tensor, return_inverse=True
            )
        )

        seg_lens_cpu = (
            torch.zeros_like(
                unique_weight_indices_tensor, dtype=torch.int32, device="cpu"
            )
            .scatter_add_(
                0,
                inverse_weight_indices_tensor,
                original_seq_lens_cpu,
            )
            .pin_memory()
        )

        seg_indptr_cpu = torch.zeros(
            (len(seg_lens_cpu) + 1,), dtype=torch.int32, pin_memory=True
        )
        seg_indptr_cpu[1:] = torch.cumsum(seg_lens_cpu, dim=0)

        # Use pinned memory to avoid synchronizations during host-to-device transfer
        weight_indices_tensor = unique_weight_indices_tensor.pin_memory()
        lora_ranks_tensor = torch.tensor(
            lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"
        )
        scalings_tensor = torch.tensor(
            scalings, dtype=torch.float, pin_memory=True, device="cpu"
        )

        bs = forward_batch.batch_size
        num_segments = len(weight_indices_tensor)

        if use_cuda_graph:
            assert (
                self.cuda_graph_batch_info is not None
            ), "CUDA Graph batch info is not initialized."
            batch_info = self.cuda_graph_batch_info
            batch_info.bs = forward_batch.batch_size
            batch_info.num_segments = num_segments
        else:
            max_len = max(seg_lens_cpu)

            batch_info = TorchNativeLoRABatchInfo(
                bs=forward_batch.batch_size,
                num_segments=num_segments,
                max_len=max_len,
                use_cuda_graph=False,
                seg_lens=torch.empty((bs,), dtype=torch.int32, device=self.device),
                seg_indptr=torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                ),
                weight_indices=torch.empty(
                    (bs,), dtype=torch.int32, device=self.device
                ),
                lora_ranks=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.int32, device=self.device
                ),
                scalings=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.float, device=self.device
                ),
                permutation=None,
            )

        # Copy to device asynchronously
        batch_info.lora_ranks[: self.max_loras_per_batch].copy_(
            lora_ranks_tensor, non_blocking=True
        )
        batch_info.scalings[: self.max_loras_per_batch].copy_(
            scalings_tensor, non_blocking=True
        )
        batch_info.weight_indices[:num_segments].copy_(
            weight_indices_tensor, non_blocking=True
        )
        batch_info.seg_indptr[: len(seg_indptr_cpu)].copy_(
            seg_indptr_cpu, non_blocking=True
        )
        batch_info.seg_lens[: len(seg_lens_cpu)].copy_(seg_lens_cpu, non_blocking=True)

        batch_info.lora_ranks_cpu = lora_ranks_tensor
        batch_info.seg_indptr_cpu = seg_indptr_cpu
        batch_info.seg_lens_cpu = seg_lens_cpu
        batch_info.weight_indices_cpu = weight_indices_tensor
        batch_info.scalings_cpu = scalings_tensor

        self.batch_info = batch_info

        # Pre-compute expanded mm infrastructure (once per batch, reused across all layers)
        if num_segments > 0 and not use_cuda_graph:
            seg_lens_gpu = batch_info.seg_lens[:num_segments]
            self._adapter_idx = torch.repeat_interleave(
                torch.arange(num_segments, device=self.device),
                seg_lens_gpu,
            )
            self._w_idx_list = weight_indices_tensor[:num_segments].tolist()
            self._gather_cache = {}
        else:
            self._adapter_idx = None
            self._w_idx_list = None
            self._gather_cache = {}
