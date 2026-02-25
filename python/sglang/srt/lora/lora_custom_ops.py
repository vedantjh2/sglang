# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Custom ops for LoRA triton kernels to enable piecewise CUDA graph (PCG).

These custom ops wrap the triton kernel launchers so that torch.compile/dynamo
can treat them as opaque operations (similar to how unified_attention_with_output
works for the attention backend). The batch_info needed by the triton kernels
is retrieved from the forward context at runtime.

During CUDA graph capture (`get_is_capture_mode() == True`), all ops return
immediately (no-op) because the captured graphs are for the base model only.
When a batch contains LoRA requests during replay, the `skip_cuda_graphs` flag
is set on the ForwardContext, causing `CUDAPiecewiseBackend` to run compiled
code directly (bypassing graph replay) so these custom ops execute normally.
"""

import torch

from sglang.srt.compilation.piecewise_context_manager import (
    get_forward_context,
    get_is_capture_mode,
)
from sglang.srt.lora.triton_ops import (
    gate_up_lora_b_fwd,
    qkv_lora_b_fwd,
    sgemm_lora_a_fwd,
    sgemm_lora_b_fwd,
)
from sglang.srt.utils import direct_register_custom_op


# ---------------------------------------------------------------------------
# 1. lora_a_sgemm: wraps sgemm_lora_a_fwd
#    Writes result into pre-allocated output tensor.
# ---------------------------------------------------------------------------
def lora_a_sgemm(
    x: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    stack_num: int,
) -> None:
    # During CUDA graph capture, skip triton kernels entirely.
    # The output tensor stays as-is (zeros from caller), which is correct
    # since captured graphs are for the base model without LoRA.
    if get_is_capture_mode():
        return
    context = get_forward_context()
    batch_info = context.lora_backend.batch_info
    ret = sgemm_lora_a_fwd(x, weights, batch_info, stack_num=stack_num)
    output.copy_(ret)


def lora_a_sgemm_fake(
    x: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    stack_num: int,
) -> None:
    return


direct_register_custom_op(
    op_name="lora_a_sgemm",
    op_func=lora_a_sgemm,
    mutates_args=["output"],
    fake_impl=lora_a_sgemm_fake,
)


# ---------------------------------------------------------------------------
# 2. lora_b_sgemm: wraps sgemm_lora_b_fwd
#    Adds LoRA B result to base_output in-place.
# ---------------------------------------------------------------------------
def lora_b_sgemm(
    x: torch.Tensor,
    weights: torch.Tensor,
    base_output: torch.Tensor,
) -> None:
    if get_is_capture_mode():
        return
    context = get_forward_context()
    batch_info = context.lora_backend.batch_info
    sgemm_lora_b_fwd(x, weights, batch_info, base_output)


def lora_b_sgemm_fake(
    x: torch.Tensor,
    weights: torch.Tensor,
    base_output: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="lora_b_sgemm",
    op_func=lora_b_sgemm,
    mutates_args=["base_output"],
    fake_impl=lora_b_sgemm_fake,
)


# ---------------------------------------------------------------------------
# 3. qkv_lora: wraps sgemm_lora_a_fwd (stack_num=3) + qkv_lora_b_fwd
#    Runs the full QKV LoRA pass and writes result into base_output.
# ---------------------------------------------------------------------------
def qkv_lora(
    x: torch.Tensor,
    qkv_lora_a: torch.Tensor,
    qkv_lora_b: torch.Tensor,
    base_output: torch.Tensor,
    output_offset: torch.Tensor,
    max_qkv_out_dim: int,
) -> None:
    if get_is_capture_mode():
        return
    context = get_forward_context()
    batch_info = context.lora_backend.batch_info
    lora_a_output = sgemm_lora_a_fwd(x, qkv_lora_a, batch_info, stack_num=3)
    qkv_lora_b_fwd(
        lora_a_output,
        qkv_lora_b,
        batch_info,
        output_offset,
        max_qkv_out_dim,
        base_output,
    )


def qkv_lora_fake(
    x: torch.Tensor,
    qkv_lora_a: torch.Tensor,
    qkv_lora_b: torch.Tensor,
    base_output: torch.Tensor,
    output_offset: torch.Tensor,
    max_qkv_out_dim: int,
) -> None:
    return


direct_register_custom_op(
    op_name="qkv_lora",
    op_func=qkv_lora,
    mutates_args=["base_output"],
    fake_impl=qkv_lora_fake,
)


# ---------------------------------------------------------------------------
# 4. gate_up_lora: wraps sgemm_lora_a_fwd (stack_num=2) + gate_up_lora_b_fwd
#    Runs the full gate_up LoRA pass and writes result into base_output.
# ---------------------------------------------------------------------------
def gate_up_lora(
    x: torch.Tensor,
    gate_up_lora_a: torch.Tensor,
    gate_up_lora_b: torch.Tensor,
    base_output: torch.Tensor,
    output_dim: int,
) -> None:
    if get_is_capture_mode():
        return
    context = get_forward_context()
    batch_info = context.lora_backend.batch_info
    lora_a_output = sgemm_lora_a_fwd(x, gate_up_lora_a, batch_info, stack_num=2)
    gate_up_lora_b_fwd(
        lora_a_output,
        gate_up_lora_b,
        batch_info,
        output_dim,
        base_output,
    )


def gate_up_lora_fake(
    x: torch.Tensor,
    gate_up_lora_a: torch.Tensor,
    gate_up_lora_b: torch.Tensor,
    base_output: torch.Tensor,
    output_dim: int,
) -> None:
    return


direct_register_custom_op(
    op_name="gate_up_lora",
    op_func=gate_up_lora,
    mutates_args=["base_output"],
    fake_impl=gate_up_lora_fake,
)
