# Copyright 2026 SGLang Team
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
"""CUDA kernels for beam-row layout conversion."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_ROW_COPY_BLOCK_SIZE = 512


@triton.jit
def _copy_grouped_beam_rows_kernel(
    source,
    output,
    num_groups,
    reverse: tl.constexpr,
    beam_width: tl.constexpr,
    row_size: tl.constexpr,
    block_size: tl.constexpr,
):
    grouped_row = tl.program_id(0)
    columns = tl.program_id(1) * block_size + tl.arange(0, block_size)
    valid = columns < row_size
    group = grouped_row // beam_width
    beam = grouped_row % beam_width
    physical_row = tl.where(
        beam == 0,
        group,
        num_groups + group * (beam_width - 1) + beam - 1,
    )
    if reverse:
        source_row = grouped_row
        output_row = physical_row
    else:
        source_row = physical_row
        output_row = grouped_row
    values = tl.load(
        source + source_row * row_size + columns,
        mask=valid,
    )
    tl.store(
        output + output_row * row_size + columns,
        values,
        mask=valid,
    )


def _copy_grouped_beam_rows(
    source: torch.Tensor,
    num_groups: int,
    beam_width: int,
    *,
    reverse: bool,
) -> torch.Tensor:
    num_rows = num_groups * beam_width
    if not reverse and source.shape[0] != num_rows:
        raise ValueError(f"Expected {num_rows} beam rows, got {source.shape[0]}")
    if source.numel() % num_rows:
        raise ValueError(
            f"Cannot reshape {source.numel()} values into {num_rows} beam rows"
        )
    flat_source = source.contiguous().view(num_rows, -1)
    output = torch.empty_like(flat_source)
    _copy_grouped_beam_rows_kernel[
        (
            num_rows,
            triton.cdiv(flat_source.shape[1], _ROW_COPY_BLOCK_SIZE),
        )
    ](
        flat_source,
        output,
        num_groups,
        reverse=reverse,
        beam_width=beam_width,
        row_size=flat_source.shape[1],
        block_size=_ROW_COPY_BLOCK_SIZE,
    )
    return output


def group_beam_rows(
    source: torch.Tensor,
    num_groups: int,
    beam_width: int,
) -> torch.Tensor:
    """Convert scheduler layout into contiguous request-major beam groups."""
    return _copy_grouped_beam_rows(
        source,
        num_groups,
        beam_width,
        reverse=False,
    )


def ungroup_beam_rows(
    source: torch.Tensor,
    num_groups: int,
    beam_width: int,
) -> torch.Tensor:
    """Restore request-major beam groups to the scheduler row layout."""
    return _copy_grouped_beam_rows(
        source,
        num_groups,
        beam_width,
        reverse=True,
    )
