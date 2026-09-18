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
"""Compact SID projection with streamed full-vocabulary normalization."""

from __future__ import annotations

import dataclasses
from typing import Callable

import torch

from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig


@dataclasses.dataclass
class BeamTrieHeadOutput:
    logits: torch.Tensor
    normalizer: torch.Tensor


class BeamTrieOutputHead:
    """Projects compact candidates while retaining legacy softmax scores."""

    def __init__(
        self,
        config: TrieOutputHeadConfig,
        vocab_size: int,
        chunk_size: int = 32768,
    ):
        self.config = config
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size

    def project_mixed(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        depths: torch.Tensor,
        project: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> BeamTrieHeadOutput:
        start = self.config.token_start
        end = self.config.token_end
        all_logits = project(hidden_states, weight[start:end]).view(
            hidden_states.shape[0],
            self.config.num_codebooks,
            self.config.codebook_size,
        )
        row_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        logits = all_logits[row_indices, depths]
        normalizer = self._stream_normalizer(
            hidden_states, weight, project, transform
        )
        return BeamTrieHeadOutput(logits, normalizer)

    def _stream_normalizer(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        project: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if self.vocab_size > weight.shape[0]:
            raise RuntimeError(
                f"Model vocabulary ({self.vocab_size}) exceeds LM-head rows "
                f"({weight.shape[0]})"
            )

        row_max = None
        row_log_sum = None
        for start in range(0, self.vocab_size, self.chunk_size):
            logical_end = min(start + self.chunk_size, self.vocab_size)
            chunk_logits = project(hidden_states, weight[start:logical_end])
            chunk_logits = transform(chunk_logits)
            if chunk_logits.is_cuda:
                from sglang.srt.layers.logsumexp import row_logsumexp

                chunk_max, chunk_log_sum = row_logsumexp(chunk_logits)
            else:
                chunk_logits = chunk_logits.float()
                chunk_max = chunk_logits.max(dim=-1).values
                chunk_log_sum = torch.log(
                    torch.exp(chunk_logits - chunk_max[:, None]).sum(dim=-1)
                )

            if row_max is None:
                row_max = chunk_max
                row_log_sum = chunk_log_sum
            else:
                next_max = torch.maximum(row_max, chunk_max)
                row_log_sum = torch.logaddexp(
                    row_log_sum + row_max - next_max,
                    chunk_log_sum + chunk_max - next_max,
                )
                row_max = next_max

        assert row_max is not None and row_log_sum is not None
        return torch.stack((row_max, row_log_sum), dim=1)
