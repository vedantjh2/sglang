import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.schedule_policy import (
    PrefillAdder,
    get_dllm_prompt_superblock_tokens,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(
    prompt_tokens: int,
    output_tokens: int,
    *,
    generated_tokens: int = 0,
    is_prefill: bool = True,
    incomplete: bool = False,
):
    return SimpleNamespace(
        origin_input_ids=array("q", range(prompt_tokens)),
        output_ids=list(range(generated_tokens)),
        sampling_params=SimpleNamespace(max_new_tokens=output_tokens),
        dllm_incomplete_ids=array("q", [1]) if incomplete else array("q"),
        is_dllm_prefill=lambda: is_prefill,
        prefix_indices=[],
        time_stats=SimpleNamespace(wait_queue_entry_time=9.95),
    )


def _make_adder(ready_reqs: int, remaining_tokens: int = 2048) -> PrefillAdder:
    adder = object.__new__(PrefillAdder)
    adder.dllm_block_size = 32
    adder.rem_dllm_tokens = remaining_tokens
    adder.dllm_ready_reqs_remaining = ready_reqs
    adder.rem_total_token_offset = 0
    adder.is_all_swa = False
    adder.is_hybrid_swa = False
    adder.is_hybrid_ssm_cache = False
    adder.token_to_kv_pool_allocator = SimpleNamespace(
        available_size=lambda: 1 << 20
    )
    adder.tree_cache = SimpleNamespace(evictable_size=lambda: 0)
    return adder


class TestPromptSuperblocks(unittest.TestCase):
    def test_eligibility_uses_complete_prompt_blocks(self):
        req = _make_req(prompt_tokens=530, output_tokens=256)

        self.assertEqual(
            get_dllm_prompt_superblock_tokens(req, prefix_len=0, block_size=32),
            512,
        )

    def test_eligibility_rejects_decode_dominant_and_incomplete_requests(self):
        decode_dominant = _make_req(prompt_tokens=256, output_tokens=1024)
        incomplete = _make_req(
            prompt_tokens=1024,
            output_tokens=256,
            incomplete=True,
        )

        self.assertEqual(
            get_dllm_prompt_superblock_tokens(
                decode_dominant,
                prefix_len=0,
                block_size=32,
            ),
            0,
        )
        self.assertEqual(
            get_dllm_prompt_superblock_tokens(
                incomplete,
                prefix_len=0,
                block_size=32,
            ),
            0,
        )

    @patch(
        "sglang.srt.managers.schedule_policy.envs."
        "SGLANG_DLLM_PREFILL_BLOCKS_PER_FORWARD.get",
        return_value=16,
    )
    def test_fair_share_preserves_request_width_before_using_spare_tokens(
        self,
        _mock_prefill_blocks,
    ):
        req = _make_req(prompt_tokens=1024, output_tokens=256)

        expected_limits = {
            64: 32,
            16: 128,
            8: 256,
            1: 512,
        }
        for ready_reqs, expected_limit in expected_limits.items():
            with self.subTest(ready_reqs=ready_reqs):
                adder = _make_adder(ready_reqs)
                self.assertEqual(
                    adder._get_dllm_req_token_limit(req, prefix_len=0),
                    expected_limit,
                )

    @patch(
        "sglang.srt.managers.schedule_policy.envs."
        "SGLANG_DLLM_PREFILL_BLOCKS_PER_FORWARD.get",
        return_value=1,
    )
    def test_disabled_policy_keeps_one_diffusion_block(self, _mock_prefill_blocks):
        req = _make_req(prompt_tokens=1024, output_tokens=256)
        adder = _make_adder(ready_reqs=1)

        self.assertEqual(adder._get_dllm_req_token_limit(req, prefix_len=0), 32)

    def test_fdfo_superblock_fast_path_returns_known_tokens_in_one_forward(self):
        algorithm = object.__new__(DllmAlgorithm)
        algorithm.block_size = 32
        model_runner = MagicMock()
        model_runner.forward.return_value = SimpleNamespace(
            logits_output="logits",
            can_run_graph=True,
        )
        forward_batch = SimpleNamespace(
            batch_size=2,
            extend_seq_lens_cpu=[64, 32],
            input_ids=torch.arange(96),
        )

        output = algorithm._run_fdfo(
            model_runner,
            forward_batch,
            algo_states=None,
        )

        self.assertEqual(output[0], "logits")
        self.assertEqual(output[1], [list(range(64)), list(range(64, 96))])
        self.assertEqual(output[2], [64, 32])
        self.assertEqual(output[3], [None, None])
        self.assertTrue(output[4])
        model_runner.forward.assert_called_once_with(
            forward_batch,
            pp_proxy_tensors=None,
        )

    def test_fdfo_superblock_fast_path_rejects_partial_blocks(self):
        algorithm = object.__new__(DllmAlgorithm)
        algorithm.block_size = 32
        forward_batch = SimpleNamespace(
            batch_size=1,
            extend_seq_lens_cpu=[48],
            input_ids=torch.arange(48),
        )

        with self.assertRaisesRegex(RuntimeError, "complete diffusion blocks"):
            algorithm._run_fdfo(
                MagicMock(),
                forward_batch,
                algo_states=None,
            )

    def test_decode_graph_rejects_variable_width_superblocks(self):
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.captured_req_width = 32
        forward_batch = SimpleNamespace(
            replace_embeds=None,
            forward_mode=SimpleNamespace(is_dllm_extend=lambda: True),
            extend_seq_lens_cpu=[64, 32],
        )

        self.assertFalse(runner.can_run_graph(forward_batch))

    @patch(
        "sglang.srt.dllm.mixin.scheduler.envs."
        "SGLANG_DLLM_PREFILL_BLOCKS_PER_FORWARD.get",
        return_value=16,
    )
    @patch("sglang.srt.dllm.mixin.scheduler.time.perf_counter", return_value=10.0)
    def test_idle_coalescing_only_delays_eligible_prompt_work(
        self,
        _mock_time,
        _mock_prefill_blocks,
    ):
        scheduler = SimpleNamespace(
            dllm_idle_coalesce_size=64,
            dllm_idle_coalesce_max_wait_seconds=0.1,
            dllm_manager=SimpleNamespace(is_empty=lambda: True),
            dllm_config=SimpleNamespace(block_size=32),
            waiting_queue=[_make_req(prompt_tokens=1024, output_tokens=256)],
        )

        self.assertTrue(
            SchedulerDllmMixin._should_coalesce_idle_dllm_requests(scheduler)
        )

        scheduler.waiting_queue = [
            _make_req(prompt_tokens=256, output_tokens=1024)
        ]
        self.assertFalse(
            SchedulerDllmMixin._should_coalesce_idle_dllm_requests(scheduler)
        )


if __name__ == "__main__":
    unittest.main()
