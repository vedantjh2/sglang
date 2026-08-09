import unittest
from types import SimpleNamespace

from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin


class TestMixedBatchPolicy(unittest.TestCase):
    def test_mixes_when_decode_can_fill_unused_prefill_rows(self):
        self.assertTrue(
            SchedulerDllmMixin._should_mix_dllm_batches(
                num_prefill_reqs=12,
                num_decode_reqs=20,
                round_capacity=64,
            )
        )

    def test_keeps_original_path_for_full_prefill_round(self):
        self.assertFalse(
            SchedulerDllmMixin._should_mix_dllm_batches(
                num_prefill_reqs=64,
                num_decode_reqs=20,
                round_capacity=64,
            )
        )

    def test_keeps_original_path_for_single_phase_rounds(self):
        for num_prefill_reqs, num_decode_reqs in ((32, 0), (0, 32), (0, 0)):
            with self.subTest(
                num_prefill_reqs=num_prefill_reqs,
                num_decode_reqs=num_decode_reqs,
            ):
                self.assertFalse(
                    SchedulerDllmMixin._should_mix_dllm_batches(
                        num_prefill_reqs=num_prefill_reqs,
                        num_decode_reqs=num_decode_reqs,
                        round_capacity=64,
                    )
                )

    def test_activates_for_later_admitted_cohorts(self):
        scheduler = object.__new__(SchedulerDllmMixin)
        scheduler.dllm_manager = SimpleNamespace(
            waiting_queue=[
                SimpleNamespace(
                    time_stats=SimpleNamespace(scheduler_recv_time=1.0)
                )
            ]
        )
        scheduler._reset_dllm_mixed_batch_admission_state()

        self.assertFalse(scheduler._has_active_dllm_admissions())
        self.assertFalse(scheduler._has_active_dllm_admissions())

        scheduler.dllm_manager.waiting_queue.append(
            SimpleNamespace(time_stats=SimpleNamespace(scheduler_recv_time=2.0))
        )
        self.assertTrue(scheduler._has_active_dllm_admissions())

        for _ in range(15):
            self.assertTrue(scheduler._has_active_dllm_admissions())
        self.assertFalse(scheduler._has_active_dllm_admissions())

        scheduler._reset_dllm_mixed_batch_admission_state()
        self.assertFalse(scheduler._has_active_dllm_admissions())


if __name__ == "__main__":
    unittest.main()
