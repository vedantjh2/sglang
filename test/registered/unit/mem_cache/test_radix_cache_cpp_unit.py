"""Unit tests for fail-closed C++ radix-cache request validation."""

import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRadixCacheCppCacheSalt(CustomTestCase):
    def test_cache_salt_is_rejected_without_loading_cpp_extension(self):
        extension_name = "sglang.srt.mem_cache.cpp_radix_tree.radix_tree"
        module_name = "sglang.srt.mem_cache.radix_cache_cpp"
        fake_extension = types.ModuleType(extension_name)
        fake_extension.IOHandle = object
        fake_extension.RadixTreeCpp = object
        fake_extension.TreeNodeCpp = object

        original_module = sys.modules.pop(module_name, None)
        try:
            with patch.dict(sys.modules, {extension_name: fake_extension}):
                module = importlib.import_module(module_name)
                module.RadixCacheCpp._reject_cache_salt(None)
                with self.assertRaisesRegex(ValueError, "experimental C\\+\\+"):
                    module.RadixCacheCpp._reject_cache_salt("tenant-a")
        finally:
            sys.modules.pop(module_name, None)
            if original_module is not None:
                sys.modules[module_name] = original_module

    def test_non_cacheable_private_prefix_is_freed_on_completion(self):
        extension_name = "sglang.srt.mem_cache.cpp_radix_tree.radix_tree"
        module_name = "sglang.srt.mem_cache.radix_cache_cpp"
        fake_extension = types.ModuleType(extension_name)
        fake_extension.IOHandle = object
        fake_extension.RadixTreeCpp = object
        fake_extension.TreeNodeCpp = object

        original_module = sys.modules.pop(module_name, None)
        try:
            with patch.dict(sys.modules, {extension_name: fake_extension}):
                module = importlib.import_module(module_name)
                cache = module.RadixCacheCpp.__new__(module.RadixCacheCpp)
                cache.page_size = 1
                cache.req_to_token_pool = SimpleNamespace(
                    req_to_token=torch.arange(8, dtype=torch.int32).unsqueeze(0)
                )
                cache.token_to_kv_pool_allocator = MagicMock()
                cache.dec_lock_ref = MagicMock()

                req = SimpleNamespace(
                    cache_salt=None,
                    origin_input_ids=[1, 2, 3, 4, 5],
                    output_ids=[6, 7],
                    prefix_indices=torch.arange(5, dtype=torch.int64),
                    extra_key=None,
                    last_node=object(),
                    skip_radix_cache_insert=True,
                    owns_private_kv=True,
                    kv=SimpleNamespace(
                        holds_kv=True,
                        req_pool_idx=0,
                        cache_protected_len=2,
                    ),
                )

                cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=7)

                freed = cache.token_to_kv_pool_allocator.free.call_args.args[0]
                self.assertTrue(
                    torch.equal(freed, torch.arange(2, 7, dtype=torch.int32))
                )
                cache.dec_lock_ref.assert_called_once_with(req.last_node)
        finally:
            sys.modules.pop(module_name, None)
            if original_module is not None:
                sys.modules[module_name] = original_module


if __name__ == "__main__":
    unittest.main()
