import unittest

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip(
        "DP MoE padding mask needs CUDA (Triton).",
        allow_module_level=True,
    )

from sglang.srt.layers.dp_attention import (  # noqa: E402
    mask_dp_pad_moe_topk_ids,
    set_dp_buffer_len,
)


class TestDpMoePaddingMask(CustomTestCase):
    def tearDown(self):
        set_dp_buffer_len(0, 0, False)

    def test_cuda_graph_replays_refreshed_valid_counts(self):
        counts = torch.tensor([3, 5], dtype=torch.int32, device="cuda")

        original = torch.arange(12 * 2, dtype=torch.int32, device="cuda").view(12, 2)
        topk_ids = original.clone()
        set_dp_buffer_len(12, 6, True, [6, 6], counts)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            mask_dp_pad_moe_topk_ids(topk_ids)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        topk_ids.copy_(original)
        with torch.cuda.graph(graph):
            mask_dp_pad_moe_topk_ids(topk_ids)

        counts.copy_(torch.tensor([2, 4], dtype=torch.int32, device="cuda"))
        topk_ids.copy_(original)
        graph.replay()
        torch.cuda.synchronize()

        expected = original.clone()
        expected[2:6] = -1
        expected[10:12] = -1
        torch.testing.assert_close(topk_ids, expected)


if __name__ == "__main__":
    unittest.main()
