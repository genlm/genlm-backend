"""
Evaluates performance differences between sequential and parallel implementations of TokenCharacterTrie
weight_sum using pytest-benchmark. Compares CPU and GPU parallel implementations against sequential baseline.

pytest benchmark_weight_sum.py --benchmark-only
"""

import torch
import pytest
import asyncio
from .util import load_trie
from genlm.backend.llm.base import MockAsyncLM
from genlm.backend.trie import AsyncTokenCharacterTrie

mock_llm = MockAsyncLM.from_name("gpt2")


@pytest.mark.benchmark(warmup=True, max_time=5, warmup_iterations=10)
@pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
@pytest.mark.parametrize("backend", ["sequential", "parallel-cpu", "parallel-gpu"])
def test_batch_weight_sum(benchmark, backend, batch_size):
    benchmark.group = f"batch_size={batch_size}"
    trie = load_trie(mock_llm, backend)
    p_llms = torch.exp(
        torch.stack([mock_llm.next_token_logprobs_sync([i]) for i in range(batch_size)])
    )
    benchmark(trie.batch_weight_sum, p_llms)


class sync_impl:
    def __init__(self, trie):
        self.trie = trie

    def __call__(self, p_llms):
        return self.trie.batch_weight_sum(p_llms)


class async_impl:
    def __init__(self, trie):
        self.async_trie = AsyncTokenCharacterTrie(trie)
        self.loop = asyncio.get_event_loop()

    def __call__(self, p_llms):
        return self.loop.run_until_complete(
            asyncio.gather(*[self.async_trie.weight_sum(p_llm) for p_llm in p_llms])
        )

    def __del__(self):
        self.async_trie.shutdown()


@pytest.mark.benchmark(warmup=True, max_time=5, warmup_iterations=10)
@pytest.mark.parametrize("impl", ["sync", "async"])
def test_async_weight_sum(benchmark, impl, backend="parallel-gpu", batch_size=32):
    trie = load_trie(mock_llm, backend)
    model_cls = async_impl if impl == "async" else sync_impl
    model = model_cls(trie)

    p_llms = torch.exp(
        torch.stack([mock_llm.next_token_logprobs_sync([i]) for i in range(batch_size)])
    )

    benchmark(model, p_llms)
