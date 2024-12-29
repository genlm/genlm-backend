"""
Evaluates performance differences between AsyncLLM (vLLM-based) and AsyncTransformer 
(HuggingFace-based) implementations using pytest-benchmark.

pytest benchmark/benchmark_backend.py --benchmark-only --benchmark-group-by=func
"""

import pytest
import asyncio
from genlm_backend.llm import AsyncVirtualLM, AsyncTransformer
from .util import get_wikitext, token_prefixes, prefix_batches

text = get_wikitext()

def load_model(model, batch_size=None):
    model_name = 'gpt2'
    if model == 'vllm':
        return AsyncVirtualLM.from_name(model_name)
    else:
        return AsyncTransformer.from_name(model_name, batch_size=batch_size)

def run_single(benchmark, llm):
    loop = asyncio.new_event_loop()

    prefixes = token_prefixes(text, llm.tokenizer)
    async def run():
        token_ids = next(prefixes)
        assert token_ids
        await llm.next_token_logprobs(token_ids)

    benchmark.pedantic(
        lambda: loop.run_until_complete(run()), 
        iterations=1, 
        rounds=20, 
        warmup_rounds=10, 
    )

    loop.close()

def run_batch(benchmark, llm, batch_size):
    loop = asyncio.new_event_loop()

    batches = prefix_batches(text, llm.tokenizer, batch_size)
    async def run():
        token_ids = next(batches)
        assert token_ids
        await llm.batch_next_token_logprobs(token_ids)

    benchmark.pedantic(
        lambda: loop.run_until_complete(run()), 
        iterations=1, 
        rounds=20, 
        warmup_rounds=10, 
    )

    loop.close()

@pytest.mark.parametrize("model", ["vllm", "transformer"])
def test_await_next_token_logprobs(benchmark, model):
    run_single(benchmark, load_model(model, batch_size=1))

@pytest.mark.parametrize("model", ["vllm", "transformer"])
def test_await_batch_next_token_logprobs(benchmark, model, batch_size=20):
    run_batch(benchmark, load_model(model, batch_size=batch_size), batch_size=batch_size)