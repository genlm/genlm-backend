"""
Evaluates the performance gains from internal modifications to the vllm engine.
Compares AsyncVirtualLM to reference implementation which does not modify the vllm engine internally.

pytest benchmark/benchmark_optimizations.py --benchmark-only --benchmark-group-by=func
"""

import pytest
import asyncio
from genlm_backend.llm import AsyncVirtualLM
from genlm_backend.llm.vllm_reference import ReferenceVirtualLM
from .util import get_wikitext, token_prefixes, prefix_batches

text = get_wikitext()

def load_model(model):
    model_name = 'gpt2'
    if model == 'optimized':
        return AsyncVirtualLM.from_name(model_name)
    else:
        return ReferenceVirtualLM.from_name(model_name)

def run_single(benchmark, llm):
    loop = asyncio.new_event_loop()

    prefixes = token_prefixes(text, llm.tokenizer)
    async def run():
        await llm.next_token_logprobs(next(prefixes))

    benchmark.pedantic(
        lambda: loop.run_until_complete(run()), 
        iterations=1, 
        rounds=200, 
        warmup_rounds=20, 
    )

    loop.close()

def run_batch(benchmark, llm, batch_size):
    loop = asyncio.new_event_loop()

    batches = prefix_batches(text, llm.tokenizer, batch_size)
    async def run():
        await llm.batch_next_token_logprobs(next(batches))

    benchmark.pedantic(
        lambda: loop.run_until_complete(run()), 
        iterations=1, 
        rounds=50, 
        warmup_rounds=5, 
    )

    loop.close()

@pytest.mark.parametrize("model", ["optimized", "reference"])
def test_await_next_token_logprobs(benchmark, model):
    run_single(benchmark, load_model(model))

@pytest.mark.parametrize("model", ["optimized", "reference"])
def test_await_batch_next_token_logprobs(benchmark, model, batch_size=20):
    run_batch(benchmark, load_model(model), batch_size=batch_size)