"""
Evaluates the performance gains from internal modifications to the vllm engine.
Compares AsyncVirtualLM to reference implementation which does not modify the vllm engine internally.

pytest benchmark/benchmark_optimizations.py --benchmark-only --benchmark-group-by=func
"""

import pytest
from .util import (
    get_wikitext,
    token_prefixes,
    token_prefix_batches,
    run_await_next_token_logprobs,
    run_await_batch_next_token_logprobs,
)
from genlm_backend.llm import AsyncVirtualLM
from genlm_backend.llm.vllm_reference import ReferenceVirtualLM

text = get_wikitext()


def load_model(model):
    model_name = "gpt2"
    if model == "optimized":
        return AsyncVirtualLM.from_name(model_name)
    else:
        return ReferenceVirtualLM.from_name(model_name)


@pytest.mark.parametrize("model", ["optimized", "reference"])
def test_await_next_token_logprobs(benchmark, model):
    llm = load_model(model)
    sequences = token_prefixes(text, tokenizer=llm.tokenizer)
    run_await_next_token_logprobs(benchmark=benchmark, llm=llm, sequences=sequences)


@pytest.mark.parametrize("model", ["optimized", "reference"])
def test_await_batch_next_token_logprobs(benchmark, model, batch_size=20):
    llm = load_model(model)
    batches = token_prefix_batches(text, tokenizer=llm.tokenizer, batch_size=batch_size)
    run_await_batch_next_token_logprobs(benchmark=benchmark, llm=llm, batches=batches)
