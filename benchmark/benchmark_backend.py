"""
Evaluates performance differences between AsyncLLM (vLLM-based) and AsyncTransformer
(HuggingFace-based) implementations using pytest-benchmark.

pytest benchmark/benchmark_backend.py --benchmark-only --benchmark-group-by=func
"""

import pytest
from .util import (
    get_wikitext,
    token_prefixes,
    token_prefix_batches,
    run_await_next_token_logprobs,
    run_await_batch_next_token_logprobs,
)

from genlm_backend.llm import AsyncVirtualLM, AsyncTransformer

text = get_wikitext()


def load_model(model, batch_size=None):
    model_name = "gpt2"
    if model == "vllm":
        return AsyncVirtualLM.from_name(model_name)
    else:
        return AsyncTransformer.from_name(model_name, batch_size=batch_size)


@pytest.mark.parametrize("model", ["vllm", "transformer"])
def test_await_next_token_logprobs(benchmark, model):
    llm = load_model(model, batch_size=1)
    sequences = token_prefixes(text, tokenizer=llm.tokenizer)
    run_await_next_token_logprobs(benchmark=benchmark, llm=llm, sequences=sequences)


@pytest.mark.parametrize("model", ["vllm", "transformer"])
def test_await_batch_next_token_logprobs(benchmark, model, batch_size=20):
    llm = load_model(model, batch_size=batch_size)
    batches = token_prefix_batches(text, tokenizer=llm.tokenizer, batch_size=batch_size)
    run_await_batch_next_token_logprobs(
        benchmark=benchmark, llm=llm, batches=batches, rounds=50, warmup_rounds=10
    )
