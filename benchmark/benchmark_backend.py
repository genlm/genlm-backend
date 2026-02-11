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

from genlm.backend.llm import load_model_by_name

text = get_wikitext()


def load_model(backend, batch_size=None):
    model_name = "gpt2"
    if backend in ["vllm", "sglang"]:
        return load_model_by_name(model_name, backend=backend)
    else:
        return load_model_by_name(model_name, backend=backend, batch_size=batch_size)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_await_next_token_logprobs(benchmark, backend):
    llm = load_model(backend, batch_size=1)
    sequences = token_prefixes(text, tokenizer=llm.tokenizer)
    run_await_next_token_logprobs(benchmark=benchmark, llm=llm, sequences=sequences)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_await_batch_next_token_logprobs(benchmark, backend, batch_size=20):
    llm = load_model(backend, batch_size=batch_size)
    batches = token_prefix_batches(text, tokenizer=llm.tokenizer, batch_size=batch_size)
    run_await_batch_next_token_logprobs(
        benchmark=benchmark, llm=llm, batches=batches, rounds=50, warmup_rounds=10
    )
