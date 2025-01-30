import torch
import pytest
import asyncio
from conftest import cuda_only
from arsenal.maths import compare
from genlm_backend.llm import AsyncVirtualLM
from genlm_backend.llm import AsyncTransformer


@pytest.fixture(scope="module")
def model_name():
    return "gpt2"


@pytest.fixture(scope="module")
def vllm_llm(model_name):
    return AsyncVirtualLM.from_name(
        model_name,
        engine_opts={
            "gpu_memory_utilization": 0.45,
            "dtype": "float16",
        },
    )


@pytest.fixture(scope="module")
def transformer_llm(model_name):
    return AsyncTransformer.from_name(
        model_name, hf_opts={"torch_dtype": torch.float16}
    )


@pytest.fixture(scope="module")
def token_ids_list(transformer_llm):
    test_prompts = [
        "There might be something wrong",
        "with the language model code",
        "It's probably this or that",
    ]
    tokenizer = transformer_llm.tokenizer
    token_ids_list = [tokenizer.encode(p) for p in test_prompts]
    return token_ids_list


@cuda_only
def test_next_token_logprobs(transformer_llm, vllm_llm, token_ids_list):
    for token_ids in token_ids_list:
        have = transformer_llm.next_token_logprobs_uncached(token_ids).cpu().numpy()
        want = asyncio.run(vllm_llm.next_token_logprobs(token_ids)).cpu().numpy()
        comparison = compare(have, want)
        assert comparison.max_rel_err < 0.03, [
            "max_rel_err",
            comparison.max_rel_err,
            token_ids,
        ]
        assert comparison.pearson > 0.99, ["corr", comparison.pearson, token_ids]


@cuda_only
def test_batch_next_token_logprobs(transformer_llm, vllm_llm, token_ids_list):
    haves = (
        asyncio.run(transformer_llm.batch_next_token_logprobs(token_ids_list))
        .cpu()
        .numpy()
    )
    wants = (
        asyncio.run(vllm_llm.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
    )
    for i, (have, want) in enumerate(zip(haves, wants)):
        comparison = compare(have, want)
        print(comparison.max_rel_err)
        assert comparison.max_rel_err < 0.03, [
            "max_rel_err",
            comparison.max_rel_err,
            token_ids_list[i],
        ]
        assert comparison.pearson > 0.99, [
            "corr",
            comparison.pearson,
            token_ids_list[i],
        ]
