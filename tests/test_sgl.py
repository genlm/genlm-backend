import pytest
import asyncio
import torch
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name
from conftest import cuda_only


@pytest.fixture(scope="module")
def model_name():
    return "gpt2"


@pytest.fixture(scope="module")
def async_llm(model_name):
    llm_opts = {
        "cache_size": 100,
        "engine_opts": {
            "disable_cuda_graph": True,
            "attention_backend": "torch_native",
        },
    }
    return load_model_by_name(model_name, backend="sgl", llm_opts=llm_opts)


@pytest.fixture(scope="module")
def reference_llm(model_name):
    return load_model_by_name(
        model_name,
        backend="hf",
        llm_opts={
            "hf_opts": {
                "device_map": None,
            }
        },
    )


# return a list of token ids for the test prompts
@pytest.fixture(scope="module")
def token_ids_list(async_llm):
    test_prompts = [
        "There might be something wrong, it may be because ",
        "with the language model code",
        "It's probably this or that",
        "with the language model code",  # Check duplicate query logic
    ]
    return [async_llm.tokenizer.encode(p) for p in test_prompts]


@cuda_only
def test_next_token_logprobs(async_llm, reference_llm, token_ids_list):
    for token_ids in token_ids_list:
        have = asyncio.run(async_llm.next_token_logprobs(token_ids)).cpu().numpy()
        want = asyncio.run(reference_llm.next_token_logprobs(token_ids)).cpu().numpy()
        max_rel_err = compare(have, want).max_rel_err
        # Allow a higher tolerance for different backends
        assert max_rel_err < 2e-2, token_ids


@cuda_only
def test_async_batching(async_llm, token_ids_list):
    async_llm.clear_cache()
    haves = (
        asyncio.run(async_llm.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
    )
    wants = [
        async_llm.next_token_logprobs_sync(token_ids).cpu().numpy()
        for token_ids in token_ids_list
    ]

    for i, (have, want) in enumerate(zip(haves, wants)):
        max_rel_err = compare(have, want).max_rel_err
        assert max_rel_err < 1e-3, [max_rel_err, token_ids_list[i]]


@cuda_only
def test_batch_next_token_logprobs_sync(async_llm, token_ids_list):
    async_llm.clear_cache()
    haves = async_llm.batch_next_token_logprobs_sync(token_ids_list)
    wants = [
        async_llm.next_token_logprobs_sync(token_ids) for token_ids in token_ids_list
    ]

    for i, (have, want) in enumerate(zip(haves, wants)):
        max_rel_err = compare(have, want).max_rel_err
        assert max_rel_err < 1e-3, [max_rel_err, token_ids_list[i]]


@cuda_only
def test_empty_input(async_llm):
    with pytest.raises(ValueError):
        asyncio.run(async_llm.next_token_logprobs([]))

    with pytest.raises(ValueError):
        async_llm.next_token_logprobs_sync([])


@cuda_only
def test_next_token_logprobs_sync(async_llm):
    async_llm.clear_cache()

    test_prompt = async_llm.tokenizer.encode("Test sync")
    have = async_llm.next_token_logprobs_sync(test_prompt)
    want = asyncio.run(async_llm.next_token_logprobs(test_prompt))

    assert torch.allclose(have, want)


@cuda_only
def test_caching(async_llm):
    async_llm.clear_cache()

    test_prompt = async_llm.tokenizer.encode("Test sync")
    have = async_llm.next_token_logprobs_sync(test_prompt)
    async_llm.clear_cache()
    want = asyncio.run(async_llm.next_token_logprobs(test_prompt))

    assert torch.allclose(have, want)
