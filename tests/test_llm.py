import pytest
import asyncio
from conftest import cuda_only
from arsenal.maths import compare
from genlm_backend.llm import AsyncVirtualLM
from genlm_backend.llm.vllm_reference import ReferenceVirtualLM


@pytest.fixture(scope="module")
def model_name():
    return "gpt2"


@pytest.fixture(scope="module")
def reference_llm(model_name):
    return ReferenceVirtualLM.from_name(
        model_name, llm_opts={"gpu_memory_utilization": 0.45}
    )


@pytest.fixture(scope="module")
def async_llm(model_name):
    return AsyncVirtualLM.from_name(
        model_name, engine_opts={"gpu_memory_utilization": 0.45}
    )


@pytest.fixture(scope="module")
def token_ids_list(async_llm):
    test_prompts = [
        "There might be something wrong",
        "with the language model code",
        "It's probably this or that",
    ]
    tokenizer = async_llm.tokenizer
    token_ids_list = [tokenizer.encode(p) for p in test_prompts]
    return token_ids_list


@cuda_only
def test_next_token_logprobs(async_llm, reference_llm, token_ids_list):
    for token_ids in token_ids_list:
        have = asyncio.run(async_llm.next_token_logprobs(token_ids))
        have = have.cpu().numpy()
        want = asyncio.run(reference_llm.next_token_logprobs(token_ids))
        assert compare(have, want).max_rel_err < 1e-5, token_ids


@cuda_only
def test_batch_next_token_logprobs(async_llm, reference_llm, token_ids_list):
    haves = asyncio.run(async_llm.batch_next_token_logprobs(token_ids_list))
    haves = haves.cpu().numpy()
    wants = asyncio.run(reference_llm.batch_next_token_logprobs(token_ids_list))
    for i, (have, want) in enumerate(zip(haves, wants)):
        assert compare(have, want).max_rel_err < 1e-5, token_ids_list[i]


@cuda_only
def test_next_token_logprobs_sync(async_llm, token_ids_list):
    # Test 1: Regular sync context.
    have = async_llm.next_token_logprobs_sync(token_ids_list[0]).cpu().numpy()
    want = asyncio.run(async_llm.next_token_logprobs(token_ids_list[0])).cpu().numpy()
    assert compare(have, want).max_rel_err == 0, "Sync context"

    # Test 2: Sync function inside async context
    # This is a non-standard but valid use case (it is used in LLaMPPL at LMContext init, which can occur inside an async context).
    async def async_context():
        have_async = async_llm.next_token_logprobs_sync(token_ids_list[0])
        return have_async.cpu().numpy()

    have_in_async = asyncio.run(async_context())
    assert compare(have_in_async, want).max_rel_err == 0, "Sync in async context"
