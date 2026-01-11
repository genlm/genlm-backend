import pytest
import asyncio
import torch
import gc
from unittest.mock import patch
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
            "chunked_prefill_size": 200,
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


@pytest.fixture(scope="module")
def token_ids_list(async_llm):
    test_prompts = [
        "There might be something wrong, it may be because ",
        "with the language model code",
        "It's probably this or that",
        "with the language model code",  # Check duplicate query logic
    ]
    return [async_llm.tokenizer.encode(p) for p in test_prompts]


@pytest.fixture(scope="module")
def long_token_ids_list(async_llm):
    test_prompts = [
        "Cat goes " + "meow " * 200,
        "Dog goes " + "woof " * 200,
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
def test_chunked_prefill(async_llm, reference_llm, long_token_ids_list):
    for token_ids in long_token_ids_list:
        have = asyncio.run(async_llm.next_token_logprobs(token_ids)).cpu().numpy()
        want = asyncio.run(reference_llm.next_token_logprobs(token_ids)).cpu().numpy()
        max_rel_err = compare(have, want).max_rel_err
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
def test_next_token_logprobs_sync(async_llm, token_ids_list):
    async_llm.clear_cache()

    have = async_llm.next_token_logprobs_sync(token_ids_list[0])
    async_llm.clear_cache()
    want = asyncio.run(async_llm.next_token_logprobs(token_ids_list[0]))

    assert torch.allclose(have, want, atol=1e-3, rtol=1e-3)


@cuda_only
def test_caching(async_llm, token_ids_list):
    async_llm.clear_cache()

    have = async_llm.next_token_logprobs_sync(token_ids_list[0])
    want = asyncio.run(async_llm.next_token_logprobs(token_ids_list[0]))

    assert torch.allclose(have, want)


@cuda_only
def test_clear_kv_cache(async_llm):
    ret = async_llm.clear_kv_cache()
    assert ret


@cuda_only
def test_reset_async_queries(async_llm):
    async_llm.reset_async_queries()
    assert async_llm._pending == {}
    assert async_llm._inflight == {}
    assert async_llm._rid_to_token_ids == {}


@cuda_only
@pytest.mark.asyncio
async def test_register_with_cancelled_future(async_llm, token_ids_list):
    fut = asyncio.get_running_loop().create_future()
    fut.cancel()
    result = async_llm._register(tuple(token_ids_list[0]), fut)
    assert result is None


@cuda_only
@pytest.mark.asyncio
async def test_reset_async_queries_with_pending_futures(async_llm, long_token_ids_list):
    async_llm.clear_cache()
    async_llm.clear_kv_cache()
    async_llm._pause_engine()
    fut = asyncio.get_running_loop().create_future()
    async_llm._queue.put_nowait((tuple(long_token_ids_list[0]), fut))

    async_llm.reset_async_queries()
    assert fut.cancelled()
    assert async_llm._pending == {}
    assert async_llm._inflight == {}
    assert async_llm._rid_to_token_ids == {}


@cuda_only
@pytest.mark.asyncio
async def test_background_loop_exception_handling(async_llm, token_ids_list):
    """Test that exceptions in the background loop are properly propagated to pending futures."""
    async_llm.clear_cache()
    async_llm.clear_kv_cache()

    test_exception = RuntimeError("Test exception in background loop")

    with patch.object(
        async_llm.model, "process_input_requests", side_effect=test_exception
    ):
        fut1 = asyncio.create_task(async_llm.next_token_logprobs(token_ids_list[0]))
        fut2 = asyncio.create_task(async_llm.next_token_logprobs(token_ids_list[1]))

        await asyncio.sleep(0.3)

        assert fut1.done()
        assert fut2.done()
        with pytest.raises(RuntimeError, match="Test exception in background loop"):
            await fut1
        with pytest.raises(RuntimeError, match="Test exception in background loop"):
            await fut2

        assert async_llm._pending == {}
        assert async_llm._inflight == {}
        assert async_llm._rid_to_token_ids == {}


@cuda_only
def test_del_cleanup(async_llm, token_ids_list):
    asyncio.run(async_llm.next_token_logprobs(token_ids_list[0]))

    assert async_llm._loop is not None
    assert async_llm._task is not None

    async_llm._cleanup_engine()

    del async_llm
    gc.collect()
