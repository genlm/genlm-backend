import pytest
import asyncio
import torch
from conftest import cuda_only
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name, AsyncTransformer
from unittest.mock import patch


@pytest.fixture(scope="module")
def model_name():
    return "gpt2"


@pytest.fixture(scope="module")
def async_llm(model_name):
    return load_model_by_name(model_name, backend="hf")


@pytest.fixture(scope="module")
def token_ids_list(async_llm):
    test_prompts = [
        "There might be something wrong",
        "with the language model code",
        "It's probably this or that",
        "with the language model code",  # Check duplicate query logic
    ]
    return [async_llm.tokenizer.encode(p) for p in test_prompts]


def test_async_batching(async_llm, token_ids_list):
    haves = (
        asyncio.run(async_llm.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
    )
    wants = [
        async_llm.next_token_logprobs_sync(token_ids).cpu().numpy()
        for token_ids in token_ids_list
    ]

    for i, (have, want) in enumerate(zip(haves, wants)):
        max_rel_err = compare(have, want).max_rel_err
        assert max_rel_err == 0, [max_rel_err, token_ids_list[i]]


def test_batch_next_token_logprobs_sync(async_llm, token_ids_list):
    haves = async_llm.batch_next_token_logprobs_sync(token_ids_list).cpu().numpy()
    wants = [
        async_llm.next_token_logprobs_sync(token_ids).cpu().numpy()
        for token_ids in token_ids_list
    ]

    for i, (have, want) in enumerate(zip(haves, wants)):
        max_rel_err = compare(have, want).max_rel_err
        assert max_rel_err == 0, [max_rel_err, token_ids_list[i]]


def test_caching(async_llm):
    async_llm.clear_cache()

    preprompt = async_llm.tokenizer.encode("There might be something wrong")
    prompt = preprompt + async_llm.tokenizer.encode(
        " with the caching logic", add_special_tokens=False
    )

    # Cache preprompt
    have = asyncio.run(async_llm.next_token_logprobs(preprompt)).cpu().numpy()
    want = async_llm.next_token_logprobs_uncached(preprompt).cpu().numpy()

    max_rel_err = compare(have, want).max_rel_err
    assert max_rel_err == 0, max_rel_err  # Sanity check

    curr = async_llm.cache
    for token_id in preprompt:
        assert curr.has_token(token_id), token_id
        curr = curr.get_token(token_id)

    have = asyncio.run(async_llm.next_token_logprobs(prompt)).cpu().numpy()
    want = async_llm.next_token_logprobs_uncached(prompt).cpu().numpy()

    max_rel_err = compare(have, want).max_rel_err
    assert max_rel_err == 0, max_rel_err


def test_empty_input(async_llm):
    # Test that empty input raises ValueError
    with pytest.raises(ValueError):
        asyncio.run(async_llm.next_token_logprobs([]))

    with pytest.raises(ValueError):
        async_llm.next_token_logprobs_sync([])

    with pytest.raises(ValueError):
        async_llm.next_token_logprobs_uncached([])


def test_cache_operations(async_llm):
    # Test cache clearing operations
    async_llm.clear_cache()
    async_llm.clear_kv_cache()

    # Cache some data
    test_prompt = async_llm.tokenizer.encode("Test prompt")
    asyncio.run(async_llm.next_token_logprobs(test_prompt))

    # Clear cache and verify it's empty
    async_llm.clear_cache()
    node, next_token_index, past, base = async_llm.walk_cache(test_prompt)
    assert next_token_index == 0  # Should not match any tokens
    assert past is None
    assert base == 0

    repr(async_llm.cache)


@pytest.mark.asyncio
async def test_reset_async_queries(async_llm):
    # Add some queries
    test_prompt = async_llm.tokenizer.encode("Test prompt")
    future = asyncio.Future()
    async_llm.add_query(test_prompt, future, None)

    # Reset queries
    async_llm.reset_async_queries()
    assert len(async_llm.queries) == 0


def test_cache_kv(async_llm):
    async_llm.clear_cache()

    # Test explicit KV caching
    test_prompt = async_llm.tokenizer.encode("Test KV caching")
    async_llm.cache_kv(test_prompt)

    # Verify the cache contains the KV pairs
    node, next_token_index, past, base = async_llm.walk_cache(test_prompt)

    assert next_token_index == len(test_prompt)
    assert node.past_key_values is not None

    async_llm.clear_kv_cache()


@pytest.mark.skip(reason="This test is flaky")
def test_walk_cache_with_past(async_llm):
    async_llm.clear_cache()

    base_prompt = async_llm.tokenizer.encode("This is a test of")
    extended_prompt = base_prompt + async_llm.tokenizer.encode(
        " caching", add_special_tokens=False
    )

    logprobs_before = asyncio.run(async_llm.next_token_logprobs(extended_prompt))
    assert logprobs_before is not None

    async_llm.cache_kv(base_prompt)
    node, next_token_index, past, base = async_llm.walk_cache(extended_prompt)
    assert past is not None
    assert base == len(base_prompt)
    assert next_token_index == len(base_prompt)

    logprobs_after = asyncio.run(async_llm.next_token_logprobs(extended_prompt))
    assert logprobs_after is not None

    assert torch.allclose(logprobs_before, logprobs_after, atol=1e-3, rtol=1e-3)


def test_next_token_logprobs_with_kv_cache(async_llm):
    async_llm.clear_cache()

    test_prompt = async_llm.tokenizer.encode("Test with past KV")

    logprobs_before = asyncio.run(async_llm.next_token_logprobs(test_prompt))
    assert logprobs_before is not None

    async_llm.cache_kv(test_prompt)

    logprobs_after = asyncio.run(async_llm.next_token_logprobs(test_prompt))
    assert logprobs_after is not None

    assert torch.allclose(logprobs_before, logprobs_after)


def test_next_token_logprobs_sync(async_llm):
    async_llm.clear_cache()

    test_prompt = async_llm.tokenizer.encode("Test sync")
    have = async_llm.next_token_logprobs_sync(test_prompt)
    want = asyncio.run(async_llm.next_token_logprobs(test_prompt))

    assert torch.allclose(have, want)


@pytest.mark.asyncio
async def test_batch_timeout(async_llm):
    # Test that queries are processed after timeout
    async_llm.clear_cache()

    test_prompt = async_llm.tokenizer.encode("Test timeout")
    future = asyncio.get_running_loop().create_future()
    async_llm.add_query(test_prompt, future, None)

    # Wait slightly longer than timeout
    await asyncio.sleep(async_llm.timeout * 1.5)

    # Future should be completed
    assert future.done()


@pytest.mark.asyncio
async def test_full_batch_size(async_llm):
    async_llm.clear_cache()

    try:
        old_batch_size = async_llm.batch_size
        old_timeout = async_llm.timeout
        async_llm.batch_size = 2
        async_llm.timeout = 10

        await asyncio.gather(
            async_llm.next_token_logprobs([0]), async_llm.next_token_logprobs([1])
        )
    finally:
        async_llm.batch_size = old_batch_size
        async_llm.timeout = old_timeout


@cuda_only
def test_from_name_with_options(model_name):
    # Test model creation with various options
    bitsandbytes_opts = {"load_in_4bit": True}
    hf_opts = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }

    model = AsyncTransformer.from_name(
        model_name,
        bitsandbytes_opts=bitsandbytes_opts,
        hf_opts=hf_opts,
        batch_size=10,
        timeout=0.01,
    )

    assert model.batch_size == 10
    assert model.timeout == 0.01


def test_batch_evaluate_empty_queries(async_llm):
    # Test batch evaluation with empty query list
    async_llm.queries = []
    async_llm.batch_evaluate_queries()
    assert len(async_llm.queries) == 0


def test_load_model_by_name_no_backend():
    with patch("torch.cuda.is_available", return_value=False):
        load_model_by_name("gpt2")


def test_sample_seeded(async_llm):
    prompt_token_ids = async_llm.tokenizer.encode("An apple a day keeps the")

    first_token_ids = asyncio.run(
        async_llm.sample(
            prompt_token_ids=prompt_token_ids,
            max_tokens=10,
            eos_token_ids=[11],
            temperature=0.5,
            seed=80808,
        )
    )

    second_token_ids = asyncio.run(
        async_llm.sample(
            prompt_token_ids=prompt_token_ids,
            max_tokens=10,
            eos_token_ids=[11],
            temperature=0.5,
            seed=80808,
        )
    )

    assert first_token_ids == second_token_ids


def test_batch_sample(async_llm):
    prompts = [
        "An apple a day keeps the",
        "The quick brown fox",
        "Jumping jacks",
    ]
    max_tokens = 5
    eos_token_ids = []
    temperature = 0.5

    prompt_token_ids = [async_llm.tokenizer.encode(p) for p in prompts]
    generated_token_ids = asyncio.run(
        async_llm.batch_sample(
            prompt_token_ids_list=prompt_token_ids,
            max_tokens=max_tokens,
            eos_token_ids=eos_token_ids,
            temperature=temperature,
        )
    )
    assert len(generated_token_ids) == len(prompts)
    assert all(len(ids) == max_tokens for ids in generated_token_ids)
