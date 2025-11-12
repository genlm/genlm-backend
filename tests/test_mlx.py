import pytest
import asyncio
import torch
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name, AsyncMlxLM
from genlm.backend.llm.mlx import Query


TOLERANCES = {
    "yujiepan/mamba2-tiny-random": 1e-1,
    "openai-community/gpt2": 1e-3,
}


@pytest.fixture(
    scope="module",
    params=["yujiepan/mamba2-tiny-random", "openai-community/gpt2"],
)
def model_name(request):
    return request.param


@pytest.fixture(scope="module")
def async_llm(model_name):
    llm_opts = {"batch_size": 5 if model_name == "openai-community/gpt2" else 1}
    return load_model_by_name(model_name, backend="mlx", llm_opts=llm_opts)


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
        "There might be something wrong",
        "with the language model code",
        "It's probably this or that",
        "with the language model code",  # Check duplicate query logic
    ]
    return [async_llm.tokenizer.encode(p) for p in test_prompts]


def test_next_token_logprobs(async_llm, reference_llm, token_ids_list, model_name):
    tolerance = TOLERANCES.get(model_name, 1e-3)
    for token_ids in token_ids_list:
        have = asyncio.run(async_llm.next_token_logprobs(token_ids)).cpu().numpy()
        want = asyncio.run(reference_llm.next_token_logprobs(token_ids)).cpu().numpy()
        assert compare(have, want).max_rel_err < tolerance, token_ids


# async and sync batching should yield the same distributions
def test_async_batching(async_llm, token_ids_list, model_name):
    tolerance = TOLERANCES.get(model_name, 1e-3)
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
        assert max_rel_err < tolerance, [max_rel_err, token_ids_list[i]]


def test_batch_next_token_logprobs_sync(async_llm, token_ids_list):
    async_llm.clear_cache()
    haves = async_llm.batch_next_token_logprobs_sync(token_ids_list)
    wants = [
        async_llm.next_token_logprobs_sync(token_ids) for token_ids in token_ids_list
    ]

    for i, (have, want) in enumerate(zip(haves, wants)):
        max_rel_err = compare(have, want).max_rel_err
        assert max_rel_err == 0, [max_rel_err, token_ids_list[i]]


# Test that empty input raises ValueError
def test_empty_input(async_llm):
    with pytest.raises(ValueError):
        asyncio.run(async_llm.next_token_logprobs([]))

    with pytest.raises(ValueError):
        async_llm.next_token_logprobs_sync([])


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
    async_llm.add_query(Query(test_prompt, future))

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


@pytest.mark.asyncio
async def test_reset_async_queries(async_llm):
    test_prompt = async_llm.tokenizer.encode("Test prompt")
    future = asyncio.get_running_loop().create_future()
    async_llm.add_query(Query(test_prompt, future))
    async_llm.reset_async_queries()
    assert len(async_llm.queries) == 0


def test_from_name_with_options(model_name):
    # Test model creation with various options

    model = AsyncMlxLM.from_name(
        model_name,
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


def test_sample_seeded(async_llm):
    prompt_token_ids = async_llm.tokenizer.encode("An apple a day keeps the")

    first_token_ids = asyncio.run(
        async_llm.sample(
            prompt_token_ids=prompt_token_ids,
            max_tokens=10,
            eos_token_ids=[async_llm.tokenizer.eos_token_id],
            temperature=0.5,
            seed=80808,
        )
    )

    second_token_ids = asyncio.run(
        async_llm.sample(
            prompt_token_ids=prompt_token_ids,
            max_tokens=10,
            eos_token_ids=[async_llm.tokenizer.eos_token_id],
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


def test_sample_eos_token_ids(async_llm):
    prompt_token_ids = async_llm.tokenizer.encode("I am the ")
    eos_token_ids = list(range(len(async_llm.tokenizer.vocab.keys())))
    generated_token_ids = asyncio.run(
        async_llm.sample(
            prompt_token_ids=prompt_token_ids,
            max_tokens=10,
            eos_token_ids=eos_token_ids,
        )
    )
    assert len(generated_token_ids) == 0


def test_caching(async_llm):
    async_llm.clear_cache()

    test_prompt = async_llm.tokenizer.encode("Test sync")
    have = async_llm.next_token_logprobs_sync(test_prompt)
    async_llm.clear_cache()
    want = asyncio.run(async_llm.next_token_logprobs(test_prompt))

    assert torch.allclose(have, want)


def test_mlx_prefix_caching(async_llm, model_name):
    if model_name == "yujiepan/mamba2-tiny-random":
        pytest.skip("This model does not support prefix caching")
    async_llm.clear_cache()
    test_prompt = async_llm.tokenizer.encode("Test prefix caching")
    async_llm.cache_kv(test_prompt)
    _, _, _, _, kv_next_token_index = async_llm.walk_cache(test_prompt)
    assert kv_next_token_index == len(test_prompt) - 1
