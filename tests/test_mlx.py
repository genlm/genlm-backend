import pytest
import asyncio
import torch
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name


# returns the default gpt2 model name
@pytest.fixture(scope="module")
def model_name():
    return "openai-community/gpt2"


# returns the instantiated async lm with the default gpt model from the hf backend
@pytest.fixture(scope="module")
def async_llm(model_name):
    return load_model_by_name(model_name, backend="mlx", llm_opts={"cache_size": 3})


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


def test_next_token_logprobs(async_llm, reference_llm, token_ids_list):
    for token_ids in token_ids_list:
        have = asyncio.run(async_llm.next_token_logprobs(token_ids)).cpu().numpy()
        want = asyncio.run(reference_llm.next_token_logprobs(token_ids)).cpu().numpy()
        assert compare(have, want).max_rel_err < 1e-3, token_ids


# async and sync batching should yield the same distributions
def test_async_batching(async_llm, token_ids_list):
    haves = asyncio.run(async_llm.batch_next_token_logprobs(token_ids_list))
    wants = [
        async_llm.next_token_logprobs_sync(token_ids) for token_ids in token_ids_list
    ]

    for i, (have, want) in enumerate(zip(haves, wants)):
        max_rel_err = compare(have, want).max_rel_err
        assert max_rel_err == 0, [max_rel_err, token_ids_list[i]]


def test_batch_next_token_logprobs_sync(async_llm, token_ids_list):
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
    want = asyncio.run(async_llm.next_token_logprobs(test_prompt))

    assert torch.allclose(have, want)


def test_mlx_cache(async_llm, token_ids_list):
    async_llm.clear_cache()
    for token_ids in token_ids_list:
        async_llm.next_token_logprobs_sync(token_ids)
    assert len(async_llm.cache) == 3
