import pytest
import asyncio
from arsenal.maths import compare
from genlm_backend.llm import AsyncTransformer


@pytest.fixture(scope="module")
def model_name():
    return "gpt2"


@pytest.fixture(scope="module")
def async_llm(model_name):
    return AsyncTransformer.from_name(model_name)


def test_async_batching(async_llm):
    test_prompts = [
        "There might be something wrong",
        "with the language model code",
        "It's probably this or that",
        "with the language model code",  # Check duplicate query logic
    ]
    token_ids_list = [async_llm.tokenizer.encode(p) for p in test_prompts]

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
