import pytest
import asyncio
import torch
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name, AsyncMlxLM


TOLERANCES = {
    "yujiepan/mamba2-tiny-random": 5e-2,
    "openai-community/gpt2": 1e-3,
}


@pytest.fixture(
    scope="module",
    params=["openai-community/gpt2", "yujiepan/mamba2-tiny-random"],
)
def model_name(request):
    return request.param


@pytest.fixture(scope="module")
def async_llm(model_name):
    llm_opts = {
        "cache_size": 4,
    }
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
        "There might be something wrong, it may be because ",
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
    haves = async_llm.batch_next_token_logprobs_sync(token_ids_list).cpu()
    wants = [
        async_llm.next_token_logprobs_sync(token_ids).cpu()
        for token_ids in token_ids_list
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
    # A batch that never fills is still run, on the timer.
    async_llm.clear_cache()

    test_prompt = async_llm.tokenizer.encode("Test timeout")
    logprobs = await asyncio.wait_for(
        async_llm.next_token_logprobs(test_prompt), timeout=60
    )

    assert logprobs.ndim == 1


@pytest.mark.asyncio
async def test_window_batches_concurrent_asks(async_llm):
    # A concurrent gather lands in one window and resolves as one batch.
    async_llm.clear_cache()
    a, b = await asyncio.gather(
        async_llm.next_token_logprobs([0]), async_llm.next_token_logprobs([1])
    )
    assert a.ndim == 1 and b.ndim == 1


@pytest.mark.asyncio
async def test_reset_async_queries(async_llm):
    # A queued query can be dropped without ever running (the window's linger
    # holds it in the queue long enough to reset).
    old_timeout = async_llm.timeout
    async_llm.timeout = 60
    try:
        test_prompt = async_llm.tokenizer.encode("Test prompt")
        task = asyncio.ensure_future(async_llm.next_token_logprobs(test_prompt))
        await asyncio.sleep(0)  # let it reach the queue
        async_llm.reset_async_queries()
        assert not task.done()
        task.cancel()
    finally:
        async_llm.timeout = old_timeout


def test_from_name_with_options(model_name):
    # Test model creation with various options

    model = AsyncMlxLM.from_name(
        model_name,
        timeout=0.01,
    )

    assert model.timeout == 0.01


def test_batch_evaluate_empty_queries(async_llm):
    # An empty cohort flushes harmlessly (a reset can empty the window's queue).
    async_llm._batch_evaluate([])
    assert len(async_llm._queries) == 0


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


def test_kv_reuse_matches_cold_prefill(async_llm, model_name, token_ids_list):
    # Walking the live KV rows forward must agree with prefilling the same contexts.
    tolerance = TOLERANCES.get(model_name, 1e-3)
    contexts = [list(c) for c in dict.fromkeys(map(tuple, token_ids_list))]
    extended = [c + [100] for c in contexts]

    async_llm.clear_cache()
    want = async_llm.batch_next_token_logprobs_sync(extended).cpu().numpy()

    async_llm.clear_cache()
    async_llm.batch_next_token_logprobs_sync(contexts)  # seed the rows
    have = async_llm.batch_next_token_logprobs_sync(extended).cpu().numpy()

    for i, context in enumerate(contexts):
        assert compare(want[i], have[i]).max_rel_err < tolerance, context


def test_kv_fork_matches_cold_prefill(async_llm, model_name, token_ids_list):
    # Two rows continuing one row -- what a resample asks for -- must agree too.
    tolerance = TOLERANCES.get(model_name, 1e-3)
    parent, other = token_ids_list[0], token_ids_list[1]
    forked = [parent + [100], parent + [101], other + [102]]

    async_llm.clear_cache()
    want = async_llm.batch_next_token_logprobs_sync(forked).cpu().numpy()

    async_llm.clear_cache()
    async_llm.batch_next_token_logprobs_sync([parent, other])
    have = async_llm.batch_next_token_logprobs_sync(forked).cpu().numpy()

    for i in range(len(forked)):
        assert compare(want[i], have[i]).max_rel_err < tolerance, i
