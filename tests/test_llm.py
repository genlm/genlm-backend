import torch
import pytest
import asyncio
from conftest import cuda_only, ReferenceVirtualLM
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name, MockAsyncLM, AsyncVirtualLM

# from hypothesis import given, strategies as st, settings


@pytest.fixture(scope="module")
def model_name():
    return "gpt2"


@pytest.fixture(scope="module")
def reference_llm(model_name):
    return ReferenceVirtualLM.from_name(
        model_name, llm_opts={"gpu_memory_utilization": 0.2, "dtype": "float16"}
    )


@pytest.fixture(scope="module")
def async_llm(model_name):
    return load_model_by_name(
        model_name,
        backend="vllm",
        llm_opts={"engine_opts": {"gpu_memory_utilization": 0.2, "dtype": "float16"}},
    )


@pytest.fixture(scope="module")
def transformer_llm(model_name):
    return load_model_by_name(
        model_name, backend="hf", llm_opts={"hf_opts": {"torch_dtype": torch.float16}}
    )


@pytest.fixture(scope="module")
def token_ids_list(async_llm):
    test_prompts = [
        "There might be something wrong",
        "It's probably this or that",
        "with the language model code",
        "It's probably this or that",
    ]
    tokenizer = async_llm.tokenizer
    token_ids_list = [tokenizer.encode(p) for p in test_prompts]
    return token_ids_list


@cuda_only
# @settings(deadline=None)
# @given(text=st.text(min_size=1, max_size=1000))
def test_next_token_logprobs(async_llm, reference_llm, token_ids_list):
    for token_ids in token_ids_list:
        have = asyncio.run(async_llm.next_token_logprobs(token_ids)).cpu().numpy()
        want = asyncio.run(reference_llm.next_token_logprobs(token_ids))
        assert compare(have, want).max_rel_err < 1e-3, token_ids


@cuda_only
def test_next_token_logprobs_sync(async_llm, reference_llm, token_ids_list):
    for token_ids in token_ids_list:
        have = async_llm.next_token_logprobs_sync(token_ids).cpu().numpy()
        want = asyncio.run(reference_llm.next_token_logprobs(token_ids))
        assert compare(have, want).max_rel_err < 1e-3, token_ids


@cuda_only
# @settings(deadline=None)
# @given(text_list=st.lists(st.text(min_size=1, max_size=1000), min_size=1, max_size=5))
def test_batch_next_token_logprobs(async_llm, reference_llm, token_ids_list):
    haves = (
        asyncio.run(async_llm.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
    )
    wants = asyncio.run(reference_llm.batch_next_token_logprobs(token_ids_list))
    for i, (have, want) in enumerate(zip(haves, wants)):
        assert compare(have, want).max_rel_err < 1e-3, token_ids_list[i]


@cuda_only
# @settings(deadline=None)
# @given(text_list=st.lists(st.text(min_size=1, max_size=1000), min_size=1, max_size=5))
def test_batch_next_token_logprobs_sync(async_llm, reference_llm, token_ids_list):
    # Test 1: Regular sync context
    haves = async_llm.batch_next_token_logprobs_sync(token_ids_list).cpu().numpy()
    wants = asyncio.run(reference_llm.batch_next_token_logprobs(token_ids_list))

    for have, want in zip(haves, wants):
        assert compare(have, want).max_rel_err < 1e-3, "Sync context"


@cuda_only
# @settings(deadline=None)
# @given(text_list=st.lists(st.text(min_size=1, max_size=1000), min_size=1, max_size=5))
def test_batch_next_token_logprobs_sync_in_async(
    async_llm, reference_llm, token_ids_list
):
    # Test 2: Sync function inside async context
    async def async_context():
        have_async = async_llm.batch_next_token_logprobs_sync(token_ids_list)
        return have_async.cpu().numpy()

    wants = asyncio.run(reference_llm.batch_next_token_logprobs(token_ids_list))
    haves = asyncio.run(async_context())

    for have, want in zip(haves, wants):
        assert compare(have, want).max_rel_err < 1e-3, "Sync in async context"


@cuda_only
def test_next_token_logprobs_agreement(transformer_llm, async_llm, token_ids_list):
    for token_ids in token_ids_list:
        have = transformer_llm.next_token_logprobs_uncached(token_ids).cpu().numpy()
        want = asyncio.run(async_llm.next_token_logprobs(token_ids)).cpu().numpy()
        comparison = compare(have, want)
        assert comparison.max_rel_err < 0.03, [
            "max_rel_err",
            comparison.max_rel_err,
            token_ids,
        ]
        assert comparison.pearson > 0.99, ["corr", comparison.pearson, token_ids]


@cuda_only
def test_batch_next_token_logprobs_agreement(
    transformer_llm, async_llm, token_ids_list
):
    haves = (
        asyncio.run(transformer_llm.batch_next_token_logprobs(token_ids_list))
        .cpu()
        .numpy()
    )
    wants = (
        asyncio.run(async_llm.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
    )
    for i, (have, want) in enumerate(zip(haves, wants)):
        comparison = compare(have, want)
        assert comparison.max_rel_err < 0.04, [
            "max_rel_err",
            comparison.max_rel_err,
            token_ids_list[i],
        ]
        assert comparison.pearson > 0.99, [
            "corr",
            comparison.pearson,
            token_ids_list[i],
        ]


@pytest.mark.asyncio
async def test_mock_async_llm():
    mock_async_llm = MockAsyncLM.from_name("gpt2")
    logprobs1 = await mock_async_llm.next_token_logprobs([0])
    logprobs2 = mock_async_llm.next_token_logprobs_sync([0])
    assert torch.allclose(logprobs1, logprobs2)
    mock_async_llm.clear_cache()  # no-op


def test_load_model_by_name_mock():
    load_model_by_name("gpt2", backend="mock")


def test_load_model_by_name_error():
    with pytest.raises(ValueError):
        load_model_by_name("gpt2", backend="invalid")


@cuda_only
def test_generate_agreement(async_llm, transformer_llm):
    prompt = async_llm.tokenizer.encode("Hello, world!")
    max_tokens = 10
    temperature = 0.01
    seed = 42
    eos_token_ids = [407]

    generated_token_ids_vllm = asyncio.run(
        async_llm.sample(
            prompt_token_ids=prompt,
            max_tokens=max_tokens,
            eos_token_ids=eos_token_ids,
            temperature=temperature,
            seed=seed,
        )
    )
    generated_token_ids_hf = asyncio.run(
        transformer_llm.sample(
            prompt_token_ids=prompt,
            max_tokens=max_tokens,
            eos_token_ids=eos_token_ids,
            temperature=temperature,
            seed=seed,
        )
    )

    assert generated_token_ids_vllm == generated_token_ids_hf


@cuda_only
def test_sample_seeded_vllm(async_llm):
    generated_token_ids = asyncio.run(
        async_llm.sample(
            prompt_token_ids=async_llm.tokenizer.encode("Hello,"),
            max_tokens=10,
            eos_token_ids=[611],
            temperature=0.01,
            seed=80808,
        )
    )
    assert (
        async_llm.tokenizer.decode(generated_token_ids)
        == " I'm sorry, but I'm not sure"
    )


@cuda_only
def test_batch_sample(async_llm):
    prompts = [
        async_llm.tokenizer.encode("Hello, world!"),
        async_llm.tokenizer.encode("An apple a day keeps the"),
    ]

    generated_token_ids_vllm = asyncio.run(
        async_llm.batch_sample(
            prompt_token_ids_list=prompts,
            max_tokens=10,
            eos_token_ids=[],
            temperature=0.01,
            seed=42,
        )
    )

    assert len(generated_token_ids_vllm) == len(prompts)
    assert len(generated_token_ids_vllm[0]) == 10
    assert len(generated_token_ids_vllm[1]) == 10


@pytest.mark.skip("This fails.")
def test_concurrent_logprobs_and_sample(async_llm):
    prompt = async_llm.tokenizer.encode("Hello, world!")

    async def logprobs_task():
        return await async_llm.next_token_logprobs(prompt)

    async def sample_task():
        return await async_llm.sample(
            prompt, max_tokens=3, eos_token_ids=[], temperature=1.0
        )

    async def both_tasks():
        return await asyncio.gather(logprobs_task(), sample_task())

    asyncio.run(both_tasks())


@cuda_only
@pytest.mark.asyncio
async def test_other(async_llm):
    async_llm_with_cache = AsyncVirtualLM(
        async_llm.async_llm_engine,
        cache_size=2,
    )

    logprobs1 = await async_llm_with_cache.next_token_logprobs([0])
    logprobs2 = await async_llm_with_cache.next_token_logprobs([1])
    assert len(async_llm_with_cache.cache) == 2

    logprobs1_post = await async_llm_with_cache.next_token_logprobs([0])
    logprobs2_post = await async_llm_with_cache.next_token_logprobs([1])
    assert torch.allclose(logprobs1, logprobs1_post)
    assert torch.allclose(logprobs2, logprobs2_post)

    # Check that we can clear the cache
    async_llm_with_cache.clear_cache()
    assert len(async_llm_with_cache.cache) == 0

    del async_llm_with_cache
