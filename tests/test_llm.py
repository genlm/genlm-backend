import torch
import pytest
import asyncio
from conftest import cuda_only, ReferenceVirtualLM
from arsenal.maths import compare
from genlm.backend.llm import AsyncVirtualLM, AsyncTransformer, MockAsyncLM
from genlm.backend.llm.vllm import LazyLogprobDict

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
    return AsyncVirtualLM.from_name(
        model_name, engine_opts={"gpu_memory_utilization": 0.2, "dtype": "float16"}
    )


@pytest.fixture(scope="module")
def transformer_llm(model_name):
    return AsyncTransformer.from_name(
        model_name, hf_opts={"torch_dtype": torch.float16}
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
        assert compare(have, want).max_rel_err < 1e-5, token_ids


@cuda_only
def test_next_token_logprobs_sync(async_llm, reference_llm, token_ids_list):
    for token_ids in token_ids_list:
        have = async_llm.next_token_logprobs_sync(token_ids).cpu().numpy()
        want = asyncio.run(reference_llm.next_token_logprobs(token_ids))
        assert compare(have, want).max_rel_err < 1e-5, token_ids


@cuda_only
# @settings(deadline=None)
# @given(text_list=st.lists(st.text(min_size=1, max_size=1000), min_size=1, max_size=5))
def test_batch_next_token_logprobs(async_llm, reference_llm, token_ids_list):
    haves = (
        asyncio.run(async_llm.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
    )
    wants = asyncio.run(reference_llm.batch_next_token_logprobs(token_ids_list))
    for i, (have, want) in enumerate(zip(haves, wants)):
        assert compare(have, want).max_rel_err < 1e-5, token_ids_list[i]


@cuda_only
# @settings(deadline=None)
# @given(text_list=st.lists(st.text(min_size=1, max_size=1000), min_size=1, max_size=5))
def test_batch_next_token_logprobs_sync(async_llm, reference_llm, token_ids_list):
    # Test 1: Regular sync context
    haves = async_llm.batch_next_token_logprobs_sync(token_ids_list).cpu().numpy()
    wants = asyncio.run(reference_llm.batch_next_token_logprobs(token_ids_list))

    for have, want in zip(haves, wants):
        assert compare(have, want).max_rel_err < 1e-5, "Sync context"


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
        assert compare(have, want).max_rel_err < 1e-5, "Sync in async context"


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


@cuda_only
@pytest.mark.asyncio
async def test_other():
    # Check that we warn when we set enable_chunked_prefill to True
    with pytest.warns(UserWarning):
        async_llm = AsyncVirtualLM.from_name(
            "gpt2",
            engine_opts={
                "enable_chunked_prefill": True,
                "gpu_memory_utilization": 0.2,
                "dtype": "float16",
            },
            cache_size=2,
        )

    logprobs1 = await async_llm.next_token_logprobs([0])
    logprobs2 = await async_llm.next_token_logprobs([1])
    assert len(async_llm.cache) == 2

    logprobs1_post = await async_llm.next_token_logprobs([0])
    logprobs2_post = await async_llm.next_token_logprobs([1])
    assert torch.allclose(logprobs1, logprobs1_post)
    assert torch.allclose(logprobs2, logprobs2_post)

    # Check that we can clear the cache
    async_llm.clear_cache()
    assert len(async_llm.cache) == 0

    del async_llm


def test_lazy_logprob_dict():
    try:
        from vllm.sequence import Logprob
    except ImportError:
        pytest.skip("vLLM is not installed")

    logprobs = torch.tensor(
        [0.1, 0.2, 0.3],
        dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    lazy_logprob_dict = LazyLogprobDict(logprobs)
    assert lazy_logprob_dict[0] == Logprob(0.1)
    assert lazy_logprob_dict[1] == Logprob(0.2)
    assert lazy_logprob_dict[2] == Logprob(0.3)

    with pytest.raises(KeyError):
        lazy_logprob_dict[3]

    assert len(lazy_logprob_dict) == 3
    assert list(lazy_logprob_dict.keys()) == [0, 1, 2]
    assert list(lazy_logprob_dict.values()) == [
        Logprob(0.1),
        Logprob(0.2),
        Logprob(0.3),
    ]
    assert list(lazy_logprob_dict.items()) == [
        (0, Logprob(0.1)),
        (1, Logprob(0.2)),
        (2, Logprob(0.3)),
    ]
    assert lazy_logprob_dict.get(0) == Logprob(0.1)
    assert lazy_logprob_dict.get(3) is None
    assert lazy_logprob_dict.get(3, 0.4) == 0.4


@pytest.mark.asyncio
async def test_mock_async_llm():
    mock_async_llm = MockAsyncLM.from_name("gpt2")
    logprobs1 = await mock_async_llm.next_token_logprobs([0])
    logprobs2 = mock_async_llm.next_token_logprobs_sync([0])
    assert torch.allclose(logprobs1, logprobs2)
    mock_async_llm.clear_cache()  # no-op
