import torch
import pytest
import asyncio
from conftest import v1_capable, ReferenceVirtualLM
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name
import numpy as np


@pytest.fixture(scope="module")
def model_name():
    return "HuggingFaceTB/SmolLM-135M"


@pytest.fixture(scope="module")
def async_llm(model_name):
    return load_model_by_name(
        model_name,
        backend="vllm",
        llm_opts={
            "engine_opts": {
                "enable_lora": True,
                "dtype": "float16",
                "gpu_memory_utilization": 0.1,
                "max_model_len": 15,
            }
        },
    )


@pytest.fixture(scope="module")
def reference_llm(model_name):
    return ReferenceVirtualLM.from_name(
        model_name,
        llm_opts={
            "enable_lora": True,
            "dtype": "float16",
            "gpu_memory_utilization": 0.1,
            "max_model_len": 15,
        },
    )


@pytest.fixture(scope="module")
def lora_path():
    return "vxef/smol_lora_toy"


@pytest.fixture(scope="module")
def transformer_llm(model_name, lora_path):
    transformer_llm_base = load_model_by_name(
        model_name, backend="hf", llm_opts={"hf_opts": {"torch_dtype": torch.float16}}
    )
    transformer_llm_base.add_new_lora(lora_path, "lora_1")
    return transformer_llm_base


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


@v1_capable
def test_async_llm_only(async_llm):
    assert async_llm is not None


@v1_capable
def test_reference_llm_only(reference_llm):
    assert reference_llm is not None


def test_unknown_lora_error(transformer_llm):
    with pytest.raises(ValueError):
        transformer_llm.next_token_logprobs_uncached([0], lora_name="lora_2")


# Note: "lora_extra_vocab_size" is 256, so async has an increased vocab size
# "lora_extra_vocab_size" will be removed in vllm v0.12.0 (genlm-backend uses vllm v0.10.0)
# This does not happen with the reference llm since the vocab size is set using the hf tokenizer (decode)
# and then logprobs=vocab_length is set in SamplingParameters in vllm
@v1_capable
def test_next_token_logprobs(async_llm, reference_llm, token_ids_list, lora_path):
    async_llm.add_new_lora(lora_path)
    reference_llm.set_lora(lora_path)
    for token_ids in token_ids_list:
        logits_async = (
            asyncio.run(async_llm.next_token_logprobs(token_ids, lora_name="lora_1"))
            .float()
            .cpu()
            .numpy()
        )

        logits_ref = asyncio.run(reference_llm.next_token_logprobs(token_ids))

        async_vocab = logits_async.shape[0]
        ref_vocab = logits_ref.shape[0]

        assert async_vocab == ref_vocab + 256, [
            "Unexpected vocab mismatch. Async must have 256 more tokens.",
            async_vocab,
            ref_vocab,
        ]
        extra_logits = logits_async[-256:]
        assert np.all(np.isneginf(extra_logits)), (
            "Async extra logits are all -inf",
            extra_logits,
        )
        trimmed_async = logits_async[:ref_vocab]
        assert trimmed_async.shape == logits_ref.shape

        assert compare(trimmed_async, logits_ref).max_rel_err < 1e-2, token_ids
    reference_llm.clear_lora()


@v1_capable
def test_next_token_logprobs_sync(async_llm, reference_llm, token_ids_list, lora_path):
    async_llm.add_new_lora(lora_path)
    reference_llm.set_lora(lora_path)
    for token_ids in token_ids_list:
        logits_async = (
            async_llm.next_token_logprobs_sync(token_ids, lora_name="lora_1")
            .float()
            .cpu()
            .numpy()
        )
        logits_ref = asyncio.run(reference_llm.next_token_logprobs(token_ids))
        async_vocab = logits_async.shape[0]
        ref_vocab = logits_ref.shape[0]

        assert async_vocab == ref_vocab + 256, [
            "Unexpected vocab mismatch. Async must have 256 more tokens because lora_extra_vocab_size=256.",
            async_vocab,
            ref_vocab,
        ]
        extra_logits = logits_async[-256:]
        assert np.all(np.isneginf(extra_logits)), (
            "Async extra logits are all -inf",
            extra_logits,
        )
        trimmed_async = logits_async[:ref_vocab]
        assert trimmed_async.shape == logits_ref.shape

        assert compare(trimmed_async, logits_ref).max_rel_err < 1e-2, token_ids
    reference_llm.clear_lora()


@v1_capable
def test_batch_next_token_logprobs_sync(
    async_llm, reference_llm, token_ids_list, lora_path
):
    async_llm.add_new_lora(lora_path)
    reference_llm.set_lora(lora_path)
    logits_async = (
        async_llm.batch_next_token_logprobs_sync(token_ids_list, lora_name="lora_1")
        .float()
        .cpu()
        .numpy()
    )

    logits_ref = asyncio.run(reference_llm.batch_next_token_logprobs(token_ids_list))

    async_vocab = logits_async.shape[1]
    ref_vocab = logits_ref.shape[1]

    assert async_vocab == ref_vocab + 256, [
        "Unexpected vocab mismatch. Async must have 256 more tokens.",
        async_vocab,
        ref_vocab,
    ]

    for logits in logits_async:
        extra_logits = logits[-256:]
        assert np.all(np.isneginf(extra_logits)), (
            "Async extra logits are all -inf",
            extra_logits,
        )
    trimmed_async = logits_async[:, :ref_vocab]
    assert trimmed_async.shape == logits_ref.shape
    for i, (logit_async, logit_ref) in enumerate(zip(trimmed_async, logits_ref)):
        assert compare(logit_async, logit_ref).max_rel_err < 1e-2, token_ids_list[i]
    reference_llm.clear_lora()


@v1_capable
def test_batch_next_token_logprobs(async_llm, reference_llm, token_ids_list, lora_path):
    async_llm.add_new_lora(lora_path)
    reference_llm.set_lora(lora_path)
    logits_async = (
        asyncio.run(
            async_llm.lora_view("lora_1").batch_next_token_logprobs(token_ids_list)
        )
        .float()
        .cpu()
        .numpy()
    )

    logits_ref = asyncio.run(reference_llm.batch_next_token_logprobs(token_ids_list))

    async_vocab = logits_async.shape[1]
    ref_vocab = logits_ref.shape[1]

    assert async_vocab == ref_vocab + 256, [
        "Unexpected vocab mismatch. Async must have 256 more tokens.",
        async_vocab,
        ref_vocab,
    ]
    for logits in logits_async:
        extra_logits = logits[-256:]
        assert np.all(np.isneginf(extra_logits)), (
            "Async extra logits are all -inf",
            extra_logits,
        )
    trimmed_async = logits_async[:, :ref_vocab]
    assert trimmed_async.shape == logits_ref.shape
    for i, (logit_async, logit_ref) in enumerate(zip(trimmed_async, logits_ref)):
        assert compare(logit_async, logit_ref).max_rel_err < 1e-2, token_ids_list[i]
    reference_llm.clear_lora()


@v1_capable
def test_swapping_lora_requests(token_ids_list, async_llm, lora_path):
    """Interleaving base and adapter requests gives the same logits as running
    each adapter in its own contiguous block."""
    async_llm.add_new_lora(lora_path)

    def logprobs(token_ids, lora_name):
        return (
            asyncio.run(async_llm.next_token_logprobs(token_ids, lora_name=lora_name))
            .float()
            .cpu()
            .numpy()
        )

    logits_noswapped_nolora = [logprobs(ids, None) for ids in token_ids_list]
    logits_noswapped_lora = [logprobs(ids, "lora_1") for ids in token_ids_list]

    logits_swapped_nolora = []
    logits_swapped_lora = []
    for token_ids in token_ids_list:
        logits_swapped_nolora.append(logprobs(token_ids, None))
        logits_swapped_lora.append(logprobs(token_ids, "lora_1"))

    for i, token_ids in enumerate(token_ids_list):
        assert (
            compare(
                logits_noswapped_lora[i][:-256], logits_swapped_lora[i][:-256]
            ).max_rel_err
            < 1e-3
        ), token_ids
    for i, token_ids in enumerate(token_ids_list):
        assert (
            compare(
                logits_noswapped_nolora[i][:-256], logits_swapped_nolora[i][:-256]
            ).max_rel_err
            < 1e-3
        ), token_ids


@v1_capable
def test_reregistration(async_llm, token_ids_list, lora_pair):
    """Re-registering a name rebinds it to the new weights: fresh engine id and
    purged logprob cache, so the cached first read can't shadow the swap."""
    identity_path, shifted_path = lora_pair
    ids = token_ids_list[0]

    base = asyncio.run(async_llm.next_token_logprobs(ids)).float().cpu().numpy()
    async_llm.add_new_lora(identity_path, "reg")
    lp_identity = (
        asyncio.run(async_llm.next_token_logprobs(ids, lora_name="reg"))
        .float()
        .cpu()
        .numpy()
    )
    finite = np.isfinite(lp_identity) & np.isfinite(base)
    assert np.abs(lp_identity[finite] - base[finite]).max() < 1e-2

    async_llm.add_new_lora(shifted_path, "reg")
    lp_shifted = (
        asyncio.run(async_llm.next_token_logprobs(ids, lora_name="reg"))
        .float()
        .cpu()
        .numpy()
    )
    finite = np.isfinite(lp_shifted) & np.isfinite(lp_identity)
    assert np.abs(lp_shifted[finite] - lp_identity[finite]).max() > 1e-2

    async_llm.remove_lora("reg")
    with pytest.raises(KeyError):
        asyncio.run(async_llm.next_token_logprobs(ids, lora_name="reg"))


@v1_capable
def test_next_token_logprobs_agreement(
    transformer_llm, async_llm, token_ids_list, lora_path
):
    async_llm.add_new_lora(lora_path)
    for token_ids in token_ids_list:
        have = (
            transformer_llm.next_token_logprobs_uncached(token_ids, lora_name="lora_1")
            .cpu()
            .numpy()
        )
        want = (
            asyncio.run(async_llm.next_token_logprobs(token_ids, lora_name="lora_1"))
            .cpu()
            .numpy()
        )

        hf_vocab = have.shape[0]

        want_trimmed = want[:hf_vocab]
        comparison = compare(have, want_trimmed)
        assert comparison.max_rel_err < 0.03, [
            "max_rel_err",
            comparison.max_rel_err,
            token_ids,
        ]
        assert comparison.pearson > 0.99, ["corr", comparison.pearson, token_ids]


@v1_capable
def test_batch_next_token_logprobs_agreement(
    transformer_llm, async_llm, token_ids_list, lora_path
):
    async_llm.add_new_lora(lora_path)
    haves = (
        asyncio.run(
            transformer_llm.batch_next_token_logprobs(
                token_ids_list, lora_name="lora_1"
            )
        )
        .cpu()
        .numpy()
    )
    wants = (
        asyncio.run(
            async_llm.lora_view("lora_1").batch_next_token_logprobs(token_ids_list)
        )
        .cpu()
        .numpy()
    )
    for i, (have, want) in enumerate(zip(haves, wants)):
        hf_vocab = have.shape[0]
        want_trimmed = want[:hf_vocab]
        comparison = compare(have, want_trimmed)
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
