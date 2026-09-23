import pytest
import asyncio
import numpy as np
import torch
from conftest import cuda_only, logprobs, batch_logprobs, assert_rows_close
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name


@pytest.fixture(scope="module")
def model_name():
    return "HuggingFaceTB/SmolLM-135M"


@pytest.fixture(scope="module")
def merged_path():
    return "vxef/smol_merged_toy"


@pytest.fixture(scope="module")
def lora_path():
    return "vxef/smol_lora_toy"


@pytest.fixture(scope="module")
def transformer_merged_llm(merged_path):
    return load_model_by_name(
        merged_path, backend="hf", llm_opts={"hf_opts": {"torch_dtype": torch.float32}}
    )


@pytest.fixture(scope="module")
def transformer_llm(model_name):
    return load_model_by_name(
        model_name, backend="hf", llm_opts={"hf_opts": {"torch_dtype": torch.float32}}
    )


@pytest.fixture(scope="module")
def transformer_llm_nolora(model_name):
    return load_model_by_name(
        model_name, backend="hf", llm_opts={"hf_opts": {"torch_dtype": torch.float32}}
    )


@pytest.fixture(scope="module", autouse=True)
def load_lora(transformer_llm, lora_path):
    transformer_llm.add_new_lora(lora_path, LORA_NAME)


@pytest.fixture(scope="module")
def token_ids_list(transformer_llm):
    test_prompts = [
        "There might be something wrong",
        "with the language model code",
        "It's probably this or that",
        "with the language model code",
    ]
    return [transformer_llm.tokenizer.encode(p) for p in test_prompts]


def test_unknown_lora_error(transformer_llm):
    with pytest.raises(ValueError):
        transformer_llm.next_token_logprobs_uncached([0], lora_name="lora_2")
    # The auto-batched path must fail the future, not leave the caller hung.
    with pytest.raises(ValueError):
        asyncio.run(transformer_llm.next_token_logprobs([0], lora_name="lora_2"))


def test_reregistration(transformer_llm, token_ids_list, lora_pair):
    """Re-registering a name rebinds it to the new weights and purges its cache
    trie — the cached sync path must not serve the old adapter's logprobs."""
    identity_path, shifted_path = lora_pair
    ids = token_ids_list[0]

    base = transformer_llm.next_token_logprobs_sync(ids).cpu().numpy()
    transformer_llm.add_new_lora(identity_path, "reg")
    lp_identity = (
        transformer_llm.next_token_logprobs_sync(ids, lora_name="reg").cpu().numpy()
    )
    assert np.abs(lp_identity - base).max() < 1e-3

    transformer_llm.add_new_lora(shifted_path, "reg")
    lp_shifted = (
        transformer_llm.next_token_logprobs_sync(ids, lora_name="reg").cpu().numpy()
    )
    assert np.abs(lp_shifted - lp_identity).max() > 1e-2

    transformer_llm.remove_lora("reg")
    with pytest.raises(ValueError):
        transformer_llm.next_token_logprobs_sync(ids, lora_name="reg")


LORA_NAME = "lora_1"


def _row(llm, token_ids, entry, **kw):
    return logprobs(llm, token_ids, entry, **kw).cpu().numpy()


def _rows(llm, token_ids_list, entry, **kw):
    return [_row(llm, ids, entry, **kw) for ids in token_ids_list]


@cuda_only
@pytest.mark.parametrize("entry", ["uncached", "async", "sync"])
def test_next_token_logprobs_lora(
    transformer_llm, transformer_merged_llm, token_ids_list, entry
):
    """An adapter applied per request matches a model with those weights merged in."""
    assert_rows_close(
        _rows(transformer_llm, token_ids_list, entry, lora_name=LORA_NAME),
        _rows(transformer_merged_llm, token_ids_list, entry),
        token_ids_list,
        rel=1e-3,
    )


@cuda_only
@pytest.mark.parametrize("entry", ["async", "sync"])
def test_batch_token_logprobs_lora(
    transformer_llm, transformer_merged_llm, token_ids_list, entry
):
    assert_rows_close(
        batch_logprobs(
            transformer_llm, token_ids_list, entry, lora_name=LORA_NAME
        ).cpu(),
        batch_logprobs(transformer_merged_llm, token_ids_list, entry).cpu(),
        token_ids_list,
        rel=1e-3,
    )


@cuda_only
@pytest.mark.parametrize("entry", ["uncached", "async", "sync"])
def test_adapter_swap(transformer_llm, token_ids_list, transformer_llm_nolora, entry):
    """Interleaving base and adapter requests on one model matches a dedicated
    base model and a contiguous adapter run."""
    lora_contiguous = _rows(transformer_llm, token_ids_list, entry, lora_name=LORA_NAME)
    base_contiguous = _rows(transformer_llm_nolora, token_ids_list, entry)

    lora_swapped, base_swapped = [], []
    for token_ids in token_ids_list:
        lora_swapped.append(
            _row(transformer_llm, token_ids, entry, lora_name=LORA_NAME)
        )
        base_swapped.append(_row(transformer_llm, token_ids, entry))

    assert_rows_close(lora_contiguous, lora_swapped, token_ids_list, rel=1e-3)
    assert_rows_close(base_contiguous, base_swapped, token_ids_list, rel=1e-3)


@cuda_only
def test_adapter_swap_mixed_batch(
    transformer_llm, token_ids_list, transformer_llm_nolora
):
    """One auto-batched dispatch containing BOTH base and adapter queries routes
    each query through its own adapter."""
    transformer_llm.clear_cache()

    async def mixed(token_ids_list):
        lora = asyncio.gather(
            *[
                transformer_llm.next_token_logprobs(t, lora_name=LORA_NAME)
                for t in token_ids_list
            ]
        )
        base = asyncio.gather(
            *[transformer_llm.next_token_logprobs(t) for t in token_ids_list]
        )
        return await lora, await base

    lora_logprobs, base_logprobs = asyncio.run(mixed(token_ids_list))

    lora_reference = (
        asyncio.run(
            transformer_llm.batch_next_token_logprobs(
                token_ids_list, lora_name=LORA_NAME
            )
        )
        .cpu()
        .numpy()
    )
    base_reference = (
        asyncio.run(transformer_llm_nolora.batch_next_token_logprobs(token_ids_list))
        .cpu()
        .numpy()
    )

    for i, token_ids in enumerate(token_ids_list):
        assert (
            compare(lora_logprobs[i].cpu().numpy(), lora_reference[i]).max_rel_err
            < 1e-3
        ), token_ids
        assert (
            compare(base_logprobs[i].cpu().numpy(), base_reference[i]).max_rel_err
            < 1e-3
        ), token_ids
