import torch
import pytest
import asyncio
from conftest import cuda_only, ReferenceVirtualLM
from arsenal.maths import compare
from genlm.backend.llm import load_model_by_name
import numpy as np

@pytest.fixture(scope="module")
def model_name():
    return "HuggingFaceTB/SmolLM-135M" 

@pytest.fixture(scope="module", params=[False, True], ids=["nolora", "lora"])
def enable_lora(request):
    return request.param

@pytest.fixture(scope="module")
def async_llm(model_name, enable_lora):
    return load_model_by_name(
        model_name,
        backend="vllm",
        llm_opts={"engine_opts": {"enable_lora": enable_lora, "dtype": "float16", "gpu_memory_utilization" : 0.1, "max_model_len":15}},
    )

@pytest.fixture(scope="module")
def reference_llm(model_name, enable_lora):
    return ReferenceVirtualLM.from_name(
        model_name, llm_opts={"enable_lora": enable_lora, "dtype": "float16","gpu_memory_utilization" : 0.1, "max_model_len":15}
    )

@pytest.fixture(scope="module")
def transformer_llm_base(model_name):
    return load_model_by_name(
        model_name, backend="hf", llm_opts={"hf_opts": {"torch_dtype": torch.float16}}
    )

@pytest.fixture(scope="module")
def lora_path():
    return "vxef/smol_lora_toy"

@pytest.fixture(scope="module")
def transformer_llm(transformer_llm_base, lora_path):
    transformer_llm_base.load_lora(lora_path, 'lora_1')
    transformer_llm_base.set_lora(lora_name='lora_1')
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

@cuda_only
def test_async_llm_only(async_llm, enable_lora):
    assert async_llm is not None

@cuda_only
def test_reference_llm_only(reference_llm, enable_lora):
    assert reference_llm is not None

# Note: "lora_extra_vocab_size" is 256, so async has an increased vocab size
# "lora_extra_vocab_size" will be removed in vllm v0.12.0 (genlm-backend uses vllm v0.10.0)
# This does not happen with the reference llm since the vocab size is set using the hf tokenizer (decode)
# and then logprobs=vocab_length is set in SamplingParameters in vllm
@cuda_only
def test_next_token_logprobs(async_llm, reference_llm, token_ids_list, enable_lora, lora_path):
    if enable_lora:
        async_llm.set_lora(lora_path)
        reference_llm.set_lora(lora_path)
        for token_ids in token_ids_list:
            logits_async = asyncio.run(async_llm.next_token_logprobs(token_ids)).float().cpu().numpy()

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
    else:
        async_llm.clear_lora()
        reference_llm.clear_lora()
        for token_ids in token_ids_list:
            have = asyncio.run(async_llm.next_token_logprobs(token_ids)).cpu().numpy()
            want = asyncio.run(reference_llm.next_token_logprobs(token_ids))
            assert compare(have, want).max_rel_err < 1e-3, token_ids

@cuda_only
def test_next_token_logprobs_sync(async_llm, reference_llm, token_ids_list, enable_lora,lora_path):
    if enable_lora:
        async_llm.set_lora(lora_path)
        reference_llm.set_lora(lora_path)
        for token_ids in token_ids_list:
            logits_async = async_llm.next_token_logprobs_sync(token_ids).float().cpu().numpy()

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
    else:
        async_llm.clear_lora()
        reference_llm.clear_lora()
        for token_ids in token_ids_list:
            have = async_llm.next_token_logprobs_sync(token_ids).cpu().numpy()
            want = asyncio.run(reference_llm.next_token_logprobs(token_ids))
            assert compare(have, want).max_rel_err < 1e-3, token_ids

@cuda_only
def test_batch_next_token_logprobs_sync(async_llm, reference_llm, token_ids_list, enable_lora,lora_path):
    if enable_lora:
        async_llm.set_lora(lora_path)
        reference_llm.set_lora(lora_path)
        logits_async = async_llm.batch_next_token_logprobs_sync(token_ids_list).float().cpu().numpy()

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
        trimmed_async = logits_async[:,:ref_vocab]
        assert trimmed_async.shape == logits_ref.shape
        for i, (logit_async, logit_ref) in enumerate(zip(trimmed_async, logits_ref)):
            assert compare(logit_async, logit_ref).max_rel_err < 1e-2, token_ids_list[i]
    else:
        async_llm.clear_lora()
        reference_llm.clear_lora()
        haves = async_llm.batch_next_token_logprobs_sync(token_ids_list).cpu().numpy()
        wants = asyncio.run(reference_llm.batch_next_token_logprobs(token_ids_list))
        for i, (have, want) in enumerate(zip(haves, wants)):
            assert compare(have, want).max_rel_err < 1e-2, token_ids_list[i]

@cuda_only
def test_batch_next_token_logprobs(async_llm, reference_llm, token_ids_list, enable_lora,lora_path):
    if enable_lora:
        async_llm.set_lora(lora_path)
        reference_llm.set_lora(lora_path)
        logits_async = (
            asyncio.run(async_llm.batch_next_token_logprobs(token_ids_list)).float().cpu().numpy()
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
        trimmed_async = logits_async[:,:ref_vocab]
        assert trimmed_async.shape == logits_ref.shape
        for i, (logit_async, logit_ref) in enumerate(zip(trimmed_async, logits_ref)):
            assert compare(logit_async, logit_ref).max_rel_err < 1e-2, token_ids_list[i]
    else:
        async_llm.clear_lora()
        reference_llm.clear_lora()
        haves = logits_async = (
            asyncio.run(async_llm.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
        )
        wants = asyncio.run(reference_llm.batch_next_token_logprobs(token_ids_list))
        for i, (have, want) in enumerate(zip(haves, wants)):
            assert compare(have, want).max_rel_err < 1e-3, token_ids_list[i]

@cuda_only
def test_swapping_lora_requests(enable_lora, token_ids_list, async_llm,lora_path):
    if enable_lora:
        async_llm.clear_lora()
        logits_noswapped_nolora = []
        logits_noswapped_lora = []
        for token_ids in token_ids_list:
            logits_noswapped_nolora.append(asyncio.run(async_llm.next_token_logprobs(token_ids)).float().cpu().numpy())
        async_llm.set_lora(lora_path)
        for token_ids in token_ids_list:
            logits_noswapped_lora.append(asyncio.run(async_llm.next_token_logprobs(token_ids)).float().cpu().numpy())
        
        logits_swapped_nolora = []
        logits_swapped_lora = []
        for token_ids in token_ids_list:
            async_llm.clear_lora()
            logits_swapped_nolora.append(asyncio.run(async_llm.next_token_logprobs(token_ids)).float().cpu().numpy())
            async_llm.set_lora(lora_path)
            logits_swapped_lora.append(asyncio.run(async_llm.next_token_logprobs(token_ids)).float().cpu().numpy())
        
        for i, token_ids in enumerate(token_ids_list):
            assert compare(logits_noswapped_lora[i][:-256], logits_swapped_lora[i][:-256]).max_rel_err < 1e-3, token_ids
        for i, token_ids in enumerate(token_ids_list):
            assert compare(logits_noswapped_nolora[i][:-256], logits_swapped_nolora[i][:-256]).max_rel_err < 1e-3, token_ids
    else:
        pytest.skip("Only run when LoRA is enabled")


@cuda_only
def test_async_llm_lora_vs_nolora_enable_no_request(model_name, token_ids_list, enable_lora):
    if enable_lora:
        pytest.skip("Only run when LoRA is not enabled")
    else:
        async_nolora = load_model_by_name(
            model_name,
            backend="vllm",
            llm_opts={"engine_opts": {"enable_lora": False, "dtype": "float16", "gpu_memory_utilization" : 0.1, "max_model_len":15}},
        )
        
        async_lora = load_model_by_name(
            model_name,
            backend="vllm",
            llm_opts={"engine_opts": {"enable_lora": True, "dtype": "float16", "gpu_memory_utilization" : 0.1, "max_model_len":15}})

        for token_ids in token_ids_list:
            logits_nolora = asyncio.run(async_nolora.next_token_logprobs(token_ids)).cpu().numpy()
            logits_lora = asyncio.run(async_lora.next_token_logprobs(token_ids)).cpu().numpy()

            nolora_vocab = logits_nolora.shape[0]
            lora_vocab = logits_lora.shape[0]
        
            assert lora_vocab == nolora_vocab + 256, [
                "Unexpected vocab mismatch. Lora enabled must have 256 more tokens.",
                nolora_vocab,
                lora_vocab,
            ]
            extra_logits = logits_lora[-256:]
            assert np.all(np.isneginf(extra_logits)), (
                "Lora enabled extra logits are all -inf",
                extra_logits,
            )
            trimmed_lora = logits_lora[:nolora_vocab]
            assert trimmed_lora.shape == logits_nolora.shape
            assert compare(logits_nolora, trimmed_lora).max_rel_err < 1e-3, token_ids

       
        for token_ids in token_ids_list:
            logits_nolora = async_nolora.next_token_logprobs_sync(token_ids).cpu().numpy()
            logits_lora = async_lora.next_token_logprobs_sync(token_ids).cpu().numpy()

            nolora_vocab = logits_nolora.shape[0]
            lora_vocab = logits_lora.shape[0]
        
            assert lora_vocab == nolora_vocab + 256, [
                "Unexpected vocab mismatch. Lora enabled must have 256 more tokens.",
                nolora_vocab,
                lora_vocab,
            ]
            extra_logits = logits_lora[-256:]
            assert np.all(np.isneginf(extra_logits)), (
                "Lora enabled extra logits are all -inf",
                extra_logits,
            )
            trimmed_lora = logits_lora[:nolora_vocab]
            assert trimmed_lora.shape == logits_nolora.shape
            assert compare(logits_nolora, trimmed_lora).max_rel_err < 1e-3, token_ids

        logits_nolora = async_nolora.batch_next_token_logprobs_sync(token_ids_list).cpu().numpy()
        logits_lora = async_lora.batch_next_token_logprobs_sync(token_ids_list).cpu().numpy()

        nolora_vocab = logits_nolora.shape[1]
        lora_vocab = logits_lora.shape[1]
        
        assert lora_vocab == nolora_vocab + 256, [
            "Unexpected vocab mismatch. Async must have 256 more tokens.",
            nolora_vocab,
            lora_vocab,
        ]
    
        for logits in logits_lora:
            extra_logits = logits[-256:]
            assert np.all(np.isneginf(extra_logits)), (
                "Lora enabled extra logits are all -inf",
                extra_logits,
            )
        trimmed_lora = logits_lora[:,:nolora_vocab]
        assert trimmed_lora.shape == logits_nolora.shape
        for i, (logit_lora, logit_nolora) in enumerate(zip(trimmed_lora, logits_nolora)):
            assert compare(logit_lora, logit_nolora).max_rel_err < 1e-3, token_ids_list[i]
        
        logits_lora = (
            asyncio.run(async_lora.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
        )
        logits_nolora = (
            asyncio.run(async_nolora.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
        )

        lora_vocab = logits_lora.shape[1]
        nolora_vocab = logits_nolora.shape[1]
        
        assert lora_vocab == nolora_vocab + 256, [
            "Unexpected vocab mismatch. Async must have 256 more tokens.",
            lora_vocab,
            nolora_vocab,
        ]
        for logits in logits_lora:
            extra_logits = logits[-256:]
            assert np.all(np.isneginf(extra_logits)), (
                "Lora extra logits are all -inf",
                extra_logits,
            )
        trimmed_lora = logits_lora[:,:nolora_vocab]
        assert trimmed_lora.shape == logits_nolora.shape
        for i, (logit_lora, logit_nolora) in enumerate(zip(trimmed_lora, logits_nolora)):
            assert compare(logit_lora, logit_nolora).max_rel_err < 1e-3, token_ids_list[i]
        

@cuda_only
def test_next_token_logprobs_agreement(transformer_llm, async_llm, token_ids_list, enable_lora,lora_path):
    if enable_lora:
        async_llm.set_lora(lora_path)
        for token_ids in token_ids_list:
            have = transformer_llm.next_token_logprobs_uncached(token_ids).cpu().numpy()
            want = asyncio.run(async_llm.next_token_logprobs(token_ids)).cpu().numpy()
            
            hf_vocab = have.shape[0]
            
            want_trimmed = want[:hf_vocab]
            comparison = compare(have, want_trimmed)
            assert comparison.max_rel_err < 0.03, [
                "max_rel_err",
                comparison.max_rel_err,
                token_ids,
            ]
            assert comparison.pearson > 0.99, ["corr", comparison.pearson, token_ids]
    else:
        pytest.skip("Only run when LoRA is enabled")

@cuda_only
def test_batch_next_token_logprobs_agreement(transformer_llm, async_llm, token_ids_list, enable_lora,lora_path):
    if enable_lora:
        async_llm.set_lora(lora_path)
        haves = (
            asyncio.run(transformer_llm.batch_next_token_logprobs(token_ids_list))
            .cpu()
            .numpy()
        )
        wants = (
            asyncio.run(async_llm.batch_next_token_logprobs(token_ids_list)).cpu().numpy()
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
    else:
        pytest.skip("Only run when LoRA is enabled")
