import pytest
import torch
import sys

try:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

import numpy as np
from genlm.backend.tokenization import decode_vocab
from contextlib import contextmanager

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="test requires CUDA"
)


@pytest.fixture(autouse=True, scope="function")
def cleanup_modules():
    yield
    # After each test, remove LLM-related modules
    # This is needed because with namespace packages, modules stay loaded in sys.modules between tests.
    # These modules can hold references to GPU memory through PyTorch's caching allocator, CUDA streams,
    # and other static resources. By removing the modules, we force them to be reloaded for each test,
    # ensuring GPU memory is properly released.
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("genlm.backend.llm"):
            sys.modules.pop(mod_name, None)

    # Also ensure GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def assert_roundtrip_bytes(test_case, tokenizer, byte_vocab):
    """Assert that encoding and decoding a test case using byte vocabulary matches the tokenizer's output.

    Args:
        test_case (str): String to test encoding/decoding roundtrip
        tokenizer: Hugging Face tokenizer instance
        byte_vocab (list): List of byte representations of tokens

    Raises:
        AssertionError: If the roundtrip result doesn't match tokenizer's direct decoding
    """
    return assert_roundtrip(test_case, tokenizer, byte_vocab, vocab_type="byte")


def assert_roundtrip_strs(test_case, tokenizer, str_vocab):
    """Assert that encoding and decoding a test case using string vocabulary matches the tokenizer's output.

    Args:
        test_case (str): String to test encoding/decoding roundtrip
        tokenizer: Hugging Face tokenizer instance
        str_vocab (list): List of string representations of tokens

    Raises:
        AssertionError: If the roundtrip result doesn't match tokenizer's direct decoding
    """
    return assert_roundtrip(test_case, tokenizer, str_vocab, vocab_type="str")


def assert_roundtrip(test_case, tokenizer, vocab, vocab_type):
    """Assert that encoding and decoding a test case matches the tokenizer's output.

    A unified function that handles both string and byte vocabularies.

    Args:
        test_case (str): String to test encoding/decoding roundtrip
        tokenizer: Hugging Face tokenizer instance
        vocab (list): List of token representations (either strings or bytes)
        vocab_type (str): Type of vocabulary - either 'str' or 'byte'

    Raises:
        AssertionError: If the roundtrip result doesn't match tokenizer's direct decoding
        ValueError: If vocab_type is not 'str' or 'byte'
    """
    with turn_off_space_cleaning(tokenizer):
        encd = tokenizer.encode(test_case)

        if vocab_type == "str":
            have = "".join([vocab[i] for i in encd])
        elif vocab_type == "byte":
            have = b"".join([vocab[i] for i in encd]).decode("utf-8")
        else:
            raise ValueError(
                f"Invalid vocab_type: {vocab_type}. Must be 'str' or 'byte'."
            )

        want = tokenizer.decode(encd)

        if have != want:
            pos = next(
                (i for i in range(min(len(have), len(want))) if have[i] != want[i]),
                min(len(have), len(want)),
            )
            context = 20

            error_msg = (
                f"\nRoundtrip assertion failed for {vocab_type} vocabulary:"
                f"\nMismatch at position {pos}"
                f"\nHave: ...{repr(have[max(0, pos - context) : pos + context])}..."
                f"\nWant: ...{repr(want[max(0, pos - context) : pos + context])}..."
            )

            raise AssertionError(error_msg)


@contextmanager
def turn_off_space_cleaning(tokenizer):
    original_value = tokenizer.clean_up_tokenization_spaces
    try:
        tokenizer.clean_up_tokenization_spaces = False
        yield
    finally:
        tokenizer.clean_up_tokenization_spaces = original_value


class ReferenceVirtualLM:
    """Reference vLLM implementation used for testing. Synchronous and significantly slower than AsyncVirtualLM (~15x slower)."""

    def __init__(self, llm):
        self.llm = llm
        self.tokenizer = llm.llm_engine.get_tokenizer()
        self.byte_vocab, self.str_vocab = decode_vocab(self.tokenizer)
        self.vocab_length = len(self.byte_vocab)
        self.llm.llm_engine.get_model_config().max_logprobs = self.vocab_length
        self.DEFAULT_SAMPLING_PARAMS = SamplingParams(
            max_tokens=1,
            n=1,
            logprobs=self.vocab_length,
            detokenize=False,
            stop=None,
            ignore_eos=True,
        )

        self.llm.llm_engine.log_stats = False

    @classmethod
    def from_name(cls, model_name, llm_opts=None):
        if not HAS_VLLM:
            raise ImportError("vLLM not installed.")
        llm_opts = {
            "enable_prefix_caching": True,
            "disable_log_stats": True,
            "dtype": "float16",
            **(llm_opts or {}),
        }
        llm = LLM(model=model_name, tokenizer=model_name, **llm_opts)
        return cls(llm)

    def next_token_logprobs_sync(self, token_ids):
        outputs = self.llm.generate(
            prompts=TokensPrompt(prompt_token_ids=token_ids),
            sampling_params=self.DEFAULT_SAMPLING_PARAMS,
            use_tqdm=False,
        )
        logprobs = np.array(
            [
                outputs[0].outputs[0].logprobs[0][i].logprob
                for i in range(self.vocab_length)
            ]
        )
        return logprobs

    async def next_token_logprobs(self, token_ids):
        # Note: async method only to support protocol, actual implementation is synchronous
        return self.next_token_logprobs_sync(token_ids)

    async def batch_next_token_logprobs(self, token_ids_list):
        # Note: async method only to support protocol, actual implementation is synchronous
        prompts = [
            TokensPrompt(prompt_token_ids=token_ids) for token_ids in token_ids_list
        ]
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=self.DEFAULT_SAMPLING_PARAMS,
            use_tqdm=False,
        )
        logprobs = np.array(
            [
                [
                    out.outputs[0].logprobs[0][i].logprob
                    for i in range(self.vocab_length)
                ]
                for out in outputs
            ]
        )
        return logprobs

    def __del__(self):
        if llm_engine := getattr(self.llm, "llm_engine"):
            if executor := getattr(llm_engine, "model_executor"):
                destroy_model_parallel()
                destroy_distributed_environment()
                del executor
