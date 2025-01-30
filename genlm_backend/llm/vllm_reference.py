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

from genlm_backend.tokenization import decode_vocab


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
