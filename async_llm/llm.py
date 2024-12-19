import torch
import asyncio
from contextlib import contextmanager
 
from vllm.utils import Counter
from vllm.inputs import TokensPrompt
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs

from async_llm.cache import OutputCache
from async_llm.sampler import DeferredSampler
from async_llm.vocabulary import decode_vocab


class AsyncLLM: 

    DEFAULT_SAMPLING_PARAMS = SamplingParams(
        max_tokens=1, n=1, logprobs=1, detokenize=False,
        stop=None, ignore_eos=True
    )

    def __init__(self, async_llm_engine, cache_size=0, cache_opts={}):
        self.async_llm_engine = async_llm_engine
        self.request_counter = Counter()
        self.custom_sampler = DeferredSampler()

        self.tokenizer = asyncio.run(async_llm_engine.get_tokenizer())
        self.byte_vocab, self.str_vocab = decode_vocab(self.tokenizer)

        self.cache = OutputCache(maxsize=cache_size, **cache_opts) if cache_size > 0 else None

    @classmethod
    def from_name(cls, model_name, engine_opts={}, **kwargs):
        """Create a AsyncLLM instance from a Hugging Face model name.
        
        This is a convenience method that handles the creation and configuration of the underlying vLLM engine.
        
        Args:
            model_name: Name of the model to load
            engine_opts: Additional options to pass to the vLLM engine
            **kwargs: Additional arguments passed to AsyncLLM constructor
            
        Returns:
            An AsyncLLM instance
        """
        default_engine_opts = {
            'enable_prefix_caching' : True,
            'disable_log_requests' : True
        }
        engine_opts = {**default_engine_opts, **engine_opts}
        engine_args = AsyncEngineArgs(
            model=model_name, 
            tokenizer=model_name, 
            **engine_opts
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        return cls(engine, **kwargs)

    async def next_token_logprobs(self, token_ids):
        """Request log probabilities of next token asynchronously with output caching.
        
        Args:
            token_ids_list (List[int]): A list of token IDs, representing the prompt to the language model.
                
        Returns:
            torch.Tensor: Normalized log probability tensor.
        """
        key = tuple(token_ids) 
        
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        
        result = await self._next_token_logprobs(key)
        
        if self.cache is not None:
            self.cache[key] = result
        
        return result

    async def _next_token_logprobs(self, token_ids):
        """Request log probabilities of next token asynchronously. 
        
        Args:
            token_ids_list (List[int]): A list of token IDs, representing the prompt to the language model.
                
        Returns:
            torch.Tensor: Normalized log probability tensor.
        """
        outputs = []
        with self.efficient_sampler():
            async for output in self.async_llm_engine.generate(
                prompt=TokensPrompt(prompt_token_ids=token_ids),
                sampling_params=self.DEFAULT_SAMPLING_PARAMS,
                request_id=str(next(self.request_counter))
            ):
                if output.finished:
                    outputs.append(output)

        return self._validate_outputs(outputs)

    async def batch_next_token_logprobs(self, token_ids_list):
        """Batch request log probabilities for multiple token sequences asynchronously.
        
        Args:
            token_ids_list (List[List[int]]): A list of token ID lists, each representing a prompt to the language model.
                
        Returns:
            torch.Tensor: A tensor of log probability tensors, one for each input sequence.
        """
        return torch.stack(await asyncio.gather(
            *[self.next_token_logprobs(token_ids) for token_ids in token_ids_list]
        ))

    def next_token_logprobs_sync(self, token_ids):
        """Request log probabilities of next token synchronously. Synchronous version of next_token_logprobs. 

        Args:
            token_ids_list (List[int]): A list of token IDs, representing the prompt to the language model.
                
        Returns:
            torch.Tensor: Normalized log probability tensor.
        """
        return asyncio.run(self.next_token_logprobs(token_ids))

    def _validate_outputs(self, outputs):
        """Validate and extract logprobs from a vLLM output."""
        assert len(outputs) == 1 # Single step decoding.
        output = outputs[0]
        assert len(output.outputs) == 1 # Single sequence.
        sequence = output.outputs[0]
        assert len(sequence.logprobs) == 1 
        return sequence.logprobs[0].logprobs

    @contextmanager
    def efficient_sampler(self):
        """Context manager that temporarily replaces the default vLLM sampler with an optimized one.
        
        This optimizes sampling by:
        1. Using a custom sampler optimized for next token log probability calculations
        2. Disabling async output processing in vLLM to prevent extra forward passes
        """
        # TODO: test with tensor_parallel_size > 1 and pipeline_parallel_size > 1
        model = self.async_llm_engine.engine.model_executor.driver_worker.model_runner.model 
        schedulers = self.async_llm_engine.engine.scheduler

        org_sampler = model.sampler
        org_async_procs = [scheduler._allow_async_output_proc for scheduler in schedulers]
        
        try:
            model.sampler = self.custom_sampler
            for scheduler in schedulers:
                scheduler._allow_async_output_proc = lambda x: False
            yield
        finally: 
            model.sampler = org_sampler
            for i, scheduler in enumerate(schedulers):
                scheduler._allow_async_output_proc = org_async_procs[i]

import numpy as np
from vllm import LLM

class ReferenceLLM:
    """ Reference implementation used for testing. Synchronous and significantly slower than AsyncLLM (~15x slower). """
    def __init__(self, llm):
        self.llm = llm
        self.byte_vocab, self.str_vocab = decode_vocab(llm.llm_engine.get_tokenizer())
        self.vocab_length = len(self.byte_vocab)
        self.llm.llm_engine.get_model_config().max_logprobs = self.vocab_length
        self.DEFAULT_SAMPLING_PARAMS = SamplingParams(
            max_tokens=1, n=1, logprobs=self.vocab_length, 
            detokenize=False, stop=None, ignore_eos=True
        )

        self.llm.llm_engine.log_stats = False

    @classmethod
    def from_name(cls, model_name, llm_opts={}):
        default_llm_opts = {
            'enable_prefix_caching' : True,
            'disable_log_stats' : True
        }
        llm_opts = {**default_llm_opts, **llm_opts}
        llm = LLM(
            model=model_name, 
            tokenizer=model_name, 
            **llm_opts
        )
        return cls(llm)

    def next_token_logprobs(self, token_ids):
        outputs = self.llm.generate(
            prompts=TokensPrompt(prompt_token_ids=token_ids), 
            sampling_params=self.DEFAULT_SAMPLING_PARAMS,
            use_tqdm=False
        )
        logprobs = np.array([
            outputs[0].outputs[0].logprobs[0][i].logprob
            for i in range(self.vocab_length)
        ])
        return logprobs

    def batch_next_token_logprobs(self, token_ids_list):
        prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in token_ids_list]
        outputs = self.llm.generate(
            prompts=prompts, sampling_params=self.DEFAULT_SAMPLING_PARAMS, use_tqdm=False
        )
        logprobs = np.array([
            [out.outputs[0].logprobs[0][i].logprob for i in range(self.vocab_length)] 
            for out in outputs
        ])
        return logprobs
