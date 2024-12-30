import torch
import asyncio
import warnings
import numpy as np
import concurrent.futures
from collections import OrderedDict
from contextlib import contextmanager

try:
    from vllm import LLM, AsyncLLMEngine, SamplingParams, AsyncEngineArgs
    from vllm.utils import Counter
    from vllm.inputs import TokensPrompt
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )
    from vllm.model_executor.layers.sampler import SamplerOutput, Sampler
    from vllm.sequence import SequenceOutput, CompletionSequenceGroupOutput, Logprob
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    warnings.warn("vLLM not installed. Run 'pip install vllm' for faster serving.")

from genlm_backend.llm.base import AsyncLM
from genlm_backend.cache import OutputCache

class AsyncVirtualLM(AsyncLM): 

    def __init__(self, async_llm_engine, eos_token=None, cache_size=0, cache_opts={}):
        """Initialize an AsyncVirtualLM instance.

        Args:
            async_llm_engine (AsyncLLMEngine): The async vLLM engine instance.
            eos_token (str, optional): End of sequence token. If not provided, uses tokenizer's eos_token.
            cache_size (int, optional): Maximum size of the output cache. If 0, caching is disabled. Defaults to 0.
            cache_opts (dict, optional): Additional options to pass to the OutputCache constructor. Defaults to {}.

        Note:
            The cache stores the log probabilities for previously seen token sequences to avoid redundant requests. 
        """
        self.async_llm_engine = async_llm_engine
        self.tokenizer = async_llm_engine.engine.get_tokenizer()
        self.request_counter = Counter()
        self.custom_sampler = DeferredSampler()
        self.cache = OutputCache(maxsize=cache_size, **cache_opts) if cache_size > 0 else None

        super().__init__(tokenizer=self.tokenizer, eos_token=eos_token)

    @classmethod
    def from_name(cls, model_name, engine_opts=None, **kwargs):
        """Create a AsyncVirtualLM instance from a model name.
        
        Args:
            model_name: Name of the model to load
            engine_opts: Additional options to pass to the AsyncLLMEngine. The engine will be 
                configured with prefix caching enabled and request logging disabled by default.
            **kwargs: Additional arguments passed to AsyncVirtualLM constructor
            
        Returns:
            An AsyncVirtualLM instance
        """
        if not HAS_VLLM:
            raise ImportError("vLLM not available. Install vLLM or use AsyncTransformer instead.")
        
        engine_opts = {
            'enable_prefix_caching': True,
            'disable_log_requests': True,
            **(engine_opts or {})
        }
        
        engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model_name, tokenizer=model_name, **engine_opts
            )
        )
        
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

    def next_token_logprobs_sync(self, token_ids):
        """Request log probabilities of next token synchronously.

        Note:
            This method is not recommended for use in async contexts. If you're already in an 
            async context, use `await next_token_logprobs(token_ids)` instead. Calling this 
            method from an async context will create a new thread with its own event loop to 
            handle the async operation.

        Args:
            token_ids_list (List[int]): A list of token IDs, representing the prompt to the language model.
                
        Returns:
            torch.Tensor: Normalized log probability tensor.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop.
            return asyncio.run(self.next_token_logprobs(token_ids))

        # We are in a running loop.
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run, self.next_token_logprobs(token_ids)
            ).result()

    default_params = SamplingParams(
        max_tokens=1, n=1, logprobs=1, detokenize=False, stop=None, ignore_eos=True
    )

    async def _next_token_logprobs(self, token_ids):
        """Request log probabilities of next token asynchronously. 
        
        Args:
            token_ids_list (List[int]): A list of token IDs, representing the prompt to the language model.
                
        Returns:
            torch.Tensor: Normalized log probability tensor.
        """
        request_id = str(next(self.request_counter))
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        outputs = []
        with self._optimized_sampling_context():
            async for output in self.async_llm_engine.generate(
                prompt=prompt, sampling_params=self.default_params, request_id=request_id
            ):
                if output.finished:
                    outputs.append(output)

        return self._validate_outputs(outputs)

    @contextmanager
    def _optimized_sampling_context(self):
        """Context manager for optimized sampling configuration."""
        model = self.async_llm_engine.engine.model_executor.driver_worker.model_runner.model
        schedulers = self.async_llm_engine.engine.scheduler
        
        original_state = {
            'sampler': model.sampler,
            'async_procs': [s._allow_async_output_proc for s in schedulers]
        }
        
        try:
            model.sampler = self.custom_sampler
            for scheduler in schedulers:
                scheduler._allow_async_output_proc = lambda x: False
            yield
        finally:
            model.sampler = original_state['sampler']
            for scheduler, orig_proc in zip(schedulers, original_state['async_procs']):
                scheduler._allow_async_output_proc = orig_proc

    def _validate_outputs(self, outputs):
        """Validate and extract logprobs from a vLLM output."""
        assert len(outputs) == 1 # Single step decoding.
        output = outputs[0]
        assert len(output.outputs) == 1 # Single sequence.
        sequence = output.outputs[0]
        assert len(sequence.logprobs) == 1 
        return sequence.logprobs[0].logprobs

    def clear_cache(self):
        self.cache.clear()

    def __del__(self):
        """Clean up resources on deletion."""
        self._cleanup_engine()

    def _cleanup_engine(self):
        """Clean up the vLLM engine and associated resources."""
        if not hasattr(self, 'async_llm_engine'):
            return
            
        async_engine = self.async_llm_engine
        async_engine.shutdown_background_loop()
        
        if executor := getattr(async_engine.engine, 'model_executor', None):
            destroy_model_parallel()
            destroy_distributed_environment()
            del executor
            
        del async_engine

class DeferredSampler(torch.nn.Module):
    """A custom vLLM sampler optimized for efficient next-token probability calculations.

    This sampler replaces vLLM's default sampling mechanism to optimize for scenarios
    where we only need the next token probabilities without actually sampling tokens.

    Note:
        While this sampler implements vLLM's expected interface, it intentionally
        avoids actual token sampling to optimize for probability calculation use cases.
        It should not be used in scenarios where actual token generation is needed.
    """
    def __init__(self): 
        super().__init__()

    def forward(self, logits, sampling_metadata):
        """Process model logits to create vLLM-compatible sampling outputs.

        This method implements the required vLLM sampler interface but optimizes for
        probability requests.

        Args:
            logits (torch.Tensor): Raw model logits with shape (num_tokens, vocab_size).
            sampling_metadata: vLLM metadata containing sequence grouping information.

        Returns:
            SamplerOutput: A vLLM-compatible output structure containing:
                - Sequence group outputs with lazy probability dictionaries
                - Placeholder values for unused sampling fields
                - No actual sampled tokens (uses dummy token_id=0)

        Note:
            The sampler uses token_id=0 as a placeholder.
        """
        assert logits is not None

        logprobs = logits.log_softmax(dim=-1, dtype=torch.float)

        sample_idx = 0
        sampler_output = []
        for seq_group in sampling_metadata.seq_groups:
            seq_ids = seq_group.seq_ids
            num_parent_seqs = len(seq_ids)
            logprobs_by_seq = logprobs[sample_idx : sample_idx + num_parent_seqs]

            assert len(logprobs_by_seq) == len(seq_ids)
            
            seq_outputs = []
            for (seq_id, seq_logprobs) in zip(seq_ids, logprobs_by_seq):
                seq_outputs.append(
                    SequenceOutput(seq_id, 0, LazyLogprobDict(seq_logprobs))
                )

            sampler_output.append(
                CompletionSequenceGroupOutput(samples=seq_outputs, prompt_logprobs=[])
            )
            
            sample_idx += 1

        sampler_outputs = SamplerOutput(
            outputs=sampler_output,
            sampled_token_probs=None,
            sampled_token_ids=None,
            logprobs=None,
            deferred_sample_results_args=None
        )

        return sampler_outputs


class LazyLogprobDict:
    """An efficient dictionary-like interface required by vLLM's output processing.
    
    vLLM's output processor expects token probabilities to be provided as a dictionary
    mapping token IDs to Logprob objects. However, creating this full dictionary is
    computationally expensive, especially when dealing with large vocabulary sizes
    (often 50k+ tokens).

    This class provides a compatible interface that satisfies vLLM's requirements while
    avoiding the overhead.
    """
    def __init__(self, logprobs):
        self.logprobs = logprobs

    def __getitem__(self, key):
        if 0 <= key < len(self.logprobs):
            return Logprob(self.logprobs[key])
        raise KeyError(key)

    def __contains__(self, key):
        return 0 <= key < len(self.logprobs)

    def __len__(self):
        return len(self.logprobs)

    def items(self):
        return ((i, Logprob(prob)) for i, prob in enumerate(self.logprobs))

    def keys(self):
        return range(len(self.logprobs))

    def values(self):
        return iter(map(Logprob, self.logprobs))

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
