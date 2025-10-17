import torch
import logging
import warnings

from genlm.backend.llm.base import AsyncLM
from genlm.backend.cache import OutputCache

try:
    # Import vLLM - need to handle potential shadowing by local vllm.py file
    import sys
    import importlib.util
    
    # Find the vLLM package (not the local vllm.py file)
    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec and vllm_spec.origin and not vllm_spec.origin.endswith("vllm.py"):
        # This is the real vLLM package
        import vllm
        AsyncLLMEngine = vllm.AsyncLLMEngine
        SamplingParams = vllm.SamplingParams
        
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.utils import Counter
        from vllm.inputs import TokensPrompt
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        HAS_VLLM = True
    else:
        # vLLM package not found or shadowed
        raise ImportError("vLLM package not found or shadowed by local vllm.py")

except (ImportError, AttributeError) as e:  # pragma: no cover
    HAS_VLLM = False  # pragma: no cover
    import sys
    print(f"WARNING: vLLM import failed: {e}", file=sys.stderr)

if not HAS_VLLM:

    class AsyncVirtualLM:  # pragma: no cover
        """Placeholder class when vLLM is not installed."""

        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "vLLM is not installed. Please install it with 'pip install vllm' "
                "to use the vLLM-based AsyncLM model."
            )

        @classmethod
        def from_name(cls, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "vLLM is not installed. Please install it with 'pip install vllm' "
                "to use the vLLM-based AsyncLM model."
            )

else:
    logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)

    
    class AsyncVirtualLM(AsyncLM):
        default_params = {
            "max_tokens": 1,
            "n": 1,
            "detokenize": False,
            "stop": None,
            "ignore_eos": True,
            "logprobs": 256,  # Request logprobs for top 1000 tokens
        }

        def __init__(self, async_llm_engine, cache_size=0, cache_opts={}, logits_processors=None):
            """Initialize an `AsyncVirtualLM` instance.

            Args:
                async_llm_engine (AsyncLLMEngine): The async vLLM engine instance.
                cache_size (int, optional): Maximum size of the output cache. If 0, caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the [`OutputCache`][genlm.backend.cache.OutputCache] constructor. Defaults to {}.

            Note:
                The cache stores the log probabilities for previously seen token sequences to avoid redundant requests. KV caching is handled internally by the vLLM engine.
            """
            self.async_llm_engine = async_llm_engine
            # Wrap v1 tokenizer to be compatible with base class
            self.tokenizer = self._wrap_tokenizer(async_llm_engine.tokenizer)
            self.request_counter = Counter()
            # Store logits processors for compatibility (not used in v1)
            self._logits_processors = list(logits_processors or [])
            self.cache = (
                OutputCache(maxsize=cache_size, **cache_opts)
                if cache_size > 0
                else None
            )

            async_llm_engine.log_stats = False

            super().__init__(tokenizer=self.tokenizer)

        def _wrap_tokenizer(self, tokenizer):
            """Wrap v1 tokenizer to be compatible with base class expectations."""
            class TokenizerWrapper:
                def __init__(self, tokenizer):
                    # Access the underlying tokenizer from TokenizerGroup
                    self._tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
                    # Add compatibility attributes
                    self.is_fast = True  # Assume fast tokenizer for v1
                    self.name_or_path = getattr(self._tokenizer, 'name_or_path', 'unknown')

                def __getattr__(self, name):
                    return getattr(self._tokenizer, name)
                
                def __len__(self):
                    return len(self._tokenizer)
            
            return TokenizerWrapper(tokenizer)

        @classmethod
        def from_name(cls, model_name, engine_opts=None, **kwargs):
            """Create a `AsyncVirtualLM` instance from a model name.

            Args:
                model_name (str): Name of the model to load.
                engine_opts (dict): Additional options to pass to the `AsyncLLMEngine`. The engine will be
                    configured with prefix caching enabled and async output processing disabled by default.
                **kwargs: Additional arguments passed to `AsyncVirtualLM` constructor.

            Returns:
                (AsyncVirtualLM): An `AsyncVirtualLM` instance.
            """
            if not HAS_VLLM:
                raise ImportError(  # pragma: no cover
                    "vLLM not available. Install vLLM or use AsyncTransformer instead."
                )

            if engine_opts is not None and "enable_chunked_prefill" in engine_opts:
                if engine_opts["enable_chunked_prefill"]:
                    warnings.warn(  # pragma: no cover
                        "Setting enable_chunked_prefill to True may interfere with AsyncVirtualLM's "
                        "custom sampling functionality."
                    )

            engine_opts = {
                "enable_prefix_caching": True,
                "disable_log_requests": True,
                "gpu_memory_utilization": 0.3,  # Reduce GPU memory usage
                "max_model_len": 512,  # Reduce max sequence length
                "max_logprobs": 1000,  # Allow up to 1000 logprobs per token
                # "disable_async_output_proc": True,  # This parameter forces vLLM to use v0, which is currently what we want to do.
                **(engine_opts or {}),
            }

            engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(model=model_name, tokenizer=model_name, **engine_opts)
            )

            return cls(engine, **kwargs)

        @property
        def underlying_model(self):
            raise NotImplementedError

        @property
        def logits_processors(self):
            return self._logits_processors

        async def next_token_logprobs(self, token_ids):
            """Request log probabilities of next token asynchronously with output caching.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                result (torch.Tensor): Normalized log probability tensor.
            """
            # Handle string input properly
            if isinstance(token_ids, str):
                key = token_ids
            else:
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
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            req_id = str(next(self.request_counter))
            print(f"request id: {req_id}")
            # For v1, use string prompt directly instead of TokensPrompt
            if isinstance(token_ids, str):
                prompt = token_ids
            else:
                # Convert token IDs back to string for v1 compatibility
                prompt = self.tokenizer.decode(token_ids)

            outputs = []
            async for output in self.async_llm_engine.generate(
                prompt=prompt,
                sampling_params=SamplingParams(**self.default_params),
                request_id=req_id,
            ):
                if output.finished:
                    outputs.append(output)

            if not outputs:
                raise RuntimeError("No outputs generated")
            
            # Extract logprobs from the output
            # v1 provides logprobs in the output when logprobs parameter is set
            output = outputs[0].outputs[0]
            logprobs = output.logprobs
            
            assert logprobs
            print(f"shape of logprobs before: {len(logprobs[0]),type(logprobs)}")
            # v1 logprobs format: list of dicts with token_id -> logprob
            # With max_logprobs=1000, we get many more logprobs
            vocab_size = len(self.tokenizer)
            logprobs_tensor = torch.full((1, vocab_size), -float('inf'), dtype=torch.float32)
            
            if isinstance(logprobs, list) and len(logprobs) > 0:
                # Get the first (and only) token's logprobs
                token_logprobs = logprobs[0]
                if isinstance(token_logprobs, dict):
                    for token_id, logprob in token_logprobs.items():
                        print(f"token_id: {token_id}, logprob: {logprob}")
                        if isinstance(token_id, int) and 0 <= token_id < vocab_size:
                            # Extract the actual logprob value from the Logprob object
                            if hasattr(logprob, 'logprob'):
                                logprobs_tensor[0, token_id] = logprob.logprob
                            else:
                                logprobs_tensor[0, token_id] = float(logprob)
            
            #Distribute the remaining mass across the tokens that are not in the top-k
            non_inf_mask = logprobs_tensor[0] != -float('inf')
            if non_inf_mask.sum() > 0:
                # Get the logprobs for the top-k tokens
                top_logprobs = logprobs_tensor[0][non_inf_mask]
                
                # Calculate the remaining probability mass
                remaining_prob = 1.0 - torch.exp(top_logprobs).sum().item()
                if remaining_prob > 0:
                    # Distribute the remaining probability uniformly among remaining tokens
                    remaining_tokens = (~non_inf_mask).sum().item()
                    if remaining_tokens > 0:
                        uniform_logprob = torch.log(torch.tensor(remaining_prob / remaining_tokens))
                        logprobs_tensor[0][~non_inf_mask] = uniform_logprob
            
            logprobs = logprobs_tensor
            
            return logprobs[0]  # Return shape (vocab_size,) instead of (1, vocab_size)

        def next_token_logprobs_sync(self, token_ids):
            """Request log probabilities of next token synchronously.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            import asyncio
            return asyncio.run(self.next_token_logprobs(token_ids))

        async def batch_next_token_logprobs(self, token_ids_list):
            """
            Request log probabilities of next tokens in a batch asynchronously.

            Args:
                token_ids_list (list[list[int]]): A list of token ID lists, each representing a prompt to the language model.

            Returns:
                (torch.Tensor): A tensor of normalized log probability tensors, one for each prompt in the input list.
            """
            # Handle empty batch
            if not token_ids_list:
                return torch.empty((0, len(self.tokenizer)), dtype=torch.float32)
            
            # Use the base class implementation
            return await super().batch_next_token_logprobs(token_ids_list)

        def batch_next_token_logprobs_sync(self, token_ids_list):
            """
            Request log probabilities of next tokens in a batch synchronously.

            Args:
                token_ids_list (list[list[int]]): A list of token ID lists, each representing a prompt to the language model.

            Returns:
                (torch.Tensor): A tensor of normalized log probability tensors, one for each prompt in the input list.
            """
            # Handle empty batch
            if not token_ids_list:
                return torch.empty((0, len(self.tokenizer)), dtype=torch.float32)
            
            # Use the base class implementation
            return super().batch_next_token_logprobs_sync(token_ids_list)

        def clear_cache(self):
            """Clear output cache."""
            if self.cache:
                self.cache.clear()

        def __del__(self):
            """Clean up resources on deletion."""
            self._cleanup_engine()

        def _cleanup_engine(self):
            """Clean up the vLLM engine and associated resources."""
            if async_engine := getattr(self, "async_llm", None):
                async_engine.shutdown()
                destroy_model_parallel()
                destroy_distributed_environment()

        async def sample(
            self,
            prompt_token_ids,
            max_tokens,
            eos_token_ids,
            temperature=1.0,
            seed=None,
        ):
            """Sample from the language model.

            Args:
                prompt_token_ids (list[int]): The token IDs of the prompt.
                eos_token_ids (list[int]): The token IDs of the end-of-sequence tokens.
                temperature (float, optional): The temperature to use to rescale the logits. Defaults to 1.0.
                max_tokens (int): The maximum number of tokens to generate.
                seed (int, optional): The seed for the random number generator. Defaults to None.

            Returns:
                (list[int]): The sampled token IDs.
            """
            async for output in self.async_llm_engine.generate(
                prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
                sampling_params=SamplingParams(
                    n=1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    stop=[self.byte_vocab[i].decode() for i in eos_token_ids],
                ),
                request_id=str(next(self.request_counter)),
            ):
                if output.finished:
                    assert len(output.outputs) == 1, (
                        "Expected exactly one sequence group"
                    )
                    token_ids = list(output.outputs[0].token_ids)
                    if token_ids[-1] in eos_token_ids:
                        token_ids = token_ids[:-1]
                    return token_ids
