import torch
import logging
import warnings
import os

from genlm.backend.llm.base import AsyncLM
from genlm.backend.cache import OutputCache

try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.utils import Counter
    from vllm.inputs import TokensPrompt

    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )

    HAS_VLLM = True
except ImportError:  # pragma: no cover
    HAS_VLLM = False  # pragma: no cover

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

    class PassThroughLogitsProcessor:
        """A logits processor that stores the logprobs and passes the logits through."""

        def __init__(self):
            self.log_probs = None

        def __call__(self, past_token_ids, logits):
            assert self.log_probs is None, (
                "Log probs already set. This should never happen."
            )
            self.log_probs = torch.log_softmax(logits, dim=-1, dtype=logits.dtype)
            return logits

    class AsyncVirtualLM(AsyncLM):
        def __init__(
            self,
            async_llm_engine,
            cache_size=0,
            cache_opts={},
            logprobs_per_request=256,
            v1=False,
        ):
            """Initialize an `AsyncVirtualLM` instance.

            Args:
                async_llm_engine (AsyncLLMEngine): The async vLLM engine instance.
                cache_size (int, optional): Maximum size of the output cache. If 0, caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the [`OutputCache`][genlm.backend.cache.OutputCache] constructor. Defaults to {}.
                v1: if true sets the engine to V1, otherwise to V0
                logprobs_per_request: used only in V1, selects the number of retrieved logprobs.

            Note:
                The cache stores the log probabilities for previously seen token sequences to avoid redundant requests. KV caching is handled internally by the vLLM engine.
            """
            self.v1 = v1
            self.async_llm_engine = async_llm_engine
            self.default_params = {
                "max_tokens": 1,
                "n": 1,
                "detokenize": False,
                "stop": None,
                "ignore_eos": True,
            }
            # Version specific modifications
            if self.v1:  # pragma: no cover
                self.default_params["logprobs"] = (  # pragma: no cover
                    logprobs_per_request  # set the retrieved logprobs
                )
                self.tokenizer = self._wrap_tokenizer(  # pragma: no cover
                    async_llm_engine.tokenizer
                )  # wrap tokenizer for V1 # pragma: no cover
                async_llm_engine.log_stats = False  # pragma: no cover
            else:
                self.tokenizer = async_llm_engine.engine.get_tokenizer()
                async_llm_engine.engine.log_stats = False
            self.request_counter = Counter()
            self.cache = (
                OutputCache(maxsize=cache_size, **cache_opts)
                if cache_size > 0
                else None
            )

            super().__init__(tokenizer=self.tokenizer)

        def _wrap_tokenizer(self, tokenizer):  # pragma: no cover
            """Wrap v1 tokenizer to be compatible with base class expectations.
            Note that in V1 async_llm_engine.tokenizer is a TokenizerGroup object"""

            class TokenizerWrapper:  # pragma: no cover
                def __init__(self, tokenizer):  # pragma: no cover
                    # Access the underlying tokenizer from TokenizerGroup
                    self._tokenizer = getattr(
                        tokenizer, "tokenizer", tokenizer
                    )  # pragma: no cover
                    # Add compatibility attributes
                    self.is_fast = (
                        True  # Assume fast tokenizer for v1 # pragma: no cover
                    )
                    self.name_or_path = getattr(
                        self._tokenizer,
                        "name_or_path",
                        "unknown",  # pragma: no cover
                    )  # pragma: no cover

                def __getattr__(  # pragma: no cover
                    self, name
                ):  # Retrieve the tokenizer from the TokenizerGroup object
                    return getattr(self._tokenizer, name)

                def __len__(self):  # pragma: no cover
                    return len(self._tokenizer)

            return TokenizerWrapper(tokenizer)

        @classmethod
        def from_name(
            cls,
            model_name,
            v1=False,
            logprobs_per_request=256,
            engine_opts=None,
            **kwargs,
        ):
            """Create a `AsyncVirtualLM` instance from a model name.

            Args:
                model_name (str): Name of the model to load.
                engine_opts (dict): Additional options to pass to the `AsyncLLMEngine`. The engine will be
                    configured with prefix caching enabled and async output processing disabled by default.
                **kwargs: Additional arguments passed to `AsyncVirtualLM` constructor.

            Returns:
                (AsyncVirtualLM): An `AsyncVirtualLM` instance.

            Note: for GPT-OSS,  vLLM >= 0.10.2 is required
            """
            if not HAS_VLLM:
                raise ImportError(  # pragma: no cover
                    "vLLM not available. Install vLLM or use AsyncTransformer instead."
                )

            if engine_opts is not None and "enable_chunked_prefill" in engine_opts:
                if engine_opts["enable_chunked_prefill"]:  # pragma: no cover
                    warnings.warn(  # pragma: no cover
                        "Setting enable_chunked_prefill to True may interfere with AsyncVirtualLM's "
                        "custom sampling functionality."
                    )

            if v1:  # pragma: no cover
                original_v1_env = os.environ.get(
                    "VLLM_USE_V1"  # pragma: no cover
                )  # The Engine Type could be set as an environmental variable so we set it to either V1 or V0 (after copying it in order to reset it later)
                os.environ["VLLM_USE_V1"] = "1"  # pragma: no cover
                from vllm.engine.arg_utils import (
                    AsyncEngineArgs,
                )  # the AsyncEngineArgs import is different in V1 and V0. # pragma: no cover

                engine_opts = {
                    "enable_prefix_caching": True,
                    "max_logprobs": logprobs_per_request,
                    **(engine_opts or {}),
                }  # pragma: no cover
            else:
                original_v1_env = os.environ.get("VLLM_USE_V1")
                os.environ["VLLM_USE_V1"] = "0"
                from vllm import (
                    AsyncEngineArgs,
                )  # the AsyncEngineArgs import is different in V1 and V0

                engine_opts = {
                    "enable_prefix_caching": True,
                    "disable_log_requests": True,  # is it possible to remove this parameter? it is cauing problems with vllm >= v 0.10.0
                    "disable_async_output_proc": True,  # This parameter forces vLLM to use v0, which is currently what we want to do.
                    **(engine_opts or {}),
                }

            engine = AsyncLLMEngine.from_engine_args(  # Set up the engine
                AsyncEngineArgs(model=model_name, tokenizer=model_name, **engine_opts)
            )

            # reset  the environmental variable, so that it does not interfere with other instances
            if original_v1_env is not None:
                os.environ["VLLM_USE_V1"] = original_v1_env
            else:
                os.environ.pop("VLLM_USE_V1", None)

            return cls(
                engine, v1=v1, logprobs_per_request=logprobs_per_request, **kwargs
            )

        @property
        def underlying_model(self):
            raise NotImplementedError  # pragma: no cover

        @property
        def logits_processors(self):
            return self._logits_processors  # pragma: no cover

        async def next_token_logprobs(self, token_ids):
            """Request log probabilities of next token asynchronously with output caching.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                result (torch.Tensor): Normalized log probability tensor.
            """

            key = tuple(token_ids)

            if self.cache is not None and key in self.cache:
                return self.cache[key]

            if self.v1:  # pragma: no cover
                result = await self._next_token_logprobs_v1(key)  # pragma: no cover
            else:
                result = await self._next_token_logprobs_v0(key)

            if self.cache is not None:
                self.cache[key] = result

            return result

        async def _next_token_logprobs_v1(self, token_ids):  # pragma: no cover
            """Request log probabilities of next token asynchronously.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            req_id = str(next(self.request_counter))

            # For v1, use string prompt directly instead of TokensPrompt
            if isinstance(token_ids, str):  # pragma: no cover
                prompt = token_ids
            else:  # pragma: no cover
                # Convert token IDs to string for v1 compatibility
                prompt = self.tokenizer.decode(token_ids)  # pragma: no cover

            outputs = []
            async for output in self.async_llm_engine.generate(
                prompt=prompt,
                sampling_params=SamplingParams(**self.default_params),
                request_id=req_id,
            ):  # pragma: no cover
                if output.finished:
                    outputs.append(output)

            # Extract logprobs from the output
            # v1 provides logprobs in the output when logprobs parameter is set
            output = outputs[0].outputs[0]  # pragma: no cover
            logprobs = output.logprobs

            assert logprobs, (
                "Log probs should have been retrieved at this point"
            )  # pragma: no cover
            # v1 logprobs format: list of dicts with token_id -> logprob
            vocab_size = len(self.tokenizer)  # pragma: no cover
            logprobs_tensor = torch.full(
                (1, vocab_size),
                -float("inf"),
                dtype=torch.float32,  # pragma: no cover
            )

            for token_id, logprob in logprobs[0].items():  # pragma: no cover
                # Assign the logprobs to the top-k retrieved tokens in the vocabulary.
                assert hasattr(logprob, "logprob"), (
                    "Logprob field is required"
                )  # pragma: no cover
                logprobs_tensor[0, token_id] = logprob.logprob

            # Right now we don't re-normalize! We might want to change this,
            # the remaining mass can either be redistributed among the remaining tokens
            # or among the selected ones.
            logprobs = logprobs_tensor  # pragma: no cover
            return logprobs[
                0
            ]  # Return shape (vocab_size,) instead of (1, vocab_size) # pragma: no cover

        async def _next_token_logprobs_v0(self, token_ids):
            """Request log probabilities of next token asynchronously.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            req_id = str(next(self.request_counter))
            prompt = TokensPrompt(prompt_token_ids=token_ids)

            outputs = []
            processor = PassThroughLogitsProcessor()
            async for output in self.async_llm_engine.generate(
                prompt=prompt,
                sampling_params=SamplingParams(
                    **self.default_params, logits_processors=[processor]
                ),
                request_id=req_id,
            ):
                if output.finished:
                    outputs.append(output)

            assert processor.log_probs is not None, (
                "Log probs should be set by the logits processor."
            )
            return processor.log_probs

        def next_token_logprobs_sync(self, token_ids):
            """Request log probabilities of next token synchronously.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            assert not self.v1  # Currently implemented only for V0
            return self.batch_next_token_logprobs_sync([token_ids])[0]

        def batch_next_token_logprobs_sync(self, token_ids_list):
            """
            Request log probabilities of next tokens in a batch synchronously.

            Args:
                token_ids_list (list[list[int]]): A list of token ID lists, each representing a prompt to the language model.

            Returns:
                (torch.Tensor): A tensor of normalized log probability tensors, one for each prompt in the input list.
            """
            assert not self.v1  # Currently implemented only for V0
            req_ids = []
            req_id2processors = {}
            for token_ids in token_ids_list:
                req_id = str(next(self.request_counter))
                req_ids.append(req_id)
                processor = PassThroughLogitsProcessor()
                req_id2processors[req_id] = processor
                self.async_llm_engine.engine.add_request(
                    prompt=TokensPrompt(prompt_token_ids=token_ids),
                    params=SamplingParams(
                        **self.default_params, logits_processors=[processor]
                    ),
                    request_id=req_id,
                )

            while self.async_llm_engine.engine.has_unfinished_requests():
                output = self.async_llm_engine.engine.step()
                for out in output:
                    if out.finished:
                        assert out.request_id in req_id2processors, (
                            f"{out.request_id} not in requested IDs"
                        )

            return torch.stack(
                [req_id2processors[req_id].log_probs for req_id in req_ids]
            )

        def clear_cache(self):
            """Clear output cache."""
            if self.cache:
                self.cache.clear()

        def __del__(self):
            """Clean up resources on deletion."""
            self._cleanup_engine()

        def _cleanup_engine(self):
            """Clean up the vLLM engine and associated resources."""
            if async_engine := getattr(self, "async_llm_engine", None):
                if self.v1:  # pragma: no cover
                    async_engine.shutdown()  # pragma: no cover
                else:
                    async_engine.shutdown_background_loop()
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
            if self.v1:  # pragma: no cover
                if isinstance(prompt_token_ids, list):  # pragma: no cover
                    prompt_token_ids = self.tokenizer.decode(
                        prompt_token_ids
                    )  # pragma: no cover
                elif isinstance(prompt_token_ids, str):  # pragma: no cover
                    pass
                else:  # pragma: no cover
                    raise ValueError(
                        f"Invalid prompt_ids_Type: {type(prompt_token_ids)}"
                    )  # pragma: no cover
            else:
                prompt_token_ids = TokensPrompt(prompt_token_ids=prompt_token_ids)

            # Question to check: Why do we need to use "byte_vocab"?
            def decode_eos(eos_token_ids):
                if self.v1:  # pragma: no cover
                    return [
                        self.tokenizer.decode([i]) for i in eos_token_ids
                    ]  # pragma: no cover
                else:  # What is the adavntage of using "byte_vocab" instead of the tokenizer. Can we do this also with V1 ?
                    [self.byte_vocab[i].decode() for i in eos_token_ids]

            async for output in self.async_llm_engine.generate(
                prompt=prompt_token_ids,
                sampling_params=SamplingParams(
                    n=1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    stop=decode_eos(eos_token_ids),
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
