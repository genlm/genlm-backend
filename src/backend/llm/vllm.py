import torch
import logging
import warnings
from contextlib import contextmanager

from genlm_backend.llm.base import AsyncLM
from genlm_backend.cache import OutputCache

try:
    from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
    from vllm.utils import Counter
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.sampler import SamplerOutput
    from vllm.sequence import SequenceOutput, CompletionSequenceGroupOutput, Logprob

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    warnings.warn(
        "vLLM not installed. Run 'pip install vllm' to use the vLLM-based AsyncLM model."
    )

if not HAS_VLLM:

    class AsyncVirtualLM:
        """Placeholder class when vLLM is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "vLLM is not installed. Please install it with 'pip install vllm' "
                "to use the vLLM-based AsyncLM model."
            )

        @classmethod
        def from_name(cls, *args, **kwargs):
            raise ImportError(
                "vLLM is not installed. Please install it with 'pip install vllm' "
                "to use the vLLM-based AsyncLM model."
            )

else:
    logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)

    class AsyncVirtualLM(AsyncLM):
        """A wrapper around vLLM's `AsyncLLMEngine` for asynchronous next token log probability computations.

        This class provides an asynchronous interface for computing log probabilities using vLLM's engine.
        It is optimized for next token log probability computations and supports caching of results (outputs and KV).
        """

        default_params = SamplingParams(
            max_tokens=1, n=1, logprobs=1, detokenize=False, stop=None, ignore_eos=True
        )

        def __init__(self, async_llm_engine, cache_size=0, cache_opts={}):
            """Initialize an `AsyncVirtualLM` instance.

            Args:
                async_llm_engine (AsyncLLMEngine): The async vLLM engine instance.
                cache_size (int, optional): Maximum size of the output cache. If 0, caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the [`OutputCache`][genlm_backend.cache.OutputCache] constructor. Defaults to {}.

            Note:
                The cache stores the log probabilities for previously seen token sequences to avoid redundant requests. KV caching is handled internally by the vLLM engine.
            """
            self.async_llm_engine = async_llm_engine
            self.tokenizer = async_llm_engine.engine.get_tokenizer()
            self.request_counter = Counter()
            self.custom_sampler = DeferredSampler()
            self.cache = (
                OutputCache(maxsize=cache_size, **cache_opts)
                if cache_size > 0
                else None
            )

            async_llm_engine.engine.log_stats = False

            super().__init__(tokenizer=self.tokenizer)

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
                raise ImportError(
                    "vLLM not available. Install vLLM or use AsyncTransformer instead."
                )

            if engine_opts is not None and "enable_chunked_prefill" in engine_opts:
                if engine_opts["enable_chunked_prefill"]:
                    warnings.warn(
                        "Setting enable_chunked_prefill to True may interfere with AsyncVirtualLM's "
                        "custom sampling functionality."
                    )

            engine_opts = {
                "enable_prefix_caching": True,
                "disable_log_requests": True,
                "disable_async_output_proc": True,
                # Need to disable chunked prefill to avoid issues
                # with our custom sampler.
                "enable_chunked_prefill": False,
                **(engine_opts or {}),
            }

            engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(model=model_name, tokenizer=model_name, **engine_opts)
            )

            return cls(engine, **kwargs)

        async def next_token_logprobs(self, token_ids):
            """Request log probabilities of next token asynchronously with output caching.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                result (torch.Tensor): Normalized log probability tensor.

            Warning:
                Do not use `asyncio.run(next_token_logprobs())` as it may interfere with vLLM's background loop.
                For synchronous usage, use the `next_token_logprobs_sync()` method instead.
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
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            req_id = str(next(self.request_counter))
            prompt = TokensPrompt(prompt_token_ids=token_ids)

            outputs = []
            with self._optimized_sampling_context():
                async for output in self.async_llm_engine.generate(
                    prompt=prompt,
                    sampling_params=self.default_params,
                    request_id=req_id,
                ):
                    if output.finished:
                        outputs.append(output)

            return self._validate_outputs(outputs)

        def next_token_logprobs_sync(self, token_ids):
            """Request log probabilities of next token synchronously.

            Args:
                token_ids_list (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            return self.batch_next_token_logprobs_sync([token_ids])[0]

        def batch_next_token_logprobs_sync(self, token_ids_list):
            """
            Request log probabilities of next tokens in a batch synchronously.

            Args:
                token_ids_list (list[list[int]]): A list of token ID lists, each representing a prompt to the language model.

            Returns:
                (torch.Tensor): A tensor of normalized log probability tensors, one for each prompt in the input list.
            """
            req_ids = []
            for token_ids in token_ids_list:
                req_id = str(next(self.request_counter))
                req_ids.append(req_id)
                self.async_llm_engine.engine.add_request(
                    prompt=TokensPrompt(prompt_token_ids=token_ids),
                    params=self.default_params,
                    request_id=req_id,
                )

            req_id2outputs = {}
            with self._optimized_sampling_context():
                while self.async_llm_engine.engine.has_unfinished_requests():
                    output = self.async_llm_engine.engine.step()
                    for out in output:
                        if out.finished:
                            assert out.request_id not in req_id2outputs, (
                                f"Duplicate outputs for request {out.request_id}"
                            )
                            assert out.request_id in req_ids, (
                                f"{out.request_id} not in requested IDs"
                            )
                            req_id2outputs[out.request_id] = out

            logprobs = [
                self._validate_outputs([req_id2outputs[req_id]]) for req_id in req_ids
            ]

            return torch.stack(logprobs)

        @contextmanager
        def _optimized_sampling_context(self):
            """Context manager for optimized sampling configuration."""
            model = self.async_llm_engine.engine.model_executor.driver_worker.model_runner.model
            original_sampler = model.sampler
            try:
                model.sampler = self.custom_sampler
                yield
            finally:
                model.sampler = original_sampler

        def _validate_outputs(self, outputs):
            """Validate and extract logprobs from a vLLM output.

            Args:
                outputs: List of sequence group outputs from vLLM generation

            Returns:
                Tensor of log probabilities for the next token

            Raises:
                AssertionError: If output structure doesn't match expected format
            """
            assert len(outputs) == 1, "Expected exactly one sequence group"
            seq_group = outputs[0]

            assert len(seq_group.outputs) == 1, (
                "Expected exactly one sequence in output"
            )
            sequence = seq_group.outputs[0]

            assert len(sequence.logprobs) == 1, "Expected exactly one set of logprobs"
            token_logprobs = sequence.logprobs[0].logprobs

            return token_logprobs

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
                async_engine.shutdown_background_loop()


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

            if not seq_group.do_sample:
                sampler_output.append(
                    CompletionSequenceGroupOutput(samples=[], prompt_logprobs=[])
                )
            else:
                assert len(logprobs_by_seq) == len(seq_ids)
                seq_outputs = []
                for seq_id, seq_logprobs in zip(seq_ids, logprobs_by_seq):
                    seq_outputs.append(
                        SequenceOutput(seq_id, 0, LazyLogprobDict(seq_logprobs))
                    )

                sampler_output.append(
                    CompletionSequenceGroupOutput(
                        samples=seq_outputs, prompt_logprobs=[]
                    )
                )

            sample_idx += 1

        sampler_outputs = SamplerOutput(
            outputs=sampler_output,
            sampled_token_probs=None,
            sampled_token_ids=None,
            logprobs=None,
            deferred_sample_results_args=None,
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
