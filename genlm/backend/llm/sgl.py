import asyncio
import threading
from typing import Dict, List, Tuple, Optional
from collections import deque

import torch


from genlm.backend.cache import OutputCache
from genlm.backend.llm.base import AsyncLM

try:
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput as Request
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    HAS_SGL = True
except ImportError:  # pragma: no cover
    HAS_SGL = False  # pragma: no cover

if not HAS_SGL:

    class AsyncSGLTransformer:  # pragma: no cover
        """Placeholder class when SGLang is not installed."""

        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "SGLang is not installed. Please install it with 'pip install sglang'"
            )

        @classmethod
        def from_name(cls, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "SGLang is not installed. Please install it with 'pip install sglang'"
            )

else:
    SP = SamplingParams(max_new_tokens=0)

    class AsyncSGLTransformer(AsyncLM):
        """Asynchronous wrapper around a SGLang inference engine.

        This class provides an asynchronous interface to SGLang inference engine with
        automatic batching and caching. It extends AsyncLM to provide efficient
        batched inference.

        The model automatically batches concurrent requests and uses a cache to store
        computed log probabilities for reuse.
        """

        def __init__(self, sgl_model, cache_size=0, cache_opts=None):
            """Initialize an `AsyncSGLTransformer` instance.

            Args:
                sgl_model: The SGLang inference engine instance.
                cache_size (int, optional): Maximum number of log probabilities to keep in memory.
                cache_opts (dict, optional): Additional configuration options for the cache.
            """
            self.model = sgl_model
            self.tokenizer = sgl_model.tokenizer

            cache_opts = {} if cache_opts is None else cache_opts
            self.cache = (
                OutputCache(maxsize=cache_size, **cache_opts)
                if cache_size > 0
                else None
            )

            self._queue: Optional[asyncio.Queue] = None
            self._task: Optional[asyncio.Task] = None

            self._pending: Dict[Tuple[int, ...], List[asyncio.Future]] = {}
            self._inflight: Dict[Tuple[int, ...], Request] = {}

            self._loop: Optional[asyncio.AbstractEventLoop] = None
            self._loop_thread_id: Optional[int] = None

            self._rid_to_token_ids: Dict[str, Tuple[int, ...]] = {}

            self._engine_paused: bool = False

            super().__init__(tokenizer=self.tokenizer)

        @classmethod
        def from_name(cls, model_id, engine_opts=None, gpu_id=0, **kwargs):
            """Create an `AsyncSGLTransformer` instance from a model name.

            Args:
                model_id (str): The name of the model to load.
                engine_opts (dict, optional): Additional configuration options for the SGLang inference engine.
                gpu_id (int, optional): The GPU ID to use for the inference engine.
                **kwargs: Additional arguments passed to the `AsyncSGLTransformer` constructor.

            Returns:
                (AsyncSGLTransformer): An initialized `AsyncSGLTransformer` instance.
            """
            _engine_opts = {
                "sampling_backend": "pytorch",
                "skip_tokenizer_init": False,
                "model_path": model_id,
                "grammar_backend": "none",
                "allow_auto_truncate": False,
                "disable_overlap_schedule": False,
                "mem_fraction_static": 0.9,  # default value is 0.9
            }
            if engine_opts:
                _engine_opts.update(engine_opts)
            server_args = ServerArgs(**_engine_opts)
            port_args = PortArgs.init_new(server_args)
            mod = Scheduler(server_args, port_args, gpu_id, 0, 0, 0, 0)
            mod.result_queue = deque()
            return cls(mod, **kwargs)

        def clear_cache(self):
            """Clear the logprobs output cache."""
            if self.cache:
                self.cache.clear()

        def clear_kv_cache(self):
            """Clear the SGLang cache."""
            return self.model.flush_cache()

        def _pause_engine(self):
            """Pause the SGLang inference engine."""
            self._engine_paused = True

        def _resume_engine(self):
            """Resume the SGLang inference engine."""
            self._engine_paused = False

        def reset_async_queries(self):
            """Clear any pending language model queries from the queue. Use this method when an exception prevented an inference algorithm from executing
            to completion."""
            self._pause_engine()

            for waiters in self._pending.values():
                for fut in waiters:
                    fut.cancel()

            self._pending.clear()
            self._inflight.clear()
            self._rid_to_token_ids.clear()

            if self._queue:
                while True:
                    try:
                        _, fut = self._queue.get_nowait()
                        fut.cancel()
                    except asyncio.QueueEmpty:
                        break

            self._resume_engine()

        def _start(self):
            """Start the background loop if it is not already running."""
            if not self._task or self._task.done():
                self._queue = asyncio.Queue()
                self._loop = asyncio.get_running_loop()
                self._loop_thread_id = threading.get_ident()
                self._task = asyncio.create_task(self._background_loop())

        def _queue_request(self, token_ids):
            """Queue a request to the SGLang inference engine.

            Args:
                token_ids (List[int]): The token IDs of the request.

            Returns:
                (asyncio.Future): A future that will be set with the result of the request.
            """
            if not self._task or self._task.done():
                self._start()
            fut = asyncio.get_running_loop().create_future()
            self._queue.put_nowait((token_ids, fut))
            return fut

        async def next_token_logprobs(self, token_ids: List[int]):
            """Request log probabilities of next token. This version is asynchronous because it automatically batches concurrent requests; use with `await`.

            Args:
                token_ids (list[int]): a list of token ids, representing a prompt to the language model.

            Returns:
                logprobs (torch.Tensor): a tensor of with the language model's log (normalized) probabilities for the next token following the prompt.
            """
            if not token_ids:
                raise ValueError("Token ids must not be empty")

            key = tuple(token_ids)

            if self.cache is not None and key in self.cache:
                return self.cache[key]

            out = await self._queue_request(key)

            if self.cache is not None:
                self.cache[key] = out

            return out

        def next_token_logprobs_sync(self, token_ids: List[int]):
            """Request log probabilities of next token synchronously.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            if not token_ids:
                raise ValueError("Token ids must not be empty")

            key = tuple(token_ids)
            if self.cache is not None and key in self.cache:
                return self.cache[key]

            out = asyncio.run(self.next_token_logprobs(token_ids))

            if self.cache is not None:
                self.cache[key] = out
            return out

        def _register(self, token_ids, future):
            """Register a request with the SGLang inference engine.

            Args:
                token_ids (List[int]): The token IDs of the request.
                future (asyncio.Future): A future that will be set with the result of the request.

            Returns:
                (Request | None): The Request object that was registered, or None if the request future was cancelled.
            """
            if future.cancelled():
                return None

            key = tuple(token_ids)

            self._pending.setdefault(key, []).append(future)

            if key in self._inflight:
                return None

            req = Request(
                input_text="",
                input_ids=list(token_ids),
                mm_inputs=None,
                sampling_params=SP,
                return_logprob=False,
                logprob_start_len=-1,
                top_logprobs_num=-1,
                token_ids_logprob=[],
                stream=False,
            )
            req.regenerate_rid()
            self._rid_to_token_ids[req.rid] = key
            self._inflight[key] = req
            return req

        async def _drain_queue(self) -> List[Request]:
            """Wait for at least one item, then drain all available items from the queue."""
            assert self._queue is not None

            requests = []

            # Wait for at least one item
            token_ids, future = await self._queue.get()
            req = self._register(token_ids, future)
            if req is not None:
                requests.append(req)

            while True:
                try:
                    token_ids, future = self._queue.get_nowait()
                    req = self._register(token_ids, future)
                    if req is not None:
                        requests.append(req)
                except asyncio.QueueEmpty:
                    break

            return requests

        async def _background_loop(self):
            """Background task that processes queued requests from the queue."""
            assert self._queue is not None
            try:
                while True:
                    requests = await self._drain_queue()
                    self.model.process_input_requests(requests)

                    if self._engine_paused:
                        await asyncio.sleep(0.01)
                        continue
                    while batch := self.model.get_next_batch_to_run():
                        with torch.inference_mode():
                            batch_result = self.model.run_batch(batch)
                            self.model.process_batch_result(batch, batch_result)
                            logprobs = torch.log_softmax(
                                batch_result.logits_output.next_token_logits, dim=-1
                            ).to("cpu", non_blocking=True)

                            for i, req in enumerate(batch.reqs):
                                if req.finished():
                                    token_ids = self._rid_to_token_ids.pop(req.rid)
                                    waiters = self._pending.pop(token_ids, [])
                                    self._inflight.pop(token_ids, None)

                                    for f in waiters:
                                        f.set_result(logprobs[i])

            except asyncio.CancelledError:
                raise
            except Exception as e:
                for waiters in self._pending.values():
                    for f in waiters:
                        f.set_exception(e)
                self._pending.clear()
                self._inflight.clear()
                self._rid_to_token_ids.clear()
                raise

        def _cleanup_engine(self):
            """Clean up the SGLang inference engine and distributed environment."""
            self.reset_async_queries()
            destroy_model_parallel()
            destroy_distributed_environment()

        def __del__(self):  # pragma: no cover
            """Clean up the SGLang inference engine when the instance is deleted."""
            self._cleanup_engine()
