import asyncio
import threading
from typing import Dict, List, Tuple, Optional

import torch

from torch.distributed import destroy_process_group

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
except ImportError:
    HAS_SGL = False

if not HAS_SGL:

    class AsyncSGLTransformer(AsyncLM):
        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "sglang is not installed. Please install it with 'pip install sglang'"
            )

        @classmethod
        def from_name(cls, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "sglang is not installed. Please install it with 'pip install sglang'"
            )

else:
    SP = SamplingParams(max_new_tokens=0)

    def _make_request(token_ids: Tuple[int, ...]) -> Request:
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
        return req

    class AsyncSGLTransformer(AsyncLM):
        def __init__(self, sgl_model, cache_size=0, cache_opts=None):
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

            super().__init__(tokenizer=self.tokenizer)

        @classmethod
        def from_name(cls, model_id, engine_opts=None, gpu_id=0, **kwargs):
            _engine_opts = {"sampling_backend": "pytorch", "skip_tokenizer_init": False}
            if engine_opts:
                _engine_opts.update(engine_opts)
            engine_opts = ServerArgs(
                model_path=model_id, grammar_backend="none", **_engine_opts
            )
            port_args = PortArgs.init_new(engine_opts)
            mod = Scheduler(engine_opts, port_args, gpu_id, 0, 0, 0, 0)
            return cls(mod, **kwargs)

        def clear_cache(self):
            if self.cache:
                self.cache.clear()

        def clear_kv_cache(self):
            return self.model.flush_cache()

        def reset_async_queries(self, *, exc: Optional[BaseException] = None):
            if self._task and not self._task.done():
                self._task.cancel()

            for waiters in self._pending.values():
                for fut in waiters:
                    if fut.done():
                        continue
                    if exc is not None:
                        fut.set_exception(exc)
                    else:
                        fut.cancel()

            self._pending.clear()
            self._inflight.clear()

            if self._queue:
                while not self._queue.empty():
                    try:
                        _, fut = self._queue.get_nowait()
                        if not fut.done():
                            if exc is not None:
                                fut.set_exception(exc)
                            else:
                                fut.cancel()
                    except asyncio.QueueEmpty:
                        break

            self._queue = None
            self._task = None
            self._loop = None
            self._loop_thread_id = None

        def _start(self):
            if not self._task or self._task.done():
                self._queue = asyncio.Queue()
                self._loop = asyncio.get_running_loop()
                self._loop_thread_id = threading.get_ident()
                self._task = asyncio.create_task(self._background_loop())

        def _queue_request(self, key: Tuple[int, ...]) -> asyncio.Future:
            self._start()
            fut = asyncio.get_running_loop().create_future()
            self._queue.put_nowait((key, fut))
            return fut

        def _register(self, key: Tuple[int, ...], fut: asyncio.Future):
            if not fut.cancelled():
                self._pending.setdefault(key, []).append(fut)

        async def next_token_logprobs(self, token_ids: List[int]) -> torch.Tensor:
            if not token_ids:
                raise ValueError("Token ids must not be empty")

            key = tuple(token_ids)

            if self.cache is not None and key in self.cache:
                return self.cache[key]

            out = await self._queue_request(key)

            if self.cache is not None:
                self.cache[key] = out

            return out

        def next_token_logprobs_sync(self, token_ids: List[int]) -> torch.Tensor:
            key = tuple(token_ids)
            if self.cache is not None and key in self.cache:
                return self.cache[key]

            if self._loop is not None and self._loop.is_running():
                if threading.get_ident() == self._loop_thread_id:
                    raise RuntimeError("sync called on event-loop thread")
                cfut = asyncio.run_coroutine_threadsafe(
                    self.next_token_logprobs(list(key)), self._loop
                )
                out = cfut.result()
            else:
                out = asyncio.run(self.next_token_logprobs(list(key)))

            if self.cache is not None:
                self.cache[key] = out
            return out

        async def _background_loop(self):
            assert self._queue is not None
            try:
                while True:
                    key, fut = await self._queue.get()
                    self._register(key, fut)

                    while True:
                        try:
                            key, fut = self._queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        self._register(key, fut)

                    to_submit = []
                    for key in list(self._pending.keys()):
                        if key not in self._inflight:
                            req = _make_request(key)
                            self._inflight[key] = req
                            to_submit.append(req)
                    if to_submit:
                        self.model.process_input_requests(to_submit)

                    while (batch := self.model.get_next_batch_to_run()) is not None:
                        with torch.inference_mode():
                            batch_result = self.model.run_batch(batch)
                            self.model.process_batch_result(batch, batch_result)

                            for i, req in enumerate(batch.reqs):
                                if not req.finished():
                                    continue
                                key = tuple(req.origin_input_ids)
                                logits = batch_result.logits_output.next_token_logits[i]
                                out = torch.log_softmax(logits, dim=-1).to("cpu")

                                waiters = self._pending.pop(key, [])
                                self._inflight.pop(key, None)
                                for f in waiters:
                                    if not f.done() and not f.cancelled():
                                        f.set_result(out)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                # never hang callers
                for waiters in self._pending.values():
                    for f in waiters:
                        if not f.done() and not f.cancelled():
                            f.set_exception(e)
                self._pending.clear()
                self._inflight.clear()
                raise

        def __del__(self):
            self.reset_async_queries()
            self._cleanup_engine()

        def _cleanup_engine(self):
            destroy_process_group()
            destroy_model_parallel()
            destroy_distributed_environment()
