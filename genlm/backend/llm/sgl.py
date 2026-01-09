import asyncio
import threading
from typing import Dict, List, Tuple, Optional
from collections import deque

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
except ImportError:  # pragma: no cover
    HAS_SGL = False  # pragma: no cover

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

            self._rid_to_token_ids: Dict[str, Tuple[int, ...]] = {}

            super().__init__(tokenizer=self.tokenizer)

        @classmethod
        def from_name(cls, model_id, engine_opts=None, gpu_id=0, **kwargs):
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
            self._rid_to_token_ids.clear()

        def _start(self):
            if not self._task or self._task.done():
                self._queue = asyncio.Queue()
                self._loop = asyncio.get_running_loop()
                self._loop_thread_id = threading.get_ident()
                self._task = asyncio.create_task(self._background_loop())

        def _queue_request(self, key: Tuple[int, ...]) -> asyncio.Future:
            if not self._task or self._task.done():
                self._start()
            fut = asyncio.get_running_loop().create_future()
            self._queue.put_nowait((key, fut))
            return fut

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
            if not token_ids:
                raise ValueError("Token ids must not be empty")

            key = tuple(token_ids)
            if self.cache is not None and key in self.cache:
                return self.cache[key]

            # If background loop is already running on some event loop, use it.
            if self._loop is not None and self._loop.is_running():
                if threading.get_ident() == self._loop_thread_id:
                    raise RuntimeError(
                        "next_token_logprobs_sync called on event-loop thread; use await"
                    )
                cfut = asyncio.run_coroutine_threadsafe(
                    self.next_token_logprobs(list(key)), self._loop
                )
                out = cfut.result()
            else:
                out = asyncio.run(self.next_token_logprobs(list(key)))

            if self.cache is not None:
                self.cache[key] = out
            return out

        def _register(
            self, token_ids: Tuple[int, ...], fut: asyncio.Future
        ) -> Optional[Request]:
            if fut.cancelled():
                return None

            self._pending.setdefault(token_ids, []).append(fut)

            if token_ids in self._inflight:
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
            self._rid_to_token_ids[req.rid] = token_ids
            self._inflight[token_ids] = req
            return req

        def _process_queue(
            self,
            first_item: Tuple[Tuple[int, ...], asyncio.Future],
        ) -> List[Request]:
            assert self._queue is not None

            to_submit: List[Request] = []

            token_ids, fut = first_item
            req = self._register(token_ids, fut)
            if req is not None:
                to_submit.append(req)

            while True:
                try:
                    token_ids, fut = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                req = self._register(token_ids, fut)
                if req is not None:
                    to_submit.append(req)

            return to_submit

        async def _background_loop(self) -> None:
            assert self._queue is not None
            try:
                while True:
                    first_item = await self._queue.get()

                    to_submit = self._process_queue(first_item)
                    if to_submit:
                        self.model.process_input_requests(to_submit)

                    while batch := self.model.get_next_batch_to_run():
                        with torch.inference_mode():
                            batch_result = self.model.run_batch(batch)
                            self.model.process_batch_result(batch, batch_result)

                            for i, req in enumerate(batch.reqs):
                                if req.finished():
                                    token_ids = self._rid_to_token_ids.pop(req.rid)
                                    print("just finished", token_ids)

                                    logits = (
                                        batch_result.logits_output.next_token_logits[i]
                                    )
                                    out = torch.log_softmax(logits, dim=-1).to("cpu")

                                    waiters = self._pending.pop(token_ids, [])
                                    self._inflight.pop(token_ids, None)

                                    for f in waiters:
                                        if not f.done() and not f.cancelled():
                                            f.set_result(out)

            except asyncio.CancelledError:
                raise
            except Exception as e:  # pragma: no cover
                # Never hang: fail everybody.
                for waiters in self._pending.values():
                    for f in waiters:
                        if not f.done() and not f.cancelled():
                            f.set_exception(e)
                self._pending.clear()
                self._inflight.clear()
                self._rid_to_token_ids.clear()
                raise

        def __del__(self):
            self.reset_async_queries()
            self._cleanup_engine()

        def _cleanup_engine(self):
            destroy_process_group()
            destroy_model_parallel()
            destroy_distributed_environment()
