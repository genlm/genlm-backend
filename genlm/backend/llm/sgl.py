import asyncio
import uuid
from typing import Dict, List, Tuple, Optional
from collections import deque
import torch

from genlm.backend.cache import OutputCache
from genlm.backend.llm.base import AsyncLM

try:
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.io_struct import (
        TokenizedGenerateReqInput as Request,
        LoadLoRAAdapterReqInput,
    )
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

    def _make_request(token_ids: Tuple[int], lora_id: Optional[str] = None) -> Request:
        """Construct a SGLang inference request object.

        Args:
            token_ids (Tuple[int]): The token IDs of the request.
            lora_id (Optional[str]): UUID of the active LoRA adapter, or None for the base model.

        Returns:
            (Request): The Request object.
        """
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
            lora_id=lora_id,
        )
        req.regenerate_rid()
        return req

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

            # _pending, _inflight, and _rid_to_token_ids are all keyed by
            # tuple(token_ids). Adapter switches wipe these (along with the
            # output cache), so the cache only ever holds entries from the
            # currently-active regime.
            self._pending: Dict = {}
            self._inflight: Dict = {}
            self._rid_to_token_ids: Dict[str, tuple] = {}

            self._lora_name_to_id: Dict[str, str] = {}
            # UUID of the currently-active LoRA adapter, or None for the
            # base model. Plumbed into every TokenizedGenerateReqInput.
            self._lora_id: Optional[str] = None

            super().__init__(tokenizer=self.tokenizer)

        @classmethod
        def from_name(cls, model_id, engine_opts=None, gpu_id=0, **kwargs):
            """Create an `AsyncSGLTransformer` instance from a model name.

            Args:
                model_id (str): The name of the model to load.
                engine_opts (dict, optional): Additional configuration options for the SGLang inference engine.
                    To enable LoRA, pass at least ``{"enable_lora": True, "max_lora_rank": <r>,
                    "lora_target_modules": [...]}``. ``max_loras_per_batch`` and
                    ``lora_backend`` are also accepted; see SGLang's ``ServerArgs``
                    for the full list. These must be set at construction time;
                    they cannot be changed later.
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

        def add_new_lora(self, lora_path, lora_name="lora_1"):
            """Load a LoRA adapter into the SGLang engine.

            Args:
                lora_path (str): Path to the adapter weights or HF hub identifier.
                lora_name (str): Name to associate with the loaded adapter.

            Notes:
                Blocks the caller for the actual disk/GPU load duration. SGLang's
                LoRA path is not lazy (unlike vLLM's), so this is a real I/O step.
                Requires ``enable_lora=True`` in the engine options at construction.
                Does not activate the adapter; call :meth:`set_lora` to enable it.
            """
            lora_id = uuid.uuid4().hex
            req = LoadLoRAAdapterReqInput(
                lora_name=lora_name,
                lora_path=lora_path,
                pinned=False,
                lora_id=lora_id,
            )
            result = self.model.load_lora_adapter(req)
            if not result.success:
                raise RuntimeError(
                    f"Failed to load LoRA adapter '{lora_name}' from '{lora_path}': "
                    f"{result.error_message}"
                )
            self._lora_name_to_id[lora_name] = lora_id

        def set_lora(self, lora_path=None, lora_name="lora_1"):
            """Activate a previously loaded LoRA adapter.

            Args:
                lora_path: Unused; accepted for signature parity with the base class.
                lora_name (str): Name of the adapter to activate (must match a prior ``add_new_lora`` call).

            Notes:
                Wipes the output cache and any in-flight tracking, so the
                cache only ever holds entries from the currently-active
                regime. Concurrent in-flight requests are cancelled.
            """
            if lora_name not in self._lora_name_to_id:
                raise ValueError(
                    f"A LoRA adapter named '{lora_name}' has not been loaded yet. "
                    "Call add_new_lora() first."
                )
            self._lora_id = self._lora_name_to_id[lora_name]
            self.reset_async_queries()
            self.clear_cache()

        def clear_lora(self):
            """Deactivate any active LoRA adapter; subsequent requests use the base model.

            Notes:
                Adapter weights stay loaded in the engine; only the active
                pointer is reset. Wipes the output cache and any in-flight
                tracking; concurrent in-flight requests are cancelled.
            """
            self._lora_id = None
            self.reset_async_queries()
            self.clear_cache()

        def clear_cache(self):
            """Clear the logprobs output cache."""
            if self.cache:
                self.cache.clear()

        def clear_kv_cache(self):
            """Clear the SGLang cache."""
            return self.model.flush_cache()

        def reset_async_queries(self):
            """Clear any pending language model queries from the queue. Use this method when an exception prevented an inference algorithm from executing
            to completion."""

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

            if self._task and not self._task.done():
                self._task.cancel()
            self._task = None
            self._queue = None

        def _start(self):
            """Start the background loop if it is not already running."""
            if not self._task or self._task.done():
                self._queue = asyncio.Queue()
                self._task = asyncio.create_task(self._background_loop())

        def _queue_request(self, token_ids: Tuple[int]):
            """Queue a request to the SGLang inference engine.

            Args:
                token_ids (tuple[int]): The token IDs of the request.

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

            token_tuple = tuple(token_ids)
            if self.cache is not None and token_tuple in self.cache:
                return self.cache[token_tuple]

            out = await self._queue_request(token_tuple)

            if self.cache is not None:
                self.cache[token_tuple] = out

            return out

        def next_token_logprobs_sync(self, token_ids: List[int]):
            """Request log probabilities of next token synchronously.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            return self.batch_next_token_logprobs_sync([token_ids])[0]

        def batch_next_token_logprobs_sync(self, token_ids_list: List[List[int]]):
            """Request log probabilities of next tokens in a batch synchronously.

            Args:
                token_ids_list (list[list[int]]): A list of token ID lists, each representing a prompt.

            Returns:
                (torch.Tensor): A tensor of normalized log probability tensors.
            """
            input_keys = [tuple(t) for t in token_ids_list]
            results = {}
            to_compute = []

            for token_tuple in input_keys:
                if not token_tuple:
                    raise ValueError("Token ids must not be empty")
                if self.cache is not None and token_tuple in self.cache:
                    results[token_tuple] = self.cache[token_tuple]
                elif token_tuple not in results:
                    to_compute.append(token_tuple)
                    results[token_tuple] = None

            if to_compute:
                requests = []
                for token_tuple in to_compute:
                    req = _make_request(token_tuple, lora_id=self._lora_id)
                    self._rid_to_token_ids[req.rid] = token_tuple
                    requests.append(req)

                for token_tuple, logprobs in self._batch_evaluate(requests):
                    results[token_tuple] = logprobs
                    if self.cache is not None:
                        self.cache[token_tuple] = logprobs

            return torch.stack([results[k] for k in input_keys])

        def _register(self, token_ids: Tuple[int], future: asyncio.Future):
            """Register a request with the SGLang inference engine.

            Args:
                token_ids (Tuple[int]): The token IDs of the request.
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

            req = _make_request(token_ids, lora_id=self._lora_id)
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

        def _batch_evaluate(self, requests: List[Request]):
            """Evaluate a batch of requests and return the token IDs and log probabilities."""
            if not requests:
                return  # pragma: no cover

            self.model.process_input_requests(requests)

            while batch := self.model.get_next_batch_to_run():
                with torch.inference_mode():
                    batch_result = self.model.run_batch(batch)
                    self.model.process_batch_result(batch, batch_result)
                    logprobs = torch.log_softmax(
                        batch_result.logits_output.next_token_logits, dim=-1
                    ).to("cpu")

                    for i, req in enumerate(batch.reqs):
                        if req.finished():
                            key = self._rid_to_token_ids.pop(req.rid, None)
                            if key is None:
                                continue  # pragma: no cover
                            yield key, logprobs[i]

        async def _background_loop(self):
            """Background task that processes queued requests from the queue."""
            assert self._queue is not None
            try:
                while True:
                    requests = await self._drain_queue()
                    for key, logprobs in self._batch_evaluate(requests):
                        waiters = self._pending.pop(key, [])
                        self._inflight.pop(key, None)
                        for f in waiters:
                            f.set_result(logprobs)

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
            if getattr(self, "model", None) is None:
                return  # pragma: no cover
            try:
                self.reset_async_queries()
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:  # pragma: no cover
                pass  # pragma: no cover

        def __del__(self):  # pragma: no cover
            """Clean up the SGLang inference engine when the instance is deleted."""
            self._cleanup_engine()
