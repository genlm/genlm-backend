import asyncio
import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch

from genlm.backend.cache import OutputCache
from genlm.backend.llm.base import AsyncLM

try:
    import mlx.core as mx
    import mlx_lm
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.generate import (
        _left_pad_prompts,
        _make_cache,
        generate_step,
        wired_limit,
    )
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import linear_to_lora_layers

    HAS_MLX = True
except ImportError:  # pragma: no cover
    HAS_MLX = False  # pragma: no cover


if not HAS_MLX:

    class AsyncMlxLM:  # pragma: no cover
        """Placeholder class when MLX is not installed."""

        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "To use the MLX-based AsyncLM model, "
                "install the package with 'pip install genlm-backend[mlx]'"
            )

        @classmethod
        def from_name(cls, *args, **kwargs):  # pragma: no cover
            raise ImportError(
                "To use the MLX-based AsyncLM model, "
                "install the package with 'pip install genlm-backend[mlx]'"
            )

else:

    def _wired(model):
        """Keep ``model``'s weights resident for the enclosed work. The stream is
        resolved per call, never stored -- MLX streams are thread-affine."""
        return wired_limit(model, [mx.default_stream(mx.default_device())])

    def _to_torch(a):
        """MLX array as a torch tensor over the same buffer. bfloat16 is narrowed
        first: callers hand these to numpy, which has no bfloat16."""
        if a.dtype == mx.bfloat16:
            a = a.astype(mx.float16)
        return torch.from_dlpack(a)

    class _SlotPool:
        """The model's live KV rows: row ``i`` holds exactly the tokens in ``seqs[i]``.

        Rows are extended, gathered, or replaced wholesale -- never rebuilt from stored
        pieces, so a batch that walks forward from the previous one copies nothing. A
        repeated source in a gather forks that row; an omitted one is dropped.
        """

        def __init__(self, model, prefill_step_size):
            self.model = model
            self.prefill_step_size = prefill_step_size
            self.seqs = []
            self.cache = None

        def reset(self):
            self.seqs, self.cache = [], None

        def logits(self, prompts):
            """Final-position logits for ``prompts``, one row each, leaving the pool
            holding exactly them. A batch whose rows each extend a live row by the same
            non-empty delta continues that row's KV; anything else is prefilled.

            Discovers what to reuse. A caller that already knows its own ancestry calls
            :meth:`advance` instead."""
            sources, shared = zip(*map(self._source, prompts))
            deltas = [p[n:] for p, n in zip(prompts, shared)]
            if self._continues(sources, deltas):
                return self.advance(list(sources), deltas)
            return self._prefill(prompts)

        def advance(self, sources, deltas):
            """Re-lay the rows as ``sources`` -- repeat an index to fork that row, omit
            one to drop it -- then forward one equal-length token block per row.

            A reorder needs row-sliceable caches; a recurrent state has none, so there the
            rows are rebuilt by prefilling what they would have held."""
            if not self._relayable(sources):
                return self._prefill(
                    [self.seqs[s] + d for s, d in zip(sources, deltas)]
                )
            self._gather(sources)
            return self._extend(deltas)

        def _relayable(self, sources):
            """Whether the rows can be re-laid as ``sources`` in place."""
            return list(sources) == list(range(len(self.seqs))) or all(
                hasattr(c, "filter") for c in self.cache
            )

        def _source(self, prompt):
            """``(row, n)`` for the live row whose tokens are the longest strict prefix
            of ``prompt``, or ``(None, 0)``."""
            best, best_n = None, 0
            for i, seq in enumerate(self.seqs):
                n = len(seq)
                if best_n < n < len(prompt) and prompt[:n] == seq:
                    best, best_n = i, n
            return best, best_n

        def _continues(self, sources, deltas):
            """Whether the pool can carry this batch forward instead of reprefilling.
            The deltas must share one non-zero length so they forward as one block;
            whether the rows can be re-laid is :meth:`advance`'s to decide."""
            if self.cache is None or None in sources:
                return False
            return len({len(d) for d in deltas}) == 1 and bool(deltas[0])

        def _gather(self, sources):
            """Rebuild the rows from existing ones: new row ``i`` continues ``sources[i]``."""
            if sources == list(range(len(self.seqs))):
                return
            idx = mx.array(sources, mx.int32)
            for c in self.cache:
                c.filter(idx)
            self.seqs = [list(self.seqs[s]) for s in sources]

        def _extend(self, deltas):
            """Forward one equal-length token block per row."""
            logits = self.model(mx.array(deltas, mx.int32), cache=self.cache)[:, -1, :]
            for seq, delta in zip(self.seqs, deltas):
                seq.extend(delta)
            return logits

        def _prefill(self, prompts):
            """Left-padded batched prefill, replacing the pool."""
            width = max(map(len, prompts))
            self.cache = _make_cache(
                self.model, [width - len(p) for p in prompts], max_kv_size=None
            )
            x = _left_pad_prompts(prompts, max_length=width)
            while x.shape[1] > 1:
                n = min(self.prefill_step_size, x.shape[1] - 1)
                self.model(x[:, :n], cache=self.cache)
                mx.eval([c.state for c in self.cache])
                x = x[:, n:]
            self.seqs = [list(p) for p in prompts]
            return self.model(x, cache=self.cache)[:, -1, :]

    class _Adapters:
        """LoRA weight sets over one model.

        MLX attaches adapters to the model itself, so the model is wrapped once, on the
        first registration, and an adapter is then only the ``lora_*`` arrays to install --
        adapters share the base weights rather than copying them. The base is the
        all-zero set the wrap starts from, which leaves base forwards bit-exact.
        """

        def __init__(self, model):
            self.model = model
            self.sets = {}  # name -> [(parameter path, array)], None being the base
            self.layout = None  # (num_layers, lora_parameters) the wrap used
            # The weight set currently on the model, by identity. ``add`` rebinds a name
            # to a fresh list, so a rebind invalidates itself.
            self.installed = None

        def add(self, name, path):
            """Register the adapter at ``path`` under ``name``, rebinding if it exists."""
            path = Path(path)
            with open(path / "adapter_config.json") as fid:
                config = json.load(fid)
            kind = config.get("fine_tune_type", "lora")
            if kind != "lora":
                raise ValueError(f"MLX adapters must be 'lora', not {kind!r}")
            layout = (config["num_layers"], config["lora_parameters"])
            if self.layout is None:
                linear_to_lora_layers(self.model, *layout)
                self.layout = layout
                self.sets[None] = self._installed()
            elif layout != self.layout:
                raise ValueError(
                    f"adapter {name!r} wants layout {layout}, but the model is already "
                    f"wrapped for {self.layout}; MLX wraps a model once, so every "
                    f"adapter on it must share a layout"
                )
            weights = mx.load(str(path / "adapters.safetensors"))
            self.sets[name] = [(k, v) for k, v in weights.items() if "lora_" in k]

        def remove(self, name):
            if name is None or self.sets.pop(name, None) is None:
                raise ValueError(f"no adapter named {name!r}")

        def select(self, name):
            """Install ``name``'s weights; ``None`` restores the base."""
            if name is not None and name not in self.sets:
                raise ValueError(f"no adapter named {name!r}")
            if self.layout is None:
                return
            weights = self.sets[name]
            if weights is not self.installed:
                self.model.update(tree_unflatten(weights))
                self.installed = weights

        def _installed(self):
            """The adapter arrays currently on the model."""
            return [
                (k, v)
                for k, v in tree_flatten(self.model.trainable_parameters())
                if "lora_" in k
            ]

    class AsyncMlxLM(AsyncLM):
        """Asynchronous MLX language model.

        Concurrent requests are batched, and a batch whose prompts each walk forward
        from the previous batch's continues the live KV rather than reprefilling.
        Next-token log-probs are memoized per exact context.
        """

        def __init__(
            self,
            mlx_lm_model,
            tokenizer,
            timeout=0.0,
            prefill_step_size=2048,
            cache_size=0,
            cache_opts=None,
        ):
            """Initialize an `AsyncMlxLM` instance.

            Args:
                mlx_lm_model: The MLX language model instance.
                tokenizer: The tokenizer for encoding/decoding text.
                timeout (float, optional): Cooperative linger in seconds spent once
                    per batch window, letting late concurrent callers join. Defaults to 0.
                prefill_step_size (int, optional): Tokens per prefill chunk.
                cache_size (int, optional): Maximum size of the output cache. If 0,
                    caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the
                    [`OutputCache`][genlm.backend.cache.OutputCache] constructor.
                    Defaults to None (no extra options).
            """
            self.mlx_lm_model = mlx_lm_model
            self.timeout = timeout
            self.prefill_step_size = prefill_step_size
            self.queries = []
            self._window_armed = False
            self.adapters = _Adapters(mlx_lm_model)
            # KV rows never cross adapters, so each adapter keeps its own pool.
            self.slots = defaultdict(
                partial(_SlotPool, mlx_lm_model, prefill_step_size)
            )
            self.cache = (
                OutputCache(maxsize=cache_size, **(cache_opts or {}))
                if cache_size > 0
                else None
            )
            # mlx_lm.load hands back a TokenizerWrapper: it adds streaming detokenize but
            # is not callable like a HuggingFace tokenizer, so unwrap to the real one.
            super().__init__(tokenizer=getattr(tokenizer, "_tokenizer", tokenizer))

        @classmethod
        def from_name(cls, model_name, **kwargs):
            """Create an `AsyncMlxLM` from a model name or local path.

            Args:
                model_name (str): HuggingFace model identifier or local path.
                **kwargs: Additional arguments passed to the constructor.

            Returns:
                AsyncMlxLM: The loaded model.
            """
            model, tokenizer = mlx_lm.load(model_name)
            return cls(model, tokenizer, **kwargs)

        def add_new_lora(self, lora_path, lora_name="lora_1"):
            """Register the adapter at ``lora_path`` under ``lora_name``, rebinding the
            name if it already exists. Every adapter on a model must share the layout
            the first one wrapped it with.

            Args:
                lora_path (str): Directory holding ``adapter_config.json`` and
                    ``adapters.safetensors``.
                lora_name (str, optional): Name to select the adapter by.
            """
            self.adapters.add(lora_name, lora_path)
            self.clear_cache()

        def remove_lora(self, lora_name):
            """Unregister ``lora_name``.

            Args:
                lora_name (str): Name of the adapter to remove.
            """
            self.adapters.remove(lora_name)
            self.clear_cache()

        def clear_cache(self):
            """Drop the memoized log-probs and every pool's live KV rows."""
            if self.cache is not None:
                self.cache.clear()
            self.slots.clear()
            mx.clear_cache()

        def reset_async_queries(self):
            """Drop queued queries. Use after an exception left them unresolved."""
            self.queries = []

        def _forward(self, prompts, lora_name):
            """Next-token log-probs under one adapter, ``[len(prompts), vocab]``."""
            with _wired(self.mlx_lm_model):
                self.adapters.select(lora_name)
                logits = self.slots[lora_name].logits(prompts).astype(mx.float32)
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                mx.eval(logprobs)
            return _to_torch(logprobs)

        def _resolve(self, keys):
            """``{key: logprobs}`` for distinct ``(context, lora_name)`` keys, forwarding
            the uncached ones one batch per adapter and memoizing them."""
            out, todo = {}, defaultdict(list)
            for key in keys:
                if self.cache is not None and key in self.cache:
                    out[key] = self.cache[key]
                else:
                    todo[key[1]].append(key)
            for lora_name, batch in todo.items():
                logprobs = self._forward([list(k[0]) for k in batch], lora_name)
                for row, key in enumerate(batch):
                    out[key] = logprobs[row]
                    if self.cache is not None:
                        self.cache[key] = logprobs[row]
            return out

        async def _collect(self):
            """Hold the batch window open: return once a full event-loop pass
            adds no new query (so a whole concurrent gather lands in one batch),
            after one cooperative ``timeout`` linger for late callers.

            Run by the window's first caller, never a background task: window
            state must not outlive the loop the callers are on."""
            lingered = not self.timeout
            while True:
                n = len(self.queries)
                await asyncio.sleep(0)
                if len(self.queries) > n:
                    continue
                if lingered:
                    return
                lingered = True
                await asyncio.sleep(self.timeout)

        def _batch_evaluate(self):
            """Resolve every queued query in one forward, deduplicating equal
            prompts. Every future gets its row or the exception — a failed
            forward must not leave co-window callers waiting forever."""
            queries, self.queries = self.queries, []
            if not queries:
                return
            futures = defaultdict(list)
            for key, future in queries:
                futures[key].append(future)
            try:
                logprobs = self._resolve(list(futures))
            except BaseException as exc:
                for waiting in futures.values():
                    for future in waiting:
                        if not future.done():
                            future.set_exception(exc)
                raise
            for key, waiting in futures.items():
                for future in waiting:
                    future.set_result(logprobs[key])

        async def next_token_logprobs(self, token_ids, lora_name=None):
            """Next-token log-probs for `token_ids`, batched with concurrent requests.

            Args:
                token_ids (list[int]): A prompt's token ids.
                lora_name (str, optional): Adapter to forward under (``None`` = base).

            Returns:
                (torch.Tensor): Normalized log-probabilities over the next token.
            """
            if not token_ids:
                raise ValueError("Token ids must not be empty")
            key = (tuple(token_ids), lora_name)
            if self.cache is not None and key in self.cache:
                return self.cache[key]
            future = asyncio.get_running_loop().create_future()
            self.queries.append((key, future))
            if not self._window_armed:
                self._window_armed = True
                try:
                    await self._collect()
                finally:
                    self._window_armed = False
                self._batch_evaluate()
            return await future

        def next_token_logprobs_sync(self, token_ids, lora_name=None):
            """Next-token log-probs for `token_ids`, evaluated immediately.

            Args:
                token_ids (list[int]): A prompt's token ids.
                lora_name (str, optional): Adapter to forward under (``None`` = base).

            Returns:
                (torch.Tensor): Normalized log-probabilities over the next token.
            """
            if not token_ids:
                raise ValueError("Token ids must not be empty")
            key = (tuple(token_ids), lora_name)
            return self._resolve([key])[key]

        def batch_next_token_logprobs_sync(self, token_ids_list, lora_name=None):
            """Next-token log-probs for each sequence, in one batched forward.

            Args:
                token_ids_list (list[list[int]]): Prompts' token ids.
                lora_name (str, optional): Adapter to forward under (``None`` = base).

            Returns:
                (torch.Tensor): ``[len(token_ids_list), vocab]`` log-probabilities.
            """
            if any(not ids for ids in token_ids_list):
                raise ValueError("Token ids must not be empty")
            keys = [(tuple(ids), lora_name) for ids in token_ids_list]
            logprobs = self._resolve(list(dict.fromkeys(keys)))
            return torch.stack([logprobs[k] for k in keys])

        async def sample(
            self,
            prompt_token_ids,
            max_tokens,
            eos_token_ids,
            temperature=1.0,
            seed=None,
            lora_name=None,
        ):
            """Sample a continuation from the model.

            Args:
                prompt_token_ids (list[int]): Token ids to continue from.
                max_tokens (int): Maximum number of tokens to generate.
                eos_token_ids (list[int]): Token ids that stop generation.
                temperature (float, optional): Logit rescaling; higher is more uniform.
                seed (int, optional): Seed for the random number generator.
                lora_name (str, optional): Adapter to sample under (``None`` = base).

            Returns:
                (list[int]): The sampled token ids, excluding any EOS.
            """
            self.adapters.select(lora_name)
            if seed is not None:
                mx.random.seed(seed)

            generated = []
            for token, _ in generate_step(
                mx.array(prompt_token_ids),
                self.mlx_lm_model,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature),
            ):
                if token in eos_token_ids:
                    break
                generated.append(token)
            return generated
