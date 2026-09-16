import asyncio
import copy
import weakref
import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch

from genlm.backend.cache import OutputCache
from genlm.backend.batching import batch_abandoned, join_batch
from genlm.backend.llm.base import AsyncLM, UNKNOWN_ADAPTER

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
        """Keep ``model``'s weights resident for the enclosed work.

        MLX streams are thread-affine, so the stream is resolved per call, never stored.
        """
        return wired_limit(model, [mx.default_stream(mx.default_device())])

    def _to_torch(a):
        """MLX array as a torch tensor over the same buffer."""
        return torch.from_dlpack(a)

    class _Row:
        """A live KV row: its tokens, and the chunk and position holding its cache."""

        __slots__ = ("tokens", "chunk", "i")

        def __init__(self, tokens, chunk, i):
            self.tokens = tokens
            self.chunk = chunk
            self.i = i

    class _Chunk:
        """One KV cache over a batch of rows; ``rows[i]`` is ``None`` once row ``i``
        has moved to a newer chunk."""

        __slots__ = ("cache", "rows", "used")

        def __init__(self, cache, used):
            self.cache = cache
            self.rows = []
            self.used = used

        @property
        def live(self):
            return [r for r in self.rows if r is not None]

    class _SlotPool:
        """The model's live KV rows, held in chunks.

        A batch assembles the rows it names into one chunk, forking a row it names
        more than once, and forwards it; rows it does not name stay where they are.
        Past ``max_rows`` live rows, the least recently written chunks are dropped.

        Attributes:
            chunks (list[_Chunk]): Row groups, oldest first.
        """

        def __init__(self, model, prefill_step_size, max_rows):
            self.model = model
            self.prefill_step_size = prefill_step_size
            self.max_rows = max_rows
            self.chunks = []
            self.batch = 0

        def reset(self):
            self.chunks = []

        def logits(self, prompts):
            """Final-position logits for ``prompts``, one row each.

            A prompt extending a live row continues it; prompts with no live prefix
            are prefilled. Continuations forward one block per delta length.

            Args:
                prompts (list[list[int]]): Token ids each row must hold.

            Returns:
                (mx.array): ``[len(prompts), vocab]`` final-position logits.
            """
            self.batch += 1
            live = [
                row for chunk in self.chunks for row in chunk.rows if row is not None
            ]
            src = [self._source(prompt, live) for prompt in prompts]
            by_delta = defaultdict(list)  # delta length (0: prefill) -> prompt indices
            for k, (prompt, row) in enumerate(zip(prompts, src)):
                by_delta[0 if row is None else len(prompt) - len(row.tokens)].append(k)
            groups = list(by_delta.items())
            blocks = []
            for g, (n, ks) in enumerate(groups):
                if n == 0:
                    blocks.append(self._prefill([prompts[k] for k in ks]))
                else:
                    keep = {src[k] for _, later in groups[g + 1 :] for k in later}
                    blocks.append(
                        self.advance(
                            [src[k] for k in ks], [prompts[k][-n:] for k in ks], keep
                        )
                    )
            self._evict()
            if len(blocks) == 1:
                return blocks[0]
            out = [None] * len(prompts)
            for (_, ks), block in zip(groups, blocks):
                for k, row in zip(ks, block):
                    out[k] = row
            return mx.stack(out)

        def advance(self, sources, deltas, keep=frozenset()):
            """Forward one equal-length token block per named row.

            Args:
                sources (list[_Row]): The row each new row continues.
                deltas (list[list[int]]): Equal-length token block to append per row.
                keep (set[_Row]): Sources a later pass still needs; they are copied,
                    never moved.

            Returns:
                (mx.array): ``[len(sources), vocab]`` final-position logits.
            """
            chunk = self._assemble(sources, keep)
            if chunk is None:
                return self._prefill([r.tokens + d for r, d in zip(sources, deltas)])
            logits = self.model(mx.array(deltas, mx.int32), cache=chunk.cache)[:, -1, :]
            for row, delta in zip(chunk.rows, deltas):
                row.tokens.extend(delta)
            chunk.used = self.batch
            return logits

        def _source(self, prompt, live):
            """The row in ``live`` whose tokens are the longest strict prefix of
            ``prompt``, or ``None``."""
            best = None
            for row in live:
                n = len(row.tokens)
                if (
                    (best is None or len(best.tokens) < n)
                    and n < len(prompt)
                    and row.tokens[-1] == prompt[n - 1]
                    and prompt[:n] == row.tokens
                ):
                    best = row
            return best

        def _assemble(self, sources, keep):
            """The chunk laid out as ``sources``, ready to forward.

            A batch naming one chunk's rows in place forwards that chunk as is.
            Otherwise the named rows' caches are gathered into a new chunk: a source
            named once moves there, one named again or in ``keep`` is forked.
            ``None`` when the caches cannot be gathered.
            """
            chunk = sources[0].chunk
            if sources == chunk.rows and not keep.intersection(sources):
                return chunk
            involved = list(dict.fromkeys(r.chunk for r in sources))
            if not all(
                hasattr(c, "filter") and hasattr(c, "extend")
                for ch in involved
                for c in ch.cache
            ):
                return None
            laid, caches = [], None
            for ch in involved:
                picks = [k for k, r in enumerate(sources) if r.chunk is ch]
                idx = mx.array([sources[k].i for k in picks], mx.int32)
                part = [copy.copy(c) for c in ch.cache]
                for c in part:
                    c.filter(idx)
                if caches is None:
                    caches = part
                else:
                    for c, other in zip(caches, part):
                        c.extend(other)
                laid.extend(picks)
            if laid != list(range(len(sources))):
                order = mx.array([laid.index(k) for k in range(len(sources))], mx.int32)
                for c in caches:
                    c.filter(order)
            new = _Chunk(caches, self.batch)
            moved = set()
            for i, row in enumerate(sources):
                if row in keep or row in moved:
                    new.rows.append(_Row(list(row.tokens), new, i))
                else:
                    row.chunk.rows[row.i] = None
                    row.chunk, row.i = new, i
                    new.rows.append(row)
                    moved.add(row)
            self.chunks = [c for c in self.chunks if c.live] + [new]
            return new

        def _prefill(self, prompts):
            """Left-padded batched prefill of ``prompts`` into a new chunk."""
            width = max(map(len, prompts))
            cache = _make_cache(
                self.model, [width - len(p) for p in prompts], max_kv_size=None
            )
            x = _left_pad_prompts(prompts, max_length=width)
            while x.shape[1] > 1:
                n = min(self.prefill_step_size, x.shape[1] - 1)
                self.model(x[:, :n], cache=cache)
                mx.eval([c.state for c in cache])
                x = x[:, n:]
            logits = self.model(x, cache=cache)[:, -1, :]
            chunk = _Chunk(cache, self.batch)
            chunk.rows = [_Row(list(p), chunk, i) for i, p in enumerate(prompts)]
            self.chunks.append(chunk)
            return logits

        def _evict(self):
            """Drop the least recently written chunks while live rows exceed
            ``max_rows``."""
            total = sum(len(c.live) for c in self.chunks)
            while total > self.max_rows and len(self.chunks) > 1:
                oldest = min(self.chunks, key=lambda c: c.used)
                self.chunks.remove(oldest)
                total -= len(oldest.live)

    class _Adapters:
        """LoRA weight sets over one model.

        MLX attaches adapters to the model itself, so the model is wrapped once, at the
        first registration, and an adapter is then only the ``lora_*`` arrays to
        install; adapters share the base weights rather than copying them. The base set
        is the all-zero one the wrap starts from, leaving base forwards bit-exact.

        Attributes:
            sets (dict): Name -> ``[(parameter path, array)]``, ``None`` being the base.
            layout (tuple|None): ``(num_layers, lora_parameters)`` the wrap used.
            installed (list|None): The weight set currently on the model, held by
                identity; `add` rebinds a name to a fresh list, so a rebind invalidates
                it.
        """

        def __init__(self, model):
            self.model = model
            self.sets = {}
            self.layout = None
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
                raise ValueError(UNKNOWN_ADAPTER.format(name))

        def select(self, name):
            """Install ``name``'s weights; ``None`` restores the base."""
            if name is not None and name not in self.sets:
                raise ValueError(UNKNOWN_ADAPTER.format(name))
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
            max_rows=128,
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
                max_rows (int, optional): Live KV rows kept per adapter; the least
                    recently used are evicted past it. Defaults to 128.
                cache_size (int, optional): Maximum size of the output cache. If 0,
                    caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the
                    [`OutputCache`][genlm.backend.cache.OutputCache] constructor.
                    Defaults to None (no extra options).
            """
            self.mlx_lm_model = mlx_lm_model
            probe = mx.zeros((1,))
            mx.eval(probe)
            self.device = _to_torch(probe).device
            self._batches = weakref.WeakKeyDictionary()  # the batch window's store
            self.timeout = timeout
            self.prefill_step_size = prefill_step_size
            self.adapters = _Adapters(mlx_lm_model)
            # KV rows never cross adapters, so each adapter keeps its own pool.
            self.slots = defaultdict(
                partial(_SlotPool, mlx_lm_model, prefill_step_size, max_rows)
            )
            self.cache = (
                OutputCache(maxsize=cache_size, **(cache_opts or {}))
                if cache_size > 0
                else None
            )
            # mlx_lm.load returns a TokenizerWrapper, which adds streaming detokenize
            # but is not callable like the HuggingFace tokenizer AsyncLM expects.
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
            """Register the adapter at ``lora_path`` under ``lora_name``.

            Re-registering a name rebinds it. Every adapter on a model must share the
            layout the first one wrapped it with.

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

        def _forward(self, prompts, lora_name):
            """Next-token log-probs under one adapter, ``[len(prompts), vocab]``."""
            with _wired(self.mlx_lm_model):
                self.adapters.select(lora_name)
                logits = self.slots[lora_name].logits(prompts)
                mx.eval(logits)
            return self._normalize(_to_torch(logits))

        def _resolve(self, keys):
            """``{key: logprobs}`` for distinct ``(context, lora_name)`` keys.

            Uncached keys are forwarded one batch per adapter, then memoized.
            """
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

        def _batch_evaluate(self, queries):
            """Resolve a batch in one forward, deduplicating equal prompts.

            Every future gets its row or a failure; a failed forward would
            otherwise leave the co-callers awaiting.
            """
            if not queries:
                return
            futures = defaultdict(list)
            for key, future in queries:
                futures[key].append(future)
            try:
                logprobs = self._resolve(list(futures))
            except Exception as exc:
                self._fail_all(futures, exc)
                raise
            except BaseException as exc:
                self._fail_all(futures, batch_abandoned(exc))
                raise
            for key, waiting in futures.items():
                for future in waiting:
                    future.set_result(logprobs[key])

        @staticmethod
        def _fail_all(futures, exc):
            for waiting in futures.values():
                for future in waiting:
                    if not future.done():
                        future.set_exception(exc)

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
            cohort = await join_batch(
                self._batches, [(key, future)], linger=self.timeout
            )
            if cohort is not None:
                self._batch_evaluate(cohort)
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
