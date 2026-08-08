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
            non-empty delta continues that row's KV; anything else is prefilled."""
            sources, shared = zip(*map(self._source, prompts))
            deltas = [p[n:] for p, n in zip(prompts, shared)]
            if self._continues(sources, deltas):
                self._gather(list(sources))
                return self._extend(deltas)
            return self._prefill(prompts)

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
            The deltas must share one non-zero length so they forward as one block, and
            a reordering needs row-sliceable caches (a recurrent state has none)."""
            if self.cache is None or None in sources:
                return False
            if len({len(d) for d in deltas}) != 1 or not deltas[0]:
                return False
            return list(sources) == list(range(len(self.seqs))) or all(
                hasattr(c, "filter") for c in self.cache
            )

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
            self.cache = _make_cache(self.model, [width - len(p) for p in prompts])
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
        first registration, and a lane is then only the ``lora_*`` arrays to install --
        adapters share the base weights rather than copying them. The base lane is the
        all-zero set the wrap starts from, which leaves base forwards bit-exact.
        """

        def __init__(self, model):
            self.model = model
            self.sets = {}  # name -> [(parameter path, array)], None being the base
            self.layout = None  # (num_layers, lora_parameters) the wrap used

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
            if self.layout is not None:
                self.model.update(tree_unflatten(self.sets[name]))

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
            batch_size=5,
            timeout=0.001,
            prefill_step_size=2048,
            cache_size=0,
            cache_opts=None,
        ):
            """Initialize an `AsyncMlxLM` instance.

            Args:
                mlx_lm_model: The MLX language model instance.
                tokenizer: The tokenizer for encoding/decoding text.
                batch_size (int, optional): Maximum number of queries to batch together.
                timeout (float, optional): Seconds to wait before running a short batch.
                prefill_step_size (int, optional): Tokens per prefill chunk.
                cache_size (int, optional): Maximum size of the output cache. If 0,
                    caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the
                    [`OutputCache`][genlm.backend.cache.OutputCache] constructor.
                    Defaults to None (no extra options).
            """
            self.mlx_lm_model = mlx_lm_model
            self.batch_size = batch_size
            self.timeout = timeout
            self.timer = None
            self.queries = []
            self.adapters = _Adapters(mlx_lm_model)
            # KV rows never cross adapters, so each lane keeps its own pool.
            self.slots = defaultdict(
                partial(_SlotPool, mlx_lm_model, prefill_step_size)
            )
            self.cache = (
                OutputCache(maxsize=cache_size, **(cache_opts or {}))
                if cache_size > 0
                else None
            )
            self.generation_stream = mx.new_stream(mx.default_device())
            super().__init__(tokenizer=tokenizer)

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
            """Drop the memoized log-probs and every lane's live KV rows."""
            if self.cache is not None:
                self.cache.clear()
            self.slots.clear()
            mx.clear_cache()

        def reset_async_queries(self):
            """Drop queued queries. Use after an exception left them unresolved."""
            self.queries = []

        def _forward(self, prompts, lora_name):
            """Next-token log-probs under one adapter, ``[len(prompts), vocab]``."""
            with wired_limit(self.mlx_lm_model, [self.generation_stream]):
                self.adapters.select(lora_name)
                logits = self.slots[lora_name].logits(prompts)
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

        def _add_query(self, key, future):
            """Queue a query, running the batch once it is full or the timer fires.

            The timer is armed on the empty-to-nonempty transition, so a steady trickle
            of queries cannot starve the batch.
            """
            self.queries.append((key, future))
            if len(self.queries) >= self.batch_size:
                if self.timer:
                    self.timer.cancel()
                    self.timer = None
                self._batch_evaluate()
            elif self.timer is None:
                self.timer = asyncio.get_running_loop().call_later(
                    self.timeout, self._batch_evaluate
                )

        def _batch_evaluate(self):
            """Resolve every queued query in one forward, deduplicating equal prompts."""
            self.timer = None
            queries, self.queries = self.queries, []
            if not queries:
                return
            futures = defaultdict(list)
            for key, future in queries:
                futures[key].append(future)
            logprobs = self._resolve(list(futures))
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
            self._add_query(key, future)
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
