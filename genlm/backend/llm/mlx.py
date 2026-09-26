import asyncio
import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch

from genlm.backend.cache import OutputCache
from genlm.backend.llm.base import (
    AsyncLM,
    UNKNOWN_ADAPTER,
    batch_abandoned,
    fail_futures,
)

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
        """MLX array as a torch tensor over the same buffer.

        bfloat16 is narrowed to float16 first: callers hand these to numpy, which has
        no bfloat16.
        """
        if a.dtype == mx.bfloat16:
            a = a.astype(mx.float16)
        return torch.from_dlpack(a)

    class _SlotPool:
        """The model's live KV rows: row ``i`` holds exactly the tokens in ``seqs[i]``.

        Attributes:
            seqs (list[list[int]]): Tokens held by each live row.
            cache (list|None): Per-layer MLX caches backing the rows.
        """

        def __init__(self, model, prefill_step_size):
            self.model = model
            self.prefill_step_size = prefill_step_size
            self.seqs = []
            self.cache = None

        def logits(self, prompts):
            """Final-position logits for ``prompts``, one row each.

            Leaves the pool holding exactly ``prompts``. A batch whose rows each extend
            a live row by the same non-empty delta continues that row's KV; anything
            else is prefilled.

            Args:
                prompts (list[list[int]]): Token ids each row must hold.

            Returns:
                (mx.array): ``[len(prompts), vocab]`` final-position logits.
            """
            sources, shared = zip(*map(self._longest_prefix_row, prompts))
            deltas = [p[n:] for p, n in zip(prompts, shared)]
            if self._can_extend(sources, deltas):
                return self.advance(list(sources), deltas)
            return self._prefill(prompts)

        def advance(self, sources, deltas):
            """Re-lay the rows as ``sources``, then forward one token block per row.

            A repeated index forks that row; an omitted one is dropped. A re-lay that
            needs a cache that cannot be row-sliced prefills the rows instead.

            Args:
                sources (list[int]): Live row each new row continues.
                deltas (list[list[int]]): Equal-length token block to append per row.

            Returns:
                (mx.array): ``[len(sources), vocab]`` final-position logits.
            """
            if not self._can_reorder(sources):
                return self._prefill(
                    [self.seqs[s] + d for s, d in zip(sources, deltas)]
                )
            self._gather(sources)
            return self._extend(deltas)

        def _can_reorder(self, sources):
            """Whether the rows can be re-laid as ``sources`` in place."""
            return list(sources) == list(range(len(self.seqs))) or all(
                hasattr(c, "filter") for c in self.cache
            )

        def _longest_prefix_row(self, prompt):
            """``(row, n)`` for the live row whose tokens are the longest strict prefix
            of ``prompt``, or ``(None, 0)``."""
            best, best_n = None, 0
            for i, seq in enumerate(self.seqs):
                n = len(seq)
                if best_n < n < len(prompt) and prompt[:n] == seq:
                    best, best_n = i, n
            return best, best_n

        def _can_extend(self, sources, deltas):
            """Whether every prompt extends a live row by one shared non-empty delta."""
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

        The model is wrapped once, at the first registration; an adapter is then only
        its ``lora_*`` arrays. The base set is the all-zero one the wrap starts from.

        Attributes:
            sets (dict): Name -> ``[(parameter path, array)]``, ``None`` being the base.
            layout (tuple|None): ``(num_layers, lora_parameters)`` the wrap used.
            installed (list|None): The weight set currently on the model, compared by
                identity; replace a set's list, never mutate it in place.
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
                # The fresh LoRA modules start in training mode, with dropout active.
                self.model.eval()
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
        """Asynchronous MLX-based language model wrapper.

        This class provides an async interface to MLX language models with
        automatic batching, caching, and KV cache management. It extends
        AsyncLM to provide efficient batched inference with prefix caching.

        The model automatically batches concurrent requests, continues the live
        KV cache when a batch extends the previous one, and optionally caches
        computed log probabilities for reuse.
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
                timeout (float, optional): Seconds to wait once for late concurrent
                    requests before processing a batch. Defaults to 0.
                prefill_step_size (int, optional): Number of tokens to process
                    per step during prompt prefilling.
                cache_size (int, optional): Maximum size of the output cache. If 0,
                    caching is disabled. Defaults to 0.
                cache_opts (dict, optional): Additional options to pass to the
                    [`OutputCache`][genlm.backend.cache.OutputCache] constructor.
                    Defaults to None (no extra options).
            """
            self.mlx_lm_model = mlx_lm_model
            self.timeout = timeout
            self.prefill_step_size = prefill_step_size
            self._adapters = _Adapters(mlx_lm_model)
            # KV rows never cross adapters, so each adapter keeps its own pool.
            self._slots = defaultdict(
                partial(_SlotPool, mlx_lm_model, prefill_step_size)
            )
            self.cache = (
                OutputCache(maxsize=cache_size, **(cache_opts or {}))
                if cache_size > 0
                else None
            )
            # mlx_lm.load returns a TokenizerWrapper; AsyncLM needs the HuggingFace
            # tokenizer it wraps.
            super().__init__(tokenizer=getattr(tokenizer, "_tokenizer", tokenizer))

        @classmethod
        def from_name(cls, model_name, **kwargs):
            """Create an `AsyncMlxLM` instance from a model name.

            Args:
                model_name (str): Name of the model to load. Can be a Hugging Face
                    model identifier or local path.
                **kwargs: Additional arguments passed to `AsyncMlxLM` constructor,
                    such as `timeout`, `prefill_step_size`, `cache_size`.

            Returns:
                AsyncMlxLM: An `AsyncMlxLM` instance with the loaded model and tokenizer.
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
            self._adapters.add(lora_name, lora_path)
            self.clear_cache()

        def remove_lora(self, lora_name):
            """Unregister ``lora_name``.

            Args:
                lora_name (str): Name of the adapter to remove.
            """
            self._adapters.remove(lora_name)
            self.clear_cache()

        def clear_cache(self):
            """Clear the output cache, the live KV rows, and the MLX device cache."""
            if self.cache is not None:
                self.cache.clear()
            self._slots.clear()
            mx.clear_cache()

        def _forward(self, prompts, lora_name):
            """Next-token log-probs under one adapter, ``[len(prompts), vocab]``."""
            with _wired(self.mlx_lm_model):
                self._adapters.select(lora_name)
                logits = self._slots[lora_name].logits(prompts).astype(mx.float32)
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                mx.eval(logprobs)
            return _to_torch(logprobs)

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
            """Resolve a batch of queries, deduplicating equal keys.

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
                fail_futures(queries, exc)
                raise
            except BaseException as exc:
                fail_futures(queries, batch_abandoned(exc))
                raise
            for key, waiting in futures.items():
                for future in waiting:
                    future.set_result(logprobs[key])

        async def next_token_logprobs(self, token_ids, lora_name=None):
            """Request log probabilities of next token. This version is asynchronous because it automatically batches concurrent requests; use with `await`.

            Args:
                token_ids (list[int]): a list of token ids, representing a prompt to the language model.
                lora_name (str, optional): Name of the LoRA adapter to use. Defaults to None (the base model).

            Returns:
                logprobs (torch.Tensor): a tensor of with the language model's log (normalized) probabilities for the next token following the prompt.
            """
            if not token_ids:
                raise ValueError("Token ids must not be empty")
            key = (tuple(token_ids), lora_name)
            if self.cache is not None and key in self.cache:
                return self.cache[key]
            future = asyncio.get_running_loop().create_future()
            batch = await self._join_batch([(key, future)], timeout=self.timeout)
            if batch is not None:
                self._batch_evaluate(batch)
            return await future

        def next_token_logprobs_sync(self, token_ids, lora_name=None):
            """Request log probabilities of next token synchronously.

            Args:
                token_ids (list[int]): A list of token IDs, representing a prompt to the language model.
                lora_name (str, optional): Name of the LoRA adapter to use. Defaults to None (the base model).

            Returns:
                (torch.Tensor): Normalized log probability tensor.
            """
            if not token_ids:
                raise ValueError("Token ids must not be empty")
            key = (tuple(token_ids), lora_name)
            return self._resolve([key])[key]

        def batch_next_token_logprobs_sync(self, token_ids_list, lora_name=None):
            """Next-token log-probs for each sequence, in one batched forward.

            Args:
                token_ids_list (list[list[int]]): Prompts' token ids.
                lora_name (str, optional): Name of the LoRA adapter to use. Defaults to None (the base model).

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
            """Sample from the language model.

            Args:
                prompt_token_ids (list[int]): The token IDs of the prompt to
                    start generation from.
                max_tokens (int): The maximum number of tokens to generate.
                eos_token_ids (list[int]): The token IDs that signal
                    end-of-sequence. Generation stops when one of these is
                    sampled.
                temperature (float, optional): The temperature to use for
                    sampling. Higher values make the distribution more uniform,
                    lower values make it more peaked. Defaults to 1.0.
                seed (int, optional): The seed for the random number generator.
                    If provided, sets the random seed before sampling.
                    Defaults to None.
                lora_name (str, optional): Name of the LoRA adapter to use. Defaults to None (the base model).

            Returns:
                (list[int]): The sampled token IDs.
            """
            self._adapters.select(lora_name)
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
