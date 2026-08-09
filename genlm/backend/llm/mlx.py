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
        resolved per call, never stored -- MLX streams are thread-affine and the burst
        runs on a worker thread."""
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

        def rows_holding(self, prefixes):
            """The row holding exactly each prefix, ``None`` where none does."""
            index = {tuple(seq): i for i, seq in enumerate(self.seqs)}
            return [index.get(tuple(p)) for p in prefixes]

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

    class _Ledger:
        """``run_burst``'s group ledger: one control handle = K lane prompts, held in
        batch order.

        MLX schedules nothing, so every ordered handle forwards every step -- there is no
        partially scheduled group to stall and no engine request id to track.
        """

        def __init__(self):
            self.order = []  # control handles, batch order
            self.prompts = {}  # handle -> committed token ids per lane
            self.loras = []  # adapter per lane, snapshotted at the first add
            self.laid = None  # the order the pools' rows currently hold

        def drain(self, control):
            """Apply the control's abort and add streams."""
            for handle in control.drain_aborts():
                if self.prompts.pop(handle, None) is not None:
                    self.order.remove(handle)
            for handle, prompts, loras in control.drain_adds():
                if not self.loras:
                    self.loras = list(loras)
                self.prompts[handle] = [list(p) for p in prompts]
                self.order.append(handle)

        def commit(self, tokens):
            """Record each ordered handle's drawn token against all K of its lanes."""
            for handle, token in zip(self.order, tokens):
                for prompt in self.prompts[handle]:
                    prompt.append(token)

        def sources(self, pool):
            """The pool row each ordered handle continues, or ``None`` to rediscover.
            Records the layout it hands out, so call it once per step.

            A handle re-added mid-burst sits exactly one token past the pool -- the
            control queues adds only after the drawn token is banked -- so looking its
            prompt's head up by exact match both finds the ancestor and proves it. A
            miss (a step that committed more than one item, a ragged unit boundary)
            falls back to :meth:`_SlotPool.logits`.
            """
            rows = (
                list(range(len(self.order)))
                if self.laid == self.order
                else pool.rows_holding([self.prompts[h][0][:-1] for h in self.order])
            )
            # Either branch the caller takes leaves the pools holding ``order``.
            self.laid = list(self.order)
            return None if None in rows else rows

    class AsyncMlxLM(AsyncLM):
        """Asynchronous MLX language model.

        Concurrent requests are batched, and a batch whose prompts each walk forward
        from the previous batch's continues the live KV rather than reprefilling.
        Next-token log-probs are memoized per exact context.
        """

        supports_burst = True  # has run_burst; drives the engine-native burst lane

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
            self.prefill_step_size = prefill_step_size
            self.timer = None
            self.queries = []
            self.adapters = _Adapters(mlx_lm_model)
            # KV rows never cross adapters, so each lane keeps its own pool.
            self.slots = defaultdict(
                partial(_SlotPool, mlx_lm_model, prefill_step_size)
            )
            # A burst's lanes hold their own rows: one pool per lane, kept across bursts
            # so a unit-grain round continues instead of reprefilling.
            self.burst_slots = []
            self.burst_lanes = []  # the lane adapters `burst_slots` was built for
            self.cache = (
                OutputCache(maxsize=cache_size, **(cache_opts or {}))
                if cache_size > 0
                else None
            )
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
            self.burst_slots, self.burst_lanes = [], []
            mx.clear_cache()

        def reset_async_queries(self):
            """Drop queued queries. Use after an exception left them unresolved."""
            self.queries = []

        def run_burst(self, control, max_steps):
            """Run one decode burst driven by an SMC ``control`` (an ``EngineControl``).

            The control owns which groups exist (its abort/add streams) and draws every
            step; this is only the loop and the KV. No SMC logic, no EOS stop, no return
            value -- committed tokens are tracked control-side.

            Args:
                control (EngineControl): the SMC control object.
                max_steps (int): maximum decode steps for the burst. A global cap, where
                    the vLLM arm's is per request; length termination is control-side.
            """
            if self.burst_active:
                raise RuntimeError(
                    "a burst already owns this model's decode loop; the two would share "
                    "one KV pool per lane and overwrite each other's rows"
                )
            ledger = _Ledger()
            self.burst_active = True
            try:
                with _wired(self.mlx_lm_model):
                    ledger.drain(control)  # seed the initial population
                    for _ in range(max_steps):
                        if not ledger.order:
                            break
                        drawn = control.draw(self._burst_step(ledger), ledger.order)
                        ledger.commit(
                            drawn.tolist() if torch.is_tensor(drawn) else list(drawn)
                        )
                        ledger.drain(control)
                    # Let the control settle what it deferred past the last step, then
                    # drain whatever that flags.
                    control.on_burst_end()
                    ledger.drain(control)
            finally:
                self.burst_active = False

        def _burst_step(self, ledger):
            """One decode step across every lane: ``[G, K, vocab]`` fp32, on the host.

            fp32 is taken in MLX rather than on the torch side, which sidesteps
            ``_to_torch``'s bfloat16 narrowing (that exists for the numpy-facing slow
            lane alone).

            The rows land on the CPU because the control composes them against
            context-only potentials whose own rows are float64 numpy, and Metal has no
            float64 -- a device row makes ``Product._compose`` raise. The slow lane
            serves host rows for the same reason, so this also keeps the two paths
            promoting through the same dtype.
            """
            if self.burst_lanes != ledger.loras:
                self.burst_slots = [
                    _SlotPool(self.mlx_lm_model, self.prefill_step_size)
                    for _ in ledger.loras
                ]
                self.burst_lanes = list(ledger.loras)
            # One source list for every lane. Lanes carry DIFFERENT prompts (a LoRA view
            # and a base view have their own prefixes), but two rows of one group share
            # each lane's prefix and a crossing only reindexes within a group -- so a
            # row's ancestor is the same row in every lane. Gathering every pool with
            # this one list is also what holds them in a single row order.
            sources = ledger.sources(self.burst_slots[0])
            per_lane = []
            for lane, pool in enumerate(self.burst_slots):
                self.adapters.select(ledger.loras[lane])
                prompts = [ledger.prompts[h][lane] for h in ledger.order]
                per_lane.append(
                    pool.advance(sources, [p[-1:] for p in prompts])
                    if sources is not None
                    else pool.logits(prompts)
                )
            warm = mx.stack(per_lane, axis=1).astype(mx.float32)
            mx.eval(warm)
            return torch.from_dlpack(warm).cpu()

        def _forward(self, prompts, lora_name):
            """Next-token log-probs under one adapter, ``[len(prompts), vocab]``."""
            if self.burst_active:
                raise RuntimeError(
                    "next-token log-probs were requested while a burst owns this "
                    "model's decode loop; the forward would reprefill into the pool the "
                    "burst is driving. This leaf should have been an injected lane."
                )
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
