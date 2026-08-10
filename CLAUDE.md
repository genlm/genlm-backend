# genlm-backend

vLLM/MLX backend layer for genlm-control. It is a batched next-token-logprobs server:
contexts in, rows out. Everything that makes SMC fast lives here, under an unchanged
`next_token_logprobs` surface — **no SMC logic (no loop, ESS, resample, or weight
bookkeeping) belongs in this repo.**

**Design in flight (2026-08-09, branch `shepard/lane-server`).** Engine serving was just
rewritten to ride vLLM's own scheduler, replacing the lane/burst machinery. Validated by
`tests/probe_vllm_server.py` on an A100 and by control's gates, but it has NOT had a quality
pass: no perf number since the rewrite, eviction policy provisional, `AsyncScheduler`
internals leaned on in a few places. Re-read the code before building on it.

## Environment

- No `.venv` here; this repo is editable-installed into genlm-control's. vLLM is
  remote-only (never on macOS).
- **rsync this repo separately from control, and push branches in lockstep.** A tree with
  only one side synced runs a serving contract the caller does not speak.
- Pre-commit hooks (ruff + ruff-format) rewrite files on commit; re-`git add` and re-commit
  if a commit aborts. genlm-control is NOT ruff-formatted — never carry `ruff format` across.
- Engine runs **in-process** (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so control-side objects
  call into it directly. `VLLM_USE_V2_MODEL_RUNNER=1` is forced: the capture shim installs
  through MRv2's `ModelState.custom_sampler` hook, so there is one runner and one seam.
  `VLLM_USE_FLASHINFER_SAMPLER=0` because flashinfer JIT-compiles and needs a `CUDA_HOME`
  the compute nodes lack. All three are set at import, before vllm loads. vLLM pin:
  `>=0.26,<0.27`.

## How serving works (`genlm/backend/llm/`)

The mechanism, in one sentence: **a running request whose next token has not been appended
has `num_new_tokens == 0`, so vLLM's scheduler steps past it — resident, KV and GPU row
intact, free.** Token append is therefore the entire stepping and cadence mechanism, and
nothing needs a barrier, a feed protocol, or an engine-side notion of a particle.

`vllm.py`, three pieces:

- **Window** (`next_token_logprobs` -> `_collect`): concurrent asks meet in one batch. The
  window's first caller holds it open until a full event-loop pass adds no new ask, so a
  whole `asyncio.gather` lands together; `timeout` adds a cooperative linger for late
  callers. Run by that caller, never a background task — window state must not outlive the
  loop its callers are on (it did, and every later test hung).
- **Residency** (`_execute`): `(context, lora) -> request`. An ask one token past a resident
  appends that token; anything else prefills a new request. A resident is extendable **once
  per window and only when it owes no row** — a window can hold both a context and its
  extension (a critic leaf trails the draw leaf) and one request cannot serve both.
- **`GenlmScheduler`** (`scheduler_cls`): owns births/appends/reaps through a thread-safe
  inbox drained at the top of `schedule()`. Three non-obvious pieces: `has_requests()`
  counts a non-empty inbox (the engine tests it *before* calling `schedule()`, so otherwise
  the inbox never drains); `num_output_placeholders` is zeroed for our requests every
  schedule (their tokens come from appends, not sampling — async run-ahead accounting);
  and every drained append is shipped on the `SchedulerOutput` for a worker-side wrap to
  write into the runner's last-sampled buffer, because that buffer — not the request's
  token list — is where a decode step reads its input token. Miss that and every decode
  forwards token 0 while all scheduler-side bookkeeping looks healthy.
- **Capture shim** (`_CaptureSampler`): publishes each step's full-vocabulary `log_softmax`
  row and reports `num_sampled = 0`, so vLLM appends nothing and checks no stop condition.
  vLLM's own sampler is unused — drawing belongs to the algorithm, never the engine.
- **The crank**: the window steps `engine.step()` on a worker thread until its rows are
  captured, single-flight; later windows reconcile and ride the running crank's frames.
  There is no daemon and no cadence of our own (the in-process engine has none either).

`mlx.py`: the same window over `_SlotPool.logits([contexts])`, which already discovers KV
reuse by content. No scheduler, no request objects — the window is the frame.

**LoRA is per-request** on every backend: forwards take `lora_name=None` (None = base);
`lora_view(name)` is the bound-handle sugar. Re-registering a name rebinds it under a fresh
monotonic id (vLLM caches adapter weights by int id, so ids are never reused), purges
caches, and reaps residents under the old adapter — their KV would outlive its weights.

## Gotchas

- `release_all()` is async and must be awaited: it waits out a turning crank before applying
  reaps. A caller that skipped that wait left requests stranded in the engine, because the
  crank task had not yet run its exit path when the awaiting caller resumed.
- `$SCRATCH/gpuvenv` on Mila is a `uv` venv with **no pip** — `python -m pip install` fails.
  Both repos are already editable there, so an rsync is the whole install step.
- `tests/probe_vllm_server.py` is the mechanism check (manual, GPU): resident appends, idle
  residents skipping and resuming, placeholder discipline, interleaved one-shots, reaping.
  Run it before a gate when something looks structurally wrong — it is minutes, not an hour,
  and it names the broken piece.
- Engine-served rows differ from fresh-prefill rows by the warm-KV residual (~5e-4 total
  variation on A100/SmolLM, argmax identical). Judge it by total variation, never by max
  log-prob difference: the full-row max is dominated by tail columns carrying no mass.
- **A same-engine consistency check can bless its own corruption**: prefix-cache blocks
  are hashed by intended token ids, so a "fresh" re-prefill APC-hits whatever KV the
  resident actually computed — TV 0.0 against garbage. Any check of the decode path
  must compare against an arm sharing NO engine state (the probe's HF fp32 cross-check).
