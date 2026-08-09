# genlm-backend

vLLM/backend layer for genlm-control. On branch `shepard/speedups` it adds the engine seam
that lets a control-side SMC driver run vLLM's decode loop. **The backend exposes seams only —
no SMC algorithm logic (no driver, ESS, resample, or weight bookkeeping) lives here.**

## Environment

- No `.venv` here; this repo is editable-installed into genlm-control's. vLLM is remote-only.
- **rsync this repo separately from control.** Syncing only control leaves a stale
  `run_burst`/`_GroupTable` serving new control code — silently wrong results, not a crash.
- **Push `shepard/speedups` in lockstep with control's.** Control's `PromptedLLM._served`
  reads `burst_active`; an unpushed backend commit leaves anything installing both repos from
  that branch broken on an `AttributeError`.
- Pre-commit hooks (ruff + ruff-format) rewrite files on commit; re-`git add` and re-commit if
  a commit aborts. Note genlm-control is NOT ruff-formatted — do not carry `ruff format` across.
- Engine runs **in-process** (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, set in `llm/vllm.py` before
  import) so control-side objects call into it directly. Run with
  `VLLM_USE_FLASHINFER_SAMPLER=0`. The `pyproject.toml` vLLM pin is stale relative to what the
  remote actually runs — check the installed version rather than trusting it.

## The engine seam (`genlm/backend/llm/`)

- `engine_control.py` — the `EngineControl` protocol the control side implements; the backend
  imports nothing from control. No `getattr`/`hasattr` probing.
- `vllm.py`:
  - `ControlSampler` (subclass of vLLM's `Sampler`) is **installed once and persistent**.
    `attach`/`detach` bind a control object per burst. Detached, it defers verbatim to stock,
    so normal generation and `next_token_logprobs` are byte-unaffected — `test_no_hub_matches_stock`
    guards that. Attached, stock logits processors run first, then `draw`. LPs run *inside*
    `Sampler.forward`, so swapping `model_runner.sampler` is the one seam needed.
  - **Group lane**: one control handle = K engine requests stepped in lockstep. `_GroupTable`
    absorbs partial scheduling (stall, flush, re-add) so control never sees a half-stepped
    group; `draw` receives `[G, K, vocab]` and returns one token per group.
  - `run_burst(control, max_steps)`: control owns what requests exist. A pre-loop drain seeds
    the initial population, then each `engine.step()` selects and drains aborts/adds. Returns
    nothing — committed tokens live control-side. **Pop-out is an explicit `abort_request`, not
    an EOS draw.** Each `engine.step()` blocks on control-side python for every live row (the
    draw hops to the main loop and waits), so the slowest row sets step latency.
  - `burst_active` / `_reject_during_burst`: while a burst owns the decode loop, any other
    logprobs forward would re-enter it through the attached sampler and deadlock (the draw hops
    to the loop the caller is blocking). Every such path raises instead. Control reads
    `burst_active` to fail earlier, with the offending potential named.
- **Row → particle mapping**: `model_runner.input_batch.req_ids[i]` is authoritative —
  `BatchUpdate.added` does not carry the request id. The id the sampler sees is **suffixed**
  (`"{external}-{8hex}"`), while `abort_request` and `output.request_id` use the plain external
  id. Control speaks opaque int handles; the engine request id is `str(handle)`. The suffixed
  format is parsed in exactly one place (`ControlSampler._row_handles`) — keep it that way.
- **LoRA is per-request** on every backend: all forwards take `lora_name=None` (None = base);
  `lora_view(name)` is the bound-handle sugar. Re-registering a name rebinds it to new weights
  with a fresh monotonic engine id (vLLM caches adapter weights by int id, so ids are never
  reused) plus a cache purge. `set_lora`/`clear_lora` are tombstones that raise with a
  migration message.
