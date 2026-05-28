"""Gated tests for the engine-native SMC arms (ControlSampler / run_burst).

These exercise the backend seams only -- no SMC logic. They require CUDA + vLLM
and are skipped otherwise.

The shipped contract under test (genlm/backend/llm/vllm.py):
  * ``ControlSampler.forward`` defers to the stock sampler when no control is attached,
    and otherwise calls ``control.draw(logits, request_ids)`` (TWO args -- no ``shape``,
    no ``sampling_metadata``). The control forms its own proposal from ``logits``.
  * ``run_burst(prompts, control, max_steps)`` submits one request
    per prompt (ids ``str(0..N-1)``, ``ignore_eos=True``), drives the decode loop,
    and -- after each step -- aborts the rows ``control.drain_aborts()`` names and
    (re-)adds the rows ``control.drain_adds()`` names. Pop-out is the explicit
    abort, NOT an EOS draw. ``run_burst`` returns ``[[] for _ in prompts]`` --
    committed tokens live control-side -- so tests assert on control-recorded state.
"""

import pytest
import torch

from conftest import v1_capable

try:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


pytestmark = v1_capable

MODEL = "gpt2"
EOS = 50256


@pytest.fixture(autouse=True, scope="function")
def cleanup_modules():
    # Override the conftest autouse fixture: these tests share one module-scoped
    # vLLM engine, so we must NOT wipe vllm/genlm modules between tests (doing so
    # triggers a duplicate opaque-type registration on re-import). No-op.
    yield


@pytest.fixture(scope="module")
def engine():
    # Import the genlm vllm module first so its env vars (VLLM_USE_V1=1,
    # VLLM_ENABLE_V1_MULTIPROCESSING=0) are set before vllm is first imported.
    import genlm.backend.llm.vllm as gvllm  # noqa: F401
    from vllm import LLM

    llm = LLM(
        model=MODEL,
        tokenizer=MODEL,
        enable_prefix_caching=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.3,
        max_model_len=256,
    )
    yield llm


@pytest.fixture(scope="module")
def vlm(engine):
    # Mirror from_name's wiring on a raw engine: register the logprobs-capture LP
    # and install the persistent ControlSampler (detached) once.
    from genlm.backend.llm.vllm import (
        AsyncVirtualLM,
        ControlSampler,
        GlobalLogprobsCapture,
    )

    logprobs_capture = GlobalLogprobsCapture()
    model_runner = AsyncVirtualLM._get_model_runner(engine)
    model_runner.input_batch.logitsprocs.argmax_invariant.append(logprobs_capture)
    control_sampler = ControlSampler(
        logprobs_mode=model_runner.model_config.logprobs_mode,
        model_runner=model_runner,
    )
    model_runner.sampler = control_sampler
    inst = AsyncVirtualLM(engine, logprobs_capture)
    inst._control_sampler = control_sampler
    yield inst


# ----- trivial controls implementing the real EngineControl contract -------------


class RecordingControl:
    """Minimal ``EngineControl``: forces a per-row token, records every draw, and
    pops a row out (via ``drain_aborts``) once it has drawn its quota.

    Implements exactly what ``ControlSampler.forward`` / ``run_burst`` call:
    ``draw(logits, request_ids)`` and ``drain_aborts()`` (and optionally
    ``drain_adds()``). No ``shape``, no ``sampling_metadata`` -- the control forms its
    proposal inside ``draw`` (here: a point mass on the forced token).
    """

    def __init__(self, token_of, quota, adds=None):
        # token_of: external-int id -> the token id to force for that row.
        # quota:    external-int id -> #draws after which the row pops out.
        # adds:     optional {after_step: [(ext_id, prompt_ids, token, quota)]} to
        #           re-add mid-burst (the Plan-B drain_adds seam).
        self.token_of = dict(token_of)
        self.quota = dict(quota)
        self._adds = adds or {}
        self.counts = {}  # external-int id -> draws so far
        self.draw_log = []  # per step: [(raw_request_id, ext_int, token), ...]
        self.seen_ids = []  # every raw request id seen, for the id-format check
        self._pending_abort = []  # raw request ids to abort on the next drain
        self._pending_add = []  # (ext_id, prompt_ids) queued for the next drain
        self._step = 0

    @staticmethod
    def _ext(rid):
        """External index for a vLLM request id. ``run_burst`` submits ``str(i)`` but
        ``input_batch.req_ids`` (what the control sees) carries a ``"{ext}-{suffix}"``
        form, while ``abort_request`` / ``output.request_id`` use the plain ``"{ext}"``.
        So drain_aborts MUST return the stripped external -- exactly what the real
        Controller's ``_burst_external`` does -- or the abort won't match the engine's
        request id. Mirroring it here is the point of the test."""
        return int(str(rid).rsplit("-", 1)[0])

    def draw(self, logits, request_ids):
        step_record = []
        out = []
        for rid in request_ids:
            self.seen_ids.append(str(rid))
            ext = self._ext(rid)
            n = self.counts.get(ext, 0) + 1
            self.counts[ext] = n
            tok = self.token_of[ext]
            out.append(tok)
            step_record.append((rid, ext, tok))
            # Once a row has drawn its quota, queue it for out-of-band pop-out. Return
            # the STRIPPED external id (run_burst does ``str(ext)``, which matches the
            # plain submitted request id) -- the suffixed ``rid`` would not match.
            if n >= self.quota[ext]:
                self._pending_abort.append(ext)
        self.draw_log.append(step_record)
        # Queue any scheduled mid-burst re-adds for this step.
        for ext_id, prompt_ids, token, quota in self._adds.get(self._step, []):
            self.token_of[ext_id] = token
            self.quota[ext_id] = quota
            self._pending_add.append((ext_id, list(prompt_ids)))
        self._step += 1
        return torch.tensor(out, dtype=torch.int64, device=logits.device)

    def drain_aborts(self):
        rows, self._pending_abort = self._pending_abort, []
        return rows

    def drain_adds(self):
        adds, self._pending_add = self._pending_add, []
        return adds

    # -- assertions helpers --
    def draws_for(self, ext):
        return [tok for step in self.draw_log for (_, e, tok) in step if e == ext]


def _assert_id_format(control):
    """The control side maps a row via ``int(rid.rsplit('-', 1)[0])`` AND
    ``run_burst`` dedups aborts via ``str(ext) in gone`` (verbatim). Both hold iff
    every request id is either the plain external string or ``"{ext}-{suffix}"``.
    This is the #8 consistency assumption -- assert it directly on the box."""
    for rid in control.seen_ids:
        ext = int(rid.rsplit("-", 1)[0])
        assert rid == str(ext) or rid.startswith(f"{ext}-"), (
            f"request id {rid!r} breaks the '{{ext}}' / '{{ext}}-{{suffix}}' "
            "assumption shared by _burst_external and run_burst's gone-filter"
        )


# ----- tests ----------------------------------------------------------------


def test_detached_matches_stock(vlm):
    """(a) The persistent ControlSampler, detached, matches stock generation."""
    from vllm.v1.sample.sampler import Sampler

    prompts = [
        TokensPrompt(prompt_token_ids=[EOS, 11, 12]),
        TokensPrompt(prompt_token_ids=[EOS, 13]),
    ]
    sp = SamplingParams(
        n=1, max_tokens=4, temperature=0.0, detokenize=False, ignore_eos=True
    )

    model_runner = vlm._get_model_runner(vlm.llm_engine)
    control_sampler = model_runner.sampler  # the persistent ControlSampler

    # Baseline through a fresh stock Sampler.
    model_runner.sampler = Sampler(
        logprobs_mode=model_runner.model_config.logprobs_mode
    )
    try:
        stock = vlm.llm_engine.generate(prompts, sp, use_tqdm=False)
        stock_toks = [list(o.outputs[0].token_ids) for o in stock]
    finally:
        model_runner.sampler = control_sampler

    # Detached ControlSampler must match stock byte-for-byte.
    control_sampler.detach()
    out = vlm.llm_engine.generate(prompts, sp, use_tqdm=False)
    out_toks = [list(o.outputs[0].token_ids) for o in out]

    assert out_toks == stock_toks


def test_draw_drives_every_step(vlm):
    """(b) ``control.draw`` (2-arg) drives the token for every row, every step; with a
    quota above ``max_steps`` the rows never pop and ``max_steps`` caps the burst."""
    forced = 198
    control = RecordingControl(token_of={0: forced, 1: forced}, quota={0: 99, 1: 99})
    ret = vlm.run_burst(
        prompts=[[EOS, 11, 12], [EOS, 13]], control=control, max_steps=4
    )
    # run_burst returns empty lists (tokens are control-side).
    assert ret == [[], []]
    # Exactly max_steps draws, both rows present each step, all forced.
    assert len(control.draw_log) == 4
    assert all(len(step) == 2 for step in control.draw_log)
    assert control.draws_for(0) == [forced] * 4
    assert control.draws_for(1) == [forced] * 4
    _assert_id_format(control)


def test_pop_out_via_abort(vlm):
    """(c) A row pops out via ``drain_aborts`` (NOT an EOS draw): once it draws its
    quota it is aborted, so it is never drawn again and the burst ends when empty."""
    forced = 198
    control = RecordingControl(token_of={0: forced, 1: forced}, quota={0: 2, 1: 2})
    vlm.run_burst(prompts=[[EOS, 11, 12], [EOS, 13]], control=control, max_steps=8)
    # Each row drawn exactly twice (aborted after the 2nd), burst ended early.
    assert control.draws_for(0) == [forced, forced]
    assert control.draws_for(1) == [forced, forced]
    assert len(control.draw_log) == 2
    _assert_id_format(control)


def test_row_request_mapping_staggered(vlm):
    """(d) Row->request mapping stays correct as rows pop out at different steps:
    row i draws its own token exactly ``quota[i]`` times, never another's."""
    token_of = {0: 100, 1: 200, 2: 300}
    quota = {0: 1, 1: 2, 2: 3}  # rows pop after 1 / 2 / 3 draws
    control = RecordingControl(token_of=token_of, quota=quota)
    vlm.run_burst(
        prompts=[[EOS, 11], [EOS, 12], [EOS, 13]],
        control=control,
        max_steps=8,
    )
    # Per-row draw counts match the quotas, and each row only ever drew its token.
    assert control.draws_for(0) == [100]
    assert control.draws_for(1) == [200, 200]
    assert control.draws_for(2) == [300, 300, 300]
    # A row never appears in a step after it has been aborted.
    assert len(control.draw_log) == 3
    assert [len(step) for step in control.draw_log] == [3, 2, 1]
    for step in control.draw_log:
        for _, ext, tok in step:
            assert tok == token_of[ext], (ext, tok)
    _assert_id_format(control)


def test_drain_adds_readds_row_midburst(vlm):
    """(e) The Plan-B ``drain_adds`` seam: a control re-adds a fresh row mid-burst and
    ``run_burst`` adds it to the live batch (it shows up in a later draw step) and
    drives it like any other -- the other rows never paused."""
    # Two initial rows (ext 0, 1) drawing >max_steps so they stay live; after the
    # first step, re-add a fresh row ext 99 that draws twice then pops.
    control = RecordingControl(
        token_of={0: 111, 1: 222},
        quota={0: 99, 1: 99},
        adds={0: [(99, [EOS, 14], 333, 2)]},
    )
    vlm.run_burst(prompts=[[EOS, 11], [EOS, 12]], control=control, max_steps=6)
    # The re-added row was driven (drew its token), proving drain_adds plumbing.
    assert control.draws_for(99) == [333, 333]
    # It first appears strictly after step 0 (it was added after the first draw).
    first_99 = next(
        i for i, step in enumerate(control.draw_log) if any(e == 99 for _, e, _ in step)
    )
    assert first_99 >= 1
    # The original rows kept being drawn across the whole burst (never paused).
    assert control.draws_for(0) == [111] * len(control.draw_log)
    _assert_id_format(control)
