"""Gated tests for the engine-native SMC arms (HubSampler / run_window).

These exercise the backend seams only -- no SMC logic. They require CUDA + vLLM
and are skipped otherwise.
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


# ----- trivial hubs implementing EngineControl -----------------------------


class ForceTokenHub:
    """Masks every row to a single forced token id (proposal is a point mass)."""

    def __init__(self, token_id):
        self.token_id = token_id
        self.shape_calls = 0
        self.seen_rows = []

    def shape(self, logits, request_ids):
        self.shape_calls += 1
        self.seen_rows.append(list(request_ids))
        logits.fill_(float("-inf"))
        logits[:, self.token_id] = 0.0

    def draw(self, logits, request_ids, sampling_metadata):
        return logits.argmax(dim=-1)


class ForceEosAtStepHub:
    """Forces a fixed token, then forces EOS once a request has drawn ``trigger``
    tokens (so that request pops out mid-window)."""

    def __init__(self, token_id, eos_id, trigger):
        self.token_id = token_id
        self.eos_id = eos_id
        self.trigger = trigger
        # internal request id -> number of draws so far
        self.counts = {}

    def shape(self, logits, request_ids):
        logits.fill_(float("-inf"))
        logits[:, self.token_id] = 0.0

    def draw(self, logits, request_ids, sampling_metadata):
        out = []
        for rid in request_ids:
            n = self.counts.get(rid, 0)
            self.counts[rid] = n + 1
            out.append(self.eos_id if n >= self.trigger else self.token_id)
        return torch.tensor(out, dtype=torch.int64, device=logits.device)


class RecordRowsHub:
    """Forces a per-particle distinct token and records the row->id ordering so we
    can verify the mapping stays correct as requests finish at different steps."""

    def __init__(self, id_to_token, eos_id, finish_after):
        # id_to_token: external particle index -> forced token
        self.id_to_token = id_to_token
        self.eos_id = eos_id
        # external index -> step at which to emit EOS
        self.finish_after = finish_after
        self.counts = {}
        # records, per step: list of (internal_rid_suffix_stripped, drawn_token)
        self.draw_log = []

    @staticmethod
    def _external(rid):
        # internal id is "{external}-{8 chars}"; strip the suffix.
        return rid.rsplit("-", 1)[0]

    def shape(self, logits, request_ids):
        logits.fill_(0.0)

    def draw(self, logits, request_ids, sampling_metadata):
        out = []
        step_record = []
        for rid in request_ids:
            ext = self._external(rid)
            idx = int(ext)
            n = self.counts.get(idx, 0)
            self.counts[idx] = n + 1
            if n >= self.finish_after[idx]:
                tok = self.eos_id
            else:
                tok = self.id_to_token[idx]
            out.append(tok)
            step_record.append((idx, tok))
        self.draw_log.append(step_record)
        return torch.tensor(out, dtype=torch.int64, device=logits.device)


# ----- tests ----------------------------------------------------------------


def test_no_hub_matches_stock(engine):
    """(a) With no hub attached, the swapped sampler matches stock generation."""
    prompts = [
        TokensPrompt(prompt_token_ids=[50256, 11, 12]),
        TokensPrompt(prompt_token_ids=[50256, 13]),
    ]
    sp = SamplingParams(
        n=1, max_tokens=4, temperature=0.0, detokenize=False, ignore_eos=True
    )

    stock = engine.generate(prompts, sp, use_tqdm=False)
    stock_toks = [list(o.outputs[0].token_ids) for o in stock]

    from genlm.backend.llm.vllm import install_hub_sampler

    hub_sampler, restore = install_hub_sampler(engine)
    try:
        # no attach -> behaves like stock
        out = engine.generate(prompts, sp, use_tqdm=False)
        hub_toks = [list(o.outputs[0].token_ids) for o in out]
    finally:
        restore()

    assert hub_toks == stock_toks


def test_force_single_token(engine):
    """(b) A hub that masks to one forced token: run_window returns that token
    every step."""
    from genlm.backend.llm.vllm import run_window

    forced = 198
    hub = ForceTokenHub(forced)
    out = run_window(
        engine,
        prompts=[[50256, 11, 12], [50256, 13]],
        control=hub,
        max_steps=4,
        eos_token_ids=[50256],
        temperature=1.0,
    )
    assert out == [[forced] * 4, [forced] * 4]
    assert hub.shape_calls == 4
    # row count per step matches number of live particles
    assert all(len(rows) == 2 for rows in hub.seen_rows)


def test_force_eos_pops_out(engine):
    """(c) A hub that forces EOS mid-window: the request finishes early and
    run_window returns the truncated (EOS-stripped) tokens."""
    from genlm.backend.llm.vllm import run_window

    forced = 198
    eos = 50256
    hub = ForceEosAtStepHub(forced, eos, trigger=2)
    out = run_window(
        engine,
        prompts=[[50256, 11, 12], [50256, 13]],
        control=hub,
        max_steps=8,
        eos_token_ids=[eos],
        temperature=1.0,
    )
    # 2 forced tokens, then EOS (stripped) -> exactly 2 tokens each
    assert out == [[forced, forced], [forced, forced]]


def test_row_request_mapping_staggered(engine):
    """(d) Mapping stays correct when requests finish at different steps."""
    from genlm.backend.llm.vllm import run_window

    eos = 50256
    # particle 0 -> token 100, particle 1 -> token 200, particle 2 -> token 300
    id_to_token = {0: 100, 1: 200, 2: 300}
    # particle 0 finishes after 1 token, 1 after 2, 2 after 3
    finish_after = {0: 1, 1: 2, 2: 3}
    hub = RecordRowsHub(id_to_token, eos, finish_after)
    out = run_window(
        engine,
        prompts=[[50256, 11], [50256, 12], [50256, 13]],
        control=hub,
        max_steps=8,
        eos_token_ids=[eos],
        temperature=1.0,
    )
    # EOS stripped: particle i emits finish_after[i] copies of its token
    assert out == [[100], [200, 200], [300, 300, 300]]

    # The draw log must never assign a particle a token that isn't its own.
    for step_record in hub.draw_log:
        for idx, tok in step_record:
            assert tok in (id_to_token[idx], eos), (idx, tok)
