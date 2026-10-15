import asyncio
import math

from hypothesis import given
from hypothesis import strategies as st

from examples.embedding_inversion._algorithms.smc import (
    SMCConfig,
    sample_smc,
    weight_increment,
)


@given(
    log_prior=st.lists(
        st.floats(min_value=-20, max_value=0, allow_nan=False), min_size=1, max_size=20
    ),
    log_proposal=st.lists(
        st.floats(min_value=-20, max_value=0, allow_nan=False), min_size=1, max_size=20
    ),
    potentials=st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=1, max_size=20
    ),
)
def test_weight_increments_telescope_to_capped_target(
    log_prior, log_proposal, potentials
):
    n = min(len(log_prior), len(log_proposal), len(potentials))
    increments = tuple(
        weight_increment(
            log_prior[i],
            log_proposal[i],
            potentials[i - 1] if i else 0.0,
            potentials[i],
        )
        for i in range(n)
    )
    expected = sum(log_prior[:n]) - sum(log_proposal[:n]) + potentials[n - 1]
    assert math.isclose(sum(increments), expected, abs_tol=1e-10)


def test_absorbing_transition_has_zero_weight_increment():
    assert weight_increment(0.0, 0.0, 3.7, 3.7) == 0.0


def test_capped_prior_and_terminal_potential_are_accounted_for_once():
    class FiniteLM:
        def __init__(self):
            self.contexts = []

        async def next_logprobs(self, tokens):
            self.contexts.append(tokens)
            return (math.log(0.2), math.log(0.8))

    class LengthScorer:
        def __init__(self):
            self.batches = []

        async def similarities(self, texts):
            self.batches.append(texts)
            return tuple(len(text) / 3 for text in texts)

    model = FiniteLM()
    scorer = LengthScorer()
    result = asyncio.run(
        sample_smc(
            model,
            scorer,
            lambda tokens: "a" * len(tokens),
            SMCConfig(
                particles=30,
                max_tokens=3,
                eos_token=0,
                beta=2.0,
                ess_fraction=0.0,
                seed=19,
                score_prefixes=True,
            ),
        )
    )

    assert len(result.particles) == 30
    assert result.lm_calls == len(model.contexts)
    assert result.encoder_calls == result.lm_calls
    assert result.resamplings == 0
    assert all(len(tokens) < 3 for tokens in model.contexts)
    assert sum(map(len, scorer.batches)) == result.encoder_calls
    for particle in result.particles:
        assert len(particle.tokens) <= 3
        assert particle.similarity == len(particle.tokens) / 3
        expected_log_prior = len(particle.tokens) * math.log(0.8)
        if particle.termination == "eos":
            expected_log_prior += math.log(0.2)
        else:
            assert len(particle.tokens) == 3
        assert math.isclose(particle.log_prior, expected_log_prior, abs_tol=1e-12)
        assert math.isclose(particle.log_weight, 2 * particle.similarity, abs_tol=1e-12)
    terminal_potentials = tuple(
        2 * particle.similarity for particle in result.particles
    )
    stable_mean = max(terminal_potentials) + math.log(
        sum(math.exp(value - max(terminal_potentials)) for value in terminal_potentials)
        / len(terminal_potentials)
    )
    assert math.isclose(result.log_normalizer_estimate, stable_mean, abs_tol=1e-12)


def test_terminal_only_scoring_and_large_potentials_survive_resampling():
    class FiniteLM:
        async def next_logprobs(self, tokens):
            return (math.log(0.4), math.log(0.6))

    class LengthScorer:
        def __init__(self):
            self.texts = []

        async def similarities(self, texts):
            self.texts.extend(texts)
            return tuple(len(text) / 4 for text in texts)

    scorer = LengthScorer()
    baseline = asyncio.run(
        sample_smc(
            FiniteLM(),
            scorer,
            lambda ids: "a" * len(ids),
            SMCConfig(
                particles=32,
                max_tokens=4,
                eos_token=0,
                beta=1000.0,
                ess_fraction=0.0,
                seed=4,
                score_prefixes=False,
            ),
        )
    )
    assert baseline.encoder_calls == 32
    assert len(scorer.texts) == 32
    assert all(math.isfinite(particle.log_weight) for particle in baseline.particles)
    assert math.isfinite(baseline.log_normalizer_estimate)

    guided = asyncio.run(
        sample_smc(
            FiniteLM(),
            LengthScorer(),
            lambda ids: "a" * len(ids),
            SMCConfig(
                particles=32,
                max_tokens=4,
                eos_token=0,
                beta=1000.0,
                ess_fraction=0.95,
                seed=4,
                score_prefixes=True,
            ),
        )
    )
    assert guided.resamplings >= 1
    assert len(guided.particles) == 32
    assert all(math.isfinite(particle.log_weight) for particle in guided.particles)
    assert math.isfinite(guided.log_normalizer_estimate)


def test_eos_proposal_has_full_support_and_correct_importance_ratio():
    class FiniteLM:
        async def next_logprobs(self, tokens):
            return (math.log(0.2), math.log(0.8))

    class LengthScorer:
        async def similarities(self, texts):
            return tuple(len(text) / 3 for text in texts)

    result = asyncio.run(
        sample_smc(
            FiniteLM(),
            LengthScorer(),
            lambda ids: "a" * len(ids),
            SMCConfig(
                particles=50,
                max_tokens=3,
                eos_token=0,
                beta=5.0,
                ess_fraction=0.0,
                seed=12,
                score_prefixes=False,
                eos_logit_bonus=2.0,
            ),
        )
    )
    normalizer = 0.2 * math.exp(2.0) + 0.8
    log_content_q = math.log(0.8 / normalizer)
    log_eos_q = math.log(0.2 * math.exp(2.0) / normalizer)
    assert any(particle.termination == "eos" for particle in result.particles)
    assert any(particle.termination == "cap" for particle in result.particles)
    for particle in result.particles:
        expected_log_q = len(particle.tokens) * log_content_q
        if particle.termination == "eos":
            expected_log_q += log_eos_q
        assert math.isclose(particle.log_proposal, expected_log_q, abs_tol=1e-12)
        expected_weight = particle.log_prior - expected_log_q + 5 * particle.similarity
        assert math.isclose(particle.log_weight, expected_weight, abs_tol=1e-12)
