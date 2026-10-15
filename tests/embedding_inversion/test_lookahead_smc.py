import asyncio
import math
from dataclasses import replace
from statistics import mean, stdev

import pytest
from hypothesis import given, settings, strategies as st

from examples.embedding_inversion._algorithms.lookahead_smc import (
    Guidance,
    LookaheadConfig,
    PositiveInt,
    sample_lookahead_smc,
)


class FiniteLM:
    def __init__(self):
        self.contexts = []

    @staticmethod
    def masses(tokens):
        return (0.2, 0.5, 0.3) if not tokens or tokens[-1] == 1 else (0.5, 0.2, 0.3)

    async def next_logprobs(self, tokens):
        self.contexts.append(tokens)
        return tuple(map(math.log, self.masses(tokens)))


class FiniteScorer:
    def __init__(self):
        self.texts = []

    @staticmethod
    def similarity(text):
        return (text.count("a") - text.count("b")) / 4

    async def similarities(self, texts):
        self.texts.extend(texts)
        return tuple(map(self.similarity, texts))


def _decode(tokens):
    return "".join("ab"[token - 1] for token in tokens)


def _config(**changes):
    return replace(
        LookaheadConfig(
            particles=PositiveInt(8),
            max_tokens=PositiveInt(4),
            rollouts=PositiveInt(2),
            lookahead_interval=PositiveInt(2),
            eos_token=0,
            beta=3.0,
            ess_fraction=0.0,
            seed=7,
        ),
        **changes,
    )


def _prior(tokens, cap):
    mass = math.prod(
        FiniteLM.masses(tokens[:i])[token] for i, token in enumerate(tokens)
    )
    return mass if len(tokens) == cap else mass * FiniteLM.masses(tokens)[0]


def _terminal_states(cap, prefix=()):
    yield prefix
    if len(prefix) < cap:
        for token in (1, 2):
            yield from _terminal_states(cap, prefix + (token,))


@given(
    cap=st.integers(1, 4),
    interval=st.integers(1, 5),
    rollouts=st.integers(1, 3),
    beta=st.floats(-5, 5, allow_nan=False),
    seed=st.integers(0, 1000),
    guidance=st.sampled_from(tuple(Guidance)),
)
@settings(max_examples=35, deadline=None)
def test_capped_target_weights_telescope_despite_random_retained_lookahead(
    cap, interval, rollouts, beta, seed, guidance
):
    lm, scorer = FiniteLM(), FiniteScorer()
    config = _config(
        max_tokens=PositiveInt(cap),
        lookahead_interval=PositiveInt(interval),
        rollouts=PositiveInt(rollouts),
        beta=beta,
        seed=seed,
        guidance=guidance,
    )
    result = asyncio.run(sample_lookahead_smc(lm, scorer, _decode, config))

    assert len(result.particles) == config.particles.value
    assert result.lm_calls == len(lm.contexts)
    assert result.encoder_calls == len(scorer.texts)
    assert all(len(context) < cap for context in lm.contexts)
    assert result.resamplings == 0
    for particle in result.particles:
        assert particle.termination == ("cap" if len(particle.tokens) == cap else "eos")
        assert particle.log_prior == pytest.approx(
            math.log(_prior(particle.tokens, cap))
        )
        assert particle.log_proposal == particle.log_prior
        assert particle.log_weight == pytest.approx(
            beta * FiniteScorer.similarity(_decode(particle.tokens))
        )
    for transition in result.trace:
        if transition.termination is None:
            assert transition.refreshed == ((transition.step - 1) % interval == 0)
            if not transition.refreshed:
                assert transition.potential == transition.previous_potential
        else:
            assert transition.potential == pytest.approx(
                beta * FiniteScorer.similarity(_decode(transition.tokens))
            )
        if transition.refreshed:
            assert len(transition.rollouts) == rollouts
            assert all(
                entry.tokens[: len(transition.tokens)] == transition.tokens
                for entry in transition.rollouts
            )
            expected_potential = (
                math.log(
                    mean(
                        math.exp(beta * entry.similarity)
                        for entry in transition.rollouts
                    )
                )
                if guidance is Guidance.COMPLETION
                else beta * FiniteScorer.similarity(_decode(transition.tokens))
            )
            assert transition.potential == pytest.approx(expected_potential)
        else:
            assert transition.rollouts == ()
    expected = mean(math.exp(p.log_weight) for p in result.particles)
    assert math.exp(result.log_normalizer_estimate) == pytest.approx(expected)
    assert result.best_cosine.similarity == max(
        entry.similarity for entry in result.archive
    )
    for entry in result.archive:
        assert entry.log_prior == pytest.approx(math.log(_prior(entry.tokens, cap)))
        assert entry.similarity == FiniteScorer.similarity(_decode(entry.tokens))
        assert entry.termination == ("cap" if len(entry.tokens) == cap else "eos")


def test_resampling_carries_realized_rollout_potential_and_records_ancestry():
    config = _config(beta=20.0, ess_fraction=0.99, lookahead_interval=PositiveInt(1))
    result = asyncio.run(
        sample_lookahead_smc(FiniteLM(), FiniteScorer(), _decode, config)
    )

    assert result.resamplings > 0
    transitions = {(t.step, t.particle_index): t for t in result.trace}
    checked_edges = 0
    for diagnostic in result.diagnostics[:-1]:
        for index, ancestor in enumerate(diagnostic.ancestors):
            next_transition = transitions.get((diagnostic.step + 1, index))
            if next_transition is not None:
                checked_edges += 1
                assert (
                    next_transition.previous_potential
                    == transitions[diagnostic.step, ancestor].potential
                )
    assert checked_edges > 0
    assert all(
        0 < diagnostic.ess <= config.particles.value + 1e-10
        for diagnostic in result.diagnostics
    )
    assert result.rollout_lm_calls > 0
    assert any(entry.source == "rollout" for entry in result.archive)


@pytest.mark.parametrize("interval", (1, 2))
@pytest.mark.parametrize("ess_fraction", (0.0, 0.9))
@pytest.mark.parametrize("guidance", tuple(Guidance))
def test_normalizer_and_weighted_moments_match_exact_enumerated_variable_length_target(
    interval, ess_fraction, guidance
):
    config = _config(
        particles=PositiveInt(4),
        max_tokens=PositiveInt(3),
        beta=2.0,
        lookahead_interval=PositiveInt(interval),
        ess_fraction=ess_fraction,
        guidance=guidance,
    )
    states = tuple(_terminal_states(config.max_tokens.value))
    assert sum(
        _prior(state, config.max_tokens.value) for state in states
    ) == pytest.approx(1.0)
    observables = (lambda tokens: 1, len, lambda tokens: int(tokens[:1] == (1,)))
    exact = tuple(
        sum(
            _prior(state, config.max_tokens.value)
            * math.exp(config.beta * FiniteScorer.similarity(_decode(state)))
            * observable(state)
            for state in states
        )
        for observable in observables
    )

    async def estimates():
        records = []
        for seed in range(1200):
            result = await sample_lookahead_smc(
                FiniteLM(), FiniteScorer(), _decode, replace(config, seed=seed)
            )
            normalizer = math.exp(result.log_normalizer_estimate)
            total_weight = sum(math.exp(p.log_weight) for p in result.particles)
            records.append(
                tuple(
                    normalizer
                    * sum(
                        math.exp(p.log_weight) * observable(p.tokens)
                        for p in result.particles
                    )
                    / total_weight
                    for observable in observables
                )
            )
        return tuple(zip(*records))

    values = asyncio.run(estimates())
    for samples, moment in zip(values, exact):
        standard_error = stdev(samples) / math.sqrt(len(samples))
        # Fixed seeds and six estimated standard errors limit stochastic false alarms.
        assert abs(mean(samples) - moment) < 6 * standard_error


@given(seed=st.integers(0, 1000), interval=st.integers(1, 5))
@settings(max_examples=25, deadline=None)
def test_guidance_without_resampling_preserves_completed_candidates_and_lm_budget(
    seed, interval
):
    config = _config(seed=seed, lookahead_interval=PositiveInt(interval))
    completion_scorer, prefix_scorer = FiniteScorer(), FiniteScorer()
    completion = asyncio.run(
        sample_lookahead_smc(FiniteLM(), completion_scorer, _decode, config)
    )
    prefix = asyncio.run(
        sample_lookahead_smc(
            FiniteLM(),
            prefix_scorer,
            _decode,
            replace(config, guidance=Guidance.PREFIX),
        )
    )
    assert prefix.archive == completion.archive
    assert prefix.lm_calls == completion.lm_calls
    assert prefix.rollout_lm_calls == completion.rollout_lm_calls
    prefix_texts = {_decode(t.tokens) for t in prefix.trace if t.refreshed}
    assert set(prefix_scorer.texts) == set(completion_scorer.texts) | prefix_texts
    assert len(prefix_scorer.texts) == len(set(prefix_scorer.texts))
    assert tuple(p.log_weight for p in prefix.particles) == pytest.approx(
        tuple(p.log_weight for p in completion.particles)
    )


@given(st.integers(max_value=0))
def test_positive_schedule_counts_exclude_zero_and_negative_values(value):
    with pytest.raises(ValueError, match="positive"):
        PositiveInt(value)
