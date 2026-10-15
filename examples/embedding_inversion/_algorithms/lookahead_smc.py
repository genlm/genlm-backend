from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal, assert_never

from .smc import (
    Active,
    Decoder,
    EmbeddingScorer,
    Finished,
    LanguageModel,
    _draw,
    _ess,
    _logmeanexp,
    weight_increment,
)


@dataclass(frozen=True)
class PositiveInt:
    """A positive count, validated at configuration construction."""

    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("count must be positive")


class Guidance(StrEnum):
    """Potential used for allocation while completed rollouts enter every archive."""

    COMPLETION = "completion"
    PREFIX = "prefix"


@dataclass(frozen=True)
class LookaheadConfig:
    """Capped-prior SMC with finite beta and an ESS threshold fraction in [0, 1]."""

    particles: PositiveInt
    max_tokens: PositiveInt
    rollouts: PositiveInt
    lookahead_interval: PositiveInt
    eos_token: int
    beta: float
    ess_fraction: float
    seed: int
    guidance: Guidance = Guidance.COMPLETION

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta):
            raise ValueError("beta must be finite")
        if not 0.0 <= self.ess_fraction <= 1.0:
            raise ValueError("ess_fraction must be in [0, 1]")


@dataclass(frozen=True)
class Completion:
    """A scored terminal text; archive membership carries no particle weight."""

    tokens: tuple[int, ...]
    log_prior: float
    similarity: float
    termination: Literal["eos", "cap"]
    source: Literal["particle", "rollout"]


@dataclass(frozen=True)
class LookaheadTransition:
    """One particle transition before any population resampling."""

    step: int
    particle_index: int
    tokens: tuple[int, ...]
    previous_potential: float
    potential: float
    log_transition_prior: float
    log_transition_proposal: float
    weight_increment: float
    termination: Literal["eos", "cap"] | None
    refreshed: bool
    rollouts: tuple[Completion, ...]


@dataclass(frozen=True)
class PopulationDiagnostic:
    """ESS before resampling and ancestor indices for the following population."""

    step: int
    ess: float
    ancestors: tuple[int, ...]
    resampled: bool


@dataclass(frozen=True)
class LookaheadResult:
    """Weighted particles and a terminal archive, with LM calls and distinct encoder texts.

    Encoder costs exclude the target query, which belongs to the caller.
    The archive retains every completion, including repeated texts.
    """

    particles: tuple[Finished, ...]
    archive: tuple[Completion, ...]
    trace: tuple[LookaheadTransition, ...]
    diagnostics: tuple[PopulationDiagnostic, ...]
    lm_calls: int
    rollout_lm_calls: int
    encoder_calls: int
    resamplings: int
    log_normalizer_estimate: float

    @property
    def best_cosine(self) -> Completion:
        """Select the highest-cosine archive member, independently of particle weights."""
        return max(self.archive, key=lambda entry: entry.similarity)


@dataclass(frozen=True)
class _Active(Active):
    lookahead: tuple[Completion, ...]


@dataclass(frozen=True)
class _Terminal:
    tokens: tuple[int, ...]
    log_prior: float
    termination: Literal["eos", "cap"]


@dataclass(frozen=True)
class _Proposed:
    index: int
    parent: _Active
    tokens: tuple[int, ...]
    logprob: float
    terminal: _Terminal | None


async def _complete(
    lm: LanguageModel, proposal: _Proposed, config: LookaheadConfig, rng: random.Random
) -> tuple[_Terminal, int]:
    tokens = proposal.tokens
    log_prior = proposal.parent.log_prior + proposal.logprob
    calls = 0
    while len(tokens) < config.max_tokens.value:
        probabilities = await lm.next_logprobs(tokens)
        calls += 1
        token, logprob = _draw(probabilities, config.eos_token, 0.0, rng)
        log_prior += logprob
        if token == config.eos_token:
            return _Terminal(tokens, log_prior, "eos"), calls
        tokens += (token,)
    return _Terminal(tokens, log_prior, "cap"), calls


async def sample_lookahead_smc(
    lm: LanguageModel,
    scorer: EmbeddingScorer,
    decode: Decoder,
    config: LookaheadConfig,
) -> LookaheadResult:
    """Estimate the capped-prior target using retained auxiliary rollout potentials.

    The extended path target contains each rollout's conditional prior law.
    Those factors cancel in importance ratios; the realized log-mean terminal
    likelihood is retained through resampling and replaced at termination.
    At the cap the target integrates to p(x) exp(beta * cosine(x)).
    """
    rng = random.Random(config.seed)
    particles: tuple[_Active | Finished, ...] = tuple(
        _Active((), 0.0, 0.0, 0.0, 0.0, ()) for _ in range(config.particles.value)
    )
    score_cache: dict[str, float] = {}
    archive: list[Completion] = []
    trace: list[LookaheadTransition] = []
    diagnostics: list[PopulationDiagnostic] = []
    lm_calls = rollout_lm_calls = resamplings = 0
    log_normalizer = 0.0

    for step in range(1, config.max_tokens.value + 1):
        active = tuple(
            (i, particle)
            for i, particle in enumerate(particles)
            if isinstance(particle, _Active)
        )
        if not active:
            break
        probabilities = await asyncio.gather(
            *(lm.next_logprobs(p.tokens) for _, p in active)
        )
        lm_calls += len(active)
        proposed: list[_Proposed] = []
        for (index, parent), distribution in zip(active, probabilities):
            token, logprob = _draw(distribution, config.eos_token, 0.0, rng)
            tokens = (
                parent.tokens if token == config.eos_token else parent.tokens + (token,)
            )
            terminal = (
                _Terminal(tokens, parent.log_prior + logprob, "eos")
                if token == config.eos_token
                else _Terminal(tokens, parent.log_prior + logprob, "cap")
                if len(tokens) == config.max_tokens.value
                else None
            )
            proposed.append(_Proposed(index, parent, tokens, logprob, terminal))

        refresh = (step - 1) % config.lookahead_interval.value == 0
        rollout_parents = tuple(p for p in proposed if p.terminal is None and refresh)
        completed = await asyncio.gather(
            *(
                _complete(lm, p, config, random.Random(rng.getrandbits(128)))
                for p in rollout_parents
                for _ in range(config.rollouts.value)
            )
        )
        rollout_lm_calls += sum(calls for _, calls in completed)
        lm_calls += sum(calls for _, calls in completed)
        terminal_paths = tuple(p.terminal for p in proposed if p.terminal is not None)
        all_paths = terminal_paths + tuple(path for path, _ in completed)
        prefix_texts = (
            tuple(decode(p.tokens) for p in rollout_parents)
            if config.guidance is Guidance.PREFIX
            else ()
        )
        missing = tuple(
            text
            for text in dict.fromkeys(
                (*tuple(decode(path.tokens) for path in all_paths), *prefix_texts)
            )
            if text not in score_cache
        )
        if missing:
            similarities = await scorer.similarities(missing)
            score_cache.update(zip(missing, similarities))
        rollout_scores = tuple(
            Completion(
                path.tokens,
                path.log_prior,
                score_cache[decode(path.tokens)],
                path.termination,
                "rollout",
            )
            for path, _ in completed
        )
        by_parent = {
            p.index: rollout_scores[
                i * config.rollouts.value : (i + 1) * config.rollouts.value
            ]
            for i, p in enumerate(rollout_parents)
        }
        archive.extend(rollout_scores)
        updated: list[_Active | Finished] = list(particles)
        changed_potential = False
        for proposal in proposed:
            parent = proposal.parent
            lookahead = by_parent.get(proposal.index, parent.lookahead)
            terminal = proposal.terminal
            potential = parent.potential
            if terminal is not None:
                potential = config.beta * score_cache[decode(terminal.tokens)]
            elif proposal.index in by_parent:
                match config.guidance:
                    case Guidance.COMPLETION:
                        potential = _logmeanexp(
                            tuple(config.beta * entry.similarity for entry in lookahead)
                        )
                    case Guidance.PREFIX:
                        potential = config.beta * score_cache[decode(proposal.tokens)]
                    case unreachable:
                        assert_never(unreachable)
            increment = weight_increment(
                proposal.logprob, proposal.logprob, parent.potential, potential
            )
            changed_potential |= potential != parent.potential
            log_prior = parent.log_prior + proposal.logprob
            log_proposal = parent.log_proposal + proposal.logprob
            log_weight = parent.log_weight + increment
            if terminal is None:
                updated[proposal.index] = _Active(
                    proposal.tokens,
                    log_prior,
                    log_proposal,
                    log_weight,
                    potential,
                    lookahead,
                )
            else:
                similarity = score_cache[decode(terminal.tokens)]
                updated[proposal.index] = Finished(
                    terminal.tokens,
                    log_prior,
                    log_proposal,
                    log_weight,
                    potential,
                    similarity,
                    terminal.termination,
                )
                archive.append(
                    Completion(
                        terminal.tokens,
                        log_prior,
                        similarity,
                        terminal.termination,
                        "particle",
                    )
                )
            trace.append(
                LookaheadTransition(
                    step,
                    proposal.index,
                    proposal.tokens,
                    parent.potential,
                    potential,
                    proposal.logprob,
                    proposal.logprob,
                    increment,
                    terminal.termination if terminal is not None else None,
                    proposal.index in by_parent,
                    by_parent.get(proposal.index, ()),
                )
            )
        particles = tuple(updated)
        log_weights = tuple(p.log_weight for p in particles)
        ess = _ess(log_weights)
        ancestors = tuple(range(len(particles)))
        resampled = (
            changed_potential
            and any(isinstance(p, _Active) for p in particles)
            and ess < config.ess_fraction * len(particles)
        )
        if resampled:
            log_normalizer += _logmeanexp(log_weights)
            maximum = max(log_weights)
            ancestors = tuple(
                rng.choices(
                    ancestors,
                    weights=tuple(math.exp(w - maximum) for w in log_weights),
                    k=len(particles),
                )
            )
            particles = tuple(replace(particles[i], log_weight=0.0) for i in ancestors)
            resamplings += 1
        diagnostics.append(PopulationDiagnostic(step, ess, ancestors, resampled))

    finished = tuple(p for p in particles if isinstance(p, Finished))
    log_normalizer += _logmeanexp(tuple(p.log_weight for p in finished))
    return LookaheadResult(
        finished,
        tuple(archive),
        tuple(trace),
        tuple(diagnostics),
        lm_calls,
        rollout_lm_calls,
        len(score_cache),
        resamplings,
        log_normalizer,
    )
