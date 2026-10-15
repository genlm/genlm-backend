"""Sequential Monte Carlo arithmetic for a capped language-model prior."""

import asyncio
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol


def weight_increment(
    log_prior: float,
    log_proposal: float,
    previous_potential: float,
    next_potential: float,
) -> float:
    """Return the importance increment for one proposal and potential transition."""
    return log_prior - log_proposal + next_potential - previous_potential


@dataclass(frozen=True)
class Active:
    tokens: tuple[int, ...]
    log_prior: float
    log_proposal: float
    log_weight: float
    potential: float


@dataclass(frozen=True)
class Finished:
    tokens: tuple[int, ...]
    log_prior: float
    log_proposal: float
    log_weight: float
    potential: float
    similarity: float
    termination: Literal["eos", "cap"]


type Particle = Active | Finished


class LanguageModel(Protocol):
    async def next_logprobs(self, tokens: tuple[int, ...]) -> Sequence[float]: ...


class EmbeddingScorer(Protocol):
    async def similarities(self, texts: tuple[str, ...]) -> tuple[float, ...]: ...


class Decoder(Protocol):
    def __call__(self, tokens: tuple[int, ...]) -> str: ...


@dataclass(frozen=True)
class SMCConfig:
    particles: int
    max_tokens: int
    eos_token: int
    beta: float
    ess_fraction: float
    seed: int
    score_prefixes: bool
    eos_logit_bonus: float = 0.0
    prefix_beta: float | None = None


@dataclass(frozen=True)
class SMCResult:
    particles: tuple[Finished, ...]
    lm_calls: int
    encoder_calls: int
    resamplings: int
    log_normalizer_estimate: float


def _draw(
    logprobs: Sequence[float],
    eos_token: int,
    eos_logit_bonus: float,
    rng: random.Random,
) -> tuple[int, float]:
    adjusted = tuple(
        value + eos_logit_bonus if token == eos_token else value
        for token, value in enumerate(logprobs)
    )
    maximum = max(adjusted)
    masses = tuple(math.exp(value - maximum) for value in adjusted)
    log_total = math.log(sum(masses))
    threshold = rng.random() * math.exp(log_total)
    cumulative = 0.0
    for token, mass in enumerate(masses):
        cumulative += mass
        if cumulative >= threshold:
            return token, adjusted[token] - maximum - log_total
    token = len(masses) - 1
    return token, adjusted[token] - maximum - log_total


def _logmeanexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(
        sum(math.exp(value - maximum) for value in values) / len(values)
    )


def _ess(log_weights: Sequence[float]) -> float:
    maximum = max(log_weights)
    weights = tuple(math.exp(value - maximum) for value in log_weights)
    total = sum(weights)
    return total * total / sum(weight * weight for weight in weights)


def _resample(
    particles: tuple[Particle, ...], rng: random.Random
) -> tuple[Particle, ...]:
    maximum = max(particle.log_weight for particle in particles)
    weights = tuple(math.exp(particle.log_weight - maximum) for particle in particles)
    ancestors = rng.choices(range(len(particles)), weights=weights, k=len(particles))
    return tuple(replace(particles[index], log_weight=0.0) for index in ancestors)


async def sample_smc(
    lm: LanguageModel,
    scorer: EmbeddingScorer,
    decode: Decoder,
    config: SMCConfig,
) -> SMCResult:
    """Sample the capped LM prior with prefix-potential resampling and terminal embedding scores."""
    rng = random.Random(config.seed)
    particles: tuple[Particle, ...] = tuple(
        Active((), 0.0, 0.0, 0.0, 0.0) for _ in range(config.particles)
    )
    lm_calls = 0
    encoder_calls = 0
    resamplings = 0
    log_normalizer = 0.0

    for _ in range(config.max_tokens):
        active = tuple(
            (index, particle)
            for index, particle in enumerate(particles)
            if isinstance(particle, Active)
        )
        if not active:
            break
        distributions = await asyncio.gather(
            *(lm.next_logprobs(particle.tokens) for _, particle in active)
        )
        lm_calls += len(active)
        proposed: list[Particle] = list(particles)
        fresh: list[tuple[int, Active, float, float, bool, bool]] = []
        for (index, particle), logprobs in zip(active, distributions):
            token, logq = _draw(logprobs, config.eos_token, config.eos_logit_bonus, rng)
            eos = token == config.eos_token
            capped = len(particle.tokens) + 1 == config.max_tokens
            tokens = particle.tokens if eos else particle.tokens + (token,)
            fresh.append(
                (
                    index,
                    replace(particle, tokens=tokens),
                    float(logprobs[token]),
                    logq,
                    eos,
                    capped,
                )
            )

        scored = tuple(
            entry for entry in fresh if entry[4] or entry[5] or config.score_prefixes
        )
        texts = tuple(decode(particle.tokens) for _, particle, _, _, _, _ in scored)
        similarities = await scorer.similarities(texts)
        encoder_calls += len(texts)
        similarities_by_index = {
            entry[0]: similarity for entry, similarity in zip(scored, similarities)
        }
        for index, particle, logp, logq, eos, capped in fresh:
            terminal = eos or capped
            similarity = similarities_by_index.get(index, 0.0)
            prefix_beta = (
                config.beta if config.prefix_beta is None else config.prefix_beta
            )
            potential = (
                config.beta * similarity
                if terminal
                else prefix_beta * similarity
                if config.score_prefixes
                else 0.0
            )
            log_weight = particle.log_weight + weight_increment(
                logp, logq, particle.potential, potential
            )
            log_prior = particle.log_prior + logp
            log_proposal = particle.log_proposal + logq
            proposed[index] = (
                Finished(
                    particle.tokens,
                    log_prior,
                    log_proposal,
                    log_weight,
                    potential,
                    similarity,
                    "eos" if eos else "cap",
                )
                if terminal
                else Active(
                    particle.tokens, log_prior, log_proposal, log_weight, potential
                )
            )
        particles = tuple(proposed)
        if (
            any(isinstance(particle, Active) for particle in particles)
            and _ess(tuple(particle.log_weight for particle in particles))
            < config.ess_fraction * config.particles
        ):
            log_normalizer += _logmeanexp(
                tuple(particle.log_weight for particle in particles)
            )
            particles = _resample(particles, rng)
            resamplings += 1

    finished = tuple(
        particle for particle in particles if isinstance(particle, Finished)
    )
    log_normalizer += _logmeanexp(tuple(particle.log_weight for particle in finished))
    return SMCResult(finished, lm_calls, encoder_calls, resamplings, log_normalizer)
