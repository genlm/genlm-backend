"""Heuristic embedding-guided beam search over a capped language-model prior."""

import asyncio
import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol, assert_never


class LanguageModel(Protocol):
    """Provide full-vocabulary next-token log probabilities for a content prefix."""

    async def next_logprobs(self, tokens: tuple[int, ...]) -> Sequence[float]: ...


class EmbeddingScorer(Protocol):
    """Score decoded strings against one fixed target embedding."""

    async def similarities(self, texts: tuple[str, ...]) -> tuple[float, ...]: ...


class Decoder(Protocol):
    """Decode content tokens without a prompt or terminal token."""

    def __call__(self, tokens: tuple[int, ...]) -> str: ...


@dataclass(frozen=True)
class BeamConfig:
    """Fix beam branching, ranking, and the heuristic repeat-token proposal filter."""

    width: int
    top_k: int
    max_tokens: int
    eos_token: int
    beta: float
    diversity_penalty: float
    rank_mode: Literal["cosine", "hybrid"] = "hybrid"
    eos_policy: Literal["unconditional", "top_k", "cap_only"] = "unconditional"
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be finite and at least 1")


@dataclass(frozen=True)
class Candidate:
    """Record a scored token expansion and whether it remains in the active beam."""

    step: int
    parent: tuple[int, ...]
    token: int
    tokens: tuple[int, ...]
    log_prior: float
    similarity: float
    rank_score: float
    termination: Literal["eos", "cap"] | None
    kept: bool = False


@dataclass(frozen=True)
class LMCall:
    """Identify a queried content prefix at one search step."""

    step: int
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class EncoderCall:
    """Record one encoder batch and its target cosine scores."""

    step: int
    texts: tuple[str, ...]
    similarities: tuple[float, ...]


@dataclass(frozen=True)
class BeamResult:
    """Hold terminal candidates, full expansion trace, and model-query records."""

    finished: tuple[Candidate, ...]
    selected_index: int
    trace: tuple[Candidate, ...]
    lm_calls: int
    encoder_calls: int
    lm_log: tuple[LMCall, ...]
    encoder_log: tuple[EncoderCall, ...]

    @property
    def selected(self) -> Candidate:
        """Return the terminal candidate selected by cosine and proposal order."""
        return self.finished[self.selected_index]


@dataclass(frozen=True)
class BeamStartsResult:
    """Hold independent fixed-prompt searches under one target scorer."""

    starts: tuple[BeamResult, ...]
    selected_start: int

    @property
    def selected(self) -> Candidate:
        """Return the highest-cosine terminal candidate across fixed prompt starts."""
        return self.starts[self.selected_start].selected

    @property
    def lm_calls(self) -> int:
        """Count all language-model queries across prompt starts."""
        return sum(start.lm_calls for start in self.starts)

    @property
    def encoder_calls(self) -> int:
        """Count all candidate encoder queries across prompt starts."""
        return sum(start.encoder_calls for start in self.starts)


@dataclass(frozen=True)
class _Prefix:
    tokens: tuple[int, ...]
    log_prior: float


def _overlap(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set, right_set = set(left), set(right)
    return len(left_set & right_set) / len(left_set | right_set)


def _select_diverse(
    candidates: tuple[Candidate, ...], width: int, penalty: float
) -> tuple[int, ...]:
    remaining = set(range(len(candidates)))
    chosen: list[int] = []
    while remaining and len(chosen) < width:
        index = max(
            remaining,
            key=lambda i: (
                candidates[i].rank_score
                - penalty
                * max(
                    (
                        _overlap(candidates[i].tokens, candidates[j].tokens)
                        for j in chosen
                    ),
                    default=0.0,
                ),
                -i,
            ),
        )
        chosen.append(index)
        remaining.remove(index)
    return tuple(chosen)


def _rank_score(log_prior: float, similarity: float, config: BeamConfig) -> float:
    match config.rank_mode:
        case "cosine":
            return similarity
        case "hybrid":
            return log_prior + config.beta * similarity
        case unreachable:
            assert_never(unreachable)


def _candidate_tokens(
    logprobs: Sequence[float], prefix: tuple[int, ...], config: BeamConfig
) -> tuple[int, ...]:
    match config.eos_policy:
        case "unconditional" | "cap_only":
            choices = (
                token for token in range(len(logprobs)) if token != config.eos_token
            )
        case "top_k":
            choices = iter(range(len(logprobs)))
        case unreachable:
            assert_never(unreachable)
    repeated = set(prefix)

    def proposal_score(token: int) -> tuple[float, int]:
        score = logprobs[token]
        if token in repeated:
            score = (
                score / config.repetition_penalty
                if score > 0
                else score * config.repetition_penalty
            )
        return score, -token

    # Proposal filtering uses the generated prefix; prior weights retain the untouched LM log probabilities.
    ranked = tuple(heapq.nlargest(config.top_k, choices, key=proposal_score))
    return (
        (*ranked, config.eos_token) if config.eos_policy == "unconditional" else ranked
    )


async def search_beam(
    lm: LanguageModel,
    scorer: EmbeddingScorer,
    decode: Decoder,
    config: BeamConfig,
) -> BeamResult:
    """Search top LM continuations with prefix guidance and select by terminal cosine.

    Cosine mode uses LM probabilities to filter each prefix's continuations.
    Hybrid mode ranks those continuations by cumulative log prior plus beta
    times cosine. Repetition penalty applies the Hugging Face sign rule to
    available normalized log probabilities for top-K filtering only; prior
    scores remain the frozen LM log probabilities. These are heuristic
    searches rather than posterior samplers; all encoder and LM invocations
    are recorded.
    """
    frontier = (_Prefix((), 0.0),)
    finished: list[Candidate] = []
    trace: list[Candidate] = []
    lm_calls = 0
    encoder_calls = 0
    lm_log: list[LMCall] = []
    encoder_log: list[EncoderCall] = []

    for step in range(1, config.max_tokens + 1):
        if not frontier:
            break
        distributions = await asyncio.gather(
            *(lm.next_logprobs(prefix.tokens) for prefix in frontier)
        )
        lm_calls += len(frontier)
        lm_log.extend(LMCall(step, prefix.tokens) for prefix in frontier)
        proposals: list[Candidate] = []
        for prefix, logprobs in zip(frontier, distributions):
            for token in _candidate_tokens(logprobs, prefix.tokens, config):
                eos = token == config.eos_token
                tokens = prefix.tokens if eos else prefix.tokens + (token,)
                termination: Literal["eos", "cap"] | None = (
                    "eos" if eos else "cap" if step == config.max_tokens else None
                )
                proposals.append(
                    Candidate(
                        step=step,
                        parent=prefix.tokens,
                        token=token,
                        tokens=tokens,
                        log_prior=prefix.log_prior + float(logprobs[token]),
                        similarity=0.0,
                        rank_score=0.0,
                        termination=termination,
                    )
                )
        texts = tuple(decode(candidate.tokens) for candidate in proposals)
        similarities = await scorer.similarities(texts)
        encoder_calls += len(proposals)
        encoder_log.append(EncoderCall(step, texts, similarities))
        scored = tuple(
            replace(
                candidate,
                similarity=similarity,
                rank_score=_rank_score(candidate.log_prior, similarity, config),
            )
            for candidate, similarity in zip(proposals, similarities)
        )
        active = tuple(
            (index, candidate)
            for index, candidate in enumerate(scored)
            if candidate.termination is None
        )
        kept_active = _select_diverse(
            tuple(candidate for _, candidate in active),
            config.width,
            config.diversity_penalty,
        )
        kept_indices = {active[index][0] for index in kept_active}
        trace.extend(
            replace(candidate, kept=index in kept_indices)
            for index, candidate in enumerate(scored)
        )
        finished.extend(
            candidate for candidate in scored if candidate.termination is not None
        )
        frontier = tuple(
            _Prefix(active[index][1].tokens, active[index][1].log_prior)
            for index in kept_active
        )

    selected_index = max(
        range(len(finished)), key=lambda index: (finished[index].similarity, -index)
    )
    return BeamResult(
        tuple(finished),
        selected_index,
        tuple(trace),
        lm_calls,
        encoder_calls,
        tuple(lm_log),
        tuple(encoder_log),
    )


async def search_beam_starts(
    models: tuple[LanguageModel, *tuple[LanguageModel, ...]],
    scorer: EmbeddingScorer,
    decode: Decoder,
    config: BeamConfig,
) -> BeamStartsResult:
    """Search fixed LM prompt starts under one target and select by terminal cosine."""
    starts = tuple(
        [await search_beam(model, scorer, decode, config) for model in models]
    )
    selected_start = max(
        range(len(starts)),
        key=lambda index: (starts[index].selected.similarity, -index),
    )
    return BeamStartsResult(starts, selected_start)
