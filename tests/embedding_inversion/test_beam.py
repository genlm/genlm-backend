import asyncio
import math

from examples.embedding_inversion._algorithms.beam import (
    BeamConfig,
    search_beam,
    search_beam_starts,
)


class FiniteLM:
    def __init__(self, distributions):
        self.distributions = distributions
        self.contexts = []

    async def next_logprobs(self, tokens):
        self.contexts.append(tokens)
        return self.distributions[tokens]


class TextScorer:
    def __init__(self, values):
        self.values = values
        self.batches = []

    async def similarities(self, texts):
        self.batches.append(texts)
        return tuple(self.values[text] for text in texts)


def test_beam_scores_top_candidates_and_explicit_eos_then_selects_by_terminal_cosine():
    lm = FiniteLM(
        {
            (): (math.log(0.7), math.log(0.2), math.log(0.1)),
            (0,): (math.log(0.8), math.log(0.1), math.log(0.1)),
        }
    )
    scorer = TextScorer({"": -0.5, "a": 0.3, "b": 0.0, "aa": 0.1, "ab": 0.9})
    result = asyncio.run(
        search_beam(
            lm,
            scorer,
            lambda ids: "".join("ab"[i] for i in ids),
            BeamConfig(
                width=1,
                top_k=2,
                max_tokens=2,
                eos_token=2,
                beta=1.0,
                diversity_penalty=0.0,
            ),
        )
    )

    assert lm.contexts == [(), (0,)]
    assert scorer.batches == [("a", "b", ""), ("aa", "ab", "a")]
    assert result.lm_calls == 2
    assert result.encoder_calls == 6
    assert [(call.step, call.tokens) for call in result.lm_log] == [(1, ()), (2, (0,))]
    assert [
        (call.step, call.texts, call.similarities) for call in result.encoder_log
    ] == [
        (1, ("a", "b", ""), (0.3, 0.0, -0.5)),
        (2, ("aa", "ab", "a"), (0.1, 0.9, 0.3)),
    ]
    assert {
        (candidate.tokens, candidate.termination) for candidate in result.finished
    } == {((), "eos"), ((0,), "eos"), ((0, 0), "cap"), ((0, 1), "cap")}
    assert result.selected.tokens == (0, 1)
    assert result.selected.similarity == 0.9
    assert math.isclose(result.selected.log_prior, math.log(0.7) + math.log(0.1))
    assert len(result.trace) == 6
    assert sum(candidate.kept for candidate in result.trace) == 1


def test_beam_diversity_retains_distinct_prefix_families():
    lm = FiniteLM(
        {
            (): (math.log(0.51), math.log(0.48), math.log(0.01)),
            (0,): (math.log(0.5), math.log(0.49), math.log(0.01)),
            (1,): (math.log(0.5), math.log(0.49), math.log(0.01)),
            (0, 0): (math.log(0.5), math.log(0.49), math.log(0.01)),
            (0, 1): (math.log(0.5), math.log(0.49), math.log(0.01)),
            (1, 0): (math.log(0.5), math.log(0.49), math.log(0.01)),
            (1, 1): (math.log(0.5), math.log(0.49), math.log(0.01)),
        }
    )
    scorer = TextScorer(
        {
            "": -1.0,
            **{
                text: 0.0
                for text in (
                    "a",
                    "b",
                    "aa",
                    "ab",
                    "ba",
                    "bb",
                    "aaa",
                    "aab",
                    "aba",
                    "abb",
                    "baa",
                    "bab",
                    "bba",
                    "bbb",
                )
            },
        }
    )
    result = asyncio.run(
        search_beam(
            lm,
            scorer,
            lambda ids: "".join("ab"[i] for i in ids),
            BeamConfig(
                width=2,
                top_k=2,
                max_tokens=3,
                eos_token=2,
                beta=0.0,
                diversity_penalty=2.0,
            ),
        )
    )

    assert lm.contexts[:3] == [(), (0,), (1,)]
    assert lm.contexts[3:] == [(0, 0), (1, 1)]
    assert len(result.finished) == 9


def test_equal_cosine_terminal_candidates_use_proposal_order_as_tie_break():
    lm = FiniteLM({(): (math.log(0.1), math.log(0.9))})
    scorer = TextScorer({"": 0.2})
    result = asyncio.run(
        search_beam(
            lm,
            scorer,
            lambda ids: "",
            BeamConfig(
                width=2,
                top_k=1,
                max_tokens=1,
                eos_token=1,
                beta=1.0,
                diversity_penalty=0.0,
            ),
        )
    )

    assert len(result.trace) == 2
    assert {candidate.termination for candidate in result.finished} == {"eos", "cap"}
    assert result.selected.termination == "cap"
    assert math.isclose(result.selected.log_prior, math.log(0.1))


def test_fixed_prompt_starts_preserve_per_start_traces_and_select_by_cosine():
    first = FiniteLM({(): (math.log(0.9), math.log(0.05), math.log(0.05))})
    second = FiniteLM({(): (math.log(0.1), math.log(0.2), math.log(0.7))})
    scorer = TextScorer({"": -1.0, "a": 0.3, "b": 0.9})
    result = asyncio.run(
        search_beam_starts(
            (first, second),
            scorer,
            lambda ids: "".join("ab"[i] for i in ids),
            BeamConfig(
                width=1,
                top_k=1,
                max_tokens=1,
                eos_token=2,
                beta=0.0,
                diversity_penalty=0.0,
            ),
        )
    )

    assert result.selected_start == 1
    assert result.selected.tokens == (1,)
    assert result.selected.log_prior < result.starts[0].selected.log_prior
    assert result.lm_calls == 2
    assert result.encoder_calls == 4
    assert len(result.starts) == 2
    assert all(len(start.trace) == 2 for start in result.starts)


def test_beta_changes_active_prefix_when_prior_and_cosine_disagree():
    distributions = {
        (): (math.log(0.7), math.log(0.2), math.log(0.1)),
        (0,): (math.log(0.4), math.log(0.4), math.log(0.2)),
        (1,): (math.log(0.4), math.log(0.4), math.log(0.2)),
    }
    scorer = TextScorer(
        {"": -1.0, "a": 0.0, "b": 0.9, "aa": 0.0, "ab": 0.0, "ba": 0.0, "bb": 0.0}
    )

    def chosen_prefix(beta, rank_mode="hybrid"):
        lm = FiniteLM(distributions)
        asyncio.run(
            search_beam(
                lm,
                scorer,
                lambda ids: "".join("ab"[i] for i in ids),
                BeamConfig(
                    width=1,
                    top_k=2,
                    max_tokens=2,
                    eos_token=2,
                    beta=beta,
                    diversity_penalty=0.0,
                    rank_mode=rank_mode,
                ),
            )
        )
        return lm.contexts[1]

    assert chosen_prefix(0.0) == (0,)
    assert chosen_prefix(2.0) == (1,)
    assert chosen_prefix(0.0, "cosine") == (1,)


def test_top_k_eos_policy_uses_lm_rank_and_cap_only_forces_content_termination():
    scorer = TextScorer({"": -1.0, "a": 0.2, "b": 0.3})

    def decode(ids):
        return "".join("ab"[i] for i in ids)

    def run(distribution, policy):
        return asyncio.run(
            search_beam(
                FiniteLM({(): tuple(map(math.log, distribution))}),
                scorer,
                decode,
                BeamConfig(
                    width=1,
                    top_k=1,
                    max_tokens=1,
                    eos_token=2,
                    beta=0.0,
                    diversity_penalty=0.0,
                    eos_policy=policy,
                ),
            )
        )

    low_eos = run((0.9, 0.05, 0.05), "top_k")
    high_eos = run((0.05, 0.05, 0.9), "top_k")
    fixed_cap = run((0.05, 0.05, 0.9), "cap_only")

    assert [
        (candidate.token, candidate.termination) for candidate in low_eos.finished
    ] == [(0, "cap")]
    assert [
        (candidate.token, candidate.termination) for candidate in high_eos.finished
    ] == [(2, "eos")]
    assert [
        (candidate.token, candidate.termination) for candidate in fixed_cap.finished
    ] == [(0, "cap")]
    assert math.isclose(fixed_cap.finished[0].log_prior, math.log(0.05))
    assert (low_eos.encoder_calls, high_eos.encoder_calls, fixed_cap.encoder_calls) == (
        1,
        1,
        1,
    )


def test_repetition_penalty_changes_top_k_filter_but_preserves_raw_prior():
    distributions = {
        (): (math.log(0.9), math.log(0.09), math.log(0.01)),
        (0,): (math.log(0.549), math.log(0.45), math.log(0.001)),
    }
    scorer = TextScorer({"a": 0.3, "aa": 0.9, "ab": 0.1})

    def run(penalty):
        lm = FiniteLM(distributions)
        result = asyncio.run(
            search_beam(
                lm,
                scorer,
                lambda ids: "".join("ab"[token] for token in ids),
                BeamConfig(
                    width=1,
                    top_k=1,
                    max_tokens=2,
                    eos_token=2,
                    beta=0.0,
                    diversity_penalty=0.0,
                    eos_policy="cap_only",
                    repetition_penalty=penalty,
                ),
            )
        )
        return result, lm.contexts

    default, default_contexts = run(1.0)
    penalized, penalized_contexts = run(1.5)

    assert default_contexts == penalized_contexts == [(), (0,)]
    assert default.finished[0].tokens == (0, 0)
    assert penalized.finished[0].tokens == (0, 1)
    assert math.isclose(default.finished[0].log_prior, math.log(0.9) + math.log(0.549))
    assert math.isclose(penalized.finished[0].log_prior, math.log(0.9) + math.log(0.45))
    assert default.selected.similarity == 0.9
    assert penalized.selected.similarity == 0.1
