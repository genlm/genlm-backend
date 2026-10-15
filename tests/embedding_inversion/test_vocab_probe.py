import asyncio
import math

import pytest
import torch

from examples.embedding_inversion._algorithms.beam import BeamConfig
from examples.embedding_inversion._algorithms.vocab_probe import (
    ProbeConfig,
    build_bank,
    load_bank,
    save_bank,
    search_probe_beam,
    select_clues,
    vocabulary_words,
)


class TinyEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        vectors = {
            "apple": (1.0, 0.0),
            "fruit": (0.8, 0.6),
            "river": (0.0, 1.0),
        }
        return torch.tensor([vectors[text] for text in texts], dtype=torch.float32)


class TinyLM:
    async def next_logprobs(self, tokens):
        return (math.log(0.8), math.log(0.1), math.log(0.1))


class TinyScorer:
    async def similarities(self, texts):
        return tuple({"": -1.0, "a": 0.9, "b": 0.2}[text] for text in texts)


def test_bank_uses_only_full_alphabetic_vocabulary_words_in_token_id_order():
    vocab = {
        "river": 8,
        "##piece": 2,
        "apple": 4,
        "[CLS]": 0,
        "fruit": 6,
        "x2": 3,
        "école": 1,
    }
    expected = (("apple", 4), ("fruit", 6), ("river", 8))

    assert vocabulary_words(vocab, {"[CLS]"}, 3) == expected
    assert (
        vocabulary_words(dict(reversed(tuple(vocab.items()))), {"[CLS]"}, 3) == expected
    )

    encoder = TinyEncoder()
    bank = build_bank(encoder, vocab, {"[CLS]"}, "facebook/contriever", "pinned-sha", 3)
    assert bank.words == ("apple", "fruit", "river")
    assert encoder.calls == [("apple", "fruit", "river")]
    assert bank.provenance.encoder_revision == "pinned-sha"
    assert bank.provenance.bank_encoder_queries == 3
    assert bank.provenance.tokenizer_vocabulary_sha256
    assert bank.provenance.vectors_sha256


def test_bank_artifact_round_trip_preserves_provenance_and_vectors(tmp_path):
    bank = build_bank(
        TinyEncoder(),
        {"apple": 0, "fruit": 1, "river": 2},
        set(),
        "encoder",
        "revision",
        3,
    )
    path = tmp_path / "bank.pt"

    artifact_hash = save_bank(bank, path)
    restored = load_bank(path)

    assert len(artifact_hash) == 64
    assert restored.words == bank.words
    assert restored.token_ids == bank.token_ids
    assert restored.provenance == bank.provenance
    assert torch.equal(restored.vectors, bank.vectors)

    raw = torch.load(path, weights_only=True)
    raw["vectors"][0, 0] = -raw["vectors"][0, 0]
    torch.save(raw, path)
    with pytest.raises(ValueError, match="vector hash mismatch"):
        load_bank(path)


def test_clue_selection_scores_target_and_penalizes_semantic_redundancy():
    encoder = TinyEncoder()
    bank = build_bank(
        encoder, {"apple": 0, "fruit": 1, "river": 2}, set(), "encoder", "revision", 3
    )
    target = torch.tensor((1.0, 0.0), dtype=torch.float32)

    clues = select_clues(bank, target, ProbeConfig(clue_count=2, diversity_penalty=1.1))

    assert tuple(clue.word for clue in clues) == ("apple", "river")
    assert clues[0].target_cosine == 1.0
    assert clues[1].target_cosine == 0.0
    assert clues[1].selection_score == 0.0
    assert encoder.calls == [("apple", "fruit", "river")]


@pytest.mark.parametrize("assistant_prefill", ("", "<think>\n\n</think>\n\n"))
def test_probe_prompt_and_beam_use_only_selected_clues_and_preserve_trace(
    assistant_prefill,
):
    encoder = TinyEncoder()
    bank = build_bank(
        encoder, {"apple": 0, "fruit": 1, "river": 2}, set(), "encoder", "revision", 3
    )
    prompts = []

    def make_model(prompt):
        prompts.append(prompt)
        return TinyLM()

    result = asyncio.run(
        search_probe_beam(
            bank,
            torch.tensor((1.0, 0.0)),
            ProbeConfig(
                clue_count=2, diversity_penalty=1.1, assistant_prefill=assistant_prefill
            ),
            make_model,
            TinyScorer(),
            lambda ids: "ab"[ids[0]] if ids else "",
            BeamConfig(
                width=1,
                top_k=2,
                max_tokens=1,
                eos_token=2,
                beta=0.0,
                diversity_penalty=0.0,
            ),
        )
    )

    assert prompts == [result.prompt]
    assert result.prompt == (
        "<|im_start|>user\n"
        "Write one standalone factual sentence related to these words. Output only the sentence.\n"
        '["apple", "river"]\n'
        "<|im_end|>\n<|im_start|>assistant\n" + assistant_prefill
    )
    assert result.beam.selected.similarity == 0.9
    assert len(result.beam.trace) == 3
    assert result.bank_encoder_queries == 3
    assert result.target_probe_encoder_queries == 0
    assert encoder.calls == [("apple", "fruit", "river")]
