"""Tokenizer-vocabulary probe bank and clue-conditioned beam search."""

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch

from .beam import (
    BeamConfig,
    BeamResult,
    Decoder,
    EmbeddingScorer,
    LanguageModel,
    search_beam,
)

BANK_SIZE = 12000


class Encoder(Protocol):
    """Encode a fixed tuple of text under the frozen victim encoder."""

    def encode(self, texts: tuple[str, ...]) -> torch.Tensor: ...


@dataclass(frozen=True)
class BankProvenance:
    """Identify the vocabulary, encoder, vector bytes, and one-time bank cost."""

    encoder: str
    encoder_revision: str
    tokenizer_vocabulary_sha256: str
    selected_words_sha256: str
    vectors_sha256: str
    bank_encoder_queries: int
    bank_encoder_batches: int


@dataclass(frozen=True)
class VocabularyBank:
    """Store canonical full-word vocabulary entries and their unit embeddings."""

    words: tuple[str, ...]
    token_ids: tuple[int, ...]
    vectors: torch.Tensor
    provenance: BankProvenance


@dataclass(frozen=True)
class ProbeConfig:
    """Fix clue selection and the assistant prefill preceding generated content."""

    clue_count: int
    diversity_penalty: float
    assistant_prefill: str = ""

    def __post_init__(self) -> None:
        if self.clue_count < 1:
            raise ValueError("clue_count must be positive")
        if self.diversity_penalty < 0:
            raise ValueError("diversity_penalty must be nonnegative")


@dataclass(frozen=True)
class Clue:
    """Record one selected vocabulary word and its target/redundancy scores."""

    word: str
    token_id: int
    target_cosine: float
    redundancy: float
    selection_score: float


@dataclass(frozen=True)
class ProbeResult:
    """Retain clue selection, exact Qwen prompt, beam trace, and bank provenance."""

    clues: tuple[Clue, ...]
    prompt: str
    beam: BeamResult
    bank_provenance: BankProvenance

    @property
    def bank_encoder_queries(self) -> int:
        """Report the bank's amortizable one-time encoder cost."""
        return self.bank_provenance.bank_encoder_queries

    @property
    def target_probe_encoder_queries(self) -> int:
        """Report target-time encoder queries used to score cached clue vectors."""
        return 0


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def vocabulary_words(
    vocab: Mapping[str, int], special_tokens: AbstractSet[str], count: int
) -> tuple[tuple[str, int], ...]:
    """Select full ASCII alphabetic tokenizer words by canonical token ID."""
    if count < 1:
        raise ValueError("count must be positive")
    # Token IDs define the tokenizer's stable vocabulary order independent of mapping insertion order.
    ordered = sorted(vocab.items(), key=lambda entry: (entry[1], entry[0]))
    words = tuple(
        (word, token_id)
        for word, token_id in ordered
        if word not in special_tokens
        and not word.startswith("##")
        and word.isascii()
        and word.isalpha()
    )
    if len(words) < count:
        raise ValueError(
            f"only {len(words)} full alphabetic tokenizer words; requested {count}"
        )
    return words[:count]


def build_bank(
    encoder: Encoder,
    vocab: Mapping[str, int],
    special_tokens: AbstractSet[str],
    encoder_name: str,
    encoder_revision: str,
    count: int = BANK_SIZE,
    encoder_batch_size: int = 32,
) -> VocabularyBank:
    """Embed a fixed tokenizer-only word bank once under the frozen encoder."""
    selected = vocabulary_words(vocab, special_tokens, count)
    words = tuple(word for word, _ in selected)
    vectors = encoder.encode(words).detach().cpu().to(torch.float32).contiguous()
    if vectors.shape[0] != count:
        raise ValueError("encoder returned an unexpected number of vectors")
    provenance = BankProvenance(
        encoder=encoder_name,
        encoder_revision=encoder_revision,
        # Canonical token order makes this hash independent of mapping insertion order.
        tokenizer_vocabulary_sha256=_json_hash(
            sorted(vocab.items(), key=lambda entry: (entry[1], entry[0]))
        ),
        selected_words_sha256=_json_hash(selected),
        vectors_sha256=hashlib.sha256(vectors.numpy().tobytes()).hexdigest(),
        bank_encoder_queries=count,
        bank_encoder_batches=(count + encoder_batch_size - 1) // encoder_batch_size,
    )
    return VocabularyBank(
        words, tuple(token_id for _, token_id in selected), vectors, provenance
    )


def save_bank(bank: VocabularyBank, path: Path) -> str:
    """Atomically store the bank and return the artifact SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(
        {
            "words": bank.words,
            "token_ids": bank.token_ids,
            "vectors": bank.vectors,
            "provenance": asdict(bank.provenance),
        },
        temporary,
    )
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bank(path: Path) -> VocabularyBank:
    """Load a bank and verify its selected vocabulary and embedding hashes."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    provenance = BankProvenance(**raw["provenance"])
    words = tuple(raw["words"])
    token_ids = tuple(raw["token_ids"])
    vectors = raw["vectors"].to(torch.float32).contiguous()
    if _json_hash(tuple(zip(words, token_ids))) != provenance.selected_words_sha256:
        raise ValueError("bank vocabulary hash mismatch")
    if (
        hashlib.sha256(vectors.numpy().tobytes()).hexdigest()
        != provenance.vectors_sha256
    ):
        raise ValueError("bank vector hash mismatch")
    return VocabularyBank(words, token_ids, vectors, provenance)


def select_clues(
    bank: VocabularyBank, target: torch.Tensor, config: ProbeConfig
) -> tuple[Clue, ...]:
    """Greedily maximize target cosine minus redundancy with earlier clues."""
    if config.clue_count > len(bank.words):
        raise ValueError("clue_count exceeds bank size")
    target_scores = bank.vectors @ target.detach().cpu().to(torch.float32)
    redundancy = torch.zeros_like(target_scores)
    available = torch.ones(len(bank.words), dtype=torch.bool)
    selected: list[Clue] = []
    for _ in range(config.clue_count):
        values = (target_scores - config.diversity_penalty * redundancy).masked_fill(
            ~available, -torch.inf
        )
        index = int(torch.argmax(values))
        selected.append(
            Clue(
                word=bank.words[index],
                token_id=bank.token_ids[index],
                target_cosine=float(target_scores[index]),
                redundancy=float(redundancy[index]),
                selection_score=float(values[index]),
            )
        )
        available[index] = False
        redundancy = torch.maximum(
            redundancy, (bank.vectors @ bank.vectors[index]).clamp_min(0)
        )
    return tuple(selected)


def probe_prompt(clues: Sequence[Clue]) -> str:
    """Construct a Qwen prompt containing selected clue words only."""
    words = json.dumps([clue.word for clue in clues], ensure_ascii=False)
    return (
        "<|im_start|>user\n"
        "Write one standalone factual sentence related to these words. Output only the sentence.\n"
        f"{words}\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


async def search_probe_beam(
    bank: VocabularyBank,
    target: torch.Tensor,
    probe_config: ProbeConfig,
    make_model: Callable[[str], LanguageModel],
    scorer: EmbeddingScorer,
    decode: Decoder,
    beam_config: BeamConfig,
) -> ProbeResult:
    """Run frozen-Qwen beam search from target-selected generic vocabulary clues."""
    clues = select_clues(bank, target, probe_config)
    prompt = probe_prompt(clues) + probe_config.assistant_prefill
    beam = await search_beam(make_model(prompt), scorer, decode, beam_config)
    return ProbeResult(clues, prompt, beam, bank.provenance)
