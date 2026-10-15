"""Run frozen embedding inversion and retain the evidence needed for offline rendering."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, assert_never

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

import genlm.backend

from ._algorithms.beam import BeamConfig, BeamResult, search_beam
from ._algorithms.lookahead_smc import (
    LookaheadConfig,
    LookaheadResult,
    PositiveInt,
    sample_lookahead_smc,
)
from ._algorithms.scoring import MemoizedScorer
from ._algorithms.vocab_probe import (
    BANK_SIZE,
    ProbeConfig,
    VocabularyBank,
    build_bank,
    load_bank,
    save_bank,
    search_probe_beam,
)
from .bundle import (
    ResultRecord,
    complete_bundle,
    load_targets,
    prepare_output,
    sha256,
    token_f1,
    write_jsonl,
)

_ENCODER = "facebook/contriever"
_ENCODER_REVISION = "2bd46a25019aeea091fd42d1f0fd4801675cf699"
_PROMPT = (
    "<|im_start|>user\nWrite a short factual passage.\n"
    "<|im_end|>\n<|im_start|>assistant\n"
)
_CAP = 40


class _Method(StrEnum):
    CLUE = "clue"
    BEAM = "beam"
    LOOKAHEAD = "lookahead"


class _Tokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(
        self,
        tokens: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


class _Backend(Protocol):
    tokenizer: _Tokenizer

    async def next_token_logprobs(self, tokens: list[int]) -> torch.Tensor: ...


@dataclass(frozen=True)
class _Profile:
    runtime: Literal["mlx", "cuda"]
    model: str
    revision: str
    encoder_device: Literal["cpu", "cuda"]
    encoder_batch_size: int


def _profile(runtime: Literal["mlx", "cuda"]) -> _Profile:
    match runtime:
        case "mlx":
            return _Profile(
                runtime,
                "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "8b403126fc14f14cfc99bb4cfa72ecbc129ea677",
                "cpu",
                32,
            )
        case "cuda":
            return _Profile(
                runtime,
                "Qwen/Qwen2.5-3B-Instruct",
                "aa8e72537993ba99e69dfaafa59ed015b17504d1",
                "cuda",
                128,
            )
        case unreachable:
            assert_never(unreachable)


def _backend_options(runtime: Literal["mlx", "cuda"]) -> dict[str, object]:
    match runtime:
        case "mlx":
            return {"cache_size": 4096}
        case "cuda":
            return {
                "engine_opts": {
                    "gpu_memory_utilization": 0.8,
                    "max_model_len": 1024,
                    "max_num_seqs": 64,
                    "language_model_only": True,
                }
            }
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True)
class _PromptModel:
    backend: _Backend
    prompt_tokens: tuple[int, ...]

    async def next_logprobs(self, tokens: tuple[int, ...]) -> tuple[float, ...]:
        values = await self.backend.next_token_logprobs(
            list(self.prompt_tokens + tokens)
        )
        return tuple(float(value) for value in values.tolist())


@dataclass(frozen=True)
class _Session:
    backend: _Backend
    release: Callable[[], Awaitable[None]]
    close: Callable[[], None]

    def model(self, prompt: str) -> _PromptModel:
        return _PromptModel(
            self.backend,
            tuple(self.backend.tokenizer.encode(prompt, add_special_tokens=False)),
        )

    def decode(self, tokens: tuple[int, ...]) -> str:
        return self.backend.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


def _load_session(profile: _Profile, model_path: str) -> _Session:
    match profile.runtime:
        case "mlx":
            backend = genlm.backend.load_model_by_name(
                model_path, backend="mlx", llm_opts=_backend_options("mlx")
            )

            async def release_mlx() -> None:
                backend.clear_cache()

            return _Session(backend, release_mlx, backend.clear_cache)
        case "cuda":
            backend = genlm.backend.load_model_by_name(
                model_path, backend="vllm", llm_opts=_backend_options("cuda")
            )
            return _Session(backend, backend.release_all, backend.cleanup)
        case unreachable:
            assert_never(unreachable)


class _Contriever:
    def __init__(self, snapshot: str, profile: _Profile):
        tokenizer = AutoTokenizer.from_pretrained(snapshot)
        if tokenizer is None:
            raise RuntimeError("the pinned encoder tokenizer could not be loaded")
        self.tokenizer = tokenizer
        self.model = (
            AutoModel.from_pretrained(snapshot).to(profile.encoder_device).eval()
        )
        self.device = profile.encoder_device
        self.batch_size = profile.encoder_batch_size

    def encode(self, texts: tuple[str, ...]) -> torch.Tensor:
        vectors: list[torch.Tensor] = []
        with torch.inference_mode():
            for begin in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    list(texts[begin : begin + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                vectors.append(torch.nn.functional.normalize(pooled, dim=-1))
        return torch.cat(vectors)


@dataclass(frozen=True)
class _TargetScorer:
    encoder: _Contriever
    target: torch.Tensor

    async def similarities(self, texts: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(
            float(value)
            for value in (self.encoder.encode(texts) @ self.target).tolist()
        )


@dataclass(frozen=True)
class _Beam:
    pass


@dataclass(frozen=True)
class _Clue:
    bank: VocabularyBank


@dataclass(frozen=True)
class _Lookahead:
    pass


_Plan = _Beam | _Clue | _Lookahead


def _plans(
    methods: tuple[_Method, ...], encoder: _Contriever, bank_cache: Path
) -> tuple[tuple[_Plan, ...], dict[str, object]]:
    plans: list[_Plan] = []
    bank_record: dict[str, object] = {
        "used": False,
        "build_encoder_queries": 0,
        "seconds": 0.0,
    }
    for method in methods:
        match method:
            case _Method.BEAM:
                plans.append(_Beam())
            case _Method.LOOKAHEAD:
                plans.append(_Lookahead())
            case _Method.CLUE:
                began = time.perf_counter()
                reused = bank_cache.exists()
                if reused:
                    bank = load_bank(bank_cache)
                else:
                    bank = build_bank(
                        encoder,
                        encoder.tokenizer.get_vocab(),
                        set(encoder.tokenizer.all_special_tokens),
                        _ENCODER,
                        _ENCODER_REVISION,
                        count=BANK_SIZE,
                        encoder_batch_size=encoder.batch_size,
                    )
                    save_bank(bank, bank_cache)
                if (bank.provenance.encoder, bank.provenance.encoder_revision) != (
                    _ENCODER,
                    _ENCODER_REVISION,
                ):
                    raise ValueError(
                        "bank encoder identity differs from the pinned encoder"
                    )
                if len(bank.words) != BANK_SIZE:
                    raise ValueError(
                        f"bank must contain the fixed {BANK_SIZE} vocabulary words"
                    )
                bank_record = {
                    "used": True,
                    "cache_reused": reused,
                    "artifact_sha256": sha256(bank_cache),
                    "cache_path": str(bank_cache.resolve()),
                    "provenance": asdict(bank.provenance),
                    "vector_dtype": str(bank.vectors.dtype),
                    "vector_shape": tuple(bank.vectors.shape),
                    "build_encoder_queries": 0
                    if reused
                    else bank.provenance.bank_encoder_queries,
                    "seconds": time.perf_counter() - began,
                    "rebuild_caveat": "Rebuilding on another device or library version can change vector rounding and near-tied clue selections.",
                }
                plans.append(_Clue(bank))
            case unreachable:
                assert_never(unreachable)
    return tuple(plans), bank_record


def _beam_config(eos_token: int) -> BeamConfig:
    return BeamConfig(
        width=8,
        top_k=16,
        max_tokens=_CAP,
        eos_token=eos_token,
        beta=32.0,
        diversity_penalty=0.2,
        rank_mode="cosine",
        eos_policy="top_k",
        repetition_penalty=1.0,
    )


def _lookahead_config(eos_token: int, seed: int) -> LookaheadConfig:
    return LookaheadConfig(
        particles=PositiveInt(8),
        max_tokens=PositiveInt(_CAP),
        rollouts=PositiveInt(2),
        lookahead_interval=PositiveInt(8),
        eos_token=eos_token,
        beta=128.0,
        ess_fraction=0.5,
        seed=seed,
    )


@dataclass(frozen=True)
class _Outcome:
    method: _Method
    tokens: tuple[int, ...]
    cosine: float
    termination: Literal["eos", "cap"]
    archive_texts: tuple[str, ...]
    artifact: dict[str, object]
    cost: dict[str, int]


def _beam_outcome(result: BeamResult, session: _Session, method: _Method) -> _Outcome:
    termination = result.selected.termination
    if termination is None:
        raise RuntimeError("beam selected an active prefix")
    archive = tuple(
        {**asdict(entry), "text": session.decode(entry.tokens)}
        for entry in result.finished
    )
    return _Outcome(
        method,
        result.selected.tokens,
        result.selected.similarity,
        termination,
        tuple(session.decode(entry.tokens) for entry in result.finished),
        {
            "selected_index": result.selected_index,
            "heuristic_archive": archive,
            "trace": tuple(
                {**asdict(entry), "text": session.decode(entry.tokens)}
                for entry in result.trace
            ),
            "lm_log": tuple(asdict(entry) for entry in result.lm_log),
            "encoder_log": tuple(asdict(entry) for entry in result.encoder_log),
        },
        {
            "lm_calls": result.lm_calls,
            "rollout_lm_calls": 0,
            "encoder_calls": result.encoder_calls,
            "completed_paths": len(result.finished),
            "resamplings": 0,
        },
    )


def _lookahead_outcome(result: LookaheadResult, session: _Session) -> _Outcome:
    maximum = max(particle.log_weight for particle in result.particles)
    weights = tuple(
        math.exp(particle.log_weight - maximum) for particle in result.particles
    )
    total = sum(weights)
    selected = result.best_cosine
    return _Outcome(
        _Method.LOOKAHEAD,
        selected.tokens,
        selected.similarity,
        selected.termination,
        tuple(session.decode(entry.tokens) for entry in result.archive),
        {
            "selected": {**asdict(selected), "text": session.decode(selected.tokens)},
            "heuristic_archive": tuple(
                {**asdict(entry), "text": session.decode(entry.tokens)}
                for entry in result.archive
            ),
            "weighted_particles": tuple(
                {
                    **asdict(particle),
                    "text": session.decode(particle.tokens),
                    "normalized_weight": weight / total,
                }
                for particle, weight in zip(result.particles, weights)
            ),
            "trace": tuple(
                {
                    **asdict(entry),
                    "text": session.decode(entry.tokens),
                    "rollouts": tuple(
                        {**asdict(rollout), "text": session.decode(rollout.tokens)}
                        for rollout in entry.rollouts
                    ),
                }
                for entry in result.trace
            ),
            "diagnostics": tuple(asdict(entry) for entry in result.diagnostics),
            "log_normalizer_estimate": result.log_normalizer_estimate,
        },
        {
            "lm_calls": result.lm_calls,
            "rollout_lm_calls": result.rollout_lm_calls,
            "encoder_calls": result.encoder_calls,
            "completed_paths": len(result.archive),
            "resamplings": result.resamplings,
        },
    )


async def _search(
    plan: _Plan,
    session: _Session,
    target: torch.Tensor,
    scorer: MemoizedScorer,
    eos_token: int,
    seed: int,
) -> _Outcome:
    match plan:
        case _Beam():
            beam = await search_beam(
                session.model(_PROMPT), scorer, session.decode, _beam_config(eos_token)
            )
            return _beam_outcome(beam, session, _Method.BEAM)
        case _Clue(bank):
            probe = await search_probe_beam(
                bank,
                target,
                ProbeConfig(clue_count=8, diversity_penalty=0.25),
                session.model,
                scorer,
                session.decode,
                _beam_config(eos_token),
            )
            result = _beam_outcome(probe.beam, session, _Method.CLUE)
            return _Outcome(
                result.method,
                result.tokens,
                result.cosine,
                result.termination,
                result.archive_texts,
                {
                    **result.artifact,
                    "prompt": probe.prompt,
                    "prompt_tokens": session.model(probe.prompt).prompt_tokens,
                    "clues": tuple(asdict(clue) for clue in probe.clues),
                    "bank_provenance": asdict(probe.bank_provenance),
                },
                result.cost,
            )
        case _Lookahead():
            result = await sample_lookahead_smc(
                session.model(_PROMPT),
                scorer,
                session.decode,
                _lookahead_config(eos_token, seed),
            )
            return _lookahead_outcome(result, session)
        case unreachable:
            assert_never(unreachable)


def _write_search(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def _source_hashes() -> dict[str, str]:
    roots = {
        "example": Path(__file__).parent,
        "genlm": Path(genlm.backend.__file__).parent,
    }
    return {
        f"{name}/{path.relative_to(root).as_posix()}": sha256(path)
        for name, root in roots.items()
        for path in root.rglob("*.py")
    }


def _environment(profile: _Profile) -> dict[str, object]:
    packages = ("genlm-backend", "torch", "transformers", "huggingface-hub")
    packages += ("mlx", "mlx-lm") if profile.runtime == "mlx" else ("vllm",)
    hardware: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    if profile.runtime == "cuda":
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        hardware.update(
            {
                "gpu": device.name,
                "gpu_memory_bytes": device.total_memory,
                "cuda": torch.version.cuda,
            }
        )
    return {
        "versions": {
            package: importlib.metadata.version(package) for package in packages
        },
        "hardware": hardware,
    }


async def run(
    inputs: Path,
    output: Path,
    runtime: Literal["mlx", "cuda"],
    methods: tuple[str, ...],
    seed: int,
    bank_cache: Path,
) -> None:
    """Perform fresh inference and write a complete, independently renderable result bundle."""
    began_wall = time.perf_counter()
    selected_methods = tuple(_Method(method) for method in methods)
    if not selected_methods or len(set(selected_methods)) != len(selected_methods):
        raise ValueError("methods must be a nonempty sequence without duplicates")
    targets = load_targets(inputs)
    profile = _profile(runtime)
    prepare_output(output)
    write_jsonl(output / "inputs.jsonl", (asdict(target) for target in targets))
    began_init = time.perf_counter()
    model_path = snapshot_download(repo_id=profile.model, revision=profile.revision)
    encoder_path = snapshot_download(repo_id=_ENCODER, revision=_ENCODER_REVISION)
    session = _load_session(profile, model_path)
    records: list[ResultRecord] = []
    try:
        encoder = _Contriever(encoder_path, profile)
        eos_token = session.backend.tokenizer.eos_token_id
        if eos_token is None:
            raise ValueError("the pinned generator tokenizer must define EOS")
        initialization_seconds = time.perf_counter() - began_init
        plans, bank_record = _plans(selected_methods, encoder, bank_cache)
        began_targets = time.perf_counter()
        vectors = encoder.encode(tuple(target.text for target in targets))
        target_seconds = time.perf_counter() - began_targets
        write_jsonl(
            output / "targets.jsonl",
            (
                {
                    "id": target.id,
                    "text": target.text,
                    "vector": vector.detach().cpu().tolist(),
                }
                for target, vector in zip(targets, vectors)
            ),
        )
        for target, vector in zip(targets, vectors):
            identity_hash = hashlib.sha256(target.id.encode()).hexdigest()
            effective_seed = seed + int(identity_hash[:8], 16)
            for plan in plans:
                scorer = MemoizedScorer(_TargetScorer(encoder, vector))
                began_search = time.perf_counter()
                outcome = await _search(
                    plan, session, vector, scorer, eos_token, effective_seed
                )
                search_seconds = time.perf_counter() - began_search
                generated = session.decode(outcome.tokens)
                cost = {
                    **outcome.cost,
                    "encoder_requested_queries": scorer.requested_queries,
                    "encoder_evaluated_queries": scorer.evaluated_queries,
                }
                artifact = f"search/{identity_hash}-{outcome.method.value}.json.gz"
                _write_search(
                    output / artifact,
                    {
                        "id": target.id,
                        "method": outcome.method.value,
                        "target_id": target.id,
                        "seed": seed,
                        "effective_seed": effective_seed,
                        "prompt": _PROMPT,
                        "prompt_tokens": session.model(_PROMPT).prompt_tokens,
                        "max_tokens": _CAP,
                        "eos_token": eos_token,
                        "search_seconds": search_seconds,
                        "cost": cost,
                        **outcome.artifact,
                    },
                )
                records.append(
                    ResultRecord(
                        id=target.id,
                        method=outcome.method.value,
                        reference_text=target.text,
                        generated_text=generated,
                        generated_tokens=outcome.tokens,
                        target_id=target.id,
                        f1=token_f1(generated, target.text),
                        cosine=outcome.cosine,
                        oracle_f1=max(
                            token_f1(text, target.text)
                            for text in outcome.archive_texts
                        ),
                        normalized_exact=re.findall(r"\w+", generated.lower())
                        == re.findall(r"\w+", target.text.lower()),
                        seed=seed,
                        effective_seed=effective_seed,
                        max_tokens=_CAP,
                        eos_token=eos_token,
                        termination=outcome.termination,
                        search_seconds=search_seconds,
                        search_artifact=artifact,
                        cost=cost,
                    )
                )
                await session.release()
                print(
                    f"{len(records)}/{len(targets) * len(plans)} "
                    f"{target.id} {outcome.method.value} "
                    f"f1={records[-1].f1:.3f} cosine={outcome.cosine:.3f} "
                    f"search={search_seconds:.2f}s",
                    flush=True,
                )
        manifest: dict[str, object] = {
            "created_utc": datetime.now(UTC).isoformat(),
            "source_inputs_sha256": sha256(inputs),
            "runtime": runtime,
            "methods": tuple(method.value for method in selected_methods),
            "seed": seed,
            "seed_derivation": "base seed + integer value of first eight SHA256 hex digits of UTF-8 target id",
            "generator": {
                "id": profile.model,
                "revision": profile.revision,
                "snapshot": model_path,
                "options": _backend_options(runtime),
            },
            "encoder": {
                "id": _ENCODER,
                "revision": _ENCODER_REVISION,
                "snapshot": encoder_path,
                "device": encoder.device,
                "dtype": str(encoder.model.dtype),
                "batch_size": encoder.batch_size,
            },
            "preprocessing": {
                "input_text": "unchanged",
                "tokenizer_max_length": 512,
                "truncation": True,
                "padding": True,
                "pooling": "attention-mask mean including special tokens",
                "normalization": "L2 unit vector",
                "generated_decode": {
                    "skip_special_tokens": True,
                    "clean_up_tokenization_spaces": False,
                },
            },
            "search_settings": {
                "generic_prompt": _PROMPT,
                "beam": asdict(_beam_config(eos_token)),
                "clue": asdict(ProbeConfig(8, 0.25)),
                "lookahead": asdict(_lookahead_config(eos_token, seed)),
                "effective_seed_override": True,
                "selection": "maximum target cosine over all terminal archive entries; proposal order breaks ties",
                "cap_semantics": "40 generated content tokens; sampled EOS is omitted from content; reaching cap terminates with probability one",
            },
            "bank": bank_record,
            "shared_target_cost": {
                "encoder_queries": len(targets),
                "encoder_batches": (len(targets) + encoder.batch_size - 1)
                // encoder.batch_size,
                "seconds": target_seconds,
            },
            "initialization_seconds": initialization_seconds,
            "source_hashes": _source_hashes(),
            **_environment(profile),
            "caveats": (
                "Archive maximum-cosine selection is a heuristic output; weighted lookahead particles are separate.",
                "Clue and generic policies use different fixed sentence/passage instructions.",
                "Logical LM requests count algorithm calls, not physical GPU forward passes or FLOPs.",
                "Search counters exclude shared target encoding and bank construction; those costs are reported once.",
                "Search time includes search-result decoding, excludes initialization, target encoding, bank preparation, artifact writes, and cache release.",
                "Models remain loaded while per-method scorer and backend caches are cleared; method order is recorded.",
                "Pinned weights and bank cache are external dependencies; the bundle contains complete rendering and score evidence.",
                "Floating-point and backend-version differences can change fresh inference despite fixed seeds.",
            ),
        }
    finally:
        session.close()
    manifest["wall_seconds"] = time.perf_counter() - began_wall
    manifest["wall_time_scope"] = (
        "Input loading through inference, artifact serialization, and generator cleanup; excludes final bundle rendering."
    )
    complete_bundle(output, manifest, tuple(records))
