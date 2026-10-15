"""Portable result bundles and model-free qualitative views."""

import hashlib
import html
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Target:
    """An unchanged reference text whose embedding is the inversion target."""

    id: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("target id must be a nonempty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("target text must be a nonempty string")


@dataclass(frozen=True)
class ResultRecord:
    """Selected inversion output; the complete search resides in its artifact."""

    id: str
    method: str
    reference_text: str
    generated_text: str
    generated_tokens: tuple[int, ...]
    target_id: str
    f1: float
    cosine: float
    oracle_f1: float
    normalized_exact: bool
    seed: int
    effective_seed: int
    max_tokens: int
    eos_token: int
    termination: str
    search_seconds: float
    search_artifact: str
    cost: dict[str, int]


def load_targets(path: Path) -> tuple[Target, ...]:
    """Read unique nonempty target identities while preserving reference text."""
    targets = tuple(
        Target(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not targets or len({target.id for target in targets}) != len(targets):
        raise ValueError("inputs must contain nonempty, unique target ids")
    return targets


def token_f1(candidate: str, reference: str) -> float:
    """Lowercase Unicode-word multiset F1, matching the frozen experiment."""
    candidate_words = re.findall(r"\w+", candidate.lower())
    reference_words = re.findall(r"\w+", reference.lower())
    denominator = len(candidate_words) + len(reference_words)
    overlap = sum((Counter(candidate_words) & Counter(reference_words)).values())
    return 2 * overlap / denominator if denominator else 1.0


def sha256(path: Path) -> str:
    """Hash artifact bytes independently of filesystem metadata."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically write strict UTF-8 JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, values: Iterable[object]) -> None:
    """Write strict UTF-8 JSON records with one record per line."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_output(output: Path) -> None:
    """Reserve an empty directory for fresh inference."""
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("fresh inference requires an empty output directory")
    (output / "search").mkdir()
    write_json(output / "manifest.json", {"schema_version": 1, "state": "running"})


def complete_bundle(
    output: Path, manifest: dict[str, object], results: tuple[ResultRecord, ...]
) -> None:
    """Seal result artifacts with content hashes and build qualitative views."""
    write_jsonl(output / "results.jsonl", (asdict(result) for result in results))
    artifacts = (
        output / "inputs.jsonl",
        output / "targets.jsonl",
        output / "results.jsonl",
        *(output / "search").glob("*.json.gz"),
    )
    write_json(
        output / "manifest.json",
        {
            **manifest,
            "schema_version": 1,
            "state": "complete",
            "task": "target_conditioned_embedding_inversion",
            "artifacts_sha256": {
                str(path.relative_to(output)): sha256(path) for path in artifacts
            },
        },
    )
    render_bundle(output, output)


def _markdown_cell(value: str) -> str:
    return html.escape(value).replace("|", "\\|").replace("\n", "<br>")


def render_bundle(bundle: Path, output: Path) -> None:
    """Render saved, hash-verified results without loading inference dependencies."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "complete":
        raise ValueError("render requires a complete bundle")
    if "results.jsonl" not in manifest["artifacts_sha256"]:
        raise ValueError("artifact hash required for results.jsonl")
    for name, expected in manifest["artifacts_sha256"].items():
        artifact = bundle / name
        if not artifact.resolve().is_relative_to(bundle.resolve()):
            raise ValueError("artifact hash path lies outside bundle")
        if sha256(artifact) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    records = tuple(
        json.loads(line)
        for line in (bundle / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    headers = (
        "Target",
        "Method",
        "Reference",
        "Generated",
        "F1 ×100",
        "Cosine ×100",
        "Stop",
        "Search s",
    )
    rows = tuple(
        (
            row["id"],
            row["method"],
            row["reference_text"],
            row["generated_text"],
            f"{100 * row['f1']:.2f}",
            f"{100 * row['cosine']:.2f}",
            row["termination"],
            f"{row['search_seconds']:.2f}",
        )
        for row in records
    )
    note = (
        "Target-conditioned embedding inversion. This table renders saved inference "
        "results. Selection uses target cosine; reference F1 is evaluation only. "
        "Search time excludes shared model loading and target encoding. "
        "Lookahead rows select the best-cosine archive entry; weighted particles "
        "are retained separately in the search artifacts."
    )
    markdown = (
        "# Embedding inversion\n\n"
        + note
        + "\n\n"
        + "| "
        + " | ".join(headers)
        + " |\n"
        + "| "
        + " | ".join("---" for _ in headers)
        + " |\n"
        + "".join("| " + " | ".join(map(_markdown_cell, row)) + " |\n" for row in rows)
    )
    header = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Embedding inversion</title><style>"
        "body{font:16px system-ui;margin:2rem;color:#17212b;background:#fafafa}"
        "p{max-width:80ch;line-height:1.5}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #cad0d7;padding:.7rem;text-align:left;vertical-align:top}"
        "th{background:#e9eff4}td{white-space:pre-wrap;overflow-wrap:anywhere}"
        "tr:nth-child(even){background:#f0f4f7}</style><h1>Embedding inversion</h1>"
        f"<p>{html.escape(note)}</p><table><thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody></table></html>\n"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "table.md").write_text(markdown, encoding="utf-8")
    (output / "table.html").write_text(document, encoding="utf-8")
