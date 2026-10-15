import json
from hashlib import sha256

import pytest
from hypothesis import given, strategies as st

from examples.embedding_inversion.bundle import (
    Target,
    load_targets,
    prepare_output,
    render_bundle,
    token_f1,
    write_json,
)


@given(st.lists(st.sampled_from(("one", "two", "three")), min_size=1))
def test_f1_counts_repeated_words(words):
    reference = " ".join(words)
    assert token_f1(reference, reference) == 1
    assert token_f1(reference, reference + " unrelated") == pytest.approx(
        2 * len(words) / (2 * len(words) + 1)
    )


def test_inputs_preserve_text_and_reject_ambiguous_ids(tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(json.dumps({"id": "a", "text": "  Two\nlines.  "}) + "\n")
    assert load_targets(inputs) == (Target("a", "  Two\nlines.  "),)
    inputs.write_text(inputs.read_text() * 2)
    with pytest.raises(ValueError, match="unique"):
        load_targets(inputs)


def test_offline_render_escapes_text_and_detects_tampering(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    record = {
        "id": "a|b",
        "method": "clue",
        "reference_text": "<script>alert(1)</script> | first\nsecond",
        "generated_text": "a & b",
        "f1": 0.25,
        "cosine": 0.75,
        "termination": "cap",
        "search_seconds": 1.5,
    }
    results = bundle / "results.jsonl"
    results.write_text(json.dumps(record) + "\n")
    write_json(
        bundle / "manifest.json",
        {
            "schema_version": 1,
            "state": "complete",
            "task": "target_conditioned_embedding_inversion",
            "artifacts_sha256": {
                "results.jsonl": sha256(results.read_bytes()).hexdigest()
            },
        },
    )
    output = tmp_path / "render"
    render_bundle(bundle, output)
    html = (output / "table.html").read_text()
    markdown = (output / "table.md").read_text()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "a &amp; b" in html
    assert "a\\|b" in markdown and "first<br>second" in markdown
    assert "25.00" in markdown and "75.00" in markdown
    assert "saved" in html.lower()
    results.write_text(results.read_text().replace("0.25", "0.99"))
    with pytest.raises(ValueError, match="hash"):
        render_bundle(bundle, tmp_path / "tampered")
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["artifacts_sha256"] = {}
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="results.jsonl"):
        render_bundle(bundle, tmp_path / "unverified")


def test_incomplete_bundle_cannot_be_presented_as_results(tmp_path):
    write_json(tmp_path / "manifest.json", {"state": "running"})
    with pytest.raises(ValueError, match="complete"):
        render_bundle(tmp_path, tmp_path / "render")


def test_fresh_run_preserves_existing_results(tmp_path):
    existing = tmp_path / "results.jsonl"
    existing.write_text("existing data\n")
    with pytest.raises(ValueError, match="empty"):
        prepare_output(tmp_path)
    assert existing.read_text() == "existing data\n"
