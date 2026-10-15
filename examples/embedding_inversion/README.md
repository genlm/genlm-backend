# Frozen embedding inversion

This example reconstructs text from a frozen Contriever embedding using a frozen
Qwen model. It supports the vocabulary-clue cosine beam, a generic cosine beam,
and completion-lookahead sequential Monte Carlo. Reference text supplies the
target embedding and evaluation metrics; generation and final selection receive
the embedding and fixed search settings. This is the embedding-reversal component
of the Lace experiments, with no clustering or unconditional-generation stage.

## Prerequisites

Run from a clone of this repository with Python **3.12 or later**. The research
kernels require Python 3.12; this does not change the backend package requirement.
Initial model downloads require network access and several GB of cache space.
Every profile pins the generator and encoder revisions in `runtime.py` and records
the resolved identities, package versions, hardware, and source hashes.

Apple silicon, using MLX Qwen2.5-1.5B-Instruct-4bit and CPU Contriever:

```bash
uv venv .venv-inversion --python 3.12
uv pip install --python .venv-inversion/bin/python -e '.[mlx]' \
  -r examples/embedding_inversion/requirements-mlx.txt
# Fresh inference: four public targets, three methods; allow several minutes
# after model downloads. The committed smoke uses the CUDA profile below.
.venv-inversion/bin/python -m examples.embedding_inversion run \
  --runtime mlx --output output/inversion-mlx
```

Linux with a CUDA GPU, using unquantized Qwen2.5-3B-Instruct and CUDA Contriever:

```bash
uv venv .venv-inversion-cuda --python 3.12
uv pip install --python .venv-inversion-cuda/bin/python -e '.[vllm]' \
  -r examples/embedding_inversion/requirements-cuda.txt
# Four targets × three methods: measured 6.9 minutes on Spark's NVIDIA GB10,
# including first-time compilation and bank construction, excluding model downloads.
.venv-inversion-cuda/bin/python -m examples.embedding_inversion run \
  --runtime cuda --output output/inversion-cuda
```

The CUDA profile allocates 80% of device memory to vLLM and also loads Contriever.
Use an otherwise idle GPU. The committed smoke ran on Spark's NVIDIA GB10 with
128 GB unified memory; earlier research runs used a 32 GB GPU.
MLX and CUDA profiles use different model sizes and quantization; compare methods
within a profile before attributing differences to search.

## Fresh inputs and saved results

`--inputs` accepts UTF-8 JSONL with unique `id` values and nonempty `text` strings.
Texts are preserved exactly. Contriever tokenizes with truncation at 512 tokens;
the manifest specifies attention-mask mean pooling and unit normalization.

```bash
printf '%s\n' '{"id":"example","text":"The museum closes at six on weekdays."}' > targets.jsonl
# One clue-beam target: roughly 10–20 search seconds after initialization.
.venv-inversion/bin/python -m examples.embedding_inversion run \
  --inputs targets.jsonl --methods clue --seed 20261013 \
  --output output/inversion-custom
```

`run` requires an empty output directory and performs fresh inference. The models
load once for the invocation, target embeddings are shared between methods, and
each method gets a fresh scoring cache. `--methods` accepts `clue`, `beam`, and
`lookahead`; the default runs all three. The fixed 12,000-word vocabulary bank is
built without experiment texts and cached at
`~/.cache/genlm-embedding-inversion/contriever.pt`; `--bank-cache` selects another
cache file. Its construction cost and content hash are recorded separately.

The committed `smoke/` bundle contains actual CUDA Qwen2.5-3B results on the public
four-target panel. Its 413-second command runtime included 147 seconds of model
initialization, 1.6 seconds of bank construction, and 251 seconds of search.
Rendering it performs no inference and needs only the Python standard library:

```bash
# Saved-result rendering: under one second for the bundled smoke panel.
python3 -m examples.embedding_inversion render \
  --bundle examples/embedding_inversion/smoke --output output/inversion-view
```

Open `output/inversion-view/table.html` or `table.md`. Rendering verifies recorded
artifact hashes and rejects incomplete or modified bundles. The same command
renders any completed fresh run by changing `--bundle`.

## Bundle contract

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Model and encoder revisions, preprocessing, search settings, environment/source provenance, shared costs, wall time, artifact hashes, completion state. |
| `inputs.jsonl` | The exact input `id` and `text` records. |
| `targets.jsonl` | `id`, `text`, and `vector` (the unit-normalized target embedding). |
| `results.jsonl` | One selected result per target/method, as defined by `bundle.ResultRecord`: reference/generated text, generated tokens, F1/cosine/oracle F1, seeds, cap/EOS, termination, search time, query costs, and search-artifact path. `target_id` joins to `targets.jsonl.id`. |
| `search/*.json.gz` | Full terminal candidates and decoded texts, search scores/traces, exact conditioning prompt/tokens; lookahead weighted particles and diagnostics remain separate from its archive. Clue records include selected clues and bank provenance. |
| `table.md`, `table.html` | Portable qualitative views regenerated from saved records. |

Model weights and the derived vocabulary bank are external, revision-pinned
caches. The result bundle contains the data needed to inspect and render the
experiment without those caches. Fresh reruns rebuild a missing bank; floating
point differences across hardware/package versions can change close search ties.

## Experimental scope

Both beams use width 8/top 16, a 40-content-token cap, cosine ranking, diversity
penalty 0.2, and EOS when it enters the proposal top-K. The clue variant selects
eight vocabulary clues with redundancy penalty 0.25. Its factual-sentence prompt
differs from the generic beam's factual-passage prompt, so their comparison
measures complete prompt policies. The beam seeds are provenance fields; these
searches are deterministic for fixed model outputs.

Completion lookahead uses eight particles, two full suffix rollouts, refresh
steps 1/9/17/25/33, terminal inverse temperature 128, and ESS threshold 0.5.
Its effective seed is the base seed plus the first eight hexadecimal digits of
SHA-256 of the target ID. The reported text is the maximum-cosine archive entry;
the weighted terminal particle population is a separate sampling output.

F1 uses lowercase Unicode-word multiset overlap. The selected result maximizes
target cosine; oracle F1 evaluates the terminal candidate archive after search.
Neither score is a measure of unconditional generation quality. Earlier
development runs found no supported F1 advantage for completion steering at these
settings. Four targets validate the runnable interface and provide qualitative
examples; they do not establish a model or algorithm ranking. The trained
ZSInvert correction stage is absent because a matching released checkpoint was
unavailable for the original experiment.

`data/public4.jsonl` reuses the frozen development canary: exclude the eight
previously tested targets, partition 24 remaining targets into quartiles by
historical 1.5B clue-beam F1, then sample one per quartile with seed 20261014.
`data/public4_selection.json` records the ranked pool and chosen IDs;
`data/public4_provenance.json` records source coordinates and original target token
IDs. The source is `microsoft/ms_marco`, configuration `v2.1`, validation revision
`a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a`; targets are the first 32 content tokens
decoded by the pinned uncased Contriever tokenizer. This approximates the paper's
32-token MS MARCO protocol; its exact evaluation subset and F1 normalization were
unspecified.

## Validation

```bash
uv pip install --python .venv-inversion/bin/python pytest hypothesis
.venv-inversion/bin/python -m pytest --confcutdir=tests/embedding_inversion \
  tests/embedding_inversion
```

`_algorithms/source_provenance.json` records the research kernel hashes and transplant
transformations. Kernel tests exercise search semantics with finite model spaces;
bundle tests cover metric behavior, input identities, preservation of existing
results, escaped views, and artifact integrity. The committed smoke supplies the
real model execution check. `smoke/validation.json` records independent artifact,
token-decoding, and CPU encoder checks against that run.
