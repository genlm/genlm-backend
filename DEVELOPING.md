# Developer's Guide

This guide describes how to complete various tasks you'll encounter when working
on the `backend` codebase.

## Local Installation

Clone the repository:
```bash
git clone git@github.com:genlm/genlm-backend.git
cd genlm-backend
```

Create a new environment. For example, with `uv`:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

> Note: You may need to install `uv` via `curl -LsSf https://astral.sh/uv/install.sh | sh`. See also [the installation methods for uv](https://docs.astral.sh/uv/getting-started/installation/).

Then, install the package with pip:

```bash
uv pip install -e ".[docs]"
uv pip install -r requirements-dev.txt
```

To build with MLX support, run:
```bash
uv pip install -e ".[mlx]"
```

## Testing

When test dependencies are installed, the test suite can be run via:

```bash
pytest tests
```

To run the test suite with coverage, run:

```bash
pytest tests --cov=genlm/backend --cov-report=term-missing
```

## GPU tests & CI

Some tests require a CUDA GPU (e.g. the vLLM and SGLang backends). They are **not** exercised by a plain `pytest tests` on a CPU-only machine.

**Running GPU tests locally.** On a machine with a CUDA GPU, install the relevant extra and run the GPU-backed suite (this mirrors CI in `.github/workflows/coverage.yml`):

```bash
# vLLM backend
uv pip install -e ".[vllm]"
pytest tests --ignore=tests/test_mlx.py --ignore=tests/test_sgl.py

# SGLang backend
uv pip install -e ".[sgl]"
pytest tests/test_sgl.py
```

**How GPU tests run in CI.** The GPU jobs (`test_vllm_coverage`, `test_sgl_coverage` in `.github/workflows/coverage.yml`) run on the self-hosted **`gpu-runners`** runner group. On every pull request and push to `main` they run **by default**.

- **Opt out with the `skip-gpu-tests` label.** A maintainer (triage/write access) can add the `skip-gpu-tests` label to a PR to skip the GPU jobs — e.g. for a docs-only change, or when you've already run the GPU suite locally (on a cloud GPU). External/fork contributors can't add labels, so GPU tests always run for their PRs.
- **`gpu-gate` is the required check.** It's a small aggregator job that passes when the GPU jobs succeed *or* are intentionally skipped, and fails only if they fail or are cancelled. Branch protection requires `gpu-gate` (not the individual GPU jobs), so a labeled-skip or a queue-stuck run never blocks a merge with a dangling `cancelled` check.
- **Queue watchdog.** A scheduled `gpu-queue-watchdog` workflow cancels GPU runs stuck in `queued` for more than 20 minutes (e.g. if no runner is available), so a PR fails fast instead of hanging at GitHub's 24-hour queue limit. If that happens: re-run the job once a runner is free, or apply `skip-gpu-tests` to proceed.

**Recommended maintainer flow:** make your change → if it touches GPU paths and you have a GPU, run the GPU tests locally → open the PR → add `skip-gpu-tests` if you've validated locally or the change is GPU-irrelevant, otherwise let CI run them on `gpu-runners`.

## Public API changes

`griffe check` runs per PR, diffing the public surface vs `main`. It will report breaking API changes as a warning + sticky comments. However, it only catches signature-level breaks, not behavioral changes under an unchanged signature (e.g., a deprecation tombstone that accepts and raises).

To surface those, put the change in the signature: annotate a removed method `-> NoReturn`; change the return annotation or a default when the contract changes.

## Documentation

Documentation is generated using [mkdocs](https://www.mkdocs.org/) and hosted on GitHub Pages. To build the documentation, run:

```bash
mkdocs build
```

To serve the documentation locally, run:

```bash
mkdocs serve
```

## Performance Benchmarking

Performance benchmarks comparing different configurations can be found in our [benchmarks directory](https://github.com/probcomp/genlm-backend/tree/main/benchmark).


## Commit Hooks

We use [pre-commit](https://pre-commit.com/) to manage a series of git
pre-commit hooks for the project; for example, each time you commit code, the
hooks will make sure that your python is formatted properly. If your code isn't,
the hook will format it, so when you try to commit the second time you'll get
past the hook.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks,
install `pre-commit` if you don't yet have it. I prefer using
[pipx](https://github.com/pipxproject/pipx) so that `pre-commit` stays globally
available.

```bash
pipx install pre-commit
```

Then install the hooks with this command:

```bash
pre-commit install
```

Now they'll run on every commit. If you want to run them manually, run the
following command:

```bash
pre-commit run --all-files
```
