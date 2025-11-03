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
