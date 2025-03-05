[![Docs](https://github.com/chisym/genlm-backend/actions/workflows/docs.yml/badge.svg)](https://probcomp.github.io/genlm-backend/)
[![Tests](https://github.com/chisym/genlm-backend/actions/workflows/pytest.yml/badge.svg)](https://github.com/probcomp/genlm-backend/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/github/chisym/genlm-backend/graph/badge.svg?token=AS70lcuXra)](https://codecov.io/github/chisym/genlm-backend)

# GenLM Backend

GenLM Backend is a high-performance backend for language model probabilistic programs in the GenLM ecosystem. It provides essential tools and functions that serve as building blocks for more complex applications. See our [documentation](https://chisym.github.io/genlm-backend/).

**Key Features**:

* **Asynchronous LLM Interfaces**: Asynchronous computation of next-token probabilities with `vllm` and `transformer` language models.
* **Tokenizer Vocabulary Decoding**: Decode Hugging Face tokenizer vocabularies into their byte and string representations.
* **Token-Character Tries**: Efficient conversion from token distributions to byte-level distributions using a trie datastructure.

## Quick Start

### Installation

Clone the repository:
```bash
git clone git@github.com:probcomp/genlm-backend.git
cd genlm_backend
```
and install with pip:

```bash
pip install .
```

This installs the package without development dependencies. For development, install in editable mode with:

```bash
pip install -e ".[docs]"
pip install -r requirements-dev.txt
```

which also installs the dependencies needed for testing (test) and documentation (docs).

## Requirements

- Python >= 3.10
- The core dependencies listed in the `setup.py` file of the repository.

> **Note**
> vLLM is not supported on macOS. On macOS systems, only CPU-based functionality (`AsyncTransformer`) will be available. GPU-accelerated features requiring vLLM (`AsyncVirtualLM`) will not work.

## Testing

When test dependencies are installed, the test suite can be run via:

```bash
pytest tests
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
