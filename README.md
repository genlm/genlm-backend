[![Docs](https://github.com/probcomp/genlm-backend/actions/workflows/docs.yml/badge.svg)](https://probcomp.github.io/genlm-backend/)
[![Tests](https://github.com/probcomp/genlm-backend/actions/workflows/pytest.yml/badge.svg)](https://github.com/probcomp/genlm-backend/actions/workflows/pytest.yml)

# GenLM Backend

GenLM Backend is a high-performance backend for language model probabilistic programs in the GenLM ecosystem. It provides essential tools and functions that serve as building blocks for more complex applications. See our [documentation](https://probcomp.github.io/genlm-backend/).

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
pip install -e .[test,docs]
```
which also installs the dependencies needed for testing (test) and documentation (docs).

## Requirements

- Python >= 3.10
- The core dependencies listed in the `setup.py` file of the repository.

## Testing

When test dependencies are installed, the test suite can be run via:
```bash
pytest tests
```

## Performance Benchmarking

Performance benchmarks comparing different configurations can be found in our [benchmarks directory](https://github.com/probcomp/genlm-backend/tree/main/benchmark).
