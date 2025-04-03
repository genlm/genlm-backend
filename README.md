[![Docs](https://github.com/genlm/backend/actions/workflows/docs.yml/badge.svg)](https://genlm.github.io/backend/)
[![Tests](https://github.com/genlm/backend/actions/workflows/pytest.yml/badge.svg)](https://github.com/genlm/backend/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/github/genlm/backend/graph/badge.svg?token=AS70lcuXra)](https://codecov.io/github/genlm/backend)

# GenLM Backend

GenLM Backend is a high-performance backend for language model probabilistic programs in the GenLM ecosystem. It provides essential tools and functions that serve as building blocks for more complex applications. See our [documentation](https://genlm.github.io/backend/).

**Key Features**:

* **LM Interfaces**: Asynchronous and synchronous interfaces to `vllm` and `transformer` language models.
* **Tokenizer Vocabulary Decoding**: Decode Hugging Face tokenizer vocabularies into their byte and string representations.
* **Token-Byte Tries**: Efficient conversion from token distributions to byte-level distributions using a trie datastructure.

## Quick Start

This library supports installation via pip:

```bash
pip install genlm.backend
```

## Development

See the [DEVELOPING.md](DEVELOPING.md) file for information on how to install the project for local development.
