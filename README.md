![Logo](logo.png)


<div align="center">

[![Docs](https://github.com/genlm/genlm-backend/actions/workflows/docs.yml/badge.svg)](https://genlm.github.io/backend/)
[![Tests](https://github.com/genlm/genlm-backend/actions/workflows/pytest.yml/badge.svg)](https://github.com/genlm/backend/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/github/genlm/genlm-backend/graph/badge.svg?token=PwmHwMJC2y)](https://codecov.io/github/genlm/genlm-backend)
[![PyPI](https://img.shields.io/pypi/v/genlm-backend?label=pypi)](https://pypi.org/project/genlm-backend/)

</div>

GenLM Backend is a high-performance backend for language model probabilistic programs, built for the GenLM ecosystem. It provides an **asynchronous** and **autobatched** interface to `vllm` and `transformers` language models, enabling scalable and efficient inference.

See our [documentation](https://genlm.github.io/genlm-backend/).

## 🚀 Key Features
- Automatic batching of concurrent log-probability requests, enabling efficient large-scale inference without having to write batching logic yourself
- Byte-level decoding of transformers tokenizers, enabling advanced token-level control
- Support for arbitrary Hugging Face models (e.g., LLaMA, DeepSeek, etc.) with fast inference and automatic KV caching using vllm
- NEW: support for MLX-LM library, allowing faster inference on Apple silicon devices.


## ⚡ Quick Start

This library supports installation via pip:

```bash
pip install genlm-backend
```

Or to install with MLX support, run:

```bash
pip install genlm-backend[mlx]
```


## 🧪 Example: Autobatched Sequential Importance Sampling with LLMs

This example demonstrates how `genlm-backend` enables concise, scalable probabilistic inference with language models. It implements a Sequential Importance Sampling (SIS) algorithm that makes asynchronous log-probabality requests which get automatically batched by the language model.


```python
import torch
import asyncio
from genlm.backend import load_model_by_name

# --- Token-level masking using the byte-level vocabulary --- #
def make_masking_function(llm, max_token_length, max_tokens):
    eos_id = llm.tokenizer.eos_token_id
    valid_ids = torch.tensor([
        token_id == eos_id or len(token) <= max_token_length
        for token_id, token in enumerate(llm.byte_vocab)
    ], dtype=torch.float).log()
    eos_one_hot = torch.nn.functional.one_hot(
        torch.tensor(eos_id), len(llm.byte_vocab)
    ).log()

    def masking_function(context):
        return eos_one_hot if len(context) >= max_tokens else valid_ids

    return masking_function

# --- Particle class for SIS --- #
class Particle:
    def __init__(self, llm, mask_function, prompt_ids):
        self.context = []
        self.prompt_ids = prompt_ids
        self.log_weight = 0.0
        self.active = True
        self.llm = llm
        self.mask_function = mask_function

    async def extend(self):
        logps = await self.llm.next_token_logprobs(self.prompt_ids + self.context)
        masked_logps = logps + self.mask_function(self.context).to(logps.device)
        logZ = masked_logps.logsumexp(dim=-1)
        self.log_weight += logZ
        next_token_id = torch.multinomial((masked_logps - logZ).exp(), 1).item()
        if next_token_id == self.llm.tokenizer.eos_token_id:
            self.active = False
        else:
            self.context.append(next_token_id)

# --- Autobatched SIS loop --- #
async def autobatched_sis(n_particles, llm, masking_function, prompt_ids):
    particles = [Particle(llm, masking_function, prompt_ids) for _ in range(n_particles)]
    while any(p.active for p in particles):
        await asyncio.gather(*[p.extend() for p in particles if p.active])
    return particles

# --- Run the example --- #
llm = load_model_by_name("gpt2") # or e.g., "meta-llama/Llama-3.2-1B" if you have access
mask_function = make_masking_function(llm, max_token_length=10, max_tokens=10)
prompt_ids = llm.tokenizer.encode("Montreal is")
particles = await autobatched_sis( # use asyncio.run(autobatched_sis(...)) if you are not in an async context
    n_particles=10, llm=llm, masking_function=mask_function, prompt_ids=prompt_ids
)

strings = [llm.tokenizer.decode(p.context) for p in particles]
log_weights = torch.tensor([p.log_weight for p in particles])
probs = torch.exp(log_weights - log_weights.logsumexp(dim=-1))

for s, p in sorted(zip(strings, probs), key=lambda x: -x[1]):
    print(f"{repr(s)} (probability: {p:.4f})")

```

This example highlights the following features:

* 🌀 **Asynchronous Inference Loop.** Each particle runs independently, but all LLM calls are scheduled concurrently via `asyncio.gather`. The backend batches them automatically, so we get the efficiency of large batched inference without having to write the batching logic.
* 🔁 **Byte-level Tokenization Support.** Token filtering is done using the model’s byte-level vocabulary, which `genlm-backend` exposes. This enables low-level control over generation in ways not possible with most high-level APIs.


## Development

See the [DEVELOPING.md](DEVELOPING.md) file for information on how to install the project for local development.
