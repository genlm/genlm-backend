import json

import pytest

pytest.importorskip("mlx")

import mlx.core as mx  # noqa: E402
import mlx_lm  # noqa: E402
import torch  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: E402

from genlm.backend.llm import AsyncMlxLM  # noqa: E402


MODEL_NAME = "openai-community/gpt2"
NUM_LAYERS = 2
LORA_PARAMETERS = {"rank": 4, "scale": 20.0, "dropout": 0.0}


def _write_adapter(directory, seed):
    """An adapter with the layout `MODEL_NAME` wraps to, and a non-zero `lora_b` so it
    actually moves the logits."""
    mx.random.seed(seed)
    model, _ = mlx_lm.load(MODEL_NAME)
    linear_to_lora_layers(model, NUM_LAYERS, LORA_PARAMETERS)
    weights = {
        k: (mx.random.normal(v.shape) * 0.02).astype(v.dtype)
        for k, v in tree_flatten(model.trainable_parameters())
        if "lora_" in k
    }
    mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": NUM_LAYERS,
                "lora_parameters": LORA_PARAMETERS,
            }
        )
    )
    return str(directory)


@pytest.fixture(scope="module")
def adapters(tmp_path_factory):
    root = tmp_path_factory.mktemp("adapters")
    built = {}
    for name, seed in (("a", 1), ("b", 2)):
        directory = root / name
        directory.mkdir()
        built[name] = _write_adapter(directory, seed)
    return built


@pytest.fixture
def llm():
    return AsyncMlxLM.from_name(MODEL_NAME)


@pytest.fixture
def token_ids(llm):
    return llm.tokenizer.encode("The capital of France is")


def test_base_is_unchanged_by_registering(llm, token_ids, adapters):
    # Wrapping starts from a zero adapter, so the base must stay bit-exact.
    before = llm.next_token_logprobs_sync(token_ids).cpu().clone()
    llm.add_new_lora(adapters["a"], "a")
    after = llm.next_token_logprobs_sync(token_ids).cpu()

    assert torch.equal(before, after)


def test_adapter_changes_the_distribution(llm, token_ids, adapters):
    llm.add_new_lora(adapters["a"], "a")

    base = llm.next_token_logprobs_sync(token_ids).cpu()
    adapted = llm.next_token_logprobs_sync(token_ids, lora_name="a").cpu()

    assert not torch.equal(base, adapted)


def test_adapters_are_independent(llm, token_ids, adapters):
    llm.add_new_lora(adapters["a"], "a")
    llm.add_new_lora(adapters["b"], "b")

    a = llm.next_token_logprobs_sync(token_ids, lora_name="a").cpu()
    b = llm.next_token_logprobs_sync(token_ids, lora_name="b").cpu()

    assert not torch.equal(a, b)


def test_kv_rows_do_not_cross_adapters(llm, token_ids, adapters):
    # A lane that walked its rows forward must not serve another lane's forward.
    llm.add_new_lora(adapters["a"], "a")
    extended = token_ids + [100]

    llm.next_token_logprobs_sync(token_ids)  # seed the base lane's rows
    base = llm.next_token_logprobs_sync(extended).cpu()
    adapted = llm.next_token_logprobs_sync(extended, lora_name="a").cpu()

    llm.clear_cache()
    want = llm.next_token_logprobs_sync(extended, lora_name="a").cpu()

    assert torch.equal(adapted, want)
    assert not torch.equal(base, adapted)


def test_rebinding_a_name(llm, token_ids, adapters):
    llm.add_new_lora(adapters["a"], "a")
    first = llm.next_token_logprobs_sync(token_ids, lora_name="a").cpu().clone()
    llm.add_new_lora(adapters["b"], "a")
    second = llm.next_token_logprobs_sync(token_ids, lora_name="a").cpu()

    assert not torch.equal(first, second)


def test_removing_an_adapter(llm, token_ids, adapters):
    llm.add_new_lora(adapters["a"], "a")
    base = llm.next_token_logprobs_sync(token_ids).cpu().clone()
    llm.remove_lora("a")

    assert torch.equal(llm.next_token_logprobs_sync(token_ids).cpu(), base)
    with pytest.raises(ValueError):
        llm.next_token_logprobs_sync(token_ids, lora_name="a")


def test_unknown_adapter(llm, token_ids):
    with pytest.raises(ValueError):
        llm.next_token_logprobs_sync(token_ids, lora_name="nope")
    with pytest.raises(ValueError):
        llm.remove_lora("nope")


def test_incompatible_layout_is_rejected(llm, tmp_path, adapters):
    other = tmp_path / "wide"
    other.mkdir()
    _write_adapter(other, seed=3)
    config = json.loads((other / "adapter_config.json").read_text())
    config["num_layers"] = NUM_LAYERS + 1
    (other / "adapter_config.json").write_text(json.dumps(config))

    llm.add_new_lora(adapters["a"], "a")
    with pytest.raises(ValueError, match="layout"):
        llm.add_new_lora(str(other), "wide")
