from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from genlm.backend import load_model_by_name
model_id = "openai/gpt-oss-20b"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load with hf which handles MXFP4 kernels under the hood
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,   # <-- upcast target
    trust_remote_code=True,
    device_map="auto",
)

# Save a BF16 copy to disk (can also push to a HF repo)
out_dir = "./gpt-oss-20b-bf16"
model.save_pretrained(out_dir, safe_serialization=True)
tok.save_pretrained(out_dir)

llm_opts = {
    "engine_opts": {
        "trust_remote_code": True,
        "dtype": "bfloat16",
        # don't set any quantization method
    }
}

model = load_model_by_name(
    "./gpt-oss-20b-bf16",
    backend="vllm",
    llm_opts=llm_opts
)

print(model)

#cant dequantize mxfp4 on load with vllm, so either support loading unquantized which we'll need triton 3.4 (and upgrade torch)
# or use hf backend or put it on hf and load it or locally and load it w vllm by loading the checkpoint

#conclusion is if gptoss default to HF (until we upgrade triton and torch) and then can load it with vllm if
#its an h100/sm90 gpu

