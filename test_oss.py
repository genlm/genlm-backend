from genlm.backend import load_model_by_name

llm_opts = {
    "hf_opts": {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True  # Add this line
    }
}

model = load_model_by_name("openai/gpt-oss-20b", backend="hf", llm_opts=llm_opts)

print("Model loaded successfully!")
print(model)

# from genlm.backend import load_model_by_name
# import torch

# llm_opts = {
#     "engine_opts": {
#         "trust_remote_code": True,

#     }
# }


# model = load_model_by_name(
#     "openai/gpt-oss-20b",
#     backend="vllm",
#     llm_opts=llm_opts
# )

# print("Model loaded successfully with vLLM backend!")
# print(model)