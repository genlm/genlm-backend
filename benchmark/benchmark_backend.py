"""
Evaluates the performance differences between AsyncLLM (vLLM-based) and AsyncTransformer 
(HuggingFace-based) implementations. Both implementations support asynchronous 
batching and caching optimizations. 

The AsyncTransformer batch_size parameter is set to the size of the input batch, meaning
that the timeout never fires.
"""

import asyncio
import argparse
import pandas as pd
from tqdm import tqdm
from arsenal.timer import Timer
from async_llm.llm import AsyncLLM
from async_llm.hf_llm import AsyncTransformer
from util import (
    get_wikitext, 
    token_prefixes, 
    prefix_batches,
    plot_features,
    save_comparison_df
)

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark AsyncLLM vs AsyncTransformer')
    parser.add_argument(
        '--model', 
        default='gpt2',
        help='Model name to benchmark (default: gpt2)'
    )
    parser.add_argument(
        '--text-length', 
        type=int, 
        default=5000,
        help='Length of text to process from wikitext (default: 5000)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Batch size for batched inference (default: 20)'
    )
    parser.add_argument(
        '--name',
        default='benchmark_backend',
        help='Name prefix for saving benchmark results (default: benchmark_backend)'
    )
    parser.add_argument(
        '--out-dir',
        default='results',
        help='Directory to save benchmark results (default: results)'
    )
    return parser.parse_args()

def load_models(model, batch_size):
    hf_model = AsyncTransformer.from_name(model, batch_size=batch_size)
    vllm_model = AsyncLLM.from_name(model, engine_opts={'gpu_memory_utilization' : 0.45})
    return vllm_model, hf_model

async def benchmark_single(our_dir, name, model, text):
    (vllm_model, hf_model) = load_models(model, batch_size=1)
    
    prefixes = token_prefixes(text, vllm_model.tokenizer)

    vllm_timer = Timer('AsyncLLM (single request)')
    hf_timer = Timer('AsyncTransformer (single request)')

    for token_ids in tqdm(prefixes):
        seq_len = len(token_ids)
        with vllm_timer(seq_len=seq_len):
            await vllm_model.next_token_logprobs(token_ids)
        with hf_timer(seq_len=seq_len):
            await hf_model.next_token_logprobs(token_ids)

    vllm_timer.compare(hf_timer)
    name = f"{name}_{model}_single"
    save_comparison_df(our_dir, name, vllm_timer, hf_timer)
    plot_features(our_dir, name, vllm_timer, hf_timer)

async def benchmark_batch(our_dir, name, model, text, batch_size):
    (vllm_model, hf_model) = load_models(model, batch_size)

    batches = prefix_batches(text, vllm_model.tokenizer, batch_size=batch_size)

    vllm_timer = Timer('AsyncLLM (batch)')
    hf_timer = Timer('AsyncTransformer (batch)')

    for token_ids in tqdm(batches):
        max_seq_len = max([len(x) for x in token_ids])
        with vllm_timer(max_seq_len=max_seq_len):
            await vllm_model.batch_next_token_logprobs(token_ids)
        with hf_timer(max_seq_len=max_seq_len):
            await hf_model.batch_next_token_logprobs(token_ids)

    vllm_timer.compare(hf_timer)
    name = f"{name}_{model}_batch{batch_size}"
    save_comparison_df(our_dir, name, vllm_timer, hf_timer)
    plot_features(our_dir, name, vllm_timer, hf_timer)

async def main():
    args = parse_args()
    text = get_wikitext()[:args.text_length]
    name = f"{args.name}_{args.text_length}"
    await benchmark_single(args.out_dir, name, args.model, text)
    await benchmark_batch(args.out_dir, name, args.model, text, args.batch_size)

if __name__ == "__main__":
    asyncio.run(main())