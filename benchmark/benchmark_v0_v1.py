#!/usr/bin/env python3
"""
Benchmark script to compare vLLM v0 vs v1 performance for genlm-backend.

Usage:
    # Run v1 benchmark (default)
    python benchmark/benchmark_v0_v1.py

    # Run v0 benchmark
    python benchmark/benchmark_v0_v1.py --v0

    # Compare both
    python benchmark/benchmark_v0_v1.py --compare

    # Custom model
    python benchmark/benchmark_v0_v1.py --model meta-llama/Llama-3.2-1B
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

# Parse args BEFORE importing anything else (to set env vars)
parser = argparse.ArgumentParser(description="Benchmark vLLM v0 vs v1")
parser.add_argument("--v0", action="store_true", help="Use vLLM v0 (default is v1)")
parser.add_argument("--compare", action="store_true", help="Run both v0 and v1 and compare")
parser.add_argument("--model", type=str, default="gpt2", help="Model to benchmark")
parser.add_argument("--gpu-mem", type=float, default=0.3, help="GPU memory utilization")
parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
parser.add_argument("--iterations", type=int, default=30, help="Benchmark iterations")
parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32", help="Batch sizes to test")
parser.add_argument("--output", type=str, help="Output JSON file for results")
args = parser.parse_args()


def setup_environment(use_v0: bool):
    """Set environment variables for v0 or v1."""
    if use_v0:
        os.environ["VLLM_USE_V1"] = "0"
        os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
        # Force v0 by disabling async output processing
        return {"disable_async_output_proc": True}
    else:
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        return {}


@dataclass
class BenchmarkResult:
    version: str
    model: str
    batch_size: int
    iterations: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    per_request_ms: float
    throughput_rps: float
    gpu_memory_mb: float
    timestamp: str


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def run_benchmark(use_v0: bool, model_name: str, gpu_mem: float, 
                  warmup: int, iterations: int, batch_sizes: list[int]) -> list[BenchmarkResult]:
    """Run benchmark for either v0 or v1."""
    import torch
    import numpy as np
    
    version = "v0" if use_v0 else "v1"
    print(f"\n{'='*60}")
    print(f"Running vLLM {version} Benchmark")
    print(f"{'='*60}")
    
    # Setup environment
    extra_opts = setup_environment(use_v0)
    
    # Import after setting environment
    if use_v0:
        # For v0, we need to use the old API
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        
        class V0Wrapper:
            """Wrapper to make v0 API compatible with our benchmark."""
            def __init__(self, model_name, gpu_mem, extra_opts):
                print(f"Loading model with v0 engine...")
                t0 = time.perf_counter()
                self.llm = LLM(
                    model=model_name,
                    gpu_memory_utilization=gpu_mem,
                    enforce_eager=True,
                    **extra_opts
                )
                self.load_time = time.perf_counter() - t0
                print(f"Model loaded in {self.load_time:.2f}s")
                self.tokenizer = self.llm.get_tokenizer()
                self.vocab_size = len(self.tokenizer)
                
                # For v0, we use logits processor
                class CaptureLogits:
                    def __init__(self):
                        self.captured = None
                    def __call__(self, past_token_ids, logits):
                        self.captured = torch.log_softmax(logits, dim=-1)
                        return logits
                self.CaptureLogits = CaptureLogits
                
            def next_token_logprobs_sync(self, token_ids):
                capture = self.CaptureLogits()
                self.llm.generate(
                    prompts=TokensPrompt(prompt_token_ids=list(token_ids)),
                    sampling_params=SamplingParams(
                        max_tokens=1, n=1, detokenize=False,
                        ignore_eos=True, logits_processors=[capture]
                    ),
                    use_tqdm=False,
                )
                return capture.captured
                
            def batch_next_token_logprobs_sync(self, token_ids_list):
                captures = [self.CaptureLogits() for _ in token_ids_list]
                prompts = [TokensPrompt(prompt_token_ids=list(t)) for t in token_ids_list]
                
                # V0 doesn't batch logits processors well, so we do sequential
                results = []
                for prompt, capture in zip(prompts, captures):
                    self.llm.generate(
                        prompts=prompt,
                        sampling_params=SamplingParams(
                            max_tokens=1, n=1, detokenize=False,
                            ignore_eos=True, logits_processors=[capture]
                        ),
                        use_tqdm=False,
                    )
                    results.append(capture.captured)
                return torch.stack(results)
        
        llm = V0Wrapper(model_name, gpu_mem, extra_opts)
    else:
        # V1 uses our optimized implementation
        from genlm.backend.llm import AsyncVirtualLM
        
        print(f"Loading model with v1 engine...")
        t0 = time.perf_counter()
        llm = AsyncVirtualLM.from_name(
            model_name,
            engine_opts={"gpu_memory_utilization": gpu_mem, "enforce_eager": True}
        )
        llm.load_time = time.perf_counter() - t0
        print(f"Model loaded in {llm.load_time:.2f}s")
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        # Create test prompts
        prompts = [llm.tokenizer.encode(f"Test prompt number {i}") for i in range(batch_size)]
        
        # Warmup
        print(f"Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            if batch_size == 1:
                llm.next_token_logprobs_sync(prompts[0])
            else:
                llm.batch_next_token_logprobs_sync(prompts)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"Benchmarking ({iterations} iterations)...")
        times = []
        for i in range(iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if batch_size == 1:
                result = llm.next_token_logprobs_sync(prompts[0])
            else:
                result = llm.batch_next_token_logprobs_sync(prompts)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            
            # Verify result shape
            if batch_size == 1:
                assert result.shape == (llm.vocab_size if hasattr(llm, 'vocab_size') else len(llm.tokenizer),)
            else:
                expected_vocab = llm.vocab_size if hasattr(llm, 'vocab_size') else len(llm.tokenizer)
                assert result.shape == (batch_size, expected_vocab)
        
        times_ms = [t * 1000 for t in times]
        avg_latency = np.mean(times_ms)
        std_latency = np.std(times_ms)
        
        result = BenchmarkResult(
            version=version,
            model=model_name,
            batch_size=batch_size,
            iterations=iterations,
            avg_latency_ms=round(avg_latency, 3),
            min_latency_ms=round(min(times_ms), 3),
            max_latency_ms=round(max(times_ms), 3),
            std_latency_ms=round(std_latency, 3),
            per_request_ms=round(avg_latency / batch_size, 3),
            throughput_rps=round(batch_size / (avg_latency / 1000), 1),
            gpu_memory_mb=round(get_gpu_memory_mb(), 1),
            timestamp=datetime.now().isoformat()
        )
        results.append(result)
        
        print(f"  Avg latency: {result.avg_latency_ms:.2f}ms ± {result.std_latency_ms:.2f}ms")
        print(f"  Per-request: {result.per_request_ms:.2f}ms")
        print(f"  Throughput:  {result.throughput_rps:.1f} req/s")
        print(f"  GPU Memory:  {result.gpu_memory_mb:.1f} MB")
    
    return results


def print_comparison(v0_results: list[BenchmarkResult], v1_results: list[BenchmarkResult]):
    """Print side-by-side comparison of v0 vs v1."""
    print(f"\n{'='*80}")
    print("COMPARISON: vLLM v0 vs v1")
    print(f"{'='*80}")
    
    print(f"\n{'Batch':<8} {'v0 Latency':<14} {'v1 Latency':<14} {'Speedup':<10} {'v0 Throughput':<15} {'v1 Throughput':<15}")
    print("-" * 80)
    
    v0_by_batch = {r.batch_size: r for r in v0_results}
    v1_by_batch = {r.batch_size: r for r in v1_results}
    
    for batch_size in sorted(set(v0_by_batch.keys()) | set(v1_by_batch.keys())):
        v0 = v0_by_batch.get(batch_size)
        v1 = v1_by_batch.get(batch_size)
        
        if v0 and v1:
            speedup = v0.avg_latency_ms / v1.avg_latency_ms
            print(f"{batch_size:<8} {v0.avg_latency_ms:>6.2f}ms      {v1.avg_latency_ms:>6.2f}ms      {speedup:>5.2f}x     {v0.throughput_rps:>8.1f} req/s   {v1.throughput_rps:>8.1f} req/s")
        elif v0:
            print(f"{batch_size:<8} {v0.avg_latency_ms:>6.2f}ms      {'N/A':<12} {'N/A':<10} {v0.throughput_rps:>8.1f} req/s   {'N/A':<15}")
        elif v1:
            print(f"{batch_size:<8} {'N/A':<12} {v1.avg_latency_ms:>6.2f}ms      {'N/A':<10} {'N/A':<15} {v1.throughput_rps:>8.1f} req/s")


def main():
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    all_results = []
    
    if args.compare:
        # Run both v0 and v1
        print("Running comparison benchmark...")
        
        # Run v0 first
        v0_results = run_benchmark(
            use_v0=True,
            model_name=args.model,
            gpu_mem=args.gpu_mem,
            warmup=args.warmup,
            iterations=args.iterations,
            batch_sizes=batch_sizes
        )
        all_results.extend(v0_results)
        
        # Clear GPU memory between runs
        import torch
        torch.cuda.empty_cache()
        
        # Need to restart Python for clean v1 import (env vars)
        print("\n⚠️  For accurate v1 comparison, run separately:")
        print(f"   python benchmark/benchmark_v0_v1.py --model {args.model}")
        
    else:
        # Run single version
        results = run_benchmark(
            use_v0=args.v0,
            model_name=args.model,
            gpu_mem=args.gpu_mem,
            warmup=args.warmup,
            iterations=args.iterations,
            batch_sizes=batch_sizes
        )
        all_results.extend(results)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Batch':<8} {'Latency':<14} {'Per-Request':<14} {'Throughput':<15}")
    print("-" * 55)
    for r in all_results:
        print(f"{r.batch_size:<8} {r.avg_latency_ms:>6.2f}ms      {r.per_request_ms:>6.2f}ms       {r.throughput_rps:>8.1f} req/s")
    
    # Save results
    if args.output:
        output_data = {
            "metadata": {
                "model": args.model,
                "gpu_mem": args.gpu_mem,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "timestamp": datetime.now().isoformat()
            },
            "results": [asdict(r) for r in all_results]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

