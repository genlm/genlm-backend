# Performance Benchmarking

This directory contains various benchmarking scripts to evaluate the system's performance.

## Backend Comparison
* `benchmark_backend.py`: Evaluates performance differences between `AsyncVirtualLM` (vLLM-based) and `AsyncTransformer` (HuggingFace-based) implementations. Run with:

    ```bash
    pytest benchmark/benchmark_backend.py --benchmark-only --benchmark-group-by=func
    ```

## Mass Sum Performance
* `benchmark_mass_sum.py`: Compares different implementations of TokenCharacterTrie mass_sum calculations:
  - Sequential vs parallel implementations
  - CPU vs GPU parallel implementations
  - Sync vs async implementations
  
  Run with:
  
    ```bash
    pytest benchmark/benchmark_mass_sum.py --benchmark-only
    ```

## Internal Optimizations
* `benchmark_optimizations.py`: Measures performance gains from internal modifications to the vLLM engine by comparing against a reference implementation. Run with:
  
    ```bash
    pytest benchmark/benchmark_optimizations.py --benchmark-only --benchmark-group-by=func
    ```

## Prefix Caching
* `benchmark_prefix_caching.py`: Evaluates the impact of vLLM's prefix caching feature using a scenario with a large prompt. Run with:
  
    ```bash
    pytest benchmark/benchmark_prefix_caching.py --benchmark-only
    ```
