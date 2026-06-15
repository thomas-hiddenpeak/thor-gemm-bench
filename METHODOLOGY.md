# Measurement Methodology

## Timing Method

All benchmarks use **CUDA Events** (`cudaEventRecord`) for GPU-side timing, not CPU wall clocks. This isolates the measurement from CPU scheduling noise and driver overhead.

- Each iteration records a start and stop CUDA event around `gemm.run()`
- Events are placed on the default stream (stream 0)
- `cudaDeviceSynchronize()` ensures all iterations complete before reading timestamps
- `cudaEventElapsedTime()` returns the GPU-observed elapsed time in milliseconds

## Warmup

Before the timed loop, the kernel is executed `--warmup` (default: 5) times. These iterations are **not timed**. This ensures:
- GPU clocks stabilize at operating frequency
- Caches (L1, L2, TMEM) reach a steady state
- Any lazy loading / JIT compilation is completed

## Outlier Handling

For >= 10 timed iterations, the top and bottom **5%** of samples are trimmed before computing mean and standard deviation. This removes:
- First-iteration cold-start effects
- Occasional system-interrupt outliers
- Thermal throttling transients

Trimmed count is reported as `trimmed` in the stats.

## Statistical Reporting

```
mean_ms    = trimmed arithmetic mean
min_ms     = overall minimum (before trimming)
max_ms     = overall maximum (before trimming)
median_ms  = 50th percentile (before trimming)
stddev_ms  = sample standard deviation (after trimming)
sem_ms     = stddev / sqrt(N)   — standard error of the mean
ci95_ms    = 1.96 × SEM        — 95% confidence interval half-width
cv_pct     = stddev / mean × 100  — coefficient of variation
```

### Confidence Interval

The 95% CI uses the normal approximation (`z = 1.96`) which is valid for `N >= 30` (default iteration count is 50, trimmed to ~45). For fewer iterations, the CI is a conservative estimate; the true interval would be wider (t-distribution).

### Coefficient of Variation

CV < 1% indicates a stable measurement. CV > 5% suggests the configuration has high run-to-run variance — the reported mean should be treated with caution.

## Reproducibility

### Environment Fingerprint

Each JSON output record includes:
- GPU name (from `cudaGetDeviceProperties`)
- SM count
- GPU clock frequency (from sysfs `devfreq` — more reliable than CUDA API on Thor)
- CUDA toolkit version (from `cudaRuntimeGetVersion`)
- CUTLASS commit SHA (compile-time injection via `-DCUTLASS_COMMIT`)

### Hardware Isolation

For the most reproducible results, use `run_isolated.sh` which:
1. Locks GPU clock to maximum frequency (sysfs `userspace` governor)
2. Sets MAXN power profile (via `nvpmodel -m 0` if available)
3. Restores original governor state on exit (via `trap`)

Without isolation, GPU boost/throttle behavior can introduce ±5–15% variance depending on thermal state and power budget.

### Software Versioning

| Component | Version | How Verified |
|-----------|---------|-------------|
| CUDA Toolkit | 13.3.33 | `cudaRuntimeGetVersion()` at runtime |
| CUTLASS | commit `32a56db1` | `git rev-parse HEAD` at build time; baked into binary via `-DCUTLASS_COMMIT` |

## Result Format

### Per-benchmark JSON (single shape)

```json
{
  "benchmark": "nvfp4_fp4_gemm",
  "m": 4096, "n": 4096, "k": 4096,
  "runtime_ms": {
    "mean": 0.241, "min": 0.238, "max": 0.246,
    "median": 0.240, "stddev": 0.002,
    "ci95": 0.0006, "cv_pct": 0.83
  },
  "gflops": 569805.7, "tflops": 569.8,
  "peak_tflops": 1032.0, "peak_pct": 55.2,
  "correctness": "passed",
  "gpu": {
    "name": "NVIDIA Thor",
    "sm_count": 20,
    "clock_mhz": 1575
  },
  "toolchain": {
    "cuda_version": 13033,
    "cutlass_commit": "32a56db1"
  }
}
```

### Search-archive JSONL

Benchmark search scripts (run_fp4_*.sh, run_tile_*.sh, run_sf_search.sh) save one JSON object per configuration to `results/results_YYYYMMDD_HHMMSS.jsonl`:

```json
{
  "config":    "M256xN128xK256 SFVec=16",
  "cluster":   "C2x2x1",
  "gflops":    576000.0,
  "tflops":    576.0,
  "peak_pct":  55.8,
  "status":    "PASS",
  "timestamp": "2026-06-15T12:00:00Z",
  "gpu":       "NVIDIA Thor"
}
```

## Known Limitations

1. **CUDA Event overhead**: For very fast kernels (< 0.1ms), CUDA event recording overhead can skew timing. All reported GEMMs run 0.2–0.8ms per iteration.
2. **No ECC state recording**: ECC can reduce effective bandwidth. If ECC is enabled/disabled between runs, results may not be comparable. Current output does not capture ECC state.
3. **Single GPU only**: The benchmark targets a single NVIDIA Thor (SM110a) GPU. Multi-GPU or MIG configurations are not supported.
4. **No power capping**: Power limits are not recorded in JSON output. If power capping varies, clock frequencies may differ even with `run_isolated.sh`.
5. **Global variables in bf16 benchmark**: `bench_nvfp4_fp4_bf16.cu` uses global `HostTensor` variables, making it unsuitable for multi-threaded benchmarking.

## References

- CUTLASS: https://github.com/nvidia/cutlass
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVFP4 specification: https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions
