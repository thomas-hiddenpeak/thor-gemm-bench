#!/usr/bin/env python3
"""Multi-run benchmark harness with cross-run aggregation.

Runs a benchmark binary N times, parses the JSON output from each run,
and produces an aggregated report with grand mean ± CI across runs.

Usage:
    # Run benchmark 5 times, aggregate results
    ./benchmark_suite.py --binary ./bench_nvfp4_fp4 --args "--m=4096 --n=4096 --k=4096 --json" --runs 5

    # Output JSON report
    ./benchmark_suite.py --binary ./bench_nvfp4_fp4 --args "--json" --runs 3 --output result.json

    # Use CPU clock isolation wrapper
    ./benchmark_suite.py --binary ./bench_nvfp4_fp4 --args "--json" --wrapper ./run_isolated.sh

    # Run multiple configs from a file (one config per line)
    ./benchmark_suite.py --configs configs.txt --runs 3 --output suite_results.json
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple


def parse_json_from_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from benchmark stdout."""
    # The benchmark outputs JSON to stdout
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def run_single(binary: str, args_list: List[str],
               wrapper: Optional[str] = None,
               timeout: int = 120) -> Tuple[Optional[Dict[str, Any]], float, str]:
    """Run benchmark once, return (parsed_json, elapsed_sec, raw_stdout)."""
    cmd = []
    if wrapper:
        cmd.append(wrapper)
    cmd.append(binary)
    cmd.extend(args_list)

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - start
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired:
        return None, time.time() - start, "(timeout)"
    except FileNotFoundError:
        print(f"  [error] Binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    parsed = parse_json_from_stdout(stdout)
    if parsed is None:
        print(f"  [warn] No JSON found in output (exit={result.returncode})", file=sys.stderr)
        if stderr:
            print(f"  stderr: {stderr.strip()[:200]}", file=sys.stderr)
    return parsed, elapsed, stdout


def aggregate_runs(runs: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate N run results into grand mean ± CI."""
    valid = [r for r in runs if r is not None]
    n = len(valid)
    if n == 0:
        return {"runs": n, "error": "no valid runs"}

    def _vals(key: str) -> List[float]:
        return [float(r.get(key, 0)) for r in valid if key in r]

    def _stats(vals: List[float]) -> Dict[str, float]:
        if len(vals) < 2:
            return {"mean": float(vals[0]) if vals else 0, "sem": 0, "ci95": 0, "cv_pct": 0}
        mean = statistics.mean(vals)
        if len(vals) < 2:
            stdev = 0
        else:
            stdev = statistics.stdev(vals)
        sem = stdev / math.sqrt(len(vals))
        ci95 = sem * 1.96
        cv = (stdev / mean * 100) if mean > 0 else 0
        return {"mean": round(mean, 3), "min": round(min(vals), 3),
                "max": round(max(vals), 3), "sem": round(sem, 3),
                "ci95": round(ci95, 3), "cv_pct": round(cv, 2)}

    # Aggregate numeric keys
    numeric_keys = ["tflops", "gflops", "peak_pct", "efficiency"]
    agg: Dict[str, Any] = {"runs": n, "config": valid[0].get("config", ""),
                           "cluster": valid[0].get("cluster", ""),
                           "gpu": valid[0].get("gpu", {}),
                           "toolchain": valid[0].get("toolchain", {})}
    for key in numeric_keys:
        vals = _vals(key)
        if vals:
            agg[key] = _stats(vals)

    # Aggregate runtime fields from runtime_ms sub-object
    runtime_keys = ["mean", "min", "max", "stddev", "ci95", "cv_pct"]
    runtime_vals: Dict[str, List[float]] = {k: [] for k in runtime_keys}
    for r in valid:
        rt = r.get("runtime_ms", {})
        if isinstance(rt, dict):
            for k in runtime_keys:
                if k in rt:
                    runtime_vals[k].append(float(rt[k]))
    if runtime_vals["mean"]:
        agg["runtime_ms"] = {}
        for k in runtime_keys:
            if runtime_vals[k]:
                agg["runtime_ms"][f"grand_{k}"] = round(statistics.mean(runtime_vals[k]), 3)
                if len(runtime_vals[k]) > 1:
                    agg["runtime_ms"][f"grand_{k}_ci95"] = round(
                        1.96 * statistics.stdev(runtime_vals[k]) / math.sqrt(len(runtime_vals[k])), 3
                    )

    # Correctness
    passed = sum(1 for r in valid if r.get("correctness") == "passed" or r.get("valid"))
    agg["correctness"] = f"{passed}/{n}"

    # Record full per-run data
    agg["per_run"] = valid
    return agg


def format_report(agg: Dict[str, Any]) -> str:
    """Human-readable summary."""
    lines = []
    cfg = agg.get("config", "")
    cluster = agg.get("cluster", "")
    label = f"{cfg} {cluster}".strip()
    lines.append(f"Config: {label}")
    lines.append(f"Runs:   {agg['runs']}")
    if "error" in agg:
        lines.append(f"Error:  {agg['error']}")
        return "\n".join(lines)

    tf = agg.get("tflops", {})
    if tf:
        lines.append(f"TFLOPS: {tf['mean']:.1f} ± {tf['ci95']:.1f} (CV: {tf['cv_pct']:.1f}%)")
    peak = agg.get("peak_pct", {})
    if peak:
        lines.append(f"Peak:   {peak['mean']:.1f}% ± {peak['ci95']:.1f}%")
    rt = agg.get("runtime_ms", {})
    if rt:
        grand = rt.get("grand_mean", 0)
        ci = rt.get("grand_mean_ci95", 0)
        lines.append(f"Time:   {grand:.3f} ± {ci:.3f} ms")
    lines.append(f"Pass:   {agg.get('correctness', '?')}")
    gpu = agg.get("gpu", {})
    if isinstance(gpu, dict) and gpu.get("name"):
        lines.append(f"GPU:    {gpu['name']}")
    tc = agg.get("toolchain", {})
    if isinstance(tc, dict) and tc.get("cutlass_commit"):
        lines.append(f"CUTLASS: {tc['cutlass_commit']}")
    return "\n".join(lines)


def run_config(binary: str, args_str: str, runs: int,
               wrapper: Optional[str] = None,
               timeout: int = 120) -> Dict[str, Any]:
    """Run one config N times, return aggregated result."""
    args_list = subprocess.list2cmdline([args_str]) if " " not in args_str else []
    # Use shlex to split properly
    import shlex
    args_list = shlex.split(args_str)

    print(f"  Running {runs}x: {binary} {' '.join(args_list[:4])}...")
    raw_outputs = []
    run_results = []
    for i in range(runs):
        parsed, elapsed, stdout = run_single(binary, args_list, wrapper, timeout)
        status = "OK" if parsed else "NO_JSON"
        print(f"    [{i+1}/{runs}] {elapsed:.1f}s {status}")
        run_results.append(parsed)
        raw_outputs.append(stdout)

    agg = aggregate_runs(run_results)
    return agg


def run_from_file(config_path: str, binary: str, runs: int,
                  wrapper: Optional[str] = None, timeout: int = 120) -> List[Dict[str, Any]]:
    """Run each config from a file."""
    with open(config_path) as f:
        configs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    results = []
    for cfg in configs:
        agg = run_config(binary, cfg, runs, wrapper, timeout)
        results.append(agg)
        print(format_report(agg))
        print()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-run benchmark harness")
    parser.add_argument("--binary", "-b", default="./bench_nvfp4_fp4", help="Benchmark binary")
    parser.add_argument("--args", default="--json", help="Benchmark arguments")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument("--wrapper", default=None, help="Wrapper script (e.g. ./run_isolated.sh)")
    parser.add_argument("--configs", default=None, help="File with one config per line")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--timeout", type=int, default=120, help="Per-run timeout (seconds)")
    args = parser.parse_args()

    if args.configs:
        results = run_from_file(args.configs, args.binary, args.runs, args.wrapper, args.timeout)
    else:
        agg = run_config(args.binary, args.args, args.runs, args.wrapper, args.timeout)
        results = [agg]
        print("\n" + format_report(agg))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
