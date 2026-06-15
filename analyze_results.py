#!/usr/bin/env python3
"""Aggregate, sort, summarize, and export benchmark results.

Usage:
    # Top-10 configurations by TFLOPS
    ./analyze_results.py results/*.jsonl

    # Group by tile config, show per-group stats
    ./analyze_results.py results/*.jsonl --group-by config

    # LaTeX table (paper-grade)
    ./analyze_results.py results/*.jsonl --latex

    # CSV export
    ./analyze_results.py results/*.jsonl --csv results.csv

    # Summary statistics
    ./analyze_results.py results/*.jsonl --summary

    # Cross-implementation comparison
    ./analyze_results.py results/fp4/*.jsonl --label FP4 --name "FP4→FP4" \
                        results/bf16/*.jsonl --label BF16 --name "BF16→BF16" \
                        --latex --compare

    # Filter by status
    ./analyze_results.py results/*.jsonl --status PASS
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def load_results(paths: List[str]) -> List[Dict[str, Any]]:
    """Load all JSONL files, skip invalid entries."""
    results = []
    for path in paths:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"  [warn] {path}: {e}", file=sys.stderr)
        except FileNotFoundError:
            print(f"  [warn] {path}: not found", file=sys.stderr)
    return results


def _tf(r: Dict[str, Any]) -> float:
    return float(r.get("tflops", r.get("gflops", 0) / 1000.0))


def _peak(r: Dict[str, Any]) -> float:
    return float(r.get("peak_pct", r.get("efficiency", 0)))


def _status(r: Dict[str, Any]) -> str:
    return r.get("status", r.get("correctness", "?")) or "?"


def _config_name(r: Dict[str, Any]) -> str:
    cfg = r.get("config", r.get("name", ""))
    cluster = r.get("cluster", "")
    return f"{cfg} {cluster}".strip() or "?"


def fmt_tf(t: float) -> str:
    if t >= 1000:
        return f"{t / 1000:.2f}P"
    return f"{t:.1f}"


def fmt_peak(pct: float) -> str:
    return f"{pct:.1f}%" if pct > 0 else "—"


def _filter(results: List[Dict[str, Any]], status: Optional[str] = None) -> List[Dict[str, Any]]:
    if status:
        return [r for r in results if _status(r) == status]
    return [r for r in results if _tf(r) > 0]


# ── Display modes ──

def show_top(results: List[Dict[str, Any]], n: int = 10, status: Optional[str] = None) -> None:
    """Tabulate top-N by TFLOPS."""
    data = _filter(results, status)
    if not data:
        print("  (no valid results)")
        return
    data.sort(key=_tf, reverse=True)

    print(f"{'Rank':<6} {'Config':<45} {'TF':>8} {'Peak%':>7} {'Status':<8}")
    print("-" * 78)
    for i, r in enumerate(data[:n], 1):
        print(f"{i:<6} {_config_name(r):<45} {fmt_tf(_tf(r)):>8} {fmt_peak(_peak(r)):>7} {_status(r):<8}")
    if len(data) > n:
        print(f"\n  ... {len(data) - n} more (total: {len(data)})")


def show_grouped(results: List[Dict[str, Any]], group_by: str = "config") -> None:
    """Per-group max/mean/count."""
    groups: Dict[str, List[float]] = defaultdict(list)
    details: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        key = str(r.get(group_by, "?"))
        t = _tf(r)
        if t > 0:
            groups[key].append(t)
            details[key].append(r)
    if not groups:
        print("  (no valid results)")
        return

    print(f"{'Group':<45} {'Count':>6} {'Max TF':>8} {'Mean TF':>8} {'Best Config':<40}")
    print("-" * 110)
    for key in sorted(groups, key=lambda k: max(groups[k]), reverse=True):
        vals = groups[key]
        best_rec = max(details[key], key=lambda r: _tf(r))
        print(f"{key:<45} {len(vals):>6} {fmt_tf(max(vals)):>8} {fmt_tf(sum(vals)/len(vals)):>8} {_config_name(best_rec):<40}")


def show_summary(results: List[Dict[str, Any]]) -> None:
    """Overall summary statistics."""
    data = _filter(results)
    if not data:
        print("  (no valid results)")
        return
    tfs = [_tf(r) for r in data]
    peaks = [_peak(r) for r in data]

    import statistics
    print(f"{'Metric':<30} {'Value':<15}")
    print("-" * 45)
    print(f"{'Total results':<30} {len(data):<15}")
    print(f"{'Passing (PASS/valid)':<30} {sum(1 for r in data if _status(r) == 'PASS' or r.get('valid')):<15}")
    print(f"{'Max TFLOPS':<30} {max(tfs):.1f}")
    print(f"{'Mean TFLOPS':<30} {statistics.mean(tfs):.1f}")
    print(f"{'Median TFLOPS':<30} {statistics.median(tfs):.1f}")
    print(f"{'Stdev TFLOPS':<30} {statistics.stdev(tfs):.2f}" if len(tfs) > 1 else "")
    print(f"{'Max peak %':<30} {max(peaks):.1f}%" if peaks else "")
    print(f"{'Mean peak %':<30} {statistics.mean(peaks):.1f}%" if peaks else "")


# ── LaTeX export ──

def escape_latex(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("→", r"$\rightarrow$")


def show_latex(results: List[Dict[str, Any]], n: int = 20, status: Optional[str] = None,
               caption: str = "Benchmark Results", label: str = "tab:benchmark") -> None:
    """Publication-grade LaTeX table (booktabs). Columns: Config, TFLOPS, Peak%, Status."""
    data = _filter(results, status)
    if not data:
        print("  (no valid results)")
        return
    data.sort(key=_tf, reverse=True)
    top = data[:n]

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lrrl}")
    print(r"\toprule")
    print(r"Configuration & {TFLOPS} & {Peak} & {Status} \\")
    print(r"\midrule")
    for r in top:
        print(f"  {escape_latex(_config_name(r))} & {_tf(r):.1f} & {_peak(r):.1f}\\% & {_status(r)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{" + escape_latex(caption) + "}")
    print(r"\label{" + label + "}")
    print(r"\end{table}")


def show_latex_comparison(groups: List[Tuple[str, str, List[Dict[str, Any]]]],
                          n: int = 10, caption: str = "Implementation Comparison",
                          label: str = "tab:comparison") -> None:
    """Side-by-side LaTeX table: config × implementation."""
    # Collect common configs
    config_tfs: Dict[str, Dict[str, float]] = defaultdict(dict)
    all_configs: set = set()
    for name, _, grp in groups:
        for r in grp:
            cfg = _config_name(r)
            t = _tf(r)
            if t > 0:
                config_tfs[cfg][name] = t
                all_configs.add(cfg)

    if not all_configs:
        print("  (no data)")
        return

    # Sort by max TFLOPS
    sorted_cfgs = sorted(all_configs, key=lambda c: max(config_tfs[c].values()), reverse=True)[:n]

    n_cols = len(groups) + 1
    col_spec = "l" + "r" * len(groups)
    headers = "Configuration"
    row_fmt = "{:<40}"
    for name, _, _ in groups:
        headers += f" & {{{name}}}"
        row_fmt += " & {:>8}"

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\toprule")
    print(headers + r" \\")
    print(r"\midrule")
    for cfg in sorted_cfgs:
        vals = [config_tfs[cfg].get(name, 0) for name, _, _ in groups]
        row = escape_latex(cfg)
        for v in vals:
            row += f" & {v:.1f}" if v > 0 else " & ---"
        print(row + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{" + escape_latex(caption) + "}")
    print(r"\label{" + label + "}")
    print(r"\end{table}")


# ── CSV export ──

def export_csv(results: List[Dict[str, Any]], path: str) -> None:
    """Flatten selected fields to CSV."""
    data = _filter(results)
    if not data:
        print("  (no data to export)")
        return
    field_map = {
        "config": "config",
        "cluster": "cluster",
        "tflops": "tflops",
        "peak_pct": "peak_pct",
        "gflops": "gflops",
        "status": "status",
        "timestamp": "timestamp",
        "gpu": "gpu",
        "m": "m",
        "n": "n",
        "k": "k",
        "efficiency": "efficiency",
        "correctness": "correctness",
    }
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(field_map.keys()), extrasaction="ignore")
        writer.writeheader()
        for r in data:
            row = {}
            for out_key, in_key in field_map.items():
                row[out_key] = r.get(in_key, "")
            writer.writerow(row)
    print(f"  Exported {len(data)} rows to {path}")


# ── Main ──

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CUDA tile benchmark JSONL")
    parser.add_argument("files", nargs="+", help="JSONL result files")
    parser.add_argument("--top", type=int, default=10, help="Number of top results")
    parser.add_argument("--status", default=None, help='Filter: PASS, BUILD_FAIL, etc.')
    parser.add_argument("--group-by", default=None, help="Group by a field (config, cluster, ...)")
    parser.add_argument("--summary", action="store_true", help="Summary statistics")
    parser.add_argument("--latex", action="store_true", help="LaTeX table output")
    parser.add_argument("--csv", default=None, help="Export to CSV")
    parser.add_argument("--label", action="append", default=[], help="Group label (for --compare)")
    parser.add_argument("--name", action="append", default=[], help="Group display name (for --compare)")
    parser.add_argument("--compare", action="store_true", help="Cross-implementation comparison (requires --label)")
    parser.add_argument("--caption", default="Benchmark Results", help="LaTeX table caption")
    args = parser.parse_args()

    if args.compare:
        # N groups: --label A ... files ... --label B ... files ...
        # Parse interleaved labels manually from the original argv
        # Actually, argparse collects all --label values first, then positional files.
        # So we can't interleave easily. Use a convention: --label L1 L2 L3,
        # files split into equal chunks.
        n_groups = max(len(args.label), 1)
        if n_groups == 1 and not args.label:
            print("  [error] --compare requires --label", file=sys.stderr)
            sys.exit(1)
        chunk = len(args.files) // n_groups
        if chunk * n_groups != len(args.files):
            print("  [error] File count must be divisible by --label count", file=sys.stderr)
            sys.exit(1)

        groups = []
        for i in range(n_groups):
            name = args.name[i] if i < len(args.name) else args.label[i]
            file_chunk = args.files[i * chunk:(i + 1) * chunk]
            res = load_results(file_chunk)
            groups.append((name, args.label[i], res))
            print(f"  {name}: {len(res)} results from {len(file_chunk)} file(s)")

        if args.latex:
            show_latex_comparison(groups, args.top, args.caption)
        else:
            # Text comparison table
            print(f"\n{'Config':<40}", end="")
            for name, _, _ in groups:
                print(f" {name:>10}", end="")
            print()
            print("-" * (40 + 12 * len(groups)))
            # Collect common configs
            config_tfs: Dict[str, Dict[str, float]] = defaultdict(dict)
            all_cfgs: set = set()
            for name, _, grp in groups:
                for r in grp:
                    cfg = _config_name(r)
                    t = _tf(r)
                    if t > 0:
                        config_tfs[cfg][name] = t
                        all_cfgs.add(cfg)
            sorted_cfgs = sorted(all_cfgs, key=lambda c: max(config_tfs[c].values()), reverse=True)[:args.top]
            for cfg in sorted_cfgs:
                print(f"{cfg:<40}", end="")
                for name, _, _ in groups:
                    v = config_tfs[cfg].get(name, 0)
                    print(f" {fmt_tf(v):>10}", end="")
                print()
        return

    results = load_results(args.files)
    if not results:
        print("No results loaded.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(results)} results from {len(args.files)} file(s)\n")

    if args.summary:
        show_summary(results)
    elif args.latex:
        show_latex(results, args.top, args.status, args.caption)
    elif args.group_by:
        show_grouped(results, args.group_by)
    elif args.csv:
        export_csv(results, args.csv)
    else:
        show_top(results, args.top, args.status)


if __name__ == "__main__":
    main()
