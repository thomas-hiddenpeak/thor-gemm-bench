#!/usr/bin/env python3
"""Generate publication-quality plots from benchmark JSONL results.

Produces:
  1. Top-N bar chart (TFLOPS per config)
  2. Scaling plot (TFLOPS vs problem size)
  3. Efficiency heatmap (tile shape x cluster layout)
  4. Comparison bar chart (multiple implementations)

Usage:
    ./plot_results.py results/*.jsonl                          # top-10 bar
    ./plot_results.py results/*.jsonl --type scaling            # scaling plot
    ./plot_results.py results/*.jsonl --type heatmap            # tile×cluster heatmap
    ./plot_results.py results/*.jsonl --output figures/bench    # custom output dir
    ./plot_results.py results/fp4/*.jsonl --label FP4 results/bf16/*.jsonl --label BF16  # compare
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# matplotlib setup — non-interactive headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Colors ──
FP4_COLOR = "#E74C3C"
BF16_COLOR = "#3498DB"
CUTLASS_COLOR = "#2ECC71"
BAR_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
              "#1ABC9C", "#E67E22", "#2980B9", "#27AE60", "#C0392B"]


def load_results(paths: List[str]) -> List[Dict[str, Any]]:
    """Load JSONL files, skip invalid lines."""
    results = []
    for path in paths:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except FileNotFoundError:
            print(f"  [warn] {path}: not found", file=sys.stderr)
    return results


def _sort_key(r: Dict[str, Any]) -> float:
    return r.get("tflops", r.get("gflops", 0) / 1000.0)


def plot_topn(
    results: List[Dict[str, Any]],
    n: int = 10,
    output: str = "figures/topn.png",
    title_prefix: str = "",
) -> None:
    """Horizontal bar chart: top-N configs by TFLOPS."""
    sorted_res = sorted(results, key=_sort_key, reverse=True)
    top = sorted_res[:n]

    labels = []
    tflops_vals = []
    peak_pcts = []
    for r in top:
        cfg = r.get("config", r.get("name", "?"))
        cluster = r.get("cluster", "")
        lbl = f"{cfg} {cluster}".strip()
        labels.append(lbl)
        tflops_vals.append(float(r.get("tflops", r.get("gflops", 0) / 1000.0)))
        peak_pcts.append(float(r.get("peak_pct", r.get("efficiency", 0))))

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.45)))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, tflops_vals, color=BAR_COLORS[:len(labels)], height=0.6)

    # Annotate each bar with peak% and TF value
    for i, (bar, tf, pct) in enumerate(zip(bars, tflops_vals, peak_pcts)):
        label = f"{tf:.0f} TF ({pct:.1f}%)" if pct > 0 else f"{tf:.0f} TF"
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("TFLOPS", fontsize=10)
    title = f"{title_prefix} Top-{n} Configurations" if title_prefix else f"Top-{n} Configurations"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(tflops_vals) * 1.3)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"  Saved: {output}")
    plt.close(fig)


def plot_scaling(
    results: List[Dict[str, Any]],
    output: str = "figures/scaling.png",
    title_prefix: str = "",
) -> None:
    """Line plot: TFLOPS vs problem size (M=N=K)."""
    # Filter for square problems
    square = [r for r in results if r.get("m", 0) == r.get("n", 0) == r.get("k", 0) and r.get("m", 0) > 0]
    if not square:
        print("  [warn] No square problem results for scaling plot", file=sys.stderr)
        return

    # Group by config name (tile shape)
    groups: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for r in square:
        cfg = r.get("config", r.get("name", "?"))
        m = int(r["m"])
        tf = float(r.get("tflops", r.get("gflops", 0) / 1000.0))
        groups[cfg].append((m, tf))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
    for i, (cfg, pts) in enumerate(sorted(groups.items())):
        pts.sort(key=lambda x: x[0])
        sizes = [p[0] for p in pts]
        tfs = [p[1] for p in pts]
        ax.plot(sizes, tfs, "o-", color=colors[i % len(colors)], label=cfg, markersize=6)

    # Peak line
    ax.axhline(y=1032, color="gray", linestyle="--", alpha=0.5, label="Peak (1032 TF)")

    ax.set_xlabel("Problem Size M=N=K", fontsize=10)
    ax.set_ylabel("TFLOPS", fontsize=10)
    title = f"{title_prefix} Scaling" if title_prefix else "Scaling vs Problem Size"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"  Saved: {output}")
    plt.close(fig)


def plot_heatmap(
    results: List[Dict[str, Any]],
    output: str = "figures/heatmap.png",
    title_prefix: str = "",
) -> None:
    """Heatmap: tile M × cluster layout, colored by TFLOPS."""
    # Extract tile_m, cluster strings
    rows: Dict[str, float] = {}
    tile_ms: set = set()
    clusters: set = set()
    for r in results:
        cfg = r.get("config", "")
        cluster = r.get("cluster", "")
        tf = float(r.get("tflops", 0))
        if not cluster:
            continue
        # Try to extract M from config string like "M128xN128xK256"
        import re
        m_match = re.search(r"M(\d+)", cfg)
        if not m_match:
            continue
        tile_m = int(m_match.group(1))
        tile_ms.add(tile_m)
        clusters.add(cluster)
        rows[(tile_m, cluster)] = tf

    if not tile_ms or not clusters:
        print("  [warn] Insufficient data for heatmap (need tile M + cluster)", file=sys.stderr)
        return

    tile_ms_sorted = sorted(tile_ms)
    clusters_sorted = sorted(clusters)

    data = np.zeros((len(tile_ms_sorted), len(clusters_sorted)))
    data[:] = np.nan
    for i, tm in enumerate(tile_ms_sorted):
        for j, cl in enumerate(clusters_sorted):
            val = rows.get((tm, cl))
            if val is not None and val > 0:
                data[i, j] = val

    fig, ax = plt.subplots(figsize=(max(6, len(clusters_sorted) * 1.5),
                                    max(4, len(tile_ms_sorted) * 0.8)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", origin="lower")

    # Labels
    ax.set_xticks(range(len(clusters_sorted)))
    ax.set_xticklabels(clusters_sorted, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(tile_ms_sorted)))
    ax.set_yticklabels([f"M={m}" for m in tile_ms_sorted], fontsize=9)
    ax.set_xlabel("Cluster Layout", fontsize=10)
    ax.set_ylabel("Tile M", fontsize=10)
    title = f"{title_prefix} Efficiency Heatmap" if title_prefix else "Tile × Cluster Efficiency (TFLOPS)"
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(len(tile_ms_sorted)):
        for j in range(len(clusters_sorted)):
            val = data[i, j]
            if not np.isnan(val) and val > 0:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if val < np.nanmax(data) * 0.5 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("TFLOPS", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist=".")
    fig.savefig(output, dpi=150)
    print(f"  Saved: {output}")
    plt.close(fig)


def plot_comparison(
    result_groups: List[Tuple[str, List[Dict[str, Any]]]],
    output: str = "figures/comparison.png",
) -> None:
    """Grouped bar chart: compare multiple implementations side by side."""
    # Find common config names across groups
    config_tfs: Dict[str, Dict[str, float]] = defaultdict(dict)
    all_configs: set = set()
    for label, group in result_groups:
        for r in group:
            cfg = r.get("config", r.get("name", "?"))
            tf = float(r.get("tflops", r.get("gflops", 0) / 1000.0))
            if tf > 0:
                config_tfs[cfg][label] = tf
                all_configs.add(cfg)

    if not all_configs:
        print("  [warn] No data for comparison plot", file=sys.stderr)
        return

    # Sort configs by mean TFLOPS across groups
    sorted_configs = sorted(all_configs, key=lambda c: np.mean(list(config_tfs[c].values()) or [0]), reverse=True)
    sorted_configs = sorted_configs[:8]  # limit to top-8 for readability

    n_groups = len(result_groups)
    x = np.arange(len(sorted_configs))
    width = 0.8 / n_groups
    colors_list = [FP4_COLOR, BF16_COLOR, CUTLASS_COLOR, "#F39C12"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, _) in enumerate(result_groups):
        vals = [config_tfs[c].get(label, 0) for c in sorted_configs]
        ax.bar(x + i * width - (n_groups - 1) * width / 2, vals, width,
               label=label, color=colors_list[i % len(colors_list)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_configs, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("TFLOPS", fontsize=10)
    ax.set_title("Implementation Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"  Saved: {output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument("files", nargs="+", help="JSONL result files")
    parser.add_argument("--type", default="topn", choices=["topn", "scaling", "heatmap", "comparison"],
                        help="Plot type (default: topn)")
    parser.add_argument("--top", type=int, default=10, help="Top-N for bar chart")
    parser.add_argument("--output", "-o", default="figures/benchmark", help="Output path stem (without extension)")
    parser.add_argument("--title", default="", help="Optional title prefix")
    parser.add_argument("--label", action="append", default=[], help="Group label for comparison plot")
    args = parser.parse_args()

    if args.type == "comparison":
        # Groups are interleaved: --label A files... --label B files...
        # We need to parse grouped labels. But argparse collects all --label values.
        # Use a convention: N groups, files split evenly.
        n_groups = max(len(args.label), 1)
        if n_groups == 1 and not any(l for l in args.label):
            n_groups = 1
        if len(args.label) > 1 and len(args.files) % len(args.label) != 0:
            print("  [error] For comparison, --label count must divide file count evenly", file=sys.stderr)
            sys.exit(1)

        chunk = len(args.files) // max(n_groups, 1)
        groups = []
        for i in range(n_groups):
            lbl = args.label[i] if i < len(args.label) else f"Group-{i}"
            file_chunk = args.files[i * chunk:(i + 1) * chunk] if n_groups > 1 else args.files
            results = load_results(file_chunk)
            groups.append((lbl, results))
            print(f"  {lbl}: {len(results)} results from {len(file_chunk)} file(s)")

        plot_comparison(groups, f"{args.output}_comparison.png")
        return

    results = load_results(args.files)
    if not results:
        print("No results loaded.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(results)} results from {len(args.files)} file(s)")

    ext = ".png"
    if args.type == "topn":
        plot_topn(results, args.top, f"{args.output}_top{args.top}{ext}", args.title)
    elif args.type == "scaling":
        plot_scaling(results, f"{args.output}_scaling{ext}", args.title)
    elif args.type == "heatmap":
        plot_heatmap(results, f"{args.output}_heatmap{ext}", args.title)


if __name__ == "__main__":
    main()
