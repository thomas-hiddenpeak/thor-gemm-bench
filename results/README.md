# Results Directory

Benchmark results are saved here as JSONL files (`results_YYYYMMDD_HHMMSS.jsonl`).
Each line is one benchmark configuration.

## Format

```json
{
  "config":      "M256xN128xK256 SFVec=16",
  "cluster":     "C2x2x1",
  "gflops":      576000.0,
  "tflops":      576.0,
  "peak_pct":    55.8,
  "status":      "PASS",
  "timestamp":   "2026-06-15T12:00:00Z",
  "gpu":         "NVIDIA Thor"
}
```

## Workflow

```bash
# Run a search → appends to a new JSONL
./run_fp4_search.sh

# Summarize all results
cat results/*.jsonl | jq -s 'sort_by(.tflops) | reverse | .[:10]'
```

`.jsonl` files are gitignored — only this README is tracked.
