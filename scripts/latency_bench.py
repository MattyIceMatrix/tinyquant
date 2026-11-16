#!/usr/bin/env python3
"""Run the compiled latency benchmark several times and report the median.

A single timing run on a shared vCPU is not a measurement. During development
repeated runs of the same binary on the same input produced INT8/FP32 ratios
between 0.48x and 0.71x purely from scheduling noise, which is wide enough to
flip a conclusion. Every timing claim in this repo therefore comes from a median
of several trials, with the full set retained so the spread is visible instead of
summarised away.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def run(exe: pathlib.Path, reps: int, trials: int) -> dict:
    runs = []
    for _ in range(trials):
        out = subprocess.run(
            [str(exe), "artifacts/model.tq8", "artifacts/model.tqf",
             "artifacts/test_vectors.tqv", str(reps)],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        runs.append(json.loads(out.stdout))

    def med(path: tuple[str, str]) -> float:
        return statistics.median(r[path[0]][path[1]] for r in runs)

    int8_p50, fp32_p50 = med(("int8", "p50_us")), med(("fp32", "p50_us"))
    return {
        "trials": trials,
        "reps_per_trial": reps,
        "samples": runs[0]["samples"],
        "skip_zeros_build": runs[0]["skip_zeros_build"],
        "int8": {k: round(med(("int8", k)), 4) for k in ("min_us", "p50_us", "p95_us", "p99_us")},
        "fp32": {k: round(med(("fp32", k)), 4) for k in ("min_us", "p50_us", "p95_us", "p99_us")},
        "int8_accuracy": runs[0]["int8"]["accuracy"],
        "fp32_accuracy": runs[0]["fp32"]["accuracy"],
        "speedup_p50": round(fp32_p50 / int8_p50, 4),
        "trial_speedups": [round(r["fp32"]["p50_us"] / r["int8"]["p50_us"], 4) for r in runs],
    }


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    for exe_name, out_name in (("bench", "latency.json"),
                               ("bench_skipzeros", "latency_skipzeros.json")):
        exe = ROOT / "build" / exe_name
        if not exe.exists():
            print(f"missing {exe}; run `make bench` first")
            raise SystemExit(1)
        rep = run(exe, reps, trials)
        (ART / out_name).write_text(json.dumps(rep, indent=2))
        spread = max(rep["trial_speedups"]) - min(rep["trial_speedups"])
        print(f"{exe_name:<16} int8 p50={rep['int8']['p50_us']:.3f}us "
              f"fp32 p50={rep['fp32']['p50_us']:.3f}us  "
              f"median speedup={rep['speedup_p50']:.3f}x  "
              f"(trial spread {spread:.3f} across {trials} trials)")
