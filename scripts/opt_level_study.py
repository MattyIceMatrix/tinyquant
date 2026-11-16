#!/usr/bin/env python3
"""Does the INT8-is-slower result survive better compiler flags?

If the slowdown were an artefact of unoptimised code, turning the optimiser up
would close the gap. This measures that instead of assuming it. -march=native is
included because it is what anyone benchmarking on their own box would reach for.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "opt"
FLAG_SETS = [
    ["-O0"], ["-O2"], ["-O3"], ["-Os"],
    ["-O3", "-march=native"],
    ["-O3", "-fno-tree-vectorize"],
]
REPS = "60"

if __name__ == "__main__":
    BUILD.mkdir(parents=True, exist_ok=True)
    rows = []
    for flags in FLAG_SETS:
        name = "_".join(f.lstrip("-") for f in flags)
        exe = BUILD / f"bench_{name}"
        subprocess.run(["gcc", "-std=c99", *flags, "-DTQ_SKIP_ZEROS=0",
                        "engine/tinyq.c", "engine/bench.c", "-o", str(exe)],
                       cwd=ROOT, check=True)
        best = None
        for _ in range(3):  # median of three, same protocol as the scaling study
            out = subprocess.run([str(exe), "artifacts/model.tq8", "artifacts/model.tqf",
                                  "artifacts/test_vectors.tqv", REPS],
                                 cwd=ROOT, capture_output=True, text=True, check=True)
            r = json.loads(out.stdout)
            cand = (r["int8"]["p50_us"], r["fp32"]["p50_us"])
            if best is None or cand[0] < best[0]:
                best = cand
        i8, f32 = best
        rows.append({"flags": " ".join(flags), "int8_p50_us": i8, "fp32_p50_us": f32,
                     "speedup_p50": round(f32 / i8, 4)})
        print(f"  {' '.join(flags):<26} int8={i8:>8.3f}us  fp32={f32:>8.3f}us  "
              f"speedup={rows[-1]['speedup_p50']:.3f}x", flush=True)

    summary = {"rows": rows,
               "int8_ever_faster": any(r["speedup_p50"] > 1.0 for r in rows),
               "best_speedup": max(r["speedup_p50"] for r in rows)}
    (ROOT / "artifacts" / "opt_levels.json").write_text(json.dumps(summary, indent=2))
    print(f"\nINT8 faster at any optimisation level: {summary['int8_ever_faster']} "
          f"(best {summary['best_speedup']}x)")
