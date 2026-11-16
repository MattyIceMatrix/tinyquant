#!/usr/bin/env python3
"""Where does INT8 actually start winning?

On the real model INT8 is *slower* than FP32 on a desktop x86 core. The usual
explanation -- "INT8 is 4x less data" -- only pays off once the weights stop
fitting in cache. Below that, both kernels read from L1 and the FP32 path wins
because the compiler vectorises 8 floats per AVX register while the INT8 path
widens each byte to int32 before accumulating, so it moves the same number of
lanes and pays extra for requantization.

This script tests that explanation instead of asserting it: it builds random
models of increasing width, measures both kernels, and reports the crossover.
Random weights are fine here -- this measures the kernel, not accuracy, and the
output is labelled as such.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tinyquant.export import export_fp32, export_int8, export_test_vectors  # noqa: E402
from tinyquant.model import MLP  # noqa: E402
from tinyquant.quantize import quantize_model  # noqa: E402

WIDTHS = [64, 128, 256, 512, 1024, 1536, 2048]
DEPTH = 3
CFLAGS = ["-std=c99", "-O3", "-march=native"]


def build_engine(width: int, out: str) -> None:
    max_w = max(512, width)
    max_bytes = max(262144, max_w * max_w * (DEPTH + 1) + 65536)
    subprocess.run(
        ["gcc", *CFLAGS, f"-DTQ_MAX_WIDTH={max_w}", f"-DTQ_MAX_WEIGHTS={max_bytes}",
         "-DTQ_SKIP_ZEROS=0", "engine/tinyq.c", "engine/bench.c", "-o", out],
        check=True,
    )


def run_width(width: int, tmp: pathlib.Path, reps: int) -> dict:
    rng = np.random.default_rng(width)
    dims = [width] * DEPTH + [10]
    W = [(rng.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i])).astype(np.float32)
         for i in range(len(dims) - 1)]
    b = [np.zeros(dims[i + 1], dtype=np.float32) for i in range(len(dims) - 1)]
    model = MLP(W, b)

    x = np.abs(rng.standard_normal((64, width)).astype(np.float32))
    qm = quantize_model(model, x, "per_channel", "minmax")

    tq8, tqf, tqv = tmp / "m.tq8", tmp / "m.tqf", tmp / "v.tqv"
    export_int8(qm, str(tq8))
    export_fp32(model, str(tqf))
    export_test_vectors(qm.quantize_input(x[:32]), x[:32], np.zeros(32, dtype=np.int64), str(tqv))

    exe = str(tmp / f"bench_{width}")
    build_engine(width, exe)

    # Three independent trials, median reported. This runs on shared cloud
    # vCPUs where a single trial can be several percent off; a median of three
    # is the minimum defensible protocol, and the spread is reported so the
    # reader can judge it rather than take the number on faith.
    trials = []
    for _ in range(3):
        out = subprocess.run([exe, str(tq8), str(tqf), str(tqv), str(reps)],
                             capture_output=True, text=True, check=True)
        trials.append(json.loads(out.stdout))
    i8 = sorted(t["int8"]["p50_us"] for t in trials)
    f32 = sorted(t["fp32"]["p50_us"] for t in trials)
    r = {"int8": {"p50_us": i8[1]}, "fp32": {"p50_us": f32[1]}}
    spread = {"int8_p50_spread_pct": round((i8[2] - i8[0]) / i8[1] * 100, 2),
              "fp32_p50_spread_pct": round((f32[2] - f32[0]) / f32[1] * 100, 2)}

    int8_bytes = sum(w.size for w in model.W)
    return {
        "width": width,
        "int8_weight_bytes": int8_bytes,
        "fp32_weight_bytes": int8_bytes * 4,
        "int8_p50_us": r["int8"]["p50_us"],
        "fp32_p50_us": r["fp32"]["p50_us"],
        "speedup_p50": round(r["fp32"]["p50_us"] / r["int8"]["p50_us"], 4),
        "trials": 3,
        **spread,
    }


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for w in WIDTHS:
            row = run_width(w, tmp, reps)
            rows.append(row)
            print(f"width={row['width']:<5} int8={row['int8_weight_bytes']/1024:>8.1f}KiB "
                  f"fp32={row['fp32_weight_bytes']/1024:>8.1f}KiB  "
                  f"int8_p50={row['int8_p50_us']:>9.3f}us fp32_p50={row['fp32_p50_us']:>9.3f}us  "
                  f"speedup={row['speedup_p50']:.3f}x "
                  f"(spread i8 {row['int8_weight_bytes'] and row['int8_p50_spread_pct']}%/"
                  f"f32 {row['fp32_p50_spread_pct']}%)", flush=True)
    pathlib.Path("artifacts/scaling_study.json").write_text(json.dumps(rows, indent=2))
