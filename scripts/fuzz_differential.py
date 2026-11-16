#!/usr/bin/env python3
"""Randomised differential testing of the C engine against the Python reference.

The fixed test set exercises one model shape. This generates many -- varying
depth, width, bit-width, granularity, calibration, and the shape of the weight
distribution itself -- and demands byte-identical output every time.

Adversarial cases are included on purpose, because they are where an INT8
pipeline breaks: a single huge outlier weight that dominates a per-tensor
scale, all-zero weights that would produce a zero scale, activations large
enough to threaten the int32 accumulator, and single-neuron layers where
vectorised code tends to read past the end.

Configurations the quantizer legitimately refuses (a bias that cannot be
represented in int32, an accumulator that would overflow) are counted and
reported rather than silently skipped -- a refusal is correct behaviour, and
hiding it would overstate the coverage.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinyquant.export import export_int8, export_test_vectors  # noqa: E402
from tinyquant.model import MLP  # noqa: E402
from tinyquant.quantize import quantize_model  # noqa: E402

MAX_W = 96
SHAPES = ("normal", "outlier", "zeros", "bimodal", "tiny", "huge")


def random_model(rng, shape: str):
    depth = int(rng.integers(1, 5))
    dims = [int(rng.integers(1, MAX_W + 1)) for _ in range(depth)] + [int(rng.integers(2, 11))]
    W, b = [], []
    for i in range(len(dims) - 1):
        n_in, n_out = dims[i], dims[i + 1]
        if shape == "normal":
            w = rng.standard_normal((n_in, n_out))
        elif shape == "outlier":
            w = rng.standard_normal((n_in, n_out)) * 0.01
            w[rng.integers(n_in), rng.integers(n_out)] = 50.0  # one dominating weight
        elif shape == "zeros":
            w = np.zeros((n_in, n_out))
        elif shape == "bimodal":
            w = rng.choice([-3.0, 3.0], size=(n_in, n_out)) + rng.standard_normal((n_in, n_out)) * 0.01
        elif shape == "tiny":
            w = rng.standard_normal((n_in, n_out)) * 1e-4
        else:  # huge
            w = rng.standard_normal((n_in, n_out)) * 1e3
        W.append(w.astype(np.float32))
        b.append((rng.standard_normal(n_out) * 0.1).astype(np.float32))
    return MLP(W, b)


def main() -> int:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261116
    rng = np.random.default_rng(seed)

    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    exe = build / "crossval_fuzz"
    subprocess.run(
        ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
         f"-DTQ_MAX_WIDTH={MAX_W}", "-DTQ_SKIP_ZEROS=0",
         "engine/tinyq.c", "engine/crossval.c", "-o", str(exe)],
        cwd=ROOT, check=True,
    )

    passed = mismatched = rejected = 0
    failures: list[dict] = []
    rejection_reasons: dict[str, int] = {}
    shape_counts: dict[str, int] = {s: 0 for s in SHAPES}

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for t in range(trials):
            shape = SHAPES[int(rng.integers(len(SHAPES)))]
            bits = int(rng.integers(2, 9))
            gran = "per_channel" if rng.random() < 0.5 else "per_tensor"
            cal = "minmax" if rng.random() < 0.5 else "percentile"
            model = random_model(rng, shape)
            n_in = model.W[0].shape[0]

            act_shape = SHAPES[int(rng.integers(len(SHAPES)))]
            scale = {"tiny": 1e-3, "huge": 50.0}.get(act_shape, 1.0)
            x = (np.abs(rng.standard_normal((48, n_in))) * scale).astype(np.float32)

            cfg = {"trial": t, "shape": shape, "bits": bits, "granularity": gran,
                   "calibration": cal, "dims": [int(w.shape[0]) for w in model.W]
                   + [int(model.W[-1].shape[1])], "act_scale": scale}
            try:
                qm = quantize_model(model, x[:32], gran, cal, weight_bits=bits)
                qx = qm.quantize_input(x[32:])
                ref = qm.forward_q(qx)
            except (OverflowError, ValueError) as exc:
                rejected += 1
                key = f"{type(exc).__name__}: {exc}"
                rejection_reasons[key] = rejection_reasons.get(key, 0) + 1
                continue

            tq8, tqv, refb = tmp / "m.tq8", tmp / "v.tqv", tmp / "ref.bin"
            export_int8(qm, str(tq8))
            export_test_vectors(qx, x[32:], np.zeros(len(qx), dtype=np.int64), str(tqv))
            refb.write_bytes(ref.astype(np.int8).tobytes(order="C"))

            proc = subprocess.run([str(exe), str(tq8), str(tqv), str(refb)],
                                  capture_output=True, text=True)
            try:
                rep = json.loads(proc.stdout)
            except json.JSONDecodeError:
                cfg["engine_error"] = (proc.stderr or "no output").strip()[-200:]
                failures.append(cfg)
                mismatched += 1
                continue

            if rep["mismatched_values"] == 0:
                passed += 1
                shape_counts[shape] += 1
            else:
                mismatched += 1
                cfg["mismatched_values"] = rep["mismatched_values"]
                cfg["worst_abs_diff"] = rep["worst_abs_diff"]
                failures.append(cfg)

            if (t + 1) % 25 == 0:
                print(f"  {t + 1}/{trials}  bit-exact={passed} "
                      f"mismatched={mismatched} rejected={rejected}", flush=True)

    report = {
        "trials": trials,
        "seed": seed,
        "bit_exact": passed,
        "mismatched": mismatched,
        "rejected_by_quantizer": rejected,
        "rejection_reasons": rejection_reasons,
        "weight_distributions_covered": shape_counts,
        "failures": failures[:20],
    }
    (ROOT / "artifacts" / "fuzz.json").write_text(json.dumps(report, indent=2))
    print(f"\n{passed} bit-exact, {mismatched} mismatched, "
          f"{rejected} refused by the quantizer, out of {trials} random configurations")
    for reason, count in sorted(rejection_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  refused {count:>4}x  {reason}")
    return 0 if mismatched == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
