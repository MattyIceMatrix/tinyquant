#!/usr/bin/env python3
"""Accuracy sweep: weight bit-width x weight granularity x calibration method.

Every cell is a full quantize-and-evaluate on the held-out test set. Nothing
here is interpolated or estimated.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tinyquant.data import load_split  # noqa: E402
from tinyquant.model import MLP  # noqa: E402
from tinyquant.quantize import quantize_model  # noqa: E402

BITS = [8, 6, 5, 4, 3, 2]
GRANS = ["per_tensor", "per_channel"]
CALS = ["minmax", "percentile"]

if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_split()
    model = MLP.load("artifacts/model_fp32.npz")
    fp32_pred = model.predict(Xte)
    fp32_acc = float((fp32_pred == yte).mean())
    fp32_bytes = int(model.n_params() * 4)

    rows = []
    for bits in BITS:
        for gran in GRANS:
            for cal in CALS:
                qm = quantize_model(model, Xtr[:256], gran, cal, weight_bits=bits)
                pred = qm.predict(Xte)
                acc = float((pred == yte).mean())
                rows.append({
                    "weight_bits": bits,
                    "granularity": gran,
                    "calibration": cal,
                    "test_acc": round(acc, 6),
                    "delta_pp": round((acc - fp32_acc) * 100, 4),
                    "agreement_with_fp32": round(float((pred == fp32_pred).mean()), 6),
                    "deployed_bytes": qm.size_bytes(),
                    "compression_x": round(fp32_bytes / qm.size_bytes(), 3),
                })
                print(f"bits={bits} {rows[-1]['granularity']:<11} {rows[-1]['calibration']:<11} "
                      f"acc={acc:.4f} delta={rows[-1]['delta_pp']:+.2f}pp "
                      f"size={rows[-1]['deployed_bytes']}B ({rows[-1]['compression_x']}x)", flush=True)

    out = {"fp32_test_acc": round(fp32_acc, 6), "fp32_weight_bytes": fp32_bytes,
           "n_test": int(len(Xte)), "rows": rows}
    pathlib.Path("artifacts/sweep.json").write_text(json.dumps(out, indent=2))
