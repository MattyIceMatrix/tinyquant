#!/usr/bin/env python3
"""Quantize the trained model and write everything the C engine needs.

Also writes reference_out.bin: the exact int8 output vector the Python
implementation produces for every test sample. engine/crossval.c must
reproduce those bytes exactly or the build fails.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tinyquant.data import load_split  # noqa: E402
from tinyquant.export import export_fp32, export_int8, export_test_vectors  # noqa: E402
from tinyquant.model import MLP  # noqa: E402
from tinyquant.quantize import quantize_model  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/model_fp32.npz")
    ap.add_argument("--granularity", default="per_channel", choices=["per_channel", "per_tensor"])
    ap.add_argument("--calibration", default="minmax", choices=["minmax", "percentile"])
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--calib-size", type=int, default=256)
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xte, yte = load_split()
    model = MLP.load(args.model)
    qm = quantize_model(model, Xtr[: args.calib_size], args.granularity,
                        args.calibration, weight_bits=args.bits)

    qx_test = qm.quantize_input(Xte)
    ref_out = qm.forward_q(qx_test)  # (n_test, 10) int8

    size_i8 = export_int8(qm, str(out / "model.tq8"))
    size_f32 = export_fp32(model, str(out / "model.tqf"))
    export_test_vectors(qx_test, Xte, yte, str(out / "test_vectors.tqv"))
    (out / "reference_out.bin").write_bytes(ref_out.astype(np.int8).tobytes(order="C"))

    fp32_acc = float((model.predict(Xte) == yte).mean())
    int8_acc = float((ref_out.argmax(axis=1) == yte).mean())
    meta = {
        "granularity": qm.granularity,
        "calibration": qm.calibration,
        "weight_bits": qm.weight_bits,
        "calib_size": args.calib_size,
        "fp32_test_acc": fp32_acc,
        "int8_test_acc": int8_acc,
        "accuracy_delta_pp": round((int8_acc - fp32_acc) * 100, 4),
        "fp32_weight_bytes": int(model.n_params() * 4),
        "int8_deployed_bytes": qm.size_bytes(),
        "compression_x": round(model.n_params() * 4 / qm.size_bytes(), 3),
        "file_bytes": {"model.tq8": size_i8, "model.tqf": size_f32},
        "activation_ranges": [round(r, 5) for r in qm.act_ranges],
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
