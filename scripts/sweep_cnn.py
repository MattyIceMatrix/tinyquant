#!/usr/bin/env python3
"""Conv accuracy sweep: weight bit-width x granularity x calibration.

The dense sweep showed per-channel scaling only mattering below 5 bits. The
interesting question for convolutions is whether it matters more, since a conv
kernel's output channels are filters with genuinely different dynamic ranges.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tinyquant.conv import ConvNet  # noqa: E402
from tinyquant.data import load_split  # noqa: E402
from tinyquant.quantize_conv import quantize_convnet  # noqa: E402

BITS = [8, 6, 5, 4, 3, 2]
GRANS = ["per_tensor", "per_channel"]
CALS = ["minmax", "percentile"]

if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_split()
    Xtr_i, Xte_i = Xtr.reshape(-1, 8, 8, 1), Xte.reshape(-1, 8, 8, 1)
    net = ConvNet.load("artifacts/model_cnn_fp32.npz")
    fp32_pred = net.predict(Xte_i)
    fp32_acc = float((fp32_pred == yte).mean())
    fp32_bytes = int(net.n_params() * 4)

    rows = []
    for bits in BITS:
        for gran in GRANS:
            for cal in CALS:
                q = quantize_convnet(net, Xtr_i[:128], gran, cal, weight_bits=bits)
                pred = q.predict(Xte_i)
                acc = float((pred == yte).mean())
                rows.append({
                    "weight_bits": bits, "granularity": gran, "calibration": cal,
                    "test_acc": round(acc, 6),
                    "delta_pp": round((acc - fp32_acc) * 100, 4),
                    "agreement_with_fp32": round(float((pred == fp32_pred).mean()), 6),
                    "deployed_bytes": q.size_bytes(),
                    "compression_x": round(fp32_bytes / q.size_bytes(), 3),
                })
                print(f"bits={bits} {gran:<11} {cal:<11} acc={acc:.4f} "
                      f"delta={rows[-1]['delta_pp']:+.2f}pp", flush=True)

    # One test sample is 1/360 = 0.278pp, so differences smaller than that are
    # not differences. Recorded here so the README cannot overstate them.
    out = {"fp32_test_acc": round(fp32_acc, 6), "fp32_weight_bytes": fp32_bytes,
           "n_test": int(len(Xte_i)),
           "one_sample_pp": round(100.0 / len(Xte_i), 4),
           "rows": rows}
    pathlib.Path("artifacts/sweep_cnn.json").write_text(json.dumps(out, indent=2))
