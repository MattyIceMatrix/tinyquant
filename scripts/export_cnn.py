#!/usr/bin/env python3
"""Quantize the CNN and write what the conv engine needs, plus the reference."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tinyquant.conv import ConvNet  # noqa: E402
from tinyquant.data import load_split  # noqa: E402
from tinyquant.export import export_graph_int8, export_image_vectors  # noqa: E402
from tinyquant.quantize_conv import quantize_convnet  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/model_cnn_fp32.npz")
    ap.add_argument("--granularity", default="per_channel", choices=["per_channel", "per_tensor"])
    ap.add_argument("--calibration", default="minmax", choices=["minmax", "percentile"])
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    Xtr, ytr, Xte, yte = load_split()
    Xtr_i, Xte_i = Xtr.reshape(-1, 8, 8, 1), Xte.reshape(-1, 8, 8, 1)

    net = ConvNet.load(args.model)
    qnet = quantize_convnet(net, Xtr_i[:128], args.granularity, args.calibration,
                            weight_bits=args.bits)

    qx = qnet.quantize_input(Xte_i)
    ref = qnet.forward_q(qx)

    size = export_graph_int8(qnet, str(out / "model_cnn.tqg"))
    export_image_vectors(qx, yte, str(out / "test_images.tqw"))
    (out / "reference_cnn_out.bin").write_bytes(ref.astype(np.int8).tobytes(order="C"))

    fp32_acc = float((net.predict(Xte_i) == yte).mean())
    int8_acc = float((ref.argmax(axis=1) == yte).mean())
    meta = {
        "granularity": qnet.granularity, "calibration": qnet.calibration,
        "weight_bits": qnet.weight_bits,
        "fp32_test_acc": fp32_acc, "int8_test_acc": int8_acc,
        "accuracy_delta_pp": round((int8_acc - fp32_acc) * 100, 4),
        "fp32_weight_bytes": int(net.n_params() * 4),
        "int8_deployed_bytes": qnet.size_bytes(),
        "compression_x": round(net.n_params() * 4 / qnet.size_bytes(), 3),
        "file_bytes": size,
        "layers": [{"kind": "conv" if l.kind == 0 else "dense",
                    "shape": list(l.qW.shape), "relu": l.relu} for l in qnet.layers],
    }
    (out / "export_cnn_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
