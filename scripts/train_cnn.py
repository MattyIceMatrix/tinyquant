#!/usr/bin/env python3
"""Train the convolutional reference model (NumPy only, deterministic)."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tinyquant.conv import ConvLayer, ConvNet, DenseLayer, col2im, im2col  # noqa: E402
from tinyquant.data import SEED, load_split  # noqa: E402

KH = KW = 3
C1, C2 = 8, 16
N_CLASSES = 10


def to_images(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, 8, 8, 1)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y, n):
    out = np.zeros((len(y), n), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def train(epochs=120, lr=0.05, batch=64, seed=SEED):
    Xtr, ytr, Xte, yte = load_split(seed)
    Xtr_i, Xte_i = to_images(Xtr), to_images(Xte)
    rng = np.random.default_rng(seed)

    def he(shape, fan_in):
        return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)

    W1 = he((KH, KW, 1, C1), KH * KW * 1)
    W2 = he((KH, KW, C1, C2), KH * KW * C1)
    W3 = he((4 * 4 * C2, N_CLASSES), 4 * 4 * C2)
    b1 = np.zeros(C1, dtype=np.float32)
    b2 = np.zeros(C2, dtype=np.float32)
    b3 = np.zeros(N_CLASSES, dtype=np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    vel = [np.zeros_like(p) for p in params]
    momentum = 0.9
    Ytr = one_hot(ytr, N_CLASSES)

    for _ in range(epochs):
        order = rng.permutation(len(Xtr_i))
        for s in range(0, len(order), batch):
            sel = order[s:s + batch]
            xb, yb = Xtr_i[sel], Ytr[sel]
            n = len(sel)

            # ---- forward, keeping the im2col buffers for the backward pass
            cols1 = im2col(xb, KH, KW)
            z1 = cols1.reshape(-1, KH * KW * 1) @ W1.reshape(-1, C1) + b1
            z1 = z1.reshape(n, 6, 6, C1)
            a1 = np.maximum(z1, 0.0)

            cols2 = im2col(a1, KH, KW)
            z2 = cols2.reshape(-1, KH * KW * C1) @ W2.reshape(-1, C2) + b2
            z2 = z2.reshape(n, 4, 4, C2)
            a2 = np.maximum(z2, 0.0)

            flat = a2.reshape(n, -1)
            logits = flat @ W3 + b3

            # ---- backward
            g = (softmax(logits) - yb) / n
            gW3 = flat.T @ g
            gb3 = g.sum(axis=0)

            ga2 = (g @ W3.T).reshape(a2.shape) * (z2 > 0)
            ga2_m = ga2.reshape(-1, C2)
            gW2 = (cols2.reshape(-1, KH * KW * C1).T @ ga2_m).reshape(W2.shape)
            gb2 = ga2_m.sum(axis=0)
            gcols2 = (ga2_m @ W2.reshape(-1, C2).T).reshape(cols2.shape)
            ga1 = col2im(gcols2, a1.shape, KH, KW) * (z1 > 0)

            ga1_m = ga1.reshape(-1, C1)
            gW1 = (cols1.reshape(-1, KH * KW * 1).T @ ga1_m).reshape(W1.shape)
            gb1 = ga1_m.sum(axis=0)

            for p, v, gr in zip(params, vel, [gW1, gb1, gW2, gb2, gW3, gb3]):
                v *= momentum
                v -= lr * gr
                p += v

    net = ConvNet([
        ConvLayer(W1, b1, relu=True),
        ConvLayer(W2, b2, relu=True),
        DenseLayer(W3, b3, relu=False),
    ])
    acc_tr = float((net.predict(Xtr_i) == ytr).mean())
    acc_te = float((net.predict(Xte_i) == yte).mean())
    meta = {"train_acc": acc_tr, "test_acc": acc_te, "epochs": epochs, "lr": lr,
            "batch": batch, "seed": seed, "params": int(net.n_params()),
            "architecture": f"conv3x3x{C1} -> conv3x3x{C2} -> dense{4*4*C2}x{N_CLASSES}"}
    return net, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default="artifacts/model_cnn_fp32.npz")
    args = ap.parse_args()
    net, meta = train(epochs=args.epochs)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    net.save(args.out)
    pathlib.Path("artifacts/train_cnn_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
