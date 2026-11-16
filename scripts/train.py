#!/usr/bin/env python3
"""Train the FP32 reference model. Deterministic: same seed, same weights."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tinyquant.data import SEED, load_split  # noqa: E402
from tinyquant.model import LAYERS, MLP  # noqa: E402


def one_hot(y: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(y), n), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(epochs: int = 300, lr: float = 0.08, batch: int = 64, seed: int = SEED):
    Xtr, ytr, Xte, yte = load_split(seed)
    rng = np.random.default_rng(seed)

    # He initialisation, since every hidden layer is ReLU.
    W = [
        (rng.standard_normal((LAYERS[i], LAYERS[i + 1])) * np.sqrt(2.0 / LAYERS[i])).astype(np.float32)
        for i in range(len(LAYERS) - 1)
    ]
    b = [np.zeros(LAYERS[i + 1], dtype=np.float32) for i in range(len(LAYERS) - 1)]
    vW = [np.zeros_like(w) for w in W]
    vb = [np.zeros_like(x) for x in b]
    model = MLP(W, b)

    Ytr = one_hot(ytr, LAYERS[-1])
    n = len(Xtr)
    momentum = 0.9

    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            sel = order[s : s + batch]
            xb, yb = Xtr[sel], Ytr[sel]
            _, acts = model.forward(xb, collect=True)

            # Softmax + cross-entropy gradient at the logits.
            g = (softmax(acts[-1]) - yb) / len(sel)
            for i in reversed(range(model.n_layers)):
                gW = acts[i].T @ g
                gb = g.sum(axis=0)
                if i > 0:
                    g = (g @ model.W[i].T) * (acts[i] > 0)
                vW[i] = momentum * vW[i] - lr * gW
                vb[i] = momentum * vb[i] - lr * gb
                model.W[i] += vW[i]
                model.b[i] += vb[i]

    acc_tr = float((model.predict(Xtr) == ytr).mean())
    acc_te = float((model.predict(Xte) == yte).mean())
    return model, {"train_acc": acc_tr, "test_acc": acc_te, "epochs": epochs,
                   "lr": lr, "batch": batch, "seed": seed,
                   "params": int(model.n_params()),
                   "n_train": int(len(Xtr)), "n_test": int(len(Xte))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", default="artifacts/model_fp32.npz")
    args = ap.parse_args()

    model, meta = train(epochs=args.epochs)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    pathlib.Path("artifacts/train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
