"""A small MLP written in NumPy only.

Why NumPy and not PyTorch: this repo is about what happens *after* training --
quantization and the inference kernel. Owning the forward pass end to end means
the Python reference and the C engine can be compared line for line, with no
framework behaviour to explain away.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LAYERS = (64, 64, 32, 10)  # input, hidden, hidden, classes


@dataclass
class MLP:
    W: list[np.ndarray]  # each (in, out), float32
    b: list[np.ndarray]  # each (out,), float32

    @property
    def n_layers(self) -> int:
        return len(self.W)

    def forward(self, x: np.ndarray, collect: bool = False):
        """FP32 forward pass. ReLU on every layer but the last."""
        acts = [x]
        h = x
        for i in range(self.n_layers):
            h = h @ self.W[i] + self.b[i]
            if i < self.n_layers - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return (h, acts) if collect else h

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).argmax(axis=1)

    def n_params(self) -> int:
        return sum(w.size for w in self.W) + sum(bb.size for bb in self.b)

    def save(self, path: str) -> None:
        flat = {}
        for i, (w, bb) in enumerate(zip(self.W, self.b)):
            flat[f"W{i}"] = w
            flat[f"b{i}"] = bb
        np.savez(path, **flat)

    @staticmethod
    def load(path: str) -> "MLP":
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("W")])
        return MLP([z[f"W{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)])
