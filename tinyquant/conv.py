"""A small convolutional network, in NumPy only.

The MLP in model.py proves the quantization pipeline works; it does not prove
much about convolutions, where per-channel scaling and memory layout matter
considerably more. This module adds a real conv stack -- including the backward
pass -- so the quantizer and the C engine can be tested against something whose
weights are shared across spatial positions.

Layout is NHWC throughout: activations are (H, W, C) and weights are
(KH, KW, C_in, C_out). Channels-last means the innermost loop in the C kernel
walks contiguous memory over output channels, which is the same reason the
dense kernel stores weights (in, out).

Padding is 'valid' and stride is 1. Both are deliberate: they keep the index
arithmetic in the C kernel small enough to read, and nothing in this repo's
argument depends on supporting more.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KIND_CONV = 0
KIND_DENSE = 1


def im2col(x: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """(N, H, W, C) -> (N, OH, OW, kh*kw*C) with 'valid' windows, stride 1."""
    n, h, w, c = x.shape
    oh, ow = h - kh + 1, w - kw + 1
    cols = np.empty((n, oh, ow, kh * kw * c), dtype=x.dtype)
    idx = 0
    for ky in range(kh):
        for kx in range(kw):
            cols[..., idx:idx + c] = x[:, ky:ky + oh, kx:kx + ow, :]
            idx += c
    return cols


def col2im(cols: np.ndarray, shape: tuple, kh: int, kw: int) -> np.ndarray:
    """Adjoint of im2col: scatter-add gradients back to input positions."""
    n, h, w, c = shape
    oh, ow = h - kh + 1, w - kw + 1
    out = np.zeros(shape, dtype=cols.dtype)
    idx = 0
    for ky in range(kh):
        for kx in range(kw):
            out[:, ky:ky + oh, kx:kx + ow, :] += cols[..., idx:idx + c]
            idx += c
    return out


@dataclass
class ConvLayer:
    W: np.ndarray   # (kh, kw, c_in, c_out) float32
    b: np.ndarray   # (c_out,) float32
    relu: bool = True
    kind: int = KIND_CONV

    def forward(self, x: np.ndarray) -> np.ndarray:
        kh, kw, c_in, c_out = self.W.shape
        cols = im2col(x, kh, kw)
        out = cols.reshape(-1, kh * kw * c_in) @ self.W.reshape(-1, c_out) + self.b
        out = out.reshape(x.shape[0], x.shape[1] - kh + 1, x.shape[2] - kw + 1, c_out)
        return np.maximum(out, 0.0) if self.relu else out


@dataclass
class DenseLayer:
    W: np.ndarray   # (in, out) float32
    b: np.ndarray   # (out,) float32
    relu: bool = False
    kind: int = KIND_DENSE

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x.reshape(x.shape[0], -1) @ self.W + self.b
        return np.maximum(out, 0.0) if self.relu else out


@dataclass
class ConvNet:
    """conv -> relu -> conv -> relu -> flatten -> dense(logits)."""
    layers: list

    def forward(self, x: np.ndarray, collect: bool = False):
        acts = [x]
        h = x
        for layer in self.layers:
            h = layer.forward(h)
            acts.append(h)
        return (h, acts) if collect else h

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).argmax(axis=1)

    def n_params(self) -> int:
        return sum(l.W.size + l.b.size for l in self.layers)

    def save(self, path: str) -> None:
        flat = {}
        for i, l in enumerate(self.layers):
            flat[f"W{i}"] = l.W
            flat[f"b{i}"] = l.b
            flat[f"kind{i}"] = np.array([l.kind])
            flat[f"relu{i}"] = np.array([1 if l.relu else 0])
        np.savez(path, **flat)

    @staticmethod
    def load(path: str) -> "ConvNet":
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("W")])
        layers = []
        for i in range(n):
            kind = int(z[f"kind{i}"][0])
            relu = bool(z[f"relu{i}"][0])
            cls = ConvLayer if kind == KIND_CONV else DenseLayer
            layers.append(cls(z[f"W{i}"], z[f"b{i}"], relu))
        return ConvNet(layers)
