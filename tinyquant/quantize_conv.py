"""INT8 quantization for the convolutional network.

The arithmetic is identical to the dense case -- symmetric INT8, int32
accumulation, Q0.31 fixed-point requantization -- which is the point: the
scheme should not need redesigning to cross from matrix multiplies to
convolutions. What changes is only which axis counts as an output channel.

For a conv weight of shape (kh, kw, c_in, c_out), per-channel scaling means one
scale per c_out, computed over the kh*kw*c_in values that feed it. Per-tensor
means one scale for the whole kernel stack. Both are supported and both are
measured, as in the dense path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conv import KIND_CONV, KIND_DENSE, ConvNet, im2col
from .fixedpoint import INT32_MAX, INT32_MIN, quantize_multiplier, requantize
from .quantize import QMAX, weight_qmax


@dataclass
class QGraphLayer:
    kind: int
    qW: np.ndarray            # int8; (kh,kw,cin,cout) for conv, (in,out) for dense
    qb: np.ndarray            # int32 (cout,)
    q_mult: np.ndarray        # int32 (cout,)
    left_shift: np.ndarray    # int32 (cout,)
    right_shift: np.ndarray   # int32 (cout,)
    relu: bool
    w_scale: np.ndarray
    out_scale: float


@dataclass
class QuantConvNet:
    layers: list[QGraphLayer]
    in_scale: float
    in_shape: tuple
    granularity: str
    calibration: str
    weight_bits: int = 8
    act_ranges: list[float] = field(default_factory=list)

    def quantize_input(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(x / self.in_scale), -QMAX, QMAX).astype(np.int8)

    def forward_q(self, qx: np.ndarray) -> np.ndarray:
        """Integer-only forward pass. qx is (N, H, W, C) int8."""
        h = qx.astype(np.int64)
        for layer in self.layers:
            if layer.kind == KIND_CONV:
                kh, kw, c_in, c_out = layer.qW.shape
                cols = im2col(h, kh, kw)
                n, oh, ow, _ = cols.shape
                acc = (cols.reshape(-1, kh * kw * c_in)
                       @ layer.qW.reshape(-1, c_out).astype(np.int64)) + layer.qb.astype(np.int64)
                out_shape = (n, oh, ow, c_out)
            else:
                c_out = layer.qW.shape[1]
                acc = (h.reshape(h.shape[0], -1)
                       @ layer.qW.astype(np.int64)) + layer.qb.astype(np.int64)
                out_shape = (h.shape[0], c_out)

            if acc.min() < INT32_MIN or acc.max() > INT32_MAX:
                raise OverflowError("int32 accumulator overflow")

            flat = acc.reshape(-1, c_out)
            out = np.empty_like(flat)
            for j in range(c_out):
                qm, ls, rs = int(layer.q_mult[j]), int(layer.left_shift[j]), int(layer.right_shift[j])
                out[:, j] = [requantize(int(v), qm, ls, rs) for v in flat[:, j]]

            lo = 0 if layer.relu else -128
            h = np.clip(out, lo, QMAX).reshape(out_shape)
        return h.astype(np.int8)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward_q(self.quantize_input(x)).argmax(axis=1)

    def size_bytes(self) -> int:
        total = 0
        for l in self.layers:
            total += (l.qW.size * self.weight_bits + 7) // 8
            total += l.qb.size * 4
            total += l.q_mult.size * 4 + l.left_shift.size * 1 + l.right_shift.size * 1
        return total


def quantize_convnet(net: ConvNet, x_calib: np.ndarray, granularity: str = "per_channel",
                     calibration: str = "minmax", percentile: float = 99.9,
                     weight_bits: int = 8) -> QuantConvNet:
    _, acts = net.forward(x_calib, collect=True)
    ranges = []
    for a in acts:
        mag = np.abs(a)
        r = float(mag.max()) if calibration == "minmax" else float(np.percentile(mag, percentile))
        ranges.append(max(r, 1e-8))
    act_scales = [r / QMAX for r in ranges]
    wmax = weight_qmax(weight_bits)

    qlayers = []
    for i, layer in enumerate(net.layers):
        W, b = layer.W, layer.b
        c_out = W.shape[-1]
        flat_w = W.reshape(-1, c_out)

        if granularity == "per_channel":
            w_absmax = np.maximum(np.abs(flat_w).max(axis=0), 1e-8)
        elif granularity == "per_tensor":
            w_absmax = np.full(c_out, max(float(np.abs(W).max()), 1e-8))
        else:
            raise ValueError(f"unknown granularity: {granularity}")

        w_scale = w_absmax / wmax
        qW = np.clip(np.rint(flat_w / w_scale[None, :]), -wmax, wmax).astype(np.int8).reshape(W.shape)

        s_x, s_y = act_scales[i], act_scales[i + 1]
        qb = np.rint(b / (s_x * w_scale)).astype(np.int64)
        if qb.min() < INT32_MIN or qb.max() > INT32_MAX:
            raise OverflowError("bias does not fit in int32")

        mults = [quantize_multiplier(float(s_x * w_scale[j] / s_y)) for j in range(c_out)]
        qlayers.append(QGraphLayer(
            kind=layer.kind, qW=qW, qb=qb.astype(np.int32),
            q_mult=np.array([m[0] for m in mults], dtype=np.int32),
            left_shift=np.array([m[1] for m in mults], dtype=np.int32),
            right_shift=np.array([m[2] for m in mults], dtype=np.int32),
            relu=layer.relu, w_scale=w_scale, out_scale=s_y,
        ))

    return QuantConvNet(layers=qlayers, in_scale=act_scales[0],
                        in_shape=tuple(x_calib.shape[1:]), granularity=granularity,
                        calibration=calibration, weight_bits=weight_bits, act_ranges=ranges)
