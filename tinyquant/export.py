"""Serialise models to the flat binary formats the C engine reads.

Both formats are little-endian, fixed-layout, and contain no pointers, offsets
or compression -- the engine maps the file and uses it in place. That is a
deliberate edge-target choice: parsing is a handful of bounds checks rather
than a dependency.

  .tq8  INT8 model     magic TQI1
  .tqf  FP32 reference magic TQF1
"""
from __future__ import annotations

import struct

import numpy as np

from .model import MLP
from .quantize import QuantModel

MAGIC_INT8 = b"TQI1"
MAGIC_FP32 = b"TQF1"


def export_int8(qm: QuantModel, path: str) -> int:
    with open(path, "wb") as f:
        f.write(MAGIC_INT8)
        f.write(struct.pack("<II", len(qm.layers), qm.layers[0].qW.shape[0]))
        f.write(struct.pack("<f", qm.in_scale))
        f.write(struct.pack("<I", qm.weight_bits))
        for l in qm.layers:
            n_in, n_out = l.qW.shape
            f.write(struct.pack("<IIII", n_in, n_out, 1 if l.relu else 0, 0))
            f.write(l.qW.astype(np.int8).tobytes(order="C"))        # (in, out) row-major
            f.write(l.qb.astype("<i4").tobytes())
            f.write(l.q_mult.astype("<i4").tobytes())
            f.write(l.left_shift.astype(np.int8).tobytes())
            f.write(l.right_shift.astype(np.int8).tobytes())
    import os
    return os.path.getsize(path)


def export_fp32(model: MLP, path: str) -> int:
    with open(path, "wb") as f:
        f.write(MAGIC_FP32)
        f.write(struct.pack("<II", model.n_layers, model.W[0].shape[0]))
        for i in range(model.n_layers):
            n_in, n_out = model.W[i].shape
            relu = 1 if i < model.n_layers - 1 else 0
            f.write(struct.pack("<IIII", n_in, n_out, relu, 0))
            f.write(model.W[i].astype("<f4").tobytes(order="C"))
            f.write(model.b[i].astype("<f4").tobytes())
    import os
    return os.path.getsize(path)


def export_test_vectors(x_q: np.ndarray, x_f: np.ndarray, y: np.ndarray, path: str) -> None:
    """Quantized inputs, float inputs and labels, so the C side scores itself."""
    with open(path, "wb") as f:
        f.write(b"TQV1")
        f.write(struct.pack("<II", x_q.shape[0], x_q.shape[1]))
        f.write(x_q.astype(np.int8).tobytes(order="C"))
        f.write(x_f.astype("<f4").tobytes(order="C"))
        f.write(y.astype("<i4").tobytes())


MAGIC_GRAPH = b"TQG1"


def export_graph_int8(qnet, path: str) -> int:
    """Write a conv/dense graph in the flat TQG1 format.

    Layout mirrors the dense format: a fixed header, then per layer a fixed
    header followed by weights, biases and requantization parameters, each
    4-byte aligned so the C side can point directly into the mapped bytes.
    """
    from .conv import KIND_CONV

    with open(path, "wb") as f:
        f.write(MAGIC_GRAPH)
        in_h, in_w, in_c = qnet.in_shape
        f.write(struct.pack("<IIIII", len(qnet.layers), in_h, in_w, in_c, qnet.weight_bits))
        f.write(struct.pack("<f", qnet.in_scale))
        for l in qnet.layers:
            if l.kind == KIND_CONV:
                kh, kw, c_in, c_out = l.qW.shape
            else:
                kh = kw = 1
                c_in, c_out = l.qW.shape
            f.write(struct.pack("<IIIIII", l.kind, 1 if l.relu else 0, kh, kw, c_in, c_out))
            f.write(l.qW.astype(np.int8).tobytes(order="C"))
            f.write(l.qb.astype("<i4").tobytes())
            f.write(l.q_mult.astype("<i4").tobytes())
            f.write(l.left_shift.astype(np.int8).tobytes())
            f.write(l.right_shift.astype(np.int8).tobytes())
    import os
    return os.path.getsize(path)


def export_image_vectors(x_q: np.ndarray, y: np.ndarray, path: str) -> None:
    """Quantized NHWC images plus labels, for the conv cross-validation harness."""
    with open(path, "wb") as f:
        f.write(b"TQW1")
        n, h, w, c = x_q.shape
        f.write(struct.pack("<IIII", n, h, w, c))
        f.write(x_q.astype(np.int8).tobytes(order="C"))
        f.write(y.astype("<i4").tobytes())
