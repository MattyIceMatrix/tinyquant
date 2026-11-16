"""Post-training INT8 quantization, implemented from scratch.

Scheme: symmetric signed INT8 for both weights and activations, zero-point 0.
  real ~= scale * q,  q in [-127, 127]

Two weight granularities are supported and both are benchmarked:
  per-tensor    one scale for the whole weight matrix
  per-channel   one scale per output neuron

Per-channel exists because a single outlier column otherwise sets the scale for
every column, and the quiet columns lose most of their resolution.

Activation ranges come from calibration -- running FP32 inference over unlabeled
samples and recording how big each tensor actually gets. Two estimators:
  minmax       absolute max seen; safe, but one outlier batch widens the range
  percentile   clip at a high percentile; narrower scale, at the cost of
               saturating the rare large value

Deliberate limitation, stated plainly: activations use zero-point 0, so a
post-ReLU tensor can only ever use [0, 127] -- one of the eight bits is spent
representing negatives that ReLU already removed. Asymmetric activation
quantization would recover it. Per-channel weights recover most of the accuracy
in practice on this model, which is why both are measured instead of assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fixedpoint import INT32_MAX, INT32_MIN, quantize_multiplier, requantize
from .model import MLP

QMAX = 127  # activation range: symmetric int8


def weight_qmax(bits: int) -> int:
    """Largest magnitude a symmetric signed weight of `bits` bits can hold."""
    if not 2 <= bits <= 8:
        raise ValueError("weight bits must be in 2..8")
    return (1 << (bits - 1)) - 1


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def collect_activation_ranges(
    model: MLP, x_calib: np.ndarray, method: str = "minmax", percentile: float = 99.9
) -> list[float]:
    """Return one absolute-max estimate per tensor: input, then each layer output."""
    _, acts = model.forward(x_calib, collect=True)
    ranges = []
    for a in acts:
        mag = np.abs(a)
        if method == "minmax":
            r = float(mag.max())
        elif method == "percentile":
            r = float(np.percentile(mag, percentile))
        else:
            raise ValueError(f"unknown calibration method: {method}")
        ranges.append(max(r, 1e-8))  # never emit a zero scale
    return ranges


# --------------------------------------------------------------------------- #
# quantized model
# --------------------------------------------------------------------------- #
@dataclass
class QuantLayer:
    qW: np.ndarray            # int8, (in, out)
    qb: np.ndarray            # int32, (out,) -- scaled by s_x * s_w
    q_mult: np.ndarray        # int32, (out,) Q0.31 requant multipliers
    left_shift: np.ndarray    # int32, (out,)
    right_shift: np.ndarray   # int32, (out,)
    relu: bool
    w_scale: np.ndarray       # float, (out,) kept for reporting only
    out_scale: float


@dataclass
class QuantModel:
    layers: list[QuantLayer]
    in_scale: float
    granularity: str
    calibration: str
    weight_bits: int = 8
    act_ranges: list[float] = field(default_factory=list)

    # ---- quantize / dequantize helpers -----------------------------------
    def quantize_input(self, x: np.ndarray) -> np.ndarray:
        q = np.rint(x / self.in_scale)
        return np.clip(q, -QMAX, QMAX).astype(np.int8)

    # ---- integer-only forward pass ---------------------------------------
    def forward_q(self, qx: np.ndarray) -> np.ndarray:
        """Integer-only inference on int8 input. Returns int8 logits.

        Accumulation is int64 in numpy for convenience but asserted to stay
        inside int32 -- the C engine uses a real int32 accumulator, and this
        assert is what guarantees the two agree.
        """
        h = qx.astype(np.int64)
        for layer in self.layers:
            acc = h @ layer.qW.astype(np.int64) + layer.qb.astype(np.int64)
            if acc.min() < INT32_MIN or acc.max() > INT32_MAX:
                raise OverflowError("int32 accumulator overflow")
            out = np.empty_like(acc)
            flat = acc.reshape(-1, acc.shape[-1])
            o = out.reshape(-1, acc.shape[-1])
            for j in range(acc.shape[-1]):
                qm = int(layer.q_mult[j])
                ls = int(layer.left_shift[j])
                rs = int(layer.right_shift[j])
                col = flat[:, j]
                o[:, j] = [requantize(int(v), qm, ls, rs) for v in col]
            lo = 0 if layer.relu else -128
            h = np.clip(out, lo, QMAX)
        return h.astype(np.int8)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward_q(self.quantize_input(x)).argmax(axis=1)

    def size_bytes(self) -> int:
        """Deployed size: bit-packed weights + int32 biases + requant parameters.

        Weights are counted at their real bit-width rather than the int8
        container they are held in here, since a deployed artifact would pack
        them. Biases and requant parameters are counted in full -- they are
        small but they are not free, and portfolio-grade numbers should not
        quietly omit them.
        """
        total = 0
        for l in self.layers:
            total += (l.qW.size * self.weight_bits + 7) // 8
            total += l.qb.size * 4
            total += l.q_mult.size * 4 + l.left_shift.size * 1 + l.right_shift.size * 1
        return total


def quantize_model(
    model: MLP,
    x_calib: np.ndarray,
    granularity: str = "per_channel",
    calibration: str = "minmax",
    percentile: float = 99.9,
    weight_bits: int = 8,
) -> QuantModel:
    act_ranges = collect_activation_ranges(model, x_calib, calibration, percentile)
    act_scales = [r / QMAX for r in act_ranges]
    wmax = weight_qmax(weight_bits)

    layers: list[QuantLayer] = []
    for i in range(model.n_layers):
        W, b = model.W[i], model.b[i]
        n_out = W.shape[1]

        if granularity == "per_channel":
            w_absmax = np.maximum(np.abs(W).max(axis=0), 1e-8)
        elif granularity == "per_tensor":
            w_absmax = np.full(n_out, max(float(np.abs(W).max()), 1e-8))
        else:
            raise ValueError(f"unknown granularity: {granularity}")

        w_scale = w_absmax / wmax
        qW = np.clip(np.rint(W / w_scale[None, :]), -wmax, wmax).astype(np.int8)

        s_x = act_scales[i]
        s_y = act_scales[i + 1]
        bias_scale = s_x * w_scale                     # bias lives in accumulator units
        qb = np.rint(b / bias_scale).astype(np.int64)
        if qb.min() < INT32_MIN or qb.max() > INT32_MAX:
            raise OverflowError("bias does not fit in int32")
        qb = qb.astype(np.int32)

        mults = [quantize_multiplier(float(s_x * w_scale[j] / s_y)) for j in range(n_out)]
        q_mult = np.array([m[0] for m in mults], dtype=np.int32)
        left_shift = np.array([m[1] for m in mults], dtype=np.int32)
        right_shift = np.array([m[2] for m in mults], dtype=np.int32)

        layers.append(
            QuantLayer(
                qW=qW, qb=qb, q_mult=q_mult, left_shift=left_shift,
                right_shift=right_shift, relu=(i < model.n_layers - 1),
                w_scale=w_scale, out_scale=s_y,
            )
        )

    return QuantModel(layers=layers, in_scale=act_scales[0], granularity=granularity,
                      calibration=calibration, act_ranges=act_ranges,
                      weight_bits=weight_bits)
