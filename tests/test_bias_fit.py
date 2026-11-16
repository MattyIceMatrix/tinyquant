"""The bias-overflow fix that randomized testing forced.

Biases are stored in accumulator units (s_x * s_w). When weights are tiny and
activations large, that scale gets small enough that an ordinary bias needs
more than 31 bits. The quantizer used to refuse; it now coarsens the weight
scale just enough to make the bias representable.
"""
from __future__ import annotations

import numpy as np
import pytest

from tinyquant.data import load_split
from tinyquant.model import MLP
from tinyquant.quantize import quantize_model


def pathological_model():
    """Tiny weights, ordinary biases -- the shape the fuzzer kept producing."""
    rng = np.random.default_rng(7)
    # 1e-9 weights against ~50-magnitude activations put the bias scale
    # (s_x * s_w) low enough that an ordinary bias exceeds int32.
    W = [(rng.standard_normal((16, 12)) * 1e-9).astype(np.float32),
         (rng.standard_normal((12, 4)) * 1e-9).astype(np.float32)]
    b = [(rng.standard_normal(12)).astype(np.float32),
         (rng.standard_normal(4)).astype(np.float32)]
    return MLP(W, b)


def big_activations():
    rng = np.random.default_rng(8)
    return (np.abs(rng.standard_normal((32, 16))) * 50.0).astype(np.float32)


def test_strict_mode_still_refuses_rather_than_wrapping():
    with pytest.raises(OverflowError):
        quantize_model(pathological_model(), big_activations(),
                       "per_channel", "minmax", bias_fit="strict")


def test_rescale_mode_produces_a_usable_model():
    qm = quantize_model(pathological_model(), big_activations(),
                        "per_channel", "minmax", bias_fit="rescale")
    assert qm.rescaled_channels > 0
    out = qm.forward_q(qm.quantize_input(big_activations()))
    assert out.shape == (32, 4)


def test_rescaling_keeps_every_bias_inside_int32():
    qm = quantize_model(pathological_model(), big_activations(),
                        "per_channel", "minmax", bias_fit="rescale")
    for layer in qm.layers:
        assert layer.qb.dtype == np.int32
        assert int(layer.qb.min()) >= -(2**31)
        assert int(layer.qb.max()) <= 2**31 - 1


def test_per_tensor_rescaling_keeps_one_shared_scale():
    """Rescaling a single channel would quietly turn per-tensor into per-channel."""
    qm = quantize_model(pathological_model(), big_activations(),
                        "per_tensor", "minmax", bias_fit="rescale")
    for layer in qm.layers:
        assert len(set(np.round(layer.w_scale, 18).tolist())) == 1


def test_the_real_model_needs_no_rescaling_at_all():
    Xtr, _, Xte, yte = load_split()
    model = MLP.load("artifacts/model_fp32.npz")
    qm = quantize_model(model, Xtr[:256], "per_channel", "minmax")
    assert qm.rescaled_channels == 0
    assert (qm.predict(Xte) == yte).mean() == (model.predict(Xte) == yte).mean()


def test_unknown_bias_fit_is_rejected():
    with pytest.raises(ValueError):
        quantize_model(pathological_model(), big_activations(), bias_fit="guess")
