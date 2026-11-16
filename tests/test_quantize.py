"""Properties the quantizer must hold, independent of the accuracy numbers."""
import numpy as np
import pytest

from tinyquant.data import load_split
from tinyquant.model import MLP
from tinyquant.quantize import quantize_model, weight_qmax

MODEL = "artifacts/model_fp32.npz"


@pytest.fixture(scope="module")
def data():
    return load_split()


@pytest.fixture(scope="module")
def model():
    return MLP.load(MODEL)


def test_weight_qmax_bounds():
    assert weight_qmax(8) == 127
    assert weight_qmax(4) == 7
    assert weight_qmax(2) == 1
    with pytest.raises(ValueError):
        weight_qmax(9)


def test_int8_is_lossless_on_this_model(model, data):
    Xtr, _, Xte, yte = data
    qm = quantize_model(model, Xtr[:256], "per_channel", "minmax")
    fp32_acc = (model.predict(Xte) == yte).mean()
    int8_acc = (qm.predict(Xte) == yte).mean()
    assert int8_acc >= fp32_acc - 1e-12


def test_quantized_weights_stay_in_range(model, data):
    Xtr = data[0]
    for bits in (8, 4, 2):
        qm = quantize_model(model, Xtr[:256], "per_channel", "minmax", weight_bits=bits)
        lim = weight_qmax(bits)
        for layer in qm.layers:
            assert layer.qW.min() >= -lim and layer.qW.max() <= lim


def test_accumulator_never_overflows_int32(model, data):
    """forward_q raises on overflow; running the whole test set proves it doesn't."""
    Xtr, _, Xte, _ = data
    qm = quantize_model(model, Xtr[:256], "per_channel", "minmax")
    qm.forward_q(qm.quantize_input(Xte))  # raises OverflowError if it does


def test_per_channel_is_never_worse_than_per_tensor(model, data):
    Xtr, _, Xte, yte = data
    for bits in (8, 5, 4, 2):
        accs = {}
        for gran in ("per_tensor", "per_channel"):
            qm = quantize_model(model, Xtr[:256], gran, "minmax", weight_bits=bits)
            accs[gran] = (qm.predict(Xte) == yte).mean()
        assert accs["per_channel"] >= accs["per_tensor"] - 1e-12, (bits, accs)


def test_deployed_size_shrinks_with_bit_width(model, data):
    Xtr = data[0]
    sizes = [quantize_model(model, Xtr[:256], "per_channel", "minmax", weight_bits=b).size_bytes()
             for b in (8, 6, 4, 2)]
    assert sizes == sorted(sizes, reverse=True)


def test_calibration_methods_both_produce_valid_scales(model, data):
    Xtr = data[0]
    for cal in ("minmax", "percentile"):
        qm = quantize_model(model, Xtr[:256], "per_channel", cal)
        assert qm.in_scale > 0
        assert all(r > 0 for r in qm.act_ranges)
    with pytest.raises(ValueError):
        quantize_model(model, Xtr[:256], "per_channel", "nonsense")
