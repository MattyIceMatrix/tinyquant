"""Convolution path: im2col correctness, quantizer properties, C engine parity."""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from tinyquant.conv import ConvNet, col2im, im2col
from tinyquant.data import load_split
from tinyquant.quantize_conv import quantize_convnet

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFLAGS = ["-std=c99", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-DTQ_SKIP_ZEROS=0"]
SRC = ["engine/tinyq.c", "engine/tinyq_graph.c", "engine/crossval_graph.c"]


@pytest.fixture(scope="module")
def data():
    Xtr, ytr, Xte, yte = load_split()
    return Xtr.reshape(-1, 8, 8, 1), ytr, Xte.reshape(-1, 8, 8, 1), yte


@pytest.fixture(scope="module")
def net():
    return ConvNet.load("artifacts/model_cnn_fp32.npz")


def test_im2col_matches_a_naive_window_extraction():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 6, 6, 3)).astype(np.float32)
    cols = im2col(x, 3, 3)
    assert cols.shape == (2, 4, 4, 27)
    # Spot-check one window against manual slicing, in the documented order.
    expected = np.concatenate([x[1, 2 + ky, 1 + kx, :] for ky in range(3) for kx in range(3)])
    np.testing.assert_allclose(cols[1, 2, 1], expected)


def test_col2im_is_the_adjoint_of_im2col():
    """<im2col(x), y> == <x, col2im(y)> -- the property the backward pass needs."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 5, 5, 2))
    y = rng.standard_normal((2, 3, 3, 18))
    lhs = float((im2col(x, 3, 3) * y).sum())
    rhs = float((x * col2im(y, x.shape, 3, 3)).sum())
    assert abs(lhs - rhs) < 1e-8 * max(1.0, abs(lhs))


def test_int8_conv_is_lossless_on_this_model(net, data):
    Xtr, _, Xte, yte = data
    q = quantize_convnet(net, Xtr[:128], "per_channel", "minmax")
    assert (q.predict(Xte) == yte).mean() >= (net.predict(Xte) == yte).mean() - 1e-12


def test_conv_accumulator_never_overflows_int32(net, data):
    Xtr, _, Xte, _ = data
    q = quantize_convnet(net, Xtr[:128], "per_channel", "minmax")
    q.forward_q(q.quantize_input(Xte))  # raises OverflowError if it does


def test_per_channel_decisively_beats_per_tensor_at_two_bits(net, data):
    """The headline conv finding: weight sharing makes granularity matter more."""
    Xtr, _, Xte, yte = data
    accs = {}
    for gran in ("per_tensor", "per_channel"):
        q = quantize_convnet(net, Xtr[:128], gran, "minmax", weight_bits=2)
        accs[gran] = (q.predict(Xte) == yte).mean()
    assert accs["per_channel"] - accs["per_tensor"] > 0.20, accs


def test_conv_quantizer_rejects_unknown_granularity(net, data):
    with pytest.raises(ValueError):
        quantize_convnet(net, data[0][:32], "per_nothing", "minmax")


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
def test_c_conv_engine_is_bit_exact_with_python(tmp_path_factory):
    exe = tmp_path_factory.mktemp("bin") / "crossval_graph"
    subprocess.run(["gcc", *CFLAGS, *SRC, "-o", str(exe)], cwd=ROOT, check=True)
    proc = subprocess.run(
        [str(exe), "artifacts/model_cnn.tqg", "artifacts/test_images.tqw",
         "artifacts/reference_cnn_out.bin"],
        cwd=ROOT, capture_output=True, text=True,
    )
    rep = json.loads(proc.stdout)
    assert rep["mismatched_values"] == 0, proc.stderr
    assert rep["argmax_disagreements"] == 0
    assert rep["bit_exact"] is True
    assert proc.returncode == 0


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
def test_conv_engine_rejects_a_bad_model_file(tmp_path, tmp_path_factory):
    exe = tmp_path_factory.mktemp("bin2") / "crossval_graph"
    subprocess.run(["gcc", *CFLAGS, *SRC, "-o", str(exe)], cwd=ROOT, check=True)
    bad = tmp_path / "bad.tqg"
    bad.write_bytes(b"NOPE" + b"\x00" * 64)
    proc = subprocess.run(
        [str(exe), str(bad), "artifacts/test_images.tqw", "artifacts/reference_cnn_out.bin"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "tqg_load failed" in proc.stderr
