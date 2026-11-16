"""The binary format is a contract between Python and C; check its header."""
import struct

import numpy as np

from tinyquant.data import load_split
from tinyquant.export import export_fp32, export_int8
from tinyquant.model import MLP
from tinyquant.quantize import quantize_model


def test_int8_header_and_size(tmp_path):
    Xtr, _, _, _ = load_split()
    model = MLP.load("artifacts/model_fp32.npz")
    qm = quantize_model(model, Xtr[:128], "per_channel", "minmax")
    path = tmp_path / "m.tq8"
    size = export_int8(qm, str(path))

    raw = path.read_bytes()
    assert raw[:4] == b"TQI1"
    n_layers, n_in = struct.unpack("<II", raw[4:12])
    in_scale, = struct.unpack("<f", raw[12:16])
    bits, = struct.unpack("<I", raw[16:20])
    assert n_layers == model.n_layers
    assert n_in == model.W[0].shape[0]
    assert bits == 8
    assert in_scale > 0
    assert size == len(raw)


def test_fp32_export_is_four_bytes_per_parameter_plus_headers(tmp_path):
    model = MLP.load("artifacts/model_fp32.npz")
    path = tmp_path / "m.tqf"
    size = export_fp32(model, str(path))
    payload = model.n_params() * 4
    header = 12 + 16 * model.n_layers
    assert size == payload + header
    assert path.read_bytes()[:4] == b"TQF1"
