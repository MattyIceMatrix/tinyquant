"""Tests that compile and run the C engine.

The bit-exactness test is the one that matters: if the C kernel and the Python
reference ever diverge, every accuracy number in the README becomes a claim
about a program nobody is running.
"""
import json
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFLAGS = ["-std=c99", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-DTQ_SKIP_ZEROS=0"]

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")


@pytest.fixture(scope="module")
def crossval_bin(tmp_path_factory):
    out = tmp_path_factory.mktemp("bin") / "crossval"
    subprocess.run(["gcc", *CFLAGS, "engine/tinyq.c", "engine/crossval.c", "-o", str(out)],
                   cwd=ROOT, check=True)
    return out


def test_engine_compiles_with_warnings_as_errors(crossval_bin):
    assert crossval_bin.exists()


def test_c_engine_is_bit_exact_with_python_reference(crossval_bin):
    proc = subprocess.run(
        [str(crossval_bin), "artifacts/model.tq8", "artifacts/test_vectors.tqv",
         "artifacts/reference_out.bin"],
        cwd=ROOT, capture_output=True, text=True,
    )
    report = json.loads(proc.stdout)
    assert report["mismatched_values"] == 0, proc.stderr
    assert report["argmax_disagreements"] == 0
    assert report["bit_exact"] is True
    assert proc.returncode == 0


def test_int8_inference_path_contains_no_floating_point_instructions():
    """The engine claims it runs on targets without an FPU. This checks it."""
    if shutil.which("objdump") is None:
        pytest.skip("objdump not available")
    obj = ROOT / "build" / "tinyq_fpcheck.o"
    obj.parent.mkdir(exist_ok=True)
    subprocess.run(["gcc", *CFLAGS, "-c", "engine/tinyq.c", "-o", str(obj)], cwd=ROOT, check=True)
    dis = subprocess.run(["objdump", "-d", "--disassemble=tq_forward", str(obj)],
                         capture_output=True, text=True, check=True).stdout
    forbidden = ("mulss", "addss", "subss", "divss", "cvtsi2ss", "cvtss2si",
                 "mulsd", "addsd", "mulps", "addps", "vmulps", "vaddps")
    found = [m for m in forbidden if m in dis]
    assert not found, f"floating-point instructions in tq_forward: {found}"


def test_engine_rejects_a_file_with_the_wrong_magic(tmp_path, crossval_bin):
    bad = tmp_path / "bad.tq8"
    bad.write_bytes(b"XXXX" + b"\x00" * 64)
    proc = subprocess.run(
        [str(crossval_bin), str(bad), "artifacts/test_vectors.tqv", "artifacts/reference_out.bin"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "tq_load failed" in proc.stderr


def test_engine_rejects_a_truncated_file(tmp_path, crossval_bin):
    full = (ROOT / "artifacts" / "model.tq8").read_bytes()
    trunc = tmp_path / "trunc.tq8"
    trunc.write_bytes(full[: len(full) // 3])
    proc = subprocess.run(
        [str(crossval_bin), str(trunc), "artifacts/test_vectors.tqv", "artifacts/reference_out.bin"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode != 0
