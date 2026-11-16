#!/usr/bin/env python3
"""Measure what the engine actually costs: stack, heap, memory errors.

Claims about running on a microcontroller are cheap to make and easy to check,
so they are checked here rather than asserted in prose:

  heap    the inference path must not reference any allocator symbol
  stack   worst-case frame of tq_forward, measured with -fstack-usage, at
          several TQ_MAX_WIDTH settings, since that knob is what a firmware
          integrator would actually turn
  memory  valgrind over the full test set: zero errors, zero leaks
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "audit"
ART = ROOT / "artifacts"
ALLOCATORS = ("malloc", "calloc", "realloc", "free", "aligned_alloc", "posix_memalign",
              "strdup", "mmap", "brk", "sbrk", "operator new")


def heap_check() -> dict:
    """Disassemble tq_forward and look for calls to any allocator."""
    obj = BUILD / "tinyq_heap.o"
    subprocess.run(["gcc", "-std=c99", "-O2", "-DTQ_SKIP_ZEROS=0", "-c",
                    "engine/tinyq.c", "-o", str(obj)], cwd=ROOT, check=True)
    found = {}
    for fn in ("tq_forward", "tq_requantize", "tq_argmax_i8"):
        dis = subprocess.run(["objdump", "-d", f"--disassemble={fn}", str(obj)],
                             capture_output=True, text=True, check=True).stdout
        hits = sorted({a for a in ALLOCATORS if a in dis})
        found[fn] = hits
    return {
        "functions_checked": list(found),
        "allocator_references": found,
        "inference_path_is_allocation_free": all(not v for v in found.values()),
    }


def stack_usage() -> dict:
    """Worst-case stack frame of tq_forward as a function of TQ_MAX_WIDTH."""
    out = {}
    for width in (32, 64, 128, 256, 512):
        d = BUILD / f"su{width}"
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcc", "-std=c99", "-O2", f"-DTQ_MAX_WIDTH={width}",
                        "-DTQ_SKIP_ZEROS=0", "-fstack-usage", "-c",
                        str(ROOT / "engine" / "tinyq.c"), "-o", str(d / "tinyq.o")],
                       cwd=d, check=True)
        su = next(d.glob("*.su"), None)
        frames = {}
        if su:
            for line in su.read_text().splitlines():
                m = re.match(r"^.*:\d+:\d+:(\w+)\t(\d+)\t(\w+)", line)
                if m:
                    frames[m.group(1)] = int(m.group(2))
        out[str(width)] = {
            "tq_forward_bytes": frames.get("tq_forward"),
            "tqf_forward_bytes": frames.get("tqf_forward"),
            "all_frames": frames,
        }
    return out


def valgrind_check() -> dict:
    if shutil.which("valgrind") is None:
        return {"skipped": "valgrind not installed"}
    exe = BUILD / "crossval_vg"
    subprocess.run(["gcc", "-std=c99", "-O1", "-g", "-DTQ_SKIP_ZEROS=0",
                    "engine/tinyq.c", "engine/crossval.c", "-o", str(exe)],
                   cwd=ROOT, check=True)
    proc = subprocess.run(
        ["valgrind", "--error-exitcode=42", "--leak-check=full", "--errors-for-leak-kinds=all",
         str(exe), "artifacts/model.tq8", "artifacts/test_vectors.tqv",
         "artifacts/reference_out.bin"],
        cwd=ROOT, capture_output=True, text=True,
    )
    err = proc.stderr
    m = re.search(r"ERROR SUMMARY: (\d+) errors", err)
    return {
        "exit_code": proc.returncode,
        "errors": int(m.group(1)) if m else None,
        "clean": proc.returncode == 0 and (m is not None and int(m.group(1)) == 0),
        "tail": err.strip().splitlines()[-4:],
    }


if __name__ == "__main__":
    BUILD.mkdir(parents=True, exist_ok=True)
    report = {"heap": heap_check(), "stack": stack_usage(), "valgrind": valgrind_check()}
    (ART / "resource_audit.json").write_text(json.dumps(report, indent=2))

    h = report["heap"]
    print(f"heap:  inference path allocation-free: {h['inference_path_is_allocation_free']}")
    for w, s in report["stack"].items():
        print(f"stack: TQ_MAX_WIDTH={w:<4} tq_forward frame = {s['tq_forward_bytes']} bytes"
              f"  (FP32 path {s['tqf_forward_bytes']} bytes)")
    v = report["valgrind"]
    print(f"valgrind: clean={v.get('clean')} errors={v.get('errors')}")
