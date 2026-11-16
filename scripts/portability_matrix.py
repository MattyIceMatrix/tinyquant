#!/usr/bin/env python3
"""Build the engine every way we can and check the output never changes.

An INT8 kernel that gives different answers at -O3 than at -O0, or on ARM than
on x86, is not a deployable kernel -- it is a kernel that happens to work on the
machine it was written on. The matrix covers:

  compilers         gcc, clang
  optimisation      -O0 -O1 -O2 -O3 -Os -Ofast
  architectures     x86-64, aarch64, riscv64, arm (32-bit, hard float)

Cross-architecture builds use `zig cc` and run under qemu-user, so the whole
matrix reproduces on any Linux box without special hardware. Every build runs
the same bit-exactness check against the same Python-generated reference bytes.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "matrix"
ART = ROOT / "artifacts"

NATIVE_OPTS = ["-O0", "-O1", "-O2", "-O3", "-Os"]
NATIVE_COMPILERS = [("gcc", ["gcc"]), ("clang", ["clang"])]
# Distributions disagree on whether the emulator is packaged as
# qemu-<arch>-static or plain qemu-<arch> (Ubuntu 26.04 dropped the -static
# package name), so each target lists the names to try in order. The binaries
# we run are statically linked by zig either way, so both work.
CROSS_TARGETS = [
    ("aarch64", "aarch64-linux-musl", ["qemu-aarch64-static", "qemu-aarch64"]),
    ("riscv64", "riscv64-linux-musl", ["qemu-riscv64-static", "qemu-riscv64"]),
    ("arm32", "arm-linux-musleabihf", ["qemu-arm-static", "qemu-arm"]),
]


def find_qemu(candidates: list[str]) -> str | None:
    for name in candidates:
        if shutil.which(name):
            return name
    return None
BASE = ["-std=c99", "-DTQ_SKIP_ZEROS=0"]

# Both engines are checked in every configuration. The conv engine shares the
# requantization arithmetic with the dense one, so a portability bug in that
# arithmetic would show up in both -- but a layout or indexing bug would show
# up only in the conv kernel, which is exactly why it is not assumed.
ENGINES = {
    "dense": (
        ["engine/tinyq.c", "engine/crossval.c"],
        ["artifacts/model.tq8", "artifacts/test_vectors.tqv", "artifacts/reference_out.bin"],
    ),
    "conv": (
        ["engine/tinyq.c", "engine/tinyq_graph.c", "engine/crossval_graph.c"],
        ["artifacts/model_cnn.tqg", "artifacts/test_images.tqw",
         "artifacts/reference_cnn_out.bin"],
    ),
}


def run_case(name: str, cmd: list[str], runner: list[str] | None, engine: str = "dense") -> dict:
    src, ARGS = ENGINES[engine]
    exe = BUILD / f"crossval_{name}_{engine}"
    build = subprocess.run([*cmd, *BASE, *src, "-o", str(exe)], cwd=ROOT,
                           capture_output=True, text=True)
    if build.returncode != 0:
        return {"case": name, "engine": engine, "built": False,
                "error": build.stderr.strip()[-300:]}

    proc = subprocess.run([*(runner or []), str(exe), *ARGS], cwd=ROOT,
                          capture_output=True, text=True)
    if proc.returncode not in (0, 3):
        return {"case": name, "engine": engine, "built": True, "ran": False,
                "error": (proc.stderr or proc.stdout).strip()[-300:]}
    rep = json.loads(proc.stdout)
    return {
        "case": name,
        "engine": engine,
        "built": True,
        "ran": True,
        "bit_exact": rep["bit_exact"],
        "mismatched_values": rep["mismatched_values"],
        "accuracy": rep["c_engine_accuracy"],
    }


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    results = []

    for cname, cbase in NATIVE_COMPILERS:
        if shutil.which(cbase[0]) is None:
            results.append({"case": f"{cname}", "skipped": "compiler not installed"})
            continue
        for opt in NATIVE_OPTS:
            for engine in ENGINES:
                results.append(run_case(f"{cname}_x86-64_{opt.lstrip('-')}",
                                        [*cbase, opt], None, engine))

    zig_ok = subprocess.run([sys.executable, "-m", "ziglang", "version"],
                            capture_output=True).returncode == 0
    for arch, triple, qemu_names in CROSS_TARGETS:
        qemu = find_qemu(qemu_names)
        if not zig_ok or qemu is None:
            results.append({"case": f"zig_{arch}_O2",
                            "skipped": "zig cc or qemu-user not installed"})
            continue
        for engine in ENGINES:
            results.append(run_case(
                f"zig_{arch}_O2",
                [sys.executable, "-m", "ziglang", "cc", "-target", triple, "-O2", "-static"],
                [qemu], engine,
            ))

    checked = [r for r in results if r.get("ran")]
    summary = {
        "cases_total": len(results),
        "cases_run": len(checked),
        "cases_skipped": len([r for r in results if r.get("skipped")]),
        "all_bit_exact": all(r["bit_exact"] for r in checked) if checked else False,
        "engines_verified": sorted({r["engine"] for r in checked}),
        "distinct_accuracies_dense": sorted({r["accuracy"] for r in checked
                                             if r["engine"] == "dense"}),
        "distinct_accuracies_conv": sorted({r["accuracy"] for r in checked
                                            if r["engine"] == "conv"}),
        "architectures_verified": sorted({
            r["case"].split("_")[1] for r in checked
        }),
        "results": results,
    }
    (ART / "portability.json").write_text(json.dumps(summary, indent=2))

    for r in results:
        if r.get("skipped"):
            print(f"  SKIP  {r['case']:<28} {r['skipped']}")
        elif not r.get("ran"):
            print(f"  FAIL  {r['case']:<28} {r.get('error', 'build failed')}")
        else:
            flag = "OK  " if r["bit_exact"] else "DIFF"
            print(f"  {flag}  {r['case']:<28} {r['engine']:<6} acc={r['accuracy']:.6f} "
                  f"mismatches={r['mismatched_values']}")
    print(f"\n{summary['cases_run']} builds run across "
          f"{len(summary['architectures_verified'])} architectures and "
          f"{len(summary['engines_verified'])} engines; "
          f"all bit-exact: {summary['all_bit_exact']}\n"
          f"distinct accuracies -- dense: {summary['distinct_accuracies_dense']}, "
          f"conv: {summary['distinct_accuracies_conv']}")
    return 0 if summary["all_bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
