#!/usr/bin/env python3
"""Render the two charts in the README from the measured JSON artifacts."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1b1b1f"
GRID = "#d9d9e0"
C_TENSOR = "#c2561f"
C_CHANNEL = "#1f6f8b"
C_BASE = "#6b6b76"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def chart_accuracy(sweep: dict, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    rows = [r for r in sweep["rows"] if r["calibration"] == "minmax"]
    for gran, colour, label in (("per_tensor", C_TENSOR, "per-tensor weight scale"),
                                ("per_channel", C_CHANNEL, "per-channel weight scale")):
        sel = sorted([r for r in rows if r["granularity"] == gran], key=lambda r: r["weight_bits"])
        xs = [r["weight_bits"] for r in sel]
        ys = [r["test_acc"] * 100 for r in sel]
        ax.plot(xs, ys, marker="o", color=colour, label=label, linewidth=2)

    base = sweep["fp32_test_acc"] * 100
    ax.axhline(base, color=C_BASE, linestyle="--", linewidth=1.2)
    ax.annotate(f"FP32 baseline  {base:.2f}%", xy=(7.6, base), xytext=(0, 6),
                textcoords="offset points", color=C_BASE, fontsize=9)

    ax.set_xlabel("weight bit-width (activations fixed at INT8)")
    ax.set_ylabel(f"test accuracy %  (n={sweep['n_test']})")
    ax.set_title("Accuracy holds to 4 bits with per-channel scales, then falls off a cliff",
                 fontsize=11, loc="left", pad=12)
    ax.invert_xaxis()
    ax.set_xticks([8, 6, 5, 4, 3, 2])
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def chart_scaling(rows: list[dict], out: pathlib.Path, l1_kib: float, l2_kib: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = [r["fp32_weight_bytes"] / 1024 for r in rows]
    ys = [r["speedup_p50"] for r in rows]

    ax.axhline(1.0, color=C_BASE, linestyle="--", linewidth=1.2)
    ax.plot(xs, ys, marker="o", color=C_CHANNEL, linewidth=2)

    for lab, kib in (("L1d 96 KiB", l1_kib), ("L2 4 MiB", l2_kib)):
        ax.axvline(kib, color=C_TENSOR, linewidth=1, alpha=0.55)
        ax.annotate(lab, xy=(kib, max(ys) * 0.99), xytext=(4, 0), textcoords="offset points",
                    rotation=90, va="top", fontsize=8, color=C_TENSOR)

    ax.set_xscale("log")
    ax.set_xlabel("FP32 weight footprint (KiB, log scale)")
    ax.set_ylabel("INT8 speedup vs FP32 (p50, higher is better)")
    ax.set_title("INT8 only gets faster once the weights stop fitting in cache",
                 fontsize=11, loc="left", pad=12)
    ax.annotate("INT8 slower", xy=(xs[0], ys[0]), xytext=(6, -14), textcoords="offset points",
                fontsize=9, color=C_BASE)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


if __name__ == "__main__":
    art = pathlib.Path("artifacts")
    docs = pathlib.Path("docs")
    docs.mkdir(exist_ok=True)
    chart_accuracy(json.loads((art / "sweep.json").read_text()), docs / "accuracy_vs_bits.svg")
    chart_scaling(json.loads((art / "scaling_study.json").read_text()),
                  docs / "int8_speedup_vs_size.svg", 96, 4096)
    print("wrote docs/accuracy_vs_bits.svg and docs/int8_speedup_vs_size.svg")
