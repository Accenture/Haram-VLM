#!/usr/bin/env python
"""
Render the Move-2 Decode-and-Look cost-accuracy Pareto figure (fig_decode_and_look.pdf).
One panel per architecture JSON (move2_decode_and_look_<arch>.json). Shows: resolution-only adaptive
frontier, Decode-and-Look frontier, the four fixed strategies, and the oracle upper bounds.
CPU only; run in 'haram' env (matplotlib).
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                 "figure.dpi": 240, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
                 "axes.linewidth": 0.8, "font.size": 9})
TEAL, GRAY, ORANGE, INK = "#2A9D8F", "#9aa0a6", "#E08E1B", "#1c1c1c"
HERE = os.path.dirname(os.path.abspath(__file__))
TITLE = {"qwen3": "Qwen3-VL", "internvl": "InternVL3-8B"}


def panel(ax, d):
    FR, FD, f = d["frontier_resolution"], d["frontier_decode_and_look"], d["fixed"]
    # adaptive frontiers
    ax.plot([p["cost"] for p in FR], [p["acc"] for p in FR], "-", color=GRAY, lw=1.6,
            label="resolution-only adaptive", zorder=2)
    ax.plot([p["cost"] for p in FD], [p["acc"] for p in FD], "-", color=TEAL, lw=2.4,
            label="Decode-and-Look (ours)", zorder=4)
    # fixed strategies
    fixed = [("low", "o", "#777"), ("low_dola", "s", ORANGE), ("high", "o", "#444"),
             ("high_dola", "s", "#b5651d")]
    lab = {"low": "low", "low_dola": "low+DoLa", "high": "high", "high_dola": "high+DoLa"}
    for k, mk, col in fixed:
        m = f[k]
        ax.scatter([m["cost"]], [m["acc"]], marker=mk, s=46, color=col, zorder=6,
                   edgecolor="white", linewidth=0.6)
        dy = 0.006 if k in ("low_dola", "high_dola") else -0.012
        ax.annotate(lab[k], (m["cost"], m["acc"]), fontsize=7.6, color=col,
                    xytext=(4, dy * 800), textcoords="offset points", zorder=7)
    # oracle bounds
    for key, mk, txt in [("mode_oracle", "*", "mode-oracle"), ("oracle_cheapest", "D", "per-item oracle")]:
        o = d[key]
        ax.scatter([o["cost"]], [o["acc"]], marker=mk, s=(120 if mk == "*" else 36), color=INK,
                   zorder=6, facecolor="none" if mk == "D" else INK, edgecolor=INK, linewidth=1.0)
        ax.annotate(txt, (o["cost"], o["acc"]), fontsize=7.2, color=INK, style="italic",
                    xytext=(5, 3), textcoords="offset points", zorder=7)
    ax.set_xlabel("avg.\\ visual tokens / query")
    ax.set_ylabel("mixed-stream accuracy")
    ax.set_title(TITLE.get(d["arch"], d["arch"]), fontsize=9.5)
    ax.grid(True, ls=":", lw=0.5, color="#dddddd")


def main():
    archs = sys.argv[1:] or ["qwen3", "internvl"]
    ds = []
    for a in archs:
        p = os.path.join(HERE, "..", "..", "results", f"move2_decode_and_look_{a}.json")
        if os.path.exists(p): ds.append(json.load(open(p)))
    assert ds, "no result JSONs found"
    fig, axes = plt.subplots(1, len(ds), figsize=(3.7 * len(ds), 3.1), squeeze=False)
    for ax, d in zip(axes[0], ds):
        panel(ax, d)
    axes[0][0].legend(loc="lower right", fontsize=7.4, frameon=True, framealpha=0.92)
    out = os.path.join(HERE, "..", "draft", "figures", "fig_decode_and_look.pdf")
    fig.savefig(out)
    print("wrote", os.path.abspath(out), "| panels:", [d["arch"] for d in ds])


if __name__ == "__main__":
    main()
