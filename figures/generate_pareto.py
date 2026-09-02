#!/usr/bin/env python
"""Rank-based adaptive-resolution Pareto, 2x2 panels across 3 architectures:
Phi-3-Vision (fine-tuned, raw), Qwen3-VL-8B, InternVL3-8B. Escalate top-fraction by
scout-confidence risk; accuracy vs avg visual tokens. Run in the 'haram' env."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                 "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 8.5,
                 "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                 "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                 "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"random": "#2a9d8f", "popular": "#1f4e79", "adversarial": "#e76f51"}
SPL = ["random", "popular", "adversarial"]
A = os.environ.get("HARAM_RESULTS", HARAM_ROOT + "/results")
PANELS = [("(a) Phi-3-Vision, fine-tuned", f"{A}/adaptive_pareto"),
          ("(b) Phi-3-Vision, raw", f"{A}/adaptive_pareto_base"),
          ("(c) Qwen3-VL-8B, raw", f"{A}/adaptive_pareto_qwen3"),
          ("(d) InternVL3-8B, raw", f"{A}/adaptive_pareto_internvl")]


def curve(d, s):
    recs = json.load(open(os.path.join(d, f"adaptive_{s}.json")))["records"]
    pl = np.array([r["pred_low"] == "yes" for r in recs]); ph = np.array([r["pred_high"] == "yes" for r in recs])
    lb = np.array([r["label"] == "yes" for r in recs]); cl = np.array([r["conf_low"] for r in recs])
    tl = np.array([r["tok_low"] for r in recs], float); th = np.array([r["tok_high"] for r in recs], float)
    order = np.argsort(-(1 - cl), kind="stable"); n = len(lb)
    xs, ys = [], []
    for f in np.linspace(0, 1, 41):
        k = int(round(f * n)); esc = np.zeros(n, bool); esc[order[:k]] = True
        xs.append((tl + np.where(esc, th, 0)).mean()); ys.append((np.where(esc, ph, pl) == lb).mean())
    return np.array(xs), np.array(ys)


fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
for ax, (title, d) in zip(axes.ravel(), PANELS):
    for s in SPL:
        xs, ys = curve(d, s); c = COL[s]
        ax.plot(xs, ys, "-", color=c, lw=1.4, label=s.capitalize(), zorder=2)
        ax.scatter([xs[0]], [ys[0]], marker="s", s=22, color=c, edgecolor="k", lw=0.4, zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], marker="*", s=64, color=c, edgecolor="k", lw=0.4, zorder=3)
        hi = ys[-1]; cand = np.where(ys >= hi - 0.005)[0]
        if len(cand):
            k = cand[np.argmin(xs[cand])]
            ax.scatter([xs[k]], [ys[k]], marker="o", s=22, facecolor="white", edgecolor=c, lw=1.2, zorder=4)
    ax.set_title(title, fontsize=8.5); ax.grid(True, ls=":", lw=0.5, alpha=0.6)
for ax in axes[1]:
    ax.set_xlabel("Avg visual tokens / query")
for ax in axes[:, 0]:
    ax.set_ylabel("Accuracy")
axes[0, 0].legend(loc="lower right", frameon=False, title="POPE split", title_fontsize=7)
mk = [Line2D([0], [0], marker="s", color="gray", ls="", mec="k", ms=5, label="always-low"),
      Line2D([0], [0], marker="*", color="gray", ls="", mec="k", ms=8, label="always-high"),
      Line2D([0], [0], marker="o", color="w", ls="", mec="gray", ms=5, label="adaptive")]
axes[0, 1].legend(handles=mk, loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_pareto.pdf"))
print("wrote fig_pareto.pdf (2x2, 3 architectures)")
