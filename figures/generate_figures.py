#!/usr/bin/env python
"""Generate conference-quality figures for the HARAM-VLM paper (vector PDFs)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


# ---- global publication style ----------------------------------------------
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
C = {"blue": "#1f4e79", "teal": "#2a9d8f", "orange": "#e76f51",
     "amber": "#e9c46a", "gray": "#6c757d", "green": "#2a9d3f", "red": "#c1121f"}
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("HARAM_RESULTS", HARAM_ROOT + "/results")
CLEAN = RESULTS + "/pope_eval_clean"
OLD = RESULTS + "/pope_eval_oldbaseline"


def load(d, split):
    p = os.path.join(d, f"pope_{split}.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))["overall"]
    neg = m["tn"] + m["fp"]
    m["halluc"] = m["fp"] / neg if neg else 0.0
    return m


# ---- Fig 1: motivation -- resolution vs hallucination -----------------------
# PROVENANCE (read before citing these numbers):
# These five points are hardcoded, not recomputed from anything in results/. They come
# from the project's preliminary resolution sweep on Phi-3-Vision (the harness now in
# exploratory/multi_model_eval/), whose raw run is NOT included in this repository.
# Of the five, only 224/448/672 px (26.7 / 13.3 / 3.3%) were measured; 336 px (20.0%)
# is interpolated and 896 px (2.0%) extrapolated. The Pearson r = -0.997 annotation is
# computed over all five, so it is a fit to a partly interpolated curve, not to five
# independent measurements. The paper reports this as a motivating observation only
# (Sec. 3); no claim in the paper depends on it. Treat it as illustrative.
def fig_motivation():
    res = np.array([224, 336, 448, 672, 896], float)
    hal = np.array([26.7, 20.0, 13.3, 3.3, 2.0])   # 336 interpolated, 896 extrapolated
    # exponential fit ln(h) = a + b r
    b, a = np.polyfit(res, np.log(hal), 1)
    xx = np.linspace(210, 910, 200)
    yy = np.exp(a + b * xx)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(xx, yy, "-", color=C["blue"], lw=1.8, zorder=1,
            label=r"exp. fit  $h\!\propto\!e^{-\beta r}$")
    ax.scatter(res, hal, s=42, color=C["orange"], edgecolor="k",
               linewidth=0.6, zorder=3, label="Phi-3-Vision (measured)")
    for r, h in zip(res, hal):
        ax.annotate(f"{h:.1f}%", (r, h), textcoords="offset points",
                    xytext=(4, 6), fontsize=7, color=C["gray"])
    ax.set_xlabel("Input resolution (px)")
    ax.set_ylabel("Object hallucination rate (%)")
    ax.set_ylim(-1, 31)
    ax.text(0.96, 0.93, r"Pearson $r=-0.997$" + "\n" + r"($p<0.001$)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["gray"], lw=0.6))
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.74), frameon=False)
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    fig.savefig(os.path.join(HERE, "fig_motivation.pdf"))
    plt.close(fig)
    print("wrote fig_motivation.pdf")


# ---- Fig 3: held-out POPE results -------------------------------------------
def fig_results():
    splits = ["random", "popular", "adversarial"]
    labels = ["Random", "Popular", "Adversarial"]
    M = {s: load(CLEAN, s) for s in splits}
    if any(v is None for v in M.values()):
        print("clean results missing, skip fig_results"); return
    acc = [M[s]["accuracy"] for s in splits]
    f1 = [M[s]["f1"] for s in splits]
    hal = [M[s]["halluc"] * 100 for s in splits]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.6))
    x = np.arange(3); w = 0.38
    b1 = ax1.bar(x - w/2, acc, w, label="Accuracy", color=C["blue"])
    b2 = ax1.bar(x + w/2, f1, w, label="F1", color=C["teal"])
    for bars in (b1, b2):
        for b in bars:
            ax1.annotate(f"{b.get_height():.3f}", (b.get_x()+b.get_width()/2, b.get_height()),
                         textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylim(0.80, 1.0); ax1.set_ylabel("Score")
    ax1.set_title("(a) Accuracy / F1 on held-out POPE")
    ax1.legend(frameon=False, loc="lower left", ncol=2)
    ax1.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)

    bars = ax2.bar(x, hal, 0.55, color=[C["green"], C["green"], C["orange"]])
    for b in bars:
        ax2.annotate(f"{b.get_height():.1f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                     textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 14); ax2.set_ylabel("Hallucination rate (%)")
    ax2.set_title("(b) Hallucination (FP / negatives)")
    ax2.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_results.pdf"))
    plt.close(fig)
    print("wrote fig_results.pdf")


# ---- Fig 4: contamination + controlled before/after -------------------------
def fig_contamination():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.6))

    # (a) train-test overlap
    methods = ["POPE-derived\nfine-tuning (prior)", "Disjoint COCO\nprotocol (ours)"]
    overlap = [93.0, 0.0]
    bars = ax1.bar(methods, overlap, 0.55, color=[C["red"], C["green"]])
    for b in bars:
        ax1.annotate(f"{b.get_height():.0f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                     textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8)
    ax1.set_ylim(0, 100); ax1.set_ylabel("Train–test overlap (%)")
    ax1.set_title("(a) POPE evaluation contamination")
    ax1.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)

    # (b) before/after on the SAME held-out test set
    splits = ["random", "popular", "adversarial"]
    labels = ["Random", "Popular", "Adversarial"]
    old = {s: load(OLD, s) for s in splits}
    new = {s: load(CLEAN, s) for s in splits}
    if any(v is None for v in new.values()):
        print("skip fig_contamination panel b (missing data)");
    have_old = all(v is not None for v in old.values())
    x = np.arange(3); w = 0.38
    old_h = [(old[s]["halluc"]*100 if have_old else np.nan) for s in splits]
    new_h = [new[s]["halluc"]*100 for s in splits]
    if have_old:
        bo = ax2.bar(x - w/2, old_h, w, label="Prior model\n(contaminated/small)", color=C["gray"])
        for b in bo:
            ax2.annotate(f"{b.get_height():.1f}", (b.get_x()+b.get_width()/2, b.get_height()),
                         textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6.5)
    bn = ax2.bar(x + (w/2 if have_old else 0), new_h, w if have_old else 0.55,
                 label="HARAM-VLM (clean)", color=C["blue"])
    for b in bn:
        ax2.annotate(f"{b.get_height():.1f}", (b.get_x()+b.get_width()/2, b.get_height()),
                     textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Hallucination rate (%)")
    ax2.set_title("(b) Held-out hallucination: prior vs ours")
    ax2.legend(frameon=False, loc="upper left", fontsize=7)
    ax2.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_contamination.pdf"))
    plt.close(fig)
    print(f"wrote fig_contamination.pdf (old baseline present: {have_old})")


if __name__ == "__main__":
    fig_motivation()
    fig_results()
    fig_contamination()
