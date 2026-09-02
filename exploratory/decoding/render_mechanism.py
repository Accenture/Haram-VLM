#!/usr/bin/env python
"""
Move 3 figure (fig_mechanism.pdf), two panels:
 (left)  MECHANISM -- mean logit-lens P(correct answer) across layers at low resolution, for greedy-wrong
         items. DoLa recovers an error iff the correct token re-emerges in late layers (final > early).
         POPE-adv prior-driven errors show this re-emergence; V*Bench perception-bound errors decay and
         never re-emerge ("the evidence is in no layer").
 (right) MODE-PROFILE -- among low-resolution errors, the fraction repaired by DoLa (prior-driven),
         by resolution (perception-bound), both, or neither. POPE-adv and V*Bench sit at opposite ends.
CPU only; run in 'haram' env.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                 "figure.dpi": 240, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
                 "axes.linewidth": 0.8, "font.size": 9})
TEAL, ORANGE, GRAY, INK = "#2A9D8F", "#E08E1B", "#9aa0a6", "#1c1c1c"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")


def mech_panel(ax):
    d = json.load(open(os.path.join(RES, "move3_layer_traj_qwen3.json")))
    R, N = d["records"], d["nlayers"]
    x = np.arange(N + 1) / N
    def mean(recs): return np.mean([r["traj"] for r in recs], axis=0)
    pope_gw = [r for r in R if r["task"] == "pope" and not r["greedy_ok"]]
    vst_gw = [r for r in R if r["task"] == "vstar" and not r["greedy_ok"]]
    pope_fix = [r for r in pope_gw if r["dola_ok"]]
    pope_fail = [r for r in pope_gw if not r["dola_ok"]]
    ax.plot(x, mean(pope_fix), "-", color=ORANGE, lw=2.6, zorder=5,
            label=f"POPE-adv, DoLa recovers ({len(pope_fix)}/{len(pope_gw)})")
    ax.plot(x, mean(pope_fail), "--", color=ORANGE, lw=1.3, alpha=0.7, zorder=3,
            label="POPE-adv, DoLa fails")
    ax.plot(x, mean(vst_gw), "-", color=TEAL, lw=2.6, zorder=5,
            label=f"V*Bench errors ({len(vst_gw)})")
    ax.axhline(0.5, color="#cccccc", lw=0.8, ls=":")
    ax.annotate("correct token\nre-emerges late", (0.93, mean(pope_fix)[-1]), fontsize=7.4, color="#9a6313",
                ha="right", va="bottom", xytext=(-2, 6), textcoords="offset points")
    ax.set_xlabel("relative layer depth (early $\\rightarrow$ output)")
    ax.set_ylabel("logit-lens $P$(correct)")
    ax.set_title("Mechanism: is the answer in any layer?", fontsize=9.3)
    ax.legend(loc="upper left", fontsize=6.9, frameon=True, framealpha=0.92)
    ax.grid(True, ls=":", lw=0.5, color="#e6e6e6")
    ax.set_ylim(-0.02, 0.62)


def profile_panel(ax):
    cats = ["prior", "both", "perc", "neither"]
    cols = {"prior": ORANGE, "both": "#b58b4a", "perc": TEAL, "neither": GRAY}
    lab = {"prior": "DoLa fixes (prior)", "both": "both", "perc": "resolution fixes (perception)",
           "neither": "neither"}
    bars = []  # (name, dict of fractions)
    for a, tag in [("qwen3", "Qwen3"), ("internvl", "InternVL3")]:
        d = json.load(open(os.path.join(RES, f"move2_decode_and_look_{a}.json")))
        for task, tname in [("pope", "POPE-adv"), ("vstar", "V*Bench")]:
            err = [r for r in d["records"] if r["task"] == task and not r["c_low"]]
            n = len(err) or 1
            frac = {
                "prior": sum(1 for r in err if r["c_low_dola"] and not r["c_high"]) / n,
                "both": sum(1 for r in err if r["c_low_dola"] and r["c_high"]) / n,
                "perc": sum(1 for r in err if r["c_high"] and not r["c_low_dola"]) / n,
                "neither": sum(1 for r in err if not r["c_low_dola"] and not r["c_high"]) / n}
            bars.append((f"{tname}\n({tag})", frac))
    y = np.arange(len(bars))[::-1]
    for i, (name, frac) in enumerate(bars):
        left = 0.0
        for c in cats:
            w = frac[c]
            ax.barh(y[i], w, left=left, color=cols[c], edgecolor="white", linewidth=0.7,
                    label=lab[c] if i == 0 else None)
            if w > 0.08:
                ax.text(left + w / 2, y[i], f"{w*100:.0f}", ha="center", va="center",
                        fontsize=7.2, color="white", fontweight="bold")
            left += w
    ax.set_yticks(y); ax.set_yticklabels([b[0] for b in bars], fontsize=7.8)
    ax.set_xlim(0, 1); ax.set_xlabel("share of low-resolution errors")
    ax.set_title("Mode-profile: what repairs each error", fontsize=9.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=6.7, frameon=False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1.05, 1.0]})
    mech_panel(axes[0]); profile_panel(axes[1])
    fig.subplots_adjust(wspace=0.32)
    out = os.path.join(HERE, "..", "draft", "figures", "fig_mechanism.pdf")
    fig.savefig(out); print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
