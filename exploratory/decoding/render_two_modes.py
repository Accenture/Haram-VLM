#!/usr/bin/env python
"""
Render the two-failure-mode qualitative figure (fig_two_modes.pdf) from dumped per-item preds.
Top panel  = PRIOR-DRIVEN (POPE-adv): low-res hallucinates YES; DoLa (decoding) flips to NO; resolution
             does not. Bottom = PERCEPTION-BOUND (V*Bench): low-res confidently wrong; DoLa can't fix;
             only high resolution recovers. Concretely illustrates the 2x2 diagonal ("match cure to cause").
Run in 'haram' env (matplotlib + PIL); CPU only.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                 "figure.dpi": 240, "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
INK, MUTE = "#1c1c1c", "#6f6f6f"
PRIOR, PERC = "#E08E1B", "#2A9D8F"
OK = dict(fc="#dff3ef", ec="#2A9D8F", tc="#1c6f63")
BAD = dict(fc="#fbe2e6", ec="#D1495B", tc="#a32d3e")
GT = dict(fc="#eeeeee", ec="#c9c9c9", tc="#3a3a3a")
TOKB = dict(fc="#eef2f7", ec="#cdd7e3", tc="#42566e")
HERE = os.path.dirname(os.path.abspath(__file__))
POPE_IMG = HARAM_ROOT + "/coco_build/images"
LETTERS = "ABCD"


def chip(ax, x, y, t, c, fs=10):
    ax.text(x, y, t, transform=ax.transAxes, fontsize=fs, fontweight="bold", color=c["tc"], va="center",
            ha="left", bbox=dict(boxstyle="round,pad=0.30", fc=c["fc"], ec=c["ec"], lw=0.9))


def badge(ax, x, y, t):
    ax.text(x, y, t, transform=ax.transAxes, fontsize=8.6, color=TOKB["tc"], va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.26", fc=TOKB["fc"], ec=TOKB["ec"], lw=0.8))


def header(ax, color, txt):
    ax.add_patch(FancyBboxPatch((0.0, 0.9), 1.0, 0.12, boxstyle="round,pad=0,rounding_size=0.018",
                                transform=ax.transAxes, fc=color, ec="none", clip_on=False))
    ax.text(0.025, 0.96, txt, transform=ax.transAxes, color="white", fontsize=10.5, fontweight="bold", va="center")


def pick_pope(recs):
    c = [r for r in recs if r["low"] == "yes" and r["low_dola"] == "no"]  # halluc fixed by DoLa
    c.sort(key=lambda r: (r["high"] != "yes", -r["low_conf"]))            # prefer high also wrong + confident
    return c[0] if c else None


def pick_vstar(recs):
    c = [r for r in recs if r["low"] != r["gt"] and r["high"] == r["gt"] and r["low_dola"] != r["gt"]]
    c.sort(key=lambda r: -r["low_conf"])                                  # most confident wrong low-res
    return c[0] if c else None


def union_box(boxes):
    if boxes and not isinstance(boxes[0], (list, tuple)): boxes = [boxes]
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    return [x0, y0, max(b[0] + b[2] for b in boxes) - x0, max(b[1] + b[3] for b in boxes) - y0]


def show_image(ax, im, color, box=None):
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor(color); s.set_linewidth(2.0)
    if box is not None:
        x, y, w, h = box
        ax.add_patch(patches.Rectangle((x, y), w, h, lw=2.0, edgecolor="#D1495B", facecolor="none"))


def main():
    pope = json.load(open(WORK_DIR + "/ex_pope.json"))
    vstar = json.load(open(WORK_DIR + "/ex_vstar.json"))
    pe = pick_pope(pope); ve = pick_vstar(vstar)
    assert pe and ve, f"no example (pope={bool(pe)}, vstar={bool(ve)})"
    print("PRIOR-DRIVEN:", pe["image"], "|", pe["query"], "| low", pe["low"], "dola", pe["low_dola"], "high", pe["high"])
    print("PERCEPTION  :", os.path.basename(ve["image"]), "|", ve["question"][:50],
          "| low", LETTERS[ve["low"]], "dola", LETTERS[ve["low_dola"]], "high", LETTERS[ve["high"]], "gt", LETTERS[ve["gt"]])

    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.4], hspace=0.32, wspace=0.06)

    # ---- Panel A: prior-driven (decoding fixes) ----
    imA = Image.open(os.path.join(POPE_IMG, pe["image"])).convert("RGB")
    axiA = fig.add_subplot(gs[0, 0]); show_image(axiA, imA, PRIOR)
    axA = fig.add_subplot(gs[0, 1]); axA.axis("off")
    header(axA, PRIOR, "PRIOR-DRIVEN  ·  decoding (DoLa) fixes it")
    axA.text(0.0, 0.80, f"Q:  {pe['query']}", transform=axA.transAxes, fontsize=9.5, color=INK, va="center")
    axA.text(0.0, 0.66, "Ground truth", transform=axA.transAxes, fontsize=9.5, color=MUTE, va="center")
    chip(axA, 0.30, 0.66, "NO", GT)
    axA.text(0.0, 0.50, "low-res", transform=axA.transAxes, fontsize=9.8, fontweight="bold", color=INK, va="center")
    badge(axA, 0.17, 0.50, "256 tok"); chip(axA, 0.40, 0.50, pe["low"].upper(), BAD)
    axA.text(0.56, 0.50, f"conf {pe['low_conf']:.2f}  (hallucinated)", transform=axA.transAxes, fontsize=8.8, color=MUTE, va="center")
    axA.text(0.0, 0.36, "+ DoLa", transform=axA.transAxes, fontsize=9.8, fontweight="bold", color=PRIOR, va="center")
    chip(axA, 0.40, 0.36, pe["low_dola"].upper(), OK); axA.text(0.56, 0.36, "decoding fixes it", transform=axA.transAxes, fontsize=8.8, color=MUTE, va="center")
    axA.text(0.0, 0.22, "+ high-res", transform=axA.transAxes, fontsize=9.8, fontweight="bold", color=INK, va="center")
    chip(axA, 0.40, 0.22, pe["high"].upper(), OK if pe["high"] == "no" else BAD)
    axA.text(0.56, 0.22, "resolution " + ("helps" if pe["high"] == "no" else "does not fix it"), transform=axA.transAxes, fontsize=8.8, color=MUTE, va="center")
    axA.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.085, boxstyle="round,pad=0,rounding_size=0.016",
                                 transform=axA.transAxes, fc="#fdf3e6", ec=PRIOR, lw=1.1, clip_on=False))
    axA.text(0.02, 0.042, "Absent object asserted from a co-occurrence prior — a language-stage error.",
             transform=axA.transAxes, fontsize=8.8, color="#9a6313", va="center")

    # ---- Panel B: perception-bound (resolution fixes) ----
    imB = Image.open(ve["image"]).convert("RGB")
    axiB = fig.add_subplot(gs[1, 0]); show_image(axiB, imB, PERC, box=union_box(ve["bbox"]))
    axB = fig.add_subplot(gs[1, 1]); axB.axis("off")
    header(axB, PERC, "PERCEPTION-BOUND  ·  perception (resolution) fixes it")
    axB.text(0.0, 0.80, f"Q:  {ve['question']}", transform=axB.transAxes, fontsize=9.0, color=INK, va="top")
    axB.text(0.0, 0.64, f"Answer: {LETTERS[ve['gt']]}. {ve['options'][ve['gt']][:40]}", transform=axB.transAxes,
             fontsize=8.6, color=OK["tc"], fontweight="bold", va="center")
    axB.text(0.0, 0.50, "low-res", transform=axB.transAxes, fontsize=9.8, fontweight="bold", color=INK, va="center")
    badge(axB, 0.17, 0.50, "256 tok"); chip(axB, 0.40, 0.50, LETTERS[ve["low"]], BAD)
    axB.text(0.55, 0.50, f"conf {ve['low_conf']:.2f}  (wrong)", transform=axB.transAxes, fontsize=8.8, color=MUTE, va="center")
    axB.text(0.0, 0.36, "+ DoLa", transform=axB.transAxes, fontsize=9.8, fontweight="bold", color=INK, va="center")
    chip(axB, 0.40, 0.36, LETTERS[ve["low_dola"]], BAD); axB.text(0.55, 0.36, "decoding cannot fix it", transform=axB.transAxes, fontsize=8.8, color=MUTE, va="center")
    axB.text(0.0, 0.22, "+ high-res", transform=axB.transAxes, fontsize=9.8, fontweight="bold", color=PERC, va="center")
    chip(axB, 0.40, 0.22, LETTERS[ve["high"]], OK); axB.text(0.55, 0.22, "perception recovers it", transform=axB.transAxes, fontsize=8.8, color=MUTE, va="center")
    axB.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.085, boxstyle="round,pad=0,rounding_size=0.016",
                                 transform=axB.transAxes, fc="#eef6f4", ec=PERC, lw=1.1, clip_on=False))
    axB.text(0.02, 0.042, "Tiny target unresolved at low res — a perception-stage error (red box).",
             transform=axB.transAxes, fontsize=8.8, color="#1c6f63", va="center")

    out = os.path.join(HERE, "..", "draft", "figures", "fig_two_modes.pdf")
    fig.savefig(out); print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
