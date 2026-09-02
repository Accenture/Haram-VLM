#!/usr/bin/env python
"""Conference-quality qualitative POPE figure: the scout-and-escalate controller's three decision
regimes (Qwen3-VL). Aligns adaptive-Pareto records with the POPE test json by order and renders one
clean example card per regime:
  (1) KEEP      : scout confident AND correct -> stay low-res, save tokens.
  (2) ESCALATE  : scout uncertain -> escalate -> high-res fixes it.
  (3) CONFIDENT MISTAKE : scout confident BUT wrong -> not escalated -> residual failure (Sec. 6.4).
Layout: framed, uniformly-sized image (left) + a styled annotation panel (right) with a colored
regime header, answer chips, and token badges. Run in the 'haram' env."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                 "figure.dpi": 240, "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
HERE = os.path.dirname(os.path.abspath(__file__))
IMG = HARAM_ROOT + "/coco_build/images"
RESULTS = os.environ.get("HARAM_RESULTS", HARAM_ROOT + "/results")
PARETO = RESULTS + "/adaptive_pareto_qwen3/adaptive_{}.json"
POPE = HARAM_ROOT + "/coco_build/data/pope_test_{}.json"

# palette
INK, MUTE = "#1c1c1c", "#6f6f6f"
REG = {"keep": "#2A9D8F", "escalate": "#E08E1B", "mistake": "#D1495B"}
REG_TITLE = {"keep": "KEEP  ·  scout confident and correct",
             "escalate": "ESCALATE  ·  scout uncertain",
             "mistake": "CONFIDENT MISTAKE  ·  not escalated"}
OK_CHIP = dict(fc="#dff3ef", ec="#2A9D8F", tc="#1c6f63")
BAD_CHIP = dict(fc="#fbe2e6", ec="#D1495B", tc="#a32d3e")
GT_CHIP = dict(fc="#eeeeee", ec="#c9c9c9", tc="#3a3a3a")
TOK_BG, TOK_EC, TOK_TC = "#eef2f7", "#cdd7e3", "#42566e"


def load(split):
    for path, hint in ((PARETO.format(split), "run controller/adaptive_infer_qwen3.py, or set HARAM_RESULTS"),
                       (POPE.format(split), "run protocol/generate_coco_pope.py -- see DATA.md \u00a72")):
        if not os.path.exists(path):
            sys.exit(f"missing {path}\n  -> {hint}")
    if not os.path.isdir(IMG):
        sys.exit(f"missing COCO images at {IMG}\n  -> see DATA.md \u00a71")
    recs = json.load(open(PARETO.format(split)))["records"]
    pope = [json.loads(l) for l in open(POPE.format(split)) if l.strip()][:len(recs)]
    assert all(r["label"] == p["label"] for r, p in zip(recs, pope)), "misaligned"
    return [{**p, **r} for r, p in zip(recs, pope)]


def pick(items, cond, key):
    return sorted([x for x in items if cond(x)], key=key)


def fit(path, tw=640, th=470, bg=(247, 247, 247)):
    """Resize preserving aspect, pad onto a fixed canvas so every image cell is identical."""
    im = Image.open(path).convert("RGB"); im.thumbnail((tw, th), Image.LANCZOS)
    canvas = Image.new("RGB", (tw, th), bg)
    canvas.paste(im, ((tw - im.width) // 2, (th - im.height) // 2))
    return canvas


def chip(ax, x, y, txt, c, fs=10.5, weight="bold"):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=fs, fontweight=weight, color=c["tc"],
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.32", fc=c["fc"], ec=c["ec"], lw=0.9))


def badge(ax, x, y, txt):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=8.8, color=TOK_TC, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.28", fc=TOK_BG, ec=TOK_EC, lw=0.8))


def draw_panel(ax, ex, regime):
    ax.axis("off")
    col = REG[regime]
    correct_low = ex["pred_low"] == ex["label"]
    escalate = ex["conf_low"] <= 0.9
    ans_chip = lambda ok: OK_CHIP if ok else BAD_CHIP
    # header band
    ax.add_patch(FancyBboxPatch((0.0, 0.9), 1.0, 0.115,
                                boxstyle="round,pad=0,rounding_size=0.018",
                                transform=ax.transAxes, fc=col, ec="none", clip_on=False))
    ax.text(0.025, 0.957, REG_TITLE[regime], transform=ax.transAxes, color="white",
            fontsize=10.5, fontweight="bold", va="center", ha="left")
    # question
    ax.text(0.0, 0.80, "Q:", transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color=INK, va="center")
    ax.text(0.065, 0.80, ex["text"], transform=ax.transAxes, fontsize=10.5, color=INK, va="center")
    # ground truth
    ax.text(0.0, 0.675, "Ground truth", transform=ax.transAxes, fontsize=10, color=MUTE, va="center")
    chip(ax, 0.30, 0.675, ex["label"].upper(), GT_CHIP)
    # scout (low-res)
    ax.text(0.0, 0.525, "Scout", transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color=INK, va="center")
    badge(ax, 0.16, 0.525, f"{ex['tok_low']} tok")
    chip(ax, 0.40, 0.525, ex["pred_low"].upper(), ans_chip(correct_low))
    ax.text(0.59, 0.525, f"conf {ex['conf_low']:.2f}", transform=ax.transAxes, fontsize=9.8,
            color=MUTE, va="center")
    # decision
    decision = "escalate" if escalate else "keep low-res"
    ax.text(0.0, 0.40, f"risk {1-ex['conf_low']:.2f}", transform=ax.transAxes, fontsize=9.8,
            color=MUTE, va="center")
    ax.annotate("", xy=(0.34, 0.40), xytext=(0.22, 0.40), transform=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
    ax.text(0.37, 0.40, decision, transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=col, va="center")
    # high-res row (only when escalated)
    if escalate:
        ax.text(0.0, 0.275, "High-res", transform=ax.transAxes, fontsize=10.5, fontweight="bold",
                color=INK, va="center")
        badge(ax, 0.19, 0.275, f"{ex['tok_high']} tok")
        chip(ax, 0.43, 0.275, ex["pred_high"].upper(), ans_chip(ex["pred_high"] == ex["label"]))
        ax.text(0.62, 0.275, f"conf {ex['conf_high']:.2f}", transform=ax.transAxes, fontsize=9.8,
                color=MUTE, va="center")
    # outcome
    if escalate:
        final_ok = ex["pred_high"] == ex["label"]
        msg = "correct after escalation" if final_ok else "still wrong after escalation"
        fc = REG["keep"] if final_ok else REG["mistake"]
        out = f"Final: {ex['pred_high'].upper()} — {msg}"
    elif correct_low:
        fc, out = REG["keep"], f"Final: {ex['pred_low'].upper()} — correct, tokens saved"
    else:
        fc = REG["mistake"]
        out = (f"Final: {ex['pred_low'].upper()} — wrong; high-res would say "
               f"{ex['pred_high'].upper()}")
    ax.add_patch(FancyBboxPatch((0.0, 0.075), 1.0, 0.085,
                                boxstyle="round,pad=0,rounding_size=0.016",
                                transform=ax.transAxes, fc="#f6f6f6", ec=fc, lw=1.1, clip_on=False))
    ax.text(0.025, 0.117, out, transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=fc, va="center", ha="left")


def main():
    data = {s: load(s) for s in ["random", "popular", "adversarial"]}
    keep = pick(data["random"],
                lambda x: x["pred_low"] == x["label"] and x["pred_high"] == x["label"]
                          and x["conf_low"] >= 0.99 and x["label"] == "yes" and "person" not in x["text"],
                key=lambda x: -x["conf_low"])
    esc = pick(data["popular"] + data["adversarial"],
               lambda x: x["pred_low"] != x["label"] and x["pred_high"] == x["label"]
                         and x["conf_low"] <= 0.70,
               key=lambda x: x["conf_low"])
    cm = pick(data["adversarial"],
              lambda x: x["pred_low"] != x["label"] and x["conf_low"] >= 0.99
                        and x["label"] == "no" and x["pred_low"] == "yes"
                        and x["pred_high"] == x["label"] and "person" not in x["text"],
              key=lambda x: -x["conf_low"])
    chosen = [("keep", keep[0]), ("escalate", esc[0]), ("mistake", cm[0])]
    for reg, c in chosen:
        print(f"{reg:9} {c['image']}  '{c['text']}' gt={c['label']} "
              f"low={c['pred_low']}(c{c['conf_low']:.2f}) high={c['pred_high']}(c{c['conf_high']:.2f})")

    n = len(chosen)
    fig = plt.figure(figsize=(7.4, 2.35 * n))
    gs = fig.add_gridspec(n, 2, width_ratios=[1.0, 1.5], hspace=0.30, wspace=0.06)
    for i, (reg, c) in enumerate(chosen):
        axi = fig.add_subplot(gs[i, 0])
        axi.imshow(fit(os.path.join(IMG, c["image"]))); axi.set_xticks([]); axi.set_yticks([])
        for s in axi.spines.values():
            s.set_edgecolor(REG[reg]); s.set_linewidth(2.0)
        axp = fig.add_subplot(gs[i, 1])
        draw_panel(axp, c, reg)
    fig.savefig(os.path.join(HERE, "fig_pope_qual.pdf"))
    print("wrote fig_pope_qual.pdf")


if __name__ == "__main__":
    main()
