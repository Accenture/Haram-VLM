#!/usr/bin/env python
"""Conference-quality qualitative V*Bench figure: cases where the low-resolution (1-tile) scout is
CONFIDENTLY WRONG on a tiny target and the high-resolution (24-tile) pass is RIGHT. Aligns result
records with the V*Bench items (same load order + Random(0) shuffle). Styled to match the POPE
qualitative figure: framed image with the target boxed (left) + an annotation panel (right) with a
colored header, multiple-choice options (correct one highlighted), and letter chips for the
low/high predictions. Run in the 'haram' env."""
import glob, json, os, random, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                 "figure.dpi": 240, "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
LETTERS = "ABCDEFGH"
_SNAPS = sorted(glob.glob(HF_CACHE + "/hub/datasets--craigwu--vstar_bench/snapshots/*/"))
if not _SNAPS:
    sys.exit("V*Bench not found under $HF_HOME. Download it first -- see DATA.md \u00a73.")
ROOT = _SNAPS[0]
HERE = os.path.dirname(os.path.abspath(__file__))

INK, MUTE = "#1c1c1c", "#6f6f6f"
ACCENT = "#3D5A80"          # header accent (resolution-decisive theme)
BOX = "#D1495B"             # target bounding box + wrong chip
OK_CHIP = dict(fc="#dff3ef", ec="#2A9D8F", tc="#1c6f63")
BAD_CHIP = dict(fc="#fbe2e6", ec="#D1495B", tc="#a32d3e")
TOK_BG, TOK_EC, TOK_TC = "#eef2f7", "#cdd7e3", "#42566e"


def load_items():
    items = []
    for sub in ["direct_attributes", "relative_position"]:
        for jf in sorted(glob.glob(os.path.join(ROOT, sub, "*.json"))):
            j = json.load(open(jf)); img = None
            for ext in (".jpg", ".JPG", ".png", ".jpeg", ".webp", ".JPEG"):
                if os.path.exists(jf[:-5] + ext):
                    img = jf[:-5] + ext; break
            if img and j.get("options"):
                items.append({"image": img, "question": j["question"], "options": j["options"],
                              "bbox": j.get("bbox", []), "sub": sub})
    return items


def chip(ax, x, y, txt, c, fs=10.5):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=fs, fontweight="bold", color=c["tc"],
            va="center", ha="left", bbox=dict(boxstyle="round,pad=0.32", fc=c["fc"], ec=c["ec"], lw=0.9))


def badge(ax, x, y, txt):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=8.8, color=TOK_TC, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.28", fc=TOK_BG, ec=TOK_EC, lw=0.8))


def draw_panel(ax, c):
    ax.axis("off")
    # header
    ax.add_patch(FancyBboxPatch((0.0, 0.9), 1.0, 0.115, boxstyle="round,pad=0,rounding_size=0.018",
                                transform=ax.transAxes, fc=ACCENT, ec="none", clip_on=False))
    ax.text(0.025, 0.957, f"RESOLUTION DECISIVE  ·  {c['sub'].replace('_', ' ')}",
            transform=ax.transAxes, color="white", fontsize=10.5, fontweight="bold", va="center")
    # question (wrap to width)
    q = textwrap.wrap(c["question"], width=44)
    y = 0.82
    ax.text(0.0, y, "Q:", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=INK, va="top")
    for k, line in enumerate(q[:2]):
        ax.text(0.065, y - 0.050 * k, line, transform=ax.transAxes, fontsize=10.5, color=INK, va="top")
    y -= 0.050 * len(q[:2]) + 0.065
    # options (correct one bold+green; "correct" tag right-aligned so it never truncates)
    for i, o in enumerate(c["opts"]):
        is_gt = (i == c["gt"])
        col = OK_CHIP["tc"] if is_gt else INK
        wt = "bold" if is_gt else "normal"
        body = textwrap.shorten(o, width=42, placeholder="…")
        ax.text(0.03, y, f"{LETTERS[i]}.  {body}", transform=ax.transAxes, fontsize=9.2,
                color=col, fontweight=wt, va="center")
        if is_gt:
            ax.text(0.99, y, "correct", transform=ax.transAxes, fontsize=8.2, fontweight="bold",
                    color=OK_CHIP["tc"], va="center", ha="right")
        y -= 0.070
    y -= 0.025
    # scout (1 tile) -- CONFIDENTLY WRONG: high confidence => the controller does NOT escalate (the failure)
    ax.text(0.0, y, "Scout", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=INK, va="center")
    badge(ax, 0.16, y, "1 tile")
    chip(ax, 0.36, y, LETTERS[c["pl"]], BAD_CHIP)
    ax.text(0.50, y, f"conf {c['cl']:.2f}", transform=ax.transAxes, fontsize=9.8, color=MUTE, va="center")
    ax.annotate("", xy=(0.81, y), xytext=(0.69, y), transform=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color=BAD_CHIP["ec"], lw=1.6))
    ax.text(0.825, y, "not escalated", transform=ax.transAxes, fontsize=9.2, fontweight="bold",
            color=BAD_CHIP["tc"], va="center")
    y -= 0.10
    # high-res (24 tiles) -- what high resolution WOULD recover (oracle / only if escalated)
    ax.text(0.0, y, "High-res", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=INK, va="center")
    badge(ax, 0.19, y, "24 tiles")
    chip(ax, 0.39, y, LETTERS[c["ph"]], OK_CHIP)
    ax.text(0.53, y, "would recover (oracle)", transform=ax.transAxes, fontsize=9.0, color=MUTE, va="center")
    # outcome -- the boundary-case failure of the training-free signal
    ax.add_patch(FancyBboxPatch((0.0, 0.02), 1.0, 0.085, boxstyle="round,pad=0,rounding_size=0.016",
                                transform=ax.transAxes, fc="#f6f6f6", ec=BAD_CHIP["ec"], lw=1.1, clip_on=False))
    ax.text(0.025, 0.062, "Confident mistake — not escalated; high-res recovers it",
            transform=ax.transAxes, fontsize=9.0, fontweight="bold", color=BAD_CHIP["tc"], va="center")


def main():
    items = load_items()
    recs = json.load(open(WORK_DIR + "/vstar_result.json"))["records"]
    assert len(items) == len(recs), f"{len(items)} items vs {len(recs)} records"
    rng = random.Random(0)
    cands = []
    for it, r in zip(items, recs):
        opts = list(it["options"]); correct = opts[0]; rng.shuffle(opts); gt = opts.index(correct)
        assert gt == r["gt"]
        if r["pred_low"] != gt and r["pred_high"] == gt:
            cands.append({**it, "opts": opts, "gt": gt, "pl": r["pred_low"], "ph": r["pred_high"],
                          "cl": r["conf_low"]})
    cands.sort(key=lambda c: -c["cl"])
    rel = [c for c in cands if c["sub"] == "relative_position"]
    att = [c for c in cands if c["sub"] == "direct_attributes"]
    chosen = (rel[:2] + att[:1])[:3] if len(rel) >= 2 else cands[:3]
    print("chosen:", [(c["sub"], round(c["cl"], 2), c["question"][:40]) for c in chosen])

    n = len(chosen)
    fig = plt.figure(figsize=(7.4, 2.75 * n))
    gs = fig.add_gridspec(n, 2, width_ratios=[1.0, 1.32], hspace=0.26, wspace=0.06)
    for row, c in enumerate(chosen):
        im = Image.open(c["image"]).convert("RGB"); W, H = im.size
        axi = fig.add_subplot(gs[row, 0]); axi.imshow(im); axi.set_xticks([]); axi.set_yticks([])
        axi.set_box_aspect(H / W); axi.set_anchor("N")
        for b in c["bbox"]:
            x, y, w, h = b
            axi.add_patch(patches.Rectangle((x, y), w, h, lw=2.0, edgecolor=BOX, facecolor="none"))
        for s in axi.spines.values():
            s.set_edgecolor(ACCENT); s.set_linewidth(2.0)
        axi.set_title(f"{W}$\\times${H}px · target boxed", fontsize=8.5, color=MUTE, pad=3)
        axp = fig.add_subplot(gs[row, 1]); draw_panel(axp, c)
    fig.savefig(os.path.join(HERE, "fig_vstar_qual.pdf"))
    print("wrote fig_vstar_qual.pdf")


if __name__ == "__main__":
    main()
