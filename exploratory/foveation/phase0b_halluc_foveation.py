#!/usr/bin/env python
"""
Foveation line, hallucination pivot — oracle feasibility test (does foveation cut absent-object hallucination?).

POPE-adversarial NEGATIVES: "Is there a {X}?" where X is ABSENT but co-occurs with present objects,
so the model hallucinates "yes". There is no GT box for X (it's absent) -- so the oracle "where to
look" is the set of PRESENT-object boxes (COCO): if foveating the right scene region lets the model
verify X's absence and flip "yes"->"no", foveation is the lever for hallucination too.

Per negative query (Qwen3-VL): low-res, high-res, and foveate each present-object region (low full +
region crop). ORACLE = "no" if low=no OR any region-foveation=no. Reports HALLUCINATION RATE (yes-rate
on negatives) for low / high / oracle-region-foveation, and tokens. GO if oracle << low at <= high tokens.
Run in 'qwen3' env.
"""
import argparse, json, os, re, time
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


TOKPX = 32 * 32


def resize_tokens(im, n):
    W, H = im.size; s = (n * TOKPX / max(1, W * H)) ** 0.5
    return im.resize((max(56, int(round(W * s))), max(56, int(round(H * s)))), Image.LANCZOS)


def expand_xyxy(b, W, H, pad=0.35, minfrac=0.12):
    x, y, w, h = b; cx, cy = x + w / 2, y + h / 2
    w2, h2 = max(w * (1 + 2 * pad), W * minfrac), max(h * (1 + 2 * pad), H * minfrac)
    return (int(max(0, cx - w2 / 2)), int(max(0, cy - h2 / 2)), int(min(W, cx + w2 / 2)), int(min(H, cy + h2 / 2)))


def yes_no_ids(tok):
    yes, no = set(), set()
    for w in ["yes", "Yes", " yes", " Yes", "YES"]:
        i = tok.encode(w, add_special_tokens=False);  yes.add(i[0]) if i else None
    for w in ["no", "No", " no", " No", "NO"]:
        i = tok.encode(w, add_special_tokens=False);  no.add(i[0]) if i else None
    return sorted(yes), sorted(no)


@torch.no_grad()
def ask(model, proc, images, q, device, yn, note=""):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": q + note}]
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=images, return_tensors="pt").to(device)
    logits = model(**inp).logits[0, -1].float(); p = torch.softmax(logits, -1)
    g = inp["image_grid_thw"]; m = getattr(proc.image_processor, "merge_size", 2)
    ntok = int(sum((g[i].prod() // (m ** 2)).item() for i in range(g.shape[0])))
    return ("yes" if p[yn[0]].sum() >= p[yn[1]].sum() else "no"), ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--pope", default=HARAM_ROOT + "/coco_build/data/pope_test_adversarial.json")
    ap.add_argument("--ann", default=HARAM_ROOT + "/coco_build/annotations/instances_val2014.json")
    ap.add_argument("--image-dir", default=HARAM_ROOT + "/coco_build/images")
    ap.add_argument("--low", type=int, default=64); ap.add_argument("--high", type=int, default=1024)
    ap.add_argument("--crop", type=int, default=512); ap.add_argument("--max-regions", type=int, default=4)
    ap.add_argument("--limit", type=int, default=600); ap.add_argument("--output", required=True)
    ap.add_argument("--random-regions", action="store_true",
                    help="CONTROL: foveate K random boxes instead of present-object boxes")
    args = ap.parse_args()
    import random as _rnd; _rng = _rnd.Random(0)
    device = "cuda"
    print("loading COCO boxes...", flush=True)
    d = json.load(open(args.ann)); cn = {c["id"]: c["name"] for c in d["categories"]}
    img2boxes = defaultdict(list)
    for a in d["annotations"]:
        if not a.get("iscrowd", 0):
            img2boxes[a["image_id"]].append(a["bbox"])   # all present-object boxes
    rows = [json.loads(l) for l in open(args.pope) if l.strip()]
    negs = [r for r in rows if r["label"] == "no"][: args.limit]   # the hallucination cases
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    yn = yes_no_ids(proc.tokenizer)
    note = " The first image is the full scene; the second is a high-resolution zoom of part of it."
    print(f"[halluc-fov] {len(negs)} adversarial NEGATIVES | low={args.low} high={args.high} crop={args.crop}", flush=True)

    recs = []; t0 = time.time()
    for i, r in enumerate(negs):
        iid = int(re.search(r"(\d+)\.jpg", r["image"]).group(1))
        im = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB"); W, H = im.size
        if args.random_regions:                       # CONTROL: K random boxes (size ~ real objects)
            boxes = []
            for _ in range(args.max_regions):
                bw, bh = _rng.uniform(0.10, 0.30) * W, _rng.uniform(0.10, 0.30) * H
                boxes.append([_rng.uniform(0, W - bw), _rng.uniform(0, H - bh), bw, bh])
        else:
            boxes = img2boxes.get(iid, [])
            boxes = sorted(boxes, key=lambda b: -b[2] * b[3])[: args.max_regions]   # largest present objects
        full_lo, full_hi = resize_tokens(im, args.low), resize_tokens(im, args.high)
        p_lo, t_lo = ask(model, proc, [full_lo], r["text"], device, yn)
        p_hi, t_hi = ask(model, proc, [full_hi], r["text"], device, yn)
        region_preds = []
        for b in boxes:
            crop = resize_tokens(im.crop(expand_xyxy(b, W, H)), args.crop)
            pr, _ = ask(model, proc, [full_lo, crop], r["text"], device, yn, note)
            region_preds.append(pr)
        oracle = "no" if (p_lo == "no" or any(pr == "no" for pr in region_preds)) else "yes"
        recs.append({"low": p_lo, "high": p_hi, "oracle": oracle, "n_regions": len(boxes),
                     "t_low": t_lo, "t_high": t_hi, "t_fov": args.low + args.crop})
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(negs)} ({time.time()-t0:.0f}s)", flush=True)

    yr = lambda k: float(np.mean([x[k] == "yes" for x in recs]))   # yes-rate on negatives = hallucination rate
    summ = {"n": len(recs),
            "hallucination_rate": {"low": yr("low"), "high": yr("high"), "oracle_fov": yr("oracle")},
            "tokens": {"low": float(np.mean([x["t_low"] for x in recs])),
                       "high": float(np.mean([x["t_high"] for x in recs])),
                       "fov_per_region": float(np.mean([x["t_fov"] for x in recs]))}}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump({"summary": summ, "records": recs}, open(args.output, "w"), indent=2)
    print("\n=== Hallucination-foveation oracle test (POPE-adversarial negatives) ===")
    h = summ["hallucination_rate"]
    print(f"  uniform-low  : halluc {h['low']:.3f}   tok {summ['tokens']['low']:.0f}")
    print(f"  uniform-high : halluc {h['high']:.3f}   tok {summ['tokens']['high']:.0f}")
    print(f"  ORACLE-region-foveation: halluc {h['oracle_fov']:.3f}   tok ~{summ['tokens']['fov_per_region']:.0f}/region")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
