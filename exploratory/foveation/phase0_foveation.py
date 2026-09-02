#!/usr/bin/env python
"""
Foveation Phase 0: ORACLE foveation feasibility (go/no-go for "look closer WHERE needed").

Tests H1: concentrating high-resolution tokens on the query-relevant region (using GROUND-TRUTH
COCO/V*Bench boxes) reduces error at equal-or-lower visual-token cost vs. uniform allocation.

Three conditions per item (Qwen3-VL-8B):
  uniform-low     : whole image at a low token budget.
  uniform-high    : whole image at a high token budget  (the cost reference).
  oracle-foveated : whole image at low budget  +  GT-region crop at high budget
                    (two images; total tokens designed <= uniform-high).

Modes:
  --mode pope  : POPE present-object subset (label=yes, SMALL GT box) -> low-res misses, foveation fixes.
  --mode vstar : V*Bench tiny-target MCQ (GT target box) -> the paper's V*Bench boundary case (Sec. 6.5).

Reports per-condition accuracy and mean visual tokens. Run in the 'qwen3' env.
"""
import argparse, glob, json, os, re, time, random
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


TOKPX = 32 * 32  # Qwen3-VL: 16px patch, 2x2 merge -> one merged visual token ~ a 32x32 px region


def resize_tokens(im, n_tok):
    """Resize (preserve aspect) so the image is ~n_tok merged visual tokens."""
    W, H = im.size
    s = (n_tok * TOKPX / max(1, W * H)) ** 0.5
    return im.resize((max(56, int(round(W * s))), max(56, int(round(H * s)))), Image.LANCZOS)


def union_box(boxes):
    """[[x,y,w,h],...] (or a single [x,y,w,h]) -> one enclosing [x,y,w,h]."""
    if boxes and not isinstance(boxes[0], (list, tuple)):
        return list(boxes)
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def expand_box(b, W, H, pad=0.35, minfrac=0.12):
    """Expand [x,y,w,h] by `pad` on each side; enforce a minimum crop size; clip to image."""
    x, y, w, h = b
    cx, cy = x + w / 2, y + h / 2
    w2, h2 = max(w * (1 + 2 * pad), W * minfrac), max(h * (1 + 2 * pad), H * minfrac)
    x0, y0 = max(0, cx - w2 / 2), max(0, cy - h2 / 2)
    x1, y1 = min(W, cx + w2 / 2), min(H, cy + h2 / 2)
    return int(x0), int(y0), int(x1), int(y1)


def yes_no_ids(tok):
    yes, no = set(), set()
    for w in ["yes", "Yes", " yes", " Yes", "YES"]:
        i = tok.encode(w, add_special_tokens=False)
        if i: yes.add(i[0])
    for w in ["no", "No", " no", " No", "NO"]:
        i = tok.encode(w, add_special_tokens=False)
        if i: no.add(i[0])
    return sorted(yes), sorted(no)


@torch.no_grad()
def run(model, proc, images, prompt, device, letter_ids=None, yn=None):
    """images: list of PIL (already resized). Returns (pred, ntok_total)."""
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=images, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = out.logits[0, -1].float()
    g = inputs["image_grid_thw"]  # [num_images, 3]
    merge = getattr(proc.image_processor, "merge_size", 2)
    ntok = int(sum((g[i].prod() // (merge ** 2)).item() for i in range(g.shape[0])))
    if yn is not None:
        yes_ids, no_ids = yn
        p = torch.softmax(logits, -1)
        pred = "yes" if p[yes_ids].sum() >= p[no_ids].sum() else "no"
        return pred, ntok
    else:  # MCQ: argmax over letter tokens
        sub = logits[letter_ids]
        return int(torch.argmax(sub).item()), ntok


# ----------------------------- data loaders -----------------------------
def load_pope(ann_path, pope_glob, image_dir, limit, max_area_frac=0.06):
    print("loading COCO val2014 boxes...", flush=True)
    d = json.load(open(ann_path))
    catid2name = {c["id"]: c["name"] for c in d["categories"]}
    img2cat2boxes = defaultdict(lambda: defaultdict(list))
    for a in d["annotations"]:
        img2cat2boxes[a["image_id"]][catid2name[a["category_id"]]].append(a["bbox"])
    pat = re.compile(r"[Ii]s there an? (.+?) in the image", re.I)
    items = []
    for pf in sorted(glob.glob(pope_glob)):
        for line in open(pf):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["label"] != "yes":      # present-object subset (foveation has a target)
                continue
            m = pat.search(r["text"])
            if not m:
                continue
            cat = m.group(1).strip().lower()
            iid = int(re.search(r"(\d+)\.jpg", r["image"]).group(1))
            boxes = img2cat2boxes.get(iid, {}).get(cat, [])
            if not boxes:
                continue
            ip = os.path.join(image_dir, r["image"])
            if not os.path.exists(ip):
                continue
            W, H = Image.open(ip).size
            small = [b for b in boxes if (b[2] * b[3]) / (W * H) <= max_area_frac]
            if not small:
                continue
            box = min(small, key=lambda b: b[2] * b[3])  # smallest present instance (hardest)
            items.append({"image": ip, "question": r["text"], "label": "yes", "box": box})
    rng = random.Random(0); rng.shuffle(items)
    return items[:limit] if limit else items


def load_vstar(root, limit):
    items = []
    for sub in ["direct_attributes", "relative_position"]:
        for jf in sorted(glob.glob(os.path.join(root, sub, "*.json"))):
            j = json.load(open(jf)); img = None
            for ext in (".jpg", ".JPG", ".png", ".jpeg", ".webp", ".JPEG"):
                if os.path.exists(jf[:-5] + ext):
                    img = jf[:-5] + ext; break
            if img and j.get("options") and j.get("bbox"):
                items.append({"image": img, "question": j["question"], "options": j["options"],
                              "bbox": j["bbox"]})
    return items[:limit] if limit else items


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--mode", choices=["pope", "vstar"], required=True)
    ap.add_argument("--ann", default=HARAM_ROOT + "/coco_build/annotations/instances_val2014.json")
    ap.add_argument("--pope-glob", default=HARAM_ROOT + "/coco_build/data/pope_test_random.json")
    ap.add_argument("--image-dir", default=HARAM_ROOT + "/coco_build/images")
    ap.add_argument("--vstar-root", default="")
    ap.add_argument("--low", type=int, default=64)     # merged tokens for low-res full image
    ap.add_argument("--high", type=int, default=1024)  # merged tokens for uniform-high full image
    ap.add_argument("--crop", type=int, default=512)   # merged tokens for the high-res region crop
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    proc.image_processor.min_pixels = 16 * TOKPX
    proc.image_processor.max_pixels = 4096 * TOKPX
    LETTERS = "ABCD"
    yn = yes_no_ids(proc.tokenizer) if args.mode == "pope" else None
    letter_ids = ([proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]
                  if args.mode == "vstar" else None)

    items = (load_pope(args.ann, args.pope_glob, args.image_dir, args.limit) if args.mode == "pope"
             else load_vstar(args.vstar_root, args.limit))
    print(f"[{args.mode}] {len(items)} items | low={args.low} high={args.high} crop={args.crop} tok", flush=True)

    recs = []
    t0 = time.time()
    rng = random.Random(0)
    for i, it in enumerate(items):
        im = Image.open(it["image"]).convert("RGB"); W, H = im.size
        if args.mode == "pope":
            q = it["question"]; gt = it["label"]
            x0, y0, x1, y1 = expand_box(it["box"], W, H)
            qa = lambda imgs, pr: run(model, proc, imgs, pr, device, yn=yn)
            qtext = q
            crop_note = (" The first image is the full scene (low resolution); the second is a "
                         "high-resolution zoom of the relevant region.")
        else:
            opts = list(it["options"]); correct = opts[0]; rng.shuffle(opts); gt = opts.index(correct)
            x0, y0, x1, y1 = expand_box(union_box(it["bbox"]), W, H)
            letters_txt = "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts))
            qtext = f"{it['question']}\n{letters_txt}\nAnswer with the letter only."
            qa = lambda imgs, pr: run(model, proc, imgs, pr, device, letter_ids=letter_ids)
            crop_note = (" The first image is the full scene (low resolution); the second is a "
                         "high-resolution zoom of the relevant region.")
        crop = im.crop((x0, y0, x1, y1))
        full_lo = resize_tokens(im, args.low)
        full_hi = resize_tokens(im, args.high)
        crop_hi = resize_tokens(crop, args.crop)

        p_lo, t_lo = qa([full_lo], qtext)
        p_hi, t_hi = qa([full_hi], qtext)
        p_fv, t_fv = qa([full_lo, crop_hi], qtext + crop_note)
        recs.append({"gt": gt, "low": p_lo, "high": p_hi, "fov": p_fv,
                     "t_low": t_lo, "t_high": t_hi, "t_fov": t_fv})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    def acc(key):
        return float(np.mean([r[key] == r["gt"] for r in recs]))
    def mt(key):
        return float(np.mean([r[key] for r in recs]))
    summary = {"mode": args.mode, "n": len(recs),
               "acc": {"low": acc("low"), "high": acc("high"), "fov": acc("fov")},
               "tokens": {"low": mt("t_low"), "high": mt("t_high"), "fov": mt("t_fov")}}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump({"summary": summary, "records": recs}, open(args.output, "w"), indent=2)
    print("\n=== Phase 0 result ===")
    print(f"  uniform-low :  acc {summary['acc']['low']:.3f}   tokens {summary['tokens']['low']:.0f}")
    print(f"  uniform-high:  acc {summary['acc']['high']:.3f}   tokens {summary['tokens']['high']:.0f}")
    print(f"  ORACLE-fov  :  acc {summary['acc']['fov']:.3f}   tokens {summary['tokens']['fov']:.0f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
