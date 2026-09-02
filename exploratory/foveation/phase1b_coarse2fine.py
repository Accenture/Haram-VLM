#!/usr/bin/env python
"""
Foveation Phase 1b (cheap, TRAINING-FREE): does coarse-to-fine self-grounding beat single-shot?

Per V*Bench item (Qwen3-VL-8B), compares on the SAME run:
  uniform-high      : whole image @ high.
  oracle-fov        : survey + GT-box crop @ high.
  selfground-1step  : survey -> ask bbox -> crop -> answer.
  selfground-2step  : survey -> coarse bbox -> ZOOM a generous region @ higher res ->
                      re-ask bbox in the zoom -> map back -> final crop -> answer.
Reports accuracy + localization (center-in-GT hit, IoU) for 1-step vs 2-step final boxes.
Decides whether iterative zoom (coarse-to-fine) lifts localization without training. Run in 'qwen3' env.
"""
import argparse, glob, json, os, re, time, random
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

TOKPX = 32 * 32
LETTERS = "ABCD"


def resize_tokens(im, n_tok):
    W, H = im.size
    s = (n_tok * TOKPX / max(1, W * H)) ** 0.5
    return im.resize((max(56, int(round(W * s))), max(56, int(round(H * s)))), Image.LANCZOS)


def union_box(boxes):
    if boxes and not isinstance(boxes[0], (list, tuple)):
        return list(boxes)
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def expand_xyxy(b_xywh, W, H, pad=0.35, minfrac=0.12):
    x, y, w, h = b_xywh
    cx, cy = x + w / 2, y + h / 2
    w2, h2 = max(w * (1 + 2 * pad), W * minfrac), max(h * (1 + 2 * pad), H * minfrac)
    return (int(max(0, cx - w2 / 2)), int(max(0, cy - h2 / 2)),
            int(min(W, cx + w2 / 2)), int(min(H, cy + h2 / 2)))


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0); inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def parse_frac(text):
    """Return (fx1,fy1,fx2,fy2) in [0,1] of the passed image, or None."""
    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) < 4:
        return None
    v = [float(x) for x in nums[:4]]
    mx = max(abs(t) for t in v)
    if mx <= 1.5:
        f = v
    elif mx <= 1000.0:
        f = [t / 1000.0 for t in v]
    else:
        return None
    x1, x2 = sorted([f[0], f[2]]); y1, y2 = sorted([f[1], f[3]])
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(1, x2), min(1, y2)
    if x2 - x1 < 0.005 or y2 - y1 < 0.005:
        return None
    return (x1, y1, x2, y2)


@torch.no_grad()
def answer_mcq(model, proc, images, prompt, device, letter_ids):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=images, return_tensors="pt").to(device)
    out = model(**inputs); logits = out.logits[0, -1].float()
    g = inputs["image_grid_thw"]; merge = getattr(proc.image_processor, "merge_size", 2)
    ntok = int(sum((g[i].prod() // (merge ** 2)).item() for i in range(g.shape[0])))
    return int(torch.argmax(logits[letter_ids]).item()), ntok


@torch.no_grad()
def ground(model, proc, im, question, device):
    pr = (f"Question: {question}\nIdentify the SINGLE rectangular image region most relevant to "
          "answering this question. Output ONLY its bounding box as four integers 'x1 y1 x2 y2', "
          "each normalized to 0-1000 (0=left/top, 1000=right/bottom). No other text.")
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": pr}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[im], return_tensors="pt").to(device)
    gen = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    return proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def load_vstar(root, limit):
    items = []
    for sub in ["direct_attributes", "relative_position"]:
        for jf in sorted(glob.glob(os.path.join(root, sub, "*.json"))):
            j = json.load(open(jf)); img = None
            for ext in (".jpg", ".JPG", ".png", ".jpeg", ".webp", ".JPEG"):
                if os.path.exists(jf[:-5] + ext):
                    img = jf[:-5] + ext; break
            if img and j.get("options") and j.get("bbox"):
                items.append({"image": img, "question": j["question"], "options": j["options"], "bbox": j["bbox"]})
    return items[:limit] if limit else items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--vstar-root", required=True)
    ap.add_argument("--survey", type=int, default=256)
    ap.add_argument("--zoom", type=int, default=512)    # generous wide-region re-encode for step 2
    ap.add_argument("--high", type=int, default=1024)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--wide", type=float, default=0.5)  # step-1 zoom region = this fraction of image
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    proc.image_processor.min_pixels = 16 * TOKPX; proc.image_processor.max_pixels = 4096 * TOKPX
    letter_ids = [proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    items = load_vstar(args.vstar_root, args.limit)
    print(f"[phase1b c2f] {len(items)} items | survey={args.survey} zoom={args.zoom} crop={args.crop} wide={args.wide}", flush=True)

    rng = random.Random(0); recs = []; t0 = time.time()
    for i, it in enumerate(items):
        im = Image.open(it["image"]).convert("RGB"); W, H = im.size
        opts = list(it["options"]); correct = opts[0]; rng.shuffle(opts); gt = opts.index(correct)
        qtext = f"{it['question']}\n" + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts)) + \
                "\nAnswer with the letter only."
        note = " The first image is the full scene; the second is a high-resolution zoom of the relevant region."
        gx, gy, gw, gh = union_box(it["bbox"]); gt_raw = (gx, gy, gx + gw, gy + gh)
        gt_xyxy = expand_xyxy([gx, gy, gw, gh], W, H)

        survey = resize_tokens(im, args.survey)
        full_hi = resize_tokens(im, args.high)
        oracle_crop = resize_tokens(im.crop(gt_xyxy), args.crop)

        def box_metrics(xyxy):
            cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
            return iou(xyxy, gt_raw), (gt_raw[0] <= cx <= gt_raw[2] and gt_raw[1] <= cy <= gt_raw[3])

        # --- step 1 (coarse) ---
        f1 = parse_frac(ground(model, proc, survey, it["question"], device))
        if f1:
            b1 = (f1[0]*W, f1[1]*H, f1[2]*W, f1[3]*H)
        else:
            b1 = (0.25*W, 0.25*H, 0.75*W, 0.75*H)
        one_xyxy = expand_xyxy([b1[0], b1[1], b1[2]-b1[0], b1[3]-b1[1]], W, H)
        crop1 = resize_tokens(im.crop(one_xyxy), args.crop)
        iou1, hit1 = box_metrics(one_xyxy)

        # --- step 2 (zoom into a generous region around the coarse guess, re-localize) ---
        c1x, c1y = (b1[0]+b1[2])/2, (b1[1]+b1[3])/2
        ww, wh = W*args.wide, H*args.wide
        wx0, wy0 = max(0, c1x-ww/2), max(0, c1y-wh/2); wx1, wy1 = min(W, c1x+ww/2), min(H, c1y+wh/2)
        zoom_im = resize_tokens(im.crop((int(wx0), int(wy0), int(wx1), int(wy1))), args.zoom)
        f2 = parse_frac(ground(model, proc, zoom_im, it["question"], device))
        if f2:  # map fractional box in the wide-crop frame back to original coords
            b2 = (wx0 + f2[0]*(wx1-wx0), wy0 + f2[1]*(wy1-wy0),
                  wx0 + f2[2]*(wx1-wx0), wy0 + f2[3]*(wy1-wy0))
            two_xyxy = expand_xyxy([b2[0], b2[1], b2[2]-b2[0], b2[3]-b2[1]], W, H)
        else:   # fall back to the wide region
            two_xyxy = (int(wx0), int(wy0), int(wx1), int(wy1))
        crop2 = resize_tokens(im.crop(two_xyxy), args.crop)
        iou2, hit2 = box_metrics(two_xyxy)

        p_hi, t_hi = answer_mcq(model, proc, [full_hi], qtext, device, letter_ids)
        p_or, t_or = answer_mcq(model, proc, [survey, oracle_crop], qtext + note, device, letter_ids)
        p_1, t_1 = answer_mcq(model, proc, [survey, crop1], qtext + note, device, letter_ids)
        p_2, t_2 = answer_mcq(model, proc, [survey, crop2], qtext + note, device, letter_ids)
        recs.append({"gt": gt, "high": p_hi, "oracle": p_or, "s1": p_1, "s2": p_2,
                     "t_high": t_hi, "t_oracle": t_or, "t_s1": t_1, "t_s2": t_2,
                     "iou1": iou1, "hit1": bool(hit1), "iou2": iou2, "hit2": bool(hit2)})
        if (i+1) % 40 == 0:
            print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    def acc(k): return float(np.mean([r[k] == r["gt"] for r in recs]))
    def mt(k): return float(np.mean([r[k] for r in recs]))
    summ = {"n": len(recs),
            "acc": {"uniform_high": acc("high"), "oracle_fov": acc("oracle"),
                    "selfground_1step": acc("s1"), "selfground_2step": acc("s2")},
            "tokens": {"uniform_high": mt("t_high"), "oracle_fov": mt("t_oracle"),
                       "selfground_1step": mt("t_s1"),
                       "selfground_2step_answer": mt("t_s2"),
                       "selfground_2step_total": args.survey + args.zoom + mt("t_s2")},
            "localize": {"center_hit_1step": float(np.mean([r["hit1"] for r in recs])),
                         "center_hit_2step": float(np.mean([r["hit2"] for r in recs])),
                         "mean_iou_1step": float(np.mean([r["iou1"] for r in recs])),
                         "mean_iou_2step": float(np.mean([r["iou2"] for r in recs]))}}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump({"summary": summ, "records": recs}, open(args.output, "w"), indent=2)
    print("\n=== Phase 1b (coarse-to-fine, training-free) ===")
    a = summ["acc"]; L = summ["localize"]
    print(f"  uniform-high     : acc {a['uniform_high']:.3f}")
    print(f"  oracle-fov       : acc {a['oracle_fov']:.3f}")
    print(f"  selfground 1-step: acc {a['selfground_1step']:.3f}   center-hit {L['center_hit_1step']:.2f}  IoU {L['mean_iou_1step']:.3f}")
    print(f"  selfground 2-step: acc {a['selfground_2step']:.3f}   center-hit {L['center_hit_2step']:.2f}  IoU {L['mean_iou_2step']:.3f}")
    print(f"  (2-step total visual tokens ~{summ['tokens']['selfground_2step_total']:.0f} vs uniform-high {summ['tokens']['uniform_high']:.0f})")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
