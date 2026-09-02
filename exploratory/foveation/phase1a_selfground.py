#!/usr/bin/env python
"""
Foveation Phase 1a: TRAINING-FREE self-grounding foveation (does the VLM know where to look?).

For each V*Bench item (Qwen3-VL-8B):
  1. survey pass at mid-res (~`--survey` merged tokens);
  2. ASK the model for the bounding box of the region most relevant to the question (generate);
  3. foveate that predicted region (crop -> high res) and re-answer (survey + crop).
Compares: uniform-high (ref), ORACLE-foveated (survey + GT-box crop), SELF-GROUNDED-foveated
(survey + predicted-box crop). Also reports localization quality vs GT: mean IoU, center-in-GT
hit rate, and box parse-success rate. Decides whether a learned localizer (Phase 1b) is needed.
Run in the 'qwen3' env.
"""
import argparse, glob, json, os, re, time, random
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

TOKPX = 32 * 32  # Qwen3-VL merged visual token ~ 32x32 px
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


def iou(a, b):  # a,b = (x1,y1,x2,y2)
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def parse_box(text, W, H):
    """Parse 4 numbers; honor the '0-1000 normalized' instruction; return (x1,y1,x2,y2) px or None."""
    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) < 4:
        return None
    v = [float(x) for x in nums[:4]]
    mx = max(abs(t) for t in v)
    if mx <= 1.5:                      # normalized 0-1
        sx, sy = W, H
    elif mx <= 1000.0:                 # our instructed 0-1000
        sx, sy = W / 1000.0, H / 1000.0
    else:                              # looks like raw pixels already
        sx, sy = 1.0, 1.0
    x1, y1, x2, y2 = v[0] * sx, v[1] * sy, v[2] * sx, v[3] * sy
    x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)


@torch.no_grad()
def answer_mcq(model, proc, images, prompt, device, letter_ids):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=images, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = out.logits[0, -1].float()
    g = inputs["image_grid_thw"]; merge = getattr(proc.image_processor, "merge_size", 2)
    ntok = int(sum((g[i].prod() // (merge ** 2)).item() for i in range(g.shape[0])))
    return int(torch.argmax(logits[letter_ids]).item()), ntok


@torch.no_grad()
def ground(model, proc, survey_im, question, device):
    """Ask the model for the relevant-region bbox (normalized 0-1000). Returns raw text."""
    pr = (f"Question: {question}\n"
          "Before answering, identify the SINGLE rectangular image region most relevant to "
          "answering this question. Output ONLY its bounding box as four integers "
          "'x1 y1 x2 y2', each normalized to 0-1000 (0=left/top, 1000=right/bottom). No other text.")
    msgs = [{"role": "user", "content": [{"type": "image", "image": survey_im}, {"type": "text", "text": pr}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[survey_im], return_tensors="pt").to(device)
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
                items.append({"image": img, "question": j["question"], "options": j["options"],
                              "bbox": j["bbox"]})
    return items[:limit] if limit else items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--vstar-root", required=True)
    ap.add_argument("--survey", type=int, default=256)  # mid-res context + grounding source
    ap.add_argument("--high", type=int, default=1024)   # uniform-high reference
    ap.add_argument("--crop", type=int, default=512)    # high-res crop budget
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
    print(f"[phase1a] {len(items)} items | survey={args.survey} high={args.high} crop={args.crop}", flush=True)
    rng = random.Random(0); recs = []; t0 = time.time()
    for i, it in enumerate(items):
        im = Image.open(it["image"]).convert("RGB"); W, H = im.size
        opts = list(it["options"]); correct = opts[0]; rng.shuffle(opts); gt = opts.index(correct)
        qtext = f"{it['question']}\n" + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts)) + \
                "\nAnswer with the letter only."
        note = (" The first image is the full scene; the second is a high-resolution zoom of the "
                "relevant region.")
        gt_xyxy = expand_xyxy(union_box(it["bbox"]), W, H)
        # raw GT (unexpanded) for IoU scoring
        gx, gy, gw, gh = union_box(it["bbox"]); gt_raw = (gx, gy, gx + gw, gy + gh)

        survey = resize_tokens(im, args.survey)
        full_hi = resize_tokens(im, args.high)
        oracle_crop = resize_tokens(im.crop(gt_xyxy), args.crop)

        # self-grounding
        gtext = ground(model, proc, survey, it["question"], device)
        pred = parse_box(gtext, W, H)
        parsed = pred is not None
        if parsed:
            pred_xyxy = expand_xyxy([pred[0], pred[1], pred[2] - pred[0], pred[3] - pred[1]], W, H)
            sg_crop = resize_tokens(im.crop(pred_xyxy), args.crop)
            iou_pred = iou(pred, gt_raw)
            cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
            center_hit = (gt_raw[0] <= cx <= gt_raw[2]) and (gt_raw[1] <= cy <= gt_raw[3])
        else:                                  # parse failed -> fallback center 50% crop
            cw, ch = W * 0.5, H * 0.5
            pred_xyxy = (int(W/2-cw/2), int(H/2-ch/2), int(W/2+cw/2), int(H/2+ch/2))
            sg_crop = resize_tokens(im.crop(pred_xyxy), args.crop); iou_pred = 0.0; center_hit = False

        p_hi, t_hi = answer_mcq(model, proc, [full_hi], qtext, device, letter_ids)
        p_or, t_or = answer_mcq(model, proc, [survey, oracle_crop], qtext + note, device, letter_ids)
        p_sg, t_sg = answer_mcq(model, proc, [survey, sg_crop], qtext + note, device, letter_ids)
        recs.append({"gt": gt, "high": p_hi, "oracle": p_or, "selfground": p_sg,
                     "t_high": t_hi, "t_oracle": t_or, "t_sg": t_sg,
                     "parsed": parsed, "iou": iou_pred, "center_hit": bool(center_hit),
                     "ground_text": gtext[:80]})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    def acc(k): return float(np.mean([r[k] == r["gt"] for r in recs]))
    def mt(k): return float(np.mean([r[k] for r in recs]))
    summ = {"n": len(recs),
            "acc": {"uniform_high": acc("high"), "oracle_fov": acc("oracle"), "selfground_fov": acc("selfground")},
            "tokens": {"uniform_high": mt("t_high"), "oracle_fov": mt("t_oracle"), "selfground_fov": mt("t_sg")},
            "localize": {"parse_rate": float(np.mean([r["parsed"] for r in recs])),
                         "mean_iou": float(np.mean([r["iou"] for r in recs])),
                         "iou>0.5": float(np.mean([r["iou"] > 0.5 for r in recs])),
                         "center_hit_rate": float(np.mean([r["center_hit"] for r in recs]))}}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump({"summary": summ, "records": recs}, open(args.output, "w"), indent=2)
    print("\n=== Phase 1a result (training-free self-grounding) ===")
    print(f"  uniform-high    : acc {summ['acc']['uniform_high']:.3f}   tok {summ['tokens']['uniform_high']:.0f}")
    print(f"  ORACLE-fov      : acc {summ['acc']['oracle_fov']:.3f}   tok {summ['tokens']['oracle_fov']:.0f}")
    print(f"  SELFGROUND-fov  : acc {summ['acc']['selfground_fov']:.3f}   tok {summ['tokens']['selfground_fov']:.0f}")
    print(f"  localize: parse {summ['localize']['parse_rate']:.2f}  meanIoU {summ['localize']['mean_iou']:.3f}  "
          f"IoU>.5 {summ['localize']['iou>0.5']:.2f}  center-hit {summ['localize']['center_hit_rate']:.2f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
