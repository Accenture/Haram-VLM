#!/usr/bin/env python
"""
Training-free adaptive-resolution inference on V*Bench (beyond yes/no), InternVL.

V*Bench: high-resolution images with tiny target objects; multiple-choice questions
(direct_attributes + relative_position). The correct answer is options[0]; we shuffle
options into A/B/C/D and read the scout's first-token distribution over the letter
tokens (predicted = argmax, confidence = letter margin). Accuracy = exact MCQ.
Scout = few tiles (target invisible); escalate = many tiles (target resolved).
Run in 'qwen3' env with HF_HOME on the share.
"""
import argparse, glob, json, os, random, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

LETTERS = "ABCDEFGH"


def letter_ids(tok):
    m = {}
    for i, L in enumerate(LETTERS):
        for w in (L, " " + L):
            ids = tok.encode(w, add_special_tokens=False)
            if ids and ids[0] not in m:
                m[ids[0]] = i
    return m  # token_id -> option index


def load_vstar(root):
    items = []
    for sub in ["direct_attributes", "relative_position"]:
        for jf in sorted(glob.glob(os.path.join(root, sub, "*.json"))):
            j = json.load(open(jf))
            img = None
            for ext in (".jpg", ".JPG", ".png", ".jpeg", ".webp", ".JPEG"):
                if os.path.exists(jf[:-5] + ext):
                    img = jf[:-5] + ext; break
            if img and j.get("options"):
                items.append({"image": img, "question": j["question"], "options": j["options"], "sub": sub})
    return items


def build_prompt(q, options):
    lines = [q] + [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    lines.append("Answer with the letter of the correct option only.")
    return "\n".join(lines)


def set_tiles(proc, max_tiles):
    ip = proc.image_processor
    for a, v in [("crop_to_patches", max_tiles > 1), ("max_patches", int(max_tiles)), ("min_patches", 1)]:
        if hasattr(ip, a):
            setattr(ip, a, v)


@torch.no_grad()
def mc_pass(model, proc, image, prompt, n_opt, max_tiles, device, lmap):
    set_tiles(proc, max_tiles)
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1].float()
    p = torch.softmax(logits, -1)
    probs = {}
    for tid, idx in lmap.items():
        if idx < n_opt:
            probs[idx] = probs.get(idx, 0.0) + p[tid].item()
    tot = sum(probs.values()) + 1e-9
    pred = max(probs, key=probs.get) if probs else 0
    conf = probs.get(pred, 0.0) / tot
    ntok = int(inputs["input_ids"].shape[1])
    return pred, conf, ntok


def sweep(R):
    tl = np.mean([r["tok_low"] for r in R]); th = np.mean([r["tok_high"] for r in R])
    risk = np.array([1 - r["conf_low"] for r in R]); order = np.argsort(-risk, kind="stable"); n = len(R)
    acc = lambda items: float(np.mean([a == b for a, b in items])) if items else 0.0
    pts = []
    for f in np.linspace(0, 1, 41):
        k = int(round(f * n)); esc = np.zeros(n, bool); esc[order[:k]] = True
        items = [(R[i]["gt"], R[i]["pred_high"] if esc[i] else R[i]["pred_low"]) for i in range(n)]
        cost = sum(R[i]["tok_low"] + (R[i]["tok_high"] if esc[i] else 0) for i in range(n)) / n
        pts.append({"frac": float(f), "esc_rate": float(esc.mean()), "acc": acc(items), "avg_tokens": cost})
    bl = {"acc": acc([(r["gt"], r["pred_low"]) for r in R]), "avg_tokens": tl}
    bh = {"acc": acc([(r["gt"], r["pred_high"]) for r in R]), "avg_tokens": th}
    return {"pareto": pts, "always_low": bl, "always_high": bh}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3-8B-hf")
    ap.add_argument("--vstar-root", required=True)
    ap.add_argument("--low-tiles", type=int, default=1); ap.add_argument("--high-tiles", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"; rng = random.Random(0)
    items = load_vstar(args.vstar_root)
    if args.limit: items = items[: args.limit]
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    lmap = letter_ids(proc.tokenizer)
    print(f"[V*Bench] {len(items)} q | {args.model} | tiles {args.low_tiles}->{args.high_tiles}", flush=True)
    R, t0 = [], time.time()
    for i, it in enumerate(items):
        opts = list(it["options"]); correct = opts[0]
        rng.shuffle(opts); gt = opts.index(correct)              # correct option index after shuffle
        prompt = build_prompt(it["question"], opts)
        img = Image.open(it["image"]).convert("RGB")
        pl, cl, tl = mc_pass(model, proc, img, prompt, len(opts), args.low_tiles, device, lmap)
        ph, ch, th = mc_pass(model, proc, img, prompt, len(opts), args.high_tiles, device, lmap)
        R.append({"gt": gt, "pred_low": pl, "conf_low": cl, "tok_low": tl,
                  "pred_high": ph, "conf_high": ch, "tok_high": th, "sub": it["sub"]})
        if (i + 1) % 40 == 0: print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    out = {"task": "vstar", "model": args.model, **sweep(R), "records": R}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    bl, bh = out["always_low"], out["always_high"]
    cand = [p for p in out["pareto"] if p["acc"] >= bh["acc"] - 0.005]
    knee = min(cand, key=lambda p: p["avg_tokens"]) if cand else max(out["pareto"], key=lambda p: p["acc"])
    print(f"  always-low : acc={bl['acc']:.3f} tok={bl['avg_tokens']:.0f}")
    print(f"  always-high: acc={bh['acc']:.3f} tok={bh['avg_tokens']:.0f}")
    print(f"  ADAPTIVE   : acc={knee['acc']:.3f} tok={knee['avg_tokens']:.0f} esc={knee['esc_rate']*100:.0f}% "
          f"(-{(1-knee['avg_tokens']/bh['avg_tokens'])*100:.0f}%)")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
