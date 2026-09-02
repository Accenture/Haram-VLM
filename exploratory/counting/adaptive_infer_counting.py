#!/usr/bin/env python
"""
Training-free adaptive-resolution inference on a COUNTING task (beyond yes/no), InternVL.

Same predict-then-allocate policy, but the answer is an integer count. We read the
scout's first-token distribution over digit tokens 1..9; predicted count = argmax,
confidence = max digit-prob / total digit-prob. Accuracy = exact-match count.
Scout = few tiles; escalate = many tiles. Run in 'qwen3' env with HF_HOME on the share.
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def digit_map(tok):
    m = {}
    for d in "123456789":
        for w in (d, " " + d):
            ids = tok.encode(w, add_special_tokens=False)
            if ids and ids[0] not in m:
                m[ids[0]] = int(d)
    return m  # token_id -> int count


def set_tiles(proc, max_tiles):
    ip = proc.image_processor
    for a, v in [("crop_to_patches", max_tiles > 1), ("max_patches", int(max_tiles)), ("min_patches", 1)]:
        if hasattr(ip, a):
            setattr(ip, a, v)


@torch.no_grad()
def count_pass(model, proc, image, q, max_tiles, device, dmap):
    set_tiles(proc, max_tiles)
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": q}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1].float()
    p = torch.softmax(logits, -1)
    ids = list(dmap.keys())
    probs = {dmap[i]: p[i].item() for i in ids}
    tot = sum(probs.values()) + 1e-9
    pred = max(probs, key=probs.get)
    conf = probs[pred] / tot
    ntok = int(inputs["input_ids"].shape[1])
    return pred, conf, ntok


def acc(items):
    return float(np.mean([r["pred"] == r["gt"] for r in items])) if items else 0.0


def sweep(R):
    tl = np.mean([r["tok_low"] for r in R]); th = np.mean([r["tok_high"] for r in R])
    pts = []
    risk = np.array([1 - r["conf_low"] for r in R]); order = np.argsort(-risk, kind="stable"); n = len(R)
    for f in np.linspace(0, 1, 41):
        k = int(round(f * n)); esc = np.zeros(n, bool); esc[order[:k]] = True
        items = [{"gt": R[i]["gt"], "pred": R[i]["pred_high"] if esc[i] else R[i]["pred_low"]} for i in range(n)]
        cost = sum(R[i]["tok_low"] + (R[i]["tok_high"] if esc[i] else 0) for i in range(n)) / n
        pts.append({"frac": float(f), "esc_rate": float(esc.mean()), "acc": acc(items), "avg_tokens": cost})
    bl = {"acc": acc([{"gt": r["gt"], "pred": r["pred_low"]} for r in R]), "avg_tokens": tl}
    bh = {"acc": acc([{"gt": r["gt"], "pred": r["pred_high"]} for r in R]), "avg_tokens": th}
    return {"pareto": pts, "always_low": bl, "always_high": bh}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3-8B-hf")
    ap.add_argument("--data", required=True); ap.add_argument("--image-dir", required=True)
    ap.add_argument("--low-tiles", type=int, default=1); ap.add_argument("--high-tiles", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit: rows = rows[: args.limit]
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    dmap = digit_map(proc.tokenizer)
    print(f"[counting] {len(rows)} q | {args.model} | tiles {args.low_tiles}->{args.high_tiles}", flush=True)
    R, t0 = [], time.time()
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
        pl, cl, tl = count_pass(model, proc, img, r["text"], args.low_tiles, device, dmap)
        ph, ch, th = count_pass(model, proc, img, r["text"], args.high_tiles, device, dmap)
        R.append({"gt": r["count"], "pred_low": pl, "conf_low": cl, "tok_low": tl,
                  "pred_high": ph, "conf_high": ch, "tok_high": th})
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    out = {"task": "counting", "model": args.model, **sweep(R), "records": R}
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
