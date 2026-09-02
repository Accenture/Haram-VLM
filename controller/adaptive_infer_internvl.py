#!/usr/bin/env python
"""
Training-free adaptive-resolution inference for InternVL (3rd architecture).

Same predict-then-allocate policy; InternVL controls resolution by DYNAMIC TILING
(number of 448px tiles). Scout = few tiles (low token budget); escalate = many tiles.
Confidence = yes/no first-token logit margin. Run in the 'qwen3' env (transformers 5.11).
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def yes_no_ids(tok):
    yes, no = set(), set()
    for w in ["yes", "Yes", " yes", " Yes", "YES"]:
        i = tok.encode(w, add_special_tokens=False)
        if i: yes.add(i[0])
    for w in ["no", "No", " no", " No", "NO"]:
        i = tok.encode(w, add_special_tokens=False)
        if i: no.add(i[0])
    return sorted(yes), sorted(no)


def set_tiles(proc, max_tiles):
    ip = proc.image_processor
    for attr, val in [("crop_to_patches", max_tiles > 1), ("max_patches", int(max_tiles)),
                      ("min_patches", 1)]:
        if hasattr(ip, attr):
            setattr(ip, attr, val)


@torch.no_grad()
def yn_pass(model, proc, image, q, max_tiles, device, yes_ids, no_ids):
    set_tiles(proc, max_tiles)
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": q}]}]
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1].float()
    p = torch.softmax(logits, -1)
    py, pn = p[yes_ids].sum().item(), p[no_ids].sum().item()
    pred = "yes" if py >= pn else "no"
    conf = max(py, pn) / (py + pn + 1e-9)
    ntok = int(inputs["input_ids"].shape[1])
    return pred, conf, ntok


def metrics(items):
    tp = sum(1 for r in items if r["l"] == "yes" and r["p"] == "yes")
    tn = sum(1 for r in items if r["l"] == "no" and r["p"] == "no")
    fp = sum(1 for r in items if r["l"] == "no" and r["p"] == "yes")
    fn = sum(1 for r in items if r["l"] == "yes" and r["p"] == "no")
    n = len(items); neg = tn + fp
    pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
    return {"acc": (tp + tn) / n, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0,
            "halluc": fp / neg if neg else 0}


def sweep(R):
    tl = np.mean([r["tok_low"] for r in R]); th = np.mean([r["tok_high"] for r in R])
    pts = []
    for tau in np.linspace(0.5, 1.0, 26):
        items, cost = [], 0.0
        for r in R:
            esc = r["conf_low"] < tau
            items.append({"l": r["label"], "p": r["pred_high"] if esc else r["pred_low"]})
            cost += r["tok_low"] + (r["tok_high"] if esc else 0)
        m = metrics(items); m.update(tau=round(float(tau), 3),
                                     esc_rate=float(np.mean([r["conf_low"] < tau for r in R])),
                                     avg_tokens=cost / len(R)); pts.append(m)
    bl = {**metrics([{"l": r["label"], "p": r["pred_low"]} for r in R]), "avg_tokens": tl}
    bh = {**metrics([{"l": r["label"], "p": r["pred_high"]} for r in R]), "avg_tokens": th}
    return {"pareto": pts, "always_low": bl, "always_high": bh}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3-8B-hf")
    ap.add_argument("--pope", required=True); ap.add_argument("--image-dir", required=True)
    ap.add_argument("--low-tiles", type=int, default=1)
    ap.add_argument("--high-tiles", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    rows = [json.loads(l) for l in open(args.pope) if l.strip()]
    if args.limit: rows = rows[: args.limit]
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    yes_ids, no_ids = yes_no_ids(proc.tokenizer)
    print(f"[{os.path.basename(args.pope)}] {len(rows)} probes | {args.model} | tiles {args.low_tiles}->{args.high_tiles}", flush=True)
    R, t0 = [], time.time()
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
        pl, cl, tl = yn_pass(model, proc, img, r["text"], args.low_tiles, device, yes_ids, no_ids)
        ph, ch, th = yn_pass(model, proc, img, r["text"], args.high_tiles, device, yes_ids, no_ids)
        R.append({"label": r["label"], "pred_low": pl, "conf_low": cl, "tok_low": tl,
                  "pred_high": ph, "conf_high": ch, "tok_high": th})
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    out = {"pope": os.path.basename(args.pope), "model": args.model, **sweep(R), "records": R}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    bl, bh = out["always_low"], out["always_high"]
    cand = [p for p in out["pareto"] if p["acc"] >= bh["acc"] - 0.005]
    knee = min(cand, key=lambda p: p["avg_tokens"]) if cand else max(out["pareto"], key=lambda p: p["acc"])
    print(f"  always-low : acc={bl['acc']:.3f} tok={bl['avg_tokens']:.0f}")
    print(f"  always-high: acc={bh['acc']:.3f} tok={bh['avg_tokens']:.0f}")
    print(f"  ADAPTIVE   : acc={knee['acc']:.3f} tok={knee['avg_tokens']:.0f} esc={knee['esc_rate']*100:.0f}% "
          f"(-{(1-knee['avg_tokens']/bh['avg_tokens'])*100:.0f}%, tau={knee['tau']})")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
