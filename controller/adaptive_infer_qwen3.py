#!/usr/bin/env python
"""
Training-free adaptive-resolution inference for Qwen3-VL (2nd architecture).

Same predict-then-allocate policy as adaptive_infer.py, but for Qwen3-VL, whose
resolution is controlled natively by a per-image pixel budget (max_pixels) rather
than by image crops. Scout = low pixel budget (~few hundred image tokens);
escalate = high pixel budget. Confidence = yes/no first-token logit margin.

Run in the 'qwen3' conda env (transformers>=4.57). Records both passes per probe;
sweep the escalation threshold offline to trace the accuracy-vs-tokens Pareto.
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PATCH = 28 * 28  # Qwen-VL: one merged visual token ~ a 28x28 pixel group


def load(model_id, device):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.to(device).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    return model, proc


def yes_no_ids(tok):
    yes, no = set(), set()
    for w in ["yes", "Yes", " yes", " Yes", "YES"]:
        ids = tok.encode(w, add_special_tokens=False)
        if ids:
            yes.add(ids[0])
    for w in ["no", "No", " no", " No", "NO"]:
        ids = tok.encode(w, add_special_tokens=False)
        if ids:
            no.add(ids[0])
    return sorted(yes), sorted(no)


@torch.no_grad()
def yn_pass(model, proc, image, question, min_pix, max_pix, device, yes_ids, no_ids):
    # bind the resolution knob on the image processor itself (message-dict pixels are
    # ignored when the raw image is passed to the processor directly)
    ip = proc.image_processor
    ip.min_pixels = int(min_pix); ip.max_pixels = int(max_pix)
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size = {**ip.size, "shortest_edge": int(min_pix), "longest_edge": int(max_pix)}
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    p_yes = probs[yes_ids].sum().item(); p_no = probs[no_ids].sum().item()
    z = p_yes + p_no + 1e-9
    pred = "yes" if p_yes >= p_no else "no"
    conf = max(p_yes, p_no) / z
    # number of image tokens actually used (from the vision grid)
    if "image_grid_thw" in inputs:
        g = inputs["image_grid_thw"][0]
        ntok = int(g.prod().item() // (getattr(proc.image_processor, "merge_size", 2) ** 2))
    else:
        ntok = int(inputs["input_ids"].shape[1])
    return pred, conf, ntok


def metrics(items):
    tp = sum(1 for r in items if r["label"] == "yes" and r["pred"] == "yes")
    tn = sum(1 for r in items if r["label"] == "no" and r["pred"] == "no")
    fp = sum(1 for r in items if r["label"] == "no" and r["pred"] == "yes")
    fn = sum(1 for r in items if r["label"] == "yes" and r["pred"] == "no")
    n = len(items); neg = tn + fp
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    return {"acc": (tp + tn) / n, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0,
            "halluc": fp / neg if neg else 0}


def sweep(records):
    tl = np.mean([r["tok_low"] for r in records]); th = np.mean([r["tok_high"] for r in records])
    pts = []
    for tau in np.linspace(0.5, 1.0, 26):
        items, cost = [], 0.0
        for r in records:
            esc = r["conf_low"] < tau
            items.append({"label": r["label"], "pred": r["pred_high"] if esc else r["pred_low"]})
            cost += r["tok_low"] + (r["tok_high"] if esc else 0)
        m = metrics(items); m.update(tau=round(float(tau), 3),
                                     esc_rate=float(np.mean([r["conf_low"] < tau for r in records])),
                                     avg_tokens=cost / len(records)); pts.append(m)
    bl = {**metrics([{"label": r["label"], "pred": r["pred_low"]} for r in records]), "avg_tokens": tl, "name": "always-low"}
    bh = {**metrics([{"label": r["label"], "pred": r["pred_high"]} for r in records]), "avg_tokens": th, "name": "always-high"}
    return {"pareto": pts, "always_low": bl, "always_high": bh}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--pope", required=True); ap.add_argument("--image-dir", required=True)
    ap.add_argument("--low-tokens", type=int, default=256)
    ap.add_argument("--high-tokens", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    rows = [json.loads(l) for l in open(args.pope) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    model, proc = load(args.model, device)
    yes_ids, no_ids = yes_no_ids(proc.tokenizer)
    print(f"[{os.path.basename(args.pope)}] {len(rows)} probes | {args.model} | tokens {args.low_tokens}->{args.high_tokens} | yes={yes_ids} no={no_ids}", flush=True)
    records, t0 = [], time.time()
    low_min, low_max = 4 * PATCH, args.low_tokens * PATCH               # cap -> downscale to ~low_tokens
    high_min, high_max = args.high_tokens * PATCH, args.high_tokens * 4 * PATCH  # floor -> force ~high_tokens
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
        pl, cl, tl = yn_pass(model, proc, img, r["text"], low_min, low_max, device, yes_ids, no_ids)
        ph, ch, th = yn_pass(model, proc, img, r["text"], high_min, high_max, device, yes_ids, no_ids)
        records.append({"label": r["label"], "pred_low": pl, "conf_low": cl, "tok_low": tl,
                        "pred_high": ph, "conf_high": ch, "tok_high": th})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    out = {"pope": os.path.basename(args.pope), "model": args.model, **sweep(records), "records": records}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    bl, bh = out["always_low"], out["always_high"]
    cand = [p for p in out["pareto"] if p["acc"] >= bh["acc"] - 0.003]
    knee = min(cand, key=lambda p: p["avg_tokens"]) if cand else max(out["pareto"], key=lambda p: p["acc"])
    print(f"  always-low : acc={bl['acc']:.3f} tok={bl['avg_tokens']:.0f}")
    print(f"  always-high: acc={bh['acc']:.3f} tok={bh['avg_tokens']:.0f}")
    print(f"  ADAPTIVE   : acc={knee['acc']:.3f} tok={knee['avg_tokens']:.0f} esc={knee['esc_rate']*100:.0f}% "
          f"(-{(1-knee['avg_tokens']/bh['avg_tokens'])*100:.0f}% tok, tau={knee['tau']})")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
