#!/usr/bin/env python
"""
Extract scout features for the learned risk head (P1), Qwen3-VL.

For each POPE probe: run the low-res scout and the high-res pass, and record
  - emb: the scout's last-token last-layer hidden state (the vector the LM uses to
         decide the answer) -- the only thing available at inference before escalating,
  - conf_low: scout yes/no confidence margin (the training-free baseline signal),
  - pred_low, pred_high, label, tok_low, tok_high.
Saved as an .npz. A risk head is then trained (on training-image features) to predict
whether the scout is wrong, and used to gate escalation on the held-out test.
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PATCH = 28 * 28


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
def scout_pass(model, proc, image, q, min_pix, max_pix, device, yes_ids, no_ids, want_emb):
    ip = proc.image_processor
    ip.min_pixels = int(min_pix); ip.max_pixels = int(max_pix)
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=want_emb)
    logits = out.logits[0, -1].float()
    p = torch.softmax(logits, -1)
    py, pn = p[yes_ids].sum().item(), p[no_ids].sum().item()
    pred = "yes" if py >= pn else "no"
    conf = max(py, pn) / (py + pn + 1e-9)
    if "image_grid_thw" in inputs:
        g = inputs["image_grid_thw"][0]
        ntok = int(g.prod().item() // (getattr(proc.image_processor, "merge_size", 2) ** 2))
    else:
        ntok = int(inputs["input_ids"].shape[1])
    emb = out.hidden_states[-1][0, -1].float().cpu().numpy() if want_emb else None
    return pred, conf, ntok, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--pope", required=True); ap.add_argument("--image-dir", required=True)
    ap.add_argument("--low-tokens", type=int, default=128)
    ap.add_argument("--high-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    rows = [json.loads(l) for l in open(args.pope) if l.strip()]
    if args.limit: rows = rows[: args.limit]
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    yes_ids, no_ids = yes_no_ids(proc.tokenizer)
    lo_min, lo_max = 4 * PATCH, args.low_tokens * PATCH
    hi_min, hi_max = args.high_tokens * PATCH, args.high_tokens * 4 * PATCH
    print(f"[{os.path.basename(args.pope)}] {len(rows)} probes", flush=True)

    E, cl, pl, ph, lb, tl, th = [], [], [], [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
        p_l, c_l, t_l, emb = scout_pass(model, proc, img, r["text"], lo_min, lo_max, device, yes_ids, no_ids, True)
        p_h, _, t_h, _ = scout_pass(model, proc, img, r["text"], hi_min, hi_max, device, yes_ids, no_ids, False)
        E.append(emb); cl.append(c_l); pl.append(1 if p_l == "yes" else 0)
        ph.append(1 if p_h == "yes" else 0); lb.append(1 if r["label"] == "yes" else 0)
        tl.append(t_l); th.append(t_h)
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, emb=np.array(E, np.float16), conf_low=np.array(cl, np.float32),
                        pred_low=np.array(pl, np.int8), pred_high=np.array(ph, np.int8),
                        label=np.array(lb, np.int8), tok_low=np.array(tl, np.int32), tok_high=np.array(th, np.int32))
    print(f"  saved {len(E)} x {len(E[0])}-d -> {args.output}")


if __name__ == "__main__":
    main()
