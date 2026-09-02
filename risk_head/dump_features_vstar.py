#!/usr/bin/env python
"""
Dump V*Bench scout features for a LEARNED risk head (does it fix the boundary case?).

Same InternVL3-8B 1->24 tile setup as Table 3, but the low-tile (scout) pass also records
its last-token last-layer hidden state. A probe is then trained (cross-validated, since
V*Bench is only ~170 items) to predict whether the scout is wrong, and used to rank
escalation -- compared to confidence and the oracle. Run in 'qwen3' env.
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
    return m


def load_vstar(root):
    items = []
    for sub in ["direct_attributes", "relative_position"]:
        for jf in sorted(glob.glob(os.path.join(root, sub, "*.json"))):
            j = json.load(open(jf)); img = None
            for ext in (".jpg", ".JPG", ".png", ".jpeg", ".webp", ".JPEG"):
                if os.path.exists(jf[:-5] + ext):
                    img = jf[:-5] + ext; break
            if img and j.get("options"):
                items.append({"image": img, "question": j["question"], "options": j["options"], "sub": sub})
    return items


def build_prompt(q, options):
    return "\n".join([q] + [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)] +
                     ["Answer with the letter of the correct option only."])


def set_tiles(proc, max_tiles):
    ip = proc.image_processor
    for a, v in [("crop_to_patches", max_tiles > 1), ("max_patches", int(max_tiles)), ("min_patches", 1)]:
        if hasattr(ip, a):
            setattr(ip, a, v)


@torch.no_grad()
def mc_pass(model, proc, image, prompt, n_opt, max_tiles, device, lmap, want_emb=False):
    set_tiles(proc, max_tiles)
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=want_emb)
    logits = out.logits[0, -1].float()
    p = torch.softmax(logits, -1)
    probs = {}
    for tid, idx in lmap.items():
        if idx < n_opt:
            probs[idx] = probs.get(idx, 0.0) + p[tid].item()
    tot = sum(probs.values()) + 1e-9
    pred = max(probs, key=probs.get) if probs else 0
    conf = probs.get(pred, 0.0) / tot
    ntok = int(inputs["input_ids"].shape[1])
    emb = out.hidden_states[-1][0, -1].float().cpu().numpy() if want_emb else None
    return pred, conf, ntok, emb


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
    print(f"[V*Bench risk] {len(items)} q | {args.model} | tiles {args.low_tiles}->{args.high_tiles}", flush=True)
    E, cl, pl, ph, gtl, tl, th, sub = [], [], [], [], [], [], [], []
    t0 = time.time()
    for i, it in enumerate(items):
        opts = list(it["options"]); correct = opts[0]; rng.shuffle(opts); gt = opts.index(correct)
        prompt = build_prompt(it["question"], opts)
        img = Image.open(it["image"]).convert("RGB")
        p_l, c_l, t_l, emb = mc_pass(model, proc, img, prompt, len(opts), args.low_tiles, device, lmap, want_emb=True)
        p_h, _, t_h, _ = mc_pass(model, proc, img, prompt, len(opts), args.high_tiles, device, lmap, want_emb=False)
        E.append(emb); cl.append(c_l); pl.append(p_l); ph.append(p_h); gtl.append(gt)
        tl.append(t_l); th.append(t_h); sub.append(0 if it["sub"] == "direct_attributes" else 1)
        if (i + 1) % 40 == 0: print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, emb=np.array(E, np.float16), conf_low=np.array(cl, np.float32),
                        pred_low=np.array(pl, np.int16), pred_high=np.array(ph, np.int16),
                        gt=np.array(gtl, np.int16), tok_low=np.array(tl, np.int32),
                        tok_high=np.array(th, np.int32), sub=np.array(sub, np.int8))
    sw = float(np.mean(np.array(pl) != np.array(gtl)))
    print(f"  saved {len(E)} x {len(E[0])}-d | scout-wrong={sw:.3f} "
          f"| always-low acc={np.mean(np.array(pl)==np.array(gtl)):.3f} "
          f"always-high acc={np.mean(np.array(ph)==np.array(gtl)):.3f} -> {args.output}")


if __name__ == "__main__":
    main()
