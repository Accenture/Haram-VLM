#!/usr/bin/env python
"""
Richer scout features for the learned risk head (P1 v3), Qwen3-VL.

NON-DESTRUCTIVE: writes to risk_features_v3/ only; does not touch the v1 dumps in
risk_features/ or any v1 script. Drop-in re-dump over the SAME probes/images as v1.

On the LOW-res scout pass (the only thing available before escalating), records:
  - emb       : last-token last-layer hidden state (4096-d)   [v1-compatible]
  - emb_ml    : last-token hidden state at 4 evenly-spaced layers (4 x 4096)
  - scalars   : [conf, py, pn, max_prob, logit_margin, entropy]  (logit-distribution risk)
  - attn      : per 4 layers, last-token attention over VISUAL tokens ->
                [vis_mass, vis_entropy, vis_max, vis_top5]  (16 scalars; eager attn)
plus the bookkeeping pred_low/conf_low/pred_high/label/tok_low/tok_high (as in v1).
The high-res pass only contributes pred_high/tok_high (no features extracted there).
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


def attn_visual_feats(attentions, vis_mask, layers):
    """For each selected layer: last-token attention (mean over heads) restricted to
    visual key positions -> [mass, entropy, max, top5]. Returns flat np.float32 (len 4*len(layers))."""
    out = []
    vis_idx = np.where(vis_mask)[0]
    for L in layers:
        a = attentions[L][0]                       # [heads, q, k]
        last = a[:, -1, :].mean(0).float().cpu().numpy()  # [k] mean over heads
        mass = float(last[vis_idx].sum())
        v = last[vis_idx]
        s = v.sum()
        if s > 1e-9 and len(v) > 1:
            p = v / s
            ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))  # normalized [0,1]
            mx = float(p.max())
            top5 = float(np.sort(p)[-5:].sum()) if len(p) >= 5 else float(p.sum())
        else:
            ent, mx, top5 = 0.0, 0.0, 0.0
        out += [mass, ent, mx, top5]
    return np.array(out, np.float32)


@torch.no_grad()
def scout_pass(model, proc, image, q, min_pix, max_pix, device, yes_ids, no_ids,
               img_token_id, want_feats, ml_layers, attn_layers):
    ip = proc.image_processor
    ip.min_pixels = int(min_pix); ip.max_pixels = int(max_pix)
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=want_feats, output_attentions=want_feats)
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
    feats = None
    if want_feats:
        hs = out.hidden_states                              # tuple len = n_layers+1
        emb = hs[-1][0, -1].float().cpu().numpy()
        emb_ml = np.stack([hs[L][0, -1].float().cpu().numpy() for L in ml_layers])  # (4,4096)
        top2 = torch.topk(logits, 2).values
        ent = float(-(p * torch.log(p + 1e-12)).sum().item() / np.log(p.numel()))
        scal = np.array([conf, py, pn, float(p.max().item()),
                         float((top2[0] - top2[1]).item()), ent], np.float32)
        vis_mask = (inputs["input_ids"][0].cpu().numpy() == img_token_id)
        try:
            attn = attn_visual_feats(out.attentions, vis_mask, attn_layers)
        except Exception as e:
            attn = np.zeros(4 * len(attn_layers), np.float32)
        feats = {"emb": emb, "emb_ml": emb_ml, "scal": scal, "attn": attn}
    return pred, conf, ntok, feats


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
    # eager attention so output_attentions is populated (scout sequence is short -> cheap)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    yes_ids, no_ids = yes_no_ids(proc.tokenizer)
    img_token_id = getattr(model.config, "image_token_id", None)
    if img_token_id is None:
        img_token_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_layers = model.config.text_config.num_hidden_layers if hasattr(model.config, "text_config") \
        else model.config.num_hidden_layers
    ml_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]      # into hidden_states (len n+1)
    attn_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]  # into attentions (len n)
    lo_min, lo_max = 4 * PATCH, args.low_tokens * PATCH
    hi_min, hi_max = args.high_tokens * PATCH, args.high_tokens * 4 * PATCH
    print(f"[{os.path.basename(args.pope)}] {len(rows)} probes | img_token={img_token_id} | "
          f"layers={n_layers} ml={ml_layers} attn={attn_layers}", flush=True)

    EMB, ML, SC, AT, cl, pl, ph, lb, tl, th = [], [], [], [], [], [], [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
        p_l, c_l, t_l, f = scout_pass(model, proc, img, r["text"], lo_min, lo_max, device,
                                      yes_ids, no_ids, img_token_id, True, ml_layers, attn_layers)
        p_h, _, t_h, _ = scout_pass(model, proc, img, r["text"], hi_min, hi_max, device,
                                    yes_ids, no_ids, img_token_id, False, ml_layers, attn_layers)
        EMB.append(f["emb"]); ML.append(f["emb_ml"]); SC.append(f["scal"]); AT.append(f["attn"])
        cl.append(c_l); pl.append(1 if p_l == "yes" else 0); ph.append(1 if p_h == "yes" else 0)
        lb.append(1 if r["label"] == "yes" else 0); tl.append(t_l); th.append(t_h)
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output,
                        emb=np.array(EMB, np.float16), emb_ml=np.array(ML, np.float16),
                        scal=np.array(SC, np.float32), attn=np.array(AT, np.float32),
                        conf_low=np.array(cl, np.float32), pred_low=np.array(pl, np.int8),
                        pred_high=np.array(ph, np.int8), label=np.array(lb, np.int8),
                        tok_low=np.array(tl, np.int32), tok_high=np.array(th, np.int32))
    print(f"  saved {len(EMB)} -> {args.output} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
