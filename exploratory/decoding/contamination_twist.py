#!/usr/bin/env python
"""
Contamination-twist: does benchmark contamination inflate decoding-side gains?

We use the paper's CONTAMINATED baseline model -- a Phi-3-V LoRA fine-tuned on a leaky POPE split where ~93% of
(image, question) test pairs also appear in training. Its POPE-adversarial test is therefore split into
  SEEN   (pair in training -> memorized; artificially low hallucination) and
  UNSEEN (pair held out      -> honest hallucination),
with the SAME model, so the only thing that varies is contamination. We apply a decoding-side fix (DoLa)
to both subsets and ask whether its apparent hallucination-reduction is larger on the contaminated
(seen) data than on the clean (unseen) data -- i.e. whether contamination inflates the reported gain.

Input = the paper's existing prediction dump (has image/text/label/seen per item). We re-run the model to
get baseline + DoLa yes/no at the same resolution (num_crops) it was evaluated at. Run in 'haram' env.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

ROOT = HARAM_ROOT + "/haram_vlm"
sys.path.insert(0, os.path.join(ROOT, "src"))
from model.Phi3_vision.modeling_phi3_v import Phi3VForCausalLM, Phi3VConfig  # noqa: E402


BASE = "microsoft/Phi-3-vision-128k-instruct"
IMG_DIRS = [HARAM_ROOT + "/coco_build/images",
            WORK_DIR + "/pope_imgs"]


def find_img(name):
    for d in IMG_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p): return p
    raise FileNotFoundError(name)


def make_processor(num_crops):
    p = AutoProcessor.from_pretrained(BASE, trust_remote_code=True, num_crops=num_crops)
    try: p.image_processor.num_crops = num_crops
    except Exception: pass
    return p


def yes_no_ids(tok):
    y, n = set(), set()
    for w in ["yes", "Yes", " yes", " Yes", "YES"]:
        i = tok.encode(w, add_special_tokens=False);  y.add(i[0]) if i else None
    for w in ["no", "No", " no", " No", "NO"]:
        i = tok.encode(w, add_special_tokens=False);  n.add(i[0]) if i else None
    return sorted(y), sorted(n)


@torch.no_grad()
def fwd(model, proc, image, question, device):
    prompt = proc.tokenizer.apply_chat_template(
        [{"role": "user", "content": f"<|image_1|>\n{question}"}], tokenize=False, add_generation_prompt=True)
    inp = proc(prompt, [image], return_tensors="pt").to(device)
    return model(**inp, output_hidden_states=True)


def dola_logit(out, lm_head, norm, cand):
    hs = out.hidden_states
    flp = torch.log_softmax(out.logits[0, -1].float(), -1)
    ll = lambda h: lm_head(norm(h) if norm is not None else h).float()
    best, sel = -1.0, None
    for L in cand:
        elp = torch.log_softmax(ll(hs[L][0, -1]), -1)
        m = 0.5 * (flp.exp() + elp.exp())
        j = 0.5 * ((flp.exp() * (flp - m.log())).sum() + (elp.exp() * (elp - m.log())).sum())
        if j.item() > best: best, sel = j.item(), elp
    return flp - sel


def yn(logit, ids):
    p = torch.softmax(logit.float(), -1)
    return "yes" if p[ids[0]].sum() >= p[ids[1]].sum() else "no"


def metrics(items):
    """items: list of (label, pred). halluc = FP / negatives; acc."""
    tp = sum(1 for l, p in items if l == "yes" and p == "yes")
    tn = sum(1 for l, p in items if l == "no" and p == "no")
    fp = sum(1 for l, p in items if l == "no" and p == "yes")
    fn = sum(1 for l, p in items if l == "yes" and p == "no")
    n = len(items); neg = tn + fp
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": n, "acc": (tp + tn) / n if n else 0.0,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "halluc": fp / neg if neg else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{ROOT}/output/haram_full_4gpu_20260610_003136")
    ap.add_argument("--input", default=HARAM_ROOT + "/pope_eval/pope_adversarial.json")
    ap.add_argument("--num-crops", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"

    cfg = Phi3VConfig.from_pretrained(BASE); cfg._attn_implementation = "eager"
    model = Phi3VForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, config=cfg,
                                             low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.ckpt); model = model.merge_and_unload()
    model = model.to(device).eval()
    proc = make_processor(args.num_crops)
    ids = yes_no_ids(proc.tokenizer)
    lm_head = model.lm_head; norm = model.model.norm
    nlayers = len(model.model.layers); cand = list(range(2, nlayers // 2, 2))
    print(f"[contam-twist] ckpt={os.path.basename(args.ckpt)} layers={nlayers} crops={args.num_crops}", flush=True)

    preds = json.load(open(args.input))["predictions"]
    if args.limit: preds = preds[: args.limit]
    recs = []; t0 = time.time()
    for i, r in enumerate(preds):
        im = Image.open(find_img(r["image"])).convert("RGB")
        out = fwd(model, proc, im, r["text"], device)
        base = yn(out.logits[0, -1], ids)
        dola = yn(dola_logit(out, lm_head, norm, cand), ids)
        recs.append({"seen": bool(r["seen"]), "label": r["label"], "base": base, "dola": dola})
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(preds)} ({time.time()-t0:.0f}s)", flush=True)

    out_obj = {"ckpt": args.ckpt, "num_crops": args.num_crops, "n": len(recs), "cells": {}}
    print("\n=== contamination-twist (POPE-adversarial, one contaminated model) ===")
    print(f"{'subset':10} {'method':8} {'n':>5} {'halluc':>7} {'acc':>6} {'F1':>6}")
    for sub, flag in [("seen", True), ("unseen", False), ("all", None)]:
        g = [r for r in recs if flag is None or r["seen"] == flag]
        mb = metrics([(r["label"], r["base"]) for r in g])
        md = metrics([(r["label"], r["dola"]) for r in g])
        out_obj["cells"][sub] = {"baseline": mb, "dola": md,
                                 "dola_halluc_reduction": mb["halluc"] - md["halluc"],
                                 "dola_acc_gain": md["acc"] - mb["acc"]}
        for nm, m in [("baseline", mb), ("DoLa", md)]:
            print(f"{sub:10} {nm:8} {m['n']:5d} {m['halluc']:.3f}  {m['acc']:.3f} {m['f1']:.3f}")
    s, u = out_obj["cells"]["seen"], out_obj["cells"]["unseen"]
    print("\n--- DoLa hallucination-reduction (baseline_halluc - dola_halluc) ---")
    print(f"  SEEN (contaminated): {s['dola_halluc_reduction']:+.3f}")
    print(f"  UNSEEN (honest)    : {u['dola_halluc_reduction']:+.3f}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out_obj, open(args.output, "w"), indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
