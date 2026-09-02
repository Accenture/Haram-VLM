#!/usr/bin/env python
"""
Dump per-item low / low+DoLa / high predictions for the two-failure-mode qualitative figure.
  pope  : POPE-adversarial negatives  -> find a PRIOR-DRIVEN case (low=YES halluc, +DoLa=NO fix).
  vstar : V*Bench                      -> find a PERCEPTION-BOUND case (low wrong, high right, DoLa can't).
Saves per-item records (image, query, preds, confidence) to JSON. Run in 'qwen3' env.
"""
import argparse, glob, json, os, random, re, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


TOKPX = 32 * 32
LETTERS = "ABCDEFGH"


def resize_tokens(im, n):
    W, H = im.size; s = (n * TOKPX / max(1, W * H)) ** 0.5
    return im.resize((max(56, int(round(W * s))), max(56, int(round(H * s)))), Image.LANCZOS)


def yes_no_ids(tok):
    y, n = set(), set()
    for w in ["yes", "Yes", " yes", " Yes", "YES"]:
        i = tok.encode(w, add_special_tokens=False);  y.add(i[0]) if i else None
    for w in ["no", "No", " no", " No", "NO"]:
        i = tok.encode(w, add_special_tokens=False);  n.add(i[0]) if i else None
    return sorted(y), sorted(n)


def letter_ids(tok):
    m = {}
    for i, L in enumerate(LETTERS):
        for w in (L, " " + L):
            ids = tok.encode(w, add_special_tokens=False)
            if ids and ids[0] not in m: m[ids[0]] = i
    return m


def find_final_norm(model):
    for name in ["model.language_model.norm", "language_model.model.norm", "model.model.norm", "model.norm"]:
        m = model; ok = True
        for p in name.split("."):
            if hasattr(m, p): m = getattr(m, p)
            else: ok = False; break
        if ok and isinstance(m, torch.nn.Module): return m
    return None


@torch.no_grad()
def fwd(model, proc, im, prompt, device, hidden=False):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt").to(device)
    return model(**inp, output_hidden_states=hidden)


def dola(out, lm_head, norm, cand):
    hs = out.hidden_states; flp = torch.log_softmax(out.logits[0, -1].float(), -1)
    ll = lambda h: lm_head(norm(h) if norm is not None else h).float()
    best, sel = -1.0, None
    for L in cand:
        elp = torch.log_softmax(ll(hs[L][0, -1]), -1)
        m = 0.5 * (flp.exp() + elp.exp())
        j = 0.5 * ((flp.exp() * (flp - m.log())).sum() + (elp.exp() * (elp - m.log())).sum())
        if j.item() > best: best, sel = j.item(), elp
    return flp - sel


def yn_pred(logit, yn):
    p = torch.softmax(logit.float(), -1); py, pn = p[yn[0]].sum().item(), p[yn[1]].sum().item()
    return ("yes" if py >= pn else "no"), max(py, pn) / (py + pn + 1e-9)


def letter_pred(logit, lmap, n):
    p = torch.softmax(logit.float(), -1); pr = {}
    for tid, idx in lmap.items():
        if idx < n: pr[idx] = pr.get(idx, 0.0) + p[tid].item()
    tot = sum(pr.values()) + 1e-9; k = max(pr, key=pr.get) if pr else 0
    return k, pr.get(k, 0.0) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct"); ap.add_argument("--mode", required=True)
    ap.add_argument("--pope", default=HARAM_ROOT + "/coco_build/data/pope_test_adversarial.json")
    ap.add_argument("--image-dir", default=HARAM_ROOT + "/coco_build/images")
    ap.add_argument("--vstar-root", default=""); ap.add_argument("--low", type=int, default=256)
    ap.add_argument("--high", type=int, default=1024); ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    lm_head = model.get_output_embeddings(); norm = find_final_norm(model)
    nlayers = len(model.model.language_model.layers) if hasattr(model.model, "language_model") else 36
    cand = list(range(2, nlayers // 2, 2)); rng = random.Random(0)
    recs = []; t0 = time.time()

    if args.mode == "pope":
        yn = yes_no_ids(proc.tokenizer)
        negs = [json.loads(l) for l in open(args.pope) if l.strip()]
        negs = [r for r in negs if r["label"] == "no"][: args.limit]
        for i, r in enumerate(negs):
            im = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
            o_lo = fwd(model, proc, resize_tokens(im, args.low), r["text"], device, hidden=True)
            o_hi = fwd(model, proc, resize_tokens(im, args.high), r["text"], device)
            lp, lc = yn_pred(o_lo.logits[0, -1], yn)
            dp, _ = yn_pred(dola(o_lo, lm_head, norm, cand), yn)
            hp, _ = yn_pred(o_hi.logits[0, -1], yn)
            recs.append({"image": r["image"], "query": r["text"], "label": "no",
                         "low": lp, "low_conf": lc, "low_dola": dp, "high": hp})
            if (i + 1) % 50 == 0: print(f"  {i+1}/{len(negs)} ({time.time()-t0:.0f}s)", flush=True)
    else:
        lmap = letter_ids(proc.tokenizer)
        items = []
        for sub in ["direct_attributes", "relative_position"]:
            for jf in sorted(glob.glob(os.path.join(args.vstar_root, sub, "*.json"))):
                j = json.load(open(jf)); img = None
                for ext in (".jpg", ".JPG", ".png", ".jpeg", ".webp", ".JPEG"):
                    if os.path.exists(jf[:-5] + ext): img = jf[:-5] + ext; break
                if img and j.get("options") and j.get("bbox"):
                    items.append({"image": img, "question": j["question"], "options": j["options"], "bbox": j["bbox"]})
        for i, it in enumerate(items):
            opts = list(it["options"]); correct = opts[0]; rng.shuffle(opts); gt = opts.index(correct)
            prompt = it["question"] + "\n" + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts)) + \
                     "\nAnswer with the letter only."
            im = Image.open(it["image"]).convert("RGB")
            o_lo = fwd(model, proc, resize_tokens(im, args.low), prompt, device, hidden=True)
            o_hi = fwd(model, proc, resize_tokens(im, args.high), prompt, device)
            lp, lc = letter_pred(o_lo.logits[0, -1], lmap, len(opts))
            dp, _ = letter_pred(dola(o_lo, lm_head, norm, cand), lmap, len(opts))
            hp, _ = letter_pred(o_hi.logits[0, -1], lmap, len(opts))
            recs.append({"image": it["image"], "question": it["question"], "options": opts, "gt": gt,
                         "bbox": it["bbox"], "low": lp, "low_conf": lc, "low_dola": dp, "high": hp})
            if (i + 1) % 40 == 0: print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    json.dump(recs, open(args.output, "w"), indent=1)
    print(f"wrote {len(recs)} -> {args.output}")


if __name__ == "__main__":
    main()
