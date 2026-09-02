#!/usr/bin/env python
"""
Deciding experiment: does DoLa (decoding-side) help on V*Bench (perception-bound), as it does on POPE?

DoLa strongly reduces POPE-adversarial hallucination by suppressing language priors (late-vs-early
layer contrast). The thesis predicts it should do NOTHING on V*Bench, where the failure is that the
model never resolved the tiny target (no layer has the information). We test low/high resolution with
and without DoLa on V*Bench (Qwen3-VL, MCQ). Run in 'qwen3' env.
"""
import argparse, glob, json, os, random, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

TOKPX = 32 * 32
LETTERS = "ABCDEFGH"


def resize_tokens(im, n):
    W, H = im.size; s = (n * TOKPX / max(1, W * H)) ** 0.5
    return im.resize((max(56, int(round(W * s))), max(56, int(round(H * s)))), Image.LANCZOS)


def letter_ids(tok):
    m = {}
    for i, L in enumerate(LETTERS):
        for w in (L, " " + L):
            ids = tok.encode(w, add_special_tokens=False)
            if ids and ids[0] not in m: m[ids[0]] = i
    return m


def load_vstar(root):
    items = []
    for sub in ["direct_attributes", "relative_position"]:
        for jf in sorted(glob.glob(os.path.join(root, sub, "*.json"))):
            j = json.load(open(jf)); img = None
            for ext in (".jpg", ".JPG", ".png", ".jpeg", ".webp", ".JPEG"):
                if os.path.exists(jf[:-5] + ext): img = jf[:-5] + ext; break
            if img and j.get("options"):
                items.append({"image": img, "question": j["question"], "options": j["options"]})
    return items


def find_final_norm(model):
    for name in ["model.language_model.norm", "language_model.model.norm", "model.model.norm", "model.norm"]:
        m = model; ok = True
        for p in name.split("."):
            if hasattr(m, p): m = getattr(m, p)
            else: ok = False; break
        if ok and isinstance(m, torch.nn.Module): return m
    return None


@torch.no_grad()
def fwd(model, proc, im, prompt, device):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt").to(device)
    return model(**inp, output_hidden_states=True)


def mcq(logit, lmap, n_opt):
    p = torch.softmax(logit.float(), -1); pr = {}
    for tid, idx in lmap.items():
        if idx < n_opt: pr[idx] = pr.get(idx, 0.0) + p[tid].item()
    return max(pr, key=pr.get) if pr else 0


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--vstar-root", required=True)
    ap.add_argument("--low", type=int, default=256); ap.add_argument("--high", type=int, default=1024)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"; rng = random.Random(0)
    items = load_vstar(args.vstar_root)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model); lmap = letter_ids(proc.tokenizer)
    lm_head = model.get_output_embeddings(); norm = find_final_norm(model)
    nlayers = len(model.model.language_model.layers) if hasattr(model.model, "language_model") else 36
    cand = list(range(2, nlayers // 2, 2))
    print(f"[vstar DoLa] {len(items)} | norm={'ok' if norm else 'MISSING'} cand={cand}", flush=True)
    C = {k: [] for k in ["low", "high", "low_dola", "high_dola"]}
    t0 = time.time()
    for i, it in enumerate(items):
        opts = list(it["options"]); correct = opts[0]; rng.shuffle(opts); gt = opts.index(correct)
        prompt = it["question"] + "\n" + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts)) + \
                 "\nAnswer with the letter only."
        im = Image.open(it["image"]).convert("RGB")
        o_lo = fwd(model, proc, resize_tokens(im, args.low), prompt, device)
        o_hi = fwd(model, proc, resize_tokens(im, args.high), prompt, device)
        C["low"].append(mcq(o_lo.logits[0, -1], lmap, len(opts)) == gt)
        C["high"].append(mcq(o_hi.logits[0, -1], lmap, len(opts)) == gt)
        C["low_dola"].append(mcq(dola(o_lo, lm_head, norm, cand), lmap, len(opts)) == gt)
        C["high_dola"].append(mcq(dola(o_hi, lm_head, norm, cand), lmap, len(opts)) == gt)
        if (i + 1) % 40 == 0: print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    out = {"n": len(items), "acc": {k: float(np.mean(v)) for k, v in C.items()}}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print("\n=== DoLa on V*Bench (perception-bound) ===")
    for k in ["low", "low_dola", "high", "high_dola"]:
        print(f"  {k:10} acc {out['acc'][k]:.3f}")
    print(f"  -> DoLa gain: low {out['acc']['low_dola']-out['acc']['low']:+.3f}, "
          f"high {out['acc']['high_dola']-out['acc']['high']:+.3f}  (thesis predicts ~0)")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
