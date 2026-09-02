#!/usr/bin/env python
"""
Move 3a: the mechanism behind the 2x2. Why does DoLa fix prior-driven but not perception-bound?

DoLa recovers an answer when it gains probability in LATE layers relative to early ones. We make this
visible with a logit lens: at LOW resolution, project every layer's last-token hidden state through the
same final-norm + lm_head DoLa uses, and record P(correct answer) at each layer.
  - Prior-driven (POPE-adv): the model says the wrong thing greedily, but the correct token should RISE
    in late layers (the late layer "knows"; the prior dominated the argmax) -> DoLa recovers it.
  - Perception-bound (V*Bench): the correct token is never resolved, so it should stay FLAT and LOW at
    EVERY layer -> no contrast can recover it. "The evidence is in no layer."
Dumps per-item trajectories (+ low-res correctness) to JSON. Run in 'qwen3' env.
"""
import argparse, glob, json, os, random, time
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


def get_backbone(model):
    for name in ["model.language_model", "language_model.model", "model.model.language_model",
                 "model.model", "language_model", "model"]:
        m = model; ok = True
        for p in name.split("."):
            if hasattr(m, p): m = getattr(m, p)
            else: ok = False; break
        if ok and hasattr(m, "layers") and hasattr(m, "norm"):
            return m.layers, m.norm
    return None, None


@torch.no_grad()
def fwd(model, proc, im, prompt, device):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt").to(device)
    return model(**inp, output_hidden_states=True)


def dola_logit(out, lm_head, norm, cand):
    hs = out.hidden_states
    flp = torch.log_softmax(out.logits[0, -1].float(), -1)
    ll = lambda h: lm_head(norm(h) if norm is not None else h).float()
    best, sel, selL = -1.0, None, None
    for L in cand:
        elp = torch.log_softmax(ll(hs[L][0, -1]), -1)
        m = 0.5 * (flp.exp() + elp.exp())
        j = 0.5 * ((flp.exp() * (flp - m.log())).sum() + (elp.exp() * (elp - m.log())).sum())
        if j.item() > best: best, sel, selL = j.item(), elp, L
    return flp - sel, selL


@torch.no_grad()
def trajectory(out, lm_head, norm, correct_ids, pool_ids):
    """P(correct) among `pool_ids` at every layer via the logit lens. The final point uses the model's
    true output logits (out.logits) to avoid double-applying the final norm to hidden_states[-1]
    (which several HF VLMs return already-normed). Intermediate layers L=0..N-1 use norm(hs[L])."""
    hs = out.hidden_states  # len = nlayers+1 (embeddings + each layer)
    def frac(logit):
        p = torch.softmax(logit.float(), -1)
        return p[correct_ids].sum().item() / (p[pool_ids].sum().item() + 1e-9)
    traj = [frac(lm_head(norm(h[0, -1]))) for h in hs[:-1]]   # logit lens, intermediate layers
    traj.append(frac(out.logits[0, -1]))                      # true final-layer distribution
    return traj


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--pope", default=HARAM_ROOT + "/coco_build/data/pope_test_adversarial.json")
    ap.add_argument("--image-dir", default=HARAM_ROOT + "/coco_build/images")
    ap.add_argument("--vstar-root", required=True)
    ap.add_argument("--low", type=int, default=256)
    ap.add_argument("--pope-per-class", type=int, default=150)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    lm_head = model.get_output_embeddings(); layers, norm = get_backbone(model)
    nlayers = len(layers); cand = list(range(2, nlayers // 2, 2))
    yn = yes_no_ids(proc.tokenizer); yes_ids, no_ids = yn; pool_yn = yes_ids + no_ids
    lmap = letter_ids(proc.tokenizer)
    print(f"[layer-traj] layers={nlayers} norm={'ok' if norm is not None else 'MISS'}", flush=True)

    recs = []; t0 = time.time()
    rows = [json.loads(l) for l in open(args.pope) if l.strip()]
    pope = [r for r in rows if r["label"] == "yes"][: args.pope_per_class] + \
           [r for r in rows if r["label"] == "no"][: args.pope_per_class]
    for i, r in enumerate(pope):
        im = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
        out = fwd(model, proc, resize_tokens(im, args.low), r["text"], device)
        cids = yes_ids if r["label"] == "yes" else no_ids
        traj = trajectory(out, lm_head, norm, cids, pool_yn)
        fl = out.logits[0, -1].float()
        greedy = "yes" if torch.softmax(fl, -1)[yes_ids].sum() >= torch.softmax(fl, -1)[no_ids].sum() else "no"
        dl, selL = dola_logit(out, lm_head, norm, cand)
        dpr = torch.softmax(dl, -1); dola = "yes" if dpr[yes_ids].sum() >= dpr[no_ids].sum() else "no"
        recs.append({"task": "pope", "label": r["label"], "low_correct": greedy == r["label"],
                     "greedy_ok": greedy == r["label"], "dola_ok": dola == r["label"],
                     "dola_layer": int(selL), "traj": traj})
        if (i + 1) % 60 == 0: print(f"  pope {i+1}/{len(pope)} ({time.time()-t0:.0f}s)", flush=True)

    rng = random.Random(0); vstar = load_vstar(args.vstar_root)
    for i, it in enumerate(vstar):
        opts = list(it["options"]); correct = opts[0]; rng.shuffle(opts); gt = opts.index(correct)
        prompt = it["question"] + "\n" + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts)) + \
                 "\nAnswer with the letter only."
        im = Image.open(it["image"]).convert("RGB")
        out = fwd(model, proc, resize_tokens(im, args.low), prompt, device)
        pool = [tid for tid, idx in lmap.items() if idx < len(opts)]
        cids = [tid for tid, idx in lmap.items() if idx == gt]
        traj = trajectory(out, lm_head, norm, cids, pool)
        def argmax_letter(logit):
            p = torch.softmax(logit.float(), -1)
            pr = {idx: sum(p[tid].item() for tid, j in lmap.items() if j == idx) for idx in range(len(opts))}
            return max(pr, key=pr.get)
        greedy = argmax_letter(out.logits[0, -1])
        dl, selL = dola_logit(out, lm_head, norm, cand); dola = argmax_letter(dl)
        recs.append({"task": "vstar", "label": None, "low_correct": greedy == gt,
                     "greedy_ok": greedy == gt, "dola_ok": dola == gt,
                     "dola_layer": int(selL), "traj": traj})
        if (i + 1) % 50 == 0: print(f"  vstar {i+1}/{len(vstar)} ({time.time()-t0:.0f}s)", flush=True)

    out_obj = {"model": args.model, "nlayers": len(layers), "low": args.low,
               "n_pope": len(pope), "n_vstar": len(vstar), "records": recs}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out_obj, open(args.output, "w"))
    # quick summary: mean final-layer P(correct) by subset, split by correctness
    for task in ["pope", "vstar"]:
        sub = [r for r in recs if r["task"] == task]
        wrong = [r for r in sub if not r["low_correct"]]
        if wrong:
            early = np.mean([np.mean(r["traj"][2:6]) for r in wrong])
            late = np.mean([r["traj"][-1] for r in wrong])
            print(f"  {task} low-WRONG (n={len(wrong)}): early-layer P(correct)={early:.3f} -> final={late:.3f} "
                  f"(rise {late-early:+.3f})")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
