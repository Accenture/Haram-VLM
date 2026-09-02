#!/usr/bin/env python
"""
Move 1 (third architecture): replicate the 2x2 on Phi-3-Vision (CLIP-crop family, distinct from
Qwen3's continuous resize and InternVL's 448px tiling).

Phi-3-V diverges from the apply_chat_template/AutoModelForImageTextToText path used by arch_2x2.py:
  - custom Phi3VForCausalLM from the local src tree (run in the 'haram' env, transformers 4.43)
  - prompt format "<|image_1|>\n{q}" via proc(prompt, [image])
  - resolution lever = num_crops (low=1 -> high=16)
The DoLa / yes-no / MCQ logic is identical to arch_2x2.py. Backbone for DoLa: model.model.{layers,norm},
lm_head = model.lm_head.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

ROOT = HARAM_ROOT + "/haram_vlm"
sys.path.insert(0, os.path.join(ROOT, "src"))
from model.Phi3_vision.modeling_phi3_v import Phi3VForCausalLM, Phi3VConfig  # noqa: E402

BASE = "microsoft/Phi-3-vision-128k-instruct"
LETTERS = "ABCDEFGH"


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


def letter_ids(tok):
    m = {}
    for i, L in enumerate(LETTERS):
        for w in (L, " " + L):
            ids = tok.encode(w, add_special_tokens=False)
            if ids and ids[0] not in m: m[ids[0]] = i
    return m


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


def ynprob(logit, yn):
    p = torch.softmax(logit.float(), -1); py, pn = p[yn[0]].sum().item(), p[yn[1]].sum().item()
    return py / (py + pn + 1e-9)


def mcq(logit, lmap, n_opt):
    p = torch.softmax(logit.float(), -1); pr = {}
    for tid, idx in lmap.items():
        if idx < n_opt: pr[idx] = pr.get(idx, 0.0) + p[tid].item()
    return max(pr, key=pr.get) if pr else 0


def prf(pred_yes, label_yes):
    pred_yes, label_yes = np.asarray(pred_yes), np.asarray(label_yes)
    tp = (pred_yes & label_yes).sum(); fp = (pred_yes & ~label_yes).sum(); fn = (~pred_yes & label_yes).sum()
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    return dict(acc=float((pred_yes == label_yes).mean()), f1=float(2 * prec * rec / (prec + rec + 1e-9)),
                halluc=float(fp / max(1, (~label_yes).sum())))


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
    ap.add_argument("--task", required=True, choices=["pope", "vstar"])
    ap.add_argument("--pope", default=HARAM_ROOT + "/coco_build/data/pope_test_adversarial.json")
    ap.add_argument("--image-dir", default=HARAM_ROOT + "/coco_build/images")
    ap.add_argument("--vstar-root", default="")
    ap.add_argument("--low-crops", type=int, default=1); ap.add_argument("--high-crops", type=int, default=16)
    ap.add_argument("--per-class", type=int, default=300); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"

    cfg = Phi3VConfig.from_pretrained(BASE); cfg._attn_implementation = "eager"
    model = Phi3VForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, config=cfg,
                                             low_cpu_mem_usage=True).to(device).eval()
    proc_lo = make_processor(args.low_crops); proc_hi = make_processor(args.high_crops)
    lm_head = model.lm_head; norm = model.model.norm
    nlayers = len(model.model.layers); cand = list(range(2, nlayers // 2, 2))
    print(f"[phi3 2x2 {args.task}] layers={nlayers} dola_cand={cand} crops {args.low_crops}->{args.high_crops}",
          flush=True)
    t0 = time.time()

    if args.task == "pope":
        yn = yes_no_ids(proc_lo.tokenizer)
        rows = [json.loads(l) for l in open(args.pope) if l.strip()]
        items = [r for r in rows if r["label"] == "yes"][: args.per_class] + \
                [r for r in rows if r["label"] == "no"][: args.per_class]
        if args.limit: items = items[: args.limit] + items[-args.limit:]
        C = {k: [] for k in ["low", "high", "low_dola", "high_dola"]}; lab = []
        for i, r in enumerate(items):
            im = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
            o_lo = fwd(model, proc_lo, im, r["text"], device)
            o_hi = fwd(model, proc_hi, im, r["text"], device)
            C["low"].append(ynprob(o_lo.logits[0, -1], yn) >= 0.5)
            C["high"].append(ynprob(o_hi.logits[0, -1], yn) >= 0.5)
            C["low_dola"].append(ynprob(dola_logit(o_lo, lm_head, norm, cand), yn) >= 0.5)
            C["high_dola"].append(ynprob(dola_logit(o_hi, lm_head, norm, cand), yn) >= 0.5)
            lab.append(r["label"] == "yes")
            if (i + 1) % 50 == 0: print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        lab = np.array(lab)
        out = {"arch": "phi3", "task": "pope", "n": len(items),
               "results": {k: prf(np.array(v), lab) for k, v in C.items()}}
        print("\n=== prior-driven (POPE-adv) ===")
        for k in ["low", "high", "low_dola", "high_dola"]:
            m = out["results"][k]; print(f"  {k:10} halluc {m['halluc']:.3f}  F1 {m['f1']:.3f}")
    else:
        lmap = letter_ids(proc_lo.tokenizer)
        import random; rng = random.Random(0)
        items = load_vstar(args.vstar_root)
        if args.limit: items = items[: args.limit]
        C = {k: [] for k in ["low", "high", "low_dola", "high_dola"]}
        for i, it in enumerate(items):
            opts = list(it["options"]); correct = opts[0]; rng.shuffle(opts); gt = opts.index(correct)
            prompt = it["question"] + "\n" + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts)) + \
                     "\nAnswer with the letter only."
            im = Image.open(it["image"]).convert("RGB")
            o_lo = fwd(model, proc_lo, im, prompt, device)
            o_hi = fwd(model, proc_hi, im, prompt, device)
            C["low"].append(mcq(o_lo.logits[0, -1], lmap, len(opts)) == gt)
            C["high"].append(mcq(o_hi.logits[0, -1], lmap, len(opts)) == gt)
            C["low_dola"].append(mcq(dola_logit(o_lo, lm_head, norm, cand), lmap, len(opts)) == gt)
            C["high_dola"].append(mcq(dola_logit(o_hi, lm_head, norm, cand), lmap, len(opts)) == gt)
            if (i + 1) % 40 == 0: print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        out = {"arch": "phi3", "task": "vstar", "n": len(items),
               "acc": {k: float(np.mean(v)) for k, v in C.items()}}
        print("\n=== perception-bound (V*Bench) ===")
        for k in ["low", "high", "low_dola", "high_dola"]:
            print(f"  {k:10} acc {out['acc'][k]:.3f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
