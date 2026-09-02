#!/usr/bin/env python
"""
Foveation line: resolution self-consistency for hallucination (FAIR test, both labels).

Earlier "any-of-K -> no" was biased toward 'no' (only helps negatives). Here we test the honest
questions on a BALANCED POPE-adversarial subset (positives + negatives), running each image at K
resolutions and asking yes/no:
  (1) Does aggregating across resolutions (majority vote / mean yes-prob) beat uniform-high F1?
      [a test-time hallucination *reducer*; costs more compute, valued for reliability]
  (2) Is cross-resolution INSTABILITY (std of yes-prob) a hallucination *detector*?
      [AUROC of instability vs. the high-res answer being wrong -> selective prediction / abstain]
Run in 'qwen3' env.
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


TOKPX = 32 * 32


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


@torch.no_grad()
def yesprob(model, proc, im, q, device, yn):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[im], return_tensors="pt").to(device)
    p = torch.softmax(model(**inp).logits[0, -1].float(), -1)
    py, pn = p[yn[0]].sum().item(), p[yn[1]].sum().item()
    return py / (py + pn + 1e-9)


def auroc(score, y):
    y = np.asarray(y, bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    order = np.argsort(score); r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def prf(pred_yes, label_yes):
    pred_yes, label_yes = np.asarray(pred_yes), np.asarray(label_yes)
    tp = (pred_yes & label_yes).sum(); fp = (pred_yes & ~label_yes).sum(); fn = (~pred_yes & label_yes).sum()
    acc = (pred_yes == label_yes).mean()
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9); f1 = 2 * prec * rec / (prec + rec + 1e-9)
    halluc = (pred_yes & ~label_yes).sum() / max(1, (~label_yes).sum())  # yes on negatives
    return dict(acc=float(acc), f1=float(f1), halluc=float(halluc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--pope", default=HARAM_ROOT + "/coco_build/data/pope_test_adversarial.json")
    ap.add_argument("--image-dir", default=HARAM_ROOT + "/coco_build/images")
    ap.add_argument("--res", default="64,128,256,512,1024")
    ap.add_argument("--per-class", type=int, default=300); ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    res = [int(x) for x in args.res.split(",")]
    rows = [json.loads(l) for l in open(args.pope) if l.strip()]
    pos = [r for r in rows if r["label"] == "yes"][: args.per_class]
    neg = [r for r in rows if r["label"] == "no"][: args.per_class]
    items = pos + neg
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model); yn = yes_no_ids(proc.tokenizer)
    print(f"[self-consistency] {len(items)} ({len(pos)}+/{len(neg)}-) | res={res}", flush=True)
    P = np.zeros((len(items), len(res))); lab = []
    t0 = time.time()
    for i, r in enumerate(items):
        im = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
        for k, rr in enumerate(res):
            P[i, k] = yesprob(model, proc, resize_tokens(im, rr), r["text"], device, yn)
        lab.append(r["label"] == "yes")
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    lab = np.array(lab)

    out = {"res": res, "per_res": {}, "n": len(items)}
    for k, rr in enumerate(res):
        out["per_res"][rr] = prf(P[:, k] >= 0.5, lab)
    hi = P[:, -1]                                   # uniform-high = the largest resolution
    out["uniform_high"] = prf(hi >= 0.5, lab)
    out["majority_vote"] = prf((P >= 0.5).mean(1) >= 0.5, lab)
    out["mean_prob"] = prf(P.mean(1) >= 0.5, lab)
    # instability as a hallucination DETECTOR (vs the high-res answer being wrong)
    err_high = (hi >= 0.5) != lab
    conf_unc = np.minimum(hi, 1 - hi)               # baseline: high-res uncertainty (1 - confidence)
    out["instability_detector"] = {"auroc_std": float(auroc(P.std(1), err_high)),
                                   "auroc_range": float(auroc(P.max(1) - P.min(1), err_high)),
                                   "auroc_high_conf": float(auroc(conf_unc, err_high)),   # baseline to beat
                                   "auroc_std+conf": float(auroc(P.std(1) + conf_unc, err_high)),
                                   "high_err_rate": float(err_high.mean())}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print("\n=== Resolution self-consistency (POPE-adversarial, balanced) ===")
    print(f"{'setting':16} {'acc':>6} {'F1':>6} {'halluc':>7}")
    for rr in res:
        m = out["per_res"][rr]; print(f"res {rr:<11} {m['acc']:.3f} {m['f1']:.3f} {m['halluc']:.3f}")
    for nm in ["uniform_high", "majority_vote", "mean_prob"]:
        m = out[nm]; print(f"{nm:16} {m['acc']:.3f} {m['f1']:.3f} {m['halluc']:.3f}")
    d = out["instability_detector"]
    print(f"\nhallucination DETECTOR (AUROC for high-res error, rate {d['high_err_rate']:.3f}):")
    print(f"  cross-res instability(std) = {d['auroc_std']:.3f}   | range = {d['auroc_range']:.3f}")
    print(f"  baseline high-res confidence = {d['auroc_high_conf']:.3f}   | std+conf = {d['auroc_std+conf']:.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
