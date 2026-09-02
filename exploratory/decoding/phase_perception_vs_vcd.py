#!/usr/bin/env python
"""
Foveation line (de-risk): PERCEPTION (resolution) vs DECODING (VCD) for hallucination, on the clean protocol.

VCD (Visual Contrastive Decoding, Leng et al. 2024) is a training-free DECODING-side mitigation:
contrast the logits on the original image vs a NOISED image (which amplifies language priors), so
  logit_vcd = (1+alpha) * logit(orig) - alpha * logit(noised).
We compare it head-to-head with the PERCEPTION-side lever (just raise resolution) on balanced clean
POPE-adversarial (Qwen3-VL). Conditions: low / mid / high resolution baselines, low+VCD, high+VCD.
Thesis confirmed if (a) high baseline cuts hallucination more than low+VCD at comparable compute, and
(b) high+VCD ≈ high (VCD adds little once the model can see). Run in 'qwen3' env.
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


def noise_image(im, std):
    rng = np.random.RandomState(0)
    a = np.asarray(im).astype(np.float32) + rng.normal(0, std * 255, (im.height, im.width, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def yes_no_ids(tok):
    y, n = set(), set()
    for w in ["yes", "Yes", " yes", " Yes", "YES"]:
        i = tok.encode(w, add_special_tokens=False);  y.add(i[0]) if i else None
    for w in ["no", "No", " no", " No", "NO"]:
        i = tok.encode(w, add_special_tokens=False);  n.add(i[0]) if i else None
    return sorted(y), sorted(n)


@torch.no_grad()
def logit_last(model, proc, im, q, device):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[im], return_tensors="pt").to(device)
    g = inp["image_grid_thw"]; m = getattr(proc.image_processor, "merge_size", 2)
    ntok = int((g[0].prod() // (m ** 2)).item())
    return model(**inp).logits[0, -1].float(), ntok


def ynprob(logit, yn):
    p = torch.softmax(logit, -1)
    py, pn = p[yn[0]].sum().item(), p[yn[1]].sum().item()
    return py / (py + pn + 1e-9)


def prf(pred_yes, label_yes):
    pred_yes, label_yes = np.asarray(pred_yes), np.asarray(label_yes)
    tp = (pred_yes & label_yes).sum(); fp = (pred_yes & ~label_yes).sum(); fn = (~pred_yes & label_yes).sum()
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    return dict(acc=float((pred_yes == label_yes).mean()), f1=float(2 * prec * rec / (prec + rec + 1e-9)),
                halluc=float(fp / max(1, (~label_yes).sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--pope", default=HARAM_ROOT + "/coco_build/data/pope_test_adversarial.json")
    ap.add_argument("--image-dir", default=HARAM_ROOT + "/coco_build/images")
    ap.add_argument("--low", type=int, default=256); ap.add_argument("--mid", type=int, default=512)
    ap.add_argument("--high", type=int, default=1024)
    ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--noise-std", type=float, default=0.6)
    ap.add_argument("--per-class", type=int, default=300); ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    rows = [json.loads(l) for l in open(args.pope) if l.strip()]
    items = [r for r in rows if r["label"] == "yes"][: args.per_class] + \
            [r for r in rows if r["label"] == "no"][: args.per_class]
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model); yn = yes_no_ids(proc.tokenizer)
    print(f"[perception vs VCD] {len(items)} items | low/mid/high={args.low}/{args.mid}/{args.high} "
          f"alpha={args.alpha} noise={args.noise_std}", flush=True)
    cond = {k: [] for k in ["low", "mid", "high", "low_vcd", "high_vcd", "low_noise", "high_noise"]}
    lab = []; t0 = time.time()
    for i, r in enumerate(items):
        im = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
        lo, t_lo = logit_last(model, proc, resize_tokens(im, args.low), r["text"], device)
        lo_n, _ = logit_last(model, proc, resize_tokens(noise_image(im, args.noise_std), args.low), r["text"], device)
        mi, _ = logit_last(model, proc, resize_tokens(im, args.mid), r["text"], device)
        hi, t_hi = logit_last(model, proc, resize_tokens(im, args.high), r["text"], device)
        hi_n, _ = logit_last(model, proc, resize_tokens(noise_image(im, args.noise_std), args.high), r["text"], device)
        cond["low"].append(ynprob(lo, yn) >= 0.5)
        cond["mid"].append(ynprob(mi, yn) >= 0.5)
        cond["high"].append(ynprob(hi, yn) >= 0.5)
        cond["low_vcd"].append(ynprob((1 + args.alpha) * lo - args.alpha * lo_n, yn) >= 0.5)
        cond["high_vcd"].append(ynprob((1 + args.alpha) * hi - args.alpha * hi_n, yn) >= 0.5)
        cond["low_noise"].append(ynprob(lo_n, yn) >= 0.5)     # VCD precondition: does noise amplify halluc?
        cond["high_noise"].append(ynprob(hi_n, yn) >= 0.5)
        lab.append(r["label"] == "yes")
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    lab = np.array(lab)
    toks = {"low": args.low, "mid": args.mid, "high": args.high,
            "low_vcd": 2 * args.low, "high_vcd": 2 * args.high,
            "low_noise": args.low, "high_noise": args.high}
    out = {"n": len(items), "alpha": args.alpha, "noise_std": args.noise_std,
           "results": {k: {**prf(np.array(v), lab), "eff_tokens": toks[k], "passes": 2 if "vcd" in k else 1}
                       for k, v in cond.items()}}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print("\n=== Perception (resolution) vs Decoding (VCD) on clean POPE-adversarial ===")
    print(f"{'condition':10} {'acc':>6} {'F1':>6} {'halluc':>7} {'tok':>6} {'passes':>7}")
    for k in ["low", "low_vcd", "mid", "high", "high_vcd"]:
        m = out["results"][k]
        print(f"{k:10} {m['acc']:.3f} {m['f1']:.3f} {m['halluc']:.3f} {m['eff_tokens']:6d} {m['passes']:7d}")
    r = out["results"]
    print(f"\nVCD precondition (does noise amplify hallucination?): "
          f"low halluc {r['low']['halluc']:.3f} -> low_noise {r['low_noise']['halluc']:.3f} "
          f"| high {r['high']['halluc']:.3f} -> high_noise {r['high_noise']['halluc']:.3f}")
    print(f"thesis checks:")
    print(f"  perception vs decoding @ ~matched compute: mid(512tok)={r['mid']['halluc']:.3f} "
          f"vs low+VCD(512tok)={r['low_vcd']['halluc']:.3f}  (lower halluc wins)")
    print(f"  does VCD add once you can see?: high={r['high']['halluc']:.3f} vs high+VCD={r['high_vcd']['halluc']:.3f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
