#!/usr/bin/env python
"""
Foveation line: broaden the decoding-side comparison on clean POPE — perception vs {VCD, DoLa}.

All are training-free DECODING-side methods that apply to POPE's single yes/no token:
  VCD  (Leng 2024): (1+a)*logit(orig) - a*logit(noised image)        [2 forward passes]
  DoLa (Chuang 2024): log p(final layer) - log p(early layer)         [1 forward pass; layer contrast,
                      early layer chosen per-token by max JSD vs final]
We compare both against the PERCEPTION lever (raise resolution) on balanced clean POPE-adversarial.
(OPERA is a generation-time beam-search method and does NOT apply to a single token -> compared on
captioning/CHAIR separately.) Run in 'qwen3' env.
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
    a = np.asarray(im).astype(np.float32) + np.random.RandomState(0).normal(0, std * 255, (im.height, im.width, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def yes_no_ids(tok):
    y, n = set(), set()
    for w in ["yes", "Yes", " yes", " Yes", "YES"]:
        i = tok.encode(w, add_special_tokens=False);  y.add(i[0]) if i else None
    for w in ["no", "No", " no", " No", "NO"]:
        i = tok.encode(w, add_special_tokens=False);  n.add(i[0]) if i else None
    return sorted(y), sorted(n)


def find_final_norm(model):
    for name in ["model.language_model.norm", "language_model.model.norm", "model.model.norm",
                 "model.norm", "language_model.norm"]:
        m = model; ok = True
        for p in name.split("."):
            if hasattr(m, p): m = getattr(m, p)
            else: ok = False; break
        if ok and isinstance(m, torch.nn.Module): return m
    return None


@torch.no_grad()
def forward(model, proc, im, q, device, want_hidden=False):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[im], return_tensors="pt").to(device)
    out = model(**inp, output_hidden_states=want_hidden)
    return out


def ynprob(logit, yn):
    p = torch.softmax(logit.float(), -1)
    py, pn = p[yn[0]].sum().item(), p[yn[1]].sum().item()
    return py / (py + pn + 1e-9)


def dola_logit(out, lm_head, norm, cand):
    hs = out.hidden_states
    final_lp = torch.log_softmax(out.logits[0, -1].float(), -1)
    llogit = lambda h: lm_head(norm(h) if norm is not None else h).float()
    best, sel = -1.0, None
    for L in cand:
        elp = torch.log_softmax(llogit(hs[L][0, -1]), -1)
        m = 0.5 * (final_lp.exp() + elp.exp())
        jsd = 0.5 * ((final_lp.exp() * (final_lp - m.log())).sum() + (elp.exp() * (elp - m.log())).sum())
        if jsd.item() > best: best, sel = jsd.item(), elp
    return final_lp - sel               # contrast (log-ratio); read yes/no off this


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
    lm_head = model.get_output_embeddings(); norm = find_final_norm(model)
    nlayers = len(model.model.language_model.layers) if hasattr(model.model, "language_model") else 36
    cand = list(range(2, nlayers // 2, 2))     # DoLa premature-layer candidates (lower half)
    print(f"[decoding compare] {len(items)} | layers={nlayers} dola_cand={cand} norm={'ok' if norm else 'MISSING'}",
          flush=True)
    C = {k: [] for k in ["low", "mid", "high", "low_vcd", "high_vcd", "low_dola", "high_dola"]}
    lab = []; t0 = time.time()
    for i, r in enumerate(items):
        im = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
        o_lo = forward(model, proc, resize_tokens(im, args.low), r["text"], device, want_hidden=True)
        o_lo_n = forward(model, proc, resize_tokens(noise_image(im, args.noise_std), args.low), r["text"], device)
        o_mi = forward(model, proc, resize_tokens(im, args.mid), r["text"], device)
        o_hi = forward(model, proc, resize_tokens(im, args.high), r["text"], device, want_hidden=True)
        o_hi_n = forward(model, proc, resize_tokens(noise_image(im, args.noise_std), args.high), r["text"], device)
        lo, hi = o_lo.logits[0, -1], o_hi.logits[0, -1]
        C["low"].append(ynprob(lo, yn) >= 0.5)
        C["mid"].append(ynprob(o_mi.logits[0, -1], yn) >= 0.5)
        C["high"].append(ynprob(hi, yn) >= 0.5)
        C["low_vcd"].append(ynprob((1 + args.alpha) * lo - args.alpha * o_lo_n.logits[0, -1], yn) >= 0.5)
        C["high_vcd"].append(ynprob((1 + args.alpha) * hi - args.alpha * o_hi_n.logits[0, -1], yn) >= 0.5)
        C["low_dola"].append(ynprob(dola_logit(o_lo, lm_head, norm, cand), yn) >= 0.5)
        C["high_dola"].append(ynprob(dola_logit(o_hi, lm_head, norm, cand), yn) >= 0.5)
        lab.append(r["label"] == "yes")
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    lab = np.array(lab)
    toks = {"low": args.low, "mid": args.mid, "high": args.high, "low_vcd": 2 * args.low,
            "high_vcd": 2 * args.high, "low_dola": args.low, "high_dola": args.high}
    passes = {"low": 1, "mid": 1, "high": 1, "low_vcd": 2, "high_vcd": 2, "low_dola": 1, "high_dola": 1}
    out = {"n": len(items), "results": {k: {**prf(np.array(v), lab), "eff_tokens": toks[k], "passes": passes[k]}
                                        for k, v in C.items()}}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print("\n=== Perception vs Decoding (VCD, DoLa) on clean POPE-adversarial ===")
    print(f"{'condition':10} {'acc':>6} {'F1':>6} {'halluc':>7} {'tok':>6} {'pass':>5}")
    for k in ["low", "low_vcd", "low_dola", "mid", "high", "high_vcd", "high_dola"]:
        m = out["results"][k]
        print(f"{k:10} {m['acc']:.3f} {m['f1']:.3f} {m['halluc']:.3f} {m['eff_tokens']:6d} {m['passes']:5d}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
