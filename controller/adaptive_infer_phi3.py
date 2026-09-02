#!/usr/bin/env python
"""
Training-free adaptive-resolution inference (scout -> risk -> escalate).

For each POPE probe we run a cheap LOW-resolution "scout" pass and a HIGH-resolution
pass, recording each pass's yes/no prediction, the scout's confidence (yes/no logit
margin), and the visual-token cost of each pass. Nothing is trained: the controller
is a single threshold on scout confidence.

Because both passes are recorded once, the escalation threshold tau can be swept
entirely offline to trace the accuracy / hallucination vs. average-token PARETO,
with the two fixed-resolution baselines as its endpoints:
  tau=0  -> never escalate  == always-low
  tau=1  -> always escalate == always-high
Honest cost model: the scout always runs; escalation adds the high-res pass.
"""
import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor

# The vendored Phi-3-V modeling code lives in the haram_vlm package.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HARAM_ROOT, "haram_vlm", "src"))
from model.Phi3_vision.modeling_phi3_v import Phi3VForCausalLM, Phi3VConfig  # noqa: E402

BASE_MODEL = "microsoft/Phi-3-vision-128k-instruct"


def load_model(ckpt, base, device):
    cfg = Phi3VConfig.from_pretrained(base)
    cfg._attn_implementation = "eager"
    model = Phi3VForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                             config=cfg, low_cpu_mem_usage=True)
    if ckpt and ckpt.lower() != "base":
        model = PeftModel.from_pretrained(model, ckpt)
        model = model.merge_and_unload()
    model.to(device).eval()
    return model


def make_processor(src, num_crops):
    p = AutoProcessor.from_pretrained(src, trust_remote_code=True, num_crops=num_crops)
    try:
        p.image_processor.num_crops = num_crops
    except Exception:
        pass
    return p


def yes_no_ids(tok):
    yes, no = set(), set()
    for w in ["yes", "Yes", " yes", " Yes", "YES"]:
        ids = tok.encode(w, add_special_tokens=False)
        if ids:
            yes.add(ids[0])
    for w in ["no", "No", " no", " No", "NO"]:
        ids = tok.encode(w, add_special_tokens=False)
        if ids:
            no.add(ids[0])
    return sorted(yes), sorted(no)


@torch.no_grad()
def yn_pass(model, proc, image, question, device, yes_ids, no_ids):
    content = f"<|image_1|>\n{question}"
    prompt = proc.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inputs = proc(prompt, [image], return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1].float()            # next-token distribution
    probs = torch.softmax(logits, dim=-1)
    p_yes = probs[yes_ids].sum().item()
    p_no = probs[no_ids].sum().item()
    z = p_yes + p_no + 1e-9
    pred = "yes" if p_yes >= p_no else "no"
    conf = max(p_yes, p_no) / z                                # decisiveness in [0.5,1]
    n_tok = int(inputs["input_ids"].shape[1])                  # cost proxy (mostly image tokens)
    return pred, conf, n_tok


def metrics(items):
    tp = sum(1 for r in items if r["label"] == "yes" and r["pred"] == "yes")
    tn = sum(1 for r in items if r["label"] == "no" and r["pred"] == "no")
    fp = sum(1 for r in items if r["label"] == "no" and r["pred"] == "yes")
    fn = sum(1 for r in items if r["label"] == "yes" and r["pred"] == "no")
    n = len(items); neg = tn + fp
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return {"acc": (tp + tn) / n, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0,
            "halluc": fp / neg if neg else 0}


def sweep(records):
    """Sweep tau over scout-confidence; return Pareto points + fixed baselines."""
    tok_low = np.mean([r["tok_low"] for r in records])
    tok_high = np.mean([r["tok_high"] for r in records])
    points = []
    for tau in np.linspace(0.5, 1.0, 26):
        items, cost = [], 0.0
        for r in records:
            escalate = r["conf_low"] < tau
            pred = r["pred_high"] if escalate else r["pred_low"]
            items.append({"label": r["label"], "pred": pred})
            cost += r["tok_low"] + (r["tok_high"] if escalate else 0)
        m = metrics(items)
        m.update(tau=round(float(tau), 3),
                 esc_rate=np.mean([r["conf_low"] < tau for r in records]),
                 avg_tokens=cost / len(records))
        points.append(m)
    base_low = {**metrics([{"label": r["label"], "pred": r["pred_low"]} for r in records]),
                "avg_tokens": tok_low, "name": "always-low"}
    base_high = {**metrics([{"label": r["label"], "pred": r["pred_high"]} for r in records]),
                 "avg_tokens": tok_high, "name": "always-high"}
    return {"pareto": points, "always_low": base_low, "always_high": base_high,
            "tok_low": tok_low, "tok_high": tok_high}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LoRA checkpoint dir, or 'base'")
    ap.add_argument("--proc-src", default=None, help="processor source (default: ckpt or base)")
    ap.add_argument("--pope", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--low-crops", type=int, default=4)
    ap.add_argument("--high-crops", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"

    rows = [json.loads(l) for l in open(args.pope) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    model = load_model(args.ckpt, BASE_MODEL, device)
    src = args.proc_src or (args.ckpt if args.ckpt.lower() != "base" else BASE_MODEL)
    proc_lo = make_processor(src, args.low_crops)
    proc_hi = make_processor(src, args.high_crops)
    yes_ids, no_ids = yes_no_ids(proc_lo.tokenizer)
    print(f"[{os.path.basename(args.pope)}] {len(rows)} probes | crops {args.low_crops}->{args.high_crops}", flush=True)

    records = []
    import time; t0 = time.time()
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
        pl, cl, tl = yn_pass(model, proc_lo, img, r["text"], device, yes_ids, no_ids)
        ph, ch, th = yn_pass(model, proc_hi, img, r["text"], device, yes_ids, no_ids)
        records.append({"label": r["label"], "pred_low": pl, "conf_low": cl, "tok_low": tl,
                        "pred_high": ph, "conf_high": ch, "tok_high": th})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    out = {"pope": os.path.basename(args.pope), "ckpt": args.ckpt,
           "low_crops": args.low_crops, "high_crops": args.high_crops,
           **sweep(records), "records": records}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    bl, bh = out["always_low"], out["always_high"]
    print(f"  always-low : acc={bl['acc']:.3f} halluc={bl['halluc']*100:.1f}% tok={bl['avg_tokens']:.0f}")
    print(f"  always-high: acc={bh['acc']:.3f} halluc={bh['halluc']*100:.1f}% tok={bh['avg_tokens']:.0f}")
    # a sample adaptive point near the knee
    knee = min(out["pareto"], key=lambda p: (p["avg_tokens"]-bl["avg_tokens"]) if p["acc"]>=bh["acc"]-0.005 else 1e9)
    print(f"  adaptive@acc~high: tau={knee['tau']} acc={knee['acc']:.3f} halluc={knee['halluc']*100:.1f}% "
          f"tok={knee['avg_tokens']:.0f} esc_rate={knee['esc_rate']:.2f}")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
