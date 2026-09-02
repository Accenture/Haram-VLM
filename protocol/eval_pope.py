#!/usr/bin/env python
"""
POPE evaluation for a trained HARAM LoRA checkpoint.

Loads base Phi-3-Vision + the trained LoRA adapter (merged), runs greedy yes/no
generation on a POPE split using the SAME prompt format as training, and reports
POPE metrics. Because the HARAM training data was built from POPE, each question
is tagged as `seen` (its (image, question) pair appears in the training json) or
`unseen`; metrics are reported overall AND split by seen/unseen so the memorized
vs. generalized performance can be told apart.

Single-GPU. Pin a GPU with CUDA_VISIBLE_DEVICES to run splits in parallel.
"""
import argparse, json, os, sys, time

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
    config = Phi3VConfig.from_pretrained(base)
    config._attn_implementation = "eager"  # robust; sequences here are short
    model = Phi3VForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, config=config, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, ckpt)
    model = model.merge_and_unload()
    model.to(device).eval()
    try:
        processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(base, trust_remote_code=True, num_crops=4)
    return model, processor


def train_pairs(train_json):
    pairs = set()
    if train_json and os.path.exists(train_json):
        for r in json.load(open(train_json)):
            q = r["conversations"][0]["value"].replace("<image>", "").strip().lower()
            pairs.add((os.path.basename(r["image"]), q))
    return pairs


@torch.no_grad()
def answer(model, processor, image, question, device):
    content = f"<|image_1|>\n{question}"
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(prompt, [image], return_tensors="pt").to(device)
    out = model.generate(
        **inputs, max_new_tokens=8, do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    gen = out[0, inputs["input_ids"].shape[1]:]
    txt = processor.tokenizer.decode(gen, skip_special_tokens=True).strip().lower()
    if txt.startswith("yes") or ("yes" in txt and "no" not in txt):
        return "yes", txt
    if txt.startswith("no") or "no" in txt:
        return "no", txt
    return "unk", txt


def metrics(rows):
    # positive class = "yes"
    tp = sum(1 for r in rows if r["label"] == "yes" and r["pred"] == "yes")
    tn = sum(1 for r in rows if r["label"] == "no" and r["pred"] == "no")
    fp = sum(1 for r in rows if r["label"] == "no" and r["pred"] == "yes")
    fn = sum(1 for r in rows if r["label"] == "yes" and r["pred"] == "no")
    n = len(rows)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    yes_ratio = sum(1 for r in rows if r["pred"] == "yes") / n if n else 0.0
    return dict(n=n, accuracy=round(acc, 4), precision=round(prec, 4),
                recall=round(rec, 4), f1=round(f1, 4), yes_ratio=round(yes_ratio, 4),
                tp=tp, tn=tn, fp=fp, fn=fn,
                unk=sum(1 for r in rows if r["pred"] == "unk"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--pope", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--train-json", default=os.path.join(HARAM_ROOT, "coco_build", "data", "haram_train_cocoLarge.json"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all questions")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda"
    rows = [json.loads(l) for l in open(args.pope) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    seen = train_pairs(args.train_json)

    print(f"[{os.path.basename(args.pope)}] loading model on {os.environ.get('CUDA_VISIBLE_DEVICES','?')} ...", flush=True)
    model, processor = load_model(args.ckpt, args.base, device)

    results, t0 = [], time.time()
    for i, r in enumerate(rows):
        img = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
        pred, raw = answer(model, processor, img, r["text"], device)
        is_seen = (r["image"], r["text"].strip().lower()) in seen
        results.append({**r, "pred": pred, "raw": raw, "seen": is_seen})
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)}  ({el:.0f}s, {(i+1)/el:.1f} q/s)", flush=True)

    overall = metrics(results)
    seen_m = metrics([r for r in results if r["seen"]])
    unseen_m = metrics([r for r in results if not r["seen"]])
    out = {
        "pope": os.path.basename(args.pope),
        "ckpt": args.ckpt,
        "overall": overall,
        "seen": seen_m,
        "unseen": unseen_m,
        "predictions": results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print(f"\n[{out['pope']}] DONE in {time.time()-t0:.0f}s")
    print(f"  overall: {overall}")
    print(f"  seen   : {seen_m}")
    print(f"  unseen : {unseen_m}")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
