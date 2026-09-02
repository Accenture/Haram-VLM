#!/usr/bin/env python
"""
Generate COCO captions under decoding/perception conditions for CHAIR evaluation (Qwen3-VL).

Conditions: baseline greedy; + DoLa (our validated layer-contrast, applied per generated token via
forward-hooks + a LogitsProcessor so HF handles the mRoPE/KV-cache while we inject the contrast); and
low- vs high-resolution baselines. COCO captioning is the prior-driven mode, so the thesis predicts
decoding (DoLa) reduces hallucination while raising resolution does not. chair.py scores the output.
Run in 'qwen3' env.
"""
import argparse, json, os, sys, time
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import LogitsProcessor, LogitsProcessorList, RepetitionPenaltyLogitsProcessor

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


TOKPX = 32 * 32
PROMPT = "Describe this image in detail."


def resize_tokens(im, n):
    W, H = im.size; s = (n * TOKPX / max(1, W * H)) ** 0.5
    return im.resize((max(56, int(round(W * s))), max(56, int(round(H * s)))), Image.LANCZOS)


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


class DoLaProcessor(LogitsProcessor):
    """Per-token DoLa: contrast final logits vs the JSD-selected early layer (captured by hooks)."""
    def __init__(self, layers, norm, lm_head, cand, capture):
        self.norm, self.lm_head, self.cand, self.capture = norm, lm_head, cand, capture

    def __call__(self, input_ids, scores):
        flp = torch.log_softmax(scores.float(), -1)              # [batch, vocab], final
        out = flp.clone()
        for b in range(scores.shape[0]):
            best, sel = -1.0, None
            for L in self.cand:
                h = self.capture.get(L)
                if h is None: continue
                elp = torch.log_softmax(self.lm_head(self.norm(h[b])).float(), -1)
                m = 0.5 * (flp[b].exp() + elp.exp())
                j = 0.5 * ((flp[b].exp() * (flp[b] - m.log())).sum() + (elp.exp() * (elp - m.log())).sum())
                if j.item() > best: best, sel = j.item(), elp
            if sel is not None: out[b] = flp[b] - sel
        return out


@torch.no_grad()
def gen_baseline(model, proc, im, device, max_new, num_beams=1):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt").to(device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, num_beams=num_beams)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().replace("\n", " ")


@torch.no_grad()
def gen_dola(model, proc, im, device, max_new, layers, norm, lm_head, cand, capture, handles, rep=1.2):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt").to(device)
    lp = LogitsProcessorList([DoLaProcessor(layers, norm, lm_head, cand, capture),
                              RepetitionPenaltyLogitsProcessor(rep)])
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, num_beams=1, logits_processor=lp)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--image-dir", default=HARAM_ROOT + "/coco_build/images")
    ap.add_argument("--instances", default=HARAM_ROOT + "/coco_build/annotations/instances_val2014.json")
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--low", type=int, default=256); ap.add_argument("--high", type=int, default=1024)
    ap.add_argument("--conditions", default="base_high,dola_high,base_low")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chair import load_gt
    img2cats, fn2id = load_gt(args.instances)
    files = sorted(f for f in fn2id if os.path.exists(os.path.join(args.image_dir, f)))
    files = files[:: max(1, len(files) // args.n_images)][: args.n_images]
    print(f"[captioning] {len(files)} images | conditions={args.conditions}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    layers, norm = get_backbone(model); lm_head = model.get_output_embeddings()
    nlayers = len(layers); cand = list(range(2, nlayers // 2, 2))

    # forward hooks to capture early-layer last-token hidden states each step
    capture = {}
    handles = []
    def mk_hook(L):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            capture[L] = h[:, -1, :].detach()
        return hook
    for L in cand:
        handles.append(layers[L].register_forward_hook(mk_hook(L)))
    print(f"  layers={nlayers} dola_cand={cand} norm={'ok' if norm is not None else 'MISS'}", flush=True)

    conds = args.conditions.split(",")
    res = {"base_high": args.high, "base_low": args.low, "dola_high": args.high,
           "base_beam_high": args.high}
    results = {c: [] for c in conds}
    t0 = time.time()
    for i, fn in enumerate(files):
        im = Image.open(os.path.join(args.image_dir, fn)).convert("RGB")
        for c in conds:
            imr = resize_tokens(im, res[c])
            if c == "dola_high":
                cap = gen_dola(model, proc, imr, device, args.max_new, layers, norm, lm_head, cand, capture, handles)
            elif c == "base_beam_high":
                cap = gen_baseline(model, proc, imr, device, args.max_new, num_beams=5)
            else:
                cap = gen_baseline(model, proc, imr, device, args.max_new, num_beams=1)
            results[c].append({"image": fn, "caption": cap})
        if (i + 1) % 25 == 0: print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

    for h in handles: h.remove()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump({"model": args.model, "n_images": len(files), "max_new": args.max_new,
               "conditions": conds, "captions": results}, open(args.output, "w"), indent=1)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
