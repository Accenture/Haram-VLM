#!/usr/bin/env python
"""
Move 1: replicate the central 2x2 (cure-depends-on-cause) on a SECOND/THIRD architecture.

For one architecture, measure the two interventions on the two failure modes:
  prior-driven   = POPE-adversarial (yes/no)  -> halluc rate (lower better)
  perception-bound = V*Bench (MCQ)            -> accuracy   (higher better)
Interventions:
  raise resolution (perception lever)  : low -> high
  DoLa             (decoding lever)     : low -> low+DoLa   (Chuang 2024, layer contrast; 1 pass)

Architectures share the apply_chat_template / AutoModelForImageTextToText API; they differ only in
  (a) the resolution lever  -- Qwen3: continuous pixel-budget resize ; InternVL: dynamic 448px tiling
  (b) the LLM backbone paths for DoLa (decoder layers + final RMSNorm), detected robustly below.
Run in the 'qwen3' env (transformers 5.x). HF_HOME must point at the shared cache.
"""
import argparse, glob, json, os, time
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


# ---------- resolution lever ----------
def resize_tokens(im, n):
    W, H = im.size; s = (n * TOKPX / max(1, W * H)) ** 0.5
    return im.resize((max(56, int(round(W * s))), max(56, int(round(H * s)))), Image.LANCZOS)


def set_tiles(proc, max_tiles):
    ip = getattr(proc, "image_processor", None)
    if ip is None: return
    for attr, val in [("crop_to_patches", max_tiles > 1), ("max_patches", int(max_tiles)), ("min_patches", 1)]:
        if hasattr(ip, attr): setattr(ip, attr, val)


def prep_image(arch, im, level, lvls):
    """Return (image_to_pass) and mutate processor tiling state as a side effect for InternVL."""
    if arch == "qwen3":
        return resize_tokens(im, lvls[level])
    return im  # internvl: tiling set separately via set_tiles


# ---------- token / letter ids ----------
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


# ---------- LLM backbone introspection for DoLa ----------
def get_backbone(model):
    """Find the text decoder stack: returns (layers_modulelist, final_norm). Robust across -hf wrappers."""
    cands = ["model.language_model", "language_model.model", "model.model.language_model",
             "model.model", "language_model", "model"]
    for name in cands:
        m = model; ok = True
        for p in name.split("."):
            if hasattr(m, p): m = getattr(m, p)
            else: ok = False; break
        if ok and hasattr(m, "layers") and hasattr(m, "norm"):
            return m.layers, m.norm
    # last resort: search for a ModuleList of decoder layers
    for mod in model.modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
            return mod, None
    return None, None


@torch.no_grad()
def fwd(model, proc, im, prompt, device, want_hidden=False):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt").to(device)
    out = model(**inp, output_hidden_states=want_hidden)
    return out, int(inp["input_ids"].shape[1])


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
    p = torch.softmax(logit.float(), -1)
    py, pn = p[yn[0]].sum().item(), p[yn[1]].sum().item()
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
    ap.add_argument("--arch", required=True, choices=["qwen3", "internvl"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["pope", "vstar"])
    ap.add_argument("--pope", default=HARAM_ROOT + "/coco_build/data/pope_test_adversarial.json")
    ap.add_argument("--image-dir", default=HARAM_ROOT + "/coco_build/images")
    ap.add_argument("--vstar-root", default="")
    ap.add_argument("--low", type=int, default=256); ap.add_argument("--high", type=int, default=1024)
    ap.add_argument("--low-tiles", type=int, default=1); ap.add_argument("--high-tiles", type=int, default=12)
    ap.add_argument("--per-class", type=int, default=300); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    lvls = {"low": args.low, "high": args.high}
    tiles = {"low": args.low_tiles, "high": args.high_tiles}

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    lm_head = model.get_output_embeddings()
    layers, norm = get_backbone(model)
    nlayers = len(layers) if layers is not None else 32
    cand = list(range(2, nlayers // 2, 2))
    print(f"[{args.arch} 2x2 {args.task}] layers={nlayers} dola_cand={cand} "
          f"norm={'ok' if norm is not None else 'MISSING'}", flush=True)

    def encode(im, level):
        if args.arch == "internvl": set_tiles(proc, tiles[level])
        return prep_image(args.arch, im, level, lvls)

    t0 = time.time()
    if args.task == "pope":
        yn = yes_no_ids(proc.tokenizer)
        rows = [json.loads(l) for l in open(args.pope) if l.strip()]
        items = [r for r in rows if r["label"] == "yes"][: args.per_class] + \
                [r for r in rows if r["label"] == "no"][: args.per_class]
        if args.limit: items = items[: args.limit] + items[-args.limit:]
        C = {k: [] for k in ["low", "high", "low_dola", "high_dola"]}; lab = []
        for i, r in enumerate(items):
            im = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
            o_lo, _ = fwd(model, proc, encode(im, "low"), r["text"], device, want_hidden=True)
            o_hi, _ = fwd(model, proc, encode(im, "high"), r["text"], device, want_hidden=True)
            C["low"].append(ynprob(o_lo.logits[0, -1], yn) >= 0.5)
            C["high"].append(ynprob(o_hi.logits[0, -1], yn) >= 0.5)
            C["low_dola"].append(ynprob(dola_logit(o_lo, lm_head, norm, cand), yn) >= 0.5)
            C["high_dola"].append(ynprob(dola_logit(o_hi, lm_head, norm, cand), yn) >= 0.5)
            lab.append(r["label"] == "yes")
            if (i + 1) % 50 == 0: print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        lab = np.array(lab)
        out = {"arch": args.arch, "task": "pope", "n": len(items),
               "results": {k: prf(np.array(v), lab) for k, v in C.items()}}
        print("\n=== prior-driven (POPE-adv) ===")
        for k in ["low", "high", "low_dola", "high_dola"]:
            m = out["results"][k]; print(f"  {k:10} halluc {m['halluc']:.3f}  F1 {m['f1']:.3f}")
    else:
        lmap = letter_ids(proc.tokenizer)
        import random; rng = random.Random(0)
        items = load_vstar(args.vstar_root)
        if args.limit: items = items[: args.limit]
        C = {k: [] for k in ["low", "high", "low_dola", "high_dola"]}
        for i, it in enumerate(items):
            opts = list(it["options"]); correct = opts[0]; rng.shuffle(opts); gt = opts.index(correct)
            prompt = it["question"] + "\n" + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts)) + \
                     "\nAnswer with the letter only."
            im = Image.open(it["image"]).convert("RGB")
            o_lo, _ = fwd(model, proc, encode(im, "low"), prompt, device, want_hidden=True)
            o_hi, _ = fwd(model, proc, encode(im, "high"), prompt, device, want_hidden=True)
            C["low"].append(mcq(o_lo.logits[0, -1], lmap, len(opts)) == gt)
            C["high"].append(mcq(o_hi.logits[0, -1], lmap, len(opts)) == gt)
            C["low_dola"].append(mcq(dola_logit(o_lo, lm_head, norm, cand), lmap, len(opts)) == gt)
            C["high_dola"].append(mcq(dola_logit(o_hi, lm_head, norm, cand), lmap, len(opts)) == gt)
            if (i + 1) % 40 == 0: print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        out = {"arch": args.arch, "task": "vstar", "n": len(items),
               "acc": {k: float(np.mean(v)) for k, v in C.items()}}
        print("\n=== perception-bound (V*Bench) ===")
        for k in ["low", "high", "low_dola", "high_dola"]:
            print(f"  {k:10} acc {out['acc'][k]:.3f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
