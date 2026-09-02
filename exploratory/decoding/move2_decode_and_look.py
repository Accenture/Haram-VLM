#!/usr/bin/env python
"""
Move 2: "Decode-and-Look" -- a unified policy that operationalizes the two-failure-mode diagnosis.

Policy: apply DoLa ALWAYS (free, same forward pass) and escalate RESOLUTION only when a cheap
perception-risk signal (low-res confidence) fires. Evaluated on a MIXED stream (POPE-adv U V*Bench)
on a common metric (per-item accuracy) vs cost (avg visual tokens), via the paper's offline tau-sweep.

For each item we compute, at low and high resolution, the greedy and the DoLa prediction (+ low-res
greedy confidence as the escalation signal), then sweep policies offline:
  fixed:   low, low+DoLa, high, high+DoLa
  adaptive (resolution only):  scout low greedy; if conf<tau escalate to high greedy
  adaptive (Decode-and-Look):  scout low+DoLa;  if conf<tau escalate to high+DoLa
  oracle bounds: per-item cheapest-correct ; mode-oracle (diagonal cure by known subset)
Cost model (honest): scout always runs (tok_low); escalation adds tok_high. DoLa is free (same pass).
Run in 'qwen3' env. arch in {qwen3, internvl}; reuses the resolution lever / backbone detection of
arch_2x2.py.
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


def set_tiles(proc, max_tiles):
    ip = getattr(proc, "image_processor", None)
    if ip is None: return
    for attr, val in [("crop_to_patches", max_tiles > 1), ("max_patches", int(max_tiles)), ("min_patches", 1)]:
        if hasattr(ip, attr): setattr(ip, attr, val)


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


def yn_decode(logit, yn):
    p = torch.softmax(logit.float(), -1); py, pn = p[yn[0]].sum().item(), p[yn[1]].sum().item()
    pred = "yes" if py >= pn else "no"; conf = max(py, pn) / (py + pn + 1e-9)
    return pred, conf


def mcq_decode(logit, lmap, n_opt):
    p = torch.softmax(logit.float(), -1); pr = {}
    for tid, idx in lmap.items():
        if idx < n_opt: pr[idx] = pr.get(idx, 0.0) + p[tid].item()
    tot = sum(pr.values()) + 1e-9; k = max(pr, key=pr.get) if pr else 0
    return k, pr.get(k, 0.0) / tot


# ---------------- offline policy sweep ----------------
def acc_cost(records, choose):
    """choose(r) -> (correct: bool, cost: float). Returns overall/pope/vstar acc + avg cost."""
    cor, cost = [], []
    sub = {"pope": [], "vstar": []}
    for r in records:
        c, k = choose(r); cor.append(c); cost.append(k); sub[r["task"]].append(c)
    return {"acc": float(np.mean(cor)), "cost": float(np.mean(cost)),
            "acc_pope": float(np.mean(sub["pope"])) if sub["pope"] else None,
            "acc_vstar": float(np.mean(sub["vstar"])) if sub["vstar"] else None}


def analyze(records, tok_low, tok_high):
    fixed = {
        "low":       acc_cost(records, lambda r: (r["c_low"],      tok_low)),
        "low_dola":  acc_cost(records, lambda r: (r["c_low_dola"], tok_low)),
        "high":      acc_cost(records, lambda r: (r["c_high"],      tok_high)),
        "high_dola": acc_cost(records, lambda r: (r["c_high_dola"], tok_high)),
    }
    def adaptive(tau, dola):
        ckey_lo = "c_low_dola" if dola else "c_low"; ckey_hi = "c_high_dola" if dola else "c_high"
        def choose(r):
            esc = r["conf_low"] < tau
            return (r[ckey_hi] if esc else r[ckey_lo], tok_low + (tok_high if esc else 0))
        m = acc_cost(records, choose); m["tau"] = round(float(tau), 3)
        m["esc"] = float(np.mean([r["conf_low"] < tau for r in records])); return m
    taus = np.linspace(0.5, 1.0, 26)
    front_res = [adaptive(t, False) for t in taus]
    front_dnl = [adaptive(t, True) for t in taus]
    # oracle: per-item cheapest correct among {low+DoLa @tok_low, high+DoLa @tok_high}
    orc = acc_cost(records, lambda r: ((True, tok_low) if r["c_low_dola"]
                                       else (r["c_high_dola"], tok_high)))
    # mode-oracle: apply the diagonal cure by known subset (pope->low+DoLa, vstar->high+DoLa)
    mode = acc_cost(records, lambda r: ((r["c_low_dola"], tok_low) if r["task"] == "pope"
                                        else (r["c_high_dola"], tok_high)))
    return {"fixed": fixed, "frontier_resolution": front_res, "frontier_decode_and_look": front_dnl,
            "oracle_cheapest": orc, "mode_oracle": mode}


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


def at(front, max_cost):
    cand = [p for p in front if p["cost"] <= max_cost + 1e-6]
    return max(cand, key=lambda p: p["acc"]) if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="qwen3", choices=["qwen3", "internvl"])
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--pope", default=HARAM_ROOT + "/coco_build/data/pope_test_adversarial.json")
    ap.add_argument("--image-dir", default=HARAM_ROOT + "/coco_build/images")
    ap.add_argument("--vstar-root", required=True)
    ap.add_argument("--low", type=int, default=256); ap.add_argument("--high", type=int, default=1024)
    ap.add_argument("--low-tiles", type=int, default=1); ap.add_argument("--high-tiles", type=int, default=12)
    ap.add_argument("--pope-per-class", type=int, default=150)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    lvls = {"low": args.low, "high": args.high}; tiles = {"low": args.low_tiles, "high": args.high_tiles}
    tok_low = args.low if args.arch == "qwen3" else args.low_tiles * 256
    tok_high = args.high if args.arch == "qwen3" else args.high_tiles * 256

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    lm_head = model.get_output_embeddings(); layers, norm = get_backbone(model)
    nlayers = len(layers) if layers is not None else 32; cand = list(range(2, nlayers // 2, 2))
    yn = yes_no_ids(proc.tokenizer); lmap = letter_ids(proc.tokenizer)
    print(f"[decode-and-look {args.arch}] layers={nlayers} norm={'ok' if norm is not None else 'MISS'} "
          f"tok {tok_low}->{tok_high}", flush=True)

    def encode(im, level):
        if args.arch == "internvl": set_tiles(proc, tiles[level])
        return resize_tokens(im, lvls[level]) if args.arch == "qwen3" else im

    # build mixed stream
    rows = [json.loads(l) for l in open(args.pope) if l.strip()]
    pope = [r for r in rows if r["label"] == "yes"][: args.pope_per_class] + \
           [r for r in rows if r["label"] == "no"][: args.pope_per_class]
    vstar = load_vstar(args.vstar_root)
    rng = random.Random(0)
    print(f"  mixed stream: {len(pope)} POPE-adv + {len(vstar)} V*Bench", flush=True)

    records = []; t0 = time.time()
    for i, r in enumerate(pope):
        im = Image.open(os.path.join(args.image_dir, r["image"])).convert("RGB")
        o_lo, _ = fwd(model, proc, encode(im, "low"), r["text"], device, want_hidden=True)
        o_hi, _ = fwd(model, proc, encode(im, "high"), r["text"], device, want_hidden=True)
        lp, lc = yn_decode(o_lo.logits[0, -1], yn)
        ldp, _ = yn_decode(dola_logit(o_lo, lm_head, norm, cand), yn)
        hp, _ = yn_decode(o_hi.logits[0, -1], yn)
        hdp, _ = yn_decode(dola_logit(o_hi, lm_head, norm, cand), yn)
        gt = r["label"]
        records.append({"task": "pope", "conf_low": lc,
                        "c_low": lp == gt, "c_low_dola": ldp == gt,
                        "c_high": hp == gt, "c_high_dola": hdp == gt})
        if (i + 1) % 50 == 0: print(f"  pope {i+1}/{len(pope)} ({time.time()-t0:.0f}s)", flush=True)
    for i, it in enumerate(vstar):
        opts = list(it["options"]); correct = opts[0]; rng.shuffle(opts); gt = opts.index(correct)
        prompt = it["question"] + "\n" + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts)) + \
                 "\nAnswer with the letter only."
        im = Image.open(it["image"]).convert("RGB")
        o_lo, _ = fwd(model, proc, encode(im, "low"), prompt, device, want_hidden=True)
        o_hi, _ = fwd(model, proc, encode(im, "high"), prompt, device, want_hidden=True)
        lp, lc = mcq_decode(o_lo.logits[0, -1], lmap, len(opts))
        ldp, _ = mcq_decode(dola_logit(o_lo, lm_head, norm, cand), lmap, len(opts))
        hp, _ = mcq_decode(o_hi.logits[0, -1], lmap, len(opts))
        hdp, _ = mcq_decode(dola_logit(o_hi, lm_head, norm, cand), lmap, len(opts))
        records.append({"task": "vstar", "conf_low": lc,
                        "c_low": lp == gt, "c_low_dola": ldp == gt,
                        "c_high": hp == gt, "c_high_dola": hdp == gt})
        if (i + 1) % 40 == 0: print(f"  vstar {i+1}/{len(vstar)} ({time.time()-t0:.0f}s)", flush=True)

    A = analyze(records, tok_low, tok_high)
    out = {"arch": args.arch, "n_pope": len(pope), "n_vstar": len(vstar),
           "tok_low": tok_low, "tok_high": tok_high, **A, "records": records}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)

    f = A["fixed"]
    print("\n=== fixed strategies (mixed stream) ===")
    print(f"{'policy':12} {'acc':>6} {'pope':>6} {'vstar':>6} {'cost':>7}")
    for k in ["low", "low_dola", "high", "high_dola"]:
        m = f[k]; print(f"{k:12} {m['acc']:.3f} {m['acc_pope']:.3f} {m['acc_vstar']:.3f} {m['cost']:7.0f}")
    # Decode-and-Look at the cost of always-high (apples-to-apples), and frontier dominance
    hd = f["high_dola"]; h = f["high"]
    dnl_at_hd = at(A["frontier_decode_and_look"], hd["cost"])
    res_at_hd = at(A["frontier_resolution"], hd["cost"])
    # cheapest DnL point reaching high_dola accuracy
    reach = [p for p in A["frontier_decode_and_look"] if p["acc"] >= hd["acc"] - 0.003]
    knee = min(reach, key=lambda p: p["cost"]) if reach else None
    print("\n=== unified policy ===")
    print(f"always high+DoLa : acc {hd['acc']:.3f}  cost {hd['cost']:.0f}")
    if knee:
        print(f"Decode-and-Look  : acc {knee['acc']:.3f}  cost {knee['cost']:.0f}  "
              f"esc {knee['esc']*100:.0f}%  (-{(1-knee['cost']/hd['cost'])*100:.0f}% tokens at matched acc)")
    if dnl_at_hd and res_at_hd:
        print(f"@cost={hd['cost']:.0f}: Decode-and-Look acc {dnl_at_hd['acc']:.3f} vs "
              f"resolution-only adaptive {res_at_hd['acc']:.3f} (frontier dominance)")
    print(f"oracle cheapest  : acc {A['oracle_cheapest']['acc']:.3f}  cost {A['oracle_cheapest']['cost']:.0f}")
    print(f"mode-oracle      : acc {A['mode_oracle']['acc']:.3f}  cost {A['mode_oracle']['cost']:.0f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
