#!/usr/bin/env python
"""
Foveation Phase 1b.2: dump frozen survey features for the localizer.

For each item (a localization triple, or a V*Bench MCQ), square-resize the image to a fixed
16x16-cell survey, run the FROZEN Qwen3-VL and extract:
  - vis  : the query-independent visual grid (N=256, d) from the embedding layer,
  - qemb : mean-pooled query token embedding (d),
  - mask : per-cell box-coverage target (N,)  [triples mode], and
  - gtbox: GT box in fractional xyxy (4,)      [for center-hit eval].
Saved as an .npz on fast local disk (regenerable). Run in 'qwen3' env.
"""
import argparse, glob, json, os, random, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

GRID = 16            # 16x16 survey cells (512x512 px square survey)
SURVEY_PX = GRID * 32  # Qwen3-VL merged token ~ 32px -> 512px


def box_mask_frac(xyxy_frac, grid=GRID):
    """Fractional (x0,y0,x1,y1) -> (grid*grid,) row-major cell-coverage mask."""
    x0, y0, x1, y1 = xyxy_frac
    m = np.zeros((grid, grid), np.uint8)
    c0, c1 = int(np.floor(x0 * grid)), int(np.ceil(x1 * grid))
    r0, r1 = int(np.floor(y0 * grid)), int(np.ceil(y1 * grid))
    m[max(0, r0):min(grid, r1), max(0, c0):min(grid, c1)] = 1
    if m.sum() == 0:  # tiny box -> mark its center cell
        m[min(grid - 1, int((y0 + y1) / 2 * grid)), min(grid - 1, int((x0 + x1) / 2 * grid))] = 1
    return m.reshape(-1)


def load_items(args):
    if args.mode == "triples":
        rows = [json.loads(l) for l in open(args.input) if l.strip()]
        if args.limit: rows = rows[: args.limit]
        out = []
        for r in rows:
            x, y, w, h = r["target_xywh"]; W, H = r["W"], r["H"]
            out.append({"path": os.path.join(args.image_dir, r["image"]), "query": r["query"],
                        "gt_frac": (x / W, y / H, (x + w) / W, (y + h) / H)})
        return out
    else:  # vstar
        items = []
        for sub in ["direct_attributes", "relative_position"]:
            for jf in sorted(glob.glob(os.path.join(args.vstar_root, sub, "*.json"))):
                j = json.load(open(jf)); img = None
                for ext in (".jpg", ".JPG", ".png", ".jpeg", ".webp", ".JPEG"):
                    if os.path.exists(jf[:-5] + ext):
                        img = jf[:-5] + ext; break
                if img and j.get("options") and j.get("bbox"):
                    bs = j["bbox"] if isinstance(j["bbox"][0], (list, tuple)) else [j["bbox"]]
                    x0 = min(b[0] for b in bs); y0 = min(b[1] for b in bs)
                    x1 = max(b[0] + b[2] for b in bs); y1 = max(b[1] + b[3] for b in bs)
                    W, H = Image.open(img).size
                    items.append({"path": img, "query": j["question"],
                                  "gt_frac": (x0 / W, y0 / H, x1 / W, y1 / H)})
        return items[: args.limit] if args.limit else items


@torch.no_grad()
def visual_grid(model, proc, img_square, device, img_token_id):
    """Query-independent visual grid (N,d) from the embedding layer + the token grid (h,w)."""
    msgs = [{"role": "user", "content": [{"type": "image", "image": img_square}, {"type": "text", "text": "."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img_square], return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True)
    vmask = (inputs["input_ids"][0] == img_token_id)
    vis = out.hidden_states[0][0, vmask].float().cpu().numpy()    # (N,d) pre-LLM = query-independent
    g = inputs["image_grid_thw"][0]; m = getattr(proc.image_processor, "merge_size", 2)
    return vis, (int(g[1] // m), int(g[2] // m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--mode", choices=["triples", "vstar"], required=True)
    ap.add_argument("--input", default=""); ap.add_argument("--image-dir", default="")
    ap.add_argument("--vstar-root", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    proc.image_processor.min_pixels = (GRID * GRID) * 1024 - 1   # ~256 tokens, square
    proc.image_processor.max_pixels = (GRID * GRID) * 1024 + 1
    embw = model.get_input_embeddings()
    img_token_id = getattr(model.config, "image_token_id", None) or proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    items = load_items(args)
    print(f"[dump {args.mode}] {len(items)} items -> {args.output}", flush=True)
    VIS, QE, MASK, GTB = [], [], [], []
    t0 = time.time(); skipped = 0
    for i, it in enumerate(items):
        try:
            im = Image.open(it["path"]).convert("RGB").resize((SURVEY_PX, SURVEY_PX), Image.LANCZOS)
        except Exception:
            skipped += 1; continue
        vis, (ht, wt) = visual_grid(model, proc, im, device, img_token_id)
        if ht * wt != GRID * GRID:   # smart_resize gave an off grid -> skip (rare)
            skipped += 1; continue
        qids = proc.tokenizer(it["query"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            qe = embw(qids)[0].mean(0).detach().float().cpu().numpy()
        VIS.append(vis.astype(np.float16)); QE.append(qe.astype(np.float16))
        MASK.append(box_mask_frac(it["gt_frac"])); GTB.append(np.array(it["gt_frac"], np.float32))
        if (i + 1) % 500 == 0: print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, vis=np.array(VIS, np.float16), qemb=np.array(QE, np.float16),
             mask=np.array(MASK, np.uint8), gtbox=np.array(GTB, np.float32))
    print(f"  saved {len(VIS)} (skipped {skipped}) | vis{np.array(VIS).shape} -> {args.output} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
