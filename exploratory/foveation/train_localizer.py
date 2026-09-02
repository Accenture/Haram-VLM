#!/usr/bin/env python
"""
Foveation Phase 1b.3: train the localizer head on frozen survey features; eval on V*Bench (Gate A).

Loads dumped features (vis grid + query emb + box mask) from /dev/shm, trains the cross-attention
head (base frozen) with per-cell BCE (pos-weighted), and reports localization on held-out COCO-val
and V*Bench each epoch:
  cell-hit    : the top-1 predicted cell overlaps the GT box (right cell at 16x16 granularity).
  contain-hit : the GT-box center falls inside the predicted (expanded) foveation region.
Gate A target: V*Bench contain-hit clearly above the training-free 2-step's 0.68. CPU-light;
needs 1 GPU for the head (features are precomputed).
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from localizer import Localizer, topk_region

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


FEAT = "/dev/shm/haram_localizer_feats"
GRID = 16


def load(split):
    d = np.load(os.path.join(FEAT, f"feat_{split}.npz"))
    return (torch.tensor(d["vis"], dtype=torch.float32), torch.tensor(d["qemb"], dtype=torch.float32),
            torch.tensor(d["mask"], dtype=torch.float32), torch.tensor(d["gtbox"], dtype=torch.float32))


def cell_overlaps_gt(cell_idx, gt, grid=GRID):
    r, c = cell_idx // grid, cell_idx % grid
    cx0, cx1, cy0, cy1 = c / grid, (c + 1) / grid, r / grid, (r + 1) / grid
    return not (cx1 <= gt[0] or cx0 >= gt[2] or cy1 <= gt[1] or cy0 >= gt[3])


@torch.no_grad()
def evaluate(model, vis, q, gt, device, bs=512):
    model.eval(); n = len(vis); cell = 0; contain = 0
    for i in range(0, n, bs):
        logits = model(vis[i:i+bs].to(device), q[i:i+bs].to(device)).cpu()
        for j in range(len(logits)):
            g = gt[i + j].tolist()
            top1 = int(torch.argmax(logits[j]))
            cell += cell_overlaps_gt(top1, g)
            x0, y0, x1, y1 = topk_region(logits[j], grid=GRID, k=3)
            gcx, gcy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
            contain += (x0 <= gcx <= x1 and y0 <= gcy <= y1)
    return cell / n, contain / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3); ap.add_argument("--pos-weight", type=float, default=40.0)
    ap.add_argument("--out", default=WORK_DIR + "/localizer.pt")
    args = ap.parse_args()
    device = "cuda"
    vtr, qtr, mtr, _ = load("train"); vva, qva, _, gva = load("val"); vvs, qvs, _, gvs = load("vstar")
    d_in = vtr.shape[-1]
    print(f"train {len(vtr)} | val {len(vva)} | vstar {len(vvs)} | d_in={d_in}", flush=True)
    model = Localizer(d_in, grid=GRID, h=args.h, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    n = len(vtr); best = -1; hist = []
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i+args.bs]
            logits = model(vtr[idx].to(device), qtr[idx].to(device))
            loss = lossf(logits, mtr[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        vc, vco = evaluate(model, vva, qva, gva, device)
        sc, sco = evaluate(model, vvs, qvs, gvs, device)
        hist.append({"epoch": ep, "loss": tot / n, "val_cell": vc, "val_contain": vco,
                     "vstar_cell": sc, "vstar_contain": sco})
        print(f"ep{ep:02d} loss={tot/n:.4f} | val cell={vc:.3f} contain={vco:.3f} "
              f"| VSTAR cell={sc:.3f} contain={sco:.3f}", flush=True)
        if sco > best:
            best = sco; torch.save({"state": model.state_dict(), "args": vars(args), "d_in": d_in}, args.out)
    print(f"\nBEST V*Bench contain-hit = {best:.3f}  (training-free 2-step = 0.68; oracle = 1.0)")
    json.dump(hist, open(args.out.replace(".pt", "_hist.json"), "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
