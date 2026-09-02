#!/usr/bin/env python
"""
Does a LEARNED risk head fix V*Bench's confident-mistake boundary case?

Loads the V*Bench scout features (vstar_risk_features.npz), trains a probe (PCA + logistic)
to predict scout-wrong via k-fold CROSS-VALIDATION (V*Bench is only ~170 items, so the probe
must be scored out-of-fold), and compares escalation rankings -- confidence (training-free),
learned (CV out-of-fold), oracle -- by AUROC and by the Table-3 metric (token saving to reach
within 0.5pt of always-high accuracy). Prints an 'Adaptive (learned)' row for Table 3.
"""
import argparse, json
import numpy as np
import torch
import os

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


FEAT = WORK_DIR + "/vstar_risk_features.npz"


def auroc(score, y):
    y = y.astype(bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    order = np.argsort(score); ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pca_fit(X, k):
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:min(k, Vt.shape[0])]


def logistic(Xtr, ytr, Xva, epochs=300, wd=3e-3, lr=5e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    lin = torch.nn.Linear(Xn.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    lf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(); lf(lin(Xn).squeeze(-1), y).backward(); opt.step()
    with torch.no_grad():
        Xvn = torch.tensor((Xva - mu) / sd, dtype=torch.float32)
        return torch.sigmoid(lin(Xvn).squeeze(-1)).numpy()


def cv_oof(emb, conf, y, k=32, folds=5, seeds=(0, 1, 2)):
    """Out-of-fold learned-risk scores (PCA(emb)+conf -> logistic), averaged over seeds."""
    n = len(y); oof = np.zeros(n)
    for s in seeds:
        idx = np.random.RandomState(s).permutation(n)
        for f in range(folds):
            va = idx[f::folds]; tr = np.setdiff1d(idx, va)
            pmu, pc = pca_fit(emb[tr], k)
            Xtr = np.concatenate([(emb[tr] - pmu) @ pc.T, conf[tr, None]], 1)
            Xva = np.concatenate([(emb[va] - pmu) @ pc.T, conf[va, None]], 1)
            oof[va] += logistic(Xtr, y[tr].astype(np.float32), Xva)
    return oof / len(seeds)


def escalate_pareto(risk, pred_low, pred_high, gt, tok_low, tok_high):
    order = np.argsort(-risk, kind="stable"); n = len(gt)
    tl, th = tok_low.astype(float), tok_high.astype(float)
    pts = []
    for f in np.linspace(0, 1, 101):
        kk = int(round(f * n)); esc = np.zeros(n, bool); esc[order[:kk]] = True
        pred = np.where(esc, pred_high, pred_low)
        pts.append((float((tl + np.where(esc, th, 0)).mean()), float((pred == gt).mean()), float(esc.mean())))
    return pts


def knee(pts, target, th_mean, tol=0.005):
    ok = [p for p in pts if p[1] >= target - tol]
    if not ok: return None
    best = min(ok, key=lambda p: p[0])
    return {"acc": best[1], "tokens": best[0], "esc": best[2], "save_pct": (1 - best[0] / th_mean) * 100}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feat", default=FEAT)
    ap.add_argument("--pca", type=int, default=32); ap.add_argument("--output", default=WORK_DIR + "/vstar_risk_report.json")
    args = ap.parse_args()
    d = np.load(args.feat)
    emb = d["emb"].astype(np.float32); conf = d["conf_low"].astype(np.float32)
    pl, ph, gt = d["pred_low"], d["pred_high"], d["gt"]
    tl, th = d["tok_low"], d["tok_high"]
    y = (pl != gt).astype(int)
    n = len(y)
    low_acc = float((pl == gt).mean()); high_acc = float((ph == gt).mean())
    print(f"V*Bench risk: n={n}, scout-wrong={y.mean():.3f} | always-low acc={low_acc:.3f} "
          f"always-high acc={high_acc:.3f} (tok {tl.mean():.0f}->{th.mean():.0f})")

    learned = cv_oof(emb, conf, y, k=args.pca)
    rng = np.random.RandomState(0)
    signals = {"confidence": 1 - conf, "learned": learned, "oracle": y + 1e-3 * rng.rand(n)}
    th_mean = th.mean()
    report = {"n": int(n), "always_low_acc": low_acc, "always_high_acc": high_acc, "signals": {}}
    print(f"\n{'signal':12} {'AUROC':>6} {'Adaptive acc':>12} {'token save':>11} {'esc%':>6}")
    for name, r in signals.items():
        a = auroc(r, y)
        kn = knee(escalate_pareto(r, pl, ph, gt, tl, th), high_acc, th_mean)
        report["signals"][name] = {"auroc": float(a), "knee": kn}
        if kn:
            print(f"{name:12} {a:6.3f} {kn['acc']:12.3f} {kn['save_pct']:10.0f}% {kn['esc']*100:5.0f}%")
        else:
            print(f"{name:12} {a:6.3f}   (cannot reach always-high)")
    json.dump(report, open(args.output, "w"), indent=2)
    print(f"\nwrote {args.output}")
    s = report["signals"]
    print("\n=== Table-3 'Adaptive' rows (token saving to match always-high acc) ===")
    for nm in ["confidence", "learned", "oracle"]:
        kn = s[nm]["knee"]
        print(f"  Adaptive ({nm:10}): acc {kn['acc']:.3f}  saving {kn['save_pct']:.0f}%  (AUROC {s[nm]['auroc']:.3f})"
              if kn else f"  Adaptive ({nm}): cannot reach always-high")


if __name__ == "__main__":
    main()
