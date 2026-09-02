#!/usr/bin/env python
"""
P1: learned risk head vs. training-free confidence.

Train a logistic risk head on TRAINING-image scout features to predict whether the
scout is wrong; on the held-out TEST splits, rank queries by predicted risk, escalate
the top fraction, and trace the accuracy-vs-tokens Pareto. Compare three escalation
rankings: confidence (baseline), learned risk (ours), and an oracle (escalate exactly
the scout-wrong queries) as an upper bound.
"""
import argparse, json, os
import numpy as np
import torch

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


FEAT = WORK_DIR + "/risk_features"


def load(npz):
    d = np.load(npz)
    return {k: d[k] for k in d.files}


def train_head(Xtr, ytr, epochs=300, wd=2e-3, lr=5e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    lin = torch.nn.Linear(Xn.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(); out = lin(Xn).squeeze(-1)
        loss = lossf(out, y); loss.backward(); opt.step()
    return lin, mu, sd


def score(lin, mu, sd, X):
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(lin(Xn).squeeze(-1)).numpy()


def pareto(risk, te):
    """Escalate top-fraction by risk; return (avg_tokens, acc) curve."""
    order = np.argsort(-risk)              # highest risk first
    pl, ph, lb = te["pred_low"], te["pred_high"], te["label"]
    tl, th = te["tok_low"].astype(float), te["tok_high"].astype(float)
    n = len(lb); pts = []
    for f in np.linspace(0, 1, 41):
        k = int(round(f * n))
        esc = np.zeros(n, bool); esc[order[:k]] = True
        pred = np.where(esc, ph, pl)
        acc = (pred == lb).mean()
        toks = (tl + np.where(esc, th, 0)).mean()
        pts.append((toks, acc, f))
    return pts


def savings_at(pts, target_acc, tol=0.005, tok_high=None):
    ok = [p for p in pts if p[1] >= target_acc - tol]
    if not ok: return None
    best = min(ok, key=lambda p: p[0])
    return (1 - best[0] / tok_high) * 100, best[2] * 100, best[1]


def main():
    tr = load(os.path.join(FEAT, "train.npz"))
    ytr = (tr["pred_low"] != tr["label"]).astype(np.float32)   # scout wrong
    Xtr = np.concatenate([tr["emb"].astype(np.float32), tr["conf_low"][:, None]], 1)
    print(f"risk-train: {len(ytr)} probes, scout-wrong rate={ytr.mean():.2f}, emb-dim={tr['emb'].shape[1]}")
    lin, mu, sd = train_head(Xtr, ytr)

    print(f"\n{'split':12} {'method':10} {'highAcc':>7} {'save@iso':>9} {'esc%':>5}")
    for s in ["random", "popular", "adversarial"]:
        te = load(os.path.join(FEAT, f"test_{s}.npz"))
        X = np.concatenate([te["emb"].astype(np.float32), te["conf_low"][:, None]], 1)
        learned = score(lin, mu, sd, X)
        conf_risk = 1 - te["conf_low"]
        oracle = (te["pred_low"] != te["label"]).astype(float) + 1e-3 * np.random.RandomState(0).rand(len(X))
        th = te["tok_high"].astype(float).mean()
        high_acc = (te["pred_high"] == te["label"]).mean()
        for name, r in [("confidence", conf_risk), ("learned", learned), ("oracle", oracle)]:
            res = savings_at(pareto(r, te), high_acc, tok_high=th)
            if res is None:
                print(f"{s:12} {name:10} {high_acc:.3f}   (cannot reach)")
            else:
                save, esc, acc = res
                print(f"{s:12} {name:10} {high_acc:.3f} {save:8.0f}% {esc:4.0f}%")
        print()


if __name__ == "__main__":
    main()
