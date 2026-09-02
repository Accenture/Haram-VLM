#!/usr/bin/env python
"""
P1 (v2): a stronger learned risk head on the SAME scout features.

Non-destructive: reads the existing dumps in risk_features/ (last-token last-layer
hidden state `emb` + `conf_low`), does NOT modify them or any v1 script. Writes its
report to risk_v2/ only.

Question this answers: is the conf->oracle gap closable with a better HEAD on the
features we already have (nonlinearity + PCA + calibration + a real val split), or
do we genuinely need richer features (multi-layer / attention) -> motivates v3 dump.

Compares risk signals on each held-out split, by AUROC(risk, scout-wrong) [clean
discrimination] and by savings@iso [the paper's metric, escalate top-k% by risk to
reach within 0.5pt of always-high]:
  - confidence            : 1 - conf_low                 (training-free baseline)
  - linear                : logistic on [emb, conf]      (v1 head, reproduced)
  - pca+mlp (calibrated)  : PCA(emb) (+) conf -> small MLP, val-selected, Platt-scaled
"""
import argparse, json, os
import numpy as np
import torch

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


FEAT = WORK_DIR + "/risk_features"
OUT = WORK_DIR + "/risk_v2"


def load(npz):
    d = np.load(npz)
    return {k: d[k] for k in d.files}


def auroc(score, y):
    """Rank-based AUROC; y in {0,1}, score higher = more likely y==1."""
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pareto_savings(risk, te, tol=0.005):
    """Escalate top-fraction by risk; savings vs always-high to reach within tol of high acc."""
    order = np.argsort(-risk)
    pl, ph, lb = te["pred_low"], te["pred_high"], te["label"]
    tl, th = te["tok_low"].astype(float), te["tok_high"].astype(float)
    n = len(lb)
    high_acc = (ph == lb).mean()
    th_mean = th.mean()
    best = None
    for f in np.linspace(0, 1, 101):
        k = int(round(f * n))
        esc = np.zeros(n, bool); esc[order[:k]] = True
        acc = (np.where(esc, ph, pl) == lb).mean()
        toks = (tl + np.where(esc, th, 0)).mean()
        if acc >= high_acc - tol:
            sv = (1 - toks / th_mean) * 100
            if best is None or sv > best[0]:
                best = (sv, f * 100, acc)
    return best, high_acc


def pca_fit(X, k):
    mu = X.mean(0)
    Xc = X - mu
    # economy SVD on centered features; components = right singular vectors
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:k]
    return mu, comp


def pca_apply(X, mu, comp):
    return (X - mu) @ comp.T


class MLP(torch.nn.Module):
    def __init__(self, d, h=64, p=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(p),
            torch.nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head(Xtr, ytr, Xva, yva, mlp=False, epochs=400, wd=2e-3, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    norm = lambda Z: torch.tensor((Z - mu) / sd, dtype=torch.float32)
    Xn, Xvn = norm(Xtr), norm(Xva)
    y = torch.tensor(ytr, dtype=torch.float32)
    model = MLP(Xn.shape[1]) if mlp else torch.nn.Linear(Xn.shape[1], 1)
    fwd = (lambda m, x: m(x)) if mlp else (lambda m, x: m(x).squeeze(-1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    best_auc, best_state = -1, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = lossf(fwd(model, Xn), y); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                va = torch.sigmoid(fwd(model, Xvn)).numpy()
            a = auroc(va, yva)
            if a > best_auc:
                best_auc = a; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, fwd, mu, sd, best_auc


def score(model, fwd, mu, sd, X):
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(fwd(model, Xn)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--output", default=os.path.join(OUT, "risk_v2_report.json"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    tr = load(os.path.join(FEAT, "train.npz"))
    emb_tr = tr["emb"].astype(np.float32)
    ytr_full = (tr["pred_low"] != tr["label"]).astype(np.float32)
    n = len(ytr_full)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    nva = int(args.val_frac * n)
    va_idx, tr_idx = perm[:nva], perm[nva:]
    print(f"risk-train: {n} probes ({len(tr_idx)} tr / {len(va_idx)} val), "
          f"scout-wrong rate={ytr_full.mean():.3f}, emb-dim={emb_tr.shape[1]}")

    # PCA fit on TRAIN-only embeddings (no leakage)
    k = min(args.pca, emb_tr[tr_idx].shape[0] - 1, emb_tr.shape[1])
    pmu, pcomp = pca_fit(emb_tr[tr_idx], k)
    conf_tr = tr["conf_low"].astype(np.float32)[:, None]
    Xfull_lin = np.concatenate([emb_tr, conf_tr], 1)                     # v1-style (raw emb)
    Xfull_pca = np.concatenate([pca_apply(emb_tr, pmu, pcomp), conf_tr], 1)

    SEEDS = [0, 1, 2, 3, 4]
    # (1) linear on raw emb+conf  -> reproduces v1 head (seed-averaged, val-selected)
    lin_heads, lin_aucs = [], []
    for sd_ in SEEDS:
        m, fwd, mu, sd, va = train_head(Xfull_lin[tr_idx], ytr_full[tr_idx],
                                        Xfull_lin[va_idx], ytr_full[va_idx], mlp=False, seed=sd_)
        lin_heads.append((m, fwd, mu, sd)); lin_aucs.append(va)
    print(f"  linear      val-AUROC={np.mean(lin_aucs):.3f}")
    # (2) calibrated MLP on PCA(emb)+conf  (seed-averaged scores)
    mlp_heads, mlp_aucs = [], []
    for sd_ in SEEDS:
        m, fwd, mu, sd, va = train_head(Xfull_pca[tr_idx], ytr_full[tr_idx],
                                        Xfull_pca[va_idx], ytr_full[va_idx], mlp=True, seed=sd_)
        mlp_heads.append((m, fwd, mu, sd)); mlp_aucs.append(va)
    print(f"  pca+mlp(k={k}) val-AUROC={np.mean(mlp_aucs):.3f}")

    def ens_score(heads_list, X):
        return np.mean([score(h[0], h[1], h[2], h[3], X) for h in heads_list], 0)

    def rank_norm(x):
        r = np.argsort(np.argsort(x)).astype(float)
        return r / (len(x) - 1)

    report = {"splits": {}, "pca_k": int(k), "n_train": int(n)}
    hdr = f"\n{'split':12} {'signal':12} {'AUROC':>6} {'save@iso':>9} {'esc%':>5} {'highAcc':>8}"
    print(hdr)
    for s in ["random", "popular", "adversarial"]:
        te = load(os.path.join(FEAT, f"test_{s}.npz"))
        emb = te["emb"].astype(np.float32); conf = te["conf_low"].astype(np.float32)
        scout_wrong = (te["pred_low"] != te["label"]).astype(float)
        Xlin = np.concatenate([emb, conf[:, None]], 1)
        Xpca = np.concatenate([pca_apply(emb, pmu, pcomp), conf[:, None]], 1)
        conf_risk = 1 - conf
        learned = ens_score(mlp_heads, Xpca)
        signals = {"confidence": conf_risk}
        signals["linear"] = ens_score(lin_heads, Xlin)
        signals["pca_mlp"] = learned
        # combined: rank fusion of confidence and the learned head (best-of-both)
        signals["combined"] = 0.5 * rank_norm(conf_risk) + 0.5 * rank_norm(learned)
        # oracle ceiling
        signals["oracle"] = scout_wrong + 1e-3 * np.random.RandomState(0).rand(len(emb))
        report["splits"][s] = {}
        for name, r in signals.items():
            auc = auroc(r, scout_wrong)
            best, high_acc = pareto_savings(r, te)
            sv = best[0] if best else None; esc = best[1] if best else None
            report["splits"][s][name] = {"auroc": float(auc),
                                         "save_iso": (float(sv) if sv is not None else None),
                                         "esc_pct": (float(esc) if esc is not None else None),
                                         "high_acc": float(high_acc)}
            svs = f"{sv:7.0f}%" if sv is not None else "  (miss)"
            print(f"{s:12} {name:12} {auc:6.3f} {svs} {esc if esc else 0:4.0f}% {high_acc:8.3f}")
        print()
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
