#!/usr/bin/env python
"""
P1 (v3): which RICHER scout features close the conf->oracle gap?

Non-destructive: reads risk_features_v3/ (richer dump: emb, emb_ml multi-layer,
scal logit-distribution, attn visual-attention). Trains the same head recipe as v2
(PCA on hidden states + calibrated MLP, val-selected, seed-averaged) on increasing
feature sets and reports AUROC(risk, scout-wrong) + savings@iso per split. The point
is the ablation: does multi-layer / logit / visual-attention add over the v2 last-layer
embedding, and how close to oracle do we get on the hard splits?
"""
import argparse, json, os
import numpy as np
import torch

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


FEAT = WORK_DIR + "/risk_features_v3"
OUT = WORK_DIR + "/risk_v3"


def load(npz):
    d = np.load(npz)
    return {k: d[k] for k in d.files}


def auroc(score, y):
    y = y.astype(bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    order = np.argsort(score); ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pareto_savings(risk, te, tol=0.005):
    order = np.argsort(-risk)
    pl, ph, lb = te["pred_low"], te["pred_high"], te["label"]
    tl, th = te["tok_low"].astype(float), te["tok_high"].astype(float)
    n = len(lb); high_acc = (ph == lb).mean(); thm = th.mean(); best = None
    for f in np.linspace(0, 1, 101):
        k = int(round(f * n)); esc = np.zeros(n, bool); esc[order[:k]] = True
        acc = (np.where(esc, ph, pl) == lb).mean()
        toks = (tl + np.where(esc, th, 0)).mean()
        if acc >= high_acc - tol:
            sv = (1 - toks / thm) * 100
            if best is None or sv > best[0]: best = (sv, f * 100, acc)
    return best, high_acc


def pca_fit(X, k):
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k]


def pca_apply(X, mu, comp):
    return (X - mu) @ comp.T


class MLP(torch.nn.Module):
    def __init__(self, d, h=64, p=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.GELU(),
                                        torch.nn.Dropout(p), torch.nn.Linear(h, 1))

    def forward(self, x): return self.net(x).squeeze(-1)


def train_head(Xtr, ytr, Xva, yva, epochs=400, wd=2e-3, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    norm = lambda Z: torch.tensor((Z - mu) / sd, dtype=torch.float32)
    Xn, Xvn = norm(Xtr), norm(Xva); y = torch.tensor(ytr, dtype=torch.float32)
    model = MLP(Xn.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss(); best_auc, best_state = -1, None
    for ep in range(epochs):
        model.train(); opt.zero_grad(); loss = lossf(model(Xn), y); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad(): va = torch.sigmoid(model(Xvn)).numpy()
            a = auroc(va, yva)
            if a > best_auc: best_auc = a; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model, mu, sd, best_auc


def score(model, mu, sd, X):
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
    with torch.no_grad(): return torch.sigmoid(model(Xn)).numpy()


# ---- feature assembly: each config returns (train_matrix, test_matrix_fn) ----
def build(dtr, tr_idx, pca_k):
    """Fit PCA bases on TRAIN rows only; return a closure mapping any split-dict -> feature matrix
    for each named config."""
    emb_tr = dtr["emb"].astype(np.float32)
    ml_tr = dtr["emb_ml"].astype(np.float32).reshape(len(emb_tr), -1)     # (N, 4*4096)
    embml_tr = np.concatenate([emb_tr, ml_tr], 1)
    pmu_e, pc_e = pca_fit(emb_tr[tr_idx], min(pca_k, len(tr_idx) - 1))
    pmu_m, pc_m = pca_fit(embml_tr[tr_idx], min(pca_k + 64, len(tr_idx) - 1))

    def feats(d, cfg):
        emb = d["emb"].astype(np.float32)
        ml = d["emb_ml"].astype(np.float32).reshape(len(emb), -1)
        embml = np.concatenate([emb, ml], 1)
        conf = d["conf_low"].astype(np.float32)[:, None]
        scal = d["scal"].astype(np.float32)
        attn = d["attn"].astype(np.float32)
        Pe = pca_apply(emb, pmu_e, pc_e)
        Pm = pca_apply(embml, pmu_m, pc_m)
        if cfg == "emb":        return np.concatenate([Pe, conf], 1)
        if cfg == "multilayer": return np.concatenate([Pm, conf], 1)
        if cfg == "logits":     return np.concatenate([Pe, conf, scal], 1)
        if cfg == "attn":       return np.concatenate([Pe, conf, attn], 1)
        if cfg == "all":        return np.concatenate([Pm, conf, scal, attn], 1)
        raise ValueError(cfg)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", type=int, default=128); ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-limit", type=int, default=0, help="first-N test probes/split (0=all); "
                    "use 2000 to match the standard split of Tables 1-2")
    ap.add_argument("--output", default=os.path.join(OUT, "risk_v3_report.json"))
    args = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    SEEDS = [0, 1, 2, 3, 4]
    CONFIGS = ["emb", "multilayer", "logits", "attn", "all"]

    dtr = load(os.path.join(FEAT, "train.npz"))
    y = (dtr["pred_low"] != dtr["label"]).astype(np.float32); n = len(y)
    rng = np.random.RandomState(0); perm = rng.permutation(n); nva = int(args.val_frac * n)
    va_idx, tr_idx = perm[:nva], perm[nva:]
    print(f"v3 risk-train: {n} probes ({len(tr_idx)}tr/{len(va_idx)}val), scout-wrong={y.mean():.3f}")
    feats = build(dtr, tr_idx, args.pca)

    tests = {s: load(os.path.join(FEAT, f"test_{s}.npz")) for s in ["random", "popular", "adversarial"]}
    if args.test_limit:
        tests = {s: {k: v[:args.test_limit] for k, v in d.items()} for s, d in tests.items()}
        print(f"  (test sliced to first {args.test_limit}/split to match Tables 1-2)")
    swrong = {s: (tests[s]["pred_low"] != tests[s]["label"]).astype(float) for s in tests}

    # train each config (seed-averaged ensemble)
    ens = {}
    for cfg in CONFIGS:
        Xtr = feats(dtr, cfg)
        heads = []
        for sd_ in SEEDS:
            m, mu, sd, va = train_head(Xtr[tr_idx], y[tr_idx], Xtr[va_idx], y[va_idx], seed=sd_)
            heads.append((m, mu, sd))
        ens[cfg] = heads
        print(f"  trained {cfg:11} (val-AUROC seed-avg shown per split below)")

    report = {"splits": {}, "pca_k": args.pca}
    print(f"\n{'split':12} {'signal':12} {'AUROC':>6} {'save@iso':>9} {'esc%':>5} {'oracle':>7}")
    for s in tests:
        te = tests[s]; report["splits"][s] = {}
        conf_risk = 1 - te["conf_low"].astype(np.float32)
        # oracle ceiling savings
        orc = swrong[s] + 1e-3 * np.random.RandomState(0).rand(len(conf_risk))
        (osv, _, _), high = pareto_savings(orc, te)
        signals = {"confidence": conf_risk}
        for cfg in CONFIGS:
            X = feats(te, cfg)
            signals[cfg] = np.mean([score(m, mu, sd, X) for (m, mu, sd) in ens[cfg]], 0)
        for name, r in signals.items():
            a = auroc(r, swrong[s]); best, _ = pareto_savings(r, te)
            sv = best[0] if best else None; esc = best[1] if best else None
            report["splits"][s][name] = {"auroc": float(a),
                "save_iso": (float(sv) if sv is not None else None),
                "esc_pct": (float(esc) if esc is not None else None)}
            svs = f"{sv:7.0f}%" if sv is not None else "  (miss)"
            print(f"{s:12} {name:12} {a:6.3f} {svs} {esc if esc else 0:4.0f}% {osv:6.0f}%")
        report["splits"][s]["oracle_save"] = float(osv); report["splits"][s]["high_acc"] = float(high)
        print()
    with open(args.output, "w") as f: json.dump(report, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
