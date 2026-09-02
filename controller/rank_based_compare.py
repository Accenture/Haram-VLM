#!/usr/bin/env python
"""
Unified, rank-based adaptive-resolution comparison across models and risk signals.

Rank-based escalation (escalate the top-k% most-risky; sweep k) is the consistent
metric. We report token savings to reach within 0.5pt of always-high accuracy for:
  confidence (training-free baseline), learned risk head, combined (rank fusion),
  and an oracle (escalate exactly the scout-wrong queries) as the ceiling.
The confidence and oracle columns for all four architectures reproduce from the saved
adaptive records in results/ with no GPU. The learned and combined columns additionally
need the Qwen3 scout-feature dumps (risk_head/dump_features_qwen3.py); they are skipped
with a note if those are absent.
"""
import argparse, json, os
import numpy as np
import torch

# Paths resolve from env vars so the scripts are portable; see DATA.md.
HARAM_ROOT = os.environ.get("HARAM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK_DIR = os.environ.get("HARAM_WORK", os.path.join(os.getcwd(), "work"))
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


FEAT = WORK_DIR + "/risk_features"
SPLITS = ["random", "popular", "adversarial"]


def savings(risk, pl, ph, lb, tl, th, tol=0.005):
    """Cheapest (max-saving) rank-based operating point within tol of always-high acc."""
    order = np.argsort(-risk, kind="stable")
    high_acc = (ph == lb).mean(); thigh = th.mean(); n = len(lb)
    best = None
    for k in range(0, n + 1, max(1, n // 100)):
        esc = np.zeros(n, bool); esc[order[:k]] = True
        pred = np.where(esc, ph, pl)
        if (pred == lb).mean() >= high_acc - tol:
            sv = (1 - (tl + np.where(esc, th, 0)).mean() / thigh) * 100
            if best is None or sv > best[0]:
                best = (sv, 100 * k / n)
    return best, high_acc


def pct_rank(x):
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(1, len(x) - 1)


def train_head(Xtr, ytr, ep=300, wd=2e-3, lr=5e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); y = torch.tensor(ytr, dtype=torch.float32)
    lin = torch.nn.Linear(Xn.shape[1], 1); opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    lf = torch.nn.BCEWithLogitsLoss()
    for _ in range(ep):
        opt.zero_grad(); lf(lin(Xn).squeeze(-1), y).backward(); opt.step()
    return lambda X: torch.sigmoid(lin(torch.tensor((X - mu) / sd, dtype=torch.float32)).squeeze(-1)).detach().numpy()


def qwen3_rows(feat=None):
    feat = feat or FEAT
    tr = np.load(os.path.join(feat, "train.npz"))
    Xtr = np.concatenate([tr["emb"].astype(np.float32), tr["conf_low"][:, None]], 1)
    head = train_head(Xtr, (tr["pred_low"] != tr["label"]).astype(np.float32))
    out = {}
    for s in SPLITS:
        te = np.load(os.path.join(feat, f"test_{s}.npz"))
        pl, ph, lb = te["pred_low"], te["pred_high"], te["label"]
        tl, th = te["tok_low"].astype(float), te["tok_high"].astype(float)
        conf = 1 - te["conf_low"]
        learned = head(np.concatenate([te["emb"].astype(np.float32), te["conf_low"][:, None]], 1))
        combined = pct_rank(conf) + pct_rank(learned)
        oracle = (pl != lb).astype(float) + 1e-3 * np.random.RandomState(0).rand(len(lb))
        out[s] = {n: savings(r, pl, ph, lb, tl, th)[0]
                  for n, r in [("conf", conf), ("learned", learned), ("combined", combined), ("oracle", oracle)]}
    return out


def record_rows(d):
    out = {}
    for s in SPLITS:
        recs = json.load(open(os.path.join(d, f"adaptive_{s}.json")))["records"]
        pl = np.array([1 if r["pred_low"] == "yes" else 0 for r in recs])
        ph = np.array([1 if r["pred_high"] == "yes" else 0 for r in recs])
        lb = np.array([1 if r["label"] == "yes" else 0 for r in recs])
        tl = np.array([r["tok_low"] for r in recs], float); th = np.array([r["tok_high"] for r in recs], float)
        conf = 1 - np.array([r["conf_low"] for r in recs])
        oracle = (pl != lb).astype(float) + 1e-3 * np.random.RandomState(0).rand(len(lb))
        out[s] = {n: savings(r, pl, ph, lb, tl, th)[0] for n, r in [("conf", conf), ("oracle", oracle)]}
    return out


def fmt(v):
    return f"{v[0]:4.0f}% (esc {v[1]:.0f}%)" if v else "  -- (can't reach)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.environ.get("HARAM_RESULTS", HARAM_ROOT + "/results"),
                    help="directory holding the adaptive_pareto* dirs (default: $HARAM_RESULTS)")
    ap.add_argument("--features", default=FEAT,
                    help="Qwen3 scout-feature dump dir, for the learned/combined columns "
                         "(default: $HARAM_WORK/risk_features)")
    args = ap.parse_args()

    print("Rank-based token savings to reach within 0.5pt of always-high accuracy.\n")

    # (label, subdir, needs_features) -- Table 1's four blocks, in paper order
    BLOCKS = [("HARAM-VLM (Phi-3, fine-tuned)", "adaptive_pareto", False),
              ("Phi-3-Vision, raw",             "adaptive_pareto_base", False),
              ("Qwen3-VL-8B, raw",              "adaptive_pareto_qwen3", True),
              ("InternVL3-8B, raw",             "adaptive_pareto_internvl", False)]

    for name, sub, wants_feats in BLOCKS:
        d = os.path.join(args.results, sub)
        if not os.path.isdir(d):
            print(f"== {name} ==\n  skipped: no {d}\n")
            continue
        rows = record_rows(d)
        if wants_feats:
            if os.path.exists(os.path.join(args.features, "train.npz")):
                for sp, v in qwen3_rows(args.features).items():
                    rows[sp].update(v)
            else:
                print(f"  note: {args.features}/train.npz absent -> learned/combined columns\n"
                      f"        skipped for {name}. Regenerate with "
                      f"risk_head/dump_features_qwen3.py.")
        print(f"== {name} ==")
        for sp in SPLITS:
            line = f"  {sp:12}"
            for sig in ("conf", "learned", "combined", "oracle"):
                if sig in rows[sp]:
                    line += f"  {sig}:{fmt(rows[sp][sig])}"
            print(line)
        print()


if __name__ == "__main__":
    main()
