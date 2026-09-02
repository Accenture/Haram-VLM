#!/usr/bin/env python3
"""
Scale up the HARAM training set by drawing from COCO val2014 + train2014.

Keeps the EXISTING 1,000-image test set fixed (so all prior results stay
comparable): it loads the test image list and excludes those images from the
new training pool, guaranteeing the larger training set is still disjoint from
the held-out test. Only the training JSON and its images are (re)generated;
the test files are untouched.

Negatives are mixed across random / popular / adversarial regimes, as before.
"""
import argparse, json, os, random, sys, urllib.request
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

RESOLUTIONS = [224, 336, 448, 672, 896]
RISK_MAP = {224: 0.267, 336: 0.20, 448: 0.133, 672: 0.033, 896: 0.02}


def url_for(fname):
    sub = "train2014" if "train2014" in fname else "val2014"
    return f"http://images.cocodataset.org/{sub}/{fname}"


def load_annotations_safe(paths):
    name2cats = defaultdict(set)
    all_cats = set()
    for p in paths:
        d = json.load(open(p))
        catid2name = {c["id"]: c["name"] for c in d["categories"]}
        all_cats |= set(catid2name.values())
        id2name = {im["id"]: im["file_name"] for im in d["images"]}
        for a in d["annotations"]:
            name2cats[id2name[a["image_id"]]].add(catid2name[a["category_id"]])
    return sorted(all_cats), name2cats


def build_stats(name2cats):
    freq = Counter()
    cooc = defaultdict(Counter)
    for cats in name2cats.values():
        cl = list(cats)
        for c in cl:
            freq[c] += 1
        for i in range(len(cl)):
            for j in range(len(cl)):
                if i != j:
                    cooc[cl[i]][cl[j]] += 1
    return freq, cooc


def pick_negatives(present, all_cats, freq, cooc, k, strategy, rng):
    absent = [c for c in all_cats if c not in present]
    if strategy == "random":
        return rng.sample(absent, k)
    if strategy == "popular":
        return sorted(absent, key=lambda c: -freq[c])[:k]
    score = {a: sum(cooc[p][a] for p in present) for a in absent}
    return sorted(absent, key=lambda c: (-score[c], -freq[c]))[:k]


def download_images(fnames, out_dir, workers=48):
    os.makedirs(out_dir, exist_ok=True)
    def fetch(fn):
        dst = os.path.join(out_dir, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return fn, True
        for _ in range(3):
            try:
                urllib.request.urlretrieve(url_for(fn), dst)
                if os.path.getsize(dst) > 0:
                    return fn, True
            except Exception:
                continue
        return fn, False
    ok = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch, fn) for fn in fnames]
        for i, f in enumerate(as_completed(futs)):
            fn, good = f.result()
            if good:
                ok.add(fn)
            if (i + 1) % 2000 == 0:
                print(f"  downloaded {i+1}/{len(fnames)} ({len(ok)} ok)", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", nargs="+", required=True, help="one or more instances_*.json")
    ap.add_argument("--n-train-images", type=int, default=36000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out", required=True, help="output training json")
    ap.add_argument("--exclude-test", nargs="+", required=True,
                    help="existing test json(s); their images are excluded from training")
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("loading annotations:", args.ann, flush=True)
    all_cats, name2cats = load_annotations_safe(args.ann)
    freq, cooc = build_stats(name2cats)
    print(f"  {len(name2cats)} images, {len(all_cats)} categories", flush=True)

    test_imgs = set()
    for tf in args.exclude_test:
        for l in open(tf):
            if l.strip():
                test_imgs.add(json.loads(l)["image"])
    print(f"  excluding {len(test_imgs)} held-out test images", flush=True)

    eligible = [n for n, c in name2cats.items()
                if len(c) >= args.k and len(c) <= len(all_cats) - args.k and n not in test_imgs]
    rng.shuffle(eligible)
    if len(eligible) < args.n_train_images:
        print(f"WARNING: only {len(eligible)} eligible, using all", flush=True)
    train_names = eligible[: args.n_train_images]
    assert not (set(train_names) & test_imgs), "train/test overlap!"
    print(f"  eligible={len(eligible)} -> train images={len(train_names)}", flush=True)

    print(f"downloading {len(train_names)} images -> {args.image_dir}", flush=True)
    ok = download_images(train_names, args.image_dir)
    print(f"  {len(ok)}/{len(train_names)} ok", flush=True)

    strategies = ["random", "popular", "adversarial"]
    rows, qid = [], 1
    for fn in train_names:
        if fn not in ok:
            continue
        present = name2cats[fn]
        pos = rng.sample(list(present), args.k)
        negs = []
        for j in range(args.k):
            cand = [c for c in pick_negatives(present, all_cats, freq, cooc, args.k,
                                              strategies[j % 3], rng) if c not in negs]
            negs.append(cand[0] if cand else
                        pick_negatives(present, all_cats, freq, cooc, 1, "random", rng)[0])
        for obj, label in [(o, "yes") for o in pos] + [(o, "no") for o in negs]:
            res = rng.choice(RESOLUTIONS)
            rows.append({
                "id": f"cocoXL_{qid}_{res}", "image": fn,
                "conversations": [
                    {"from": "human", "value": f"<image>\nIs there a {obj} in the image?"},
                    {"from": "gpt", "value": label}],
                "metadata": {"resolution": res, "query_type": "simple_yes_no",
                             "expected_hallucination_risk": RISK_MAP[res], "source": "coco_generated"},
            })
            qid += 1
    rng.shuffle(rows)
    json.dump(rows, open(args.out, "w"))
    yn = Counter(r["conversations"][1]["value"] for r in rows)
    used = len(set(r["image"] for r in rows))
    print(f"\nSUMMARY: {len(rows)} train questions over {used} images, balance={dict(yn)}")
    print(f"  test overlap (must be 0): {len(set(r['image'] for r in rows) & test_imgs)}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
