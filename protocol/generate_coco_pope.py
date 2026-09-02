#!/usr/bin/env python3
"""
Generate a large, CLEAN (disjoint train/test) POPE-style yes/no dataset from
COCO val2014 object annotations.

- Train set: balanced 3 positive + 3 negative questions per image, with negatives
  MIXED across random / popular / adversarial sampling so the model sees hard
  negatives during training. Emitted in HARAM training format (with metadata).
- Test set: three separate POPE-style files (random / popular / adversarial),
  each balanced 3 pos + 3 neg per image, in the original POPE line-JSON format
  ({question_id, image, text, label}) so eval_pope.py consumes them directly.

Train and test images are DISJOINT, so the test set is a true held-out benchmark.
Only the selected images are downloaded from cocodataset.org.
"""
import argparse, json, os, random, sys, urllib.request
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

RESOLUTIONS = [224, 336, 448, 672, 896]
RISK_MAP = {224: 0.267, 336: 0.20, 448: 0.133, 672: 0.033, 896: 0.02}
COCO_URL = "http://images.cocodataset.org/val2014/{}"


def load_annotations(path):
    d = json.load(open(path))
    catid2name = {c["id"]: c["name"] for c in d["categories"]}
    all_cats = sorted(catid2name.values())
    img2name = {im["id"]: im["file_name"] for im in d["images"]}
    img2cats = defaultdict(set)
    for a in d["annotations"]:
        img2cats[a["image_id"]].add(catid2name[a["category_id"]])
    return all_cats, img2name, {i: c for i, c in img2cats.items()}


def build_stats(img2cats, all_cats):
    freq = Counter()                       # images containing category c
    cooc = defaultdict(Counter)            # cooc[a][b] = images containing both
    for cats in img2cats.values():
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
    if strategy == "adversarial":
        score = {a: sum(cooc[p][a] for p in present) for a in absent}
        ranked = sorted(absent, key=lambda c: (-score[c], -freq[c]))
        return ranked[:k]
    raise ValueError(strategy)


def questions_for_image(fname, present, all_cats, freq, cooc, k, strategy, rng, qid_start):
    pos = rng.sample(list(present), k)
    neg = pick_negatives(present, all_cats, freq, cooc, k, strategy, rng)
    rows, qid = [], qid_start
    for obj, label in [(o, "yes") for o in pos] + [(o, "no") for o in neg]:
        rows.append({"question_id": qid, "image": fname,
                     "text": f"Is there a {obj} in the image?", "label": label})
        qid += 1
    return rows


def to_haram(rows, rng):
    """POPE-format rows -> HARAM training samples (with metadata)."""
    out = []
    for r in rows:
        res = rng.choice(RESOLUTIONS)
        out.append({
            "id": f"cocoL_{r['question_id']}_{res}",
            "image": r["image"],
            "conversations": [
                {"from": "human", "value": f"<image>\n{r['text']}"},
                {"from": "gpt", "value": r["label"]},
            ],
            "metadata": {"resolution": res, "query_type": "simple_yes_no",
                         "expected_hallucination_risk": RISK_MAP[res], "source": "coco_generated"},
        })
    return out


def download_images(fnames, out_dir, workers=32):
    os.makedirs(out_dir, exist_ok=True)
    def fetch(fn):
        dst = os.path.join(out_dir, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return fn, True
        for _ in range(3):
            try:
                urllib.request.urlretrieve(COCO_URL.format(fn), dst)
                if os.path.getsize(dst) > 0:
                    return fn, True
            except Exception:
                continue
        return fn, False
    ok = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, fn): fn for fn in fnames}
        for i, f in enumerate(as_completed(futs)):
            fn, good = f.result()
            if good:
                ok.add(fn)
            if (i + 1) % 500 == 0:
                print(f"  downloaded {i+1}/{len(fnames)} ({len(ok)} ok)", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--n-train-images", type=int, default=6000)
    ap.add_argument("--n-test-images", type=int, default=1000)
    ap.add_argument("--k", type=int, default=3, help="pos (= neg) questions per image")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("loading annotations...", flush=True)
    all_cats, img2name, img2cats = load_annotations(args.ann)
    freq, cooc = build_stats(img2cats, all_cats)

    # eligible: enough present categories for k positives, and room for k negatives
    eligible = [i for i, cats in img2cats.items() if len(cats) >= args.k and len(cats) <= len(all_cats) - args.k]
    rng.shuffle(eligible)
    need = args.n_train_images + args.n_test_images
    if len(eligible) < need:
        sys.exit(f"only {len(eligible)} eligible images, need {need}")
    train_ids = eligible[: args.n_train_images]
    test_ids = eligible[args.n_train_images: need]
    assert not (set(train_ids) & set(test_ids)), "train/test image overlap!"
    print(f"eligible={len(eligible)}  train_imgs={len(train_ids)}  test_imgs={len(test_ids)}", flush=True)

    # download just the selected images
    sel_names = [img2name[i] for i in train_ids + test_ids]
    print(f"downloading {len(sel_names)} images -> {args.image_dir} ...", flush=True)
    ok = download_images(sel_names, args.image_dir)
    print(f"  {len(ok)}/{len(sel_names)} images downloaded ok", flush=True)

    # TRAIN: mixed negatives (cycle strategies across the k negatives)
    strategies = ["random", "popular", "adversarial"]
    train_rows, qid = [], 1
    for i in train_ids:
        fn = img2name[i]
        if fn not in ok:
            continue
        present = img2cats[i]
        pos = rng.sample(list(present), args.k)
        negs = []
        for j in range(args.k):
            s = strategies[j % len(strategies)]
            cand = pick_negatives(present, all_cats, freq, cooc, args.k, s, rng)
            cand = [c for c in cand if c not in negs]
            negs.append(cand[0] if cand else pick_negatives(present, all_cats, freq, cooc, 1, "random", rng)[0])
        for obj, label in [(o, "yes") for o in pos] + [(o, "no") for o in negs]:
            train_rows.append({"question_id": qid, "image": fn,
                               "text": f"Is there a {obj} in the image?", "label": label})
            qid += 1
    train_haram = to_haram(train_rows, rng)
    rng.shuffle(train_haram)
    json.dump(train_haram, open(os.path.join(args.out_dir, "haram_train_cocoLarge.json"), "w"), indent=1)

    # TEST: three POPE-style files, disjoint images
    qid = 1
    test_counts = {}
    for strat in strategies:
        rows = []
        for i in test_ids:
            fn = img2name[i]
            if fn not in ok:
                continue
            rows.extend(questions_for_image(fn, img2cats[i], all_cats, freq, cooc, args.k, strat, rng, qid))
            qid += 2 * args.k
        with open(os.path.join(args.out_dir, f"pope_test_{strat}.json"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        test_counts[strat] = len(rows)

    print("\n=== SUMMARY ===")
    print(f"train: {len(train_haram)} questions over {len(set(r['image'] for r in train_rows))} images")
    yn = Counter(r["label"] for r in train_rows)
    print(f"  train label balance: {dict(yn)}")
    for s, n in test_counts.items():
        print(f"test[{s}]: {n} questions")
    # explicit disjointness check at filename level
    tr_imgs = set(r["image"] for r in train_rows)
    te_imgs = set(img2name[i] for i in test_ids) & ok
    print(f"train/test image overlap: {len(tr_imgs & te_imgs)} (must be 0)")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
