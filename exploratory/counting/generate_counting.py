#!/usr/bin/env python
"""
Generate a COUNTING benchmark (beyond yes/no) from COCO annotations on the HELD-OUT
test images (disjoint from training). Question: "How many <category> are in the image?"
with ground-truth integer counts from instance annotations. Counts capped at 1..6
(single digit, tractable, resolution-sensitive). Self-contained on the share.
"""
import argparse, json, os, random, shutil
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)            # instances_val2014.json
    ap.add_argument("--test-pope", required=True)      # any pope_test_*.json (defines held-out images)
    ap.add_argument("--src-images", required=True)     # where the images currently live (local)
    ap.add_argument("--out-dir", required=True)        # cloudfiles benchmark dir
    ap.add_argument("--max-count", type=int, default=6)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)

    test_imgs = set()
    for l in open(args.test_pope):
        if l.strip():
            test_imgs.add(json.loads(l)["image"])
    print(f"held-out test images: {len(test_imgs)}")

    d = json.load(open(args.ann))
    catid2name = {c["id"]: c["name"] for c in d["categories"]}
    name2file = {im["file_name"]: im["id"] for im in d["images"]}
    # per held-out image: category -> instance count
    counts = defaultdict(lambda: defaultdict(int))
    test_ids = {name2file[f]: f for f in test_imgs if f in name2file}
    for a in d["annotations"]:
        if a["image_id"] in test_ids:
            counts[a["image_id"]][catid2name[a["category_id"]]] += 1

    # collect (file, category, count) with count in 1..max_count, group by count for balance
    by_count = defaultdict(list)
    for iid, fn in test_ids.items():
        for cat, c in counts[iid].items():
            if 1 <= c <= args.max_count:
                by_count[c].append((fn, cat, c))
    print("available per count:", {k: len(v) for k, v in sorted(by_count.items())})

    # sample ~n, spread across counts (oversample higher counts where possible)
    per = max(1, args.n // args.max_count)
    sel = []
    for c in range(1, args.max_count + 1):
        items = by_count.get(c, [])
        rng.shuffle(items)
        sel.extend(items[:per])
    rng.shuffle(sel)
    sel = sel[: args.n]

    rows = []
    needed_imgs = set()
    for i, (fn, cat, c) in enumerate(sel):
        rows.append({"question_id": i + 1, "image": fn,
                     "text": f"How many {cat} are in the image? Answer with a single number.",
                     "count": c})
        needed_imgs.add(fn)
    with open(os.path.join(args.out_dir, "counting_test.json"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # copy the referenced images onto the share (self-contained benchmark)
    copied = 0
    for fn in needed_imgs:
        dst = os.path.join(args.out_dir, "images", fn)
        if not os.path.exists(dst):
            src = os.path.join(args.src_images, fn)
            if os.path.exists(src):
                shutil.copyfile(src, dst); copied += 1   # copyfile: no chmod (CIFS-safe)
    from collections import Counter
    print(f"wrote {len(rows)} counting questions over {len(needed_imgs)} images")
    print("count distribution:", dict(sorted(Counter(r['count'] for r in rows).items())))
    print(f"copied {copied} images to {args.out_dir}/images")
    print(f"-> {args.out_dir}/counting_test.json")


if __name__ == "__main__":
    main()
