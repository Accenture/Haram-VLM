#!/usr/bin/env python
"""
Foveation Phase 1b.1: generate fine-grained LOCALIZATION training data from COCO.

Each example is a (image, query, target-box) triple in the V*Bench regime: a query that
references a SMALL, unambiguous (single-instance) object in a multi-object scene, with the
target = that object's box (the region a localizer must point at to answer). Two query types
mirror V*Bench: `attribute` ("what color/shape is the X?") and `relative-position`
("is the X left or right of the Y?"). CPU-only, deterministic (seeded).

The base model is never asked the answer here -- this trains *where to look*, supervised by
the COCO box. Output: jsonl of {image, image_id, query, qtype, target_xywh, W, H, cat}.
"""
import argparse, json, os, random
from collections import defaultdict

ATTR = [
    "What color is the {cat}?",
    "What is the color of the {cat}?",
    "What is the shape of the {cat}?",
    "What material is the {cat} made of?",
    "What kind of {cat} is in the image?",
]
RELPOS = [
    "Is the {a} to the left or right of the {b}?",
    "Is the {a} on the left or right side of the {b}?",
    "On which side of the {b} is the {a}, left or right?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="COCO instances_*2014.json")
    ap.add_argument("--img-prefix", default="COCO_train2014_", help="filename prefix for this split")
    ap.add_argument("--image-dir", default="", help="if set, only emit triples whose image file exists here")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--min-frac", type=float, default=0.0008, help="min box area / image area (skip degenerate)")
    ap.add_argument("--max-frac", type=float, default=0.05, help="max box area / image area (small-object regime)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print(f"loading {args.ann} ...", flush=True)
    d = json.load(open(args.ann))
    catid2name = {c["id"]: c["name"] for c in d["categories"]}
    imgwh = {im["id"]: (im["width"], im["height"], im["file_name"]) for im in d["images"]}
    # image_id -> {catname: [bbox,...]}
    by_img = defaultdict(lambda: defaultdict(list))
    for a in d["annotations"]:
        if a.get("iscrowd", 0):
            continue
        by_img[a["image_id"]][catid2name[a["category_id"]]].append(a["bbox"])

    present = set(os.listdir(args.image_dir)) if args.image_dir else None
    if present is not None:
        print(f"  filtering to {len(present)} images present in {args.image_dir}")
    rng = random.Random(args.seed)
    img_ids = list(by_img.keys()); rng.shuffle(img_ids)
    out = []
    n_attr = n_rel = 0
    for iid in img_ids:
        if iid not in imgwh:
            continue
        W, H, fname = imgwh[iid]
        if present is not None and fname not in present:
            continue
        area = W * H
        by_cat = by_img[iid]
        if len(by_cat) < 2:                      # need ≥2 distinct categories so the query disambiguates
            continue
        # small, single-instance categories = clean unambiguous targets
        singles = [(c, b[0]) for c, b in by_cat.items()
                   if len(b) == 1 and args.min_frac <= (b[0][2] * b[0][3]) / area <= args.max_frac]
        if not singles:
            continue
        cat, box = rng.choice(singles)
        others = [c for c in by_cat if c != cat]
        if rng.random() < 0.5:                   # attribute query
            q = rng.choice(ATTR).format(cat=cat); qtype = "attribute"; n_attr += 1
        else:                                    # relative-position query (target = the small object A)
            b = rng.choice(others)
            q = rng.choice(RELPOS).format(a=cat, b=b); qtype = "relpos"; n_rel += 1
        out.append({"image": fname, "image_id": iid, "query": q, "qtype": qtype,
                    "target_xywh": [round(v, 1) for v in box], "W": W, "H": H, "cat": cat})
        if len(out) >= args.n:
            break

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    fr = [r["target_xywh"][2] * r["target_xywh"][3] / (r["W"] * r["H"]) for r in out]
    print(f"wrote {len(out)} triples ({n_attr} attribute / {n_rel} relpos) -> {args.output}")
    print(f"  target area frac: mean={sum(fr)/len(fr):.4f}, "
          f"<1%={sum(x<0.01 for x in fr)/len(fr):.0%}, distinct images={len(set(r['image_id'] for r in out))}")


if __name__ == "__main__":
    main()
