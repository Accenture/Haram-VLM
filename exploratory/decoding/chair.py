#!/usr/bin/env python
"""
CHAIR (Caption Hallucination Assessment with Image Relevance, Rohrbach et al. 2018) scorer.

Given generated captions and COCO val2014 instance annotations, compute:
  CHAIR_s = fraction of captions containing >=1 hallucinated object
  CHAIR_i = fraction of mentioned object instances that are hallucinated
plus avg recall (GT objects mentioned / GT objects present) and avg #objects mentioned, so we can verify
a method does not merely shorten captions. An object is "mentioned" if any of its synonyms appears in the
caption; "hallucinated" if mentioned but absent from the image's COCO GT objects.

Absolute CHAIR values depend on the synonym list and parser; we hold both FIXED across methods, so the
RELATIVE comparison between decoding/perception conditions is what matters. Pure Python (no nltk).
"""
import json, re
from collections import defaultdict

# COCO-category synonym list (canonical CHAIR mapping; word -> COCO 80 category name).
SYNONYMS = {
    "person": ["person", "girl", "boy", "man", "woman", "kid", "child", "chef", "baker", "people",
               "adult", "rider", "children", "baby", "worker", "passenger", "sister", "biker",
               "policeman", "cop", "officer", "lady", "cowboy", "bride", "groom", "male", "female",
               "guy", "traveler", "mother", "father", "gentleman", "pitcher", "player", "skier",
               "snowboarder", "skater", "skateboarder", "foreigner", "caller", "offender", "coworker",
               "trainer", "bartender", "knight", "mom", "dad", "men", "women", "teenager", "girlfriend",
               "boyfriend", "guard", "surfer", "musician", "pedestrian", "couple", "crowd", "tourist",
               "human", "humans"],
    "bicycle": ["bicycle", "bike", "bicycles", "bikes", "unicycle", "tricycle"],
    "car": ["car", "automobile", "van", "minivan", "sedan", "suv", "hatchback", "cab", "jeep", "coupe",
            "taxicab", "limo", "taxi", "cars"],
    "motorcycle": ["motorcycle", "scooter", "motor bike", "motor cycle", "motorbike", "moped",
                   "motorcycles"],
    "airplane": ["airplane", "jet", "plane", "aircraft", "airbus", "biplane", "seaplane", "planes",
                 "airplanes"],
    "bus": ["bus", "minibus", "trolley", "buses"],
    "train": ["train", "locomotive", "tramway", "caboose", "trains"],
    "truck": ["truck", "pickup", "lorry", "hauler", "firetruck", "trucks"],
    "boat": ["boat", "ship", "liner", "sailboat", "motorboat", "dinghy", "yacht", "catamaran",
             "canoe", "gondola", "speedboat", "raft", "kayak", "boats", "ships"],
    "traffic light": ["traffic light", "stop light", "street light", "traffic signal", "stoplight",
                      "traffic lights"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter"],
    "bench": ["bench", "pew", "benches"],
    "bird": ["bird", "ostrich", "owl", "seagull", "goose", "duck", "parakeet", "falcon", "robin",
             "pelican", "waterfowl", "heron", "hummingbird", "mallard", "finch", "pigeon", "sparrow",
             "seabird", "osprey", "blackbird", "fowl", "egret", "cormorant", "parrot", "dove", "crow",
             "eagle", "hawk", "swan", "turkey", "penguin", "birds"],
    "cat": ["cat", "kitten", "feline", "tabby", "cats", "kitty"],
    "dog": ["dog", "puppy", "beagle", "pup", "chihuahua", "schnauzer", "dachshund", "rottweiler",
            "canine", "pitbull", "collie", "pug", "terrier", "poodle", "labrador", "doggie", "doberman",
            "mutt", "doggy", "spaniel", "bulldog", "sheepdog", "weimaraner", "corgi", "dogs", "hound"],
    "horse": ["horse", "colt", "pony", "stallion", "mare", "foal", "horses"],
    "sheep": ["sheep", "lamb", "ram", "lambs", "goat", "ewe"],
    "cow": ["cow", "cattle", "oxen", "ox", "calf", "cows", "bull", "holstein", "heifer", "buffalo",
            "bovine", "yak"],
    "elephant": ["elephant", "elephants"],
    "bear": ["bear", "panda", "bears"],
    "zebra": ["zebra", "zebras"],
    "giraffe": ["giraffe", "giraffes"],
    "backpack": ["backpack", "knapsack", "backpacks"],
    "umbrella": ["umbrella", "parasol", "umbrellas"],
    "handbag": ["handbag", "purse", "wallet", "satchel", "pocketbook", "handbags", "wristlet"],
    "tie": ["tie", "necktie", "bowtie", "ties"],
    "suitcase": ["suitcase", "suit case", "luggage", "baggage", "briefcase", "suitcases"],
    "frisbee": ["frisbee", "frisbees"],
    "skis": ["skis", "ski"],
    "snowboard": ["snowboard", "snowboards"],
    "sports ball": ["sports ball", "ball", "football", "soccer", "basketball", "baseball", "volleyball",
                    "tennis ball", "balls"],
    "kite": ["kite", "kites"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "mitt", "glove"],
    "skateboard": ["skateboard", "skateboards"],
    "surfboard": ["surfboard", "longboard", "surf board", "bodyboard", "boards", "surfboards"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "bottle": ["bottle", "bottles", "flask"],
    "wine glass": ["wine glass", "wine glasses", "wineglass"],
    "cup": ["cup", "cups", "mug", "teacup"],
    "fork": ["fork", "forks"],
    "knife": ["knife", "knives", "pocketknife", "machete"],
    "spoon": ["spoon", "spoons", "teaspoon"],
    "bowl": ["bowl", "bowls", "container"],
    "banana": ["banana", "bananas"],
    "apple": ["apple", "apples"],
    "sandwich": ["sandwich", "sandwiches", "burger", "hamburger", "cheeseburger"],
    "orange": ["orange", "oranges"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot", "carrots"],
    "hot dog": ["hot dog", "hotdog", "hot dogs", "hotdogs"],
    "pizza": ["pizza", "pizzas"],
    "donut": ["donut", "doughnut", "donuts", "doughnuts", "bagel"],
    "cake": ["cake", "cakes", "cheesecake", "cupcake", "pastry"],
    "chair": ["chair", "chairs", "seat", "stool", "armchair", "recliner"],
    "couch": ["couch", "sofa", "loveseat", "couches", "settee"],
    "potted plant": ["potted plant", "houseplant", "potted plants", "plant"],
    "bed": ["bed", "beds", "bunk", "cot"],
    "dining table": ["dining table", "table", "desk", "tables"],
    "toilet": ["toilet", "urinal", "commode", "lavatory", "potty", "toilets"],
    "tv": ["tv", "television", "televisions", "telly", "monitor", "tvs"],
    "laptop": ["laptop", "computer", "notebook", "netbook", "laptops", "macbook"],
    "mouse": ["mouse", "computer mouse", "mice"],
    "remote": ["remote", "remote control", "controller", "remotes"],
    "keyboard": ["keyboard", "keyboards", "keypad"],
    "cell phone": ["cell phone", "cellphone", "mobile phone", "phone", "iphone", "smartphone",
                   "cell phones", "phones"],
    "microwave": ["microwave", "microwaves"],
    "oven": ["oven", "stove", "stovetop", "ovens", "range"],
    "toaster": ["toaster", "toasters"],
    "sink": ["sink", "sinks", "basin"],
    "refrigerator": ["refrigerator", "fridge", "freezer", "refrigerators"],
    "book": ["book", "books", "novel", "booklet"],
    "clock": ["clock", "clocks", "wristwatch", "watch"],
    "vase": ["vase", "vases"],
    "scissors": ["scissors", "shears"],
    "teddy bear": ["teddy bear", "teddy", "teddy bears", "stuffed animal", "stuffed bear", "plush"],
    "hair drier": ["hair drier", "hair dryer", "blow dryer", "hairdryer"],
    "toothbrush": ["toothbrush", "toothbrushes"],
}

# build word -> category, longest-phrase-first for matching
_WORD2CAT = {}
for cat, words in SYNONYMS.items():
    for w in words:
        _WORD2CAT[w] = cat
_PHRASES = sorted(_WORD2CAT.keys(), key=lambda w: -len(w.split()))


def load_gt(instances_path):
    d = json.load(open(instances_path))
    catid2name = {c["id"]: c["name"] for c in d["categories"]}
    img2cats = defaultdict(set)
    for a in d["annotations"]:
        img2cats[a["image_id"]].add(catid2name[a["category_id"]])
    fn2id = {im["file_name"]: im["id"] for im in d["images"]}
    return img2cats, fn2id


def mentioned_objects(caption):
    """Return the set of COCO categories mentioned in the caption (longest-phrase-first, word-boundary)."""
    text = " " + re.sub(r"[^a-z ]", " ", caption.lower()) + " "
    text = re.sub(r"\s+", " ", text)
    found, used = set(), text
    for phrase in _PHRASES:
        if re.search(r"(?<= )" + re.escape(phrase) + r"(?= )", used):
            found.add(_WORD2CAT[phrase])
            used = re.sub(r"(?<= )" + re.escape(phrase) + r"(?= )", " ", used)  # consume to avoid sub-phrase double count
    return found


def score(captions, img2cats, fn2id):
    """captions: list of {image, caption}. Returns CHAIR metrics."""
    n_caps, n_hall_caps = 0, 0
    n_mentions, n_hall_mentions = 0, 0
    recalls, n_obj = [], []
    per = []
    for r in captions:
        gid = fn2id.get(r["image"])
        if gid is None: continue
        gt = img2cats.get(gid, set())
        ment = mentioned_objects(r["caption"])
        hall = ment - gt
        n_caps += 1; n_hall_caps += (1 if hall else 0)
        n_mentions += len(ment); n_hall_mentions += len(hall)
        if gt: recalls.append(len(ment & gt) / len(gt))
        n_obj.append(len(ment))
        per.append({"image": r["image"], "caption": r["caption"], "mentioned": sorted(ment),
                    "hallucinated": sorted(hall)})
    return {"n": n_caps,
            "CHAIR_s": n_hall_caps / max(1, n_caps),
            "CHAIR_i": n_hall_mentions / max(1, n_mentions),
            "avg_objects": sum(n_obj) / max(1, len(n_obj)),
            "avg_recall": sum(recalls) / max(1, len(recalls)),
            "per_caption": per}


if __name__ == "__main__":
    # quick self-test of the parser
    for c in ["A man riding a bicycle next to a dog and a car.",
              "A dining table with a wine glass and a teddy bear on it."]:
        print(c, "->", sorted(mentioned_objects(c)))
