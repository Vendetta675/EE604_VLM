"""Image pool for labelling: which images of which source go to the teacher, with a fixed review/test split.

Sources (data/raw):
  roboflow/subway-guide      indoor metro guide boards (images with a guide board or an arrow)
  roboflow/wayfinding-sign   indoor wayfinding signs (all)
  roboflow/exit-sign         exit signs with left/right arrows (all)
  roboflow/guide-signs       highway guide signs (all)
  roboflow/road-signs-us     road scenes, only images with a "Guide Signs" box
  roboflow/road-sign-mw      road scenes, only images with a "Guide" box               (if downloaded)
  roboflow/cambodia          road scenes, only images with a "guide_sign" box          (if downloaded)
  roboflow/taipei-station    indoor station boards (random subset)
  kaggle/indian-signboard-image-dataset, kaggle/road-and-traffic-signboards (all)
  local/shop-signboards      shop fronts: negatives, the model must output no direction (random subset)

  python -m signdir.pool        -> data/pool.jsonl  ({"image", "source", "split", "regions"}; split = test | train,
                                   regions = the dataset's own sign boxes, which the teacher also reads zoomed in)
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
RAW = ROOT / "raw"
IMG = (".jpg", ".jpeg", ".png")
# per source: (glob or COCO filter, max images); COCO filter = class names an image must contain
SOURCES = {
    "subway-guide": ({"guide board", "left arrow", "right arrow", "straight arrow", "under arrow"}, None),
    "wayfinding-sign": (None, None),
    "exit-sign": (None, None),
    "guide-signs": (None, None),
    "road-signs-us": ({"Guide Signs"}, None),
    "road-sign-mw": ({"Guide"}, 400),
    "cambodia": ({"guide_sign"}, 300),
    "taipei-station": (None, 300),
    "kaggle/indian-signboard-image-dataset": ("**/*", None),
    "kaggle/road-and-traffic-signboards": ("**/*", None),
    "local/shop-signboards": ("*/images/*", 200),
}
TEST_FRAC = 0.08  # held out for human review: the clean test set
SEED = 0


def coco_images(name, classes):
    """[(path, sign boxes [x1, y1, x2, y2] of the selected classes)] of a Roboflow COCO export."""
    out = []
    for ann in sorted((RAW / "roboflow" / name).glob("*/_annotations.coco.json")):
        j = json.loads(ann.read_text())
        cat = {c["id"]: c["name"] for c in j["categories"]}
        keep = None if classes is None else {a["image_id"] for a in j["annotations"] if cat[a["category_id"]] in classes}
        boxes = {}
        for a in j["annotations"]:
            if classes is not None and cat[a["category_id"]] in classes:
                x, y, w, h = a["bbox"]
                boxes.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
        out += [(ann.parent / im["file_name"], boxes.get(im["id"], []))
                for im in j["images"] if keep is None or im["id"] in keep]
    return out


def main():
    rows = []
    for name, (sel, cap) in SOURCES.items():
        if isinstance(sel, str):
            files = sorted((Path(p), []) for p in glob.glob(str(RAW / name / sel), recursive=True)
                           if p.lower().endswith(IMG))
        else:
            files = coco_images(name, sel)
        if not files:
            print(f"{name:40s} missing")
            continue
        rng = random.Random(f"{SEED}-{name}")  # per source, so adding a source never changes another's split
        rng.shuffle(files)
        files = files[:cap] if cap else files
        n_test = max(1, round(TEST_FRAC * len(files)))
        rows += [{"image": str(f.relative_to(RAW)), "source": name, "split": "test" if i < n_test else "train",
                  "regions": regions} for i, (f, regions) in enumerate(files)]
        print(f"{name:40s} {len(files):5d} images ({n_test} test)")
    random.Random(SEED).shuffle(rows)  # labelling order: sources interleaved, so a partial run is still balanced
    (ROOT / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "images ->", ROOT / "pool.jsonl")


if __name__ == "__main__":
    main()
