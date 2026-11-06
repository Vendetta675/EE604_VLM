"""Public stairs + indoor-obstacle datasets (Roboflow Universe, CC BY 4.0, forked to workspace irws) -> the navigation
task's SFT rows: data/sft/nav_{train,test}.jsonl, prompt build_sft.PROMPT_NAV, labels "<name> | <direction> | <type>":

  Stairs | up | stairs        Stairs | down | stairs        Chair | none | obstacle

  stairs-updown   (smart-object-identification/dataset-stairs)  a 'stairs' box per staircase + per-step up_steps /
                  down_steps boxes: the staircase's direction = majority of the step boxes inside it (no steps: dropped)
  stair-d         (new-workspace-qqd80/stair_d)                 upstair / downstair, one box per image
  ar-obstacles    (surya-ausgm/ar-navigation-obstacles)          furniture etc.; OBSTACLES maps the classes kept
  indoor-objects  (hani-fyp-2025/indoor-object-detection)        same kind of data; images seen in ar-obstacles skipped
  stairs-tomlai is downloaded but unused: its stairs have no up / down label.
Classes that are part of a wall (door, window, cabinet / fridge doors) are not obstacles and are not labelled.
Roboflow splits: train + valid -> train, test -> test. Every 3rd obstacle-dataset image with only doors / windows
is kept as a negative ("<box>none</box>": nothing in the way).

  python -m signdir.public_nav [--long_side 1536]
"""
from __future__ import annotations

import argparse
import collections
import json
import re

from PIL import Image

from signdir.build_sft import PROMPT_NAV, SFT, answer
from signdir.pool import RAW

RF = RAW / "roboflow"
OBSTACLES = {"chair": "Chair", "table": "Table", "couch": "Sofa", "sofa": "Sofa", "cabinet": "Cabinet",
             "pole": "Pole", "stoppole": "Pole", "box": "Box", "dustbin": "Dustbin", "bed": "Bed", "person": "Person",
             "peoples": "Person", "cycle": "Bicycle", "vending machine": "Vending machine", "fridge": "Fridge",
             "drawers": "Drawers", "closet": "Cupboard", "cupboard": "Cupboard", "obstacle": "Obstacle"}


def coco(name):
    """(split, image path, [(class, [x1, y1, x2, y2])]) of a downloaded COCO export."""
    for f in sorted((RF / name).glob("*/_annotations.coco.json")):
        c = json.loads(f.read_text())
        cat = {x["id"]: x["name"] for x in c["categories"]}
        by = collections.defaultdict(list)
        for a in c["annotations"]:
            x, y, w, h = a["bbox"]
            by[a["image_id"]].append((cat[a["category_id"]], [x, y, x + w, y + h]))
        for im in c["images"]:
            yield f.parent.name, f.parent / im["file_name"], by[im["id"]]


def inside(b, big, frac=0.6):
    ix = max(0, min(b[2], big[2]) - max(b[0], big[0])) * max(0, min(b[3], big[3]) - max(b[1], big[1]))
    return ix >= frac * (b[2] - b[0]) * (b[3] - b[1])


def items_for(name, anns):
    if name == "stairs-updown":
        out = []
        for c, b in anns:
            if c != "stairs":
                continue
            votes = collections.Counter("up" if s == "up_steps" else "down" for s, sb in anns
                                        if s in ("up_steps", "down_steps") and inside(sb, b))
            if votes:
                out.append({"text": "Stairs", "direction": votes.most_common(1)[0][0], "type": "stairs", "bbox": b})
        if not out and any(c == "stairs" for c, _ in anns):
            return None                                    # a staircase without step labels: direction unknown
        return out
    if name == "stair-d":
        return [{"text": "Stairs", "direction": "up" if c == "upstair" else "down", "type": "stairs", "bbox": b}
                for c, b in anns if c in ("upstair", "downstair")]
    return [{"text": OBSTACLES[c.lower()], "direction": "none", "type": "obstacle", "bbox": b}
            for c, b in anns if c.lower() in OBSTACLES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long_side", type=int, default=1536)
    a = ap.parse_args()
    out, seen, stats = {"train": [], "test": []}, set(), collections.Counter()
    (SFT / "images").mkdir(parents=True, exist_ok=True)
    for name in ("stairs-updown", "stair-d", "ar-obstacles", "indoor-objects"):
        for split, path, anns in coco(name):
            orig = re.sub(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+\.jpg$", "", path.name, flags=re.I)
            if name in ("ar-obstacles", "indoor-objects"):
                if orig in seen:
                    stats[f"{name}: duplicate"] += 1
                    continue
                seen.add(orig)
            items = items_for(name, anns)
            if items is None:
                stats[f"{name}: no step labels"] += 1
                continue
            if not items and name in ("ar-obstacles", "indoor-objects"):
                stats[f"{name}: nothing kept"] += 1
                if stats[f"{name}: nothing kept"] % 3:          # every 3rd: a "nothing in the way" negative
                    continue
                stats[f"{name}: negative"] += 1
            im = Image.open(path).convert("RGB")
            W, H = im.size
            s = min(1.0, a.long_side / max(im.size))
            if s < 1:
                im = im.resize((round(W * s), round(H * s)))
            img = f"nav__{name}__{path.stem}.jpg"
            im.save(SFT / "images" / img, quality=92)
            sp = "test" if split == "test" else "train"
            out[sp].append({"image": img, "source": name, "reviewed": False,
                            "conversations": [{"from": "human", "value": PROMPT_NAV},
                                              {"from": "gpt", "value": answer(items, W, H)}]})
            stats[f"{name}: {sp}"] += 1
            for it in items:
                stats[f"label {it['text']} | {it['direction']}"] += 1
    for sp, rows in out.items():
        (SFT / f"nav_{sp}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"nav_{sp}.jsonl: {len(rows)} images")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
