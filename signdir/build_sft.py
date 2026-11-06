"""Teacher labels (data/teacher.jsonl, reviewed labels in data/reviewed.jsonl win) -> LocateAnything SFT data.

One sample per image:
  human: PROMPT
  gpt:   <ref>{destination} | {direction}</ref><box><x1><y1><x2><y2></box> ...   (coords 0..1000, reading order)
         or <box>none</box> when the image has no direction sign
  (the all-signs task, PROMPT_ALL: <ref>{text} | {direction} | {type}</ref>, types in TYPES)
Images are EXIF-rotated and resized to a long side of LONG_SIDE (the 6 GB training budget) into data/sft/images.

  python -m signdir.build_sft [--direction vote|crop|full|agree] [--long_side 1536] [--synth] [--all]
    --synth  also add the synthetic signs of data/sft/synth.jsonl (signdir.synth) to the train split
    --all    also add the all-signs task (PROMPT_ALL): teacher_all.jsonl (signdir.label_all; same direction rule,
             an item with an arrow is a direction sign) and synth_signs_{train,test}.jsonl (signdir.synth_signs);
             its test rows go to test_all.jsonl, so test.jsonl stays the direction-task test set
    vote   the zoomed read, but "none" unless an arrow-presence check says there is an arrow   (default; best on
           the reviewed test set: 64% vs 59% for crop, half the false arrows)
    crop   use the zoomed re-read of the arrow when it parsed, else the full-image read
    full   the full-image read only
    agree  drop items whose two reads disagree (and the images that then lose items)
"""
from __future__ import annotations

import argparse
import json

from signdir import teacher as T
from signdir.pool import RAW, ROOT

PROMPT = "Read the direction signs: locate every destination with the direction of its arrow."
# all-signs task (signdir.synth_signs, signdir.label_all): every sign, "<text> | <direction> | <type>"; direction
# signs stay one item per destination, any other sign is one item with its text (or, for a pictogram, its meaning)
PROMPT_ALL = "Read every sign: locate each sign's text with the direction of its arrow and the sign type."
TYPES = ("direction", "information", "facility", "warning", "prohibition", "mandatory", "room", "handwritten")
# navigation task (--nav): stairs (direction up / down / none) and obstacles in the way, its own prompt, because the
# sign images (synthetic signs pasted on room photos) contain unlabelled furniture and stairs
PROMPT_NAV = "Find the stairs and obstacles in the way: locate each with whether the stairs go up or down."
NAV_TYPES = ("stairs", "obstacle")
LONG_SIDE = 1024  # default; --long_side 1536 fits the 6 GB QLoRA budget (peak 5.2 GiB on the longest samples)
SFT = ROOT / "sft"


def answer(items, w, h):
    if not items:
        return "<box>none</box>"
    items = sorted(items, key=lambda it: (round(it["bbox"][1] / h * 10), it["bbox"][0]))  # rows, then left to right
    out = []
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        q = [min(1000, max(0, round(v))) for v in (x1 / w * 1000, y1 / h * 1000, x2 / w * 1000, y2 / h * 1000)]
        text = " ".join(it["text"].split()).replace("<", "").replace(">", "")
        label = f"{text} | {it['direction']}" + (f" | {it['type']}" if it.get("type") else "")
        out.append(f"<ref>{label}</ref><box>" + "".join(f"<{v}>" for v in q) + "</box>")
    return "".join(out)


def final_items(items, rule):
    out = []
    for it in items:
        d = it["direction"]
        if rule in ("crop", "vote") and it.get("direction_crop"):
            d = it["direction_crop"]
        if rule == "vote":
            arrow = T.has_arrow(it)
            if arrow is None:
                return None  # not re-checked yet (signdir.recheck)
            d = d if arrow else "none"
        elif rule == "agree" and it.get("direction_crop") not in (None, d):
            return None
        out.append({**it, "direction": d})
    return out


def load_labels():
    """image -> label row; human-reviewed rows replace teacher rows."""
    rows = {}
    for name in ("teacher.jsonl", "reviewed.jsonl"):
        p = ROOT / name
        if p.exists():
            for r in map(json.loads, p.open()):
                if "error" not in r:
                    rows[r["image"]] = {**r, "reviewed": name == "reviewed.jsonl"}
    return rows


def all_signs_items(items, rule):
    """final_items for the all-signs task: the direction rule applies to items with a claimed arrow (only those
    have the arrow checks), and the type is made consistent with the result: an item with an arrow is a direction
    sign, a "direction" item without one is a plain name (information)."""
    out = []
    for it in items:
        if it["direction"] != "none":
            it = final_items([it], rule)
            if it is None:
                return None  # not re-checked
            it = it[0]
        it = dict(it)
        if it["direction"] != "none":
            it["type"] = "direction"
        elif it.get("type") == "direction":
            it["type"] = "information"
        out.append(it)
    return out


def save_image(img, long_side):
    """Resized copy of a raw image in SFT/images (once); returns its name."""
    name = img.replace("/", "__").rsplit(".", 1)[0] + ".jpg"
    if not (SFT / "images" / name).exists():
        im = T.open_image(RAW / img)
        s = min(1.0, long_side / max(im.size))
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)))
        im.save(SFT / "images" / name, quality=92)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["vote", "crop", "full", "agree"], default="vote")
    ap.add_argument("--long_side", type=int, default=LONG_SIDE)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--all", action="store_true", help="add the all-signs task (teacher_all + synth_signs)")
    ap.add_argument("--paper", action="store_true",
                    help="with --all: add room plates + handwritten notes (synth_paper_{train,test}.jsonl)")
    ap.add_argument("--replay", type=int, default=None,
                    help="keep only this many random rows of the earlier training data (direction + all-signs tasks, "
                         "run 4's) next to every new row (--paper / --nav / --realsense): a fine-tune from run 4's "
                         "adapter that fits one Kaggle session")
    ap.add_argument("--nav", action="store_true",
                    help="add the navigation task: nav_{train,test}.jsonl (signdir.public_nav) -> test_nav.jsonl")
    ap.add_argument("--realsense", action="store_true",
                    help="add the labelled RealSense photos: realsense_{train,test}.jsonl (signdir.label_capture); "
                         "their test rows go to test_real.jsonl")
    a = ap.parse_args()
    (SFT / "images").mkdir(parents=True, exist_ok=True)
    out = {"train": [], "test": []}
    dropped = 0
    for img, r in sorted(load_labels().items()):
        items = r["items"] if r["reviewed"] else final_items(r["items"], a.direction)
        if items is None:
            dropped += 1
            continue
        im = T.open_image(RAW / img)
        s = min(1.0, a.long_side / max(im.size))
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)))
        name = img.replace("/", "__").rsplit(".", 1)[0] + ".jpg"
        im.save(SFT / "images" / name, quality=92)
        W, H = r["width"], r["height"]  # labels are in the coordinates of the full-size (EXIF-rotated) image
        out[r["split"]].append({"image": name, "source": r["source"], "reviewed": r["reviewed"],
                                "conversations": [{"from": "human", "value": PROMPT},
                                                  {"from": "gpt", "value": answer(items, W, H)}]})
    if a.synth:
        out["train"] += [json.loads(l) for l in (SFT / "synth.jsonl").open()]
    if a.all:
        out["test_all"] = []
        dropped_all = 0
        for r in map(json.loads, (ROOT / "teacher_all.jsonl").open()):
            if "error" in r:
                continue
            items = all_signs_items(r["items"], a.direction)
            if items is None:
                dropped_all += 1
                continue
            row = {"image": save_image(r["image"], a.long_side), "source": r["source"], "reviewed": False,
                   "conversations": [{"from": "human", "value": PROMPT_ALL},
                                     {"from": "gpt", "value": answer(items, r["width"], r["height"])}]}
            out["train" if r["split"] == "train" else "test_all"].append(row)
        out["train"] += [json.loads(l) for l in (SFT / "synth_signs_train.jsonl").open()]
        out["test_all"] += [json.loads(l) for l in (SFT / "synth_signs_test.jsonl").open()]
        print("all-signs: dropped (not re-checked):", dropped_all)
    if a.replay is not None and len(out["train"]) > a.replay:
        import random
        random.Random(0).shuffle(out["train"])
        out["train"] = out["train"][:a.replay]
        print(f"replay: {a.replay} rows of the earlier training data kept")
    if a.all and a.paper:
        out["train"] += [json.loads(l) for l in (SFT / "synth_paper_train.jsonl").open()]
        out["test_all"] += [json.loads(l) for l in (SFT / "synth_paper_test.jsonl").open()]
    if a.nav:
        out["train"] += [json.loads(l) for l in (SFT / "nav_train.jsonl").open()]
        out["test_nav"] = [json.loads(l) for l in (SFT / "nav_test.jsonl").open()]
    if a.realsense:
        out["train"] += [json.loads(l) for l in (SFT / "realsense_train.jsonl").open()]
        out["test_real"] = [json.loads(l) for l in (SFT / "realsense_test.jsonl").open()]
    for split, rows in out.items():
        (SFT / f"{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        neg = sum(x["conversations"][1]["value"] == "<box>none</box>" for x in rows)
        print(f"{split}: {len(rows)} images ({neg} without a direction sign)")
    print("dropped (reads disagree / not re-checked):", dropped)
    recipe = {"signdir": {"annotation": str(SFT / "train.jsonl"), "root": str(SFT / "images"),
                          "repeat_time": 1.0, "data_augment": False}}
    (SFT / "recipe.json").write_text(json.dumps(recipe, indent=2))


if __name__ == "__main__":
    main()
