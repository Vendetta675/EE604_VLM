"""Teacher labels (data/teacher.jsonl, reviewed labels in data/reviewed.jsonl win) -> LocateAnything SFT data.

One sample per image:
  human: PROMPT
  gpt:   <ref>{destination} | {direction}</ref><box><x1><y1><x2><y2></box> ...   (coords 0..1000, reading order)
         or <box>none</box> when the image has no direction sign
Images are EXIF-rotated and resized to a long side of LONG_SIDE (the 6 GB training budget) into data/sft/images.

  python -m signdir.build_sft [--direction vote|crop|full|agree] [--long_side 1536] [--synth]
    --synth  also add the synthetic signs of data/sft/synth.jsonl (signdir.synth) to the train split
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
        out.append(f"<ref>{text} | {it['direction']}</ref><box>" + "".join(f"<{v}>" for v in q) + "</box>")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["vote", "crop", "full", "agree"], default="vote")
    ap.add_argument("--long_side", type=int, default=LONG_SIDE)
    ap.add_argument("--synth", action="store_true")
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
