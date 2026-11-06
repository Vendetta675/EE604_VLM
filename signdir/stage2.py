"""Second-stage test: caption ground-truth sign crops with a sign-meaning adapter (e.g. work_dirs/ee604_zip, which
answers "This sign indicates <X>." and outputs no boxes) and score <X> against the labels.

Crops: every ground-truth item of data/sft/test.jsonl (direction task) and test_all.jsonl (all-signs task, with a sign
type), box + PAD on each side, enlarged to a long side of at least MIN_SIDE. Per crop:
  text_ok   <X> matches the label text: edit similarity >= 0.8, or one contains the other (case-insensitive)
  dir_ok    <X> is a direction word and equals the label's direction (only counted when <X> is a direction word)
Reported overall, per task, per source and per sign type; raw captions in data/sft/stage2_<adapter>.json.

  python -m signdir.stage2 --adapter work_dirs/ee604_zip [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from signdir import la
from signdir.build_sft import SFT
from signdir.evaluate import similarity

PROMPT = "What does this sign indicate?"
PAD, MIN_SIDE = 0.15, 448
DIRS = {"left", "right", "straight", "up", "down", "back", "up-left", "up-right", "down-left", "down-right",
        "ahead", "forward"}
SAY = re.compile(r"^this (?:\w+ )?sign (?:indicates|is used to|is for|is|means|shows)\s+(?:an? |the )?(.*?)\.?$",
                 re.I)


def meaning(caption: str) -> str:
    c = caption.replace("<|im_end|>", "").strip()
    m = SAY.match(c)
    return (m.group(1) if m else c).strip().lower()


def crop(im, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    c = im.crop((max(0, x1 - PAD * w), max(0, y1 - PAD * h), min(im.width, x2 + PAD * w), min(im.height, y2 + PAD * h)))
    s = MIN_SIDE / max(c.size)
    return c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), resample=3) if s > 1 else c


def text_ok(x, t):
    x, t = x.lower().strip(), t.lower().strip()
    return bool(x) and (similarity(x, t) >= 0.8 or (len(x) >= 3 and (x in t or t in x)))


def summary(rows):
    n = len(rows)
    dir_rows = [r for r in rows if r["meaning"] in DIRS]
    return {"crops": n, "text_ok": sum(r["text_ok"] for r in rows) / max(1, n),
            "dir_word_frac": len(dir_rows) / max(1, n),
            "dir_ok_when_dir_word": sum(r["dir_ok"] for r in dir_rows) / max(1, len(dir_rows)) if dir_rows else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="work_dirs/ee604_zip")
    ap.add_argument("--limit", type=int, default=None, help="images per test file")
    a = ap.parse_args()
    model, processor, tokenizer = la.load(adapter=a.adapter)
    model.eval()
    out = []
    for task, fname in (("direction", "test.jsonl"), ("all", "test_all.jsonl")):
        tests = [json.loads(l) for l in (SFT / fname).open()][:a.limit]
        for k, t in enumerate(tests):
            im = Image.open(SFT / "images" / t["image"]).convert("RGB")
            for g in la.parse(t["conversations"][1]["value"], im.width, im.height):
                cap = la.predict(model, processor, tokenizer, crop(im, g["bbox"]), PROMPT, mode="slow",
                                 max_new_tokens=48)
                x = meaning(cap)
                out.append({"task": task, "image": t["image"], "source": t["source"], "gt_text": g["text"],
                            "gt_direction": g["direction"], "gt_type": g.get("type", ""), "caption": cap,
                            "meaning": x, "text_ok": text_ok(x, g["text"]),
                            "dir_ok": x in DIRS and x.replace("ahead", "straight").replace("forward", "straight")
                            == g["direction"]})
            if k % 25 == 0:
                print(f"{task} {k}/{len(tests)} images, {len(out)} crops", flush=True)
    res = {"all": summary(out)}
    for key in ("task", "source", "gt_type"):
        groups = defaultdict(list)
        for r in out:
            groups[r[key]].append(r)
        res[key] = {g: summary(rs) for g, rs in sorted(groups.items())}
    path = SFT / f"stage2_{Path(a.adapter).name}.json"
    path.write_text(json.dumps({"metrics": res, "rows": out}, ensure_ascii=False, indent=1))
    print(json.dumps(res, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
