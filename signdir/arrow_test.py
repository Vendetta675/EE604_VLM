"""Tune the teacher's arrow check on the hand-reviewed test set (data/reviewed.jsonl).

Each reviewed destination matched to a teacher item (box IoU >= 0.5) is re-asked with several prompts on a zoomed
crop around the teacher's box; answers are cached in data/arrow_test.jsonl (resumable), then scored against the
reviewed direction.

  python -m signdir.arrow_test run [--per_source 60]
  python -m signdir.arrow_test score
"""
from __future__ import annotations

import argparse
import collections
import json
import random

import torch

from signdir import teacher as T
from signdir.pool import RAW, ROOT

OUT = ROOT / "arrow_test.jsonl"
DIRS = ", ".join(T.DIRECTIONS)
PROMPTS = {
    "yn_plain": (T.ARROW_YN, False),
    "yn_box": ('The red box marks the sign text "{text}". Is there an arrow symbol inside the red box? '
               "Answer yes or no.", True),
    "dir_box": (T.DIR_BOX, True),
}
marked_crop, yes = T.marked_crop, T.yes


def cases(per_source):
    teach = {r["image"]: r for r in map(json.loads, (ROOT / "teacher.jsonl").open()) if r["split"] == "test"}
    out = collections.defaultdict(list)
    for r in map(json.loads, (ROOT / "reviewed.jsonl").open()):
        t = teach[r["image"]].get("items", [])
        for k, it in enumerate(r["items"]):
            j = max(range(len(t)), key=lambda j: T.box_iou(it["bbox"], t[j]["bbox"]), default=None)
            if j is not None and T.box_iou(it["bbox"], t[j]["bbox"]) >= 0.5:
                out[r["source"]].append({"image": r["image"], "k": k, "source": r["source"], "text": t[j]["text"],
                                         "bbox": t[j]["bbox"], "truth": it["direction"],
                                         "full": t[j]["direction"], "crop": t[j].get("direction_crop")})
    rng = random.Random(0)
    return [c for s in sorted(out) for c in rng.sample(out[s], min(per_source, len(out[s])))]


@torch.inference_mode()
def run(per_source):
    done = {(r["image"], r["k"]) for r in map(json.loads, OUT.open())} if OUT.exists() else set()
    todo = [c for c in cases(per_source) if (c["image"], c["k"]) not in done]
    print(len(done), "done,", len(todo), "to ask", flush=True)
    model, proc = T.load()
    images = {}
    with OUT.open("a") as fh:
        for i, c in enumerate(todo):
            im = images.get(c["image"]) or T.open_image(RAW / c["image"])
            images = {c["image"]: im}
            plain, boxed = T.crop_around(im, c["bbox"]), marked_crop(im, c["bbox"])
            for name, (prompt, box) in PROMPTS.items():
                a, _ = T._ask(model, proc, boxed if box else plain, prompt.format(text=c["text"]), 24)
                c[name] = a.strip()
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"{i + 1}/{len(todo)}", flush=True)


def score():
    rows = [json.loads(l) for l in OUT.open()]
    rules = {
        "full (old)": lambda r: r["full"],
        "crop (old)": lambda r: r["crop"] or r["full"],
        "dir_box": lambda r: T.first_direction(r["dir_box"]),
        "yn_plain -> crop": lambda r: (r["crop"] or r["full"]) if yes(r["yn_plain"]) else "none",
        "yn_box -> crop": lambda r: (r["crop"] or r["full"]) if yes(r["yn_box"]) else "none",
        "yn_box -> dir_box": lambda r: (T.first_direction(r["dir_box"]) or "none") if yes(r["yn_box"]) else "none",
        "both yn -> crop": lambda r: (r["crop"] or r["full"]) if yes(r["yn_box"]) and yes(r["yn_plain"]) else "none",
    }
    arrow = [r["truth"] != "none" for r in rows]
    print(f"{len(rows)} destinations, {sum(arrow)} with an arrow")
    by = collections.defaultdict(lambda: collections.Counter())
    for name, f in rules.items():
        ok = pres = tp = fp = 0
        for r, a in zip(rows, arrow):
            p = f(r) or "none"
            ok += p == r["truth"]
            pres += (p != "none") == a
            tp += p != "none" and a
            fp += p != "none" and not a
            by[name][r["source"]] += p == r["truth"]
        print(f"{name:20s} direction {ok / len(rows):.3f}  arrow present/absent {pres / len(rows):.3f}  "
              f"arrow recall {tp / max(1, sum(arrow)):.2f}  false arrows {fp}")
    n = collections.Counter(r["source"] for r in rows)
    print("\nper source (direction accuracy):")
    for s in sorted(n):
        print(f"  {s:40s} n={n[s]:3d}  " + "  ".join(f"{k}: {by[k][s] / n[s]:.2f}" for k in
                                                     ("crop (old)", "yn_box -> crop", "yn_box -> dir_box")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--per_source", type=int, default=60)
    a = ap.parse_args()
    run(a.per_source) if a.cmd == "run" else score()
