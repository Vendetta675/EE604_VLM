"""Score teacher label files against the human review (data/reviewed.jsonl) on the images they all cover.

Labels go through build_sft.final_items (direction rule "vote", as in training); scoring is signdir.evaluate.score.
Note the review was made by correcting data/teacher.jsonl, which favours that file slightly.

  python -m signdir.compare_teacher teacher.jsonl teacher_hr.jsonl
"""
from __future__ import annotations

import argparse
import json

from signdir.build_sft import final_items
from signdir.evaluate import score
from signdir.pool import ROOT

KEYS = ("det_P", "det_R", "det_F1", "text_sim", "direction_acc", "e2e_F1", "neg_FP_rate")


def load(name):
    out = {}
    for r in map(json.loads, (ROOT / name).open()):
        items = final_items(r.get("items", []), "vote") if "error" not in r else None
        if items is not None:
            out[r["image"]] = items
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    review = {r["image"]: r for r in map(json.loads, (ROOT / "reviewed.jsonl").open())}
    labels = {f: load(f) for f in a.files}
    common = sorted(set(review).intersection(*labels.values()))
    print(len(common), "reviewed images labelled by every file")
    for src in [None] + sorted({review[i]["source"] for i in common}):
        imgs = [i for i in common if src is None or review[i]["source"] == src]
        print(f"\n{src or 'ALL'} ({len(imgs)} images, {sum(len(review[i]['items']) for i in imgs)} reviewed items)")
        for f in a.files:
            m = score([{"pred": labels[f][i], "gt": review[i]["items"]} for i in imgs])
            print(f"  {f:22s} " + "  ".join(f"{k} {m[k]:.3f}" for k in KEYS) + f"  items {m['pred_items']}")


if __name__ == "__main__":
    main()
