"""Run 6 training set: real signs first. From the full build (build_sft --long_side 1536 --synth --all --paper --nav
--realsense, no --replay) -> data/sft/train_run6.jsonl:
  real direction-task rows (teacher)   x2      real all-signs rows (teacher_all)   x2
  RealSense photos (human-reviewed)    x3      synthetic_signs 300, synthetic_paper 300, synthetic 100, nav 500
Copies stand in for epochs (train with --epochs 1). Run 4 / 5 found synthetic rows easy (synthetic_signs e2e .84) while
real-photo detection stayed at F1 ~.39 vs the teacher's ~.84, so the real rows get most of the budget.

  python -m signdir.build_run6
"""
from __future__ import annotations

import collections
import json
import random

from signdir.build_sft import PROMPT, PROMPT_NAV, SFT

COPIES = {"real_dir": 2, "real_all": 2, "realsense": 3}
SAMPLE = {"synthetic_signs": 300, "synthetic_paper": 300, "synthetic": 100, "nav": 500}


def group(r):
    s, p = r["source"], r["conversations"][0]["value"]
    if s.startswith("realsense"):
        return "realsense"
    if p == PROMPT_NAV:
        return "nav"
    if s in SAMPLE:
        return s
    return "real_dir" if p == PROMPT else "real_all"


def main():
    by = collections.defaultdict(list)
    for r in map(json.loads, (SFT / "train.jsonl").open()):
        by[group(r)].append(r)
    rng = random.Random(6)
    out = []
    for g, rows in sorted(by.items()):
        if g in COPIES:
            out += rows * COPIES[g]
        else:
            rows = rows[:]
            rng.shuffle(rows)
            out += rows[:SAMPLE[g]]
        print(f"{g:16s} {len(rows):5d} -> {len(rows) * COPIES[g] if g in COPIES else min(len(rows), SAMPLE[g]):5d}")
    rng.shuffle(out)
    (SFT / "train_run6.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(len(out), "rows ->", SFT / "train_run6.jsonl")


if __name__ == "__main__":
    main()
