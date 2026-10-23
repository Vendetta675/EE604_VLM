"""Add the arrow-presence checks (teacher.arrow_checks) to teacher.jsonl rows labelled before they existed.

Rewrites data/teacher.jsonl in place (atomically, every --every images), so it must not run while signdir.label is
appending to it. Resumable: items that already have "arrow_yn" are skipped.

  python -m signdir.recheck [--split train]
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from signdir import teacher as T
from signdir.label import OUT
from signdir.pool import RAW


def save(rows):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    os.replace(tmp, OUT)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--every", type=int, default=20)
    a = ap.parse_args()
    rows = [json.loads(l) for l in OUT.open()]
    todo = [i for i, r in enumerate(rows) if r["split"] == a.split and any("arrow_yn" not in it for it in r.get("items", []))]
    print(len(todo), "images to re-check", flush=True)
    model, proc = T.load()
    for n, i in enumerate(todo, 1):
        im = T.open_image(RAW / rows[i]["image"])
        for it in rows[i]["items"]:
            if "arrow_yn" not in it:
                T.arrow_checks(model, proc, im, it)
        if n % a.every == 0 or n == len(todo):
            save(rows)
            print(f"{n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
