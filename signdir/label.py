"""Run the teacher (signdir.teacher) over data/pool.jsonl -> data/teacher.jsonl (resumable, one line per image).

Each line: {"image", "source", "split", "width", "height", "items": [{"text", "direction", "direction_crop",
"agree", "bbox"}], "raw", "seconds"}. Images already in teacher.jsonl are skipped, so the run can be stopped and
restarted at any time, and new sources added to the pool later are picked up.

  python -m signdir.label [--limit N] [--split test|train]
  python -m signdir.label --sources cambodia guide-signs --grid auto --out teacher_hr.jsonl   (tiled high-res reads)
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from signdir import teacher as T
from signdir.pool import RAW, ROOT

OUT = ROOT / "teacher.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", choices=["test", "train"], default=None, help="only this split (test first = review set)")
    ap.add_argument("--sources", nargs="+", default=None, help="only these sources")
    ap.add_argument("--grid", default="1", help='tiles per side (teacher.label grid), or "auto": 3 above 4 MP, else 2')
    ap.add_argument("--out", default=OUT.name, help="output file in data/")
    a = ap.parse_args()
    out = ROOT / a.out
    done = {json.loads(l)["image"] for l in out.open()} if out.exists() else set()
    todo = [r for r in map(json.loads, (ROOT / "pool.jsonl").open())
            if r["image"] not in done and (a.split is None or r["split"] == a.split)
            and (a.sources is None or r["source"] in a.sources)][:a.limit]
    print(f"{len(done)} done, {len(todo)} to label", flush=True)
    model, proc = T.load()
    t_all = time.time()
    with out.open("a") as fh:
        for i, r in enumerate(todo):
            t0 = time.time()
            try:
                im = T.open_image(RAW / r["image"])
                grid = (3 if im.width * im.height > 4e6 else 2) if a.grid == "auto" else int(a.grid)
                items, raw = T.label(model, proc, im, regions=r.get("regions", []), grid=grid)
            except (torch.OutOfMemoryError, OSError, ValueError, ZeroDivisionError) as e:  # huge / broken image: record and move on
                torch.cuda.empty_cache()
                print("skip", r["image"], type(e).__name__, flush=True)
                fh.write(json.dumps({**r, "error": type(e).__name__}) + "\n")
                continue
            r = {k: v for k, v in r.items() if k != "regions"}
            fh.write(json.dumps({**r, "width": im.width, "height": im.height, "items": items, "raw": raw,
                                 "seconds": round(time.time() - t0, 1), "grid": grid}, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0:
                rate = (time.time() - t_all) / (i + 1)
                print(f"{i + 1}/{len(todo)} {rate:.1f} s/img, ETA {rate * (len(todo) - i - 1) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
