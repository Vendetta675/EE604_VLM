"""Draft labels for the RealSense photos (signdir.capture) with the Qwen2.5-VL-7B teacher, for the human review
(signdir.review_capture) -> data/realsense_draft.jsonl.

Scenes: photos are taken in bursts, so consecutive photos with the same tag less than GAP_S apart form one scene.
Per tag, every other scene goes to the test split (the real test set), the rest to train, so no scene is in both.
Photos listed in data/raw/realsense/exclude.txt (one file stem per line; faces by default) are skipped.

Prompts by tag:
  room / handwritten / sign   the all-signs prompt (label_all.PROMPT) + how the new types and drawn symbols are labelled
  stairs-up / stairs-down     the navigation prompt; the tag sets the stairs' direction (the photographer knows it)
  obstacle                    the navigation prompt

  python -m signdir.label_capture            (GPU, ~15-30 s per photo; resumes: photos already drafted are skipped)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch

from signdir import label_all as LA
from signdir import teacher as T
from signdir.pool import RAW, ROOT

CAP = RAW / "realsense"
OUT = ROOT / "realsense_draft.jsonl"
GAP_S = 5
SIGN_TAGS = ("room", "handwritten", "sign")

SIGN_PROMPT = LA.PROMPT.replace(
    "Output [] only if",
    "Type room = a room / door number plate or label (e.g. \"Room 214\", \"4i\", \"Lab 3\"). Type handwritten = "
    "anything written or drawn by hand with a pen or marker on paper or a board. A drawn arrow alone has text "
    "\"Arrow\" and its direction; an arrow inside a circle has text \"Go straight\" (or its direction); an empty "
    "drawn triangle has text \"Warning\"; written words keep their text (e.g. \"EXIT\", \"NO ENTRY\"). Output [] only if")
NAV_PROMPT = (
    "Find the stairs and the obstacles a walking person could bump into or trip over in this image (furniture, "
    "plants, bins, boxes, poles, machines, people). Output a JSON list with one object per staircase or obstacle, "
    'with keys "text" (its name, e.g. "Stairs", "Chair", "Plant", "Dustbin"), "direction" (for stairs: up or down '
    'as seen from the camera; else none), "type" (stairs or obstacle) and "bbox_2d" ([x1, y1, x2, y2]). Output [] '
    "if there is none. Write compact JSON on one line.")


def photos():
    excl = set()
    ex = CAP / "exclude.txt"
    if ex.exists():
        excl = {l.strip() for l in ex.read_text().splitlines() if l.strip() and not l.startswith("#")}
    rows = []
    for j in sorted(CAP.glob("*/*.json")):
        meta = json.loads(j.read_text())
        if j.stem in excl or not j.with_suffix(".jpg").exists():
            continue
        rows.append({"image": str(j.with_suffix(".jpg").relative_to(RAW)), "tag": meta["tag"], "time": meta["time"]})
    rows.sort(key=lambda r: r["time"])
    scene, last = -1, None
    for r in rows:
        if last is None or r["tag"] != last["tag"] or r["time"] - last["time"] > GAP_S:
            scene += 1
        r["scene"], last = scene, r
    for tag in {r["tag"] for r in rows}:
        scenes = sorted({r["scene"] for r in rows if r["tag"] == tag})
        test = set(scenes[1::2])                       # every other scene of this tag: test
        for r in rows:
            if r["tag"] == tag:
                r["split"] = "test" if r["scene"] in test else "train"
    return rows, excl


def parse_nav(answer, sx, sy, tag):
    s = answer[answer.find("["): answer.rfind("]") + 1]
    try:
        raw = json.loads(s)
    except json.JSONDecodeError:
        raw = [json.loads(m) for m in re.findall(r"\{[^{}]*\}", s) if T._ok(m)]
    out = []
    for it in raw if isinstance(raw, list) else []:
        if not isinstance(it, dict) or not isinstance(it.get("bbox_2d"), list) or len(it["bbox_2d"]) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in it["bbox_2d"])
        if x2 <= x1 or y2 <= y1:
            continue
        typ = "stairs" if str(it.get("type", "")).lower().startswith("stair") else "obstacle"
        d = str(it.get("direction", "none")).lower()
        if typ == "stairs":
            d = "up" if tag == "stairs-up" else "down" if tag == "stairs-down" else (d if d in ("up", "down") else "none")
        else:
            d = "none"
        text = "Stairs" if typ == "stairs" else (str(it.get("text", "")).strip().capitalize() or "Obstacle")
        out.append({"text": text, "direction": d, "type": typ, "bbox": [x1 * sx, y1 * sy, x2 * sx, y2 * sy]})
    return out


@torch.inference_mode()
def draft(model, proc, image, tag):
    prompt = SIGN_PROMPT if tag in SIGN_TAGS else NAV_PROMPT
    answer, inputs = T._ask(model, proc, image, prompt, 768)
    gh, gw = inputs.image_grid_thw[0, 1].item() * 14, inputs.image_grid_thw[0, 2].item() * 14
    sx, sy = image.width / gw, image.height / gh
    return (LA.parse(answer, sx, sy) if tag in SIGN_TAGS else parse_nav(answer, sx, sy, tag)), answer


def main():
    rows, excl = photos()
    done = {json.loads(l)["image"] for l in OUT.open()} if OUT.exists() else set()
    todo = [r for r in rows if r["image"] not in done]
    n_sc = len({r["scene"] for r in rows})
    print(f"{len(rows)} photos ({len(excl)} excluded), {n_sc} scenes; test {sum(r['split'] == 'test' for r in rows)}; "
          f"{len(todo)} to draft", flush=True)
    model, proc = T.load()
    t_all = time.time()
    with OUT.open("a") as fh:
        for i, r in enumerate(todo):
            t0 = time.time()
            im = T.open_image(RAW / r["image"])
            try:
                items, raw = draft(model, proc, im, r["tag"])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                items, raw = [], "OOM"
            task = "all" if r["tag"] in SIGN_TAGS else "nav"
            fh.write(json.dumps({**r, "task": task, "width": im.width, "height": im.height, "items": items,
                                 "raw": raw, "seconds": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0:
                rate = (time.time() - t_all) / (i + 1)
                print(f"{i + 1}/{len(todo)} {rate:.1f} s/img, ETA {rate * (len(todo) - i - 1) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
