"""Teacher labels for the all-signs task (build_sft.PROMPT_ALL) -> data/teacher_all.jsonl (resumable).

Qwen2.5-VL-7B (signdir.teacher) reads every sign, not only direction signs: {"text", "direction", "type", "bbox"},
type in build_sft.TYPES. As in signdir.teacher.label, the dataset's own sign boxes are also read zoomed in, and an
item with an arrow gets the arrow checks (direction_crop, arrow_yn, direction_box) that build_sft's "vote" rule uses.

Images: the real sources of data/pool.jsonl in SOURCES (same test split), plus the DataCluster Indian traffic-scene
photos (kaggle/indian-traffic-sign, VOC boxes as regions, 8 % test).

  python -m signdir.label_all [--limit N] [--split test|train]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import xml.etree.ElementTree as ET

import torch

from signdir import teacher as T
from signdir.build_sft import TYPES
from signdir.pool import RAW, ROOT, TEST_FRAC

OUT = ROOT / "teacher_all.jsonl"
# pool source -> max images (None = all)
SOURCES = {"kaggle/indian-signboard-image-dataset": None, "kaggle/road-and-traffic-signboards": None,
           "local/shop-signboards": None, "wayfinding-sign": None, "exit-sign": None, "subway-guide": 150}

PROMPT = (
    "Read every sign in this image: direction and wayfinding signs, room / department / facility signs, warning, "
    "prohibition and mandatory signs, shop and name boards, traffic signs. "
    "Output a JSON list with one object per sign, with keys "
    '"text" (all the text on the sign as written, in its original language and script, lines joined with " / "; '
    'for a sign with only a symbol, its meaning in English, e.g. "No parking"), '
    '"direction" (one of ' + ", ".join(T.DIRECTIONS) + '; straight = arrow pointing up/ahead; none = no arrow), '
    '"type" (one of ' + ", ".join(TYPES) + ") and "
    '"bbox_2d" ([x1, y1, x2, y2] around the sign). '
    "A direction sign listing several destinations gets one object per destination, with type direction and a box "
    "around that destination and its arrow. Output [] only if the image has no sign. Write compact JSON on one line."
)
# type names the teacher writes outside TYPES (it echoes the prompt's list of sign kinds)
TYPE_ALIASES = {"shop": "information", "name": "information", "traffic": "information"}


def sign_type(t):
    t = str(t).strip().lower()
    return t if t in TYPES else next((v for k, v in TYPE_ALIASES.items() if t.startswith(k)), None)


def parse(answer, sx, sy):
    """T.parse plus the type (items without a valid type are dropped)."""
    s = answer[answer.find("["): answer.rfind("]") + 1]
    try:
        raw = json.loads(s)
    except json.JSONDecodeError:
        raw = [json.loads(m) for m in re.findall(r"\{[^{}]*\}", s) if T._ok(m)]
    out = []
    for it in raw if isinstance(raw, list) else []:
        if not isinstance(it, dict):
            continue
        typ = sign_type(it.get("type", ""))
        if typ is None:
            continue
        p = T.parse(json.dumps([it]), sx, sy)
        if p:
            p[0]["type"] = typ
            if typ == "direction" and p[0]["direction"] == "none":
                p[0]["type"] = "information"  # a "direction" sign without an arrow is just a name
            out.append(p[0])
    return out


def read(model, proc, image, max_new_tokens=768):
    answer, inputs = T._ask(model, proc, image, PROMPT, max_new_tokens)
    gh, gw = inputs.image_grid_thw[0, 1].item() * 14, inputs.image_grid_thw[0, 2].item() * 14
    return parse(answer, image.width / gw, image.height / gh), answer


@torch.inference_mode()
def label(model, proc, image, regions=(), max_regions=4):
    items, answer = read(model, proc, image)
    for it in items:
        it["from"] = "full"
    for r in sorted(regions, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:max_regions]:
        if (r[2] - r[0]) * (r[3] - r[1]) > 0.5 * image.width * image.height:
            continue
        crop = T.crop_around(image, r, pad=0.3, min_side=448)
        ox, oy = T.crop_origin(image, r, pad=0.3, min_side=448)
        found, _ = read(model, proc, crop)
        for it in found:
            it["bbox"] = [it["bbox"][0] + ox, it["bbox"][1] + oy, it["bbox"][2] + ox, it["bbox"][3] + oy]
            it["from"] = "region"
            items = [o for o in items if o["from"] == "region" or T.box_iou(o["bbox"], it["bbox"]) <= 0.3]
            if not any(T.box_iou(o["bbox"], it["bbox"]) > 0.5 for o in items):
                items.append(it)
    out, seen = [], set()
    for it in items:
        key = (it["text"].lower(), tuple(round(v) for v in it["bbox"]))
        if key in seen:
            continue
        seen.add(key)
        if it["direction"] != "none":  # arrow checks only where an arrow is claimed
            a, _ = T._ask(model, proc, T.crop_around(image, it["bbox"]), T.CHECK.format(text=it["text"]), 24)
            it["direction_crop"] = T.first_direction(a)
            it["agree"] = it["direction_crop"] == it["direction"]
            T.arrow_checks(model, proc, image, it)
        out.append(it)
    return out, answer


def voc_images():
    """DataCluster Indian traffic scenes: [{"image", "source", "split", "regions"}]."""
    rows = []
    for x in sorted((RAW / "kaggle/indian-traffic-sign").rglob("*.xml")):
        root = ET.parse(x).getroot()
        name = root.findtext("filename")
        img = next((RAW / "kaggle/indian-traffic-sign").rglob(name), None)
        if img is None:
            continue
        boxes = [[float(o.find("bndbox").findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                 for o in root.iter("object")]
        rows.append({"image": str(img.relative_to(RAW)), "source": "kaggle/indian-traffic-sign", "regions": boxes})
    rng = random.Random(0)
    for r in rows:
        r["split"] = "test" if rng.random() < TEST_FRAC else "train"
    return rows


def todo_rows():
    rng = random.Random(0)
    pool = [r for r in map(json.loads, (ROOT / "pool.jsonl").open()) if r["source"] in SOURCES]
    rows = []
    for src, cap in SOURCES.items():
        rs = [r for r in pool if r["source"] == src]
        if cap is not None and len(rs) > cap:
            test = [r for r in rs if r["split"] == "test"]
            rs = test + rng.sample([r for r in rs if r["split"] == "train"], cap - len(test))
        rows += rs
    return rows + voc_images()


def reparse():
    """Recover items dropped by the first parse (types outside TYPES, before TYPE_ALIASES) from each row's full-image
    answer. Recovered arrows have no arrow checks, so they are kept as "none" (build_sft's vote rule needs the checks)."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    rows, added = [json.loads(l) for l in OUT.open()], 0
    for r in rows:
        if "raw" not in r:
            continue
        gh, gw = smart_resize(r["height"], r["width"], 28, 256 * 28 * 28, T.MAX_PIXELS)  # as T.load's processor
        have = {it["text"].lower() for it in r["items"]}
        for it in re.findall(r"\{[^{}]*\}", r["raw"]):
            if not T._ok(it):
                continue
            d = json.loads(it)
            if str(d.get("type", "")).strip().lower() in TYPES or sign_type(d.get("type", "")) is None:
                continue
            p = T.parse(json.dumps([d]), r["width"] / gw, r["height"] / gh)
            if not p or p[0]["text"].lower() in have:
                continue
            p[0].update(type=sign_type(d["type"]), direction="none", **{"from": "full", "reparsed": True})
            r["items"].append(p[0])
            have.add(p[0]["text"].lower())
            added += 1
        merged = {}  # one board read line by line (same box) -> one item, lines joined as the prompt asks
        for it in r["items"]:
            k = tuple(round(v) for v in it["bbox"]) if it.get("reparsed") else id(it)
            if k in merged:
                merged[k]["text"] += " / " + it["text"]
                added -= 1
            else:
                merged[k] = it
        r["items"] = list(merged.values())
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(OUT)
    print(f"reparse: {added} items added")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", choices=["test", "train"], default=None)
    ap.add_argument("--reparse", action="store_true", help="recover dropped items in teacher_all.jsonl, then exit")
    a = ap.parse_args()
    if a.reparse:
        return reparse()
    done = {json.loads(l)["image"] for l in OUT.open()} if OUT.exists() else set()
    todo = [r for r in todo_rows() if r["image"] not in done and a.split in (None, r["split"])]
    todo = sorted(todo, key=lambda r: r["split"] != "test")[:a.limit]  # test split first
    print(f"{len(done)} done, {len(todo)} to label", flush=True)
    model, proc = T.load()
    t_all = time.time()
    with OUT.open("a") as fh:
        for i, r in enumerate(todo):
            t0 = time.time()
            try:
                im = T.open_image(RAW / r["image"])
                items, raw = label(model, proc, im, regions=r.get("regions", []))
            except (torch.OutOfMemoryError, OSError, ValueError, ZeroDivisionError) as e:
                torch.cuda.empty_cache()
                print("skip", r["image"], type(e).__name__, flush=True)
                fh.write(json.dumps({**r, "error": type(e).__name__}) + "\n")
                continue
            r = {k: v for k, v in r.items() if k != "regions"}
            fh.write(json.dumps({**r, "width": im.width, "height": im.height, "items": items, "raw": raw,
                                 "seconds": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0:
                rate = (time.time() - t_all) / (i + 1)
                print(f"{i + 1}/{len(todo)} {rate:.1f} s/img, ETA {rate * (len(todo) - i - 1) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
