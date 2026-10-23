"""Synthetic direction signs with exact arrow labels -> data/sft/synth.jsonl (+ data/sft/images/synth__*.jpg).

The teacher almost never labels an arrow "straight" (it reads true straight arrows as up-left / up-right / right), so
the run-2 adapter never says straight. These images add exactly labelled arrows, weighted towards straight and the
diagonals: a panel (road green / blue / white / airport yellow / ...) with 1-4 rows of "arrow + destination", the
destinations drawn from the real train labels (Latin and CJK), pasted with a perspective warp onto a train image
without a direction sign (so nothing in the background is an unlabelled arrow). Each item's box covers its text plus
its arrow, as in the teacher labels.

  python -m signdir.synth [--n 800] [--seed 0]       then  python -m signdir.build_sft --long_side 1536 --synth
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from signdir.build_sft import PROMPT, SFT, answer

# arrow angle in degrees clockwise from "up"
ANGLE = {"straight": 0, "up-right": 45, "right": 90, "down-right": 135, "down": 180, "down-left": 225, "left": 270,
         "up-left": 315}
WEIGHT = {"straight": 0.32, "up-left": 0.12, "up-right": 0.12, "left": 0.14, "right": 0.14, "down": 0.05,
          "down-left": 0.03, "down-right": 0.03, "none": 0.05}
# (panel, ink) colours
STYLES = [((0, 112, 60), (255, 255, 255)), ((20, 70, 160), (255, 255, 255)), ((245, 245, 245), (20, 20, 20)),
          ((250, 200, 0), (15, 15, 15)), ((25, 25, 25), (250, 200, 0)), ((110, 60, 30), (255, 255, 255)),
          ((60, 60, 65), (255, 255, 255)), ((0, 90, 90), (255, 255, 255)), ((200, 30, 30), (255, 255, 255))]
FONTS = {"latin": ["/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
                   "/usr/share/fonts/truetype/noto/NotoSans-SemiBold.ttf",
                   "/usr/share/fonts/truetype/noto/NotoSans-Medium.ttf",
                   "/usr/share/fonts/truetype/noto/NotoSans-CondensedBold.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"],
         "cjk": ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"]}
CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")
OTHER = re.compile(r"[^\u0000-ɏ -⯿　-〿぀-ヿ一-鿿가-힯＀-￯]")


def texts():
    """Destinations of the real train labels that the fonts can draw."""
    out = set()
    for r in map(json.loads, (SFT / "train.jsonl").open()):
        if r["source"] == "synthetic":
            continue
        for t in re.findall(r"<ref>(.*?) \| [a-z-]+</ref>", r["conversations"][1]["value"]):
            if 1 <= len(t) <= 28 and not OTHER.search(t):
                out.add(t)
    return sorted(out)


def backgrounds():
    return [r["image"] for r in map(json.loads, (SFT / "train.jsonl").open())
            if r["source"] != "synthetic" and r["conversations"][1]["value"] == "<box>none</box>"]


def arrow(size, angle, ink, rng):
    """RGBA square of side size with an arrow pointing at angle (degrees clockwise from up)."""
    s = 4 * size  # draw large, then downsample (antialiasing)
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c, kind = s / 2, rng.choice(["block", "block", "line", "boxed"])
    fill = ink
    if kind == "boxed":  # arrow knocked out of a filled square, as on airport / metro signs
        d.rounded_rectangle([0, 0, s - 1, s - 1], radius=s // 8, fill=ink + (255,))
        fill = (255, 255, 255) if sum(ink) < 300 else (20, 20, 20)
        r = 0.36 * s
    else:
        r = 0.48 * s
    head_w, head_h, shaft = rng.uniform(0.75, 1.0) * r, rng.uniform(0.6, 0.9) * r, rng.uniform(0.22, 0.38) * r
    if kind == "line":
        w = max(2, int(shaft * 0.6))
        pts = [(0, r), (0, -r + w / 2)]
        wing = [(-head_w * 0.8, -r + head_h), (0, -r), (head_w * 0.8, -r + head_h)]
    else:
        pts = [(-shaft, r), (-shaft, -r + head_h), (-head_w, -r + head_h), (0, -r), (head_w, -r + head_h),
               (shaft, -r + head_h), (shaft, r)]
    a = math.radians(angle)
    rot = lambda p: (c + p[0] * math.cos(a) - p[1] * math.sin(a), c + p[0] * math.sin(a) + p[1] * math.cos(a))
    if kind == "line":
        d.line([rot(p) for p in pts], fill=fill + (255,), width=w)
        d.line([rot(p) for p in wing], fill=fill + (255,), width=w, joint="curve")
    else:
        d.polygon([rot(p) for p in pts], fill=fill + (255,))
    return im.resize((size, size), Image.LANCZOS)


def panel(items, rng):
    """Sign panel (RGBA) and the item boxes on it. items: [(text, direction)]."""
    bg, ink = rng.choice(STYLES)
    th = rng.randint(40, 72)  # text height
    pad, gap = int(th * rng.uniform(0.4, 0.8)), int(th * rng.uniform(0.35, 0.8))
    rows = []
    for text, d in items:
        font = ImageFont.truetype(rng.choice(FONTS["cjk" if CJK.search(text) else "latin"]), th)
        l, t, r, b = font.getbbox(text)
        size = int(th * rng.uniform(1.0, 1.6))
        side = None if d == "none" else ("left" if "left" in d else "right" if "right" in d
                                         else rng.choice(["left", "right"]))
        if side and rng.random() < 0.15:
            side = "left" if side == "right" else "right"  # arrow on the "wrong" side happens on real signs
        rows.append((text, d, font, (l, t, r, b), size, side))
    row_h = max(max(rw[4], rw[3][3] - rw[3][1]) for rw in rows)
    width = max(rw[3][2] - rw[3][0] + (rw[4] + gap if rw[5] else 0) for rw in rows) + 2 * pad
    height = len(rows) * row_h + (len(rows) - 1) * gap + 2 * pad
    im = Image.new("RGBA", (width, height), bg + (255,))
    d = ImageDraw.Draw(im)
    if rng.random() < 0.6:  # inset border
        bw = max(2, th // 12)
        d.rounded_rectangle([bw, bw, width - bw - 1, height - bw - 1], radius=th // 4, outline=ink, width=bw)
    boxes = []
    align = rng.choice(["edge", "edge", "center"])
    for k, (text, dirn, font, (l, t, r, b), size, side) in enumerate(rows):
        y0 = pad + k * (row_h + gap)
        tw, tht = r - l, b - t
        if side == "left":
            ax, tx = pad, pad + size + gap
        elif side == "right":
            ax = width - pad - size
            tx = ax - gap - tw if align == "edge" else pad
        else:
            ax, tx = None, (width - tw) // 2 if align == "center" else pad
        ty = y0 + (row_h - tht) // 2 - t
        d.text((tx, ty), text, font=font, fill=ink)
        x1, x2 = tx + l, tx + r
        if ax is not None:
            ay = y0 + (row_h - size) // 2
            im.alpha_composite(arrow(size, ANGLE[dirn] + rng.uniform(-4, 4), ink, rng), (ax, ay))
            x1, x2 = min(x1, ax), max(x2, ax + size)
        y1, y2 = y0 + (row_h - max(size if ax is not None else 0, tht)) // 2, y0 + (row_h + max(size if ax is not None else 0, tht)) // 2
        boxes.append([x1 - 3, y1 - 3, x2 + 3, y2 + 3])
    return im, boxes


def place(bgim, sign, boxes, rng):
    """Warp the sign into the background; returns the image and the warped (axis-aligned) item boxes."""
    s = rng.randint(1024, 1536) / max(bgim.size)  # some backgrounds are tiny (US road signs ~0.05 MP)
    bgim = bgim.resize((round(bgim.width * s), round(bgim.height * s)), Image.BICUBIC)
    W, H = bgim.size
    sw, sh = sign.size
    s = rng.uniform(0.25, 0.75) * W / sw
    s = min(s, 0.85 * H / sh)
    w, h = sw * s, sh * s
    x0, y0 = rng.uniform(0, W - w), rng.uniform(0, H - h)
    j = 0.08 * min(w, h)
    dst = np.float32([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]])
    dst += np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(np.asarray(sign), M, (W, H), flags=cv2.INTER_AREA)
    a = warped[..., 3:4].astype(np.float32) / 255
    out = np.asarray(bgim.convert("RGB")).astype(np.float32) * (1 - a) + warped[..., :3] * a
    out_boxes = []
    for x1, y1, x2, y2 in boxes:
        p = cv2.perspectiveTransform(np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]), M)[0]
        out_boxes.append([max(0, p[:, 0].min()), max(0, p[:, 1].min()), min(W, p[:, 0].max()), min(H, p[:, 1].max())])
    return out, out_boxes


def degrade(img, rng):
    img = img * rng.uniform(0.7, 1.15) + rng.uniform(-20, 20)  # exposure
    if rng.random() < 0.4:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:
        img = img + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(2, 8), img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pool, bgs = texts(), backgrounds()
    dirs, w = list(WEIGHT), list(WEIGHT.values())
    print(f"{len(pool)} destinations, {len(bgs)} backgrounds", flush=True)
    rows, count = [], {d: 0 for d in dirs}
    for i in range(a.n):
        k = rng.choices([1, 2, 3, 4], [0.3, 0.35, 0.25, 0.1])[0]
        items = [(rng.choice(pool), rng.choices(dirs, w)[0]) for _ in range(k)]
        if all(d == "none" for _, d in items):
            items[0] = (items[0][0], "straight")
        sign, boxes = panel(items, rng)
        bg = Image.open(SFT / "images" / rng.choice(bgs)).convert("RGB")
        img, boxes = place(bg, sign, boxes, rng)
        img = degrade(img, rng)
        name = f"synth__{a.seed}_{i:05d}.jpg"
        img.save(SFT / "images" / name, quality=rng.randint(70, 95))
        labelled = [{"text": t, "direction": d, "bbox": b} for (t, d), b in zip(items, boxes)]
        for _, d in items:
            count[d] += 1
        rows.append({"image": name, "source": "synthetic", "reviewed": False,
                     "conversations": [{"from": "human", "value": PROMPT},
                                       {"from": "gpt", "value": answer(labelled, *img.size)}]})
    (SFT / "synth.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} images -> {SFT / 'synth.jsonl'}; items per direction: {count}")


if __name__ == "__main__":
    main()
