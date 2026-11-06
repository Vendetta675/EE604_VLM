"""Synthetic room-number plates and pen-on-paper notes for the all-signs task -> data/sft/synth_paper_{train,test}.jsonl.

Two new sign types (build_sft.TYPES):
  room         door / room plates: "Room 214", "G-12", "Lab 3", "Dr. A. Sharma / Room 305", "कक्ष 204 / Room 204"
  handwritten  a note written with a pen or marker on paper (plain, lined, squared, sticky note): a place name, a room,
               an instruction, usually with a hand-drawn arrow -> "Library | right | handwritten"
Printed signs from signdir.synth_signs are mixed in (about 1 in 3 panels) so the adapter keeps the old types.

Handwriting = Google handwriting fonts in data/fonts_hand (Latin; Kalam / Amita also Devanagari), each glyph line
jittered (baseline wobble, slant, size), ink colour + pressure noise; arrows are wobbly strokes with a drawn head.
Labels follow PROMPT_ALL: "<text> | <direction> | <type>", text as written (languages joined by " / ").

  python -m signdir.synth_paper [--n 2400] [--test 200] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from signdir.build_sft import PROMPT_ALL, SFT, answer
from signdir.pool import ROOT
from signdir.synth import degrade
from signdir.synth_signs import (DEV, LAT, PLACES, backgrounds, compose, direction_panel, fill, sign_panel,
                                 text_block)

FONTS = ROOT / "fonts_hand"
DEV_HAND = ("Kalam", "Amita")          # handwriting fonts that also cover Devanagari
DIRS = {"left": 0.2, "right": 0.2, "straight": 0.2, "up-left": 0.06, "up-right": 0.06, "down": 0.04, "back": 0.02,
        "none": 0.22}

# room plates: (english, hindi or None); {n} = room number, {L} = block letter
ROOMS = [("Room {n}", "कक्ष {n}"), ("{n}", None), ("{L}-{n}", None), ("{L}{n}", None), ("Lab {s}", "प्रयोगशाला {s}"),
         ("Lab {n}", None), ("Classroom {n}", "कक्षा {n}"), ("Lecture Hall {s}", None), ("LH-{s}", None),
         ("Seminar Hall {s}", "सेमिनार हॉल {s}"), ("Conference Room {s}", None), ("Meeting Room {s}", None),
         ("Faculty Room {n}", None), ("Staff Room", "स्टाफ कक्ष"), ("HOD Office", "विभागाध्यक्ष कार्यालय"),
         ("Principal's Office", "प्राचार्य कार्यालय"), ("Office {n}", None), ("Store Room", "भंडार कक्ष"),
         ("Server Room", None), ("Computer Lab {s}", "कंप्यूटर लैब {s}"), ("Ward {s}", "वार्ड {s}"),
         ("OPD {s}", "ओपीडी {s}"), ("Cabin {s}", None), ("Hostel Room {n}", None), ("Flat {L}-{n}", None),
         ("CSE-{n}", None), ("ECE-{n}", None), ("ME-{n}", None), ("Room No. {n}", "कमरा नं. {n}")]
PEOPLE = ["Dr. A. Sharma", "Prof. R. Iyer", "Dr. S. Gupta", "Mr. V. Rao", "Ms. P. Nair", "Dr. K. Singh",
          "Prof. M. Das", "Dr. N. Verma", "Mrs. L. Menon", "Dr. H. Khan"]
# handwritten note texts: place names / instructions written on paper (english, hindi or None)
NOTES = [("Library", "पुस्तकालय"), ("Canteen", "कैंटीन"), ("Washroom", "शौचालय"), ("Toilet", "शौचालय"),
         ("Exit", "निकास"), ("Lift", "लिफ्ट"), ("Stairs", "सीढ़ियाँ"), ("Office", "कार्यालय"), ("Reception", "स्वागत"),
         ("Lab", None), ("Room {n}", "कमरा {n}"), ("Lab {n}", None), ("Hall {s}", None), ("Exam Hall", "परीक्षा हॉल"),
         ("Registration", "पंजीकरण"), ("Workshop", None), ("Meeting Room", None), ("Parking", "पार्किंग"),
         ("Drinking Water", "पीने का पानी"), ("Waiting Area", None), ("Seminar this way", None),
         ("Event this way", None), ("Interview Room", None), ("Help Desk", "सहायता"), ("Pharmacy", "दवाखाना"),
         ("Main Gate", "मुख्य द्वार"), ("Back Door", None), ("Wet Floor", None), ("Do not enter", "प्रवेश निषेध"),
         ("Out of order", "खराब है"), ("Keep door closed", None), ("Silence please", "कृपया शांति रखें"),
         ("Room {n} shifted", None), ("Workshop in {L}-{n}", None), ("Lab {s} moved to Room {n}", None),
         ("Turn here for Library", None), ("Canteen upstairs", None), ("Toilet downstairs", None)]
PAPERS = ["white", "white", "lined", "squared", "sticky", "card"]
INKS = [(25, 40, 140), (20, 20, 25), (20, 20, 25), (170, 25, 30), (20, 110, 50), (60, 60, 160)]


def hand_fonts():
    fs = sorted(str(p) for p in FONTS.glob("*.ttf"))
    if not fs:
        raise SystemExit(f"no handwriting fonts in {FONTS}")
    return fs, [f for f in fs if Path(f).name.startswith(DEV_HAND)]


def fill2(text, rng):
    return fill(text, rng).replace("{s}", str(rng.randint(1, 6)) if rng.random() < 0.6 else rng.choice("ABCD"))


def fill_pair(en, hi, rng):
    """The same number / letter in both languages."""
    if hi is None:
        return fill2(en, rng), None
    a, b = fill2(en + "\t" + hi, rng).split("\t")
    return a, b


# ---- room plates ---------------------------------------------------------------------------------

PLATE = [((245, 245, 245), (25, 25, 25)), ((20, 60, 130), (255, 255, 255)), ((180, 150, 80), (30, 25, 15)),
         ((200, 200, 205), (20, 20, 20)), ((30, 30, 30), (240, 240, 240)), ((0, 100, 70), (255, 255, 255)),
         ((120, 30, 40), (245, 235, 215))]


def room_panel(rng):
    """A door plate: (RGBA, [(text, 'none', 'room', box)])."""
    en, hi = fill_pair(*rng.choice(ROOMS), rng)
    lines = [("en", en)]
    if hi and rng.random() < 0.45:
        lines = [("hi", hi), ("en", en)] if rng.random() < 0.5 else [("en", en), ("hi", hi)]
    if rng.random() < 0.2:
        lines = [("en", rng.choice(PEOPLE))] + lines
    bg, ink = rng.choice(PLATE)
    th = rng.randint(30, 70)
    block = text_block(lines, th, ink, rng)
    pad = int(th * rng.uniform(0.35, 0.8))
    W, H = block.width + 2 * pad, block.height + 2 * pad
    im = Image.new("RGBA", (W, H), bg + (255,))
    d = ImageDraw.Draw(im)
    if rng.random() < 0.5:
        bw = max(2, th // 14)
        d.rounded_rectangle([bw, bw, W - bw - 1, H - bw - 1], radius=th // 5, outline=ink, width=bw)
    if rng.random() < 0.3:                                     # screws
        for x, y in ((pad // 2, pad // 2), (W - pad // 2, pad // 2), (pad // 2, H - pad // 2), (W - pad // 2, H - pad // 2)):
            r = max(2, th // 12)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(150, 150, 150), outline=(90, 90, 90))
    im.alpha_composite(block, (pad, pad))
    if rng.random() < 0.6:                                     # a strip of door around the plate
        m = int(th * rng.uniform(0.6, 2.0))
        door = rng.choice([(120, 80, 50), (160, 120, 80), (200, 200, 195), (90, 60, 40), (60, 90, 120)])
        out = Image.new("RGBA", (W + 2 * m, H + 2 * m), door + (255,))
        out.alpha_composite(im, (m, m))
        return out, [(" / ".join(t for _, t in lines), "none", "room", [m, m, m + W, m + H])]
    return im, [(" / ".join(t for _, t in lines), "none", "room", [0, 0, W, H])]


# ---- handwriting -----------------------------------------------------------------------------------

def hand_line(text, path, size, ink, rng):
    """One handwritten line as RGBA: per-word baseline wobble, size jitter and slant."""
    f = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    words = text.split(" ")
    imgs = []
    for w in words:
        s = int(size * rng.uniform(0.92, 1.08))
        fw = ImageFont.truetype(path, s, layout_engine=ImageFont.Layout.RAQM) if s != size else f
        b = fw.getbbox(w)
        im = Image.new("RGBA", (b[2] - b[0] + 8, int(size * 1.9)), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((4 - b[0], int(size * 0.35)), w, font=fw, fill=ink + (255,))
        imgs.append(im)
    space = int(size * rng.uniform(0.25, 0.45))
    W = sum(i.width for i in imgs) + space * (len(imgs) - 1)
    H = int(size * 2.3)
    out = Image.new("RGBA", (W + 4, H), (0, 0, 0, 0))
    x, drift = 2, 0.0
    for i in imgs:
        drift += rng.uniform(-0.06, 0.06) * size
        out.alpha_composite(i, (x, max(0, min(H - i.height, int(size * 0.2 + drift)))))
        x += i.width + space
    sl = rng.uniform(-0.18, 0.12)                              # slant (shear)
    out = out.transform((out.width + int(abs(sl) * H), H), Image.AFFINE,
                        (1, sl, -sl * H if sl > 0 else 0, 0, 1, 0), Image.BICUBIC)
    bb = out.getbbox()
    return out.crop(bb) if bb else out


def hand_arrow(size, direction, ink, rng, width):
    """A hand-drawn arrow (wobbly shaft + two strokes for the head) in an RGBA square."""
    s = 3 * size
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ang = {"straight": 0, "up-right": 45, "right": 90, "down-right": 135, "down": 180, "down-left": 225,
           "left": 270, "up-left": 315, "back": 180}[direction]
    a = math.radians(ang + rng.uniform(-8, 8))
    c, r = s / 2, 0.42 * s
    n = 12
    bend = rng.uniform(-0.08, 0.08) * s
    pts = []
    for k in range(n + 1):
        t = k / n
        along = -r + 2 * r * t
        side = bend * math.sin(math.pi * t) + rng.uniform(-0.01, 0.01) * s
        pts.append((side, -along))                             # up = -y
    rot = lambda p: (c + p[0] * math.cos(a) - p[1] * math.sin(a), c + p[0] * math.sin(a) + p[1] * math.cos(a))
    w = 3 * width
    d.line([rot(p) for p in pts], fill=ink + (255,), width=w, joint="curve")
    tip = pts[-1]
    hl, spread = rng.uniform(0.25, 0.4) * s, rng.uniform(0.45, 0.75)
    for sgn in (-1, 1):
        wing = (tip[0] + sgn * hl * math.sin(spread) + rng.uniform(-0.02, 0.02) * s, tip[1] + hl * math.cos(spread))
        d.line([rot(tip), rot(wing)], fill=ink + (255,), width=w)
    if direction == "back" and rng.random() < 0.5:             # U-turn mark: a hook at the tail
        d.arc([c - r / 2, c + r / 2, c + r / 2, c + 1.3 * r], 0, 180, fill=ink + (255,), width=w)
    return im.resize((size, size), Image.LANCZOS)


def paper(W, H, kind, rng):
    base = {"white": (246, 246, 240), "lined": (248, 248, 244), "squared": (247, 247, 243), "card": (235, 228, 205),
            "sticky": rng.choice([(252, 240, 120), (255, 200, 120), (170, 230, 250), (250, 180, 210),
                                  (190, 240, 160)])}[kind]
    a = np.full((H, W, 3), base, np.float32)
    a += np.random.default_rng(rng.randint(0, 2**31)).normal(0, 3, (H, W, 1))         # paper grain
    g = np.linspace(rng.uniform(-12, 0), rng.uniform(0, 8), W)[None, :, None]           # uneven light
    a = np.clip(a + g, 0, 255)
    im = Image.fromarray(a.astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im)
    step = max(18, H // rng.randint(10, 18))
    if kind == "lined":
        for y in range(step * 2, H, step):
            d.line([(0, y), (W, y)], fill=(150, 180, 220, 255), width=1)
        if rng.random() < 0.6:
            d.line([(int(W * 0.1), 0), (int(W * 0.1), H)], fill=(220, 120, 120, 255), width=1)
    elif kind == "squared":
        for y in range(step, H, step):
            d.line([(0, y), (W, y)], fill=(185, 200, 225, 255), width=1)
        for x in range(step, W, step):
            d.line([(x, 0), (x, H)], fill=(185, 200, 225, 255), width=1)
    if rng.random() < 0.4:                                     # tape on the top corners / edge
        tc = (230, 225, 200, 170)
        for x in ((0, int(W * 0.12)), (W - int(W * 0.12), W)) if rng.random() < 0.6 else ((int(W * 0.4), int(W * 0.6)),):
            d.rectangle([x[0], 0, x[1], int(H * 0.06) + 6], fill=tc)
    if rng.random() < 0.3:                                     # a fold line
        x = int(W * rng.uniform(0.3, 0.7))
        d.line([(x, 0), (x, H)], fill=(205, 205, 200, 255), width=2)
    return im


def note_rows(rng):
    """1-3 rows of (lines [(lang, text)], direction)."""
    k = rng.choices([1, 2, 3], [0.6, 0.3, 0.1])[0]
    rows = []
    for _ in range(k):
        if rng.random() < 0.25:
            en, hi, _, _ = rng.choice(PLACES)
            en, hi = fill_pair(en, hi, rng)
        else:
            en, hi = fill_pair(*rng.choice(NOTES), rng)
        lines = [("en", en)]
        if hi and rng.random() < 0.3:
            lines = [("hi", hi)] if rng.random() < 0.5 else [("en", en), ("hi", hi)]
        rows.append((lines, rng.choices(list(DIRS), list(DIRS.values()))[0]))
    return rows


def note_panel(rng, fonts, dev_fonts):
    """A paper note: (RGBA, items), one item per written row ("Library | right | handwritten")."""
    ink = rng.choice(INKS)
    if rng.random() < 0.25:                                    # marker: thick, darker
        ink = rng.choice([(15, 15, 20), (20, 30, 120), (150, 20, 25)])
        width = rng.randint(4, 7)
    else:
        width = rng.randint(2, 4)
    path_lat = rng.choice(fonts)
    path_dev = rng.choice(dev_fonts) if dev_fonts else None
    th = rng.randint(40, 90)
    rows = []
    for lines, d in note_rows(rng):
        if path_dev is None:
            lines = [l for l in lines if l[0] == "en"] or [("en", "Room 1")]
        ims = [hand_line(t, path_dev if lang == "hi" else path_lat, th, ink, rng) for lang, t in lines]
        rows.append((lines, d, ims))
    asz = int(th * rng.uniform(1.6, 2.6))
    pad = int(th * rng.uniform(0.6, 1.2))
    gap = int(th * 0.5)
    blocks = []
    for lines, d, ims in rows:
        bw = max(i.width for i in ims)
        bh = sum(i.height for i in ims) + int(th * 0.2) * (len(ims) - 1)
        blocks.append((bw, bh))
    has_arrow = [d != "none" for _, d, _ in rows]
    layout = rng.choice(["side", "below"])
    W = max(bw + (asz + gap if a and layout == "side" else 0) for (bw, _), a in zip(blocks, has_arrow))
    W = max(W, asz) + 2 * pad
    rh = [max(bh, asz) if a and layout == "side" else bh + (asz + gap // 2 if a else 0)
          for (_, bh), a in zip(blocks, has_arrow)]
    H = sum(rh) + gap * (len(rows) - 1) + 2 * pad
    kind = rng.choice(PAPERS)
    if kind == "sticky":
        W, H = max(W, H), max(W, H)
    im = paper(W, H, kind, rng)
    items, y = [], pad + (H - 2 * pad - sum(rh) - gap * (len(rows) - 1)) // 2
    for (lines, d, ims), (bw, bh), h in zip(rows, blocks, rh):
        x0 = pad + rng.randint(0, max(0, W - 2 * pad - bw - (asz + gap if d != "none" and layout == "side" else 0)))
        if d != "none" and layout == "side":
            left = "left" in d or (d in ("straight", "down", "back") and rng.random() < 0.5)
            ax = x0 if left else x0 + bw + gap
            tx = x0 + asz + gap if left else x0
            ay = y + (h - asz) // 2
            ty = y + (h - bh) // 2
        elif d != "none":
            tx, ty = x0, y
            ax, ay = x0 + rng.randint(0, max(0, bw - asz)), y + bh + gap // 2
        else:
            tx, ty, ax, ay = x0, y, None, None
        yy = ty
        for i in ims:
            im.alpha_composite(i, (tx, yy))
            yy += i.height + int(th * 0.2)
        x1, y1, x2, y2 = tx, ty, tx + bw, ty + bh
        if ax is not None:
            im.alpha_composite(hand_arrow(asz, d, ink, rng, width), (ax, ay))
            x1, y1, x2, y2 = min(x1, ax), min(y1, ay), max(x2, ax + asz), max(y2, ay + asz)
        if rng.random() < 0.15:                                # underline
            ImageDraw.Draw(im).line([(tx, ty + bh + 2), (tx + bw, ty + bh + 4)], fill=ink + (255,), width=width)
        items.append((" / ".join(t for _, t in lines), d, "handwritten", [x1 - 4, y1 - 4, x2 + 4, y2 + 4]))
        y += h + gap
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.6)))
    return im, items


# ---- drawn symbols (as in the RealSense photos: outline arrows, circled arrows, triangles, boxed words) -------

BOXED = ["EXIT", "NO ENTRY", "CLOSED", "STOP", "DANGER", "KEEP LEFT", "KEEP RIGHT", "PUSH", "PULL", "OPEN", "LAB",
         "OFFICE", "TOILET", "LIFT", "STAIRS", "WAY OUT", "ENTRY", "NO EXIT", "WET FLOOR", "SILENCE", "OUT OF ORDER",
         "DO NOT ENTER", "ROOM {n}", "LAB {n}", "{L}-{n}"]
CIRCLED = {"straight": "Go straight", "left": "Turn left", "right": "Turn right"}
ANG = {"straight": 0, "up-right": 45, "right": 90, "down-right": 135, "down": 180, "down-left": 225, "left": 270,
       "up-left": 315}


def wobble(pts, amp, rng, n=6):
    """Points along the closed polygon with hand jitter (n per side)."""
    out = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        for k in range(n):
            t = k / n
            out.append((x0 + (x1 - x0) * t + rng.uniform(-amp, amp), y0 + (y1 - y0) * t + rng.uniform(-amp, amp)))
    return out


def stroke(d, pts, ink, w, rng, closed=True):
    """A pen outline: sometimes traced twice (a second, slightly offset pass) and left a little open."""
    p = pts + pts[:1] if closed else pts
    if closed and rng.random() < 0.3:
        p = p[:-max(1, len(p) // 12)]                       # the pen stops short of the start
    d.line(p, fill=ink + (255,), width=w, joint="curve")
    if rng.random() < 0.3:
        o = rng.uniform(-1.5, 1.5) * w
        d.line([(x + o, y + o) for x, y in p], fill=ink + (200,), width=max(1, w // 2), joint="curve")


def block_arrow_pts(c, r, double=False, rng=None):
    """Outline of a block arrow pointing up (y down), centred at c, half-length r."""
    sw, hw, hh = r * rng.uniform(0.18, 0.3), r * rng.uniform(0.45, 0.65), r * rng.uniform(0.4, 0.6)
    top = [(0, -r), (hw, -r + hh), (sw, -r + hh)]
    if double:
        bot = [(sw, r - hh), (hw, r - hh), (0, r), (-hw, r - hh), (-sw, r - hh)]
    else:
        bot = [(sw, r), (-sw, r)]
    pts = top + bot + [(-sw, -r + hh), (-hw, -r + hh)]
    return [(c[0] + x, c[1] + y) for x, y in pts]


def rotate(pts, c, deg):
    a = math.radians(deg)
    return [(c[0] + (x - c[0]) * math.cos(a) - (y - c[1]) * math.sin(a),
             c[1] + (x - c[0]) * math.sin(a) + (y - c[1]) * math.cos(a)) for x, y in pts]


def symbol(kind, size, ink, rng, width, fonts):
    """RGBA square (or wider, for words) with one drawn symbol; returns (image, text, direction)."""
    s = 2 * size
    if kind == "word":
        word = fill2(rng.choice(BOXED), rng)
        line = hand_line(word, rng.choice(fonts), int(size * rng.uniform(0.35, 0.5)), ink, rng)
        pad = int(size * 0.12)
        im = Image.new("RGBA", (line.width + 2 * pad, line.height + 2 * pad), (0, 0, 0, 0))
        im.alpha_composite(line, (pad, pad))
        if rng.random() < 0.8:
            W, H = im.size
            stroke(ImageDraw.Draw(im), wobble([(2, 2), (W - 3, 2), (W - 3, H - 3), (2, H - 3)], width, rng),
                   ink, width, rng)
        return im, word, "none"
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c, r, w = (s / 2, s / 2), 0.42 * s, 2 * width
    if kind == "triangle":
        pts = [(c[0], c[1] - r), (c[0] + r * 0.95, c[1] + r * 0.75), (c[0] - r * 0.95, c[1] + r * 0.75)]
        stroke(d, wobble(pts, s * 0.01, rng), ink, w, rng)
        if rng.random() < 0.4:                              # "!" inside
            d.line([(c[0], c[1] - r * 0.3), (c[0] + rng.uniform(-3, 3), c[1] + r * 0.3)], fill=ink + (255,), width=w)
            d.ellipse([c[0] - w, c[1] + r * 0.45 - w, c[0] + w, c[1] + r * 0.45 + w], fill=ink + (255,))
        return im.resize((size, size), Image.LANCZOS), "Warning", "none"
    double = kind == "double"
    direction = rng.choice(["left", "right"]) if double else (
        rng.choice(list(CIRCLED)) if kind == "circled" else
        rng.choices(list(ANG), [0.22, 0.08, 0.22, 0.02, 0.08, 0.02, 0.22, 0.08])[0])
    rr = r * (0.62 if kind == "circled" else 1.0)
    pts = rotate(block_arrow_pts(c, rr, double, rng), c, ANG[direction] + rng.uniform(-6, 6))
    if kind == "circled" or rng.random() < 0.7:
        stroke(d, wobble(pts, s * 0.008, rng, n=3), ink, w, rng)
    else:                                                   # filled in
        d.polygon(wobble(pts, s * 0.006, rng, n=3), fill=ink + (255,))
    if kind == "circled":
        e = [(c[0] + r * math.cos(t) * rng.uniform(0.97, 1.03), c[1] + r * 1.08 * math.sin(t))
             for t in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
        stroke(d, e, ink, w, rng)
    im = im.resize((size, size), Image.LANCZOS)
    if double:
        return im, "Both ways", "none"
    if kind == "circled":
        return im, CIRCLED[direction], direction
    return im, "Arrow", direction


def symbol_panel(rng, fonts, page=None):
    """A notebook page with 1-3 drawn symbols / boxed words (the RealSense photos): (RGBA, items). page = (W, H)."""
    ink = rng.choice(INKS)
    width = rng.randint(2, 4) if rng.random() < 0.75 else rng.randint(4, 7)
    k = rng.choices([1, 2, 3], [0.55, 0.35, 0.1])[0]
    size = int(page[1] * rng.uniform(0.25, 0.5)) if page else rng.randint(140, 320)
    syms = []
    for _ in range(k):
        kind = rng.choices(["arrow", "circled", "triangle", "word", "double"], [0.4, 0.15, 0.1, 0.27, 0.08])[0]
        syms.append(symbol(kind, size, ink, rng, width, fonts))
    pad, gap = int(size * 0.4), int(size * 0.3)
    W = max(s[0].width for s in syms) + 2 * pad
    H = sum(s[0].height for s in syms) + gap * (k - 1) + 2 * pad
    if page:
        W, H = max(W, page[0]), max(H, page[1])
    im = paper(W, H, rng.choice(["lined", "lined", "white", "squared"]), rng)
    items, y = [], (H - sum(s[0].height for s in syms) - gap * (k - 1)) // 2
    for sim, text, d in syms:
        x = rng.randint(pad // 2, max(pad // 2, W - sim.width - pad // 2))
        im.alpha_composite(sim, (x, y))
        bb = sim.getbbox() or (0, 0, sim.width, sim.height)
        items.append((text, d, "handwritten", [x + bb[0] - 4, y + bb[1] - 4, x + bb[2] + 4, y + bb[3] + 4]))
        y += sim.height + gap
    return im.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.7))), items


def closeup(bg, panel, rng):
    """The page photographed from close: it covers most of the frame, tilted, maybe running off an edge."""
    import cv2
    W, H = rng.choice([(1280, 720), (1280, 960), (1024, 768)])
    bg = np.asarray(bg.convert("RGB").resize((W, H)), np.float32)
    sign, its = panel
    sw, sh = sign.size
    f = rng.uniform(0.75, 1.3) * min(W / sw, H / sh)
    cx, cy = W / 2 + rng.uniform(-0.15, 0.15) * W, H / 2 + rng.uniform(-0.15, 0.15) * H
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    dst = np.float32([[cx + (x - sw / 2) * f, cy + (y - sh / 2) * f] for x, y in src])
    dst = np.float32(rotate([tuple(p) for p in dst], (cx, cy), rng.uniform(-12, 12)))   # small: labels are in
                                                                                       # image directions
    j = 0.08 * min(W, H)
    dst += np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(np.asarray(sign), M, (W, H), flags=cv2.INTER_AREA)
    a = warped[..., 3:4].astype(np.float32) / 255
    shade = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.85, 1.1), W)[None, :, None]   # uneven light
    out = bg * (1 - a) + warped[..., :3] * a * shade
    items = []
    for text, d, typ, (x1, y1, x2, y2) in its:
        p = cv2.perspectiveTransform(np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]), M)[0]
        b = [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
        area = (b[2] - b[0]) * (b[3] - b[1])
        c = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
        if area <= 0 or (c[2] - c[0]) * (c[3] - c[1]) < 0.6 * area:   # mostly off the frame: drop the item,
            if (c[2] - c[0]) * (c[3] - c[1]) > 0.05 * area:           # and if partly visible, drop the image
                return None, None
            continue
        items.append({"text": text, "direction": d, "type": typ, "bbox": c})
    return out, items


# ---- scenes -----------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2400, help="train images")
    ap.add_argument("--test", type=int, default=200, help="held-out images (own backgrounds and seed)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--closeup", type=float, default=0.3, help="share of images that are a page filling the frame")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    fonts, dev_fonts = hand_fonts()
    bgs = backgrounds()
    rng.shuffle(bgs)
    cut = len(bgs) // 10
    print(f"{len(bgs)} backgrounds, {len(fonts)} handwriting fonts ({len(dev_fonts)} with Devanagari)", flush=True)
    (SFT / "images").mkdir(parents=True, exist_ok=True)
    for split, n, pool in (("test", a.test, bgs[:cut]), ("train", a.n, bgs[cut:])):
        rows, count = [], {}
        for i in range(n):
            img = None
            if rng.random() < a.closeup:                       # a notebook page filling the frame
                page = (rng.randint(900, 1400), rng.randint(700, 1100))
                panel = symbol_panel(rng, fonts, page) if rng.random() < 0.75 else note_panel(rng, fonts, dev_fonts)
                img, items = closeup(Image.open(rng.choice(pool)), panel, rng)
            if img is None:
                panels = []
                for _ in range(rng.choices([1, 2, 3], [0.5, 0.35, 0.15])[0]):
                    kind = rng.choices(["note", "symbol", "room", "sign", "direction"],
                                       [0.27, 0.18, 0.27, 0.14, 0.14])[0]
                    panels.append(note_panel(rng, fonts, dev_fonts) if kind == "note" else symbol_panel(rng, fonts)
                                  if kind == "symbol" else room_panel(rng) if kind == "room" else sign_panel(rng)
                                  if kind == "sign" else direction_panel(rng))
                if rng.random() < 0.04:
                    panels = []
                img, items = compose(Image.open(rng.choice(pool)).convert("RGB"), panels, rng)
            img = degrade(img, rng)
            name = f"synthpaper__{split}_{a.seed}_{i:05d}.jpg"
            img.save(SFT / "images" / name, quality=rng.randint(70, 95))
            for it in items:
                count[it["type"]] = count.get(it["type"], 0) + 1
            rows.append({"image": name, "source": "synthetic_paper", "reviewed": False,
                         "conversations": [{"from": "human", "value": PROMPT_ALL},
                                           {"from": "gpt", "value": answer(items, *img.size)}]})
        p = SFT / f"synth_paper_{split}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{split}: {len(rows)} images -> {p}; items per type {count}", flush=True)


if __name__ == "__main__":
    main()
