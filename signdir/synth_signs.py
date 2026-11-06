"""Synthetic Indian signs for the all-signs task -> data/sft/synth_signs_{train,test}.jsonl (+ images).

Public data has almost no Indian indoor signboards, so they are rendered: railway / hospital / campus / mall / office
signs in English, Hindi (Devanagari, shaped by raqm) or both, of every type in build_sft.TYPES, with simple drawn
pictograms (no smoking, high voltage, toilets, wheelchair ...), plus real Indian traffic-sign photos from the 85-class
Kaggle set (sarangdilipjodh/indian-traffic-signs-prediction85-classes; class -> meaning + type in TRAFFIC). Signs
are pasted with a perspective warp onto MIT Indoor Scenes photos of rooms that rarely show signs (BACKGROUND_ROOMS),
so nothing in the background is an unlabelled sign.

Labels follow build_sft.PROMPT_ALL: one item per sign ("No Smoking / धूम्रपान निषेध | none | prohibition"); a
direction panel gives one item per row ("Platform 2 / प्लेटफार्म 2 | left | direction"). Text is what is printed,
top to bottom, languages joined by " / "; a pictogram-only sign (traffic) is labelled with its meaning.

  python -m signdir.synth_signs [--n 2400] [--test 200] [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from signdir.build_sft import PROMPT_ALL, SFT, answer
from signdir.pool import RAW
from signdir.synth import ANGLE, arrow, degrade

DEV = ["/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
       "/usr/share/fonts/truetype/noto/NotoSansDevanagari-SemiBold.ttf",
       "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Medium.ttf"]
LAT = ["/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", "/usr/share/fonts/truetype/noto/NotoSans-SemiBold.ttf",
       "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
       "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
       "/usr/share/fonts/truetype/noto/NotoSans-CondensedBold.ttf"]

# (english, hindi, type, pictogram); {n} is replaced by a number
SIGNS = [
    # facility
    ("Toilet", "शौचालय", "facility", "toilet"), ("Gents Toilet", "पुरुष शौचालय", "facility", "man"),
    ("Ladies Toilet", "महिला शौचालय", "facility", "woman"), ("Drinking Water", "पेयजल", "facility", None),
    ("Lift", "लिफ्ट", "facility", "lift"), ("Waiting Room", "प्रतीक्षालय", "facility", None),
    ("Waiting Hall", "प्रतीक्षा कक्ष", "facility", None), ("Ticket Counter", "टिकट काउंटर", "facility", None),
    ("Reservation Office", "आरक्षण कार्यालय", "facility", None), ("Enquiry", "पूछताछ", "facility", None),
    ("Pharmacy", "दवा की दुकान", "facility", None), ("Emergency", "आपातकालीन", "facility", None),
    ("Reception", "स्वागत कक्ष", "facility", None), ("Canteen", "कैंटीन", "facility", None),
    ("Cloak Room", "अमानती सामान घर", "facility", None), ("ATM", "एटीएम", "facility", None),
    ("Parking", "पार्किंग", "facility", None), ("Exit", "निकास", "facility", None),
    ("Entry", "प्रवेश", "facility", None), ("Emergency Exit", "आपातकालीन निकास", "facility", "run"),
    ("Staircase", "सीढ़ियाँ", "facility", None), ("Escalator", "एस्केलेटर", "facility", None),
    ("First Aid", "प्राथमिक चिकित्सा", "facility", "cross"), ("Help Desk", "सहायता केंद्र", "facility", None),
    ("Wheelchair Accessible", "व्हीलचेयर सुलभ", "facility", "wheelchair"),
    ("Prayer Room", "प्रार्थना कक्ष", "facility", None), ("Food Court", "फूड कोर्ट", "facility", None),
    ("Billing Counter", "बिलिंग काउंटर", "facility", None), ("Fire Extinguisher", "अग्निशामक", "facility", None),
    ("Baby Care Room", "शिशु देखभाल कक्ष", "facility", None),
    # information
    ("OPD", "ओपीडी", "information", None), ("Casualty", "आकस्मिक विभाग", "information", None),
    ("Blood Bank", "ब्लड बैंक", "information", None), ("X-Ray", "एक्स-रे", "information", None),
    ("Laboratory", "प्रयोगशाला", "information", None), ("Cardiology", "हृदय रोग विभाग", "information", None),
    ("Orthopaedics", "हड्डी रोग विभाग", "information", None), ("Paediatrics", "बाल रोग विभाग", "information", None),
    ("Ward No. {n}", "वार्ड संख्या {n}", "information", None), ("Room No. {n}", "कमरा संख्या {n}", "information", None),
    ("Platform No. {n}", "प्लेटफार्म संख्या {n}", "information", None), ("Ground Floor", "भूतल", "information", None),
    ("First Floor", "पहली मंजिल", "information", None), ("Second Floor", "दूसरी मंजिल", "information", None),
    ("Library", "पुस्तकालय", "information", None), ("Principal's Office", "प्रधानाचार्य कार्यालय", "information", None),
    ("Staff Room", "स्टाफ कक्ष", "information", None), ("Computer Lab", "कंप्यूटर लैब", "information", None),
    ("Admin Block", "प्रशासनिक भवन", "information", None), ("Accounts Section", "लेखा अनुभाग", "information", None),
    ("Office Hours 10 AM - 5 PM", "कार्यालय समय 10 बजे - 5 बजे", "information", None),
    ("Welcome", "स्वागत है", "information", None), ("Gate No. {n}", "गेट नंबर {n}", "information", None),
    ("CCTV Surveillance", "सीसीटीवी निगरानी", "information", None), ("Coach H{n}", "कोच एच{n}", "information", None),
    # warning
    ("Caution: Wet Floor", "सावधान: फर्श गीला है", "warning", "exclaim"),
    ("Danger: High Voltage", "खतरा: उच्च वोल्टेज", "warning", "bolt"),
    ("Mind the Gap", "अंतराल का ध्यान रखें", "warning", "exclaim"),
    ("Mind Your Step", "कदम संभाल कर रखें", "warning", "exclaim"),
    ("Watch Your Head", "सिर का ध्यान रखें", "warning", "exclaim"),
    ("Caution: Slippery Floor", "सावधान: फिसलन", "warning", "exclaim"), ("Danger", "खतरा", "warning", "bolt"),
    ("Caution: Work in Progress", "सावधान: कार्य प्रगति पर है", "warning", "exclaim"),
    ("Beware of Dogs", "कुत्तों से सावधान", "warning", "exclaim"),
    # prohibition
    ("No Smoking", "धूम्रपान निषेध", "prohibition", "cigarette"),
    ("No Entry", "प्रवेश निषेध", "prohibition", "noentry"), ("No Spitting", "थूकना मना है", "prohibition", None),
    ("No Parking", "पार्किंग निषेध", "prohibition", "P"), ("Do Not Touch", "छूना मना है", "prohibition", None),
    ("Mobile Phones Not Allowed", "मोबाइल फोन वर्जित है", "prohibition", "phone"),
    ("Photography Prohibited", "फोटोग्राफी निषेध", "prohibition", "camera"),
    ("Footwear Not Allowed", "जूते चप्पल वर्जित", "prohibition", None),
    ("Authorised Persons Only", "केवल अधिकृत व्यक्ति", "prohibition", None),
    ("Do Not Use Lift in Case of Fire", "आग लगने पर लिफ्ट का उपयोग न करें", "prohibition", None),
    ("No Littering", "कचरा न फैलाएं", "prohibition", None),
    ("Outside Food Not Allowed", "बाहर का खाना वर्जित है", "prohibition", None),
    # mandatory
    ("Wear Mask", "मास्क पहनें", "mandatory", "person"), ("Keep Left", "बाएं चलें", "mandatory", "keepleft"),
    ("Queue Here", "कतार में लगें", "mandatory", "person"), ("Maintain Silence", "शांति बनाए रखें", "mandatory", None),
    ("Switch Off Mobile Phones", "मोबाइल फोन बंद रखें", "mandatory", None),
    ("Wear Helmet", "हेलमेट पहनें", "mandatory", "person"), ("Remove Footwear", "जूते उतारें", "mandatory", None),
    ("Use Dustbin", "कूड़ेदान का प्रयोग करें", "mandatory", None), ("Wash Your Hands", "हाथ धोएं", "mandatory", None),
    ("Show Your Ticket", "टिकट दिखाएं", "mandatory", None), ("Keep Door Closed", "दरवाजा बंद रखें", "mandatory", None),
]
# places that appear on direction panels (with an arrow per row)
PLACES = [s for s in SIGNS if s[2] in ("facility", "information") and "{n}" not in s[0] and len(s[0]) < 20] + [
    ("Platform No. {n}", "प्लेटफार्म संख्या {n}", "information", None), ("Ward {n}", "वार्ड {n}", "information", None),
    ("Block {L}", "ब्लॉक {L}", "information", None), ("Way Out", "बाहर जाने का रास्ता", "facility", None),
    ("Metro", "मेट्रो", "facility", None), ("Bus Stand", "बस अड्डा", "facility", None),
    ("Departure", "प्रस्थान", "facility", None), ("Arrival", "आगमन", "facility", None)]
DIRS = {"left": 0.24, "right": 0.24, "straight": 0.24, "up-left": 0.08, "up-right": 0.08, "down": 0.04,
        "down-left": 0.04, "down-right": 0.04}
# panel styles per type: (background, ink)
STYLE = {"facility": [((20, 70, 160), (255, 255, 255)), ((0, 112, 60), (255, 255, 255)), ((245, 245, 245), (20, 20, 20))],
         "information": [((245, 245, 245), (20, 20, 20)), ((20, 70, 160), (255, 255, 255)), ((110, 60, 30), (255, 255, 255)),
                         ((60, 60, 65), (255, 255, 255)), ((250, 205, 0), (15, 15, 15))],
         "warning": [((250, 205, 0), (15, 15, 15)), ((245, 245, 245), (15, 15, 15))],
         "prohibition": [((245, 245, 245), (190, 20, 20)), ((245, 245, 245), (20, 20, 20)), ((190, 20, 20), (255, 255, 255))],
         "mandatory": [((20, 70, 160), (255, 255, 255)), ((245, 245, 245), (20, 70, 160))],
         "direction": [((250, 205, 0), (15, 15, 15)), ((20, 70, 160), (255, 255, 255)), ((0, 112, 60), (255, 255, 255)),
                       ((25, 25, 25), (250, 205, 0)), ((245, 245, 245), (20, 20, 20))]}
# MIT Indoor Scenes rooms that seldom contain signs
BACKGROUND_ROOMS = ["bedroom", "livingroom", "kitchen", "dining_room", "bathroom", "children_room", "closet",
                    "corridor", "classroom", "office", "meeting_room", "waitingroom", "lobby", "hospitalroom",
                    "operating_room", "dentaloffice", "studiomusic", "artstudio", "pantry", "nursery", "laundromat",
                    "gameroom", "locker_room", "computerroom", "stairscase", "elevator", "lobby"]
# 85-class traffic signs (IRC categories): regulatory red-circle signs and limits -> prohibition, blue / stop /
# give-way -> mandatory, the rest are cautionary -> warning
def traffic_type(name):
    if "PROHIBITED" in name or name.startswith("NO_") or "LIMIT" in name:
        return "prohibition"
    if name.startswith("COMPULSARY") or name in ("STOP", "GIVE_WAY", "PASS_EITHER_SIDE", "TURN_RIGHT",
                                                  "PRIORITY_FOR_ONCOMING_VEHICLES"):
        return "mandatory"
    if name in ("RESTRICTION_ENDS", "DIRECTION"):
        return "information"
    return "warning"


def fill(text, rng):
    return text.replace("{n}", str(rng.randint(1, 12))).replace("{L}", rng.choice("ABCDEFGH"))


def font(path, size):
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)


def pictogram(kind, size, ink, rng):
    """RGBA square pictogram, drawn at 4x and downsampled."""
    s = 4 * size
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    k = s / 100
    black, red, white, blue = (20, 20, 20, 255), (200, 20, 20, 255), (255, 255, 255, 255), (20, 70, 160, 255)
    def person(x, y, sc, col, skirt=False):
        d.ellipse([x - 7 * sc, y, x + 7 * sc, y + 14 * sc], fill=col)
        body = [(x - 11 * sc, y + 18 * sc), (x + 11 * sc, y + 18 * sc), (x + 15 * sc if skirt else x + 9 * sc, y + 50 * sc),
                (x - 15 * sc if skirt else x - 9 * sc, y + 50 * sc)]
        d.polygon(body, fill=col)
        d.rectangle([x - 8 * sc, y + 50 * sc, x - 2 * sc, y + 75 * sc], fill=col)
        d.rectangle([x + 2 * sc, y + 50 * sc, x + 8 * sc, y + 75 * sc], fill=col)
    if kind in ("cigarette", "phone", "camera", "P"):  # prohibition: glyph + red ring and slash
        d.ellipse([4 * k, 4 * k, 96 * k, 96 * k], fill=white)
        if kind == "cigarette":
            d.rectangle([18 * k, 46 * k, 70 * k, 56 * k], fill=black)
            d.rectangle([70 * k, 46 * k, 80 * k, 56 * k], fill=(230, 120, 30, 255))
            d.arc([60 * k, 18 * k, 80 * k, 44 * k], 180, 360, fill=black, width=int(3 * k))
        elif kind == "phone":
            d.rounded_rectangle([36 * k, 20 * k, 64 * k, 80 * k], radius=int(5 * k), fill=black)
            d.rectangle([40 * k, 28 * k, 60 * k, 66 * k], fill=white)
        elif kind == "camera":
            d.rounded_rectangle([22 * k, 34 * k, 78 * k, 72 * k], radius=int(6 * k), fill=black)
            d.ellipse([38 * k, 40 * k, 62 * k, 64 * k], fill=white)
            d.rectangle([34 * k, 28 * k, 48 * k, 34 * k], fill=black)
        else:
            d.text((50 * k, 52 * k), "P", font=font(LAT[0], int(60 * k)), fill=black, anchor="mm")
        d.ellipse([4 * k, 4 * k, 96 * k, 96 * k], outline=red, width=int(10 * k))
        d.line([(20 * k, 20 * k), (80 * k, 80 * k)], fill=red, width=int(10 * k))
    elif kind == "noentry":
        d.ellipse([4 * k, 4 * k, 96 * k, 96 * k], fill=red)
        d.rectangle([20 * k, 42 * k, 80 * k, 58 * k], fill=white)
    elif kind in ("exclaim", "bolt"):  # warning triangle
        tri = [(50 * k, 6 * k), (96 * k, 90 * k), (4 * k, 90 * k)]
        d.polygon(tri, fill=(250, 205, 0, 255))
        d.line(tri + [tri[0]], fill=black, width=int(7 * k), joint="curve")
        if kind == "exclaim":
            d.rectangle([46 * k, 34 * k, 54 * k, 66 * k], fill=black)
            d.ellipse([45 * k, 72 * k, 55 * k, 82 * k], fill=black)
        else:
            d.polygon([(56 * k, 30 * k), (40 * k, 60 * k), (52 * k, 60 * k), (44 * k, 84 * k), (62 * k, 52 * k),
                       (50 * k, 52 * k)], fill=black)
    elif kind in ("person", "keepleft"):  # mandatory: white glyph on a blue disc
        d.ellipse([4 * k, 4 * k, 96 * k, 96 * k], fill=blue)
        if kind == "person":
            person(50 * k, 14 * k, k, white)
        else:
            d.polygon([(28 * k, 72 * k), (28 * k, 44 * k), (40 * k, 56 * k), (66 * k, 30 * k), (74 * k, 38 * k),
                       (48 * k, 64 * k), (60 * k, 72 * k)], fill=white)
    elif kind in ("toilet", "man", "woman"):
        if kind in ("toilet", "man"):
            person((30 if kind == "toilet" else 50) * k, 12 * k, k, ink + (255,))
        if kind in ("toilet", "woman"):
            person((70 if kind == "toilet" else 50) * k, 12 * k, k, ink + (255,), skirt=True)
        if kind == "toilet":
            d.line([(50 * k, 8 * k), (50 * k, 92 * k)], fill=ink + (255,), width=int(3 * k))
    elif kind == "lift":
        d.rectangle([14 * k, 8 * k, 86 * k, 92 * k], outline=ink + (255,), width=int(5 * k))
        d.polygon([(50 * k, 20 * k), (64 * k, 40 * k), (36 * k, 40 * k)], fill=ink + (255,))
        d.polygon([(50 * k, 80 * k), (64 * k, 60 * k), (36 * k, 60 * k)], fill=ink + (255,))
    elif kind == "wheelchair":
        d.text((50 * k, 52 * k), "♿", font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                                              int(80 * k)), fill=ink + (255,), anchor="mm")
    elif kind == "cross":
        d.rectangle([12 * k, 12 * k, 88 * k, 88 * k], fill=(0, 140, 70, 255))
        d.rectangle([40 * k, 22 * k, 60 * k, 78 * k], fill=white)
        d.rectangle([22 * k, 40 * k, 78 * k, 60 * k], fill=white)
    elif kind == "run":
        d.rectangle([4 * k, 4 * k, 96 * k, 96 * k], fill=(0, 140, 70, 255))
        person(46 * k, 14 * k, k, white)
    return im.resize((size, size), Image.LANCZOS)


def lines_for(en, hi, rng):
    mode = rng.choices(["both", "en", "hi"], [0.55, 0.3, 0.15])[0]
    if mode == "en":
        return [("en", en)]
    if mode == "hi":
        return [("hi", hi)]
    return [("hi", hi), ("en", en)] if rng.random() < 0.6 else [("en", en), ("hi", hi)]


def text_block(lines, th, ink, rng):
    """RGBA block of the given (lang, text) lines, centred; returns the image."""
    fonts = [font(rng.choice(DEV if lang == "hi" else LAT), th if lang == "en" else int(th * 1.05))
             for lang, _ in lines]
    sizes = [f.getbbox(t) for f, (_, t) in zip(fonts, lines)]
    gap = int(th * 0.3)
    w = max(b[2] - b[0] for b in sizes)
    h = sum(b[3] - b[1] for b in sizes) + gap * (len(lines) - 1)
    im = Image.new("RGBA", (w + 4, h + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    y = 2
    for f, (_, t), b in zip(fonts, lines, sizes):
        d.text((2 + (w - (b[2] - b[0])) // 2 - b[0], y - b[1]), t, font=f, fill=ink + (255,))
        y += b[3] - b[1] + gap
    return im


def sign_panel(rng):
    """One non-direction sign: (RGBA image, [(text, direction, type, box)])."""
    en, hi, typ, pic = rng.choice(SIGNS)
    en, hi = fill(en + "\t" + hi, rng).split("\t")  # the same number in both languages
    lines = lines_for(en, hi, rng)
    bg, ink = rng.choice(STYLE[typ])
    th = rng.randint(36, 64)
    block = text_block(lines, th, ink, rng)
    pad = int(th * rng.uniform(0.4, 0.8))
    picsize = int(th * rng.uniform(2.0, 3.2)) if pic and rng.random() < 0.85 else 0
    layout = rng.choice(["top", "left"]) if picsize else None
    if layout == "top":
        W, H = max(block.width, picsize) + 2 * pad, picsize + block.height + 3 * pad
    elif layout == "left":
        W, H = picsize + block.width + 3 * pad, max(picsize, block.height) + 2 * pad
    else:
        W, H = block.width + 2 * pad, block.height + 2 * pad
    im = Image.new("RGBA", (W, H), bg + (255,))
    d = ImageDraw.Draw(im)
    if rng.random() < 0.6:
        bw = max(2, th // 12)
        d.rounded_rectangle([bw, bw, W - bw - 1, H - bw - 1], radius=th // 4, outline=ink, width=bw)
    if layout == "top":
        im.alpha_composite(pictogram(pic, picsize, ink, rng), ((W - picsize) // 2, pad))
        im.alpha_composite(block, ((W - block.width) // 2, picsize + 2 * pad))
    elif layout == "left":
        im.alpha_composite(pictogram(pic, picsize, ink, rng), (pad, (H - picsize) // 2))
        im.alpha_composite(block, (picsize + 2 * pad, (H - block.height) // 2))
    else:
        im.alpha_composite(block, (pad, pad))
    text = " / ".join(t for _, t in lines)
    return im, [(text, "none", typ, [0, 0, W, H])]


def direction_panel(rng):
    """Panel of 1-4 'arrow + place' rows: (RGBA image, items), one item per row."""
    bg, ink = rng.choice(STYLE["direction"])
    th = rng.randint(34, 56)
    rows = []
    for _ in range(rng.choices([1, 2, 3, 4], [0.3, 0.35, 0.25, 0.1])[0]):
        en, hi, _, _ = rng.choice(PLACES)
        en, hi = fill(en + "\t" + hi, rng).split("\t")
        rows.append((lines_for(en, hi, rng), rng.choices(list(DIRS), list(DIRS.values()))[0]))
    blocks = [text_block(lines, th, ink, rng) for lines, _ in rows]
    asz = int(th * rng.uniform(1.4, 2.2))
    pad, gap = int(th * 0.5), int(th * 0.5)
    rh = [max(asz, b.height) for b in blocks]
    W = max(b.width for b in blocks) + asz + 3 * pad
    H = sum(rh) + gap * (len(rows) - 1) + 2 * pad
    im = Image.new("RGBA", (W, H), bg + (255,))
    items, y = [], pad
    for (lines, d), b, h in zip(rows, blocks, rh):
        side = "left" if "left" in d else "right" if "right" in d else rng.choice(["left", "right"])
        ax = pad if side == "left" else W - pad - asz
        tx = ax + asz + pad if side == "left" else ax - pad - b.width
        im.alpha_composite(arrow(asz, ANGLE[d] + rng.uniform(-4, 4), ink, rng), (ax, y + (h - asz) // 2))
        im.alpha_composite(b, (tx, y + (h - b.height) // 2))
        x1, x2 = min(ax, tx), max(ax + asz, tx + b.width)
        items.append((" / ".join(t for _, t in lines), d, "direction", [x1 - 3, y - 3, x2 + 3, y + h + 3]))
        y += h + gap
    return im, items


def traffic_signs():
    """[(path, meaning, type)] from the 85-class Indian traffic-sign crops."""
    root = RAW / "kaggle/indian-traffic-85"
    out = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir() and p.parent.name in ("train", "test")):
        typ = traffic_type(d.name.upper())
        meaning = d.name.replace("_", " ").replace("COMPULSARY", "COMPULSORY").capitalize()
        out += [(str(f), meaning, typ) for f in sorted(d.glob("*.jpg"))[:40]]
    return out


def traffic_panel(signs, rng):
    path, meaning, typ = rng.choice(signs)
    im = Image.open(path).convert("RGBA")
    s = rng.randint(140, 320) / max(im.size)
    im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BICUBIC)
    return im, [(meaning, "none", typ, [0, 0, im.width, im.height])]


def compose(bg, panels, rng):
    """Paste panels (warped) without overlap; returns the image and the labelled items in image pixels."""
    import cv2
    s = rng.randint(1024, 1536) / max(bg.size)
    bg = bg.resize((round(bg.width * s), round(bg.height * s)), Image.BICUBIC)
    W, H = bg.size
    out = np.asarray(bg.convert("RGB")).astype(np.float32)
    taken, items = [], []
    for sign, its in panels:
        sw, sh = sign.size
        for _ in range(20):
            f = rng.uniform(0.18, 0.55) * W / sw
            f = min(f, 0.7 * H / sh)
            w, h = sw * f, sh * f
            x0, y0 = rng.uniform(0, W - w), rng.uniform(0, H - h)
            if not any(x0 < b[2] and x0 + w > b[0] and y0 < b[3] and y0 + h > b[1] for b in taken):
                break
        else:
            continue
        taken.append([x0, y0, x0 + w, y0 + h])
        j = 0.07 * min(w, h)
        dst = np.float32([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]) + np.float32(
            [[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]]), dst)
        warped = cv2.warpPerspective(np.asarray(sign), M, (W, H), flags=cv2.INTER_AREA)
        a = warped[..., 3:4].astype(np.float32) / 255
        out = out * (1 - a) + warped[..., :3] * a
        for text, d, typ, (x1, y1, x2, y2) in its:
            p = cv2.perspectiveTransform(np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]), M)[0]
            items.append({"text": text, "direction": d, "type": typ,
                          "bbox": [max(0, p[:, 0].min()), max(0, p[:, 1].min()), min(W, p[:, 0].max()),
                                   min(H, p[:, 1].max())]})
    return out, items


def backgrounds():
    root = RAW / "kaggle/mit-indoor"
    out = []
    for room in BACKGROUND_ROOMS:
        for d in root.rglob(room):
            if d.is_dir():
                out += sorted(str(p) for p in d.glob("*.jpg"))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2400, help="train images")
    ap.add_argument("--test", type=int, default=200, help="held-out images (own backgrounds and seed)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    bgs, signs = backgrounds(), traffic_signs()
    rng.shuffle(bgs)
    cut = len(bgs) // 10
    print(f"{len(bgs)} backgrounds, {len(signs)} traffic-sign crops, {len(SIGNS)} sign texts", flush=True)
    (SFT / "images").mkdir(parents=True, exist_ok=True)
    for split, n, pool in (("test", a.test, bgs[:cut]), ("train", a.n, bgs[cut:])):
        rows, count = [], {}
        for i in range(n):
            panels = []
            for _ in range(rng.choices([1, 2, 3], [0.45, 0.4, 0.15])[0]):
                kind = rng.choices(["sign", "direction", "traffic"], [0.55, 0.3, 0.15 if signs else 0])[0]
                panels.append(sign_panel(rng) if kind == "sign" else direction_panel(rng) if kind == "direction"
                              else traffic_panel(signs, rng))
            if rng.random() < 0.04:
                panels = []  # a few images with no sign at all
            img, items = compose(Image.open(rng.choice(pool)).convert("RGB"), panels, rng)
            img = degrade(img, rng)
            name = f"synthsigns__{split}_{a.seed}_{i:05d}.jpg"
            img.save(SFT / "images" / name, quality=rng.randint(70, 95))
            for it in items:
                count[it["type"]] = count.get(it["type"], 0) + 1
            rows.append({"image": name, "source": "synthetic_signs", "reviewed": False,
                         "conversations": [{"from": "human", "value": PROMPT_ALL},
                                           {"from": "gpt", "value": answer(items, *img.size)}]})
        p = SFT / f"synth_signs_{split}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{split}: {len(rows)} images -> {p}; items per type {count}", flush=True)


if __name__ == "__main__":
    main()
