"""Teacher labeller: Qwen2.5-VL (4-bit) reads every destination on a direction sign together with its arrow.

For each image it returns a list of {"text", "direction", "bbox"} in pixel coordinates of the original image, where
bbox covers the destination text plus its arrow. Directions are one of DIRECTIONS.

  python -m signdir.teacher img1.jpg img2.jpg ...      (or one path per line on stdin; prints the parsed labels)
"""
from __future__ import annotations

import json
import os
import re
import sys

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

MODEL = os.environ.get("TEACHER", "Qwen/Qwen2.5-VL-7B-Instruct")
DIRECTIONS = ("left", "right", "straight", "back", "up-left", "up-right", "down-left", "down-right", "down", "none")
MAX_PIXELS = int(os.environ.get("TEACHER_PIXELS", 768 * 28 * 28))  # ~0.6 MP keeps the 4-bit 7B within 6 GB

PROMPT = (
    "Read the direction signs in this image (road guide signs, exit signs, wayfinding boards). "
    "List every destination written on them together with the direction of the arrow next to it.\n"
    "Output a JSON list of objects with keys "
    '"text" (destination as written, original language), '
    '"direction" (one of ' + ", ".join(DIRECTIONS) + '; straight = arrow pointing up/ahead; none = no arrow) and '
    '"bbox_2d" ([x1, y1, x2, y2] around the destination and its arrow). '
    "Output [] only if the image has no direction sign. Write compact JSON on one line."
)


def load(model=MODEL):
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True,
                           llm_int8_skip_modules=["visual.merger"],  # also quantize lm_head (default keeps it bf16)
                           llm_int8_enable_fp32_cpu_offload=True)
    # the 7B's untied 152k x 3584 input embedding alone is 1 GB: keep it on the CPU (a lookup), the rest on the GPU
    dmap = {"model.language_model.embed_tokens": "cpu", "model.visual": 0, "model.language_model.layers": 0,
            "model.language_model.norm": 0, "model.language_model.rotary_emb": 0, "lm_head": 0}
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, quantization_config=q, device_map=dmap,
                                                           torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    p = AutoProcessor.from_pretrained(model, min_pixels=256 * 28 * 28, max_pixels=MAX_PIXELS)
    return m.eval(), p


def parse(answer: str, sx: float, sy: float):
    """JSON list from the answer (tolerates ```json fences / trailing text); boxes rescaled to the original image."""
    s = answer[answer.find("["): answer.rfind("]") + 1]
    try:
        items = json.loads(s)
    except json.JSONDecodeError:
        items = [json.loads(m) for m in re.findall(r"\{[^{}]*\}", s) if _ok(m)]
    out = []
    for it in items if isinstance(items, list) else []:
        b, t, d = it.get("bbox_2d"), str(it.get("text", "")).strip(), str(it.get("direction", "")).strip().lower()
        if not (isinstance(b, list) and len(b) == 4 and t and d in DIRECTIONS):
            continue
        x1, y1, x2, y2 = (float(v) for v in b)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append({"text": t, "direction": d, "bbox": [x1 * sx, y1 * sy, x2 * sx, y2 * sy]})
    return out


def _ok(s):
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


CHECK = ("Look at the sign text \"{text}\". Which way does the arrow that belongs to it point? "
         "Answer with one of: " + ", ".join(DIRECTIONS) + " (straight = up/ahead, none = it has no arrow). Reply with that one word only.")


# arrow-presence checks, tuned on the reviewed test set (signdir.arrow_test): the zoomed CHECK read always names some
# direction, so an item keeps its arrow only if ARROW_YN says yes or DIR_BOX (box drawn on the crop) names one
ARROW_YN = ('Is there an arrow symbol that belongs to the sign text "{text}" (right next to it, on the same line '
            "or in the same sign panel)? Answer yes or no.")
DIR_BOX = ('The red box marks the sign text "{text}". If there is an arrow symbol inside the red box, which way '
           "does it point? Answer with one of: " + ", ".join(DIRECTIONS) + " (straight = up/ahead). If there is no "
           "arrow inside the red box, answer none. Reply with that one word only.")


def yes(answer):
    a = answer.strip().lower()
    return True if a.startswith("yes") else False if a.startswith("no") else None


def marked_crop(image, box):
    """Zoomed crop around box (as crop_around) with the box drawn in red."""
    from PIL import ImageDraw
    x0, y0, x1, y1 = crop_box(image, box)
    crop = image.crop((x0, y0, x1, y1)).copy()
    w = max(2, round(max(crop.size) / 150))
    ImageDraw.Draw(crop).rectangle([box[0] - x0 - w, box[1] - y0 - w, box[2] - x0 + w, box[3] - y0 + w],
                                   outline=(255, 0, 0), width=w)
    return crop


def arrow_checks(model, proc, image, it):
    """Adds "arrow_yn" (True / False / None) and "direction_box" to an item."""
    a, _ = _ask(model, proc, crop_around(image, it["bbox"]), ARROW_YN.format(text=it["text"]), 24)
    it["arrow_yn"] = yes(a)
    a, _ = _ask(model, proc, marked_crop(image, it["bbox"]), DIR_BOX.format(text=it["text"]), 24)
    it["direction_box"] = first_direction(a)


def has_arrow(it):
    """Arrow-presence vote of the checks (items labelled before the checks existed: None = unknown)."""
    if "arrow_yn" not in it:
        return None
    return bool(it["arrow_yn"]) or it.get("direction_box") not in (None, "none")


def first_direction(answer):
    """The first direction word in a free-text answer ("The arrow points up-left." -> "up-left")."""
    a = answer.lower().replace("_", "-").replace(" ", "-")
    hits = [(a.find(d), -len(d), d) for d in DIRECTIONS if d in a]
    return min(hits)[2] if hits else None


def open_image(path):
    """RGB image with the EXIF rotation applied (phone photos), as the training images will be."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _ask(model, proc, image, prompt, max_new_tokens):
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, max_time=180)  # no hour-long stalls
    return proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0], inputs


def crop_box(image, box, pad=0.6, min_side=192):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = max(pad * w, (min_side - w) / 2, 0), max(pad * h, (min_side - h) / 2, 0)
    a, b = max(0, x1 - px), max(0, y1 - py)
    c, d = min(image.width, x2 + px), min(image.height, y2 + py)
    # a box partly or wholly outside the image would give an empty crop: keep at least 28 px inside it
    a, b = min(a, image.width - 28), min(b, image.height - 28)
    return max(0, a), max(0, b), max(c, a + 28), max(d, b + 28)


def crop_around(image, box, pad=0.6, min_side=192):
    return image.crop(crop_box(image, box, pad, min_side))


def crop_origin(image, box, pad=0.6, min_side=192):
    return crop_box(image, box, pad, min_side)[:2]


def _read(model, proc, image, max_new_tokens):
    answer, inputs = _ask(model, proc, image, PROMPT, max_new_tokens)
    # Qwen2.5-VL boxes are absolute pixels of the resized model input
    gh, gw = inputs.image_grid_thw[0, 1].item() * 14, inputs.image_grid_thw[0, 2].item() * 14
    return parse(answer, image.width / gw, image.height / gh), answer


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def tile_boxes(image, grid, overlap=0.2):
    """grid x grid overlapping tiles covering the image: [(x1, y1, x2, y2)]."""
    out = []
    for gy in range(grid):
        for gx in range(grid):
            tw, th = image.width / (grid - (grid - 1) * overlap), image.height / (grid - (grid - 1) * overlap)
            x1, y1 = gx * tw * (1 - overlap), gy * th * (1 - overlap)
            out.append((round(x1), round(y1), round(min(image.width, x1 + tw)), round(min(image.height, y1 + th))))
    return out


def _on_inner_edge(box, tile, image, margin=0.02):
    """Item box touching a tile edge that is not an image edge: probably cut off by the tile."""
    x1, y1, x2, y2 = box
    a, b, c, d = tile
    mx, my = margin * (c - a), margin * (d - b)
    return ((a > 0 and x1 <= a + mx) or (b > 0 and y1 <= b + my) or
            (c < image.width and x2 >= c - mx) or (d < image.height and y2 >= d - my))


@torch.inference_mode()
def label(model, proc, image: Image.Image, regions=(), max_new_tokens=512, check=True, max_regions=4, grid=1):
    """Labels of one image.

    regions: known sign boxes (from the source dataset); each (up to max_regions, largest first) is also read on a
    zoomed crop, which finds small / distant signs the downscaled full image misses. A zoomed item replaces a
    full-image item with IoU > 0.3 (it is read at higher resolution). With check, each arrow direction is re-read
    on a crop around the item ("direction_crop"), "agree" says whether both reads match, and arrow_checks adds the
    arrow-presence answers ("arrow_yn", "direction_box").
    grid > 1 also reads grid x grid overlapping tiles (each at the full input resolution), merged like region reads;
    tile items touching an inner tile edge are dropped (cut-off signs: the neighbouring tile or the full read has them)."""
    items, answer = _read(model, proc, image, max_new_tokens)
    for it in items:
        it["from"] = "full"
    if grid > 1:
        for t in tile_boxes(image, grid):
            found, _ = _read(model, proc, image.crop(t), max_new_tokens)
            for it in found:
                it["bbox"] = [it["bbox"][0] + t[0], it["bbox"][1] + t[1], it["bbox"][2] + t[0], it["bbox"][3] + t[1]]
                if _on_inner_edge(it["bbox"], t, image):
                    continue
                it["from"] = "tile"
                items = [o for o in items if o["from"] != "full" or box_iou(o["bbox"], it["bbox"]) <= 0.3]
                if not any(box_iou(o["bbox"], it["bbox"]) > 0.5 for o in items):
                    items.append(it)
    regions = sorted(regions, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:max_regions]
    for r in regions:
        if (r[2] - r[0]) * (r[3] - r[1]) > 0.5 * image.width * image.height:
            continue  # already read at good resolution in the full image
        crop = crop_around(image, r, pad=0.3, min_side=448)
        ox, oy = crop_origin(image, r, pad=0.3, min_side=448)
        found, _ = _read(model, proc, crop, max_new_tokens)
        for it in found:
            it["bbox"] = [it["bbox"][0] + ox, it["bbox"][1] + oy, it["bbox"][2] + ox, it["bbox"][3] + oy]
            it["from"] = "region"
            items = [o for o in items if o["from"] == "region" or box_iou(o["bbox"], it["bbox"]) <= 0.3]
            if not any(box_iou(o["bbox"], it["bbox"]) > 0.5 for o in items):
                items.append(it)
    out, seen = [], set()
    for it in items:
        key = (it["text"].lower(), tuple(round(v) for v in it["bbox"]))
        if key in seen:
            continue
        seen.add(key)
        if check:
            a, _ = _ask(model, proc, crop_around(image, it["bbox"]), CHECK.format(text=it["text"]), 24)
            it["direction_crop"] = first_direction(a)
            it["agree"] = it["direction_crop"] == it["direction"]
            arrow_checks(model, proc, image, it)
        out.append(it)
    return out, answer


if __name__ == "__main__":
    import time
    m, p = load()
    files = sys.argv[1:] or [l.strip() for l in sys.stdin if l.strip()]
    for f in files:
        t0 = time.time()
        labels, raw = label(m, p, open_image(f))
        print(f"{time.time() - t0:.1f}s", f, json.dumps(labels, ensure_ascii=False))
    print("peak GPU MiB", torch.cuda.max_memory_allocated() // 2**20)
