"""Camera-frame preprocessing before the sign model (RealSense / webcam frames are dim, low-contrast and 1280 px wide,
while training images were ~1536 px). Ops, applied in this order, given as a '+'-joined spec (e.g. "clahe+up1536"):

  wb        grey-world white balance (indoor tungsten / LED casts)
  gamma     auto gamma: brighten so the median luminance is ~0.45 (dark corridors), never darken
  clahe     contrast-limited histogram equalisation of the L channel (LAB), clip 2, 8x8 tiles
  sharp     unsharp mask (radius 2, 80 %)
  upN       bicubic enlargement to a long side of at least N px (as la._upscaled / build_upscaled)

Boxes come back in the coordinates of the returned image; callers rescale by the returned factor.

  python -m signdir.evaluate --adapter ... --mode slow --test test_real.jsonl --pre clahe+up1536
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

ORDER = ("wb", "gamma", "clahe", "sharp")


def _wb(a):
    m = a.reshape(-1, 3).mean(0)
    return np.clip(a * (m.mean() / np.maximum(m, 1e-3)), 0, 255)


def _gamma(a):
    y = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    med = float(np.median(y))
    if med <= 0.02 or med >= 0.45:
        return a
    g = np.log(0.45) / np.log(med)  # < 1: brightens
    return 255.0 * (a / 255.0) ** g


def _clahe(a):
    lab = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)


def apply(image: Image.Image, spec: str | None):
    """(processed RGB image, scale of the processed image relative to the input)."""
    if not spec or spec == "none":
        return image, 1.0
    ops = spec.split("+")
    a = np.asarray(image.convert("RGB")).astype(np.float32)
    for op in ORDER:
        if op in ops:
            a = {"wb": _wb, "gamma": _gamma, "clahe": _clahe}.get(op, lambda x: x)(a)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if "sharp" in ops:
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
    s = 1.0
    for op in ops:
        if op.startswith("up"):
            n = int(op[2:])
            if max(out.size) < n:
                s = n / max(out.size)
                out = out.resize((round(out.width * s), round(out.height * s)), resample=Image.BICUBIC)
    unknown = [o for o in ops if o not in ORDER and not o.startswith("up")]
    if unknown:
        raise ValueError(f"unknown preprocessing op(s): {unknown}")
    return out, s
