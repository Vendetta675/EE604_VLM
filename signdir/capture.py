"""Capture RealSense photos for the fine-tune's real test (and train) set: room plates, pen-on-paper notes, printed
indoor signs, stairs, obstacles.

A window shows the live colour image with a depth inset. Keys:
  space   save colour (JPEG) + aligned depth (16-bit PNG, millimetres) + a JSON with intrinsics, depth scale, the tag
  1..6    set the tag of the next shots: 1 room  2 handwritten  3 sign  4 stairs-up  5 stairs-down  6 obstacle
  q/Esc   quit
Files go to data/raw/realsense/<YYYY-MM-DD>/<tag>_<HHMMSS>_<n>.{jpg,png,json}. The tag only helps the labelling
(signdir.label_capture); labels themselves come from the teacher + your check in review.

  python -m signdir.capture [--out data/raw/realsense]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from signdir.pool import RAW

TAGS = {"1": "room", "2": "handwritten", "3": "sign", "4": "stairs-up", "5": "stairs-down", "6": "obstacle"}


def main():
    from signdir.live import Viewer
    from signdir.realsense import RealSense
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RAW / "realsense"))
    a = ap.parse_args()
    out = Path(a.out) / time.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    cam = RealSense()
    intr = cam.intrinsics
    meta = {"camera": cam.name, "depth_scale": cam.depth_scale, "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx,
            "ppy": intr.ppy, "width": intr.width, "height": intr.height}
    viewer, tag, n = Viewer("capture: space = save, 1-6 = tag, q = quit"), "sign", len(list(out.glob("*.jpg")))
    print(f"{cam.name} -> {out}  ({n} photos there already); tag keys: " +
          "  ".join(f"{k} {v}" for k, v in TAGS.items()), flush=True)
    try:
        while True:
            bgr, depth_m = cam.latest()
            show = bgr.copy()
            dv = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(depth_m, 0, 5), alpha=255 / 5), cv2.COLORMAP_JET)
            dv[depth_m == 0] = 0
            dv = cv2.resize(dv, (show.shape[1] // 4, show.shape[0] // 4))
            show[:dv.shape[0], -dv.shape[1]:] = dv
            cv2.putText(show, f"tag: {tag}   saved: {n}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
            cv2.putText(show, f"tag: {tag}   saved: {n}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
            viewer.show(show)                            # processes Tk events: the key pressed since the last frame
            k, viewer.key = viewer.key, None
            if viewer.closed or k in ("q", "Escape"):
                viewer.root.destroy()
                break
            if k in TAGS:
                tag = TAGS[k]
            elif k == "space":
                stem = out / f"{tag}_{time.strftime('%H%M%S')}_{n:04d}"
                cv2.imwrite(str(stem.with_suffix(".jpg")), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(stem.with_suffix(".png")), np.round(depth_m * 1000).astype(np.uint16))
                stem.with_suffix(".json").write_text(json.dumps({**meta, "tag": tag, "time": time.time()}))
                n += 1
                print(f"saved {stem.name}", flush=True)
    finally:
        cam.close()


if __name__ == "__main__":
    main()
