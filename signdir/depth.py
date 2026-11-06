"""Distance to detected signs with Depth Anything V2 (metric, transformers port), calibrated to the webcam.

The metric checkpoints predict depth in metres per pixel: "indoor" was fine-tuned on Hypersim (up to ~20 m),
"outdoor" on Virtual KITTI (up to ~80 m). A sign's distance is the median depth over the middle of its box (the
box edges often catch the wall or sky behind the board).

Metric depth from one image assumes the training cameras' field of view, so on a webcam it is off by a factor that
can also drift with distance. `calibrate` fixes that per webcam: hold a flat target (a sign, a sheet of paper on a
wall, a door) at a few measured distances inside the centre box; it fits true = a * estimate^b (b = 1 with fewer
than 3 distances) and saves it to signdir/webcam_depth.json under the camera's name and the scene. live.py and
DepthEstimator(..., camera=index) pick it up automatically.

  python -m signdir.depth calibrate --scene indoor --at 1 2 3 4     # SPACE takes each distance, q aborts
  python -m signdir.depth show                                      # saved calibrations

  est = DepthEstimator("indoor", camera=0)    # calibrated for /dev/video0 if it has been
  depth = est.depth_map(rgb)                  # (H, W) metres, same size as the PIL image
  d = sign_distance(depth, [x1, y1, x2, y2])  # metres, or None if the box is empty
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CALIB = Path(__file__).with_name("webcam_depth.json")

MODELS = {
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
}


def camera_name(index: int) -> str:
    """The V4L2 name of /dev/video<index> ("Integrated Camera: Integrated C"), or "video<index>"."""
    p = Path(f"/sys/class/video4linux/video{index}/name")
    return p.read_text().strip() if p.exists() else f"video{index}"


def load_calibration(camera: int | None, scene: str):
    """{"a", "b", ...} saved for this camera and scene, or None."""
    if camera is None or not CALIB.exists():
        return None
    return json.loads(CALIB.read_text()).get(camera_name(camera), {}).get(scene)


class DepthEstimator:
    """scale: a fixed factor (overrides any calibration); camera: webcam index whose saved calibration to apply."""

    def __init__(self, scene="indoor", scale=None, input_size=518, device=None, camera=None):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        model_id = MODELS.get(scene, scene)  # a scene name or any Depth Anything V2 checkpoint id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(
            model_id, size={"height": input_size, "width": input_size})
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(self.device).eval()
        self.calibration = None if scale is not None else load_calibration(camera, scene)
        self.a, self.b = ((float(scale), 1.0) if scale is not None else
                          (self.calibration["a"], self.calibration["b"]) if self.calibration else (1.0, 1.0))

    def correct(self, d):
        return self.a * np.power(np.maximum(d, 1e-3), self.b)

    @torch.inference_mode()
    def depth_map(self, rgb: Image.Image) -> np.ndarray:
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device, self.dtype)
        pred = self.model(**inputs).predicted_depth  # (1, h, w) metres at the network's resolution
        pred = torch.nn.functional.interpolate(pred[:, None].float(), size=(rgb.height, rgb.width),
                                               mode="bilinear", align_corners=False)[0, 0]
        return self.correct(pred.cpu().numpy())

    def raw_map(self, rgb: Image.Image) -> np.ndarray:
        a, b, self.a, self.b = self.a, self.b, 1.0, 1.0
        try:
            return self.depth_map(rgb)
        finally:
            self.a, self.b = a, b


def sign_distance(depth: np.ndarray, bbox, inner=0.5) -> float | None:
    """Median depth over the central `inner` fraction (per side) of the box, in the depth map's units."""
    h, w = depth.shape
    x1, y1, x2, y2 = bbox
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * inner / 2, (y2 - y1) * inner / 2
    xa, ya = max(0, int(cx - hw)), max(0, int(cy - hh))
    xb, yb = min(w, max(xa + 1, int(np.ceil(cx + hw)))), min(h, max(ya + 1, int(np.ceil(cy + hh))))
    if xb <= xa or yb <= ya:
        return None
    return float(np.median(depth[ya:yb, xa:xb]))


class Smoother:
    """Per-sign exponential smoothing of distances across webcam frames. A sign is matched to the previous one with
    the same text; tracks unseen for `forget` seconds are dropped, so a sign seen again starts fresh."""

    def __init__(self, alpha=0.5, forget=3.0):
        self.alpha, self.forget, self.tracks = alpha, forget, {}

    def __call__(self, key: str, d: float | None, now: float | None = None) -> float | None:
        now = time.time() if now is None else now
        self.tracks = {k: v for k, v in self.tracks.items() if now - v[1] <= self.forget}
        if d is None:
            return self.tracks[key][0] if key in self.tracks else None
        if key in self.tracks:
            d = self.alpha * d + (1 - self.alpha) * self.tracks[key][0]
        self.tracks[key] = (d, now)
        return d


def fit(true, est):
    """(a, b) of true = a * est^b by least squares in log space; b = 1 with fewer than 3 distinct distances."""
    lt, le = np.log(np.asarray(true, float)), np.log(np.asarray(est, float))
    if len(set(np.round(true, 3))) < 3:
        return float(np.exp(np.mean(lt - le))), 1.0
    b, la = np.polyfit(le, lt, 1)
    return float(np.exp(la)), float(b)


def calibrate(a):
    import cv2
    from signdir.live import Viewer, frames
    est = DepthEstimator(a.scene, scale=1.0)
    viewer, samples = Viewer("depth calibration"), []
    targets = list(a.at)
    stream = frames(argparse.Namespace(image=None, camera=a.camera, width=a.width))
    box_frac = a.box
    for frame in stream:
        h, w = frame.shape[:2]
        box = [w * (1 - box_frac) / 2, h * (1 - box_frac) / 2, w * (1 + box_frac) / 2, h * (1 + box_frac) / 2]
        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        raw = sign_distance(est.raw_map(rgb), box, inner=1.0)
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f"target at {targets[0]:.2f} m in the box, SPACE to take  (raw {raw:.2f} m)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        viewer.show(frame)
        key = viewer.key  # a key pressed during the (slow) show; wait() resets it
        if not viewer.wait():
            raise SystemExit("aborted, nothing saved")
        if (key or viewer.key) != "space":
            continue
        reads = []
        for frame in stream:  # average a few fresh frames (webcam depth flickers)
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            reads.append(sign_distance(est.raw_map(rgb), box, inner=1.0))
            if len(reads) == a.frames:
                break
        samples.append({"true_m": targets.pop(0), "estimate_m": float(np.median(reads)),
                        "spread_m": float(np.std(reads))})
        print(f"  {samples[-1]['true_m']:.2f} m -> raw {samples[-1]['estimate_m']:.2f} m "
              f"(+- {samples[-1]['spread_m']:.2f})", flush=True)
        if not targets:
            break
    viewer.root.destroy()
    ka, kb = fit([s["true_m"] for s in samples], [s["estimate_m"] for s in samples])
    for s in samples:
        s["corrected_m"] = ka * s["estimate_m"] ** kb
    worst = max(abs(s["corrected_m"] / s["true_m"] - 1) for s in samples)
    allc = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    name = camera_name(a.camera)
    allc.setdefault(name, {})[a.scene] = {"a": ka, "b": kb, "width": a.width, "model": MODELS.get(a.scene, a.scene),
                                          "date": time.strftime("%Y-%m-%d"), "samples": samples}
    CALIB.write_text(json.dumps(allc, indent=1))
    print(f"{name} / {a.scene}: true = {ka:.3f} * estimate^{kb:.3f}; worst error after correction "
          f"{worst:.0%}; saved to {CALIB}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--scene", default="indoor", help="indoor, outdoor or a checkpoint id (as live.py --depth)")
    c.add_argument("--at", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0], help="target distances, metres")
    c.add_argument("--camera", type=int, default=0)
    c.add_argument("--width", type=int, default=1280, help="as live.py --width (the field of view can change)")
    c.add_argument("--box", type=float, default=0.25, help="centre box size, fraction of the frame")
    c.add_argument("--frames", type=int, default=10, help="frames averaged per distance")
    sub.add_parser("show")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a)
    else:
        print(CALIB.read_text() if CALIB.exists() else f"no calibration yet ({CALIB})")


if __name__ == "__main__":
    main()
