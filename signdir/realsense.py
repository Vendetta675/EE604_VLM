"""Intel RealSense RGB-D source for signdir.live: the colour stream feeds the VLM, the depth stream (aligned to colour)
gives each sign's metric distance, replacing the monocular Depth Anything estimate (no calibration, no extra model).

The VLM itself stays RGB-only (LocateAnything was never trained on depth); depth is used after detection:
  distance(depth_m, bbox)   median of the VALID (non-zero) depth inside the inner part of the box; RealSense marks
                            holes (glossy / backlit / out-of-range surfaces) as 0, which a plain median would count.
Accuracy: stereo depth error grows ~ quadratically with range (D435: ~2 % at 2 m); beyond ~6-8 m treat it as coarse.

  python -m signdir.live --realsense                    colour 1280x720 + aligned depth, 30 fps
  python -m signdir.realsense                           check: stream info and the depth at the image centre
"""
from __future__ import annotations

import threading
import time

import numpy as np


class RealSense:
    """Colour (BGR) + depth aligned to colour, in metres. A background thread keeps only the newest frameset, so
    several readers each get the freshest frame:
      latest()        colour + depth aligned to colour (filtered); aligning costs CPU, so it is done here, on demand
                      (the sign VLM asks every ~2 s), not for every frame
      latest_depth()  the raw depth image in metres, in the depth camera's own pixels (`depth_intrinsics`); cheap,
                      for the obstacle check at camera rate
    Both wait for a frame newer than `after` (the frame number the caller saw last)."""

    def __init__(self, width=1280, height=720, fps=30, filters=True, serial=None):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipe.start(cfg)
        dev = profile.get_device()
        self.name = f"{dev.get_info(rs.camera_info.name)} ({dev.get_info(rs.camera_info.serial_number)})"
        self.depth_scale = dev.first_depth_sensor().get_depth_scale()      # z16 units -> metres
        self.align = rs.align(rs.stream.color)
        # edge-preserving spatial + temporal smoothing; no hole filling (an invented depth is worse than none)
        self.filters = [rs.spatial_filter(), rs.temporal_filter()] if filters else []
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = intr
        self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        for _ in range(15):                                                 # let auto-exposure settle
            self.pipe.wait_for_frames()
        self._cv, self._fs, self._depth, self._n, self._stop, self.error = threading.Condition(), None, None, 0, False, None
        self._align_lock = threading.Lock()
        self._thread = threading.Thread(target=self._grab, daemon=True)
        self._thread.start()

    def _grab(self):
        try:
            while not self._stop:
                fs = self.pipe.wait_for_frames(5000)
                fs.keep()                                                   # hold it past the SDK's frame pool
                raw = np.asanyarray(fs.get_depth_frame().get_data())
                with self._cv:
                    self._fs, self._depth, self._n = fs, raw, self._n + 1
                    self._cv.notify_all()
        except Exception as e:                                              # unplugged / stopped: wake the readers
            self.error = e
            with self._cv:
                self._cv.notify_all()

    def _wait(self, after, timeout_s):
        ok = self._cv.wait_for(lambda: self._n > max(after, 0) or self.error is not None, timeout_s)
        if self.error is not None:
            raise RuntimeError(f"RealSense stopped: {self.error}")
        if not ok:
            raise RuntimeError("RealSense: no frame within the timeout")

    def latest(self, after: int = -1, timeout_s=5.0):
        """(bgr, depth_m aligned to colour) newer than frame number `after`; with `after` also the frame number."""
        with self._cv:
            self._wait(after, timeout_s)
            fs, n = self._fs, self._n
        with self._align_lock:
            fs = self.align.process(fs)
            color, depth = fs.get_color_frame(), fs.get_depth_frame()
            for f in self.filters:
                depth = f.process(depth)
            bgr = np.asanyarray(color.get_data()).copy()
            depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        return (bgr, depth_m) if after < 0 else (bgr, depth_m, n)

    def latest_depth(self, after: int = -1, timeout_s=5.0):
        """(raw depth in metres in the depth camera's pixels, frame number), newer than `after`."""
        with self._cv:
            self._wait(after, timeout_s)
            raw, n = self._depth, self._n
        return raw.astype(np.float32) * self.depth_scale, n

    def bearing_deg(self, x: float) -> float:
        """Horizontal angle of image column x from the optical axis (+ = right)."""
        return float(np.degrees(np.arctan2(x - self.intrinsics.ppx, self.intrinsics.fx)))

    def close(self):
        self._stop = True
        self._thread.join(timeout=2)
        try:
            self.pipe.stop()
        except RuntimeError:
            pass


def distance(depth_m: np.ndarray, bbox, inner=0.5, min_valid=0.2) -> float | None:
    """Median depth (m) of the valid pixels in the central `inner` fraction of the box; None when fewer than
    `min_valid` of those pixels have depth (the sign is a hole in the depth image: too far, glossy or backlit)."""
    h, w = depth_m.shape
    x1, y1, x2, y2 = bbox
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * inner / 2, (y2 - y1) * inner / 2
    xa, ya = max(0, int(cx - hw)), max(0, int(cy - hh))
    xb, yb = min(w, max(xa + 1, int(np.ceil(cx + hw)))), min(h, max(ya + 1, int(np.ceil(cy + hh))))
    if xb <= xa or yb <= ya:
        return None
    patch = depth_m[ya:yb, xa:xb]
    valid = patch[patch > 0]
    if valid.size < min_valid * patch.size:
        return None
    return float(np.median(valid))


def main():
    cam = RealSense()
    try:
        bgr, d = cam.latest()
        h, w = d.shape
        c = distance(d, [w * 0.45, h * 0.45, w * 0.55, h * 0.55])
        print(f"{cam.name}: colour {bgr.shape[1]}x{bgr.shape[0]}, depth scale {cam.depth_scale:g} m/unit, "
              f"valid depth {np.mean(d > 0):.0%}, centre {c if c is None else round(c, 2)} m")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
