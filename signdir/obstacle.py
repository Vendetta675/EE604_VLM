"""Obstacle and drop-off warnings from RealSense depth, at camera rate (no model, no training).

Geometry per frame (depth aligned to colour, colour intrinsics):
  1. back-project every `step`-th pixel to camera coordinates (x right, y down, z forward)
  2. floor = the largest plane among the points below the optical axis whose normal is within `max_tilt_deg` of the
     camera's "up" (RANSAC; the camera is held roughly upright). Its distance to the camera = camera height.
  3. each point gets: height above the floor, forward distance along the floor, lateral offset (+ = right)
  4. obstacle = >= `min_points` points standing `min_h`..`max_h` above the floor inside the walking corridor
     (|lateral| < `half_width`, forward < `warn_m`); reported at its near edge (10th percentile), with its bearing.
     A low, wide obstacle (top below `step_h`) is called a step up.
  5. drop = >= `min_points` points more than `drop_h` BELOW the floor inside the corridor (stairs going down, a hole,
     a platform edge), at its near edge.
  When no floor is found (camera pointed at a wall or the ceiling, or too close), anything in the central part of the
  image nearer than `near_m` is reported as an obstacle.

  python -m signdir.obstacle                    print the warnings live (Ctrl-C stops)
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np


class Navigator:
    def __init__(self, intrinsics, half_width=0.35, warn_m=1.5, min_h=0.12, max_h=1.9, step_h=0.3, drop_h=0.15,
                 drop_from=0.5, drop_to=2.0, near_m=0.6, min_points=30, step=6, max_tilt_deg=40, cam_h=(0.5, 2.2), seed=0):
        self.fx, self.fy, self.cx, self.cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        self.half_width, self.warn_m, self.min_h, self.max_h = half_width, warn_m, min_h, max_h
        self.step_h, self.drop_h, self.drop_from, self.drop_to = step_h, drop_h, drop_from, drop_to
        self.near_m, self.min_points, self.step = near_m, min_points, step
        self.cos_tilt = math.cos(math.radians(max_tilt_deg))
        self.cam_h = cam_h
        self.rng = np.random.default_rng(seed)
        self.plane = None                                   # last floor (n, d), reused as a RANSAC seed

    def points(self, depth_m):
        d = depth_m[::self.step, ::self.step]
        v, u = np.mgrid[0:depth_m.shape[0]:self.step, 0:depth_m.shape[1]:self.step]
        ok = (d > 0.15) & (d < 6.0)
        z = d[ok]
        return np.stack([(u[ok] - self.cx) * z / self.fx, (v[ok] - self.cy) * z / self.fy, z], 1), v[ok]

    def floor(self, P, rows, h_img, iters=120, thr=0.03):
        """(n, d): unit normal pointing up (towards the camera side), n.p + d = height above the floor."""
        cand = P[(rows > h_img * 0.45) & (P[:, 1] > 0.05)]
        if len(cand) < 200:
            return None
        if len(cand) > 4000:
            cand = cand[self.rng.choice(len(cand), 4000, replace=False)]
        best, best_n = None, 0
        tries = [self.plane] if self.plane is not None else []
        for _ in range(iters):
            a, b, c = cand[self.rng.choice(len(cand), 3, replace=False)]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            n = n / nn
            if n[1] > 0:                                    # make it point up (camera y is down)
                n = -n
            tries.append((n, -float(n @ a)))
        for n, d in tries:
            if -n[1] < self.cos_tilt or not self.cam_h[0] <= d <= self.cam_h[1]:   # too steep / implausible height
                continue                            # (a desk top 0.2 m below the camera is not the floor)
            k = int((np.abs(cand @ n + d) < thr).sum())
            if k > best_n:
                best, best_n = (n, d), k
        if best is None or best_n < 0.25 * len(cand):
            return None
        n, d = best                                         # least-squares refine on the inliers
        inl = cand[np.abs(cand @ n + d) < thr]
        c0 = inl.mean(0)
        n = np.linalg.svd(inl - c0)[2][-1]
        n = -n if n[1] > 0 else n
        d = -float(n @ c0)
        if not self.cam_h[0] <= d <= self.cam_h[1]:
            self.plane = None
            return None
        self.plane = (n, d)
        return self.plane

    def analyse(self, depth_m) -> dict:
        P, rows = self.points(depth_m)
        out = {"floor": None, "camera_height": None, "obstacle": None, "drop": None}
        if len(P) < 100:
            return out
        pl = self.floor(P, rows, depth_m.shape[0])
        if pl is None:                                      # no floor: plain nearness in the image centre
            c = (np.abs(P[:, 0]) < self.half_width) & (P[:, 2] < self.near_m)
            if c.sum() >= self.min_points:
                q = P[c]
                out["obstacle"] = {"distance": float(np.percentile(q[:, 2], 10)), "kind": "obstacle",
                                   "bearing": float(np.degrees(np.arctan2(np.median(q[:, 0]),
                                                                          np.median(q[:, 2]))))}
            return out
        n, d = pl
        fwd = np.array([0.0, 0.0, 1.0]) - n[2] * n
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, n)
        h, f, lat = P @ n + d, P @ fwd, P @ right
        out["floor"], out["camera_height"] = True, float(d)
        corridor = (np.abs(lat) < self.half_width) & (f > 0.15)
        ob = corridor & (f < self.warn_m) & (h > self.min_h) & (h < self.max_h)
        if ob.sum() >= self.min_points:
            dist = float(np.percentile(f[ob], 10))
            near = ob & (f < dist + 0.3)                    # the obstacle's front part
            kind = "step up" if np.percentile(h[near], 90) < self.step_h and np.ptp(lat[near]) > 0.4 else "obstacle"
            out["obstacle"] = {"distance": dist, "kind": kind,
                               "bearing": float(np.degrees(np.arctan2(np.median(lat[near]), dist))),
                               "height": float(np.percentile(h[near], 90))}
        dr = corridor & (f > self.drop_from) & (f < self.drop_to) & (h < -self.drop_h)
        if dr.sum() >= self.min_points:
            out["drop"] = {"distance": float(np.percentile(f[dr], 10)), "depth": float(-np.median(h[dr]))}
        return out


def side(bearing):
    if abs(bearing) < 8:
        return "ahead"
    return f"slightly {'right' if bearing > 0 else 'left'}" if abs(bearing) < 20 else ("on your right" if bearing > 0
                                                                                      else "on your left")


def warning(res) -> str | None:
    """Spoken warning for one analysis, drop first (it is the more dangerous)."""
    if res.get("drop"):
        return f"Careful, drop ahead, {res['drop']['distance']:.1f} metres."
    ob = res.get("obstacle")
    if ob:
        what = "Step up" if ob["kind"] == "step up" else "Obstacle"
        return f"{what} {side(ob['bearing'])}, {ob['distance']:.1f} metres."
    return None


class Watcher:
    """Runs the Navigator on every new RealSense frame in a thread; calls `say(text)` when a warning appears or changes
    (same warning not repeated within `repeat` s). `state` holds the latest analysis for drawing."""

    def __init__(self, cam, say, repeat=6.0, hold=3, window=5, gap=1.5, max_hz=12.0, closer_m=0.5, **kw):
        self.cam, self.say, self.repeat, self.hold, self.window, self.gap = cam, say, repeat, hold, window, gap
        self.period, self.closer_m = 1.0 / max_hz, closer_m
        self.nav = Navigator(cam.depth_intrinsics, **kw)          # raw depth: the depth camera's own pixels
        self.state, self.stop, self.fps = {}, False, 0.0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def run(self):
        n, last, last_t, last_d, t0, k, recent = 0, None, 0.0, None, time.time(), 0, []
        while not self.stop:
            t_frame = time.time()
            try:
                depth_m, n = self.cam.latest_depth(after=n)
            except RuntimeError:
                return
            self.state = self.nav.analyse(depth_m)
            time.sleep(max(0.0, self.period - (time.time() - t_frame)))     # cap the rate: leave CPU to the VLM
            k += 1
            if k % 15 == 0:
                self.fps, t0 = 15 / (time.time() - t0), time.time()
            w = warning(self.state)
            key = None if w is None else w.split(" ")[0]          # Obstacle / Step / Careful: not the side or metres
            recent = (recent + [key])[-self.window:]
            if w is None or recent.count(key) < self.hold:      # debounce: the same warning in hold of window frames
                if recent.count(None) >= self.hold:
                    last = None
                continue
            now = time.time()
            if now - last_t < self.gap:
                continue
            hit = self.state.get("drop") or self.state.get("obstacle")
            d = hit["distance"] if hit else None
            closer = last_d is not None and d is not None and d < last_d - self.closer_m
            if key != last or now - last_t > self.repeat or closer:
                self.say(w)
                last, last_t, last_d = key, now, d


def main():
    from signdir.realsense import RealSense
    cam = RealSense()
    print(f"{cam.name}: obstacle check (Ctrl-C stops)", flush=True)
    w = Watcher(cam, lambda t: print("SAY:", t, flush=True)).start()
    try:
        while True:
            time.sleep(1.0)
            s = w.state
            ch = s.get("camera_height")
            print(f"{w.fps:4.1f} fps  floor {'yes, camera %.2f m up' % ch if ch else 'no'}  obstacle "
                  f"{s.get('obstacle')}  drop {s.get('drop')}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        w.stop = True
        cam.close()


if __name__ == "__main__":
    main()
