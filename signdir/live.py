"""Live demo: Intel RealSense (default when one is plugged in), webcam or image files -> LocateAnything + sign adapter -> spoken directions.

Each frame is read like the training images (long side <= --long_side), the answer is parsed into
destinations + arrow directions, turned into a short command ("Exit: turn right. Platform 2: go straight.") and
spoken with speech-dispatcher (spd-say). The same command is not repeated within --repeat seconds. A window shows the
frame with the boxes (q, Esc or closing it quits); it uses Tkinter, since the installed OpenCV is the headless build.
With --depth (default indoor) Depth Anything V2 estimates each sign's distance (signdir/depth.py): it is drawn on the
box, printed and spoken ("Exit, 6 metres: turn right."), and signs are ordered nearest first. On the webcam the
calibration saved by `python -m signdir.depth calibrate` for that camera and scene is applied, and each sign's
distance is smoothed over frames.

  python -m signdir.live                                   RealSense if connected, else webcam 0; adapter
                                                           work_dirs/signdir_r16_all1536 (run 4), --task all
  python -m signdir.live --source webcam --camera 1 --no-show
  python -m signdir.live --image a.jpg b.jpg               files instead of the webcam
  python -m signdir.live --depth outdoor                   road signs (calibrate it too: signdir.depth calibrate)
  python -m signdir.live --depth none                      no distance
  python -m signdir.live --source realsense                Intel RealSense: colour -> VLM, aligned stereo depth ->
                                                           metric distance (no Depth Anything, no calibration), and
                                                           the bearing from its intrinsics ("Exit, 4 metres,
                                                           slightly to your right: turn right.")
  python -m signdir.live --task direction                  direction signs only
  python -m signdir.live --no-obstacles                    RealSense without the obstacle / drop-off warnings

With the RealSense, signdir.obstacle runs on every depth frame in its own thread (~15-30 fps, independent of the
~2 s sign model): an obstacle in the walking corridor or a drop in the floor ahead is spoken at once, with priority
over the sign sentences ("Obstacle slightly left, 0.8 metres." / "Careful, drop ahead, 1.2 metres.").
"""
from __future__ import annotations

import argparse
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from signdir import la
from signdir.build_sft import PROMPT, PROMPT_ALL, PROMPT_NAV

# up-left / up-right are spoken as "go straight": the teacher labels (and so the run-2 adapter) call most straight
# arrows up-left / up-right and never say straight; --literal_up speaks them as "bear left / right"
SAY = {"left": "turn left", "right": "turn right", "straight": "go straight", "back": "go back",
       "up-left": "go straight", "up-right": "go straight", "down-left": "down and to the left",
       "down-right": "down and to the right", "down": "go down", "up": "go up"}
LITERAL_UP = {"up-left": "bear left", "up-right": "bear right"}


def metres(d):
    """Distance as spoken / drawn: 0.1 m steps below 3 m, whole metres above."""
    return f"{d:.1f} metres" if d < 3 else f"{d:.0f} metres"


def where(b):
    """Bearing (deg, + = right of the optical axis) as spoken: coarse bands, a precise angle is not useful aloud."""
    if abs(b) < 10:
        return "ahead"
    side = "right" if b > 0 else "left"
    return f"slightly to your {side}" if abs(b) < 30 else f"to your {side}"


def named(it, text):
    """'Exit, 4 metres, slightly to your right': the sign, then where it is (distance, bearing when known)."""
    out = text
    if it.get("distance") is not None:
        out += f", {metres(it['distance'])}"
    if it.get("bearing") is not None:
        out += f", {where(it['bearing'])}"
    return out


# --task all: which signs are spoken first, and how a type is announced
TYPE_ORDER = {"warning": 0, "prohibition": 1, "mandatory": 2, "direction": 3, "handwritten": 4, "room": 5,
              "facility": 6, "information": 7}
TYPE_SAY = {"warning": "Warning: {}.", "prohibition": "{}.", "mandatory": "{}.", "facility": "{}.",
            "information": "{}.", "room": "{}.", "handwritten": "Note: {}."}


def spoken_text(text):
    """The Latin-script part of a bilingual sign ("No Smoking / धूम्रपान निषेध" -> "No Smoking"), else the text."""
    parts = [p.strip() for p in text.split(" / ") if p.strip()]
    latin = [p for p in parts if sum(c.isascii() and c.isalpha() for c in p) >= max(1, len(p) // 2)]
    return latin[0] if latin else (parts[0] if parts else text)


def command_all(items, max_items=3):
    """Spoken sentence for the all-signs task: warnings / prohibitions / mandatory signs first, then directions, then
    facilities and names; nearer signs first within a type."""
    def key(it):
        return (TYPE_ORDER.get(it.get("type"), 9), it.get("distance") is None, it.get("distance") or 0)
    out = []
    for it in sorted((it for it in items if it["text"].strip()), key=key)[:max_items]:
        name = named(it, spoken_text(it["text"]))
        if it["direction"] in SAY:
            out.append(f"{name}: {SAY[it['direction']]}.")
        else:
            out.append(TYPE_SAY.get(it.get("type"), "{}.").format(name))
    return " ".join(out)


def command_nav(items, max_items=2):
    """Spoken sentence for the navigation pass: stairs first (with up / down), then named obstacles, nearest first."""
    def key(it):
        return (it.get("type") != "stairs", it.get("distance") is None, it.get("distance") or 0)
    out = []
    for it in sorted(items, key=key)[:max_items]:
        if it.get("type") == "stairs":
            way = {"up": "going up", "down": "going down"}.get(it["direction"], "")
            out.append(named(it, f"Stairs {way}".strip()) + ".")
        else:
            out.append(named(it, it["text"] or "Obstacle") + ".")
    return " ".join(out)


def command(items, max_items=3):
    """Spoken sentence for the parsed items, nearest signs first (by estimated distance when there is one, else
    largest box); items without an arrow only if nothing has one."""
    items = sorted(items, key=lambda it: (it.get("distance") is None, it.get("distance") or 0,
                                          -(it["bbox"][2] - it["bbox"][0]) * (it["bbox"][3] - it["bbox"][1])))
    name = lambda it: named(it, it["text"])
    arrows = [it for it in items if it["direction"] in SAY and it["text"].strip()]
    if arrows:
        return " ".join(f"{name(it)}: {SAY[it['direction']]}." for it in arrows[:max_items])
    if items:
        return "I see " + "; ".join(name(it) for it in items[:max_items]) + ", no arrow."
    return ""


def describe(it):
    """One printed line per sign: text | arrow | type | distance | bearing."""
    parts = [it["text"], f"arrow {it['direction']}"]
    if it.get("type"):
        parts.append(it["type"])
    parts.append("distance ?" if it.get("distance") is None else f"{it['distance']:.2f} m")
    if it.get("bearing") is not None:
        parts.append(f"bearing {it['bearing']:+.1f} deg ({where(it['bearing'])})")
    return " | ".join(parts)


def realsense_connected():
    try:
        import pyrealsense2 as rs
        return len(rs.context().query_devices()) > 0
    except Exception:
        return False


def speak_now(text):
    """Obstacle warnings: speech-dispatcher's 'important' priority cuts into a sign sentence being spoken."""
    subprocess.Popen(["spd-say", "-P", "important", "-r", "10", text])


def speak(text, proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()  # newer command replaces one still being spoken
    return subprocess.Popen(["spd-say", "-w", "-r", "-10", text])


def draw(frame, items):
    for it in items:
        x1, y1, x2, y2 = (int(v) for v in it["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{it['text']} | {it['direction']}"  # cv2 draws ASCII only; non-Latin text shows as ?
        if it.get("type"):
            label += f" | {it['type']}"
        if it.get("distance") is not None:
            label += f" | {it['distance']:.1f} m"
        if it.get("bearing") is not None:
            label += f" | {it['bearing']:+.0f} deg"
        cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return frame


class Viewer:
    """Tkinter window showing BGR frames (opencv-python-headless has no cv2.imshow)."""

    def __init__(self, title="signdir"):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root)
        self.label.pack()
        self.key, self.closed = None, False
        self.root.bind("<Key>", lambda e: setattr(self, "key", e.keysym))
        self.root.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "closed", True))

    def show(self, frame):
        from PIL import ImageTk
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.thumbnail((1280, 800))
        self.photo = ImageTk.PhotoImage(img)  # keep a reference, or Tk shows nothing
        self.label.configure(image=self.photo)
        self.root.update()

    def wait(self, block=False):
        """False once the user quits (q / Esc / window closed); with block, waits for any key first."""
        self.key = None
        while True:
            self.root.update()
            if self.closed or self.key in ("q", "Escape"):
                self.root.destroy()
                return False
            if not block or self.key:
                return True
            time.sleep(0.02)


def draw_nav(frame, st):
    """Obstacle / drop state at the bottom of the frame."""
    if not st:
        return frame
    msgs = []
    if st.get("drop"):
        msgs.append((f"DROP {st['drop']['distance']:.1f} m", (0, 0, 255)))
    ob = st.get("obstacle")
    if ob:
        msgs.append((f"{ob['kind'].upper()} {ob['distance']:.1f} m {ob['bearing']:+.0f} deg", (0, 140, 255)))
    msgs.append(("floor: camera %.2f m up" % st["camera_height"] if st.get("camera_height") else "floor: not found",
                 (200, 200, 200)))
    y = frame.shape[0] - 15
    for text, col in msgs:
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        y -= 36
    return frame


NAV = {"watcher": None}


def frames(a):
    """(BGR frame, depth in metres aligned to it or None, column -> bearing function or None) per step: image files,
    the RealSense (depth + calibrated intrinsics) or the webcam."""
    if a.image:
        for f in a.image:
            yield cv2.imread(f), None, None
        return
    if a.source == "realsense":
        from signdir.realsense import RealSense
        cam = RealSense(a.rs_width, a.rs_height, a.rs_fps)
        print(f"realsense: {cam.name}, {a.rs_width}x{a.rs_height} colour + aligned depth", flush=True)
        if not a.no_obstacles:
            from signdir.obstacle import Watcher
            say = (lambda t: print("WARN:", t, flush=True)) if a.no_voice else (
                lambda t: (print("WARN:", t, flush=True), speak_now(t)))
            NAV["watcher"] = Watcher(cam, say).start()
            print("obstacles: depth corridor check running", flush=True)
        try:
            while True:
                yield (*cam.latest(), cam.bearing_deg)
        finally:
            if NAV["watcher"] is not None:
                NAV["watcher"].stop = True
            cam.close()
    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.width * 9 // 16)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {a.camera}")
    try:
        while True:
            for _ in range(4):  # drop frames buffered during the last (slow) inference
                cap.grab()
            ok, f = cap.read()
            if not ok:
                raise SystemExit("camera read failed")
            yield f, None, None
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="work_dirs/signdir_r16_all1536")
    ap.add_argument("--task", default="all", choices=["direction", "all"],
                    help="direction signs only, or every sign with its type (needs an adapter trained with build_sft --all)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280, help="requested webcam width (16:9)")
    ap.add_argument("--image", nargs="+", help="image files instead of the webcam")
    ap.add_argument("--long_side", type=int, default=1536, help="as in training (build_sft --long_side)")
    ap.add_argument("--mode", default="slow", choices=["slow", "hybrid", "fast"])
    ap.add_argument("--repeat", type=float, default=8.0, help="seconds before the same command is spoken again")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--frames", type=int, default=None, help="stop after this many frames")
    ap.add_argument("--literal_up", action="store_true", help="speak up-left / up-right as bear left / right")
    ap.add_argument("--depth", default="indoor",
                    help="Depth Anything V2 metric model for sign distance: indoor, outdoor, a checkpoint id, or none")
    ap.add_argument("--depth_scale", type=float, default=None,
                    help="fixed factor on estimated distances instead of the saved webcam calibration")
    ap.add_argument("--source", default="auto", choices=["auto", "realsense", "webcam"],
                    help="auto = the RealSense when one is connected, else the webcam. RealSense: colour for the VLM, "
                         "its aligned depth for distances and its intrinsics for the bearing (Depth Anything unused)")
    ap.add_argument("--realsense", action="store_true", help="same as --source realsense")
    ap.add_argument("--no-obstacles", action="store_true", help="RealSense: no obstacle / drop-off warnings")
    ap.add_argument("--nav_every", type=int, default=0,
                    help="also run the navigation prompt (stairs up/down, named obstacles) every N-th frame; needs an "
                         "adapter trained with build_sft --nav (0 = off)")
    ap.add_argument("--rs_width", type=int, default=1280)
    ap.add_argument("--rs_height", type=int, default=720)
    ap.add_argument("--rs_fps", type=int, default=30)
    a = ap.parse_args()
    if a.realsense:
        a.source = "realsense"
    if a.source == "auto":
        a.source = "realsense" if realsense_connected() else "webcam"
    if a.image:
        a.source = "files"
    print(f"source: {a.source}", flush=True)
    if a.literal_up:
        SAY.update(LITERAL_UP)
    model, processor, tokenizer = la.load(adapter=a.adapter)
    model.eval()
    depth = None
    if a.source == "realsense":
        from signdir.depth import Smoother
        smooth = Smoother()
        print("depth: RealSense stereo (metric, aligned to colour)", flush=True)
    elif a.depth != "none":
        from signdir.depth import DepthEstimator
        from signdir.depth import Smoother
        depth = DepthEstimator(a.depth, scale=a.depth_scale, camera=None if a.image else a.camera)
        smooth = Smoother() if not a.image else (lambda key, d: d)
        c = depth.calibration
        print(f"depth: {a.depth}, " + (f"webcam calibration true = {c['a']:.3f} * estimate^{c['b']:.3f} ({c['date']})"
                                       if c else f"scale {depth.a:g} (not calibrated: python -m signdir.depth "
                                       "calibrate)"), flush=True)
    last, last_t, voice = "", 0.0, None
    viewer = None if a.no_show else Viewer()
    for n, (frame, depth_m, bearing) in enumerate(frames(a)):
        if a.frames is not None and n >= a.frames:
            break
        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        s = min(1.0, a.long_side / max(rgb.size))
        if s < 1:
            rgb = rgb.resize((round(rgb.width * s), round(rgb.height * s)))
        t0 = time.time()
        nav = a.nav_every > 0 and n % a.nav_every == a.nav_every - 1
        prompt = PROMPT_NAV if nav else PROMPT_ALL if a.task == "all" else PROMPT
        answer = la.predict(model, processor, tokenizer, rgb, prompt, mode=a.mode)
        items = la.parse(answer, rgb.width, rgb.height)
        for it in items:
            it["bbox"] = [v / s for v in it["bbox"]]
        t1 = time.time()
        if bearing is not None:  # RealSense colour intrinsics: angle of the box centre from the optical axis
            for it in items:
                it["bearing"] = bearing((it["bbox"][0] + it["bbox"][2]) / 2)
        if depth_m is not None and items:  # RealSense: metric depth, already aligned to the colour frame
            from signdir.realsense import distance
            for it in items:
                it["distance"] = smooth(it["text"].strip().lower(), distance(depth_m, it["bbox"]))
        elif depth is not None and items:
            from signdir.depth import sign_distance
            dmap = depth.depth_map(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            for it in items:
                it["distance"] = smooth(it["text"].strip().lower(), sign_distance(dmap, it["bbox"]))
        cmd = command_nav(items) if nav else command_all(items) if a.task == "all" else command(items)
        print(f"{t1 - t0:.1f}s + depth {time.time() - t1:.2f}s  {cmd or '(no sign)'}", flush=True)
        for it in items:
            print("   " + describe(it), flush=True)
        if cmd and not a.no_voice and (cmd != last or time.time() - last_t > a.repeat):
            voice, last, last_t = speak(cmd, voice), cmd, time.time()
        if viewer is not None:
            if NAV["watcher"] is not None:
                draw_nav(frame, NAV["watcher"].state)
            viewer.show(draw(frame, items))
            if not viewer.wait(block=bool(a.image)):
                break
    if voice is not None:
        voice.wait()


if __name__ == "__main__":
    main()
