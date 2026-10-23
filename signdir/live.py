"""Live demo: webcam (or image files) -> LocateAnything + sign adapter -> spoken directions.

Each frame is read like the training images (long side <= --long_side), the answer is parsed into
destinations + arrow directions, turned into a short command ("Exit: turn right. Platform 2: go straight.") and
spoken with speech-dispatcher (spd-say). The same command is not repeated within --repeat seconds. A window shows the
frame with the boxes (q, Esc or closing it quits); it uses Tkinter, since the installed OpenCV is the headless build.

  python -m signdir.live                                   webcam 0, adapter work_dirs/signdir_r16_full1536
  python -m signdir.live --camera 1 --no-show
  python -m signdir.live --image a.jpg b.jpg               files instead of the webcam
"""
from __future__ import annotations

import argparse
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from signdir import la
from signdir.build_sft import PROMPT

# up-left / up-right are spoken as "go straight": the teacher labels (and so the run-2 adapter) call most straight
# arrows up-left / up-right and never say straight; --literal_up speaks them as "bear left / right"
SAY = {"left": "turn left", "right": "turn right", "straight": "go straight", "back": "go back",
       "up-left": "go straight", "up-right": "go straight", "down-left": "down and to the left",
       "down-right": "down and to the right", "down": "go down"}
LITERAL_UP = {"up-left": "bear left", "up-right": "bear right"}


def command(items, max_items=3):
    """Spoken sentence for the parsed items, largest (nearest) signs first; items without an arrow only if nothing
    has one."""
    items = sorted(items, key=lambda it: -(it["bbox"][2] - it["bbox"][0]) * (it["bbox"][3] - it["bbox"][1]))
    arrows = [it for it in items if it["direction"] in SAY and it["text"].strip()]
    if arrows:
        return " ".join(f"{it['text']}: {SAY[it['direction']]}." for it in arrows[:max_items])
    if items:
        return "I see " + ", ".join(it["text"] for it in items[:max_items]) + ", no arrow."
    return ""


def speak(text, proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()  # newer command replaces one still being spoken
    return subprocess.Popen(["spd-say", "-w", "-r", "-10", text])


def draw(frame, items):
    for it in items:
        x1, y1, x2, y2 = (int(v) for v in it["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{it['text']} | {it['direction']}"  # cv2 draws ASCII only; non-Latin text shows as ?
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


def frames(a):
    if a.image:
        for f in a.image:
            yield cv2.imread(f)
        return
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
            yield f
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="work_dirs/signdir_r16_full1536")
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
    a = ap.parse_args()
    if a.literal_up:
        SAY.update(LITERAL_UP)
    model, processor, tokenizer = la.load(adapter=a.adapter)
    model.eval()
    last, last_t, voice = "", 0.0, None
    viewer = None if a.no_show else Viewer()
    for n, frame in enumerate(frames(a)):
        if a.frames is not None and n >= a.frames:
            break
        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        s = min(1.0, a.long_side / max(rgb.size))
        if s < 1:
            rgb = rgb.resize((round(rgb.width * s), round(rgb.height * s)))
        t0 = time.time()
        answer = la.predict(model, processor, tokenizer, rgb, PROMPT, mode=a.mode)
        items = la.parse(answer, rgb.width, rgb.height)
        for it in items:
            it["bbox"] = [v / s for v in it["bbox"]]
        cmd = command(items)
        print(f"{time.time() - t0:.1f}s  {cmd or '(no sign)'}", flush=True)
        if cmd and not a.no_voice and (cmd != last or time.time() - last_t > a.repeat):
            voice, last, last_t = speak(cmd, voice), cmd, time.time()
        if viewer is not None:
            viewer.show(draw(frame, items))
            if not viewer.wait(block=bool(a.image)):
                break
    if voice is not None:
        voice.wait()


if __name__ == "__main__":
    main()
