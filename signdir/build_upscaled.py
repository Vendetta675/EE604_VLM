"""Training set for an indoor fine-tune at the inference resolution of la.detect(upscale=...).

Low-resolution indoor photos (taipei-station 512 px, wayfinding-sign ~470 px, part of subway-guide) reach the vision
encoder as a few hundred tokens; enlarging them at inference (la.detect upscale=1536) raised indoor detection F1
(0.367 -> 0.398 on the reviewed test set) but lowered text / direction accuracy, as the model never saw enlarged
images in training. This builds data/sft/train_up.jsonl (+ recipe_up.json) for a continuation run:

  every indoor sample of train.jsonl (both tasks) + a replay subset of the other sources (direction task, incl.
  the shop-front negatives) and of the synthetic all-signs task, with each image enlarged exactly as
  la._upscaled does (bicubic, long side >= --upscale) into data/sft/images_up/. Boxes are 0..1000 normalized,
  so the answers are unchanged.

  python -m signdir.build_upscaled [--upscale 1536] [--replay 400] [--replay_all 200]
  python -m signdir.qlora --recipe data/sft/recipe_up.json --init work_dirs/signdir_r16_all1536 --out ... --epochs 1
"""
from __future__ import annotations

import argparse
import json
import random

from PIL import Image

from signdir import la
from signdir.build_sft import PROMPT, SFT
from signdir.evaluate import INDOOR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upscale", type=int, default=1536)
    ap.add_argument("--replay", type=int, default=400, help="direction-task samples of non-indoor sources")
    ap.add_argument("--replay_all", type=int, default=200, help="synthetic all-signs samples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (SFT / "train.jsonl").open()]
    indoor = [r for r in rows if r["source"] in INDOOR]
    other = [r for r in rows if r["source"] not in INDOOR and r["source"] != "synthetic_signs"
             and r["conversations"][0]["value"] == PROMPT]
    synth_all = [r for r in rows if r["source"] == "synthetic_signs"]
    rng = random.Random(a.seed)
    out = indoor + rng.sample(other, min(a.replay, len(other))) + rng.sample(synth_all, min(a.replay_all, len(synth_all)))
    rng.shuffle(out)
    dst = SFT / "images_up"
    dst.mkdir(exist_ok=True)
    n_up = 0
    for r in out:
        p = dst / r["image"]
        if not p.exists():
            im = Image.open(SFT / "images" / r["image"]).convert("RGB")
            up = la._upscaled(im, a.upscale)
            n_up += up is not im
            up.save(p, quality=95)
    (SFT / "train_up.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    recipe = {"signdir": {"annotation": str(SFT / "train_up.jsonl"), "root": str(dst), "repeat_time": 1.0,
                          "data_augment": False}}
    (SFT / "recipe_up.json").write_text(json.dumps(recipe, indent=2))
    print(f"{len(out)} samples ({len(indoor)} indoor, replay {len(out) - len(indoor)}), {n_up} images enlarged "
          f"-> {SFT / 'train_up.jsonl'}")


if __name__ == "__main__":
    main()
