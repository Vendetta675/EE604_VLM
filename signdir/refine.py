"""Zoomed re-read of predicted signs: for every predicted box of an eval json (signdir.evaluate), crop the box with
padding (arrows sit beside the text), enlarge it and ask the same adapter the row's own prompt again; the read whose
box best overlaps the original box (in crop coordinates) gives a second text / direction / type for the item.

Scored variants (boxes always the first pass's): orig, text (re-read text), dir (re-read direction), both.

  python -m signdir.refine --adapter work_dirs/signdir_r16_all1536 --eval data/sft/eval_..._slow_up1536.json
      [--test test.jsonl] [--limit N] [--pad 1.0] [--side 768]
  -> <eval>_refine.json (rows with "reread" per pred), metrics per variant
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from signdir import la
from signdir.build_sft import SFT
from signdir.evaluate import score


def crop_box(W, H, b, pad):
    x1, y1, x2, y2 = b
    h = max(1.0, y2 - y1)
    px, py = pad * h + 0.1 * (x2 - x1), 0.6 * h
    return (max(0, round(x1 - px)), max(0, round(y1 - py)), min(W, round(x2 + px)), min(H, round(y2 + py)))


def reread(model, processor, tokenizer, im, it, prompt, pad, side):
    W, H = im.size
    c = crop_box(W, H, it["bbox"], pad)
    cw, ch = c[2] - c[0], c[3] - c[1]
    if cw < 4 or ch < 4:
        return None
    crop = la._upscaled(im.crop(c), side)
    ans = la.predict(model, processor, tokenizer, crop, prompt, mode="slow", max_new_tokens=256)
    own = [it["bbox"][0] - c[0], it["bbox"][1] - c[1], it["bbox"][2] - c[0], it["bbox"][3] - c[1]]
    best, bv = None, 0.0
    for r in la.parse(ans, cw, ch, la.MAX_REPEAT):
        v = la._iou(r["bbox"], own)
        cx, cy = (r["bbox"][0] + r["bbox"][2]) / 2, (r["bbox"][1] + r["bbox"][3]) / 2
        if own[0] <= cx <= own[2] and own[1] <= cy <= own[3]:
            v += 0.5  # centre inside the original box
        if v > bv:
            best, bv = r, v
    return {"answer": ans, **({k: best[k] for k in ("text", "direction", "type") if k in best} if best else {})}


def variant(rows, use_text, use_dir):
    out = []
    for r in rows:
        preds = []
        for p in r["pred"]:
            q = dict(p)
            rr = p.get("reread") or {}
            if use_text and rr.get("text"):
                q["text"] = rr["text"]
            if use_dir and rr.get("direction"):
                q["direction"] = rr["direction"]
            preds.append(q)
        out.append({**r, "pred": preds})
    return out


def report(rows):
    for name, t, d in (("orig", 0, 0), ("text", 1, 0), ("dir", 0, 1), ("both", 1, 1)):
        m = score(variant(rows, t, d))
        print(f"{name:5s} det F1 {m['det_F1']:.3f}  text sim {m['text_sim']:.3f} exact {m['text_exact']:.3f}  "
              f"dir {m['direction_acc']:.3f}  e2e F1 {m['e2e_F1']:.3f}" +
              (f"  type {m['type_acc']:.3f}" if m["type_acc"] is not None else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pad", type=float, default=1.0, help="horizontal padding in box heights")
    ap.add_argument("--side", type=int, default=768, help="crops enlarged to at least this long side")
    ap.add_argument("--report", action="store_true", help="only score an existing *_refine.json")
    a = ap.parse_args()
    out = Path(a.eval).with_name(Path(a.eval).stem + "_refine.json")
    if a.report:
        report(json.load(open(out))["rows"])
        return
    prompts = {t["image"]: t["conversations"][0]["value"] for t in map(json.loads, (SFT / a.test).open())}
    rows = json.load(open(a.eval))["rows"][:a.limit]
    model, processor, tokenizer = la.load(a.adapter)
    model.eval()
    for k, r in enumerate(rows):
        im = Image.open(SFT / "images" / r["image"]).convert("RGB")
        for p in r["pred"]:
            p["reread"] = reread(model, processor, tokenizer, im, p, prompts[r["image"]], a.pad, a.side)
        if k % 20 == 0:
            print(f"{k}/{len(rows)}", flush=True)
            json.dump({"rows": rows[:k + 1]}, open(out, "w"), ensure_ascii=False)
    json.dump({"rows": rows}, open(out, "w"), ensure_ascii=False)
    report(rows)
    print("->", out)


if __name__ == "__main__":
    main()
