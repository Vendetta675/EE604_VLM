"""Evaluate LocateAnything (base or + adapter) on data/sft/test.jsonl: destinations, their text and arrow direction.

Per image, predictions are matched one-to-one to ground-truth items (greedy by IoU, IoU >= IOU) and scored:
  det P/R/F1     box found (IoU >= IOU)
  text           mean normalized edit similarity of matched texts (1 = exact), and exact-match rate
  direction      accuracy of the direction on matched items
  e2e P/R/F1     a prediction counts only if box, text (similarity >= TEXT_OK) and direction are all right
  neg FP rate    images without a direction sign that got any prediction
Reported for all test images and for the human-reviewed ones only (once data/reviewed.jsonl exists).

  python -m signdir.evaluate [--adapter work_dirs/signdir_r16] [--mode hybrid|fast|slow] [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from signdir import la
from signdir.build_sft import PROMPT, SFT

IOU, TEXT_OK = 0.3, 0.8


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def similarity(a, b):
    a, b = " ".join(a.lower().split()), " ".join(b.lower().split())
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b))


def match(pred, gt):
    pairs = sorted(((iou(p["bbox"], g["bbox"]), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)),
                   reverse=True)
    used_p, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if v < IOU:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i), used_g.add(j)
            out.append((pred[i], gt[j]))
    return out


def score(rows):
    s = dict(n_pred=0, n_gt=0, det=0, e2e=0, text_sim=0.0, text_exact=0, dir_ok=0, neg=0, neg_fp=0)
    for r in rows:
        pred, gt = r["pred"], r["gt"]
        s["n_pred"] += len(pred)
        s["n_gt"] += len(gt)
        if not gt:
            s["neg"] += 1
            s["neg_fp"] += bool(pred)
        for p, g in match(pred, gt):
            sim = similarity(p["text"], g["text"])
            s["det"] += 1
            s["text_sim"] += sim
            s["text_exact"] += sim == 1.0
            s["dir_ok"] += p["direction"] == g["direction"]
            s["e2e"] += sim >= TEXT_OK and p["direction"] == g["direction"]
    f1 = lambda tp: 2 * tp / (s["n_pred"] + s["n_gt"]) if s["n_pred"] + s["n_gt"] else 0.0
    d = max(1, s["det"])
    return {"images": len(rows), "gt_items": s["n_gt"], "pred_items": s["n_pred"],
            "det_P": s["det"] / max(1, s["n_pred"]), "det_R": s["det"] / max(1, s["n_gt"]), "det_F1": f1(s["det"]),
            "text_sim": s["text_sim"] / d, "text_exact": s["text_exact"] / d, "direction_acc": s["dir_ok"] / d,
            "e2e_P": s["e2e"] / max(1, s["n_pred"]), "e2e_R": s["e2e"] / max(1, s["n_gt"]), "e2e_F1": f1(s["e2e"]),
            "neg_images": s["neg"], "neg_FP_rate": s["neg_fp"] / max(1, s["neg"])}


def confusion(rows):
    """Lines "truth -> {predicted: n}" for the directions of matched items."""
    c = {}
    for r in rows:
        for p, g in match(r["pred"], r["gt"]):
            c.setdefault(g["direction"], {}).setdefault(p["direction"], 0)
            c[g["direction"]][p["direction"]] += 1
    return [f"  {g:10s} -> " + str(dict(sorted(c[g].items(), key=lambda x: -x[1])))
            for g in sorted(c, key=lambda g: -sum(c[g].values()))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "fast", "slow"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    model, processor, tokenizer = la.load(adapter=a.adapter)
    model.eval()
    rows = []
    tests = [json.loads(l) for l in (SFT / "test.jsonl").open()][:a.limit]
    for k, t in enumerate(tests):
        im = Image.open(SFT / "images" / t["image"]).convert("RGB")
        ans = la.predict(model, processor, tokenizer, im, PROMPT, mode=a.mode)
        gt_ans = t["conversations"][1]["value"]
        rows.append({"image": t["image"], "source": t["source"], "reviewed": t["reviewed"], "answer": ans,
                     "pred": la.parse(ans, im.width, im.height), "gt": la.parse(gt_ans, im.width, im.height)})
        if k % 20 == 0:
            print(f"{k}/{len(tests)}", flush=True)
    name = Path(a.adapter).name if a.adapter else "base"
    out = SFT / f"eval_{name}_{a.mode}.json"
    res = {"all": score(rows), "reviewed": score([r for r in rows if r["reviewed"]])}
    res["by_source"] = {s: score([r for r in rows if r["source"] == s]) for s in sorted({r["source"] for r in rows})}
    out.write_text(json.dumps({"metrics": res, "rows": rows}, ensure_ascii=False, indent=1))
    for k in ("all", "reviewed"):
        m = res[k]
        print(f"{k:9s} imgs {m['images']:4d}  det F1 {m['det_F1']:.3f}  text sim {m['text_sim']:.3f} "
              f"exact {m['text_exact']:.3f}  dir acc {m['direction_acc']:.3f}  e2e F1 {m['e2e_F1']:.3f}  "
              f"neg FP {m['neg_FP_rate']:.2f}")
    print("direction confusion (truth -> predicted):", *confusion(rows), sep="\n")
    print("->", out)


if __name__ == "__main__":
    main()
