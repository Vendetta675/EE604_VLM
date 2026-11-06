"""Post-processing of predicted sign items (pixel boxes): the model often splits one labelled destination into
fragments ("Puertas Gates" + "D,F") or repeats a fragment inside a full read. Rules:
  contain   drop an item lying inside another (> CONTAIN of its area) whose text already contains its text
  hmerge    join items on one line (vertical overlap >= VOV of the lower height, similar heights, horizontal gap
            <= HGAP line heights) when at most one of them has an arrow
  vmerge    join stacked lines (horizontal overlap >= HOV of the narrower, vertical gap <= VGAP line heights, similar
            heights) when the lower one has no arrow (direction boards stack separate destinations, each with an arrow)
Merged text is in reading order; the direction is the arrowed item's (else none); type the first item's.

  python -m signdir.merge data/sft/eval_..._slow_up1536.json   tune on even images, report on odd + all
"""
from __future__ import annotations

import itertools
import json
import sys

CONTAIN = 0.8


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _norm(t):
    return " ".join(t.lower().split())


def _join(a, b, sep=" "):
    first, second = (a, b) if (a["bbox"][1], a["bbox"][0]) <= (b["bbox"][1], b["bbox"][0]) else (b, a)
    d = next((x["direction"] for x in (first, second) if x["direction"] not in ("", "none")), first["direction"])
    out = {**first, "text": f"{first['text']}{sep}{second['text']}".strip(), "direction": d,
           "bbox": [min(a["bbox"][0], b["bbox"][0]), min(a["bbox"][1], b["bbox"][1]),
                    max(a["bbox"][2], b["bbox"][2]), max(a["bbox"][3], b["bbox"][3])]}
    return out


def _arrow(it):
    return it["direction"] not in ("", "none")


def _hmerge_ok(a, b, hgap, vov):
    ha, hb = a["bbox"][3] - a["bbox"][1], b["bbox"][3] - b["bbox"][1]
    if min(ha, hb) <= 0 or max(ha, hb) / min(ha, hb) > 1.6 or (_arrow(a) and _arrow(b)):
        return False
    ov = min(a["bbox"][3], b["bbox"][3]) - max(a["bbox"][1], b["bbox"][1])
    gap = max(a["bbox"][0], b["bbox"][0]) - min(a["bbox"][2], b["bbox"][2])
    return ov >= vov * min(ha, hb) and gap <= hgap * max(ha, hb)


def _vmerge_ok(a, b, vgap, hov):
    up, lo = (a, b) if a["bbox"][1] <= b["bbox"][1] else (b, a)
    hu, hl = up["bbox"][3] - up["bbox"][1], lo["bbox"][3] - lo["bbox"][1]
    if min(hu, hl) <= 0 or max(hu, hl) / min(hu, hl) > 1.8 or _arrow(lo):
        return False
    wu, wl = up["bbox"][2] - up["bbox"][0], lo["bbox"][2] - lo["bbox"][0]
    ov = min(up["bbox"][2], lo["bbox"][2]) - max(up["bbox"][0], lo["bbox"][0])
    gap = lo["bbox"][1] - up["bbox"][3]
    return ov >= hov * min(wu, wl) and -0.3 * min(hu, hl) <= gap <= vgap * min(hu, hl)


def merge(items, contain=True, hgap=0.0, vov=0.6, vgap=0.0, hov=0.5):
    """hgap / vgap 0 switch hmerge / vmerge off."""
    items = [dict(it) for it in items]
    if contain:
        keep = []
        for i, b in enumerate(items):
            inside = any(j != i and _area(a["bbox"]) > _area(b["bbox"]) and
                         _inter(a["bbox"], b["bbox"]) > CONTAIN * max(1e-6, _area(b["bbox"])) and
                         _norm(b["text"]) in _norm(a["text"]) for j, a in enumerate(items))
            if not inside:
                keep.append(b)
        items = keep
    for on, ok, arg in ((hgap > 0, _hmerge_ok, (hgap, vov)), (vgap > 0, _vmerge_ok, (vgap, hov))):
        if not on:
            continue
        changed = True
        while changed:
            changed = False
            for i, j in itertools.combinations(range(len(items)), 2):
                if ok(items[i], items[j], *arg):
                    m = _join(items[i], items[j])
                    items = [x for k, x in enumerate(items) if k not in (i, j)] + [m]
                    changed = True
                    break
    return items


def apply(rows, **kw):
    return [{**r, "pred": merge(r["pred"], **kw)} for r in rows]


def main():
    from signdir.evaluate import score
    rows = json.load(open(sys.argv[1]))["rows"]
    tune, hold = rows[0::2], rows[1::2]
    grid = [dict(contain=c, hgap=h, vgap=v) for c in (False, True) for h in (0, 0.5, 1.0, 2.0) for v in (0, 0.5, 1.0)]
    res = sorted(((score(apply(tune, **kw))["det_F1"], score(apply(tune, **kw))["e2e_F1"], i)
                  for i, kw in enumerate(grid)), reverse=True)
    print("tune half (det F1, e2e F1, params), top 5:")
    for d, e, i in res[:5]:
        print(f"  {d:.3f} {e:.3f} {grid[i]}")
    best = grid[res[0][2]]
    for name, part in (("held-out half", hold), ("all", rows)):
        a, b = score(part), score(apply(part, **best))
        print(f"{name:14s} det F1 {a['det_F1']:.3f} -> {b['det_F1']:.3f}  (P {a['det_P']:.3f}->{b['det_P']:.3f} "
              f"R {a['det_R']:.3f}->{b['det_R']:.3f})  e2e F1 {a['e2e_F1']:.3f} -> {b['e2e_F1']:.3f}  "
              f"text {a['text_sim']:.3f}->{b['text_sim']:.3f}")
    print("best:", best)


if __name__ == "__main__":
    main()
