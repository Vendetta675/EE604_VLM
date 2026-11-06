"""Human review of the teacher's draft labels on the RealSense photos (signdir.label_capture), then the SFT rows.

  python -m signdir.review_capture make        -> data/review_capture.html (open in a browser; images from data/raw)
  python -m signdir.review_capture import ~/Downloads/reviewed_capture.json
        -> data/sft/realsense_{train,test}.jsonl (+ images in data/sft/images); then build_sft --realsense

The page is signdir.review's, with a type per item and the photo's tag shown. Photos with task "all" (tags room /
handwritten / sign) are labelled for the all-signs prompt (types in build_sft.TYPES); task "nav" (stairs-up /
stairs-down / obstacle) for the navigation prompt (types stairs / obstacle; stairs direction up / down).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from signdir.build_sft import NAV_TYPES, PROMPT_ALL, PROMPT_NAV, SFT, TYPES, answer
from signdir.pool import RAW, ROOT
from signdir.review import PAGE
from signdir.teacher import DIRECTIONS

DRAFT = ROOT / "realsense_draft.jsonl"
DIRS = list(DIRECTIONS) + ["up"]
HELP = {
    "all": "Signs: one box per sign (a direction board row = one box per destination + its arrow). Type room = a room "
           "or door number plate; handwritten = anything drawn or written by hand. Drawn arrow alone: text \"Arrow\" + "
           "its direction; arrow in a circle: \"Go straight\" / \"Turn left\" / \"Turn right\"; empty triangle: "
           "\"Warning\"; double-headed arrow: \"Both ways\"; written words as written (\"EXIT\").",
    "nav": "Stairs and obstacles: one box per staircase (type stairs, direction up / down) and per obstacle a walking "
           "person could bump into (type obstacle, text = its name: Chair, Plant, Dustbin ...; direction none).",
}


def page():
    p = PAGE.replace("signreview-v1", "capturereview-v1").replace("Direction-sign review", "RealSense photo review")
    p = p.replace("grid-template-columns:22px 1fr 110px 28px", "grid-template-columns:22px 1fr 96px 110px 28px")
    p = p.replace('const DATA = __DATA__, DIRS = __DIRS__,', 'const DATA = __DATA__, DIRS = __DIRS__, TYPES = __TYPES__,')
    p = p.replace("""    <select data-k="${k}" class="dr">""", """    <select data-k="${k}" class="ty">${TYPES[DATA[i].task].map(o => `<option ${o == t.type ? "selected" : ""}>${o}</option>`).join("")}</select>
    <select data-k="${k}" class="dr">""")
    p = p.replace("""if (e.target.classList.contains("dr")) { cur().items[k].direction = e.target.value; save(); render(); } };""",
                  """if (e.target.classList.contains("dr")) { cur().items[k].direction = e.target.value; save(); render(); }
  if (e.target.classList.contains("ty")) { cur().items[k].type = e.target.value; save(); render(); } };""")
    p = p.replace("""cur().items.push({text: "", direction: "none", bbox: b});""",
                  """cur().items.push({text: DATA[i].task == "nav" ? (DATA[i].tag.startsWith("stairs") ? "Stairs" : "") : "",
    direction: DATA[i].tag == "stairs-up" ? "up" : DATA[i].tag == "stairs-down" ? "down" : "none",
    type: DATA[i].task == "nav" ? (DATA[i].tag.startsWith("stairs") ? "stairs" : "obstacle") : (DATA[i].tag == "sign" ? "direction" : DATA[i].tag), bbox: b});""")
    p = p.replace("""$("meta").innerHTML = `${d.source} · ${d.image}`""",
                  """$("meta").innerHTML = `<b>tag ${d.tag}</b> · ${d.split} · ${d.image}<br><span class="help">${d.help}</span>`""")
    p = p.replace(""": "<p>No destinations (image has no direction sign).</p>";""", """: "<p>Nothing labelled (no sign / no stairs or obstacle).</p>";""")
    p = p.replace("""<button id="none">No direction sign</button>""", """<button id="none">Nothing here</button>""")
    p = p.replace("""Box = destination text + its arrow. Direction = where its arrow points ("straight" = up/ahead, "none" = no arrow).""",
                  """Direction = where an arrow points ("straight" = up/ahead, "none" = no arrow); for stairs up / down.""")
    p = p.replace('a.download = "reviewed.json"', 'a.download = "reviewed_capture.json"')
    return p


def make():
    data = []
    for r in map(json.loads, DRAFT.open()):
        data.append({"image": r["image"], "src": "raw/" + r["image"], "tag": r["tag"], "task": r["task"],
                     "split": r["split"], "width": r["width"], "height": r["height"], "help": HELP[r["task"]],
                     "items": [{"text": it["text"], "direction": it["direction"], "type": it.get("type", ""),
                                "bbox": it["bbox"]} for it in r["items"]]})
    html = (page().replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__DIRS__", json.dumps(DIRS))
            .replace("__TYPES__", json.dumps({"all": list(TYPES), "nav": list(NAV_TYPES)})))
    (ROOT / "review_capture.html").write_text(html)
    print(len(data), "photos ->", ROOT / "review_capture.html")


def import_(path):
    reviewed = {r["image"]: r["items"] for r in json.loads(Path(path).expanduser().read_text())}
    out = {"train": [], "test": []}
    for r in map(json.loads, DRAFT.open()):
        if r["image"] not in reviewed:
            continue
        items = [it for it in reviewed[r["image"]] if it["text"].strip() or it.get("type") == "obstacle"]
        for it in items:
            it["text"] = it["text"].strip() or "Obstacle"
        name = "realsense__" + Path(r["image"]).name
        shutil.copy(RAW / r["image"], SFT / "images" / name)
        out[r["split"]].append({"image": name, "source": f"realsense_{r['tag']}", "reviewed": True,
                                "conversations": [{"from": "human", "value": PROMPT_ALL if r["task"] == "all"
                                                   else PROMPT_NAV},
                                                  {"from": "gpt", "value": answer(items, r["width"], r["height"])}]})
    for sp, rows in out.items():
        (SFT / f"realsense_{sp}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        print(f"realsense_{sp}.jsonl: {len(rows)} photos")
    missing = sum(1 for _ in DRAFT.open()) - len(reviewed)
    if missing:
        print(f"note: {missing} drafted photos not marked done in the review (left out)")


if __name__ == "__main__":
    make() if sys.argv[1] == "make" else import_(sys.argv[2])
