"""Human review of the teacher's labels on the test split (the clean test set).

  python -m signdir.review make      -> data/review.html (open it in a browser; it reads images from data/raw)
  python -m signdir.review import ~/Downloads/reviewed.json   -> data/reviewed.jsonl (then rerun signdir.build_sft)

In the page: fix the text, pick the direction, delete wrong items, drag on the image to add a missed destination,
"No direction sign" clears the image. Progress is kept in the browser (localStorage); "Export" downloads
reviewed.json with every image marked done.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from signdir.pool import ROOT
from signdir.teacher import DIRECTIONS

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Sign review</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#d9d9d4;--accent:#1f6feb;--bad:#c62828;--ok:#2e7d32}
body{margin:0;font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:2}
header b{font-size:15px} .sp{flex:1} button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--card);cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)} button:disabled{opacity:.4}
main{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:16px;padding:16px}
#wrap{position:relative;background:#222;border-radius:8px;overflow:hidden;align-self:start}
#wrap img{display:block;max-width:100%;max-height:calc(100vh - 110px);margin:auto;user-select:none}
#svg{position:absolute;left:0;top:0;width:100%;height:100%;cursor:crosshair}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;align-self:start}
.item{display:grid;grid-template-columns:22px 1fr 110px 28px;gap:6px;align-items:center;padding:6px 0;border-bottom:1px solid var(--line)}
.item input,.item select{font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:4px;width:100%;box-sizing:border-box}
.item.sel{background:#eef4ff} .num{font-weight:600;color:var(--accent)} .x{color:var(--bad);border:none;background:none;font-size:18px;padding:0}
.meta{color:var(--muted);font-size:12px;word-break:break-all;margin-bottom:8px} .warn{color:var(--bad)} .done{color:var(--ok)}
.help{color:var(--muted);font-size:12px;margin-top:10px;line-height:1.5}
</style></head><body>
<header><b>Direction-sign review</b><span id="pos"></span><span id="state"></span><span class="sp"></span>
<button id="prev">◀ Prev</button><button id="next" class="primary">Done ▶</button><button id="exp">Export</button></header>
<main><div id="wrap"><img id="img"><svg id="svg"></svg></div>
<div class="panel"><div class="meta" id="meta"></div><div id="items"></div>
<canvas id="zoom" width="396" height="220" style="width:100%;background:#222;border-radius:6px;margin-top:8px"></canvas>
<p><button id="none">No direction sign</button> <button id="reset">Reset to teacher</button></p>
<div class="help">Drag on the image to add a destination box. Click a box or row to select it.<br>
Keys: <b>Enter</b> done + next, <b>←/→</b> prev/next, <b>Del</b> delete selected (outside text fields).<br>
Box = destination text + its arrow. Direction = where its arrow points ("straight" = up/ahead, "none" = no arrow).</div></div></main>
<script>
const DATA = __DATA__, DIRS = __DIRS__, KEY = "signreview-v1";
let st = {}; try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let i = Math.max(0, DATA.findIndex(d => !(st[d.image] || {}).done)); if (i < 0) i = 0;
let sel = -1, drag = null;
const $ = id => document.getElementById(id), img = $("img"), svg = $("svg");
const cur = () => st[DATA[i].image] || (st[DATA[i].image] = {items: JSON.parse(JSON.stringify(DATA[i].items)), done: false});
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
function show() {
  const d = DATA[i]; img.src = d.src; sel = -1;
  $("pos").textContent = `${i + 1} / ${DATA.length}`;
  $("meta").innerHTML = `${d.source} · ${d.image}` + (d.disagree ? ` <span class="warn">· teacher's two reads disagreed: ${d.disagree}</span>` : "");
  const n = DATA.filter(x => (st[x.image] || {}).done).length;
  $("state").innerHTML = `<span class="done">${n} done</span>` + (cur().done ? " · this one ✓" : "");
  render();
}
function render() {
  const its = cur().items, W = DATA[i].width, H = DATA[i].height;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const sw = Math.max(W, H) / 300;
  svg.innerHTML = its.map((t, k) => { const [x1, y1, x2, y2] = t.bbox; const c = k == sel ? "#ffd400" : "#00e5ff";
    return `<rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" fill="none" stroke="${c}" stroke-width="${sw}" data-k="${k}"/>
            <text x="${x1}" y="${y1 - sw}" fill="${c}" font-size="${sw * 9}" font-weight="700">${k + 1}</text>`; }).join("");
  zoom();
  $("items").innerHTML = its.length ? its.map((t, k) => `<div class="item ${k == sel ? "sel" : ""}" data-k="${k}">
    <span class="num">${k + 1}</span><input value="${t.text.replace(/"/g, "&quot;")}" data-k="${k}" class="tx">
    <select data-k="${k}" class="dr">${DIRS.map(o => `<option ${o == t.direction ? "selected" : ""}>${o}</option>`).join("")}</select>
    <button class="x" data-k="${k}" title="delete">×</button></div>`).join("") : "<p>No destinations (image has no direction sign).</p>";
}
function zoom() {  // the selected item (else all items) enlarged
  const c = $("zoom"), g = c.getContext("2d"), its = cur().items; g.fillStyle = "#222"; g.fillRect(0, 0, c.width, c.height);
  if (!its.length || !img.complete || !img.naturalWidth) return;
  const bs = sel >= 0 ? [its[sel].bbox] : its.map(t => t.bbox);
  let [x1, y1, x2, y2] = [Math.min(...bs.map(b => b[0])), Math.min(...bs.map(b => b[1])), Math.max(...bs.map(b => b[2])), Math.max(...bs.map(b => b[3]))];
  const sx = img.naturalWidth / DATA[i].width, sy = img.naturalHeight / DATA[i].height, px = (x2 - x1) * .25 + 10, py = (y2 - y1) * .25 + 10;
  x1 = Math.max(0, x1 - px); y1 = Math.max(0, y1 - py); x2 = Math.min(DATA[i].width, x2 + px); y2 = Math.min(DATA[i].height, y2 + py);
  const k = Math.min(c.width / (x2 - x1), c.height / (y2 - y1)), w = (x2 - x1) * k, h = (y2 - y1) * k, ox = (c.width - w) / 2, oy = (c.height - h) / 2;
  g.drawImage(img, x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy, ox, oy, w, h);
  g.lineWidth = 2;
  its.forEach((t, n) => { const b = t.bbox; g.strokeStyle = n == sel ? "#ffd400" : "#00e5ff";
    g.strokeRect(ox + (b[0] - x1) * k, oy + (b[1] - y1) * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k); });
}
function rel(e) { const r = img.getBoundingClientRect(), W = DATA[i].width, H = DATA[i].height;
  return [Math.min(W, Math.max(0, (e.clientX - r.left) / r.width * W)), Math.min(H, Math.max(0, (e.clientY - r.top) / r.height * H))]; }
function fit() { const r = img.getBoundingClientRect(), w = $("wrap").getBoundingClientRect();
  Object.assign(svg.style, {left: r.left - w.left + "px", top: r.top - w.top + "px", width: r.width + "px", height: r.height + "px"}); }
img.onload = () => { fit(); zoom(); }; window.onresize = fit;
svg.onmousedown = e => { if (e.target.dataset.k !== undefined) { sel = +e.target.dataset.k; render(); return; } drag = rel(e); };
window.onmouseup = e => { if (!drag) return; const p = rel(e), b = [Math.min(drag[0], p[0]), Math.min(drag[1], p[1]), Math.max(drag[0], p[0]), Math.max(drag[1], p[1])];
  drag = null; if (b[2] - b[0] < 4 || b[3] - b[1] < 4) return;
  cur().items.push({text: "", direction: "none", bbox: b}); sel = cur().items.length - 1; save(); render();
  const inp = document.querySelector(`.tx[data-k="${sel}"]`); if (inp) inp.focus(); };
$("items").oninput = e => { const k = +e.target.dataset.k; if (e.target.classList.contains("tx")) cur().items[k].text = e.target.value; save(); };
$("items").onchange = e => { const k = +e.target.dataset.k; if (e.target.classList.contains("dr")) { cur().items[k].direction = e.target.value; save(); render(); } };
$("items").onclick = e => { if (e.target.classList.contains("x")) { cur().items.splice(+e.target.dataset.k, 1); sel = -1; save(); render(); }
  else if (e.target.dataset.k !== undefined && e.target.tagName == "DIV") { sel = +e.target.dataset.k; render(); } };
$("none").onclick = () => { cur().items = []; save(); render(); };
$("reset").onclick = () => { delete st[DATA[i].image]; save(); show(); };
$("prev").onclick = () => { if (i > 0) { i--; show(); } };
$("next").onclick = () => { cur().done = true; save(); if (i < DATA.length - 1) i++; show(); };
document.onkeydown = e => { const t = e.target.tagName; if (e.key == "Enter") { $("next").click(); e.preventDefault(); return; }
  if (t == "INPUT" || t == "SELECT") return;
  if (e.key == "ArrowLeft") $("prev").click(); else if (e.key == "ArrowRight") { if (i < DATA.length - 1) { i++; show(); } }
  else if ((e.key == "Delete" || e.key == "Backspace") && sel >= 0) { cur().items.splice(sel, 1); sel = -1; save(); render(); } };
$("exp").onclick = () => { const out = DATA.filter(d => (st[d.image] || {}).done).map(d => ({image: d.image, items: st[d.image].items}));
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(out)], {type: "application/json"}));
  a.download = "reviewed.json"; a.click(); };
show();
</script></body></html>"""


def make():
    rows = [json.loads(l) for l in (ROOT / "teacher.jsonl").open()]
    data = []
    for r in rows:
        if r["split"] != "test" or "error" in r:
            continue
        items = [{"text": it["text"], "direction": it.get("direction_crop") or it["direction"], "bbox": it["bbox"]}
                 for it in r["items"]]
        dis = "; ".join(f'{it["text"]}: {it["direction"]} vs {it["direction_crop"]}' for it in r["items"]
                        if it.get("direction_crop") not in (None, it["direction"]))
        data.append({"image": r["image"], "src": "raw/" + r["image"], "source": r["source"], "width": r["width"],
                     "height": r["height"], "items": items, "disagree": dis})
    html = PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__DIRS__", json.dumps(DIRECTIONS))
    (ROOT / "review.html").write_text(html)
    print(len(data), "test images ->", ROOT / "review.html")


def import_(path):
    reviewed = {r["image"]: r["items"] for r in json.loads(Path(path).expanduser().read_text())}
    rows = [json.loads(l) for l in (ROOT / "teacher.jsonl").open()]
    out = [{**{k: r[k] for k in ("image", "source", "split", "width", "height")},
            "items": [{"text": it["text"].strip(), "direction": it["direction"], "bbox": it["bbox"]}
                      for it in reviewed[r["image"]] if it["text"].strip()]}
           for r in rows if r["image"] in reviewed]
    (ROOT / "reviewed.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(len(out), "reviewed images ->", ROOT / "reviewed.jsonl")


if __name__ == "__main__":
    make() if sys.argv[1] == "make" else import_(sys.argv[2])
