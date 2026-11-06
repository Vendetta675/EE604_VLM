"""Sweep camera-frame preprocessing (signdir.preprocess) on the RealSense sign test photos (test_real.jsonl, sources
realsense_sign / _handwritten / _room) with one model load. A spec ending in ":t2" also reads 2 x 2 zoomed tiles.

  python -m signdir.pre_sweep work_dirs/signdir_r16_nav1536 none,up1536,gamma+clahe,gamma+clahe+up1536
  -> one metrics line per spec, data/sft/eval_pre_<adapter>_<spec>.json
"""
import json, sys, time
from PIL import Image
from signdir import la, preprocess
from signdir.build_sft import SFT
from signdir.evaluate import score
adapter = sys.argv[1]
specs = sys.argv[2].split(",")
tests = [json.loads(l) for l in (SFT / "test_real.jsonl").open()]
tests = [t for t in tests if t["source"] in ("realsense_sign", "realsense_handwritten", "realsense_room")]
model, processor, tokenizer = la.load(adapter=adapter); model.eval()
for spec in specs:
    tiles = 2 if spec.endswith(":t2") else 0
    pre = spec.split(":")[0]
    rows, t0 = [], time.time()
    for t in tests:
        im = Image.open(SFT / "images" / t["image"]).convert("RGB")
        x, sc = preprocess.apply(im, pre)
        pred, ans = la.detect(model, processor, tokenizer, x, t["conversations"][0]["value"], mode="slow", tiles=tiles)
        for p in pred:
            p["bbox"] = [v / sc for v in p["bbox"]]
        rows.append({"image": t["image"], "source": t["source"], "reviewed": True, "answer": ans[0], "pred": pred,
                     "gt": la.parse(t["conversations"][1]["value"], im.width, im.height)})
    m = score(rows); dt = (time.time() - t0) / len(tests)
    print(f"{spec:28s} n {m['images']} gt {m['gt_items']} pred {m['pred_items']:3d}  det P {m['det_P']:.3f} R {m['det_R']:.3f} "
          f"F1 {m['det_F1']:.3f}  text {m['text_sim']:.3f} dir {m['direction_acc']:.3f} type {m['type_acc'] or 0:.3f} "
          f"e2e {m['e2e_F1']:.3f}  neg {m['neg_FP_rate']:.2f}  {dt:.1f}s/img", flush=True)
    json.dump({"metrics": m, "rows": rows}, open(SFT / f"eval_pre_{adapter.split('/')[-1]}_{spec.replace(':','_')}.json", "w"), ensure_ascii=False)
