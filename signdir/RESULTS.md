# signdir results

All numbers: slow decoding, IoU >= 0.3 matching (`signdir/evaluate.py`). Raw eval JSONs and adapters are not in the
repo (`data/`, `work_dirs/` are git-ignored; LocateAnything weights are non-commercial).

## Run 5 vs run 4 (2026-11-06)

Run 5 = `kaggle.py train5` (Kaggle 2xT4, FP16, init run 4 adapter, 1 epoch, 880 steps, 5.5 h): synthetic signs +
door plates / handwritten notes, stairs/obstacle task (`PROMPT_NAV`), 71 own RealSense photos, replay of the old data.
Adapter `work_dirs/signdir_r16_nav1536`.

| Test set | Metric | Run 4 | Run 5 |
|---|---|---|---|
| test_real (68 own RealSense photos)* | det F1 / recall | 0.554 / 0.437 | **0.691 / 0.622** |
| | direction / type acc | 0.525 / 0.424 | **0.667 / 0.619** |
| | e2e F1 / neg FP | 0.113 / 0 | **0.230 / 0** |
| test_nav (220, stairs + obstacles) | det F1 / e2e F1 | 0 / 0 | **0.772 / 0.668** |
| test_all (537) | det F1 / e2e F1 | 0.704 / 0.428 | **0.791 / 0.656** |
| | type acc | 0.731 | **0.942** |
| reviewed test (247, native res.) | det F1 / e2e F1 | 0.309 / 0.113 | **0.395 / 0.132** |

\* test_real labels corrected: 5 circled-arrow photos (`handwritten_0146`-`0150`) had been left unlabelled in review and
are now "Go straight | straight | handwritten" (this alone had shown run 5 a false-alert rate of 0.71).

Previous best on the reviewed test, `signdir_r16_up1536` at 1536 px: det F1 0.389, e2e 0.145.

## Camera-frame preprocessing (`signdir/preprocess.py`, `python -m signdir.pre_sweep`)

`live.py` never enlarged RealSense frames (1280x720), while training images are ~1536 px. Run 5 adapter, 52
RealSense sign / handwritten / room photos (106 signs):

| `--pre` | det F1 | recall | precision | direction | e2e F1 | s/img |
|---|---|---|---|---|---|---|
| none | 0.743 | 0.708 | 0.781 | 0.627 | 0.257 | 3.8 |
| up1536 | 0.714 | 0.670 | 0.763 | **0.732** | **0.281** | 4.1 |
| gamma+clahe | 0.734 | 0.689 | 0.785 | 0.685 | 0.261 | 2.6 |
| **gamma+clahe+up1536** | **0.746** | **0.708** | **0.789** | 0.680 | 0.279 | 3.8 |
| wb+gamma+clahe+sharp+up1536 | 0.734 | 0.689 | 0.785 | 0.685 | **0.281** | 3.8 |

Detection changes are within 1-2 signs (noise); enlarging clearly helps arrow direction. Chosen:
`--pre gamma+clahe+up1536`. Not yet run: sharp+up1536, up1920, up1536 with 2x2 tiles.
(Run 4 adapter, first 3 specs: det F1 0.659 / 0.634 / 0.640 - the detection gain came from run 5, not preprocessing.)

## Live RealSense test (run 5, `--pre gamma+clahe+up1536`)

    python -m signdir.live --source realsense --adapter work_dirs/signdir_r16_nav1536 --pre gamma+clahe+up1536

61 frames, ~3.3 s/frame (RTX 3050 6 GB), signs at 0.4-0.5 m. Read: Drinking Water | right | direction (7/8 frames),
Do Not Touch | prohibition, APRON MUST BE WORN | mandatory, WATERJET MACHINING | information, WC | facility,
PUSH | information, a hand-drawn arrow | right. Single-frame truncations ("Do Not T", "MUST BE WORN") recovered on the
next frame. No obstacle warnings fired: the floor must be visible 0.5-2.2 m below the camera.
