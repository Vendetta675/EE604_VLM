# EE604 VLM — sign-direction reading with LocateAnything

Course project built on NVIDIA's [LocateAnything-3B](https://github.com/NVlabs/Eagle/tree/main/Embodied)
(the `Embodied/` folder of NVlabs/Eagle, Apache-2.0; the original README is in
[README_LocateAnything.md](README_LocateAnything.md)). We QLoRA-fine-tune the model to find direction signs,
read their destinations and the arrow direction, and speak the instruction from a webcam.

Our code is in [`signdir/`](signdir):

| Module | What it does |
|---|---|
| `build_sft` | builds `data/sft/{train,test}.jsonl` from teacher-labelled images (`--synth` adds synthetic signs) |
| `synth` | renders synthetic direction signs (straight / left / right / ...) onto background images |
| `qlora` | NF4 QLoRA training (LoRA r16), single GPU or `torchrun` data-parallel, fp16 via `SIGNDIR_FP16=1` |
| `kaggle` | packs the code into a Kaggle kernel (2x T4) for training + evaluation |
| `evaluate` | detection F1, direction accuracy, end-to-end F1, direction confusion |
| `live` | webcam / image demo with spoken directions |

## Setup

```bash
python -m venv .venv_la && . .venv_la/bin/activate
pip install -e .
cp .env.example .env            # only if you use the Roboflow data tools
```

Data and adapters are not in git:
- data: private Kaggle dataset `gouravsingh2005/signdir-sft` (ask for access), unpack into `data/sft/`
  (`train.jsonl`, `test.jsonl`, `images/`)
- trained adapters (`work_dirs/<run>/adapter.pt`): shared separately (see the releases page)

## Results

Best adapter: **run 5** (`work_dirs/signdir_r16_nav1536`; Kaggle 2x T4, 1 epoch from run 4). Slow decoding, IoU >= 0.3;
details, the preprocessing sweep and the live RealSense test are in [`signdir/RESULTS.md`](signdir/RESULTS.md).

| Test set | det F1 | det recall | text exact | direction acc | type acc | e2e F1 |
|---|---|---|---|---|---|---|
| test_real: 68 own RealSense photos | 0.691 | 0.622 | 0.655 | 0.667 | 0.619 | 0.230 |
| test_nav: 220 stairs / obstacle images | 0.772 | 0.697 | 0.955 | 0.910 | 1.000 | 0.668 |
| test_all: 537 all-signs images | 0.791 | 0.736 | 0.833 | 0.946 | 0.942 | 0.656 |
| reviewed test: 247 hand-checked direction-sign images | 0.395 | 0.357 | 0.444 | 0.593 | - | 0.132 |

Reviewed test across runs (e2e F1 needs box, text and arrow all right):

| Model | det F1 | e2e F1 |
|---|---|---|
| base LocateAnything-3B | 0.105 | 0.000 |
| run 2: real images only | 0.300 | 0.085 |
| run 3: + synthetic signs | 0.295 | 0.092 |
| run 4: + all-signs task | 0.309 | 0.113 |
| run 4-up: trained on 1536 px enlarged images (eval at 1536 px) | 0.389 | 0.145 |
| run 5: + RealSense photos, door plates / handwritten, stairs + obstacles | 0.395 | 0.132 |

On RealSense frames, `--pre gamma+clahe+up1536` (contrast + enlarge to the training resolution) gives det F1 0.746 and
direction accuracy 0.680 vs 0.743 / 0.627 without preprocessing (52 sign photos).

## Usage

```bash
python -m signdir.live --source realsense --adapter work_dirs/signdir_r16_nav1536 \
    --pre gamma+clahe+up1536                                                # RealSense: signs + distance + obstacles
python -m signdir.live --adapter work_dirs/signdir_r16_synth1536            # webcam
python -m signdir.live --adapter work_dirs/... --image a.jpg b.jpg          # images
python -m signdir.evaluate --adapter work_dirs/... --mode slow              # test-set metrics
python -m signdir.qlora --out work_dirs/my_run --epochs 2 --accum 8         # train (6 GB GPU works)
python -m signdir.kaggle train                                               # build the Kaggle kernel
```

## Contributing

Work on a branch, open a pull request into `main`. Never commit `.env`, data, or `work_dirs/`.
