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

## Usage

```bash
python -m signdir.live --adapter work_dirs/signdir_r16_synth1536            # webcam
python -m signdir.live --adapter work_dirs/... --image a.jpg b.jpg          # images
python -m signdir.evaluate --adapter work_dirs/... --mode slow              # test-set metrics
python -m signdir.qlora --out work_dirs/my_run --epochs 2 --accum 8         # train (6 GB GPU works)
python -m signdir.kaggle train                                               # build the Kaggle kernel
```

## Contributing

Work on a branch, open a pull request into `main`. Never commit `.env`, data, or `work_dirs/`.
