"""Kaggle kernel for training on Kaggle's GPUs (2x T4, 16 GB each, float16: T4 has no bf16 tensor cores).

The data (data/sft train/test jsonl + the images they name, as images.tar) is the private dataset
gouravsingh2005/signdir-sft; signdir/ and eaglevl/ are embedded in the kernel script, so a code change needs only a
re-push. Writes .tmp/kaggle/kernel_<mode>/ (run.py + kernel-metadata.json); push it with
  .venv_data/bin/kaggle kernels push -p .tmp/kaggle/kernel_<mode> --accelerator NvidiaTeslaT4

  python -m signdir.kaggle smoke   bf16 vs fp16 smoke (time / loss / memory), 3 data-parallel steps, 4-image eval
  python -m signdir.kaggle train   run 3 (real + synthetic, 2 epochs, batch 8 over the GPUs) + slow-mode eval;
                                   the adapter, logs and eval json land in the kernel output
  python -m signdir.kaggle train4  run 4: from run 3's adapter (init/adapter.pt in the dataset), 1 epoch over the
                                   direction task + the all-signs task (build_sft --all), eval on test.jsonl and
                                   test_all.jsonl
  python -m signdir.kaggle train6  run 6: real signs first (signdir.build_run6: real rows x2, RealSense x3, small
                                   synthetic + nav replay), init = best earlier adapter, eval test_real only
  python -m signdir.kaggle train5  run 5: from run 4's adapter (init/adapter.pt), 1 epoch over room plates +
                                   handwritten notes (synth_paper), the navigation task (stairs up/down, obstacles;
                                   public_nav), the labelled RealSense photos, and 2,500 replayed run-4 rows
                                   (build_sft --all --paper --nav --realsense --replay 2500); eval on test_all,
                                   test_nav, test_real and test
"""
from __future__ import annotations

import base64
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "gouravsingh2005"
DATASET = f"{USER}/signdir-sft"
OUT = {"smoke": "smoke", "train": "signdir_r16_synth1536", "train4": "signdir_r16_all1536",
       "train5": "signdir_r16_nav1536", "train5fast": "signdir_r16_nav1536",
       "train6": "signdir_r16_real6"}

RUN = r'''
import glob, os, subprocess, sys, tarfile, base64, io, json, time
MODE, CODE, OUT = __MODE__, __CODE__, __OUT__
W = "/tmp/Embodied"
LOG = open("/kaggle/working/run.log", "a")


def sh(cmd, check=True, env=None):
    print("$", cmd, flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=True, cwd=W if os.path.isdir(W) else None, env={**os.environ, **(env or {})},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        sys.stdout.write(line)
        LOG.write(line)
        LOG.flush()
    p.wait()
    print(f"[{time.time() - t0:.0f} s, exit {p.returncode}]", flush=True)
    if check and p.returncode:
        raise SystemExit(f"failed: {cmd}")
    return p.returncode


import torch
n_gpu = torch.cuda.device_count()
print(torch.__version__, torch.version.cuda, sys.version.split()[0], n_gpu,
      [torch.cuda.get_device_name(i) for i in range(n_gpu)], flush=True)

# dependencies (torch stays Kaggle's)
open("/tmp/cons.txt", "w").write(f"torch=={torch.__version__}\n")
sh("pip install -q -c /tmp/cons.txt transformers==4.57.1 tokenizers==0.22.0 peft==0.12.0 accelerate==1.5.2 "
   "bitsandbytes")
# Kaggle's own datasets / pyarrow are fine (only imported); the rest one by one, best effort (Python 3.13 there)
for pkg in ["datasets", "bitstring==4.2.3", "sortedcontainers", "nvidia-ml-py", "python-dotenv", "einops", "timm",
            "filetype", "lmdb", "av", "ebmlite", "hjson", "py-cpuinfo"]:
    sh(f"pip install -q -c /tmp/cons.txt '{pkg}'", check=False)
sh("pip install -q --no-deps liger_kernel==0.3.1 decord", check=False)
sh("pip install -q --no-deps --no-build-isolation deepspeed==0.15.4", check=False)

# modules only the video / deepspeed / liger paths use: stub whatever did not install
STUBS = "/tmp/stubs"
for mod in ["decord", "deepspeed.comm", "liger_kernel.ops.cross_entropy", "ebmlite"]:
    if subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode == 0:
        continue
    print("stub", mod, flush=True)
    parts = mod.split(".")
    for k in range(1, len(parts) + 1):
        d = os.path.join(STUBS, *parts[:k])
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "__init__.py"), "w").write("def __getattr__(name):\n    return type(name, (), {})\n")

# code and data
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE))) as t:
    t.extractall(W)
src = os.path.dirname(glob.glob("/kaggle/input/**/train.jsonl", recursive=True)[0])
print("data", src, os.listdir(src), flush=True)
sft = f"{W}/data/sft"
os.makedirs(sft, exist_ok=True)
for f in ("train.jsonl", "test.jsonl", "test_all.jsonl", "test_nav.jsonl", "test_real.jsonl"):
    if os.path.exists(f"{src}/{f}"):
        sh(f"cp {src}/{f} {sft}/")
init = glob.glob("/kaggle/input/**/adapter.pt", recursive=True)  # init/adapter.pt (zipped dir may unpack nested)
print("init adapter:", init, flush=True)
if MODE == "train5fast":  # fits ~3.5 h: subsample run 5's train rows per group, RealSense photos twice
    import random
    rows = [json.loads(l) for l in open(f"{sft}/train.jsonl")]
    def grp(r):
        if r["source"].startswith("realsense"):
            return "realsense"
        if r["conversations"][0]["value"].startswith("Find the stairs"):
            return "nav"
        return "paper" if r["source"] == "synthetic_paper" else "replay"
    keep = {"replay": 800, "paper": 900, "nav": 1200}
    by = {}
    for r in rows:
        by.setdefault(grp(r), []).append(r)
    rng = random.Random(0)
    out = by.get("realsense", []) * 2
    for g, n in keep.items():
        rng.shuffle(by[g])
        out += by[g][:n]
    rng.shuffle(out)
    open(f"{sft}/train.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print("train5fast rows:", {g: len(v) for g, v in by.items()}, "->", len(out), flush=True)
if MODE in ("train4", "train5", "train5fast", "train6") and not init:
    raise SystemExit("init adapter.pt missing from the dataset (upload with --dir-mode zip)")
if init:
    os.makedirs("/tmp/init", exist_ok=True)
    sh(f"cp {init[0]} /tmp/init/adapter.pt")
if os.path.exists(f"{src}/images.tar"):
    os.makedirs(f"{sft}/images", exist_ok=True)
    sh(f"tar -xf {src}/images.tar -C {sft}/images")
else:  # Kaggle unpacked the archive
    d = [p for p in glob.glob(f"{src}/**/", recursive=True) if glob.glob(p + "*.jpg")][0]
    os.symlink(d, f"{sft}/images")
json.dump({"signdir": {"annotation": f"{sft}/train.jsonl", "root": f"{sft}/images", "repeat_time": 1.0,
                       "data_augment": False}}, open(f"{sft}/recipe.json", "w"), indent=2)
os.makedirs("/kaggle/working/work_dirs", exist_ok=True)
os.symlink("/kaggle/working/work_dirs", f"{W}/work_dirs")
os.environ.update(PYTHONPATH=f"{W}:{STUBS}", SIGNDIR_FP16="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  TOKENIZERS_PARALLELISM="false")
sh("python -c 'from signdir.la import snapshot; print(snapshot())'")
tr = f"torchrun --nproc_per_node {n_gpu} -m signdir.qlora" if n_gpu > 1 else "python -m signdir.qlora"
accum = max(1, 8 // n_gpu)  # batch 8 as in run 2

if MODE == "smoke":
    sh("python -m signdir.qlora --smoke 6", check=False, env={"SIGNDIR_FP16": "0"})
    sh("python -m signdir.qlora --smoke 6")
    sh(f"{tr} --out work_dirs/smoke --max_steps 3 --accum 2")
    sh("python -m signdir.evaluate --adapter work_dirs/smoke --mode slow --limit 4")
elif MODE == "train":
    sh(f"{tr} --out work_dirs/{OUT} --epochs 2 --accum {accum} --save 25")
    sh(f"python -m signdir.evaluate --adapter work_dirs/{OUT} --mode slow")
elif MODE == "train5":
    sh(f"{tr} --out work_dirs/{OUT} --epochs 1 --accum {accum} --save 25 --init /tmp/init")
    for t in ("test_real.jsonl", "test_nav.jsonl", "test_all.jsonl", "test.jsonl"):
        if os.path.exists(f"{sft}/{t}"):
            sh(f"python -m signdir.evaluate --adapter work_dirs/{OUT} --mode slow --test {t}", check=False)
elif MODE == "train6":  # train_run6.jsonl (signdir.build_run6) is packaged as train.jsonl; big evals run locally
    sh(f"{tr} --out work_dirs/{OUT} --epochs 1 --accum {accum} --save 25 --init /tmp/init")
    sh(f"python -m signdir.evaluate --adapter work_dirs/{OUT} --mode slow --test test_real.jsonl", check=False)
elif MODE == "train5fast":
    sh(f"{tr} --out work_dirs/{OUT} --epochs 1 --accum {accum} --save 25 --init /tmp/init")
    # one test set per GPU in parallel; test_all / test are evaluated locally on the downloaded adapter
    ps = [subprocess.Popen(f"python -m signdir.evaluate --adapter work_dirs/{OUT} --mode slow --test {t}"
                           f" > /kaggle/working/eval_{t}.log 2>&1", shell=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g % max(1, n_gpu))})
          for g, t in enumerate(("test_nav.jsonl", "test_real.jsonl"))]
    for p in ps:
        p.wait()
else:
    sh(f"{tr} --out work_dirs/{OUT} --epochs 1 --accum {accum} --save 25 --init /tmp/init")
    sh(f"python -m signdir.evaluate --adapter work_dirs/{OUT} --mode slow --test test_all.jsonl", check=False)
    sh(f"python -m signdir.evaluate --adapter work_dirs/{OUT} --mode slow", check=False)
sh(f"cp {sft}/eval_*.json /kaggle/working/", check=False)
print("RUN_DONE", flush=True)
'''


def code_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for d in ("signdir", "eaglevl"):
            t.add(ROOT / d, arcname=d, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    assert mode in OUT, mode
    d = ROOT / ".tmp" / "kaggle" / f"kernel_{mode}"
    d.mkdir(parents=True, exist_ok=True)
    run = RUN.replace("__MODE__", repr(mode)).replace("__OUT__", repr(OUT[mode])).replace("__CODE__", repr(code_b64()))
    (d / "run.py").write_text(run)
    meta = {"id": f"{USER}/signdir-{mode}", "title": f"signdir-{mode}", "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [],
            "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(d, f"{(d / 'run.py').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
