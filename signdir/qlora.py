"""QLoRA fine-tuning of LocateAnything-3B on data/sft (signdir.build_sft) on a single 6 GB GPU (or data-parallel).

Samples are built by the repo's own LazySupervisedDatasetMTP (the parallel-box-decoding targets of the official
recipe), one image per step (no packing), gradient checkpointing, LoRA on all linear layers of the language model,
gradient accumulation. The adapter (lora_ tensors only) is saved to <out>/adapter.pt every --save steps and at the end.

Under torchrun each rank takes every world-th sample of the same order and the LoRA gradients are averaged before each
optimizer step, so the batch is accum x world (Kaggle 2x T4: --accum 4 gives run 2's batch of 8). With SIGNDIR_FP16=1
the loss is scaled (GradScaler) and the LoRA weights stay in float32.

  python -m signdir.qlora --out work_dirs/signdir_r16 [--epochs 2 --lr 1e-4 --accum 8 --r 16]
  torchrun --nproc_per_node 2 -m signdir.qlora --out ... --accum 4
  python -m signdir.qlora --smoke [N]   forward/backward on the first N samples (default 1): time, loss, memory
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from signdir import la
from signdir.build_sft import SFT


def dataset(processor, tokenizer, max_len):
    import eaglevl.train.locany_finetune_magi_stream as F
    tokenizer.model_max_length = max_len
    processor.tokenizer = tokenizer
    meta = json.loads((SFT / "recipe.json").read_text())["signdir"]
    return F.LazySupervisedDatasetMTP("signdir", meta, processor, block_size=6)


def batch(item):
    """One unpacked sample in the layout of packed_collate_fn_mtp (sub_sample_lengths = [L], no attention mask)."""
    L = item["input_ids"].shape[0]
    grid = item["image_grid_hws"]
    grid = torch.as_tensor(grid)
    return dict(input_ids=item["input_ids"][None].cuda(), labels=item["labels"][None].cuda(),
                position_ids=item["position_ids"][None].cuda(), attention_mask=None,
                pixel_values=item["pixel_values"].to("cuda", la.DTYPE), image_flags=item["image_flags"].cuda(),
                image_grid_hws=grid.cuda(), sub_sample_lengths=[torch.tensor([L])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work_dirs/signdir_r16")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8, help="samples per optimizer step on each rank")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--save", type=int, default=100, help="optimizer steps between adapter saves")
    ap.add_argument("--max_steps", type=int, default=None, help="stop after this many optimizer steps (testing)")
    ap.add_argument("--smoke", type=int, nargs="?", const=1, default=0)
    a = ap.parse_args()
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(0)
    model, processor, tokenizer = la.load(train=True, r=a.r)
    model.language_model.config.use_cache = False
    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.train()
    model.vision_model.eval()  # frozen
    ds = dataset(processor, tokenizer, a.max_len)
    params = [p for p in model.parameters() if p.requires_grad]
    fp16 = la.DTYPE == torch.float16
    if fp16:
        for p in params:  # GradScaler cannot unscale float16 gradients
            p.data = p.data.float()
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0**10, enabled=fp16)
    print(f"{len(ds)} samples, {sum(p.numel() for p in params) / 1e6:.1f} M trainable params, {la.DTYPE}, "
          f"rank {rank}/{world}", flush=True)

    if a.smoke:
        torch.cuda.reset_peak_memory_stats()
        times = []
        for j in range(a.smoke):
            b = batch(ds[j])
            torch.cuda.synchronize()
            t0 = time.time()
            loss = model(**b).loss
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            print(f"smoke {j}: seq {b['input_ids'].shape[1]}, loss {loss.item():.4f}, {times[-1]:.1f} s", flush=True)
        grads = [p.grad for p in params if p.grad is not None]
        finite = all(torch.isfinite(g).all() for g in grads)
        print(f"smoke: mean {sum(times[1:] or times) / len(times[1:] or times):.2f} s/sample (after the first), "
              f"grads finite {finite}, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (out / "args.json").write_text(json.dumps({**vars(a), "world": world, "dtype": str(la.DTYPE)}, indent=2))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(ds) * a.epochs / (a.accum * world))
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    order = []
    rng = random.Random(0)
    while len(order) < total * a.accum * world:
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        order += idx
    order = order[:total * a.accum * world][rank::world]
    log = (out / "log.jsonl").open("a") if rank == 0 else None
    t0, run, n_skip = time.time(), 0.0, 0
    for i, j in enumerate(order):
        try:
            loss = model(**batch(ds[j])).loss / a.accum
            scaler.scale(loss).backward()
            run += loss.item()
        except torch.OutOfMemoryError:  # an unusually long sample: skip it, keep the accumulated gradients
            loss = None
            torch.cuda.empty_cache()
            n_skip += 1
        if (i + 1) % a.accum:
            continue
        if world > 1:  # every rank reaches this point after the same number of samples, skipped or not
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
                p.grad /= world
            stats = torch.tensor([run, n_skip], device="cuda", dtype=torch.float64)
            dist.all_reduce(stats)
            run, skipped = stats[0].item() / world, int(stats[1].item())
        else:
            skipped = n_skip
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step = (i + 1) // a.accum
        if rank == 0:
            rec = {"step": step, "loss": round(run, 4), "lr": sched.get_last_lr()[0], "skipped": skipped,
                   "min": round((time.time() - t0) / 60, 1)}
            if fp16:
                rec["scale"] = scaler.get_scale()
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 10 == 0 or step == a.max_steps:
                eta = (time.time() - t0) / step * (total - step) / 3600
                print(f"step {step}/{total} loss {run:.4f} lr {rec['lr']:.2e} ETA {eta:.1f} h", flush=True)
            if step % a.save == 0 or step == total:
                torch.save(la.lora_state(model), out / "adapter.pt")
        run = 0.0
        if step == a.max_steps:
            break
    if rank == 0:
        torch.save(la.lora_state(model), out / "adapter.pt")
        print("done", out / "adapter.pt", "skipped samples:", skipped)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
