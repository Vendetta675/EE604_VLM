"""LocateAnything-3B in 4-bit (QLoRA) for a 6 GB GPU: loading, LoRA injection, prompting and output parsing.

The Qwen2.5-3B language model is quantized to NF4; the MoonViT vision encoder, the MLP projector and the (tied)
embedding / lm_head stay in bf16 (the training loss reads lm_head.weight directly, so it cannot be quantized).
LoRA adapters are injected in place (peft.inject_adapter_in_model), so the model's own forward / generate code
paths are unchanged; an adapter is just the state dict of the lora_ tensors.

  train=True   eaglevl classes (the training code's model, with the parallel-box-decoding training masks)
  train=False  the checkpoint's remote code (its hybrid fast/slow generate)

SIGNDIR_FP16=1 runs in float16 instead of bfloat16 (GPUs without bf16 tensor cores: Kaggle's T4 / P100). The model
loads on the current CUDA device (torch.cuda.set_device per rank for data-parallel training).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "nvidia/LocateAnything-3B"
SKIP = ["vision_model", "mlp1", "lm_head"]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DTYPE = torch.float16 if os.environ.get("SIGNDIR_FP16") == "1" else torch.bfloat16


def segment_sdpa(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    """MoonViT attention without the dense [L, L] block mask of its "sdpa" version (1 GB at 1024 px): each image's
    patches attend within that image, so run SDPA per segment, which picks the memory-efficient kernel."""
    import torch.nn.functional as F
    out = []
    for a, b in zip(q_cu_seqlens[:-1].tolist(), q_cu_seqlens[1:].tolist()):
        qs, ks, vs = (t[a:b].transpose(0, 1)[None] for t in (q, k, v))
        out.append(F.scaled_dot_product_attention(qs, ks, vs)[0].transpose(0, 1).reshape(b - a, -1))
    return torch.cat(out)


def patch_vit(model):
    """Point every loaded MoonViT module's "sdpa" entry at segment_sdpa."""
    import sys
    for name, mod in list(sys.modules.items()):
        if name.endswith("modeling_vit") and hasattr(mod, "VL_VISION_ATTENTION_FUNCTIONS"):
            mod.VL_VISION_ATTENTION_FUNCTIONS["sdpa"] = segment_sdpa


class _TorchFP16:
    """Stand-in for the torch module inside mask_sdpa_utils, whose additive attention masks are hard-coded bf16."""
    bfloat16 = torch.float16

    def __getattr__(self, name):
        return getattr(torch, name)


def patch_dtype():
    """Under SIGNDIR_FP16, make every loaded mask_sdpa_utils build float16 masks (SDPA needs the query dtype)."""
    import sys
    if DTYPE != torch.float16:
        return
    for name, mod in list(sys.modules.items()):
        if name.endswith("mask_sdpa_utils") and getattr(mod, "torch", None) is torch:
            mod.torch = _TorchFP16()


def device():
    return f"cuda:{torch.cuda.current_device()}"


def _no_flash_attn():
    """eaglevl's model __init__ hard-codes flash_attention_2 for MoonViT (not installed, no gain on a 3050): map any
    flash_attention_2 request to sdpa (then served by segment_sdpa)."""
    import transformers.modeling_utils as mu
    if getattr(mu.PreTrainedModel, "_signdir_patched", False):
        return
    orig = mu.PreTrainedModel._check_and_adjust_attn_implementation

    def check(self, attn_implementation, is_init_check=False):
        if attn_implementation == "flash_attention_2":
            attn_implementation = "sdpa"
        return orig(self, attn_implementation, is_init_check=is_init_check)
    mu.PreTrainedModel._check_and_adjust_attn_implementation = check
    mu.PreTrainedModel._signdir_patched = True


def snapshot():
    return snapshot_download(MODEL_ID, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")


def quant():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE,
                              bnb_4bit_use_double_quant=True, llm_int8_skip_modules=SKIP)


def lora_config(r=16, alpha=32, dropout=0.05):
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS)


def load(adapter: str | Path | None = None, train=False, r=16):
    path = snapshot()
    if train:
        _no_flash_attn()
        from eaglevl.model.locany.configuration_locateanything import LocateAnythingConfig
        from eaglevl.model.locany.modeling_locateanything import LocateAnythingForConditionalGeneration
        cfg = LocateAnythingConfig.from_pretrained(path)
        for c in (cfg, cfg.text_config):
            c._attn_implementation, c._attn_implementation_autoset = "sdpa", False
        cfg.vision_config._attn_implementation, cfg.vision_config._attn_implementation_autoset = "sdpa", False
        model = LocateAnythingForConditionalGeneration.from_pretrained(
            path, config=cfg, quantization_config=quant(), torch_dtype=DTYPE, device_map=device(),
            attn_implementation="sdpa")
        model.text_mask_token_id = cfg.text_config.text_mask_token_id
        model.language_model.block_size = int(cfg.text_config.block_size)
        model.language_model.causal_attn = cfg.text_config.causal_attn
    else:
        model = AutoModel.from_pretrained(path, trust_remote_code=True, quantization_config=quant(),
                                          torch_dtype=DTYPE, device_map=device(), attn_implementation="sdpa")
    patch_vit(model)
    patch_dtype()
    if adapter is not None or train:
        inject_adapter_in_model(lora_config(r=r), model.language_model)
        if adapter is not None:
            sd = torch.load(Path(adapter) / "adapter.pt", map_location="cpu")
            missing = model.language_model.load_state_dict(sd, strict=False)
            assert not missing.unexpected_keys, missing.unexpected_keys[:5]
        model.language_model.to(device())
    for n, p in model.named_parameters():
        p.requires_grad = train and "lora_" in n
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, processor, tokenizer


def lora_state(model):
    return {k: v.detach().cpu() for k, v in model.language_model.state_dict().items() if "lora_" in k}


@torch.no_grad()
def predict(model, processor, tokenizer, image, prompt, mode="hybrid", max_new_tokens=512):
    """Greedy answer of the remote-code model (as locateanything_worker.py, but deterministic)."""
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(msgs)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(device())
    # generate reads only the last <= 6 positions of the logits; computing (and upcasting to fp32) the logits of the
    # whole prefill (~2.4k tokens x 150k vocab at 1536 px) does not fit in 6 GB
    hook = model.language_model.lm_head.register_forward_pre_hook(lambda m, args: (args[0][:, -8:],))
    try:
        out = _generate(model, inputs, tokenizer, mode, max_new_tokens)
    finally:
        hook.remove()
    return out[0] if isinstance(out, tuple) else out


def _generate(model, inputs, tokenizer, mode, max_new_tokens):
    return model.generate(pixel_values=inputs["pixel_values"].to(DTYPE), input_ids=inputs["input_ids"],
                         attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws"),
                         tokenizer=tokenizer, max_new_tokens=max_new_tokens, use_cache=True, generation_mode=mode,
                         temperature=0.0, do_sample=False, verbose=False)


ITEM = re.compile(r"<ref>(.*?)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def _iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def parse(answer: str, w: int, h: int):
    """[{"text", "direction", "bbox"}] from '<ref>Delhi | left</ref><box><x1><y1><x2><y2></box>...' (pixels).
    Repeats of an earlier item (same text, box IoU > 0.5: the model sometimes loops) are dropped."""
    out = []
    for m in ITEM.finditer(answer):
        label = m.group(1)
        text, _, d = label.rpartition("|")
        text, d = (text.strip(), d.strip().lower()) if text else (label.strip(), "")
        x1, y1, x2, y2 = (int(g) for g in m.groups()[1:])
        it = {"text": text, "direction": d, "bbox": [x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h]}
        if not any(o["text"].lower() == text.lower() and _iou(o["bbox"], it["bbox"]) > 0.5 for o in out):
            out.append(it)
    return out
