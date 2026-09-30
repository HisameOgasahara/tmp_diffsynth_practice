from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer, T5TokenizerFast

from .anima_dit import AnimaDiT
from .lora import fuse_lora
from .text_encoder import ZImageTextEncoder
from .vae import WanVideoVAE


ANIMA_REPO = "circlestone-labs/Anima"
ANIMA_REVISION = "f973fc41ec7545364ac9776c2440285f43ff2a30"

WEIGHT_FILES = {
    "dit": "split_files/diffusion_models/anima-base-v1.0.safetensors",
    "text_encoder": "split_files/text_encoders/qwen_3_06b_base.safetensors",
    "vae": "split_files/vae/qwen_image_vae.safetensors",
}


def download_weights(cache_dir=None):
    paths = {}
    for name, filename in WEIGHT_FILES.items():
        paths[name] = Path(
            hf_hub_download(
                repo_id=ANIMA_REPO,
                filename=filename,
                revision=ANIMA_REVISION,
                cache_dir=cache_dir,
            )
        )
    return paths


def _stream_load(model, path, map_key, skip_key=lambda key: False):
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    loaded = set()
    unexpected = []

    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        for source_key in checkpoint.keys():
            if skip_key(source_key):
                continue
            target_key = map_key(source_key)
            target = targets.get(target_key)
            if target is None:
                unexpected.append((source_key, target_key))
                continue
            tensor = checkpoint.get_tensor(source_key)
            target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded.add(target_key)

    required = set(model.state_dict().keys())
    missing = sorted(required - loaded)
    if unexpected or missing:
        raise RuntimeError(
            f"state_dict mismatch: missing={missing[:12]}, unexpected={unexpected[:12]}"
        )
    return model


def load_text_encoder(path, device="cuda", dtype=torch.float16):
    with torch.device(device):
        model = ZImageTextEncoder(model_size="0.6B")
    model = model.to(dtype=dtype).eval().requires_grad_(False)
    return _stream_load(
        model,
        path,
        map_key=lambda key: key,
        skip_key=lambda key: key == "lm_head.weight",
    )


def load_dit(path, device="cuda", dtype=torch.float16, lora_path=None, lora_scale=1.0):
    model = AnimaDiT(device=device, dtype=dtype).eval().requires_grad_(False)
    model = _stream_load(
        model,
        path,
        map_key=lambda key: key.removeprefix("net."),
    )
    if lora_path:
        count = fuse_lora(model, lora_path, scale=lora_scale)
        print(f"LoRA 적용: {lora_path}, scale={lora_scale}, Linear {count}개")
    return model


def load_vae(path, device="cuda", dtype=torch.float16):
    with torch.device(device):
        model = WanVideoVAE()
    model = model.to(dtype=dtype).eval().requires_grad_(False)
    return _stream_load(
        model,
        path,
        map_key=lambda key: "model." + key,
    )


def load_tokenizers():
    qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    t5 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")
    return qwen, t5


@torch.inference_mode()
def encode_prompt(
    text_encoder,
    qwen_tokenizer,
    t5_tokenizer,
    prompt,
    device="cuda",
    dtype=torch.float16,
    max_sequence_length=512,
):
    qwen_inputs = qwen_tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = qwen_inputs.input_ids.to(device)
    attention_mask = qwen_inputs.attention_mask.to(device).bool()
    prompt_embeds = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states[-1].to(dtype)

    t5_inputs = t5_tokenizer(
        [prompt],
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    t5_ids = t5_inputs.input_ids.to(device)
    return prompt_embeds, t5_ids


@torch.inference_mode()
def adapt_conditioning(dit, prompt_embeds, t5_ids):
    return dit.preprocess_text_embeds(prompt_embeds, t5_ids)


def z_image_schedule(steps, denoise=1.0, shift=3.0):
    sigmas = torch.linspace(float(denoise), 0.0, int(steps) + 1, dtype=torch.float32)[:-1]
    sigmas = float(shift) * sigmas / (1.0 + (float(shift) - 1.0) * sigmas)
    timesteps = sigmas * 1000.0
    return sigmas, timesteps


@torch.inference_mode()
def sample_euler(
    dit,
    positive,
    negative,
    height,
    width,
    seed,
    steps,
    cfg_scale,
    denoise=1.0,
    shift=3.0,
    device="cuda",
    dtype=torch.float16,
):
    generator = torch.Generator("cpu").manual_seed(int(seed))
    latents = torch.randn(
        (1, 16, int(height) // 8, int(width) // 8),
        generator=generator,
        device="cpu",
        dtype=dtype,
    ).to(device)

    sigmas, timesteps = z_image_schedule(steps, denoise=denoise, shift=shift)

    for i, timestep in enumerate(timesteps):
        t = timestep.reshape(1).to(device=device, dtype=dtype) / 1000.0

        positive_pred = dit(
            x=latents.unsqueeze(2),
            timesteps=t,
            context=positive,
            t5xxl_ids=None,
        ).squeeze(2)

        if float(cfg_scale) == 1.0:
            velocity = positive_pred
        else:
            negative_pred = dit(
                x=latents.unsqueeze(2),
                timesteps=t,
                context=negative,
                t5xxl_ids=None,
            ).squeeze(2)
            velocity = negative_pred + float(cfg_scale) * (positive_pred - negative_pred)

        sigma = sigmas[i].to(device=device, dtype=latents.dtype)
        sigma_next = (
            sigmas[i + 1].to(device=device, dtype=latents.dtype)
            if i + 1 < len(sigmas)
            else torch.zeros((), device=device, dtype=latents.dtype)
        )
        latents = latents + velocity * (sigma_next - sigma)

    return latents


@torch.inference_mode()
def decode_image(vae, latents, device="cuda"):
    decoded = vae.decode(latents.unsqueeze(2), device=device).squeeze(2)
    pixels = ((decoded.float() + 1.0) / 2.0).clamp(0, 1)
    array = (
        pixels[0]
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    return Image.fromarray(np.asarray(array))
