from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer, T5TokenizerFast

from .anima_dit import AnimaDiT
from .lora import fuse_lora
from .logging_utils import configure_logging, log_stage, logger, track_progress
from .profiling import configure_profiler, advance_profile_step, profile_range, profile_stage
from .text_encoder import ZImageTextEncoder
from .vae import WanVideoVAE
from .prompt_weights import tokenize_weighted_prompt


ANIMA_REPO = "circlestone-labs/Anima"
ANIMA_REVISION = "f973fc41ec7545364ac9776c2440285f43ff2a30"

WEIGHT_FILES = {
    "dit": "split_files/diffusion_models/anima-base-v1.0.safetensors",
    "text_encoder": "split_files/text_encoders/qwen_3_06b_base.safetensors",
    "vae": "split_files/vae/qwen_image_vae.safetensors",
}


def configure_diagnostics(config):
    from .model_cache import clear_model_cache

    configure_logging(config.get("logging"))
    if not config.get("use_model_cache", False):
        clear_model_cache()
    configure_profiler(config.get("profiler"))


@log_stage("모델 가중치 다운로드/캐시 확인")
def download_weights(cache_dir=None):
    paths = {}
    for name, filename in WEIGHT_FILES.items():
        logger.info("%s 가중치 확인: %s", name, filename)
        paths[name] = Path(
            hf_hub_download(
                repo_id=ANIMA_REPO,
                filename=filename,
                revision=ANIMA_REVISION,
                cache_dir=cache_dir,
            )
        )
        logger.info("%s 가중치 준비 완료", name)
    return paths


def _stream_load(model, path, map_key, skip_key=lambda key: False):
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    loaded = set()
    unexpected = []

    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        for source_key in track_progress(checkpoint.keys(), f"{model.__class__.__name__} 가중치"):
            if skip_key(source_key):
                continue
            target_key = map_key(source_key)
            target = targets.get(target_key)
            if target is None:
                unexpected.append((source_key, target_key))
                continue
            with profile_range("weights/read_and_copy"):
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


@log_stage("Qwen text encoder 로딩")
@profile_stage("load_text_encoder")
def load_text_encoder(path, device="cuda", dtype=torch.float16):
    logger.info("device=%s, dtype=%s", device, dtype)
    with profile_range("model/initialize"), torch.device(device):
        model = ZImageTextEncoder(model_size="0.6B")
        model = model.to(dtype=dtype).eval().requires_grad_(False)
    return _stream_load(
        model,
        path,
        map_key=lambda key: key,
        skip_key=lambda key: key == "lm_head.weight",
    )


@log_stage("Anima DiT 로딩")
@profile_stage("load_dit")
def load_dit(path, device="cuda", dtype=torch.float16, lora_path=None, lora_scale=1.0):
    logger.info("device=%s, dtype=%s", device, dtype)
    with profile_range("model/initialize"):
        model = AnimaDiT(device=device, dtype=dtype).eval().requires_grad_(False)
    model = _stream_load(
        model,
        path,
        map_key=lambda key: key.removeprefix("net."),
    )
    if lora_path:
        logger.info("LoRA 적용 시작: %s, scale=%s", lora_path, lora_scale)
        with profile_range("model/lora_fusion"):
            count = fuse_lora(model, lora_path, scale=lora_scale)
        logger.info("LoRA 적용 완료: Linear %d개", count)
    return model


@log_stage("VAE 로딩")
@profile_stage("load_vae")
def load_vae(path, device="cuda", dtype=torch.float16):
    logger.info("device=%s, dtype=%s", device, dtype)
    with profile_range("model/initialize"), torch.device(device):
        model = WanVideoVAE()
        model = model.to(dtype=dtype).eval().requires_grad_(False)
    return _stream_load(
        model,
        path,
        map_key=lambda key: "model." + key,
    )


@log_stage("Qwen/T5 tokenizer 로딩")
def load_tokenizers():
    qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    t5 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")
    return qwen, t5


@torch.inference_mode()
@log_stage("Qwen prompt encoding")
@profile_stage("encode_prompt")
def encode_prompt(
    text_encoder,
    qwen_tokenizer,
    t5_tokenizer,
    prompt,
    device="cuda",
    dtype=torch.float16,
    max_sequence_length=512,
    use_token_weights=False,
    return_token_weights=False,
):
    if use_token_weights:
        qwen_inputs, t5_ids, t5_weights = tokenize_weighted_prompt(
            qwen_tokenizer, t5_tokenizer, prompt, max_sequence_length
        )
    else:
        qwen_inputs = qwen_tokenizer(
            [prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        t5_inputs = t5_tokenizer(
            [prompt], max_length=max_sequence_length, truncation=True, return_tensors="pt"
        )
        t5_ids = t5_inputs.input_ids
        t5_weights = torch.ones((*t5_ids.shape, 1), dtype=torch.float32)
    input_ids = qwen_inputs.input_ids.to(device)
    attention_mask = qwen_inputs.attention_mask.to(device).bool()
    with profile_range("text/qwen_forward"):
        prompt_embeds = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1].to(dtype)

    t5_ids = t5_ids.to(device)
    if return_token_weights:
        return prompt_embeds, t5_ids, t5_weights.to(device=device, dtype=dtype)
    return prompt_embeds, t5_ids


@torch.inference_mode()
@log_stage("Anima text adapter conditioning")
@profile_stage("adapt_conditioning")
def adapt_conditioning(dit, prompt_embeds, t5_ids, t5_weights=None):
    with profile_range("text/anima_adapter"):
        return dit.preprocess_text_embeds(prompt_embeds, t5_ids, t5xxl_weights=t5_weights)


def z_image_schedule(steps, denoise=1.0, shift=3.0):
    sigmas = torch.linspace(float(denoise), 0.0, int(steps) + 1, dtype=torch.float32)[:-1]
    sigmas = float(shift) * sigmas / (1.0 + (float(shift) - 1.0) * sigmas)
    timesteps = sigmas * 1000.0
    return sigmas, timesteps


@torch.inference_mode()
@log_stage("Euler 이미지 생성")
@profile_stage("sample_euler", stepped=True)
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
    logger.info("생성 설정: %sx%s, steps=%s, CFG=%s, seed=%s", width, height, steps, cfg_scale, seed)
    generator = torch.Generator("cpu").manual_seed(int(seed))
    latents = torch.randn(
        (1, 16, int(height) // 8, int(width) // 8),
        generator=generator,
        device="cpu",
        dtype=dtype,
    ).to(device)

    sigmas, timesteps = z_image_schedule(steps, denoise=denoise, shift=shift)

    for i, timestep in enumerate(track_progress(timesteps, "Euler 생성", total=len(timesteps))):
        t = timestep.reshape(1).to(device=device, dtype=dtype) / 1000.0

        with profile_range("sampling/positive_dit"):
            positive_pred = dit(
                x=latents.unsqueeze(2),
                timesteps=t,
                context=positive,
                t5xxl_ids=None,
            ).squeeze(2)

        if float(cfg_scale) == 1.0:
            velocity = positive_pred
        else:
            with profile_range("sampling/negative_dit"):
                negative_pred = dit(
                    x=latents.unsqueeze(2),
                    timesteps=t,
                    context=negative,
                    t5xxl_ids=None,
                ).squeeze(2)
            with profile_range("sampling/cfg"):
                velocity = negative_pred + float(cfg_scale) * (positive_pred - negative_pred)

        sigma = sigmas[i].to(device=device, dtype=latents.dtype)
        sigma_next = (
            sigmas[i + 1].to(device=device, dtype=latents.dtype)
            if i + 1 < len(sigmas)
            else torch.zeros((), device=device, dtype=latents.dtype)
        )
        with profile_range("sampling/euler_update"):
            latents = latents + velocity * (sigma_next - sigma)
        logger.debug("생성 스텝 %d/%d 완료", i + 1, len(timesteps))
        advance_profile_step()

    return latents


@torch.inference_mode()
@log_stage("VAE 이미지 디코딩")
@profile_stage("decode_image")
def decode_image(vae, latents, device="cuda"):
    with profile_range("vae/decode"):
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
