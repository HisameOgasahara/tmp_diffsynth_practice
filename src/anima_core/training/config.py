"""TOML 학습 설정을 읽고 실행 가능한 값으로 확인합니다."""

from pathlib import Path
import json
import math
import tomllib


def load_config(path):
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    validate_config(config)
    return config


def write_config(path, config):
    validate_config(config)
    lines = []
    for section, values in config.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            encoded = json.dumps(value, ensure_ascii=False) if isinstance(value, (str, bool)) else str(value)
            lines.append(f"{key} = {encoded}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def resolve_preprocess_target_res(config):
    from .anima_image import ALLOWED_TARGET_RES

    max_pixels = config["dataset"]["max_pixels"]
    edge = math.isqrt(max_pixels)
    if edge * edge != max_pixels or edge not in ALLOWED_TARGET_RES:
        raise ValueError(f"max_pixels에는 anima-lora 해상도 기준 {ALLOWED_TARGET_RES}의 제곱을 지정하세요.")
    return [edge]


def validate_config(config):
    fields = {
        "dataset": {"max_pixels", "batch_size", "repeat", "caption_dropout_rate"},
        "lora": {"base_model", "rank", "alpha", "target_modules"},
        "training": {"optimizer", "learning_rate", "weight_decay", "lr_scheduler",
                     "warmup_ratio", "max_steps", "gradient_accumulation_steps", "seed",
                     "max_grad_norm", "sigmoid_scale", "sigmoid_bias"},
        "runtime": {"mixed_precision", "use_gradient_checkpointing"},
        "checkpoint": {"save_steps"},
    }
    if config.get("training", {}).get("optimizer") == "Muon":
        fields["training"] |= {"momentum", "nesterov", "ns_steps", "adjust_lr_fn"}
    if set(config) != set(fields):
        raise ValueError(f"설정 섹션은 {sorted(fields)}이어야 합니다.")
    for section, keys in fields.items():
        if set(config[section]) != keys:
            raise ValueError(f"{section} 설정 항목은 {sorted(keys)}이어야 합니다.")
    for section, key in (
        ("dataset", "max_pixels"), ("dataset", "batch_size"), ("dataset", "repeat"),
        ("lora", "rank"), ("training", "max_steps"),
        ("training", "gradient_accumulation_steps"), ("checkpoint", "save_steps"),
    ):
        value = config[section][key]
        if type(value) is not int or value < 1:
            raise ValueError(f"{section}.{key}는 양의 정수여야 합니다.")
    resolve_preprocess_target_res(config)
    for section, key, minimum, maximum in (
        ("dataset", "caption_dropout_rate", 0, 1),
        ("training", "warmup_ratio", 0, 1),
        ("training", "weight_decay", 0, math.inf),
        ("training", "learning_rate", 0, math.inf),
        ("lora", "alpha", 0, math.inf),
        ("training", "max_grad_norm", 0, math.inf),
        ("training", "sigmoid_scale", 0, math.inf),
        ("training", "sigmoid_bias", -math.inf, math.inf),
    ):
        value = config[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{section}.{key}는 유한한 수여야 합니다.")
        if not minimum <= value <= maximum:
            raise ValueError(f"{section}.{key} 범위: {minimum} ~ {maximum}")
    if config["training"]["learning_rate"] == 0 or config["lora"]["alpha"] == 0:
        raise ValueError("learning_rate와 alpha는 0보다 커야 합니다.")
    if config["training"]["warmup_ratio"] == 1:
        raise ValueError("warmup_ratio는 1보다 작아야 합니다.")
    if config["lora"]["base_model"] != "dit":
        raise ValueError("현재 학습 대상은 dit입니다.")
    if not isinstance(config["lora"]["target_modules"], str):
        raise ValueError("target_modules는 쉼표로 구분한 문자열이어야 합니다.")
    if config["training"]["optimizer"] not in {"AdamW", "Muon"}:
        raise ValueError("optimizer는 AdamW 또는 Muon입니다.")
    if config["training"]["optimizer"] == "Muon":
        training = config["training"]
        momentum = training["momentum"]
        if isinstance(momentum, bool) or not isinstance(momentum, (int, float)) or not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum은 0 이상 1 미만의 유한한 수여야 합니다.")
        if type(training["nesterov"]) is not bool:
            raise ValueError("nesterov는 bool이어야 합니다.")
        if type(training["ns_steps"]) is not int or not 1 <= training["ns_steps"] < 100:
            raise ValueError("ns_steps는 1 이상 100 미만의 정수여야 합니다.")
        if training["adjust_lr_fn"] not in {"original", "match_rms_adamw"}:
            raise ValueError("adjust_lr_fn은 original 또는 match_rms_adamw입니다.")
    if config["training"]["lr_scheduler"] not in {"cosine", "constant"}:
        raise ValueError("lr_scheduler는 cosine 또는 constant입니다.")
    if type(config["training"]["seed"]) is not int or config["training"]["seed"] < 0:
        raise ValueError("seed는 0 이상의 정수여야 합니다.")
    if config["runtime"]["mixed_precision"] not in {"bf16", "fp16", "no"}:
        raise ValueError("mixed_precision은 bf16, fp16, no 중 하나입니다.")
    if type(config["runtime"]["use_gradient_checkpointing"]) is not bool:
        raise ValueError("use_gradient_checkpointing은 bool이어야 합니다.")
    return config


def resolve_dtype(config, device):
    import torch

    precision = config["runtime"]["mixed_precision"]
    if precision == "fp16" and torch.device(device).type != "cuda":
        raise ValueError("fp16 학습은 CUDA에서 실행하세요.")
    if precision == "bf16" and torch.device(device).type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("현재 GPU는 bf16을 지원하지 않습니다. TOML 또는 노트북에서 fp16을 선택하세요.")
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[precision]
