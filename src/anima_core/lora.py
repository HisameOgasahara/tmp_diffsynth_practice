"""Anima 추론용 LoRA 파일을 읽고 PyTorch 가중치에 합칩니다."""

import math
import re
from pathlib import Path

import torch
from safetensors.torch import load_file


_WEIGHT_KEY = re.compile(
    r"^(.*)\.(lora_A|lora_B|lora_down|lora_up)(?:\.default)?\.weight$"
)
_MODEL_PREFIXES = (
    "base_model.model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "net.",
)


def _resolve_layer(source_name, modules):
    name = source_name
    while True:
        if name in modules:
            return name
        prefix = next((p for p in _MODEL_PREFIXES if name.startswith(p)), None)
        if prefix is None:
            break
        name = name.removeprefix(prefix)

    # 실제 모듈 이름에서 별칭을 만들어 q_proj 등의 밑줄을 보존합니다.
    flattened = name.removeprefix("lora_unet_")
    matches = [n for n in modules if n.replace(".", "_") == flattened]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Anima 레이어를 찾을 수 없거나 이름이 모호합니다: {source_name}")


def _read_lora(path):
    path = Path(path).expanduser()
    if path.suffix.lower() != ".safetensors":
        raise ValueError("LoRA는 .safetensors 파일 경로로 전달하세요.")
    return load_file(str(path), device="cpu")


def _prepare_layers(model, state_dict):
    modules = dict(model.named_modules())
    groups = {}
    for key, tensor in state_dict.items():
        match = _WEIGHT_KEY.fullmatch(key)
        if match:
            source_name, kind = match.groups()
            side = "A" if kind in ("lora_A", "lora_down") else "B"
        elif key.endswith(".alpha"):
            source_name, side = key.removesuffix(".alpha"), "alpha"
        else:
            raise ValueError(f"지원하지 않는 LoRA 텐서입니다: {key}")
        group = groups.setdefault(source_name, {})
        if side in group:
            raise ValueError(f"중복된 LoRA 가중치입니다: {key}")
        group[side] = tensor

    layers = []
    resolved = set()
    for source_name, group in groups.items():
        if "A" not in group or "B" not in group:
            raise ValueError(f"LoRA A/B 쌍이 누락되었습니다: {source_name}")
        name = _resolve_layer(source_name, modules)
        if name in resolved:
            raise ValueError(f"동일 레이어에 중복된 LoRA 쌍이 있습니다: {name}")
        resolved.add(name)
        module = modules[name]
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"Linear 레이어용 LoRA만 지원합니다: {name}")

        down, up = group["A"], group["B"]
        if down.ndim != 2 or up.ndim != 2:
            raise ValueError(f"LoRA 행렬은 2차원이어야 합니다: {source_name}")
        rank = down.shape[0]
        if (
            rank == 0
            or down.shape[1] != module.in_features
            or up.shape != (module.out_features, rank)
        ):
            raise ValueError(
                f"LoRA 크기 불일치: {name}, A={tuple(down.shape)}, "
                f"B={tuple(up.shape)}, weight={tuple(module.weight.shape)}"
            )
        if not down.is_floating_point() or not up.is_floating_point():
            raise ValueError(f"LoRA 행렬은 부동소수점이어야 합니다: {source_name}")
        alpha = group.get("alpha")
        if alpha is not None and alpha.numel() != 1:
            raise ValueError(f"LoRA alpha는 스칼라여야 합니다: {source_name}")
        factor = float(alpha.item()) / rank if alpha is not None else 1.0
        if not math.isfinite(factor):
            raise ValueError(f"LoRA alpha가 유한하지 않습니다: {source_name}")
        layers.append((module, down, up, factor))

    if not layers:
        raise ValueError("파일에 적용할 LoRA A/B 가중치가 없습니다.")
    return layers


@torch.no_grad()
def fuse_lora(model, path, scale=1.0):
    """W += scale * (alpha / rank) * B @ A; alpha가 없으면 배율은 1입니다.

    DiffSynth GeneralLoRALoader의 변환 및 fusion 계산을 따릅니다.
    모델을 제자리에서 수정하며, 반환값은 적용한 Linear 레이어 수입니다.
    원본 가중치로 돌아가려면 기본 모델을 다시 로드하세요.
    """
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError("LoRA scale은 유한한 수여야 합니다.")
    state_dict = _read_lora(path)
    layers = _prepare_layers(model, state_dict)
    if scale == 0.0:
        return 0
    for module, down, up, factor in layers:
        # 레이어별 FP32 곱셈 후 모델 dtype으로 변환합니다.
        down = down.to(device=module.weight.device, dtype=torch.float32)
        up = up.to(device=module.weight.device, dtype=torch.float32)
        delta = (up @ down).mul_(scale * factor)
        module.weight.copy_((module.weight.float() + delta).to(module.weight.dtype))
    return len(layers)
