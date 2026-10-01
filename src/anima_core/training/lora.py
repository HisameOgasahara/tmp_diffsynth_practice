"""DiffSynth의 대상 탐색과 표준 A/B LoRA를 PyTorch만으로 구성합니다.

참고: DiffSynth-Studio/diffsynth/diffusion/training_module.py (Apache-2.0).
DiffSynth가 PEFT에 맡기는 A의 Kaiming 초기화와 B의 0 초기화를 직접 구현합니다.
"""

import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False,
                                device=base.weight.device, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False,
                                device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, inputs):
        output = self.base(inputs)
        update = self.lora_B(self.lora_A(inputs.to(self.lora_A.weight.dtype)))
        return output + update.to(output.dtype) * (self.alpha / self.rank)


def find_target_modules(model, search_for_linear=False, prefix=""):
    # DiffSynth auto_detect_lora_target_modules의 ModuleList/차원 선택 규칙.
    if search_for_linear:
        return [
            f"{prefix}.{name}".strip(".")
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and min(module.in_features, module.out_features) >= 512
        ]
    targets = []
    for name, module in model.named_children():
        if name == "llm_adapter":
            continue
        path = f"{prefix}.{name}".strip(".")
        targets.extend(find_target_modules(
            module, isinstance(module, nn.ModuleList) and len(module) > 1, path
        ))
    return targets


def inject_lora(model, config):
    requested = config["target_modules"]
    if requested:
        suffixes = [name.strip() for name in requested.split(",") if name.strip()]
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, nn.Linear)
                   and not name.startswith("llm_adapter.")
                   and any(name == suffix or name.endswith("." + suffix) for suffix in suffixes)]
        missing = [suffix for suffix in suffixes
                   if not any(name == suffix or name.endswith("." + suffix) for name in targets)]
        if missing:
            raise ValueError(f"LoRA 대상 레이어가 없습니다: {missing}")
    else:
        targets = find_target_modules(model)
    if not targets:
        raise ValueError("LoRA 대상 Linear 레이어가 없습니다.")
    model.requires_grad_(False)
    for path in targets:
        parent_path, _, leaf = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf, LoRALinear(model.get_submodule(path), config["rank"], config["alpha"]))
    return targets


def collect_lora_weights(model):
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            weights[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu().contiguous()
            weights[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu().contiguous()
            weights[f"{name}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
    return weights


def restore_lora_weights(model, weights):
    expected = collect_lora_weights(model)
    if weights.keys() != expected.keys():
        raise ValueError("LoRA 체크포인트의 학습 대상 레이어가 현재 모델과 다릅니다.")
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            if float(weights[f"{name}.alpha"]) != module.alpha:
                raise ValueError(f"LoRA alpha 불일치: {name}")
            for side in ("A", "B"):
                source = weights[f"{name}.lora_{side}.weight"]
                target = getattr(module, f"lora_{side}").weight
                if source.shape != target.shape:
                    raise ValueError(f"LoRA rank/크기 불일치: {name}")
                target.copy_(source)
