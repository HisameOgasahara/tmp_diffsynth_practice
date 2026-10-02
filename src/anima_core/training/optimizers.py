"""학습 설정으로 AdamW 또는 LoRA A/B 행렬용 Muon을 생성합니다."""

import torch


def create_optimizer(parameters, training):
    options = {"lr": training["learning_rate"], "weight_decay": training["weight_decay"]}
    if training["optimizer"] == "AdamW":
        return torch.optim.AdamW(parameters, **options)
    if training["optimizer"] == "Muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("Muon을 지원하는 PyTorch가 필요합니다. PyTorch 2.11 이상 환경을 사용하세요.")
        return torch.optim.Muon(
            parameters, **options, momentum=training["momentum"],
            nesterov=training["nesterov"], ns_steps=training["ns_steps"],
            adjust_lr_fn=training["adjust_lr_fn"],
        )
    raise ValueError(f"지원하지 않는 optimizer: {training['optimizer']}")
