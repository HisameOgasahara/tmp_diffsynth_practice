"""생성용 LoRA와 optimizer/RNG/데이터 진행 상태를 함께 저장합니다."""

import json
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch

from .config import write_config
from .lora import collect_lora_weights, restore_lora_weights


def save_checkpoint(output_dir, model, optimizer, scheduler, scaler, step, stream, config, fingerprint, loss_recorder):
    target = Path(output_dir) / f"step-{step:07d}"
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"저장 경로가 이미 있습니다: {target}")
    partial.mkdir(parents=True)
    save_file(collect_lora_weights(model), str(partial / "lora.safetensors"),
              metadata={"format": "anima_lora", "step": str(step)})
    state = {
        "version": 1, "step": step, "config": config, "fingerprint": fingerprint,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "stream": stream.state_dict(),
        "loss_recorder": loss_recorder.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(state, partial / "state.pt")
    write_config(partial / "training.toml", config)
    (partial / "complete.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    partial.rename(target)
    return target


def load_checkpoint(path, model, optimizer, scheduler, scaler, config, fingerprint, device):
    path = Path(path)
    if not (path / "complete.json").is_file():
        raise ValueError("저장이 완료된 step 폴더를 지정하세요.")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if state["version"] != 1 or state["config"] != config:
        raise ValueError("재개 설정이 저장 당시 설정과 다릅니다. 저장된 training.toml을 사용하세요.")
    if state["fingerprint"] != fingerprint:
        raise ValueError("데이터 캐시 또는 기본 모델이 저장 당시와 다릅니다.")
    restore_lora_weights(model, load_file(str(path / "lora.safetensors")))
    optimizer.load_state_dict(state["optimizer"])
    # FP32 LoRA 파라미터의 optimizer 상태를 학습 장치로 이동합니다.
    for values in optimizer.state.values():
        for name, value in values.items():
            if isinstance(value, torch.Tensor) and name != "step":
                values[name] = value.to(device)
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["rng_cpu"])
    if torch.device(device).type == "cuda" and state["rng_cuda"]:
        if len(state["rng_cuda"]) != torch.cuda.device_count():
            raise ValueError("저장 당시와 CUDA 장치 수가 다릅니다.")
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return state
