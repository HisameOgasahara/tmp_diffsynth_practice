"""캐시를 읽어 일반 LoRA를 학습하고 업데이트 경계에서 저장합니다."""

import json
import math
from pathlib import Path
import time

import torch
from tqdm.auto import tqdm

from .checkpoints import load_checkpoint, save_checkpoint
from .config import resolve_dtype, validate_config, write_config
from .dataset import CachedDataset, SampleStream, release_cuda_memory
from .flow_matching import FlowMatchingLoss
from .lora import inject_lora


def create_scheduler(optimizer, training):
    total = training["max_steps"]
    warmup = round(total * training["warmup_ratio"])

    def multiplier(completed):
        if warmup and completed < warmup:
            return (completed + 1) / warmup
        if training["lr_scheduler"] == "constant":
            return 1.0
        progress = min(1.0, max(0.0, (completed - warmup) / max(1, total - warmup)))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train_model(model, config, dataset, output_dir, resume_from=None, device="cuda", tensorboard=True):
    validate_config(config)
    device = torch.device(device)
    dtype = resolve_dtype(config, device)
    output_dir = Path(output_dir).expanduser().resolve()
    if not resume_from and any(output_dir.glob("step-*")):
        raise FileExistsError("기존 학습 결과가 있습니다. 새 output_dir 또는 resume_from을 지정하세요.")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_config(output_dir / "training.toml", config)
    fingerprint = dataset.manifest["fingerprint"]
    training = config["training"]
    torch.manual_seed(training["seed"])
    model.to(device=device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("학습할 LoRA 파라미터가 없습니다.")
    optimizer = torch.optim.AdamW(parameters, lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = create_scheduler(optimizer, training)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    step = 0
    stream_state = {"epoch": 0, "cursor": 0}
    if resume_from:
        state = load_checkpoint(resume_from, model, optimizer, scheduler, scaler,
                                config, fingerprint, device)
        step, stream_state = state["step"], state["stream"]
    stream = SampleStream(dataset, config["dataset"]["repeat"], training["seed"], **stream_state)
    loss_fn = FlowMatchingLoss()
    writer = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output_dir / "logs"), purge_step=step + 1 if resume_from else None)
    log_file = (output_dir / "metrics.jsonl").open("a", encoding="utf-8")
    last_checkpoint = Path(resume_from) if resume_from else None
    batch_size = config["dataset"]["batch_size"]
    accumulation = training["gradient_accumulation_steps"]
    dropout = config["dataset"]["caption_dropout_rate"]
    print(f"학습 가능 파라미터: {sum(p.numel() for p in parameters):,}")
    print(f"batch={batch_size}, accumulation={accumulation}, 유효 batch={batch_size * accumulation}")
    print("캡션 길이를 유지하기 위해 batch 안의 샘플을 순차 계산해 gradient를 평균합니다.")
    progress = tqdm(total=training["max_steps"], initial=step, desc="LoRA optimizer 업데이트")
    try:
        while step < training["max_steps"]:
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            lr = optimizer.param_groups[0]["lr"]
            for _ in range(accumulation):
                for sample in stream.take(batch_size):
                    condition = dataset.empty if torch.rand(()).item() < dropout else sample
                    latent = sample["latent"].unsqueeze(0).to(device=device, dtype=dtype)
                    embeds = condition["prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype)
                    ids = condition["t5_ids"].unsqueeze(0).to(device=device)
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                        loss = loss_fn(model, latent, embeds, ids,
                                       config["runtime"]["use_gradient_checkpointing"])
                    if not torch.isfinite(loss):
                        raise RuntimeError("loss에 비유한 값이 있습니다. precision과 학습률을 확인하세요.")
                    total_loss += float(loss.detach()) / (batch_size * accumulation)
                    scaler.scale(loss / (batch_size * accumulation)).backward()
                    del loss, latent, embeds, ids
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < old_scale:
                print("FP16 gradient overflow: optimizer 업데이트를 건너뛰고 scale을 줄였습니다.")
                continue
            scheduler.step()
            step += 1
            metrics = {"step": step, "loss": total_loss, "learning_rate": lr,
                       "seconds": time.monotonic() - started, "epoch": stream.epoch}
            log_file.write(json.dumps(metrics) + "\n")
            log_file.flush()
            if writer:
                for key in ("loss", "learning_rate", "seconds"):
                    writer.add_scalar(f"train/{key}", metrics[key], step)
                writer.flush()
            progress.update(1)
            progress.set_postfix(loss=f"{total_loss:.5f}", lr=f"{lr:.2g}")
            if step % config["checkpoint"]["save_steps"] == 0 or step == training["max_steps"]:
                last_checkpoint = save_checkpoint(output_dir, model, optimizer, scheduler, scaler,
                                                   step, stream, config, fingerprint)
                print("저장 완료:", last_checkpoint)
    except KeyboardInterrupt:
        print("학습이 중단되었습니다. 완료된 마지막 정기 체크포인트:", last_checkpoint)
        raise
    finally:
        progress.close()
        log_file.close()
        if writer:
            writer.close()
    return {"step": step, "checkpoint": str(last_checkpoint),
            "lora": str(last_checkpoint / "lora.safetensors")}


def train(config, cache_path, dit_path, output_dir, resume_from=None, device="cuda", tensorboard=True):
    from ..runtime import load_dit

    dtype = resolve_dtype(config, device)
    dataset = CachedDataset(cache_path)
    cached_weights = dataset.manifest["identity"]["weights"]["dit"]
    dit_path = Path(dit_path).resolve()
    stat = dit_path.stat()
    if {"path": str(dit_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != cached_weights:
        raise ValueError("캐시를 준비할 때 사용한 DiT 가중치를 지정하세요.")
    torch.manual_seed(config["training"]["seed"])
    model = load_dit(dit_path, device=device, dtype=dtype)
    try:
        targets = inject_lora(model, config["lora"])
        print(f"LoRA 적용 레이어: {len(targets)}개")
        (Path(output_dir).expanduser().resolve()).mkdir(parents=True, exist_ok=True)
        (Path(output_dir).expanduser().resolve() / "lora_targets.json").write_text(
            json.dumps(targets, indent=2), encoding="utf-8"
        )
        return train_model(model, config, dataset, output_dir, resume_from, device, tensorboard)
    finally:
        del model
        release_cuda_memory()
