"""단계 사이에는 모델을 CPU RAM에 보관하고 실행할 때만 장치로 이동합니다."""

import gc
from contextlib import contextmanager
from pathlib import Path

import torch

from .logging_utils import logger, log_stage
from .profiling import profile_range, profile_stage
from .runtime import load_dit, load_text_encoder, load_vae


_models = {}


def _file_identity(path):
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


@log_stage("캐시 모델 장치 이동")
@profile_stage("cache_transfer")
def _move_model(model, device):
    logger.info("%s -> %s", model.__class__.__name__, device)
    with profile_range("cache/model_transfer"):
        model.to(device=device)


def _clear_dit_state(model):
    # forward가 진단용으로 저장한 GPU 텐서 참조를 단계 종료 시 해제합니다.
    model.affline_scale_log_info = {}
    model.affline_emb = None
    model.crossattn_emb = None


def _clear_vae_state(model):
    # VAE의 이전 프레임 feature cache도 GPU 텐서를 보유합니다.
    model.model.clear_cache()


def clear_model_cache():
    """메모리에 보관한 모델 참조를 해제합니다. 파일은 수정하지 않습니다."""
    if _models:
        logger.info("모델 캐시 비우기: %s", ", ".join(_models))
        _models.clear()
        gc.collect()


@contextmanager
def _use_model(name, loader, path, device, dtype, enabled, cleanup=None, **options):
    if not enabled:
        clear_model_cache()
        logger.info("%s: 기존 로딩 경로", name)
        yield loader(path, device=device, dtype=dtype, **options)
        return

    lora_path = options.get("lora_path")
    key = (
        _file_identity(path),
        dtype,
        _file_identity(lora_path) if lora_path else None,
        float(options.get("lora_scale", 1.0)) if lora_path else None,
    )
    entry = _models.get(name)
    if entry is not None and entry[0] != key:
        logger.info("%s 캐시 교체: 가중치 또는 LoRA 설정 변경", name)
        del _models[name]
        del entry
        gc.collect()
        entry = None
    if entry is None:
        logger.info("%s 캐시 miss: CPU RAM에 최초 로딩", name)
        model = loader(path, device="cpu", dtype=dtype, **options)
        _models[name] = key, model
    else:
        logger.info("%s 캐시 hit: 파일 재로딩 없이 재사용", name)
        model = entry[1]

    try:
        _move_model(model, device)
        yield model
    finally:
        if cleanup is not None:
            cleanup(model)
        _move_model(model, "cpu")
        logger.info("%s: CPU RAM에 보관", name)


def use_text_encoder(path, device="cuda", dtype=torch.float16, enabled=False):
    return _use_model("text_encoder", load_text_encoder, path, device, dtype, enabled)


def use_dit(path, device="cuda", dtype=torch.float16, enabled=False, lora_path=None, lora_scale=1.0):
    return _use_model(
        "dit", load_dit, path, device, dtype, enabled,
        cleanup=_clear_dit_state, lora_path=lora_path, lora_scale=lora_scale,
    )


def use_vae(path, device="cuda", dtype=torch.float16, enabled=False):
    return _use_model("vae", load_vae, path, device, dtype, enabled, cleanup=_clear_vae_state)
