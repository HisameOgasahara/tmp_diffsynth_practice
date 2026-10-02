"""선택한 추론 단계의 PyTorch CPU/CUDA trace와 연산별 요약 저장."""

from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from pathlib import Path
from uuid import uuid4

import torch

from .logging_utils import logger


@dataclass(frozen=True)
class ProfileOptions:
    enabled: bool = False
    stages: tuple = ("sample_latents",)
    output_dir: str = "profiles"
    wait: int = 0
    warmup: int = 1
    active: int = 3
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    row_limit: int = 20


_options = ProfileOptions()
_current_profiler = ContextVar("anima_profiler", default=None)
_stage_names = set()


def configure_profiler(options=None):
    """options는 ProfileOptions 필드와 동일한 키를 가진 설정입니다."""
    global _options
    values = dict(options or {})
    if "stages" in values:
        values["stages"] = tuple(values["stages"])
    configured = ProfileOptions(**values)
    if configured.wait < 0 or configured.warmup < 0 or configured.active < 1:
        raise ValueError("profiler wait/warmup은 0 이상, active는 1 이상이어야 합니다.")
    if configured.row_limit < 1:
        raise ValueError("profiler row_limit은 1 이상이어야 합니다.")
    unknown = set(configured.stages) - _stage_names
    if unknown:
        raise ValueError(f"알 수 없는 profiler 단계: {sorted(unknown)}, 사용 가능: {sorted(_stage_names)}")
    _options = configured


def profile_range(name):
    if _current_profiler.get() is None:
        return nullcontext()
    return torch.profiler.record_function(name)


def advance_profile_step():
    profiler = _current_profiler.get()
    if profiler is not None:
        profiler.step()


def _save_report(profiler, directory, name, options, uses_cuda):
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"{name}_{uuid4().hex}"
    trace_path = stem.with_suffix(".trace.json")
    summary_path = stem.with_suffix(".txt")
    profiler.export_chrome_trace(str(trace_path))
    averages = profiler.key_averages(group_by_input_shape=options.record_shapes)
    tables = ["CPU 연산 시간 (자식 연산 제외)\n" + averages.table(
        sort_by="self_cpu_time_total", row_limit=options.row_limit
    )]
    if uses_cuda:
        tables.append("CUDA 연산 시간 (자식 연산 제외)\n" + averages.table(
            sort_by="self_device_time_total", row_limit=options.row_limit
        ))
    summary = "\n\n".join(tables)
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("프로파일 요약 %s\n%s", name, summary)
    logger.info("프로파일 저장: %s / %s", trace_path, summary_path)


def profile_stage(name, stepped=False):
    _stage_names.add(name)
    def decorate(function):
        function_signature = signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            options = _options
            if not options.enabled or name not in options.stages or _current_profiler.get() is not None:
                return function(*args, **kwargs)
            arguments = function_signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            tensor_device = next(
                (value.device for value in arguments.arguments.values() if isinstance(value, torch.Tensor)),
                "cpu",
            )
            device = torch.device(arguments.arguments.get("device", tensor_device))
            uses_cuda = device.type == "cuda" and torch.cuda.is_available()
            activities = [torch.profiler.ProfilerActivity.CPU]
            if uses_cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            schedule = None
            if stepped:
                available = int(arguments.arguments["steps"]) - options.wait - options.warmup
                if available < 1:
                    logger.warning("%s 프로파일 생략: 생성 스텝이 wait + warmup보다 커야 합니다.", name)
                    return function(*args, **kwargs)
                active = min(options.active, available)
                schedule = torch.profiler.schedule(
                    wait=options.wait, warmup=options.warmup, active=active, repeat=1
                )
                logger.info("%s 프로파일: wait=%d, warmup=%d, active=%d", name, options.wait, options.warmup, active)
            directory = Path(options.output_dir).expanduser()
            profiler = torch.profiler.profile(
                activities=activities,
                schedule=schedule,
                record_shapes=options.record_shapes,
                profile_memory=options.profile_memory,
                with_stack=options.with_stack,
                on_trace_ready=lambda p: _save_report(p, directory, name, options, uses_cuda),
            )
            logger.info("%s 프로파일 시작: %s", name, device)
            with profiler:
                token = _current_profiler.set(profiler)
                try:
                    return function(*args, **kwargs)
                finally:
                    _current_profiler.reset(token)
        return wrapped
    return decorate
