"""런타임 단계 로그와 노트북/터미널 진행률 표시."""

import logging
import sys
from functools import wraps
from time import perf_counter

from tqdm.auto import tqdm


logger = logging.getLogger("anima_core")
_show_progress = True


def configure_logging(options=None):
    """level과 progress 옵션으로 패키지 전용 출력을 설정합니다."""
    global _show_progress
    options = options or {}
    logger.setLevel(options.get("level", "INFO"))
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    _show_progress = bool(options.get("progress", True))


def log_stage(label):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            started = perf_counter()
            logger.info("%s 시작", label)
            try:
                result = function(*args, **kwargs)
            except Exception:
                logger.exception("%s 실패", label)
                raise
            logger.info("%s 완료 (경과 %.2f초)", label, perf_counter() - started)
            return result
        return wrapped
    return decorate


def track_progress(iterable, label, total=None):
    return tqdm(iterable, desc=label, total=total, disable=not _show_progress)


configure_logging()
