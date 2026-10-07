# Source: sorryhyun/anime_tools, commit 74aa014ba12db74286b17e634056683f7f2b44d7 (v0.7.5).
# License: licenses/anime-tools-LICENSE.


from __future__ import annotations

import math


EDGE_TOKEN_BANDS: dict[int, tuple[int, int]] = {
    512: (1008, 1024),
    768: (2160, 2160),
    896: (3000, 3024),
    1024: (4032, 4200),
    1280: (6300, 6300),
    1536: (8640, 8640),
}
ALLOWED_TARGET_RES: tuple[int, ...] = tuple(sorted(EDGE_TOKEN_BANDS))
DEFAULT_TARGET_RES: tuple[int, ...] = (1024,)

DEFAULT_FREEFIT_MAX_RATIO = 4.0


FREEFIT_BAND_TOLERANCE = 0.025


FREEFIT_FROZEN_EDGES: tuple[int, ...] = (1024,)


def band_for_tier(edge: int) -> tuple[int, int]:
    try:
        return EDGE_TOKEN_BANDS[edge]
    except KeyError:
        raise ValueError(
            f"target_res {edge} not in allowed tiers {list(ALLOWED_TARGET_RES)}"
        ) from None


def choose_edge(width: int, height: int, target_res) -> int:
    tiers = list(target_res)
    if len(tiers) == 1:
        return tiers[0]
    native_tokens = (width / 16.0) * (height / 16.0)
    best_edge: int | None = None
    best_cost = float("inf")
    for edge in tiers:
        lo, hi = band_for_tier(edge)
        cost = abs(math.log(((lo + hi) / 2.0) / native_tokens))
        if cost < best_cost:
            best_cost, best_edge = cost, edge
    if best_edge is None:
        raise ValueError("choose_edge requires at least one tier")
    return best_edge


def freefit_band_for_edge(
    edge: int, tol: float = FREEFIT_BAND_TOLERANCE
) -> tuple[int, int]:
    lo, hi = band_for_tier(edge)
    if edge in FREEFIT_FROZEN_EDGES:
        return lo, hi
    return round(lo * (1.0 - tol)), round(hi * (1.0 + tol))


def freefit_bucket(
    width: int,
    height: int,
    band: tuple[int, int],
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
    patch: int = 16,
    rope_cap: int = 256,
) -> tuple[int, int]:
    lo, hi = int(band[0]), int(band[1])
    if lo <= 0 or hi < lo:
        raise ValueError(f"invalid free-fit band {band}")
    aspect = min(max(width / height, 1.0 / max_ratio), float(max_ratio))

    best: tuple[float, float, int, int] | None = None
    for hp in range(1, min(rope_cap, hi) + 1):
        wp_lo = max(1, -(-lo // hp))
        wp_hi = min(rope_cap, hi // hp)
        for wp in range(wp_lo, wp_hi + 1):
            cover_scale = max(wp * patch / width, hp * patch / height)

            key = (abs(wp / hp - aspect), abs(math.log(cover_scale)), hp, wp)
            if best is None or key < best:
                best = key
    if best is None:
        raise ValueError(f"free-fit band {band} admits no grid under {rope_cap=}")
    _, _, hp, wp = best
    return wp * patch, hp * patch


from collections.abc import Iterable


def normalize_target_res(target_res: Iterable[int] | int | str | None) -> list[int]:
    if target_res is None:
        return list(DEFAULT_TARGET_RES)
    if isinstance(target_res, int):
        return [target_res]
    if isinstance(target_res, str):
        raw = target_res.strip()
        if not raw:
            return list(DEFAULT_TARGET_RES)
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    values = [int(value) for value in target_res]
    return values or list(DEFAULT_TARGET_RES)


def select_bucket(
    width: int,
    height: int,
    target_res: Iterable[int] | int | str | None = None,
    *,
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
) -> tuple[int, tuple[int, int]]:
    edge = choose_edge(width, height, normalize_target_res(target_res))
    return edge, freefit_bucket(
        width, height, freefit_band_for_edge(edge), max_ratio=max_ratio
    )


