# Source: sorryhyun/anime_tools, commit 74aa014ba12db74286b17e634056683f7f2b44d7 (v0.7.5).
# License: licenses/anime-tools-LICENSE.
# Bucket and resize implementations below are copied unchanged from upstream.

"""Free-fit ("free-aspect token-band") resize geometry — the tier + bucket math.

Free-fit preserves an image's native aspect ratio and lands its patch-grid token
count anywhere inside its tier's band, so the cropped residual on the covering
axis is under one patch. Token count is ``(W//16) * (H//16)``.

This module is the one owner of the geometry: the trainer's
``library/datasets/buckets.py`` re-exports these names (the way it re-exports
the PE tower), so both sides land an image on the same ``(W, H)`` by
construction and neither pass re-resizes the other's PNGs. Torch-free and
numpy-free — the trainer's GUI reads it on the UI thread.
"""

from __future__ import annotations

import math

# Per-tier token-count bands. Single-family tiers (768/1280/1536) have lo == hi;
# 512/896/1024 carry two families. Every count is inside the rope per-axis cap.
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

# Widens every non-frozen tier's natural band so the solver has aspect freedom:
# a band with lo == hi leaves free-fit only that count's coarse divisor grids
# and it crops.
FREEFIT_BAND_TOLERANCE = 0.025  # ±2.5%

# 1024 stays at its natural (4032, 4200): the trainer's frozen top-5 aspect set
# (``DCW_ASPECT_BUCKETS``, consumed by CNS calibration + mod-distill) is drawn
# from this tier. Bump only with those consumers in mind.
FREEFIT_FROZEN_EDGES: tuple[int, ...] = (1024,)


def band_for_tier(edge: int) -> tuple[int, int]:
    """The natural ``(lo, hi)`` token band for a tier edge, or a clear error."""
    try:
        return EDGE_TOKEN_BANDS[edge]
    except KeyError:
        raise ValueError(
            f"target_res {edge} not in allowed tiers {list(ALLOWED_TARGET_RES)}"
        ) from None


def choose_edge(width: int, height: int, target_res) -> int:
    """Assign an image to the tier that resizes it the *least*.

    Minimizes ``|log(band_midpoint / native_tokens)|``, so it is
    scale-symmetric; a single-element ``target_res`` is a no-op.
    """
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
    """Token-count band ``(lo, hi)`` for one tier — the free-fit search range.

    The natural band, widened symmetrically by ``tol`` except for
    :data:`FREEFIT_FROZEN_EDGES`. A wider band crops less.
    """
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
    """Native-aspect resize target whose patch grid fills the token ``band``.

    Returns pixel ``(W, H)``, both multiples of ``patch``, whose patch grid lies
    in ``[lo, hi]`` and whose aspect is as close as possible to the source's,
    clamped to ``[1/max_ratio, max_ratio]`` and subject to ``max(W//patch,
    H//patch) <= rope_cap``. Deterministic in its inputs.

    Crop is zero unless the ratio clamp fired, in which case the caller
    cover-crops to the clamped aspect. The search is exhaustive over the band,
    tie-broken toward the grid that rescales the image the least.
    """
    lo, hi = int(band[0]), int(band[1])
    if lo <= 0 or hi < lo:
        raise ValueError(f"invalid free-fit band {band}")
    aspect = min(max(width / height, 1.0 / max_ratio), float(max_ratio))

    best: tuple[float, float, int, int] | None = None
    for hp in range(1, min(rope_cap, hi) + 1):
        wp_lo = max(1, -(-lo // hp))  # ceil(lo / hp)
        wp_hi = min(rope_cap, hi // hp)  # floor(hi / hp)
        for wp in range(wp_lo, wp_hi + 1):
            cover_scale = max(wp * patch / width, hp * patch / height)
            # aspect first, then least rescale, then a deterministic shape key.
            key = (abs(wp / hp - aspect), abs(math.log(cover_scale)), hp, wp)
            if best is None or key < best:
                best = key
    if best is None:
        raise ValueError(f"free-fit band {band} admits no grid under {rope_cap=}")
    _, _, hp, wp = best
    return wp * patch, hp * patch


from collections.abc import Iterable
from PIL import Image

DEFAULT_CROP_ANCHOR = "center"


CROP_ANCHORS: dict[str, tuple[float, float]] = {
    "top_left": (0.0, 0.0),
    "top": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0),
    "bottom": (0.5, 1.0),
    "bottom_right": (1.0, 1.0),
}


def normalize_target_res(target_res: Iterable[int] | int | str | None) -> list[int]:
    """Normalize a config/CLI ``target_res`` into a non-empty tier list."""
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


def normalize_crop_anchor(crop_anchor: str | None) -> str:
    value = str(crop_anchor or DEFAULT_CROP_ANCHOR).strip().lower()
    return value if value in CROP_ANCHORS else DEFAULT_CROP_ANCHOR


def select_bucket(
    width: int,
    height: int,
    target_res: Iterable[int] | int | str | None = None,
    *,
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
) -> tuple[int, tuple[int, int]]:
    """``(tier_edge, (W, H))`` for a source size — the whole geometry decision."""
    edge = choose_edge(width, height, normalize_target_res(target_res))
    return edge, freefit_bucket(
        width, height, freefit_band_for_edge(edge), max_ratio=max_ratio
    )


def resize_to_bucket(
    img: Image.Image,
    bucket: tuple[int, int],
    *,
    crop_anchor: str = DEFAULT_CROP_ANCHOR,
) -> Image.Image:
    """Cover-scale ``img`` to ``bucket`` (LANCZOS) then anchor-crop to it.

    ``img`` is the already-transposed, margin-cropped working region. This is
    the trainer's exact pixel geometry.
    """
    bw, bh = bucket
    anchor_x, anchor_y = CROP_ANCHORS[normalize_crop_anchor(crop_anchor)]
    w, h = img.size
    if w / h > bw / bh:
        new_h, new_w = bh, round(bh * w / h)
    else:
        new_w, new_h = bw, round(bw * h / w)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = round((new_w - bw) * anchor_x)
    top = round((new_h - bh) * anchor_y)
    return img.crop((left, top, left + bw, top + bh))

