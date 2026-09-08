"""Sentinel-2 scene classification masking and compositing reducers.

Cloud and cloud-shadow handling is the single largest source of false
positives in optical change detection: an unmasked cloud shadow looks exactly
like a canopy that has been removed. The Sen2Cor scene classification layer
(SCL) is used here, dilated outwards, because SCL consistently under-calls
thin cloud edges and shadow.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
from scipy import ndimage

# Sen2Cor scene classification codes shipped in the Sentinel-2 L2A SCL band.
SCL_NO_DATA = 0
SCL_SATURATED = 1
SCL_DARK_AREA = 2
SCL_CLOUD_SHADOW = 3
SCL_VEGETATION = 4
SCL_NOT_VEGETATED = 5
SCL_WATER = 6
SCL_UNCLASSIFIED = 7
SCL_CLOUD_MEDIUM_PROB = 8
SCL_CLOUD_HIGH_PROB = 9
SCL_THIN_CIRRUS = 10
SCL_SNOW = 11

SCL_NAMES = {
    SCL_NO_DATA: "no data",
    SCL_SATURATED: "saturated / defective",
    SCL_DARK_AREA: "dark area / topographic shadow",
    SCL_CLOUD_SHADOW: "cloud shadow",
    SCL_VEGETATION: "vegetation",
    SCL_NOT_VEGETATED: "not vegetated",
    SCL_WATER: "water",
    SCL_UNCLASSIFIED: "unclassified",
    SCL_CLOUD_MEDIUM_PROB: "cloud, medium probability",
    SCL_CLOUD_HIGH_PROB: "cloud, high probability",
    SCL_THIN_CIRRUS: "thin cirrus",
    SCL_SNOW: "snow / ice",
}

#: Codes treated as usable land observations.
DEFAULT_VALID_SCL: tuple[int, ...] = (
    SCL_VEGETATION,
    SCL_NOT_VEGETATED,
    SCL_WATER,
    SCL_UNCLASSIFIED,
)

#: Codes that get buffered outwards before masking.
DEFAULT_DILATED_SCL: tuple[int, ...] = (
    SCL_CLOUD_SHADOW,
    SCL_CLOUD_MEDIUM_PROB,
    SCL_CLOUD_HIGH_PROB,
    SCL_THIN_CIRRUS,
    SCL_SATURATED,
)


def scl_valid_mask(
    scl: np.ndarray,
    valid_codes: Sequence[int] = DEFAULT_VALID_SCL,
    dilate_codes: Sequence[int] = DEFAULT_DILATED_SCL,
    dilation_pixels: int = 3,
) -> np.ndarray:
    """Boolean mask of usable pixels, with cloud and shadow buffered outwards.

    Parameters
    ----------
    scl:
        Scene classification band, ``(..., y, x)``. Leading axes (e.g. time)
        are handled independently.
    dilation_pixels:
        Radius in pixels to grow the ``dilate_codes`` classes by. At 10 m,
        3 pixels (~30 m) removes most of the soft cloud edge that Sen2Cor
        leaves behind. Set to 0 to disable.
    """
    scl = np.asarray(scl)
    valid = np.isin(scl, np.asarray(valid_codes))
    if dilation_pixels and len(dilate_codes):
        bad = np.isin(scl, np.asarray(dilate_codes))
        bad = dilate_mask(bad, dilation_pixels)
        valid &= ~bad
    return valid


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Grow a boolean mask by ``pixels`` using a disc, per 2-D slice."""
    if pixels <= 0:
        return np.asarray(mask, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    structure = _disc(pixels)
    if mask.ndim == 2:
        return ndimage.binary_dilation(mask, structure=structure)
    out = np.empty_like(mask)
    for index in np.ndindex(mask.shape[:-2]):
        out[index] = ndimage.binary_dilation(mask[index], structure=structure)
    return out


def _disc(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def cloud_fraction(scl: np.ndarray, valid_codes: Sequence[int] = DEFAULT_VALID_SCL) -> np.ndarray:
    """Per-scene fraction of pixels that are *not* usable land observations."""
    scl = np.asarray(scl)
    valid = np.isin(scl, np.asarray(valid_codes))
    axes = tuple(range(valid.ndim - 2, valid.ndim))
    return 1.0 - valid.mean(axis=axes)


def masked_median(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median over axis 0 of ``stack`` where ``valid``; NaN where nothing is.

    Median rather than mean: a single missed cloud in a small stack drags a
    mean far more than it drags a median.
    """
    stack = np.asarray(stack, dtype=np.float32)
    data = np.where(valid, stack, np.nan)
    with np.errstate(invalid="ignore"):
        return _nanmedian(data)


def masked_mean(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    stack = np.asarray(stack, dtype=np.float32)
    data = np.where(valid, stack, np.nan)
    with np.errstate(invalid="ignore"):
        return _nanmean(data)


def _nanmedian(data: np.ndarray) -> np.ndarray:
    if data.shape[0] == 0:
        return np.full(data.shape[1:], np.nan, dtype=np.float32)
    all_nan = np.all(np.isnan(data), axis=0)
    safe = np.where(all_nan[None, ...], 0.0, data)
    out = np.nanmedian(safe, axis=0).astype(np.float32)
    return np.where(all_nan, np.nan, out)


def _nanmean(data: np.ndarray) -> np.ndarray:
    if data.shape[0] == 0:
        return np.full(data.shape[1:], np.nan, dtype=np.float32)
    all_nan = np.all(np.isnan(data), axis=0)
    safe = np.where(all_nan[None, ...], 0.0, data)
    out = np.nanmean(safe, axis=0).astype(np.float32)
    return np.where(all_nan, np.nan, out)


def masked_medoid(
    stack: Dict[str, np.ndarray], valid: np.ndarray
) -> Dict[str, np.ndarray]:
    """Per-pixel medoid composite: the single real observation closest to the median.

    A band-wise median is the obvious reducer and it is quietly wrong. Each
    band's median is taken independently, so over a pixel that swings hard
    through the year - a cropping paddock going bare, green, harvested - the
    median red can come from a bare date and the median NIR from a green one.
    The result is a spectrum that no acquisition ever recorded, and its NDVI
    can sit far above anything the pixel actually reached. On a real NSW
    cropping scene that inflates composite NDVI by more than 0.3 at the 90th
    percentile, which is enough to walk crop straight through a woody-cover
    gate.

    The medoid avoids it by choosing, per pixel, the observation whose whole
    spectrum is closest to the per-band median vector. Every output pixel is
    then a real, internally consistent observation.

    Parameters
    ----------
    stack:
        Band name -> ``(time, y, x)`` reflectance.
    valid:
        ``(time, y, x)`` boolean mask of usable observations.
    """
    names = list(stack)
    if not names:
        return {}
    data = np.stack([np.asarray(stack[name], dtype=np.float32) for name in names])
    valid = np.asarray(valid, dtype=bool)
    masked = np.where(valid[None, ...], data, np.nan)

    reference = _nanmedian_axis1(masked)
    with np.errstate(invalid="ignore"):
        distance = np.nansum((masked - reference[:, None, ...]) ** 2, axis=0)
    # Unusable observations must never win the argmin.
    distance = np.where(valid, distance, np.inf)

    nothing_valid = ~np.any(valid, axis=0)
    chosen = np.argmin(np.where(np.isfinite(distance), distance, np.inf), axis=0)
    rows, cols = np.indices(chosen.shape)
    out = {}
    for index, name in enumerate(names):
        picked = data[index][chosen, rows, cols].astype(np.float32)
        out[name] = np.where(nothing_valid, np.nan, picked)
    return out


def _nanmedian_axis1(data: np.ndarray) -> np.ndarray:
    """NaN-aware median over axis 1 of a ``(band, time, y, x)`` array."""
    all_nan = np.all(np.isnan(data), axis=1)
    safe = np.where(all_nan[:, None, ...], 0.0, data)
    with np.errstate(invalid="ignore"):
        out = np.nanmedian(safe, axis=1).astype(np.float32)
    return np.where(all_nan, np.nan, out)


def observation_count(valid: np.ndarray) -> np.ndarray:
    """Number of valid observations per pixel, as int16."""
    return np.asarray(valid, dtype=bool).sum(axis=0).astype(np.int16)


def boxcar(values: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware boxcar mean, used as the Sentinel-1 multi-look filter."""
    if window <= 1:
        return np.asarray(values, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    kernel = np.ones((window, window), dtype=np.float32)
    total = ndimage.convolve(filled, kernel, mode="nearest")
    count = ndimage.convolve(finite.astype(np.float32), kernel, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    return np.where(count > 0, out, np.nan).astype(np.float32)


def describe_scl(codes: Iterable[int]) -> str:
    return ", ".join(f"{c} ({SCL_NAMES.get(int(c), 'unknown')})" for c in codes)


__all__ = [
    "DEFAULT_DILATED_SCL",
    "DEFAULT_VALID_SCL",
    "SCL_NAMES",
    "boxcar",
    "cloud_fraction",
    "describe_scl",
    "dilate_mask",
    "masked_mean",
    "masked_median",
    "masked_medoid",
    "observation_count",
    "scl_valid_mask",
]
