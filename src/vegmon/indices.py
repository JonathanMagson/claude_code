"""Spectral indices and Sentinel-1 unit conversions.

All optical functions take surface reflectance as floats in 0-1 (Sentinel-2
L2A scaled by 1/10000) and return float32 arrays with NaN wherever the inputs
were NaN or the denominator collapses.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a - b) / denom
    out = np.where(np.abs(denom) < _EPS, np.nan, out)
    return out.astype(np.float32)


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalised Difference Vegetation Index, (NIR - Red) / (NIR + Red).

    Sentinel-2 B08 and B04. Saturates over dense canopy, which is why it is
    paired with NBR here rather than used alone.
    """
    return _normalised_difference(nir, red)


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalised Burn Ratio, (NIR - SWIR2) / (NIR + SWIR2).

    Sentinel-2 B08 and B12. The workhorse for clearing: removing woody
    material drops NIR and raises SWIR2 at once, so NBR moves further than
    NDVI for the same event and keeps moving after NDVI has saturated.
    """
    return _normalised_difference(nir, swir2)


def ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Normalised Difference Moisture Index, (NIR - SWIR1) / (NIR + SWIR1).

    Sentinel-2 B08 and B11. Tracks canopy water content; useful for telling a
    drought-stressed but intact canopy from a removed one.
    """
    return _normalised_difference(nir, swir1)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters NDWI, (Green - NIR) / (Green + NIR). Used to mask open water."""
    return _normalised_difference(green, nir)


# Sentinel-2 tasselled cap coefficients (Shi & Xu 2019, 10 m + 20 m bands
# ordered blue, green, red, nir, swir1, swir2).
_TC_BRIGHTNESS = np.array([0.3510, 0.3813, 0.3437, 0.7196, 0.2396, 0.1949], dtype=np.float32)
_TC_GREENNESS = np.array([-0.3599, -0.3533, -0.4734, 0.6633, 0.0087, -0.2856], dtype=np.float32)
_TC_WETNESS = np.array([0.2578, 0.2305, 0.0883, 0.1071, -0.7611, -0.5308], dtype=np.float32)


def _tasselled_cap(
    coeffs: np.ndarray,
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
) -> np.ndarray:
    stack = np.stack(
        [np.asarray(b, dtype=np.float32) for b in (blue, green, red, nir, swir1, swir2)]
    )
    return np.tensordot(coeffs, stack, axes=(0, 0)).astype(np.float32)


def tc_brightness(blue, green, red, nir, swir1, swir2) -> np.ndarray:
    """Tasselled cap brightness - rises sharply when soil is exposed."""
    return _tasselled_cap(_TC_BRIGHTNESS, blue, green, red, nir, swir1, swir2)


def tc_greenness(blue, green, red, nir, swir1, swir2) -> np.ndarray:
    """Tasselled cap greenness - the photosynthetic-vegetation axis."""
    return _tasselled_cap(_TC_GREENNESS, blue, green, red, nir, swir1, swir2)


def tc_wetness(blue, green, red, nir, swir1, swir2) -> np.ndarray:
    """Tasselled cap wetness - canopy and soil moisture.

    Wetness is the single most reliable tasselled-cap axis for woody change in
    Australian landscapes: it separates structural loss from the seasonal
    green-up that confounds NDVI in cropping country.
    """
    return _tasselled_cap(_TC_WETNESS, blue, green, red, nir, swir1, swir2)


def tc_disturbance_index(
    brightness: np.ndarray, greenness: np.ndarray, wetness: np.ndarray
) -> np.ndarray:
    """Healey's Disturbance Index on already-normalised TC components.

    DI = brightness - (greenness + wetness). Inputs are expected to be
    z-scores against a stable-forest population; use :func:`zscore_stable` to
    produce them.
    """
    return (
        np.asarray(brightness, dtype=np.float32)
        - (np.asarray(greenness, dtype=np.float32) + np.asarray(wetness, dtype=np.float32))
    ).astype(np.float32)


def zscore_stable(values: np.ndarray, stable_mask: np.ndarray) -> np.ndarray:
    """Z-score ``values`` against the population inside ``stable_mask``.

    Standardising against undisturbed vegetation, rather than the whole scene,
    is what makes the Disturbance Index comparable between dates.
    """
    values = np.asarray(values, dtype=np.float32)
    sample = values[stable_mask & np.isfinite(values)]
    if sample.size < 2:
        return np.full_like(values, np.nan, dtype=np.float32)
    mean = float(np.mean(sample))
    std = float(np.std(sample))
    if std < _EPS:
        return np.full_like(values, np.nan, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def linear_to_db(linear: np.ndarray) -> np.ndarray:
    """Sentinel-1 backscatter from linear power to decibels."""
    linear = np.asarray(linear, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 10.0 * np.log10(np.where(linear > 0, linear, np.nan))
    return out.astype(np.float32)


def db_to_linear(db: np.ndarray) -> np.ndarray:
    """Sentinel-1 backscatter from decibels to linear power."""
    return np.power(10.0, np.asarray(db, dtype=np.float32) / 10.0).astype(np.float32)


def rvi(vv_linear: np.ndarray, vh_linear: np.ndarray) -> np.ndarray:
    """Dual-pol Radar Vegetation Index, 4*VH / (VV + VH), in linear power.

    Ranges 0-1 and rises with volume scattering, so it drops when canopy is
    removed. Cross-checks the raw VH drop without needing a clear sky.
    """
    vv = np.asarray(vv_linear, dtype=np.float32)
    vh = np.asarray(vh_linear, dtype=np.float32)
    denom = vv + vh
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (4.0 * vh) / denom
    return np.where(np.abs(denom) < _EPS, np.nan, out).astype(np.float32)


__all__ = [
    "db_to_linear",
    "linear_to_db",
    "ndmi",
    "ndvi",
    "ndwi",
    "nbr",
    "rvi",
    "tc_brightness",
    "tc_disturbance_index",
    "tc_greenness",
    "tc_wetness",
    "zscore_stable",
]
