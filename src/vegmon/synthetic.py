"""A physically-motivated synthetic Sentinel-1/2 datacube with ground truth.

Real STAC catalogues are not always reachable (corporate proxies, air-gapped
review environments, or simply an outage the morning of a demo). This module
generates a datacube that behaves like the real thing where it matters:

* reflectance comes from **linear spectral unmixing** of green vegetation,
  non-photosynthetic vegetation, bare soil, char and water endmembers, so
  NDVI/NBR values and their responses to clearing are in the right range;
* Sentinel-1 backscatter follows a saturating function of *structural* cover,
  with gamma-distributed speckle at the ~4.4 looks of a GRD IW product and a
  soil-moisture term that hits low-cover pixels hardest;
* Sentinel-2 scenes carry spatially-correlated cloud with an offset shadow,
  encoded in a real Sen2Cor SCL band;
* the scene contains **decoys that should not be called clearing**: cropping
  paddocks whose seasonal cycle mimics a clearing event in any single date
  pair, and a fire scar that collapses NBR while leaving the woody structure
  (and therefore the radar signal) largely intact.

That last point is the reason the demo fuses two sensors rather than one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from vegmon.config import PipelineConfig, demo_config
from vegmon.grid import Grid
from vegmon.masking import (
    SCL_CLOUD_HIGH_PROB,
    SCL_CLOUD_SHADOW,
    SCL_NOT_VEGETATED,
    SCL_THIN_CIRRUS,
    SCL_VEGETATION,
    SCL_WATER,
)
from vegmon.series import RadarSeries, OpticalSeries, S2_BANDS, Scene

BAND_ORDER = ("blue", "green", "red", "nir", "swir1", "swir2")

# Endmember surface reflectance, in canonical band order. Values are typical
# of the NSW inland slopes: red-brown duplex soils, sclerophyll canopy.
ENDMEMBERS: Dict[str, np.ndarray] = {
    # Woody canopy: deep, shadowed and moist, so NIR is high and SWIR is low.
    "gv": np.array([0.030, 0.060, 0.030, 0.420, 0.180, 0.080], dtype=np.float32),
    # Herbaceous green: the same chlorophyll signal, but a thin single-layer
    # canopy over bright soil with far less water in it. NIR is slightly lower
    # and SWIR much higher, which is why grass regrowth restores NDVI almost
    # fully while leaving NBR well short - the single most important thing to
    # get right when measuring whether woody vegetation is actually returning.
    "gv_herb": np.array([0.035, 0.075, 0.045, 0.400, 0.260, 0.140], dtype=np.float32),
    "npv": np.array([0.060, 0.100, 0.150, 0.280, 0.350, 0.300], dtype=np.float32),
    "soil": np.array([0.100, 0.140, 0.200, 0.280, 0.360, 0.330], dtype=np.float32),
    "char": np.array([0.030, 0.035, 0.040, 0.060, 0.070, 0.060], dtype=np.float32),
    "water": np.array([0.040, 0.050, 0.030, 0.015, 0.008, 0.005], dtype=np.float32),
}

# Land cover codes used internally by the generator.
LC_WOODLAND = 0
LC_CROP = 1
LC_PASTURE = 2
LC_WATER = 3

DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class PatchSpec:
    """One managed feature placed into the synthetic landscape.

    Coordinates are fractions of the grid so the same layout works at any
    raster size. ``kind`` drives the physics:

    ``clearing``  woody structure removed; both sensors respond
    ``fire``      canopy scorched, structure retained; optical only
    ``crop``      annual cropping cycle, no event at all
    """

    patch_id: int
    name: str
    kind: str
    centre: Tuple[float, float]
    area_ha: float
    event_date: Optional[date] = None
    recovery_tau_years: float = 3.5
    recovery_target: float = 0.9
    reclear_date: Optional[date] = None
    aspect: float = 1.0
    shape: str = "blob"


#: The bundled demo layout. Sizes span the range that matters for NSW
#: reporting: from below the 0.5 ha minimum mapping unit up to a whole paddock.
DEMO_PATCHES: Tuple[PatchSpec, ...] = (
    PatchSpec(1, "Paddock expansion", "clearing", (0.24, 0.22), 40.0,
              date(2020, 1, 15), recovery_tau_years=3.5, recovery_target=0.85),
    PatchSpec(2, "Stalled regrowth block", "clearing", (0.70, 0.20), 12.0,
              date(2020, 2, 10), recovery_tau_years=6.0, recovery_target=0.22),
    PatchSpec(3, "Fast regrowth block", "clearing", (0.46, 0.40), 5.0,
              date(2020, 1, 20), recovery_tau_years=1.8, recovery_target=0.95),
    PatchSpec(4, "Small clearing", "clearing", (0.82, 0.46), 2.0,
              date(2020, 2, 1), recovery_tau_years=4.0, recovery_target=0.8),
    PatchSpec(5, "Sub-hectare clearing", "clearing", (0.16, 0.55), 0.8,
              date(2020, 1, 25), recovery_tau_years=3.0, recovery_target=0.9),
    PatchSpec(6, "Below-MMU clearing", "clearing", (0.35, 0.62), 0.3,
              date(2020, 2, 5), recovery_tau_years=3.0, recovery_target=0.9),
    PatchSpec(7, "Re-cleared block", "clearing", (0.68, 0.72), 18.0,
              date(2020, 1, 5), recovery_tau_years=2.5, recovery_target=0.9,
              reclear_date=date(2023, 6, 1)),
    PatchSpec(8, "Fenceline widening", "clearing", (0.30, 0.86), 7.0,
              date(2020, 2, 20), recovery_tau_years=5.0, recovery_target=0.7,
              aspect=9.0, shape="strip"),
    PatchSpec(20, "Fire scar", "fire", (0.86, 0.80), 25.0,
              date(2020, 1, 10), recovery_tau_years=2.0, recovery_target=1.0),
    PatchSpec(30, "Winter crop paddock A", "crop", (0.10, 0.36), 30.0, None),
    PatchSpec(31, "Winter crop paddock B", "crop", (0.55, 0.08), 22.0, None),
    PatchSpec(32, "Winter crop paddock C", "crop", (0.90, 0.14), 16.0, None),
    PatchSpec(33, "Cropped then fallowed", "crop_fallow", (0.50, 0.55), 20.0,
              date(2020, 1, 1)),
)


@dataclass
class SyntheticTruth:
    """Ground truth that ships alongside the synthetic datacube."""

    labels: np.ndarray
    """``(y, x)`` int32 raster of patch ids; 0 is background."""

    patches: List[dict] = field(default_factory=list)
    """One record per placed patch, with kind, event date and true area."""

    grid: Optional[Grid] = None

    @property
    def clearing_mask(self) -> np.ndarray:
        """Boolean mask of pixels that were genuinely cleared."""
        ids = [p["patch_id"] for p in self.patches if p["kind"] == "clearing"]
        return np.isin(self.labels, ids) if ids else np.zeros(self.labels.shape, bool)

    @property
    def fire_mask(self) -> np.ndarray:
        ids = [p["patch_id"] for p in self.patches if p["kind"] == "fire"]
        return np.isin(self.labels, ids) if ids else np.zeros(self.labels.shape, bool)

    @property
    def crop_mask(self) -> np.ndarray:
        ids = [p["patch_id"] for p in self.patches if p["kind"].startswith("crop")]
        return np.isin(self.labels, ids) if ids else np.zeros(self.labels.shape, bool)

    @property
    def decoy_mask(self) -> np.ndarray:
        """Everything that changes but is *not* land clearing."""
        return self.fire_mask | self.crop_mask

    def clearing_patches(self) -> List[dict]:
        return [p for p in self.patches if p["kind"] == "clearing"]

    def total_cleared_ha(self) -> float:
        return float(sum(p["area_ha"] for p in self.clearing_patches()))


# ---------------------------------------------------------------------------
# noise helpers
# ---------------------------------------------------------------------------


def fractal_field(
    rng: np.random.Generator,
    shape: Tuple[int, int],
    scales: Sequence[float] = (2.0, 6.0, 18.0, 48.0),
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Sum of band-limited noise at several scales, normalised to 0-1.

    A cheap stand-in for the multi-scale spatial structure of real landscapes:
    broad soil and terrain trends with fine canopy texture on top.
    """
    if weights is None:
        weights = [s**0.85 for s in scales]
    out = np.zeros(shape, dtype=np.float32)
    for scale, weight in zip(scales, weights):
        noise = rng.standard_normal(shape).astype(np.float32)
        out += weight * ndimage.gaussian_filter(noise, sigma=scale, mode="wrap")
    out -= out.min()
    denom = out.max()
    return (out / denom).astype(np.float32) if denom > 0 else out


def _ar1(rng: np.random.Generator, n: int, rho: float = 0.62) -> np.ndarray:
    """A first-order autoregressive series, unit variance - the climate driver."""
    out = np.zeros(n, dtype=np.float32)
    innovation_sd = float(np.sqrt(1.0 - rho**2))
    for i in range(1, n):
        out[i] = rho * out[i - 1] + innovation_sd * rng.standard_normal()
    return out


def _blob_mask(
    shape: Tuple[int, int],
    centre: Tuple[float, float],
    area_pixels: float,
    rng: np.random.Generator,
    aspect: float = 1.0,
    roughness: float = 0.22,
) -> np.ndarray:
    """An irregular convex-ish patch of approximately ``area_pixels`` pixels.

    Real clearing follows paddock boundaries and terrain, so a perfect
    rectangle would make the sieve and polygonisation steps look better than
    they are. This perturbs an ellipse boundary with smooth noise, then
    rescales so the area still lands where it was asked to.
    """
    height, width = shape
    cy = centre[1] * height
    cx = centre[0] * width
    radius = np.sqrt(area_pixels * aspect / np.pi)
    ry = max(radius / np.sqrt(aspect), 1.0)
    rx = max(radius * np.sqrt(aspect) / aspect * aspect**0.5, 1.0)
    rx = max(np.sqrt(area_pixels * aspect / np.pi), 1.0)
    ry = max(rx / aspect, 1.0)

    yy, xx = np.mgrid[0:height, 0:width]
    dy = (yy - cy) / ry
    dx = (xx - cx) / rx
    dist = np.sqrt(dy * dy + dx * dx)

    wobble = ndimage.gaussian_filter(
        rng.standard_normal((height, width)).astype(np.float32), sigma=max(rx, ry) / 4.0
    )
    wobble_sd = float(np.std(wobble))
    if wobble_sd > 0:
        wobble = wobble / wobble_sd
    field = dist * (1.0 + roughness * wobble)

    target = int(round(area_pixels))
    target = max(1, min(target, height * width))
    flat = np.partition(field.ravel(), target - 1)
    cutoff = flat[target - 1]
    return field <= cutoff


def _strip_mask(
    shape: Tuple[int, int],
    centre: Tuple[float, float],
    area_pixels: float,
    rng: np.random.Generator,
    aspect: float = 9.0,
) -> np.ndarray:
    """A long thin clearing - a widened fenceline, track or pipeline easement."""
    height, width = shape
    half_width = max(np.sqrt(area_pixels / aspect) / 2.0, 1.0)
    length = area_pixels / (2.0 * half_width)
    angle = rng.uniform(-0.5, 0.5)
    cy = centre[1] * height
    cx = centre[0] * width
    yy, xx = np.mgrid[0:height, 0:width]
    along = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
    across = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
    jitter = ndimage.gaussian_filter(
        rng.standard_normal((height, width)).astype(np.float32), sigma=6.0
    )
    jitter_sd = float(np.std(jitter))
    if jitter_sd > 0:
        jitter = jitter / jitter_sd
    return (np.abs(across + 0.6 * jitter) <= half_width) & (np.abs(along) <= length / 2.0)


# ---------------------------------------------------------------------------
# landscape
# ---------------------------------------------------------------------------


@dataclass
class Landscape:
    """Static layers the temporal model draws on."""

    grid: Grid
    woody_cover: np.ndarray
    land_class: np.ndarray
    soil_brightness: np.ndarray
    labels: np.ndarray
    patches: List[dict]


def build_landscape(
    grid: Grid,
    rng: np.random.Generator,
    patches: Sequence[PatchSpec] = DEMO_PATCHES,
    area_scale: float = 1.0,
) -> Landscape:
    """Lay out woodland, cropping, pasture, water and the managed patches."""
    shape = grid.shape
    structure = fractal_field(rng, shape, scales=(1.5, 5.0, 16.0, 40.0))

    # Woody cover: a broad gradient with fine canopy texture, so the woodland
    # has realistic internal variation and soft edges rather than a hard step.
    woody = np.clip(0.15 + 1.25 * (structure - 0.32), 0.0, 0.92).astype(np.float32)
    woody += 0.06 * (fractal_field(rng, shape, scales=(1.0, 3.0)) - 0.5)
    woody = np.clip(woody, 0.0, 0.92).astype(np.float32)

    land_class = np.where(woody > 0.35, LC_WOODLAND, LC_PASTURE).astype(np.int8)

    soil = (0.75 + 0.5 * fractal_field(rng, shape, scales=(8.0, 30.0))).astype(np.float32)

    # A farm dam in the south-west corner.
    water = _blob_mask(shape, (0.12, 0.12), 0.012 * shape[0] * shape[1], rng, roughness=0.15)
    land_class[water] = LC_WATER
    woody[water] = 0.0

    labels = np.zeros(shape, dtype=np.int32)
    records: List[dict] = []
    pixel_ha = grid.pixel_area_ha

    for spec in patches:
        area_pixels = spec.area_ha * area_scale / pixel_ha
        if spec.shape == "strip":
            mask = _strip_mask(shape, spec.centre, area_pixels, rng, aspect=spec.aspect)
        else:
            mask = _blob_mask(shape, spec.centre, area_pixels, rng, aspect=spec.aspect)
        mask &= land_class != LC_WATER
        if not mask.any():
            continue
        # Later patches never overwrite earlier ones.
        mask &= labels == 0
        if not mask.any():
            continue
        labels[mask] = spec.patch_id

        if spec.kind in ("clearing", "fire"):
            # Guarantee the pre-event state really is woody, so the detector's
            # pre-condition gates are exercised rather than accidentally met.
            woody[mask] = np.clip(
                0.72 + 0.10 * (fractal_field(rng, shape, scales=(1.0, 4.0))[mask] - 0.5),
                0.55,
                0.9,
            )
            land_class[mask] = LC_WOODLAND
        elif spec.kind in ("crop", "crop_fallow"):
            woody[mask] = 0.0
            land_class[mask] = LC_CROP

        records.append(
            {
                "patch_id": spec.patch_id,
                "name": spec.name,
                "kind": spec.kind,
                "event_date": spec.event_date.isoformat() if spec.event_date else None,
                "reclear_date": spec.reclear_date.isoformat() if spec.reclear_date else None,
                "recovery_tau_years": spec.recovery_tau_years,
                "recovery_target": spec.recovery_target,
                "requested_area_ha": spec.area_ha * area_scale,
                "area_ha": float(mask.sum()) * pixel_ha,
                "pixels": int(mask.sum()),
            }
        )

    return Landscape(
        grid=grid,
        woody_cover=woody,
        land_class=land_class,
        soil_brightness=soil,
        labels=labels,
        patches=records,
    )


# ---------------------------------------------------------------------------
# temporal model
# ---------------------------------------------------------------------------


def _years_since(times: np.ndarray, event: date) -> np.ndarray:
    return (times - np.datetime64(event, "D")).astype("timedelta64[D]").astype(
        np.float32
    ) / DAYS_PER_YEAR


def _cover_trajectory(
    pre_cover: np.ndarray,
    years: float,
    tau: float,
    target: float,
    residual: float = 0.03,
) -> np.ndarray:
    """Saturating-exponential recovery of woody cover after removal."""
    if years < 0:
        return pre_cover
    asymptote = pre_cover * target
    return (asymptote + (residual - asymptote) * np.exp(-years / max(tau, 1e-3))).astype(
        np.float32
    )


def _herbaceous_response(years: float) -> float:
    """Post-clearing herbaceous multiplier.

    Ground cover is scraped bare at the event, then flushes above its normal
    level for a few years while light and nutrients are abundant, and settles
    back as woody cover closes over. This is why NDVI recovers much faster
    than NBR after clearing - and why NDVI alone overstates recovery.
    """
    if years < 0:
        return 1.0
    if years < 0.4:
        return 0.28
    return float(0.85 + 0.35 * np.exp(-(years - 0.4) / 2.2))


def _seasonal_herbaceous(doy: np.ndarray, peak_doy: float, amplitude: float, base: float) -> np.ndarray:
    phase = 2.0 * np.pi * (doy - peak_doy) / DAYS_PER_YEAR
    return np.clip(base + amplitude * np.cos(phase), 0.0, 1.0).astype(np.float32)


def _unmix(fractions: Dict[str, np.ndarray]) -> np.ndarray:
    """Linear spectral mixing to a ``(band, y, x)`` reflectance stack."""
    shape = next(iter(fractions.values())).shape
    out = np.zeros((len(BAND_ORDER),) + shape, dtype=np.float32)
    for name, frac in fractions.items():
        spectrum = ENDMEMBERS[name]
        out += spectrum[:, None, None] * frac[None, ...]
    return out


def _cloud_scene(
    rng: np.random.Generator, shape: Tuple[int, int], cover: float
) -> Tuple[np.ndarray, np.ndarray]:
    """A spatially-correlated cloud mask and its offset shadow."""
    if cover <= 0.001:
        empty = np.zeros(shape, dtype=bool)
        return empty, empty
    field = fractal_field(rng, shape, scales=(6.0, 20.0, 55.0))
    cutoff = float(np.quantile(field, 1.0 - min(cover, 1.0)))
    cloud = field >= cutoff
    shift_y = int(rng.integers(4, 14))
    shift_x = int(rng.integers(-12, -3))
    shadow = np.roll(np.roll(cloud, shift_y, axis=0), shift_x, axis=1) & ~cloud
    return cloud, shadow


def _cloud_cover_for(doy: float, rng: np.random.Generator) -> float:
    """Draw a scene cloud fraction; summer storms make the wet season cloudier."""
    seasonal = 0.30 + 0.16 * np.cos(2.0 * np.pi * (doy - 15.0) / DAYS_PER_YEAR)
    concentration = 2.4
    alpha = max(seasonal * concentration, 0.05)
    beta = max((1.0 - seasonal) * concentration, 0.05)
    return float(rng.beta(alpha, beta))


def _speckle_db(rng: np.random.Generator, shape: Tuple[int, int], looks: float = 4.4) -> np.ndarray:
    """Multiplicative speckle for a multi-looked SAR intensity image, in dB."""
    gamma = rng.gamma(shape=looks, scale=1.0 / looks, size=shape).astype(np.float32)
    return (10.0 * np.log10(np.maximum(gamma, 1e-6))).astype(np.float32)


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------


def _date_range(start: date, end: date, step_days: int, offset_days: int = 0) -> List[date]:
    first = start + timedelta(days=offset_days)
    n = max(0, (end - first).days // step_days + 1)
    return [first + timedelta(days=step_days * i) for i in range(n)]


def _state_at(
    land: Landscape,
    specs: Dict[int, PatchSpec],
    when: date,
    climate: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Cover fractions and structural cover for a single acquisition date."""
    shape = land.grid.shape
    woody = land.woody_cover.copy()
    structural = land.woody_cover.copy()
    char = np.zeros(shape, dtype=np.float32)
    herb_mult = np.ones(shape, dtype=np.float32)

    doy = float(when.timetuple().tm_yday)

    for patch_id, spec in specs.items():
        if spec.event_date is None:
            continue
        mask = land.labels == patch_id
        if not mask.any():
            continue
        years = (when - spec.event_date).days / DAYS_PER_YEAR
        if years < 0:
            continue
        pre = land.woody_cover[mask]

        if spec.kind == "clearing":
            cover = _cover_trajectory(pre, years, spec.recovery_tau_years, spec.recovery_target)
            elapsed = years
            if spec.reclear_date is not None and when >= spec.reclear_date:
                years2 = (when - spec.reclear_date).days / DAYS_PER_YEAR
                at_reclear = _cover_trajectory(
                    pre,
                    (spec.reclear_date - spec.event_date).days / DAYS_PER_YEAR,
                    spec.recovery_tau_years,
                    spec.recovery_target,
                )
                cover = _cover_trajectory(
                    at_reclear, years2, spec.recovery_tau_years, spec.recovery_target
                )
                elapsed = years2
            woody[mask] = cover
            structural[mask] = cover
            herb_mult[mask] = _herbaceous_response(elapsed)

        elif spec.kind == "crop_fallow":
            # A paddock cropped in the baseline year and left bare afterwards.
            # Spectrally this is a large, persistent NDVI and NBR drop with no
            # woody vegetation anywhere near it - the classic false positive.
            herb_mult[mask] = 0.12

        elif spec.kind == "fire":
            # Canopy scorched but stems standing: optical greenness collapses,
            # structure - and therefore radar backscatter - barely moves.
            scorch = float(np.exp(-years / 1.5))
            woody[mask] = pre * (1.0 - 0.92 * scorch)
            structural[mask] = pre * (1.0 - 0.16 * scorch)
            char[mask] = 0.68 * scorch
            herb_mult[mask] = 0.35 + 0.65 * (1.0 - scorch)

    # Herbaceous ground cover: seasonal cycle by land class, scaled by climate.
    herb = np.zeros(shape, dtype=np.float32)
    doy_arr = np.full(shape, doy, dtype=np.float32)
    crop = land.land_class == LC_CROP
    pasture = land.land_class == LC_PASTURE
    woodland = land.land_class == LC_WOODLAND
    herb[crop] = _seasonal_herbaceous(doy_arr[crop], peak_doy=275.0, amplitude=0.45, base=0.50)
    herb[pasture] = _seasonal_herbaceous(doy_arr[pasture], peak_doy=60.0, amplitude=0.22, base=0.35)
    herb[woodland] = _seasonal_herbaceous(doy_arr[woodland], peak_doy=60.0, amplitude=0.10, base=0.25)
    herb *= np.clip(1.0 + 0.16 * climate, 0.4, 1.6)
    herb *= herb_mult
    herb = np.clip(herb + 0.03 * rng.standard_normal(shape).astype(np.float32), 0.0, 1.0)

    water = land.land_class == LC_WATER
    return {
        "woody": np.clip(woody, 0.0, 0.95).astype(np.float32),
        "structural": np.clip(structural, 0.0, 0.95).astype(np.float32),
        "char": char,
        "herb": herb.astype(np.float32),
        "water": water,
    }


def _reflectance(state: Dict[str, np.ndarray], land: Landscape) -> np.ndarray:
    """Turn a cover state into a ``(band, y, x)`` reflectance stack."""
    woody = state["woody"]
    herb = state["herb"]
    char = state["char"]
    open_ground = 1.0 - woody

    gv_woody = woody * 0.65
    gv_herb = open_ground * herb * 0.80
    npv = woody * 0.28 + open_ground * herb * 0.20 + open_ground * (1.0 - herb) * 0.35
    # Char displaces whatever else was there rather than adding to it.
    scale = np.clip(1.0 - char, 0.0, 1.0)
    gv_woody = gv_woody * scale
    gv_herb = gv_herb * scale
    npv = npv * scale
    soil = np.clip(1.0 - gv_woody - gv_herb - npv - char, 0.0, 1.0)

    water = state["water"]
    fractions = {
        "gv": np.where(water, 0.0, gv_woody).astype(np.float32),
        "gv_herb": np.where(water, 0.0, gv_herb).astype(np.float32),
        "npv": np.where(water, 0.0, npv).astype(np.float32),
        "soil": np.where(water, 0.0, soil * land.soil_brightness).astype(np.float32),
        "char": np.where(water, 0.0, char).astype(np.float32),
        "water": water.astype(np.float32),
    }
    return _unmix(fractions)


def _backscatter(
    state: Dict[str, np.ndarray],
    climate: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sentinel-1 VV and VH in dB from structural cover and soil moisture."""
    structural = state["structural"]
    herb = state["herb"]
    effective = np.clip(structural + 0.18 * (1.0 - structural) * herb, 0.0, 1.0)

    # Saturating rise of volume scattering with structural cover.
    vh = -20.5 + 6.2 * np.power(effective, 0.7)
    # Soil moisture lifts backscatter, and does so most where cover is thin.
    vh = vh + climate * (0.25 + 0.85 * (1.0 - effective))
    # Cross-ratio narrows as cover increases.
    vv = vh + 5.5 + 3.0 * (1.0 - effective)

    water = state["water"]
    vh = np.where(water, -26.0, vh)
    vv = np.where(water, -20.0, vv)

    shape = structural.shape
    vh = vh + _speckle_db(rng, shape)
    vv = vv + _speckle_db(rng, shape)
    return vv.astype(np.float32), vh.astype(np.float32)


def _scl_from_state(
    reflectance: np.ndarray,
    state: Dict[str, np.ndarray],
    cloud: np.ndarray,
    shadow: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    red = reflectance[BAND_ORDER.index("red")]
    nir = reflectance[BAND_ORDER.index("nir")]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    scl = np.where(ndvi > 0.30, SCL_VEGETATION, SCL_NOT_VEGETATED).astype(np.uint8)
    scl[state["water"]] = SCL_WATER
    scl[shadow] = SCL_CLOUD_SHADOW
    # A thin cirrus fringe around the thick cloud, as Sen2Cor tends to produce.
    fringe = ndimage.binary_dilation(cloud, iterations=2) & ~cloud
    scl[fringe] = SCL_THIN_CIRRUS
    scl[cloud] = SCL_CLOUD_HIGH_PROB
    return scl


def generate_scene(
    config: Optional[PipelineConfig] = None,
    size: int = 320,
    seed: int = 20240917,
    optical_revisit_days: int = 5,
    radar_revisit_days: int = 12,
    patches: Sequence[PatchSpec] = DEMO_PATCHES,
    area_scale: Optional[float] = None,
    progress: bool = False,
) -> Scene:
    """Generate a full synthetic Sentinel-1 + Sentinel-2 datacube.

    Parameters
    ----------
    size:
        Raster side length in pixels. At the default 10 m resolution, 320
        pixels is a 3.2 km square (1024 ha) - big enough to hold a 40 ha
        clearing and small enough to run in about a minute.
    seed:
        Everything is reproducible from this.
    area_scale:
        Multiplier on every patch area. Defaults to ``(size / 320) ** 2`` so
        that a smaller raster holds the same *proportion* of disturbance
        rather than the same hectares - without it, a 96-pixel test grid ends
        up more than half cleared, which breaks the assumption that most of
        the scene is stable.

    Returns
    -------
    Scene
        With ``truth`` populated: a label raster and per-patch records.
    """
    config = config or demo_config()
    rng = np.random.default_rng(seed)

    # Place the grid on the AOI centroid in the configured projected CRS.
    grid = _grid_for(config, size)
    if area_scale is None:
        area_scale = (size / 320.0) ** 2
    land = build_landscape(grid, rng, patches, area_scale=area_scale)
    specs = {spec.patch_id: spec for spec in patches}

    periods = config.periods
    optical_dates = _date_range(periods.series_start, periods.series_end, optical_revisit_days)
    radar_dates = sorted(
        _date_range(periods.series_start, periods.series_end, radar_revisit_days)
        + _date_range(periods.series_start, periods.series_end, radar_revisit_days, offset_days=6)
    )
    orbit_state = [
        "ascending" if ((d - periods.series_start).days // 6) % 2 == 0 else "descending"
        for d in radar_dates
    ]

    # One shared climate driver, sampled monthly and interpolated to each date.
    months = max(1, int((periods.series_end - periods.series_start).days / 30.4) + 2)
    climate_monthly = _ar1(rng, months)
    origin = periods.series_start

    def climate_at(when: date) -> float:
        position = (when - origin).days / 30.4
        return float(np.interp(position, np.arange(months), climate_monthly))

    n_optical = len(optical_dates)
    bands = {
        name: np.empty((n_optical,) + grid.shape, dtype=np.float32) for name in S2_BANDS
    }
    scl = np.empty((n_optical,) + grid.shape, dtype=np.uint8)

    for i, when in enumerate(optical_dates):
        state = _state_at(land, specs, when, climate_at(when), rng)
        reflectance = _reflectance(state, land)
        reflectance += rng.normal(0.0, 0.006, reflectance.shape).astype(np.float32)
        np.clip(reflectance, 0.0, 1.0, out=reflectance)
        cover = _cloud_cover_for(float(when.timetuple().tm_yday), rng)
        cloud, shadow = _cloud_scene(rng, grid.shape, cover)
        scl[i] = _scl_from_state(reflectance, state, cloud, shadow, rng)
        # Cloud is bright and cold; shadow is dark. Leave the values in place so
        # a consumer that ignores SCL gets burned exactly like they would in real life.
        reflectance[:, cloud] = np.clip(
            reflectance[:, cloud] + rng.uniform(0.25, 0.55), 0.0, 1.0
        )
        reflectance[:, shadow] *= 0.45
        for b, name in enumerate(BAND_ORDER):
            bands[name][i] = reflectance[b]
        if progress and i % 50 == 0:
            print(f"  optical {i + 1}/{n_optical} {when}", flush=True)

    n_radar = len(radar_dates)
    vv = np.empty((n_radar,) + grid.shape, dtype=np.float32)
    vh = np.empty((n_radar,) + grid.shape, dtype=np.float32)
    for i, when in enumerate(radar_dates):
        state = _state_at(land, specs, when, climate_at(when), rng)
        vv[i], vh[i] = _backscatter(state, climate_at(when), rng)
        if progress and i % 50 == 0:
            print(f"  radar {i + 1}/{n_radar} {when}", flush=True)

    optical = OpticalSeries(grid=grid, times=optical_dates, bands=bands, scl=scl,
                            sensor="synthetic-sentinel-2-l2a")
    radar = RadarSeries(grid=grid, times=radar_dates, vv_db=vv, vh_db=vh,
                        orbit_state=orbit_state, sensor="synthetic-sentinel-1-grd")
    truth = SyntheticTruth(labels=land.labels, patches=land.patches, grid=grid)
    return Scene(optical=optical, radar=radar, grid=grid, truth={"synthetic": truth})


def _grid_for(config: PipelineConfig, size: int) -> Grid:
    """A projected grid of ``size`` pixels centred on the AOI."""
    lon, lat = config.aoi.centroid
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", config.aoi.crs, always_xy=True)
        east, north = transformer.transform(lon, lat)
    except Exception:  # pragma: no cover - pyproj is optional for the demo
        east, north = 700000.0, 6580000.0
    half = size * config.resolution / 2.0
    return Grid.from_origin(
        crs=config.aoi.crs,
        west=float(east) - half,
        north=float(north) + half,
        resolution=config.resolution,
        width=size,
        height=size,
    )


def scene_truth(scene: Scene) -> SyntheticTruth:
    """Extract the synthetic ground truth from a scene, or raise."""
    if not scene.truth or "synthetic" not in scene.truth:
        raise ValueError("scene carries no synthetic ground truth")
    return scene.truth["synthetic"]  # type: ignore[return-value]


__all__ = [
    "BAND_ORDER",
    "DEMO_PATCHES",
    "ENDMEMBERS",
    "Landscape",
    "PatchSpec",
    "SyntheticTruth",
    "build_landscape",
    "fractal_field",
    "generate_scene",
    "scene_truth",
]


# ---------------------------------------------------------------------------
# persistence - generating a scene is the slow part, so cache it
# ---------------------------------------------------------------------------


def save_scene(scene: Scene, path) -> "object":
    """Write a scene (and its truth) to a compressed ``.npz``."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    truth = scene.truth["synthetic"] if scene.truth else None
    payload = {
        "optical_times": scene.optical.times.astype("datetime64[D]").astype("int64"),
        "scl": scene.optical.scl,
        "radar_times": scene.radar.times.astype("datetime64[D]").astype("int64"),
        "vv_db": scene.radar.vv_db,
        "vh_db": scene.radar.vh_db,
        "meta": np.frombuffer(
            json.dumps(
                {
                    "grid": {
                        "crs": scene.grid.crs,
                        "transform": list(scene.grid.transform),
                        "width": scene.grid.width,
                        "height": scene.grid.height,
                    },
                    "orbit_state": list(scene.radar.orbit_state or []),
                    "patches": truth.patches if truth else [],
                }
            ).encode("utf-8"),
            dtype=np.uint8,
        ),
    }
    for name, arr in scene.optical.bands.items():
        payload[f"band_{name}"] = arr
    if truth is not None:
        payload["labels"] = truth.labels
    np.savez_compressed(path, **payload)
    return path


def load_scene(path) -> Scene:
    """Read back a scene written by :func:`save_scene`."""
    import json

    data = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(data["meta"]).decode("utf-8"))
    grid = Grid(
        crs=meta["grid"]["crs"],
        transform=tuple(meta["grid"]["transform"]),
        width=meta["grid"]["width"],
        height=meta["grid"]["height"],
    )
    optical = OpticalSeries(
        grid=grid,
        times=data["optical_times"].astype("datetime64[D]"),
        bands={name: data[f"band_{name}"] for name in S2_BANDS},
        scl=data["scl"],
        sensor="synthetic-sentinel-2-l2a",
    )
    radar = RadarSeries(
        grid=grid,
        times=data["radar_times"].astype("datetime64[D]"),
        vv_db=data["vv_db"],
        vh_db=data["vh_db"],
        orbit_state=meta["orbit_state"] or None,
        sensor="synthetic-sentinel-1-grd",
    )
    truth = None
    if "labels" in data:
        truth = SyntheticTruth(labels=data["labels"], patches=meta["patches"], grid=grid)
    return Scene(
        optical=optical,
        radar=radar,
        grid=grid,
        truth={"synthetic": truth} if truth is not None else None,
    )


__all__ += ["load_scene", "save_scene"]
