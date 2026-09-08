"""Configuration objects for the clearing-detection / regrowth pipeline.

Defaults are tuned for woody vegetation on the NSW inland slopes and plains
(box-gum woodland, brigalow, cypress pine), 10 m Sentinel-2 and 10-20 m
Sentinel-1. They are a starting point for calibration, not gospel: every
threshold here should be re-derived against local reference sites before the
pipeline is used operationally.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"cannot interpret {value!r} as a date")


@dataclass(frozen=True)
class AOI:
    """Area of interest as a lon/lat bounding box plus a working CRS.

    ``crs`` is the projected CRS the analysis is done in. Use the local UTM
    zone so that pixel areas are metric and comparable: zone 55S (EPSG:32755)
    covers 147-153 E, which is most of eastern NSW; zone 56S (EPSG:32756) picks
    up the coast east of 150 E.
    """

    name: str
    west: float
    south: float
    east: float
    north: float
    crs: str = "EPSG:32755"

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.west + self.east) / 2.0, (self.south + self.north) / 2.0)

    def __post_init__(self) -> None:
        if self.west >= self.east or self.south >= self.north:
            raise ValueError(f"AOI {self.name!r} has an empty or inverted bounding box")


@dataclass(frozen=True)
class Periods:
    """The three time windows the pipeline needs.

    ``pre_start``..``pre_end``   baseline composite, before the suspected clearing
    ``post_start``..``post_end`` detection composite, after it
    ``series_start``..``series_end`` the full record used for regrowth fitting

    The pre/post windows should each be long enough to guarantee several
    cloud-free Sentinel-2 observations (a full season is typical for inland
    NSW) but short enough not to straddle the event itself.
    """

    pre_start: date
    pre_end: date
    post_start: date
    post_end: date
    series_start: date
    series_end: date

    def __post_init__(self) -> None:
        object.__setattr__(self, "pre_start", _as_date(self.pre_start))
        object.__setattr__(self, "pre_end", _as_date(self.pre_end))
        object.__setattr__(self, "post_start", _as_date(self.post_start))
        object.__setattr__(self, "post_end", _as_date(self.post_end))
        object.__setattr__(self, "series_start", _as_date(self.series_start))
        object.__setattr__(self, "series_end", _as_date(self.series_end))
        if self.pre_end > self.post_start:
            raise ValueError("pre-period must end before the post-period starts")
        if self.series_start > self.pre_start or self.series_end < self.post_end:
            raise ValueError("series window must span both the pre- and post-periods")

    @property
    def event_window(self) -> Tuple[date, date]:
        """The interval the clearing event must fall inside."""
        return (self.pre_end, self.post_start)

    def phenology_offset_days(self) -> int:
        """How far the post-window's seasonal position differs from the pre's.

        Comparing a March-August composite against a March-November one is the
        most common way to manufacture false positives in cropping country: the
        baseline picks up the spring green peak and the later window does not,
        so every paddock in the district reads as cleared. Composites should
        cover the same months in both years.
        """
        pre_mid = self.pre_start + (self.pre_end - self.pre_start) / 2
        post_mid = self.post_start + (self.post_end - self.post_start) / 2
        offset = abs(pre_mid.timetuple().tm_yday - post_mid.timetuple().tm_yday)
        return int(min(offset, 365 - offset))

    def check_phenology(self, tolerance_days: int = 21) -> None:
        """Warn if the pre and post windows are not seasonally aligned."""
        offset = self.phenology_offset_days()
        if offset > tolerance_days:
            warnings.warn(
                f"pre- and post-period composites are {offset} days apart in the "
                "seasonal cycle; expect phenological false positives over crop "
                "and pasture. Match the months in both windows.",
                UserWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class DetectionConfig:
    """Thresholds for the Sentinel-2, Sentinel-1 and fusion stages."""

    # --- Sentinel-2 -------------------------------------------------------
    min_pre_ndvi: float = 0.40
    """Pre-event NDVI floor. Screens out crop, pasture and bare ground so the
    detector only ever fires on something that was woody to begin with.

    Set for the ~20% foliage-cover definition of woody vegetation used in NSW
    reporting, not for closed forest. A closed-canopy threshold would quietly
    exclude the grassy box woodland and open cypress country where most
    clearing actually happens."""

    min_pre_nbr: float = 0.18
    """Pre-event NBR floor. Second woody-cover gate, less sensitive to the
    green flush that makes winter crop look like forest in NDVI. Also set for
    open woodland rather than closed forest."""

    min_pre_ndvi_persistent: float = 0.35
    """Persistence gate: the low percentile of pre-event NDVI must clear this.

    This is the single most valuable false-positive filter in cropping
    country. A winter-crop paddock reaches NDVI 0.7 in September and drops
    below 0.25 after harvest, so its seasonal *median* can pass a woody
    threshold - but its low percentile cannot. Woody vegetation stays green
    all year, so it does."""

    pre_percentile: float = 10.0
    """Percentile used for the persistence gate above."""

    dnbr_threshold: float = 0.20
    """Minimum NBR drop (pre - post) to call a pixel disturbed. ~0.10-0.25 is
    the usual 'low severity' band in the fire literature; clearing sits well
    above it, so this is deliberately conservative."""

    dndvi_threshold: float = 0.15
    """Minimum NDVI drop, applied jointly with dNBR to suppress false alarms
    from soil-moisture and burn-scar darkening."""

    min_clear_observations: int = 3
    """Minimum cloud-free observations required in each of the pre and post
    windows before a pixel is eligible. Fewer than this and the composite is
    too noisy to trust."""

    max_cloud_cover: float = 0.6
    """Scene-level cloud fraction above which a Sentinel-2 acquisition is
    dropped before compositing."""

    # --- Sentinel-1 -------------------------------------------------------
    vh_drop_db: float = 2.0
    """Minimum VH backscatter drop in dB. Removing woody structure collapses
    volume scattering, so VH falls sharply; 2 dB clears the residual speckle
    after multi-looking without needing a clear sky."""

    min_pre_vh_db: float = -20.0
    """Pre-event VH floor in dB. Below this the target was already too smooth
    (water, bare soil, saltpan) to be standing vegetation."""

    speckle_window: int = 5
    """Side length in pixels of the boxcar multi-look filter applied to each
    S1 composite. Trades spatial detail for a usable signal-to-noise ratio."""

    min_s1_observations: int = 3
    """Minimum S1 acquisitions per window. Temporal median over >=3 passes is
    what makes the dB drop robust to speckle and to soil-moisture spikes."""

    # --- Fusion and cleanup ----------------------------------------------
    min_mapping_unit_ha: float = 0.5
    """Patches smaller than this are dropped. NSW clearing reporting works in
    hectares; sub-0.5 ha detections at 10 m are mostly edge noise."""

    opening_iterations: int = 1
    """Binary opening passes applied before sieving, to shave off one-pixel
    filaments along paddock edges and drainage lines."""

    confirm_overlap: float = 0.30
    """Fraction of a patch that both sensors must agree on before it is called
    CONFIRMED. Agreement is never pixel-perfect - Sentinel-1 is multi-looked
    and geolocated slightly differently - so requiring a majority overlap
    rather than every pixel keeps real events out of the single-sensor tiers."""

    fill_holes: bool = True
    """Fill interior holes in patches (retained scattered trees read as holes
    but are still part of the same clearing event)."""


@dataclass(frozen=True)
class RegrowthConfig:
    """Parameters for the per-patch recovery analysis."""

    season_months: int = 3
    """Length of the temporal bins the NDVI series is aggregated into before
    fitting. Quarterly medians damp the phenological cycle without erasing the
    recovery signal."""

    min_seasons_post: int = 4
    """Minimum post-event seasonal composites needed to attempt a fit."""

    recovered_fraction: float = 0.80
    """Fraction of the pre-event NDVI baseline that counts as recovered. 0.8 is
    the R80P convention from the disturbance-recovery literature."""

    stalled_fraction: float = 0.30
    """Below this fraction of baseline, with no meaningful positive trend, a
    patch is called stalled rather than recovering."""

    stalled_slope_per_year: float = 0.01
    """NDVI per year. A recent trend below this is treated as flat."""

    recent_years: float = 2.0
    """Length of the trailing window used for the recent-trend estimate."""

    reclear_drop: float = 0.15
    """Index drop below the trailing seasonal level that flags a re-clearing.

    In NBR units. Must clear the year-to-year swing that drought alone
    produces over intact vegetation, which is why the drop also has to be
    sustained into the following season before it counts."""

    min_recovery_for_reclear: float = 0.10
    """A patch must have regained at least this much index value above its
    post-clearing trough before a subsequent drop can be called a
    re-clearing - otherwise the site never recovered in the first place."""


@dataclass(frozen=True)
class PipelineConfig:
    """Everything the pipeline needs for one AOI."""

    aoi: AOI
    periods: Periods
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    regrowth: RegrowthConfig = field(default_factory=RegrowthConfig)
    resolution: float = 10.0
    """Working pixel size in metres. Sentinel-1 is resampled to match."""

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)

    def to_json(self, path: str | Path, indent: int = 2) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=indent))
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineConfig":
        return cls(
            aoi=AOI(**data["aoi"]),
            periods=Periods(**data["periods"]),
            detection=DetectionConfig(**data.get("detection", {})),
            regrowth=RegrowthConfig(**data.get("regrowth", {})),
            resolution=float(data.get("resolution", 10.0)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# A worked example: the Pilliga / Narrabri area of the NSW north-west slopes,
# used by the bundled demo. Roughly 5 x 5 km of mixed cypress-pine woodland,
# cropping and cleared paddock.
PILLIGA_DEMO_AOI = AOI(
    name="Pilliga East (demo)",
    west=149.10,
    south=-30.95,
    east=149.16,
    north=-30.90,
    crs="EPSG:32755",
)

DEMO_PERIODS = Periods(
    pre_start=date(2019, 3, 1),
    pre_end=date(2019, 11, 30),
    post_start=date(2020, 3, 1),
    post_end=date(2020, 11, 30),
    series_start=date(2019, 1, 1),
    series_end=date(2025, 12, 31),
)


def demo_config() -> PipelineConfig:
    """The configuration used by ``vegmon demo``."""
    return PipelineConfig(aoi=PILLIGA_DEMO_AOI, periods=DEMO_PERIODS)


__all__ = [
    "AOI",
    "DEMO_PERIODS",
    "DetectionConfig",
    "PILLIGA_DEMO_AOI",
    "Periods",
    "PipelineConfig",
    "RegrowthConfig",
    "demo_config",
]
