"""Characterise an AOI before detecting anything in it.

This exists because of a mistake. The first real run of this pipeline was
pointed at an AOI chosen for having the most persistently-green pixels, with
thresholds carried over from synthetic data, and it confidently reported 80 ha
of land clearing. The AOI was the Namoi irrigated cotton district and the
"clearing" was a paddock rotation. Nothing about the maps looked wrong.

What would have caught it was looking at the distributions first: how green
this landscape is, how hard it swings through the year, how much of it is
actually woody, and whether the two years being compared are even comparable.
That is what this module reports, and it should be run before any detection
run on new country.

Nothing here makes a decision. It prints the numbers a person needs to choose
a baseline year, a detection year, and a starting set of thresholds - and
flags the specific conditions that are known to produce confident nonsense.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vegmon.config import DetectionConfig
from vegmon.grid import Grid
from vegmon.series import OpticalSeries

#: A year is only usable as a baseline with at least this many clear views.
MIN_CLEAR_SCENES = 12

#: Two years further apart than this in median greenness are not a fair pair:
#: the difference between them is climate, not change.
MAX_COMPARABLE_NDVI_GAP = 0.12


@dataclass
class YearStats:
    """What one year of Sentinel-2 says about a landscape."""

    year: int
    clear_scenes: int
    ndvi_median: float
    ndvi_p90: float
    nbr_median: float
    persistent_ndvi_p85: float
    """85th percentile of the low temporal percentile - the level an adaptive
    woody gate would land on."""

    amplitude_median: float
    """Median within-year NDVI swing. High means cropping dominates."""

    woody_fraction: float
    """Share of the AOI that is persistently green *and* steady through the
    year. Persistence alone is not a woody test - irrigated cropping passes
    it."""

    crop_fraction: float
    mean_cloud: float

    @classmethod
    def empty(cls, year: int) -> "YearStats":
        """A year with no usable observations. Named rather than positional so
        that adding a field cannot silently shift the others."""
        nan = float("nan")
        return cls(
            year=year, clear_scenes=0, ndvi_median=nan, ndvi_p90=nan, nbr_median=nan,
            persistent_ndvi_p85=nan, amplitude_median=nan, woody_fraction=nan,
            crop_fraction=nan, mean_cloud=nan,
        )

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class PairAdvice:
    """Whether two consecutive years can fairly be compared."""

    baseline: int
    detection: int
    ndvi_gap: float
    min_clear_scenes: int
    comparable: bool
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "baseline": self.baseline,
            "detection": self.detection,
            "ndvi_gap": round(self.ndvi_gap, 4),
            "min_clear_scenes": self.min_clear_scenes,
            "comparable": self.comparable,
            "notes": "; ".join(self.notes),
        }


@dataclass
class SurveyResult:
    aoi_name: str
    area_ha: float
    years: List[YearStats]
    pairs: List[PairAdvice]
    warnings: List[str] = field(default_factory=list)
    suggested_thresholds: Dict[str, float] = field(default_factory=dict)

    def best_pair(self) -> Optional[PairAdvice]:
        usable = [p for p in self.pairs if p.comparable]
        if not usable:
            return None
        return min(usable, key=lambda p: (p.ndvi_gap, -p.min_clear_scenes))

    def as_dict(self) -> dict:
        return {
            "aoi": self.aoi_name,
            "area_ha": round(self.area_ha, 1),
            "years": [y.as_dict() for y in self.years],
            "pairs": [p.as_dict() for p in self.pairs],
            "warnings": self.warnings,
            "suggested_thresholds": {
                k: round(v, 4) for k, v in self.suggested_thresholds.items()
            },
        }


def _temporal_percentile(series: OpticalSeries, index: str, percentile: float) -> np.ndarray:
    stack = series.index_stack(index)
    all_nan = np.all(~np.isfinite(stack), axis=0)
    safe = np.where(np.isfinite(stack), stack, np.nan)
    safe = np.where(all_nan[None, ...], 0.0, safe)
    with np.errstate(invalid="ignore"):
        out = np.nanpercentile(safe, percentile, axis=0).astype(np.float32)
    return np.where(all_nan, np.nan, out)


def describe_year(
    series: OpticalSeries,
    year: int,
    config: Optional[DetectionConfig] = None,
) -> YearStats:
    """Summarise one calendar year of a loaded optical series."""
    config = config or DetectionConfig()
    window = series.select(date(year, 1, 1), date(year, 12, 31),
                           max_cloud_cover=config.max_cloud_cover)
    if len(window) == 0:
        return YearStats.empty(year)

    ndvi_typical = _temporal_percentile(window, "ndvi", 50.0)
    nbr_typical = _temporal_percentile(window, "nbr", 50.0)
    persistent = _temporal_percentile(window, "ndvi", config.pre_percentile)
    amplitude = (
        _temporal_percentile(window, "ndvi", 90.0) - _temporal_percentile(window, "ndvi", 10.0)
    )

    finite = np.isfinite(persistent) & np.isfinite(amplitude)
    woody = finite & (persistent >= config.min_pre_ndvi_persistent) & (amplitude <= 0.35)
    crop = finite & (amplitude > 0.45)

    def pct(values: np.ndarray, q: float) -> float:
        sample = values[np.isfinite(values)]
        return float(np.percentile(sample, q)) if sample.size else float("nan")

    return YearStats(
        year=year,
        clear_scenes=len(window),
        ndvi_median=pct(ndvi_typical, 50),
        ndvi_p90=pct(ndvi_typical, 90),
        nbr_median=pct(nbr_typical, 50),
        persistent_ndvi_p85=pct(persistent, 85),
        amplitude_median=pct(amplitude, 50),
        woody_fraction=float(woody.sum() / max(finite.sum(), 1)),
        crop_fraction=float(crop.sum() / max(finite.sum(), 1)),
        mean_cloud=float(np.mean(window.scene_cloud_fraction())),
    )


def compare_years(earlier: YearStats, later: YearStats) -> PairAdvice:
    """Decide whether two consecutive years form a fair before/after pair."""
    gap = abs(earlier.ndvi_median - later.ndvi_median)
    fewest = min(earlier.clear_scenes, later.clear_scenes)
    notes: List[str] = []
    comparable = True

    if not np.isfinite(gap):
        return PairAdvice(earlier.year, later.year, float("nan"), fewest, False,
                          ["one of the years has no usable observations"])
    if gap > MAX_COMPARABLE_NDVI_GAP:
        comparable = False
        direction = "greener" if later.ndvi_median > earlier.ndvi_median else "browner"
        notes.append(
            f"the landscape is {gap:.2f} NDVI {direction} in {later.year}; that is a "
            "climate difference, and it will read as change everywhere"
        )
    if fewest < MIN_CLEAR_SCENES:
        comparable = False
        notes.append(f"only {fewest} clear views in the thinner year")
    if earlier.woody_fraction < 0.05:
        comparable = False
        notes.append(
            f"only {earlier.woody_fraction:.1%} of the baseline year is woody - there is "
            "almost nothing here to lose"
        )
    if not notes:
        notes.append("comparable")
    return PairAdvice(earlier.year, later.year, gap, fewest, comparable, notes)


def survey(
    series: OpticalSeries,
    aoi_name: str,
    grid: Grid,
    years: Sequence[int],
    config: Optional[DetectionConfig] = None,
) -> SurveyResult:
    """Characterise an AOI year by year and advise on usable epoch pairs."""
    config = config or DetectionConfig()
    stats = [describe_year(series, year, config) for year in years]
    usable = [s for s in stats if s.clear_scenes > 0]
    pairs = [compare_years(a, b) for a, b in zip(stats, stats[1:])]

    warnings: List[str] = []
    if usable:
        typical_woody = float(np.median([s.woody_fraction for s in usable]))
        typical_crop = float(np.median([s.crop_fraction for s in usable]))
        if typical_woody < 0.05:
            warnings.append(
                f"Only {typical_woody:.1%} of this AOI is woody. There is little here to "
                "monitor; move the window onto a woodland-cropping margin."
            )
        if typical_crop > 0.4:
            warnings.append(
                f"{typical_crop:.1%} of this AOI swings hard through the year - it is "
                "dominated by cropping. Keep the seasonal-amplitude gate on; without it "
                "a rotation reads as clearing."
            )
        spread = max(s.ndvi_median for s in usable) - min(s.ndvi_median for s in usable)
        if spread > 0.25:
            warnings.append(
                f"Median NDVI moves {spread:.2f} across the record. Absolute thresholds "
                "will not transfer between years here; use the adaptive options."
            )
        thin = [s.year for s in stats if s.clear_scenes < MIN_CLEAR_SCENES]
        if thin:
            warnings.append(
                f"Thin optical coverage in {', '.join(str(y) for y in thin)} - do not use "
                "as a baseline."
            )

    suggested: Dict[str, float] = {}
    if usable:
        suggested = {
            "min_pre_ndvi_persistent": round(
                float(np.median([s.persistent_ndvi_p85 for s in usable])), 3
            ),
            "max_pre_seasonal_amplitude": 0.35,
            "adaptive_change_mad": 3.0,
            "adaptive_vh_mad": 5.0,
        }

    return SurveyResult(
        aoi_name=aoi_name,
        area_ha=grid.width * grid.height * grid.pixel_area_ha,
        years=stats,
        pairs=pairs,
        warnings=warnings,
        suggested_thresholds=suggested,
    )


def format_survey(result: SurveyResult) -> str:
    """The survey as a table a person can read in a terminal."""
    lines = [
        f"{result.aoi_name}  -  {result.area_ha:,.0f} ha",
        "",
        f"{'year':<6}{'clear':>6}{'NDVI p50':>10}{'NDVI p90':>10}{'NBR p50':>9}"
        f"{'persist':>9}{'swing':>8}{'woody':>8}{'crop':>7}{'cloud':>7}",
    ]
    for stat in result.years:
        if stat.clear_scenes == 0:
            lines.append(f"{stat.year:<6}{'-':>6}   no usable observations")
            continue
        lines.append(
            f"{stat.year:<6}{stat.clear_scenes:>6}{stat.ndvi_median:>10.3f}"
            f"{stat.ndvi_p90:>10.3f}{stat.nbr_median:>9.3f}{stat.persistent_ndvi_p85:>9.3f}"
            f"{stat.amplitude_median:>8.3f}{stat.woody_fraction:>8.1%}"
            f"{stat.crop_fraction:>7.1%}{stat.mean_cloud:>7.1%}"
        )

    lines += ["", "epoch pairs:"]
    for pair in result.pairs:
        mark = "ok  " if pair.comparable else "AVOID"
        lines.append(
            f"  {mark} {pair.baseline}->{pair.detection}  "
            f"NDVI gap {pair.ndvi_gap:.3f}  -  {'; '.join(pair.notes)}"
        )

    best = result.best_pair()
    lines += ["", f"suggested epoch: {best.baseline} -> {best.detection}" if best else
              "suggested epoch: none of these year pairs is fair; widen the record"]
    if result.suggested_thresholds:
        lines.append("suggested starting thresholds (calibrate before trusting):")
        for key, value in result.suggested_thresholds.items():
            lines.append(f"  {key:<30} {value}")
    if result.warnings:
        lines += ["", "warnings:"]
        lines += [f"  ! {w}" for w in result.warnings]
    return "\n".join(lines)


__all__ = [
    "MAX_COMPARABLE_NDVI_GAP",
    "MIN_CLEAR_SCENES",
    "PairAdvice",
    "SurveyResult",
    "YearStats",
    "compare_years",
    "describe_year",
    "format_survey",
    "survey",
]
