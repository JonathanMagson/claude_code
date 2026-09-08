"""Temporal persistence as confirming evidence when radar is unavailable.

The pipeline's normal second opinion is Sentinel-1: canopy loss that is real
takes the woody structure with it, and a burn scar or a bare paddock does not.
Where no analysis-ready Sentinel-1 is reachable - which is the situation over
Australia for several of the free catalogues - the next best independent check
available from the optical record alone is **persistence**.

A clearing event does not grow back. A harvested paddock does, next season. A
drought dip recovers when it rains. An unmasked cloud shadow is gone by the
next acquisition. Requiring the spectral drop to still be there several
seasons later rejects all three, using observations that played no part in the
original detection.

What it does **not** reject is fire. A burn scar persists for years, and it
looks like clearing in every optical index. That is the specific failure this
substitutes away, and it is why persistence is a fallback rather than a
replacement: if Sentinel-1 is reachable, use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from vegmon.config import DetectionConfig, Periods
from vegmon.detect import OpticalChange, robust_offset
from vegmon.grid import Grid
from vegmon.series import OpticalSeries


@dataclass
class PersistenceEvidence:
    """Confirming evidence built from post-event seasonal composites.

    Deliberately shaped like :class:`~vegmon.detect.RadarChange` so that
    :func:`vegmon.fuse.fuse` can consume either without caring which it got.
    """

    grid: Grid
    mask: np.ndarray
    seasons_checked: int
    seasons_below: np.ndarray
    """Per-pixel count of post-event seasons that stayed below the level."""

    seasons_observed: np.ndarray
    persistent_drop: np.ndarray
    """Per-pixel dNBR against the *last* post-event season, not the first."""

    windows: List[Tuple[date, date]] = field(default_factory=list)
    gates: Dict[str, np.ndarray] = field(default_factory=dict)
    offsets: Dict[str, float] = field(default_factory=dict)

    #: Persistence is corroboration, never a detector in its own right: it asks
    #: whether a drop that was already detected is still there. A pixel that
    #: never showed a drop cannot "persist", so this layer contributes no
    #: detections of its own and the single-evidence tier for it is empty by
    #: construction.
    is_independent: bool = False

    #: What the confirmed tier means when this is the confirming source.
    source_name: str = "temporal persistence"
    source_description: str = (
        "the spectral loss was still present in later seasons, so it is not "
        "harvest, a drought dip or an unmasked shadow. Fire is NOT excluded by "
        "this check - only Sentinel-1 does that."
    )

    def rejection_summary(self) -> Dict[str, int]:
        return {name: int((~gate).sum()) for name, gate in self.gates.items()}

    def as_stats(self) -> Dict[str, np.ndarray]:
        """Per-pixel layers to summarise onto each patch."""
        return {
            "persistent_dnbr": self.persistent_drop,
            "seasons_below": self.seasons_below.astype(np.float32),
            "seasons_observed": self.seasons_observed.astype(np.float32),
        }


def seasonal_windows(
    start: date, count: int, months: int = 3, skip_months: int = 3
) -> List[Tuple[date, date]]:
    """``count`` consecutive windows of ``months``, beginning after a gap.

    The gap matters. The season immediately after clearing is when a burnt or
    windrowed site is at its darkest and a harvested paddock has not yet been
    re-sown, so both look identical. Starting a season later is what lets the
    two separate.
    """
    windows: List[Tuple[date, date]] = []
    cursor = _add_months(start, skip_months)
    for _ in range(count):
        end = _add_months(cursor, months) - timedelta(days=1)
        windows.append((cursor, end))
        cursor = _add_months(cursor, months)
    return windows


def _add_months(when: date, months: int) -> date:
    total = when.year * 12 + (when.month - 1) + months
    return date(total // 12, total % 12 + 1, 1)


def detect_persistence(
    series: OpticalSeries,
    optical: OpticalChange,
    periods: Periods,
    config: Optional[DetectionConfig] = None,
    seasons: int = 4,
    season_months: int = 3,
    skip_months: int = 3,
    min_seasons_below: Optional[int] = None,
    min_seasons_observed: int = 2,
    tolerance: float = 0.6,
) -> PersistenceEvidence:
    """Check whether each pixel's spectral drop was still there later on.

    Parameters
    ----------
    seasons, season_months, skip_months:
        How many post-event windows to test, how long each is, and how long to
        wait before the first one.
    min_seasons_below:
        How many of the observed seasons must still be down. Defaults to all
        but one, so a single cloudy or anomalous season does not veto a real
        detection.
    tolerance:
        Fraction of the detection threshold a later season must still be below.
        Below 1.0 because regrowth begins immediately: demanding the *full*
        original drop years later would reject every site that started to
        recover, which is most of them.
    """
    config = config or DetectionConfig()
    if min_seasons_below is None:
        min_seasons_below = max(1, seasons - 1)

    threshold = optical.thresholds.get("dnbr_threshold", config.dnbr_threshold)
    level = threshold * tolerance
    pre_nbr = optical.pre.nbr

    windows = seasonal_windows(
        periods.post_start, seasons, months=season_months, skip_months=skip_months
    )
    below = np.zeros(series.grid.shape, np.int16)
    observed = np.zeros(series.grid.shape, np.int16)
    last_drop = np.full(series.grid.shape, np.nan, np.float32)

    for start, end in windows:
        if start > periods.series_end:
            break
        composite = series.composite(start, end, max_cloud_cover=config.max_cloud_cover)
        valid = composite.n_obs >= 1
        drop = pre_nbr - composite.nbr
        # Each season is normalised against its own stable population, exactly
        # as the detection pair is - otherwise a wet season reads as recovery
        # everywhere and a dry one as clearing everywhere.
        drop = drop - robust_offset(drop, valid & np.isfinite(drop), threshold)
        observed += valid.astype(np.int16)
        below += (valid & (drop >= level)).astype(np.int16)
        last_drop = np.where(valid & np.isfinite(drop), drop, last_drop)

    checked = len(windows)
    required = np.minimum(min_seasons_below, np.maximum(observed - 0, 1))
    gates = {
        "seasons_observed": observed >= min_seasons_observed,
        "still_down": below >= required,
    }
    mask = np.ones(series.grid.shape, bool)
    for gate in gates.values():
        mask &= gate

    return PersistenceEvidence(
        grid=series.grid,
        mask=mask,
        seasons_checked=checked,
        seasons_below=below,
        seasons_observed=observed,
        persistent_drop=last_drop.astype(np.float32),
        windows=windows,
        gates=gates,
        offsets={"level": float(level)},
    )


__all__ = ["PersistenceEvidence", "detect_persistence", "seasonal_windows"]
