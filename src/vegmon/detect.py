"""Bitemporal change detection for Sentinel-2 and Sentinel-1.

Both detectors follow the same shape: build a seasonal composite either side
of the suspected event, difference it, normalise the difference against the
stable population, then apply a set of gates that encode what land clearing
actually looks like. Each returns the gate results as well as the decision, so
a rejected pixel can always be explained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from vegmon.config import DetectionConfig, Periods
from vegmon.grid import Grid
from vegmon.series import OpticalComposite, OpticalSeries, RadarComposite, RadarSeries

DAYS_PER_YEAR = 365.25


def robust_offset(
    delta: np.ndarray,
    population: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    iterations: int = 2,
) -> float:
    """Median of ``delta`` over the stable population - the inter-date offset.

    Two composites of the same unchanged ground are never identical: sun
    angle, aerosol residuals and BRDF move optical indices by a few hundredths,
    and soil moisture moves Sentinel-1 by more than a decibel. Subtracting the
    median of the *stable* population removes that common-mode shift, which is
    the difference between a threshold that transfers between dates and one
    that has to be re-tuned for every pair. This is relative radiometric
    normalisation against pseudo-invariant features, done the cheap way.

    When ``threshold`` is given the estimate is refined: pixels that look
    changed at the current offset are dropped and the median retaken. That
    matters because the median only survives contamination below 50%, and an
    AOI drawn tightly around a clearing event can breach that. It cannot
    rescue a truly change-dominated AOI - for those, pass a ``population``
    mask of known-stable ground, or widen the AOI.
    """
    values = delta[population] if population is not None else delta
    values = values[np.isfinite(values)]
    if values.size < 32:
        values = delta[np.isfinite(delta)]
    if values.size == 0:
        return 0.0
    offset = float(np.median(values))
    if threshold is None:
        return offset
    for _ in range(max(0, iterations - 1)):
        keep = values < offset + threshold
        if keep.sum() < max(32, 0.05 * values.size):
            break
        refined = float(np.median(values[keep]))
        if abs(refined - offset) < 1e-4:
            offset = refined
            break
        offset = refined
    return offset


@dataclass
class OpticalChange:
    """Result of the Sentinel-2 bitemporal detector."""

    grid: Grid
    pre: OpticalComposite
    post: OpticalComposite
    dnbr: np.ndarray
    dndvi: np.ndarray
    pre_ndvi_persistent: np.ndarray
    mask: np.ndarray
    gates: Dict[str, np.ndarray] = field(default_factory=dict)
    offsets: Dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> np.ndarray:
        """How far past its threshold the weaker of the two indices got."""
        return self._score

    _score: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def rejection_summary(self) -> Dict[str, int]:
        """Pixel counts failing each gate, for diagnosing a quiet detector."""
        return {name: int((~gate).sum()) for name, gate in self.gates.items()}


@dataclass
class RadarChange:
    """Result of the Sentinel-1 backscatter-drop detector."""

    grid: Grid
    pre: RadarComposite
    post: RadarComposite
    vh_drop_db: np.ndarray
    vv_drop_db: np.ndarray
    rvi_drop: np.ndarray
    mask: np.ndarray
    gates: Dict[str, np.ndarray] = field(default_factory=dict)
    offsets: Dict[str, float] = field(default_factory=dict)
    orbit_drops: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def orbit_agreement(self) -> Optional[np.ndarray]:
        """True where every available orbit direction saw at least half the drop."""
        if len(self.orbit_drops) < 2:
            return None
        stacked = np.stack(list(self.orbit_drops.values()))
        return np.all(np.isfinite(stacked), axis=0) & np.all(stacked > 0, axis=0)

    def rejection_summary(self) -> Dict[str, int]:
        return {name: int((~gate).sum()) for name, gate in self.gates.items()}


def detect_optical(
    series: OpticalSeries,
    periods: Periods,
    config: Optional[DetectionConfig] = None,
    check_phenology: bool = True,
) -> OpticalChange:
    """Detect canopy loss from a pair of Sentinel-2 seasonal composites.

    The gates, in order:

    1. **enough observations** either side, or the composite is noise;
    2. **was woody** - median NDVI and NBR above a floor in the pre-period;
    3. **persistently woody** - the low temporal percentile of pre-period NDVI
       also above a floor, which is what separates woody vegetation from a
       crop that happens to be green when the baseline was taken;
    4. **lost enough** - both dNBR and dNDVI past their thresholds.

    Requiring NBR *and* NDVI to move together is deliberate. NBR alone fires on
    burn scars and on wet soil; NDVI alone fires on every harvested paddock.
    """
    config = config or DetectionConfig()
    if check_phenology:
        periods.check_phenology()

    pre = series.composite(
        periods.pre_start, periods.pre_end, max_cloud_cover=config.max_cloud_cover
    )
    post = series.composite(
        periods.post_start, periods.post_end, max_cloud_cover=config.max_cloud_cover
    )

    pre_ndvi_persistent = _temporal_percentile(
        series, periods.pre_start, periods.pre_end, config, "ndvi", config.pre_percentile
    )

    enough_pre = pre.n_obs >= config.min_clear_observations
    enough_post = post.n_obs >= config.min_clear_observations
    was_woody = (pre.ndvi >= config.min_pre_ndvi) & (pre.nbr >= config.min_pre_nbr)
    persistent = pre_ndvi_persistent >= config.min_pre_ndvi_persistent
    not_water = pre.ndwi < 0.0

    # Normalise against every valid land pixel, not just the woody ones.
    # Restricting the population to pixels that pass the woody gate sounds
    # tighter but is self-defeating: those are exactly the pixels under test,
    # so on an AOI drawn around a clearing event they are mostly the change
    # itself and the median walks into the signal. Change is a minority of any
    # sensibly-sized AOI, so the whole-scene median is the safer reference.
    stable_population = enough_pre & enough_post & not_water
    raw_dnbr = pre.nbr - post.nbr
    raw_dndvi = pre.ndvi - post.ndvi
    nbr_offset = robust_offset(raw_dnbr, stable_population, config.dnbr_threshold)
    ndvi_offset = robust_offset(raw_dndvi, stable_population, config.dndvi_threshold)
    dnbr = (raw_dnbr - nbr_offset).astype(np.float32)
    dndvi = (raw_dndvi - ndvi_offset).astype(np.float32)

    lost_nbr = dnbr >= config.dnbr_threshold
    lost_ndvi = dndvi >= config.dndvi_threshold

    gates = {
        "observations_pre": enough_pre,
        "observations_post": enough_post,
        "woody_before": was_woody,
        "persistently_woody": persistent,
        "not_water": not_water,
        "nbr_drop": lost_nbr,
        "ndvi_drop": lost_ndvi,
    }
    mask = np.ones(series.grid.shape, dtype=bool)
    for gate in gates.values():
        mask &= np.nan_to_num(gate, nan=False).astype(bool)

    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.minimum(
            dnbr / max(config.dnbr_threshold, 1e-6),
            dndvi / max(config.dndvi_threshold, 1e-6),
        )
    score = np.where(np.isfinite(score), score, 0.0).astype(np.float32)

    return OpticalChange(
        grid=series.grid,
        pre=pre,
        post=post,
        dnbr=dnbr,
        dndvi=dndvi,
        pre_ndvi_persistent=pre_ndvi_persistent,
        mask=mask,
        gates=gates,
        offsets={"dnbr": nbr_offset, "dndvi": ndvi_offset},
        _score=score,
    )


def _temporal_percentile(
    series: OpticalSeries,
    start: date,
    end: date,
    config: DetectionConfig,
    index_name: str,
    percentile: float,
) -> np.ndarray:
    """Per-pixel percentile of an index through time over a window."""
    window = series.select(start, end, max_cloud_cover=config.max_cloud_cover)
    if len(window) == 0:
        return np.full(series.grid.shape, np.nan, dtype=np.float32)
    stack = window.index_stack(index_name)
    all_nan = np.all(~np.isfinite(stack), axis=0)
    safe = np.where(np.isfinite(stack), stack, np.nan)
    safe = np.where(all_nan[None, ...], 0.0, safe)
    with np.errstate(invalid="ignore"):
        out = np.nanpercentile(safe, percentile, axis=0).astype(np.float32)
    return np.where(all_nan, np.nan, out)


def detect_radar(
    series: RadarSeries,
    periods: Periods,
    config: Optional[DetectionConfig] = None,
    stable_population: Optional[np.ndarray] = None,
) -> RadarChange:
    """Detect canopy loss from a pair of Sentinel-1 seasonal composites.

    Removing woody structure collapses the volume scattering that dominates
    cross-polarised return, so VH falls by several decibels and stays down.
    The advantage over optical is that it does not need a clear sky - which in
    a NSW summer, under cloud and bushfire smoke, is what decides whether an
    event is seen in February or in May.

    The traps this guards against:

    * **speckle**, handled by a temporal median over at least three passes and
      a spatial boxcar multi-look;
    * **soil moisture**, which lifts backscatter over thin cover after rain and
      is removed as a scene-wide offset;
    * **orbit geometry**, where ascending and descending passes differ by more
      than the signal being looked for, so per-orbit drops are reported
      separately and can be required to agree.
    """
    config = config or DetectionConfig()

    pre = series.composite(periods.pre_start, periods.pre_end, config.speckle_window)
    post = series.composite(periods.post_start, periods.post_end, config.speckle_window)

    raw_vh = pre.vh_db - post.vh_db
    raw_vv = pre.vv_db - post.vv_db
    if stable_population is None:
        stable_population = pre.vh_db >= config.min_pre_vh_db
    vh_offset = robust_offset(raw_vh, stable_population, config.vh_drop_db)
    vv_offset = robust_offset(raw_vv, stable_population, config.vh_drop_db)
    vh_drop = (raw_vh - vh_offset).astype(np.float32)
    vv_drop = (raw_vv - vv_offset).astype(np.float32)

    raw_rvi = pre.rvi - post.rvi
    rvi_drop = (raw_rvi - robust_offset(raw_rvi, stable_population)).astype(np.float32)

    orbit_drops: Dict[str, np.ndarray] = {}
    if series.orbit_state is not None:
        for state in sorted(set(s.lower() for s in series.orbit_state)):
            single = series.by_orbit(state)
            if len(single.select(periods.pre_start, periods.pre_end)) < 2:
                continue
            if len(single.select(periods.post_start, periods.post_end)) < 2:
                continue
            o_pre = single.composite(periods.pre_start, periods.pre_end, config.speckle_window)
            o_post = single.composite(periods.post_start, periods.post_end, config.speckle_window)
            drop = o_pre.vh_db - o_post.vh_db
            orbit_drops[state] = (
                drop - robust_offset(drop, stable_population, config.vh_drop_db)
            ).astype(np.float32)

    gates = {
        "observations_pre": pre.n_obs >= config.min_s1_observations,
        "observations_post": post.n_obs >= config.min_s1_observations,
        "structure_before": pre.vh_db >= config.min_pre_vh_db,
        "vh_drop": vh_drop >= config.vh_drop_db,
    }
    mask = np.ones(series.grid.shape, dtype=bool)
    for gate in gates.values():
        mask &= np.nan_to_num(gate, nan=False).astype(bool)

    return RadarChange(
        grid=series.grid,
        pre=pre,
        post=post,
        vh_drop_db=vh_drop,
        vv_drop_db=vv_drop,
        rvi_drop=rvi_drop,
        mask=mask,
        gates=gates,
        offsets={"vh_db": vh_offset, "vv_db": vv_offset},
        orbit_drops=orbit_drops,
    )


# ---------------------------------------------------------------------------
# event timing
# ---------------------------------------------------------------------------


@dataclass
class EventTiming:
    """When a patch was cleared, and when each sensor could first have said so."""

    patch_id: int
    event_date: Optional[date]
    s2_detection_date: Optional[date]
    s1_detection_date: Optional[date]
    first_detection_date: Optional[date]
    s2_latency_days: Optional[int]
    s1_latency_days: Optional[int]

    def as_dict(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "s2_detection_date": (
                self.s2_detection_date.isoformat() if self.s2_detection_date else None
            ),
            "s1_detection_date": (
                self.s1_detection_date.isoformat() if self.s1_detection_date else None
            ),
            "first_detection_date": (
                self.first_detection_date.isoformat() if self.first_detection_date else None
            ),
            "s2_latency_days": self.s2_latency_days,
            "s1_latency_days": self.s1_latency_days,
        }


def _to_date(value) -> date:
    return np.datetime64(value, "D").astype("datetime64[D]").astype(date)


def estimate_event_timing(
    optical: OpticalSeries,
    radar: RadarSeries,
    labels: np.ndarray,
    patch_ids: List[int],
    periods: Periods,
    config: Optional[DetectionConfig] = None,
    search_buffer_days: int = 60,
    drop_fraction: float = 0.5,
    min_clear_fraction: float = 0.6,
) -> List[EventTiming]:
    """Date each patch's clearing, and how soon each sensor could see it.

    For every patch the mean NBR (Sentinel-2) and mean VH (Sentinel-1) are
    tracked through the event window. The detection date is the first
    acquisition on which the patch mean has fallen past halfway between its
    pre-event baseline and its post-event level and stays there.

    ``min_clear_fraction`` is what keeps the optical answer honest. An
    acquisition where 95% of the patch is under cloud still yields a patch
    mean from the handful of clear pixels, and scoring that as a detection
    would credit Sentinel-2 with alerts no operator would act on. Requiring
    most of the patch to be visible is the realistic bar, and it is the reason
    the optical latency here is measured in weeks rather than days.

    The gap between the two answers the question that decides whether a
    monitoring program is useful: how long after the dozer does an alert
    arrive? Sentinel-1 usually wins that race outright in summer, because it
    does not wait for a clear sky.
    """
    config = config or DetectionConfig()
    start = periods.pre_end - timedelta(days=search_buffer_days)
    end = periods.post_start + timedelta(days=search_buffer_days)

    optical_window = optical.select(start, end)
    nbr_stack = optical_window.index_stack("nbr") if len(optical_window) else None
    radar_window = radar.select(start, end)

    pre_optical = optical.composite(
        periods.pre_start, periods.pre_end, max_cloud_cover=config.max_cloud_cover
    )
    post_optical = optical.composite(
        periods.post_start, periods.post_end, max_cloud_cover=config.max_cloud_cover
    )
    pre_radar = radar.composite(periods.pre_start, periods.pre_end, config.speckle_window)
    post_radar = radar.composite(periods.post_start, periods.post_end, config.speckle_window)

    results: List[EventTiming] = []
    for patch_id in patch_ids:
        mask = labels == patch_id
        if not mask.any():
            continue

        s2_date = None
        if nbr_stack is not None and len(optical_window):
            baseline = float(np.nanmean(pre_optical.nbr[mask]))
            floor = float(np.nanmean(post_optical.nbr[mask]))
            s2_date = _first_sustained_crossing(
                optical_window.times,
                nbr_stack,
                mask,
                baseline,
                floor,
                drop_fraction,
                min_clear_fraction=min_clear_fraction,
            )

        s1_date = None
        if len(radar_window):
            baseline = float(np.nanmean(pre_radar.vh_db[mask]))
            floor = float(np.nanmean(post_radar.vh_db[mask]))
            s1_date = _first_sustained_crossing(
                radar_window.times, radar_window.vh_db, mask, baseline, floor, drop_fraction
            )

        candidates = [d for d in (s2_date, s1_date) if d is not None]
        first = min(candidates) if candidates else None
        results.append(
            EventTiming(
                patch_id=int(patch_id),
                event_date=None,
                s2_detection_date=s2_date,
                s1_detection_date=s1_date,
                first_detection_date=first,
                s2_latency_days=None,
                s1_latency_days=None,
            )
        )
    return results


def _first_sustained_crossing(
    times: np.ndarray,
    stack: np.ndarray,
    mask: np.ndarray,
    baseline: float,
    floor: float,
    drop_fraction: float,
    sustain: int = 2,
    min_clear_fraction: float = 0.0,
) -> Optional[date]:
    """First acquisition whose patch mean crosses the halfway level and stays.

    Requiring the crossing to be sustained across the next observations is what
    keeps a single cloud-contaminated scene, or one wet Sentinel-1 pass, from
    dating the event weeks early. Acquisitions where less than
    ``min_clear_fraction`` of the patch is visible are skipped entirely.
    """
    if not np.isfinite(baseline) or not np.isfinite(floor) or baseline <= floor:
        return None
    level = floor + (baseline - floor) * (1.0 - drop_fraction)

    sample = stack[:, mask]
    usable = np.isfinite(sample)
    clear_fraction = usable.mean(axis=1)
    enough = clear_fraction >= min_clear_fraction
    totals = np.where(usable, sample, 0.0).sum(axis=1)
    counts = usable.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan).astype(np.float32)

    below = means < level
    finite = np.isfinite(means) & enough
    for i in range(len(means)):
        if not finite[i] or not below[i]:
            continue
        following = [j for j in range(i + 1, len(means)) if finite[j]][:sustain]
        if len(following) < min(sustain, max(0, int(finite[i + 1 :].sum()))):
            continue
        if all(below[j] for j in following):
            return _to_date(times[i])
    return None


def attach_true_event_dates(
    timings: List[EventTiming], truth_patches: List[dict]
) -> List[EventTiming]:
    """Fill in the true event dates and latencies from synthetic ground truth."""
    lookup = {p["patch_id"]: p.get("event_date") for p in truth_patches}
    out: List[EventTiming] = []
    for timing in timings:
        raw = lookup.get(timing.patch_id)
        event = date.fromisoformat(raw) if raw else None
        s2_latency = (
            (timing.s2_detection_date - event).days
            if event and timing.s2_detection_date
            else None
        )
        s1_latency = (
            (timing.s1_detection_date - event).days
            if event and timing.s1_detection_date
            else None
        )
        out.append(
            EventTiming(
                patch_id=timing.patch_id,
                event_date=event,
                s2_detection_date=timing.s2_detection_date,
                s1_detection_date=timing.s1_detection_date,
                first_detection_date=timing.first_detection_date,
                s2_latency_days=s2_latency,
                s1_latency_days=s1_latency,
            )
        )
    return out


__all__ = [
    "EventTiming",
    "OpticalChange",
    "RadarChange",
    "attach_true_event_dates",
    "detect_optical",
    "detect_radar",
    "estimate_event_timing",
    "robust_offset",
]
