"""Post-clearing recovery trajectories, fitted per patch.

Three independent recovery tracks are followed, because they disagree in ways
that matter:

``ndvi``  greenness. Recovers fastest and overstates woody recovery badly -
          grass and forbs colonise a cleared paddock within one season and
          push NDVI back towards its pre-event value while there is still no
          woody vegetation on the site at all.
``nbr``   greenness plus canopy moisture and structure. Slower, and the better
          single proxy for whether woody cover is actually returning, so it is
          what the classification is based on.
``vh``    Sentinel-1 cross-polarised backscatter. Tracks woody structure, and
          keeps climbing in these data after the optical tracks have levelled
          off - but C-band saturates at modest biomass, so it responds steeply
          to the first regrowth and weakly to the difference between ten-year
          and twenty-year regrowth. Sensitive early, blunt late.

Reporting all three, rather than picking one, is the point: a patch whose NDVI
has recovered but whose NBR and VH have not is grassland, not regrowth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

from vegmon.config import Periods, RegrowthConfig
from vegmon.fuse import ClearingMap
from vegmon.series import OpticalSeries, RadarSeries

DAYS_PER_YEAR = 365.25

CLASS_RECOVERED = "recovered"
CLASS_RECOVERING = "recovering"
CLASS_STALLED = "stalled"
CLASS_RECLEARED = "recleared"
CLASS_UNKNOWN = "insufficient_data"

CLASS_DESCRIPTIONS = {
    CLASS_RECOVERED: "Back to at least 80% of the pre-clearing baseline (R80P).",
    CLASS_RECOVERING: "Trending up but not yet at 80% of baseline.",
    CLASS_STALLED: "Well below baseline with no meaningful upward trend - "
    "the site is being held open, by grazing, repeat cultivation or drought.",
    CLASS_RECLEARED: "Recovered part-way, then dropped again: a second "
    "clearing or disturbance event on the same ground.",
    CLASS_UNKNOWN: "Too few clear observations after the event to fit a trajectory.",
}


@dataclass
class SeasonalSeries:
    """A patch's index values binned into seasons."""

    dates: List[date]
    values: np.ndarray
    counts: np.ndarray

    def before(self, when: date) -> np.ndarray:
        return self.values[np.array([d < when for d in self.dates], dtype=bool)]

    def since(self, when: date) -> Tuple[np.ndarray, np.ndarray]:
        """(years since ``when``, values) for finite seasons at or after ``when``."""
        _, years, values = self.since_with_dates(when)
        return years, values

    def since_with_dates(self, when: date) -> Tuple[List[date], np.ndarray, np.ndarray]:
        """Dates, years-since and values for finite seasons at or after ``when``.

        Dates stay aligned with the values after non-finite seasons are
        dropped, which is what lets a re-clearing be dated rather than just
        counted.
        """
        selected = [
            (d, (d - when).days / DAYS_PER_YEAR, v)
            for d, v in zip(self.dates, self.values)
            if d >= when and np.isfinite(v)
        ]
        if not selected:
            return [], np.zeros(0, np.float32), np.zeros(0, np.float32)
        dates = [d for d, _, _ in selected]
        years = np.array([y for _, y, _ in selected], dtype=np.float32)
        values = np.array([v for _, _, v in selected], dtype=np.float32)
        return dates, years, values


@dataclass
class RecoveryFit:
    """A saturating-exponential fit to one recovery track."""

    index: str
    baseline: Optional[float] = None
    trough: Optional[float] = None
    current: Optional[float] = None
    asymptote: Optional[float] = None
    tau_years: Optional[float] = None
    half_life_years: Optional[float] = None
    recovery_fraction: Optional[float] = None
    r80p: Optional[float] = None
    years_to_80pct: Optional[float] = None
    recent_slope_per_year: Optional[float] = None
    rmse: Optional[float] = None
    n_seasons: int = 0
    converged: bool = False
    at_bound: bool = False
    """True if the optimiser pinned tau against its bound - the fit is a
    floor or ceiling, not an estimate, and any extrapolation from it is
    meaningless."""

    recovery_at_1yr: Optional[float] = None
    recovery_at_2yr: Optional[float] = None
    recovery_at_5yr: Optional[float] = None
    """Observed recovery fraction at fixed ages, interpolated from the
    seasonal series rather than the fit. Comparing these across the three
    tracks is what exposes grass masquerading as regrowth: NDVI runs ahead of
    NBR in the first two years by margins that never close if the site is not
    actually regrowing woody cover."""


@dataclass
class PatchRecovery:
    """Everything known about one patch's post-clearing trajectory."""

    patch_id: int
    event_date: Optional[date]
    area_ha: float
    tier: str
    classification: str = CLASS_UNKNOWN
    reclear_date: Optional[date] = None
    fits: Dict[str, RecoveryFit] = field(default_factory=dict)
    series: Dict[str, SeasonalSeries] = field(default_factory=dict, repr=False)

    @property
    def primary(self) -> Optional[RecoveryFit]:
        """The NBR fit, which the classification is based on."""
        return self.fits.get("nbr")

    def as_record(self) -> dict:
        """Flat dict for tables and reports."""
        row: dict = {
            "patch_id": self.patch_id,
            "tier": self.tier,
            "area_ha": round(self.area_ha, 3),
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "classification": self.classification,
            "reclear_date": self.reclear_date.isoformat() if self.reclear_date else None,
        }
        for name, fit in self.fits.items():
            for key, value in asdict(fit).items():
                if key == "index":
                    continue
                row[f"{name}_{key}"] = (
                    round(value, 4) if isinstance(value, float) and np.isfinite(value) else value
                )
        return row


@dataclass
class RegrowthResult:
    patches: List[PatchRecovery]
    config: RegrowthConfig

    def records(self) -> List[dict]:
        return [p.as_record() for p in self.patches]

    def summary(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for name in (
            CLASS_RECOVERED,
            CLASS_RECOVERING,
            CLASS_STALLED,
            CLASS_RECLEARED,
            CLASS_UNKNOWN,
        ):
            selected = [p for p in self.patches if p.classification == name]
            if not selected:
                continue
            out[name] = {
                "patches": len(selected),
                "area_ha": round(float(sum(p.area_ha for p in selected)), 2),
                "description": CLASS_DESCRIPTIONS[name],
            }
        return out

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.records())


# ---------------------------------------------------------------------------
# series extraction
# ---------------------------------------------------------------------------


def patch_means(
    stack: np.ndarray,
    labels: np.ndarray,
    patch_ids: Sequence[int],
) -> Dict[int, np.ndarray]:
    """Mean of each time slice over each patch, ignoring NaN.

    Indexing the stack once per patch and reducing the resulting ``(time,
    pixels)`` array is far cheaper than masking the full raster per time step.
    """
    out: Dict[int, np.ndarray] = {}
    for patch_id in patch_ids:
        mask = labels == patch_id
        if not mask.any():
            out[patch_id] = np.full(stack.shape[0], np.nan, dtype=np.float32)
            continue
        sample = stack[:, mask]
        with np.errstate(invalid="ignore"):
            all_nan = np.all(~np.isfinite(sample), axis=1)
            safe = np.where(np.isfinite(sample), sample, np.nan)
            safe = np.where(all_nan[:, None], 0.0, safe)
            means = np.nanmean(safe, axis=1).astype(np.float32)
        out[patch_id] = np.where(all_nan, np.nan, means)
    return out


def seasonalise(
    times: np.ndarray,
    values: np.ndarray,
    season_months: int = 3,
    min_observations: int = 1,
) -> SeasonalSeries:
    """Bin a time series into fixed-length seasons and take the median of each.

    Quarterly medians damp the phenological cycle - which would otherwise
    dominate any trend fit - without smearing out the event itself. The median
    also absorbs the occasional cloud-contaminated observation that survived
    masking.
    """
    dates = [np.datetime64(t, "D").astype("datetime64[D]").astype(date) for t in times]
    if not dates:
        return SeasonalSeries([], np.zeros(0, np.float32), np.zeros(0, np.int16))
    origin = min(dates)
    bins: Dict[int, List[float]] = {}
    for when, value in zip(dates, values):
        if not np.isfinite(value):
            continue
        months = (when.year - origin.year) * 12 + (when.month - origin.month)
        bins.setdefault(months // season_months, []).append(float(value))

    out_dates: List[date] = []
    out_values: List[float] = []
    out_counts: List[int] = []
    for index in sorted(bins):
        sample = bins[index]
        if len(sample) < min_observations:
            continue
        month_offset = index * season_months + season_months // 2
        year = origin.year + (origin.month - 1 + month_offset) // 12
        month = (origin.month - 1 + month_offset) % 12 + 1
        out_dates.append(date(year, month, 15))
        out_values.append(float(np.median(sample)))
        out_counts.append(len(sample))
    return SeasonalSeries(
        dates=out_dates,
        values=np.array(out_values, dtype=np.float32),
        counts=np.array(out_counts, dtype=np.int16),
    )


# ---------------------------------------------------------------------------
# curve fitting
# ---------------------------------------------------------------------------


def recovery_curve(years: np.ndarray, asymptote: float, trough: float, tau: float) -> np.ndarray:
    """Saturating exponential: ``A + (R - A) * exp(-t / tau)``.

    Chosen over a straight line because recovery is not linear - it is fast
    while there is light and space, and slows as the canopy closes - and over
    a logistic because post-clearing regrowth from stumps and soil seed banks
    has no lag phase to model.
    """
    return asymptote + (trough - asymptote) * np.exp(-years / max(tau, 1e-3))


def fit_recovery(
    series: SeasonalSeries,
    event_date: date,
    config: RegrowthConfig,
    index_name: str = "nbr",
) -> RecoveryFit:
    """Fit one recovery track and derive its metrics."""
    fit = RecoveryFit(index=index_name)

    pre_values = series.before(event_date)
    if pre_values.size:
        fit.baseline = float(np.median(pre_values))
    years, post_values = series.since(event_date)
    finite = np.isfinite(post_values)
    years, post_values = years[finite], post_values[finite]
    fit.n_seasons = int(post_values.size)
    if fit.baseline is None or post_values.size < config.min_seasons_post:
        return fit

    # The trough is the low point of the first year, not the global minimum:
    # a later minimum usually means a second disturbance, which is handled
    # separately rather than being folded into the recovery baseline.
    first_year = post_values[years <= 1.0]
    fit.trough = float(np.min(first_year)) if first_year.size else float(np.min(post_values))
    fit.current = float(np.median(post_values[-2:]))

    span = fit.baseline - fit.trough
    if span > 1e-6:
        fit.recovery_fraction = float((fit.current - fit.trough) / span)
    # R80P is a *ratio* to the pre-event value, so it only means anything for
    # an index with a natural zero. Sentinel-1 VH is in decibels, where the
    # baseline is negative and "80% of baseline" is a larger number than the
    # baseline itself - so it is left undefined there rather than reported as
    # a number that reads plausible and is nonsense.
    fit.r80p = (
        float(fit.current / (config.recovered_fraction * fit.baseline))
        if fit.baseline > 1e-6
        else None
    )

    for age, attr in ((1.0, "recovery_at_1yr"), (2.0, "recovery_at_2yr"), (5.0, "recovery_at_5yr")):
        if span > 1e-6 and years.size and years.max() >= age:
            observed = float(np.interp(age, years, post_values))
            setattr(fit, attr, round(float((observed - fit.trough) / span), 4))

    recent = years >= max(0.0, float(years.max()) - config.recent_years)
    if recent.sum() >= 3:
        # Theil-Sen rather than least squares: one bad season should not set
        # the trend that decides whether a site is called stalled.
        slope, _, _, _ = stats.theilslopes(post_values[recent], years[recent])
        fit.recent_slope_per_year = float(slope)

    # Bounds are derived from the observed span, not from assumed index units.
    # Hard-coding them for a 0-1 index silently pins the fit for any track on
    # another scale - decibels above all, where the values are negative and a
    # "+0.6" ceiling sits below the answer.
    guess = [max(fit.current, fit.trough + 1e-3 * span), fit.trough, 2.5]
    lower = [fit.trough - 0.2 * span, fit.trough - 0.5 * span, 0.15]
    upper = [fit.baseline + 0.6 * span, fit.trough + 0.5 * span, 25.0]
    guess = [float(np.clip(g, lo, hi)) for g, lo, hi in zip(guess, lower, upper)]
    try:
        result = optimize.least_squares(
            lambda p: recovery_curve(years, *p) - post_values,
            x0=guess,
            bounds=(lower, upper),
            max_nfev=2000,
        )
        asymptote, trough_fit, tau = (float(v) for v in result.x)
        fit.asymptote = asymptote
        fit.tau_years = tau
        fit.half_life_years = float(tau * np.log(2.0))
        residual = recovery_curve(years, asymptote, trough_fit, tau) - post_values
        fit.rmse = float(np.sqrt(np.mean(residual**2)))
        fit.converged = bool(result.success)
        fit.at_bound = bool(tau <= lower[2] * 1.01 or tau >= upper[2] * 0.99)

        # The recovery target is expressed on the span from trough to
        # baseline, matching the recovery fraction the classification uses.
        # A multiplicative target would disagree with it, and is undefined for
        # a negative-valued track anyway.
        target = fit.trough + config.recovered_fraction * span
        if fit.at_bound:
            # Extrapolating a time-to-recovery from a fit that hit its bound
            # would put a number on a curve the data does not support.
            fit.years_to_80pct = None
        elif asymptote > target and abs(trough_fit - asymptote) > 1e-6:
            ratio = (target - asymptote) / (trough_fit - asymptote)
            if ratio > 0:
                fit.years_to_80pct = float(-tau * np.log(ratio))
        else:
            # The curve levels off below the target: on this trajectory the
            # site does not get back to 80% of baseline at all.
            fit.years_to_80pct = None
    except Exception:  # pragma: no cover - optimiser blow-ups are not fatal
        fit.converged = False
    return fit


def detect_reclearing(
    series: SeasonalSeries,
    event_date: date,
    config: RegrowthConfig,
) -> Optional[date]:
    """Find a second disturbance: a sustained drop below the recent level.

    Two details stop this from firing on noise. The reference level is a
    trailing *median* of the preceding seasons rather than a running maximum -
    the maximum of a noisy series drifts upwards, so every drought season
    afterwards reads as a drop. And the drop has to persist into the following
    season, because a real clearing event does not grow back within three
    months while a dry quarter does.

    The gate on how much the site had recovered first also matters: without
    it, a site that never recovered at all reads as re-cleared every time the
    season turns.
    """
    dates, years, values = series.since_with_dates(event_date)
    if values.size < 6:
        return None
    trough = float(np.min(values[years <= 1.0])) if np.any(years <= 1.0) else float(values[0])

    lookback = 4
    for i in range(lookback, values.size - 1):
        reference = float(np.median(values[max(0, i - lookback) : i]))
        if reference - trough < config.min_recovery_for_reclear:
            continue
        drop = reference - values[i]
        sustained = reference - values[i + 1]
        if drop >= config.reclear_drop and sustained >= config.reclear_drop * 0.7:
            return dates[i] if i < len(dates) else None
    return None


def classify(
    fit: Optional[RecoveryFit],
    reclear_date: Optional[date],
    config: RegrowthConfig,
) -> str:
    """Turn recovery metrics into a class a report can use."""
    if reclear_date is not None:
        return CLASS_RECLEARED
    if fit is None or fit.recovery_fraction is None:
        return CLASS_UNKNOWN
    if fit.recovery_fraction >= config.recovered_fraction:
        return CLASS_RECOVERED
    slope = fit.recent_slope_per_year
    if fit.recovery_fraction < config.stalled_fraction and (
        slope is None or slope < config.stalled_slope_per_year
    ):
        return CLASS_STALLED
    return CLASS_RECOVERING


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def analyse_regrowth(
    optical: OpticalSeries,
    radar: Optional[RadarSeries],
    clearing: ClearingMap,
    periods: Periods,
    config: Optional[RegrowthConfig] = None,
    event_dates: Optional[Dict[int, date]] = None,
    tiers: Sequence[str] = ("confirmed",),
) -> RegrowthResult:
    """Fit recovery trajectories for every patch in the requested tiers.

    ``event_dates`` maps patch id to a clearing date; where a patch is missing
    the midpoint of the pre/post gap is used, which is the best that can be
    said from a bitemporal pair alone.
    """
    config = config or RegrowthConfig()
    records = [r for r in clearing.records if not tiers or r["tier"] in tiers]
    patch_ids = [r["patch_id"] for r in records]
    if not patch_ids:
        return RegrowthResult(patches=[], config=config)

    gap_start, gap_end = periods.event_window
    default_event = gap_start + timedelta(days=(gap_end - gap_start).days // 2)

    window = optical.select(periods.series_start, periods.series_end)
    tracks: Dict[str, Dict[int, np.ndarray]] = {}
    for index_name in ("ndvi", "nbr"):
        stack = window.index_stack(index_name)
        tracks[index_name] = patch_means(stack, clearing.components, patch_ids)
        del stack

    radar_window = None
    if radar is not None:
        radar_window = radar.select(periods.series_start, periods.series_end)
        tracks["vh"] = patch_means(radar_window.vh_db, clearing.components, patch_ids)

    out: List[PatchRecovery] = []
    for record in records:
        patch_id = record["patch_id"]
        event = (event_dates or {}).get(patch_id) or default_event
        recovery = PatchRecovery(
            patch_id=patch_id,
            event_date=event,
            area_ha=record["area_ha"],
            tier=record["tier"],
        )
        for index_name, means in tracks.items():
            times = radar_window.times if index_name == "vh" else window.times
            series = seasonalise(times, means[patch_id], config.season_months)
            recovery.series[index_name] = series
            recovery.fits[index_name] = fit_recovery(series, event, config, index_name)

        nbr_series = recovery.series.get("nbr")
        recovery.reclear_date = (
            detect_reclearing(nbr_series, event, config) if nbr_series else None
        )
        recovery.classification = classify(recovery.primary, recovery.reclear_date, config)
        out.append(recovery)

    return RegrowthResult(patches=out, config=config)


__all__ = [
    "CLASS_DESCRIPTIONS",
    "CLASS_RECLEARED",
    "CLASS_RECOVERED",
    "CLASS_RECOVERING",
    "CLASS_STALLED",
    "CLASS_UNKNOWN",
    "PatchRecovery",
    "RecoveryFit",
    "RegrowthResult",
    "SeasonalSeries",
    "analyse_regrowth",
    "classify",
    "detect_reclearing",
    "fit_recovery",
    "patch_means",
    "recovery_curve",
    "seasonalise",
]
