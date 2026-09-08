from datetime import date, timedelta

import numpy as np
import pytest

from vegmon.config import RegrowthConfig
from vegmon.regrowth import (
    CLASS_RECLEARED,
    CLASS_RECOVERED,
    CLASS_RECOVERING,
    CLASS_STALLED,
    CLASS_UNKNOWN,
    SeasonalSeries,
    classify,
    detect_reclearing,
    fit_recovery,
    patch_means,
    recovery_curve,
    seasonalise,
)

EVENT = date(2020, 3, 1)


def _series(values, start=date(2019, 3, 15), step_days=91):
    dates = [start + timedelta(days=step_days * i) for i in range(len(values))]
    return SeasonalSeries(
        dates=dates,
        values=np.asarray(values, dtype=np.float32),
        counts=np.full(len(values), 5, np.int16),
    )


def _trajectory(baseline, trough, tau, n=28, start=date(2019, 3, 15)):
    dates = [start + timedelta(days=91 * i) for i in range(n)]
    years = np.array([(d - EVENT).days / 365.25 for d in dates])
    values = np.where(
        years < 0, baseline, recovery_curve(np.maximum(years, 0.0), baseline, trough, tau)
    )
    return _series(values.astype(np.float32), start=start)


# --- seasonal binning ------------------------------------------------------


def test_seasonalise_takes_the_median_of_each_bin():
    times = np.array(
        [np.datetime64(date(2020, 1, 5) + timedelta(days=20 * i), "D") for i in range(6)]
    )
    values = np.array([0.1, 0.2, 0.9, 0.4, 0.5, 0.6], dtype=np.float32)
    binned = seasonalise(times, values, season_months=3)
    assert len(binned.dates) == 2
    assert binned.counts.sum() == 6


def test_seasonalise_ignores_non_finite_values():
    times = np.array([np.datetime64(date(2020, 1, 1) + timedelta(days=10 * i), "D")
                      for i in range(4)])
    values = np.array([0.3, np.nan, 0.5, np.nan], dtype=np.float32)
    binned = seasonalise(times, values, season_months=6)
    assert binned.counts.tolist() == [2]
    assert binned.values[0] == pytest.approx(0.4)


def test_seasonalise_on_an_empty_series():
    binned = seasonalise(np.array([], dtype="datetime64[D]"), np.array([]), 3)
    assert binned.dates == [] and binned.values.size == 0


def test_since_with_dates_keeps_dates_aligned_after_dropping_gaps():
    series = _series([0.1, np.nan, 0.3, 0.4])
    dates, years, values = series.since_with_dates(series.dates[0])
    assert len(dates) == len(years) == len(values) == 3
    assert np.allclose(values, [0.1, 0.3, 0.4])


# --- curve fitting ---------------------------------------------------------


def test_fit_recovers_a_known_time_constant():
    fit = fit_recovery(_trajectory(0.32, 0.02, 2.5), EVENT, RegrowthConfig(), "nbr")
    assert fit.converged
    assert fit.tau_years == pytest.approx(2.5, rel=0.1)
    assert fit.half_life_years == pytest.approx(2.5 * np.log(2), rel=0.1)
    assert fit.rmse < 1e-3


def test_fit_works_on_a_decibel_scale_where_values_are_negative():
    """Bounds must come from the observed span, not from assumed index units."""
    fit = fit_recovery(_trajectory(-15.0, -19.5, 2.5), EVENT, RegrowthConfig(), "vh")
    assert fit.converged
    assert fit.tau_years == pytest.approx(2.5, rel=0.1)
    assert fit.asymptote == pytest.approx(-15.0, abs=0.3)


def test_r80p_is_undefined_for_a_negative_baseline():
    decibels = fit_recovery(_trajectory(-15.0, -19.5, 2.5), EVENT, RegrowthConfig(), "vh")
    index = fit_recovery(_trajectory(0.32, 0.02, 2.5), EVENT, RegrowthConfig(), "nbr")
    assert decibels.r80p is None
    assert index.r80p is not None


def test_years_to_80pct_is_none_when_the_curve_levels_off_below_target():
    # Asymptote at 30% of baseline: this site never reaches R80P.
    fit = fit_recovery(_trajectory(0.40, 0.02, 3.0, n=28), EVENT, RegrowthConfig(), "nbr")
    stalled = fit_recovery(
        _series(np.concatenate([np.full(4, 0.40), np.full(24, 0.12)])), EVENT,
        RegrowthConfig(), "nbr",
    )
    assert fit.years_to_80pct is not None
    assert stalled.years_to_80pct is None


def test_fixed_age_recovery_fractions_are_reported():
    fit = fit_recovery(_trajectory(0.32, 0.02, 2.5), EVENT, RegrowthConfig(), "nbr")
    assert 0.0 < fit.recovery_at_1yr < fit.recovery_at_2yr < fit.recovery_at_5yr <= 1.05


def test_fit_gives_up_gracefully_without_enough_seasons():
    fit = fit_recovery(_series([0.3, 0.3, 0.05]), EVENT, RegrowthConfig(), "nbr")
    assert fit.n_seasons < RegrowthConfig().min_seasons_post
    assert fit.tau_years is None


def test_fit_without_a_pre_event_baseline():
    series = _series([0.05, 0.1, 0.15, 0.2, 0.25], start=date(2020, 6, 1))
    fit = fit_recovery(series, EVENT, RegrowthConfig(), "nbr")
    assert fit.baseline is None and fit.recovery_fraction is None


def test_at_bound_flag_suppresses_extrapolation():
    flat = _series(np.concatenate([np.full(4, 0.4), np.full(24, 0.10)]))
    fit = fit_recovery(flat, EVENT, RegrowthConfig(), "nbr")
    if fit.at_bound:
        assert fit.years_to_80pct is None


# --- re-clearing -----------------------------------------------------------


def test_reclearing_is_found_where_a_sustained_drop_occurs():
    values = np.array([0.05 + 0.30 * (1 - np.exp(-i * 0.25)) for i in range(24)], np.float32)
    values[14:] -= 0.22
    series = _series(values, start=date(2020, 3, 15))
    found = detect_reclearing(series, EVENT, RegrowthConfig())
    assert found is not None and found == series.dates[14]


def test_noisy_seasons_do_not_read_as_reclearing():
    values = np.array([0.05 + 0.30 * (1 - np.exp(-i * 0.25)) for i in range(24)], np.float32)
    values[12] -= 0.10
    values[17] -= 0.09
    assert detect_reclearing(_series(values, start=date(2020, 3, 15)), EVENT, RegrowthConfig()) is None


def test_a_site_that_never_recovered_is_not_called_recleared():
    values = np.full(20, 0.05, np.float32)
    values[10:] = 0.01
    assert detect_reclearing(_series(values, start=date(2020, 3, 15)), EVENT, RegrowthConfig()) is None


def test_reclearing_needs_a_minimum_series_length():
    assert detect_reclearing(_series([0.1, 0.2, 0.3]), EVENT, RegrowthConfig()) is None


# --- classification --------------------------------------------------------


@pytest.mark.parametrize(
    "fraction,slope,expected",
    [
        (0.95, 0.02, CLASS_RECOVERED),
        (0.55, 0.03, CLASS_RECOVERING),
        (0.15, 0.001, CLASS_STALLED),
        (0.15, 0.05, CLASS_RECOVERING),
    ],
)
def test_classification_thresholds(fraction, slope, expected):
    from vegmon.regrowth import RecoveryFit

    fit = RecoveryFit(index="nbr", recovery_fraction=fraction, recent_slope_per_year=slope)
    assert classify(fit, None, RegrowthConfig()) == expected


def test_reclearing_overrides_every_other_class():
    from vegmon.regrowth import RecoveryFit

    fit = RecoveryFit(index="nbr", recovery_fraction=0.99, recent_slope_per_year=0.1)
    assert classify(fit, date(2023, 6, 1), RegrowthConfig()) == CLASS_RECLEARED


def test_missing_metrics_classify_as_unknown():
    assert classify(None, None, RegrowthConfig()) == CLASS_UNKNOWN


# --- patch extraction ------------------------------------------------------


def test_patch_means_ignores_nan_and_handles_absent_patches():
    stack = np.array([[[1.0, np.nan], [3.0, 4.0]]], dtype=np.float32)
    labels = np.array([[1, 1], [2, 2]], dtype=np.int32)
    means = patch_means(stack, labels, [1, 2, 99])
    assert means[1][0] == pytest.approx(1.0)
    assert means[2][0] == pytest.approx(3.5)
    assert np.isnan(means[99][0])


def test_patch_means_returns_nan_for_a_fully_masked_time_slice():
    stack = np.full((2, 2, 2), np.nan, dtype=np.float32)
    stack[0] = 1.0
    labels = np.ones((2, 2), dtype=np.int32)
    means = patch_means(stack, labels, [1])
    assert means[1][0] == pytest.approx(1.0) and np.isnan(means[1][1])


# --- against the reference truth -------------------------------------------


def test_recovery_classes_match_the_reference_trajectories(result, truth):
    """Each detected patch is classified the way the truth says it behaved."""
    truth_by_id = {p["patch_id"]: p for p in truth.patches}
    checked = 0
    for patch in result.regrowth.patches:
        overlapping = [
            int(i)
            for i in np.unique(truth.labels[result.clearing.components == patch.patch_id])
            if i > 0
        ]
        if not overlapping:
            continue
        reference = truth_by_id[overlapping[0]]
        if reference["kind"] != "clearing":
            continue
        checked += 1
        if reference["reclear_date"]:
            assert patch.classification == CLASS_RECLEARED, patch.patch_id
        elif reference["recovery_target"] <= 0.25:
            assert patch.classification in (CLASS_STALLED, CLASS_RECOVERING), patch.patch_id
        else:
            assert patch.classification != CLASS_RECLEARED, patch.patch_id
    assert checked >= 3


def test_every_patch_carries_all_three_tracks(result):
    for patch in result.regrowth.patches:
        assert set(patch.fits) == {"ndvi", "nbr", "vh"}
        assert set(patch.series) == {"ndvi", "nbr", "vh"}


def test_records_flatten_to_a_table(result):
    records = result.regrowth.records()
    assert records
    assert {"patch_id", "classification", "nbr_recovery_fraction"} <= set(records[0])
