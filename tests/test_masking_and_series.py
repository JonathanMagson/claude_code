from datetime import date, timedelta

import numpy as np
import pytest

from vegmon import masking
from vegmon.grid import Grid
from vegmon.series import OpticalSeries, RadarSeries, S2_BANDS


def _grid(size=16):
    return Grid.from_origin("EPSG:32755", 700000, 6580000, 10.0, size, size)


def _optical(size=16, n=6, start=date(2020, 1, 1), step=10):
    grid = _grid(size)
    times = [start + timedelta(days=step * i) for i in range(n)]
    rng = np.random.default_rng(0)
    bands = {k: rng.uniform(0.05, 0.4, (n, size, size)).astype(np.float32) for k in S2_BANDS}
    scl = np.full((n, size, size), masking.SCL_VEGETATION, np.uint8)
    return OpticalSeries(grid=grid, times=times, bands=bands, scl=scl)


def test_scl_valid_mask_excludes_cloud_and_buffers_it():
    scl = np.full((1, 9, 9), masking.SCL_VEGETATION, np.uint8)
    scl[0, 4, 4] = masking.SCL_CLOUD_HIGH_PROB
    undilated = masking.scl_valid_mask(scl, dilation_pixels=0)
    dilated = masking.scl_valid_mask(scl, dilation_pixels=2)
    assert undilated[0].sum() == 80
    assert dilated[0].sum() < undilated[0].sum()
    assert not dilated[0, 4, 4]


def test_scl_valid_mask_keeps_water_and_bare_ground():
    scl = np.array([[[masking.SCL_WATER, masking.SCL_NOT_VEGETATED,
                      masking.SCL_CLOUD_SHADOW, masking.SCL_NO_DATA]]], np.uint8)
    valid = masking.scl_valid_mask(scl, dilation_pixels=0)
    assert list(valid[0, 0]) == [True, True, False, False]


def test_cloud_fraction_counts_unusable_pixels():
    scl = np.full((2, 4, 4), masking.SCL_VEGETATION, np.uint8)
    scl[1] = masking.SCL_CLOUD_HIGH_PROB
    fractions = masking.cloud_fraction(scl)
    assert fractions[0] == pytest.approx(0.0)
    assert fractions[1] == pytest.approx(1.0)


def test_masked_median_ignores_invalid_and_returns_nan_when_nothing_is_valid():
    stack = np.array([[[1.0]], [[100.0]], [[3.0]]], dtype=np.float32)
    valid = np.array([[[True]], [[False]], [[True]]])
    assert masking.masked_median(stack, valid)[0, 0] == pytest.approx(2.0)
    assert np.isnan(masking.masked_median(stack, np.zeros_like(valid))[0, 0])


def test_masked_median_on_an_empty_stack():
    empty = np.zeros((0, 3, 3), dtype=np.float32)
    out = masking.masked_median(empty, np.zeros((0, 3, 3), bool))
    assert out.shape == (3, 3) and np.all(np.isnan(out))


def test_boxcar_is_nan_aware():
    # A hole in the middle of an otherwise uniform field is filled from its
    # neighbours rather than poisoning the output, which is what lets the
    # multi-look filter run over Sentinel-1 composites with gaps in them.
    values = np.full((5, 5), 2.0, dtype=np.float32)
    values[2, 2] = np.nan
    smoothed = masking.boxcar(values, 3)
    assert np.all(np.isfinite(smoothed))
    assert smoothed[2, 2] == pytest.approx(2.0)


def test_boxcar_returns_nan_only_where_no_neighbour_is_finite():
    values = np.full((3, 3), np.nan, dtype=np.float32)
    assert np.all(np.isnan(masking.boxcar(values, 3)))


def test_boxcar_window_of_one_is_a_no_op():
    values = np.array([[1.0, 2.0]], dtype=np.float32)
    assert np.array_equal(masking.boxcar(values, 1), values)


def test_optical_series_rejects_a_missing_band():
    series = _optical()
    with pytest.raises(ValueError, match="missing bands"):
        OpticalSeries(
            grid=series.grid, times=series.times,
            bands={k: v for k, v in series.bands.items() if k != "swir2"},
            scl=series.scl,
        )


def test_optical_series_rejects_a_mismatched_shape():
    series = _optical()
    bad = dict(series.bands)
    bad["red"] = bad["red"][:, :4, :4]
    with pytest.raises(ValueError, match="shape"):
        OpticalSeries(grid=series.grid, times=series.times, bands=bad, scl=series.scl)


def test_optical_series_sorts_unordered_acquisitions():
    series = _optical(n=3)
    shuffled = OpticalSeries(
        grid=series.grid,
        times=[series.times[2], series.times[0], series.times[1]],
        bands={k: v[[2, 0, 1]] for k, v in series.bands.items()},
        scl=series.scl[[2, 0, 1]],
    )
    assert list(shuffled.times) == sorted(shuffled.times)
    assert np.allclose(shuffled.bands["red"], series.bands["red"])


def test_composite_counts_only_clear_observations():
    series = _optical(n=6)
    series.scl[2] = masking.SCL_CLOUD_HIGH_PROB
    composite = series.composite(date(2020, 1, 1), date(2020, 3, 1), max_cloud_cover=0.6)
    assert int(composite.n_obs.max()) == 5


def test_composite_drops_scenes_over_the_cloud_limit():
    series = _optical(n=4)
    series.scl[0] = masking.SCL_CLOUD_HIGH_PROB
    kept = series.select(date(2020, 1, 1), date(2020, 12, 1), max_cloud_cover=0.5)
    assert len(kept) == 3


def test_index_stack_masks_cloud_as_nan():
    series = _optical(n=3)
    series.scl[1] = masking.SCL_CLOUD_HIGH_PROB
    stack = series.index_stack("ndvi")
    assert np.all(np.isnan(stack[1]))
    assert np.all(np.isfinite(stack[0]))


def test_index_stack_rejects_an_unknown_index():
    with pytest.raises(KeyError):
        _optical().index_stack("nope")


def test_radar_composite_median_is_invariant_to_speckle_outliers():
    grid = _grid(8)
    times = [date(2020, 1, 1) + timedelta(days=12 * i) for i in range(5)]
    vh = np.full((5, 8, 8), -15.0, dtype=np.float32)
    vh[2] = -4.0  # one very bright pass
    vv = vh + 6.0
    series = RadarSeries(grid=grid, times=times, vv_db=vv, vh_db=vh)
    composite = series.composite(times[0], times[-1], speckle_window=1)
    assert composite.vh_db.mean() == pytest.approx(-15.0, abs=0.01)


def test_radar_by_orbit_splits_and_requires_metadata():
    grid = _grid(8)
    times = [date(2020, 1, 1) + timedelta(days=6 * i) for i in range(4)]
    shape = (4, 8, 8)
    series = RadarSeries(
        grid=grid, times=times,
        vv_db=np.zeros(shape, np.float32), vh_db=np.zeros(shape, np.float32),
        orbit_state=["ascending", "descending", "ascending", "descending"],
    )
    assert len(series.by_orbit("ascending")) == 2
    bare = RadarSeries(grid=grid, times=times, vv_db=np.zeros(shape, np.float32),
                       vh_db=np.zeros(shape, np.float32))
    with pytest.raises(ValueError, match="orbit"):
        bare.by_orbit("ascending")


def test_radar_cross_ratio_is_vh_minus_vv():
    grid = _grid(4)
    times = [date(2020, 1, 1), date(2020, 1, 13)]
    vh = np.full((2, 4, 4), -16.0, np.float32)
    vv = np.full((2, 4, 4), -9.0, np.float32)
    composite = RadarSeries(grid=grid, times=times, vv_db=vv, vh_db=vh).composite(
        times[0], times[-1], speckle_window=1
    )
    assert composite.cross_ratio_db.mean() == pytest.approx(-7.0, abs=0.01)
