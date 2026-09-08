import json
import warnings
from datetime import date

import pytest

from vegmon.config import AOI, DetectionConfig, Periods, PipelineConfig, demo_config
from vegmon.grid import Grid


def test_grid_geometry_and_area_conversions():
    grid = Grid.from_origin("EPSG:32755", 700000, 6580000, 10.0, 512, 256)
    assert grid.shape == (256, 512)
    assert grid.resolution == (10.0, 10.0)
    assert grid.pixel_area_ha == pytest.approx(0.01)
    west, south, east, north = grid.bounds
    assert (west, north) == (700000, 6580000)
    assert east == pytest.approx(705120) and south == pytest.approx(6577440)
    assert grid.area_ha(50) == pytest.approx(0.5)
    assert grid.pixels_for_ha(0.5) == 50


def test_pixels_for_ha_never_returns_zero():
    grid = Grid.from_origin("EPSG:32755", 0, 0, 10.0, 4, 4)
    assert grid.pixels_for_ha(0.0001) == 1


def test_xy_centres_are_pixel_centres():
    grid = Grid.from_origin("EPSG:32755", 1000, 2000, 10.0, 3, 2)
    xs, ys = grid.xy_centres()
    assert list(xs) == [1005.0, 1015.0, 1025.0]
    assert list(ys) == [1995.0, 1985.0]


def test_aoi_rejects_an_inverted_bounding_box():
    with pytest.raises(ValueError):
        AOI(name="bad", west=150.0, south=-30.0, east=149.0, north=-31.0)


def test_periods_rejects_overlapping_windows():
    with pytest.raises(ValueError):
        Periods(
            pre_start=date(2019, 1, 1), pre_end=date(2020, 6, 1),
            post_start=date(2020, 1, 1), post_end=date(2020, 12, 1),
            series_start=date(2019, 1, 1), series_end=date(2021, 1, 1),
        )


def test_periods_rejects_a_series_that_does_not_span_the_windows():
    with pytest.raises(ValueError):
        Periods(
            pre_start=date(2019, 1, 1), pre_end=date(2019, 6, 1),
            post_start=date(2020, 1, 1), post_end=date(2020, 6, 1),
            series_start=date(2019, 3, 1), series_end=date(2021, 1, 1),
        )


def test_periods_accepts_iso_strings():
    periods = Periods("2019-01-01", "2019-06-01", "2020-01-01", "2020-06-01",
                      "2019-01-01", "2021-01-01")
    assert periods.pre_start == date(2019, 1, 1)


def test_phenology_check_warns_on_mismatched_seasons():
    periods = Periods(
        pre_start=date(2019, 3, 1), pre_end=date(2019, 11, 30),
        post_start=date(2020, 3, 1), post_end=date(2020, 8, 31),
        series_start=date(2019, 1, 1), series_end=date(2021, 1, 1),
    )
    assert periods.phenology_offset_days() > 21
    with pytest.warns(UserWarning, match="seasonal cycle"):
        periods.check_phenology()


def test_phenology_check_is_silent_when_windows_are_aligned():
    periods = demo_config().periods
    assert periods.phenology_offset_days() <= 21
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        periods.check_phenology()


def test_phenology_offset_wraps_around_the_year():
    periods = Periods(
        pre_start=date(2018, 12, 1), pre_end=date(2019, 1, 31),
        post_start=date(2019, 12, 1), post_end=date(2020, 1, 31),
        series_start=date(2018, 12, 1), series_end=date(2020, 2, 1),
    )
    assert periods.phenology_offset_days() <= 2


def test_config_json_roundtrip(tmp_path):
    original = demo_config()
    path = original.to_json(tmp_path / "config.json")
    restored = PipelineConfig.from_json(path)
    assert restored.aoi == original.aoi
    assert restored.periods == original.periods
    assert restored.detection == original.detection
    assert json.loads(path.read_text())["aoi"]["crs"] == "EPSG:32755"


def test_detection_defaults_target_open_woodland_not_closed_forest():
    # NSW reporting treats ~20% foliage cover as woody; a closed-forest
    # threshold would silently exclude the country most clearing happens in.
    config = DetectionConfig()
    assert config.min_pre_ndvi <= 0.45
    assert config.min_pre_nbr <= 0.20
