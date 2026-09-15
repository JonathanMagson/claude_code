"""The pre-flight AOI survey.

These tests encode the two mistakes the survey exists to prevent, both of
which were made for real before it existed: pointing the detector at an AOI
with almost no woody vegetation in it, and comparing a drought year against a
wet one.
"""

import numpy as np
import pytest

from vegmon.config import DetectionConfig
from vegmon.survey import (
    MAX_COMPARABLE_NDVI_GAP,
    MIN_CLEAR_SCENES,
    PairAdvice,
    YearStats,
    compare_years,
    describe_year,
    format_survey,
    survey,
)


def _year(year=2022, clear=40, ndvi=0.55, woody=0.5, crop=0.05, amplitude=0.2):
    return YearStats(
        year=year, clear_scenes=clear, ndvi_median=ndvi, ndvi_p90=ndvi + 0.1,
        nbr_median=ndvi - 0.3, persistent_ndvi_p85=ndvi - 0.05,
        amplitude_median=amplitude, woody_fraction=woody, crop_fraction=crop,
        mean_cloud=0.1,
    )


def test_two_similar_years_are_comparable():
    advice = compare_years(_year(2021), _year(2022))
    assert advice.comparable
    assert advice.notes == ["comparable"]


def test_a_drought_year_against_a_wet_one_is_rejected():
    """The 2019 to 2020 drought break, which fooled the first real run."""
    # The real gap measured at the Boggabri site across the drought break.
    advice = compare_years(_year(2019, ndvi=0.516), _year(2020, ndvi=0.642))
    assert not advice.comparable
    assert advice.ndvi_gap > MAX_COMPARABLE_NDVI_GAP
    assert "climate difference" in "; ".join(advice.notes)


def test_a_greener_later_year_is_described_as_greener():
    assert "greener" in "; ".join(compare_years(_year(ndvi=0.3), _year(ndvi=0.7)).notes)
    assert "browner" in "; ".join(compare_years(_year(ndvi=0.7), _year(ndvi=0.3)).notes)


def test_an_aoi_with_no_woody_cover_is_rejected():
    """The Namoi irrigated cotton district: 2.7% woody, and every pair AVOID."""
    advice = compare_years(_year(2022, woody=0.02), _year(2023, woody=0.03))
    assert not advice.comparable
    assert "almost nothing here to lose" in "; ".join(advice.notes)


def test_thin_optical_coverage_is_rejected():
    advice = compare_years(_year(2022, clear=MIN_CLEAR_SCENES - 1), _year(2023))
    assert not advice.comparable
    assert "clear views" in "; ".join(advice.notes)


def test_a_year_with_no_observations_is_handled():
    advice = compare_years(YearStats.empty(2020), _year(2021))
    assert not advice.comparable
    assert "no usable observations" in "; ".join(advice.notes)


def test_empty_year_is_constructed_by_name_not_position():
    """Guards the constructor against a field being added in the middle."""
    empty = YearStats.empty(2020)
    assert empty.year == 2020 and empty.clear_scenes == 0
    assert np.isnan(empty.mean_cloud) and np.isnan(empty.woody_fraction)


def test_describe_year_reports_the_synthetic_landscape(scene, config):
    stats = describe_year(scene.optical, 2021, config.detection)
    assert stats.clear_scenes > 0
    assert 0.0 <= stats.woody_fraction <= 1.0
    assert 0.0 <= stats.mean_cloud <= 1.0
    assert np.isfinite(stats.ndvi_median)


def test_describe_year_outside_the_record_is_empty_not_an_error(scene, config):
    assert describe_year(scene.optical, 1999, config.detection).clear_scenes == 0


def test_survey_runs_over_the_synthetic_scene(scene, config):
    result = survey(scene.optical, "synthetic", scene.grid, [2020, 2021, 2022], config.detection)
    assert len(result.years) == 3
    assert len(result.pairs) == 2
    assert result.area_ha > 0
    assert set(result.suggested_thresholds) >= {
        "min_pre_ndvi_persistent", "max_pre_seasonal_amplitude"
    }


def test_survey_serialises_and_formats(scene, config):
    import json

    result = survey(scene.optical, "synthetic", scene.grid, [2020, 2021], config.detection)
    json.dumps(result.as_dict())
    text = format_survey(result)
    assert "epoch pairs:" in text
    assert "suggested epoch:" in text


def test_best_pair_prefers_the_closest_years():
    from vegmon.survey import SurveyResult

    pairs = [
        PairAdvice(2020, 2021, 0.09, 40, True, ["comparable"]),
        PairAdvice(2021, 2022, 0.01, 40, True, ["comparable"]),
        PairAdvice(2022, 2023, 0.30, 40, False, ["climate difference"]),
    ]
    outcome = SurveyResult("a", 100.0, [], pairs)
    best = outcome.best_pair()
    assert (best.baseline, best.detection) == (2021, 2022)


def test_best_pair_is_none_when_nothing_is_comparable():
    from vegmon.survey import SurveyResult

    outcome = SurveyResult("a", 100.0, [], [PairAdvice(2020, 2021, 0.4, 40, False, ["no"])])
    assert outcome.best_pair() is None
    assert "none of these year pairs is fair" in format_survey(outcome)
