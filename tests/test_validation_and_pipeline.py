import json

import numpy as np
import pytest

from vegmon.validation import (
    SIZE_CLASSES,
    pixel_accuracy,
    size_class_for,
    summarise_size_classes,
)


# --- metric arithmetic -----------------------------------------------------


def test_pixel_accuracy_arithmetic():
    detected = np.array([True, True, False, False])
    reference = np.array([True, False, True, False])
    accuracy = pixel_accuracy("t", detected, reference)
    assert (accuracy.true_positive, accuracy.false_positive, accuracy.false_negative) == (1, 1, 1)
    assert accuracy.precision == pytest.approx(0.5)
    assert accuracy.recall == pytest.approx(0.5)
    assert accuracy.f1 == pytest.approx(0.5)
    assert accuracy.iou == pytest.approx(1 / 3)


def test_pixel_accuracy_with_no_detections_is_nan_not_zero():
    accuracy = pixel_accuracy("t", np.zeros(4, bool), np.array([True, False, False, False]))
    assert np.isnan(accuracy.precision)
    assert accuracy.recall == pytest.approx(0.0)


@pytest.mark.parametrize(
    "area,expected",
    [(0.2, "<0.5 ha"), (0.5, "0.5-2 ha"), (2.0, "2-10 ha"), (25.0, "10-50 ha"), (400.0, ">50 ha")],
)
def test_size_class_boundaries(area, expected):
    assert size_class_for(area) == expected


def test_size_classes_are_contiguous():
    for (_, _, high), (_, low, _) in zip(SIZE_CLASSES, SIZE_CLASSES[1:]):
        assert high == low


def test_summarise_size_classes_skips_empty_classes():
    from vegmon.validation import PatchMatch

    matches = [
        PatchMatch(1, 5.0, "2-10 ha", detected=True, iou=0.9),
        PatchMatch(2, 6.0, "2-10 ha", detected=False, iou=0.05),
    ]
    summary = summarise_size_classes(matches)
    assert set(summary) == {"2-10 ha"}
    assert summary["2-10 ha"]["recall"] == pytest.approx(0.5)


# --- against the reference truth -------------------------------------------


def test_confirmed_tier_is_precise(result):
    accuracy = result.validation.pixel["fused_confirmed"]
    assert accuracy.precision > 0.90
    assert accuracy.recall > 0.80


def test_fusion_beats_either_sensor_alone_on_precision(result):
    """The central claim of the design, measured."""
    validation = result.validation
    fused = validation.pixel["fused_confirmed"].precision
    optical = validation.pixel["sentinel2_only"].precision
    assert fused > optical
    assert fused >= validation.pixel["fused_any_tier"].precision


def test_total_area_bias_is_small(result):
    assert abs(result.validation.area["bias_percent"]) < 20.0


def test_no_look_alike_class_reaches_the_confirmed_tier(result):
    for name, entry in result.validation.decoys.items():
        if name == "stable background":
            assert entry["confirmed_fraction"] < 0.01, name
        else:
            assert entry["confirmed_fraction"] == 0.0, name


def test_the_fire_scar_is_caught_by_sentinel2_and_rejected_by_fusion(result):
    fire = result.validation.decoys["fire scar"]
    assert fire["s2_only_fraction"] > 0.5
    assert fire["confirmed_fraction"] == 0.0


def test_recall_rises_with_patch_size(result):
    recall = result.validation.size_class_recall
    below_mmu = recall.get("<0.5 ha")
    if below_mmu:
        # Removed by the sieve by design, not missed by the detector.
        assert below_mmu["recall"] == 0.0
    large = [v["recall"] for k, v in recall.items() if k not in ("<0.5 ha",)]
    assert large and min(large) >= 0.5


def test_validation_serialises_to_json(result):
    payload = json.dumps(result.validation.as_dict())
    assert "size_class_recall" in payload


def test_headline_is_human_readable(result):
    headline = result.validation.headline()
    assert "precision" in headline and "Area:" in headline


# --- pipeline --------------------------------------------------------------


def test_pipeline_summary_has_the_expected_shape(result):
    summary = result.summary()
    assert set(summary) >= {
        "aoi", "crs", "area_ha", "tiers", "recovery", "validation",
        "normalisation_offsets", "runtime_seconds", "periods",
    }
    json.dumps(summary, default=str)


def test_pipeline_dates_each_patch_individually(result):
    dates = {p.patch_id: p.event_date for p in result.regrowth.patches}
    assert len(set(dates.values())) > 1, "every patch got the same fallback date"
    for event in dates.values():
        assert result.config.periods.pre_end <= event <= result.config.periods.post_start


def test_pipeline_skips_validation_without_truth(scene, config):
    from vegmon.pipeline import run_pipeline
    from vegmon.series import Scene

    bare = Scene(optical=scene.optical, radar=scene.radar, grid=scene.grid)
    outcome = run_pipeline(bare, config)
    assert outcome.validation is None
    assert "validate" not in outcome.timings_seconds


def test_pipeline_can_carry_extra_tiers_into_regrowth(scene, config):
    from vegmon.pipeline import run_pipeline

    outcome = run_pipeline(scene, config, regrowth_tiers=("confirmed", "s2_only"))
    tiers = {p.tier for p in outcome.regrowth.patches}
    assert "s2_only" in tiers
