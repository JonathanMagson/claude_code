from datetime import date, timedelta

import numpy as np
import pytest

from vegmon.config import DetectionConfig
from vegmon.detect import (
    attach_true_event_dates,
    detect_optical,
    detect_radar,
    estimate_event_timing,
    robust_offset,
)
from vegmon.fuse import (
    TIER_CONFIRMED,
    TIER_OPTICAL_ONLY,
    TIER_RADAR_ONLY,
    clean_mask,
    fuse,
    sieve,
)
from vegmon.grid import Grid


# --- normalisation ---------------------------------------------------------


def test_robust_offset_is_the_median_of_the_stable_population():
    delta = np.concatenate([np.full(900, 0.05), np.full(100, 0.9)])
    assert robust_offset(delta) == pytest.approx(0.05, abs=1e-6)


def test_robust_offset_refines_away_from_contamination():
    # 45% of the population changed: the plain median survives, but only just.
    delta = np.concatenate([np.full(550, 0.02), np.full(450, 0.6)])
    plain = robust_offset(delta)
    refined = robust_offset(delta, threshold=0.2)
    assert plain == pytest.approx(0.02, abs=1e-6)
    assert refined == pytest.approx(0.02, abs=1e-6)


def test_robust_offset_falls_back_when_the_population_is_tiny():
    delta = np.full(500, 0.3)
    population = np.zeros(500, bool)
    population[:4] = True
    assert robust_offset(delta, population) == pytest.approx(0.3)


def test_robust_offset_handles_an_all_nan_input():
    assert robust_offset(np.full(10, np.nan)) == 0.0


# --- optical detector ------------------------------------------------------


def test_optical_detector_finds_the_cleared_patches(scene, truth, config):
    change = detect_optical(scene.optical, config.periods, config.detection)
    assert float(change.mask[truth.clearing_mask].mean()) > 0.85


def test_optical_detector_rejects_cropping_and_fallow(scene, truth, config):
    change = detect_optical(scene.optical, config.periods, config.detection)
    assert float(change.mask[truth.crop_mask].mean()) < 0.02


def test_optical_detector_is_fooled_by_fire_on_purpose(scene, truth, config):
    """Sentinel-2 alone cannot tell a burn scar from a clearing. It should not
    quietly succeed here - the whole design rests on it failing."""
    change = detect_optical(scene.optical, config.periods, config.detection)
    assert float(change.mask[truth.fire_mask].mean()) > 0.5


def test_optical_offsets_are_small_after_normalisation(scene, config):
    change = detect_optical(scene.optical, config.periods, config.detection)
    assert abs(change.offsets["dnbr"]) < 0.1
    assert abs(change.offsets["dndvi"]) < 0.1


def test_optical_gate_names_are_stable_and_all_report(scene, config):
    change = detect_optical(scene.optical, config.periods, config.detection)
    assert set(change.gates) == {
        "observations_pre", "observations_post", "woody_before",
        "persistently_woody", "not_water", "steady_through_the_year",
        "nbr_drop", "ndvi_drop",
    }
    assert set(change.rejection_summary()) == set(change.gates)


def test_persistence_gate_is_what_stops_the_fallow_paddock(scene, truth, config):
    strict = detect_optical(scene.optical, config.periods, config.detection)
    relaxed = detect_optical(
        scene.optical,
        config.periods,
        DetectionConfig(min_pre_ndvi_persistent=-1.0),
    )
    fallow = np.isin(truth.labels, [p["patch_id"] for p in truth.patches
                                    if p["kind"] == "crop_fallow"])
    assert float(strict.mask[fallow].mean()) < 0.02
    assert float(relaxed.mask[fallow].mean()) > float(strict.mask[fallow].mean())


def test_amplitude_gate_rejects_a_pixel_that_swings_through_the_year(scene, truth, config):
    """Persistent greenness is not a woody test - irrigated cropping passes it.

    What separates woody vegetation from any crop is the shape of the year:
    woodland barely moves, a crop swings between planting and harvest.
    """
    from dataclasses import replace

    strict = detect_optical(
        scene.optical,
        config.periods,
        replace(config.detection, max_pre_seasonal_amplitude=0.35),
    )
    assert "steady_through_the_year" in strict.gates
    assert strict.pre_amplitude is not None
    # The cleared patches were woodland before the event, so they must survive
    # the gate; the cropping paddocks must not.
    assert float(strict.gates["steady_through_the_year"][truth.clearing_mask].mean()) > 0.8
    crop = truth.crop_mask
    assert float(strict.gates["steady_through_the_year"][crop].mean()) < float(
        strict.gates["steady_through_the_year"][truth.clearing_mask].mean()
    )
    assert float(strict.mask[truth.clearing_mask].mean()) > 0.80


def test_amplitude_gate_is_off_by_default(scene, config):
    change = detect_optical(scene.optical, config.periods, config.detection)
    assert change.pre_amplitude is None
    assert bool(change.gates["steady_through_the_year"].all())


def test_optical_detector_warns_on_misaligned_phenology(scene, config):
    from vegmon.config import Periods

    periods = Periods(
        pre_start=date(2019, 3, 1), pre_end=date(2019, 11, 30),
        post_start=date(2020, 3, 1), post_end=date(2020, 8, 31),
        series_start=config.periods.series_start, series_end=config.periods.series_end,
    )
    with pytest.warns(UserWarning, match="seasonal cycle"):
        detect_optical(scene.optical, periods, config.detection)


# --- radar detector --------------------------------------------------------


def test_radar_detector_finds_clearing_and_ignores_fire(scene, truth, config):
    change = detect_radar(scene.radar, config.periods, config.detection)
    assert float(change.mask[truth.clearing_mask].mean()) > 0.85
    assert float(change.mask[truth.fire_mask].mean()) < 0.02


def test_radar_detector_reports_per_orbit_drops(scene, config):
    change = detect_radar(scene.radar, config.periods, config.detection)
    assert set(change.orbit_drops) == {"ascending", "descending"}
    assert change.orbit_agreement is not None


def test_radar_offset_removes_the_soil_moisture_shift(scene, config):
    change = detect_radar(scene.radar, config.periods, config.detection)
    assert abs(change.offsets["vh_db"]) < 3.0


# --- cleanup and fusion ----------------------------------------------------


def test_sieve_removes_components_below_the_threshold():
    mask = np.zeros((20, 20), bool)
    mask[2:4, 2:4] = True     # 4 pixels
    mask[10:16, 10:16] = True  # 36 pixels
    kept = sieve(mask, min_pixels=10)
    assert kept.sum() == 36


def test_sieve_with_a_threshold_of_one_is_a_no_op():
    mask = np.array([[True, False]])
    assert np.array_equal(sieve(mask, 1), mask)


def test_clean_mask_fills_holes_and_drops_filaments():
    grid = Grid.from_origin("EPSG:32755", 0, 0, 10.0, 40, 40)
    mask = np.zeros((40, 40), bool)
    mask[5:25, 5:25] = True
    mask[14:16, 14:16] = False  # retained trees inside the paddock
    mask[30, 2:30] = True       # a one-pixel filament
    cleaned = clean_mask(mask, grid, DetectionConfig(min_mapping_unit_ha=0.5))
    assert cleaned[14, 14]
    assert not cleaned[30, 10]


def test_fusion_tiers_follow_which_sensors_agreed(scene, config):
    optical = detect_optical(scene.optical, config.periods, config.detection)
    radar = detect_radar(scene.radar, config.periods, config.detection)
    clearing = fuse(optical, radar, config.detection)

    assert clearing.summary()["confirmed"]["patches"] >= 1
    for record in clearing.records:
        if record["tier"] == "confirmed":
            assert record["agreement_fraction"] >= config.detection.confirm_overlap
        else:
            assert record["agreement_fraction"] < config.detection.confirm_overlap


def test_fusion_puts_the_fire_scar_in_the_optical_only_tier(scene, truth, config):
    optical = detect_optical(scene.optical, config.periods, config.detection)
    radar = detect_radar(scene.radar, config.periods, config.detection)
    clearing = fuse(optical, radar, config.detection)
    assert not (clearing.mask("confirmed") & truth.fire_mask).any()
    assert (clearing.mask("s2_only") & truth.fire_mask).any()


def test_fusion_on_empty_detections_returns_an_empty_map(scene, config):
    optical = detect_optical(scene.optical, config.periods, config.detection)
    radar = detect_radar(scene.radar, config.periods, config.detection)
    optical.mask[:] = False
    radar.mask[:] = False
    clearing = fuse(optical, radar, config.detection)
    assert len(clearing) == 0
    assert clearing.area_ha() == 0.0
    assert clearing.summary()["confirmed"]["patches"] == 0


def test_clearing_map_filtering_and_masks(result):
    clearing = result.clearing
    confirmed = clearing.filter_tier("confirmed")
    assert all(r["tier"] == "confirmed" for r in confirmed.records)
    assert confirmed.mask().sum() == clearing.mask("confirmed").sum()
    assert clearing.area_ha("confirmed") == pytest.approx(confirmed.area_ha(), abs=1e-6)


def test_geodataframe_carries_geometry_and_shape_metrics(result):
    frame = result.clearing.to_geodataframe()
    assert len(frame) == len(result.clearing)
    assert str(frame.crs).upper().endswith("32755")
    assert (frame["shape_index"] >= 1.0).all()
    assert frame["area_ha"].is_monotonic_decreasing


def test_write_vector_produces_a_readable_geopackage(result, tmp_path):
    import geopandas as gpd

    path = result.clearing.write_vector(tmp_path / "clearing.gpkg")
    assert len(gpd.read_file(path)) == len(result.clearing)


# --- event timing ----------------------------------------------------------


def test_event_timing_recovers_the_true_dates(scene, truth, config):
    ids = [p["patch_id"] for p in truth.patches if p["kind"] == "clearing"]
    timings = attach_true_event_dates(
        estimate_event_timing(
            scene.optical, scene.radar, truth.labels, ids, config.periods, config.detection
        ),
        truth.patches,
    )
    dated = [t for t in timings if t.s1_latency_days is not None]
    assert dated
    assert all(0 <= t.s1_latency_days <= 30 for t in dated)


def test_radar_alerts_sooner_than_optical_under_cloud(scene, truth, config):
    """The claim the whole demo is built on, asserted rather than asserted-at."""
    ids = [p["patch_id"] for p in truth.patches if p["kind"] == "clearing"]
    timings = attach_true_event_dates(
        estimate_event_timing(
            scene.optical, scene.radar, truth.labels, ids, config.periods, config.detection
        ),
        truth.patches,
    )
    optical = [t.s2_latency_days for t in timings if t.s2_latency_days is not None]
    radar = [t.s1_latency_days for t in timings if t.s1_latency_days is not None]
    assert optical and radar
    assert np.median(radar) < np.median(optical)


def test_event_timing_skips_patches_with_no_pixels(scene, config):
    labels = np.zeros(scene.grid.shape, np.int32)
    assert estimate_event_timing(
        scene.optical, scene.radar, labels, [42], config.periods, config.detection
    ) == []
