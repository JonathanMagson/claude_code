"""The synthetic datacube is the reference the rest of the suite trusts.

If its physics drift, every accuracy assertion downstream becomes vacuous, so
these tests pin the properties the detectors actually depend on rather than
just checking that arrays come out the right shape.
"""

import numpy as np
import pytest

from vegmon.synthetic import (
    DEMO_PATCHES,
    fractal_field,
    generate_scene,
    load_scene,
    save_scene,
    scene_truth,
)


def test_patch_areas_match_what_was_requested(truth):
    for record in truth.patches:
        assert record["area_ha"] == pytest.approx(record["requested_area_ha"], rel=0.05)


def test_area_scaling_keeps_the_disturbed_fraction_stable(config):
    small = scene_truth(generate_scene(config, size=64, seed=3))
    large = scene_truth(generate_scene(config, size=128, seed=3))
    small_fraction = float((small.labels > 0).mean())
    large_fraction = float((large.labels > 0).mean())
    assert small_fraction == pytest.approx(large_fraction, abs=0.05)
    assert 0.10 < large_fraction < 0.30


def test_truth_masks_are_disjoint(truth):
    assert not (truth.clearing_mask & truth.fire_mask).any()
    assert not (truth.clearing_mask & truth.crop_mask).any()


def test_every_patch_kind_is_present(truth):
    kinds = {record["kind"] for record in truth.patches}
    assert {"clearing", "fire", "crop", "crop_fallow"} <= kinds


def test_clearing_patches_were_woody_before_the_event(scene, truth, config):
    pre = scene.optical.composite(
        config.periods.pre_start, config.periods.pre_end, max_cloud_cover=0.6
    )
    mask = truth.clearing_mask
    assert float(np.nanmean(pre.ndvi[mask])) > config.detection.min_pre_ndvi
    assert float(np.nanmean(pre.nbr[mask])) > config.detection.min_pre_nbr


def test_clearing_produces_a_spectral_and_a_structural_drop(scene, truth, config):
    periods = config.periods
    pre = scene.optical.composite(periods.pre_start, periods.pre_end, max_cloud_cover=0.6)
    post = scene.optical.composite(periods.post_start, periods.post_end, max_cloud_cover=0.6)
    radar_pre = scene.radar.composite(periods.pre_start, periods.pre_end)
    radar_post = scene.radar.composite(periods.post_start, periods.post_end)

    stable = truth.labels == 0
    dnbr = (pre.nbr - post.nbr) - float(np.nanmedian((pre.nbr - post.nbr)[stable]))
    dvh = (radar_pre.vh_db - radar_post.vh_db) - float(
        np.nanmedian((radar_pre.vh_db - radar_post.vh_db)[stable])
    )
    cleared = truth.clearing_mask
    assert float(np.nanmean(dnbr[cleared])) > config.detection.dnbr_threshold
    assert float(np.nanmean(dvh[cleared])) > config.detection.vh_drop_db


def test_fire_scar_collapses_optically_but_keeps_its_structure(scene, truth, config):
    """The reason two sensors are better than one, asserted directly."""
    periods = config.periods
    pre = scene.optical.composite(periods.pre_start, periods.pre_end, max_cloud_cover=0.6)
    post = scene.optical.composite(periods.post_start, periods.post_end, max_cloud_cover=0.6)
    radar_pre = scene.radar.composite(periods.pre_start, periods.pre_end)
    radar_post = scene.radar.composite(periods.post_start, periods.post_end)

    stable = truth.labels == 0
    dnbr = (pre.nbr - post.nbr) - float(np.nanmedian((pre.nbr - post.nbr)[stable]))
    dvh = (radar_pre.vh_db - radar_post.vh_db) - float(
        np.nanmedian((radar_pre.vh_db - radar_post.vh_db)[stable])
    )
    fire = truth.fire_mask
    assert float(np.nanmean(dnbr[fire])) > config.detection.dnbr_threshold
    assert float(np.nanmean(dvh[fire])) < config.detection.vh_drop_db


def test_cropping_is_not_persistently_green(scene, truth, config):
    """The persistence gate has something real to catch."""
    window = scene.optical.select(
        config.periods.pre_start, config.periods.pre_end, max_cloud_cover=0.6
    )
    stack = window.index_stack("ndvi")
    finite = np.where(np.isfinite(stack), stack, np.nan)
    all_nan = np.all(~np.isfinite(stack), axis=0)
    percentile = np.nanpercentile(np.where(all_nan[None], 0.0, finite), 10, axis=0)

    crop = truth.crop_mask
    cleared = truth.clearing_mask
    assert float(np.nanmean(percentile[crop])) < config.detection.min_pre_ndvi_persistent
    assert float(np.nanmean(percentile[cleared])) > config.detection.min_pre_ndvi_persistent


def test_clouds_are_present_and_spatially_correlated(scene):
    fractions = scene.optical.scene_cloud_fraction()
    assert 0.1 < float(fractions.mean()) < 0.8
    assert float(fractions.max()) > 0.5


def test_radar_carries_both_orbit_directions(scene):
    assert set(scene.radar.orbit_state) == {"ascending", "descending"}


def test_generation_is_reproducible(config):
    a = generate_scene(config, size=48, seed=99)
    b = generate_scene(config, size=48, seed=99)
    assert np.allclose(a.optical.bands["nir"], b.optical.bands["nir"])
    assert np.array_equal(scene_truth(a).labels, scene_truth(b).labels)


def test_different_seeds_give_different_scenes(config):
    a = generate_scene(config, size=48, seed=1)
    b = generate_scene(config, size=48, seed=2)
    assert not np.allclose(a.optical.bands["nir"], b.optical.bands["nir"])


def test_save_and_load_roundtrip(tiny_scene, tmp_path):
    path = save_scene(tiny_scene, tmp_path / "scene.npz")
    restored = load_scene(path)
    assert len(restored.optical) == len(tiny_scene.optical)
    assert np.allclose(restored.optical.bands["swir2"], tiny_scene.optical.bands["swir2"])
    assert np.allclose(restored.radar.vh_db, tiny_scene.radar.vh_db)
    assert np.array_equal(
        scene_truth(restored).labels, scene_truth(tiny_scene).labels
    )
    assert restored.grid == tiny_scene.grid


def test_reflectance_stays_physical(tiny_scene):
    for name, band in tiny_scene.optical.bands.items():
        assert float(np.nanmin(band)) >= 0.0, name
        assert float(np.nanmax(band)) <= 1.0, name


def test_backscatter_is_in_a_plausible_decibel_range(tiny_scene):
    vh = tiny_scene.radar.vh_db
    assert -40.0 < float(np.nanpercentile(vh, 1)) < -10.0
    assert -30.0 < float(np.nanpercentile(vh, 99)) < 0.0
    # Cross-pol always returns less than co-pol.
    assert float(np.nanmean(tiny_scene.radar.vv_db - vh)) > 0.0


def test_fractal_field_is_normalised():
    field = fractal_field(np.random.default_rng(0), (32, 32))
    assert field.min() == pytest.approx(0.0)
    assert field.max() == pytest.approx(1.0)


def test_scene_without_truth_raises_a_clear_error(tiny_scene):
    from vegmon.series import Scene

    bare = Scene(optical=tiny_scene.optical, radar=tiny_scene.radar, grid=tiny_scene.grid)
    with pytest.raises(ValueError, match="ground truth"):
        scene_truth(bare)


def test_demo_layout_spans_the_reporting_size_classes():
    areas = sorted(p.area_ha for p in DEMO_PATCHES if p.kind == "clearing")
    assert areas[0] < 0.5  # below the minimum mapping unit, deliberately
    assert areas[-1] > 20.0
