"""Tests for the geolocation offset measurement.

Built on synthetic rasters with a shift put in deliberately, so the tool can be
checked against a known answer rather than against a guess about real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import measure_shift  # noqa: E402

PIXEL = 20.0
ORIGIN_X, ORIGIN_Y = 500000.0, 6500000.0
CRS = "EPSG:32755"


def speckled_scene(size: int = 768, seed: int = 0) -> np.ndarray:
    """Blobby linear-power field, so correlation has real structure to lock onto."""
    rng = np.random.default_rng(seed)
    coarse = rng.gamma(shape=3.0, scale=0.03, size=(size // 8, size // 8))
    scene = np.kron(coarse, np.ones((8, 8)))
    return (scene[:size, :size] * rng.gamma(shape=8.0, scale=0.125, size=(size, size))).astype("float32")


def write(path: Path, data: np.ndarray, row_offset: int = 0, col_offset: int = 0) -> Path:
    """Write *data* georeferenced so that it sits row/col_offset from the origin."""
    transform = from_origin(
        ORIGIN_X + col_offset * PIXEL, ORIGIN_Y - row_offset * PIXEL, PIXEL, PIXEL
    )
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32", crs=CRS, transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
    return path


def test_no_shift_is_reported_as_none(tmp_path: Path):
    scene = speckled_scene()
    reference = write(tmp_path / "ref.tif", scene)
    target = write(tmp_path / "tgt.tif", scene)
    rows, cols, sharpness, pixel, _ = measure_shift.measure(reference, target)
    assert (rows, cols) == (0, 0)
    assert pixel == PIXEL
    assert sharpness > 8


@pytest.mark.parametrize("row_offset,col_offset", [(4, 0), (-4, 0), (0, 5), (0, -5), (3, -2)])
def test_a_known_shift_is_recovered(tmp_path: Path, row_offset: int, col_offset: int):
    """Move the target's georeferencing north/south/east/west by a known amount.

    Shifting the geotransform by +row means the same imagery is now declared to
    be further south, so the tool should report the target as south.
    """
    scene = speckled_scene()
    reference = write(tmp_path / "ref.tif", scene)
    target = write(tmp_path / "tgt.tif", scene, row_offset=row_offset, col_offset=col_offset)
    rows, cols, sharpness, _, _ = measure_shift.measure(reference, target)
    assert (rows, cols) == (row_offset, col_offset)
    assert sharpness > 8


def test_describe_names_the_direction():
    assert measure_shift.describe(4, 0, 20.0) == "80 m south"
    assert measure_shift.describe(-4, 0, 20.0) == "80 m north"
    assert measure_shift.describe(0, 5, 20.0) == "100 m east"
    assert measure_shift.describe(0, -5, 20.0) == "100 m west"
    assert measure_shift.describe(2, -3, 20.0) == "40 m south and 60 m west"
    assert measure_shift.describe(0, 0, 20.0) == "no measurable offset"


def test_non_overlapping_rasters_are_an_error(tmp_path: Path):
    scene = speckled_scene(size=256)
    reference = write(tmp_path / "ref.tif", scene)
    target = write(tmp_path / "tgt.tif", scene, col_offset=100_000)
    with pytest.raises(measure_shift.MeasureError, match="do not overlap"):
        measure_shift.measure(reference, target)


def test_mostly_empty_patch_is_refused(tmp_path: Path):
    scene = speckled_scene(size=256)
    blank = np.full_like(scene, np.nan)
    blank[:20, :20] = scene[:20, :20]
    reference = write(tmp_path / "ref.tif", scene)
    target = write(tmp_path / "tgt.tif", blank)
    with pytest.raises(measure_shift.MeasureError, match="too little to correlate"):
        measure_shift.measure(reference, target)


def test_to_db_floors_zero_and_keeps_nan(tmp_path: Path):
    values = np.array([[1.0, 0.0, np.nan, 0.1]], dtype="float32")
    result = measure_shift.to_db(values)
    assert result[0, 0] == pytest.approx(0.0)
    assert np.isnan(result[0, 1]), "zero power is no data, not -100 dB"
    assert np.isnan(result[0, 2])
    assert result[0, 3] == pytest.approx(-10.0)


def test_resolve_band_passes_through_a_plain_raster(tmp_path: Path):
    image = tmp_path / "Gamma0_VH.img"
    image.touch()
    assert measure_shift.resolve_band(image) == image


def test_resolve_band_prefers_backscatter_over_a_mask(tmp_path: Path):
    dim = tmp_path / "scene.dim"
    dim.touch()
    data = tmp_path / "scene.data"
    data.mkdir()
    (data / "layover_shadow_mask.img").touch()
    (data / "Gamma0_VH.img").touch()
    assert measure_shift.resolve_band(dim).stem == "Gamma0_VH"


def test_resolve_band_honours_an_explicit_choice(tmp_path: Path):
    dim = tmp_path / "scene.dim"
    dim.touch()
    data = tmp_path / "scene.data"
    data.mkdir()
    (data / "Gamma0_VH.img").touch()
    (data / "Gamma0_VV.img").touch()
    assert measure_shift.resolve_band(dim, "Gamma0_VV").stem == "Gamma0_VV"


def test_resolve_band_lists_options_when_the_name_is_wrong(tmp_path: Path):
    dim = tmp_path / "scene.dim"
    dim.touch()
    data = tmp_path / "scene.data"
    data.mkdir()
    (data / "Gamma0_VH.img").touch()
    with pytest.raises(measure_shift.MeasureError, match="Gamma0_VH"):
        measure_shift.resolve_band(dim, "Sigma0_VV")


def test_unfinished_run_is_reported_clearly(tmp_path: Path):
    """A .dim with no .data folder is a crashed job."""
    dim = tmp_path / "scene.dim"
    dim.touch()
    with pytest.raises(measure_shift.MeasureError, match="did not finish"):
        measure_shift.resolve_band(dim)


def test_common_patch_avoids_an_empty_centre():
    """A GA NRB burst is a parallelogram in a north-up box, so the middle of the
    box is often entirely nodata."""
    reference = np.full((400, 400), np.nan, dtype="float32")
    target = np.full((400, 400), np.nan, dtype="float32")
    reference[10:110, 10:110] = 1.0
    target[10:110, 10:110] = 1.0
    ref_patch, tgt_patch = measure_shift.common_patch(reference, target, 64)
    assert np.isfinite(ref_patch).all()
    assert np.isfinite(tgt_patch).all()


def test_common_patch_cuts_both_from_the_same_window():
    """Centring each patch on its own valid data would cancel the very shift
    being measured."""
    reference = np.full((200, 200), np.nan, dtype="float32")
    target = np.full((200, 200), np.nan, dtype="float32")
    reference[20:120, 20:120] = 1.0
    target[40:140, 40:140] = 1.0  # valid region offset from the reference's
    ref_patch, tgt_patch = measure_shift.common_patch(reference, target, 32)
    # Same window means the two patches line up index for index.
    assert ref_patch.shape == tgt_patch.shape
    assert np.isfinite(ref_patch).all() and np.isfinite(tgt_patch).all()


def test_common_patch_refuses_disjoint_valid_regions():
    reference = np.full((200, 200), np.nan, dtype="float32")
    target = np.full((200, 200), np.nan, dtype="float32")
    reference[:50, :50] = 1.0
    target[150:, 150:] = 1.0
    with pytest.raises(measure_shift.MeasureError, match="never have data at the same pixel"):
        measure_shift.common_patch(reference, target, 32)


def test_common_patch_requires_a_common_grid():
    with pytest.raises(measure_shift.MeasureError, match="common grid"):
        measure_shift.common_patch(np.zeros((10, 10)), np.zeros((12, 12)), 4)


def test_shift_is_still_recovered_when_the_centre_is_empty(tmp_path: Path):
    """The real-data case: valid pixels off-centre, and a genuine offset."""
    scene = speckled_scene(size=768)
    framed = np.full((768, 768), np.nan, dtype="float32")
    framed[40:440, 40:440] = scene[40:440, 40:440]
    reference = write(tmp_path / "ref.tif", framed)
    target = write(tmp_path / "tgt.tif", framed, row_offset=3, col_offset=-2)
    rows, cols, sharpness, _, _ = measure_shift.measure(reference, target, patch=256)
    assert (rows, cols) == (3, -2)
    assert sharpness > 8
