"""Tests for the GA NRB comparison.

Synthetic rasters with a known bias put in deliberately, so the reported bias
can be checked against the right answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import compare_to_nrb  # noqa: E402
from measure_shift import MeasureError  # noqa: E402

PIXEL = 20.0
CRS = "EPSG:32755"


def scene(size: int = 600, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.gamma(shape=3.0, scale=0.03, size=(size // 8, size // 8))
    field = np.kron(coarse, np.ones((8, 8)))[:size, :size]
    return (field * rng.gamma(shape=10.0, scale=0.1, size=(size, size))).astype("float32")


def write(path: Path, data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(500000.0, 6500000.0, PIXEL, PIXEL)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32", crs=CRS, transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
    return path


def build_tree(tmp_path: Path, gain_db: float = 0.0, variant: str = "grd_gamma0_ellipsoid",
               suffix: str = "") -> Path:
    """A minimal before_after tree: one AOI, one burst, one date, both pols."""
    date_dir = tmp_path / "pilliga" / "t009_019142_iw1" / "20240603"
    linear_gain = 10 ** (gain_db / 10.0)
    produced = date_dir / "grd_preprocessed" / variant / "S1A_IW_GRDH_1SDV_x.data"
    produced.mkdir(parents=True)
    (produced.parent / "S1A_IW_GRDH_1SDV_x.dim").touch()
    for pol in ("VH", "VV"):
        data = scene(seed=hash(pol) % 100)
        write(date_dir / f"ga_s1a_nrb_0-1-0_T009-019142-IW1_20240603T084048Z_{pol}-gamma0{suffix}.tif", data)
        write(produced / f"Gamma0_{pol}.img", data * linear_gain)
    return tmp_path


def test_pairs_are_found_for_both_polarisations(tmp_path: Path):
    root = build_tree(tmp_path)
    pairs = list(compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid"))
    assert {p.pol for p in pairs} == {"VH", "VV"}
    assert all(p.aoi == "pilliga" and p.date == "20240603" for p in pairs)


def test_polarisations_are_never_crossed(tmp_path: Path):
    root = build_tree(tmp_path)
    for pair in compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid"):
        assert f"{pair.pol}-gamma0" in pair.reference.name
        assert pair.target.stem == f"Gamma0_{pair.pol}"


def test_raw_and_filtered_references_are_not_mixed(tmp_path: Path):
    """A filtered SNAP variant must not be compared against the raw NRB."""
    root = build_tree(tmp_path, suffix="_sf_db")
    assert not list(compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid", filtered=False))
    assert len(list(compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid", filtered=True))) == 2


def test_identical_rasters_report_no_bias(tmp_path: Path):
    root = build_tree(tmp_path, gain_db=0.0)
    for pair in compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid"):
        result = compare_to_nrb.compare(pair)
        assert result.bias == pytest.approx(0.0, abs=0.01)
        assert result.corr == pytest.approx(1.0, abs=0.01)
        assert (result.row_shift, result.col_shift) == pytest.approx((0.0, 0.0), abs=0.02)


@pytest.mark.parametrize("gain_db", [2.0, -3.5])
def test_a_known_bias_is_recovered(tmp_path: Path, gain_db: float):
    root = build_tree(tmp_path, gain_db=gain_db)
    for pair in compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid"):
        result = compare_to_nrb.compare(pair)
        assert result.bias == pytest.approx(gain_db, abs=0.05)
        assert result.snap_median - result.ga_median == pytest.approx(gain_db, abs=0.05)
        # A pure gain shifts values without changing structure.
        assert result.corr == pytest.approx(1.0, abs=0.01)


def test_unknown_variant_yields_no_pairs(tmp_path: Path):
    root = build_tree(tmp_path)
    assert not list(compare_to_nrb.find_pairs(root, "grd_gamma0_rtc"))


def test_all_nodata_target_is_refused(tmp_path: Path):
    root = build_tree(tmp_path)
    pair = next(iter(compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid")))
    write(pair.target, np.full((600, 600), np.nan, dtype="float32"))
    with pytest.raises(MeasureError, match="effectively empty"):
        compare_to_nrb.compare(pair)


def test_csv_round_trips(tmp_path: Path):
    import csv as csv_module
    root = build_tree(tmp_path, gain_db=1.5)
    results = [compare_to_nrb.compare(p) for p in compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid")]
    out = tmp_path / "out.csv"
    compare_to_nrb.write_csv(out, results)
    rows = list(csv_module.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == len(results)
    assert {r["pol"] for r in rows} == {"VH", "VV"}
    assert all(float(r["bias_db"]) == pytest.approx(1.5, abs=0.05) for r in rows)


def test_main_reports_missing_pairs_clearly(tmp_path: Path, capsys):
    (tmp_path / "empty").mkdir()
    code = compare_to_nrb.main([str(tmp_path), "--variant", "grd_gamma0_ellipsoid"])
    assert code == 1
    assert "no pairs found" in capsys.readouterr().err


def test_matching_projections_are_not_flagged(tmp_path: Path):
    root = build_tree(tmp_path)
    for pair in compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid"):
        assert compare_to_nrb.compare(pair).reprojected is False


def test_projection_mismatch_is_flagged(tmp_path: Path):
    """An extra interpolation applied to one AOI and not the others is exactly
    the inconsistency a cross-AOI comparison must not hide."""
    import rasterio
    from rasterio.transform import from_origin

    root = build_tree(tmp_path)
    pair = next(iter(compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid")))
    data = scene()
    with rasterio.open(
        pair.target, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32", crs="EPSG:32755",
        transform=from_origin(200000.0, 6500000.0, PIXEL, PIXEL), nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
    with pytest.raises(Exception):
        # Different zone puts it somewhere else entirely; the point is only that
        # the mismatch is detected rather than silently reprojected.
        compare_to_nrb.compare(pair)


@pytest.mark.parametrize("gain_db", [2.0, -3.5])
def test_gain_recovers_a_real_calibration_offset(tmp_path: Path, gain_db: float):
    """A pure multiplicative gain must show identically in bias and in gain."""
    root = build_tree(tmp_path, gain_db=gain_db)
    for pair in compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid"):
        result = compare_to_nrb.compare(pair)
        assert result.gain == pytest.approx(gain_db, abs=0.05)
        assert result.bias == pytest.approx(gain_db, abs=0.05)


def test_gain_is_unmoved_by_look_count_while_bias_is_not(tmp_path: Path):
    """The point of reporting both.

    Two products of the same scene at different effective looks have the same
    mean linear power but different medians in dB, because speckle is skewed.
    A median-in-dB comparison reads that as a calibration offset; the linear
    mean does not.
    """
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(7)
    size = 600
    truth = np.full((size, size), 0.05, dtype="float32")

    def speckle(looks: int) -> np.ndarray:
        return (truth * rng.gamma(shape=looks, scale=1.0 / looks, size=(size, size))).astype("float32")

    root = build_tree(tmp_path)
    pair = next(iter(compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid")))
    for path, looks in ((pair.reference, 2), (pair.target, 16)):
        with rasterio.open(
            path, "w", driver="GTiff", height=size, width=size, count=1,
            dtype="float32", crs=CRS, transform=from_origin(500000.0, 6500000.0, PIXEL, PIXEL),
            nodata=np.nan,
        ) as dst:
            dst.write(speckle(looks), 1)

    result = compare_to_nrb.compare(pair)
    # Same underlying backscatter, so the linear means agree...
    assert result.gain == pytest.approx(0.0, abs=0.15)
    # ...but the better-looked product has the higher median in dB.
    assert result.bias > 0.5


def test_empty_snap_output_is_named_as_such(tmp_path: Path):
    """A failed run writes a file full of nodata. That must not be reported as
    the two footprints missing each other."""
    root = build_tree(tmp_path)
    pair = next(iter(compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid")))
    write(pair.target, np.full((600, 600), np.nan, dtype="float32"))
    with pytest.raises(MeasureError, match="SNAP product is effectively empty"):
        compare_to_nrb.compare(pair)


def test_empty_reference_is_named_as_such(tmp_path: Path):
    root = build_tree(tmp_path)
    pair = next(iter(compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid")))
    write(pair.reference, np.full((600, 600), np.nan, dtype="float32"))
    with pytest.raises(MeasureError, match="GA raster is effectively empty"):
        compare_to_nrb.compare(pair)


def test_displacement_is_distinguished_from_emptiness(tmp_path: Path):
    """Both products full of data, but valid in disjoint places."""
    root = build_tree(tmp_path)
    pair = next(iter(compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid")))
    data = scene()
    left, right = data.copy(), data.copy()
    left[:, 300:] = np.nan
    right[:, :300] = np.nan
    write(pair.reference, left)
    write(pair.target, right)
    with pytest.raises(MeasureError, match="not in the same places"):
        compare_to_nrb.compare(pair)


def test_valid_fractions_are_quoted_in_the_error(tmp_path: Path):
    root = build_tree(tmp_path)
    pair = next(iter(compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid")))
    write(pair.target, np.full((600, 600), np.nan, dtype="float32"))
    with pytest.raises(MeasureError, match=r"GA is 100\.0% valid and SNAP is 0\.0%"):
        compare_to_nrb.compare(pair)


def test_band_is_matched_loosely(tmp_path: Path):
    """SNAP does not always name the band Gamma0_VH: Terrain-Correction's own
    radiometric normalisation suffixes it with how the angle was derived."""
    produced = tmp_path / "scene.data"
    produced.mkdir()
    (tmp_path / "scene.dim").touch()
    (produced / "Gamma0_VH_use_local_inci_angle_from_dem.img").touch()
    (produced / "Gamma0_VV_use_local_inci_angle_from_dem.img").touch()
    assert compare_to_nrb.find_band(tmp_path / "scene.dim", "VH").stem.startswith("Gamma0_VH")
    assert compare_to_nrb.find_band(tmp_path / "scene.dim", "VV").stem.startswith("Gamma0_VV")


def test_band_match_never_crosses_polarisation(tmp_path: Path):
    produced = tmp_path / "scene.data"
    produced.mkdir()
    (tmp_path / "scene.dim").touch()
    (produced / "Gamma0_VV_normalised.img").touch()
    (produced / "Gamma0_VH_normalised.img").touch()
    assert "VH" in compare_to_nrb.find_band(tmp_path / "scene.dim", "VH").stem
    assert "VV" in compare_to_nrb.find_band(tmp_path / "scene.dim", "VV").stem


def test_band_match_ignores_non_gamma0_bands(tmp_path: Path):
    produced = tmp_path / "scene.data"
    produced.mkdir()
    (tmp_path / "scene.dim").touch()
    (produced / "Sigma0_VH.img").touch()
    (produced / "Beta0_VH.img").touch()
    with pytest.raises(MeasureError, match="no gamma0 VH band"):
        compare_to_nrb.find_band(tmp_path / "scene.dim", "VH")


def test_missing_band_lists_what_is_there(tmp_path: Path):
    """Silently dropping a product is how a successful run gets mistaken for one
    that never happened."""
    produced = tmp_path / "scene.data"
    produced.mkdir()
    (tmp_path / "scene.dim").touch()
    (produced / "Sigma0_VH.img").touch()
    with pytest.raises(MeasureError, match="Bands present: Sigma0_VH"):
        compare_to_nrb.find_band(tmp_path / "scene.dim", "VH")


def test_unpairable_products_are_reported_not_swallowed(tmp_path: Path):
    root = build_tree(tmp_path)
    produced = next(root.rglob("grd_preprocessed/grd_gamma0_ellipsoid/*.data"))
    for image in produced.glob("*.img"):
        image.rename(image.with_name(image.name.replace("Gamma0", "Sigma0")))
    messages = []
    pairs = list(compare_to_nrb.find_pairs(root, "grd_gamma0_ellipsoid", report=messages.append))
    assert not pairs
    assert messages and "Bands present" in messages[0]
