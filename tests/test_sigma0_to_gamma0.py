"""Tests for deriving gamma0 from sigma0 and the projected local incidence angle."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import sigma0_to_gamma0 as converter  # noqa: E402

CRS = "EPSG:32756"


def write_band(path: Path, data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="ENVI", height=data.shape[0], width=data.shape[1], count=1,
        dtype="float32", crs=CRS, transform=from_origin(500000.0, 6500000.0, 20, 20),
        nodata=np.nan,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def build_product(tmp_path: Path, angle_degrees=30.0, mask_value=0.0, size=32) -> Path:
    product = tmp_path / "scene.dim"
    product.touch()
    data = tmp_path / "scene.data"
    write_band(data / "Sigma0_VH.img", np.full((size, size), 0.05))
    write_band(data / "Sigma0_VV.img", np.full((size, size), 0.20))
    write_band(data / "projectedLocalIncidenceAngle.img", np.full((size, size), angle_degrees))
    write_band(data / "layoverShadowMask.img", np.full((size, size), mask_value))
    return product


def test_conversion_matches_the_definition():
    sigma0 = np.array([[0.05, 0.20]], dtype="float32")
    angle = np.array([[30.0, 60.0]], dtype="float32")
    gamma0, dropped = converter.convert(sigma0, angle)
    assert gamma0[0, 0] == pytest.approx(0.05 / math.cos(math.radians(30)), rel=1e-5)
    assert gamma0[0, 1] == pytest.approx(0.20 / math.cos(math.radians(60)), rel=1e-5)
    assert dropped == 0.0


def test_gamma0_is_never_below_sigma0():
    """Dividing by a cosine can only raise the value."""
    rng = np.random.default_rng(0)
    sigma0 = rng.uniform(0.01, 0.5, size=(64, 64)).astype("float32")
    angle = rng.uniform(0.0, 70.0, size=(64, 64)).astype("float32")
    gamma0, _ = converter.convert(sigma0, angle)
    assert np.all(gamma0[np.isfinite(gamma0)] >= sigma0[np.isfinite(gamma0)] - 1e-6)


def test_steep_angles_are_dropped_not_amplified():
    """As the angle approaches 90 the quotient explodes. GA caps the equivalent
    quantity with rtc_min_value_db; this drops the pixel."""
    sigma0 = np.array([[0.05, 0.05]], dtype="float32")
    angle = np.array([[30.0, 89.5]], dtype="float32")
    gamma0, dropped = converter.convert(sigma0, angle, max_angle=80.0)
    assert np.isfinite(gamma0[0, 0])
    assert np.isnan(gamma0[0, 1])
    assert dropped == pytest.approx(0.5)


def test_layover_and_shadow_are_dropped():
    sigma0 = np.full((1, 4), 0.05, dtype="float32")
    angle = np.full((1, 4), 30.0, dtype="float32")
    mask = np.array([[0, 1, 2, 3]], dtype="float32")  # clear, layover, shadow, both
    gamma0, dropped = converter.convert(sigma0, angle, mask)
    assert np.isfinite(gamma0[0, 0])
    assert np.isnan(gamma0[0, 1:]).all()
    assert dropped == pytest.approx(0.75)


def test_nodata_in_sigma0_stays_nodata():
    sigma0 = np.array([[0.05, np.nan]], dtype="float32")
    angle = np.array([[30.0, 30.0]], dtype="float32")
    gamma0, _ = converter.convert(sigma0, angle)
    assert np.isfinite(gamma0[0, 0])
    assert np.isnan(gamma0[0, 1])


def test_product_conversion_writes_both_polarisations(tmp_path: Path):
    product = build_product(tmp_path)
    written = converter.convert_product(product)
    assert {pol for pol, _, _ in written} == {"VH", "VV"}
    for _, target, _ in written:
        assert target.exists()


def test_written_band_is_readable_and_correct(tmp_path: Path):
    product = build_product(tmp_path, angle_degrees=45.0)
    converter.convert_product(product)
    with rasterio.open(tmp_path / "scene.data" / "Gamma0_VH.img") as src:
        values = src.read(1)
    assert values[0, 0] == pytest.approx(0.05 / math.cos(math.radians(45)), rel=1e-4)


def test_written_band_keeps_the_georeferencing(tmp_path: Path):
    product = build_product(tmp_path)
    converter.convert_product(product)
    with rasterio.open(tmp_path / "scene.data" / "Sigma0_VH.img") as source, \
         rasterio.open(tmp_path / "scene.data" / "Gamma0_VH.img") as result:
        assert result.crs == source.crs
        assert result.transform == source.transform
        assert result.shape == source.shape


def test_existing_bands_are_not_silently_redone(tmp_path: Path):
    product = build_product(tmp_path)
    converter.convert_product(product)
    marker = tmp_path / "scene.data" / "Gamma0_VH.img"
    before = marker.stat().st_mtime_ns
    written = converter.convert_product(product)
    assert marker.stat().st_mtime_ns == before
    assert any(dropped is None for _, _, dropped in written)


def test_a_product_without_sigma0_is_skipped_not_failed(tmp_path: Path):
    """Other variants' outputs sit in sibling folders under the same search root
    and carry the same filename. They are not candidates, not failures."""
    product = build_product(tmp_path)
    for pol in ("VH", "VV"):
        (tmp_path / "scene.data" / f"Sigma0_{pol}.img").unlink()
    assert converter.convert_product(product) is None


def test_sigma0_without_an_angle_band_is_still_an_error(tmp_path: Path):
    product = build_product(tmp_path)
    (tmp_path / "scene.data" / "projectedLocalIncidenceAngle.img").unlink()
    with pytest.raises(converter.ConversionError, match="no projectedLocalIncidenceAngle"):
        converter.convert_product(product)


def test_messages_name_the_variant_folder(tmp_path: Path):
    """Every variant holds a product of the same name, so the bare filename is
    ambiguous."""
    product = tmp_path / "grd_gamma0_tcnorm" / "scene.dim"
    product.parent.mkdir()
    build = build_product(product.parent)
    assert converter.label(build) == "grd_gamma0_tcnorm/scene.dim"


def test_converted_product_can_then_be_paired(tmp_path: Path):
    """The point of writing into .data: compare_to_nrb must find the result."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import compare_to_nrb

    product = build_product(tmp_path)
    converter.convert_product(product)
    assert compare_to_nrb.find_band(product, "VH").stem == "Gamma0_VH"
    assert compare_to_nrb.find_band(product, "VV").stem == "Gamma0_VV"
