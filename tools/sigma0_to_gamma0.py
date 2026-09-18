#!/usr/bin/env python3
"""Derive gamma0 from a terrain-corrected sigma0 product and its incidence angle.

SNAP's Range-Doppler Terrain Correction, run with radiometric normalisation,
writes ``Sigma0_*`` bands alongside ``projectedLocalIncidenceAngle`` even when
asked for gamma0. The conversion it withheld is a definition::

    gamma0 = sigma0 / cos(local incidence angle)

Using the *projected local* incidence angle -- the one the DEM gives, not the
ellipsoid one -- makes this a terrain normalisation rather than a flat-earth
one. It is a weaker correction than true area-projection RTC, which integrates
the illuminated area rather than taking a cosine, but it is the same intent.

Two guards, because the division is unstable where the geometry is:

* Pixels the product flags as layover or shadow are dropped. The backscatter
  there does not belong to the pixel it landed in, whatever the maths says.
* The correction is capped. As the local incidence angle approaches 90 degrees
  the cosine approaches zero and the quotient explodes; GA's own pipeline caps
  the equivalent quantity with ``rtc_min_value_db: -30``.

Writes ``Gamma0_<pol>.img`` into the product's ``.data`` folder, where
``compare_to_nrb.py`` will find it. It does NOT update the ``.dim`` header, so
SNAP itself will not list the new bands; GDAL-based tools read them fine.

Example
-------
    python tools/sigma0_to_gamma0.py D:/scratch/.../grd_gamma0_tcnorm
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio

DEFAULT_MAX_ANGLE = 80.0
ANGLE_BAND = "projectedlocalincidenceangle"
MASK_BAND = "layovershadowmask"


class ConversionError(RuntimeError):
    """Raised when a product lacks what the conversion needs."""


def band_path(data: Path, name: str) -> Optional[Path]:
    for image in sorted(data.glob("*.img")):
        if image.stem.lower() == name:
            return image
    return None


def sigma0_bands(data: Path) -> dict:
    """Polarisation -> sigma0 image, for every sigma0 band in the product."""
    found = {}
    for image in sorted(data.glob("*.img")):
        stem = image.stem.lower()
        if not stem.startswith("sigma0"):
            continue
        for pol in ("VH", "VV", "HH", "HV"):
            if pol.lower() in stem:
                found[pol] = image
                break
    return found


def read(path: Path):
    with rasterio.open(path) as src:
        return src.read(1, masked=True).filled(np.nan).astype("float32"), src.profile


def convert(
    sigma0: np.ndarray,
    angle_degrees: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_angle: float = DEFAULT_MAX_ANGLE,
):
    """gamma0 from sigma0 and the projected local incidence angle.

    Returns (gamma0, dropped_fraction).
    """
    cosine = np.cos(np.radians(angle_degrees))
    floor = math.cos(math.radians(max_angle))
    unusable = ~np.isfinite(cosine) | (cosine < floor)
    if mask is not None:
        # SNAP's layover/shadow mask: 0 is clear, anything else is not.
        unusable |= np.nan_to_num(mask, nan=1.0) != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma0 = sigma0 / np.where(unusable, np.nan, cosine)
    gamma0 = np.where(np.isfinite(gamma0), gamma0, np.nan).astype("float32")
    valid_before = np.isfinite(sigma0)
    dropped = float((valid_before & ~np.isfinite(gamma0)).sum() / max(valid_before.sum(), 1))
    return gamma0, dropped


def convert_product(product: Path, max_angle: float = DEFAULT_MAX_ANGLE,
                    overwrite: bool = False) -> list:
    data = product.with_suffix(".data")
    if not data.is_dir():
        raise ConversionError(f"{product.name} has no .data folder")
    angle = band_path(data, ANGLE_BAND)
    if angle is None:
        raise ConversionError(
            f"{product.name} has no projectedLocalIncidenceAngle band. Terrain "
            "correction must be run with saveProjectedLocalIncidenceAngle enabled."
        )
    bands = sigma0_bands(data)
    if not bands:
        raise ConversionError(f"{product.name} has no sigma0 bands")

    angle_degrees, _ = read(angle)
    mask_file = band_path(data, MASK_BAND)
    mask = read(mask_file)[0] if mask_file else None

    written = []
    for pol, source in sorted(bands.items()):
        target = data / f"Gamma0_{pol}.img"
        if target.exists() and not overwrite:
            written.append((pol, target, None))
            continue
        sigma0, profile = read(source)
        gamma0, dropped = convert(sigma0, angle_degrees, mask, max_angle)
        profile.update(driver="ENVI", dtype="float32", count=1, nodata=np.nan)
        with rasterio.open(target, "w", **profile) as dst:
            dst.write(gamma0, 1)
        written.append((pol, target, dropped))
    return written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", help="a .dim product, or a folder to search for them")
    parser.add_argument("--max-angle", type=float, default=DEFAULT_MAX_ANGLE,
                        help=f"drop pixels steeper than this local incidence angle "
                             f"(default {DEFAULT_MAX_ANGLE} deg)")
    parser.add_argument("--overwrite", action="store_true", help="redo bands already written")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    products = [root] if root.suffix.lower() == ".dim" else sorted(root.rglob("*.dim"))
    if not products:
        print(f"no .dim products under {root}", file=sys.stderr)
        return 1

    failures = 0
    for product in products:
        try:
            written = convert_product(product, args.max_angle, args.overwrite)
        except ConversionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(product.name)
        for pol, target, dropped in written:
            if dropped is None:
                print(f"  {target.name}  (already present, skipped)")
            else:
                print(f"  {target.name}  dropped {dropped:.1%} to layover/shadow "
                      f"and angles > {args.max_angle:.0f} deg")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
