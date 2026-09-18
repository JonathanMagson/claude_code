#!/usr/bin/env python3
"""Measure the geolocation offset between a SNAP product and a GA NRB raster.

"The scene looks slightly south" is not something you can act on. This reports
the offset in pixels and metres, and which way, by phase-correlating the two
images over the area they share.

Direction matters for diagnosis. Sentinel-1 flies a near-polar orbit, so:

  * an ALONG-TRACK (roughly north-south) shift comes from orbit timing or the
    geocoding, not from the DEM;
  * a CROSS-TRACK (roughly east-west) shift is what a DEM height error causes,
    since height error displaces a pixel in slant range.

So a north-south offset largely exonerates terrain flattening and the DEM, and
points at Apply-Orbit-File or Terrain-Correction instead.

The reference is the GA NRB GeoTIFF; the target is a SNAP output. For BEAM-DIMAP
pass either the ``.dim`` or the band ``.img`` inside the ``.data`` folder -- the
``.img`` files are plain ENVI rasters that GDAL reads directly.

Example
-------
    python tools/measure_shift.py \\
        --reference .../ga_s1a_nrb_..._VH-gamma0.tif \\
        --target    .../grd_gamma0_rtc/S1A_...dim
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject, transform_bounds

DEFAULT_PATCH = 512
EPSILON = 1e-10


class MeasureError(RuntimeError):
    """Raised when the comparison cannot be made at all."""


def resolve_band(path: Path, band: Optional[str] = None) -> Path:
    """Turn a BEAM-DIMAP .dim into the .img holding a band; pass anything else through."""
    if path.suffix.lower() != ".dim":
        return path
    data = path.with_suffix(".data")
    if not data.is_dir():
        raise MeasureError(f"{path.name} has no .data folder -- the run did not finish")
    images = sorted(data.glob("*.img"))
    if not images:
        raise MeasureError(f"no .img bands in {data}")
    if band:
        for image in images:
            if image.stem.lower() == band.lower():
                return image
        raise MeasureError(
            f"no band {band!r} in {data.name}. Available: "
            + ", ".join(i.stem for i in images)
        )
    # Prefer a real backscatter band over a mask when the caller did not choose.
    backscatter = [i for i in images if i.stem.lower().startswith(("gamma0", "sigma0", "beta0"))]
    return (backscatter or images)[0]


def to_db(values: np.ndarray) -> np.ndarray:
    """dB, so the correlation is driven by structure rather than by bright outliers."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 10.0 * np.log10(np.clip(values, EPSILON, None))
    return np.where(np.isfinite(values) & (values > 0), result, np.nan).astype("float32")


def overlap_window(reference, target_bounds) -> Tuple[int, int, int, int]:
    """Row/col window of the reference covering both rasters, as (row0, col0, rows, cols)."""
    left = max(reference.bounds.left, target_bounds[0])
    bottom = max(reference.bounds.bottom, target_bounds[1])
    right = min(reference.bounds.right, target_bounds[2])
    top = min(reference.bounds.top, target_bounds[3])
    if left >= right or bottom >= top:
        raise MeasureError(
            "the two rasters do not overlap. Check they are the same burst/scene "
            "and the same CRS."
        )
    row0, col0 = reference.index(left, top)
    row1, col1 = reference.index(right, bottom)
    return max(row0, 0), max(col0, 0), max(row1 - row0, 0), max(col1 - col0, 0)


def centred_patch(array: np.ndarray, size: int) -> np.ndarray:
    """The most central size x size block, or the whole array if it is smaller."""
    rows, cols = array.shape
    size = min(size, rows, cols)
    row0 = (rows - size) // 2
    col0 = (cols - size) // 2
    return array[row0:row0 + size, col0:col0 + size]


def common_patch(reference: np.ndarray, target: np.ndarray, size: int):
    """Matching size x size blocks cut where BOTH rasters have data.

    The geometric centre of the overlap is often empty: a GA NRB burst is a
    parallelogram inside a north-up bounding box, so the middle of that box can
    be entirely nodata.

    Both patches must come from the SAME window. Centring each on its own valid
    data would move them independently and cancel out exactly the displacement
    being measured.
    """
    if reference.shape != target.shape:
        raise MeasureError("patches must be cut from rasters on a common grid")
    usable = np.isfinite(reference) & np.isfinite(target)
    if not usable.any():
        raise MeasureError(
            "the two rasters never have data at the same pixel; nothing to correlate"
        )
    rows, cols = reference.shape
    size = min(size, rows, cols)
    row_centre, col_centre = (int(round(axis.mean())) for axis in np.nonzero(usable))
    row0 = int(np.clip(row_centre - size // 2, 0, rows - size))
    col0 = int(np.clip(col_centre - size // 2, 0, cols - size))
    window = (slice(row0, row0 + size), slice(col0, col0 + size))
    return reference[window], target[window]


def prepare(patch: np.ndarray) -> np.ndarray:
    """Mean-remove, zero-fill gaps and window, ready for phase correlation."""
    finite = np.isfinite(patch)
    if finite.sum() < 0.2 * patch.size:
        raise MeasureError(
            f"only {100 * finite.mean():.0f}% of the compared patch has valid data; "
            "too little to correlate. Try --patch with a larger window."
        )
    centred = np.where(finite, patch - np.nanmean(patch), 0.0)
    rows, cols = centred.shape
    window = np.hanning(rows)[:, None] * np.hanning(cols)[None, :]
    return centred * window


def phase_correlate(reference: np.ndarray, target: np.ndarray) -> Tuple[int, int, float]:
    """Integer (row, col) shift of *target* relative to *reference*, and peak sharpness."""
    # fft(target) * conj(fft(reference)) peaks at the TARGET's displacement.
    # The other order gives the same magnitude with the sign flipped.
    spectrum = np.fft.fft2(target) * np.conj(np.fft.fft2(reference))
    spectrum /= np.abs(spectrum) + EPSILON
    surface = np.fft.ifft2(spectrum).real
    peak = np.unravel_index(int(np.argmax(surface)), surface.shape)
    rows, cols = surface.shape
    # The FFT wraps, so a peak past halfway is a negative shift.
    row_shift = peak[0] - rows if peak[0] > rows // 2 else peak[0]
    col_shift = peak[1] - cols if peak[1] > cols // 2 else peak[1]
    sharpness = float(surface[peak] / (surface.std() + EPSILON))
    return int(row_shift), int(col_shift), sharpness


def describe(row_shift: int, col_shift: int, pixel: float) -> str:
    """Plain words for which way the target sits relative to the reference."""
    parts = []
    if row_shift:
        # Row increases downwards, i.e. southwards, in a north-up raster.
        parts.append(f"{abs(row_shift * pixel):.0f} m {'south' if row_shift > 0 else 'north'}")
    if col_shift:
        parts.append(f"{abs(col_shift * pixel):.0f} m {'east' if col_shift > 0 else 'west'}")
    return " and ".join(parts) if parts else "no measurable offset"


def align(reference_path: Path, target_path: Path):
    """Put *target* on *reference*'s exact grid over the area they share.

    Returns (reference_array, target_array, pixel_size, (cols, rows)). Both
    arrays are linear power with NaN for no data, on identical grids, so a
    difference between them is a real difference and not a gridding artefact.
    """
    with rasterio.open(reference_path) as reference, rasterio.open(target_path) as target:
        row0, col0, rows, cols = overlap_window(
            reference, transform_bounds(target.crs, reference.crs, *target.bounds)
        )
        if rows < 32 or cols < 32:
            raise MeasureError(f"overlap is only {cols}x{rows} pixels -- too small to compare")

        window = Window(col0, row0, cols, rows)
        reference_data = reference.read(1, window=window, masked=True).filled(np.nan)
        reference_transform = reference.window_transform(window)

        resampled = np.full((rows, cols), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(target, 1),
            destination=resampled,
            dst_transform=reference_transform,
            dst_crs=reference.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        pixel = abs(reference_transform.a)
    return reference_data, resampled, pixel, (cols, rows)


def measure(reference_path: Path, target_path: Path, patch: int = DEFAULT_PATCH):
    reference_data, resampled, pixel, shape = align(reference_path, target_path)
    reference_patch, target_patch = common_patch(
        to_db(reference_data), to_db(resampled), patch
    )
    row_shift, col_shift, sharpness = phase_correlate(
        prepare(reference_patch), prepare(target_patch)
    )
    return row_shift, col_shift, sharpness, pixel, shape


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", required=True, help="GA NRB GeoTIFF")
    parser.add_argument("--target", required=True, help="SNAP .dim or band .img")
    parser.add_argument("--band", help="band name inside a .dim, e.g. Gamma0_VH")
    parser.add_argument("--patch", type=int, default=DEFAULT_PATCH,
                        help=f"size of the correlated window (default {DEFAULT_PATCH})")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        target = resolve_band(Path(args.target), args.band)
        row_shift, col_shift, sharpness, pixel, shape = measure(
            Path(args.reference), target, args.patch
        )
    except MeasureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"reference : {Path(args.reference).name}")
    print(f"target    : {target.name}")
    print(f"overlap   : {shape[0]} x {shape[1]} px at {pixel:.0f} m")
    print(f"offset    : {row_shift:+d} rows, {col_shift:+d} cols")
    print(f"            target sits {describe(row_shift, col_shift, pixel)} of the reference")
    print(f"peak      : {sharpness:.1f}x noise", end="")
    if sharpness < 8:
        print("  <- weak; treat the offset as unreliable")
    else:
        print()

    if abs(row_shift) > abs(col_shift) and abs(row_shift) > 1:
        print("\nMostly along-track (north-south). That is orbit timing or terrain")
        print("correction, NOT the DEM or terrain flattening -- a DEM height error")
        print("displaces pixels across-track.")
    elif abs(col_shift) > 1:
        print("\nMostly across-track (east-west), which is the direction a DEM height")
        print("error displaces pixels. Suspect the DEM or terrain flattening.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
