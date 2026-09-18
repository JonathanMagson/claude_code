#!/usr/bin/env python3
"""Compare SNAP-processed GRD output against the GA Sentinel-1 NRB.

Walks the before/after tree, pairs each GA NRB raster with the SNAP product for
the same scene and polarisation, puts both on the GA grid over the area they
share, and reports how far apart they are radiometrically and geometrically.

Reported per pair, in dB over the valid overlap:

  bias    median(SNAP - GA) in dB. Sensitive to how many looks each product
          carries: speckle is skewed, so a product with more looks has a higher
          median in dB even at identical true backscatter. Do not read a small
          bias as a calibration difference without checking `gain`.
  gain    10*log10(mean linear SNAP / mean linear GA). The same offset measured
          on linear power, where the mean is unbiased by look count. This is the
          number to quote for calibration agreement.
  rmse    spread of the difference in dB. Where a missing terrain correction
          shows up: RTC redistributes energy per pixel rather than shifting the
          scene mean, so it inflates the spread while leaving bias near zero.
  corr    Pearson correlation of the two dB images. Structure agreement.
  shift   geolocation offset in whole pixels, from phase correlation.

Compare like with like. The GA NRB is delivered UNFILTERED, so a SNAP variant
carrying a speckle filter belongs against the ``_sf_db`` rasters that
``postprocess_nrb.py`` writes, not against the raw product.

Example
-------
    python tools/compare_to_nrb.py D:/scratch/.../before_after \\
        --variant grd_gamma0_ellipsoid --csv comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_shift import (  # noqa: E402
    MeasureError,
    align,
    common_patch,
    phase_correlate,
    prepare,
    resolve_band,
    to_db,
)

OUTPUT_DIRNAME = "grd_preprocessed"
# ga_s1a_nrb_0-1-0_T009-019142-IW1_20240603T084048Z_VH-gamma0.tif
GA_RASTER = re.compile(r"_(?P<pol>VH|VV)-gamma0(?P<suffix>|_sf_db)\.tif$", re.IGNORECASE)


class Pair(NamedTuple):
    aoi: str
    date: str
    pol: str
    reference: Path
    target: Path


class Result(NamedTuple):
    pair: Pair
    valid: int
    coverage: float
    ga_median: float
    snap_median: float
    bias: float
    gain: float
    rmse: float
    corr: float
    row_shift: int
    col_shift: int
    reprojected: bool


def find_pairs(root: Path, variant: str, filtered: bool = False) -> Iterator[Pair]:
    """Pair every GA NRB raster with the SNAP band for the same scene and pol.

    The SNAP output lives in ``grd_preprocessed/<variant>/`` beside the GA
    rasters, so a pair is always within one date folder.
    """
    wanted_suffix = "_sf_db" if filtered else ""
    for produced in sorted(root.rglob(f"{OUTPUT_DIRNAME}/{variant}/*.dim")):
        date_dir = produced.parent.parent.parent
        for reference in sorted(date_dir.glob("*.tif")):
            match = GA_RASTER.search(reference.name)
            if not match or match.group("suffix").lower() != wanted_suffix:
                continue
            pol = match.group("pol").upper()
            try:
                target = resolve_band(produced, f"Gamma0_{pol}")
            except MeasureError:
                continue
            yield Pair(
                aoi=date_dir.parent.parent.name,
                date=date_dir.name,
                pol=pol,
                reference=reference,
                target=target,
            )


def compare(pair: Pair, patch: int = 512) -> Result:
    reference, target, _, shape, reprojected = align(pair.reference, pair.target)
    ga = to_db(reference)
    snap = to_db(target)

    usable = np.isfinite(ga) & np.isfinite(snap)
    count = int(usable.sum())
    coverage = count / ga.size if ga.size else 0.0
    if count < 1000:
        raise MeasureError(
            f"only {count} pixels valid in both rasters; nothing to compare. "
            "The GA burst and the GRD slice may barely overlap."
        )

    difference = snap[usable] - ga[usable]
    bias = float(np.median(difference))
    # Same comparison on linear power. The mean of linear power does not move
    # with look count, so this separates a real calibration offset from the
    # median-in-dB artefact that differing looks produce.
    ga_power = np.asarray(reference)[usable]
    snap_power = np.asarray(target)[usable]
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = float(10.0 * np.log10(np.mean(snap_power) / np.mean(ga_power)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    corr = float(np.corrcoef(ga[usable], snap[usable])[0, 1])

    try:
        ga_patch, snap_patch = common_patch(ga, snap, patch)
        row_shift, col_shift, _ = phase_correlate(prepare(ga_patch), prepare(snap_patch))
    except MeasureError:
        row_shift = col_shift = 0  # too little valid data in the centre to correlate

    return Result(
        pair=pair,
        valid=count,
        coverage=coverage,
        ga_median=float(np.median(ga[usable])),
        snap_median=float(np.median(snap[usable])),
        bias=bias,
        gain=gain,
        rmse=rmse,
        corr=corr,
        row_shift=row_shift,
        col_shift=col_shift,
        reprojected=reprojected,
    )


def write_csv(path: Path, results: Sequence[Result]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "aoi", "date", "pol", "valid_px", "coverage",
            "ga_median_db", "snap_median_db", "bias_db", "gain_db", "rmse_db", "corr",
            "row_shift_px", "col_shift_px", "reference", "target",
        ])
        for r in results:
            writer.writerow([
                r.pair.aoi, r.pair.date, r.pair.pol, r.valid, f"{r.coverage:.4f}",
                f"{r.ga_median:.2f}", f"{r.snap_median:.2f}", f"{r.bias:.2f}",
                f"{r.gain:.2f}", f"{r.rmse:.2f}", f"{r.corr:.3f}", r.row_shift, r.col_shift,
                r.pair.reference.name, r.pair.target.name,
            ])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", help="folder holding the AOI sub-folders")
    parser.add_argument("--variant", required=True, help="which grd_preprocessed sub-folder to compare")
    parser.add_argument(
        "--filtered", action="store_true",
        help="compare against the _sf_db NRB rasters instead of the raw ones "
             "(use when the SNAP variant applies a speckle filter)",
    )
    parser.add_argument("--csv", help="also write the table to this file")
    parser.add_argument("--patch", type=int, default=512, help="window used for the shift estimate")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    pairs = list(find_pairs(root, args.variant, args.filtered))
    if not pairs:
        print(
            f"no pairs found for variant {args.variant!r} under {root}.\n"
            f"Expected SNAP output in <date>/{OUTPUT_DIRNAME}/{args.variant}/*.dim "
            f"beside the GA {'_sf_db ' if args.filtered else ''}rasters.",
            file=sys.stderr,
        )
        return 1

    print(f"{'AOI':10} {'DATE':10} {'POL':4} {'COVER':>6} {'GA dB':>7} {'SNAP dB':>8} "
          f"{'BIAS':>7} {'GAIN':>7} {'RMSE':>6} {'CORR':>6} {'SHIFT px':>9}")
    results = []
    for pair in pairs:
        try:
            result = compare(pair, args.patch)
        except MeasureError as exc:
            print(f"{pair.aoi:10} {pair.date:10} {pair.pol:4} skipped: {exc}")
            continue
        results.append(result)
        print(
            f"{result.pair.aoi:10} {result.pair.date:10} {result.pair.pol:4} "
            f"{result.coverage:>5.1%} {result.ga_median:>7.2f} {result.snap_median:>8.2f} "
            f"{result.bias:>+7.2f} {result.gain:>+7.2f} {result.rmse:>6.2f} {result.corr:>6.3f} "
            f"{result.row_shift:>+4d},{result.col_shift:>+4d}"
        )

    if not results:
        print("\nno pair could be compared", file=sys.stderr)
        return 1

    biases = [r.bias for r in results]
    gains = [r.gain for r in results]
    print(f"\n{len(results)} pair(s). Median bias {np.median(biases):+.2f} dB "
          f"(range {min(biases):+.2f} to {max(biases):+.2f}); "
          f"median gain {np.median(gains):+.2f} dB "
          f"(range {min(gains):+.2f} to {max(gains):+.2f}).")
    if abs(np.median(biases) - np.median(gains)) > 0.15:
        print("bias and gain disagree: much of the dB offset is a difference in")
        print("effective looks, not in calibration. Quote gain.")

    by_aoi = {}
    for r in results:
        by_aoi.setdefault(r.pair.aoi, []).append(r)
    if len(by_aoi) > 1:
        print("\nPer AOI (RMSE and CORR are where a missing terrain correction shows):")
        for aoi, group in sorted(by_aoi.items(), key=lambda kv: np.median([r.rmse for r in kv[1]])):
            print(f"  {aoi:10} bias {np.median([r.bias for r in group]):+.2f}  "
                  f"gain {np.median([r.gain for r in group]):+.2f}  "
                  f"rmse {np.median([r.rmse for r in group]):.2f}  "
                  f"corr {np.median([r.corr for r in group]):.3f}")
    mismatched = sorted({r.pair.aoi for r in results if r.reprojected})
    if mismatched:
        print(f"\nProjection mismatch in: {', '.join(mismatched)}. The SNAP output is not")
        print("in the same CRS as the GA product, so it was reprojected as well as")
        print("resampled - an extra interpolation applied to these AOIs and not the")
        print("others. Reprocess them (--overwrite) to take GA's projection at source.")
    if any(r.row_shift or r.col_shift for r in results):
        print("Non-zero shift on at least one pair: the products are not co-registered,")
        print("so the bias and RMSE above include a misregistration component.")

    if args.csv:
        write_csv(Path(args.csv), results)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
