#!/usr/bin/env python3
"""Speckle-filter GA NRB gamma0 and convert it to decibels.

GA ship the NRB product deliberately unfiltered, in linear power - their own
metadata records ``speckle_filter_applied: False`` and gives the conversion as
``10*log10(backscatter_linear)``. Both steps are the user's to apply, and both
are lossy, which is why they are not baked into the product.

Order is not a matter of taste. The Lee filter models speckle as multiplicative
noise on **linear** intensity, so filtering must happen before the log. Running
it on decibels averages logarithms instead of powers and biases the result low.

The filter itself is imported from ``de_sar_demo`` - Geoscience Australia's own
implementation from the de-sar-sample-data repository - rather than rewritten,
so results match what their notebooks produce. It is NaN-aware by normalised
convolution, which matters here: a GA burst grid is roughly 70% nodata, and a
plain uniform filter would drag that nodata across every valid edge.

Outputs land beside their inputs with a suffix, so a burst folder ends up
holding the delivered product and the processed version together.

Usage::

    python tools/postprocess_nrb.py data/before_after
    python tools/postprocess_nrb.py data/before_after --window 5 --apply-mask
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

DEFAULT_WINDOW = 7          # de_sar_demo's own default for the xarray wrapper
DEFAULT_EPSILON = 1e-10     # -100 dB; a floor below any real backscatter
DEFAULT_SUFFIX = "_sf_db"

#: 0 valid, 1 shadow, 2 layover, 3 both, 255 invalid.
MASK_KEEP = (0,)


def load_lee_filter(de_sar_demo: Optional[Path] = None):
    """GA's Lee filter, from an install or a path to the demo repository.

    Preferring their implementation over a local rewrite keeps these outputs
    comparable with the published notebooks.
    """
    if de_sar_demo:
        sys.path.insert(0, str(Path(de_sar_demo).resolve()))
    try:
        from de_sar_demo.speckle_filters import lee_filter

        return lee_filter
    except ImportError as exc:
        missing = str(exc)
        # The module imports xarray at the top even though lee_filter itself
        # only needs numpy and scipy, so a missing dependency and a missing
        # repository fail identically. Say which one happened.
        if "de_sar_demo" in missing:
            raise SystemExit(
                f"cannot find de_sar_demo ({missing}).\n"
                "  point at the cloned repo:\n"
                "    --de-sar-demo C:\\Data\\GIT\\de-sar-sample-data\n"
                "  or install it:\n"
                "    cd C:\\Data\\GIT\\de-sar-sample-data && python -m pip install ."
            )
        raise SystemExit(
            f"found de_sar_demo, but it cannot import: {missing}\n"
            "  its speckle_filters module imports xarray at the top, so that has\n"
            "  to be present even though the Lee filter only uses numpy/scipy:\n"
            "    conda install -c conda-forge xarray -y"
        )


def _require_rasterio():
    try:
        import rasterio  # noqa: F401
    except ImportError:
        raise SystemExit(
            "needs rasterio:\n  conda install -c conda-forge rasterio -y"
        )


def to_db(values, epsilon: float = DEFAULT_EPSILON):
    """Linear power to decibels, floored so zeros do not become -inf.

    GA's rasters contain exact zeros. ``10*log10(0)`` is ``-inf``, and a single
    one of those poisons every mean and percentile computed downstream, so the
    clip is load-bearing rather than cosmetic.
    """
    import numpy as np

    with np.errstate(divide="ignore", invalid="ignore"):
        return (10.0 * np.log10(np.clip(values, epsilon, None))).astype("float32")


def find_inputs(root: Path, suffix: str) -> List[Path]:
    """Every gamma0 raster under ``root``, excluding previous outputs."""
    return sorted(
        p for p in root.rglob("*.tif")
        if "gamma0" in p.name.lower() and suffix not in p.stem
    )


def sibling_mask(path: Path) -> Optional[Path]:
    hits = sorted(path.parent.glob("*_mask.tif"))
    return hits[0] if hits else None


def process(
    path: Path,
    lee_filter,
    window: int,
    epsilon: float,
    suffix: str,
    apply_mask: bool,
    overwrite: bool,
) -> Optional[Tuple[Path, dict]]:
    import numpy as np
    import rasterio

    out_path = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if out_path.exists() and not overwrite:
        print(f"  {path.name}: output exists, skipped")
        return None

    with rasterio.open(path) as src:
        linear = src.read(1).astype("float64")
        profile = src.profile.copy()

    before_valid = float(np.isfinite(linear).mean())

    if apply_mask:
        mask_path = sibling_mask(path)
        if mask_path is None:
            print(f"  {path.name}: no sibling mask found, continuing unmasked",
                  file=sys.stderr)
        else:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            linear = np.where(np.isin(mask, MASK_KEEP), linear, np.nan)

    # Filter in linear power, then convert. Not the other way round.
    # de_sar_demo guards some of its NaN arithmetic with errstate and not all
    # of it, so a 70%-nodata burst emits warnings that are expected and not
    # actionable. Silence them rather than burying the per-file report.
    with np.errstate(invalid="ignore", divide="ignore"):
        filtered = lee_filter(linear, size=window)
    decibels = to_db(filtered, epsilon)
    # Keep nodata as nodata rather than letting the floor fill it in.
    decibels = np.where(np.isfinite(filtered), decibels, np.nan).astype("float32")

    profile.update(
        dtype="float32", count=1, nodata=float("nan"),
        compress="deflate", tiled=True, blockxsize=512, blockysize=512,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(decibels, 1)
        dst.update_tags(
            speckle_filter="Lee (de_sar_demo)",
            speckle_filter_window=str(window),
            units="dB",
            conversion_eq="10*log10(backscatter_linear)",
            source=path.name,
        )

    finite = decibels[np.isfinite(decibels)]
    stats = {
        "valid_before": before_valid,
        "valid_after": float(np.isfinite(decibels).mean()),
        "median_db": float(np.median(finite)) if finite.size else float("nan"),
        "p5_db": float(np.percentile(finite, 5)) if finite.size else float("nan"),
        "p95_db": float(np.percentile(finite, 95)) if finite.size else float("nan"),
        "bytes": out_path.stat().st_size,
    }
    return out_path, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", type=Path, default=Path("data/before_after"))
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help=f"Lee filter window in pixels (default {DEFAULT_WINDOW})")
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                    help="floor applied before the log, to avoid -inf")
    ap.add_argument("--suffix", default=DEFAULT_SUFFIX)
    ap.add_argument("--apply-mask", action="store_true",
                    help="also null out layover and shadow using the sibling mask")
    ap.add_argument("--de-sar-demo", type=Path, default=None,
                    help="path to a cloned de-sar-sample-data, if not installed")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"not found: {args.root}", file=sys.stderr)
        return 1

    _require_rasterio()
    lee_filter = load_lee_filter(args.de_sar_demo)

    inputs = find_inputs(args.root, args.suffix)
    if not inputs:
        print(f"no gamma0 rasters under {args.root}", file=sys.stderr)
        return 1

    print(f"{len(inputs)} raster(s), Lee window {args.window} px, "
          f"{'masked, ' if args.apply_mask else ''}output suffix {args.suffix}\n")

    written = 0
    for path in inputs:
        print(f"{path.relative_to(args.root)}")
        result = process(path, lee_filter, args.window, args.epsilon,
                         args.suffix, args.apply_mask, args.overwrite)
        if result is None:
            continue
        out_path, stats = result
        written += 1
        print(f"  -> {out_path.name}")
        print(f"     valid {stats['valid_after'] * 100:.1f}%"
              + (f" (was {stats['valid_before'] * 100:.1f}%)"
                 if abs(stats["valid_after"] - stats["valid_before"]) > 0.001 else "")
              + f"   p5 {stats['p5_db']:.1f}  median {stats['median_db']:.1f}"
                f"  p95 {stats['p95_db']:.1f} dB")

    print(f"\n{written} file(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
