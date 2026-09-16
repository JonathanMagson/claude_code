#!/usr/bin/env python3
"""Check downloaded GA NRB rasters: how much is valid, and is the signal sane.

Two things are worth knowing before building anything on these files, and
neither is visible from the file listing.

**How much of the burst is actually data.** GA's own caveat for this release is
that burst geometries extend past their valid data, so a burst grid is not a
burst of pixels. A sample measured while building this tooling was 31.6% valid.
If the valid area over the part of the burst you care about is thin, pick a
different burst rather than discovering it downstream.

**Whether the backscatter is plausible.** Gamma0 over inland NSW woodland and
cropping sits around -18 to -8 dB. A median far outside that, or a valid
fraction that changes sharply between two dates of the same burst, means
something is wrong with the pair before any differencing starts.

The mask classes are GA's: 0 valid, 1 shadow, 2 layover, 3 both, 255 invalid.

Usage::

    python tools/check_rasters.py data/before_after
    python tools/check_rasters.py data/before_after --per-file
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MASK_CLASSES = {
    0: "valid",
    1: "shadow",
    2: "layover",
    3: "layover+shadow",
    255: "invalid",
}


def _require_rasterio():
    try:
        import rasterio  # noqa: F401
    except ImportError:
        raise SystemExit(
            "needs rasterio:\n"
            "  conda install -c conda-forge rasterio -y\n"
            "(conda rather than pip - the Windows wheels pull GDAL with them)"
        )


def describe_mask(path: Path) -> dict:
    import numpy as np
    import rasterio

    with rasterio.open(path) as ds:
        values = ds.read(1)
        shape = (ds.height, ds.width)
        crs = str(ds.crs)
        res = ds.res
    total = values.size
    counts = {int(k): int(v) for k, v in zip(*np.unique(values, return_counts=True))}
    return {
        "shape": shape,
        "crs": crs,
        "resolution": res,
        "total": total,
        "counts": counts,
        "valid_fraction": counts.get(0, 0) / total if total else 0.0,
        # Layover and shadow are still data, just geometrically compromised.
        "usable_fraction": sum(counts.get(k, 0) for k in (0, 1, 2, 3)) / total if total else 0.0,
    }


def describe_backscatter(path: Path) -> dict:
    import numpy as np
    import rasterio

    with rasterio.open(path) as ds:
        values = ds.read(1).astype("float64")
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    out = {
        "shape": values.shape,
        "valid_fraction": finite.size / values.size if values.size else 0.0,
        "median_db": None,
        "p5_db": None,
        "p95_db": None,
    }
    if positive.size:
        db = 10 * np.log10(positive)
        out["median_db"] = float(np.median(db))
        out["p5_db"] = float(np.percentile(db, 5))
        out["p95_db"] = float(np.percentile(db, 95))
    return out


def scan(root: Path) -> Dict[Tuple[str, ...], Dict[str, Path]]:
    """Group rasters by the directory they sit in."""
    groups: Dict[Tuple[str, ...], Dict[str, Path]] = defaultdict(dict)
    for path in sorted(root.rglob("*.tif")):
        key = tuple(path.relative_to(root).parts[:-1])
        name = path.name.lower()
        if name.endswith("_mask.tif"):
            groups[key]["mask"] = path
        elif "vv-gamma0" in name or "vv_gamma0" in name:
            groups[key]["vv"] = path
        elif "vh-gamma0" in name or "vh_gamma0" in name:
            groups[key]["vh"] = path
        elif "hh-gamma0" in name or "hh_gamma0" in name:
            groups[key]["hh"] = path
        elif "local-incidence" in name or "local_incidence" in name:
            groups[key]["lia"] = path
    return groups


def report(root: Path, per_file: bool = False) -> int:
    _require_rasterio()
    groups = scan(root)
    if not groups:
        print(f"no .tif found under {root}")
        return 1

    # A static folder holds geometry layers only - no mask, no backscatter - so
    # it has nothing to report in these columns and would show as a blank row.
    statics = {k: v for k, v in groups.items()
               if "mask" not in v and not {"vv", "vh", "hh"} & set(v)}
    groups = {k: v for k, v in groups.items() if k not in statics}
    if not groups:
        print(f"no acquisition folders under {root} (only static layers)")
        return 1

    print(f"{len(groups)} acquisition folder(s) under {root}"
          + (f", {len(statics)} static folder(s)" if statics else "") + "\n")
    header = f"{'folder':<52}{'valid%':>8}{'usable%':>9}{'VV med dB':>11}{'VH med dB':>11}"
    print(header)
    print("-" * len(header))

    valid_by_burst: Dict[str, List[float]] = defaultdict(list)
    rows = []
    for key, files in sorted(groups.items()):
        label = "/".join(key)
        mask = describe_mask(files["mask"]) if "mask" in files else None
        vv = describe_backscatter(files["vv"]) if "vv" in files else None
        vh = describe_backscatter(files["vh"]) if "vh" in files else None

        valid = f"{mask['valid_fraction'] * 100:7.1f}" if mask else "      -"
        usable = f"{mask['usable_fraction'] * 100:8.1f}" if mask else "       -"
        vv_db = f"{vv['median_db']:10.1f}" if vv and vv["median_db"] is not None else "         -"
        vh_db = f"{vh['median_db']:10.1f}" if vh and vh["median_db"] is not None else "         -"
        print(f"{label:<52}{valid:>8}{usable:>9}{vv_db:>11}{vh_db:>11}")

        if mask and len(key) >= 2:
            valid_by_burst["/".join(key[:-1])].append(mask["valid_fraction"])
        rows.append((label, files, mask, vv, vh))

    # --- pair consistency -------------------------------------------------
    drift = {b: v for b, v in valid_by_burst.items()
             if len(v) > 1 and (max(v) - min(v)) > 0.05}
    if drift:
        print("\nvalid fraction differs between dates of the same burst:")
        for burst, values in sorted(drift.items()):
            span = ", ".join(f"{v * 100:.1f}%" for v in values)
            print(f"  {burst}: {span}")
        print("  a difference here is a difference in what can be compared;")
        print("  intersect the masks before differencing the pair.")
    elif valid_by_burst:
        print("\nvalid fraction is consistent across dates of each burst.")

    # --- detail -----------------------------------------------------------
    if per_file:
        for label, files, mask, vv, vh in rows:
            print(f"\n{label}")
            if mask:
                print(f"  grid {mask['shape'][1]}x{mask['shape'][0]}  {mask['crs']}  "
                      f"{mask['resolution'][0]:g} m")
                for code, count in sorted(mask["counts"].items()):
                    name = MASK_CLASSES.get(code, f"class {code}")
                    print(f"    {name:<16}{count:>12,}  {count / mask['total'] * 100:5.1f}%")
            for band, stats in (("VV", vv), ("VH", vh)):
                if stats and stats["median_db"] is not None:
                    print(f"  {band}: p5 {stats['p5_db']:.1f}  median "
                          f"{stats['median_db']:.1f}  p95 {stats['p95_db']:.1f} dB"
                          f"   ({stats['valid_fraction'] * 100:.1f}% finite)")

    if statics:
        print("\nstatic layers (geometry only, shared across dates):")
        for key, files in sorted(statics.items()):
            print(f"  {'/'.join(key)}: {', '.join(sorted(files))}")

    print("\nmask classes: 0 valid, 1 shadow, 2 layover, 3 both, 255 invalid")
    print("gamma0 over inland NSW woodland/cropping is typically -18 to -8 dB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", type=Path, default=Path("data/before_after"))
    ap.add_argument("--per-file", action="store_true",
                    help="full mask class breakdown and dB percentiles")
    args = ap.parse_args()
    if not args.root.exists():
        print(f"not found: {args.root}")
        return 1
    return report(args.root, per_file=args.per_file)


if __name__ == "__main__":
    raise SystemExit(main())
