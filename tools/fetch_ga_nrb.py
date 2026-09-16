#!/usr/bin/env python3
"""Download Geoscience Australia NRB bursts covering an AOI.

Pulls the per-acquisition rasters (gamma0 per polarisation, the layover/shadow
mask) and, once per burst id, the static layers those acquisitions reference
but do not carry: incidence angle, local incidence angle, number of looks and
the gamma0->beta0/sigma0 ratios.

The static layers are the reason to bother with the burst layout. They are
what let you convert convention or undo the terrain normalisation, and they
are published once per burst rather than per date, so fetching them separately
is the intended access pattern, not a workaround.

Usage::

    python tools/fetch_ga_nrb.py --aoi pilliga --out data/ga_nrb
    python tools/fetch_ga_nrb.py --aoi pilliga --bursts t045_095772_iw2 --no-static
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vegmon import ga_nrb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_comparison_scenes import AOIS  # noqa: E402

#: Everything except the thumbnail, which is a 7 MB PNG of no analytic use.
DEFAULT_ASSETS = (
    "VV-gamma0.tif",
    "VH-gamma0.tif",
    "HH-gamma0.tif",
    "HV-gamma0.tif",
    "mask.tif",
    "stac-item.json",
    "metadata.xml",
    "checksum.sha1",
)


def _download(url: str, dest: Path, timeout: float = 600.0) -> Tuple[Path, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest, dest.stat().st_size
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest, dest.stat().st_size


def fetch(
    aoi_name: str,
    out: Path,
    burst_filter: Optional[Sequence[str]] = None,
    assets: Sequence[str] = DEFAULT_ASSETS,
    with_static: bool = True,
    db_path: Optional[str] = None,
    workers: int = 8,
) -> List[Path]:
    aoi = AOIS[aoi_name]["bbox"]
    bursts = ga_nrb.bursts_for_aoi(aoi, db_path=db_path)
    if burst_filter:
        wanted = set(burst_filter)
        bursts = [b for b in bursts if b.burst_id in wanted]
    if not bursts:
        raise ga_nrb.CoverageError(
            f"GA has published no NRB bursts over {aoi_name}; "
            "its Collection 1 release is a sample, not continental coverage"
        )

    jobs: List[Tuple[str, Path]] = []
    for burst in sorted(bursts, key=lambda b: b.burst_id):
        for acq in ga_nrb.acquisitions(burst.burst_id):
            for suffix, url in acq.assets.items():
                if suffix not in assets:
                    continue
                jobs.append((url, out / burst.burst_id / acq.datetime_key / url.split("/")[-1]))
        if with_static:
            for _layer, url in ga_nrb.static_layers(burst.burst_id).items():
                jobs.append((url, out / burst.burst_id / "static" / url.split("/")[-1]))

    print(f"{len(bursts)} burst(s), {len(jobs)} file(s) -> {out}", flush=True)
    written: List[Path] = []
    total = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for path, size in pool.map(lambda j: _download(*j), jobs):
            written.append(path)
            total += size
            print(f"  {size:>12,}  {path.relative_to(out)}", flush=True)
    print(f"{total:,} bytes total", flush=True)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aoi", default="pilliga", choices=list(AOIS))
    ap.add_argument("--bursts", nargs="*", default=None,
                    help="restrict to these burst ids (default: all covering the AOI)")
    ap.add_argument("--out", default="data/ga_nrb", type=Path)
    ap.add_argument("--no-static", action="store_true", help="skip the static layers")
    ap.add_argument("--burst-db", default=None)
    args = ap.parse_args()

    try:
        fetch(args.aoi, args.out, args.bursts, with_static=not args.no_static,
              db_path=args.burst_db)
    except ga_nrb.CoverageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
