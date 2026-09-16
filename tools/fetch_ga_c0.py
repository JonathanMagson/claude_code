#!/usr/bin/env python3
"""Download GA Collection 0 NRB rasters for an AOI, via the STAC API.

The Collection 1 fetcher (``fetch_ga_nrb.py``) lists a public bucket. Collection
0 has no such bucket, so this one searches the STAC API and downloads from the
asset hrefs each item carries.

Asset names are not the same between the two collections - Collection 0 has
``VV_gamma0`` and ``mask`` where Collection 1 has ``vv_gamma0`` and
``oa_layover_shadow_mask`` - so assets are matched on a normalised name rather
than an exact key, and the same ``--assets`` argument works against either.

Static layers come back as assets on the item itself, pointing into the
separate static product. They are identical for every date of a burst, so
``--with-static`` fetches them once per burst rather than once per acquisition.

Usage::

    python tools/fetch_ga_c0.py --aoi hunter --track 9 --dates 3 --out data/c0
    python tools/fetch_ga_c0.py --aoi pilliga --track 9 --dates 10 --with-static
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from find_comparison_scenes import AOIS  # noqa: E402
from find_ga_c0_scenes import (  # noqa: E402
    C0_END,
    C0_START,
    CATALOG,
    COLLECTIONS_C0,
    CatalogUnreachable,
    open_catalog,
    search_aoi,
)

#: What to pull by default: both polarisations and the validity mask. The
#: thumbnail is deliberately absent - it is a multi-megabyte PNG of no
#: analytic use.
DEFAULT_ASSETS = ("vv_gamma0", "vh_gamma0", "hh_gamma0", "hv_gamma0", "mask")

#: Geometry-only layers, shared by every acquisition of a burst.
STATIC_ASSETS = (
    "incidence_angle",
    "local_incidence_angle",
    "number_of_looks",
    "gamma0_to_beta0_ratio",
    "gamma0_to_sigma0_ratio",
)


def normalise_asset(name: str) -> str:
    """Collapse the Collection 0 and Collection 1 spellings onto one name."""
    key = name.lower().removeprefix("oa_")
    if key in {"layover_shadow_mask", "mask"}:
        return "mask"
    return key


def select_assets(item: dict, wanted: Sequence[str]) -> Dict[str, str]:
    """Asset hrefs from an item whose normalised name is in ``wanted``."""
    want = {normalise_asset(w) for w in wanted}
    out: Dict[str, str] = {}
    for key, asset in (item.get("assets") or {}).items():
        norm = normalise_asset(key)
        href = asset.get("href") if isinstance(asset, dict) else None
        if norm in want and href:
            out[norm] = href
    return out


def _download(url: str, dest: Path, timeout: float = 900.0) -> Tuple[Path, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest, dest.stat().st_size
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest, dest.stat().st_size


def plan(
    items: Sequence[dict],
    out: Path,
    assets: Sequence[str],
    with_static: bool,
    static_assets: Sequence[str] = STATIC_ASSETS,
) -> List[Tuple[str, Path]]:
    """Every (url, destination) to fetch, statics deduplicated per burst."""
    jobs: List[Tuple[str, Path]] = []
    static_done: set = set()
    for item in items:
        props = item.get("properties", {})
        burst = props.get("sarard:burst_id", "unknown_burst")
        stamp = (props.get("datetime") or "")[:19].replace(":", "").replace("-", "")
        for _norm, href in sorted(select_assets(item, assets).items()):
            jobs.append((href, out / burst / stamp / href.split("/")[-1]))
        if with_static and burst not in static_done:
            static_done.add(burst)
            for _norm, href in sorted(select_assets(item, static_assets).items()):
                jobs.append((href, out / burst / "static" / href.split("/")[-1]))
    return jobs


def _content_length(url: str, timeout: float = 30.0) -> int:
    """Size of an asset without fetching it."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def measure(jobs: Sequence[Tuple[str, Path]], workers: int = 16) -> int:
    """Total download size, from HEAD requests.

    Worth knowing before committing: the static layers are ~193 MB per burst
    against ~33 MB for a burst-date of VV+VH, so a request that looks small in
    file count can be dominated by geometry layers that never change.
    """
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(lambda j: _content_length(j[0]), jobs))


def fetch(
    aoi_name: str,
    out: Path,
    track: Optional[int] = None,
    start: str = C0_START,
    end: str = C0_END,
    dates: Optional[int] = None,
    assets: Sequence[str] = DEFAULT_ASSETS,
    with_static: bool = False,
    static_assets: Sequence[str] = STATIC_ASSETS,
    max_bursts: Optional[int] = None,
    catalog: str = CATALOG,
    workers: int = 8,
    dry_run: bool = False,
) -> List[Path]:
    client = open_catalog(catalog)
    items = search_aoi(client, AOIS[aoi_name]["bbox"], start, end,
                       COLLECTIONS_C0, track=track)
    if not items:
        raise SystemExit(
            f"no Collection 0 items for {aoi_name}"
            + (f" on track {track}" if track else "")
            + f" between {start} and {end}"
        )

    by_date: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        by_date[(item.get("properties", {}).get("datetime") or "")[:10]].append(item)
    chosen = sorted(by_date)
    if dates is not None:
        chosen = chosen[:dates]
    picked = [i for d in chosen for i in by_date[d]]

    if max_bursts is not None:
        keep = sorted({i.get("properties", {}).get("sarard:burst_id") for i in picked})
        keep = set(k for k in keep if k)
        keep = set(sorted(keep)[:max_bursts])
        picked = [i for i in picked
                  if i.get("properties", {}).get("sarard:burst_id") in keep]

    bursts = {i.get("properties", {}).get("sarard:burst_id") for i in picked}
    print(f"{aoi_name}: {len(items)} items over {len(by_date)} dates; "
          f"taking {len(picked)} items over {len(chosen)} dates "
          f"across {len(bursts)} burst(s)", flush=True)

    jobs = plan(picked, out, assets, with_static, static_assets)
    print(f"{len(jobs)} file(s) -> {out}", flush=True)
    if dry_run:
        for _url, dest in jobs[:12]:
            print(f"  would fetch {dest.relative_to(out)}")
        if len(jobs) > 12:
            print(f"  ... +{len(jobs) - 12} more")
        print("  measuring...", flush=True)
        total = measure(jobs)
        statics = [j for j in jobs if "/static/" in j[1].as_posix()]
        static_bytes = measure(statics) if statics else 0
        print(f"\n  TOTAL {total / 1e9:.2f} GB"
              f"  ({static_bytes / 1e9:.2f} GB of that is static layers,"
              f" fetched once per burst and reusable across every date)")
        return []

    written: List[Path] = []
    total = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for path, size in pool.map(lambda j: _download(*j), jobs):
            written.append(path)
            total += size
            print(f"  {size:>13,}  {path.relative_to(out)}", flush=True)
    print(f"{total:,} bytes total", flush=True)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aoi", default="hunter", choices=list(AOIS))
    ap.add_argument("--track", type=int, default=None,
                    help="restrict to one relative orbit (recommended: 9 covers all three)")
    ap.add_argument("--start", default=C0_START)
    ap.add_argument("--end", default=C0_END)
    ap.add_argument("--dates", type=int, default=None,
                    help="only the first N dates in the window")
    ap.add_argument("--assets", nargs="*", default=list(DEFAULT_ASSETS))
    ap.add_argument("--with-static", action="store_true",
                    help="also fetch the per-burst geometry layers (~193 MB per burst for all five)")
    ap.add_argument("--static-assets", nargs="*", default=list(STATIC_ASSETS),
                    help="which geometry layers; local_incidence_angle alone is "
                         "enough to explain a terrain-correction difference")
    ap.add_argument("--max-bursts", type=int, default=None,
                    help="only the first N bursts, for a pilot download")
    ap.add_argument("--out", type=Path, default=Path("data/ga_c0"))
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be fetched and stop")
    args = ap.parse_args()

    try:
        fetch(args.aoi, args.out, track=args.track, start=args.start, end=args.end,
              dates=args.dates, assets=args.assets, with_static=args.with_static,
              static_assets=args.static_assets, max_bursts=args.max_bursts,
              catalog=args.catalog, dry_run=args.dry_run)
    except (CatalogUnreachable, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
