#!/usr/bin/env python3
"""Find GA Collection 0 NRB coverage via DEA's STAC API, and match it to Level-1.

Collection 0 and Collection 1 are different archives with different reach, and
the difference decides whether an AOI has GA data at all:

* **Collection 1** is the current production run. It lives in a public bucket
  and :mod:`vegmon.ga_nrb` reads it directly. Over NSW it is a single track,
  one date - see ``docs/ga-sar-comparison.md``.
* **Collection 0** is the 2025 early-access release: continental Australia for
  1 Jun 2024 - 30 Jun 2025, plus deep time series over Griffith NSW, Kununurra
  WA, and Injune/Starcke/Mungalla QLD. It is **not** in any publicly listable
  bucket that could be found, so unlike Collection 1 it cannot be reached by
  listing S3. It is served by DEA's development STAC API, and that is the only
  route to it.

So this script needs ``explorer.dev.dea.ga.gov.au`` to be reachable. It is
blocked from some networks - including the one this was written on, which is
why the Collection 1 tooling takes the bucket route instead. Run this from
somewhere that can reach it.

Once items are found, the Level-1 matching is the same as for Collection 1 and
reuses the same tested code: the NRB item names its parent SLC in
``sarard:scene_id``, and the GRD twin is the scene sharing its absolute orbit
and data-take id.

Usage::

    python tools/find_ga_c0_scenes.py --out c0_manifest.json
    python tools/find_ga_c0_scenes.py --aoi hunter bluemtns --start 2024-06-01
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from find_comparison_scenes import AOIS, match_grd_for_slc, slc_url  # noqa: E402

CATALOG = "https://explorer.dev.dea.ga.gov.au/stac"

#: The four Collection 0 products, one per polarisation mode. VV+VH is the
#: primary mode over Australia; the others are listed so an AOI that only has
#: an off-nominal acquisition still turns something up.
COLLECTIONS_C0 = [
    "ga_s1_nrb_iw_vv_vh_0",
    "ga_s1_nrb_iw_vv_0",
    "ga_s1_nrb_iw_hh_0",
    "ga_s1_nrb_iw_hh_hv_0",
]

#: Continental Collection 0 window from GA's early-access announcement. The
#: extended sites (Griffith, Kununurra, Injune/Starcke/Mungalla) run wider, so
#: --start/--end override this.
C0_START = "2024-06-01"
C0_END = "2025-06-30"


class CatalogUnreachable(RuntimeError):
    """The STAC API could not be opened - usually an egress policy, not an outage."""


def open_catalog(url: str = CATALOG, timeout: float = 60.0):
    try:
        from pystac_client import Client
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError("needs pystac-client: pip install pystac-client") from exc
    try:
        return Client.open(url)
    except Exception as exc:
        raise CatalogUnreachable(
            f"could not open {url}: {exc}\n"
            "DEA's dev STAC API is blocked on some networks. Collection 1 can be "
            "read from S3 instead (tools/find_comparison_scenes.py); Collection 0 "
            "cannot - it has no public bucket."
        ) from exc


def search_aoi(
    client,
    bbox: Sequence[float],
    start: str,
    end: str,
    collections: Sequence[str] = COLLECTIONS_C0,
    limit: Optional[int] = None,
    track: Optional[int] = None,
) -> List[dict]:
    """Every Collection 0 item intersecting an AOI, as plain dicts.

    ``track`` restricts to one relative orbit. Worth using: an AOI is normally
    seen by three or four tracks, and ascending and descending passes view the
    canopy from opposite sides. Mixing them puts a viewing-geometry difference
    straight into any change signal, so a time series should be built from one
    track, not from everything that overlaps.
    """
    search = client.search(
        collections=list(collections),
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        max_items=limit,
    )
    items = [item.to_dict() for item in search.items()]
    if track is not None:
        items = [
            i for i in items if i.get("properties", {}).get("sat:relative_orbit") == track
        ]
    return items


def summarise(items: Sequence[dict]) -> dict:
    """Dates, tracks and parent SLCs for a set of NRB items."""
    by_collection: Dict[str, int] = {}
    dates, tracks, bursts, slcs = set(), set(), set(), set()
    for item in items:
        props = item.get("properties", {})
        by_collection[item.get("collection", "?")] = (
            by_collection.get(item.get("collection", "?"), 0) + 1
        )
        if props.get("datetime"):
            dates.add(props["datetime"][:10])
        if props.get("sat:relative_orbit") is not None:
            tracks.add(props["sat:relative_orbit"])
        if props.get("sarard:burst_id"):
            bursts.add(props["sarard:burst_id"])
        if props.get("sarard:scene_id"):
            slcs.add(props["sarard:scene_id"])
    return {
        "items": len(items),
        "by_collection": by_collection,
        "dates": sorted(dates),
        "tracks": sorted(tracks),
        "bursts": sorted(bursts),
        "slc_scene_ids": sorted(slcs),
    }


def build(
    names: Sequence[str],
    start: str,
    end: str,
    match_level1: bool = True,
    catalog: str = CATALOG,
    track: Optional[int] = None,
    max_slcs: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    client = open_catalog(catalog)
    out: Dict[str, dict] = {}

    for name in names:
        aoi = AOIS[name]
        if verbose:
            print(f"\n=== {name}: {aoi['label']} ===", flush=True)
        items = search_aoi(client, aoi["bbox"], start, end, track=track)
        summary = summarise(items)
        if verbose:
            print(f"  {summary['items']} Collection 0 items "
                  f"across {len(summary['dates'])} dates, tracks {summary['tracks']}",
                  flush=True)
            print(f"  products: {summary['by_collection']}", flush=True)

        pairs = []
        if match_level1:
            # One GRD lookup per SLC costs a day listing each, so pair the
            # distinct scenes rather than every burst item.
            slcs = summary["slc_scene_ids"]
            if max_slcs is not None:
                slcs = slcs[:max_slcs]
            if verbose:
                print(f"  pairing {len(slcs)} of {len(summary['slc_scene_ids'])} SLCs "
                      "to Level-1 (one archive-day listing per distinct date)", flush=True)
            for slc in slcs:
                grds = match_grd_for_slc(slc)
                twin = next((g for g in grds if g["exact_twin"]), grds[0] if grds else None)
                pairs.append(
                    {
                        "slc_scene_id": slc,
                        "slc_url": slc_url(slc),
                        "grd_twin": twin,
                        "grd_pass_slices": len(grds),
                    }
                )
            if verbose:
                matched = sum(1 for p in pairs if p["grd_twin"])
                print(f"  matched {matched}/{len(pairs)} SLCs to a GRD twin", flush=True)

        out[name] = {
            "label": aoi["label"],
            "bbox": list(aoi["bbox"]),
            "collection_0": summary,
            "sentinel1": {"matched_to_ga": pairs},
        }

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "catalog": catalog,
        "window": [start, end],
        "track": track,
        "aois": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aoi", nargs="*", default=list(AOIS), choices=list(AOIS))
    ap.add_argument("--start", default=C0_START)
    ap.add_argument("--end", default=C0_END)
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--no-level1", action="store_true",
                    help="skip GRD/SLC matching (much faster)")
    ap.add_argument("--track", type=int, default=None,
                    help="restrict to one relative orbit; a time series should use one")
    ap.add_argument("--max-slcs", type=int, default=None,
                    help="cap how many SLCs get paired to Level-1")
    ap.add_argument("--out", default="c0_manifest.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        manifest = build(args.aoi, args.start, args.end,
                         match_level1=not args.no_level1,
                         catalog=args.catalog, track=args.track,
                         max_slcs=args.max_slcs, verbose=not args.quiet)
    except (CatalogUnreachable, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    Path(args.out).write_text(json.dumps(manifest, indent=2))
    if not args.quiet:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
