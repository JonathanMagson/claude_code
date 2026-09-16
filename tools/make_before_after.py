#!/usr/bin/env python3
"""Two GA NRB images per area, before and after, with their SLC and GRD names.

A minimal comparison set. For each area this picks one burst, takes it on two
dates, downloads the NRB rasters, and reports the Sentinel-1 Level-1 products
behind each date so they can be pulled from an existing archive.

Two choices matter:

* **One burst, both dates.** The pair is only differenceable if it is the same
  burst on the same track - same footprint, same grid, same viewing geometry.
  So the burst is chosen from those present on *both* dates, not from either
  date alone. It does not need to fill the area; a burst is ~25 x 100 km and
  that is plenty to compare products on.
* **The widest separation available**, by default the first and last dates in
  the window, since a before/after pair is more use the further apart it is.
  ``--before`` and ``--after`` override.

Usage::

    python tools/make_before_after.py --track 9 --out data/before_after
    python tools/make_before_after.py --aoi hunter --before 2024-06-03 --after 2025-06-22
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_ga_c0 import (  # noqa: E402
    DEFAULT_ASSETS,
    STATIC_ASSETS,
    _download,
    measure,
    select_assets,
)
from find_comparison_scenes import AOIS, match_grd_for_slc, slc_url  # noqa: E402
from find_ga_c0_scenes import (  # noqa: E402
    C0_END,
    C0_START,
    CATALOG,
    COLLECTIONS_C0,
    CatalogUnreachable,
    open_catalog,
    search_aoi,
)
from ga_l1_index import decompose_burst  # noqa: E402

FIELDS = [
    "aoi", "role", "acquired", "ga_burst_id", "esa_burst_id", "track", "subswath",
    "ga_item_id", "platform", "absolute_orbit", "orbit_state",
    "slc_scene_id", "grd_scene_id", "grd_all_slices", "slc_url", "ga_files",
]


def pick_pair(
    items: Sequence[dict],
    before: Optional[str] = None,
    after: Optional[str] = None,
    burst: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Choose (burst_id, before_date, after_date) covered by a single burst."""
    by_date: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        by_date[(item.get("properties", {}).get("datetime") or "")[:10]].append(item)
    dates = sorted(d for d in by_date if d)
    if len(dates) < 2:
        raise SystemExit(f"need two dates, found {len(dates)}")

    first = before or dates[0]
    last = after or dates[-1]
    for label, value in (("--before", first), ("--after", last)):
        if value not in by_date:
            raise SystemExit(f"{label} {value} is not an available date; have "
                             f"{dates[0]} .. {dates[-1]} ({len(dates)} dates)")

    def bursts_on(day: str) -> set:
        return {i.get("properties", {}).get("sarard:burst_id") for i in by_date[day]}

    common = sorted(b for b in bursts_on(first) & bursts_on(last) if b)
    if not common:
        raise SystemExit(f"no burst appears on both {first} and {last}")
    if burst:
        if burst not in common:
            raise SystemExit(f"burst {burst} is not on both dates; options: {common}")
        return burst, first, last
    return common[len(common) // 2], first, last  # middle burst, least edge-clipped


def collect(
    items: Sequence[dict], burst_id: str, day: str
) -> Optional[dict]:
    for item in items:
        props = item.get("properties", {})
        if props.get("sarard:burst_id") == burst_id and (props.get("datetime") or "")[:10] == day:
            return item
    return None


def build(
    names: Sequence[str],
    out: Path,
    track: Optional[int],
    start: str,
    end: str,
    before: Optional[str],
    after: Optional[str],
    burst: Optional[str],
    with_static: bool,
    catalog: str,
    dry_run: bool,
) -> List[dict]:
    client = open_catalog(catalog)
    rows: List[dict] = []

    for name in names:
        items = search_aoi(client, AOIS[name]["bbox"], start, end, COLLECTIONS_C0, track=track)
        if not items:
            print(f"{name}: no Collection 0 items", file=sys.stderr)
            continue

        burst_id, day_before, day_after = pick_pair(items, before, after, burst)
        print(f"\n=== {name} ===")
        print(f"  burst  {burst_id}")
        print(f"  before {day_before}")
        print(f"  after  {day_after}")

        for role, day in (("before", day_before), ("after", day_after)):
            item = collect(items, burst_id, day)
            if item is None:
                print(f"  {role}: burst missing on {day}", file=sys.stderr)
                continue
            props = item.get("properties", {})
            slc = props.get("sarard:scene_id", "")
            matches = match_grd_for_slc(slc) if slc else []
            twin = next((m for m in matches if m["exact_twin"]), matches[0] if matches else None)

            wanted = list(DEFAULT_ASSETS) + (list(STATIC_ASSETS) if with_static else [])
            assets = select_assets(item, wanted)
            dest = out / name / burst_id / day.replace("-", "")
            jobs = [(href, dest / href.split("/")[-1]) for href in assets.values()]

            if dry_run:
                size = measure(jobs)
                print(f"  {role}: {len(jobs)} file(s), {size / 1e6:.0f} MB")
            else:
                for url, path in jobs:
                    _, size = _download(url, path)
                    print(f"  {role}: {size:>12,}  {path.name}")

            row = {
                "aoi": name,
                "role": role,
                "acquired": day,
                "ga_burst_id": burst_id,
                "ga_item_id": item.get("id", ""),
                "platform": props.get("platform", ""),
                "absolute_orbit": props.get("sat:absolute_orbit", ""),
                "orbit_state": props.get("sat:orbit_state", ""),
                "slc_scene_id": slc,
                "grd_scene_id": twin["scene_id"] if twin else "",
                "grd_all_slices": "|".join(m["scene_id"] for m in matches),
                "slc_url": slc_url(slc) or "",
                "ga_files": "|".join(p.name for _u, p in jobs),
            }
            row.update(decompose_burst(burst_id))
            rows.append(row)
            print(f"    SLC {row['slc_scene_id']}")
            print(f"    GRD {row['grd_scene_id'] or '(none found)'}")

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aoi", nargs="*", default=list(AOIS), choices=list(AOIS))
    ap.add_argument("--track", type=int, default=9,
                    help="relative orbit; 9 covers all three areas (default 9)")
    ap.add_argument("--start", default=C0_START)
    ap.add_argument("--end", default=C0_END)
    ap.add_argument("--before", default=None, help="YYYY-MM-DD; default earliest")
    ap.add_argument("--after", default=None, help="YYYY-MM-DD; default latest")
    ap.add_argument("--burst", default=None,
                    help="burst id to use; default one present on both dates")
    ap.add_argument("--with-static", action="store_true",
                    help="also fetch the geometry layers for the burst")
    ap.add_argument("--out", type=Path, default=Path("data/before_after"))
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        rows = build(args.aoi, args.out, args.track, args.start, args.end,
                     args.before, args.after, args.burst, args.with_static,
                     args.catalog, args.dry_run)
    except (CatalogUnreachable, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("nothing selected", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "before_after.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})

    scenes = args.out / "scenes_to_pull.txt"
    lines = []
    for row in rows:
        lines.append(f"{row['aoi']:<10} {row['role']:<7} {row['acquired']}  "
                     f"SLC {row['slc_scene_id']}  GRD {row['grd_scene_id']}")
    scenes.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{len(rows)} acquisitions ({len(rows) // 2} before/after pairs)")
    print(f"  {manifest}")
    print(f"  {scenes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
