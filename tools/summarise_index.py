#!/usr/bin/env python3
"""Summarise a burst_join.csv: coverage, Level-1 pairing, and where it is thin.

The index is 1,000+ rows, which is the wrong shape for reading. This answers
the questions actually worth asking of it:

* Did everything pair to a Level-1 product, and if not, which rows did not?
* How many distinct SLCs and GRDs is that really, once acquisitions shared
  between areas are deduplicated? That is the number to pull from an archive,
  not the per-area total.
* Which dates need more than one SLC? An area sitting across an ESA slice
  boundary needs two per date, and a burst on the seam needs two GRD slices.
* How does each area sit across IW1/IW2/IW3, and which ESA burst ids does it
  use - the join key into burst ids read from SLC annotation.

Standard library only, so it runs wherever the index does.

Usage::

    python tools/summarise_index.py data/l1_index/burst_join.csv
    python tools/summarise_index.py data/l1_index/burst_join.csv --bursts
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


def load(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarise(rows: Sequence[dict], show_bursts: bool = False) -> None:
    if not rows:
        print("empty index")
        return

    by_aoi: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_aoi[row.get("aoi", "?")].append(row)

    print(f"{len(rows):,} burst-acquisitions across {len(by_aoi)} area(s)\n")

    header = f"{'area':<10}{'bursts':>8}{'dates':>7}{'SLCs':>7}{'GRDs':>7}{'bursts/date':>13}{'SLCs/date':>11}"
    print(header)
    print("-" * len(header))
    for aoi, group in sorted(by_aoi.items()):
        dates = {r["acquired"] for r in group}
        slcs = {r["slc_scene_id"] for r in group if r["slc_scene_id"]}
        grds = {r["grd_scene_id"] for r in group if r["grd_scene_id"]}
        print(f"{aoi:<10}{len(group):>8}{len(dates):>7}{len(slcs):>7}{len(grds):>7}"
              f"{len(group) / max(len(dates), 1):>13.1f}{len(slcs) / max(len(dates), 1):>11.2f}")

    # --- what to actually pull -------------------------------------------
    all_slc = {r["slc_scene_id"] for r in rows if r["slc_scene_id"]}
    all_grd = {r["grd_scene_id"] for r in rows if r["grd_scene_id"]}
    per_aoi_total = sum(
        len({r["slc_scene_id"] for r in g if r["slc_scene_id"]}) for g in by_aoi.values()
    )
    print(f"\nLevel-1 products to pull: {len(all_slc)} SLC, {len(all_grd)} GRD")
    if per_aoi_total > len(all_slc):
        print(f"  ({per_aoi_total} counted per-area, so {per_aoi_total - len(all_slc)} "
              "acquisitions are shared between areas - fetch the deduplicated list)")

    # --- pairing health ---------------------------------------------------
    unmatched = [r for r in rows if not r["grd_scene_id"]]
    print(f"\nunmatched (no GRD twin): {len(unmatched)}")
    for row in unmatched[:10]:
        print(f"  {row['aoi']:<10} {row['acquired']}  {row['ga_burst_id']}  {row['slc_scene_id']}")
    if len(unmatched) > 10:
        print(f"  ... +{len(unmatched) - 10} more")

    # Every IW pass is cut into a dozen or more slices, so "the pass has more
    # than one slice" is true of everything and says nothing about any burst.
    # Report the distribution instead, and be explicit that identifying the
    # bursts actually sitting on a seam needs footprint intersection, which
    # this index does not carry.
    per_pass = Counter(_int(r.get("grd_slice_count")) for r in rows
                       if _int(r.get("grd_slice_count")))
    if per_pass:
        spread = ", ".join(f"{n} slices x{c} burst-rows" for n, c in sorted(per_pass.items()))
        print(f"\nGRD slices per pass: {spread}")
        print("  every pass is sliced, so this is about the pass, not the burst;")
        print("  grd_all_slices lists them all, and a burst near a slice edge needs two.")

    # Areas on one track can genuinely share a burst.
    burst_areas = defaultdict(set)
    for row in rows:
        if row.get("ga_burst_id"):
            burst_areas[row["ga_burst_id"]].add(row.get("aoi", "?"))
    shared = {b: a for b, a in burst_areas.items() if len(a) > 1}
    print(f"\nbursts shared between areas: {len(shared)} of {len(burst_areas)}")
    for burst, areas in sorted(shared.items())[:10]:
        print(f"  {burst}  {' + '.join(sorted(areas))}")
    if len(shared) > 10:
        print(f"  ... +{len(shared) - 10} more")

    # --- slice-boundary areas --------------------------------------------
    print("\nSLCs needed per date:")
    for aoi, group in sorted(by_aoi.items()):
        per_date = defaultdict(set)
        for row in group:
            if row["slc_scene_id"]:
                per_date[row["acquired"]].add(row["slc_scene_id"])
        spread = Counter(len(v) for v in per_date.values())
        detail = ", ".join(f"{n} SLC on {c} date(s)" for n, c in sorted(spread.items()))
        flag = "  <- sits across an SLC boundary" if max(spread, default=0) > 1 else ""
        print(f"  {aoi:<10} {detail}{flag}")

    # --- geometry ---------------------------------------------------------
    print("\nsub-swath spread:")
    for aoi, group in sorted(by_aoi.items()):
        counts = Counter(r["subswath"] for r in group if r["subswath"])
        print(f"  {aoi:<10} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print("\nESA burst ids (join key for burst ids read from SLC annotation):")
    for aoi, group in sorted(by_aoi.items()):
        ids = sorted({_int(r.get("esa_burst_id")) for r in group if r.get("esa_burst_id")})
        shown = ids if show_bursts else ids[:8]
        tail = "" if show_bursts or len(ids) <= 8 else f" ... +{len(ids) - 8} more"
        print(f"  {aoi:<10} {len(ids)} distinct: {shown}{tail}")

    # --- time span --------------------------------------------------------
    dates = sorted({r["acquired"] for r in rows if r["acquired"]})
    print(f"\nacquisition span: {dates[0]} to {dates[-1]} over {len(dates)} distinct dates")
    gaps = _repeat_gaps(dates)
    if gaps:
        common = gaps.most_common(3)
        print("  spacing between consecutive dates: "
              + ", ".join(f"{d}d x{n}" for d, n in common))


def _int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _repeat_gaps(dates: Sequence[str]) -> Counter:
    from datetime import date as _date

    parsed = []
    for text in dates:
        try:
            parsed.append(_date(*(int(p) for p in text.split("-"))))
        except Exception:
            continue
    return Counter((b - a).days for a, b in zip(parsed, parsed[1:]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="?", type=Path,
                    default=Path("data/l1_index/burst_join.csv"))
    ap.add_argument("--bursts", action="store_true",
                    help="print every ESA burst id rather than the first few")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"not found: {args.csv}\nrun tools/ga_l1_index.py first")
        return 1
    summarise(load(args.csv), show_bursts=args.bursts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
