#!/usr/bin/env python3
"""Index which Sentinel-1 Level-1 products each GA NRB burst came from.

For a workflow where the GRD and SLC archive already exists elsewhere - an HPC
filesystem, say - there is no reason to download Level-1 again. What is needed
is the join: for every GA NRB burst, the name of the SLC it was cut from and
the GRD covering the same acquisition, so those files can be located in the
existing archive.

The join is exact rather than inferred:

* **NRB -> SLC** is published. GA's STAC item carries ``sarard:scene_id``,
  which is the SLC name verbatim.
* **SLC -> GRD** is matched on absolute orbit *and* data-take id. The file
  names cannot be derived from one another - slice boundaries and the trailing
  product CRC differ - so the GRD archive is listed once per acquisition date
  and the twin picked out by its start time.
* **NRB burst -> ESA burst** is arithmetic. GA's ``t009_019126_iw2`` is
  track 009, ESA burst 19126, sub-swath IW2; the middle field is the ESA burst
  id, 1..375887, three sub-swath rows each across the 1,127,661-row OPERA
  database. So a burst can be tied to a specific sub-swath of a specific SLC,
  not just to the scene.

A pass is archived as several GRD slices. Most bursts sit inside one, but a
burst near a slice boundary needs its neighbour too, so both the twin and the
full slice list are written.

Outputs, into ``--out``:

============================  =================================================
``burst_join.csv``            one row per GA burst-acquisition: GA product,
                              its SLC, and the GRD covering it
``scene_pairs.csv``           one row per acquisition: SLC to GRD on its own,
                              independent of any GA product
``scenes_slc.txt``            unique SLC scene ids, one per line
``scenes_grd.txt``            unique GRD twin scene ids, one per line
``scenes_grd_all_slices.txt`` every GRD slice of every matched pass
============================  =================================================

Usage::

    python tools/ga_l1_index.py --aoi hunter --track 9 --out data/index_hunter
    python tools/ga_l1_index.py --aoi hunter pilliga bluemtns --track 9 --dates 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from find_comparison_scenes import AOIS, match_grd_for_slc  # noqa: E402
from find_ga_c0_scenes import (  # noqa: E402
    C0_END,
    C0_START,
    CATALOG,
    COLLECTIONS_C0,
    CatalogUnreachable,
    open_catalog,
    search_aoi,
)

FIELDS = [
    "aoi",
    "ga_burst_id",
    "esa_burst_id",
    "track",
    "subswath",
    "acquired",
    "datetime",
    "ga_item_id",
    "ga_collection",
    "platform",
    "absolute_orbit",
    "datatake",
    "orbit_state",
    "slc_scene_id",
    "grd_scene_id",
    "grd_slice_count",
    "grd_all_slices",
]

#: Scene-level schema: the SLC-to-GRD pairing on its own.
PAIR_FIELDS = [
    "slc_scene_id",
    "grd_scene_id",
    "grd_slice_count",
    "grd_all_slices",
    "acquired",
    "platform",
    "track",
    "absolute_orbit",
    "datatake",
    "orbit_state",
    "aois",
    "ga_burst_count",
]


def decompose_burst(burst_id: str) -> Dict[str, object]:
    """``t009_019126_iw2`` -> track 9, ESA burst 19126, sub-swath IW2."""
    parts = (burst_id or "").split("_")
    if len(parts) != 3:
        return {"track": "", "esa_burst_id": "", "subswath": ""}
    return {
        "track": int(parts[0].lstrip("t")),
        "esa_burst_id": int(parts[1]),
        "subswath": parts[2].upper(),
    }


def _datatake_of(scene_id: str) -> str:
    bits = (scene_id or "").split("_")
    return bits[-2] if len(bits) >= 2 else ""


def rows_for_aoi(
    client,
    aoi_name: str,
    track: Optional[int],
    start: str,
    end: str,
    dates: Optional[int] = None,
    verbose: bool = True,
) -> List[dict]:
    items = search_aoi(client, AOIS[aoi_name]["bbox"], start, end,
                       COLLECTIONS_C0, track=track)
    if dates is not None:
        keep = sorted({(i.get("properties", {}).get("datetime") or "")[:10] for i in items})[:dates]
        items = [i for i in items
                 if (i.get("properties", {}).get("datetime") or "")[:10] in keep]

    slcs = sorted({i.get("properties", {}).get("sarard:scene_id") for i in items} - {None})
    if verbose:
        n_dates = len({(i.get("properties", {}).get("datetime") or "")[:10] for i in items})
        print(f"\n=== {aoi_name} ===")
        print(f"  {len(items)} NRB bursts over {n_dates} dates, {len(slcs)} distinct SLCs")
        print(f"  matching GRD twins (one archive-day listing per date)...", flush=True)

    # One lookup per SLC; the day listings behind it are cached.
    grd_for_slc: Dict[str, dict] = {}
    for slc in slcs:
        matches = match_grd_for_slc(slc)
        twin = next((m for m in matches if m["exact_twin"]), matches[0] if matches else None)
        grd_for_slc[slc] = {"twin": twin, "all": matches}

    matched = sum(1 for v in grd_for_slc.values() if v["twin"])
    if verbose:
        print(f"  {matched}/{len(slcs)} SLCs matched to a GRD twin", flush=True)

    rows: List[dict] = []
    for item in items:
        props = item.get("properties", {})
        burst = props.get("sarard:burst_id", "")
        slc = props.get("sarard:scene_id", "")
        pairing = grd_for_slc.get(slc, {"twin": None, "all": []})
        twin = pairing["twin"]
        row = {
            "aoi": aoi_name,
            "ga_burst_id": burst,
            "acquired": (props.get("datetime") or "")[:10],
            "datetime": props.get("datetime", ""),
            "ga_item_id": item.get("id", ""),
            "ga_collection": item.get("collection", ""),
            "platform": props.get("platform", ""),
            "absolute_orbit": props.get("sat:absolute_orbit", ""),
            "datatake": _datatake_of(slc),
            "orbit_state": props.get("sat:orbit_state", ""),
            "slc_scene_id": slc,
            "grd_scene_id": twin["scene_id"] if twin else "",
            "grd_slice_count": len(pairing["all"]),
            "grd_all_slices": "|".join(m["scene_id"] for m in pairing["all"]),
        }
        row.update(decompose_burst(burst))
        rows.append(row)

    rows.sort(key=lambda r: (r["acquired"], r["ga_burst_id"]))
    return rows


def scene_pairs(rows: Sequence[dict]) -> List[dict]:
    """Collapse the burst rows to one row per acquisition: SLC to GRD.

    Useful on its own. Many GA bursts share one SLC, so the burst table repeats
    the same pairing dozens of times; this is the distinct list, and it stands
    without reference to any GA product.
    """
    pairs: "OrderedDict[str, dict]" = OrderedDict()
    for row in rows:
        slc = row.get("slc_scene_id")
        if not slc:
            continue
        if slc not in pairs:
            pairs[slc] = {
                "slc_scene_id": slc,
                "grd_scene_id": row.get("grd_scene_id", ""),
                "grd_slice_count": row.get("grd_slice_count", 0),
                "grd_all_slices": row.get("grd_all_slices", ""),
                "acquired": row.get("acquired", ""),
                "platform": row.get("platform", ""),
                "absolute_orbit": row.get("absolute_orbit", ""),
                "datatake": row.get("datatake", ""),
                "track": row.get("track", ""),
                "orbit_state": row.get("orbit_state", ""),
                "aois": set(),
                "ga_burst_count": 0,
            }
        pairs[slc]["ga_burst_count"] += 1
        if row.get("aoi"):
            pairs[slc]["aois"].add(row["aoi"])
    out = []
    for pair in pairs.values():
        pair["aois"] = "|".join(sorted(pair["aois"]))
        out.append(pair)
    return out


def write_outputs(rows: Sequence[dict], out: Path) -> Dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    paths["csv"] = out / "burst_join.csv"
    with open(paths["csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})

    pairs = scene_pairs(rows)
    paths["pairs"] = out / "scene_pairs.csv"
    with open(paths["pairs"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({k: pair.get(k, "") for k in PAIR_FIELDS})

    # OrderedDict rather than set so the lists stay in acquisition order.
    slcs = list(OrderedDict.fromkeys(r["slc_scene_id"] for r in rows if r["slc_scene_id"]))
    twins = list(OrderedDict.fromkeys(r["grd_scene_id"] for r in rows if r["grd_scene_id"]))
    every = list(OrderedDict.fromkeys(
        s for r in rows for s in (r["grd_all_slices"] or "").split("|") if s
    ))

    for key, name, values in (
        ("slc", "scenes_slc.txt", slcs),
        ("grd", "scenes_grd.txt", twins),
        ("grd_all", "scenes_grd_all_slices.txt", every),
    ):
        paths[key] = out / name
        paths[key].write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")

    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aoi", nargs="*", default=list(AOIS), choices=list(AOIS))
    ap.add_argument("--track", type=int, default=None,
                    help="restrict to one relative orbit (9 covers all three AOIs)")
    ap.add_argument("--start", default=C0_START)
    ap.add_argument("--end", default=C0_END)
    ap.add_argument("--dates", type=int, default=None,
                    help="only the first N dates")
    ap.add_argument("--out", type=Path, default=Path("data/l1_index"))
    ap.add_argument("--catalog", default=CATALOG)
    args = ap.parse_args()

    try:
        client = open_catalog(args.catalog)
    except (CatalogUnreachable, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rows: List[dict] = []
    for name in args.aoi:
        rows += rows_for_aoi(client, name, args.track, args.start, args.end,
                             dates=args.dates)

    if not rows:
        print("no NRB bursts matched the request", file=sys.stderr)
        return 1

    paths = write_outputs(rows, args.out)
    unmatched = [r for r in rows if not r["grd_scene_id"]]

    print(f"\n{len(rows)} burst-acquisitions indexed")
    print(f"  {len(set(r['slc_scene_id'] for r in rows))} distinct SLCs")
    print(f"  {len(set(r['grd_scene_id'] for r in rows if r['grd_scene_id']))} distinct GRD twins")
    print(f"  {len(scene_pairs(rows))} SLC/GRD acquisition pairs")
    if unmatched:
        print(f"  {len(unmatched)} burst(s) with no GRD match - see blank column in the CSV")
    print()
    for key in ("csv", "pairs", "slc", "grd", "grd_all"):
        print(f"  {paths[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
