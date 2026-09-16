#!/usr/bin/env python3
"""Build a matched GA NRB / Sentinel-1 GRD / Sentinel-1 SLC manifest for an AOI.

The point is a like-for-like comparison: the same acquisition seen three ways.

* **GA NRB** - terrain-corrected, CEOS-ARD gamma0, burst-level, from
  ``dea-public-data-dev``. Its STAC item names the SLC it was cut from, which
  is what ties the three together.
* **Sentinel-1 SLC** - the Level-1 input GA processed. The scene id comes free
  from the NRB metadata; where GA has no coverage it has to be found from the
  orbit instead.
* **Sentinel-1 GRD** - the Level-1 product most people actually use, from
  ``sentinel-s1-l1c``. Detected and multi-looked but *not* terrain corrected,
  which is exactly the difference worth measuring against the NRB.

A GRD and an SLC belong to the same acquisition when they share an absolute
orbit *and* a data-take id. That pairing is exact; the file names cannot be
derived from one another because the slice boundaries and the product CRC
differ, so it has to be matched, not constructed.

Usage::

    python tools/find_comparison_scenes.py --out manifest.json
    python tools/find_comparison_scenes.py --aoi pilliga --days 60
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vegmon import ga_nrb  # noqa: E402
from vegmon.s1grd import BUCKET_URL as GRD_BUCKET_URL  # noqa: E402
from vegmon.s1grd import GrdScene, SceneSearchError, find_scenes, list_day  # noqa: E402

#: Three areas in NSW with different clearing/regrowth character: intensive
#: agriculture and mining in the Hunter, dry woodland under active clearing
#: pressure in the Pilliga, and steep forested terrain in the Blue Mountains
#: where terrain correction matters most.
AOIS: Dict[str, dict] = {
    "hunter": {
        "label": "Hunter Valley (Singleton - Muswellbrook)",
        "bbox": (150.90, -32.80, 151.45, -32.25),
        "centre": (151.17, -32.53),
    },
    "pilliga": {
        "label": "Pilliga Forest (Narrabri - Baradine)",
        "bbox": (148.90, -31.10, 149.75, -30.35),
        "centre": (149.32, -30.72),
    },
    "bluemtns": {
        "label": "Blue Mountains (Katoomba - Blackheath)",
        "bbox": (150.10, -33.85, 150.60, -33.45),
        "centre": (150.35, -33.65),
    },
}

#: The SLC archive on AWS stopped being updated after 2022. Anything more
#: recent has a valid scene id but no open bucket to pull it from.
SLC_BUCKET = "sentinel1-slc"
SLC_REGION = "eu-west-1"
SLC_BUCKET_URL = f"https://{SLC_BUCKET}.s3.{SLC_REGION}.amazonaws.com"
SLC_LAST_YEAR = 2022

_TAKE_RE = re.compile(r"_(\d{6})_([0-9A-F]{6})_([0-9A-F]{4})$")


def _datatake(scene_id: str) -> Optional[tuple]:
    """(absolute orbit, data-take id) - the pair that identifies an acquisition."""
    match = _TAKE_RE.search(scene_id)
    return (match.group(1), match.group(2)) if match else None


@lru_cache(maxsize=512)
def _day_scenes(day: date, mode: str, pols: str) -> tuple:
    """One day of the global GRD archive, cached.

    A pass is cut into a dozen or more slices and every burst of that pass
    names the same few SLCs, so an uncached lookup re-lists the same day once
    per SLC. Over a Collection 0 time series that is the difference between
    minutes and hours.
    """
    try:
        return tuple(list_day(day, mode, pols))
    except Exception:
        return ()


def match_grd_for_slc(slc_id: str, pols: str = "DV") -> List[dict]:
    """GRD slices from the same acquisition as an SLC, closest in time first.

    Same day, same absolute orbit, same data-take. The slice whose start time
    equals the SLC's is the direct twin; the neighbours are the rest of the
    pass and are listed because a burst near a slice edge may need them.
    """
    take = _datatake(slc_id)
    if take is None:
        return []
    orbit, take_id = take
    stamp = re.search(r"(\d{8})T(\d{6})", slc_id)
    if not stamp:
        return []
    day = datetime.strptime(stamp.group(1), "%Y%m%d").date()
    slc_start = stamp.group(2)

    scenes = _day_scenes(day, "IW", pols)

    out = []
    for scene in scenes:
        scene_take = _datatake(scene.scene_id)
        if scene_take != (orbit, take_id):
            continue
        grd_start = scene.start_time.strftime("%H%M%S")
        out.append(
            {
                "scene_id": scene.scene_id,
                "start_time": scene.start_time.isoformat(),
                "relative_orbit": scene.relative_orbit,
                "url": f"{GRD_BUCKET_URL}/{scene.prefix}",
                "exact_twin": grd_start == slc_start,
                "seconds_from_slc": abs(
                    int(grd_start[:2]) * 3600 + int(grd_start[2:4]) * 60 + int(grd_start[4:])
                    - (int(slc_start[:2]) * 3600 + int(slc_start[2:4]) * 60 + int(slc_start[4:]))
                ),
            }
        )
    return sorted(out, key=lambda s: s["seconds_from_slc"])


def slc_url(slc_id: str) -> Optional[str]:
    """Where the SLC SAFE lives, if the open archive still carries that year."""
    stamp = re.search(r"(\d{4})(\d{2})(\d{2})T", slc_id)
    if not stamp:
        return None
    year = int(stamp.group(1))
    if year > SLC_LAST_YEAR:
        return None
    return f"{SLC_BUCKET_URL}/{year}/{stamp.group(2)}/{stamp.group(3)}/IW/{slc_id}/"


def survey_ga(names: Sequence[str], db_path: Optional[str]) -> Dict[str, List[ga_nrb.Burst]]:
    aois = {n: AOIS[n]["bbox"] for n in names}
    return ga_nrb.survey_coverage(aois, db_path=db_path)


def build(names: Sequence[str], days: int, db_path: Optional[str], verbose: bool = True) -> dict:
    coverage = survey_ga(names, db_path)
    manifest: Dict[str, dict] = {}

    for name in names:
        aoi = AOIS[name]
        lon, lat = aoi["centre"]
        bursts = coverage.get(name, [])
        entry: Dict[str, object] = {
            "label": aoi["label"],
            "bbox": list(aoi["bbox"]),
            "centre": [lon, lat],
            "ga_nrb": {
                "covered": bool(bursts),
                "product": ga_nrb.PRODUCTS["vv_vh"],
                "bursts": [
                    {"burst_id": b.burst_id, "track": b.track, "bbox": [round(v, 4) for v in b.bbox]}
                    for b in sorted(bursts, key=lambda b: b.burst_id)
                ],
                "acquisitions": [],
            },
            "sentinel1": {},
        }
        if verbose:
            print(f"\n=== {name}: {aoi['label']} ===", flush=True)
            print(f"  GA NRB bursts covering AOI: {len(bursts)}", flush=True)

        # --- GA acquisitions, and the SLC ids they name -------------------
        slc_ids: List[str] = []
        for burst in sorted(bursts, key=lambda b: b.burst_id):
            acqs = ga_nrb.load_items(ga_nrb.acquisitions(burst.burst_id))
            for acq in acqs:
                record = {
                    "burst_id": acq.burst_id,
                    "datetime": acq.datetime_key,
                    "scene_id": acq.scene_id,
                    "absolute_orbit": acq.absolute_orbit,
                    "relative_orbit": acq.relative_orbit,
                    "orbit_state": acq.orbit_state,
                    "prefix": f"{ga_nrb.BUCKET_URL}/{acq.prefix}",
                    "assets": acq.assets,
                }
                entry["ga_nrb"]["acquisitions"].append(record)
                if acq.scene_id:
                    slc_ids.append(acq.scene_id)
            if verbose and acqs:
                print(f"    {burst.burst_id}: {len(acqs)} acquisition(s)", flush=True)

        slc_ids = sorted(set(slc_ids))
        entry["ga_nrb"]["slc_scene_ids"] = slc_ids

        # --- Sentinel-1 Level-1 for the same acquisitions ------------------
        pairs = []
        for slc in slc_ids:
            grds = match_grd_for_slc(slc)
            twin = next((g for g in grds if g["exact_twin"]), grds[0] if grds else None)
            pairs.append(
                {
                    "slc_scene_id": slc,
                    "slc_url": slc_url(slc),
                    "slc_note": None if slc_url(slc) else
                        f"open SLC archive stops at {SLC_LAST_YEAR}; use CDSE or ASF for this one",
                    "grd_twin": twin,
                    "grd_pass_slices": len(grds),
                }
            )
        entry["sentinel1"]["matched_to_ga"] = pairs
        if verbose and pairs:
            for p in pairs:
                twin = p["grd_twin"]["scene_id"] if p["grd_twin"] else "none found"
                print(f"    SLC {p['slc_scene_id']}\n      GRD {twin}", flush=True)

        # --- Independent GRD search, for AOIs GA does not cover ------------
        end = date.today()
        start = end - timedelta(days=days)
        try:
            if verbose:
                print(f"  scanning GRD archive {start}..{end} for the AOI centre", flush=True)
            scenes = find_scenes(lon, lat, start, end, progress=verbose)
            entry["sentinel1"]["independent_grd"] = {
                "relative_orbit": scenes[0].relative_orbit,
                "window": [start.isoformat(), end.isoformat()],
                "scenes": [
                    {
                        "scene_id": s.scene_id,
                        "acquired": s.acquired.isoformat(),
                        "start_time": s.start_time.isoformat(),
                        "absolute_orbit": s.absolute_orbit,
                        "relative_orbit": s.relative_orbit,
                        "mission": s.mission,
                        "url": f"{GRD_BUCKET_URL}/{s.prefix}",
                    }
                    for s in scenes
                ],
            }
            if verbose:
                print(f"    {len(scenes)} scenes on relative orbit "
                      f"{scenes[0].relative_orbit}", flush=True)
        except SceneSearchError as exc:
            entry["sentinel1"]["independent_grd"] = {"error": str(exc)}
            if verbose:
                print(f"    no coverage: {exc}", flush=True)

        manifest[name] = entry

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "ga_nrb_bucket": ga_nrb.BUCKET_URL,
        "grd_bucket": GRD_BUCKET_URL,
        "slc_bucket": SLC_BUCKET_URL,
        "aois": manifest,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aoi", nargs="*", default=list(AOIS), choices=list(AOIS))
    ap.add_argument("--days", type=int, default=60,
                    help="how far back to scan the GRD archive (default 60)")
    ap.add_argument("--burst-db", default=None, help="path to a cached OPERA burst db")
    ap.add_argument("--out", default="comparison_manifest.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    manifest = build(args.aoi, args.days, args.burst_db, verbose=not args.quiet)
    Path(args.out).write_text(json.dumps(manifest, indent=2))
    if not args.quiet:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
