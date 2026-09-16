"""Sentinel-1 GRD from the raw SAFE products on AWS, calibrated in-process.

Every analysis-ready Sentinel-1 product this pipeline would rather use - the
Planetary Computer RTC collection, CDSE, ASF - sits behind a host that a
restrictive egress policy blocks, and the one public RTC bucket on AWS
(``sentinel-s1-rtc-indigo``) covers UTM zones 10-19, which is the continental
United States. Australia has no analysis-ready option there at all.

What *is* open, anonymous and complete is the Level-1 GRD archive in
``s3://sentinel-s1-l1c``: full SAFE products, measurement GeoTIFFs and the
annotation needed to calibrate them. That is enough. This module does the
short version of the processing chain:

1. **Find the scenes.** There is no search API, so coverage is worked out from
   the orbit: one repeat cycle is scanned against each scene's ``productInfo``
   footprint to learn which relative orbits see the AOI, and everything after
   that follows the 12-day repeat.
2. **Calibrate.** ``gamma0 = (DN^2 - noise) / A^2`` with the per-product
   calibration and thermal-noise LUTs, applied in radar geometry where they
   are defined.
3. **Geocode.** The measurement TIFFs carry the 210-point geolocation grid as
   GCPs, so the warp to a projected grid is a GCP warp, done at the native
   10 m and then power-averaged to the working resolution - which is where the
   looks come from.

What this does **not** do is radiometric terrain correction. That matters less
than it sounds here, for a specific reason: the detector compares two
acquisitions **from the same relative orbit**, so the viewing geometry, the
local incidence angle and the terrain-driven part of the backscatter are the
same in both and divide out of the difference. Ellipsoid gamma0 from a fixed
orbit is a sound basis for change detection. It is *not* a sound basis for
comparing absolute levels between orbits or against another sensor, and this
module refuses to mix orbits by default for that reason.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import re
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

import numpy as np

from vegmon.grid import Grid
from vegmon.series import RadarSeries

BUCKET = "sentinel-s1-l1c"
REGION = "eu-central-1"
BUCKET_URL = f"https://{BUCKET}.s3.{REGION}.amazonaws.com"
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

#: Sentinel-1 repeats its ground track every 12 days. With Sentinel-1B lost in
#: December 2021 that is also the revisit for most of the useful record.
REPEAT_DAYS = 12

#: Cycle length in relative orbits, and the mission-specific offsets that turn
#: an absolute orbit number into a relative one. The offsets are per-satellite
#: and not guessable from each other: each was checked against acquisitions
#: whose relative orbit is stated independently (Geoscience Australia's NRB
#: metadata for S1A/S1B/S1C, and footprint-to-burst-database matching for
#: S1D, which no published NRB covers yet).
_CYCLE = 175
_ORBIT_OFFSET = {"S1A": 73, "S1B": 27, "S1C": 172, "S1D": 42}

_SCENE_RE = re.compile(
    r"(?P<mission>S1[ABCD])_(?P<mode>\w{2})_(?P<ptype>GRD[HM])_1S(?P<pols>\w{2})_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_(?P<absorbit>\d{6})_"
    r"(?P<takeid>\w{6})_(?P<uid>\w{4})"
)


class SceneSearchError(RuntimeError):
    """No Sentinel-1 coverage could be found for the request."""


@dataclass(frozen=True)
class GrdScene:
    prefix: str
    scene_id: str
    acquired: date
    start_time: datetime
    absolute_orbit: int
    relative_orbit: int
    mission: str
    pass_direction: str = "unknown"

    def asset_url(self, path: str) -> str:
        return f"{BUCKET_URL}/{self.prefix}{path}"

    def measurement_url(self, pol: str) -> str:
        return self.asset_url(f"measurement/iw-{pol.lower()}.tiff")

    @property
    def orbit_state(self) -> str:
        return self.pass_direction.lower()


def relative_orbit(scene_id: str) -> int:
    """Relative orbit (1-175) from a scene id's absolute orbit number."""
    match = _SCENE_RE.search(scene_id)
    if not match:
        raise ValueError(f"not a Sentinel-1 scene id: {scene_id!r}")
    mission = match["mission"]
    if mission not in _ORBIT_OFFSET:
        # Falling back to another satellite's offset would not fail loudly, it
        # would quietly label the scene with the wrong track and the caller
        # would mix orbits without knowing.
        raise ValueError(f"no orbit offset known for {mission}: {scene_id!r}")
    return ((int(match["absorbit"]) - _ORBIT_OFFSET[mission]) % _CYCLE) + 1


def _scene_from_prefix(prefix: str) -> Optional[GrdScene]:
    scene_id = prefix.rstrip("/").split("/")[-1]
    match = _SCENE_RE.search(scene_id)
    if not match:
        return None
    start = datetime.strptime(match["start"], "%Y%m%dT%H%M%S")
    return GrdScene(
        prefix=prefix,
        scene_id=scene_id,
        acquired=start.date(),
        start_time=start,
        absolute_orbit=int(match["absorbit"]),
        relative_orbit=relative_orbit(scene_id),
        mission=match["mission"],
    )


def _list_prefixes(prefix: str, timeout: float = 60.0) -> List[str]:
    out: List[str] = []
    token: Optional[str] = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET_URL}/?{urlencode(query)}", timeout=timeout) as r:
            root = ElementTree.fromstring(r.read())
        out.extend(
            n.findtext(f"{_S3_NS}Prefix", "") for n in root.findall(f"{_S3_NS}CommonPrefixes")
        )
        if root.findtext(f"{_S3_NS}IsTruncated", "false") != "true":
            return out
        token = root.findtext(f"{_S3_NS}NextContinuationToken")


def list_day(day: date, mode: str = "IW", pols: str = "DV") -> List[GrdScene]:
    """Every GRD scene archived for one day in a mode/polarisation combination."""
    prefix = f"GRD/{day.year}/{day.month}/{day.day}/{mode}/{pols}/"
    scenes = [_scene_from_prefix(p) for p in _list_prefixes(prefix)]
    return [s for s in scenes if s is not None]


def _product_info(scene: GrdScene, timeout: float = 40.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(scene.asset_url("productInfo.json"), timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _covers(scene: GrdScene, lon: float, lat: float) -> Optional[GrdScene]:
    """Test a scene's published footprint against a point."""
    from shapely.geometry import Point, shape

    info = _product_info(scene)
    if not info or "footprint" not in info:
        return None
    try:
        if not shape(info["footprint"]).contains(Point(lon, lat)):
            return None
    except Exception:
        return None
    return GrdScene(**{**scene.__dict__})


def overpass_bands(lon: float, width_hours: float = 2.0) -> List[Tuple[int, int]]:
    """UTC hour ranges in which Sentinel-1 can see a given longitude.

    Sentinel-1 is sun-synchronous with an 18:00 local-solar-time ascending
    node, so a longitude is only ever imaged in two narrow windows a day.
    Filtering the day's ~440 scenes down to those windows is what makes the
    coverage search cheap enough to run without an API behind it.
    """
    solar_offset = lon / 15.0
    bands = []
    for local_hour in (18.0, 6.0):
        centre = (local_hour - solar_offset) % 24.0
        bands.append(
            (
                int(np.floor((centre - width_hours / 2) % 24)),
                int(np.floor((centre + width_hours / 2) % 24)),
            )
        )
    return bands


def _in_bands(scene: GrdScene, bands: Sequence[Tuple[int, int]]) -> bool:
    hour = scene.start_time.hour
    for low, high in bands:
        if low <= high:
            if low <= hour <= high:
                return True
        elif hour >= low or hour <= high:  # band wraps midnight
            return True
    return False


def find_relative_orbits(
    lon: float,
    lat: float,
    seed_start: date,
    cycle_days: int = REPEAT_DAYS,
    mode: str = "IW",
    pols: str = "DV",
    workers: int = 20,
    progress: bool = False,
) -> List[GrdScene]:
    """Scan one repeat cycle to learn which relative orbits see a point.

    Costs a few hundred metadata fetches, once. Everything afterwards follows
    the 12-day repeat and needs only a handful.
    """
    bands = overpass_bands(lon)
    candidates: List[GrdScene] = []
    for offset in range(cycle_days):
        day = seed_start + timedelta(days=offset)
        try:
            candidates.extend(s for s in list_day(day, mode, pols) if _in_bands(s, bands))
        except Exception:
            continue
    if progress:
        print(f"  {len(candidates)} candidate scenes in the overpass windows", flush=True)
    if not candidates:
        raise SceneSearchError(
            f"no {mode}/{pols} Sentinel-1 scenes archived between {seed_start} and "
            f"{seed_start + timedelta(days=cycle_days)}"
        )

    hits: List[GrdScene] = []
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for found in pool.map(lambda s: _covers(s, lon, lat), candidates):
            if found is not None:
                hits.append(found)
    if not hits:
        raise SceneSearchError(
            f"no Sentinel-1 {mode}/{pols} scene covering {lat:.4f},{lon:.4f} in the "
            f"repeat cycle starting {seed_start}"
        )
    return sorted(hits, key=lambda s: s.start_time)


def find_scenes(
    lon: float,
    lat: float,
    start: date,
    end: date,
    relative_orbit_number: Optional[int] = None,
    mode: str = "IW",
    pols: str = "DV",
    workers: int = 20,
    progress: bool = False,
) -> List[GrdScene]:
    """Every acquisition of one relative orbit covering a point in a date range.

    Pinning a single relative orbit is not a convenience. Ascending and
    descending passes view the canopy from opposite sides and differ over the
    same intact woodland by more than the drop the detector is looking for, and
    without terrain correction the local incidence angle differs too. Mixing
    them would put that difference straight into the change signal.
    """
    seed = find_relative_orbits(lon, lat, start, mode=mode, pols=pols,
                                workers=workers, progress=progress)
    if relative_orbit_number is None:
        relative_orbit_number = seed[0].relative_orbit
    on_orbit = [s for s in seed if s.relative_orbit == relative_orbit_number]
    if not on_orbit:
        available = sorted({s.relative_orbit for s in seed})
        raise SceneSearchError(
            f"relative orbit {relative_orbit_number} does not cover this point; "
            f"available: {available}"
        )
    anchor = on_orbit[0]
    if progress:
        print(
            f"  anchored on relative orbit {relative_orbit_number} "
            f"({anchor.start_time:%H:%M} UTC), stepping {REPEAT_DAYS} days",
            flush=True,
        )

    # Walk the repeat cycle outwards from the anchor in both directions.
    days: List[date] = []
    step = anchor.acquired
    while step >= start:
        days.append(step)
        step -= timedelta(days=REPEAT_DAYS)
    step = anchor.acquired + timedelta(days=REPEAT_DAYS)
    while step <= end:
        days.append(step)
        step += timedelta(days=REPEAT_DAYS)
    days = sorted(d for d in days if start <= d <= end)

    def candidates_for(day: date) -> List[GrdScene]:
        try:
            return [
                s
                for s in list_day(day, mode, pols)
                if s.relative_orbit == relative_orbit_number
            ]
        except Exception:
            return []

    found: List[GrdScene] = []
    with futures.ThreadPoolExecutor(max_workers=min(workers, 12)) as pool:
        per_day = list(pool.map(candidates_for, days))
    flat = [s for group in per_day for s in group]
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for hit in pool.map(lambda s: _covers(s, lon, lat), flat):
            if hit is not None:
                found.append(hit)

    # A pass can be archived as several consecutive slices; keep one per date.
    unique: Dict[date, GrdScene] = {}
    for scene in sorted(found, key=lambda s: s.start_time):
        unique.setdefault(scene.acquired, scene)
    if not unique:
        raise SceneSearchError(
            f"relative orbit {relative_orbit_number} yielded no scenes between "
            f"{start} and {end}"
        )
    return [unique[k] for k in sorted(unique)]


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


@dataclass
class Lut:
    """A calibration or noise look-up table on its own coarse (line, pixel) grid."""

    lines: np.ndarray
    pixels: np.ndarray
    values: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.values.shape


def _fetch(url: str, timeout: float = 90.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def parse_calibration(xml: bytes, quantity: str = "gamma") -> Lut:
    """Read a calibration LUT. ``quantity`` is sigmaNought, betaNought or gamma.

    gamma0 is the default: it is the convention for vegetation backscatter, and
    it already divides out the cosine of the incidence angle that otherwise
    puts a range-dependent ramp across every scene.
    """
    root = ElementTree.fromstring(xml)
    vectors = root.findall(".//calibrationVector")
    if not vectors:
        raise ValueError("calibration annotation carries no vectors")
    lines = np.array([int(v.findtext("line")) for v in vectors], dtype=np.int64)
    pixels = np.array(vectors[0].findtext("pixel").split(), dtype=np.int64)
    values = np.array(
        [np.array(v.findtext(quantity).split(), dtype=np.float32) for v in vectors]
    )
    return Lut(lines=lines, pixels=pixels, values=values.astype(np.float32))


def parse_noise(xml: bytes) -> Optional[Lut]:
    """Read the thermal-noise LUT, in DN-squared units.

    Noise removal matters for the cross-polarised channel specifically: VH over
    woodland sits only a few decibels above the noise floor, and the floor
    ramps across the swath. Leaving it in puts a range-dependent bias into
    exactly the band the detector reads.
    """
    root = ElementTree.fromstring(xml)
    vectors = root.findall(".//noiseRangeVector")
    key = "noiseRangeLut"
    if not vectors:  # older baselines
        vectors = root.findall(".//noiseVector")
        key = "noiseLut"
    if not vectors:
        return None
    lines = np.array([int(v.findtext("line")) for v in vectors], dtype=np.int64)
    pixels = np.array(vectors[0].findtext("pixel").split(), dtype=np.int64)
    values = np.array(
        [np.array(v.findtext(key).split(), dtype=np.float32) for v in vectors]
    )
    return Lut(lines=lines, pixels=pixels, values=values.astype(np.float32))


def _lut_gcps(lut: Lut, gcps, height: int, width: int):
    """Re-express a scene's GCPs in the LUT's own coarse index space.

    The LUT is defined on a subsampled (line, pixel) grid. Rescaling the GCPs
    onto that grid lets the LUT be warped to the map with exactly the same
    geometry as the image, without ever building it at full resolution.
    """
    from rasterio.control import GroundControlPoint

    line_index = np.interp(
        [g.row for g in gcps], lut.lines, np.arange(lut.lines.size, dtype=np.float64)
    )
    pixel_index = np.interp(
        [g.col for g in gcps], lut.pixels, np.arange(lut.pixels.size, dtype=np.float64)
    )
    return [
        GroundControlPoint(row=float(r), col=float(c), x=g.x, y=g.y, z=g.z)
        for r, c, g in zip(line_index, pixel_index, gcps)
    ]


def _warp(source, gcps, gcp_crs, grid: Grid, resampling_name: str, dtype="float32"):
    import numpy as _np
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = _np.zeros(grid.shape, dtype=dtype)
    reproject(
        source=source,
        destination=destination,
        src_crs=gcp_crs,
        gcps=gcps,
        src_nodata=0,
        dst_transform=grid.rasterio_transform(),
        dst_crs=grid.crs,
        dst_nodata=0,
        resampling=getattr(Resampling, resampling_name),
        num_threads=2,
    )
    return destination


def read_scene(
    scene: GrdScene,
    grid: Grid,
    pol: str = "vh",
    quantity: str = "gamma",
    remove_noise: bool = True,
    native_resolution: float = 10.0,
) -> np.ndarray:
    """Calibrated backscatter in dB for one scene, on ``grid``.

    The warp is done at the product's native 10 m and the result is averaged
    down to the working grid **in linear power**, not in decibels. Averaging
    decibels would be averaging logarithms, which biases every multi-looked
    pixel low and is one of the easier ways to get a plausible-looking and
    wrong backscatter image.
    """
    import rasterio

    fine = _fine_grid(grid, native_resolution)
    with rasterio.open(scene.measurement_url(pol)) as src:
        gcps, gcp_crs = src.gcps
        if not gcps:
            raise ValueError(f"{scene.scene_id} carries no geolocation GCPs")
        height, width = src.height, src.width
        digital_number = _warp(
            rasterio.band(src, 1), gcps, gcp_crs, fine, "nearest", dtype="float32"
        )

    power = digital_number.astype(np.float32) ** 2
    valid = digital_number > 0

    calibration = parse_calibration(_fetch(scene.asset_url(
        f"annotation/calibration/calibration-iw-{pol.lower()}.xml")), quantity)
    gain = _warp(calibration.values, _lut_gcps(calibration, gcps, height, width),
                 gcp_crs, fine, "bilinear")

    if remove_noise:
        noise_lut = parse_noise(_fetch(scene.asset_url(
            f"annotation/calibration/noise-iw-{pol.lower()}.xml")))
        if noise_lut is not None:
            noise = _warp(noise_lut.values, _lut_gcps(noise_lut, gcps, height, width),
                          gcp_crs, fine, "bilinear")
            power = power - noise

    with np.errstate(invalid="ignore", divide="ignore"):
        backscatter = np.where(
            valid & (gain > 0), power / np.square(gain, dtype=np.float32), np.nan
        )
    # Thermal-noise subtraction can drive a genuinely dark pixel negative.
    backscatter = np.where(backscatter > 0, backscatter, np.nan).astype(np.float32)

    aggregated = _block_mean_power(backscatter, grid, fine)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (10.0 * np.log10(aggregated)).astype(np.float32)


def _fine_grid(grid: Grid, native_resolution: float) -> Grid:
    """The same extent as ``grid``, at the product's native pixel size."""
    factor = max(1, int(round(grid.resolution[0] / native_resolution)))
    if factor == 1:
        return grid
    return Grid(
        crs=grid.crs,
        transform=(
            grid.transform[0] / factor, 0.0, grid.transform[2],
            0.0, grid.transform[4] / factor, grid.transform[5],
        ),
        width=grid.width * factor,
        height=grid.height * factor,
    )


def _block_mean_power(values: np.ndarray, grid: Grid, fine: Grid) -> np.ndarray:
    """Average linear power over each output cell - where the looks come from."""
    factor = fine.width // grid.width
    if factor <= 1:
        return values
    blocks = values.reshape(grid.height, factor, grid.width, factor)
    with np.errstate(invalid="ignore"):
        counts = np.sum(np.isfinite(blocks), axis=(1, 3))
        totals = np.nansum(blocks, axis=(1, 3))
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan).astype(np.float32)


def load_sentinel1(
    grid: Grid,
    lon: float,
    lat: float,
    start: date,
    end: date,
    relative_orbit_number: Optional[int] = None,
    workers: int = 6,
    progress: bool = False,
    quantity: str = "gamma",
) -> RadarSeries:
    """Build a :class:`RadarSeries` for an AOI from raw GRD products."""
    scenes = find_scenes(lon, lat, start, end,
                         relative_orbit_number=relative_orbit_number,
                         workers=20, progress=progress)
    if progress:
        print(f"  {len(scenes)} Sentinel-1 acquisitions on relative orbit "
              f"{scenes[0].relative_orbit}; calibrating...", flush=True)

    def process(scene: GrdScene):
        try:
            vh = read_scene(scene, grid, "vh", quantity=quantity)
            vv = read_scene(scene, grid, "vv", quantity=quantity)
        except Exception as exc:
            return scene, None, None, exc
        return scene, vv, vh, None

    kept: List[Tuple[GrdScene, np.ndarray, np.ndarray]] = []
    failures = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (scene, vv, vh, error) in enumerate(pool.map(process, scenes)):
            if error is not None or vv is None:
                failures += 1
                continue
            kept.append((scene, vv, vh))
            if progress and (index + 1) % 10 == 0:
                print(f"    {index + 1}/{len(scenes)} scenes", flush=True)
    if not kept:
        raise SceneSearchError("every Sentinel-1 scene failed to calibrate")
    if progress and failures:
        print(f"  ({failures} scenes skipped)", flush=True)

    kept.sort(key=lambda item: item[0].start_time)
    return RadarSeries(
        grid=grid,
        times=[s.acquired for s, _, _ in kept],
        vv_db=np.stack([vv for _, vv, _ in kept]).astype(np.float32),
        vh_db=np.stack([vh for _, _, vh in kept]).astype(np.float32),
        orbit_state=["ascending" if s.start_time.hour < 12 else "descending"
                     for s, _, _ in kept],
        sensor=f"aws-sentinel-s1-l1c:GRD:rel{kept[0][0].relative_orbit}",
    )


__all__ = [
    "BUCKET_URL",
    "GrdScene",
    "Lut",
    "SceneSearchError",
    "find_relative_orbits",
    "find_scenes",
    "list_day",
    "load_sentinel1",
    "overpass_bands",
    "parse_calibration",
    "parse_noise",
    "read_scene",
    "relative_orbit",
]
