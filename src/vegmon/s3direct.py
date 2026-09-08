"""Sentinel-2 L2A straight from the AWS open-data bucket, with no STAC API.

The usual way to find Sentinel-2 scenes is a STAC search. That fails in a lot
of the places this pipeline is actually useful: government and corporate
networks routinely allow the AWS data buckets while blocking the third-party
STAC endpoints that index them, and a STAC outage takes the whole workflow
down with it.

The AWS ``sentinel-cogs`` bucket does not need an index. Its keys are
deterministic::

    sentinel-s2-l2a-cogs/{zone}/{band}/{square}/{year}/{month}/{scene}/B04.tif

so scenes can be enumerated with anonymous S3 ``ListObjectsV2`` calls and read
with windowed HTTP range requests straight out of the COGs. No account, no
token, no API in the middle.

Two things make this fast enough to be practical:

* **Cloud pre-screening.** The 20 m scene classification layer is fetched
  first, for every candidate date, and dates too cloudy over the AOI are
  dropped before any reflectance band is touched. Over inland NSW in summer
  that skips most of the archive.
* **Threaded reads.** Every read is network-bound, so a modest thread pool
  turns a serial half-hour into a couple of minutes.
"""

from __future__ import annotations

import concurrent.futures as futures
import os
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from vegmon.config import DetectionConfig, PipelineConfig
from vegmon.grid import Grid
from vegmon.masking import DEFAULT_VALID_SCL
from vegmon.series import OpticalSeries, S2_BANDS

BUCKET = "sentinel-cogs"
REGION = "us-west-2"
BUCKET_URL = f"https://{BUCKET}.s3.{REGION}.amazonaws.com"
PREFIX = "sentinel-s2-l2a-cogs"

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

#: Canonical band name -> asset file and its native ground sample distance.
BAND_ASSETS: Dict[str, Tuple[str, int]] = {
    "blue": ("B02", 10),
    "green": ("B03", 10),
    "red": ("B04", 10),
    "nir": ("B08", 10),
    "swir1": ("B11", 20),
    "swir2": ("B12", 20),
}
SCL_ASSET = ("SCL", 20)

_SCENE_RE = re.compile(
    r"(?P<mission>S2[ABC])_(?P<tile>\d{2}[A-Z]{3})_(?P<date>\d{8})_(?P<sequence>\d+)_L2A"
)


class SceneListingError(RuntimeError):
    """The bucket could not be listed, or held nothing for the request."""


@dataclass(frozen=True)
class SceneRef:
    """One Sentinel-2 acquisition in the bucket."""

    tile: str
    acquired: date
    sequence: int
    mission: str
    prefix: str

    def asset_url(self, asset: str) -> str:
        return f"{BUCKET_URL}/{self.prefix}{asset}.tif"


def configure_gdal() -> None:
    """Point GDAL at the same proxy and CA bundle the rest of the session uses.

    Without this, reads fail with a TLS error behind an intercepting proxy -
    GDAL keeps its own HTTP settings and does not read ``HTTPS_PROXY``.
    """
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        os.environ.setdefault("GDAL_HTTP_PROXY", proxy)
    for name, value in (
        ("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt"),
        ("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"),
        ("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif"),
        ("GDAL_HTTP_MAX_RETRY", "3"),
        ("GDAL_HTTP_RETRY_DELAY", "2"),
        ("VSI_CACHE", "TRUE"),
        ("GDAL_CACHEMAX", "256"),
    ):
        if name == "CURL_CA_BUNDLE" and not os.path.exists(value):
            continue
        os.environ.setdefault(name, value)


def _list_prefixes(prefix: str, timeout: float = 60.0) -> List[str]:
    """Anonymous ``ListObjectsV2`` with a delimiter, following continuations."""
    out: List[str] = []
    token: Optional[str] = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        try:
            with urlopen(f"{BUCKET_URL}/?{urlencode(query)}", timeout=timeout) as response:
                root = ElementTree.fromstring(response.read())
        except Exception as exc:
            raise SceneListingError(
                f"could not list s3://{BUCKET}/{prefix} - the AWS Sentinel-2 open-data "
                f"bucket must be reachable ({BUCKET_URL})"
            ) from exc
        out.extend(
            node.findtext(f"{_S3_NS}Prefix", "")
            for node in root.findall(f"{_S3_NS}CommonPrefixes")
        )
        if root.findtext(f"{_S3_NS}IsTruncated", "false") != "true":
            return out
        token = root.findtext(f"{_S3_NS}NextContinuationToken")
        if not token:
            return out


def parse_tile(tile: str) -> Tuple[str, str, str]:
    """``'55JGG'`` -> ``('55', 'J', 'GG')``, matching the bucket's key layout."""
    tile = tile.upper().lstrip("T")
    if not re.fullmatch(r"\d{2}[A-Z]{3}", tile):
        raise ValueError(f"{tile!r} is not an MGRS tile id like '55JGG'")
    return tile[:2].lstrip("0") or "0", tile[2], tile[3:]


def tile_for(lon: float, lat: float) -> str:
    """MGRS tile id containing a lon/lat, for picking an AOI's tile."""
    try:
        import mgrs
    except ImportError as exc:  # pragma: no cover - optional convenience
        raise ImportError("resolving a tile from lon/lat needs: pip install mgrs") from exc
    return mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=0)


def list_scenes(tile: str, start: date, end: date) -> List[SceneRef]:
    """Every acquisition for a tile between two dates, newest processing only.

    The bucket keeps more than one processing baseline for some dates, as
    ``..._0_L2A`` and ``..._1_L2A``. Only the highest sequence is kept: they
    are the same acquisition, and stacking both would double-weight it in
    every composite.
    """
    zone, band, square = parse_tile(tile)
    scenes: Dict[date, SceneRef] = {}
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        month_prefix = f"{PREFIX}/{zone}/{band}/{square}/{cursor.year}/{cursor.month}/"
        for scene_prefix in _list_prefixes(month_prefix):
            match = _SCENE_RE.search(scene_prefix)
            if not match:
                continue
            acquired = date(
                int(match["date"][:4]), int(match["date"][4:6]), int(match["date"][6:])
            )
            if not (start <= acquired <= end):
                continue
            sequence = int(match["sequence"])
            existing = scenes.get(acquired)
            if existing is None or sequence > existing.sequence:
                scenes[acquired] = SceneRef(
                    tile=match["tile"], acquired=acquired, sequence=sequence,
                    mission=match["mission"], prefix=scene_prefix,
                )
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)

    if not scenes:
        raise SceneListingError(
            f"no Sentinel-2 scenes for tile {tile} between {start} and {end}"
        )
    return [scenes[k] for k in sorted(scenes)]


def grid_for_aoi(
    tile: str,
    centre_lonlat: Tuple[float, float],
    size: int,
    resolution: float,
    crs: Optional[str] = None,
) -> Grid:
    """A square grid of ``size`` pixels centred on a lon/lat, snapped to the tile.

    Snapping to the source pixel edges matters: an unsnapped window forces
    GDAL to resample every read, which both costs time and softens the sharp
    boundaries that clearing detection depends on.
    """
    from pyproj import Transformer

    zone, band, _ = parse_tile(tile)
    epsg = f"EPSG:{'327' if band < 'N' else '326'}{int(zone):02d}"
    crs = crs or epsg
    east, north = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        *centre_lonlat
    )
    half = size * resolution / 2.0
    # Snap the origin to a whole multiple of the working resolution.
    west = np.floor((east - half) / resolution) * resolution
    top = np.ceil((north + half) / resolution) * resolution
    return Grid.from_origin(crs, float(west), float(top), resolution, size, size)


def _read_window(url: str, grid: Grid, resampling_name: str) -> np.ndarray:
    """Read one COG into ``grid``, resampling if the asset is coarser."""
    import warnings

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(url) as source:
        destination = np.zeros(grid.shape, dtype=source.dtypes[0])
        # The destination is a bare array, so rasterio warns that it carries no
        # geotransform of its own. It does not need one - dst_transform and
        # dst_crs below are what place it.
        warnings.filterwarnings(
            "ignore", category=rasterio.errors.NotGeoreferencedWarning
        )
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=grid.rasterio_transform(),
            dst_crs=grid.crs,
            resampling=getattr(Resampling, resampling_name),
            src_nodata=source.nodata,
            dst_nodata=0,
        )
        return destination


def _safe_read(url: str, grid: Grid, resampling_name: str) -> Optional[np.ndarray]:
    try:
        return _read_window(url, grid, resampling_name)
    except Exception:
        # A missing or truncated asset costs one date, not the whole run.
        return None


def screen_scenes(
    scenes: Sequence[SceneRef],
    grid: Grid,
    max_cloud_cover: float,
    workers: int = 8,
    progress: bool = False,
) -> List[Tuple[SceneRef, np.ndarray, float]]:
    """Fetch each scene's SCL over the AOI and drop the too-cloudy ones.

    Screening on the AOI rather than on the scene-level metadata is the whole
    point: a tile can be 70% cloudy and still perfectly clear over the paddock
    being monitored, and the scene-level figure would throw it away.
    """
    configure_gdal()
    kept: List[Tuple[SceneRef, np.ndarray, float]] = []

    def fetch(scene: SceneRef):
        return scene, _safe_read(scene.asset_url(SCL_ASSET[0]), grid, "nearest")

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (scene, scl) in enumerate(pool.map(fetch, scenes)):
            if scl is None:
                continue
            usable = float(np.isin(scl, np.asarray(DEFAULT_VALID_SCL)).mean())
            cloud = 1.0 - usable
            if cloud <= max_cloud_cover:
                kept.append((scene, scl.astype(np.uint8), cloud))
            if progress and index % 50 == 0:
                print(f"    screened {index + 1}/{len(scenes)}", flush=True)
    return kept


def load_sentinel2(
    tile: str,
    grid: Grid,
    start: date,
    end: date,
    detection: Optional[DetectionConfig] = None,
    workers: int = 8,
    progress: bool = False,
) -> OpticalSeries:
    """Build an :class:`OpticalSeries` for an AOI straight from the bucket."""
    detection = detection or DetectionConfig()
    configure_gdal()

    scenes = list_scenes(tile, start, end)
    if progress:
        print(f"  {len(scenes)} acquisitions on tile {tile}; screening cloud over the AOI...",
              flush=True)
    screened = screen_scenes(
        scenes, grid, detection.max_cloud_cover, workers=workers, progress=progress
    )
    if not screened:
        raise SceneListingError(
            f"every one of the {len(scenes)} acquisitions for {tile} was cloudier than "
            f"{detection.max_cloud_cover:.0%} over this AOI; widen the window or the limit"
        )
    if progress:
        print(f"  {len(screened)} usable after cloud screening; reading reflectance...",
              flush=True)

    n = len(screened)
    bands = {name: np.full((n,) + grid.shape, np.nan, np.float32) for name in S2_BANDS}
    scl_stack = np.zeros((n,) + grid.shape, np.uint8)
    times = [scene.acquired for scene, _, _ in screened]
    for index, (_, scl, _) in enumerate(screened):
        scl_stack[index] = scl

    jobs = [
        (index, name, scene.asset_url(BAND_ASSETS[name][0]))
        for index, (scene, _, _) in enumerate(screened)
        for name in S2_BANDS
    ]

    def fetch(job):
        index, name, url = job
        return index, name, _safe_read(url, grid, "bilinear")

    done = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for index, name, raw in pool.map(fetch, jobs):
            done += 1
            if raw is None:
                continue
            # L2A is uint16 scaled by 10000 with 0 as nodata.
            value = raw.astype(np.float32)
            bands[name][index] = np.where(value > 0, value / 10000.0, np.nan)
            if progress and done % 200 == 0:
                print(f"    read {done}/{len(jobs)} assets", flush=True)

    return OpticalSeries(
        grid=grid, times=times, bands=bands, scl=scl_stack,
        sensor=f"aws-sentinel-cogs:{tile}",
    )


def cloud_free_days(series: OpticalSeries, threshold: float = 0.6) -> int:
    """How many acquisitions cleared the AOI cloud threshold - a coverage check."""
    return int((series.scene_cloud_fraction() <= threshold).sum())


__all__ = [
    "BAND_ASSETS",
    "BUCKET_URL",
    "SceneListingError",
    "SceneRef",
    "cloud_free_days",
    "configure_gdal",
    "grid_for_aoi",
    "list_scenes",
    "load_sentinel2",
    "parse_tile",
    "screen_scenes",
    "tile_for",
]
