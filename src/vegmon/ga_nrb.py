"""Geoscience Australia's Sentinel-1 NRB product, straight from the DEA bucket.

GA's Digital Earth branch publishes a Sentinel-1 Normalised Radar Backscatter
(NRB) product: CEOS-ARD certified, radiometrically terrain corrected gamma0,
produced by a fork of NASA JPL's OPERA RTC processor. That is the analysis
ready Sentinel-1 this pipeline would otherwise have to build itself in
:mod:`vegmon.s1grd`, so it is worth being able to read.

The documented way in is DEA's development STAC API at
``explorer.dev.dea.ga.gov.au``. Like every other STAC endpoint this pipeline
has tried, it is routinely blocked where the AWS data buckets are not, and GA
themselves warn it is development infrastructure that should not be relied on.
It is also not needed: the products sit in a public, anonymously listable
bucket under a deterministic key layout::

    dea-public-data-dev/baseline/{product}/{burst_id}/{yyyy}/{mm}/{dd}/{datetime}/

so the archive can be enumerated with ``ListObjectsV2`` and each acquisition's
STAC item read directly out of the bucket beside its rasters.

Two things about the product shape the code here:

* **It is delivered as bursts, not scenes.** A Sentinel-1 IW scene is three
  sub-swaths of 20-30 bursts each. Bursts repeat to near-identical footprints
  every cycle where scenes do not, which makes them the right unit for a time
  series - but it means "which data covers my AOI" is a burst-geometry
  question. The OPERA burst database (also in the bucket) answers it.
* **The static layers live somewhere else.** Incidence angle, local incidence
  angle, number of looks and the gamma0->beta0/sigma0 ratios depend only on
  the viewing geometry, so they are published once per burst id under
  ``ga_s1_nrb_iw_static_*`` and referenced by every acquisition of that burst.

Coverage is a sample, not a continental archive - see :func:`survey_coverage`.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import sqlite3
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

BUCKET = "dea-public-data-dev"
REGION = "ap-southeast-2"
BUCKET_URL = f"https://{BUCKET}.s3.{REGION}.amazonaws.com"
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

#: The four IW products, split by the polarisation mode of the acquisition.
#: Sentinel-1 flies VV+VH over Australia and HH over the polar regions, so
#: ``ga_s1_nrb_iw_vv_vh_1`` is the only one that matters for NSW.
PRODUCTS = {
    "vv_vh": "ga_s1_nrb_iw_vv_vh_1",
    "vv": "ga_s1_nrb_iw_vv_1",
    "hh": "ga_s1_nrb_iw_hh_1",
    "hh_hv": "ga_s1_nrb_iw_hh_hv_1",
    "static": "ga_s1_nrb_iw_static_1",
}

#: OPERA burst-id database: burst id -> bounding box, in the burst's own UTM
#: zone. 52 MB, cached on first use.
BURST_DB_KEY = "projects/s1_nrb/burst_db/0.18.0/opera-iw-burst-bbox-only.sqlite3"


class CoverageError(RuntimeError):
    """The GA NRB archive holds nothing for the requested area."""


# ---------------------------------------------------------------------------
# bucket plumbing
# ---------------------------------------------------------------------------


def _list(prefix: str, delimiter: str = "/", timeout: float = 60.0) -> Tuple[List[str], List[Tuple[str, int]]]:
    """Anonymous ``ListObjectsV2``, following continuation tokens to the end."""
    prefixes: List[str] = []
    keys: List[Tuple[str, int]] = []
    token: Optional[str] = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            query["delimiter"] = delimiter
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET_URL}/?{urlencode(query)}", timeout=timeout) as r:
            root = ElementTree.fromstring(r.read())
        prefixes.extend(
            n.findtext(f"{_S3_NS}Prefix", "") for n in root.findall(f"{_S3_NS}CommonPrefixes")
        )
        keys.extend(
            (n.findtext(f"{_S3_NS}Key", ""), int(n.findtext(f"{_S3_NS}Size", "0")))
            for n in root.findall(f"{_S3_NS}Contents")
        )
        if root.findtext(f"{_S3_NS}IsTruncated", "false") != "true":
            return prefixes, keys
        token = root.findtext(f"{_S3_NS}NextContinuationToken")


def fetch_burst_db(path: str = "opera-iw-burst-bbox-only.sqlite3") -> str:
    """Download the OPERA IW burst database to ``path`` unless it is there."""
    import os

    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        urllib.request.urlretrieve(f"{BUCKET_URL}/{BURST_DB_KEY}", path)
    return path


# ---------------------------------------------------------------------------
# burst geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Burst:
    """One OPERA burst id and its footprint in lon/lat."""

    burst_id: str
    bbox: Tuple[float, float, float, float]  # west, south, east, north

    @property
    def track(self) -> int:
        return int(self.burst_id.split("_")[0].lstrip("t"))

    @property
    def sub_swath(self) -> str:
        return self.burst_id.split("_")[-1].upper()


def _burst_boxes(db_path: str, burst_ids: Optional[Iterable[str]] = None) -> List[Burst]:
    """Read burst bounding boxes out of the OPERA database, reprojected to 4326."""
    from pyproj import Transformer

    con = sqlite3.connect(db_path)
    if burst_ids is None:
        cursor = con.execute("SELECT burst_id_jpl,epsg,xmin,ymin,xmax,ymax FROM burst_id_map")
    else:
        ids = tuple(burst_ids)
        placeholders = ",".join("?" * len(ids))
        cursor = con.execute(
            "SELECT burst_id_jpl,epsg,xmin,ymin,xmax,ymax FROM burst_id_map "
            f"WHERE burst_id_jpl IN ({placeholders})",
            ids,
        )

    transformers: Dict[int, object] = {}
    out: List[Burst] = []
    for burst_id, epsg, xmin, ymin, xmax, ymax in cursor:
        if epsg not in transformers:
            transformers[epsg] = Transformer.from_crs(epsg, 4326, always_xy=True)
        lon0, lat0 = transformers[epsg].transform(xmin, ymin)
        lon1, lat1 = transformers[epsg].transform(xmax, ymax)
        out.append(
            Burst(
                burst_id,
                (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1)),
            )
        )
    con.close()
    return out


def _intersects(box: Sequence[float], aoi: Sequence[float]) -> bool:
    return box[0] <= aoi[2] and box[2] >= aoi[0] and box[1] <= aoi[3] and box[3] >= aoi[1]


def bursts_in_archive(product: str = PRODUCTS["vv_vh"]) -> List[str]:
    """Every burst id GA has actually published for a product."""
    prefixes, _ = _list(f"baseline/{product}/")
    return sorted(p.rstrip("/").split("/")[-1] for p in prefixes)


def bursts_for_aoi(
    aoi: Sequence[float],
    product: str = PRODUCTS["vv_vh"],
    db_path: Optional[str] = None,
) -> List[Burst]:
    """Published bursts whose footprint intersects ``aoi`` (west, south, east, north).

    Empty is a meaningful answer, not an error: GA's release is a sample and
    most of Australia is outside it.
    """
    db_path = db_path or fetch_burst_db()
    published = bursts_in_archive(product)
    if not published:
        return []
    return [b for b in _burst_boxes(db_path, published) if _intersects(b.bbox, aoi)]


# ---------------------------------------------------------------------------
# acquisitions
# ---------------------------------------------------------------------------


@dataclass
class Acquisition:
    """One burst on one date, with the assets GA published for it."""

    burst_id: str
    product: str
    datetime_key: str  # e.g. 20260420T192317
    prefix: str
    assets: Dict[str, str] = field(default_factory=dict)
    item: Optional[dict] = None

    @property
    def stac_url(self) -> Optional[str]:
        return self.assets.get("stac-item.json")

    @property
    def scene_id(self) -> Optional[str]:
        """The parent Sentinel-1 SLC this burst was cut from."""
        return (self.item or {}).get("properties", {}).get("sarard:scene_id")

    @property
    def absolute_orbit(self) -> Optional[int]:
        return (self.item or {}).get("properties", {}).get("sat:absolute_orbit")

    @property
    def relative_orbit(self) -> Optional[int]:
        return (self.item or {}).get("properties", {}).get("sat:relative_orbit")

    @property
    def orbit_state(self) -> Optional[str]:
        return (self.item or {}).get("properties", {}).get("sat:orbit_state")


def acquisitions(burst_id: str, product: str = PRODUCTS["vv_vh"]) -> List[Acquisition]:
    """Every acquisition GA has published for one burst, oldest first.

    The key layout nests year/month/day/datetime, so this walks four levels
    rather than doing one flat listing - the flat listing of a whole product
    is tens of thousands of keys.
    """
    root = f"baseline/{product}/{burst_id}/"
    found: List[Acquisition] = []
    years, _ = _list(root)
    for year in years:
        months, _ = _list(year)
        for month in months:
            days, _ = _list(month)
            for day in days:
                stamps, _ = _list(day)
                for stamp in stamps:
                    _, keys = _list(stamp, delimiter="")
                    assets = {k.split("/")[-1].split("_")[-1]: f"{BUCKET_URL}/{k}" for k, _ in keys}
                    found.append(
                        Acquisition(
                            burst_id=burst_id,
                            product=product,
                            datetime_key=stamp.rstrip("/").split("/")[-1],
                            prefix=stamp,
                            assets=assets,
                        )
                    )
    return sorted(found, key=lambda a: a.datetime_key)


def load_items(acqs: Sequence[Acquisition], workers: int = 12) -> List[Acquisition]:
    """Fetch and attach each acquisition's STAC item in place."""

    def grab(acq: Acquisition) -> Acquisition:
        url = acq.stac_url
        if url:
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    acq.item = json.load(r)
            except Exception:
                acq.item = None
        return acq

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(grab, acqs))


def static_layers(burst_id: str) -> Dict[str, str]:
    """Asset URLs for a burst's static layers, keyed by layer name.

    These are published once per burst id under a validity date and DEM, not
    per acquisition, so every date of a burst shares them.
    """
    root = f"baseline/{PRODUCTS['static']}/{burst_id}/"
    validity, _ = _list(root)
    out: Dict[str, str] = {}
    for period in validity:
        dems, _ = _list(period)
        for dem in dems:
            _, keys = _list(dem, delimiter="")
            for key, _size in keys:
                out[key.split("/")[-1].split("_")[-1]] = f"{BUCKET_URL}/{key}"
    return out


# ---------------------------------------------------------------------------
# coverage survey
# ---------------------------------------------------------------------------


def survey_coverage(
    aois: Dict[str, Sequence[float]],
    product: str = PRODUCTS["vv_vh"],
    db_path: Optional[str] = None,
) -> Dict[str, List[Burst]]:
    """Which of several AOIs the GA archive actually covers.

    Worth running before planning anything around this product. The Collection
    1 release is a soft release over selected tracks, and an AOI that returns
    nothing here has no GA NRB data at all - not a gap in the time series.
    """
    db_path = db_path or fetch_burst_db()
    published = bursts_in_archive(product)
    boxes = _burst_boxes(db_path, published)
    return {
        name: [b for b in boxes if _intersects(b.bbox, aoi)] for name, aoi in aois.items()
    }
