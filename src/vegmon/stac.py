"""Live Sentinel-1 and Sentinel-2 loaders backed by public STAC catalogues.

Everything here is optional: the pipeline runs end to end on the synthetic
datacube without ``pystac-client`` or ``odc-stac`` installed. When they are
available, these loaders produce exactly the same :class:`OpticalSeries` and
:class:`RadarSeries` objects, so nothing downstream changes.

Catalogue notes
---------------

``earth-search`` (Element 84, on AWS)
    Sentinel-2 L2A as cloud-optimised GeoTIFFs in the AWS open data
    programme. No account, no token, no egress cost from ``us-west-2``. This
    is the default and the one to reach for first.

``planetary-computer`` (Microsoft)
    Sentinel-2 L2A *and* Sentinel-1 RTC - radiometrically terrain-corrected
    gamma-nought, which is what you actually want over the escarpment country
    where slope effects otherwise swamp a 2 dB threshold. Assets need signing;
    the ``planetary-computer`` package does it anonymously.

``cdse`` (Copernicus Data Space Ecosystem)
    The authoritative ESA source, and the one to use if provenance matters for
    a regulatory process. Needs registration.

If a catalogue is unreachable the loaders raise :class:`CatalogueUnavailable`
with the endpoint named, rather than failing deep inside a HTTP library -
corporate proxies and allow-lists are the usual cause, not the catalogue
being down.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from vegmon.config import AOI, DetectionConfig, PipelineConfig
from vegmon.grid import Grid
from vegmon.indices import linear_to_db
from vegmon.series import OpticalSeries, RadarSeries, S2_BANDS, Scene


class CatalogueUnavailable(RuntimeError):
    """A STAC endpoint could not be reached or returned nothing usable."""


@dataclass(frozen=True)
class Catalogue:
    name: str
    url: str
    optical_collection: Optional[str]
    radar_collection: Optional[str]
    requires_signing: bool = False
    notes: str = ""


CATALOGUES: Dict[str, Catalogue] = {
    "earth-search": Catalogue(
        name="earth-search",
        url="https://earth-search.aws.element84.com/v1",
        optical_collection="sentinel-2-l2a",
        radar_collection="sentinel-1-grd",
        notes="Anonymous, no account. Sentinel-1 here is GRD, not terrain-corrected.",
    ),
    "planetary-computer": Catalogue(
        name="planetary-computer",
        url="https://planetarycomputer.microsoft.com/api/stac/v1",
        optical_collection="sentinel-2-l2a",
        radar_collection="sentinel-1-rtc",
        requires_signing=True,
        notes="Sentinel-1 RTC is terrain-corrected gamma-nought - prefer it in hill country.",
    ),
    "cdse": Catalogue(
        name="cdse",
        url="https://catalogue.dataspace.copernicus.eu/stac",
        optical_collection="SENTINEL-2",
        radar_collection="SENTINEL-1",
        notes="Authoritative ESA source; needs a (free) Copernicus account.",
    ),
}

#: Sentinel-2 asset keys, per catalogue, for the bands the pipeline needs.
_S2_ASSETS: Dict[str, Dict[str, str]] = {
    "earth-search": {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "nir": "nir",
        "swir1": "swir16",
        "swir2": "swir22",
        "scl": "scl",
    },
    "planetary-computer": {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir": "B08",
        "swir1": "B11",
        "swir2": "B12",
        "scl": "SCL",
    },
}


def open_catalogue(name: str = "earth-search"):
    """Open a STAC client, translating connection failures into a clear error."""
    if name not in CATALOGUES:
        raise KeyError(f"unknown catalogue {name!r}; choose from {sorted(CATALOGUES)}")
    catalogue = CATALOGUES[name]
    try:
        from pystac_client import Client
    except ImportError as exc:
        raise ImportError(
            "live loading needs pystac-client and odc-stac: pip install 'vegmon[stac]'"
        ) from exc

    modifier = None
    if catalogue.requires_signing:
        try:
            import planetary_computer

            modifier = planetary_computer.sign_inplace
        except ImportError:  # pragma: no cover - optional
            modifier = None

    try:
        return Client.open(catalogue.url, modifier=modifier)
    except Exception as exc:
        raise CatalogueUnavailable(
            f"could not reach the {catalogue.name} STAC endpoint at {catalogue.url}. "
            "In a restricted network this is usually an egress policy rather than an "
            "outage - the host has to be on the allow-list. The synthetic datacube "
            "(vegmon demo) needs no network at all."
        ) from exc


def search_items(
    catalogue_name: str,
    collection: str,
    aoi: AOI,
    start: date,
    end: date,
    query: Optional[dict] = None,
    limit: Optional[int] = None,
) -> List:
    """Return the STAC items intersecting an AOI over a date range."""
    client = open_catalogue(catalogue_name)
    search = client.search(
        collections=[collection],
        bbox=list(aoi.bbox),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query=query,
        max_items=limit,
    )
    try:
        items = list(search.items())
    except Exception as exc:
        raise CatalogueUnavailable(
            f"search against {catalogue_name}/{collection} failed"
        ) from exc
    if not items:
        raise CatalogueUnavailable(
            f"no {collection} items over {aoi.name} between {start} and {end}"
        )
    return items


def _load_cube(items, bands: Sequence[str], aoi: AOI, resolution: float, chunks=None):
    try:
        from odc.stac import load as odc_load
    except ImportError as exc:
        raise ImportError(
            "live loading needs odc-stac: pip install 'vegmon[stac]'"
        ) from exc
    return odc_load(
        items,
        bands=list(bands),
        crs=aoi.crs,
        resolution=resolution,
        bbox=list(aoi.bbox),
        chunks=chunks or {},
        groupby="solar_day",
    )


def load_sentinel2(
    config: PipelineConfig,
    catalogue: str = "earth-search",
    max_cloud_cover: Optional[float] = None,
    detection: Optional[DetectionConfig] = None,
) -> OpticalSeries:
    """Load a Sentinel-2 L2A series for the configured AOI and full time span.

    Scene-level cloud cover is filtered at the catalogue rather than after
    download - there is no point pulling a 95% cloudy acquisition across the
    network to throw it away, and over inland NSW in summer that is most of
    them.
    """
    detection = detection or config.detection
    cloud_limit = (
        max_cloud_cover if max_cloud_cover is not None else detection.max_cloud_cover
    ) * 100.0
    catalogue_def = CATALOGUES[catalogue]
    if catalogue_def.optical_collection is None:
        raise ValueError(f"catalogue {catalogue!r} carries no Sentinel-2 collection")

    items = search_items(
        catalogue,
        catalogue_def.optical_collection,
        config.aoi,
        config.periods.series_start,
        config.periods.series_end,
        query={"eo:cloud_cover": {"lt": cloud_limit}},
    )

    assets = _S2_ASSETS.get(catalogue)
    if assets is None:
        raise ValueError(f"no Sentinel-2 asset mapping for catalogue {catalogue!r}")
    cube = _load_cube(items, list(assets.values()), config.aoi, config.resolution)

    grid = _grid_from_cube(cube, config.aoi.crs)
    times = [np.datetime64(t, "D") for t in cube.time.values]
    # L2A is stored as uint16 scaled by 10000, with 0 as nodata.
    bands = {}
    for name in S2_BANDS:
        raw = cube[assets[name]].values.astype(np.float32)
        bands[name] = np.where(raw > 0, raw / 10000.0, np.nan).astype(np.float32)
    scl = cube[assets["scl"]].values.astype(np.uint8)

    return OpticalSeries(
        grid=grid, times=times, bands=bands, scl=scl, sensor=f"{catalogue}:sentinel-2-l2a"
    )


def load_sentinel1(
    config: PipelineConfig,
    catalogue: str = "planetary-computer",
    orbit_state: Optional[str] = None,
) -> RadarSeries:
    """Load a Sentinel-1 series, converted to dB, on the optical grid.

    Pass ``orbit_state`` to pin a single pass direction. It is worth doing:
    ascending and descending acquisitions view the canopy from opposite sides,
    and the resulting backscatter difference over the same intact woodland can
    exceed the 2 dB drop the detector is looking for.
    """
    catalogue_def = CATALOGUES[catalogue]
    if catalogue_def.radar_collection is None:
        raise ValueError(f"catalogue {catalogue!r} carries no Sentinel-1 collection")

    query = {}
    if orbit_state:
        query["sat:orbit_state"] = {"eq": orbit_state.lower()}
    items = search_items(
        catalogue,
        catalogue_def.radar_collection,
        config.aoi,
        config.periods.series_start,
        config.periods.series_end,
        query=query or None,
    )

    cube = _load_cube(items, ["vv", "vh"], config.aoi, config.resolution)
    grid = _grid_from_cube(cube, config.aoi.crs)
    times = [np.datetime64(t, "D") for t in cube.time.values]

    # RTC and GRD assets are both linear power; the pipeline works in dB.
    vv = linear_to_db(cube["vv"].values.astype(np.float32))
    vh = linear_to_db(cube["vh"].values.astype(np.float32))

    states = [
        str(item.properties.get("sat:orbit_state", "unknown")).lower() for item in items
    ]
    states = states[: len(times)] if len(states) >= len(times) else None

    return RadarSeries(
        grid=grid,
        times=times,
        vv_db=vv,
        vh_db=vh,
        orbit_state=states,
        sensor=f"{catalogue}:{catalogue_def.radar_collection}",
    )


def load_scene(
    config: PipelineConfig,
    optical_catalogue: str = "earth-search",
    radar_catalogue: str = "planetary-computer",
    orbit_state: Optional[str] = None,
) -> Scene:
    """Load both sensors for one AOI and return a pipeline-ready scene."""
    optical = load_sentinel2(config, catalogue=optical_catalogue)
    radar = load_sentinel1(config, catalogue=radar_catalogue, orbit_state=orbit_state)
    if radar.grid.shape != optical.grid.shape:
        raise ValueError(
            "Sentinel-1 and Sentinel-2 loaded onto different grids; "
            "pass the same resolution and AOI to both"
        )
    return Scene(optical=optical, radar=radar, grid=optical.grid)


def _grid_from_cube(cube, crs: str) -> Grid:
    """Recover a :class:`Grid` from an odc-stac xarray cube."""
    x = cube.x.values
    y = cube.y.values
    resolution_x = float(abs(x[1] - x[0])) if x.size > 1 else 10.0
    resolution_y = float(abs(y[1] - y[0])) if y.size > 1 else resolution_x
    west = float(x[0]) - resolution_x / 2.0
    north = float(y[0]) + resolution_y / 2.0
    return Grid(
        crs=str(getattr(cube, "odc", None).crs if hasattr(cube, "odc") else crs) or crs,
        transform=(resolution_x, 0.0, west, 0.0, -resolution_y, north),
        width=int(x.size),
        height=int(y.size),
    )


def catalogue_report() -> str:
    """A short human-readable description of the catalogue options."""
    lines = ["Available STAC catalogues:"]
    for catalogue in CATALOGUES.values():
        lines.append(f"  {catalogue.name:<20} {catalogue.url}")
        lines.append(
            f"  {'':<20} S2: {catalogue.optical_collection}  S1: {catalogue.radar_collection}"
        )
        if catalogue.notes:
            lines.append(f"  {'':<20} {catalogue.notes}")
    return "\n".join(lines)


__all__ = [
    "CATALOGUES",
    "Catalogue",
    "CatalogueUnavailable",
    "catalogue_report",
    "load_scene",
    "load_sentinel1",
    "load_sentinel2",
    "open_catalogue",
    "search_items",
]
