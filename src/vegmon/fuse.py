"""Fuse the optical and radar detections into mapped clearing patches.

Neither sensor is trusted on its own. Sentinel-2 sees the spectral collapse
but cannot tell canopy *removal* from canopy *scorching* or from a paddock
taken out of production. Sentinel-1 sees the structural collapse but is
speckly, coarse at the edges, and blind to anything that does not change
scattering geometry. Agreement between them is a much stronger statement than
either alone, so the output is tiered by how many sensors saw the event rather
than collapsing to a single yes/no.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

from vegmon.config import DetectionConfig
from vegmon.detect import OpticalChange, RadarChange
from vegmon.grid import Grid

# Confidence tiers, ordered so that a higher number is a stronger claim.
TIER_NONE = 0
TIER_RADAR_ONLY = 1
TIER_OPTICAL_ONLY = 2
TIER_CONFIRMED = 3

TIER_NAMES = {
    TIER_NONE: "none",
    TIER_RADAR_ONLY: "s1_only",
    TIER_OPTICAL_ONLY: "s2_only",
    TIER_CONFIRMED: "confirmed",
}

TIER_DESCRIPTIONS = {
    "confirmed": (
        "Both lines of evidence agree. This is the tier to act on."
    ),
    "s2_only": (
        "Spectral loss with no matching structural loss. Typically fire, "
        "drought dieback, a harvested or fallowed paddock, or an unmasked "
        "cloud shadow. Review before treating as clearing."
    ),
    "s1_only": (
        "Structural loss with no matching spectral loss. Often thinning or "
        "selective removal under retained canopy, an inundated surface, or a "
        "gap in cloud-free optical coverage. Review. Empty by construction "
        "when the confirming evidence is corroborating rather than independent."
    ),
}


@dataclass
class ClearingMap:
    """Tiered clearing patches with per-patch attributes."""

    grid: Grid
    components: np.ndarray
    """``(y, x)`` int32 raster of patch ids; 0 is background."""

    tier: np.ndarray
    """``(y, x)`` uint8 raster of confidence tiers, one value per patch."""

    records: List[dict] = field(default_factory=list)
    detection_config: Optional[DetectionConfig] = None
    confirming_source: str = "Sentinel-1 structural change"
    """What supplied the second opinion. Normally Sentinel-1; where no radar is
    reachable this says so, because "confirmed" means something weaker then."""

    confirming_description: str = (
        "the canopy went and the woody structure went with it"
    )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def patch_ids(self) -> List[int]:
        return [r["patch_id"] for r in self.records]

    def filter_tier(self, *tiers: str) -> "ClearingMap":
        """A copy keeping only patches in the named tiers."""
        keep = {r["patch_id"] for r in self.records if r["tier"] in tiers}
        components = np.where(np.isin(self.components, list(keep)), self.components, 0)
        return ClearingMap(
            grid=self.grid,
            components=components.astype(np.int32),
            tier=np.where(components > 0, self.tier, TIER_NONE).astype(np.uint8),
            records=[r for r in self.records if r["patch_id"] in keep],
            detection_config=self.detection_config,
            confirming_source=self.confirming_source,
            confirming_description=self.confirming_description,
        )

    def mask(self, *tiers: str) -> np.ndarray:
        """Boolean mask of pixels in the named tiers (all tiers if none given)."""
        if not tiers:
            return self.components > 0
        ids = [r["patch_id"] for r in self.records if r["tier"] in tiers]
        return np.isin(self.components, ids) if ids else np.zeros(self.grid.shape, bool)

    def area_ha(self, *tiers: str) -> float:
        records = self.records if not tiers else [r for r in self.records if r["tier"] in tiers]
        return float(sum(r["area_ha"] for r in records))

    def summary(self) -> Dict[str, dict]:
        """Patch count and area per tier - the headline table."""
        out: Dict[str, dict] = {}
        for name in ("confirmed", "s2_only", "s1_only"):
            records = [r for r in self.records if r["tier"] == name]
            out[name] = {
                "patches": len(records),
                "area_ha": round(float(sum(r["area_ha"] for r in records)), 2),
                "description": TIER_DESCRIPTIONS[name],
            }
        return out

    def to_geodataframe(self):
        """Polygonise to a GeoDataFrame in the working CRS.

        Requires geopandas and rasterio; raises a clear error if either is
        missing, since the rest of the pipeline does not need them.
        """
        try:
            import geopandas as gpd
            from rasterio import features
            from shapely.geometry import shape
        except ImportError as exc:  # pragma: no cover - exercised only without extras
            raise ImportError(
                "polygonising needs geopandas, rasterio and shapely: "
                "pip install 'vegmon[geo]'"
            ) from exc

        by_id = {r["patch_id"]: r for r in self.records}
        geometries = []
        rows = []
        transform = self.grid.rasterio_transform()
        for geom, value in features.shapes(
            self.components, mask=self.components > 0, transform=transform
        ):
            record = by_id.get(int(value))
            if record is None:
                continue
            polygon = shape(geom)
            row = dict(record)
            row["perimeter_m"] = round(float(polygon.length), 1)
            # 1.0 for a circle; a long thin fenceline clearing scores much higher.
            row["shape_index"] = round(
                float(polygon.length / (2.0 * np.sqrt(np.pi * max(polygon.area, 1e-9)))), 2
            )
            geometries.append(polygon)
            rows.append(row)

        frame = gpd.GeoDataFrame(rows, geometry=geometries, crs=self.grid.crs)
        return frame.sort_values("area_ha", ascending=False).reset_index(drop=True)

    def write_vector(self, path, layer: str = "clearing"):
        """Write patches to GeoPackage (or any format fiona can handle)."""
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.to_geodataframe()
        if path.suffix.lower() == ".gpkg":
            frame.to_file(path, layer=layer, driver="GPKG")
        else:
            frame.to_file(path)
        return path


def clean_mask(
    mask: np.ndarray,
    grid: Grid,
    config: DetectionConfig,
) -> np.ndarray:
    """Morphological cleanup and minimum-mapping-unit sieve.

    Opening first, to shave the one-pixel filaments that thresholding leaves
    along paddock edges and drainage lines; then hole filling, because
    scattered retained trees inside a cleared paddock read as holes but are
    part of the same event; then the sieve, because sub-MMU detections at 10 m
    are mostly edge noise and reporting works in hectares anyway.
    """
    mask = np.asarray(mask, dtype=bool)
    if config.opening_iterations > 0:
        mask = ndimage.binary_opening(mask, iterations=config.opening_iterations)
    if config.fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    return sieve(mask, grid.pixels_for_ha(config.min_mapping_unit_ha))


def sieve(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Drop connected components smaller than ``min_pixels``."""
    if min_pixels <= 1:
        return np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.asarray(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = np.zeros(sizes.shape, dtype=bool)
    keep[1:] = sizes[1:] >= min_pixels
    return keep[labels]


def fuse(
    optical: OpticalChange,
    radar,
    config: Optional[DetectionConfig] = None,
) -> ClearingMap:
    """Combine the two detectors into tiered, sieved, attributed patches.

    Tiering is done per *patch*, not per pixel. Pixel-level agreement between a
    10 m optical detection and a multi-looked radar one is never complete
    along edges, so a patch is called confirmed when the two sensors overlap
    across a decent fraction of it - which is the claim actually being made.
    """
    config = config or DetectionConfig()
    grid = optical.grid

    # The confirming evidence is either a Sentinel-1 change layer or, where no
    # radar is reachable, a temporal-persistence layer. Both expose a mask and
    # a dict of per-pixel statistics, and nothing below needs to know which.
    if isinstance(radar, RadarChange):
        confirming_stats = {
            "mean_vh_drop_db": radar.vh_drop_db,
            "max_vh_drop_db": radar.vh_drop_db,
            "mean_vv_drop_db": radar.vv_drop_db,
            "pre_vh_db": radar.pre.vh_db,
            "post_vh_db": radar.post.vh_db,
        }
        confirming_source = "Sentinel-1 structural change"
        confirming_description = "the canopy went and the woody structure went with it"
    else:
        confirming_stats = radar.as_stats()
        confirming_source = getattr(radar, "source_name", "second evidence layer")
        confirming_description = getattr(radar, "source_description", "")

    # An independent sensor can detect clearing the optical detector missed, so
    # its detections join the union. Corroborating evidence cannot: it only
    # ever answers a question about a pixel the optical detector already
    # flagged, so letting it contribute its own patches would invent
    # detections out of a test that was never run on that ground.
    independent = getattr(radar, "is_independent", True)
    union = (optical.mask | radar.mask) if independent else optical.mask
    cleaned = clean_mask(union, grid, config)

    components, n_components = ndimage.label(cleaned)
    components = components.astype(np.int32)
    tier_raster = np.zeros(grid.shape, dtype=np.uint8)
    records: List[dict] = []

    if n_components == 0:
        return ClearingMap(
            grid, components, tier_raster, records, config,
            confirming_source, confirming_description,
        )

    xs, ys = grid.xy_centres()
    x_grid, y_grid = np.meshgrid(xs, ys)

    for patch_id in range(1, n_components + 1):
        patch = components == patch_id
        n_pixels = int(patch.sum())
        s2_fraction = float(optical.mask[patch].mean())
        s1_fraction = float(radar.mask[patch].mean())
        both_fraction = float((optical.mask & radar.mask)[patch].mean())

        if both_fraction >= config.confirm_overlap:
            tier = TIER_CONFIRMED
        elif s2_fraction >= s1_fraction:
            tier = TIER_OPTICAL_ONLY
        else:
            tier = TIER_RADAR_ONLY
        tier_raster[patch] = tier

        records.append(
            {
                "patch_id": patch_id,
                "tier": TIER_NAMES[tier],
                "tier_code": int(tier),
                "area_ha": round(grid.area_ha(n_pixels), 3),
                "n_pixels": n_pixels,
                "s2_fraction": round(s2_fraction, 3),
                "s1_fraction": round(s1_fraction, 3),
                "agreement_fraction": round(both_fraction, 3),
                "mean_dnbr": _stat(optical.dnbr, patch),
                "max_dnbr": _stat(optical.dnbr, patch, np.nanmax),
                "mean_dndvi": _stat(optical.dndvi, patch),
                "pre_ndvi": _stat(optical.pre.ndvi, patch),
                "pre_nbr": _stat(optical.pre.nbr, patch),
                "pre_ndvi_persistent": _stat(optical.pre_ndvi_persistent, patch),
                "post_ndvi": _stat(optical.post.ndvi, patch),
                "centroid_x": round(float(x_grid[patch].mean()), 1),
                "centroid_y": round(float(y_grid[patch].mean()), 1),
                **{
                    name: _stat(layer, patch, np.nanmax if name.startswith("max_") else np.nanmean)
                    for name, layer in confirming_stats.items()
                },
            }
        )

    return ClearingMap(
        grid=grid,
        components=components,
        tier=tier_raster,
        records=records,
        detection_config=config,
        confirming_source=confirming_source,
        confirming_description=confirming_description,
    )


def _stat(values: np.ndarray, mask: np.ndarray, reducer=np.nanmean) -> Optional[float]:
    sample = values[mask]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return None
    return round(float(reducer(sample)), 4)


__all__ = [
    "ClearingMap",
    "TIER_CONFIRMED",
    "TIER_DESCRIPTIONS",
    "TIER_NAMES",
    "TIER_NONE",
    "TIER_OPTICAL_ONLY",
    "TIER_RADAR_ONLY",
    "clean_mask",
    "fuse",
    "sieve",
]
