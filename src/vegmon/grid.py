"""Minimal raster geometry: a CRS + affine transform + shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

M2_PER_HA = 10_000.0


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid.

    ``transform`` is the 6-tuple of affine coefficients in rasterio order
    ``(a, b, c, d, e, f)`` mapping (col, row) -> (x, y), i.e. pixel width,
    row rotation, x origin, column rotation, pixel height (negative for
    north-up), y origin.
    """

    crs: str
    transform: Tuple[float, float, float, float, float, float]
    width: int
    height: int

    @classmethod
    def from_origin(
        cls,
        crs: str,
        west: float,
        north: float,
        resolution: float,
        width: int,
        height: int,
    ) -> "Grid":
        return cls(
            crs=crs,
            transform=(resolution, 0.0, west, 0.0, -resolution, north),
            width=width,
            height=height,
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def resolution(self) -> Tuple[float, float]:
        """(x size, y size) in CRS units, both positive."""
        return (abs(self.transform[0]), abs(self.transform[4]))

    @property
    def pixel_area_m2(self) -> float:
        xs, ys = self.resolution
        return xs * ys

    @property
    def pixel_area_ha(self) -> float:
        return self.pixel_area_m2 / M2_PER_HA

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(west, south, east, north) in CRS units."""
        a, _, c, _, e, f = self.transform
        west, north = c, f
        east = west + a * self.width
        south = north + e * self.height
        return (min(west, east), min(south, north), max(west, east), max(south, north))

    def area_ha(self, pixel_count: int | float | np.integer) -> float:
        return float(pixel_count) * self.pixel_area_ha

    def pixels_for_ha(self, hectares: float) -> int:
        """Number of whole pixels covering ``hectares`` (the minimum mapping unit)."""
        return max(1, int(round(hectares / self.pixel_area_ha)))

    def rasterio_transform(self):
        """Return this grid's transform as an ``affine.Affine`` for rasterio."""
        from affine import Affine

        return Affine(*self.transform)

    def xy_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """1-D arrays of pixel-centre x and y coordinates."""
        a, _, c, _, e, f = self.transform
        xs = c + a * (np.arange(self.width) + 0.5)
        ys = f + e * (np.arange(self.height) + 0.5)
        return xs, ys
