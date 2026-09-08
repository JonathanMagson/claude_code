"""In-memory time-series containers for Sentinel-2 and Sentinel-1.

Both the synthetic generator and the live STAC loaders produce these, so every
downstream stage is identical whether the data came off AWS or out of a random
number generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from vegmon import indices as ix
from vegmon import masking
from vegmon.grid import Grid

#: Canonical optical band names and the Sentinel-2 bands they map to.
S2_BANDS: dict[str, str] = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
    "swir1": "B11",
    "swir2": "B12",
}

_TC_ORDER = ("blue", "green", "red", "nir", "swir1", "swir2")


def to_datetime64(values: Iterable) -> np.ndarray:
    return np.array([np.datetime64(v, "D") for v in values], dtype="datetime64[D]")


def _window_mask(times: np.ndarray, start: date, end: date) -> np.ndarray:
    return (times >= np.datetime64(start, "D")) & (times <= np.datetime64(end, "D"))


@dataclass
class OpticalComposite:
    """A cloud-free reduction of an optical series over one time window."""

    grid: Grid
    start: date
    end: date
    bands: Dict[str, np.ndarray]
    n_obs: np.ndarray
    _cache: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def band(self, name: str) -> np.ndarray:
        return self.bands[name]

    def _tc_inputs(self) -> list[np.ndarray]:
        return [self.bands[b] for b in _TC_ORDER]

    def index(self, name: str) -> np.ndarray:
        """Compute (and memoise) a named index over this composite."""
        if name in self._cache:
            return self._cache[name]
        b = self.bands
        if name == "ndvi":
            out = ix.ndvi(b["red"], b["nir"])
        elif name == "nbr":
            out = ix.nbr(b["nir"], b["swir2"])
        elif name == "ndmi":
            out = ix.ndmi(b["nir"], b["swir1"])
        elif name == "ndwi":
            out = ix.ndwi(b["green"], b["nir"])
        elif name == "tc_brightness":
            out = ix.tc_brightness(*self._tc_inputs())
        elif name == "tc_greenness":
            out = ix.tc_greenness(*self._tc_inputs())
        elif name == "tc_wetness":
            out = ix.tc_wetness(*self._tc_inputs())
        else:
            raise KeyError(f"unknown optical index {name!r}")
        self._cache[name] = out
        return out

    @property
    def ndvi(self) -> np.ndarray:
        return self.index("ndvi")

    @property
    def nbr(self) -> np.ndarray:
        return self.index("nbr")

    @property
    def ndmi(self) -> np.ndarray:
        return self.index("ndmi")

    @property
    def ndwi(self) -> np.ndarray:
        return self.index("ndwi")


@dataclass
class RadarComposite:
    """A temporal reduction of a Sentinel-1 series over one time window."""

    grid: Grid
    start: date
    end: date
    vv_db: np.ndarray
    vh_db: np.ndarray
    n_obs: np.ndarray

    @property
    def rvi(self) -> np.ndarray:
        return ix.rvi(ix.db_to_linear(self.vv_db), ix.db_to_linear(self.vh_db))

    @property
    def cross_ratio_db(self) -> np.ndarray:
        """VH - VV in dB. Rises with volume scattering from canopy."""
        return (self.vh_db - self.vv_db).astype(np.float32)


@dataclass
class OpticalSeries:
    """A stack of Sentinel-2 L2A acquisitions on a common grid.

    ``bands`` holds ``(time, y, x)`` float32 surface reflectance in 0-1, keyed
    by the canonical names in :data:`S2_BANDS`. ``scl`` is the matching scene
    classification stack.
    """

    grid: Grid
    times: np.ndarray
    bands: Dict[str, np.ndarray]
    scl: np.ndarray
    sensor: str = "sentinel-2-l2a"
    _valid: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.times = to_datetime64(self.times)
        order = np.argsort(self.times)
        if not np.array_equal(order, np.arange(len(order))):
            self.times = self.times[order]
            self.bands = {k: v[order] for k, v in self.bands.items()}
            self.scl = self.scl[order]
        missing = set(S2_BANDS) - set(self.bands)
        if missing:
            raise ValueError(f"optical series is missing bands: {sorted(missing)}")
        expected = (len(self.times),) + self.grid.shape
        for name, arr in self.bands.items():
            if arr.shape != expected:
                raise ValueError(f"band {name!r} has shape {arr.shape}, expected {expected}")
        if self.scl.shape != expected:
            raise ValueError(f"scl has shape {self.scl.shape}, expected {expected}")

    def __len__(self) -> int:
        return int(len(self.times))

    def valid_mask(self, dilation_pixels: int = 3) -> np.ndarray:
        """Cached per-pixel validity mask derived from SCL."""
        if self._valid is None:
            self._valid = masking.scl_valid_mask(self.scl, dilation_pixels=dilation_pixels)
        return self._valid

    def scene_cloud_fraction(self) -> np.ndarray:
        return masking.cloud_fraction(self.scl)

    def select(self, start: date, end: date, max_cloud_cover: float = 1.0) -> "OpticalSeries":
        """Subset to acquisitions inside a date window and under a cloud limit."""
        keep = _window_mask(self.times, start, end)
        if max_cloud_cover < 1.0:
            keep &= self.scene_cloud_fraction() <= max_cloud_cover
        return OpticalSeries(
            grid=self.grid,
            times=self.times[keep],
            bands={k: v[keep] for k, v in self.bands.items()},
            scl=self.scl[keep],
            sensor=self.sensor,
        )

    def composite(
        self,
        start: date,
        end: date,
        max_cloud_cover: float = 1.0,
        dilation_pixels: int = 3,
        reducer: str = "medoid",
    ) -> OpticalComposite:
        """Cloud-free composite over a window.

        ``reducer="medoid"`` (the default) picks the single real observation
        per pixel closest to the per-band median, so every composite pixel is
        a spectrally consistent spectrum. ``reducer="median"`` takes each band
        independently, which is faster but manufactures spectra that no
        acquisition recorded - see :func:`vegmon.masking.masked_medoid`.
        """
        sub = self.select(start, end, max_cloud_cover=max_cloud_cover)
        valid = sub.valid_mask(dilation_pixels=dilation_pixels)
        if reducer == "medoid":
            bands = masking.masked_medoid(sub.bands, valid)
        elif reducer == "median":
            bands = {name: masking.masked_median(arr, valid) for name, arr in sub.bands.items()}
        else:
            raise ValueError(f"unknown reducer {reducer!r}; use 'medoid' or 'median'")
        return OpticalComposite(
            grid=self.grid,
            start=start,
            end=end,
            bands=bands,
            n_obs=masking.observation_count(valid),
        )

    def index_stack(self, name: str, dilation_pixels: int = 3) -> np.ndarray:
        """``(time, y, x)`` stack of a named index, NaN where masked."""
        b = self.bands
        if name == "ndvi":
            out = ix.ndvi(b["red"], b["nir"])
        elif name == "nbr":
            out = ix.nbr(b["nir"], b["swir2"])
        elif name == "ndmi":
            out = ix.ndmi(b["nir"], b["swir1"])
        elif name == "ndwi":
            out = ix.ndwi(b["green"], b["nir"])
        elif name in ("tc_brightness", "tc_greenness", "tc_wetness"):
            func = {
                "tc_brightness": ix.tc_brightness,
                "tc_greenness": ix.tc_greenness,
                "tc_wetness": ix.tc_wetness,
            }[name]
            out = func(*[b[k] for k in _TC_ORDER])
        else:
            raise KeyError(f"unknown optical index {name!r}")
        valid = self.valid_mask(dilation_pixels=dilation_pixels)
        return np.where(valid, out, np.nan).astype(np.float32)


@dataclass
class RadarSeries:
    """A stack of Sentinel-1 acquisitions, in dB, on a common grid.

    Keeping a single orbit direction is strongly preferred: ascending and
    descending passes see different sides of the canopy and differ by more
    than the clearing signal itself. :meth:`by_orbit` splits them.
    """

    grid: Grid
    times: np.ndarray
    vv_db: np.ndarray
    vh_db: np.ndarray
    orbit_state: Optional[Sequence[str]] = None
    sensor: str = "sentinel-1-grd"

    def __post_init__(self) -> None:
        self.times = to_datetime64(self.times)
        order = np.argsort(self.times)
        if not np.array_equal(order, np.arange(len(order))):
            self.times = self.times[order]
            self.vv_db = self.vv_db[order]
            self.vh_db = self.vh_db[order]
            if self.orbit_state is not None:
                self.orbit_state = [self.orbit_state[i] for i in order]
        expected = (len(self.times),) + self.grid.shape
        for name, arr in (("vv_db", self.vv_db), ("vh_db", self.vh_db)):
            if arr.shape != expected:
                raise ValueError(f"{name} has shape {arr.shape}, expected {expected}")
        if self.orbit_state is not None and len(self.orbit_state) != len(self.times):
            raise ValueError("orbit_state length does not match the number of acquisitions")

    def __len__(self) -> int:
        return int(len(self.times))

    def by_orbit(self, state: str) -> "RadarSeries":
        if self.orbit_state is None:
            raise ValueError("series carries no orbit metadata")
        keep = np.array([s.lower() == state.lower() for s in self.orbit_state])
        return self._subset(keep)

    def select(self, start: date, end: date) -> "RadarSeries":
        return self._subset(_window_mask(self.times, start, end))

    def _subset(self, keep: np.ndarray) -> "RadarSeries":
        return RadarSeries(
            grid=self.grid,
            times=self.times[keep],
            vv_db=self.vv_db[keep],
            vh_db=self.vh_db[keep],
            orbit_state=(
                [s for s, k in zip(self.orbit_state, keep) if k]
                if self.orbit_state is not None
                else None
            ),
            sensor=self.sensor,
        )

    def composite(self, start: date, end: date, speckle_window: int = 5) -> RadarComposite:
        """Temporal median in dB, then a spatial boxcar multi-look.

        The median is rank-based, so taking it in dB is identical to taking it
        in linear power and converting - unlike a mean, which must be done in
        linear power.
        """
        sub = self.select(start, end)
        finite = np.isfinite(sub.vv_db) & np.isfinite(sub.vh_db)
        vv = masking.masked_median(sub.vv_db, finite)
        vh = masking.masked_median(sub.vh_db, finite)
        if speckle_window > 1:
            vv = masking.boxcar(vv, speckle_window)
            vh = masking.boxcar(vh, speckle_window)
        return RadarComposite(
            grid=self.grid,
            start=start,
            end=end,
            vv_db=vv,
            vh_db=vh,
            n_obs=masking.observation_count(finite),
        )


@dataclass
class Scene:
    """The pair of series that the pipeline consumes, plus optional truth."""

    optical: OpticalSeries
    radar: Optional[RadarSeries]
    """May be ``None``. Several free catalogues carry no analysis-ready
    Sentinel-1 over Australia, and the pipeline falls back to temporal
    persistence as its confirming evidence rather than refusing to run."""

    grid: Grid
    truth: Optional[Mapping[str, object]] = None

    def __post_init__(self) -> None:
        if self.radar is not None and self.optical.grid.shape != self.radar.grid.shape:
            raise ValueError("optical and radar series are on different grids")


__all__ = [
    "OpticalComposite",
    "OpticalSeries",
    "RadarComposite",
    "RadarSeries",
    "S2_BANDS",
    "Scene",
    "to_datetime64",
]
