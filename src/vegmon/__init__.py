"""vegmon - Sentinel-1/2 clearing detection and regrowth monitoring.

A self-contained demo pipeline for woody vegetation change monitoring:

  1. Sentinel-2 bitemporal spectral change detection (dNBR / dNDVI)
  2. Sentinel-1 VH backscatter drop detection (cloud-independent)
  3. Fusion of the two into confidence-tiered clearing patches
  4. Per-patch regrowth trajectory fitting and recovery classification

The pipeline runs against live STAC catalogues or a physically-motivated
synthetic datacube that ships with known ground truth, so every stage can be
validated without network access.
"""

__version__ = "0.1.0"

from vegmon.config import (
    AOI,
    DetectionConfig,
    Periods,
    PipelineConfig,
    RegrowthConfig,
)
from vegmon.grid import Grid

__all__ = [
    "AOI",
    "DetectionConfig",
    "Grid",
    "Periods",
    "PipelineConfig",
    "RegrowthConfig",
    "__version__",
]
