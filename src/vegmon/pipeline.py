"""End-to-end orchestration: detect, fuse, date, fit recovery, validate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from vegmon.config import PipelineConfig
from vegmon.detect import (
    EventTiming,
    OpticalChange,
    RadarChange,
    attach_true_event_dates,
    detect_optical,
    detect_radar,
    estimate_event_timing,
)
from vegmon.fuse import ClearingMap, fuse
from vegmon.regrowth import RegrowthResult, analyse_regrowth
from vegmon.series import Scene
from vegmon.validation import ValidationReport, validate


@dataclass
class PipelineResult:
    """Everything one run produces."""

    config: PipelineConfig
    scene: Scene
    optical: OpticalChange
    radar: RadarChange
    clearing: ClearingMap
    regrowth: RegrowthResult
    timing: List[EventTiming] = field(default_factory=list)
    validation: Optional[ValidationReport] = None
    timings_seconds: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict:
        """The headline numbers, ready for a report or a JSON dump."""
        out = {
            "aoi": self.config.aoi.name,
            "crs": self.config.aoi.crs,
            "area_ha": round(
                self.scene.grid.width * self.scene.grid.height * self.scene.grid.pixel_area_ha, 1
            ),
            "optical_acquisitions": len(self.scene.optical),
            "radar_acquisitions": len(self.scene.radar),
            "periods": {
                "pre": [self.config.periods.pre_start.isoformat(), self.config.periods.pre_end.isoformat()],
                "post": [self.config.periods.post_start.isoformat(), self.config.periods.post_end.isoformat()],
                "series": [
                    self.config.periods.series_start.isoformat(),
                    self.config.periods.series_end.isoformat(),
                ],
            },
            "normalisation_offsets": {**self.optical.offsets, **self.radar.offsets},
            "tiers": self.clearing.summary(),
            "recovery": self.regrowth.summary(),
            "runtime_seconds": {k: round(v, 2) for k, v in self.timings_seconds.items()},
        }
        if self.validation is not None:
            out["validation"] = self.validation.as_dict()
        return out


def run_pipeline(
    scene: Scene,
    config: PipelineConfig,
    validate_against_truth: bool = True,
    regrowth_tiers: tuple = ("confirmed",),
    verbose: bool = False,
) -> PipelineResult:
    """Run every stage against an already-loaded scene.

    The scene can come from :mod:`vegmon.synthetic` or :mod:`vegmon.stac` -
    the stages below cannot tell the difference and do not need to.
    """
    clock: Dict[str, float] = {}

    def step(name: str, func):
        start = time.perf_counter()
        result = func()
        clock[name] = time.perf_counter() - start
        if verbose:
            print(f"  {name}: {clock[name]:.1f}s", flush=True)
        return result

    optical = step(
        "detect_optical", lambda: detect_optical(scene.optical, config.periods, config.detection)
    )
    radar = step(
        "detect_radar", lambda: detect_radar(scene.radar, config.periods, config.detection)
    )
    clearing = step("fuse", lambda: fuse(optical, radar, config.detection))

    # Date each detected patch from the time series rather than assuming the
    # midpoint of the gap. A regrowth curve fitted from the wrong start date
    # mixes pre-event observations into the recovery limb and flatters it.
    timing = step(
        "estimate_event_dates",
        lambda: estimate_event_timing(
            scene.optical,
            scene.radar,
            clearing.components,
            [r["patch_id"] for r in clearing.records if r["tier"] in regrowth_tiers],
            config.periods,
            config.detection,
        ),
    )
    event_dates: Dict[int, date] = {
        t.patch_id: t.first_detection_date for t in timing if t.first_detection_date
    }

    regrowth = step(
        "analyse_regrowth",
        lambda: analyse_regrowth(
            scene.optical,
            scene.radar,
            clearing,
            config.periods,
            config.regrowth,
            event_dates=event_dates,
            tiers=regrowth_tiers,
        ),
    )

    report = None
    if validate_against_truth and scene.truth and "synthetic" in scene.truth:
        truth = scene.truth["synthetic"]

        def _validate():
            reference_ids = [r["patch_id"] for r in truth.patches if r["kind"] == "clearing"]
            reference_timing = attach_true_event_dates(
                estimate_event_timing(
                    scene.optical,
                    scene.radar,
                    truth.labels,
                    reference_ids,
                    config.periods,
                    config.detection,
                ),
                truth.patches,
            )
            return validate(
                clearing,
                truth,
                optical_mask=optical.mask,
                radar_mask=radar.mask,
                latency=[t.as_dict() for t in reference_timing],
            )

        report = step("validate", _validate)

    return PipelineResult(
        config=config,
        scene=scene,
        optical=optical,
        radar=radar,
        clearing=clearing,
        regrowth=regrowth,
        timing=timing,
        validation=report,
        timings_seconds=clock,
    )


__all__ = ["PipelineResult", "run_pipeline"]
