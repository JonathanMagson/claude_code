"""End-to-end orchestration: detect, fuse, date, fit recovery, validate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from vegmon.config import Periods, PipelineConfig
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
            "radar_acquisitions": len(self.scene.radar) if self.scene.radar else 0,
            "periods": {
                "pre": [self.config.periods.pre_start.isoformat(), self.config.periods.pre_end.isoformat()],
                "post": [self.config.periods.post_start.isoformat(), self.config.periods.post_end.isoformat()],
                "series": [
                    self.config.periods.series_start.isoformat(),
                    self.config.periods.series_end.isoformat(),
                ],
            },
            "normalisation_offsets": {**self.optical.offsets, **self.radar.offsets},
            "resolved_thresholds": self.optical.thresholds,
            "confirming_source": self.clearing.confirming_source,
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
    if scene.radar is not None:
        confirming = step(
            "detect_radar", lambda: detect_radar(scene.radar, config.periods, config.detection)
        )
    else:
        # No reachable Sentinel-1 for this AOI. Persistence is a weaker second
        # opinion - it does not reject fire - but it is independent of the
        # observations that produced the detection, which is the property that
        # matters. The tier description carried on the map says which was used.
        from vegmon.persistence import detect_persistence

        confirming = step(
            "detect_persistence",
            lambda: detect_persistence(
                scene.optical, optical, config.periods, config.detection
            ),
        )
    radar = confirming
    clearing = step("fuse", lambda: fuse(optical, confirming, config.detection))

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
            # Measure recovery against undisturbed *woody* ground, not against
            # the whole scene. On a cropping-belt AOI the scene median is
            # mostly paddock, and normalising regrowth against a crop rotation
            # imports exactly the cycle the normalisation exists to remove.
            reference_mask=(
                optical.gates["woody_before"]
                & optical.gates["persistently_woody"]
                & (clearing.components == 0)
            ),
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


def scan_epochs(
    scene: Scene,
    config: PipelineConfig,
    first_year: int,
    last_year: int,
    start_month_day: tuple = (3, 1),
    end_month_day: tuple = (9, 30),
) -> List[dict]:
    """Run the detector over every consecutive year pair in the record.

    A single before/after pair answers "was this cleared between these two
    dates". A monitoring programme needs the other question - "what happened
    each year" - and the answer is a different shape: an annual table where
    the interesting rows are the ones that stand out from their neighbours.
    Running the epochs together also makes the confounds visible. A year pair
    straddling a drought onset lights up across the whole scene in small
    patches; a clearing year produces a few large ones. Neither is obvious
    from one pair alone.
    """
    from vegmon.detect import detect_optical, detect_radar
    from vegmon.fuse import fuse

    rows: List[dict] = []
    for year in range(first_year, last_year):
        periods = Periods(
            pre_start=date(year, *start_month_day),
            pre_end=date(year, *end_month_day),
            post_start=date(year + 1, *start_month_day),
            post_end=date(year + 1, *end_month_day),
            series_start=config.periods.series_start,
            series_end=config.periods.series_end,
        )
        try:
            optical = detect_optical(
                scene.optical, periods, config.detection, check_phenology=False
            )
            if scene.radar is not None:
                confirming = detect_radar(scene.radar, periods, config.detection)
            else:
                from vegmon.persistence import detect_persistence

                confirming = detect_persistence(
                    scene.optical, optical, periods, config.detection
                )
            clearing = fuse(optical, confirming, config.detection)
        except Exception as exc:  # a short or cloud-starved epoch is not fatal
            rows.append({"epoch": f"{year}-{year + 1}", "error": str(exc)[:120]})
            continue

        summary = clearing.summary()
        woody = (
            optical.gates["woody_before"]
            & optical.gates["persistently_woody"]
            & optical.gates["steady_through_the_year"]
        )
        confirmed = [r for r in clearing.records if r["tier"] == "confirmed"]
        rows.append(
            {
                "epoch": f"{year}-{year + 1}",
                "woody_fraction": round(float(woody.mean()), 4),
                "dnbr_threshold": round(optical.thresholds["dnbr_threshold"], 4),
                "confirmed_patches": summary["confirmed"]["patches"],
                "confirmed_ha": summary["confirmed"]["area_ha"],
                "review_ha": round(
                    summary["s2_only"]["area_ha"] + summary["s1_only"]["area_ha"], 2
                ),
                "largest_patch_ha": round(
                    max((r["area_ha"] for r in confirmed), default=0.0), 2
                ),
                "median_patch_ha": round(
                    float(np.median([r["area_ha"] for r in confirmed])) if confirmed else 0.0, 2
                ),
            }
        )
    return rows


__all__ = ["PipelineResult", "run_pipeline", "scan_epochs"]
