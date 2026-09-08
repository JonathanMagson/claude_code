"""Accuracy assessment against reference data.

The reference here is the synthetic ground truth, but the shape of the answer
is what a real validation against SLATS woody-change layers, aerial photo
interpretation or field sites would look like: pixel agreement, patch-level
matching, area bias, per-size-class recall, and - because it is the number
that decides whether a monitoring program is worth running - how long after
the event each sensor could first have raised an alert.

Reporting a single overall accuracy for a change map is close to meaningless.
Change is rare, so a map that detects nothing scores over 99% correct. What
matters is recall by patch size, precision against the specific things that
mimic clearing, and how much total area is over- or under-called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vegmon.fuse import ClearingMap
from vegmon.grid import Grid

#: Reporting size classes in hectares, chosen around the 0.5 ha minimum
#: mapping unit and the patch sizes that dominate NSW clearing statistics.
SIZE_CLASSES: Tuple[Tuple[str, float, float], ...] = (
    ("<0.5 ha", 0.0, 0.5),
    ("0.5-2 ha", 0.5, 2.0),
    ("2-10 ha", 2.0, 10.0),
    ("10-50 ha", 10.0, 50.0),
    (">50 ha", 50.0, float("inf")),
)


@dataclass
class PixelAccuracy:
    """Confusion-matrix statistics for one detection mask."""

    label: str
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else float("nan")

    @property
    def iou(self) -> float:
        denom = self.true_positive + self.false_positive + self.false_negative
        return self.true_positive / denom if denom else float("nan")

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "iou": round(self.iou, 4),
        }


@dataclass
class PatchMatch:
    """One reference patch and the detection that best covers it."""

    reference_id: int
    reference_area_ha: float
    size_class: str
    detected: bool
    detected_id: Optional[int] = None
    detected_area_ha: Optional[float] = None
    iou: float = 0.0
    tier: Optional[str] = None
    fragments: int = 0

    def as_dict(self) -> dict:
        return {
            "reference_id": self.reference_id,
            "reference_area_ha": round(self.reference_area_ha, 3),
            "size_class": self.size_class,
            "detected": self.detected,
            "detected_id": self.detected_id,
            "detected_area_ha": (
                round(self.detected_area_ha, 3) if self.detected_area_ha else None
            ),
            "iou": round(self.iou, 3),
            "tier": self.tier,
            "fragments": self.fragments,
        }


@dataclass
class ValidationReport:
    pixel: Dict[str, PixelAccuracy] = field(default_factory=dict)
    matches: List[PatchMatch] = field(default_factory=list)
    size_class_recall: Dict[str, dict] = field(default_factory=dict)
    decoys: Dict[str, dict] = field(default_factory=dict)
    area: Dict[str, float] = field(default_factory=dict)
    latency: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "pixel": {k: v.as_dict() for k, v in self.pixel.items()},
            "matches": [m.as_dict() for m in self.matches],
            "size_class_recall": self.size_class_recall,
            "decoys": self.decoys,
            "area": self.area,
            "latency": self.latency,
        }

    def headline(self) -> str:
        fused = self.pixel.get("fused_confirmed")
        lines = []
        if fused:
            lines.append(
                f"Confirmed tier: precision {fused.precision:.1%}, "
                f"recall {fused.recall:.1%}, IoU {fused.iou:.1%} (pixels)"
            )
        detected = sum(1 for m in self.matches if m.detected)
        lines.append(f"Reference patches found: {detected}/{len(self.matches)}")
        if self.area:
            lines.append(
                f"Area: {self.area['detected_ha']:.1f} ha detected vs "
                f"{self.area['reference_ha']:.1f} ha reference "
                f"({self.area['bias_percent']:+.1f}%)"
            )
        return "\n".join(lines)


def size_class_for(area_ha: float) -> str:
    for name, low, high in SIZE_CLASSES:
        if low <= area_ha < high:
            return name
    return SIZE_CLASSES[-1][0]


def pixel_accuracy(label: str, detected: np.ndarray, reference: np.ndarray) -> PixelAccuracy:
    detected = np.asarray(detected, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    return PixelAccuracy(
        label=label,
        true_positive=int((detected & reference).sum()),
        false_positive=int((detected & ~reference).sum()),
        false_negative=int((~detected & reference).sum()),
        true_negative=int((~detected & ~reference).sum()),
    )


def match_patches(
    clearing: ClearingMap,
    reference_labels: np.ndarray,
    reference_records: Sequence[dict],
    grid: Grid,
    min_iou: float = 0.20,
    tiers: Sequence[str] = ("confirmed",),
) -> List[PatchMatch]:
    """Match each reference patch to the detection that best overlaps it.

    Matching on IoU rather than on any overlap at all is deliberate: a
    detection that clips one corner of a 40 ha paddock should not be scored as
    having found it. ``fragments`` records how many separate detections landed
    inside the reference patch, which is the number that tells you whether the
    morphological cleanup is breaking real events apart.
    """
    tier_by_id = {r["patch_id"]: r["tier"] for r in clearing.records}
    keep_ids = {pid for pid, tier in tier_by_id.items() if tier in tiers}
    components = np.where(np.isin(clearing.components, list(keep_ids)), clearing.components, 0)

    matches: List[PatchMatch] = []
    for record in reference_records:
        reference_id = record["patch_id"]
        reference_mask = reference_labels == reference_id
        reference_pixels = int(reference_mask.sum())
        if reference_pixels == 0:
            continue

        overlapping = components[reference_mask]
        overlapping = overlapping[overlapping > 0]
        best_id, best_iou, best_pixels = None, 0.0, 0
        for candidate in np.unique(overlapping):
            candidate_mask = components == candidate
            intersection = int((candidate_mask & reference_mask).sum())
            union = int((candidate_mask | reference_mask).sum())
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_id, best_iou, best_pixels = int(candidate), iou, int(candidate_mask.sum())

        matches.append(
            PatchMatch(
                reference_id=reference_id,
                reference_area_ha=record["area_ha"],
                size_class=size_class_for(record["area_ha"]),
                detected=best_iou >= min_iou,
                detected_id=best_id if best_iou >= min_iou else None,
                detected_area_ha=grid.area_ha(best_pixels) if best_iou >= min_iou else None,
                iou=best_iou,
                tier=tier_by_id.get(best_id) if best_id else None,
                fragments=int(np.unique(overlapping).size),
            )
        )
    return matches


def summarise_size_classes(matches: Sequence[PatchMatch]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name, _, _ in SIZE_CLASSES:
        selected = [m for m in matches if m.size_class == name]
        if not selected:
            continue
        found = [m for m in selected if m.detected]
        out[name] = {
            "reference_patches": len(selected),
            "detected": len(found),
            "recall": round(len(found) / len(selected), 3),
            "mean_iou": round(float(np.mean([m.iou for m in selected])), 3),
            "reference_ha": round(float(sum(m.reference_area_ha for m in selected)), 2),
        }
    return out


def assess_decoys(
    clearing: ClearingMap,
    decoy_masks: Dict[str, np.ndarray],
    grid: Grid,
) -> Dict[str, dict]:
    """How much of each look-alike class each tier called clearing.

    This is the table that justifies fusing two sensors. A burn scar and a
    fallowed paddock both read as clearing to Sentinel-2 alone; what they do
    to the confirmed tier is the whole argument.
    """
    out: Dict[str, dict] = {}
    for name, mask in decoy_masks.items():
        total = int(mask.sum())
        if total == 0:
            continue
        entry = {"reference_ha": round(grid.area_ha(total), 2)}
        for tier in ("confirmed", "s2_only", "s1_only"):
            hit = int((clearing.mask(tier) & mask).sum())
            entry[f"{tier}_ha"] = round(grid.area_ha(hit), 2)
            entry[f"{tier}_fraction"] = round(hit / total, 4)
        out[name] = entry
    return out


def validate(
    clearing: ClearingMap,
    truth,
    optical_mask: Optional[np.ndarray] = None,
    radar_mask: Optional[np.ndarray] = None,
    latency: Optional[List[dict]] = None,
    min_iou: float = 0.20,
) -> ValidationReport:
    """Full accuracy assessment of a clearing map against synthetic truth."""
    grid = clearing.grid
    reference = truth.clearing_mask
    report = ValidationReport()

    report.pixel["fused_confirmed"] = pixel_accuracy(
        "confirmed tier (S1 + S2)", clearing.mask("confirmed"), reference
    )
    report.pixel["fused_any_tier"] = pixel_accuracy(
        "any tier", clearing.mask(), reference
    )
    if optical_mask is not None:
        report.pixel["sentinel2_only"] = pixel_accuracy(
            "Sentinel-2 detector alone", optical_mask, reference
        )
    if radar_mask is not None:
        report.pixel["sentinel1_only"] = pixel_accuracy(
            "Sentinel-1 detector alone", radar_mask, reference
        )

    reference_records = [
        r for r in truth.patches if r["kind"] == "clearing" and r["pixels"] > 0
    ]
    report.matches = match_patches(
        clearing, truth.labels, reference_records, grid, min_iou=min_iou
    )
    report.size_class_recall = summarise_size_classes(report.matches)

    decoys = {}
    fire = truth.fire_mask
    if fire.any():
        decoys["fire scar"] = fire
    crop_ids = [r["patch_id"] for r in truth.patches if r["kind"] == "crop"]
    if crop_ids:
        decoys["cropping"] = np.isin(truth.labels, crop_ids)
    fallow_ids = [r["patch_id"] for r in truth.patches if r["kind"] == "crop_fallow"]
    if fallow_ids:
        decoys["cropped then fallowed"] = np.isin(truth.labels, fallow_ids)
    decoys["stable background"] = truth.labels == 0
    report.decoys = assess_decoys(clearing, decoys, grid)

    detected_ha = clearing.area_ha("confirmed")
    reference_ha = float(sum(r["area_ha"] for r in reference_records))
    report.area = {
        "detected_ha": round(detected_ha, 2),
        "reference_ha": round(reference_ha, 2),
        "bias_ha": round(detected_ha - reference_ha, 2),
        "bias_percent": round(
            100.0 * (detected_ha - reference_ha) / reference_ha if reference_ha else 0.0, 2
        ),
    }

    if latency:
        s2 = [r["s2_latency_days"] for r in latency if r.get("s2_latency_days") is not None]
        s1 = [r["s1_latency_days"] for r in latency if r.get("s1_latency_days") is not None]
        report.latency = {
            "per_patch": latency,
            "s2_median_days": float(np.median(s2)) if s2 else None,
            "s1_median_days": float(np.median(s1)) if s1 else None,
            "s2_max_days": float(np.max(s2)) if s2 else None,
            "s1_max_days": float(np.max(s1)) if s1 else None,
            "advantage_days": (
                float(np.median(s2) - np.median(s1)) if s2 and s1 else None
            ),
        }
    return report


__all__ = [
    "PatchMatch",
    "PixelAccuracy",
    "SIZE_CLASSES",
    "ValidationReport",
    "assess_decoys",
    "match_patches",
    "pixel_accuracy",
    "size_class_for",
    "summarise_size_classes",
    "validate",
]
