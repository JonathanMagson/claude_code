"""Figures: change-detection map panels and recovery trajectories.

Two rules shape everything here. Magnitude panels use one hue light-to-dark,
never a rainbow, so that "more change" always reads as "darker". And the three
recovery tracks are plotted as *recovery fraction* on a single shared axis
rather than in their native units, because NDVI, NBR and decibels have no
common scale and a second y-axis would invent a relationship between them that
does not exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from vegmon import palette as pal
from vegmon.detect import OpticalChange, RadarChange
from vegmon.fuse import ClearingMap
from vegmon.regrowth import RegrowthResult, recovery_curve
from vegmon.series import Scene


def _configure(matplotlib_module) -> None:
    matplotlib_module.rcParams.update(
        {
            "figure.facecolor": pal.SURFACE_LIGHT,
            "axes.facecolor": pal.SURFACE_LIGHT,
            "savefig.facecolor": pal.SURFACE_LIGHT,
            "font.family": ["DejaVu Sans"],
            "text.color": pal.INK_PRIMARY,
            "axes.labelcolor": pal.INK_SECONDARY,
            "axes.edgecolor": pal.BASELINE,
            "xtick.color": pal.INK_MUTED,
            "ytick.color": pal.INK_MUTED,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": pal.GRIDLINE,
            "grid.linewidth": 0.6,
        }
    )


def stretch_limits(band: np.ndarray, low: float = 2.0, high: float = 98.0):
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = np.nanpercentile(finite, [low, high])
    return (float(lo), float(hi if hi > lo else lo + 1e-6))


def _stretch(band: np.ndarray, limits=None, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Percentile stretch to 0-1 for display."""
    lo, hi = limits if limits is not None else stretch_limits(band, low, high)
    if hi <= lo:
        return np.zeros_like(band)
    return np.clip((band - lo) / (hi - lo), 0.0, 1.0)


FALSE_COLOUR_BANDS = ("swir2", "nir", "red")


def false_colour(composite, limits=None) -> np.ndarray:
    """SWIR2 / NIR / Red composite - the standard woody-vegetation view.

    Healthy woody vegetation reads green, bare and recently cleared ground
    reads magenta to pink, water reads near-black. It separates cleared from
    intact far better than true colour, which is why it is what gets put in
    front of an assessor.

    ``limits`` must be shared between a before/after pair. Stretching each
    date on its own percentiles is the classic way to manufacture a change
    that is not there - the display rescales, and unchanged ground shifts
    colour between the two panels.
    """
    return np.dstack(
        [
            _stretch(composite.band(band), None if limits is None else limits[band])
            for band in FALSE_COLOUR_BANDS
        ]
    )


def shared_stretch(*composites) -> Dict[str, tuple]:
    """Percentile limits per band, pooled across composites."""
    limits = {}
    for band in FALSE_COLOUR_BANDS:
        pooled = np.concatenate([c.band(band).ravel() for c in composites])
        limits[band] = stretch_limits(pooled)
    return limits


def plot_change_maps(
    scene: Scene,
    optical: OpticalChange,
    radar: RadarChange,
    clearing: ClearingMap,
    path: str | Path,
    truth=None,
    dpi: int = 130,
) -> Path:
    """Six-panel figure: before, after, both change surfaces, result, reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _configure(matplotlib)
    n_panels = 6 if truth is not None else 5
    rows = 2
    cols = 3
    figure, axes = plt.subplots(rows, cols, figsize=(13.5, 9.0))
    axes = axes.ravel()
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    limits = shared_stretch(optical.pre, optical.post)
    axes[0].imshow(false_colour(optical.pre, limits))
    axes[0].set_title(
        f"Sentinel-2 before\n{optical.pre.start} to {optical.pre.end}", loc="left"
    )
    axes[1].imshow(false_colour(optical.post, limits))
    axes[1].set_title(
        f"Sentinel-2 after\n{optical.post.start} to {optical.post.end}", loc="left"
    )

    dnbr_image = axes[2].imshow(
        optical.dnbr, cmap=pal.colormap(pal.BLUE_RAMP, "dnbr"), vmin=0.0, vmax=0.5
    )
    axes[2].set_title("Spectral change\ndNBR (pre - post, normalised)", loc="left")
    _colourbar(figure, dnbr_image, axes[2], "dNBR")

    vh_image = axes[3].imshow(
        radar.vh_drop_db, cmap=pal.colormap(pal.ORANGE_RAMP, "vh"), vmin=0.0, vmax=6.0
    )
    axes[3].set_title("Structural change\nSentinel-1 VH drop (dB)", loc="left")
    _colourbar(figure, vh_image, axes[3], "dB")

    axes[4].imshow(_tier_rgb(clearing), interpolation="nearest")
    summary = clearing.summary()
    axes[4].set_title(
        "Fused detection by confidence tier\n"
        f"{summary['confirmed']['patches']} confirmed patches, "
        f"{summary['confirmed']['area_ha']:.1f} ha",
        loc="left",
    )
    axes[4].legend(
        handles=[
            Patch(facecolor=pal.TIER_COLOURS[tier], label=f"{label} ({summary[tier]['area_ha']:.1f} ha)")
            for tier, label in (
                ("confirmed", "Confirmed: S1 + S2"),
                ("s2_only", "S2 only - review"),
                ("s1_only", "S1 only - review"),
            )
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.02),
        fontsize=7.5,
        labelcolor=pal.INK_SECONDARY,
    )

    if truth is not None:
        axes[5].imshow(_truth_rgb(truth), interpolation="nearest")
        axes[5].set_title(
            f"Reference truth\n{truth.total_cleared_ha():.1f} ha cleared, plus look-alikes",
            loc="left",
        )
        axes[5].legend(
            handles=[
                Patch(facecolor=pal.TRUTH_COLOURS["clearing"], label="Cleared (true positive target)"),
                Patch(facecolor=pal.TRUTH_COLOURS["fire"], label="Fire scar (not clearing)"),
                Patch(facecolor=pal.TRUTH_COLOURS["crop"], label="Cropping (not clearing)"),
                Patch(facecolor=pal.TRUTH_COLOURS["crop_fallow"], label="Fallowed crop (not clearing)"),
            ],
            loc="upper left",
            bbox_to_anchor=(0.0, -0.02),
            fontsize=7.5,
            labelcolor=pal.INK_SECONDARY,
            ncol=2,
        )
    else:
        axes[5].set_visible(False)

    figure.suptitle(
        "Land clearing detection: Sentinel-2 spectral change confirmed by Sentinel-1 structural change",
        x=0.012,
        ha="left",
        fontsize=13,
        color=pal.INK_PRIMARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def _colourbar(figure, image, axis, label: str) -> None:
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    bar.set_label(label, color=pal.INK_SECONDARY, fontsize=8)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, color=pal.INK_MUTED, labelcolor=pal.INK_MUTED)


def _hex_to_rgb(colour: str) -> np.ndarray:
    value = colour.lstrip("#")
    return np.array([int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def _tier_rgb(clearing: ClearingMap) -> np.ndarray:
    canvas = np.ones(clearing.grid.shape + (3,), dtype=np.float32) * _hex_to_rgb(pal.SURFACE_LIGHT)
    for tier, colour in pal.TIER_COLOURS.items():
        mask = clearing.mask(tier)
        if mask.any():
            canvas[mask] = _hex_to_rgb(colour)
    return canvas


def _truth_rgb(truth) -> np.ndarray:
    canvas = np.ones(truth.labels.shape + (3,), dtype=np.float32) * _hex_to_rgb(pal.SURFACE_LIGHT)
    for record in truth.patches:
        colour = pal.TRUTH_COLOURS.get(record["kind"])
        if colour is None:
            continue
        canvas[truth.labels == record["patch_id"]] = _hex_to_rgb(colour)
    return canvas


def plot_recovery(
    regrowth: RegrowthResult,
    path: str | Path,
    max_patches: int = 9,
    dpi: int = 130,
) -> Optional[Path]:
    """Small multiples of recovery trajectory, one panel per patch.

    All three tracks are shown as recovery fraction on a shared 0-1 axis - 0 is
    the post-clearing trough, 1 is the pre-clearing baseline. Indexing to a
    common base is what makes NDVI, NBR and decibels comparable in one frame
    without a second y-axis.
    """
    if not regrowth.patches:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _configure(matplotlib)
    patches = sorted(regrowth.patches, key=lambda p: -p.area_ha)[:max_patches]
    cols = min(3, len(patches))
    rows = int(np.ceil(len(patches) / cols))
    figure, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 3.1 * rows), squeeze=False, sharex=True, sharey=True
    )
    flat = axes.ravel()

    threshold = regrowth.config.recovered_fraction
    for index, patch in enumerate(patches):
        axis = flat[index]
        axis.grid(axis="y", linewidth=0.6, color=pal.GRIDLINE)
        axis.set_axisbelow(True)
        axis.axhline(
            threshold, color=pal.BASELINE, linewidth=1.0, zorder=1
        )
        axis.text(
            0.02,
            threshold + 0.03,
            f"R80P ({threshold:.0%} of baseline)",
            fontsize=7,
            color=pal.INK_MUTED,
            transform=axis.get_yaxis_transform(),
        )

        endpoints = []
        for track in ("ndvi", "nbr", "vh"):
            fit = patch.fits.get(track)
            series = patch.series.get(track)
            if fit is None or series is None or fit.baseline is None or fit.trough is None:
                continue
            span = fit.baseline - fit.trough
            if abs(span) < 1e-6:
                continue
            years, values = series.since(patch.event_date)
            if years.size == 0:
                continue
            fraction = (values - fit.trough) / span
            colour = pal.TRACK_COLOURS[track]
            # Observations stay visible but recessive: the quarterly medians
            # still carry a phenological cycle, and showing it is honest -
            # it is the noise the fitted trajectory has to see through.
            axis.plot(
                years, fraction, color=colour, linewidth=1.0, alpha=0.45,
                marker="o", markersize=2.6, markeredgewidth=0, zorder=3,
            )
            if fit.tau_years and fit.asymptote is not None:
                smooth = np.linspace(0.0, float(years.max()), 120)
                curve = (
                    recovery_curve(smooth, fit.asymptote, fit.trough, fit.tau_years) - fit.trough
                ) / span
                axis.plot(smooth, curve, color=colour, linewidth=2.0, zorder=4)
                endpoints.append((float(curve[-1]), track, colour))
            else:
                endpoints.append((float(fraction[-1]), track, colour))

        _label_endpoints(axis, endpoints, x=float(axis.get_xlim()[1]))

        if patch.reclear_date is not None and patch.event_date is not None:
            when = (patch.reclear_date - patch.event_date).days / 365.25
            axis.axvline(when, color=pal.STATUS["critical"], linewidth=1.4, zorder=2)
            axis.annotate(
                "re-cleared",
                xy=(when, 1.30),
                xytext=(3, 0),
                textcoords="offset points",
                fontsize=7,
                color=pal.STATUS["critical"],
                va="top",
                zorder=5,
            )

        axis.set_title(
            f"Patch {patch.patch_id} - {patch.area_ha:.1f} ha - {patch.classification.replace('_', ' ')}",
            loc="left",
            color=pal.INK_PRIMARY,
        )
        axis.set_ylim(-0.15, 1.35)

    for axis in flat[len(patches) :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Years since clearing")
    for row in axes:
        row[0].set_ylabel("Recovery fraction")

    figure.legend(
        handles=[
            Line2D([0], [0], color=pal.TRACK_COLOURS[t], linewidth=2.0, label=pal.TRACK_LABELS[t])
            for t in ("ndvi", "nbr", "vh")
        ],
        loc="lower center",
        ncol=3,
        fontsize=9,
        labelcolor=pal.INK_SECONDARY,
        bbox_to_anchor=(0.5, -0.005),
    )
    figure.suptitle(
        "Post-clearing recovery, indexed to the pre-clearing baseline",
        x=0.012,
        ha="left",
        fontsize=13,
        color=pal.INK_PRIMARY,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def _label_endpoints(axis, endpoints, x: float, min_gap: float = 0.14) -> None:
    """Direct-label series endpoints, nudged apart so they never overlap.

    Three series is few enough to name in place, which keeps identity off
    colour alone - but only if the labels stay legible. Collided labels are
    worse than no labels, so they are pushed apart along y before drawing.
    """
    if not endpoints:
        return
    ordered = sorted(endpoints, key=lambda item: item[0])
    positions = [value for value, _, _ in ordered]
    for index in range(1, len(positions)):
        if positions[index] - positions[index - 1] < min_gap:
            positions[index] = positions[index - 1] + min_gap
    for position, (value, track, colour) in zip(positions, ordered):
        axis.annotate(
            track.upper(),
            xy=(x, position),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7,
            color=colour,
            va="center",
            annotation_clip=False,
            zorder=6,
        )


__all__ = [
    "false_colour",
    "plot_change_maps",
    "plot_recovery",
    "shared_stretch",
    "stretch_limits",
]
