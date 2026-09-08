"""The plotting palette, and the reasoning behind it.

Three categorical slots carry identity throughout: the detection tier in the
maps and the sensor track in the recovery plots. Three is not an arbitrary cap
- it is the largest set from this palette that clears colour-vision-deficiency
separation on *every* pair rather than only on adjacent ones, which is what a
map needs, since any two classes can end up side by side.

Magnitude uses one hue, light to dark. Two magnitude panels appear together
(spectral change and backscatter change), so the second takes the next
categorical hue as its own single-hue ramp rather than introducing a rainbow.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# --- categorical: identity -------------------------------------------------
SERIES_LIGHT: Tuple[str, str, str] = ("#2a78d6", "#eb6834", "#1baf7a")
SERIES_DARK: Tuple[str, str, str] = ("#3987e5", "#d95926", "#199e70")

#: Detection tiers, mapped to fixed slots. The colour follows the tier, never
#: its rank in a particular run, so a map with no s1_only patches still shows
#: confirmed in blue.
TIER_COLOURS: Dict[str, str] = {
    "confirmed": SERIES_LIGHT[0],
    "s2_only": SERIES_LIGHT[1],
    "s1_only": SERIES_LIGHT[2],
}

#: Recovery tracks, same fixed-slot rule.
TRACK_COLOURS: Dict[str, str] = {
    "ndvi": SERIES_LIGHT[0],
    "nbr": SERIES_LIGHT[1],
    "vh": SERIES_LIGHT[2],
}

TRACK_LABELS: Dict[str, str] = {
    "ndvi": "NDVI (greenness)",
    "nbr": "NBR (canopy + moisture)",
    "vh": "Sentinel-1 VH (structure)",
}

# --- sequential: magnitude -------------------------------------------------
BLUE_RAMP: Tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
ORANGE_RAMP: Tuple[str, ...] = (
    "#fce4d6", "#fad3bd", "#f7c1a5", "#f4b08c", "#f19f74", "#ee8b56",
    "#eb6834", "#d95926", "#c04e21", "#a6431c", "#8c3817", "#732d13", "#59230e",
)

# --- reference classes in the truth panel ----------------------------------
# These are not series - they label what the ground actually is - so they use
# the neutral ink and status tokens rather than borrowing categorical slots.
TRUTH_COLOURS: Dict[str, str] = {
    "clearing": "#2a78d6",
    "fire": "#d03b3b",
    "crop": "#eda100",
    "crop_fallow": "#8a6d3b",
}

# --- chrome ----------------------------------------------------------------
SURFACE_LIGHT = "#fcfcfb"
SURFACE_DARK = "#1a1a19"
PAGE_LIGHT = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

STATUS: Dict[str, str] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

#: Recovery classes. These *are* states, so they wear status tokens - and are
#: always shipped with a text label, never colour alone.
CLASS_COLOURS: Dict[str, str] = {
    "recovered": STATUS["good"],
    "recovering": STATUS["warning"],
    "stalled": STATUS["serious"],
    "recleared": STATUS["critical"],
    "insufficient_data": INK_MUTED,
}


def colormap(ramp: Sequence[str], name: str = "vegmon"):
    """Build a matplotlib colormap from a single-hue ramp."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(ramp))


def relative_luminance(hex_colour: str) -> float:
    """WCAG relative luminance, used to assert ramps are monotonic."""
    value = hex_colour.lstrip("#")
    channels = [int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def is_monotonic(ramp: Sequence[str]) -> bool:
    """True if a sequential ramp darkens monotonically.

    A ramp that reverses anywhere makes two different magnitudes read as the
    same value, which is the whole failure mode sequential encoding exists to
    avoid.
    """
    luminances = [relative_luminance(c) for c in ramp]
    return all(b < a for a, b in zip(luminances, luminances[1:]))


__all__ = [
    "BLUE_RAMP",
    "CLASS_COLOURS",
    "GRIDLINE",
    "INK_MUTED",
    "INK_PRIMARY",
    "INK_SECONDARY",
    "ORANGE_RAMP",
    "SERIES_DARK",
    "SERIES_LIGHT",
    "STATUS",
    "SURFACE_DARK",
    "SURFACE_LIGHT",
    "TIER_COLOURS",
    "TRACK_COLOURS",
    "TRACK_LABELS",
    "TRUTH_COLOURS",
    "colormap",
    "is_monotonic",
    "relative_luminance",
]
