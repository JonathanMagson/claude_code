"""Self-contained HTML report.

No CDN, no build step, no network: one file that opens anywhere, which is
what makes it usable as an attachment on a compliance matter or a briefing.

Chart conventions follow the same rules as the figures - one axis per chart,
colour fixed to the entity rather than to its rank, a legend plus direct
labels so identity never rests on colour alone, and a table view behind every
chart for screen readers, printing and anyone who wants the numbers.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from vegmon import palette as pal
from vegmon.fuse import TIER_DESCRIPTIONS
from vegmon.pipeline import PipelineResult
from vegmon.regrowth import CLASS_DESCRIPTIONS

_CSS = """
:root { color-scheme: light; }
.viz-root {
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --gridline: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --status-good: #0ca30c; --status-warning: #fab219;
  --status-serious: #ec835a; --status-critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --gridline: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
  --gridline: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px; line-height: 1.55; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 24px 80px; }
h1 { font-size: 28px; letter-spacing: -0.015em; margin: 0 0 6px; font-weight: 620; }
h2 { font-size: 19px; margin: 44px 0 6px; font-weight: 600; letter-spacing: -0.01em; }
h3 { font-size: 15px; margin: 26px 0 8px; font-weight: 600; }
p  { color: var(--text-secondary); margin: 8px 0 14px; max-width: 74ch; }
.lede { font-size: 16px; color: var(--text-secondary); max-width: 74ch; }
.card { background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px 22px; margin: 16px 0; }
.tiles { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); margin: 20px 0 8px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; }
.tile .k { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); }
.tile .v { font-size: 27px; font-weight: 620; letter-spacing: -0.02em; margin-top: 2px; line-height: 1.15; }
.tile .s { font-size: 12.5px; color: var(--text-secondary); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 12px 7px 0; border-bottom: 1px solid var(--gridline); }
th { color: var(--text-muted); font-weight: 600; font-size: 12px;
  text-transform: uppercase; letter-spacing: 0.05em; }
td.num, th.num { text-align: right; padding-right: 18px; }
details { margin: 10px 0 0; } summary { cursor: pointer; color: var(--text-secondary); font-size: 13px; }
figure { margin: 18px 0; background: #fcfcfb; border: 1px solid var(--border);
  border-radius: 12px; padding: 12px 12px 4px; }
figure img { width: 100%; border-radius: 6px; display: block; }
figcaption { font-size: 13px; color: #52514e; margin: 8px 4px 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 4px 0 14px; font-size: 13px; color: var(--text-secondary); }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
.chip { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px;
  padding: 2px 9px; border-radius: 999px; border: 1px solid var(--border); color: var(--text-secondary); }
.note { border-left: 3px solid var(--series-1); padding: 2px 0 2px 14px; margin: 16px 0;
  color: var(--text-secondary); font-size: 14px; max-width: 74ch; }
.warn { border-left-color: var(--status-warning); }
#tt { position: fixed; pointer-events: none; opacity: 0; transition: opacity .09s;
  background: var(--surface-1); color: var(--text-primary); border: 1px solid var(--border);
  border-radius: 8px; padding: 7px 11px; font-size: 12.5px; box-shadow: 0 6px 22px rgba(0,0,0,.14);
  z-index: 40; white-space: nowrap; }
.toggle { float: right; font-size: 12.5px; background: var(--surface-1); color: var(--text-secondary);
  border: 1px solid var(--border); border-radius: 999px; padding: 5px 13px; cursor: pointer; }
svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
footer { margin-top: 54px; padding-top: 18px; border-top: 1px solid var(--gridline);
  color: var(--text-muted); font-size: 12.5px; }
"""

_JS = """
const tt = document.getElementById('tt');
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', e => {
    tt.textContent = el.getAttribute('data-tip');
    tt.style.opacity = 1;
    tt.style.left = Math.min(e.clientX + 14, window.innerWidth - tt.offsetWidth - 10) + 'px';
    tt.style.top = (e.clientY - 34) + 'px';
  });
  el.addEventListener('mouseleave', () => { tt.style.opacity = 0; });
});
document.querySelector('.toggle').addEventListener('click', () => {
  const root = document.documentElement;
  const dark = getComputedStyle(document.body).backgroundColor === 'rgb(13, 13, 13)';
  root.setAttribute('data-theme', dark ? 'light' : 'dark');
});
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _rounded_bar(x: float, y: float, width: float, height: float, radius: float = 4.0) -> str:
    """A bar with only the data end rounded, anchored flat to the baseline."""
    radius = max(0.0, min(radius, width, height / 2.0))
    if width <= 0:
        return ""
    return (
        f"M{x},{y} H{x + width - radius} "
        f"Q{x + width},{y} {x + width},{y + radius} "
        f"V{y + height - radius} "
        f"Q{x + width},{y + height} {x + width - radius},{y + height} "
        f"H{x} Z"
    )


def bar_chart(
    rows: Sequence[dict],
    max_value: Optional[float] = None,
    unit: str = "",
    width: int = 700,
    row_height: int = 34,
    label_width: int = 190,
) -> str:
    """Horizontal bars: one colour per entity, value labelled at the bar end."""
    if not rows:
        return "<p>No data.</p>"
    max_value = max_value or max((r["value"] for r in rows), default=1.0) or 1.0
    plot_width = width - label_width - 74
    height = row_height * len(rows) + 10
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="bar chart">'
    ]
    for index, row in enumerate(rows):
        y = index * row_height + 5
        bar_height = row_height - 14
        length = (row["value"] / max_value) * plot_width if max_value else 0.0
        colour = row.get("colour", "var(--series-1)")
        tip = row.get("tip", f"{row['label']}: {row['value']:g}{unit}")
        parts.append(
            f'<text x="0" y="{y + bar_height / 2 + 4}" font-size="13" '
            f'fill="var(--text-secondary)">{_esc(row["label"])}</text>'
        )
        if length > 0.5:
            parts.append(
                f'<path d="{_rounded_bar(label_width, y, length, bar_height)}" '
                f'fill="{colour}" data-tip="{_esc(tip)}"><title>{_esc(tip)}</title></path>'
            )
        value_label = row.get("value_label") or f"{row['value']:g}{unit}"
        parts.append(
            f'<text x="{label_width + max(length, 0) + 8}" y="{y + bar_height / 2 + 4}" '
            f'font-size="12.5" fill="var(--text-secondary)" '
            f'style="font-variant-numeric:tabular-nums">{_esc(value_label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def dumbbell_chart(
    rows: Sequence[dict],
    width: int = 700,
    row_height: int = 30,
    label_width: int = 150,
    unit: str = " days",
) -> str:
    """Paired dots joined by a rule - the right form for a two-point gap.

    Two bars per row would put the reader's eye on the bar lengths; the number
    that matters here is the *distance between* them.
    """
    if not rows:
        return "<p>No data.</p>"
    max_value = max(max(r["a"], r["b"]) for r in rows) or 1.0
    plot_width = width - label_width - 96
    height = row_height * len(rows) + 30

    def scale(value: float) -> float:
        return label_width + (value / max_value) * plot_width

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="paired dot chart">'
    ]
    for tick in range(0, int(max_value) + 1, max(1, int(max_value) // 4)):
        x = scale(tick)
        parts.append(
            f'<line x1="{x}" y1="6" x2="{x}" y2="{height - 22}" stroke="var(--gridline)" stroke-width="1"/>'
            f'<text x="{x}" y="{height - 6}" font-size="11" fill="var(--text-muted)" '
            f'text-anchor="middle">{tick}</text>'
        )
    for index, row in enumerate(rows):
        y = index * row_height + 18
        xa, xb = scale(row["a"]), scale(row["b"])
        tip = (
            f"{row['label']}: Sentinel-1 {row['b']:g}{unit}, "
            f"Sentinel-2 {row['a']:g}{unit} ({row['a'] - row['b']:+g}{unit})"
        )
        parts.append(
            f'<text x="0" y="{y + 4}" font-size="13" fill="var(--text-secondary)">{_esc(row["label"])}</text>'
            f'<line x1="{min(xa, xb)}" y1="{y}" x2="{max(xa, xb)}" y2="{y}" '
            f'stroke="var(--baseline)" stroke-width="2"/>'
            f'<circle cx="{xb}" cy="{y}" r="5.5" fill="var(--series-3)" stroke="var(--surface-1)" '
            f'stroke-width="2" data-tip="{_esc(tip)}"><title>{_esc(tip)}</title></circle>'
            f'<circle cx="{xa}" cy="{y}" r="5.5" fill="var(--series-1)" stroke="var(--surface-1)" '
            f'stroke-width="2" data-tip="{_esc(tip)}"><title>{_esc(tip)}</title></circle>'
        )
    parts.append(
        f'<text x="{width - 88}" y="{height - 6}" font-size="11" fill="var(--text-muted)">'
        f'days after clearing</text></svg>'
    )
    return "".join(parts)


def _table(headers: Sequence[str], rows: Sequence[Sequence], numeric_from: int = 1) -> str:
    head = "".join(
        f'<th class="{"num" if i >= numeric_from else ""}">{_esc(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{_esc(cell)}</td>'
            for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _details_table(summary: str, headers, rows, numeric_from: int = 1) -> str:
    return (
        f"<details><summary>{_esc(summary)}</summary>"
        f"{_table(headers, rows, numeric_from)}</details>"
    )


def _tile(key: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="tile"><div class="k">{_esc(key)}</div>'
        f'<div class="v">{_esc(value)}</div>'
        f'<div class="s">{_esc(sub)}</div></div>'
    )


def _embed_image(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img alt="{_esc(caption)}" src="data:image/png;base64,{encoded}"/>'
        f"<figcaption>{_esc(caption)}</figcaption></figure>"
    )


def build_report(
    result: PipelineResult,
    path: str | Path,
    figures: Optional[Dict[str, Path]] = None,
    title: str = "Land clearing and regrowth monitoring",
) -> Path:
    """Render the full pipeline result to one self-contained HTML file."""
    figures = figures or {}
    config = result.config
    clearing = result.clearing
    tiers = clearing.summary()
    validation = result.validation

    tier_rows = [
        {
            "label": label,
            "value": tiers[key]["area_ha"],
            "value_label": f"{tiers[key]['area_ha']:.1f} ha  ({tiers[key]['patches']})",
            "colour": f"var(--series-{slot})",
            "tip": f"{label}: {tiers[key]['area_ha']:.1f} ha across {tiers[key]['patches']} patches",
        }
        for key, label, slot in (
            ("confirmed", "Confirmed (S1 + S2)", 1),
            ("s2_only", "Sentinel-2 only", 2),
            ("s1_only", "Sentinel-1 only", 3),
        )
    ]

    recovery = result.regrowth.summary()
    recovery_rows = [
        {
            "label": name.replace("_", " ").title(),
            "value": entry["area_ha"],
            "value_label": f"{entry['area_ha']:.1f} ha  ({entry['patches']})",
            "colour": pal.CLASS_COLOURS.get(name, pal.INK_MUTED),
            "tip": f"{name}: {entry['area_ha']:.1f} ha across {entry['patches']} patches",
        }
        for name, entry in recovery.items()
    ]

    parts: List[str] = []
    parts.append(f"<h1>{_esc(title)}</h1>")
    parts.append(
        f'<p class="lede">{_esc(config.aoi.name)} &middot; '
        f'{_esc(config.aoi.crs)} &middot; {result.summary()["area_ha"]:.0f} ha analysed &middot; '
        f'baseline {config.periods.pre_start}&ndash;{config.periods.pre_end}, '
        f'detection {config.periods.post_start}&ndash;{config.periods.post_end}, '
        f'recovery record to {config.periods.series_end}.</p>'
    )

    # --- headline tiles ---------------------------------------------------
    tile_html = [
        _tile(
            "Confirmed clearing",
            f"{tiers['confirmed']['area_ha']:.1f} ha",
            f"{tiers['confirmed']['patches']} patches, both sensors",
        ),
        _tile(
            "Flagged for review",
            f"{tiers['s2_only']['area_ha'] + tiers['s1_only']['area_ha']:.1f} ha",
            "single-sensor detections",
        ),
        _tile(
            "Observations used",
            f"{len(result.scene.optical) + len(result.scene.radar):,}",
            f"{len(result.scene.optical)} Sentinel-2 + {len(result.scene.radar)} Sentinel-1",
        ),
    ]
    if validation:
        fused = validation.pixel["fused_confirmed"]
        tile_html.append(
            _tile("Precision", f"{fused.precision:.1%}", "confirmed tier, pixel level")
        )
        tile_html.append(_tile("Recall", f"{fused.recall:.1%}", "confirmed tier, pixel level"))
        advantage = (validation.latency or {}).get("advantage_days")
        if advantage is not None:
            tile_html.append(
                _tile(
                    "Radar time advantage",
                    f"{advantage:.0f} days",
                    "median, ahead of the first usable optical view",
                )
            )
    parts.append(f'<div class="tiles">{"".join(tile_html)}</div>')

    # --- detection --------------------------------------------------------
    parts.append("<h2>Detection</h2>")
    parts.append(
        "<p>Sentinel-2 supplies the spectral evidence and Sentinel-1 the structural "
        "evidence. Only where the two agree is a patch called clearing; a detection "
        "on one sensor alone is reported, not discarded, because the single-sensor "
        "tiers are informative in themselves.</p>"
    )
    parts.append('<div class="legend">')
    for key, label, slot in (
        ("confirmed", "Confirmed (S1 + S2)", 1),
        ("s2_only", "Sentinel-2 only", 2),
        ("s1_only", "Sentinel-1 only", 3),
    ):
        parts.append(
            f'<span><i class="swatch" style="background:var(--series-{slot})"></i>'
            f"{_esc(label)} &mdash; {_esc(TIER_DESCRIPTIONS[key].split('.')[0])}.</span>"
        )
    parts.append("</div>")
    parts.append(f'<div class="card">{bar_chart(tier_rows, unit=" ha")}</div>')
    parts.append(
        _details_table(
            "Table view: area by confidence tier",
            ["Tier", "Patches", "Area (ha)"],
            [[r["label"], tiers[k]["patches"], f"{tiers[k]['area_ha']:.2f}"]
             for r, k in zip(tier_rows, ("confirmed", "s2_only", "s1_only"))],
        )
    )
    parts.append(
        _details_table(
            "Table view: every detected patch",
            ["Patch", "Tier", "Area (ha)", "dNBR", "VH drop (dB)", "Agreement", "Pre NDVI"],
            [
                [
                    r["patch_id"], r["tier"], f"{r['area_ha']:.2f}",
                    r["mean_dnbr"], r["mean_vh_drop_db"],
                    f"{r['agreement_fraction']:.0%}", r["pre_ndvi"],
                ]
                for r in sorted(clearing.records, key=lambda r: -r["area_ha"])
            ],
            numeric_from=2,
        )
    )
    if "change_maps" in figures:
        parts.append(
            _embed_image(
                Path(figures["change_maps"]),
                "Before and after false colour (SWIR2/NIR/Red, shared stretch), the two "
                "change surfaces, the fused result, and the reference truth.",
            )
        )

    parts.append(
        f'<div class="note">Inter-date normalisation removed a '
        f'{result.optical.offsets["dnbr"]:+.3f} dNBR and '
        f'{result.radar.offsets["vh_db"]:+.2f} dB common-mode offset before thresholding. '
        "Without that step the same thresholds would have to be re-tuned for every "
        "date pair.</div>"
    )

    # --- regrowth ---------------------------------------------------------
    parts.append("<h2>Regrowth</h2>")
    parts.append(
        "<p>Each confirmed patch is dated from its own time series, then followed on "
        "three tracks: NDVI for greenness, NBR for canopy condition, and Sentinel-1 VH "
        "for structure. They are deliberately reported side by side. Grass colonises a "
        "cleared paddock within a season and pushes NDVI back towards baseline while no "
        "woody vegetation has returned at all, so a patch whose NDVI has recovered but "
        "whose NBR and VH have not is grassland, not regrowth.</p>"
    )
    parts.append('<div class="legend">')
    for name, entry in recovery.items():
        parts.append(
            f'<span><i class="swatch" style="background:{pal.CLASS_COLOURS.get(name, pal.INK_MUTED)}">'
            f"</i>{_esc(name.replace('_', ' ').title())} &mdash; "
            f"{_esc(CLASS_DESCRIPTIONS.get(name, ''))}</span>"
        )
    parts.append("</div>")
    parts.append(f'<div class="card">{bar_chart(recovery_rows, unit=" ha")}</div>')

    recovery_records = result.regrowth.records()
    parts.append(
        _details_table(
            "Table view: recovery metrics per patch",
            [
                "Patch", "Class", "Area (ha)", "Cleared", "NBR recovery",
                "NDVI recovery", "VH recovery", "NBR half-life (yr)", "Years to 80%",
            ],
            [
                [
                    r["patch_id"], r["classification"], f"{r['area_ha']:.2f}",
                    r["event_date"] or "-",
                    _fmt(r.get("nbr_recovery_fraction"), "{:.0%}"),
                    _fmt(r.get("ndvi_recovery_fraction"), "{:.0%}"),
                    _fmt(r.get("vh_recovery_fraction"), "{:.0%}"),
                    _fmt(r.get("nbr_half_life_years"), "{:.1f}"),
                    _fmt(r.get("nbr_years_to_80pct"), "{:.1f}", "not on this trajectory"),
                ]
                for r in sorted(recovery_records, key=lambda r: -r["area_ha"])
            ],
            numeric_from=2,
        )
    )
    if "recovery" in figures:
        parts.append(
            _embed_image(
                Path(figures["recovery"]),
                "Recovery trajectories per patch. All three tracks are indexed to the same "
                "0-1 scale (0 = post-clearing trough, 1 = pre-clearing baseline) so they "
                "share one axis; faint lines are the quarterly observations, solid lines "
                "the fitted saturating curve.",
            )
        )

    early = _early_recovery_rows(recovery_records)
    if early:
        parts.append("<h3>Where the three tracks disagree</h3>")
        parts.append(
            "<p>Observed recovery fraction at fixed ages, averaged across patches. "
            "NDVI sits consistently above NBR: greenness returns faster than canopy "
            "condition does, because ground cover colonises the site long before woody "
            "vegetation does. VH sits above both, for a different reason - C-band "
            "backscatter responds steeply to the first increments of woody structure "
            "and then saturates, which makes it a sensitive early-regrowth indicator "
            "and a weak one for separating mature regrowth. No single track is the "
            "answer; the disagreement between them is the information.</p>"
        )
        parts.append(f'<div class="card">{bar_chart(early, max_value=1.0)}</div>')
        parts.append(
            _details_table(
                "Table view: mean recovery fraction by age",
                ["Track and age", "Recovery fraction"],
                [[r["label"], f"{r['value']:.0%}"] for r in early],
            )
        )

    # --- validation -------------------------------------------------------
    if validation:
        parts.append("<h2>Validation</h2>")
        parts.append(
            "<p>Assessed against the reference truth that ships with the synthetic "
            "datacube. A single overall accuracy would be meaningless here - change is "
            "rare, so a map detecting nothing would score above 99%. What follows is "
            "recall by patch size, precision against the specific things that mimic "
            "clearing, and total area bias.</p>"
        )
        parts.append(
            _table(
                ["Detector", "Precision", "Recall", "F1", "IoU"],
                [
                    [
                        accuracy.label,
                        f"{accuracy.precision:.1%}",
                        f"{accuracy.recall:.1%}",
                        f"{accuracy.f1:.3f}",
                        f"{accuracy.iou:.3f}",
                    ]
                    for accuracy in validation.pixel.values()
                ],
            )
        )
        parts.append(
            f'<div class="note">Total area: {validation.area["detected_ha"]:.1f} ha detected '
            f'against {validation.area["reference_ha"]:.1f} ha reference, a bias of '
            f'{validation.area["bias_percent"]:+.1f}%.</div>'
        )

        size_rows = [
            {
                "label": name,
                "value": entry["recall"],
                "value_label": f"{entry['recall']:.0%}  ({entry['detected']}/{entry['reference_patches']})",
                "colour": "var(--series-1)",
                "tip": f"{name}: {entry['detected']} of {entry['reference_patches']} patches found",
            }
            for name, entry in validation.size_class_recall.items()
        ]
        if size_rows:
            parts.append("<h3>Recall by patch size</h3>")
            parts.append(
                "<p>Where the map stops working. The smallest class sits below the "
                f"{config.detection.min_mapping_unit_ha} ha minimum mapping unit and is "
                "removed by the sieve by design, not missed by the detector.</p>"
            )
            parts.append(f'<div class="card">{bar_chart(size_rows, max_value=1.0)}</div>')
            parts.append(
                _details_table(
                    "Table view: recall by size class",
                    ["Size class", "Reference patches", "Detected", "Recall", "Mean IoU"],
                    [
                        [n, e["reference_patches"], e["detected"], f"{e['recall']:.0%}", e["mean_iou"]]
                        for n, e in validation.size_class_recall.items()
                    ],
                )
            )

        if validation.decoys:
            parts.append("<h3>What the look-alikes did</h3>")
            parts.append(
                "<p>This is the argument for fusing two sensors, in one table. A burn "
                "scar and a fallowed paddock both read as clearing to Sentinel-2 alone; "
                "neither survives the requirement that Sentinel-1 agree.</p>"
            )
            parts.append(
                _table(
                    ["Look-alike class", "Extent (ha)", "Called confirmed", "Called S2-only", "Called S1-only"],
                    [
                        [
                            name,
                            f"{entry['reference_ha']:.1f}",
                            f"{entry['confirmed_ha']:.1f} ha ({entry['confirmed_fraction']:.1%})",
                            f"{entry['s2_only_ha']:.1f} ha ({entry['s2_only_fraction']:.1%})",
                            f"{entry['s1_only_ha']:.1f} ha ({entry['s1_only_fraction']:.1%})",
                        ]
                        for name, entry in validation.decoys.items()
                    ],
                )
            )

        latency_rows = [
            {
                "label": f"Patch {entry['patch_id']}",
                "a": entry["s2_latency_days"],
                "b": entry["s1_latency_days"],
            }
            for entry in (validation.latency or {}).get("per_patch", [])
            if entry.get("s2_latency_days") is not None
            and entry.get("s1_latency_days") is not None
        ]
        if latency_rows:
            parts.append("<h3>How soon could each sensor have said so?</h3>")
            parts.append(
                "<p>Days between the clearing event and the first acquisition on which "
                "that sensor could have raised the alert. Sentinel-2 has to wait for a "
                "view of the ground; over inland NSW in summer, under cloud and bushfire "
                "smoke, that wait is what sets the response time of the whole program.</p>"
            )
            parts.append(
                '<div class="legend">'
                '<span><i class="swatch" style="background:var(--series-3)"></i>Sentinel-1 (radar)</span>'
                '<span><i class="swatch" style="background:var(--series-1)"></i>Sentinel-2 (optical)</span>'
                "</div>"
            )
            parts.append(f'<div class="card">{dumbbell_chart(latency_rows)}</div>')
            latency = validation.latency
            parts.append(
                f'<div class="note">Median latency: Sentinel-1 '
                f'{latency["s1_median_days"]:.0f} days (worst {latency["s1_max_days"]:.0f}), '
                f'Sentinel-2 {latency["s2_median_days"]:.0f} days '
                f'(worst {latency["s2_max_days"]:.0f}).</div>'
            )
            parts.append(
                _details_table(
                    "Table view: detection latency per patch",
                    ["Patch", "Event", "S1 alert", "S1 days", "S2 alert", "S2 days"],
                    [
                        [
                            e["patch_id"], e["event_date"], e["s1_detection_date"],
                            e["s1_latency_days"], e["s2_detection_date"], e["s2_latency_days"],
                        ]
                        for e in latency["per_patch"]
                    ],
                )
            )

    # --- method -----------------------------------------------------------
    parts.append("<h2>Method and settings</h2>")
    parts.append(
        _details_table(
            "Full configuration",
            ["Setting", "Value"],
            [[k, json.dumps(v) if isinstance(v, (dict, list)) else v]
             for k, v in _flatten(config.to_dict())],
        )
    )
    parts.append(
        _details_table(
            "Stage runtimes",
            ["Stage", "Seconds"],
            [[k, f"{v:.2f}"] for k, v in result.timings_seconds.items()],
        )
    )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "".join(parts)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_esc(title)}</title><style>{_CSS}</style></head>
<body class="viz-root"><div id="tt" role="status"></div><div class="wrap">
<button class="toggle" type="button">Toggle theme</button>
{body}
<footer>Generated {_esc(generated)} by vegmon. Sentinel-1 and Sentinel-2 data are
free and open under the Copernicus programme. Thresholds in this run are defaults
and should be calibrated against local reference sites before operational use.</footer>
</div><script>{_JS}</script></body></html>"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


def _fmt(value, spec: str, fallback: str = "-") -> str:
    if value is None:
        return fallback
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return fallback


def _early_recovery_rows(records: Sequence[dict]) -> List[dict]:
    import numpy as np

    rows: List[dict] = []
    slots = {"ndvi": 1, "nbr": 2, "vh": 3}
    for age in ("1yr", "2yr", "5yr"):
        for track, slot in slots.items():
            values = [
                r.get(f"{track}_recovery_at_{age}")
                for r in records
                if r.get(f"{track}_recovery_at_{age}") is not None
            ]
            if not values:
                continue
            mean = float(np.mean(values))
            rows.append(
                {
                    "label": f"{track.upper()} at {age.replace('yr', ' year')}",
                    "value": mean,
                    "value_label": f"{mean:.0%}",
                    "colour": f"var(--series-{slot})",
                    "tip": f"{track.upper()} at {age}: {mean:.1%} of baseline recovered, mean of {len(values)} patches",
                }
            )
    return rows


def _flatten(data: dict, prefix: str = "") -> List[tuple]:
    out: List[tuple] = []
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.extend(_flatten(value, f"{name}."))
        else:
            out.append((name, value))
    return out


__all__ = ["bar_chart", "build_report", "dumbbell_chart"]
