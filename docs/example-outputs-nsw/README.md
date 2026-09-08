# Example outputs, real NSW Sentinel-2

Boggabri / Maules Creek woodland margin, MGRS tile 55JGG, centred
150.1325 E, 30.0634 S. 512 x 512 px at 20 m = 10,486 ha in EPSG:32755.
367 usable Sentinel-2 acquisitions from 2019 to 2025, read straight from the
AWS `sentinel-cogs` bucket with no STAC API. Detection epoch 2022 vs 2023.

No Sentinel-1: see [`../nsw-real-data.md`](../nsw-real-data.md) for why, and
for what temporal persistence substitutes in its place.

Reproduce with:

```bash
vegmon aws --tile 55JGG --lon 150.1325 --lat -30.0634 \
  --name "Boggabri / Maules Creek woodland margin (55JGG)" \
  --size 512 --resolution 20 \
  --pre 2022-03-01 2022-09-30 --post 2023-03-01 2023-09-30 \
  --series 2019-01-01 2025-12-31 \
  --max-amplitude 0.35 --adaptive-woody 0 --adaptive-mad 3 --epochs \
  --outputs docs/example-outputs-nsw
```

| File | What it is |
|---|---|
| `report.html` | The full report for the 2022-2023 epoch. Self-contained. |
| `epochs.csv` | The annual monitoring table: every consecutive year pair, 2019 to 2025. |
| `change_maps.png` | Before/after false colour, the change surface, the persistence layer, the fused result. |
| `recovery.png` | Recovery trajectories for each confirmed patch. |
| `clearing.gpkg` | Patch polygons in EPSG:32755, with area, tier and change magnitudes. |
| `regrowth.csv` | Recovery metrics per patch, including the within-year swing that separates regrowth from cultivation. |
| `summary.json` | The whole run, including the thresholds actually applied. |

## What it found

24 confirmed patches totalling 29.9 ha, plus 13.8 ha in the review tier.
Patches are small (median 0.9 ha) and elongated (shape index 1.5-2.2),
scattered through the woodland rather than squared off against paddock
boundaries.

The regrowth stage is what makes this interpretable: 9.0 ha had already
**recovered** and 16.8 ha was **recovering** within two and a half years, on
low within-year swings that confirm the returning cover is woody. Mechanical
clearing does not come back like that. This epoch straddles the break from the
2022 La Nina floods to the 2023 El Nino, and drought canopy thinning is the
better explanation than clearing.

The annual table puts that in context - and shows the 2019-2020 epoch placing
59.6 ha in the review tier against 5.3 ha confirmed, which is what a Black
Summer burn scar that subsequently recovered looks like to a detector with no
radar.
