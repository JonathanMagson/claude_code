# Example outputs, real NSW Sentinel-2

Boggabri / Maules Creek woodland margin, MGRS tile 55JGG, centred
150.1325 E, 30.0634 S. 512 x 512 px at 20 m = 10,486 ha in EPSG:32755.
367 usable Sentinel-2 acquisitions from 2019 to 2025, read straight from the
AWS `sentinel-cogs` bucket with no STAC API. Detection epoch 2022 vs 2023.

Sentinel-1 is included: 196 acquisitions on relative orbit 45, calibrated to
gamma0 in-process from the raw GRD archive on AWS, since no analysis-ready
Sentinel-1 product for Australia is on a reachable host. See
[`../nsw-real-data.md`](../nsw-real-data.md).

Reproduce with:

```bash
vegmon aws --tile 55JGG --lon 150.1325 --lat -30.0634 \
  --name "Boggabri / Maules Creek woodland margin (55JGG)" \
  --size 512 --resolution 20 \
  --pre 2022-03-01 2022-09-30 --post 2023-03-01 2023-09-30 \
  --series 2019-01-01 2025-12-31 \
  --max-amplitude 0.35 --adaptive-woody 0 --adaptive-mad 3 \
  --adaptive-vh-mad 5 --s1 --s1-orbit 45 --epochs \
  --outputs docs/example-outputs-nsw
```

| File | What it is |
|---|---|
| `report.html` | The full report for the 2022-2023 epoch. Self-contained. |
| `epochs.csv` | The annual monitoring table: every consecutive year pair, 2019 to 2025. |
| `change_maps.png` | Before/after false colour, the change surface, the persistence layer, the fused result. |
| `recovery-persistence-variant.png` | Recovery trajectories from the **persistence-confirmed** variant of this run (drop `--s1`). The fused run confirms nothing, so it produces no trajectories; these are what showed the 2022-2023 signal recovering, and are kept because that is half the argument. |
| `clearing.gpkg` | Patch polygons in EPSG:32755, with area, tier and change magnitudes. |
| `regrowth.csv` | Recovery metrics per patch. Empty for the fused run, which confirms no patches to follow. |
| `summary.json` | The whole run, including the thresholds actually applied. |

## What it found

**Zero confirmed clearing**, in this epoch and in every other epoch of the
2019-2025 record. Sentinel-2 found 44.2 ha of spectral loss and Sentinel-1
found 27.6 ha of structural loss, and they are not the same ground.

That is the fusion working, not failing. Across all six epochs the median VH
drop *inside* the optical detections is 0.1-0.4 dB (1.34 dB in 2022-2023),
against an adaptive threshold of 1.5-3 dB: where the canopy browned, the woody
structure stayed standing. Combined with the regrowth trajectories - most of
the 2022-2023 signal had recovered or was recovering within two and a half
years - the reading is drought canopy thinning at the 2022 La Nina to 2023
El Nino break, not clearing.

A null result cannot demonstrate sensitivity. That the detector finds real
clearing has only been shown against synthetic truth; validation against SLATS
woody change layers is the necessary next step.
