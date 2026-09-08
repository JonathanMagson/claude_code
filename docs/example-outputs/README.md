# Example outputs

Produced by `vegmon demo` on the bundled synthetic Sentinel-1/2 datacube
(320 x 320 px at 10 m = 1024 ha, 512 Sentinel-2 and 427 Sentinel-1
acquisitions, 2019-2025). Regenerate with:

```bash
vegmon demo --outputs docs/example-outputs
```

| File | What it is |
|---|---|
| `report.html` | The full report. Self-contained — open it directly in a browser. |
| `change_maps.png` | Before/after false colour, both change surfaces, the fused result, the reference truth. |
| `recovery.png` | Recovery trajectories for each confirmed patch, three tracks each. |
| `clearing.gpkg` | Clearing patches as polygons (EPSG:32755), with area, tier and change magnitudes. |
| `regrowth.csv` | Every recovery metric for every patch. |
| `summary.json` | The whole run, machine-readable, including the validation report. |

Headline result on this scene: 84.0 ha confirmed across 9 patches against
84.9 ha of reference clearing (-1.1% area bias), 99.5% precision and 98.5%
recall at the pixel level, no look-alike class reaching the confirmed tier,
and Sentinel-1 raising the alert a median 13 days ahead of Sentinel-2.
