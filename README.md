# vegmon

Land clearing detection from **Sentinel-1 and Sentinel-2 together**, with
post-clearing **regrowth monitoring** on top of the detections.

Built as a self-contained demo for woody vegetation monitoring in inland NSW:
detect the clearing event, confirm it across two sensors, date it, then follow
what grows back. It runs against live public STAC catalogues, and — because
network access is not always a given — against a synthetic Sentinel-1/2
datacube with known ground truth that ships with the package.

```bash
pip install -e '.[all]'
vegmon demo
```

That single command generates a datacube, runs every stage, and writes a map
figure, recovery trajectories, a GeoPackage of clearing patches, a CSV of
recovery metrics and a self-contained HTML report. No account, no token, no
network.

---

## Why two sensors

Optical change detection on its own cannot tell canopy **removal** from canopy
**scorching**, or from a paddock taken out of production. Both collapse NDVI
and NBR exactly the way clearing does. Sentinel-1 can tell them apart, because
removing woody structure collapses the volume scattering that dominates
cross-polarised radar return, and a burn scar or a bare paddock does not.

The demo scene contains those look-alikes deliberately. Measured against its
ground truth:

| Detector | Precision | Recall |
|---|---|---|
| Sentinel-2 alone | 78.9% | 96.7% |
| Sentinel-1 alone | 99.5% | 96.1% |
| **Both, where they agree** | **99.5%** | **98.5%** |

Sentinel-2 alone loses a fifth of its precision to a 25 ha fire scar it
scores as clearing. Requiring the two sensors to agree removes it entirely,
and the fire scar is not discarded — it is reported in an `s2_only` tier for
review, which is a more useful answer than either "clearing" or silence.

The second reason is timing. On the same scene, radar raised the alert a
median of **13 days** before the first Sentinel-2 acquisition that could see
enough of the patch to act on — worst case 5 days versus 51.

## What comes out

`vegmon demo` writes to `outputs/`. A committed example run is in
[`docs/example-outputs/`](docs/example-outputs/) — open `report.html` in a
browser to see what the pipeline produces without running anything.

| File | What it is |
|---|---|
| `report.html` | Self-contained report: headline numbers, charts, tables, both figures. Opens anywhere, no network. |
| `change_maps.png` | Before/after false colour, both change surfaces, the fused result, the reference truth. |
| `recovery.png` | Per-patch recovery trajectories, three tracks, fitted curves. |
| `clearing.gpkg` | Clearing patches with area, tier, change magnitudes and shape metrics. Opens in QGIS or ArcGIS. |
| `regrowth.csv` | Every recovery metric per patch. |
| `summary.json` | The whole run, machine-readable. |

## How it works

### 1. Composite

Seasonal composites either side of the suspected event, built from
cloud-masked Sentinel-2 (Sen2Cor SCL, with cloud and shadow dilated outwards,
because SCL consistently under-calls both) and multi-looked Sentinel-1.

The optical reducer is a **medoid**, not a band-wise median: it picks, per
pixel, the single real observation closest to the per-band median vector, so
every composite pixel is an internally consistent spectrum rather than one
assembled from different dates.

**The pre- and post-period must cover the same months.** Comparing a
March–August composite against a March–November one is the most reliable way
to manufacture false positives in cropping country: the baseline catches the
spring green peak and the later window does not, so every paddock in the
district reads as cleared. `Periods.check_phenology()` warns when the two
windows drift more than three weeks apart in the seasonal cycle, and the
optical detector calls it automatically.

### 2. Normalise

Two composites of the same unchanged ground are never identical — sun angle,
aerosols and BRDF move optical indices by a few hundredths, and soil moisture
moves Sentinel-1 by more than a decibel. The median difference over the stable
population is subtracted before thresholding. This is relative radiometric
normalisation against pseudo-invariant features, done cheaply, and it is the
difference between thresholds that transfer between date pairs and thresholds
that have to be re-tuned for every one.

### 3. Detect

**Sentinel-2** gates, in order: enough clear observations either side → was
woody (temporal median NDVI and NBR above a floor) → *persistently* woody (the
10th percentile of pre-event NDVI also above a floor) → **steady through the
year** (within-year NDVI swing below a ceiling) → lost enough (dNBR **and**
dNDVI past threshold).

The gates read *temporal percentiles of the index*, never the index of a band
composite. Any band-wise composite reduces each band independently, so over a
pixel that swings through the year the composite NDVI can sit far above
anything the pixel actually reached — on a real NSW cropping scene, 0.42
against a true temporal median of 0.20.

That third gate does most of the work in cropping country. A winter-crop
paddock reaches NDVI 0.7 in September and drops below 0.25 after harvest, so
its seasonal *median* can pass a woody threshold — its low percentile cannot.
Woody vegetation stays green all year, so it does.

The fourth gate is what persistence alone cannot do. Irrigated cotton holds
NDVI above 0.6 for most of the year and clears any persistence test
comfortably, and a paddock going from cotton to fallow produces a dNBR
indistinguishable from clearing. What separates woody vegetation from any crop
is the *shape* of the year: woody vegetation here is largely evergreen and
moves little, while a crop swings hard between planting and harvest.

Requiring NBR and NDVI to move *together* is also deliberate. NBR alone fires
on wet soil; NDVI alone fires on every harvested paddock.

**Sentinel-1** gates: enough passes either side → had structure (pre-event VH
above a floor) → VH dropped at least 2 dB. Speckle is handled by a temporal
median over at least three passes plus a spatial boxcar multi-look; soil
moisture is removed as a scene-wide offset. Per-orbit drops are computed and
reported separately, because ascending and descending passes view the canopy
from opposite sides and can differ by more than the signal being looked for.

### 4. Fuse

Morphological opening (shaves the one-pixel filaments thresholding leaves
along paddock edges), hole filling (retained scattered trees read as holes but
are part of the same event), then a minimum-mapping-unit sieve — 0.5 ha by
default, because NSW reporting works in hectares and sub-0.5 ha detections at
10 m are mostly edge noise.

Surviving patches are tiered by which sensors saw them:

| Tier | Meaning |
|---|---|
| `confirmed` | Both sensors agree. The canopy went and the structure went with it. Act on this. |
| `s2_only` | Spectral loss, no structural loss. Fire, drought dieback, harvest, fallow, or unmasked shadow. Review. |
| `s1_only` | Structural loss, no spectral loss. Thinning under retained canopy, inundation, or an optical coverage gap. Review. |

Tiering is done per *patch*, not per pixel: agreement between a 10 m optical
detection and a multi-looked radar one is never complete along edges, so the
claim being made is that the two overlap across a decent fraction of the
patch.

### 5. Date

Each patch is dated from its own time series — the first acquisition where the
patch mean crosses halfway between its pre- and post-event levels and stays
there. An optical acquisition only counts if most of the patch is actually
visible; without that, a scene 95% under cloud still yields a patch mean from a
handful of clear pixels and credits Sentinel-2 with alerts nobody could act on.

Dating patches individually also matters for step 6: a recovery curve fitted
from an assumed midpoint mixes pre-event observations into the recovery limb
and flatters it.

### 6. Follow the regrowth

Per patch, three tracks, quarterly medians, saturating-exponential fits:

- **NDVI** — greenness. Recovers fastest and **overstates woody recovery**.
  Grass and forbs colonise a cleared paddock within a season.
- **NBR** — greenness plus canopy moisture and structure. The classification
  is based on this one.
- **Sentinel-1 VH** — structure. C-band saturates at modest biomass, so it
  responds steeply to the first regrowth and weakly to the difference between
  ten- and twenty-year regrowth. Sensitive early, blunt late.

Reporting all three rather than picking one is the point. **A patch whose NDVI
has recovered but whose NBR and VH have not is grassland, not regrowth.**

Trajectories are expressed as **anomalies against undisturbed woody ground in
the same scene**, and the within-year cycle is subtracted before fitting.
Both matter on real data: raw NBR over inland NSW swings further between a wet
year and a dry one than a clearing event moves it, so an unnormalised
trajectory reports every drought as a second clearing.

Metrics per track: recovery fraction, R80P, fitted time constant and
half-life, extrapolated years to 80% of baseline, observed recovery fraction
at 1/2/5 years, a Theil–Sen recent trend, and the median within-year swing.
Patches are classified `recovered` / `recovering` / `stalled` / `recleared` /
`cultivated`.

That last class carries a lot of weight. Cropping and returning woody cover
reach similar annual-average greenness, so a recovery fraction alone cannot
tell them apart — only the size of the within-year swing can. A cleared site
that went into production is not regrowing, and on real data it is the most
common outcome.

Re-clearing is a sustained drop below a *trailing median* of the preceding
seasons — not below a running maximum, which drifts upwards on a noisy series
so that every drought season afterwards reads as a second event.

### 7. Validate

Reporting a single overall accuracy for a change map is close to meaningless:
change is rare, so a map that detects nothing scores over 99% correct. What
the harness reports instead is recall **by patch size**, precision against the
**specific things that mimic clearing**, total **area bias**, and **detection
latency** per sensor.

## Using it on real data

Two routes. The first needs a reachable STAC API:

```bash
vegmon catalogues                       # what's available
vegmon config-template my-aoi.json      # starter config
$EDITOR my-aoi.json                     # set the bbox, CRS and periods
vegmon run --config my-aoi.json --orbit ascending
```

The second needs nothing but the AWS data bucket, which is often the only one
reachable from a government or corporate network:

```bash
vegmon aws --tile 55JGG --lon 150.1325 --lat -30.0634 \
  --size 512 --resolution 20 \
  --pre 2022-03-01 2022-09-30 --post 2023-03-01 2023-09-30 \
  --series 2019-01-01 2025-12-31 --epochs
```

`vegmon aws` enumerates Sentinel-2 scenes by listing S3 prefixes and reads
windowed COGs directly, so it works when `earth-search`, Planetary Computer
and CDSE are all blocked. `--epochs` additionally scans every consecutive year
pair in the record and writes the annual monitoring table.

**A real NSW run, and everything that broke doing it, is written up in
[`docs/nsw-real-data.md`](docs/nsw-real-data.md)** — worth reading before
pointing this at your own AOI. Outputs are in
[`docs/example-outputs-nsw/`](docs/example-outputs-nsw/). The short version:
the detection machinery transferred unchanged, but absolute thresholds do not
survive an Australian drought, band-wise median compositing manufactures
spectra that walk cropping through a woody-cover gate, and persistent
greenness is not a test for woody vegetation — irrigated cotton passes it and
produced 80 ha of confidently-reported "clearing" that was paddock rotation.

| Catalogue | Sentinel-2 | Sentinel-1 | Notes |
|---|---|---|---|
| `earth-search` | L2A | GRD | Anonymous, no account. AWS open data. Start here. |
| `planetary-computer` | L2A | **RTC** | Terrain-corrected gamma-nought — prefer it in hill country, where slope effects otherwise swamp a 2 dB threshold. |
| `cdse` | L2A | GRD/SLC | Authoritative ESA source. Needs a free Copernicus account. |

Pin a single orbit direction with `--orbit`. It matters more than it sounds.

**On restricted networks**, `vegmon run` fails with a message naming the
endpoint and the likely cause — an egress policy rather than an outage. The
hosts to allow-list are in the table above. `vegmon demo` needs no network at
all, which is why it exists.

## Calibrate before you trust it

Every threshold in `DetectionConfig` and `RegrowthConfig` is a documented
starting point, not a finding. The defaults are set for the ~20% foliage-cover
definition of woody vegetation used in NSW reporting rather than for closed
forest — a closed-canopy threshold quietly excludes the grassy box woodland
and open cypress country where most clearing actually happens.

Re-derive them against local reference sites (SLATS woody change layers,
aerial photo interpretation, field sites) before operational use. The
validation harness in `vegmon.validation` takes any reference mask, not just
the synthetic truth, so the same tables can be produced against real
reference data.

## The synthetic datacube

Not noise with a rectangle drawn on it. Reflectance comes from linear spectral
unmixing of green canopy, **herbaceous** green, non-photosynthetic vegetation,
bare soil, char and water endmembers — herbaceous green carries its own
spectrum, higher in SWIR and lower in NIR than woody canopy, which is what
makes grass restore NDVI almost fully while leaving NBR short. Sentinel-1
backscatter follows a saturating function of structural cover with
gamma-distributed speckle at the ~4.4 looks of a GRD IW product, plus a
soil-moisture term weighted towards low-cover pixels. Cloud is spatially
correlated with an offset shadow, encoded in a real SCL band, and heavier in
summer.

The scene contains eight clearing events spanning 0.3 to 40 ha with different
recovery trajectories (including one stalled and one re-cleared), plus three
look-alikes that must **not** be called clearing: a fire scar, cropping
paddocks, and a paddock cropped in the baseline year then fallowed. Patch
areas scale with grid size, so a small test raster holds the same *proportion*
of disturbance as the full one.

```python
from vegmon.synthetic import generate_scene, scene_truth
from vegmon.config import demo_config
from vegmon.pipeline import run_pipeline

scene = generate_scene(demo_config(), size=320, seed=42)
result = run_pipeline(scene, demo_config())
print(result.validation.headline())
```

## Layout

```
src/vegmon/
  config.py      AOI, periods, thresholds — every default documented and justified
  grid.py        raster geometry and hectare conversions
  indices.py     NDVI, NBR, NDMI, tasselled cap, dB conversions, RVI
  masking.py     SCL masking with cloud/shadow dilation, compositing reducers
  series.py      optical and radar time-series containers
  synthetic.py   the synthetic datacube and its ground truth
  stac.py        live loaders (earth-search, Planetary Computer, CDSE)
  s3direct.py    Sentinel-2 straight from the AWS bucket, no STAC API needed
  persistence.py temporal persistence as confirming evidence without radar
  detect.py      the two detectors, normalisation, event dating
  fuse.py        cleanup, sieve, tiering, polygonisation
  regrowth.py    seasonal binning, curve fitting, classification
  validation.py  accuracy assessment against any reference
  viz.py         map panels and recovery figures
  palette.py     the plotting palette and why it is what it is
  report.py      the self-contained HTML report
  pipeline.py    orchestration
  cli.py         the command line
```

## Tests

```bash
pytest -m "not network"
```

The suite pins the properties the detectors depend on, not just array shapes —
that clearing drops both sensors past threshold, that the fire scar drops the
optical one and *not* the radar one, that cropping is not persistently green,
that fusion beats either sensor alone on precision, and that radar alerts
sooner than optical.

## Licence and data

MIT. Sentinel-1 and Sentinel-2 data are free and open under the EU Copernicus
programme.
