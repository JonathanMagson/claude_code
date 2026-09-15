# Monitoring workflow

**Status: draft.** Written from one full pass over real NSW imagery. The
sequence and the checks are sound; the numbers in it are starting points, and
step 4 is the one nobody can skip.

This is the operational procedure for running a clearing-detection and
regrowth cycle over an area. It exists because the tooling on its own is not
enough: every serious error made while building this was a *process* error,
not a code error, and each one is now a step below.

---

## What this produces, and what it does not

**Produces** — per monitored area, per epoch: a GeoPackage of woody-change
patches tiered by how many independent lines of evidence support them; an
annual change table; recovery trajectories for confirmed patches classified
`recovered` / `recovering` / `stalled` / `recleared` / `cultivated`; and a
self-contained HTML report.

**Does not produce** — a legally defensible determination. Every patch is a
*candidate* requiring assessment against imagery, property boundaries and
approvals. The tiers say how much the data supports a claim; they do not say
whether clearing was lawful, and they cannot distinguish clearing from
several things that look like it (see step 6).

---

## Stage A — Set up (once)

```bash
pip install -e '.[all]'
vegmon demo                 # end-to-end on synthetic data, no network
pytest -m "not network"
```

`vegmon demo` must report ~99% precision and ~98% recall against the bundled
truth. If it does not, stop — the installation is wrong, and nothing
downstream is interpretable.

**Network.** The pipeline needs one of:

| Route | Hosts required | Command |
|---|---|---|
| STAC (preferred) | `earth-search.aws.element84.com`, `planetarycomputer.microsoft.com` | `vegmon run` |
| AWS buckets only | `sentinel-cogs.s3.us-west-2.amazonaws.com`, `sentinel-s1-l1c.s3.eu-central-1.amazonaws.com` | `vegmon aws` |

Ask IT for the second set if the first is blocked — it is a smaller ask, and
`vegmon aws --s1` gets both sensors from it. Confirm before planning work:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://sentinel-cogs.s3.us-west-2.amazonaws.com/
```

---

## Stage B — Per monitored area (once per area)

### 1. Choose the area, and choose it on evidence

Clearing happens on the **woodland–cropping margin**. An area that is already
fully cleared has nothing to detect; an irrigated floodplain will generate
false positives for years.

Do not pick coordinates by eye. Survey a whole tile and take the window with
the most balanced mix — the procedure in
[`nsw-real-data.md`](nsw-real-data.md) found two of three candidate tiles were
essentially fully cleared already (tile-wide woody fraction 0.4% and 5.5%).

Record the MGRS tile, centre, size and resolution. 512 px at 20 m
(10,486 ha) is a reasonable unit of work: about 10 minutes to load seven years
of Sentinel-2, and small enough to review by hand.

### 2. **Characterise the area before detecting anything in it**

```bash
vegmon survey --tile 55JGG --lon 150.1325 --lat -30.0634 \
  --years 2019 2025 --outputs sites/boggabri/
```

This is the step that would have prevented every wrong answer this pipeline
has produced. It reports, per year: clear views, median and 90th-percentile
NDVI, persistence, within-year swing, woody fraction, cropping fraction and
cloud — then flags which year pairs can fairly be compared.

**Stop and move the area if:**

- woody fraction is **below ~5%** — there is nothing here to lose;
- cropping fraction is **above ~40%** — a rotation will dominate everything;
- no epoch pair is marked `ok`.

**Record the suggested epoch and thresholds** in the site folder. Both are
inputs to every later run and both need to be revisited when the record grows.

Worked example — the survey output that would have stopped the first bad run:

```
Namoi irrigated cotton
year   clear  NDVI p50  persist   swing   woody   crop
2022      46     0.212    0.180   0.593    3.0%  55.5%
  AVOID 2022->2023  -  only 3.0% of the baseline year is woody
  ! 49.0% of this AOI swings hard through the year - dominated by cropping
```

### 3. Choose the epoch

The baseline and detection windows must **cover the same months**. Comparing
March–August against March–November is the most reliable way to manufacture
false positives in cropping country: the baseline catches the spring green
peak and the later window does not, so every paddock in the district reads as
cleared. `Periods.check_phenology()` warns automatically; do not proceed past
the warning.

Prefer a window that **ends before the clearing season** so the event falls in
the gap. If detected event dates cluster inside the baseline window, the
baseline is contaminated — shorten it and re-run. That happened on the first
real run and the fix moved the pre-window from March–November to March–September.

Avoid year pairs that straddle a drought break. The survey flags them.

### 4. Calibrate against local reference sites — **do not skip**

Every threshold shipped is a documented starting point, not a finding. Before
an area's output is used for anything:

1. Assemble reference patches — SLATS woody change layers, aerial photo
   interpretation, known approved clearing, and at least as many **known
   non-events**: burn scars, harvested paddocks, drought-affected woodland.
2. Run the epoch over them.
3. Use `vegmon.validation` — it takes any reference mask, not just synthetic
   truth — to produce recall by patch size, precision against each look-alike
   class, and area bias.
4. Adjust and re-run. Record the final values, the reference set and the date
   in the site folder.

Target the **minimum detectable patch size**, not a single accuracy number.
Change is rare, so a map detecting nothing scores above 99% correct.

Re-calibrate when the area changes character, when a new sensor baseline
lands, or annually — whichever is sooner.

---

## Stage C — Per cycle

### 5. Run

```bash
vegmon aws --tile 55JGG --lon 150.1325 --lat -30.0634 \
  --name "Boggabri margin" --size 512 --resolution 20 \
  --pre  2024-03-01 2024-09-30 \
  --post 2025-03-01 2025-09-30 \
  --series 2019-01-01 2025-12-31 \
  --max-amplitude 0.35 --adaptive-mad 3 --adaptive-vh-mad 5 \
  --s1 --s1-orbit 45 --epochs \
  --cache sites/boggabri/s2.npz --s1-cache sites/boggabri/s1.npz \
  --outputs sites/boggabri/2025Q3/
```

Notes that matter:

- **`--s1-orbit` is not optional in practice.** Ascending and descending
  passes view the canopy from opposite sides and differ over the same intact
  woodland by more than the drop being looked for. Pin one orbit and keep it
  pinned for the life of the area.
- **`--epochs`** writes the annual table. Read it every cycle: a row only
  means something against its neighbours.
- **Caches** make re-runs minutes instead of tens of minutes. Delete them when
  the series end date moves.

### 6. Triage the output

Work the tiers. **Never send an untriaged patch list to anyone.**

| Tier | What it means | Action |
|---|---|---|
| `confirmed` | Both sensors agree: canopy gone *and* woody structure gone | Assess. This is the actionable tier. |
| `s2_only` | Spectral loss, no structural loss | Review. Fire, drought dieback, harvest, fallow, or unmasked cloud shadow. |
| `s1_only` | Structural loss, no spectral loss | Review. Thinning under retained canopy, inundation, or an optical coverage gap. |

For each `confirmed` patch, before it goes any further:

1. **Look at the before/after imagery.** Every time.
2. **Check the shape.** Clearing follows paddock boundaries — squared off,
   shape index near 1.3–1.7. Many small elongated patches scattered through
   woodland is dieback, not a dozer.
3. **Check the size distribution.** A few large patches reads as clearing;
   dozens of sub-hectare patches reads as a landscape-wide process.
4. **Check the event dates.** Real clearing clusters in time. Dates spread
   evenly across the window suggest a gradual process.
5. **Check the epoch table.** If every epoch shows similar area, the detector
   is measuring the landscape, not an event.

If a patch survives all five, it is a genuine candidate for assessment.

### 7. Follow the regrowth

For `confirmed` patches carried forward, the recovery classification is a
second, independent read on whether the event was what it looked like:

- `recovered` / `recovering` **within two or three years, on a low within-year
  swing** — probably was not mechanical clearing. Drought thinning recovers;
  a cleared paddock does not.
- `cultivated` — the site is in production. Cropping and returning woody cover
  reach the same annual-average greenness; only the within-year swing
  separates them, and this is the most consequential distinction the pipeline
  makes.
- `stalled` — cleared and being held open. The tier that usually matters most.
- `recleared` — a second event on the same ground.

Recovery needs several years of post-event record. For a current-cycle event
it will read `insufficient_data`; revisit it in later cycles rather than
concluding anything.

### 8. QA before anything leaves the team

Check, every cycle:

- [ ] `vegmon demo` still passes at ~99%/98% — the code has not regressed.
- [ ] Phenology alignment produced no warning.
- [ ] Both windows have **at least 12 clear views** over the area.
- [ ] Normalisation offsets in `summary.json` are small. A large optical
      offset (>0.1 dNBR) or radar offset (>3 dB) means the two epochs are not
      comparable — usually a drought break.
- [ ] Applied thresholds in `summary.json` match the calibrated values, and
      any adaptive value is in a plausible range. An adaptive VH threshold of
      8 dB means that epoch's radar was too noisy to use.
- [ ] Woody fraction is close to the surveyed value. A big move means the
      amplitude gate is reading a different landscape.
- [ ] Detected area is plausible against the epoch table and against previous
      cycles.
- [ ] Every `confirmed` patch has been eyeballed against imagery.

### 9. Record and hand off

Keep, per cycle, under `sites/<area>/<cycle>/`:

`summary.json` (the whole run including applied thresholds) ·
`clearing.gpkg` (patches, EPSG:32755, ready for QGIS/ArcGIS) ·
`epochs.csv` · `regrowth.csv` · `report.html` · the triage decisions and who
made them.

`summary.json` is the provenance record — it carries the AOI, the periods, the
normalisation offsets and the thresholds actually applied, which is what
makes a past cycle reproducible and defensible.

GeoPackage attributes worth knowing: `tier`, `area_ha`, `mean_dnbr`,
`mean_vh_drop_db`, `agreement_fraction` (how much of the patch both sensors
saw), `shape_index` (1.0 = circular, higher = elongated), `pre_ndvi` and
`pre_ndvi_persistent` (what the site looked like before).

---

## Cadence

| Cycle | What runs | Why |
|---|---|---|
| **Quarterly** | Rolling epoch over the last full season pair; triage | Keeps latency to a season. Radar alerts within days; the limit is the seasonal composite, not the sensor. |
| **Annual** | Full epoch scan over the whole record; re-calibrate | Epochs only mean something against their neighbours, and thresholds drift. |
| **On demand** | Single AOI around a report or complaint | Use the longest available baseline. |

---

## Known failure modes

Each of these was hit for real while building this.

| Symptom | Cause | Fix |
|---|---|---|
| Large confident detections over green, flat country | Irrigated cropping passing the woody gate | `--max-amplitude 0.35`. Irrigated cotton is persistently green; only the within-year swing separates it from woodland. |
| Every paddock in the district reads as cleared | Baseline and detection windows cover different months | Match the months. Heed the phenology warning. |
| Detections everywhere, or nowhere at all | Absolute thresholds carried from another district or year | `--adaptive-woody`, `--adaptive-mad`, `--adaptive-vh-mad`. NDVI at one NSW site swung 0.18→0.54→0.18→0.35 over four years. |
| Large single-sensor radar area, no agreement | Fixed decibel threshold | `--adaptive-vh-mad 5`. Real VH change has MAD 0.43 dB but a 99th percentile of 2.7 dB; a fixed 2 dB caught 4.6% of one AOI. |
| Many small elongated patches through woodland | Drought canopy thinning | Check regrowth. If it recovers in 2–3 years it was not clearing. |
| Every patch reads `recleared` | Trajectories not normalised, or seasonal cycle not removed | Already handled, but check the reference population is *woody* undisturbed ground, not the whole scene. |
| Event dates inside the baseline window | Contaminated baseline | Shorten the pre-window so the event falls in the gap. |
| Composite NDVI far above what the pixel ever reached | Band-wise median compositing | Already fixed (medoid + gating on temporal percentiles). Do not revert the reducer. |

---

## Open items before this leaves draft

1. **Validation against SLATS.** The detector's ability to find real clearing
   has only been demonstrated against synthetic truth. The one real fused run
   returned a null, and a null cannot demonstrate sensitivity. This is the
   single most important gap.
2. **Minimum detectable patch size on real imagery**, per district.
3. **Terrain correction** if the programme extends into the escarpment.
   Pinning one relative orbit makes ellipsoid gamma0 sound for change
   detection on the slopes and plains; it is not sound in steep country.
4. **Fire layer integration.** Without Sentinel-1 persistence cannot exclude
   fire; with it, a burn scar is still a `s2_only` detection needing review.
   Joining to NPWS/RFS fire history would close that automatically.
5. **Cycle automation** once cadence is agreed.
