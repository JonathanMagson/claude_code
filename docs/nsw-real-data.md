# Running on real NSW Sentinel-2

This is the log of taking the pipeline from synthetic data to real inland NSW,
including everything that broke. The short version: the detection machinery
transferred, the *thresholds* did not, and three methodological problems only
became visible on real data.

## Getting the data without a STAC API

The usual route to Sentinel-2 is a STAC search against `earth-search`,
Planetary Computer or CDSE. All three were blocked at the network egress
policy in the environment this was developed in — a `403` on `CONNECT`, which
is exactly what a government or corporate proxy does — while the underlying
AWS data bucket was reachable.

That turns out not to matter, because `sentinel-cogs` needs no index. Its keys
are deterministic:

```
sentinel-s2-l2a-cogs/{zone}/{band}/{square}/{year}/{month}/{scene}/B04.tif
```

so scenes can be enumerated with anonymous S3 `ListObjectsV2` calls and read
with windowed HTTP range requests straight out of the COGs. `vegmon.s3direct`
does that, and `vegmon aws` runs the whole pipeline on it:

```bash
vegmon aws --tile 55JGG --lon 150.1325 --lat -30.0634 \
  --size 512 --resolution 20 \
  --pre 2022-03-01 2022-09-30 --post 2023-03-01 2023-09-30 \
  --series 2019-01-01 2025-12-31 --epochs
```

Over a 512 × 512 window at 20 m, listing 1,021 acquisitions and reading the
367 that cleared the AOI cloud screen took about ten minutes on twelve
threads. Two things make it practical: the 20 m scene classification layer is
fetched first and too-cloudy dates are dropped before any reflectance band is
touched, and cloud is screened **over the AOI** rather than on the scene-level
metadata — a tile can be 70% cloudy and perfectly clear over the paddock in
question.

## No Sentinel-1

There is no free, analysis-ready Sentinel-1 for Australia reachable from this
environment. Planetary Computer's `sentinel-1-rtc`, CDSE and ASF are all
blocked; the public `sentinel-s1-rtc-indigo` bucket on AWS covers UTM zones
10–19 only, which is the continental United States. The raw
`sentinel-s1-l1c` GRD bucket is reachable, but turning GRD into calibrated,
terrain-corrected gamma-nought is a processing chain in its own right and not
something to do quickly and claim confidence in.

So the real runs have **no radar confirmation**, which removes the pipeline's
main defence. In its place `vegmon.persistence` supplies a weaker but
independent second opinion: a clearing event does not grow back, so the
spectral drop must still be there several seasons later. That rejects harvest,
drought dips and unmasked cloud shadow — using observations that played no
part in the original detection. It does **not** reject fire. Where Sentinel-1
is reachable, use it.

## Three things that only broke on real data

### 1. Absolute thresholds do not survive an Australian drought

The synthetic scene was tuned to my own assumptions about what NSW woodland
looks like. Real numbers, over a 6.4 km window in the 2019 baseline year:

| Site | NDVI (season median, p50) | Persistently green (temporal p10 > 0.35) |
|---|---|---|
| Moree plains | 0.18 | 9.3% |
| Croppa Creek | 0.27 | 1.6% |
| Pilliga East | 0.34 | 16.9% |

2019 was the peak of the drought. NBR over persistently-green pixels ran
0.11–0.35 depending on the site, against the 0.18 floor the synthetic had
suggested — so the same gate either passed almost everything or rejected
almost everything depending on which year and which district it was applied
in. At Croppa Creek the season-median NDVI swung 0.18 → 0.54 → 0.18 → 0.35
across four consecutive years.

Fixed thresholds cannot work across that. `DetectionConfig` grew
`adaptive_woody_quantile` (gate on the scene's own distribution) and
`adaptive_change_mad` (a robust `median + k·MAD` outlier threshold over the
woody population). Note that `k` around **3** is right on real data, not the
4–6 a Gaussian intuition suggests — the change distribution is heavy-tailed,
and `k = 5` rejected everything.

### 2. Band-wise median compositing manufactures spectra

Compositing by taking each band's median independently is the obvious choice
and it is quietly wrong. Over a pixel that swings through the year, the median
red can come from a bare date and the median NIR from a green one, and the
result is a spectrum no acquisition ever recorded. Measured on the Namoi
scene:

| | NDVI p50 | NDVI p85 |
|---|---|---|
| Band-wise median composite | 0.424 | 0.608 |
| Medoid composite | 0.426 | 0.674 |
| **Temporal median of NDVI** | **0.199** | **0.445** |

The composite reads 0.42 where the pixel's actual typical NDVI is 0.20. That
gap is enough to walk a cropping paddock through a woody-cover gate.

Two changes followed. `masked_medoid` picks, per pixel, the single real
observation closest to the per-band median vector, so every composite pixel is
an internally consistent spectrum — the approach Flood (2013) established for
Australian Landsat compositing, and now the default reducer here. And the
**woody gates were moved off the composite entirely** onto temporal
percentiles of the index, which is the only thing that answers "how green is
this pixel typically".

### 3. Persistent greenness is not a test for woody vegetation

The first properly-run real site was selected by looking for the largest
fraction of persistently-green pixels. It found the Namoi irrigated cotton
district, and the pipeline reported **80 ha of confirmed clearing** in seven
patches, dNBR 0.52–0.79, pre-event NDVI 0.62–0.76, corroborated by 3–4
seasons of persistence, all dating to a single operation in early November
2020.

None of it was clearing. Irrigated cotton holds NDVI above 0.6 for most of the
year and clears any persistence gate comfortably; a paddock going from cotton
to fallow produces a dNBR indistinguishable from clearing on a bitemporal
pair. The regrowth stage caught it — every patch classified `cultivated`, on
a within-year swing of 0.36–0.66 — but the detector should never have admitted
them.

What separates woody vegetation from any crop, irrigated or not, is the
*shape* of the year: woody vegetation in these landscapes is largely evergreen
and moves little, while a crop swings hard between planting and harvest. That
is now a gate: `max_pre_seasonal_amplitude`, the 90th minus the 10th
percentile of NDVI over the 24 months before the baseline, defaulting to 0.35
on the `aws` command.

Its effect on that same site:

| | Woody fraction of AOI | Confirmed clearing, all six epochs |
|---|---|---|
| Before the amplitude gate | 42% | 80 ha |
| After | 0.4–2.6% | **0 ha** |

The 80 ha was cotton rotation, and it is now correctly rejected in every year.

A fourth, smaller version of the same lesson applied to the regrowth stage:
raw NBR over inland NSW swings further between a wet year and a dry one than
a clearing event moves it, so trajectories are now expressed as anomalies
against **undisturbed woody ground in the same scene**, and the within-year
cycle is subtracted before fitting. Without both, every drought year read as
a second clearing — which is precisely what the first real run reported.

## Results

### Site selection

Picking coordinates by eye does not work: two of three tiles turned out to be
essentially fully cleared already (tile-wide woody fraction 0.4% at Moree,
5.5% at Croppa Creek). Sites were instead chosen by surveying whole 110 km
tiles at 100 m and finding the 10 km window with the most balanced
woodland-versus-cropping mix, using the amplitude definition of woody above.

| Site | Tile | Centre | Woody | Cropped |
|---|---|---|---|---|
| Boggabri / Maules Creek | 55JGG | 150.1325 E, 30.0634 S | 34.7% | 35.1% |
| Croppa Creek / Yallaroi | 56JKN | 150.8398 E, 29.1761 S | 30.9% | 31.0% |
| Moree plains | 55JGH | 149.8299 E, 29.4433 S | 7.4% | 49.6% |

### Annual epoch scan, Boggabri margin (10,486 ha)

A single before/after pair answers "was this cleared between these dates". A
monitoring programme needs the annual table, where the interesting rows are
the ones that stand out from their neighbours:

| Epoch | Woody | dNBR threshold | Confirmed patches | Confirmed ha | Review ha | Largest patch | Median patch |
|---|---|---|---|---|---|---|---|
| 2019&ndash;2020 | 56.1% | 0.393 | 3 | 5.3 | 59.6 | 3.0 | 1.2 |
| 2020&ndash;2021 | 42.2% | 0.150 | 0 | 0.0 | 0.0 | &ndash; | &ndash; |
| 2021&ndash;2022 | 30.1% | 0.126 | 1 | 0.6 | 0.0 | 0.6 | 0.6 |
| 2022&ndash;2023 | 78.8% | 0.137 | 24 | 29.9 | 13.8 | 6.1 | 0.9 |
| 2023&ndash;2024 | 66.8% | 0.178 | 0 | 0.0 | 0.0 | &ndash; | &ndash; |
| 2024&ndash;2025 | 74.2% | 0.151 | 3 | 9.1 | 1.7 | 6.5 | 1.5 |

Reading it:

* **2019–2020** put 59.6 ha in the review tier against only 5.3 ha confirmed.
  That epoch spans the Black Summer fires. Spectral change with no persistence
  is what a burn scar that recovered looks like, and the tier separation is
  doing exactly the job it exists for.
* **2022–2023** is the largest signal, and its *shape* argues against
  clearing: 24 patches with a median of 0.9 ha and shape indices of 1.5–2.2,
  scattered through the woodland rather than squared off against paddock
  boundaries. That epoch straddles the break from the 2022 La Niña floods to
  the 2023 El Niño. Drought canopy thinning is the better explanation, and the
  regrowth stage supports it: of the 29.9 ha, 9.0 ha had already **recovered**
  and 16.8 ha was **recovering** within two and a half years, on low
  within-year swings (0.05–0.25) that confirm the returning cover is woody.
  Mechanical clearing does not come back like that.
* **2024–2025** is the more clearing-shaped result: three patches, 9.1 ha, the
  largest 6.5 ha. There is not yet enough post-event record to say what is
  happening on it.
* The woody fraction moves between epochs (30–79%) because the amplitude gate
  reads the 24 months before each baseline, and how strongly the landscape
  cycled in those two years changes what counts as steady. Worth knowing
  before comparing epochs on area alone.

The same scan over the Namoi irrigated site returns **zero confirmed clearing
in all six epochs**, which is the correct answer for a fully-cultivated
floodplain and the clearest evidence that the amplitude gate works.

Outputs for the 2022–2023 epoch are in
[`docs/example-outputs-nsw/`](example-outputs-nsw/).

## What this says about using it operationally

The detection machinery transferred to real data without changes. Every
problem was a *calibration* problem, and each one was found by looking at
distributions rather than at maps — the maps looked plausible in every one of
the wrong runs, including the 80 ha of cotton.

Three specific cautions:

1. **Recalibrate per district and per epoch, not once.** The adaptive
   thresholds handle a lot, but the amplitude and woody gates still need
   reference sites. Every number in `DetectionConfig` is a starting point.
2. **Never accept a detection without looking at what happened next.** The
   regrowth trajectory is what distinguished cotton rotation from clearing,
   and drought thinning from clearing, on real data. Both were invisible in
   the bitemporal pair.
3. **Without Sentinel-1, fire is not excluded.** The 2019–2020 epoch above is
   readable only because persistence separates a recovered burn scar from a
   clearing. A scar that has *not* recovered would still pass. Getting radar
   into the pipeline — CDSE or Planetary Computer RTC from a network that can
   reach them — is the single highest-value addition.

Validation against SLATS woody change layers or aerial photo interpretation is
the obvious next step. `vegmon.validation` takes any reference mask, so the
same recall-by-size-class and decoy tables can be produced against real
reference data rather than synthetic truth.
