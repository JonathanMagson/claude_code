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

## Sentinel-1, calibrated from the raw archive

Every *analysis-ready* Sentinel-1 product is out of reach here. Planetary
Computer's `sentinel-1-rtc`, CDSE and ASF are all blocked at the egress
policy, and the one open RTC bucket on AWS (`sentinel-s1-rtc-indigo`) covers
UTM zones 10-19 — the continental United States. Australia has no
analysis-ready option on any reachable host.

What *is* open, anonymous and complete is the **Level-1 GRD archive** in
`s3://sentinel-s1-l1c`: full SAFE products, measurement GeoTIFFs, calibration
and thermal-noise LUTs, and the geolocation grid. That is enough to build the
product rather than download it, and `vegmon.s1grd` does:

1. **Find the scenes without a search API.** There is no index, so coverage
   comes from the orbit. Sentinel-1 is sun-synchronous with an 18:00
   local-solar-time ascending node, so a given longitude is only ever imaged
   in two narrow UTC windows a day — which cuts the ~440 daily IW/DV scenes
   to a few dozen. One repeat cycle is scanned against each scene's published
   footprint to learn which relative orbits see the AOI; everything after that
   follows the 12-day repeat and costs a handful of requests.
2. **Calibrate.** `gamma0 = (DN² − noise) / A²`, with the per-product
   calibration and noise LUTs applied in radar geometry where they are
   defined. Noise removal matters specifically for the cross-polarised
   channel: VH over woodland sits only a few decibels above a noise floor
   that ramps across the swath, so leaving it in puts a range-dependent bias
   straight into the band the detector reads.
3. **Geocode.** The measurement TIFFs are tiled and carry the 210-point
   geolocation grid as GCPs, so a windowed GCP warp pulls just the AOI out of
   a 500 MB swath in about two seconds. The warp runs at the native 10 m and
   the result is averaged down to the working grid **in linear power, not in
   decibels** — averaging logarithms biases every multi-looked pixel low, and
   is one of the easier ways to produce a plausible-looking backscatter image
   that is quietly wrong. The 2×2 aggregation is also where the looks come
   from.

Loading 196 acquisitions (2019-2025, relative orbit 45) over a 10,486 ha AOI
took 15 minutes.

### No terrain correction, and why it is defensible here

This does not do radiometric terrain correction. That matters less than it
sounds, for one specific reason: the detector compares two acquisitions
**from the same relative orbit**, so the viewing geometry, the local incidence
angle and the terrain-driven part of the backscatter are identical in both and
divide out of the difference. Ellipsoid gamma0 from a pinned orbit is a sound
basis for change detection. It is *not* a sound basis for comparing absolute
levels between orbits, and `s1grd` refuses to mix orbits for that reason —
ascending and descending passes view the canopy from opposite sides and differ
over the same intact woodland by more than the drop being looked for.

### Does the calibration hold up?

Checked against the Sentinel-2 woody mask over the same AOI, and against
itself through time:

| Check | Result | Expected |
|---|---|---|
| Woody VH gamma0 | −16.2 dB | −15 to −18 dB for woodland |
| Cropping VH gamma0 | −17.6 dB | lower than woodland |
| Woody minus cropping, VH | **+1.4 dB** | positive: canopy volume scattering |
| Cross-ratio VH−VV, woody | **−5.7 dB** | less negative than cropping |
| Cross-ratio VH−VV, cropping | −7.3 dB | surface scattering dominates |
| Stable-woodland VH, 55 dates | **sd 0.73 dB** | ~0.5-1 dB from soil moisture |

The last row is the one that matters most: a broken calibration shows
multi-decibel jumps between scenes, and 0.73 dB of scatter over 55 dates is
what correctly-calibrated Sentinel-1 does over stable forest.

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

### 4. A fixed decibel threshold does not transfer either

The same trap as the optical thresholds, on the radar side. Measured over the
real NSW scene, the VH change distribution has a MAD of only **0.43 dB** but a
99th percentile of **2.7 dB** — speckle residual and soil moisture give it a
heavy tail the optical indices do not have. A fixed 2 dB drop, which is a
perfectly defensible number in the literature and the one the synthetic scene
suggested, selected **4.6% of the whole AOI** and produced 440 ha of
single-sensor "detections". `adaptive_vh_mad` applies the same robust
`median + k·MAD` rule, and the right `k` there is about **5** where **3** is
right on the optical side. The difference is the tail, not the noise level.
That change cut the spurious radar-only area from 440 ha to 28 ha.

A fifth, smaller version of the same lesson applied to the regrowth stage:
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
| 2019&ndash;2020 | 56.1% | 0.393 | 0 | 0.0 | 93.9 | &ndash; | &ndash; |
| 2020&ndash;2021 | 42.2% | 0.150 | 0 | 0.0 | 93.6 | &ndash; | &ndash; |
| 2021&ndash;2022 | 30.1% | 0.126 | 0 | 0.0 | 30.4 | &ndash; | &ndash; |
| 2022&ndash;2023 | 78.8% | 0.137 | 0 | 0.0 | 71.8 | &ndash; | &ndash; |
| 2023&ndash;2024 | 66.8% | 0.178 | 0 | 0.0 | 0.8 | &ndash; | &ndash; |
| 2024&ndash;2025 | 74.2% | 0.151 | 0 | 0.0 | 69.9 | &ndash; | &ndash; |

Reading it (this table is the **fused** run, with real Sentinel-1 supplying the
confirming evidence):

* **Nothing is confirmed, in any epoch.** Everything the optical detector
  found sits in the review tier because the radar did not corroborate it. The
  per-epoch breakdown of what each sensor saw is in the next section, and it
  is the substantive result of this whole exercise.
* **2022–2023** is by far the largest optical signal (171 ha before sieving,
  71.8 ha of patches after). Its *shape* already argued against clearing:
  small patches, median under 1 ha, shape indices 1.5–2.2, scattered through
  the woodland rather than squared off against paddock boundaries. That epoch
  straddles the break from the 2022 La Niña floods to the 2023 El Niño.
* Run with **temporal persistence** instead of radar, the same epoch returns
  24 confirmed patches over 29.9 ha — and then the regrowth stage undercuts
  them anyway: 9.0 ha had already **recovered** and 16.8 ha was **recovering**
  within two and a half years, on low within-year swings (0.05–0.25) that
  confirm the returning cover is woody. Mechanical clearing does not come
  back like that. Persistence rules out harvest and drought *dips*; it cannot
  rule out a multi-year drought *decline*, and here it did not. The radar
  does.
* The woody fraction moves between epochs (30–79%) because the amplitude gate
  reads the 24 months before each baseline, and how strongly the landscape
  cycled in those two years changes what counts as steady. Worth knowing
  before comparing epochs on area alone.

The same scan over the Namoi irrigated site returns **zero confirmed clearing
in all six epochs**, which is the correct answer for a fully-cultivated
floodplain and the clearest evidence that the amplitude gate works.

### What Sentinel-1 says about all of it

With real radar in the pipeline, the fusion returns **zero confirmed clearing
in every epoch at this site**. That is not the detector failing to fire — it is
the two sensors disagreeing, which is the answer the design exists to produce.
The table below is the whole argument:

| Epoch | Sentinel-2 ha | Sentinel-1 ha | Both ha | Median VH drop *inside* the optical detections |
|---|---|---|---|---|
| 2019&ndash;2020 | 92.5 | 34.5 | 0.6 | 0.20 dB |
| 2020&ndash;2021 | 7.6 | 118.4 | 0.1 | 0.40 dB |
| 2021&ndash;2022 | 4.9 | 35.7 | 0.0 | &minus;0.02 dB |
| 2022&ndash;2023 | 171.4 | 34.8 | 4.2 | 1.34 dB |
| 2023&ndash;2024 | 3.7 | 1.7 | 0.0 | 0.12 dB |
| 2024&ndash;2025 | 27.4 | 66.0 | 0.8 | 0.10 dB |

The last column is the finding. Across seven years, where Sentinel-2 sees
spectral loss, Sentinel-1 sees essentially no structural loss — a few tenths
of a decibel, against an adaptive threshold of 1.5-3 dB. Even the large
2022-2023 signal only reaches 1.34 dB. Woody structure did not leave those
sites.

That is an independent confirmation of what the regrowth trajectories already
said: the 2022-2023 signal is **drought canopy thinning**, not clearing.
Browning drops NBR hard and leaves the stems standing, so the radar does not
move; and the sites greened back up within two and a half years. Two
unrelated lines of evidence, the same conclusion.

**What this run does not establish** is that the detector finds real clearing
on real data — there was apparently none to find in this AOI over this period,
and a null result cannot demonstrate sensitivity. That has only been shown
against synthetic truth. Validation against SLATS woody change layers remains
the necessary next step.

Outputs for the 2022–2023 epoch are in
[`docs/example-outputs-nsw/`](example-outputs-nsw/), including the recovery
trajectories from the persistence-confirmed variant of the same run.

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
