# SNAP GRD pre-processing

Four templated SNAP graphs in `graphs/`, plus `tools/run_snap_grd.py` to batch
them over the downloaded GRD zips. Outputs are written one level below each
GRD, in a sub-folder per variant, so methods can be compared side by side.

```
before_after/pilliga/t009_019142_iw1/20240603/
    S1A_IW_GRDH_..._D146.zip            <- the GRD
    grd_preprocessed/
        grd_gamma0_rtc/                 <- variant 1
        grd_gamma0_rtc_reflee/          <- variant 2
        grd_gamma0_rtc_leesigma/        <- variant 3
        grd_sigma0_ellipsoid/           <- variant 4
```

## The variants

| Graph | Calibration | Speckle | Flattening | Geocoded | Output |
|---|---|---|---|---|---|
| `grd_gamma0_rtc` | beta0 | none | yes | yes | gamma0 RTC |
| `grd_gamma0_rtc_reflee` | beta0 | Refined Lee | yes | yes | gamma0 RTC |
| `grd_gamma0_rtc_leesigma` | beta0 | Lee Sigma 7x7 | yes | yes | gamma0 RTC |
| `grd_gamma0_ellipsoid` | gamma0 | none | **no** | yes | gamma0, ellipsoid |
| `grd_gamma0_ellipsoid_reflee` | gamma0 | Refined Lee | **no** | yes | gamma0, ellipsoid |
| `grd_gamma0_ellipsoid_leesigma` | gamma0 | Lee Sigma 7x7 | **no** | yes | gamma0, ellipsoid |
| `grd_sigma0_ellipsoid` | sigma0 | none | **no** | yes | sigma0, ellipsoid |
| `grd_gamma0_rtc_defaults` | beta0 | none | yes, SNAP defaults | yes | gamma0 RTC |
| `grd_gamma0_tcnorm` | beta0 | none | in terrain correction | yes | gamma0, normalised |
| `grd_tf_diagnostic` | beta0 | none | yes | **no** | flattened + simulated image |

**Use `grd_gamma0_ellipsoid` for now.** `grd_gamma0_rtc` is in principle the
closest SNAP equivalent to the GA NRB, but SNAP's Terrain-Flattening on these
GRD products displaces the output geographically -- see "Terrain flattening
shifts GRD geolocation" below. The two `_ellipsoid` graphs are controls that show what
terrain flattening buys you over plain terrain correction; `grd_gamma0_ellipsoid`
is also the "turn flattening off for now" option, since it keeps the gamma0
convention and so stays at least dimensionally comparable to the GA product.

Common settings, chosen to match the GA NRB so no reprojection is needed before
comparing: **20 m** pixel spacing, **Copernicus 30m Global DEM**, output grid
aligned to a standard grid, and the **projection read from the GA raster beside
the scene**.

Do not assume the UTM zone from the AOI's longitude. GA chooses it per burst,
and it is not always the zone the AOI centre falls in -- the Pilliga centre is
in zone 55, but GA delivers that burst in zone 56. The runner reads the CRS off
`ga_*gamma0.tif` when rasterio is available, falls back to a per-AOI table
otherwise, and `--crs` overrides both.

Speckle filtering, where present, runs in **radar geometry** (after calibration,
before flattening or geocoding). That is where speckle statistics hold; after
geocoding, resampling has already correlated neighbouring pixels.

**Match filtered to filtered.** The GA NRB is delivered unfiltered, so compare it
against an unfiltered variant (`grd_gamma0_ellipsoid`). To compare filtered
products, use the `_sf_db` rasters that `postprocess_nrb.py` writes from the NRB
against one of the `_reflee` / `_leesigma` variants. Comparing a filtered GRD
against a raw NRB attributes your filter's smoothing to a processing
difference.

## Linear, not dB

None of the graphs end in `LinearToFromdB`. Output stays in linear power so it
can be differenced and averaged correctly, and so the filter-then-convert order
is guaranteed. Convert to dB in the comparison step, the same way
`tools/postprocess_nrb.py` does for the NRB side.

## Running it

Dry run first:

```
python tools/run_snap_grd.py D:\scratch\sentinel_1_data_project\before_after --dry-run
```

One variant, for real:

```
python tools/run_snap_grd.py D:\scratch\sentinel_1_data_project\before_after --variant grd_gamma0_rtc
```

`--gpt` is optional: the runner checks PATH, then the usual SNAP install
locations, including the user-scope directory under `LOCALAPPDATA` that SNAP's
Windows installer falls back to when it cannot write to Program Files. It prints
which one it picked. Pass `--gpt` explicitly if you have several SNAP versions.

Useful flags: `--print-commands` emits `cmd.exe` one-liners instead of running
anything; `--overwrite` re-runs finished jobs; `--crs` overrides the projection
that is otherwise inferred from the AOI folder name; `--list-variants` shows
what is available. Finished jobs are skipped automatically, and a `.dim` with no
matching `.data/` counts as unfinished rather than done.

## Two further attempts at terrain flattening

After `grd_gamma0_rtc` failed (below), two variants test different explanations.

**`grd_gamma0_rtc_defaults`** keeps Terrain-Flattening but reverts every setting
that `grd_gamma0_rtc` had tuned away from SNAP's defaults -- `oversamplingMultiple`
2.0 back to 1.0, `additionalOverlap` 0.2 back to 0.1, `nodataValueAtSea` false
back to true.

**Result: worse, and conclusive.** On the Blue Mountains it produced a raster
only 5.6% valid, sharing *zero* pixels with the GA burst's 30.5%. Two faults at
once: `nodataValueAtSea: true` masks about 94% of the product against a
Copernicus DEM, and what survived is still displaced. Since the displacement
persists at SNAP's own defaults, the tuning was never the cause -- Terrain
Flattening on these GRD products displaces regardless of parameters. Do not use
this variant; it is kept only as the record of that test.

**`grd_gamma0_tcnorm`** drops Terrain-Flattening and normalises inside
Range-Doppler Terrain Correction instead, in a single pass. On the Blue
Mountains it lifted correlation against the GA NRB from 0.48 to 0.65 with
sub-half-pixel registration, so the terrain normalisation works.

There is deliberately **no Calibration node**: the normalisation calibrates
internally from the aux file, and calibrating beforehand applies the LUT twice.
The first attempt did exactly that and came out 53 dB dark, which is
`10*log10(1/A^2)` for a Sentinel-1 calibration constant of a few hundred.

**It needs a second step:** SNAP writes `Sigma0_*` bands rather than `Gamma0_*` regardless of
`saveGammaNought`, along with `projectedLocalIncidenceAngle`. Run

```
python tools/sigma0_to_gamma0.py <the grd_gamma0_tcnorm folder>
```

to finish the conversion -- `gamma0 = sigma0 / cos(local incidence angle)` --
which writes `Gamma0_<pol>.img` into each product where `compare_to_nrb.py`
finds it. Pixels flagged as layover or shadow, and local incidence angles above
80 degrees, are dropped rather than amplified: the quotient explodes as the
cosine approaches zero, which is the same instability GA caps with
`rtc_min_value_db: -30`. The `.dim` header is not updated, so SNAP itself will
not list the added bands. This is weaker
radiometrically -- a local-incidence-angle correction rather than true area
integration, so a looser approximation of GA's `area_projection` RTC -- but it
structurally cannot suffer the failure that broke `grd_gamma0_rtc`, because
there is no intermediate product to resample and re-geocode. If this works and
the other does not, the fault is in Terrain-Flattening on GRD itself.

Never enable Terrain-Correction's `applyRadiometricNormalization` while
Terrain-Flattening is also running: that normalises slopes twice. A test pins it.

## Terrain flattening shifts GRD geolocation

Confirmed on the Pilliga scenes, 2025-09: running `grd_gamma0_rtc` produced a
product displaced slightly south of the GA NRB, with edge artefacts. Running the
identical graph without Terrain-Flattening (`grd_gamma0_ellipsoid`) removed both.
Precise orbits were confirmed applied in both runs, so this is not the orbit.

This is worth writing down because the obvious reasoning is wrong. Terrain
flattening is usually described as a radiometric correction, and Sentinel-1's
near-polar orbit means a DEM height error displaces pixels across-track
(east-west) rather than along-track (north-south) -- which together suggest
flattening could not cause a southward shift. It did. SNAP's Terrain-Flattening
resamples into the geometry of the DEM-derived simulated image, so it moves
pixels as well as rescaling them, and on GRD the result is not guaranteed to
come back to where it started.

**What this costs.** `grd_gamma0_ellipsoid` is gamma0 from the ellipsoid
incidence angle, not radiometrically terrain-flattened. Against the GA NRB it
will differ on slopes by several dB, and the difference is correlated with the
terrain -- steeply so in the Blue Mountains, much less in the Pilliga. Any
comparison has to state that, because it is exactly the quantity RTC exists to
remove.

**The likely real fix** is to stop using GRD. GA derives its NRB from SLC
bursts, where terrain flattening is well posed; on GRD the ground-range
detection has already happened before the terrain is accounted for, so SNAP's
implementation is an approximation to begin with. The matching SLCs are already
downloaded alongside each GRD.

## Debugging terrain flattening

Terrain flattening producing something that "looks geometrically wrong" almost
never means the flattening parameters are wrong. Work through it in this order.

**0. Check what you are looking at.** Terrain-Flattening output is still in
radar geometry. It is *supposed* to look skewed. Only after Range-Doppler
Terrain-Correction does it become a map. `grd_tf_diagnostic` deliberately stops
before geocoding, so its output looks wrong by design.

**1. The orbit.** SNAP's `continueOnFail` on Apply-Orbit-File silently falls
back to the predicted orbit shipped in the product when the precise orbit cannot
be downloaded. Flattening then divides the real image by a simulated image built
from a misregistered orbit, which shows up as smeared or doubled terrain edges
and bright/dark banding that follows the topography. Every step still reports
success. All the graphs here set `continueOnFail=false` so this fails loudly
instead.

**2. Look at the simulated image.**

```
python tools/run_snap_grd.py <root> --variant grd_tf_diagnostic --gpt <path to gpt>
```

That writes the flattened bands *and* `simulatedImage` in radar geometry. Open
both in SNAP and flicker between them. If the simulated terrain does not sit on
top of the real terrain, the problem is the orbit or the DEM, not flattening.

**3. The DEM.** SRTM 1Sec has voids over water and steep terrain, and its
auto-download endpoint is unreliable; a void becomes a hole in the simulated
area and a hole or spike in the output. Copernicus 30m is void-filled and is
what GA uses. Swap without editing any XML:

```
python tools/run_snap_grd.py <root> --dem "SRTM 1Sec HGT"
```

**4. Oversampling.** `oversamplingMultiple` defaults to 1.0 in SNAP, which
undersamples the simulated image relative to the SAR grid; over high relief the
result is holed, striped or blocky. These graphs default to 2.0. Tune with
`--oversampling` and `--overlap`.

**5. Only then, turn it off.** `grd_gamma0_ellipsoid` removes flattening
entirely. That isolates whether the fault is in flattening or upstream of it.
It is a diagnostic, not a replacement: on slopes it differs from the GA NRB by
several dB, and the error is correlated with the terrain, which is exactly the
signal most analyses are trying to measure.

## Comparing against the GA NRB

`tools/compare_to_nrb.py` walks the tree, pairs every GA NRB raster with the
SNAP band for the same scene and polarisation, puts both on the GA grid over
their shared area, and reports the difference:

```
python tools/compare_to_nrb.py <root> --variant grd_gamma0_ellipsoid --csv comparison.csv
```

| Column | Meaning |
|---|---|
| `bias` | median(SNAP - GA) in dB. Moves with effective look count, so not a calibration figure on its own. |
| `gain` | the same offset on linear power, where the mean is unbiased by look count. **Quote this for calibration.** |
| `rmse` | spread of the difference in dB. Where a missing terrain correction shows up. |
| `corr` | Pearson correlation of the two dB images. Structure agreement. |
| `shift` | geolocation offset in pixels, estimated to ~0.04 px. Above ~0.5 px the radiometry is contaminated by misregistration. |

**Read the spread, not the offset, for terrain.** RTC redistributes energy per
pixel -- slopes facing the sensor down, slopes facing away up -- so over a scene
it largely cancels in the median and leaves `bias` near zero however rugged the
ground is. It inflates `rmse` and depresses `corr` instead. Measured on these
six scenes with no terrain flattening, against the GA NRB:

| AOI | bias | rmse | corr |
|---|---|---|---|
| Pilliga (flat) | +0.30 | 2.0 | 0.70 |
| Hunter (undulating) | +0.40 | 2.6 | 0.53 |
| Blue Mountains (steep) | +0.28 | 3.4 | 0.48 |

`bias` carries no terrain signal at all; `rmse` and `corr` order by terrain
monotonically. Note this ordering is consistent with RTC driving it but does not
prove it: rugged terrain also decorrelates more from sub-pixel registration
differences and from higher scene contrast.

**Why both `bias` and `gain`.** Speckle is skewed, so the median sits below the
mean, and a product with more looks has a higher median in dB at identical true
backscatter. Comparing medians in dB across products with different look counts
reads that as a calibration offset. The linear mean does not move with looks. If
the two disagree, the difference between them is the look-count artefact.

Pass `--filtered` when the SNAP variant applies a speckle filter: it then pairs
against the `_sf_db` rasters from `postprocess_nrb.py` instead of the raw NRB.
Filtered-against-raw attributes your filter's smoothing to a processing
difference, so the tool refuses to mix them by accident.

## What `shift` does and does not tell you

`shift` is the **relative** offset between the two products, not the absolute
accuracy of either. If both sat 30 m from truth in the same direction it would
still read zero. Establishing absolute accuracy needs an independent reference:
corner reflectors, a surveyed water body or coastline, or high-accuracy optical
imagery.

On theory GA should be the more absolutely accurate of the two. Its config
applies two geometric corrections that SNAP's Range-Doppler Terrain Correction
does not:

```yaml
apply_bistatic_delay_correction: True          # S1_RTC_IW.yaml:158
apply_static_tropospheric_delay_correction: True   # S1_RTC_IW.yaml:161
```

Both are metre-scale. At 20 m posting they are well under a pixel, which is
consistent with the measured agreement and also means this comparison cannot
resolve them.

## Measuring a geolocation offset

"The scene looks slightly south" is not actionable. `tools/measure_shift.py`
phase-correlates a SNAP output against the GA NRB over the area they share and
reports the offset in pixels and metres:

```
python tools/measure_shift.py --reference <GA ..._VH-gamma0.tif> --target <SNAP .dim>
```

The direction is the diagnosis. Sentinel-1 flies a near-polar orbit, so a DEM
height error displaces pixels **across-track** (roughly east-west), while orbit
timing and the geocoding move them **along-track** (roughly north-south). A
north-south offset therefore largely exonerates the DEM and terrain flattening.

## Notes

- `Apply-Orbit-File` uses `continueOnFail=false`. See "Debugging terrain
  flattening" below -- this is the setting most likely to produce a product that
  looks geometrically wrong while every step reports success.
- `Remove-GRD-Border-Noise` is kept, but for IPF >= 2.90 (all of these 2024-25
  scenes) border noise is already handled by thermal noise removal.
- The graphs are templated. `gpt` does not default `${...}` placeholders, so all
  five (`input`, `output`, `crs`, `spacing`, `dem`) must be supplied on every
  invocation. The runner always supplies all five.
- SNAP is not installed in the sandbox these graphs were written in. They are
  validated for structure, connectivity and templating by
  `tests/test_snap_graphs.py`; they have not been executed.
