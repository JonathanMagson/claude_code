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
| `grd_sigma0_ellipsoid` | sigma0 | none | **no** | yes | sigma0, ellipsoid |
| `grd_tf_diagnostic` | beta0 | none | yes | **no** | flattened + simulated image |

`grd_gamma0_rtc` is the baseline: it is the closest SNAP equivalent to the GA
NRB, so it is what any "how does my processing compare to GA's" question should
be asked against. The two `_ellipsoid` graphs are controls that show what
terrain flattening buys you over plain terrain correction; `grd_gamma0_ellipsoid`
is also the "turn flattening off for now" option, since it keeps the gamma0
convention and so stays at least dimensionally comparable to the GA product.

Common settings, chosen to match the GA NRB so no reprojection is needed before
comparing: **20 m** pixel spacing, **UTM** (EPSG:32755 for Pilliga, 32756 for
Hunter and the Blue Mountains), **Copernicus 30m Global DEM**, output grid
aligned to standard grid.

Speckle filtering, where present, runs in **radar geometry** (after calibration,
before terrain flattening). That is where speckle statistics hold; after
geocoding, resampling has already correlated neighbouring pixels.

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
