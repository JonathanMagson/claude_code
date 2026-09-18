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

| Graph | Calibration | Speckle | Terrain flattening | Output |
|---|---|---|---|---|
| `grd_gamma0_rtc` | beta0 | none | yes | gamma0 RTC |
| `grd_gamma0_rtc_reflee` | beta0 | Refined Lee | yes | gamma0 RTC |
| `grd_gamma0_rtc_leesigma` | beta0 | Lee Sigma 7x7 | yes | gamma0 RTC |
| `grd_sigma0_ellipsoid` | sigma0 | none | **no** | sigma0, geocoded only |

`grd_gamma0_rtc` is the baseline: it is the closest SNAP equivalent to the GA
NRB, so it is what any "how does my processing compare to GA's" question should
be asked against. `grd_sigma0_ellipsoid` is the control that shows what terrain
flattening buys you over plain terrain correction.

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
python tools/run_snap_grd.py D:\scratch\sentinel_1_data_project\before_after ^
    --variant grd_gamma0_rtc ^
    --gpt "C:\Program Files\esa-snap\bin\gpt.exe"
```

Useful flags: `--print-commands` emits `cmd.exe` one-liners instead of running
anything; `--overwrite` re-runs finished jobs; `--crs` overrides the projection
that is otherwise inferred from the AOI folder name; `--list-variants` shows
what is available. Finished jobs are skipped automatically, and a `.dim` with no
matching `.data/` counts as unfinished rather than done.

## Notes

- `Apply-Orbit-File` uses `continueOnFail=false`. Precise orbits are published
  ~20 days after acquisition and are available for all six scenes; failing loudly
  beats silently geocoding against the predicted orbit.
- `Remove-GRD-Border-Noise` is kept, but for IPF >= 2.90 (all of these 2024-25
  scenes) border noise is already handled by thermal noise removal.
- The graphs are templated. `gpt` does not default `${...}` placeholders, so all
  five (`input`, `output`, `crs`, `spacing`, `dem`) must be supplied on every
  invocation. The runner always supplies all five.
- SNAP is not installed in the sandbox these graphs were written in. They are
  validated for structure, connectivity and templating by
  `tests/test_snap_graphs.py`; they have not been executed.
