# Comparing GA's Sentinel-1 NRB against Level-1 GRD and SLC

Geoscience Australia publishes a terrain-corrected, CEOS-ARD certified
Sentinel-1 backscatter product (NRB). This note records what is actually in
that archive over NSW, how to pull it, and how to line each acquisition up
against the Level-1 GRD and SLC it was made from, so the three can be compared
on the same ground.

Three NSW areas were checked, chosen for different terrain and clearing
character:

| AOI | Area | Bounding box (W, S, E, N) |
| --- | --- | --- |
| `hunter` | Hunter Valley, Singleton – Muswellbrook | 150.90, -32.80, 151.45, -32.25 |
| `pilliga` | Pilliga Forest, Narrabri – Baradine | 148.90, -31.10, 149.75, -30.35 |
| `bluemtns` | Blue Mountains, Katoomba – Blackheath | 150.10, -33.85, 150.60, -33.45 |

## Two collections, and which one you are looking at

This matters more than anything else here, because the two GA collections have
completely different reach and only one of them can be read from S3.

| | Collection 1 | Collection 0 |
| --- | --- | --- |
| What | current production run | 2025 early-access release |
| Coverage | selected tracks, sample | **continental Australia**, 1 Jun 2024 – 30 Jun 2025 |
| Extended sites | none | Griffith NSW (Apr 2024 – Apr 2025), Kununurra WA (2015 – 2025), Injune/Starcke/Mungalla QLD (2018 – 2025) |
| Access | public bucket, listable | **STAC API only** |
| Tooling here | `vegmon.ga_nrb`, `tools/find_comparison_scenes.py` | `tools/find_ga_c0_scenes.py` |

Everything in the next section is **Collection 1**, read directly from the
bucket. Its shape was confirmed against GA's own burst-completeness report in
`dea-public-data-dev/projects/s1_nrb/monitoring/`, which names
`dea-public-data-dev` / `baseline` / collection `1` as the live target — so
this is the right place, not a stale copy.

**Collection 0 is a different archive and is not in any publicly listable
bucket.** Searched for and not found: `baseline/`, `derivative/`,
`projects/`, `experimental/` and the bucket roots of `dea-public-data`,
`dea-public-data-dev` and `deant-data-public-dev`. The only Collection 0
VV+VH data in reachable S3 is a single CEOS-ARD certification example under
`deant-data-public-dev/persistent/CEOS-ARD/data/example_2/`. GA's own
dev-bucket processing record lists 21,365 IW scenes of which 21,201 are
single-HH (Antarctic) and only 150 VV+VH — the continental run is not in
there either.

So a Collection 1 coverage gap is **not** evidence that GA has no data for an
area. **Confirmed by running `tools/find_ga_c0_scenes.py` from a network that
can reach the API** — Collection 0 covers all three, with real time series:

| AOI | Collection 0 items | Dates | Tracks |
| --- | --- | --- | --- |
| Hunter | 895 | 97 | 9, 74, 147 |
| Pilliga | 946 | 95 | 9, 45, 118 |
| Blue Mountains | 444 | 64 | 9, 147 |

All VV+VH, over 1 Jun 2024 – 30 Jun 2025. Those tracks match what the OPERA
burst database predicts for each AOI, which is a useful cross-check that the
two independent routes agree.

**Track 9 covers all three areas.** Since ascending and descending passes view
the canopy from opposite sides, a time series should be built from one track —
and track 9 is the one that lets all three areas share the same geometry. Pass
`--track 9`.

## Collection 1 covers one of the three

Collection 1 is a **soft-release sample over selected tracks, not
continental coverage**. Checking every published burst in all four IW products
against the three AOIs:

| AOI | Collection 1 bursts | Track | Nearest published burst |
| --- | --- | --- | --- |
| Pilliga | **17** | 45 | on target |
| Hunter | **0** | – | ~134 km west |
| Blue Mountains | **0** | – | ~91 km west |

None of GA's own extended-time-series sites are in Collection 1 either —
Griffith NSW and Injune QLD both return 0 bursts — which is another way of
seeing that Collection 1 and Collection 0 are separate archives.

Over mainland Australia the VV+VH product is 1,761 bursts on 13 tracks. In
NSW that is a **single descending swath on relative orbit 45**, running
Moree → Narrabri → Dubbo/Mudgee → Bathurst/Orange → Cowra → Wagga. It passes
inland of both the Hunter and the Blue Mountains. That is a statement about
Collection 1 only; Collection 0's continental run is the thing to check for
those two areas.

The other IW products do not help: `ga_s1_nrb_iw_hh_1` (12,744 bursts) and
`ga_s1_nrb_iw_hh_hv_1` (287) are polar, and Sentinel-1 flies VV+VH over
Australia anyway.

If Collection 0 turns out to be unavailable to you and a Collection 1
substitute is needed, the two nearest on-track options are:

* **Upper Hunter around Merriwa/Cassilis** — bursts `t045_095779_iw1`,
  `t045_095780_iw1`. Same catchment, grazing and woodland rather than the
  Singleton coal/vineyard mix.
* **Oberon / Kanangra, west of the Blue Mountains** — bursts
  `t045_095787_iw1`, `t045_095788_iw1`. Steep forested terrain with active
  plantation forestry, which exercises terrain correction similarly.

## What is in the Pilliga

Seventeen bursts on track 45, and **one acquisition each**: 2026-04-20.
There is no time series in this product yet. For change detection, GA NRB is
currently a single-date reference image, not a source of series.

Both SLCs behind those bursts, with their GRD twins:

| GA burst datetime | Parent SLC | GRD twin |
| --- | --- | --- |
| 20260420T192256–192323 | `S1A_IW_SLC__1SDV_20260420T192256_20260420T192323_064167_081391_1104` | `S1A_IW_GRDH_1SDV_20260420T192256_20260420T192321_064167_081391_D3B4` |
| 20260420T192321–192348 | `S1A_IW_SLC__1SDV_20260420T192321_20260420T192348_064167_081391_D1D8` | `S1A_IW_GRDH_1SDV_20260420T192321_20260420T192346_064167_081391_A830` |

Absolute orbit 64167, relative orbit 45, descending.

### How the pairing is established

A GRD and an SLC belong to the same acquisition when they share an **absolute
orbit** *and* a **data-take id** (`064167_081391` above). That match is exact.
The file names cannot be derived from one another — the slice boundaries and
the trailing product CRC differ — so it has to be looked up, not constructed.

The tie from GA's product back to Level-1 is free: the NRB STAC item carries
`sarard:scene_id`, which is the SLC name verbatim.

Verified for `t045_095772_iw2`: the D3B4 slice covers **91.6%** of the burst,
the neighbouring A830 slice covers the remaining **8.4%**, and together they
cover it completely. A burst near a slice edge needs both — which is why the
manifest records how many slices the pass was cut into.

## Level-1 coverage for all three areas

GA's gap is not a Sentinel-1 gap. Scanning the GRD archive for 2026-08-17 to
2026-09-16 finds current coverage over every AOI:

| AOI | Relative orbit | Scenes in 30 days | Dates |
| --- | --- | --- | --- |
| Hunter | 147 | 3 | 2026-08-20, 09-01, 09-13 |
| Pilliga | 45 | 2 | 2026-08-25, 09-06 |
| Blue Mountains | 147 | 3 | 2026-08-20, 09-01, 09-13 |

Hunter and the Blue Mountains sit on the **same pass** — identical absolute
orbits (4213, 4388, 4563), adjacent slices a few seconds apart. Anything
comparing those two areas is comparing them under identical viewing geometry
on the same day, which is about as clean as a cross-site comparison gets.

Every one of these is **Sentinel-1D**, which is the point of the fix below:
before it, this table was empty.

The full per-scene manifest, with bucket URLs, is in
[`comparison_manifest.json`](./comparison_manifest.json).

## Getting the data

```bash
# every burst covering an AOI, plus that burst's static layers
python tools/fetch_ga_nrb.py --aoi pilliga --out data/ga_nrb

# just the two bursts over the forest core
python tools/fetch_ga_nrb.py --aoi pilliga --out data/ga_nrb \
    --bursts t045_095772_iw2 t045_095771_iw2
```

Confirmed working: 483 MB for those two bursts, every file matching the
`.sha1` GA publishes beside it.

```bash
# matched GA / SLC / GRD manifest for all three AOIs
python tools/find_comparison_scenes.py --out comparison_manifest.json
```

Both go straight at the public buckets with anonymous `ListObjectsV2` — no
account, no token, and nothing through a STAC API.

Collection 0 has no bucket to list, so it goes through the STAC API instead:

```bash
# needs explorer.dev.dea.ga.gov.au to be reachable
python tools/find_ga_c0_scenes.py --out c0_manifest.json

# coverage check only, skipping the per-SLC GRD lookups
python tools/find_ga_c0_scenes.py --aoi hunter bluemtns --no-level1
```

Restrict to one track and cap the pairing, or the run is long — an AOI has
~900 items over ~95 dates, and each distinct SLC needs a day of the global GRD
archive listed:

```bash
python tools/find_ga_c0_scenes.py --track 9 --max-slcs 20 --out c0_manifest.json
```

Day listings are cached, so several SLCs from one pass cost one listing rather
than one each (measured: 1.8 s cold, 0.00 s warm).

| What | Where |
| --- | --- |
| GA NRB products | `s3://dea-public-data-dev/baseline/ga_s1_nrb_iw_*` (ap-southeast-2) |
| OPERA burst database | `s3://dea-public-data-dev/projects/s1_nrb/burst_db/0.18.0/` |
| Sentinel-1 GRD | `s3://sentinel-s1-l1c/GRD/{y}/{m}/{d}/IW/DV/` (eu-central-1), 2014–present |
| Sentinel-1 SLC | `s3://sentinel1-slc/{y}/{m}/{d}/IW/` (eu-west-1), **2014–2022 only** |

### The SLC gap

The open SLC archive on AWS stops at 2022. The 2026 Pilliga SLCs are named
correctly in the manifest but cannot be pulled from it — they need CDSE or ASF,
both of which want credentials. The GRD twins have no such problem; that
archive is current.

## What each product actually gives you

Measured on `t045_095772_iw2`, the real downloaded rasters:

* **Grid** — EPSG:32755 (UTM 55S), 20 m, 5160 × 2834.
* **VV gamma0** — float32 linear power, NaN nodata. Median −13.5 dB,
  5th–95th percentile −18.9 to −9.6 dB. Plausible for dry inland woodland.
* **Layover/shadow mask** — uint8. **Only 31.6% of the burst grid is valid
  data**; the other 68.4% is 255 (invalid). This is the documented Collection 1
  caveat that burst geometries overrun their valid data. Budget for it: a burst
  is not a full 103 × 57 km of usable pixels.
* **Static layers** — 100% valid across the full grid, since they come from
  DEM geometry rather than the acquisition. Local incidence angle spans
  4.8°–71.6°.

The comparison worth making is the one the mask and the static layers set up:
GA NRB is radiometrically terrain corrected against a real DEM, the GRD is
not. Over the Blue Mountains or Oberon that difference is large and
systematic; over the flat Pilliga it should nearly vanish. Differencing NRB
gamma0 against a GRD calibrated through `vegmon.s1grd` — which does ellipsoid
gamma0 and explicitly skips terrain correction — isolates exactly the terrain
term, with the static `local-incidence-angle` layer as the explanatory
variable.

## A bug this turned up

Scanning the 2026 GRD archive found **no Sentinel-1A at all** — the recent
record is entirely S1C and S1D. Two problems in `vegmon/s1grd.py` followed:

* The scene-id regex matched `S1[ABC]`, so every **S1D** scene was silently
  dropped. In a 12-day window in September 2026, S1D was 5,454 of 9,946
  scenes. Sentinel-1 search was returning nothing for recent dates.
* The **S1C** orbit offset was set to 73, copied from S1A. It is **172**.
  Every S1C scene was being labelled with a track 99 out, and since the
  detector deliberately pins one relative orbit, that silently mixed orbits.

Both are fixed. The offsets are now anchored to independent evidence rather
than inference:

| Satellite | Offset | Anchored on |
| --- | --- | --- |
| S1A | 73 | GA NRB metadata (abs 55082 → track 60, and 5 more) |
| S1B | 27 | GA NRB metadata (abs 25010 → track 134, and 5 more) |
| S1C | 172 | GA NRB metadata (abs 7232 → track 61) |
| S1D | 42 | footprint-to-burst-database matching, 6 independent orbits, tracks 16–104 |

`relative_orbit()` now raises on a satellite it has no offset for, instead of
falling back to S1A's and quietly returning a wrong track.

## Reaching the archives

The documented route into GA's product is DEA's development STAC API at
`explorer.dev.dea.ga.gov.au`. It is unreachable from this environment, as are
ASF, CDSE, Earth Search, Planetary Computer and NASA CMR. The AWS data buckets
behind all of them are open. Everything here works by listing those buckets
directly, which is the same approach `vegmon.s3direct` and `vegmon.s1grd`
already take for Sentinel-2 and Sentinel-1 — and which also sidesteps GA's own
warning that the dev STAC API should not be relied on.
