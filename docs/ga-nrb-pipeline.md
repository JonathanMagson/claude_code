# How GA builds the Sentinel-1 NRB product

Read from `GeoscienceAustralia/sar-pipeline` at `a292888`: the entrypoint
`scripts/run_isce3_rtc_pipeline.sh`, the run-config templates in
`sar_pipeline/configs/isce3_rtc/`, and `Docker/isce3_rtc/Dockerfile`.

One run processes **one SLC** and emits **one product per burst**. Nothing is
mosaicked — `save_bursts: True`, `save_mosaics: False`.

A rendered copy of the diagram below is at
[`ga-nrb-pipeline.png`](./ga-nrb-pipeline.png), for viewers that do not draw
Mermaid.

```mermaid
flowchart TB
    subgraph SRC["EXTERNAL INPUTS — downloaded per run"]
        direction LR
        SLC["<b>Sentinel-1 SLC</b><br/>IW, single scene<br/><i>.zip → .SAFE, 4–8 GB</i><br/><br/>AUS_COP_HUB → ASF → CDSE<br/>first source that answers"]
        ORB["<b>Orbit file</b><br/>RESORB or POEORB<br/><i>.EOF, XML, ~4 MB</i><br/><br/>AUS_COP_HUB → ASF → CDSE"]
        DEM["<b>DEM</b><br/>cop_glo30 | REMA_2/10/32<br/><i>GeoTIFF / COG</i><br/><br/>AWS Open Data registry<br/>or data.pgc.umn.edu"]
        GEO["<b>Geoid</b><br/>EGM2008<br/><i>GeoTIFF</i><br/><br/>aria-geoid S3"]
    end

    subgraph BAKED["BAKED INTO THE CONTAINER"]
        direction LR
        BDB["<b>OPERA burst database</b><br/><i>SQLite, 53 MB</i><br/>burst_id → bbox + EPSG<br/><br/>dea-public-data-dev S3"]
        TPL["<b>Run-config template</b><br/><i>YAML</i><br/>S1_RTC_IW.yaml<br/>S1_RTC_STATIC_IW.yaml"]
    end

    subgraph S1["STAGE 1 · env: sar-pipeline"]
        GET["<b>isce3-rtc-get-data-for-scene-and-make-run-config</b><br/>downloads the four inputs above<br/>resolves burst ids, picks DEM, chooses output CRS<br/>fills the template"]
        CFG["<b>OPERA-RTC_runconfig.yaml</b><br/><i>YAML, ~12 KB</i><br/>the only handover between environments"]
    end

    subgraph S2["STAGE 2 · env: RTC"]
        RTC["<b>rtc_s1.py</b><br/>GA fork of opera-adt/RTC v1.0.6, on ISCE3<br/><br/>thermal-noise removal · absolute radiometric calibration<br/>bistatic delay · static tropospheric delay<br/>beta0 → gamma0 by <b>area projection</b><br/>geocode to DEM · layover and shadow masking"]
    end

    subgraph S3["STAGE 3 · env: sar-pipeline"]
        META["<b>isce3-rtc-make-metadata-and-upload-bursts</b><br/>STAC + ISO XML + HDF5 · SHA1 · thumbnail<br/>validates STAC before upload"]
    end

    subgraph OUT["OUTPUT — one folder per burst"]
        direction LR
        RAST["<b>Rasters</b><br/><i>COG GeoTIFF, float32</i><br/>VV/VH-gamma0.tif · linear power<br/>mask.tif · uint8<br/>20 m, UTM or EPSG:3031"]
        MD["<b>Metadata</b><br/><i>stac-item.json</i><br/><i>metadata.xml · metadata.h5</i><br/><i>proc-config.yaml</i><br/><i>checksum.sha1 · thumbnail.png</i>"]
    end

    S3B[("<b>AWS S3</b><br/>dea-public-data-dev/baseline/<br/>{product}/{burst_id}/{y}/{m}/{d}/")]

    SLC --> GET
    ORB --> GET
    DEM --> GET
    GEO --> GET
    BDB --> GET
    TPL --> GET
    GET --> CFG
    CFG --> RTC
    SLC -. "read directly<br/>by the processor" .-> RTC
    RTC --> RAST
    RTC ==> META
    RAST -. "read back<br/>for checksums" .-> META
    META --> MD
    MD --> S3B
    RAST --> S3B

    classDef ext fill:#1f4e5f,stroke:#0d2b35,color:#fff
    classDef baked fill:#4a3f6b,stroke:#2a2340,color:#fff
    classDef stage fill:#8a5a30,stroke:#4d3119,color:#fff
    classDef out fill:#2d5f3a,stroke:#173020,color:#fff
    classDef store fill:#5f2d43,stroke:#30161f,color:#fff
    classDef box fill:#f7f7f4,stroke:#b8b8ae,color:#333
    class SRC,BAKED,S1,S2,S3,OUT box
    class SLC,ORB,DEM,GEO ext
    class BDB,TPL baked
    class GET,CFG,RTC,META stage
    class RAST,MD out
    class S3B store
```

## Inputs, by source and format

| Input | Format | Where from | Size |
| --- | --- | --- | --- |
| Sentinel-1 SLC | `.zip` → `.SAFE` | AUS_COP_HUB → ASF → CDSE, in preference order | 4–8 GB |
| Orbit | `.EOF` (XML) | same three | ~4 MB |
| DEM | GeoTIFF / COG | `registry.opendata.aws/copernicus-dem` or `data.pgc.umn.edu` (REMA) | varies |
| Geoid | GeoTIFF | `aria-geoid` S3, EGM2008 | ~1 GB |
| Burst database | SQLite | `dea-public-data-dev` S3, baked into the image | 53 MB |
| Run-config template | YAML | in-repo | 12 KB |

Credentials are needed for the SLC and orbit sources only: Earthdata for ASF,
CDSE for Copernicus, and internal credentials for the Australasian hub. The
DEM, geoid and burst database are all anonymous.

## Why three conda environments

`RTC` and `sar-pipeline` have dependency conflicts, and `pygssearch` (the
Australasian hub client) conflicts with both. The container carries all three
and the shell script switches between them. **The run config is the only thing
passed across** — stage 1 writes a YAML, stage 2 reads it, which is also why
the config ships with the product as `proc-config.yaml`.

## What the processor actually does

From `S1_RTC_IW.yaml`, in order:

1. **Thermal noise removal** — `apply_thermal_noise_correction: True`
2. **Absolute radiometric calibration** — to beta0
3. **Bistatic delay correction** and **static tropospheric delay correction**
4. **Radiometric terrain flattening** — `input_terrain_radiometry: beta0` →
   `output_type: gamma0`, `algorithm_type: area_projection`
5. **Geocoding** — `area_projection` onto the DEM grid, `biquintic` DEM
   interpolation, with layover/shadow and valid-sample sub-swath masking

`area_projection` rather than `bilinear_distribution` is the choice that makes
this CEOS-ARD: it computes the true illuminated area per pixel instead of
approximating it from the cosine of the incidence angle.

## RTC_S1 and RTC_S1_STATIC are the same run, different save flags

|  | `RTC_S1` | `RTC_S1_STATIC` |
| --- | --- | --- |
| `save_mask` | True | False |
| `save_incidence_angle` | False | True |
| `save_local_inc_angle` | False | True |
| `save_nlooks` | False | True |
| `save_rtc_anf` | False | True |

The static layers depend only on viewing geometry and terrain, so they are
produced once per burst id and referenced by every acquisition of that burst —
which is why `fetch_ga_c0.py` fetches them once per burst rather than per date.

## One SLC in, many bursts out

`cli.py:565` passes a single-element list:

```python
RTC_RUN_CONFIG.set(f"{gk}.input_file_group.safe_file_path", [str(SCENE_PATH)])
```

A burst is the atomic unit of IW acquisition — roughly 3 seconds of azimuth
data — and ESA cuts SLC slices *at burst boundaries*, so a burst is always
wholly inside one slice. GRD slices are cut at arbitrary intervals instead,
which is why a burst can span two of those: measured on `t045_095772_iw2`, one
GRD slice covered 91.6% of the burst and its neighbour the remaining 8.4%.
