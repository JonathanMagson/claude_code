"""Command line entry point.

    vegmon demo                 run the whole thing on synthetic data, no network
    vegmon run --config c.json  run against live Sentinel-1/2 via STAC
    vegmon catalogues           what data sources are available
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from vegmon import __version__


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--outputs", type=Path, default=Path("outputs"), help="directory for outputs"
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="skip the PNG figures and HTML report"
    )
    parser.add_argument(
        "--tiers",
        default="confirmed",
        help="comma-separated tiers to carry into the regrowth analysis",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vegmon",
        description=(
            "Land clearing detection from Sentinel-1 and Sentinel-2, with "
            "post-clearing regrowth monitoring."
        ),
    )
    parser.add_argument("--version", action="version", version=f"vegmon {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo",
        help="run end to end on the bundled synthetic datacube (no network needed)",
    )
    demo.add_argument("--size", type=int, default=320, help="raster side length in pixels")
    demo.add_argument("--seed", type=int, default=20240917, help="random seed")
    demo.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="npz path to cache the generated scene in and reuse next run",
    )
    _add_common(demo)

    run = subparsers.add_parser("run", help="run against live Sentinel-1/2 through STAC")
    run.add_argument("--config", type=Path, required=True, help="pipeline config JSON")
    run.add_argument("--optical-catalogue", default="earth-search")
    run.add_argument("--radar-catalogue", default="planetary-computer")
    run.add_argument(
        "--orbit",
        default=None,
        choices=["ascending", "descending"],
        help="pin Sentinel-1 to one orbit direction (recommended)",
    )
    _add_common(run)

    aws = subparsers.add_parser(
        "aws",
        help=(
            "run against Sentinel-2 read straight from the AWS open-data bucket, "
            "with no STAC API in the middle"
        ),
    )
    aws.add_argument("--tile", required=True, help="MGRS tile id, e.g. 55JGG")
    aws.add_argument("--lon", type=float, required=True, help="AOI centre longitude")
    aws.add_argument("--lat", type=float, required=True, help="AOI centre latitude")
    aws.add_argument("--name", default=None, help="label for the AOI in the report")
    aws.add_argument("--size", type=int, default=512, help="AOI side length in pixels")
    aws.add_argument("--resolution", type=float, default=20.0, help="working pixel size, m")
    aws.add_argument("--pre", nargs=2, metavar=("START", "END"), required=True)
    aws.add_argument("--post", nargs=2, metavar=("START", "END"), required=True)
    aws.add_argument("--series", nargs=2, metavar=("START", "END"), required=True)
    aws.add_argument("--workers", type=int, default=12, help="parallel COG reads")
    aws.add_argument(
        "--adaptive-woody",
        type=float,
        default=0.85,
        help=(
            "quantile of the scene's own persistent-NDVI distribution to use as the "
            "woody gate; 0 disables and falls back to the fixed thresholds"
        ),
    )
    aws.add_argument(
        "--adaptive-mad",
        type=float,
        default=3.0,
        help="MAD multiplier for the adaptive change threshold; 0 disables",
    )
    aws.add_argument(
        "--max-amplitude",
        type=float,
        default=0.35,
        help=(
            "largest within-year NDVI swing before the event that still counts as "
            "woody; rejects cropping, irrigated cropping above all. 0 disables"
        ),
    )
    aws.add_argument("--max-cloud", type=float, default=0.45,
                     help="AOI cloud fraction above which an acquisition is dropped")
    aws.add_argument("--cache", type=Path, default=None,
                     help="npz path to cache the loaded series in")
    aws.add_argument(
        "--epochs",
        action="store_true",
        help=(
            "also scan every consecutive year pair across the record and write "
            "epochs.csv - the annual monitoring table"
        ),
    )
    _add_common(aws)

    subparsers.add_parser("catalogues", help="list the available STAC catalogues")

    template = subparsers.add_parser(
        "config-template", help="write a starter config JSON for your own AOI"
    )
    template.add_argument("path", type=Path)

    return parser


def _write_outputs(result, args, scene) -> dict:
    from vegmon.report import build_report
    from vegmon.viz import plot_change_maps, plot_recovery

    outputs = Path(args.outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    written: dict = {}

    summary_path = outputs / "summary.json"
    summary_path.write_text(json.dumps(result.summary(), indent=2, default=str))
    written["summary"] = summary_path

    try:
        gpkg = result.clearing.write_vector(outputs / "clearing.gpkg")
        written["vector"] = gpkg
    except Exception as exc:  # geopandas/fiona are optional
        if not args.quiet:
            print(f"  (skipped vector output: {exc})", file=sys.stderr)

    patches_path = outputs / "regrowth.csv"
    _write_csv(patches_path, result.regrowth.records())
    written["regrowth"] = patches_path

    if args.no_figures:
        return written

    truth = (scene.truth or {}).get("synthetic")
    figures = {
        "change_maps": plot_change_maps(
            scene, result.optical, result.radar, result.clearing,
            outputs / "change_maps.png", truth=truth,
        )
    }
    recovery = plot_recovery(result.regrowth, outputs / "recovery.png")
    if recovery:
        figures["recovery"] = recovery
    written.update(figures)
    written["report"] = build_report(result, outputs / "report.html", figures=figures)
    return written


def _write_csv(path: Path, records: Sequence[dict]) -> None:
    import csv

    if not records:
        path.write_text("")
        return
    fields: list = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _report(result, quiet: bool) -> None:
    if quiet:
        return
    tiers = result.clearing.summary()
    print("\nDetection")
    for name, entry in tiers.items():
        print(f"  {name:<10} {entry['patches']:>3} patches  {entry['area_ha']:>8.2f} ha")
    print("\nRegrowth")
    for name, entry in result.regrowth.summary().items():
        print(f"  {name:<18} {entry['patches']:>3} patches  {entry['area_ha']:>8.2f} ha")
    if result.validation is not None:
        print("\nValidation")
        for line in result.validation.headline().splitlines():
            print(f"  {line}")
        latency = result.validation.latency or {}
        if latency.get("advantage_days") is not None:
            print(
                f"  Sentinel-1 raised the alert a median "
                f"{latency['advantage_days']:.0f} days before Sentinel-2"
            )


def command_demo(args) -> int:
    from vegmon.config import demo_config
    from vegmon.pipeline import run_pipeline
    from vegmon.synthetic import generate_scene, load_scene, save_scene

    config = demo_config()
    cache = args.cache
    if cache and Path(cache).exists():
        if not args.quiet:
            print(f"Loading cached scene from {cache}")
        scene = load_scene(cache)
    else:
        if not args.quiet:
            print(
                f"Generating a synthetic Sentinel-1/2 datacube "
                f"({args.size}x{args.size} px at {config.resolution:.0f} m, "
                f"{config.periods.series_start} to {config.periods.series_end})..."
            )
        scene = generate_scene(config, size=args.size, seed=args.seed, progress=not args.quiet)
        if cache:
            save_scene(scene, cache)

    if not args.quiet:
        print(
            f"Scene: {len(scene.optical)} Sentinel-2 and {len(scene.radar)} "
            f"Sentinel-1 acquisitions"
        )
    result = run_pipeline(
        scene,
        config,
        regrowth_tiers=tuple(t.strip() for t in args.tiers.split(",") if t.strip()),
        verbose=not args.quiet,
    )
    written = _write_outputs(result, args, scene)
    _report(result, args.quiet)
    if not args.quiet:
        print("\nWrote:")
        for name, path in written.items():
            print(f"  {name:<12} {path}")
    return 0


def command_run(args) -> int:
    from vegmon.config import PipelineConfig
    from vegmon.pipeline import run_pipeline
    from vegmon.stac import CatalogueUnavailable, load_scene

    config = PipelineConfig.from_json(args.config)
    config.periods.check_phenology()
    if not args.quiet:
        print(f"Loading {config.aoi.name} from STAC...")
    try:
        scene = load_scene(
            config,
            optical_catalogue=args.optical_catalogue,
            radar_catalogue=args.radar_catalogue,
            orbit_state=args.orbit,
        )
    except CatalogueUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("\nTry 'vegmon demo' - it needs no network at all.", file=sys.stderr)
        return 2

    result = run_pipeline(
        scene,
        config,
        validate_against_truth=False,
        regrowth_tiers=tuple(t.strip() for t in args.tiers.split(",") if t.strip()),
        verbose=not args.quiet,
    )
    written = _write_outputs(result, args, scene)
    _report(result, args.quiet)
    if not args.quiet:
        print("\nWrote:")
        for name, path in written.items():
            print(f"  {name:<12} {path}")
    return 0


def command_aws(args) -> int:
    from dataclasses import replace

    from vegmon.config import AOI, DetectionConfig, Periods, PipelineConfig
    from vegmon.pipeline import run_pipeline
    from vegmon.s3direct import (
        SceneListingError,
        cloud_free_days,
        grid_for_aoi,
        load_sentinel2,
    )
    from vegmon.series import Scene

    grid = grid_for_aoi(args.tile, (args.lon, args.lat), args.size, args.resolution)
    west, south, east, north = _bounds_lonlat(grid)
    aoi = AOI(
        name=args.name or f"{args.tile} @ {args.lat:.4f},{args.lon:.4f}",
        west=west, south=south, east=east, north=north, crs=grid.crs,
    )
    periods = Periods(*args.pre, *args.post, *args.series)
    periods.check_phenology()

    detection = replace(
        DetectionConfig(),
        max_cloud_cover=args.max_cloud,
        adaptive_woody_quantile=args.adaptive_woody or None,
        adaptive_change_mad=args.adaptive_mad or None,
        max_pre_seasonal_amplitude=args.max_amplitude or None,
    )
    config = PipelineConfig(
        aoi=aoi, periods=periods, detection=detection, resolution=args.resolution
    )

    if not args.quiet:
        area = grid.width * grid.height * grid.pixel_area_ha
        print(
            f"AOI {aoi.name}: {args.size}x{args.size} px at {args.resolution:.0f} m "
            f"= {area:,.0f} ha in {grid.crs}"
        )
        print(f"Reading Sentinel-2 from the AWS open-data bucket (no STAC API)...")

    cache = args.cache
    if cache and Path(cache).exists():
        from vegmon.synthetic import load_scene as load_cached

        if not args.quiet:
            print(f"  loading cached series from {cache}")
        cached = load_cached(cache)
        # The cache round-trips an empty radar series; restore it as absent so
        # the pipeline falls back to persistence rather than "confirming"
        # against a stack with nothing in it.
        radar = cached.radar if cached.radar is not None and len(cached.radar) else None
        scene = Scene(optical=cached.optical, radar=radar, grid=cached.grid)
    else:
        try:
            optical = load_sentinel2(
                args.tile, grid, periods.series_start, periods.series_end,
                detection, workers=args.workers, progress=not args.quiet,
            )
        except SceneListingError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        scene = Scene(optical=optical, radar=None, grid=grid)
        if cache:
            _cache_optical(scene, cache)

    if not args.quiet:
        print(
            f"  {len(scene.optical)} usable acquisitions, "
            f"{cloud_free_days(scene.optical, 0.2)} of them under 20% cloud over the AOI"
        )
        print("  no analysis-ready Sentinel-1 is reachable for Australia from the free")
        print("  catalogues, so temporal persistence supplies the confirming evidence.")

    result = run_pipeline(
        scene,
        config,
        validate_against_truth=False,
        regrowth_tiers=tuple(t.strip() for t in args.tiers.split(",") if t.strip()),
        verbose=not args.quiet,
    )
    written = _write_outputs(result, args, scene)

    if args.epochs:
        from vegmon.pipeline import scan_epochs

        if not args.quiet:
            print("\nScanning every consecutive year pair...")
        rows = scan_epochs(
            scene, config, periods.series_start.year, periods.series_end.year
        )
        path = Path(args.outputs) / "epochs.csv"
        _write_csv(path, rows)
        written["epochs"] = path
        if not args.quiet:
            print(f"  {'epoch':<11}{'woody':>7}{'thr':>7}{'patches':>9}{'conf ha':>9}"
                  f"{'review ha':>11}{'largest':>9}{'median':>8}")
            for row in rows:
                if "error" in row:
                    print(f"  {row['epoch']:<11} {row['error']}")
                    continue
                print(
                    f"  {row['epoch']:<11}{row['woody_fraction']:>6.1%}"
                    f"{row['dnbr_threshold']:>7.3f}{row['confirmed_patches']:>9}"
                    f"{row['confirmed_ha']:>9.1f}{row['review_ha']:>11.1f}"
                    f"{row['largest_patch_ha']:>9.1f}{row['median_patch_ha']:>8.1f}"
                )

    _report(result, args.quiet)
    if not args.quiet:
        print("\nThresholds actually applied:")
        for key, value in sorted(result.optical.thresholds.items()):
            print(f"  {key:<28} {value:.3f}")
        print("\nWrote:")
        for name, path in written.items():
            print(f"  {name:<12} {path}")
    return 0


def _bounds_lonlat(grid):
    from pyproj import Transformer

    transformer = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    west, south, east, north = grid.bounds
    lons, lats = transformer.transform([west, east, west, east], [south, south, north, north])
    return min(lons), min(lats), max(lons), max(lats)


def _cache_optical(scene, path) -> None:
    """Cache an optical-only scene, reusing the synthetic scene serialiser."""
    import numpy as np

    from vegmon.grid import Grid
    from vegmon.series import RadarSeries
    from vegmon.synthetic import save_scene
    from vegmon.series import Scene

    grid = scene.grid
    empty = RadarSeries(
        grid=grid, times=[], vv_db=np.zeros((0,) + grid.shape, np.float32),
        vh_db=np.zeros((0,) + grid.shape, np.float32),
    )
    save_scene(Scene(optical=scene.optical, radar=empty, grid=grid), path)


def command_catalogues(args) -> int:
    from vegmon.stac import catalogue_report

    print(catalogue_report())
    return 0


def command_config_template(args) -> int:
    from vegmon.config import demo_config

    path = demo_config().to_json(args.path)
    print(f"Wrote a starter configuration to {path}")
    print("Edit the aoi bounds, crs and periods, then: vegmon run --config", path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "aws": command_aws,
        "demo": command_demo,
        "run": command_run,
        "catalogues": command_catalogues,
        "config-template": command_config_template,
    }
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
