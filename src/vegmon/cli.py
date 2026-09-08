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
        "demo": command_demo,
        "run": command_run,
        "catalogues": command_catalogues,
        "config-template": command_config_template,
    }
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
