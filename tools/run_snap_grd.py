#!/usr/bin/env python3
"""Batch-run SNAP ``gpt`` pre-processing graphs over downloaded Sentinel-1 GRD zips.

Outputs land one level below each GRD, in a per-variant sub-folder, so several
pre-processing methods can sit side by side for comparison::

    .../pilliga/t009_019142_iw1/20240603/
        S1A_IW_GRDH_..._D146.zip
        grd_preprocessed/
            grd_gamma0_rtc/          S1A_IW_GRDH_..._D146.dim (+ .data/)
            grd_gamma0_rtc_reflee/   S1A_IW_GRDH_..._D146.dim (+ .data/)
            ...

The graphs in ``graphs/`` are templated: every one needs ``input``, ``output``,
``crs``, ``spacing`` and ``dem`` supplied via ``-P``. This script supplies all
five. ``gpt`` does not default ``${...}`` placeholders -- a missing one is an
error, not a silent skip.

Examples
--------
List what would run, without running it::

    python tools/run_snap_grd.py D:\\scratch\\sentinel_1_data_project\\before_after --dry-run

Run one variant over everything::

    python tools/run_snap_grd.py D:\\scratch\\...\\before_after --variant grd_gamma0_rtc

Emit cmd.exe one-liners instead of running them::

    python tools/run_snap_grd.py D:\\scratch\\...\\before_after --print-commands
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

OUTPUT_DIRNAME = "grd_preprocessed"
DEFAULT_SPACING = "20.0"
DEFAULT_DEM = "Copernicus 30m Global DEM"
# Terrain-Flattening simulates the illuminated area from the DEM. At the SNAP
# default of 1.0 that simulation is undersampled relative to the SAR grid, and
# high relief comes out holed, striped or blocky. 2.0 costs runtime, not quality.
DEFAULT_OVERSAMPLING = "2.0"
DEFAULT_OVERLAP = "0.2"

# Fallback only, used when no GA raster can be read beside the scene. GA picks
# the projection per burst, which is not always the zone the AOI centre sits in:
# the Pilliga centre is zone 55, but GA delivers that burst in zone 56.
AOI_CRS = {
    "hunter": "EPSG:32756",
    "pilliga": "EPSG:32756",
    "bluemtns": "EPSG:32756",
}


# SNAP installs wherever it can. Without admin rights its Windows installer
# falls back under LOCALAPPDATA, which is not on PATH and not in Program Files.
GPT_CANDIDATES = (
    "{localappdata}/snap/bin/gpt.exe",
    "{localappdata}/esa-snap/bin/gpt.exe",
    "{localappdata}/Programs/snap/bin/gpt.exe",
    "{localappdata}/Programs/esa-snap/bin/gpt.exe",
    "C:/Program Files/snap/bin/gpt.exe",
    "C:/Program Files/esa-snap/bin/gpt.exe",
    "{home}/snap/bin/gpt",
    "{home}/esa-snap/bin/gpt",
    "/opt/snap/bin/gpt",
    "/opt/esa-snap/bin/gpt",
    "/usr/local/snap/bin/gpt",
)


class ConfigError(RuntimeError):
    """Raised when the run cannot be set up (bad paths, unknown variant, no CRS)."""


def locate_gpt(explicit: Optional[str] = None) -> str:
    """Resolve the SNAP gpt executable, or explain where it was looked for."""
    if explicit:
        if Path(explicit).exists() or shutil.which(explicit):
            return explicit
        raise ConfigError(f"gpt not found at {explicit}")
    found = shutil.which("gpt")
    if found:
        return found
    localappdata = os.environ.get("LOCALAPPDATA", "")
    home = str(Path.home())
    searched = []
    for template in GPT_CANDIDATES:
        candidate = Path(template.format(localappdata=localappdata, home=home))
        searched.append(str(candidate))
        if candidate.exists():
            return str(candidate)
    raise ConfigError(
        "SNAP's gpt was not found on PATH or in any usual install location.\n"
        "Pass it explicitly with --gpt. Looked in:\n  " + "\n  ".join(searched)
    )


def graphs_dir(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit)
    else:
        path = Path(__file__).resolve().parent.parent / "graphs"
    if not path.is_dir():
        raise ConfigError(f"graphs directory not found: {path}")
    return path


def placeholders(graph: Path) -> set[str]:
    """The ${...} names a graph actually uses.

    Graphs differ: the diagnostic stops before terrain correction and so has no
    crs or spacing, while the ellipsoid variants have no terrain flattening and
    so no oversampling. Passing a -P a graph does not use is noise at best, so
    supply exactly what each one asks for.
    """
    return set(re.findall(r"\$\{(\w+)\}", graph.read_text(encoding="utf-8")))


def available_variants(directory: Path) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.xml"))


def resolve_variants(requested: Sequence[str], directory: Path) -> list[str]:
    known = available_variants(directory)
    if not known:
        raise ConfigError(f"no .xml graphs in {directory}")
    if not requested:
        return known
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ConfigError(
            f"unknown variant(s): {', '.join(unknown)}. Available: {', '.join(known)}"
        )
    return list(requested)


def find_grd_zips(root: Path) -> list[Path]:
    """Every Sentinel-1 GRD zip under *root*, excluding anything already produced."""
    if not root.exists():
        raise ConfigError(f"root does not exist: {root}")
    scenes = [
        path
        for path in sorted(root.rglob("S1*_GRD*.zip"))
        if OUTPUT_DIRNAME not in path.parts
    ]
    return scenes


def crs_from_reference(scene: Path) -> Optional[str]:
    """CRS of a GA NRB raster sitting beside *scene*, if one is readable.

    Matching GA's projection at source is what makes the products directly
    comparable: otherwise the comparison has to reproject, adding a resampling
    step between two things that are supposed to be measured against each other.

    Needs rasterio. This module is otherwise stdlib-only and still works without
    it, so the import is deliberately local and optional.
    """
    candidates = sorted(scene.parent.glob("ga_*gamma0.tif"))
    if not candidates:
        return None
    try:
        import rasterio  # noqa: PLC0415
    except ImportError:
        return None
    try:
        with rasterio.open(candidates[0]) as reference:
            code = reference.crs.to_string() if reference.crs else None
    except Exception:
        return None
    return code


def crs_for(scene: Path, override: Optional[str] = None) -> str:
    """Projection for a scene: explicit, else GA's own, else the AOI fallback.

    The fallback is a guess from the AOI's centre longitude and has been wrong
    before -- GA's Pilliga burst is zone 56 although the AOI centre sits in
    zone 55 -- so the GA raster wins whenever one can be read.
    """
    if override:
        return override
    from_reference = crs_from_reference(scene)
    if from_reference:
        return from_reference
    lowered = {part.lower() for part in scene.parts}
    for aoi, code in AOI_CRS.items():
        if aoi in lowered:
            return code
    raise ConfigError(
        f"cannot infer a CRS for {scene}: no GA raster beside it to take one "
        f"from, and no known AOI folder in its path "
        f"({', '.join(sorted(AOI_CRS))}). Pass --crs explicitly."
    )


def output_path(scene: Path, variant: str) -> Path:
    return scene.parent / OUTPUT_DIRNAME / variant / (scene.stem + ".dim")


def is_complete(target: Path) -> bool:
    """A BEAM-DIMAP product is only usable if both the header and .data/ exist."""
    return target.is_file() and target.with_suffix(".data").is_dir()


def build_command(
    gpt: str,
    graph: Path,
    memory: str,
    threads: str,
    values: dict,
) -> list[str]:
    """gpt invocation supplying exactly the placeholders *graph* declares."""
    needed = placeholders(graph)
    missing = needed - values.keys()
    if missing:
        raise ConfigError(
            f"{graph.name} needs {', '.join(sorted(missing))}, which the runner "
            "cannot supply. Add it to the runner or remove it from the graph."
        )
    command = [gpt, str(graph), "-c", memory, "-q", threads]
    command += [f"-P{name}={values[name]}" for name in sorted(needed)]
    return command


def quote_for_cmd(command: Iterable[str]) -> str:
    parts = []
    for item in command:
        if item.startswith("-P") and "=" in item:
            flag, value = item.split("=", 1)
            parts.append(f'{flag}="{value}"')
        elif " " in item:
            parts.append(f'"{item}"')
        else:
            parts.append(item)
    return " ".join(parts)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="folder to search recursively for S1 GRD .zip files")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME",
        help="graph to run (repeatable). Default: every graph in graphs/",
    )
    parser.add_argument("--graphs", help="directory holding the graph XML (default: <repo>/graphs)")
    parser.add_argument("--gpt", help="path to the SNAP gpt executable "
                                     "(default: PATH, then the usual install locations)")
    parser.add_argument("--crs", help="override the projection, e.g. EPSG:32755")
    parser.add_argument("--spacing", default=DEFAULT_SPACING, help=f"output pixel spacing in metres (default {DEFAULT_SPACING})")
    parser.add_argument("--oversampling", default=DEFAULT_OVERSAMPLING,
                        help=f"Terrain-Flattening DEM oversampling (default {DEFAULT_OVERSAMPLING}; "
                             "raise for steep terrain, 1.0 is the SNAP default)")
    parser.add_argument("--overlap", default=DEFAULT_OVERLAP,
                        help=f"Terrain-Flattening additional overlap (default {DEFAULT_OVERLAP})")
    parser.add_argument("--dem", default=DEFAULT_DEM, help=f"SNAP DEM name (default: {DEFAULT_DEM!r})")
    parser.add_argument("--memory", default="8G", help="gpt tile cache, -c (default 8G)")
    parser.add_argument("--threads", default="8", help="gpt parallelism, -q (default 8)")
    parser.add_argument("--dry-run", action="store_true", help="list the jobs, run nothing")
    parser.add_argument("--print-commands", action="store_true", help="emit cmd.exe one-liners, run nothing")
    parser.add_argument("--overwrite", action="store_true", help="re-run jobs whose output already exists")
    parser.add_argument("--list-variants", action="store_true", help="show the available graphs and exit")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        directory = graphs_dir(args.graphs)
        if args.list_variants:
            for name in available_variants(directory):
                print(name)
            return 0
        variants = resolve_variants(args.variant, directory)
        scenes = find_grd_zips(Path(args.root))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not scenes:
        print(f"no GRD zips found under {args.root}")
        return 1

    planning_only = args.dry_run or args.print_commands
    try:
        gpt = locate_gpt(args.gpt)
    except ConfigError as exc:
        if not planning_only:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        gpt = args.gpt or "gpt"  # planning only, so a real gpt is not needed yet
    else:
        if not args.gpt:
            print(f"using gpt: {gpt}")

    # The diagnostic graph stops before terrain correction, so it needs no CRS.
    # Don't make an unrecognised AOI folder an error for a run that never geocodes.
    needs_crs = any("crs" in placeholders(directory / f"{name}.xml") for name in variants)

    jobs = []
    for scene in scenes:
        try:
            crs = crs_for(scene, args.crs) if needs_crs else ""
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for variant in variants:
            target = output_path(scene, variant)
            if is_complete(target) and not args.overwrite:
                continue
            jobs.append((scene, variant, target, crs))

    print(f"{len(scenes)} scene(s) x {len(variants)} variant(s) -> {len(jobs)} job(s) to run")
    if not jobs:
        print("nothing to do (all outputs present; use --overwrite to force)")
        return 0

    failures = 0
    for index, (scene, variant, target, crs) in enumerate(jobs, start=1):
        try:
            command = build_command(
                gpt,
                directory / f"{variant}.xml",
                args.memory,
                args.threads,
                {
                    "input": scene,
                    "output": target,
                    "crs": crs,
                    "spacing": args.spacing,
                    "dem": args.dem,
                    "oversampling": args.oversampling,
                    "overlap": args.overlap,
                },
            )
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.print_commands:
            print(quote_for_cmd(command))
            continue
        print(f"[{index}/{len(jobs)}] {variant}  {scene.name}  -> {target}")
        if args.dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        result = subprocess.run(command)
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            failures += 1
            print(f"    FAILED (exit {result.returncode}) after {elapsed / 60:.1f} min", file=sys.stderr)
        else:
            print(f"    done in {elapsed / 60:.1f} min")

    if failures:
        print(f"{failures} of {len(jobs)} job(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
