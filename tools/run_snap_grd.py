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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

OUTPUT_DIRNAME = "grd_preprocessed"
DEFAULT_SPACING = "20.0"
DEFAULT_DEM = "Copernicus 30m Global DEM"

# The GA NRB products are delivered in UTM. Matching the projection at source
# avoids a reproject-and-resample step before any comparison.
AOI_CRS = {
    "hunter": "EPSG:32756",
    "pilliga": "EPSG:32755",
    "bluemtns": "EPSG:32756",
}


class ConfigError(RuntimeError):
    """Raised when the run cannot be set up (bad paths, unknown variant, no CRS)."""


def graphs_dir(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit)
    else:
        path = Path(__file__).resolve().parent.parent / "graphs"
    if not path.is_dir():
        raise ConfigError(f"graphs directory not found: {path}")
    return path


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


def crs_for(scene: Path, override: Optional[str] = None) -> str:
    """UTM CRS for a scene, from the AOI folder name in its path."""
    if override:
        return override
    lowered = {part.lower() for part in scene.parts}
    for aoi, code in AOI_CRS.items():
        if aoi in lowered:
            return code
    raise ConfigError(
        f"cannot infer a CRS for {scene}: no known AOI folder in its path "
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
    scene: Path,
    target: Path,
    crs: str,
    spacing: str,
    dem: str,
    memory: str,
    threads: str,
) -> list[str]:
    return [
        gpt,
        str(graph),
        "-c",
        memory,
        "-q",
        threads,
        f"-Pinput={scene}",
        f"-Poutput={target}",
        f"-Pcrs={crs}",
        f"-Pspacing={spacing}",
        f"-Pdem={dem}",
    ]


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
    parser.add_argument("--gpt", default="gpt", help="path to the SNAP gpt executable")
    parser.add_argument("--crs", help="override the projection, e.g. EPSG:32755")
    parser.add_argument("--spacing", default=DEFAULT_SPACING, help=f"output pixel spacing in metres (default {DEFAULT_SPACING})")
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
    if not planning_only and shutil.which(args.gpt) is None and not Path(args.gpt).exists():
        print(
            f"error: gpt not found ({args.gpt}). Point --gpt at it, e.g.\n"
            r'  --gpt "C:\Program Files\esa-snap\bin\gpt.exe"',
            file=sys.stderr,
        )
        return 2

    jobs = []
    for scene in scenes:
        try:
            crs = crs_for(scene, args.crs)
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
        command = build_command(
            args.gpt, directory / f"{variant}.xml", scene, target,
            crs, args.spacing, args.dem, args.memory, args.threads,
        )
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
