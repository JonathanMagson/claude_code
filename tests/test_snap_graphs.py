"""Structural checks on the SNAP graphs and the gpt batch runner.

SNAP itself is not installed here, so these assert what can be asserted without
it: the XML is well formed, every node's source resolves, the templating gpt
needs is present, and nothing scene-specific has been left baked in.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import run_snap_grd as runner  # noqa: E402

GRAPHS = sorted((Path(__file__).resolve().parent.parent / "graphs").glob("*.xml"))
FULL_EXTENT = "0,0,2147483647,2147483647"
REQUIRED_PARAMS = {"input", "output", "crs", "spacing", "dem"}


def nodes(graph: ET.Element) -> dict[str, ET.Element]:
    return {node.get("id"): node for node in graph.findall("node")}


def parameter(node: ET.Element, name: str) -> str | None:
    found = node.find(f"parameters/{name}")
    return None if found is None else (found.text or "")


def test_graphs_exist():
    assert GRAPHS, "no graphs found"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_every_node_source_resolves(path: Path):
    graph = ET.parse(path).getroot()
    known = nodes(graph)
    for node_id, node in known.items():
        if node_id == "Read":
            continue
        refs = node.findall("sources/sourceProduct")
        assert refs, f"{node_id} has no source -- the graph is disconnected"
        for ref in refs:
            assert ref.get("refid") in known, f"{node_id} points at unknown node {ref.get('refid')}"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_write_is_connected(path: Path):
    """The draft graph shipped with an empty <sources/> on Write. Never again."""
    graph = ET.parse(path).getroot()
    write = nodes(graph)["Write"]
    assert write.findall("sources/sourceProduct"), "Write is disconnected"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_templated_for_gpt(path: Path):
    text = path.read_text(encoding="utf-8")
    placeholders = set(re.findall(r"\$\{(\w+)\}", text))
    assert REQUIRED_PARAMS <= placeholders, f"missing {REQUIRED_PARAMS - placeholders}"
    graph = ET.parse(path).getroot()
    known = nodes(graph)
    assert parameter(known["Read"], "file") == "${input}"
    assert parameter(known["Write"], "file") == "${output}"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_no_scene_specific_pixel_region(path: Path):
    """A hardcoded pixelRegion crops or breaks every scene of a different size."""
    graph = ET.parse(path).getroot()
    assert parameter(nodes(graph)["Read"], "pixelRegion") == FULL_EXTENT


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_no_hardcoded_drive_letters(path: Path):
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z]:\\\\?[\w\\]", text), "a local path is baked into the graph"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_output_is_linear(path: Path):
    """dB belongs in the comparison step, not the graph -- see postprocess_nrb.py."""
    graph = ET.parse(path).getroot()
    assert "LinearToFromdB" not in nodes(graph)


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_terrain_flattening_is_fed_beta0(path: Path):
    """Terrain-Flattening needs beta0; the draft fed it a sigma0-only product."""
    graph = ET.parse(path).getroot()
    known = nodes(graph)
    flattening = known.get("Terrain-Flattening")
    if flattening is None:
        return
    assert parameter(flattening, "sourceBands") == "Beta0_VH,Beta0_VV"
    calibration = known["Calibration"]
    assert parameter(calibration, "outputBetaBand") == "true"
    assert parameter(calibration, "outputSigmaBand") == "false"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_terrain_correction_bands_are_produced_upstream(path: Path):
    graph = ET.parse(path).getroot()
    known = nodes(graph)
    requested = set(parameter(known["Terrain-Correction"], "sourceBands").split(","))
    convention = "Gamma0" if "Terrain-Flattening" in known else "Sigma0"
    assert requested == {f"{convention}_VH", f"{convention}_VV"}


def test_speckle_filter_runs_before_terrain_flattening():
    """Speckle statistics are only valid before geocoding correlates neighbours."""
    for path in GRAPHS:
        graph = ET.parse(path).getroot()
        known = nodes(graph)
        if "Speckle-Filter" not in known:
            continue
        assert known["Speckle-Filter"].find("sources/sourceProduct").get("refid") == "Calibration"


# --- runner -----------------------------------------------------------------


def test_output_path_nests_by_variant(tmp_path: Path):
    scene = tmp_path / "pilliga" / "t009_019142_iw1" / "20240603" / "S1A_IW_GRDH_x.zip"
    target = runner.output_path(scene, "grd_gamma0_rtc")
    assert target == scene.parent / "grd_preprocessed" / "grd_gamma0_rtc" / "S1A_IW_GRDH_x.dim"


def test_variants_do_not_collide(tmp_path: Path):
    scene = tmp_path / "S1A_IW_GRDH_x.zip"
    first = runner.output_path(scene, "grd_gamma0_rtc")
    second = runner.output_path(scene, "grd_gamma0_rtc_reflee")
    assert first != second and first.name == second.name


def test_crs_inferred_from_aoi_folder(tmp_path: Path):
    scene = tmp_path / "before_after" / "pilliga" / "t009" / "S1A_IW_GRDH_x.zip"
    assert runner.crs_for(scene) == "EPSG:32755"
    hunter = tmp_path / "before_after" / "hunter" / "t009" / "S1A_IW_GRDH_x.zip"
    assert runner.crs_for(hunter) == "EPSG:32756"


def test_crs_override_wins(tmp_path: Path):
    scene = tmp_path / "pilliga" / "S1A_IW_GRDH_x.zip"
    assert runner.crs_for(scene, "EPSG:4326") == "EPSG:4326"


def test_unknown_aoi_is_an_error_not_a_guess(tmp_path: Path):
    scene = tmp_path / "somewhere" / "S1A_IW_GRDH_x.zip"
    with pytest.raises(runner.ConfigError):
        runner.crs_for(scene)


def test_find_grd_zips_skips_its_own_outputs(tmp_path: Path):
    scene = tmp_path / "pilliga" / "S1A_IW_GRDH_x.zip"
    scene.parent.mkdir(parents=True)
    scene.touch()
    stray = tmp_path / "pilliga" / "grd_preprocessed" / "v1" / "S1A_IW_GRDH_y.zip"
    stray.parent.mkdir(parents=True)
    stray.touch()
    assert runner.find_grd_zips(tmp_path) == [scene]


def test_find_grd_zips_ignores_slc(tmp_path: Path):
    grd = tmp_path / "S1A_IW_GRDH_1SDV_x.zip"
    grd.touch()
    (tmp_path / "S1A_IW_SLC__1SDV_x.zip").touch()
    assert runner.find_grd_zips(tmp_path) == [grd]


def test_incomplete_dimap_is_not_treated_as_done(tmp_path: Path):
    """A .dim with no .data/ is a crashed run, not a finished one."""
    target = tmp_path / "S1A.dim"
    target.touch()
    assert not runner.is_complete(target)
    target.with_suffix(".data").mkdir()
    assert runner.is_complete(target)


def test_build_command_supplies_every_placeholder(tmp_path: Path):
    command = runner.build_command(
        "gpt", tmp_path / "g.xml", tmp_path / "s.zip", tmp_path / "o.dim",
        "EPSG:32755", "20.0", "Copernicus 30m Global DEM", "8G", "8",
    )
    supplied = {item[2:].split("=", 1)[0] for item in command if item.startswith("-P")}
    assert supplied == REQUIRED_PARAMS


def test_quote_for_cmd_quotes_values_with_spaces(tmp_path: Path):
    command = runner.build_command(
        "gpt", tmp_path / "g.xml", tmp_path / "s.zip", tmp_path / "o.dim",
        "EPSG:32755", "20.0", "Copernicus 30m Global DEM", "8G", "8",
    )
    rendered = runner.quote_for_cmd(command)
    assert '-Pdem="Copernicus 30m Global DEM"' in rendered


def test_unknown_variant_is_rejected():
    directory = Path(__file__).resolve().parent.parent / "graphs"
    with pytest.raises(runner.ConfigError):
        runner.resolve_variants(["not_a_graph"], directory)
