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

SUPPLIED = {
    "input": "in.zip",
    "output": "out.dim",
    "crs": "EPSG:32755",
    "spacing": "20.0",
    "dem": "Copernicus 30m Global DEM",
    "oversampling": "2.0",
    "overlap": "0.2",
}

GRAPHS = sorted((Path(__file__).resolve().parent.parent / "graphs").glob("*.xml"))
FULL_EXTENT = "0,0,2147483647,2147483647"
# Every graph reads and writes; the rest depend on which operators it has.
UNIVERSAL_PARAMS = {"input", "output"}
GEOCODING_PARAMS = {"crs", "spacing"}
FLATTENING_PARAMS = {"oversampling", "overlap"}


def nodes(graph: ET.Element) -> dict[str, ET.Element]:
    return {node.get("id"): node for node in graph.findall("node")}


def parameter(node: ET.Element, name: str) -> str | None:
    found = node.find(f"parameters/{name}")
    return None if found is None else (found.text or "")


def test_graphs_exist():
    assert GRAPHS, "no graphs found"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_xml_is_well_formed(path: Path):
    """Notably, '--' is illegal inside an XML comment."""
    ET.parse(path)


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
    declared = runner.placeholders(path)
    assert UNIVERSAL_PARAMS <= declared, f"missing {UNIVERSAL_PARAMS - declared}"
    graph = ET.parse(path).getroot()
    known = nodes(graph)
    assert parameter(known["Read"], "file") == "${input}"
    assert parameter(known["Write"], "file") == "${output}"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_placeholders_match_the_operators_present(path: Path):
    """A graph declares a placeholder if and only if it has the operator using it."""
    declared = runner.placeholders(path)
    known = nodes(ET.parse(path).getroot())
    assert (GEOCODING_PARAMS <= declared) is ("Terrain-Correction" in known)
    assert (FLATTENING_PARAMS <= declared) is ("Terrain-Flattening" in known)
    assert ("dem" in declared) is bool({"Terrain-Correction", "Terrain-Flattening"} & known.keys())


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_runner_can_supply_every_declared_placeholder(path: Path):
    """Guards against adding a ${...} to a graph the runner knows nothing about."""
    runner.build_command("gpt", path, "8G", "8", SUPPLIED)


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
    """Terrain-Correction must ask for bands that actually reach it: either the
    gamma0 that Terrain-Flattening emits, or whatever Calibration was told to
    output when there is no flattening."""
    graph = ET.parse(path).getroot()
    known = nodes(graph)
    correction = known.get("Terrain-Correction")
    if correction is None:
        return
    requested = set(parameter(correction, "sourceBands").split(","))
    if "Terrain-Flattening" in known:
        available = {"Gamma0_VH", "Gamma0_VV"}
    else:
        calibration = known["Calibration"]
        available = {
            f"{name}_{pol}"
            for name, flag in (("Sigma0", "outputSigmaBand"),
                               ("Gamma0", "outputGammaBand"),
                               ("Beta0", "outputBetaBand"))
            if parameter(calibration, flag) == "true"
            for pol in ("VH", "VV")
        }
    assert requested <= available, f"asks for {requested - available}, never produced"
    assert requested, "no bands selected"


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


def test_build_command_supplies_exactly_what_the_graph_declares(tmp_path: Path):
    graph = tmp_path / "g.xml"
    graph.write_text("<graph>${input} ${output} ${dem}</graph>")
    command = runner.build_command("gpt", graph, "8G", "8", SUPPLIED)
    supplied = {item[2:].split("=", 1)[0] for item in command if item.startswith("-P")}
    assert supplied == {"input", "output", "dem"}, "passed a -P the graph never uses"


def test_build_command_rejects_a_placeholder_it_cannot_fill(tmp_path: Path):
    graph = tmp_path / "g.xml"
    graph.write_text("<graph>${input} ${output} ${wavelength}</graph>")
    with pytest.raises(runner.ConfigError, match="wavelength"):
        runner.build_command("gpt", graph, "8G", "8", SUPPLIED)


def test_quote_for_cmd_quotes_values_with_spaces(tmp_path: Path):
    graph = tmp_path / "g.xml"
    graph.write_text("<graph>${input} ${output} ${dem}</graph>")
    rendered = runner.quote_for_cmd(runner.build_command("gpt", graph, "8G", "8", SUPPLIED))
    assert '-Pdem="Copernicus 30m Global DEM"' in rendered


def test_unknown_variant_is_rejected():
    directory = Path(__file__).resolve().parent.parent / "graphs"
    with pytest.raises(runner.ConfigError):
        runner.resolve_variants(["not_a_graph"], directory)


# --- the terrain-flattening fixes -------------------------------------------


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_orbit_failure_is_never_silent(path: Path):
    """continueOnFail lets SNAP fall back to the predicted orbit, which
    misregisters the terrain-flattening simulated image and looks like a
    geometry fault in the final product."""
    graph = ET.parse(path).getroot()
    assert parameter(nodes(graph)["Apply-Orbit-File"], "continueOnFail") == "false"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_terrain_flattening_never_receives_gamma0_or_sigma0(path: Path):
    graph = ET.parse(path).getroot()
    known = nodes(graph)
    if "Terrain-Flattening" not in known:
        return
    bands = parameter(known["Terrain-Flattening"], "sourceBands")
    assert "Gamma0" not in bands and "Sigma0" not in bands
    calibration = known["Calibration"]
    assert parameter(calibration, "outputGammaBand") == "false"
    assert parameter(calibration, "outputSigmaBand") == "false"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_oversampling_is_tunable_not_hardcoded(path: Path):
    graph = ET.parse(path).getroot()
    flattening = nodes(graph).get("Terrain-Flattening")
    if flattening is None:
        return
    assert parameter(flattening, "oversamplingMultiple") == "${oversampling}"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_border_limit_is_the_default_not_5000(path: Path):
    graph = ET.parse(path).getroot()
    assert parameter(nodes(graph)["Remove-GRD-Border-Noise"], "borderLimit") == "500"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_inland_dems_reading_zero_are_not_masked_as_sea(path: Path):
    graph = ET.parse(path).getroot()
    for node_id in ("Terrain-Flattening", "Terrain-Correction"):
        node = nodes(graph).get(node_id)
        if node is not None:
            assert parameter(node, "nodataValueAtSea") == "false"


def test_a_no_flattening_variant_exists_for_diagnosis():
    """Removing terrain flattening has to be a one-flag experiment, not an edit."""
    assert (Path(__file__).resolve().parent.parent / "graphs" / "grd_gamma0_ellipsoid.xml").exists()


def test_diagnostic_graph_stops_before_geocoding_and_emits_the_simulation():
    path = Path(__file__).resolve().parent.parent / "graphs" / "grd_tf_diagnostic.xml"
    known = nodes(ET.parse(path).getroot())
    assert "Terrain-Correction" not in known, "the diagnostic must stay in radar geometry"
    assert parameter(known["Terrain-Flattening"], "outputSimulatedImage") == "true"
    assert known["Write"].find("sources/sourceProduct").get("refid") == "Terrain-Flattening"
