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
# Graphs that deliberately hold SNAP's own defaults, to test whether the tuned
# values were what broke terrain flattening. They are not meant to be tunable.
DEFAULTS_VARIANTS = {"grd_gamma0_rtc_defaults"}


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
    if path.stem not in DEFAULTS_VARIANTS:
        assert (FLATTENING_PARAMS <= declared) is ("Terrain-Flattening" in known)
    else:
        assert not (FLATTENING_PARAMS & declared), "a defaults variant must not be tunable"
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
    selected = parameter(correction, "sourceBands")
    if not selected:
        return  # empty means every band, which is always satisfiable
    requested = set(selected.split(","))
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
    """Fallback path, used only when no GA raster can be read beside the scene."""
    scene = tmp_path / "before_after" / "pilliga" / "t009" / "S1A_IW_GRDH_x.zip"
    assert runner.crs_for(scene) == "EPSG:32756"
    hunter = tmp_path / "before_after" / "hunter" / "t009" / "S1A_IW_GRDH_x.zip"
    assert runner.crs_for(hunter) == "EPSG:32756"


def test_crs_is_taken_from_the_ga_raster_not_the_aoi_guess(tmp_path: Path):
    """GA picks the projection per burst, and it is not always the zone the AOI
    centre falls in -- the Pilliga centre is zone 55, GA delivers zone 56."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    scene_dir = tmp_path / "pilliga" / "t009_019142_iw1" / "20240603"
    scene_dir.mkdir(parents=True)
    scene = scene_dir / "S1A_IW_GRDH_x.zip"
    reference = scene_dir / "ga_s1a_nrb_0-1-0_T009-019142-IW1_x_VH-gamma0.tif"
    with rasterio.open(
        reference, "w", driver="GTiff", height=4, width=4, count=1,
        dtype="float32", crs="EPSG:32756", transform=from_origin(0, 0, 20, 20),
    ) as dst:
        dst.write(np.zeros((4, 4), dtype="float32"), 1)
    assert runner.crs_for(scene) == "EPSG:32756"


def test_explicit_crs_beats_the_ga_raster(tmp_path: Path):
    scene_dir = tmp_path / "pilliga" / "t009" / "20240603"
    scene_dir.mkdir(parents=True)
    (scene_dir / "ga_x_VH-gamma0.tif").touch()  # unreadable, must not be consulted
    assert runner.crs_for(scene_dir / "S1A_IW_GRDH_x.zip", "EPSG:4326") == "EPSG:4326"


def test_unreadable_ga_raster_falls_back_rather_than_crashing(tmp_path: Path):
    scene_dir = tmp_path / "pilliga" / "t009" / "20240603"
    scene_dir.mkdir(parents=True)
    (scene_dir / "ga_x_VH-gamma0.tif").write_text("not a geotiff")
    assert runner.crs_for(scene_dir / "S1A_IW_GRDH_x.zip") == "EPSG:32756"


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
    if flattening is None or path.stem in DEFAULTS_VARIANTS:
        return
    assert parameter(flattening, "oversamplingMultiple") == "${oversampling}"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_border_limit_is_the_default_not_5000(path: Path):
    graph = ET.parse(path).getroot()
    assert parameter(nodes(graph)["Remove-GRD-Border-Noise"], "borderLimit") == "500"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_inland_dems_reading_zero_are_not_masked_as_sea(path: Path):
    if path.stem in DEFAULTS_VARIANTS:
        return
    graph = ET.parse(path).getroot()
    for node_id in ("Terrain-Flattening", "Terrain-Correction"):
        node = nodes(graph).get(node_id)
        if node is not None:
            assert parameter(node, "nodataValueAtSea") == "false"


def test_defaults_variant_really_holds_snap_defaults():
    """Its whole purpose is to differ from grd_gamma0_rtc only by reverting the
    three tuned settings, so it must not drift."""
    path = Path(__file__).resolve().parent.parent / "graphs" / "grd_gamma0_rtc_defaults.xml"
    flattening = nodes(ET.parse(path).getroot())["Terrain-Flattening"]
    assert parameter(flattening, "oversamplingMultiple") == "1.0"
    assert parameter(flattening, "additionalOverlap") == "0.1"
    assert parameter(flattening, "nodataValueAtSea") == "true"


@pytest.mark.parametrize("path", GRAPHS, ids=lambda p: p.stem)
def test_terrain_normalisation_is_never_applied_twice(path: Path):
    """Terrain-Correction has its own radiometric normalisation. Running it as
    well as Terrain-Flattening normalises slopes twice and overcorrects them."""
    known = nodes(ET.parse(path).getroot())
    correction = known.get("Terrain-Correction")
    if correction is None:
        return
    normalising = parameter(correction, "applyRadiometricNormalization") == "true"
    assert not (normalising and "Terrain-Flattening" in known)


def test_tcnorm_variant_does_not_calibrate_twice():
    """Terrain-Correction's radiometric normalisation calibrates internally.
    Calibrating first applies the LUT twice, which measured 53 dB dark."""
    path = Path(__file__).resolve().parent.parent / "graphs" / "grd_gamma0_tcnorm.xml"
    known = nodes(ET.parse(path).getroot())
    assert "Terrain-Flattening" not in known, "the point is a single pass"
    assert "Calibration" not in known, "normalisation calibrates; doing both is twice"
    correction = known["Terrain-Correction"]
    assert correction.find("sources/sourceProduct").get("refid") == "Remove-GRD-Border-Noise"
    assert parameter(correction, "applyRadiometricNormalization") == "true"
    # The conversion to gamma0 needs this band; SNAP writes sigma0 regardless.
    assert parameter(correction, "saveProjectedLocalIncidenceAngle") == "true"


def test_a_no_flattening_variant_exists_for_diagnosis():
    """Removing terrain flattening has to be a one-flag experiment, not an edit."""
    assert (Path(__file__).resolve().parent.parent / "graphs" / "grd_gamma0_ellipsoid.xml").exists()


def test_diagnostic_graph_stops_before_geocoding_and_emits_the_simulation():
    path = Path(__file__).resolve().parent.parent / "graphs" / "grd_tf_diagnostic.xml"
    known = nodes(ET.parse(path).getroot())
    assert "Terrain-Correction" not in known, "the diagnostic must stay in radar geometry"
    assert parameter(known["Terrain-Flattening"], "outputSimulatedImage") == "true"
    assert known["Write"].find("sources/sourceProduct").get("refid") == "Terrain-Flattening"


# --- locating SNAP ----------------------------------------------------------


def test_locate_gpt_accepts_an_explicit_path(tmp_path: Path):
    gpt = tmp_path / "gpt.exe"
    gpt.touch()
    assert runner.locate_gpt(str(gpt)) == str(gpt)


def test_locate_gpt_rejects_a_bad_explicit_path(tmp_path: Path):
    with pytest.raises(runner.ConfigError, match="not found"):
        runner.locate_gpt(str(tmp_path / "nope.exe"))


def test_locate_gpt_finds_a_user_scope_windows_install(tmp_path: Path, monkeypatch):
    """Without admin rights SNAP installs under LOCALAPPDATA, which is not on
    PATH and not in Program Files."""
    gpt = tmp_path / "snap" / "bin" / "gpt.exe"
    gpt.parent.mkdir(parents=True)
    gpt.touch()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(runner.shutil, "which", lambda _: None)
    assert runner.locate_gpt() == str(gpt)


def test_locate_gpt_prefers_path_over_the_candidate_list(monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda _: "/usr/bin/gpt")
    assert runner.locate_gpt() == "/usr/bin/gpt"


def test_locate_gpt_reports_where_it_looked(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(runner.shutil, "which", lambda _: None)
    with pytest.raises(runner.ConfigError, match="Looked in"):
        runner.locate_gpt()


def test_a_filtered_variant_exists_without_terrain_flattening():
    """Both original filtered variants sat behind terrain flattening, which
    displaces GRD geolocation, leaving no usable filtered option."""
    directory = Path(__file__).resolve().parent.parent / "graphs"
    unflattened_filtered = [
        path for path in directory.glob("*.xml")
        if "Speckle-Filter" in nodes(ET.parse(path).getroot())
        and "Terrain-Flattening" not in nodes(ET.parse(path).getroot())
    ]
    assert unflattened_filtered, "no speckle-filtered graph avoids terrain flattening"
