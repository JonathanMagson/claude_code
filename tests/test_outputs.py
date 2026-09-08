"""Palette, figures, report and CLI."""

import re

import numpy as np
import pytest

from vegmon import palette as pal


# --- palette ---------------------------------------------------------------


def test_sequential_ramps_darken_monotonically():
    assert pal.is_monotonic(pal.BLUE_RAMP)
    assert pal.is_monotonic(pal.ORANGE_RAMP)


def test_categorical_slots_are_distinct_and_fixed():
    assert len(set(pal.SERIES_LIGHT)) == 3
    assert len(set(pal.SERIES_DARK)) == 3
    # Colour follows the tier, not its rank in a given run.
    assert pal.TIER_COLOURS["confirmed"] == pal.SERIES_LIGHT[0]
    assert pal.TIER_COLOURS["s2_only"] == pal.SERIES_LIGHT[1]
    assert pal.TIER_COLOURS["s1_only"] == pal.SERIES_LIGHT[2]


def test_status_colours_are_not_reused_as_series_colours():
    assert not set(pal.STATUS.values()) & set(pal.SERIES_LIGHT)


def test_recovery_classes_all_have_a_colour():
    from vegmon.regrowth import CLASS_DESCRIPTIONS

    assert set(CLASS_DESCRIPTIONS) <= set(pal.CLASS_COLOURS)


def test_relative_luminance_endpoints():
    assert pal.relative_luminance("#ffffff") == pytest.approx(1.0)
    assert pal.relative_luminance("#000000") == pytest.approx(0.0)


def test_is_monotonic_rejects_a_reversing_ramp():
    assert not pal.is_monotonic(["#000000", "#ffffff", "#000000"])


# --- figures ---------------------------------------------------------------


def test_shared_stretch_makes_before_and_after_comparable(result):
    from vegmon.viz import false_colour, shared_stretch

    limits = shared_stretch(result.optical.pre, result.optical.post)
    before = false_colour(result.optical.pre, limits)
    after = false_colour(result.optical.post, limits)
    assert before.shape == after.shape == result.scene.grid.shape + (3,)
    assert 0.0 <= float(before.min()) and float(after.max()) <= 1.0


def test_change_map_figure_is_written(result, tmp_path):
    from vegmon.viz import plot_change_maps

    path = plot_change_maps(
        result.scene, result.optical, result.radar, result.clearing,
        tmp_path / "maps.png", truth=result.scene.truth["synthetic"], dpi=60,
    )
    assert path.exists() and path.stat().st_size > 20_000


def test_change_map_figure_works_without_truth(result, tmp_path):
    from vegmon.viz import plot_change_maps

    path = plot_change_maps(
        result.scene, result.optical, result.radar, result.clearing,
        tmp_path / "maps.png", truth=None, dpi=60,
    )
    assert path.exists()


def test_recovery_figure_is_written(result, tmp_path):
    from vegmon.viz import plot_recovery

    path = plot_recovery(result.regrowth, tmp_path / "recovery.png", dpi=60)
    assert path is not None and path.exists()


def test_recovery_figure_returns_none_with_no_patches(result, tmp_path):
    from vegmon.regrowth import RegrowthResult
    from vegmon.viz import plot_recovery

    empty = RegrowthResult(patches=[], config=result.regrowth.config)
    assert plot_recovery(empty, tmp_path / "none.png") is None


# --- report ----------------------------------------------------------------


def test_report_is_self_contained_and_covers_every_section(result, tmp_path):
    from vegmon.report import build_report

    path = build_report(result, tmp_path / "report.html")
    html = path.read_text()

    assert html.startswith("<!doctype html>")
    # No external requests: everything inlined.
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "<script src=" not in html and "<link " not in html

    for heading in ("Detection", "Regrowth", "Validation", "Method and settings"):
        assert f">{heading}<" in html

    # Every chart ships a table view alongside it.
    assert html.count("Table view:") >= 4
    # Identity is never colour alone.
    assert html.count("class=\"legend\"") >= 2
    # Dark mode is selected, not an automatic flip.
    assert 'data-theme="dark"' in html and "prefers-color-scheme: dark" in html


def test_report_embeds_figures_as_data_uris(result, tmp_path):
    from vegmon.report import build_report
    from vegmon.viz import plot_change_maps

    figure = plot_change_maps(
        result.scene, result.optical, result.radar, result.clearing,
        tmp_path / "maps.png", dpi=60,
    )
    html = build_report(result, tmp_path / "report.html", figures={"change_maps": figure}).read_text()
    assert "data:image/png;base64," in html


def test_report_escapes_untrusted_text(result, tmp_path):
    from vegmon.report import build_report

    result.config.aoi.__dict__  # frozen dataclass; patch the title instead
    html = build_report(
        result, tmp_path / "report.html", title="<script>alert(1)</script>"
    ).read_text()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_bar_chart_handles_an_empty_series():
    from vegmon.report import bar_chart

    assert "No data" in bar_chart([])


def test_bar_chart_marks_carry_hover_and_title():
    from vegmon.report import bar_chart

    svg = bar_chart([{"label": "a", "value": 2.0}, {"label": "b", "value": 1.0}], unit=" ha")
    assert svg.count("data-tip=") == 2
    assert svg.count("<title>") == 2


def test_dumbbell_chart_draws_a_pair_per_row():
    from vegmon.report import dumbbell_chart

    svg = dumbbell_chart([{"label": "Patch 1", "a": 20, "b": 3}])
    assert svg.count("<circle") == 2
    assert "Sentinel-1" in svg


# --- CLI -------------------------------------------------------------------


def test_cli_demo_runs_end_to_end(tmp_path, capsys):
    from vegmon.cli import main

    code = main(["demo", "--size", "48", "--outputs", str(tmp_path), "--quiet"])
    assert code == 0
    for name in ("summary.json", "regrowth.csv", "change_maps.png", "report.html"):
        assert (tmp_path / name).exists(), name


def test_cli_demo_caches_and_reuses_a_scene(tmp_path):
    from vegmon.cli import main

    cache = tmp_path / "scene.npz"
    main(["demo", "--size", "48", "--outputs", str(tmp_path / "a"), "--cache", str(cache),
          "--no-figures", "--quiet"])
    assert cache.exists()
    main(["demo", "--size", "48", "--outputs", str(tmp_path / "b"), "--cache", str(cache),
          "--no-figures", "--quiet"])
    assert (tmp_path / "b" / "summary.json").exists()


def test_cli_config_template_roundtrips(tmp_path, capsys):
    from vegmon.cli import main
    from vegmon.config import PipelineConfig

    path = tmp_path / "config.json"
    assert main(["config-template", str(path)]) == 0
    assert PipelineConfig.from_json(path).aoi.crs == "EPSG:32755"


def test_cli_catalogues_lists_the_sources(capsys):
    from vegmon.cli import main

    assert main(["catalogues"]) == 0
    out = capsys.readouterr().out
    assert "earth-search" in out and "planetary-computer" in out


def test_cli_version(capsys):
    from vegmon.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert re.search(r"vegmon \d+\.\d+\.\d+", capsys.readouterr().out)
