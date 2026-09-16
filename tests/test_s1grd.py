"""The raw-GRD Sentinel-1 reader.

Everything here is offline: the parts that touch the network are exercised in
the real runs, and the parts that decide whether the numbers come out right -
orbit arithmetic, overpass windows, LUT parsing, power averaging - are pure
functions and are tested as such.
"""

from datetime import date, datetime

import numpy as np
import pytest

from vegmon.grid import Grid
from vegmon.s1grd import (
    Lut,
    _block_mean_power,
    _fine_grid,
    _in_bands,
    _scene_from_prefix,
    overpass_bands,
    parse_calibration,
    parse_noise,
    relative_orbit,
)

SCENE = (
    "GRD/2023/6/11/IW/DV/"
    "S1A_IW_GRDH_1SDV_20230611T192312_20230611T192337_048942_05E2AA_6EBE/"
)


def test_relative_orbit_matches_the_published_formula():
    # Absolute orbit 48942 on Sentinel-1A is relative orbit 45.
    assert relative_orbit(SCENE.rstrip("/").split("/")[-1]) == 45


def test_relative_orbit_is_in_range_for_a_long_run_of_orbits():
    for absolute in range(40000, 40400):
        scene = f"S1A_IW_GRDH_1SDV_20230611T192312_20230611T192337_{absolute:06d}_05E2AA_6EBE"
        assert 1 <= relative_orbit(scene) <= 175


def test_relative_orbit_repeats_every_175_orbits():
    a = "S1A_IW_GRDH_1SDV_20230611T192312_20230611T192337_048942_05E2AA_6EBE"
    b = "S1A_IW_GRDH_1SDV_20230623T192312_20230623T192337_049117_05E2AA_6EBE"
    assert relative_orbit(a) == relative_orbit(b)


def test_relative_orbit_rejects_a_non_scene():
    with pytest.raises(ValueError):
        relative_orbit("not-a-scene")


# Each satellite has its own offset from absolute to relative orbit, and they
# are not interchangeable: using S1A's for S1C puts the track out by 99. These
# cases are anchored on acquisitions whose relative orbit is stated
# independently - Geoscience Australia's NRB metadata for S1A/S1B/S1C, and
# footprint-against-burst-database matching for S1D.
@pytest.mark.parametrize(
    "scene,expected",
    [
        ("S1A_IW_GRDH_1SDV_20260420T192256_20260420T192321_064167_081391_D3B4", 45),
        ("S1A_IW_GRDH_1SDV_20240805T200431_20240805T200458_055082_06B621_601A", 60),
        ("S1B_IW_GRDH_1SDV_20200101T192312_20200101T192337_025010_05E2AA_6EBE", 134),
        ("S1C_IW_GRDH_1SDV_20260415T213909_20260415T213940_007232_00EA8A_0AE3", 61),
        ("S1D_IW_GRDH_1SDV_20260906T191917_20260906T191947_004461_008100_ABCD", 45),
        ("S1D_IW_GRDH_1SDV_20260908T190441_20260908T190510_004490_008100_ABCD", 74),
    ],
)
def test_relative_orbit_is_right_for_every_satellite(scene, expected):
    assert relative_orbit(scene) == expected


def test_sentinel_1d_scenes_are_not_silently_dropped():
    # S1D carries most of the recent archive; a parser that only knows A/B/C
    # returns None here and the scene vanishes from every search.
    prefix = (
        "GRD/2026/9/6/IW/DV/"
        "S1D_IW_GRDH_1SDV_20260906T191917_20260906T191947_004461_008100_ABCD/"
    )
    scene = _scene_from_prefix(prefix)
    assert scene is not None
    assert scene.mission == "S1D"
    assert scene.relative_orbit == 45


def test_an_unknown_satellite_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        relative_orbit(
            "S1E_IW_GRDH_1SDV_20260906T191917_20260906T191947_004461_008100_ABCD"
        )


def test_scene_parsed_from_its_prefix():
    scene = _scene_from_prefix(SCENE)
    assert scene is not None
    assert scene.mission == "S1A"
    assert scene.acquired == date(2023, 6, 11)
    assert scene.start_time == datetime(2023, 6, 11, 19, 23, 12)
    assert scene.absolute_orbit == 48942
    assert scene.relative_orbit == 45
    assert scene.measurement_url("vh").endswith("measurement/iw-vh.tiff")


def test_unparseable_prefix_is_skipped_not_fatal():
    assert _scene_from_prefix("GRD/2023/6/11/IW/DV/rubbish/") is None


def test_overpass_bands_bracket_the_real_nsw_acquisition_times():
    """The two real overpasses over 150 E are 08:40 and 19:23 UTC."""
    bands = overpass_bands(150.13)
    scene_morning = _scene_from_prefix(SCENE.replace("T192312", "T084034"))
    scene_evening = _scene_from_prefix(SCENE)
    assert _in_bands(scene_morning, bands)
    assert _in_bands(scene_evening, bands)


def test_overpass_bands_exclude_the_far_side_of_the_planet():
    bands = overpass_bands(150.13)
    # Local midnight over NSW is 14:00 UTC; nothing is imaged then.
    midday = _scene_from_prefix(SCENE.replace("T192312", "T140000"))
    assert not _in_bands(midday, bands)


def test_overpass_bands_wrap_around_midnight():
    bands = overpass_bands(-60.0)  # South America
    assert all(0 <= low <= 23 and 0 <= high <= 23 for low, high in bands)


# --- calibration -----------------------------------------------------------


def _calibration_xml(quantity="gamma"):
    return f"""<?xml version="1.0"?><calibration><calibrationVectorList>
    <calibrationVector><line>0</line><pixel>0 10 20</pixel>
      <sigmaNought>100 110 120</sigmaNought><betaNought>200 210 220</betaNought>
      <{quantity}>300 310 320</{quantity}></calibrationVector>
    <calibrationVector><line>50</line><pixel>0 10 20</pixel>
      <sigmaNought>101 111 121</sigmaNought><betaNought>201 211 221</betaNought>
      <{quantity}>301 311 321</{quantity}></calibrationVector>
    </calibrationVectorList></calibration>""".encode()


def test_calibration_lut_parses_onto_its_own_grid():
    lut = parse_calibration(_calibration_xml(), "gamma")
    assert lut.shape == (2, 3)
    assert list(lut.lines) == [0, 50]
    assert list(lut.pixels) == [0, 10, 20]
    assert lut.values[0, 0] == pytest.approx(300.0)


def test_calibration_can_select_a_different_quantity():
    lut = parse_calibration(_calibration_xml(), "sigmaNought")
    assert lut.values[1, 2] == pytest.approx(121.0)


def test_calibration_without_vectors_is_an_error():
    with pytest.raises(ValueError):
        parse_calibration(b"<calibration></calibration>")


def test_noise_lut_parses_both_annotation_baselines():
    modern = b"""<?xml version="1.0"?><noise><noiseRangeVectorList>
      <noiseRangeVector><line>0</line><pixel>0 10</pixel>
        <noiseRangeLut>3000 2900</noiseRangeLut></noiseRangeVector>
      </noiseRangeVectorList></noise>"""
    legacy = b"""<?xml version="1.0"?><noise><noiseVectorList>
      <noiseVector><line>0</line><pixel>0 10</pixel>
        <noiseLut>3000 2900</noiseLut></noiseVector>
      </noiseVectorList></noise>"""
    for xml in (modern, legacy):
        lut = parse_noise(xml)
        assert lut is not None and lut.values[0, 0] == pytest.approx(3000.0)


def test_noise_absent_returns_none_rather_than_raising():
    assert parse_noise(b"<noise></noise>") is None


# --- geometry and multi-looking --------------------------------------------


def _grid(size=8, resolution=20.0):
    return Grid.from_origin("EPSG:32755", 700000, 6580000, resolution, size, size)


def test_fine_grid_matches_the_extent_at_native_resolution():
    coarse = _grid(8, 20.0)
    fine = _fine_grid(coarse, 10.0)
    assert fine.width == 16 and fine.height == 16
    assert fine.resolution == (10.0, 10.0)
    assert fine.bounds == pytest.approx(coarse.bounds)


def test_fine_grid_is_a_no_op_at_native_resolution():
    coarse = _grid(8, 10.0)
    assert _fine_grid(coarse, 10.0) is coarse


def test_block_mean_power_averages_in_linear_units():
    coarse = _grid(2, 20.0)
    fine = _fine_grid(coarse, 10.0)
    values = np.array([[1.0, 3.0, 0.0, 0.0],
                       [5.0, 7.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    out = _block_mean_power(values, coarse, fine)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(4.0)


def test_block_mean_power_ignores_nan_and_reports_nan_when_all_are():
    coarse = _grid(1, 20.0)
    fine = _fine_grid(coarse, 10.0)
    partial = np.array([[2.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    assert _block_mean_power(partial, coarse, fine)[0, 0] == pytest.approx(3.0)
    assert np.isnan(_block_mean_power(np.full((2, 2), np.nan, np.float32), coarse, fine)[0, 0])


def test_averaging_power_then_logging_is_not_the_same_as_averaging_decibels():
    """The bias this ordering avoids, stated as a test.

    Averaging decibels averages logarithms, which sits below the true mean
    power for any pixel with real variation - the classic way to produce a
    plausible-looking backscatter image that is quietly wrong.
    """
    coarse = _grid(1, 20.0)
    fine = _fine_grid(coarse, 10.0)
    power = np.array([[0.001, 0.1], [0.01, 0.05]], dtype=np.float32)
    correct = 10 * np.log10(_block_mean_power(power, coarse, fine)[0, 0])
    naive = float(np.mean(10 * np.log10(power)))
    assert correct > naive + 3.0


# ---------------------------------------------------------------------------
# GA Collection 0 / Collection 1 asset naming
# ---------------------------------------------------------------------------

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "tools"))


def _assets(names):
    return {"assets": {n: {"href": f"https://example/{n}.tif"} for n in names}}


# Collection 0 spells these VV_gamma0/mask, Collection 1 spells the same layers
# vv_gamma0/oa_layover_shadow_mask. A fetcher matching on the exact key silently
# downloads nothing for one of the two.
C0_KEYS = ["VV_gamma0", "VH_gamma0", "mask", "thumbnail", "number_of_looks"]
C1_KEYS = ["vv_gamma0", "vh_gamma0", "oa_layover_shadow_mask", "thumbnail",
           "oa_number_of_looks"]


@pytest.mark.parametrize("keys", [C0_KEYS, C1_KEYS])
def test_both_collections_resolve_to_the_same_asset_names(keys):
    from fetch_ga_c0 import DEFAULT_ASSETS, select_assets

    got = select_assets(_assets(keys), DEFAULT_ASSETS)
    assert set(got) == {"vv_gamma0", "vh_gamma0", "mask"}


def test_thumbnail_is_not_fetched_by_default():
    from fetch_ga_c0 import DEFAULT_ASSETS, select_assets

    assert "thumbnail" not in select_assets(_assets(C0_KEYS), DEFAULT_ASSETS)


def test_static_layers_are_fetched_once_per_burst_not_per_date():
    from fetch_ga_c0 import DEFAULT_ASSETS, plan

    keys = C0_KEYS + ["gamma0_to_beta0_ratio", "gamma0_to_sigma0_ratio",
                      "incidence_angle", "local_incidence_angle"]
    items = [
        dict(_assets(keys),
             properties={"sarard:burst_id": "t009_019126_iw2",
                         "datetime": f"2024-07-{day:02d}T19:12:34Z"})
        for day in (3, 15, 27)
    ]
    jobs = plan(items, _Path("out"), DEFAULT_ASSETS, with_static=True)
    statics = [j for j in jobs if "/static/" in j[1].as_posix()]
    assert len(statics) == 5


def test_static_asset_selection_can_be_narrowed():
    # All five geometry layers are ~193 MB per burst; local incidence angle
    # alone is ~46 MB and is the one that explains a terrain-correction
    # difference, so narrowing has to actually narrow.
    from fetch_ga_c0 import DEFAULT_ASSETS, plan

    keys = C0_KEYS + ["gamma0_to_beta0_ratio", "gamma0_to_sigma0_ratio",
                      "incidence_angle", "local_incidence_angle"]
    items = [
        dict(_assets(keys),
             properties={"sarard:burst_id": "t009_019126_iw2",
                         "datetime": "2024-07-03T08:40:00Z"})
    ]
    narrowed = plan(items, _Path("out"), DEFAULT_ASSETS, True,
                    ["local_incidence_angle"])
    statics = [j for j in narrowed if "/static/" in j[1].as_posix()]
    assert len(statics) == 1
    # GA's filenames hyphenate where the asset key underscores, so normalise
    # before asserting which layer came back.
    assert "local_incidence_angle" in statics[0][0].replace("-", "_")
