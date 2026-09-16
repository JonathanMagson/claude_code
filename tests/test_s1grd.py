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


# ---------------------------------------------------------------------------
# GA burst id -> ESA burst id
# ---------------------------------------------------------------------------


def test_burst_id_decomposes_to_track_esa_burst_and_swath():
    # GA's t009_019126_iw2 is track 9, ESA burst 19126, IW2. The middle field
    # is the ESA burst id: the OPERA database holds 1,127,661 IW rows over
    # 375,887 distinct middle values, three sub-swath rows each, which is
    # exactly ESA's IW burst-id space.
    from ga_l1_index import decompose_burst

    assert decompose_burst("t009_019126_iw2") == {
        "track": 9, "esa_burst_id": 19126, "subswath": "IW2"
    }
    assert decompose_burst("t045_095772_iw3")["esa_burst_id"] == 95772


def test_burst_id_decomposition_survives_rubbish():
    from ga_l1_index import decompose_burst

    assert decompose_burst("rubbish")["track"] == ""
    assert decompose_burst("")["esa_burst_id"] == ""


def test_index_writes_one_row_per_burst_and_deduplicates_scene_lists(tmp_path):
    from ga_l1_index import FIELDS, write_outputs

    slc = "S1A_IW_SLC__1SDV_20260420T192256_20260420T192323_064167_081391_1104"
    grd = "S1A_IW_GRDH_1SDV_20260420T192256_20260420T192321_064167_081391_D3B4"
    rows = [
        {"aoi": "pilliga", "ga_burst_id": b, "esa_burst_id": 0, "track": 45,
         "subswath": "IW2", "acquired": "2026-04-20", "datetime": "", "ga_item_id": "",
         "ga_collection": "", "platform": "", "absolute_orbit": "", "datatake": "",
         "orbit_state": "", "slc_scene_id": slc, "grd_scene_id": grd,
         "grd_slice_count": 2, "grd_all_slices": f"{grd}|other"}
        for b in ("t045_095772_iw2", "t045_095773_iw1", "t045_095774_iw1")
    ]
    paths = write_outputs(rows, tmp_path)

    body = paths["csv"].read_text().strip().splitlines()
    assert body[0] == ",".join(FIELDS)
    assert len(body) == 4  # header + three bursts

    # three bursts, one shared acquisition: the scene lists must collapse
    assert paths["slc"].read_text().strip().splitlines() == [slc]
    assert paths["grd"].read_text().strip().splitlines() == [grd]
    assert paths["grd_all"].read_text().strip().splitlines() == [grd, "other"]


def test_scene_pairs_collapse_many_bursts_to_one_acquisition():
    # The burst table repeats a pairing once per burst; the scene table is the
    # SLC-to-GRD list on its own, which is what you want when the question is
    # about Level-1 products rather than GA ones.
    from ga_l1_index import PAIR_FIELDS, scene_pairs

    slc = "S1A_IW_SLC__1SDV_20260420T192256_20260420T192323_064167_081391_1104"
    grd = "S1A_IW_GRDH_1SDV_20260420T192256_20260420T192321_064167_081391_D3B4"
    rows = [
        {"aoi": aoi, "slc_scene_id": slc, "grd_scene_id": grd, "acquired": "2026-04-20",
         "grd_slice_count": 2, "grd_all_slices": f"{grd}|other", "platform": "Sentinel-1A",
         "absolute_orbit": 64167, "datatake": "081391", "track": 45,
         "orbit_state": "descending"}
        for aoi in ("pilliga", "pilliga", "hunter")
    ]
    pairs = scene_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["ga_burst_count"] == 3
    assert pairs[0]["aois"] == "hunter|pilliga"
    assert set(PAIR_FIELDS) >= set(pairs[0])


def test_scene_pairs_skip_bursts_with_no_slc():
    from ga_l1_index import scene_pairs

    assert scene_pairs([{"slc_scene_id": "", "grd_scene_id": ""}]) == []


def test_repeat_gaps_finds_the_twelve_day_cycle():
    from summarise_index import _repeat_gaps

    dates = ["2024-06-03", "2024-06-15", "2024-06-27", "2024-07-09"]
    assert _repeat_gaps(dates).most_common(1) == [(12, 3)]


def test_repeat_gaps_ignores_unparseable_dates():
    from summarise_index import _repeat_gaps

    assert _repeat_gaps(["2024-06-03", "", "2024-06-15"]).most_common(1) == [(12, 1)]


def test_int_coercion_treats_blanks_as_zero():
    # grd_slice_count is blank for an unmatched row; the seam count must not
    # blow up on it.
    from summarise_index import _int

    assert _int("2") == 2
    assert _int("") == 0
    assert _int(None) == 0


# ---------------------------------------------------------------------------
# before/after pair selection
# ---------------------------------------------------------------------------


def _ba_item(burst, day):
    return {"properties": {"sarard:burst_id": burst,
                           "datetime": f"{day}T08:40:00Z"}}


BA_ITEMS = (
    [_ba_item(b, "2024-06-03") for b in ("a", "b", "c")]
    + [_ba_item(b, "2024-09-01") for b in ("a", "b")]
    + [_ba_item(b, "2025-06-22") for b in ("a", "b", "d")]
)


def test_before_after_picks_a_burst_present_on_both_dates():
    # 'c' is only on the first date and 'd' only on the last; choosing either
    # gives a pair that cannot be differenced.
    from make_before_after import pick_pair

    burst, first, last = pick_pair(BA_ITEMS)
    assert burst in ("a", "b")
    assert (first, last) == ("2024-06-03", "2025-06-22")


def test_before_after_rejects_a_burst_missing_from_one_date():
    from make_before_after import pick_pair

    with pytest.raises(SystemExit):
        pick_pair(BA_ITEMS, burst="c")


def test_before_after_rejects_a_date_with_no_acquisition():
    from make_before_after import pick_pair

    with pytest.raises(SystemExit):
        pick_pair(BA_ITEMS, before="2024-07-04")


def test_before_after_needs_two_dates():
    from make_before_after import pick_pair

    with pytest.raises(SystemExit):
        pick_pair([_ba_item("a", "2024-06-03")])


# ---------------------------------------------------------------------------
# raster checking
# ---------------------------------------------------------------------------


def test_raster_scan_groups_by_folder_and_recognises_both_spellings(tmp_path):
    # Collection 0 hyphenates (VV-gamma0), Collection 1 underscores, and the
    # scan has to see both or a whole collection reports as having no bands.
    from check_rasters import scan

    acq = tmp_path / "hunter" / "t009_019128_iw3" / "20240603"
    acq.mkdir(parents=True)
    (acq / "ga_s1a_nrb_0-1-0_T009_20240603Z_VV-gamma0.tif").touch()
    (acq / "ga_s1a_nrb_0-1-0_T009_20240603Z_VH-gamma0.tif").touch()
    (acq / "ga_s1a_nrb_0-1-0_T009_20240603Z_mask.tif").touch()
    static = tmp_path / "hunter" / "t009_019128_iw3" / "static"
    static.mkdir()
    (static / "ga_s1_nrb-static_0-1-0_T009_local-incidence-angle.tif").touch()

    groups = scan(tmp_path)
    acq_key = ("hunter", "t009_019128_iw3", "20240603")
    assert set(groups[acq_key]) == {"vv", "vh", "mask"}
    assert set(groups[("hunter", "t009_019128_iw3", "static")]) == {"lia"}


def test_raster_scan_reads_collection_1_underscored_names(tmp_path):
    from check_rasters import scan

    acq = tmp_path / "burst" / "date"
    acq.mkdir(parents=True)
    (acq / "x_vv_gamma0.tif").touch()
    (acq / "x_mask.tif").touch()
    assert set(scan(tmp_path)[("burst", "date")]) == {"vv", "mask"}


# ---------------------------------------------------------------------------
# Level-1 download planning
# ---------------------------------------------------------------------------


def _manifest(tmp_path, rows):
    import csv as _csv

    from fetch_s1_level1 import DEFAULT_MANIFEST  # noqa: F401  (import check)

    fields = ["aoi", "role", "acquired", "ga_burst_id", "slc_scene_id", "grd_scene_id"]
    root = tmp_path / "before_after"
    root.mkdir()
    path = root / "before_after.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return path


def test_level1_downloads_land_beside_the_ga_rasters(tmp_path):
    # The zips have to go into the acquisition folder the NRB rasters are in,
    # which means rebuilding the YYYYMMDD folder name from the CSV's date.
    from fetch_s1_level1 import read_manifest

    path = _manifest(tmp_path, [
        {"aoi": "hunter", "role": "before", "acquired": "2024-06-03",
         "ga_burst_id": "t009_019128_iw3", "slc_scene_id": "SLC_A",
         "grd_scene_id": "GRD_A"},
    ])
    jobs = read_manifest(path, ["slc", "grd"])
    expected = path.parent / "hunter" / "t009_019128_iw3" / "20240603"
    assert [j["dest"] for j in jobs] == [expected, expected]


def test_level1_skips_rows_with_no_scene_id(tmp_path):
    from fetch_s1_level1 import read_manifest

    path = _manifest(tmp_path, [
        {"aoi": "a", "role": "before", "acquired": "2024-06-03",
         "ga_burst_id": "b", "slc_scene_id": "SLC_A", "grd_scene_id": ""},
    ])
    jobs = read_manifest(path, ["slc", "grd"])
    assert [j["kind"] for j in jobs] == ["slc"]


def test_level1_product_selection_is_honoured(tmp_path):
    from fetch_s1_level1 import read_manifest

    path = _manifest(tmp_path, [
        {"aoi": "a", "role": "before", "acquired": "2024-06-03",
         "ga_burst_id": "b", "slc_scene_id": "SLC_A", "grd_scene_id": "GRD_A"},
    ])
    assert [j["kind"] for j in read_manifest(path, ["grd"])] == ["grd"]


def test_earthdata_credentials_come_from_the_environment(monkeypatch, tmp_path):
    from fetch_s1_level1 import credentials

    monkeypatch.setenv("EARTHDATA_USERNAME", "user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "secret")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    user, password, source = credentials()
    assert (user, password) == ("user", "secret")
    assert "EARTHDATA_USERNAME" in source


def test_windows_style_netrc_is_read_too(monkeypatch, tmp_path):
    # Windows tooling writes _netrc, not .netrc; missing it sends the user
    # hunting for credentials the script could have found.
    from fetch_s1_level1 import EARTHDATA_HOST, credentials

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    (tmp_path / "_netrc").write_text(
        f"machine {EARTHDATA_HOST} login jdoe password s3cret\n"
    )
    user, password, source = credentials()
    assert (user, password) == ("jdoe", "s3cret")
    assert "_netrc" in source


def test_auth_help_covers_the_usual_causes():
    from fetch_s1_level1 import AUTH_HELP

    # The email-instead-of-username mistake fails identically to a wrong
    # password, so it has to be named explicitly.
    assert "not the email address" in AUTH_HELP
    assert "--username" in AUTH_HELP


def test_missing_credentials_raise_with_a_usable_message(monkeypatch):
    from fetch_s1_level1 import AuthError, credentials

    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", "/nonexistent")
    monkeypatch.setenv("USERPROFILE", "/nonexistent")
    with pytest.raises(AuthError) as excinfo:
        credentials()
    assert "urs.earthdata.nasa.gov" in str(excinfo.value)


def _asf_product(**props):
    class _P:
        def __init__(self, properties):
            self.properties = properties

    return _P(props)


def test_metadata_companion_is_not_mistaken_for_the_data_product():
    # An ASF granule carries both the archive and a small metadata record under
    # one sceneName. Keying on the name alone lets the metadata record win and
    # a 9 MB XML stands in for a 4 GB SLC.
    from fetch_s1_level1 import _is_metadata

    data = _asf_product(processingLevel="SLC", url="https://x/S1A.zip")
    meta = _asf_product(processingLevel="METADATA_SLC", url="https://x/S1A.iso.xml")
    assert not _is_metadata(data)
    assert _is_metadata(meta)


def test_lookup_keeps_the_data_product_when_metadata_arrives_last(monkeypatch):
    import sys as _sys
    import types as _types

    data = _asf_product(sceneName="SLC_X", processingLevel="SLC",
                        bytes=4_100_000_000, url="https://x/SLC_X.zip")
    meta = _asf_product(sceneName="SLC_X", processingLevel="METADATA_SLC",
                        bytes=9_000_000, url="https://x/SLC_X.iso.xml")
    stub = _types.ModuleType("asf_search")
    stub.granule_search = lambda batch: [data, meta]  # metadata last
    monkeypatch.setitem(_sys.modules, "asf_search", stub)

    from fetch_s1_level1 import lookup, product_bytes

    assert product_bytes(lookup(["SLC_X"])["SLC_X"]) == 4_100_000_000


def test_product_bytes_handles_the_ways_asf_reports_size():
    from fetch_s1_level1 import product_bytes

    assert product_bytes(_asf_product(bytes=1_000)) == 1_000
    assert product_bytes(_asf_product(bytes="1000")) == 1_000
    assert product_bytes(_asf_product(sizeMB=4_100)) == 4_100_000_000
    assert product_bytes(_asf_product(bytes=None)) == 0
    assert product_bytes(_asf_product()) == 0
    assert product_bytes(None) == 0


def test_explicit_ca_bundle_is_exported_for_requests(tmp_path, monkeypatch):
    # Both variables matter: requests reads REQUESTS_CA_BUNDLE, and anything
    # going through plain ssl reads SSL_CERT_FILE.
    from fetch_s1_level1 import use_system_certs

    pem = tmp_path / "corporate-root.pem"
    pem.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    note = use_system_certs(pem)
    assert str(pem) in note
    import os as _os
    assert _os.environ["REQUESTS_CA_BUNDLE"] == str(pem.resolve())
    assert _os.environ["SSL_CERT_FILE"] == str(pem.resolve())


def test_tls_help_names_the_fix_and_refuses_the_shortcut():
    from fetch_s1_level1 import TLS_HELP

    assert "--ca-bundle" in TLS_HELP
    assert "root CA" in TLS_HELP
    # Disabling verification would make the error go away and is not a fix.
    assert "Do not disable certificate verification" in TLS_HELP


def test_ca_export_degrades_where_the_os_store_is_not_enumerable(monkeypatch):
    # ssl.enum_certificates is Windows-only; elsewhere this must return None
    # rather than raising, since Python already uses the platform roots there.
    import ssl as _ssl

    from fetch_s1_level1 import export_system_ca_bundle

    monkeypatch.delattr(_ssl, "enum_certificates", raising=False)
    assert export_system_ca_bundle() is None


def test_ca_export_merges_the_os_store_with_certifi(tmp_path, monkeypatch):
    # A network that inspects HTTPS usually inspects only some hosts, so the
    # bundle has to keep the public roots as well as the corporate one.
    import ssl as _ssl

    from fetch_s1_level1 import export_system_ca_bundle

    fake = _ssl.PEM_cert_to_DER_cert(
        "-----BEGIN CERTIFICATE-----\n" + "MIIBIjANBgkq" * 4 + "\n-----END CERTIFICATE-----\n"
    )
    monkeypatch.setattr(
        _ssl, "enum_certificates",
        lambda store: [(fake, "x509_asn", True)] if store in ("ROOT", "CA") else [],
        raising=False,
    )
    bundle = export_system_ca_bundle(tmp_path / "ca-bundle.pem")
    assert bundle is not None
    text = bundle.read_text(encoding="utf-8")
    # two injected OS certs plus the whole certifi set
    assert text.count("BEGIN CERTIFICATE") > 10
