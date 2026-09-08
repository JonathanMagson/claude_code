import numpy as np

from vegmon import indices as ix


def test_ndvi_and_nbr_have_expected_sign_and_range():
    red = np.array([[0.05, 0.20]], dtype=np.float32)
    nir = np.array([[0.40, 0.22]], dtype=np.float32)
    swir2 = np.array([[0.08, 0.30]], dtype=np.float32)

    ndvi = ix.ndvi(red, nir)
    nbr = ix.nbr(nir, swir2)

    assert ndvi[0, 0] > 0.5 and ndvi[0, 1] < 0.1
    assert nbr[0, 0] > 0.5 and nbr[0, 1] < 0.0
    assert np.all(np.abs(ndvi) <= 1.0) and np.all(np.abs(nbr) <= 1.0)


def test_zero_denominator_gives_nan_not_an_exception():
    assert np.isnan(ix.ndvi(np.array([0.0]), np.array([0.0])))[0]
    assert np.isnan(ix.nbr(np.array([0.0]), np.array([0.0])))[0]


def test_nan_inputs_propagate():
    assert np.isnan(ix.ndvi(np.array([np.nan]), np.array([0.4])))[0]


def test_db_roundtrip():
    linear = np.array([0.001, 0.01, 0.1, 1.0], dtype=np.float32)
    assert np.allclose(ix.db_to_linear(ix.linear_to_db(linear)), linear, rtol=1e-4)


def test_linear_to_db_known_values():
    assert np.isclose(ix.linear_to_db(np.array([0.01]))[0], -20.0, atol=1e-4)
    assert np.isnan(ix.linear_to_db(np.array([0.0]))[0])


def test_rvi_is_bounded_and_rises_with_cross_pol():
    low = ix.rvi(np.array([0.05]), np.array([0.002]))[0]
    high = ix.rvi(np.array([0.05]), np.array([0.02]))[0]
    assert 0.0 <= low < high <= 4.0


def test_tasselled_cap_wetness_falls_when_swir_rises():
    bands = [np.array([0.05]), np.array([0.08]), np.array([0.06]), np.array([0.35])]
    wet_canopy = ix.tc_wetness(*bands, np.array([0.15]), np.array([0.07]))
    dry_soil = ix.tc_wetness(*bands, np.array([0.36]), np.array([0.33]))
    assert wet_canopy[0] > dry_soil[0]


def test_zscore_stable_returns_nan_without_a_population():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.all(np.isnan(ix.zscore_stable(values, np.zeros(3, bool))))


def test_zscore_stable_centres_on_the_stable_population():
    values = np.array([1.0, 2.0, 3.0, 40.0], dtype=np.float32)
    stable = np.array([True, True, True, False])
    scores = ix.zscore_stable(values, stable)
    # Standardised against the stable population, its own mean lands at zero
    # and the outlier is many standard deviations away.
    assert np.isclose(scores[1], 0.0, atol=1e-5)
    assert scores[3] > 10.0


def test_zscore_stable_returns_nan_when_the_population_has_no_spread():
    values = np.array([1.0, 1.0, 1.0, 9.0], dtype=np.float32)
    scores = ix.zscore_stable(values, np.array([True, True, True, False]))
    assert np.all(np.isnan(scores))
