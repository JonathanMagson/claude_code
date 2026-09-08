"""The live loaders, tested without a network.

Anything that needs a reachable catalogue is marked ``network`` and skipped by
default - a test suite that fails because a corporate proxy blocks AWS is a
test suite people learn to ignore.
"""

import pytest

from vegmon.config import demo_config
from vegmon.stac import CATALOGUES, CatalogueUnavailable, catalogue_report, open_catalogue


def test_every_catalogue_declares_its_collections_and_url():
    for name, catalogue in CATALOGUES.items():
        assert catalogue.url.startswith("https://"), name
        assert catalogue.optical_collection or catalogue.radar_collection, name


def test_catalogue_report_names_every_source():
    report = catalogue_report()
    for name in CATALOGUES:
        assert name in report


def test_unknown_catalogue_raises_with_the_valid_choices():
    with pytest.raises(KeyError, match="earth-search"):
        open_catalogue("does-not-exist")


def test_unknown_catalogue_is_rejected_before_loading():
    config = demo_config()
    from vegmon.stac import load_sentinel2

    with pytest.raises(KeyError):
        load_sentinel2(config, catalogue="nope")


def test_unreachable_catalogue_raises_an_actionable_error(monkeypatch):
    import vegmon.stac as stac

    class _Client:
        @staticmethod
        def open(*args, **kwargs):
            raise OSError("connection refused")

    monkeypatch.setattr(
        stac, "open_catalogue", stac.open_catalogue
    )
    monkeypatch.setitem(
        __import__("sys").modules, "pystac_client", type("m", (), {"Client": _Client})
    )
    with pytest.raises(CatalogueUnavailable) as exc:
        open_catalogue("earth-search")
    message = str(exc.value)
    assert "earth-search" in message
    # The message has to name the likely cause and the offline fallback,
    # because a blocked egress policy is far more common than an outage.
    assert "egress" in message or "allow-list" in message
    assert "vegmon demo" in message


@pytest.mark.network
def test_earth_search_is_reachable():
    open_catalogue("earth-search")
