import pytest

from agents import gen3_data
from agents.gen3_data import _base


def test_facade_exposes_every_namespace():
    for name in ("moves", "species", "items", "abilities", "natures", "type_chart", "priors"):
        assert hasattr(gen3_data, name), f"gen3_data is missing the {name} namespace"


def test_dexes_are_singletons():
    # Parsed once: the same object comes back each call (no re-read, no re-parse).
    assert gen3_data.species.raw() is gen3_data.species.raw()
    assert gen3_data.moves._dex() is gen3_data.moves._dex()
    assert gen3_data.type_chart.chart() is gen3_data.type_chart.chart()


def test_base_singleton_runs_builder_once():
    calls = []
    f = _base.singleton(lambda: (calls.append(1), "value")[1])
    assert f() == "value"
    assert f() == "value"
    assert len(calls) == 1  # built once, cached thereafter


def test_base_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        _base.load_json("does_not_exist.json")
