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


@pytest.mark.parametrize("row", [{"name": "No Num"}, {"num": None}, {"num": True}, {"num": "7"}, "not a row"])
def test_a_dex_row_without_a_numeric_num_is_a_load_error_not_a_silent_zero(tmp_path, monkeypatch, row):
    """F-X5-5: `load_dex_json` refuses a row with no numeric `num` (it used to read as 0 through
    `get("num", 0)`, and so silently encode as the dex's index 0). The Rust encoder tables mirror it."""
    import json

    monkeypatch.setattr(_base, "_DATA_DIR", tmp_path)
    (tmp_path / "dex.json").write_text(json.dumps({"good": {"num": 3}, "bad": row}))
    with pytest.raises(ValueError, match="row 'bad' has no numeric `num`"):
        _base.load_dex_json("dex.json")
    (tmp_path / "dex.json").write_text(json.dumps({"good": {"num": 3}, "zero": {"num": 0}}))
    assert _base.load_dex_json("dex.json")["zero"]["num"] == 0      # a REAL zero is fine


def test_every_shipped_dex_row_has_a_numeric_num():
    for ns in (gen3_data.species, gen3_data.moves, gen3_data.items, gen3_data.abilities):
        assert ns.raw()                                  # loads through load_dex_json: raises on a bad row
