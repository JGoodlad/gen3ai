"""The ENGINE PIN (``utils.team_corpora.engine_pin``): a frozen corpus's builder decides by the engine-coverage verdict
recorded at its build, so an engine that GAINS coverage moves no corpus, while one that LOSES coverage throws.

The regression this guards: the gen-3 parity batches taught the engine Blast Burn (``2016f5c2e``) and 40-odd more ids,
and ``--check`` of ``w_forum_v1``, ``d3_setmut_v1`` and ``n1_proc_v1`` all failed on main although no shipped file
changed. The fake oracle below answers like that grown engine (every id runs) without Node or Rust."""
import json
import sys
from pathlib import Path

import pytest

from utils import team_corpora as TC
from utils.team_corpora import engine_pin as EP

#: Where each corpus's builder recorded the oracle's answer in its manifest (the pin's provenance).
RECORDED = {"w_forum_v1": ("engine", "unplayable_ids"), "d3_setmut_v1": ("engine", "prefilter_unplayable"),
            "n1_proc_v1": ("engine", "prefilter_unplayable")}


def _manifest(name):
    return json.loads((TC.CORPORA[name] / TC.MANIFEST_NAME).read_text())


def test_every_registered_corpus_has_a_pin_equal_to_what_its_manifest_recorded():
    pins = EP.load()
    assert set(pins) == set(TC.CORPORA) == set(RECORDED), "a new corpus needs an engine pin (and a RECORDED row here)"
    for name, (block, field) in RECORDED.items():
        rec = _manifest(name)[block][field]
        assert {k: sorted(v) for k, v in EP.pinned(name).items()} == \
            {k: sorted(rec.get(k, [])) for k in EP.KINDS}, f"{name}: the pin differs from its manifest's {block}.{field}"
        assert pins[name]["recorded_in"].endswith(f":: {block}.{field}")


def test_a_GROWN_engine_moves_no_decision():
    pin = EP.pinned("w_forum_v1")
    assert "blastburn" in pin["move"] and "shellbell" in pin["item"]
    live = {"move": {"blastburn": "ran", "tackle": "ran", "skyattack": "ran"}, "item": {"shellbell": "ran", "leftovers": "ran"}}
    assert EP.unplayable("w_forum_v1", live) == {"move": ["blastburn", "skyattack"], "item": ["shellbell"]}


def test_a_LOST_engine_id_throws():
    with pytest.raises(EP.EnginePinError, match="lost coverage"):
        EP.unplayable("w_forum_v1", {"move": {"tackle": "panic", "blastburn": "panic"}})


def test_an_unpinned_corpus_or_an_unknown_kind_is_refused():
    with pytest.raises(EP.EnginePinError, match="no engine pin"):
        EP.pinned("w_forum_v0")
    with pytest.raises(EP.EnginePinError, match="unknown oracle kind"):
        EP.unplayable("w_forum_v1", {"moves": {"tackle": "ran"}})


@pytest.fixture
def grown_engine(tmp_path, monkeypatch):
    """A stand-in ``scan_move_probe`` that runs EVERY id (the engine after the parity batches, and beyond)."""
    probe = Path(tmp_path, "scan_move_probe")
    probe.write_text(f"#!{sys.executable}\nimport json, sys\n"
                     "for ln in sys.stdin:\n    i = ln.strip()\n"
                     "    if i: print(json.dumps({'id': i, 'move': i, 'verdict': 'ran'}))\n")
    probe.chmod(0o755)
    from utils.ladder_corpus import build as LB

    monkeypatch.setattr(LB, "scan_probe_bin", lambda: str(probe))


def test_the_w_forum_build_drops_by_its_pin_under_a_grown_engine(grown_engine):
    from utils.team_corpora import build_w_forum as B

    rows = [{"verdict": "ok", "species": ["charizard"], "moves": ["blastburn", "earthquake"], "items": ["shellbell"],
             "abilities": ["blaze"]}]
    assert B.engine_unplayable(rows) == {"species": [], "move": ["blastburn"], "item": ["shellbell"], "ability": []}


@pytest.mark.parametrize("name", ["d3_setmut_v1", "n1_proc_v1"])
def test_the_set_drawing_prefilter_excludes_by_its_pin_under_a_grown_engine(grown_engine, name):
    from utils.team_corpora import build_d3_setmut as D

    assert "blastburn" in EP.pinned(name)["move"]
    assert D._engine_unplayable(name, "move", ["blastburn", "earthquake"]) == ["blastburn"]
    assert D._engine_unplayable(name, "item", []) == []

