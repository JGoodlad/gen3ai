"""The Rust encoder's layout (``src/rust_sim/src/encoder/layout.rs``) and the layout the MODEL reads agree.

``layout.rs`` is RUST-OWNED source since T27 P6 slice 6d-2 (2026-10-08): its generator
(``agents.observation.rust_core_obs_layout``, which read the deleted Python encoder's internals) is
deleted and the ``.rs`` file is edited directly — the P1 precedent for ``schema.rs`` /
``present/tables.rs``. The Python side keeps only what the MODEL reads: ``agents/observation/constants.py``
(every offset and dim the extractor slices by), ``gen3_effects.VOLATILE_SLOTS`` / ``CANT_REASONS_LIVE``,
the faint-cause vocabulary, ``TypeEncoder.TYPE_TO_IDX`` and ``pokemon._STATUS_STR_IDX``.

This gate PARSES ``layout.rs`` and holds every value that exists on BOTH sides equal, so a model-side
layout change the encoder does not mirror — or an encoder change the model does not — FAILS here,
before a single row is read at the wrong offset. The compared set is NAMED (``COMPARED``) and asserted
non-trivial: at least every block offset and ``OBS_DIM``.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pytest

from utils.paths import repo_path

LAYOUT_RS = repo_path("src", "rust_sim", "src", "encoder", "layout.rs")

#: The block offsets the extractor slices the row by — the floor of the compared set.
BLOCK_OFFSETS = ("OFFSET_OUR_TEAM", "OFFSET_OPP_TEAM", "OFFSET_CONTEXT", "OFFSET_GLOBAL", "OFFSET_REACTIVE",
                 "OFFSET_PAIR_HISTORY", "OFFSET_EVENT_WINDOW", "OFFSET_OBS_FACTS")

#: The NON-integer tables compared besides every shared ``usize`` constant: layout.rs name -> its Python source.
TABLES = {
    "EV_* (EventCol)": "constants.EventCol",
    "EVENT_STATUS_IDS": "constants.EVENT_STATUS_IDS",
    "FACTS_VOL_EFFECTS": "constants.FACTS_VOL_EFFECTS",
    "FACTS_VOL_DURATION": "constants.FACTS_VOL_DURATION",
    "FACTS_SCREENS": "constants.FACTS_SCREENS",
    "FACTS_TURN_NORM": "constants.FACTS_TURN_NORM",
    "VOLATILE_SLOTS": "gen3_effects.VOLATILE_SLOTS",
    "CANT_REASONS_LIVE": "gen3_effects.CANT_REASONS_LIVE",
    "FAINT_CAUSE_VOCAB": "faint-cause vocabulary (FAINT_CAUSE_VOCAB_LIVE)",
    "TYPE_TO_IDX": "types.TypeEncoder.TYPE_TO_IDX",
    "CONDITION_STATUS_IDX": "pokemon._STATUS_STR_IDX",
}

_USIZE = re.compile(r"^pub const ([A-Z0-9_]+): usize = (\d+);$", re.M)
_F64 = re.compile(r"^pub const ([A-Z0-9_]+): f64 = ([-0-9.eE+]+);$", re.M)


def _text() -> str:
    return LAYOUT_RS.read_text()


def _usize_consts(text: str) -> Dict[str, int]:
    return {k: int(v) for k, v in _USIZE.findall(text)}


def _static_body(text: str, name: str) -> str:
    m = re.search(rf"^pub static {name}: \[[^\]]*\] = \[(.*?)\];$", text, re.M | re.S)
    assert m, f"layout.rs has no `pub static {name}` table"
    return m.group(1)


def _str_table(text: str, name: str) -> List[str]:
    return re.findall(r'"((?:[^"\\]|\\.)*)"', _static_body(text, name))


def _pairs(text: str, name: str) -> List[Tuple[str, int]]:
    return [(k, int(v)) for k, v in re.findall(r'\("([^"]*)", (\d+)\)', _static_body(text, name))]


def _shared_usize_names() -> List[str]:
    from agents.observation import constants as C

    rs = _usize_consts(_text())
    return sorted(n for n, v in vars(C).items()
                  if n in rs and isinstance(v, int) and not isinstance(v, bool))


def test_the_compared_set_is_named_and_not_trivial():
    shared = _shared_usize_names()
    missing = [n for n in BLOCK_OFFSETS if n not in shared]
    assert not missing, f"block offsets absent from one side: {missing}"
    assert "OBS_FACTS_DIM" in shared and "TEAM_SIZE" in shared and "POKEMON_FULL_DIM" in shared
    assert len(shared) >= 100, f"only {len(shared)} shared integer constants — the parse is broken"
    for name in TABLES:
        if not name.startswith("EV_"):
            assert re.search(rf"^pub (static|const) {name}\b", _text(), re.M), name


@pytest.mark.parametrize("name", _shared_usize_names())
def test_every_shared_integer_constant_is_equal(name):
    from agents.observation import constants as C

    rs = _usize_consts(_text())[name]
    assert rs == getattr(C, name), (
        f"{name}: layout.rs says {rs}, agents/observation/constants.py says {getattr(C, name)} — the Rust "
        f"encoder and the model disagree on the row's layout. Edit BOTH (layout.rs is Rust-owned source).")


def test_obs_dim_is_the_models_row_length():
    from agents.observation import constants as C
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    text = _text()
    obs_dim = _usize_consts(text)["OBS_DIM"]
    assert obs_dim == C.OFFSET_OBS_FACTS + C.OBS_FACTS_DIM == Gen3ObservationEncoder(load_mappings()).dimension
    assert f'\\"shape\\":[{obs_dim}]' in text, "FRAME_HEAD does not spell OBS_DIM"


def test_the_event_columns_are_equal():
    from agents.observation import constants as C

    rs = {k[len("EV_"):]: v for k, v in _usize_consts(_text()).items() if k.startswith("EV_")}
    assert rs == {c.name: int(c) for c in C.EventCol}


def test_the_constants_tables_are_equal():
    from agents.observation import constants as C

    text = _text()
    assert dict(_pairs(text, "EVENT_STATUS_IDS")) == C.EVENT_STATUS_IDS
    assert tuple(_str_table(text, "FACTS_VOL_EFFECTS")) == tuple(C.FACTS_VOL_EFFECTS)
    assert tuple(_str_table(text, "FACTS_SCREENS")) == tuple(C.FACTS_SCREENS)
    assert dict(_F64.findall(text))["FACTS_TURN_NORM"] and \
        float(dict(_F64.findall(text))["FACTS_TURN_NORM"]) == float(C.FACTS_TURN_NORM)
    dur = re.findall(r"\((-?\d+), (-?\d+), (true|false)\)", _static_body(text, "FACTS_VOL_DURATION"))
    assert [(int(a), int(b), c == "true") for a, b, c in dur] == \
        [tuple(C.FACTS_VOL_DURATION[k]) for k in C.FACTS_VOL_EFFECTS]


def test_the_model_read_vocabularies_are_equal():
    from agents.battle.faint_causes import FAINT_CAUSE_VOCAB_LIVE
    from agents.observation import gen3_effects as E
    from agents.observation import pokemon as PK
    from agents.observation.types import TypeEncoder

    text = _text()
    assert tuple(_str_table(text, "VOLATILE_SLOTS")) == tuple(E.VOLATILE_SLOTS)
    assert tuple(_str_table(text, "CANT_REASONS_LIVE")) == tuple(E.CANT_REASONS_LIVE)
    assert tuple(_str_table(text, "FAINT_CAUSE_VOCAB")) == tuple(FAINT_CAUSE_VOCAB_LIVE)
    assert dict(_pairs(text, "TYPE_TO_IDX")) == dict(TypeEncoder.TYPE_TO_IDX)
    assert dict(_pairs(text, "CONDITION_STATUS_IDX")) == dict(PK._STATUS_STR_IDX)


def test_the_gate_has_teeth(monkeypatch):
    """A one-cell disagreement fails: a perturbed parse of a shared constant is caught."""
    from agents.observation import constants as C

    name = "OFFSET_REACTIVE"
    perturbed = _text().replace(f"pub const {name}: usize = {C.OFFSET_REACTIVE};",
                                f"pub const {name}: usize = {C.OFFSET_REACTIVE + 1};")
    assert perturbed != _text()
    assert _usize_consts(perturbed)[name] != getattr(C, name)


def test_the_header_says_rust_owned():
    head = "\n".join(_text().splitlines()[:6])
    assert "Rust-OWNED" in head and "@generated" not in head
