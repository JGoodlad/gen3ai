"""The PER-SIDE oracle reveal's rules (``main.h2h.reveal``): which (p1, p2) levels a cell plays at under each mode,
the typed refusals, the row stamp, the protocol per mode, and the spec's per-side form. Pure: no engine, no game
(the real games: ``play_reveal_integration_test.py``; the Rust rows: ``rust_env/tests/oracle_reveal_test.rs``)."""
from __future__ import annotations

import json

import pytest

from agents.training import eval_ledger as L
from main.h2h import play as PL
from main.h2h import reveal as RV
from utils.rust_env import protocol as P

TS = "ts:" + "0" * 16


@pytest.mark.parametrize("mode,player,opponent,want", [
    ("off", "off", "off", ("off", "off")),
    ("one_sided", "species", "off", ("species", "off")),
    ("one_sided", "off", "species", ("off", "species")),
    ("one_sided", "full", "off", ("full", "off")),
    ("one_sided", "off", "full", ("off", "full")),
    ("both_sided", "species", "off", ("species", "species")),
    ("both_sided", "off", "full", ("full", "full")),
])
def test_the_cell_levels_follow_the_mode_and_the_recorded_levels(mode, player, opponent, want):
    assert RV.side_levels(mode, player, opponent) == want
    RV.check_side_levels(want, mode, player, opponent)          # what the mode requires is accepted


@pytest.mark.parametrize("mode,player,opponent,match", [
    ("off", "species", "off", "one_sided"),             # an oracle checkpoint under off: told which mode to declare
    ("off", "off", "full", "one_sided"),
    ("one_sided", "off", "off", "needs an ORACLE"),     # a reveal mode with no oracle side
    ("both_sided", "off", "off", "needs an ORACLE"),
    ("one_sided", "species", "full", "both sides are oracle"),
    ("both_sided", "full", "full", "both sides are oracle"),
])
def test_a_mode_inconsistent_with_the_checkpoints_is_a_typed_refusal(mode, player, opponent, match):
    with pytest.raises(RV.RevealModeError, match=match):
        RV.side_levels(mode, player, opponent)


def test_an_unknown_mode_is_refused():
    with pytest.raises(RV.RevealModeError, match="not one of"):
        RV.side_levels("sideways", "species", "off")


@pytest.mark.parametrize("levels,mode,player,opponent,match", [
    (("full", "off"), "one_sided", "species", "off", "trained under --oracle-reveal species"),   # told more
    (("off", "off"), "one_sided", "species", "off", "trained under --oracle-reveal species"),    # told nothing
    (("off", "species"), "one_sided", "off", "full", "trained under --oracle-reveal full"),       # the p2 oracle
    (("species", "species"), "one_sided", "species", "off", "not what mode one_sided requires"),  # non-oracle told
    (("species", "off"), "both_sided", "species", "off", "not what mode both_sided requires"),    # non-oracle not told
    (("species", "full"), "both_sided", "species", "off", "not what mode both_sided requires"),
])
def test_a_side_at_the_wrong_level_is_refused(levels, mode, player, opponent, match):
    with pytest.raises(RV.RevealLevelMismatch, match=match):
        RV.check_side_levels(levels, mode, player, opponent)


def test_each_mode_is_its_own_closed_list_protocol_and_off_is_the_h2h_protocol_unchanged():
    assert RV.PROTOCOL_OF_MODE["off"] == PL.PROTOCOL == "gen3_eval_protocol_v1_h2h"
    assert set(RV.PROTOCOL_OF_MODE.values()) <= set(L.PROTOCOLS)
    assert len(set(RV.PROTOCOL_OF_MODE.values())) == len(RV.MODES)
    regimes = {m: PL.regime_for(300, TS, mode=m) for m in RV.MODES}
    assert regimes["off"] == PL.regime_for(300, TS), "the default regime IS off's"
    assert len({r["regime_id"] for r in regimes.values()}) == 3, "the modes never pool: three regimes"
    for m, r in regimes.items():
        assert r["protocol"] == RV.PROTOCOL_OF_MODE[m]
        assert {k: v for k, v in r.items() if k not in ("protocol", "regime_id")} == \
            {k: v for k, v in regimes["off"].items() if k not in ("protocol", "regime_id")}, "only the protocol differs"


def test_the_row_stamp_is_absent_under_off_and_names_both_levels_otherwise():
    assert RV.row_block("off", RV.OFF_OFF) is None
    assert RV.row_block("one_sided", ("off", "full")) == {"mode": "one_sided", "player": "off", "opponent": "full"}


def test_the_spec_writes_a_symmetric_reveal_as_the_string_and_a_split_one_as_the_pair():
    kw = dict(n=1, threads=1, teams=["X"], names=("a", "b"), turn_limit=300, refusal_budget=0, bank_dir=None)
    assert json.loads(P.spec_json(**kw))["oracle_reveal"] == "off"
    assert P.spec_json(**kw) == P.spec_json(**kw, oracle_reveal=("off", "off")), "off/off is the spec of before"
    assert json.loads(P.spec_json(**kw, oracle_reveal="species"))["oracle_reveal"] == "species"
    assert json.loads(P.spec_json(**kw, oracle_reveal=("species", "off")))["oracle_reveal"] == ["species", "off"]
    assert json.loads(P.spec_json(**kw, oracle_reveal=("full", "full")))["oracle_reveal"] == "full"
    for bad in ("x", ("off",), ("off", "x"), ("off", "off", "off")):
        with pytest.raises(ValueError, match="oracle_reveal"):
            P.spec_json(**kw, oracle_reveal=bad)
