"""``main.h2h --panel`` (B-D7, module ``main.h2h.panel``) WITHOUT an engine: the regime a panel read is planned and
stamped under, the refusals, and the pool path left exactly as it was. The validator is stubbed where a test needs a
builder (no Node); the real pack and a real game are ``play_panel_integration_test.py``."""
from __future__ import annotations

import json
import re

import pytest

from agents.training import eval_ledger as L
from main import heldout_panels as HP
from main.h2h import cli as CLI
from main.h2h import panel as PN
from main.h2h import play as PL

TURNS = 250
POOL_TS = "ts:0123456789abcdef"


@pytest.fixture
def no_pool_read(monkeypatch):
    """The pool's team-set id without reading the pool (the pool path's regime is otherwise unchanged)."""
    monkeypatch.setattr(PL, "eval_team_set", lambda team_pool="repo": POOL_TS)


@pytest.fixture
def fresh_registry():
    HP.load_registry.cache_clear()
    HP.panel_team_texts.cache_clear()
    yield
    HP.load_registry.cache_clear()
    HP.panel_team_texts.cache_clear()


def test_without_a_panel_the_plan_is_the_pool_regime_byte_for_byte(no_pool_read):
    for mode in ("off", "one_sided"):
        assert PL.plan_regime(TURNS, mode, None) == PL.regime_for(TURNS, mode=mode)
    assert PL.plan_regime(TURNS)["team_set"] == POOL_TS
    assert PL.panel_builder(None) is None


def test_a_panel_regime_stamps_the_registered_content_digest(no_pool_read):
    ent = HP.panel_entry("home72")
    reg = PL.plan_regime(TURNS, "off", "home72")
    pool = PL.plan_regime(TURNS, "off", None)
    assert re.fullmatch(r"ts:[0-9a-f]{16}", reg["team_set"])          # the v2 schema's team_set form
    assert reg["team_set"] != pool["team_set"] and reg["regime_id"] != pool["regime_id"]
    assert ent["content_digest"] in reg["team_source"] and "home72" in reg["team_source"]
    assert reg["regime_id"] == L.with_regime_id(reg)["regime_id"]
    # everything but the team set and its label is the pool regime's (the game protocol is unchanged)
    same = {k: v for k, v in reg.items() if k not in ("team_set", "team_source", "regime_id")}
    assert same == {k: v for k, v in pool.items() if k not in ("team_set", "team_source", "regime_id")}
    assert reg["protocol"] == PL.PROTOCOL


def test_each_panel_is_its_own_regime():
    sets = {name: PN.panel_regime(TURNS, name)["team_set"] for name in ("home72", "away", "forum72", "p72",
                                                                         "heldout72_hpfix")}
    assert len(set(sets.values())) == len(sets)


def test_an_unregistered_panel_is_refused_naming_the_registered_ones():
    with pytest.raises(PN.PanelRefused, match="not a registered panel.*home72"):
        PL.plan_regime(TURNS, "off", "home73")
    with pytest.raises(PN.PanelRefused):
        PN.check_panel("home73")
    assert PN.check_panel(None) is None


def test_the_cli_refuses_an_unregistered_panel_before_resolving_any_player(monkeypatch):
    def boom(_spec):
        raise AssertionError("a player was resolved before the panel was checked")

    monkeypatch.setattr(PL, "resolve_player", boom)
    for argv in (["play", "--player", "a.zip", "--opponent", "b.zip", "--pairs", "1", "--panel", "nope"],
                 ["play-many", "--players", "a.zip", "--opponents", "b.zip", "--pairs", "1", "--panel", "nope"]):
        with pytest.raises(PN.PanelRefused, match="nope"):
            CLI.main(argv)


def test_a_panel_whose_registered_digest_moved_is_refused(tmp_path, monkeypatch, fresh_registry):
    reg = json.loads(HP.registry_path().read_text())
    reg["panels"]["home72"]["content_digest"] = "0" * 64
    p = tmp_path / "heldout_panels.json"
    p.write_text(json.dumps(reg))
    monkeypatch.setattr(HP, "registry_path", lambda: p)
    with pytest.raises(PN.PanelRefused, match="not the registered panel"):
        PL.plan_regime(TURNS, "off", "home72")


def _validator(invalid=()):
    def fake(fmt, teams):
        return [{"valid": i not in invalid, "errors": ["stub"] if i in invalid else []} for i in range(len(teams))]
    return fake


def test_the_builder_draws_the_whole_panel_and_stamps_the_planned_regime(monkeypatch):
    from utils.bridge import team_validator

    monkeypatch.setattr(team_validator, "validate_teams_locally", _validator())
    tb = PN.PanelTeams("home72")
    assert len(tb.packed_teams) == HP.panel_entry("home72")["n"] == 72
    assert tb.team_set_id == PN.panel_team_set_id("home72")
    assert PL.builder_regime(TURNS, tb) == PN.panel_regime(TURNS, "home72")
    assert not getattr(tb, "bias_packed_teams", None)                 # no bias: a uniform draw from the panel
    # the eval core's keyed draw (re-seeded per game key and side) only ever yields a panel team
    from agents.training.rust_eval import seeds as SD

    packed = set(tb.packed_teams)
    assert all(SD.draw_team(tb, (7, "h2h", g), side) in packed for g in range(40) for side in (SD.TRAINEE,
                                                                                                SD.OPPONENT))


def test_a_panel_the_validator_does_not_pack_whole_is_refused(monkeypatch):
    from utils.bridge import team_validator

    monkeypatch.setattr(team_validator, "validate_teams_locally", _validator(invalid={5}))
    with pytest.raises(PN.PanelRefused, match="1 of 72 teams fail"):
        PN.PanelTeams("home72")
