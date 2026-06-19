"""StrictBattleView + LegalActions tests.

The contract under test:
  1. StrictBattleView exposes ONLY the vetted surfaces (.live / .turn_view(n) / .history /
     .legal / .events_since / .event_cursor + scalar meta incl. .turn:int) and hides the
     raw battle.
  2. Any other attribute access raises a helpful AttributeError naming the boundary —
     the static "lock" (Phase 4) builds on this runtime guarantee.
  3. LegalActions mirrors poke-env's server-parsed request fields faithfully (move slots
     with pp/disabled, switches with team slots, force_switch/trapped/wait/struggle).
"""

import logging
from types import MappingProxyType, SimpleNamespace

import pytest

from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions, LiveView
from agents.battle.strict_view import StrictBattleView
from agents.battle.turn_view import TurnView

LOG = logging.getLogger("strict-view-test")


def _battle(extra=None):
    b = Gen3Battle("battle-gen3ou-strict", "p1user", LOG, gen=3)
    base = [
        ["", "player", "p1", "p1user", "", ""],
        ["", "player", "p2", "p2user", "", ""],
        ["", "teamsize", "p1", "6"],
        ["", "teamsize", "p2", "6"],
        ["", "gen", "3"],
        ["", "start"],
        ["", "switch", "p1a: Zappy", "Zapdos, L100", "100/100"],
        ["", "switch", "p2a: Tyra", "Tyranitar, L100, M", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Tyra"],
        ["", "-damage", "p2a: Tyra", "60/100"],
        ["", "turn", "2"],
    ]
    for line in base + (extra or []):
        b.parse_message(line)
    return b


# ───────────────────────────── the boundary ────────────────────────────────
def test_strict_view_exposes_only_vetted_surfaces():
    sv = _battle().strict_view()
    assert isinstance(sv, StrictBattleView)
    assert isinstance(sv.live, LiveView)
    assert isinstance(sv.legal, LegalActions)
    assert isinstance(sv.turn_view(2), TurnView)
    assert all(isinstance(t, TurnView) for t in sv.history)
    assert len(sv.history) == 2  # turns 1 and 2 played
    assert isinstance(sv.event_cursor, int)
    assert isinstance(sv.events_since(0), list)


def test_strict_view_meta_passthroughs():
    sv = _battle().strict_view()
    assert sv.battle_tag == "battle-gen3ou-strict"
    assert sv.finished is False
    assert sv.won is None and sv.lost is None
    # `turn` is the scalar current-turn int — same name/meaning as battle.turn / live.turn,
    # so migrating a consumer off battle.turn is a drop-in (turn_view(n) fetches a TurnView)
    assert sv.turn == 2
    assert sv.turn == sv.live.turn == sv._battle.turn
    assert isinstance(sv.turn, int)


_FORBIDDEN_RAW_ATTRS = [
    "active_pokemon", "opponent_active_pokemon", "team", "available_moves",
    "available_switches", "last_request", "parse_message", "events",
    "opponent_team", "side_conditions", "force_switch", "trapped",
]


@pytest.mark.parametrize("attr", _FORBIDDEN_RAW_ATTRS)
def test_strict_view_hides_raw_battle(attr):
    """Every raw battle attribute is unreachable through the boundary; the error names
    the right accessor instead of silently proxying to poke-env."""
    sv = _battle().strict_view()
    with pytest.raises(AttributeError) as ei:
        getattr(sv, attr)
    msg = str(ei.value)
    assert ".live" in msg and ".legal" in msg  # the error guides to the accessors


def test_strict_view_never_returns_raw_battle():
    b = _battle()
    sv = b.strict_view()
    # none of the public surfaces hand back the raw battle object
    assert sv.live is not b
    assert sv.legal is not b
    assert sv.turn_view(1) is not b
    assert all(t is not b for t in sv.history)


# ───────────────────────────── LegalActions ────────────────────────────────
def _stub_battle(*, request, switches, moves, team_species,
                 force_switch=False, trapped=False, maybe_trapped=False, wait=False,
                 active_moves=None):
    """A minimal duck-typed battle exposing exactly the server-parsed fields
    LegalActions.from_battle reads. Pure-logic transform → a stub is the right tool
    (the real server request is exercised by the e2e fuzz). ``active_moves`` (key → Move.id) sets
    the active mon's live moveset so the own-Hidden-Power typed-id resolution can be exercised."""
    mons = {f"p1: {s}": SimpleNamespace(species=s) for s in team_species}
    mon_by_species = {m.species: m for m in mons.values()}
    active = next(iter(mons.values()), None)  # only _own_hp_typed_id reads battle.active_pokemon
    if active is not None:
        active.moves = {k: SimpleNamespace(id=mid) for k, mid in (active_moves or {}).items()}
    return SimpleNamespace(
        last_request=request,
        active_pokemon=active,
        available_switches=[mon_by_species[s] for s in switches],
        available_moves=[SimpleNamespace(id=mid) for mid in moves],
        team=mons,
        force_switch=force_switch,
        trapped=trapped,
        maybe_trapped=maybe_trapped,
        wait=wait,
    )


def test_legal_actions_from_request():
    req = {
        "active": [{
            "moves": [
                {"id": "thunderbolt", "pp": 24, "maxpp": 24, "target": "normal"},
                {"id": "hiddenpowerice", "pp": 3, "maxpp": 24, "disabled": True,
                 "target": "normal"},
            ]
        }],
    }
    stub = _stub_battle(
        request=req,
        switches=["blissey", "skarmory"],
        moves=["thunderbolt", "hiddenpowerice"],
        team_species=["zapdos", "blissey", "skarmory"],
    )
    legal = LegalActions.from_battle(stub)

    assert legal.move_ids == ("thunderbolt", "hiddenpowerice")
    tb, hp = legal.move_slots
    assert (tb.current_pp, tb.max_pp, tb.disabled) == (24, 24, False)
    assert (hp.current_pp, hp.max_pp, hp.disabled) == (3, 24, True)
    assert tb.target == "normal"

    # switches carry the team-slot index the action space uses (0-based team order)
    assert legal.switch_species == ("blissey", "skarmory")
    assert legal.switch_slots == (1, 2)  # zapdos is slot 0 (active), not switchable

    assert legal.force_switch is False and legal.trapped is False
    assert legal.wait is False and legal.struggle is False
    # last_request is a read-only mirror
    assert isinstance(legal.last_request, MappingProxyType)
    with pytest.raises(TypeError):
        legal.last_request["active"] = []  # type: ignore[index]


def test_legal_actions_own_hidden_power_typed_for_display_only():
    """gen3 has no team preview, so the wire request keys OUR Hidden Power BARE ("hiddenpower")
    while the live Move object keeps the IV-derived typed id ("hiddenpowergrass"). move_slots /
    move_ids stay wire-truth (bare) — the mask/mapper/serialization depend on it — while
    display_move_ids surfaces the typed id for human/forensic LABELS (we always know our own HP
    type). own_hp_typed_id carries the resolved id."""
    req = {"active": [{"moves": [
        {"id": "icebeam", "pp": 16, "maxpp": 16},
        {"id": "hiddenpower", "pp": 24, "maxpp": 24},   # BARE on the wire
    ]}]}
    stub = _stub_battle(
        request=req, switches=[], moves=["icebeam", "hiddenpower"],
        team_species=["tyranitar"],
        # live moveset keyed bare but each Move carries the typed id (poke-env's raw_id patch)
        active_moves={"icebeam": "icebeam", "hiddenpower": "hiddenpowergrass"},
    )
    legal = LegalActions.from_battle(stub)
    assert legal.move_ids == ("icebeam", "hiddenpower")          # WIRE-TRUTH stays bare
    assert legal.own_hp_typed_id == "hiddenpowergrass"
    assert legal.display_move_ids == ("icebeam", "hiddenpowergrass")  # typed for the label


def test_legal_actions_no_hidden_power_display_equals_wire():
    """No Hidden Power → own_hp_typed_id is None and display_move_ids is a no-op (== move_ids)."""
    req = {"active": [{"moves": [{"id": "icebeam", "pp": 16, "maxpp": 16}]}]}
    stub = _stub_battle(request=req, switches=[], moves=["icebeam"],
                        team_species=["tyranitar"], active_moves={"icebeam": "icebeam"})
    legal = LegalActions.from_battle(stub)
    assert legal.own_hp_typed_id is None
    assert legal.display_move_ids == legal.move_ids == ("icebeam",)


def test_legal_actions_force_switch_has_no_active_block():
    """On a forced switch the request carries forceSwitch but no 'active' block —
    move_slots must be empty and force_switch True without crashing."""
    stub = _stub_battle(
        request={"forceSwitch": [True]},
        switches=["blissey"],
        moves=[],
        team_species=["zapdos", "blissey"],
        force_switch=True,
    )
    legal = LegalActions.from_battle(stub)
    assert legal.move_slots == ()
    assert legal.move_ids == ()
    assert legal.force_switch is True
    assert legal.switch_species == ("blissey",)


def test_legal_actions_struggle_detected():
    stub = _stub_battle(
        request={"active": [{"moves": [{"id": "struggle", "pp": 1, "maxpp": 1}]}]},
        switches=[],
        moves=["struggle"],
        team_species=["zapdos"],
        trapped=True,
    )
    legal = LegalActions.from_battle(stub)
    assert legal.struggle is True
    assert legal.trapped is True


def test_legal_actions_no_request_is_empty_not_crash():
    stub = _stub_battle(
        request=None, switches=[], moves=[], team_species=["zapdos"],
    )
    legal = LegalActions.from_battle(stub)
    assert legal.move_slots == () and legal.switches == ()
    assert legal.last_request is None
    assert legal.struggle is False and legal.force_switch is False
