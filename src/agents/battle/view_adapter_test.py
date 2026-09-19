"""Unit gate for the ONE-SIDED VIEW adapter (`gen3_one_sided_view_v1`).

The differential gate (``one_sided_view_parity_fuzz_test.py``) drives real battles and
answers "do the two roads agree". THIS file answers the question that one cannot: **is each
poke-env PRESENTATION RULE the adapter re-applies actually the rule poke-env has** — checked
against a HAND-BUILT payload, one rule per test, and against the poke-env objects themselves
rather than against a second copy of the rule.

It also pins the ONE INPUT EACH SUB-ENCODER TAKES from the adapter, because four of them
(``items`` / ``abilities`` / ``types`` / ``moves``) still read a poke-env ``Pokemon`` rather than
a ``live_mon`` — deferral D2 — so what :class:`ViewMon` supplies is a real interface, not an
implementation detail.
"""

from __future__ import annotations

import pytest

from agents.battle.view_adapter import (ViewBattle, ViewMon, legal_actions_from_payload,
                                        live_view_from_payload, read_models_from_payload)


def _mon_row(species, **kw):
    """A payload mon row with the shape `view.rs` emits, defaults filled in."""
    row = {
        "species": species, "active": False, "fainted": False, "revealed": True,
        "hp_fraction": 1.0, "current_hp": 100, "max_hp": 100,
        "status": None, "status_counter": 0, "protect_counter": 0,
        "types": ["normal"], "moves": [], "item": None, "consumed_item": None,
        "ability": None, "ability_events": [], "boosts": {}, "volatiles": [], "base_stats": {
            "hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        "ivs": None, "evs": None, "nature": None, "spread_known": False, "stats": None,
    }
    row.update(kw)
    return row


def _payload(ours_mons, opp_mons, **kw):
    p = {
        "side": "p1", "turn": 5, "finished": False, "won": None, "lost": None,
        "weather": {"weather": None, "is_permanent": False, "turns_active": 0},
        "ours": {"team_size": 6, "active": None, "side_conditions": {}, "mons": ours_mons},
        "opp": {"team_size": 6, "active": None, "side_conditions": {}, "mons": opp_mons},
        "request": None,
    }
    p.update(kw)
    return p


# ---------------------------------------------------------------------------
# Presentation rule 1 — the single-possible-ability inference
# ---------------------------------------------------------------------------

def test_a_species_with_ONE_possible_ability_is_known_without_being_disclosed():
    """``Pokemon._update_from_pokedex`` sets ``_ability`` outright when the dex lists exactly one
    ability and ``gen >= 3`` — public knowledge, not a leak. The port therefore sends only what
    the PROTOCOL disclosed and the inference is applied here."""
    from poke_env.data.gen_data import GenData

    dex = GenData.from_gen(3).pokedex
    assert len(dex["tyranitar"]["abilities"]) == 1, "fixture: Tyranitar must be single-ability"
    assert len(dex["salamence"]["abilities"]) == 1, "fixture: Salamence must be single-ability"
    assert len(dex["skarmory"]["abilities"]) > 1, "fixture: Skarmory must be multi-ability"

    live = live_view_from_payload(_payload(
        [], [_mon_row("tyranitar"), _mon_row("skarmory")]))
    tt = live.opp.get("tyranitar")
    sk = live.opp.get("skarmory")
    assert tt is not None and tt.ability == "sandstream", "the inference must fire"
    assert sk is not None and sk.ability is None, (
        "a multi-ability species stays unknown until the protocol says otherwise")


def test_a_TRACED_ability_reverts_to_the_tracer_when_the_mon_leaves_the_field():
    """poke-env keeps TWO ability slots and the getter prefers the temporary one, which
    ``switch_out`` clears. A Porygon2 that Traces Magnet Pull reads ``magnetpull`` while it is out
    and ``trace`` once it pivots — carrying the copied ability forever was a 14-occurrence class,
    found only on the THIRD fresh seed.

    NOTE the fixture: poke-env's pokedex is NOT gen-filtered, so Porygon2 lists Trace AND Download
    and the single-ability inference does NOT fire. The base slot is therefore filled by the Trace
    special case itself — which is exactly the path that was wrong."""
    from poke_env.data.gen_data import GenData

    assert len(GenData.from_gen(3).pokedex["porygon2"]["abilities"]) > 1, (
        "fixture: poke-env lists Porygon2 with more than one ability, so no inference fires")
    traced = live_view_from_payload(_payload([], [_mon_row(
        "porygon2", ability_events=[{"id": "magnetpull", "trace": True}])]))
    assert traced.opp.get("porygon2").ability == "magnetpull"

    reverted = live_view_from_payload(_payload([], [_mon_row(
        "porygon2", ability_events=[{"id": "magnetpull", "trace": True}, {"id": "", "trace": False}])]))
    assert reverted.opp.get("porygon2").ability == "trace", (
        "a switch-out clears the TEMPORARY slot and the tracer's own ability is what remains")


def test_a_plain_ability_announcement_on_a_MULTI_ability_species_fills_the_base_slot():
    """With no inference to fire, the FIRST announcement is the base — so it survives a
    switch-out, unlike a traced overlay."""
    live = live_view_from_payload(_payload([], [_mon_row(
        "skarmory", ability_events=[{"id": "keeneye", "trace": False},
                                    {"id": "", "trace": False}])]))
    assert live.opp.get("skarmory").ability == "keeneye"


def test_a_DISCLOSED_ability_always_wins_over_the_inference():
    """Our OWN side sends the engine's value in ``ability`` (our ``|request|`` states it)."""
    live = live_view_from_payload(_payload([_mon_row("skarmory", ability="keeneye")], []))
    assert live.ours.get("skarmory").ability == "keeneye"


# ---------------------------------------------------------------------------
# Presentation rule 2 — the announced volatiles
# ---------------------------------------------------------------------------

def test_an_ends_on_turn_volatile_is_DROPPED_once_a_turn_boundary_has_passed():
    """``Pokemon.end_turn`` deletes every ``ends_on_turn`` effect. Focus Punch is the gen-3 one
    that bites: carried forever, a Snorlax reads a volatile poke-env dropped the same turn."""
    from poke_env.battle.effect import Effect

    assert Effect.FOCUS_PUNCH.ends_on_turn, "fixture: Focus Punch must be an ends_on_turn effect"
    fresh = live_view_from_payload(_payload(
        [], [_mon_row("snorlax", volatiles=[{"name": "move: Focus Punch", "turns": 0,
                                             "restarts": 0}])]))
    assert "focuspunch" in fresh.opp.get("snorlax").volatiles
    stale = live_view_from_payload(_payload(
        [], [_mon_row("snorlax", volatiles=[{"name": "move: Focus Punch", "turns": 1,
                                             "restarts": 0}])]))
    assert stale.opp.get("snorlax").volatiles == {}


def test_a_turn_countable_volatile_carries_its_TURN_count_and_the_rest_carry_zero():
    from poke_env.battle.effect import Effect

    assert Effect.TAUNT.is_turn_countable, "fixture: Taunt must be turn-countable"
    assert not Effect.LEECH_SEED.is_turn_countable, "fixture: Leech Seed must not be"
    live = live_view_from_payload(_payload([], [_mon_row(
        "blissey",
        volatiles=[{"name": "move: Taunt", "turns": 2, "restarts": 0},
                   {"name": "move: Leech Seed", "turns": 4, "restarts": 0}])]))
    vol = live.opp.get("blissey").volatiles
    assert vol == {"taunt": 2, "leechseed": 0}


def test_an_action_countable_volatile_carries_its_RESTART_count():
    from poke_env.battle.effect import Effect

    assert Effect.STOCKPILE.is_action_countable, "fixture: Stockpile must be action-countable"
    live = live_view_from_payload(_payload([], [_mon_row(
        "swampert", volatiles=[{"name": "Stockpile", "turns": 1, "restarts": 2}])]))
    assert live.opp.get("swampert").volatiles == {"stockpile": 2}


# ---------------------------------------------------------------------------
# PP — the sighting counter, and the Pressure correction
# ---------------------------------------------------------------------------

def _pp_of(live, side, species, move_id):
    mon = (live.ours if side == "ours" else live.opp).get(species)
    return next(m.current_pp for m in mon.moves if m.id == move_id)


def test_a_WATCHED_move_reports_max_pp_minus_the_sightings():
    live = live_view_from_payload(_payload([], [_mon_row(
        "skarmory", moves=[{"id": "drillpeck", "move_id": "drillpeck", "uses": 3,
                            "max_pp": 32, "uses_vs": {}}])]))
    assert _pp_of(live, "opp", "skarmory", "drillpeck") == 29


def test_a_move_aimed_at_a_PRESSURE_holder_costs_TWO_per_sighting():
    """``Pokemon.moved`` passes ``pressure`` into ``Move.use``, which decrements by 2. Whether it
    fires depends on the ability poke-env KNOWS — inference included — so the payload carries the
    sightings by target and the decision is made against the read-model."""
    live = live_view_from_payload(_payload(
        [_mon_row("zapdos", spread_known=True)],   # gen3 Zapdos: Pressure, inferred
        [_mon_row("skarmory", moves=[{"id": "drillpeck", "move_id": "drillpeck", "uses": 3,
                                      "max_pp": 32, "uses_vs": {"zapdos": 3}}])]))
    assert live.ours.get("zapdos").ability == "pressure", "fixture: the inference must fire"
    assert _pp_of(live, "opp", "skarmory", "drillpeck") == 26


def test_a_SELF_targeting_move_is_never_pressured():
    """``_pressure_on`` gates on the move's ``target`` — Swords Dance (``self``) is exempt even
    when the foe holds Pressure, which is what makes the rule a gate rather than a multiplier."""
    live = live_view_from_payload(_payload(
        [_mon_row("zapdos")],
        [_mon_row("heracross", moves=[{"id": "swordsdance", "move_id": "swordsdance", "uses": 2,
                                       "max_pp": 48, "uses_vs": {"zapdos": 2}}])]))
    assert _pp_of(live, "opp", "heracross", "swordsdance") == 46


def test_an_OWN_move_reports_the_engines_pp_verbatim():
    """Our own side sends ``current_pp`` — the wire's own number, which poke-env asserts its
    counter against (``check_move_consistency``). See deferral D7."""
    live = live_view_from_payload(_payload(
        [_mon_row("jirachi", moves=[{"id": "wish", "move_id": "wish", "current_pp": 11,
                                     "max_pp": 16}])], []))
    assert _pp_of(live, "ours", "jirachi", "wish") == 11


# ---------------------------------------------------------------------------
# The read-model shapes the encoders actually consume
# ---------------------------------------------------------------------------

def test_an_opponents_stats_dict_is_all_None_not_empty():
    """``Pokemon._stats`` is seeded with six ``None`` keys and only OUR request ever fills it, so
    ``dict(mon.stats) if mon.stats else {}`` keeps the None-valued dict — truthy. An empty dict
    would be a different object and the incoming-damage belief reads it."""
    live = live_view_from_payload(_payload([], [_mon_row("snorlax")]))
    assert live.opp.get("snorlax").stats == {
        "hp": None, "atk": None, "def": None, "spa": None, "spd": None, "spe": None}


def test_the_moves_tuple_is_sorted_by_the_poke_env_DICT_KEY():
    """``LiveView.moves`` is sorted by the moves-dict key, and a typed Hidden Power sits under the
    BARE key — so it sorts as ``hiddenpower``, not as ``hiddenpowerice``."""
    live = live_view_from_payload(_payload([_mon_row("zapdos", moves=[
        {"id": "thunderbolt", "move_id": "thunderbolt", "current_pp": 24, "max_pp": 24},
        {"id": "hiddenpower", "move_id": "hiddenpowerice", "current_pp": 24, "max_pp": 24},
        {"id": "batonpass", "move_id": "batonpass", "current_pp": 64, "max_pp": 64}])], []))
    assert [m.id for m in live.ours.get("zapdos").moves] == [
        "batonpass", "hiddenpower", "thunderbolt"]


def test_the_ViewMon_moveset_restores_the_TYPED_hidden_power_the_wire_hid():
    """``MovesEncoder`` reads ``Move.id`` (deferral D2) and the dex row behind it, so a typed HP
    must come back typed — a bare one encodes dex num 237 (type unknown) instead of the real
    typed num, which is a silently wrong move-slot id for one of our OWN mons."""
    live = live_view_from_payload(_payload([_mon_row("zapdos", moves=[
        {"id": "hiddenpower", "move_id": "hiddenpowerice", "current_pp": 24, "max_pp": 24}])], []))
    payload = _payload([_mon_row("zapdos", moves=[
        {"id": "hiddenpower", "move_id": "hiddenpowerice", "current_pp": 24, "max_pp": 24}])], [])
    vb = ViewBattle(live, None, payload)
    mv = vb.team["zapdos"].moves["hiddenpower"]
    assert mv.id == "hiddenpowerice", f"the typed id was lost: {mv.id}"
    assert mv.current_pp == 24


def test_a_MIRROR_match_does_not_let_one_sides_row_overwrite_the_others():
    """Both teams running the same species is ordinary in gen3ou. Keyed by species over BOTH
    sides at once, the opponent's row (whose Hidden Power is unknowable, so BARE) replaced ours —
    and our own mon's move-slot id silently became the bare one."""
    ours = _mon_row("metagross", moves=[
        {"id": "hiddenpower", "move_id": "hiddenpowerfire", "current_pp": 24, "max_pp": 24}])
    opp = _mon_row("metagross", moves=[
        {"id": "hiddenpower", "move_id": "hiddenpower", "uses": 1, "max_pp": 24, "uses_vs": {}}])
    payload = _payload([ours], [opp])
    live, _, vb = read_models_from_payload(payload)
    assert vb.team["metagross"].moves["hiddenpower"].id == "hiddenpowerfire"
    assert vb.opponent_team["metagross"].moves["hiddenpower"].id == "hiddenpower"


def test_the_ViewMon_supplies_exactly_the_four_D2_encoder_inputs():
    """``items`` / ``abilities`` / ``types`` / ``moves`` read the raw mon. Each field below is a
    direct copy of its ``LivePokemon`` counterpart — assert that, so a future field that is
    DERIVED here instead of copied shows up as a failure."""
    from poke_env.battle.pokemon_type import PokemonType

    row = _mon_row("skarmory", types=["steel", "flying"], item="leftovers",
                   consumed_item=None, ability="keeneye")
    live = live_view_from_payload(_payload([], [row]))
    lm = live.opp.get("skarmory")
    vm = ViewMon.of(lm, row)
    assert (vm.species, vm.active, vm.fainted) == (lm.species, lm.active, lm.fainted)
    assert vm.item == lm.item and vm.consumed_item == lm.consumed_item
    assert vm.ability == lm.ability
    assert (vm.type_1, vm.type_2) == (PokemonType.STEEL, PokemonType.FLYING)


def test_an_itemless_OWN_mon_reads_the_empty_string_and_an_unrevealed_one_reads_None():
    """poke-env takes an own item straight from the request, where "no item" is ``""``; only the
    ``unknownitem`` SENTINEL becomes ``None``. Both spell "encode zeros", but the read-models are
    compared field by field."""
    live = live_view_from_payload(_payload([_mon_row("skarmory", item="")],
                                           [_mon_row("snorlax", item=None)]))
    assert live.ours.get("skarmory").item == ""
    assert live.opp.get("snorlax").item is None


# ---------------------------------------------------------------------------
# LegalActions
# ---------------------------------------------------------------------------

_REQUEST = {
    "active": [{"moves": [
        {"move": "Drill Peck", "id": "drillpeck", "pp": 30, "maxpp": 32, "target": "any",
         "disabled": False},
        {"move": "Spikes", "id": "spikes", "pp": 32, "maxpp": 32, "target": "foeSide",
         "disabled": True}]}],
    "side": {"name": "P1", "id": "p1", "pokemon": [
        {"ident": "p1: Skarmory", "details": "Skarmory, M", "condition": "100/100",
         "active": True, "moves": ["drillpeck", "spikes"]},
        {"ident": "p1: Zapdos", "details": "Zapdos", "condition": "0 fnt", "active": False,
         "moves": ["thunderbolt", "hiddenpowerice"]},
        {"ident": "p1: Snorlax", "details": "Snorlax, M", "condition": "300/400",
         "active": False, "moves": ["bodyslam"]}]},
}


def _legal_payload(**req):
    r = {k: v for k, v in _REQUEST.items()}
    r.update(req)
    live = live_view_from_payload(_payload(
        [_mon_row("skarmory", active=True), _mon_row("zapdos", fainted=True),
         _mon_row("snorlax")], []))
    return legal_actions_from_payload(_payload([], [], request=r), live), live


def test_the_switch_SLOT_is_the_stable_team_index_not_the_requests_own_order():
    """🚨 Showdown floats the ACTIVE mon to ``side.pokemon[0]`` on every request, while poke-env's
    ``battle.team`` dict is fixed at the FIRST one — and ``LegalSwitch.slot`` indexes the latter,
    which is what the 11-dim action space maps onto. Here the request lists Skarmory first
    (active) while the stable order is Skarmory, Zapdos, Snorlax: Snorlax must be slot 2."""
    legal, _ = _legal_payload()
    assert legal.switch_species == ("snorlax",), "the FAINTED Zapdos is not switchable"
    assert legal.switch_slots == (2,)


def test_a_disabled_slot_is_kept_at_its_request_position():
    legal, _ = _legal_payload()
    assert [m.id for m in legal.move_slots] == ["drillpeck", "spikes"]
    assert [m.disabled for m in legal.move_slots] == [False, True]
    assert legal.move_slots[0].current_pp == 30 and legal.move_slots[0].max_pp == 32


def test_struggle_is_the_FLAG_and_never_a_move_slot():
    """The server sends a lone ``struggle`` entry when all PP is gone; keeping it in
    ``move_slots`` re-introduces the historical struggle double-enabling bug."""
    legal, _ = _legal_payload(active=[{"moves": [
        {"move": "Struggle", "id": "struggle", "pp": 1, "maxpp": 1, "target": "randomNormal",
         "disabled": False}]}])
    assert legal.move_slots == () and legal.struggle is True


def test_a_trapped_side_offers_no_switches():
    legal, _ = _legal_payload(active=[{"moves": [], "trapped": True}])
    assert legal.trapped is True and legal.switches == ()


def test_maybe_trapped_and_force_switch_and_wait_come_off_the_request():
    legal, _ = _legal_payload(active=[{"moves": [], "maybeTrapped": True}],
                              forceSwitch=[True], wait=True)
    assert (legal.maybe_trapped, legal.force_switch, legal.wait) == (True, True, True)


def test_the_own_typed_hidden_power_is_read_off_the_ACTIVE_roster_row():
    """The ``active`` block re-keys our HP bare; the ROSTER row carries the typed id
    (``gen3_own_typed_hp_request_roster_v1``). OURS only — the roster IS our side."""
    legal, _ = _legal_payload(side={"name": "P1", "id": "p1", "pokemon": [
        {"ident": "p1: Zapdos", "details": "Zapdos", "condition": "100/100", "active": True,
         "moves": ["thunderbolt", "hiddenpowerice"]}]})
    assert legal.own_hp_typed_id == "hiddenpowerice"


def test_no_outstanding_request_is_None_and_never_an_empty_legality():
    """A side with no open boundary has NO decision. An empty ``LegalActions`` would read as an
    all-zero mask, which the live player treats as a DEFERRED decision rather than a forfeit —
    a different thing, and the caller must be able to tell them apart."""
    assert legal_actions_from_payload(_payload([], [])) is None


# ---------------------------------------------------------------------------
# The shim's own contract
# ---------------------------------------------------------------------------

def test_the_view_battle_REFUSES_an_attribute_the_payload_cannot_answer():
    """A poke-env field the encoder starts reading must fail LOUDLY here, naming the contract —
    never silently return a default that encodes as a plausible zero."""
    live = live_view_from_payload(_payload([_mon_row("skarmory", active=True)], []))
    vb = ViewBattle(live, None, None)
    with pytest.raises(AttributeError, match="one_sided_view"):
        _ = vb.last_request
    with pytest.raises(AttributeError, match="one_sided_view"):
        _ = vb.strict_view().events_since


def test_the_opponent_team_ORDER_is_the_payloads_and_never_re_sorted():
    """``get_team_list`` walks ``battle.opponent_team`` and that order IS the obs slot
    assignment — poke-env's dict is in REVEAL order, which the port emits."""
    live = live_view_from_payload(_payload(
        [], [_mon_row("swampert"), _mon_row("metagross"), _mon_row("celebi")]))
    vb = ViewBattle(live, None, None)
    assert list(vb.opponent_team) == ["swampert", "metagross", "celebi"]
