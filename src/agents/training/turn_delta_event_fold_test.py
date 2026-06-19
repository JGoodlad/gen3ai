"""
Unit tests for Step 4 — TurnDelta.build_from_events():

  - attempted action (move id / switch species), preserved even when the move
    never fired (faint, cant)
  - multi-KO faint counts + cause multi-hots
  - missing → 0 (no-event window, empty-delta)
  - encoder round-trip for new faint + attempted fields
"""
from __future__ import annotations
import itertools
import numpy as np
import pytest

from agents.battle.battle_event import OPP, OURS, BattleEvent, EventKind
from agents.battle.live_view import LegalActions
from agents.battle.turn_view import FAINT_CAUSE_DIM, FAINT_CAUSE_VOCAB
from agents.gen3_mechanics import BOOST_DIM
from agents.observation.turn_delta_encoder import (
    TurnDeltaEncoder,
    OFFSET_OUR_FAINT_CAUSES,
    OFFSET_OPP_FAINT_CAUSES,
    OFFSET_OUR_ATTEMPTED_MOVE,
    OFFSET_OUR_ACTOR_SPECIES,
    FAINT_CAUSE_VOCAB as _ENC_CAUSE_VOCAB,
)
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_seq = itertools.count()


def ev(kind, *, side=None, actor=None, target=None, turn=1, **value):
    return BattleEvent(
        seq=next(_seq),
        turn=turn,
        kind=kind,
        side=side,
        actor_species=actor,
        target_species=target,
        value=value,
    )


_MOVES = {
    "thunderbolt": {"num": 85, "basePower": 95, "type": "Electric",
                    "hasSecondary": True, "hasRecoil": False},
    "rockslide":   {"num": 100, "basePower": 75, "type": "Rock",
                    "hasSecondary": True, "hasRecoil": False},
    "explosion":   {"num": 153, "basePower": 250, "type": "Normal",
                    "hasSecondary": False, "hasRecoil": False},
}
_SPECIES = {
    "zapdos":    {"num": 145},
    "tyranitar": {"num": 248},
    "forretress": {"num": 205},
    "blissey":   {"num": 242},
    "skarmory":  {"num": 227},
}


def _enc():
    return TurnDeltaEncoder(_MOVES, _SPECIES)


def _zero_ctx(**kwargs):
    """Minimal BattleContext (all-zero HP, empty maps)."""
    defaults = dict(
        turn=1,
        phase="move_selection",
        mask=np.ones(11, dtype=np.int8),
        our_slot_map={},
        opp_slot_map={},
        our_hp=np.zeros(6, dtype=np.float32),
        opp_hp=np.zeros(6, dtype=np.float32),
        our_active="zapdos",
        opp_active="tyranitar",
        our_fainted_count=0,
        opp_fainted_count=1,
        active_move_ids=["thunderbolt", "icebeam", "rest", "sleeptalk"],
        opp_last_move_id=None,
        opp_all_last_move_ids={},
        opp_active_revealed_moves=frozenset(),
        our_cant_reason=None,
        opp_cant_reason=None,
        our_boosts=np.zeros(BOOST_DIM, dtype=np.int8),
        opp_boosts=np.zeros(BOOST_DIM, dtype=np.int8),
        our_last_effectiveness=None,
        opp_last_effectiveness=None,
        we_moved_first=None,
        our_team_order=("zapdos", "skarmory"),
    )
    defaults.update(kwargs)
    return BattleContext(**defaults)


# ---------------------------------------------------------------------------
# Attempted action
# ---------------------------------------------------------------------------

def test_attempted_move_preserved_when_move_fires():
    """Action index 6 → move slot 0 → thunderbolt; event confirms it fired."""
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar",
           move_id="thunderbolt"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.5),
    ]
    prev = _zero_ctx()
    curr = _zero_ctx(turn=2)
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    assert d.our_attempted_move_id == "thunderbolt"


def test_attempted_move_preserved_when_faint_before_act():
    """Action index 6 → thunderbolt; but opp KO'd us before we acted."""
    events = [
        ev(EventKind.MOVE, side=OPP, actor="tyranitar", target="zapdos",
           move_id="rockslide"),
        ev(EventKind.DAMAGE, side=OURS, actor="zapdos", amount=-1.0),
        ev(EventKind.FAINT, side=OURS, actor="zapdos"),
    ]
    prev = _zero_ctx()
    curr = _zero_ctx(turn=2, our_fainted_count=1, our_active="NONE")
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    # We pressed thunderbolt (action 6) but our mon fainted before it fired.
    assert d.our_attempted_move_id == "thunderbolt"
    assert d.our_move_id is None          # no MOVE event for our side


def _legal_with_own_hp(typed_id="hiddenpowergrass"):
    """A minimal LegalActions snapshot carrying the typed own-HP id (what from_battle resolves)."""
    return LegalActions(move_slots=(), switches=(), force_switch=False, trapped=False,
                        maybe_trapped=False, wait=False, struggle=False, last_request=None,
                        own_hp_typed_id=typed_id)


def test_our_hidden_power_folds_typed_from_legal_snapshot():
    """Our OWN Hidden Power folds with its TYPED id. The wire `active` block (→ active_move_ids) and
    the protocol |move| line both report it BARE, but the decision-time LegalActions snapshot carries
    the typed id, so the history's move-TYPE channel encodes our real HP type (vs the type-0 'unknown'
    sentinel that is correct only for the opponent)."""
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="hiddenpower"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.3),
    ]
    prev = _zero_ctx(active_move_ids=["hiddenpower", "thunderbolt", "rest", "sleeptalk"],
                     legal=_legal_with_own_hp("hiddenpowergrass"))
    curr = _zero_ctx(turn=2)
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    assert d.our_move_id == "hiddenpowergrass"            # the move that fired
    assert d.our_attempted_move_id == "hiddenpowergrass"  # the move we pressed (slot 0)


def test_hidden_power_stays_bare_without_snapshot_and_opp_never_typed():
    """No own-HP typed id (legal=None) → our HP stays bare; an opponent's HP is NEVER typed — the
    fold only resolves OUR move id, from OUR own snapshot, so there is no opponent-info leak."""
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="hiddenpower"),
        ev(EventKind.MOVE, side=OPP, actor="tyranitar", target="zapdos", move_id="hiddenpower"),
        ev(EventKind.DAMAGE, side=OURS, actor="zapdos", amount=-0.3),
    ]
    prev = _zero_ctx(active_move_ids=["hiddenpower", "thunderbolt", "rest", "sleeptalk"], legal=None)
    curr = _zero_ctx(turn=2)
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    assert d.our_move_id == "hiddenpower"   # no typed id available → unchanged
    assert d.opp_move_id == "hiddenpower"   # opponent HP always bare


def test_typed_hidden_power_encodes_real_type_not_unknown():
    """End-to-end: a typed own-HP fold ENCODES our HP with its DISTINCT num + real type
    (gen3_typed_hidden_power_ids_v1: hiddenpowergrass → num 363, type GRASS), where the OPPONENT's
    bare HP stays num 237 / type-0 unknown."""
    from agents.gen3_data import moves as _g3moves
    enc = TurnDeltaEncoder(dict(_g3moves.raw()), _SPECIES)
    grass_num = _g3moves.get("hiddenpowergrass").num   # 363 (distinct, our known HP)
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="hiddenpower"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.3),
    ]
    prev = _zero_ctx(active_move_ids=["hiddenpower", "thunderbolt", "rest", "sleeptalk"],
                     legal=_legal_with_own_hp("hiddenpowergrass"))
    curr = _zero_ctx(turn=2)
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    assert d.our_move_id == "hiddenpowergrass"
    desc = enc.describe_vector(enc.encode(d))
    assert desc["our_move"]["move_id"] == grass_num != 237   # OUR HP gets its distinct num (363)
    assert desc["our_move"]["type_id"] == 4                  # GRASS — the real type, not 0 (unknown)


def test_pressed_switch_has_no_attempted_move():
    """A pressed switch (action < 6) sets no attempted_move — switches always
    execute, so the switch itself is captured by our_switch_to."""
    events = [
        ev(EventKind.SWITCH, side=OURS, actor="skarmory"),
    ]
    prev = _zero_ctx(our_team_order=("zapdos", "skarmory"))
    curr = _zero_ctx(turn=2, our_active="skarmory")
    d = TurnDelta.build_from_events(prev, curr, action=1, events=events)
    assert d.our_attempted_move_id is None
    assert d.our_switch_to == "skarmory"


def test_attempted_struggle():
    """Action index 10 → struggle."""
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar",
           move_id="struggle"),
    ]
    prev = _zero_ctx()
    curr = _zero_ctx(turn=2)
    d = TurnDelta.build_from_events(prev, curr, action=10, events=events)
    assert d.our_attempted_move_id == "struggle"


def test_attempted_move_zero_when_no_events():
    """No events → all-zero attempted action (missing → 0)."""
    prev = _zero_ctx()
    curr = _zero_ctx(turn=2, phase="forced_switch")
    d = TurnDelta.build_from_events(prev, curr, action=7, events=[])
    assert d.our_attempted_move_id == "icebeam"  # still decoded from action
    assert d.our_move_id is None                  # but no event confirmed it


# ---------------------------------------------------------------------------
# Multi-KO + causes
# ---------------------------------------------------------------------------

def test_single_faint_attack():
    """One KO by direct move (no [from]) → our_faint_count=0, opp_faint_count=1, cause=attack."""
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar",
           move_id="thunderbolt"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-1.0),  # no reason
        ev(EventKind.FAINT, side=OPP, actor="tyranitar"),
    ]
    prev = _zero_ctx()
    curr = _zero_ctx(turn=2, opp_fainted_count=2)
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    assert d.opp_faint_count == 1
    assert d.our_faint_count == 0
    assert d.opp_fainted is True
    idx = list(FAINT_CAUSE_VOCAB).index("attack")
    assert d.opp_faint_causes[idx] == pytest.approx(1.0)
    assert d.opp_faint_causes.sum() == pytest.approx(1.0)


def test_double_ko_explosion_both_sides():
    """Explosion KOs both sides: user→selfko, target→attack."""
    events = [
        ev(EventKind.MOVE, side=OURS, actor="forretress", target="blissey",
           move_id="explosion"),
        ev(EventKind.DAMAGE, side=OPP, actor="blissey", amount=-1.0),
        ev(EventKind.FAINT, side=OURS, actor="forretress"),
        ev(EventKind.FAINT, side=OPP, actor="blissey"),
    ]
    prev = _zero_ctx(our_active="forretress")
    curr = _zero_ctx(turn=2, our_fainted_count=1, opp_fainted_count=2,
                     our_active="NONE")
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    assert d.our_faint_count == 1
    assert d.opp_faint_count == 1
    selfko_idx = list(FAINT_CAUSE_VOCAB).index("selfko")
    attack_idx = list(FAINT_CAUSE_VOCAB).index("attack")
    assert d.our_faint_causes[selfko_idx] == pytest.approx(1.0)
    assert d.opp_faint_causes[attack_idx] == pytest.approx(1.0)


def test_faint_cause_hazard_spikes():
    events = [
        ev(EventKind.DAMAGE, side=OURS, actor="zapdos", amount=-0.25, reason="Spikes"),
        ev(EventKind.FAINT, side=OURS, actor="zapdos"),
    ]
    prev = _zero_ctx()
    curr = _zero_ctx(turn=2, our_fainted_count=1)
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    assert d.our_faint_count == 1
    hazard_idx = list(FAINT_CAUSE_VOCAB).index("hazard")
    assert d.our_faint_causes[hazard_idx] == pytest.approx(1.0)


def test_no_faints_all_zero_counts():
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar",
           move_id="thunderbolt"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.5),
    ]
    prev = _zero_ctx()
    curr = _zero_ctx(turn=2)
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    assert d.our_faint_count == 0
    assert d.opp_faint_count == 0
    assert d.our_faint_causes.sum() == pytest.approx(0.0)
    assert d.opp_faint_causes.sum() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Encoder round-trip for new fields
# ---------------------------------------------------------------------------

def test_encoder_faint_count_and_causes_round_trip():
    """Faint count and causes make it through encode()→describe_vector() intact."""
    enc = _enc()
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar",
           move_id="thunderbolt"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-1.0),
        ev(EventKind.FAINT, side=OPP, actor="tyranitar"),
    ]
    prev = _zero_ctx()
    curr = _zero_ctx(turn=2, opp_fainted_count=2)
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    # faint_count is kept on the dataclass (for reward) but NOT encoded into obs.
    assert d.opp_faint_count == 1
    assert d.our_faint_count == 0
    vec = enc.encode(d)

    # faint cause multi-hot
    attack_idx = list(_ENC_CAUSE_VOCAB).index("attack")
    assert vec[OFFSET_OPP_FAINT_CAUSES + attack_idx] == pytest.approx(1.0)
    assert vec[OFFSET_OPP_FAINT_CAUSES : OFFSET_OPP_FAINT_CAUSES + FAINT_CAUSE_DIM].sum() == pytest.approx(1.0)

    desc = enc.describe_vector(vec)
    assert "attack" in desc["opp_faint_causes"]


def test_encoder_attempted_move_id_round_trip():
    """our_attempted_move_id raw int is stored at OFFSET_OUR_ATTEMPTED_MOVE."""
    enc = _enc()
    events = [
        ev(EventKind.CANT, side=OURS, actor="zapdos", reason="slp"),
    ]
    prev = _zero_ctx()
    curr = _zero_ctx(turn=2, our_cant_reason="slp")
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    # action 6 → move slot 0 → thunderbolt (num=85)
    assert d.our_attempted_move_id == "thunderbolt"
    vec = enc.encode(d)
    assert vec[OFFSET_OUR_ATTEMPTED_MOVE] == pytest.approx(85.0)  # thunderbolt num

    desc = enc.describe_vector(vec)
    assert desc["our_attempted_move"] == "thunderbolt"


def test_pressed_switch_no_attempted_move_in_vec():
    """A pressed switch leaves attempted_move_id = 0 in the encoded vector
    (the switch is captured by our_switch_to_species, not an attempted move)."""
    enc = _enc()
    events = [
        ev(EventKind.SWITCH, side=OURS, actor="skarmory"),
    ]
    prev = _zero_ctx(our_team_order=("zapdos", "skarmory"))
    curr = _zero_ctx(turn=2, our_active="skarmory")
    d = TurnDelta.build_from_events(prev, curr, action=1, events=events)
    assert d.our_attempted_move_id is None
    vec = enc.encode(d)
    assert vec[OFFSET_OUR_ATTEMPTED_MOVE] == pytest.approx(0.0)


def test_empty_delta_missing_zeros():
    """TurnDelta.empty() encodes all new fields as zero."""
    enc = _enc()
    vec = enc.encode(TurnDelta.empty())
    assert vec[OFFSET_OUR_FAINT_CAUSES:OFFSET_OUR_FAINT_CAUSES + FAINT_CAUSE_DIM].sum() == 0.0
    assert vec[OFFSET_OPP_FAINT_CAUSES:OFFSET_OPP_FAINT_CAUSES + FAINT_CAUSE_DIM].sum() == 0.0
    assert vec[OFFSET_OUR_ATTEMPTED_MOVE] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Status transitions folded into the TurnDelta (Lum-Berry-enables-DD pattern)
# ---------------------------------------------------------------------------

def test_status_applied_and_cured_folded():
    """A STATUS event → our_status_applied; a CURESTATUS event → our_status_cured.
    These are the per-turn transition events the history needs (the snapshot only
    shows current status, not 'cured THIS turn')."""
    from poke_env.battle.status import Status
    # Opp Toxics our tyranitar, then our tyranitar's Lum Berry cures it.
    events_applied = [
        ev(EventKind.MOVE, side=OPP, actor="blissey", target="tyranitar", move_id="toxic"),
        ev(EventKind.STATUS, side=OURS, actor="tyranitar", status="tox"),
    ]
    prev = _zero_ctx(our_active="tyranitar")
    curr = _zero_ctx(turn=2, our_active="tyranitar")
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events_applied)
    assert d.our_status_applied == Status.TOX
    assert d.our_status_cured is None

    events_cured = [
        ev(EventKind.ENDITEM, side=OURS, actor="tyranitar", item="lumberry"),
        ev(EventKind.CURESTATUS, side=OURS, actor="tyranitar", status="tox"),
    ]
    d2 = TurnDelta.build_from_events(prev, curr, action=6, events=events_cured)
    assert d2.our_status_cured == Status.TOX
    assert d2.our_status_applied is None


def test_status_transition_encoder_round_trip():
    from poke_env.battle.status import Status
    from agents.observation.turn_delta_encoder import (
        OFFSET_OUR_STATUS_CURED, OFFSET_OPP_STATUS_APPLIED,
    )
    enc = _enc()
    events = [
        ev(EventKind.CURESTATUS, side=OURS, actor="zapdos", status="par"),
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="thunderbolt"),
        ev(EventKind.STATUS, side=OPP, actor="tyranitar", status="par"),
    ]
    prev = _zero_ctx()
    curr = _zero_ctx(turn=2)
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    assert d.our_status_cured == Status.PAR
    assert d.opp_status_applied == Status.PAR
    vec = enc.encode(d)
    desc = enc.describe_vector(vec)
    assert desc["our_status_cured"] == "PAR"
    assert desc["opp_status_applied"] == "PAR"
    # PAR is index 3 in _STATUS_ORDER (BRN,FNT,FRZ,PAR,...)
    assert vec[OFFSET_OUR_STATUS_CURED + 3] == pytest.approx(1.0)
    assert vec[OFFSET_OPP_STATUS_APPLIED + 3] == pytest.approx(1.0)


def test_no_status_change_is_zeros():
    enc = _enc()
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="thunderbolt"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.4),
    ]
    d = TurnDelta.build_from_events(_zero_ctx(), _zero_ctx(turn=2), action=6, events=events)
    assert d.our_status_applied is None and d.opp_status_cured is None
    desc = enc.describe_vector(enc.encode(d))
    assert desc["our_status_applied"] is None
    assert desc["opp_status_cured"] is None


# ---------------------------------------------------------------------------
# Item-used bit folded into the TurnDelta (parity with ability_activated)
# ---------------------------------------------------------------------------

def test_item_used_bit_folded():
    """|-enditem| → item_lost on the dataclass (id kept for reward/replay) and a
    single 'item used' BIT in the encoded vector (the WHICH lives in the item block)."""
    from agents.observation.turn_delta_encoder import (
        OFFSET_OUR_ITEM_USED, OFFSET_OPP_ITEM_USED,
    )
    enc = _enc()
    events = [
        ev(EventKind.ENDITEM, side=OURS, actor="zapdos", item="lumberry"),
        ev(EventKind.CURESTATUS, side=OURS, actor="zapdos", status="par"),
    ]
    prev = _zero_ctx()
    curr = _zero_ctx(turn=2)
    d = TurnDelta.build_from_events(prev, curr, action=6, events=events)
    # dataclass keeps the identity (for reward/replay)
    assert d.our_item_lost == "lumberry"
    assert d.opp_item_lost is None
    vec = enc.encode(d)
    # encoder emits only a bit, no item id
    assert vec[OFFSET_OUR_ITEM_USED] == pytest.approx(1.0)
    assert vec[OFFSET_OPP_ITEM_USED] == pytest.approx(0.0)
    desc = enc.describe_vector(vec)
    assert desc["our_item_used"] is True
    assert desc["opp_item_used"] is False


def test_no_item_used_is_zero():
    enc = _enc()
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="thunderbolt"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.3),
    ]
    d = TurnDelta.build_from_events(_zero_ctx(), _zero_ctx(turn=2), action=6, events=events)
    assert d.our_item_lost is None and d.opp_item_lost is None
    assert enc.describe_vector(enc.encode(d))["our_item_used"] is False
