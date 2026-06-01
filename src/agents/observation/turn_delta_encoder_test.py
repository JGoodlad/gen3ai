import numpy as np
import pytest
from agents.observation.turn_delta_encoder import (
    TurnDeltaEncoder,
    TURN_DELTA_DIM,
    STATUS_DIM,
    CANT_DIM,
    OUTCOME_DIM,
    OFFSET_OUR_SWITCHED,
    OFFSET_OPP_SWITCHED,
    OFFSET_OUR_FAILED_TO_MOVE,
    OFFSET_OPP_FAILED_TO_MOVE,
    OFFSET_OUR_CANT,
    OFFSET_OPP_CANT,
    OFFSET_OUR_HP_DELTA_SUM,
    OFFSET_OPP_HP_DELTA_SUM,
    OFFSET_WE_FAINTED,
    OFFSET_OPP_FAINTED,
    OFFSET_OPP_MOVE_KNOWN,
    OFFSET_OUR_EFF,
    OFFSET_OPP_EFF,
    OFFSET_ORDER,
    OFFSET_OUR_BOOST_DELTA,
    OFFSET_OPP_BOOST_DELTA,
    OFFSET_PHASE_FORCED_SWITCH,
    OFFSET_OUR_TARGET_HP_DELTA,
    OFFSET_OPP_TARGET_HP_DELTA,
    OFFSET_OUR_HP_LEVELS,
    OFFSET_OPP_HP_LEVELS,
    OFFSET_OUR_TARGET_STATUS,
    OFFSET_OPP_TARGET_STATUS,
    OFFSET_OUR_MOVE_OUTCOME,
    OFFSET_OPP_MOVE_OUTCOME,
    OFFSET_OUR_CRIT,
    OFFSET_OPP_CRIT,
    OFFSET_OUR_ACTOR_SPECIES,
    OFFSET_OPP_ACTOR_SPECIES,
    OFFSET_OUR_TARGET_SPECIES,
    OFFSET_OPP_TARGET_SPECIES,
    OFFSET_OUR_SWITCH_TO_SPEC,
    OFFSET_OPP_SWITCH_TO_SPEC,
)
from agents.training.turn_delta import TurnDelta
from poke_env.battle.abstract_battle import DamagingMoveEvent
from poke_env.battle.status import Status

# Minimal move mapping covering the test cases.
_MOVES = {
    "rockslide": {"num": 100, "basePower": 75, "type": "Rock", "hasSecondary": True, "hasRecoil": False},
    "surf":      {"num": 57,  "basePower": 95, "type": "Water", "hasSecondary": False, "hasRecoil": False},
    "struggle":  {"num": 165, "basePower": 50, "type": "Normal", "hasSecondary": False, "hasRecoil": True},
}

# Minimal species mapping covering the test cases.
_SPECIES = {
    "tyranitar": {"num": 248},
    "salamence": {"num": 373},
    "skarmory":  {"num": 227},
    "blissey":   {"num": 242},
    "snorlax":   {"num": 143},
}


def _enc():
    return TurnDeltaEncoder(_MOVES, _SPECIES)


def _empty_delta():
    return TurnDelta.empty()


def _delta(**kwargs):
    base = dict(
        our_move_id=None, our_switch_to=None, our_prev_active="tyranitar",
        opp_move_id=None, opp_switch_to=None, opp_prev_active="salamence",
        opp_move_known=True,
        our_hp_delta=np.zeros(6, dtype=np.float32),
        opp_hp_delta=np.zeros(6, dtype=np.float32),
        we_fainted=False, opp_fainted=False,
        our_failed_to_move=False, our_cant_reason=None,
        opp_failed_to_move=False, opp_cant_reason=None,
        our_boost_delta=np.zeros(7, dtype=np.int8),
        opp_boost_delta=np.zeros(7, dtype=np.int8),
        our_effectiveness=None,
        opp_effectiveness=None,
        we_moved_first=None,
    )
    base.update(kwargs)
    return TurnDelta(**base)


def test_dimension():
    assert _enc().dimension == TURN_DELTA_DIM
    assert TURN_DELTA_DIM == 159  # gen3_trapping_signals_v1: + attempted_switch_rejected bit + attempted-switch species id


def test_empty_delta_is_all_zeros_except_opp_move_known():
    enc = _enc()
    vec = enc.encode(_empty_delta())
    assert vec.shape == (TURN_DELTA_DIM,)
    assert vec.dtype == np.float32
    # empty() has opp_move_known=False, prev_active="NULL" → species id 0,
    # all numeric fields zero.
    assert vec.sum() == pytest.approx(0.0)


def test_known_move_populates_features():
    enc = _enc()
    delta = _delta(our_move_id="rockslide")
    vec = enc.encode(delta)
    # our_move block: [id_norm, power_norm, has_secondary, has_recoil, type_id_norm]
    our_block = vec[:5]
    assert our_block[0] > 0.0      # id_norm
    assert our_block[1] == pytest.approx(75.0 / 200.0)   # power
    assert our_block[2] == pytest.approx(1.0)             # has_secondary
    assert our_block[3] == pytest.approx(0.0)             # no recoil
    assert our_block[4] > 0.0                             # type_id_norm (Rock != 0)


def test_unknown_move_is_zeros():
    enc = _enc()
    delta = _delta(our_move_id=None)
    vec = enc.encode(delta)
    assert vec[:5].sum() == pytest.approx(0.0)


def test_opp_move_block_offset():
    enc = _enc()
    delta = _delta(opp_move_id="surf")
    vec = enc.encode(delta)
    opp_block = vec[5:10]
    assert opp_block[0] > 0.0
    assert opp_block[1] == pytest.approx(95.0 / 200.0)


def test_switched_flag():
    enc = _enc()
    delta = _delta(our_switch_to="skarmory")
    vec = enc.encode(delta)
    assert vec[OFFSET_OUR_SWITCHED] == pytest.approx(1.0)
    assert vec[OFFSET_OPP_SWITCHED] == pytest.approx(0.0)


def test_failed_to_move_flag():
    enc = _enc()
    delta = _delta(our_failed_to_move=True, our_cant_reason="par")
    vec = enc.encode(delta)
    assert vec[OFFSET_OUR_FAILED_TO_MOVE] == pytest.approx(1.0)
    assert vec[OFFSET_OPP_FAILED_TO_MOVE] == pytest.approx(0.0)
    # our_cant_reason onehot: par = index 2 in gen3_effects.CANT_REASONS
    # (slp=0, frz=1, par=2, flinch=3, ...)
    assert vec[OFFSET_OUR_CANT + 2] == pytest.approx(1.0)


def test_opp_cant_reason_onehot():
    enc = _enc()
    delta = _delta(opp_failed_to_move=True, opp_cant_reason="flinch")
    vec = enc.encode(delta)
    # flinch is index 3 in [par, slp, frz, flinch, confusion, ...]
    assert vec[OFFSET_OPP_CANT + 3] == pytest.approx(1.0)
    assert vec[OFFSET_OPP_CANT + 0] == pytest.approx(0.0)


def test_cant_reason_prefix_normalized():
    """Showdown's 'move: Taunt' / 'ability: Truant' must map to the bare index.

    Indices are from gen3_effects.CANT_REASONS:
      slp=0, frz=1, par=2, flinch=3, recharge=4, attract=5, disable=6,
      taunt=7, imprison=8, focuspunch=9, nopp=10, truant=11
    """
    enc = _enc()
    # taunt is index 7 in gen3_effects.CANT_REASONS
    vec = enc.encode(_delta(our_failed_to_move=True, our_cant_reason="move: Taunt"))
    assert vec[OFFSET_OUR_CANT + 7] == pytest.approx(1.0)
    assert enc.describe_vector(vec)["our_cant"] == "taunt"
    # truant is index 11, prefixed with 'ability: '
    vec2 = enc.encode(_delta(opp_failed_to_move=True, opp_cant_reason="ability: Truant"))
    assert vec2[OFFSET_OPP_CANT + 11] == pytest.approx(1.0)
    # recharge (Hyper Beam) is a bare reason at index 4
    vec3 = enc.encode(_delta(our_failed_to_move=True, our_cant_reason="recharge"))
    assert vec3[OFFSET_OUR_CANT + 4] == pytest.approx(1.0)


def test_unknown_cant_reason_raises():
    """Crash-don't-drop: a cant reason outside the gen3_effects vocab must RAISE,
    not silently encode all-zeros (which is indistinguishable from 'moved
    normally'). The encoder is the only validation point — gen3_battle records
    the |cant| reason verbatim."""
    from agents.observation.gen3_effects import UnknownCantReasonError
    enc = _enc()
    with pytest.raises(UnknownCantReasonError):
        enc.encode(_delta(our_failed_to_move=True, our_cant_reason="move: Bogus"))


def test_hp_delta():
    enc = _enc()
    our_hp = np.zeros(6, dtype=np.float32)
    opp_hp = np.zeros(6, dtype=np.float32)
    our_hp[0] = -0.3   # we took 30% damage
    opp_hp[0] = -0.5   # opp took 50%
    delta = _delta(our_hp_delta=our_hp, opp_hp_delta=opp_hp)
    vec = enc.encode(delta)
    assert vec[OFFSET_OUR_HP_DELTA_SUM] == pytest.approx(-0.3)
    assert vec[OFFSET_OPP_HP_DELTA_SUM] == pytest.approx(-0.5)


def test_faint_flags():
    enc = _enc()
    delta = _delta(opp_fainted=True)
    vec = enc.encode(delta)
    assert vec[OFFSET_WE_FAINTED] == pytest.approx(0.0)
    assert vec[OFFSET_OPP_FAINTED] == pytest.approx(1.0)


def test_opp_move_known_flag():
    enc = _enc()
    delta = _delta(opp_move_known=True)
    vec = enc.encode(delta)
    assert vec[OFFSET_OPP_MOVE_KNOWN] == pytest.approx(1.0)

    delta2 = _delta(opp_move_known=False)
    vec2 = enc.encode(delta2)
    assert vec2[OFFSET_OPP_MOVE_KNOWN] == pytest.approx(0.0)


def test_move_outcome_onehot():
    enc = _enc()
    # hit / miss / fail map to indices 0 / 1 / 2
    vec = enc.encode(_delta(our_move_id="rockslide", our_move_outcome="hit",
                            opp_move_id="surf", opp_move_outcome="miss"))
    assert vec[OFFSET_OUR_MOVE_OUTCOME + 0] == pytest.approx(1.0)
    assert vec[OFFSET_OPP_MOVE_OUTCOME + 1] == pytest.approx(1.0)
    d = enc.describe_vector(vec)
    assert d["our_move_outcome"] == "hit"
    assert d["opp_move_outcome"] == "miss"

    # fail
    vec2 = enc.encode(_delta(our_move_id="rockslide", our_move_outcome="fail"))
    assert vec2[OFFSET_OUR_MOVE_OUTCOME + 2] == pytest.approx(1.0)

    # None → all-zeros outcome block
    vec3 = enc.encode(_delta(our_switch_to="skarmory"))
    blk = vec3[OFFSET_OUR_MOVE_OUTCOME:OFFSET_OUR_MOVE_OUTCOME + OUTCOME_DIM]
    assert blk.sum() == pytest.approx(0.0)
    assert enc.describe_vector(vec3)["our_move_outcome"] is None


def test_crit_bits():
    enc = _enc()
    vec = enc.encode(_delta(our_move_id="rockslide", our_move_outcome="hit", our_move_crit=True))
    assert vec[OFFSET_OUR_CRIT] == pytest.approx(1.0)
    assert vec[OFFSET_OPP_CRIT] == pytest.approx(0.0)
    d = enc.describe_vector(vec)
    assert d["our_move_crit"] is True
    assert d["opp_move_crit"] is False


def test_unknown_move_id_gracefully_zeros():
    enc = _enc()
    delta = _delta(our_move_id="shadowball")  # not in _MOVES
    vec = enc.encode(delta)
    assert vec[:5].sum() == pytest.approx(0.0)


def test_struggle_recoil_flag():
    enc = _enc()
    delta = _delta(our_move_id="struggle")
    vec = enc.encode(delta)
    assert vec[3] == pytest.approx(1.0)  # has_recoil


# ---------------------------------------------------------------------------
# Extended (gen3_unified_v2) fields
# ---------------------------------------------------------------------------

def test_boost_deltas_encoded():
    enc = _enc()
    bd_our = np.array([2, 0, 1, 0, 0, 0, 0], dtype=np.int8)   # +2 atk, +1 def
    bd_opp = np.array([0, -1, 0, 0, -2, 0, 0], dtype=np.int8) # -1 def, -2 spe
    delta = _delta(our_boost_delta=bd_our, opp_boost_delta=bd_opp)
    vec = enc.encode(delta)
    assert np.allclose(vec[OFFSET_OUR_BOOST_DELTA:OFFSET_OUR_BOOST_DELTA + 7], bd_our.astype(np.float32))
    assert np.allclose(vec[OFFSET_OPP_BOOST_DELTA:OFFSET_OPP_BOOST_DELTA + 7], bd_opp.astype(np.float32))


def test_phase_forced_switch_flag():
    enc = _enc()
    vec_move = enc.encode(_delta(phase_is_forced_switch=False))
    vec_fs = enc.encode(_delta(phase_is_forced_switch=True))
    assert vec_move[OFFSET_PHASE_FORCED_SWITCH] == pytest.approx(0.0)
    assert vec_fs[OFFSET_PHASE_FORCED_SWITCH] == pytest.approx(1.0)


def test_target_hp_delta_signed():
    enc = _enc()
    delta = _delta(our_target_hp_delta=-0.4, opp_target_hp_delta=-0.85)
    vec = enc.encode(delta)
    assert vec[OFFSET_OUR_TARGET_HP_DELTA] == pytest.approx(-0.4)
    assert vec[OFFSET_OPP_TARGET_HP_DELTA] == pytest.approx(-0.85)


def test_target_hp_delta_none_is_zero():
    enc = _enc()
    delta = _delta(our_target_hp_delta=None, opp_target_hp_delta=None)
    vec = enc.encode(delta)
    assert vec[OFFSET_OUR_TARGET_HP_DELTA] == pytest.approx(0.0)
    assert vec[OFFSET_OPP_TARGET_HP_DELTA] == pytest.approx(0.0)


def test_hp_levels_vector():
    enc = _enc()
    our_hp = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    opp_hp = np.array([0.7, 1.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    delta = _delta(our_hp_after=our_hp, opp_hp_after=opp_hp)
    vec = enc.encode(delta)
    assert np.allclose(vec[OFFSET_OUR_HP_LEVELS:OFFSET_OUR_HP_LEVELS + 6], our_hp)
    assert np.allclose(vec[OFFSET_OPP_HP_LEVELS:OFFSET_OPP_HP_LEVELS + 6], opp_hp)


def test_actor_species_falls_back_to_prev_active():
    """No damaging event → encoder uses prev_active species for the actor IDs."""
    enc = _enc()
    delta = _delta()  # our_prev_active=tyranitar (248), opp_prev_active=salamence (373)
    vec = enc.encode(delta)
    assert int(vec[OFFSET_OUR_ACTOR_SPECIES]) == 248
    assert int(vec[OFFSET_OPP_ACTOR_SPECIES]) == 373


def test_actor_species_prefers_damaging_event_user():
    """A damaging event's user_species wins over prev_active (mirror-match attribution)."""
    enc = _enc()
    event = DamagingMoveEvent(
        user_species="skarmory",   # 227
        target_species="snorlax",  # 143
        target_status=None,
        move_id="surf",
        effectiveness=1.0,
    )
    delta = _delta(our_damaging_event=event)
    vec = enc.encode(delta)
    assert int(vec[OFFSET_OUR_ACTOR_SPECIES]) == 227   # event.user_species wins
    # OPP actor still falls back to opp_prev_active because opp has no event.
    assert int(vec[OFFSET_OPP_ACTOR_SPECIES]) == 373


def test_target_species_from_damaging_event():
    """`our_target_species_id` follows the "this-side-this-turn" convention:
    it's the species of OUR mon that got hit = sourced from the OPPONENT's
    damaging_event.target_species. Mirror for opp.
    """
    enc = _enc()
    our_event = DamagingMoveEvent(
        user_species="tyranitar", target_species="blissey",  # 242 (opp mon we hit)
        target_status=None, move_id="rockslide", effectiveness=2.0,
    )
    opp_event = DamagingMoveEvent(
        user_species="salamence", target_species="skarmory",  # 227 (our mon they hit)
        target_status=None, move_id="surf", effectiveness=0.5,
    )
    delta = _delta(our_damaging_event=our_event, opp_damaging_event=opp_event)
    vec = enc.encode(delta)
    # our_target_species_id = mon ON our side that opp hit = skarmory
    assert int(vec[OFFSET_OUR_TARGET_SPECIES]) == 227
    # opp_target_species_id = mon ON opp side that we hit = blissey
    assert int(vec[OFFSET_OPP_TARGET_SPECIES]) == 242


def test_switch_to_species_encoded():
    enc = _enc()
    delta = _delta(our_switch_to="skarmory", opp_switch_to="blissey")
    vec = enc.encode(delta)
    assert int(vec[OFFSET_OUR_SWITCH_TO_SPEC]) == 227
    assert int(vec[OFFSET_OPP_SWITCH_TO_SPEC]) == 242


def test_switch_to_species_zero_when_no_switch():
    enc = _enc()
    delta = _delta(our_switch_to=None, opp_switch_to=None)
    vec = enc.encode(delta)
    assert int(vec[OFFSET_OUR_SWITCH_TO_SPEC]) == 0
    assert int(vec[OFFSET_OPP_SWITCH_TO_SPEC]) == 0


def test_unknown_species_raises():
    """A non-sentinel species name that isn't in the mapping must raise —
    silently embedding it as id 0 would collide with the unrevealed sentinel
    and miscode training data.
    """
    enc = _enc()
    delta = _delta(our_switch_to="mewthree")  # not in _SPECIES fixture
    with pytest.raises(ValueError, match="Unrecognized species"):
        enc.encode(delta)


def test_sentinel_species_encode_to_zero():
    """None / "NONE" / "NULL" are the legitimate "no species this turn"
    sentinels and must encode as id 0 without raising."""
    enc = _enc()
    # No actor/target/switch — prev_active sentinel + None switch.
    delta = _delta(
        our_prev_active="NONE",
        opp_prev_active="NULL",
        our_switch_to=None,
        opp_switch_to=None,
    )
    vec = enc.encode(delta)
    for offset in (
        OFFSET_OUR_ACTOR_SPECIES, OFFSET_OPP_ACTOR_SPECIES,
        OFFSET_OUR_TARGET_SPECIES, OFFSET_OPP_TARGET_SPECIES,
        OFFSET_OUR_SWITCH_TO_SPEC, OFFSET_OPP_SWITCH_TO_SPEC,
    ):
        assert int(vec[offset]) == 0


def test_target_status_onehot_encoded():
    """`our_target_status` is the status of OUR mon (skarmory) when opp hit
    it = opp_damaging_event.target_status. Same "this-side-this-turn"
    convention as our_target_species_id / our_target_hp_delta.
    """
    enc = _enc()
    our_event = DamagingMoveEvent(
        user_species="tyranitar", target_species="blissey",
        target_status=Status.SLP,  # opp mon blissey was asleep when we hit it
        move_id="rockslide", effectiveness=1.0,
    )
    opp_event = DamagingMoveEvent(
        user_species="salamence", target_species="skarmory",
        target_status=Status.FRZ,  # our mon skarmory was frozen when opp hit it
        move_id="surf", effectiveness=2.0,
    )
    delta = _delta(our_damaging_event=our_event, opp_damaging_event=opp_event)
    vec = enc.encode(delta)
    our_status = vec[OFFSET_OUR_TARGET_STATUS:OFFSET_OUR_TARGET_STATUS + STATUS_DIM]
    opp_status = vec[OFFSET_OPP_TARGET_STATUS:OFFSET_OPP_TARGET_STATUS + STATUS_DIM]
    # _STATUS_ORDER = (BRN, FNT, FRZ, PAR, PSN, SLP, TOX) → FRZ=2, SLP=5
    # our_target_status = status of OUR mon (skarmory, FRZ) when opp hit us
    assert our_status[2] == pytest.approx(1.0)
    assert our_status.sum() == pytest.approx(1.0)
    # opp_target_status = status of OPP mon (blissey, SLP) when we hit them
    assert opp_status[5] == pytest.approx(1.0)
    assert opp_status.sum() == pytest.approx(1.0)


def test_target_status_onehot_zero_when_no_status():
    """No damaging event OR target_status=None → all-zeros."""
    enc = _enc()
    # No event at all
    vec1 = enc.encode(_delta())
    assert vec1[OFFSET_OUR_TARGET_STATUS:OFFSET_OUR_TARGET_STATUS + STATUS_DIM].sum() == 0
    # Event but target_status=None
    event = DamagingMoveEvent(
        user_species="tyranitar", target_species="blissey",
        target_status=None, move_id="rockslide", effectiveness=1.0,
    )
    vec2 = enc.encode(_delta(our_damaging_event=event))
    assert vec2[OFFSET_OUR_TARGET_STATUS:OFFSET_OUR_TARGET_STATUS + STATUS_DIM].sum() == 0


def test_describe_vector_decodes_extended_fields():
    enc = _enc()
    # Set up both sides' events so we exercise the full set of fields:
    #   - our event: tyranitar→rockslide→blissey (TOX) — "we" attacker
    #   - opp event: salamence→surf→skarmory (PAR) — "opp" attacker
    our_event = DamagingMoveEvent(
        user_species="tyranitar", target_species="blissey",
        target_status=Status.TOX, move_id="rockslide", effectiveness=2.0,
    )
    opp_event = DamagingMoveEvent(
        user_species="salamence", target_species="skarmory",
        target_status=Status.PAR, move_id="surf", effectiveness=0.5,
    )
    delta = _delta(
        our_damaging_event=our_event,
        opp_damaging_event=opp_event,
        our_switch_to="skarmory",
        phase_is_forced_switch=True,
        our_target_hp_delta=-0.7,
        our_boost_delta=np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        our_hp_after=np.array([1.0, 0.5, 0, 0, 0, 0], dtype=np.float32),
    )
    vec = enc.encode(delta)
    desc = enc.describe_vector(vec)
    assert desc["phase_is_forced_switch"] is True
    assert desc["our_target_hp_delta"] == pytest.approx(-0.7)
    # "this-side-this-turn" convention:
    #   our_actor    = our attacker          = tyranitar
    #   our_target   = mon ON our side hit   = skarmory  (from opp event)
    #   our_status   = its status when hit   = PAR       (from opp event)
    #   opp_actor    = opp attacker          = salamence
    #   opp_target   = mon ON opp side hit   = blissey   (from our event)
    #   opp_status   = its status when hit   = TOX       (from our event)
    assert desc["our_actor_species"] == "tyranitar"
    assert desc["our_target_species"] == "skarmory"
    assert desc["our_target_status"] == "PAR"
    assert desc["opp_actor_species"] == "salamence"
    assert desc["opp_target_species"] == "blissey"
    assert desc["opp_target_status"] == "TOX"
    assert desc["our_switch_to_species"] == "skarmory"
    assert desc["our_boost_delta"][0] == pytest.approx(1.0)
    assert desc["our_hp_levels"][0] == pytest.approx(1.0)
    assert desc["our_hp_levels"][1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Embedded-ID manifest invariants (the layout-driven contract)
# ---------------------------------------------------------------------------

def test_embedded_manifest_partitions_the_slot():
    """TURN_DELTA_EMBEDDED_IDS + TURN_DELTA_SCALAR_OFFSETS must EXACTLY partition
    [0, TURN_DELTA_DIM): every position is either an embedded ID or a scalar,
    never both, never neither. This is what guarantees a raw id can't silently
    leak through as a scalar (or vice versa)."""
    from agents.observation.turn_delta_encoder import (
        TURN_DELTA_EMBEDDED_IDS, TURN_DELTA_SCALAR_OFFSETS, TURN_DELTA_DIM,
    )
    embedded = {pos for pos, _ in TURN_DELTA_EMBEDDED_IDS}
    scalars = set(TURN_DELTA_SCALAR_OFFSETS)
    # no overlap
    assert embedded.isdisjoint(scalars)
    # covers exactly the full slot
    assert embedded | scalars == set(range(TURN_DELTA_DIM))
    # no duplicate embedded positions
    assert len(embedded) == len(TURN_DELTA_EMBEDDED_IDS)
    # every kind is a known table
    assert all(kind in ("move", "type", "species") for _, kind in TURN_DELTA_EMBEDDED_IDS)


def test_embedded_manifest_positions_in_range():
    from agents.observation.turn_delta_encoder import (
        TURN_DELTA_EMBEDDED_IDS, TURN_DELTA_DIM,
    )
    for pos, _ in TURN_DELTA_EMBEDDED_IDS:
        assert 0 <= pos < TURN_DELTA_DIM
