"""Integrity tests for move/team ordering alignment between the model's view
(sorted-by-id move slots, the request-order block) and the action space (request order).

The row guard `check_obs_move_order` (gen3_move_legality_by_id_v1) is exercised on REAL rows
(the learner golden buffer + the compile parity fixture) and on targeted corruptions of them.
"""
import types
import numpy as np
import pytest
from unittest.mock import MagicMock

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move

from agents.action.mapper import Gen3ActionMapper
from agents.action.ordering_integrity import (
    OrderingMismatchError,
    row_offsets,
    check_obs_move_order,
    check_switch_ordering_alignment,
)
from agents.battle.live_view import (
    LegalActions, LiveView, LiveSide, LivePokemon, LiveMove, LiveWeather,
)
from agents.action.constants import MOVE_START


_NO_WEATHER = LiveWeather(weather=None, is_permanent=False, turns_active=0)


def _livemon(species, move_ids=(), active=False):
    """A minimal LivePokemon carrying just the fields the integrity guards read
    (species + revealed move ids)."""
    return LivePokemon(
        species=species, active=active, fainted=False, revealed=True,
        hp_fraction=1.0, status=None, types=("ghost",),
        moves=tuple(LiveMove(id=m, current_pp=16, max_pp=16) for m in sorted(move_ids)),
        item=None, ability=None, boosts={}, volatiles={},
    )


def _live(active_move_ids=(), team_species=("gengar", "metagross", "starmie"),
          active_species="gengar", turn=30):
    """A LiveView whose ``ours.mons`` mirrors a ``_battle()`` team (same order), the
    active mon carrying ``active_move_ids``. Drives the integrity checks + get_mask
    through the LiveView boundary instead of a raw battle read."""
    mons, active_obj = [], None
    for s in team_species:
        is_active = (s == active_species)
        m = _livemon(s, active_move_ids if is_active else (), active=is_active)
        mons.append(m)
        if is_active:
            active_obj = m
    ours = LiveSide(team_size=len(mons), active=active_obj, mons=tuple(mons), side_conditions={})
    opp = LiveSide(team_size=0, active=None, mons=(), side_conditions={})
    return LiveView(turn=turn, weather=_NO_WEATHER, ours=ours, opp=opp)


def _move(move_id):
    m = MagicMock(spec=Move)
    m.id = move_id
    return m


def _pokemon(species, move_ids, active=False):
    mon = MagicMock(spec=Pokemon)
    mon.species = species
    mon.moves = {mid: _move(mid) for mid in move_ids}
    mon.active = active
    mon.fainted = False
    return mon


def _battle(active_move_ids, disabled_id=None, struggle=False):
    """Build a minimal battle whose request (action) move order is
    `active_move_ids` and whose active mon owns the same moves (sorted-order is
    derived by the masker/extractor independently)."""
    active = _pokemon("gengar", active_move_ids, active=True)
    bench = [
        _pokemon("metagross", ["meteormash"]),
        _pokemon("starmie", ["surf"]),
    ]
    team = {p.species: p for p in [active, *bench]}

    battle = MagicMock(spec=AbstractBattle)
    battle.turn = 30
    battle.team = team
    battle.active_pokemon = active
    # Bench is switchable; active is not.
    battle.available_switches = bench
    battle.available_moves = (
        [_move("struggle")] if struggle else list(active.moves.values())
    )
    # LegalActions.from_battle reads these poke-env-derived legality flags.
    battle.force_switch = False
    battle.trapped = False
    battle.maybe_trapped = False
    battle.wait = False
    # last_request drives the mask: request order + disabled flags.
    battle.last_request = {
        "active": [{
            "moves": [
                {"id": mid, "disabled": (mid == disabled_id)}
                for mid in active_move_ids
            ]
        }]
    }
    return battle










def test_switch_ordering_aligned_by_default():
    """Our team uses list(battle.team.values()) order everywhere (mirrored by
    live.ours.mons) -> aligned, no raise."""
    battle = _battle(["icepunch", "taunt", "thunderbolt", "willowisp"])
    legal = LegalActions.from_battle(battle)
    live = _live(["icepunch", "taunt", "thunderbolt", "willowisp"])
    check_switch_ordering_alignment(live, np.ones(11, dtype=np.int8), legal)  # explicit


def test_switch_ordering_mismatch_raises():
    """If the encoder's team order ever diverges from the mask/mapper's captured
    snapshot order, a switch action would target the wrong mon -> must raise."""
    battle = _battle(["icepunch", "taunt", "thunderbolt", "willowisp"])
    legal = LegalActions.from_battle(battle)  # capture switches at the original order
    # Simulate drift: the encoder's live team order (LiveView) no longer matches the
    # snapshot's slot mapping (here, reversed).
    live = _live(
        active_move_ids=["icepunch", "taunt", "thunderbolt", "willowisp"],
        team_species=("starmie", "metagross", "gengar"), active_species="gengar",
    )
    with pytest.raises(OrderingMismatchError):
        check_switch_ordering_alignment(live, np.ones(11, dtype=np.int8), legal)




# --------------------------------------------------------------------------
# 2b — mapper resolves action index -> the move/switch that index means
# --------------------------------------------------------------------------

def test_mapper_resolves_action_index_to_request_slot():
    """Action 6+k must execute request move k (the contract the model trains on)."""
    battle = _battle(["thunderbolt", "willowisp", "icepunch", "taunt"])
    for k, expected in enumerate(["thunderbolt", "willowisp", "icepunch", "taunt"]):
        order = Gen3ActionMapper.action_to_order(MOVE_START + k, battle)
        assert order.order.id == expected


def test_mapper_raises_on_stale_state():
    """If the server move list changes after the snapshot was captured, acting is
    refused rather than sending the wrong move (fail-loud)."""
    battle = _battle(["thunderbolt", "willowisp", "icepunch", "taunt"])
    snap = LegalActions.from_battle(battle)
    ctx = types.SimpleNamespace(turn=battle.turn, legal=snap, mask=np.ones(11, dtype=np.int8))
    # Server now reports a different move order:
    battle.last_request["active"][0]["moves"] = [
        {"id": m, "disabled": False} for m in ["surf", "icebeam", "thunderbolt", "taunt"]
    ]
    with pytest.raises(RuntimeError):
        Gen3ActionMapper.assert_decision_current(ctx, battle)


# --------------------------------------------------------------------------
# gen3_move_legality_by_id_v1 — the ROW guard on real observation rows
# --------------------------------------------------------------------------

def _real_rows():
    """REAL rows + the masks they were served with: the learner golden buffer (a Rust-core rollout)
    and the compile parity fixture (real eval states)."""
    from pathlib import Path
    from agents.model.compile_parity_fixture import FIXTURE_PATH
    gold = Path(__file__).resolve().parents[1] / "training" / "learner_golden_buffer.npz"
    with np.load(gold) as z:
        obs = z["obs:observation"]
        obs = obs.reshape(-1, obs.shape[-1]).astype(np.float32)
        mask = z["action_masks"].reshape(-1, z["action_masks"].shape[-1])
    with np.load(FIXTURE_PATH) as z:
        obs2, mask2 = np.asarray(z["obs"], dtype=np.float32), np.asarray(z["action_mask"], dtype=bool)
    return np.concatenate([obs, obs2]), np.concatenate([mask.astype(bool), mask2])


def _misaligned_row_index(obs):
    """A row whose request order differs from its sorted slot order AND carries an illegal move —
    exactly where a positional application lands legality on the wrong move."""
    o = row_offsets()
    for i, row in enumerate(obs):
        req, legal = row[o.req_ids], row[o.req_legal]
        team = row[o.team0:o.team0 + 6 * o.mon_dim].reshape(6, o.mon_dim)
        act = np.flatnonzero(team[:, o.active_col] > 0.5)
        if (req > 0).sum() == 4 and len(act) == 1 and (legal < 0.5).any():
            if not np.array_equal(team[act[0], o.slot_id_cols], req):
                return i
    raise AssertionError("no misaligned real row — the corruption tests below would be vacuous")


def test_guard_offsets_match_the_extractor_unpack():
    """The guard's absolute offsets are the ones the model reads: the schema's request block and our
    team block, and `slice_pokemon_categoricals`'s move ids + active flag (no model build needed)."""
    import torch
    from agents.model.extractor_ctx import slice_pokemon_categoricals
    from agents.observation.constants import POKEMON_ACTIVE_OFFSET, POKEMON_FULL_DIM
    from agents.observation.schema import build_schema
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    sl = build_schema(layout).slices()
    o = row_offsets()
    assert sl["reactive.active_req_moves"].start == o.req0 == o.req_ids.start
    assert sl["reactive.active_req_moves"].stop == o.req_legal.stop
    assert (o.mon_dim, o.active_col, o.team0) == (POKEMON_FULL_DIM, POKEMON_ACTIVE_OFFSET, 0)
    assert sl["our_team"].start == 0 and (sl["our_team"].stop - sl["our_team"].start) == 6 * POKEMON_FULL_DIM
    obs, _ = _real_rows()
    team = torch.as_tensor(obs[:, :6 * POKEMON_FULL_DIM]).reshape(len(obs), 6, POKEMON_FULL_DIM)
    ids = slice_pokemon_categoricals(team, layout)
    assert torch.equal(ids["all_move_ids"], team[:, :, torch.as_tensor(o.slot_id_cols)].long())
    assert torch.equal(ids["hp_and_active"][..., -1], team[:, :, POKEMON_ACTIVE_OFFSET])


def test_guard_passes_every_real_row():
    obs, mask = _real_rows()
    check_obs_move_order(obs, mask)


def test_guard_raises_when_a_request_move_has_no_sorted_slot():
    obs, mask = _real_rows()
    i = _misaligned_row_index(obs)
    bad = obs[i:i + 1].copy()
    bad[0, row_offsets().req0] = 9999.0                       # request slot 0 names a move our active does not hold
    with pytest.raises(OrderingMismatchError, match="one-to-one"):
        check_obs_move_order(bad, mask[i:i + 1])


def test_guard_raises_when_legality_disagrees_with_the_mask():
    obs, mask = _real_rows()
    i = _misaligned_row_index(obs)
    bad = mask[i:i + 1].copy()
    bad[0, MOVE_START:MOVE_START + 4] = ~bad[0, MOVE_START:MOVE_START + 4]
    with pytest.raises(OrderingMismatchError, match="disagree with"):
        check_obs_move_order(obs[i:i + 1], bad)


def test_guard_raises_on_a_choosable_move_with_no_identity():
    obs, mask = _real_rows()
    i = _misaligned_row_index(obs)
    row = obs[i:i + 1].copy()
    row[0, row_offsets().req0:row_offsets().req_legal.stop] = 0.0               # the request block wiped, the mask still allows moves
    m = mask[i:i + 1].copy()
    m[0, MOVE_START] = True
    with pytest.raises(OrderingMismatchError, match="no identity"):
        check_obs_move_order(row, m)


def test_guard_raises_without_exactly_one_active():
    from agents.observation.constants import POKEMON_ACTIVE_OFFSET, POKEMON_FULL_DIM
    obs, mask = _real_rows()
    i = _misaligned_row_index(obs)
    row = obs[i:i + 1].copy()
    for k in range(6):
        row[0, k * POKEMON_FULL_DIM + POKEMON_ACTIVE_OFFSET] = 0.0
    with pytest.raises(OrderingMismatchError, match="active mons"):
        check_obs_move_order(row, mask[i:i + 1])


def test_guard_exempts_a_single_forced_action_like_the_recharge_turn():
    """Hyper Beam's recharge turn: the request names only `recharge` (no moveset holds it), so the request
    block is all-zero while the mask allows action 6 alone. One legal action = nothing to choose: no raise.
    The same row with a second legal action IS a decision and raises."""
    obs, mask = _real_rows()
    i = _misaligned_row_index(obs)
    row = obs[i:i + 1].copy()
    o = row_offsets()
    row[0, o.req0:o.req_legal.stop] = 0.0
    forced = np.zeros((1, mask.shape[1]), dtype=bool)
    forced[0, MOVE_START] = True
    check_obs_move_order(row, forced)
    forced[0, 1] = True                          # + a legal switch: now a choice exists
    with pytest.raises(OrderingMismatchError, match="no identity"):
        check_obs_move_order(row, forced)
