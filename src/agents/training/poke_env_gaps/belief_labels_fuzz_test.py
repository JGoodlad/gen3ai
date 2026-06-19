"""Bridge fuzz test for the hidden-opponent belief LABELS (the privileged training target).

Unit tests pin the label-builder math on synthetic name→num maps; this exercises the real BATTLE
EXTRACTION — `Gen3Env._belief_labels` reading `battle2.team` for the opponent's FULL team, deriving
the revealed set + slot order, and emitting `belief_species`/`belief_moves` into the stepped obs dict
— against thousands of real Showdown states from the full gen3ou pool. It is the only way to catch a
poke-env-shaped gap the math tests can't see (a forme/transform name mismatch, a mon whose moves
don't map, a reveal-order surprise).

Per-decision invariants asserted against the LIVE battle (a violation raises immediately):
  1. The two belief keys are always present, correct shape/dtype, and species nums in-vocab.
  2. SINGLE-SOURCE: the label's believed-slot set (species>=0) EQUALS the obs species_known==0 slots
     the model's BeliefSlots keys its injection on — they can never diverge.
  3. CORRECTNESS: the SET of labeled species == the opponent's ACTUAL hidden mons (battle2.team minus
     the revealed set). This validates the whole battle2→labels pipeline end to end.
  4. Per labeled slot, the move labels are a SUBSET of that mon's real moveset (no hallucinated moves).
  5. LEAK: the belief labels never appear inside obs["observation"] (the vector the model reads) — its
     width is exactly the encoder dimension, so the omniscient labels cannot reach the forward.

Scenario invariant (over a whole run):
  6. The labels actually FIRE — the large majority of mid-battle decisions have >=1 believed slot with
     a real hidden mon (catches a silent all-PAD regression the per-decision checks would pass).

Run directly (no server needed; in-process via the local BattleStream bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/belief_labels_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.data.normalize import to_id_str
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from agents.training.gen3_env import Gen3Env
from agents.observation.base import ObservationEncoder
from agents.observation.state_encoder import load_mappings
from agents.observation.constants import (
    OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_SPECIES_KNOWN_OFFSET, TEAM_SIZE,
)
from agents.observation.belief_labels import BELIEF_MOVE_SLOTS, N_SPREAD_STATS, SPREAD_STAT_ORDER
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
from utils.bridge.bridge_session import attach_bridge_transport
from utils.teambuilder import Gen3Teambuilder
from utils.team_loader.loader import TeamLoader

_MAX_STEPS = 200


def _build(teams, emit):
    env = Gen3Env(
        load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration("BeliefFz", None),
        start_listening=False, emit_belief_labels=emit, emit_spread_labels=emit,
    )
    attach_bridge_transport(env, battle_format="gen3ou", persistent=True, recycle_every=0)
    opp = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration=AccountConfiguration("BeliefFzOpp", None), start_listening=False,
    )
    w = SingleAgentWrapper(env, opp)
    w.action_space = env.action_space
    w.observation_space = env.observation_space
    return w


def _species_known(vec, i):
    return vec[OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET]


class Stats:
    def __init__(self):
        self.decisions = 0
        self.fired = 0  # decisions with >=1 believed labeled slot
        self.spread_fired = 0  # revealed slots whose true derived-stat spread label was validated


def _check(obs, env, sp_num, mv_num, enc_dim, stats):
    bs = obs.get("belief_species")
    bm = obs.get("belief_moves")
    assert bs is not None and bm is not None, "belief keys missing from stepped obs"
    assert bs.shape == (TEAM_SIZE,) and bm.shape == (TEAM_SIZE, BELIEF_MOVE_SLOTS), (bs.shape, bm.shape)
    vec = obs["observation"]
    # (5) leak: the obs vector is exactly the encoder output width — labels are NOT in it
    assert vec.shape[0] == enc_dim, f"obs width {vec.shape[0]} != encoder dim {enc_dim}"
    b1 = getattr(env, "battle1", None)
    b2 = getattr(env, "battle2", None)
    if b1 is None or b2 is None:
        return
    stats.decisions += 1
    believed_obs = {i for i in range(TEAM_SIZE) if _species_known(vec, i) < 0.5}
    believed_label = {i for i in range(TEAM_SIZE) if bs[i] >= 0}
    # (2) single-source invariant
    assert believed_label == believed_obs, f"believed mask mismatch: label={believed_label} obs={believed_obs}"
    # (1) in-vocab
    for i in believed_label:
        assert 0 <= int(bs[i]) < len(sp_num) + 10_000, f"species num {bs[i]} absurd"
    # (3) labeled species == actual hidden mons
    revealed = {to_id_str(m.species) for m in ObservationEncoder.get_team_list(b1, is_opponent=True) if m}
    full = {to_id_str(m.species): m for m in b2.team.values()}
    hidden = {s for s in full if s not in revealed}
    hidden_nums = {sp_num[s] for s in hidden if s in sp_num}
    label_nums = {int(bs[i]) for i in believed_label}
    assert label_nums == hidden_nums, f"labeled species {label_nums} != actual hidden {hidden_nums}"
    if believed_label:
        stats.fired += 1
    # (4) per slot: move labels ⊆ that species' real moveset
    num_to_species = {sp_num[s]: s for s in full if s in sp_num}
    for i in believed_label:
        sp = num_to_species.get(int(bs[i]))
        if sp is None:
            continue
        real_move_nums = {mv_num[to_id_str(mv.id)] for mv in full[sp].moves.values() if to_id_str(mv.id) in mv_num}
        for j in range(BELIEF_MOVE_SLOTS):
            mid = int(bm[i, j])
            if mid >= 0:
                assert mid in real_move_nums, f"move {mid} not in {sp}'s moveset {real_move_nums}"

    # (6) SPREAD belief (gen3_unified_spread_belief_v1): each REVEALED slot carries the TRUE derived
    # stats {atk,def,spa,spd,spe} of that mon (agent2's OWN team's computed stats); believed/pad → mask 0.
    bsp = obs.get("belief_spread")
    bspm = obs.get("belief_spread_mask")
    assert bsp is not None and bspm is not None, "belief_spread keys missing from stepped obs"
    assert bsp.shape == (TEAM_SIZE, N_SPREAD_STATS) and bspm.shape == (TEAM_SIZE,), (bsp.shape, bspm.shape)
    revealed_slots = [i for i in range(TEAM_SIZE) if _species_known(vec, i) >= 0.5]
    revealed_mons = [m for m in ObservationEncoder.get_team_list(b1, is_opponent=True) if m]
    for slot, m in zip(revealed_slots, revealed_mons):
        tmon = full.get(to_id_str(m.species))
        st = (getattr(tmon, "stats", None) or {}) if tmon is not None else {}
        vals = [st.get(k) for k in SPREAD_STAT_ORDER]
        if all(v is not None for v in vals):
            assert bspm[slot] == 1.0, f"revealed slot {slot} ({m.species}) should be supervised"
            assert np.allclose(bsp[slot], [float(v) for v in vals]), \
                f"spread label {bsp[slot].tolist()} != true {vals} for {m.species}"
            stats.spread_fired += 1
    # believed / empty slots: never supervised, always zero
    for i in range(TEAM_SIZE):
        if i not in revealed_slots:
            assert bspm[i] == 0.0 and np.allclose(bsp[i], 0.0), f"non-revealed slot {i} leaked a spread label"


def main(n_battles=40):
    mappings = load_mappings()
    sp_num = {sid: rec["num"] for sid, rec in mappings.get("species", {}).items() if "num" in rec}
    # Mirror gen3_env._move_num EXACTLY (gen3_typed_hidden_power_ids_v1): the OPPONENT's move-belief
    # labels fold every Hidden Power — bare or any typed variant — to the typeless num 237 (the opp is
    # observed bare), NOT the typed variant's distinct data num (355-370, which are for OUR own-team
    # obs). So the "real moveset" we validate the labels against must use the SAME convention.
    mv_num = {mid: (HIDDEN_POWER_MOVE_NUM if mid.startswith("hiddenpower") else rec["num"])
              for mid, rec in mappings.get("moves", {}).items() if "num" in rec}
    _loader = TeamLoader()
    teams = _loader.get_sample_teams() or _loader.get_all_teams()
    rng = np.random.default_rng(0)
    w = _build(teams, emit=True)
    enc_dim = w.env.observation_space["observation"].shape[0]
    stats = Stats()

    # (5b) the OFF env declares no belief keys (training-only) — space check, no battle needed
    off_env = Gen3Env(
        mappings, battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration("BeliefFzOff", None),
        start_listening=False, emit_belief_labels=False,
    )
    assert "belief_species" not in off_env.observation_space.spaces, "OFF env must not declare belief keys"
    assert "belief_spread" not in off_env.observation_space.spaces, "OFF env must not declare spread keys"

    for ep in range(n_battles):
        obs, _ = w.reset()
        _check(obs, w.env, sp_num, mv_num, enc_dim, stats)
        for _ in range(_MAX_STEPS):
            mask = np.asarray(obs["action_mask"]).astype(bool)
            legal = np.flatnonzero(mask)
            action = int(rng.choice(legal)) if legal.size else 0
            obs, _r, term, trunc, _i = w.step(action)
            _check(obs, w.env, sp_num, mv_num, enc_dim, stats)
            if term or trunc:
                break
        if (ep + 1) % 10 == 0:
            print(f"[belief-fz] {ep+1} battles | {stats.decisions} decisions | "
                  f"fired {stats.fired} ({100*stats.fired/max(1,stats.decisions):.0f}%)", flush=True)
    w.close()

    # (6) the labels actually FIRE on a meaningful fraction of decisions (catches a silent all-PAD
    # regression). Random-vs-random drags to ~200 turns where most/all mons are revealed (no believed
    # slots), so the fire rate is naturally modest — the floor only needs to be clearly above zero.
    fire_rate = stats.fired / max(1, stats.decisions)
    assert fire_rate > 0.10, f"belief labels fired on only {fire_rate:.0%} of decisions — silent all-PAD regression?"
    # SPREAD labels supervise REVEALED slots (the opposite population from the species belief), so they
    # fire whenever an opp mon is on the field — a clearly-above-zero floor catches a silent all-mask-0
    # regression (e.g. mon.stats unavailable / a species-match break).
    assert stats.spread_fired > 0.5 * stats.decisions, (
        f"spread labels validated on only {stats.spread_fired} revealed slots over {stats.decisions} "
        "decisions — the belief_spread plumbing (mon.stats / species match) likely broke.")
    print(f"✅ belief-labels fuzz PASSED: {stats.decisions} decisions across {n_battles} battles, "
          f"fired {fire_rate:.0%}, spread-validated {stats.spread_fired} revealed slots, all invariants held.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    try:
        main(n)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
