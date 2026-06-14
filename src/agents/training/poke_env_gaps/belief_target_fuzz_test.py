"""Bridge fuzz test for the LATENT-belief target (`belief_target_slots`, the privileged per-mon encode).

Unit tests pin the latent loss math + the model wiring on synthetic tensors; this exercises the real
BATTLE EXTRACTION — `Gen3Env._build_belief_target_slots` reading `battle2.team`, assigning each hidden
mon to its believed slot (the SAME `assign_hidden_to_slots` as the species labels), and emitting the
fresh per-mon obs encode — against thousands of real Showdown states from the full gen3ou pool.

Per-decision invariants asserted against the LIVE battle (a violation raises immediately):
  1. `belief_target_slots` is present, shape [6,107] float32, when emit_belief_target=True.
  2. SAME ASSIGNMENT: every slot with a target (any non-zero) is exactly a believed slot
     (belief_species>=0) — targets and species labels name the SAME slots.
  3. CORRECTNESS: each believed slot's target == an INDEPENDENT fresh encode of the mon the species
     label names (pokemon_encoder.encode(mon, battle2, is_own=True) + active=0). This validates the
     env's per-battle cache + the canonical assignment end to end.
  4. PAD slots (revealed / non-target) are all-zero.
  5. LEAK: belief_target_slots is NOT inside obs["observation"] (its width is exactly the encoder dim,
     so the privileged target cannot reach the forward).

Scenario invariant (over a whole run):
  6. The target actually FIRES on a meaningful fraction of mid-battle decisions.

Run directly (no server needed; in-process via the local BattleStream bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/belief_target_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import sys
import traceback

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.data.normalize import to_id_str
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from agents.training.gen3_env import Gen3Env
from agents.observation.state_encoder import load_mappings
from agents.observation.constants import POKEMON_FULL_DIM, TEAM_SIZE
from utils.bridge.bridge_session import attach_bridge_transport
from utils.teambuilder import Gen3Teambuilder
from utils.team_loader.loader import TeamLoader

_MAX_STEPS = 200


def _build(teams):
    env = Gen3Env(
        load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration("TgtFz", None),
        start_listening=False, emit_belief_labels=True, emit_belief_target=True,
    )
    attach_bridge_transport(env, battle_format="gen3ou", persistent=True, recycle_every=0)
    opp = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration=AccountConfiguration("TgtFzOpp", None), start_listening=False,
    )
    w = SingleAgentWrapper(env, opp)
    w.action_space = env.action_space
    w.observation_space = env.observation_space
    return w


class Stats:
    def __init__(self):
        self.decisions = 0
        self.fired = 0


def _fresh_encode(env, mon, b2):
    """Independent recompute of a mon's fresh 107-d slot (the env caches; this re-derives)."""
    enc = env.observation_encoder.pokemon_encoder.encode(mon, b2, is_own=True)
    slot = np.zeros(POKEMON_FULL_DIM, dtype=np.float32)
    slot[: len(enc)] = enc
    return slot


def _check(obs, env, sp_num, enc_dim, stats):
    tgt = obs.get("belief_target_slots")
    bs = obs.get("belief_species")
    assert tgt is not None and bs is not None, "belief_target_slots / belief_species missing"
    assert tgt.shape == (TEAM_SIZE, POKEMON_FULL_DIM) and tgt.dtype == np.float32, (tgt.shape, tgt.dtype)
    vec = obs["observation"]
    # (5) leak: the obs vector is exactly the encoder width — the target is NOT in it
    assert vec.shape[0] == enc_dim, f"obs width {vec.shape[0]} != encoder dim {enc_dim}"
    b2 = getattr(env, "battle2", None)
    if getattr(env, "battle1", None) is None or b2 is None:
        return
    stats.decisions += 1
    by_species_num = {sp_num.get(to_id_str(m.species)): m for m in b2.team.values()}
    nonzero_slots = {i for i in range(TEAM_SIZE) if np.any(tgt[i] != 0.0)}
    believed = {i for i in range(TEAM_SIZE) if bs[i] >= 0}
    # (2) same assignment: a target exists exactly where the species label exists (modulo a mon whose
    # fresh encode happens to be all-zero, which never occurs — species_known=1 alone makes it nonzero)
    assert nonzero_slots == believed, f"target slots {nonzero_slots} != believed slots {believed}"
    if believed:
        stats.fired += 1
    for i in believed:
        # (3) the target IS the fresh encode of the mon the SPECIES label names (same mon, both heads)
        mon = by_species_num.get(int(bs[i]))
        assert mon is not None, f"species num {int(bs[i])} not in battle2.team"
        expected = _fresh_encode(env, mon, b2)
        assert np.allclose(tgt[i], expected, atol=1e-6), (
            f"slot {i}: target != fresh encode of {mon.species} (max |Δ| "
            f"{np.abs(tgt[i] - expected).max():.3g})")
    # (4) non-target slots are all-zero
    for i in range(TEAM_SIZE):
        if i not in believed:
            assert not np.any(tgt[i]), f"non-target slot {i} is not all-zero"


def main(n_battles=40):
    mappings = load_mappings()
    sp_num = {sid: rec["num"] for sid, rec in mappings.get("species", {}).items() if "num" in rec}
    _loader = TeamLoader()
    teams = _loader.get_sample_teams() or _loader.get_all_teams()
    rng = np.random.default_rng(0)
    w = _build(teams)
    enc_dim = w.env.observation_space["observation"].shape[0]
    stats = Stats()

    # OFF env declares no target key (training-only)
    off_env = Gen3Env(
        mappings, battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration("TgtFzOff", None),
        start_listening=False, emit_belief_labels=True, emit_belief_target=False,
    )
    assert "belief_target_slots" not in off_env.observation_space.spaces, "OFF env must not declare the target key"

    for ep in range(n_battles):
        obs, _ = w.reset()
        _check(obs, w.env, sp_num, enc_dim, stats)
        for _ in range(_MAX_STEPS):
            mask = np.asarray(obs["action_mask"]).astype(bool)
            legal = np.flatnonzero(mask)
            action = int(rng.choice(legal)) if legal.size else 0
            obs, _r, term, trunc, _i = w.step(action)
            _check(obs, w.env, sp_num, enc_dim, stats)
            if term or trunc:
                break
        if (ep + 1) % 10 == 0:
            print(f"[target-fz] {ep+1} battles | {stats.decisions} decisions | "
                  f"fired {stats.fired} ({100*stats.fired/max(1,stats.decisions):.0f}%)", flush=True)
    w.close()

    fire_rate = stats.fired / max(1, stats.decisions)
    assert fire_rate > 0.10, f"belief target fired on only {fire_rate:.0%} of decisions — silent all-PAD regression?"
    print(f"✅ belief-target fuzz PASSED: {stats.decisions} decisions across {n_battles} battles, "
          f"fired {fire_rate:.0%}, target==fresh-encode + no-leak held.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    try:
        main(n)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
