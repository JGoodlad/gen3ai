"""Fuzz: the per-team LUT's team signature is INVARIANT within a battle and matches the offline table.

`gen3_zarch_lut_v1` (v46). The LUT identifies OUR team from the OBSERVATION — sorted species(6) ⊕
moves(24) — so eval / frozen opponents / the prober need no plumbing. Two things must hold, and
neither can be established by reading the encoder:

  1. **INVARIANCE** — the signature computed from the live obs must be IDENTICAL at every decision
     of a battle. If it drifted (a move slot zeroed at 0 PP, a species re-encoded after a faint, a
     Knock Off touching something we read), the policy would silently be re-conditioned onto a
     DIFFERENT team's code mid-game. That is the GIGO class this test exists to make impossible.
  2. **AGREEMENT** — it must equal what `agents.model.team_signature.team_signature` computes
     OFFLINE from the same team's Showdown export. A producer/consumer mismatch would send every
     decision to row 0 (unconditioned) and quietly turn the whole experiment into a no-op — the
     failure that looks exactly like "the LUT didn't help".

The reader here is a DELIBERATELY separate implementation from the extractor's `_zarch_lut_index`;
an independent decode is what makes the agreement check meaningful.

Real battles over the in-process bridge (no server), per the project's fuzz convention:

    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/team_signature_fuzz_test.py [n_battles] [teams_glob]
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from agents.model.team_signature import (
    TEAM_SIGNATURE_MOVES, TEAM_SIGNATURE_SPECIES, team_signature)
from agents.observation.constants import POKEMON_FULL_DIM
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.gen3_env import Gen3Env
from utils.bridge.bridge_session import attach_bridge_transport
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

DEFAULT_TEAMS = "tmp/pool_cluster_n20/*.txt"
_MAX_STEPS = 200


def _sig_offsets(layout):
    pk = layout["pokemon"]
    sp = pk["species"]
    mv = pk["moves"]
    return (sp["offset"] + sp["layout"]["species_id"]["offset"],
            [mv["offset"] + s["offset"] for s in mv["layout"]["slots"]])


def _obs_signature(vec: np.ndarray, sp_idx: int, mv_idx) -> tuple:
    """Decode the signature the extractor computes, straight from the raw observation vector."""
    species, moves = [], []
    for slot in range(TEAM_SIGNATURE_SPECIES):
        base = slot * POKEMON_FULL_DIM
        species.append(int(vec[base + sp_idx]))
        for k in mv_idx:
            moves.append(int(vec[base + k]))
    return tuple(sorted(species)) + tuple(sorted(moves[:TEAM_SIGNATURE_MOVES]))


def main(n_battles: int = 4, teams_glob: str = DEFAULT_TEAMS) -> int:
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()["layout"]
    sp_idx, mv_idx = _sig_offsets(layout)

    team_files = sorted(glob.glob(teams_glob))
    if not team_files:
        print(f"[team-sig-fz] no teams matched {teams_glob!r}", file=sys.stderr)
        return 2
    pool = TeamLoader().get_all_teams()
    rng = np.random.default_rng(0)

    total = 0
    for i, path in enumerate(team_files):
        team_str = open(path, encoding="utf-8").read()
        expected = team_signature(team_str, mappings)

        env = Gen3Env(
            mappings, battle_format="gen3ou", team=Gen3Teambuilder([team_str]),
            opponent_team=Gen3Teambuilder(pool),
            account_configuration1=AccountConfiguration(f"TSigFz{i:03d}", None),
            start_listening=False)
        attach_bridge_transport(env, battle_format="gen3ou", persistent=True, recycle_every=0)
        opp = RandomPlayer(
            battle_format="gen3ou", team=Gen3Teambuilder(pool),
            account_configuration=AccountConfiguration(f"TSigFzO{i:03d}", None),
            start_listening=False)
        w = SingleAgentWrapper(env, opp)
        w.action_space = env.action_space
        w.observation_space = env.observation_space

        checked = 0
        for ep in range(n_battles):
            obs, _ = w.reset()
            first = None
            for _ in range(_MAX_STEPS):
                sig = _obs_signature(np.asarray(obs["observation"], np.float32), sp_idx, mv_idx)
                if first is None:
                    first = sig
                elif sig != first:
                    raise AssertionError(
                        f"[{os.path.basename(path)} ep{ep}] the team signature CHANGED MID-BATTLE.\n"
                        f"  first = {first}\n  now   = {sig}\n"
                        "The LUT would re-condition the policy onto a different team's code mid-game.")
                if sig != expected:
                    raise AssertionError(
                        f"[{os.path.basename(path)} ep{ep}] the LIVE signature does not match the "
                        "OFFLINE table entry — every decision would fall through to the unknown row "
                        f"(0) and the LUT would be a silent no-op.\n"
                        f"  live    = {sig}\n  offline = {expected}")
                checked += 1
                mask = np.asarray(obs["action_mask"]).astype(bool)
                legal = np.flatnonzero(mask)
                obs, _r, term, trunc, _i = w.step(int(rng.choice(legal)) if legal.size else 0)
                if term or trunc:
                    break
        w.close()
        total += checked
        print(f"  ✓ {os.path.basename(path):24s} {checked:5d} decisions — stable + matches offline",
              flush=True)

    print(f"\n[team-sig-fz] PASSED — {len(team_files)} team(s) × {n_battles} battles, {total} "
          "decisions: every live signature was CONSTANT within its battle and EQUAL to the offline "
          "table entry.")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    g = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TEAMS
    sys.exit(main(n, g))
