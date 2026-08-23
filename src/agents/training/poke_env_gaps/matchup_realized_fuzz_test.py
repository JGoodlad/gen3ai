"""Bridge fuzz test — the REALIZED matchup equals the DECLARED MatchupSpec.

The permanent form of the probe that caught the training-mirror bug: a `--trainee-team` pin used to
silently pin the OPPONENTS too (PokeEnv's single `team=` feeds both env agents), so every specialist
run trained a single-team mirror at ~100% WR without any test noticing — the declaration and the
realized battles had no reconciliation. This drives the REAL construction path (MatchupSpec →
teambuilders → Gen3Env(team=, opponent_team=) → bridge → MaskableAgentWrapper) over real battles,
parses the trainee's own protocol stream, and asserts per episode:

  1. p1 (the trainee) fields EXACTLY the declared pinned team — every episode.
  2. p2 (the opponent) does NOT field the pinned team (the mirror signature); across the run the
     opponents' rosters VARY (a pool draw, not any single fixed team).

Run directly (no server; in-process bridge):
    python src/agents/training/poke_env_gaps/matchup_realized_fuzz_test.py [n_episodes]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import re
import sys
import traceback
from types import SimpleNamespace

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from agents.observation.state_encoder import load_mappings
from agents.training.gen3_env import Gen3Env
from agents.training.matchup_spec import MatchupSpec
from utils.team_loader.loader import TeamLoader
from utils.bridge.bridge_session import attach_bridge_transport

BATTLE_FORMAT = "gen3ou"
TSS = "data/teams/specialist/tss_starmie.txt"
_SWITCH_RE = re.compile(r"\|(?:switch|drag)\|(p[12])[a-z]?: [^|]*\|([^,|]+)")


def _species(lines, side):
    return {m.group(2).strip() for ln in lines for m in [_SWITCH_RE.match(ln)] if m and m.group(1) == side}


def main(n_episodes: int = 8) -> int:
    loader = TeamLoader()
    all_teams, sample_teams = loader.get_all_teams(), loader.get_sample_teams()
    spec = MatchupSpec.from_args(SimpleNamespace(
        trainee_team=TSS, exploiter=None, self_play=False, bot_weights=None,
        exploiter_keep_bots=False, exploiter_bot_fraction=0.5,
        exploiter_temp_start=None, exploiter_temp_mode="fixed", stable_opponent_temp=1.0))
    pinned = {ln.split("@")[0].split("(")[0].strip()
              for ln in spec.trainee_teams.pin_str.splitlines() if ln.strip() and "@" in ln}
    assert len(pinned) == 6, pinned

    env = Gen3Env(
        load_mappings(), battle_format=BATTLE_FORMAT,
        team=spec.trainee_teams.build(all_teams, sample_teams),
        opponent_team=spec.opponent_teams.build(all_teams, sample_teams),
        account_configuration1=AccountConfiguration("MatchupFz", None),
        start_listening=False,
    )
    lines: list = []
    orig = env.agent1._handle_battle_message

    async def capture(split_messages):
        for msg in split_messages:
            lines.append("|" + "|".join(str(x) for x in msg[1:]) if len(msg) > 1 else "|")
        await orig(split_messages)

    env.agent1._handle_battle_message = capture     # BEFORE attach (bound-handler capture)
    attach_bridge_transport(env, battle_format=BATTLE_FORMAT, persistent=True, recycle_every=10000)
    opponent = RandomPlayer(battle_format=BATTLE_FORMAT,
                            account_configuration=AccountConfiguration("MatchupFzOpp", None),
                            start_listening=False)
    wrapped = SingleAgentWrapper(env, opponent)
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space

    rng = np.random.default_rng(0)
    opp_rosters = []
    try:
        for ep in range(n_episodes):
            lines.clear()
            obs, _ = wrapped.reset()
            for _ in range(600):
                mask = np.asarray(obs["action_mask"]).astype(bool)
                legal = np.flatnonzero(mask)
                obs, _r, term, trunc, _i = wrapped.step(int(rng.choice(legal)) if legal.size else 0)
                if term or trunc:
                    break
            p1, p2 = _species(lines, "p1"), _species(lines, "p2")
            # Which side is the trainee? Resolve by the pinned roster (role can be p1 or p2).
            ours, theirs = (p1, p2) if p1 <= pinned else (p2, p1)
            # (1) the trainee fields ONLY the declared team
            assert ours <= pinned and ours, (
                f"REALIZED≠DECLARED ep {ep}: trainee fielded {sorted(ours)} ⊄ pinned {sorted(pinned)}")
            # (2) the opponent is NOT the pinned team (the mirror signature)
            assert not (theirs <= pinned), (
                f"MIRROR ep {ep}: opponent fielded a subset of the PINNED team {sorted(theirs)} — "
                "the opponent_team seam is not taking effect")
            opp_rosters.append(frozenset(theirs))
        # (3) opponents VARY across episodes (a pool draw, not one fixed team)
        assert len(set(opp_rosters)) >= max(2, n_episodes // 4), (
            f"opponent rosters did not vary: {[sorted(r) for r in opp_rosters]}")
    except Exception:
        traceback.print_exc()
        return 1
    print(f"PASS — {n_episodes} episodes: trainee == declared pin every episode; "
          f"{len(set(opp_rosters))} distinct opponent rosters (pool draws)")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
