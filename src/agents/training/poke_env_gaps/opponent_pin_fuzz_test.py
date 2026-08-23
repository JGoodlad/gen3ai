"""Bridge fuzz test — a PINNED opponent really pilots ITS OWN team, per episode (fold-back).

The league fold-back contract: a specialist stable/exploiter opponent (its run trained on ONE
pinned team) must bring THAT team on the episodes it plays, while bot/pool episodes keep the
shared pool — switched per episode by ``MaskableAgentWrapper._apply_opponent_team`` (env.agent2
does the opponent-side networking, so its ``_team`` decides the opponent's REAL team — the
training-mirror lesson applied to the opponent side). This drives the REAL construction path
(Gen3Env(team=, opponent_team=) → MaskableAgentWrapper(exploiter_team=…) → bridge) over real
battles with keep-bots 0.5 alternating pinned-target and bot episodes, parses the protocol
stream, and asserts per episode:

  1. The trainee always fields its own pin (identifies the sides unambiguously).
  2. On EXPLOITER episodes (ground truth: ``wrapped.opponent is exploiter``) the opponent fields
     EXACTLY the opponent pin.
  3. On BOT episodes the opponent does NOT field the opponent pin (a pool draw), and rosters vary.

Run directly (no server; in-process bridge):
    python src/agents/training/poke_env_gaps/opponent_pin_fuzz_test.py [n_episodes]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import re
import sys
import traceback

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

from agents.observation.state_encoder import load_mappings
from agents.training.gen3_env import Gen3Env
from agents.training.wrappers import MaskableAgentWrapper
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from utils.bridge.bridge_session import attach_bridge_transport

BATTLE_FORMAT = "gen3ou"
TSS = "data/teams/specialist/tss_starmie.txt"
_SWITCH_RE = re.compile(r"\|(?:switch|drag)\|(p[12])[a-z]?: [^|]*\|([^,|]+)")


def _species(lines, side):
    return {m.group(2).strip() for ln in lines for m in [_SWITCH_RE.match(ln)] if m and m.group(1) == side}


def _pin_species(pin_str):
    return {ln.split("@")[0].split("(")[0].strip()
            for ln in pin_str.splitlines() if ln.strip() and "@" in ln}


def main(n_episodes: int = 10) -> int:
    loader = TeamLoader()
    all_teams, sample_teams = loader.get_all_teams(), loader.get_sample_teams()
    with open(TSS, encoding="utf-8") as f:
        trainee_pin = f.read()
    # The opponent's pin: a sample team DIFFERENT from the trainee's (so sides classify exactly).
    opp_pin = next(t for t in sample_teams if _pin_species(t) != _pin_species(trainee_pin))
    trainee_species, opp_pin_species = _pin_species(trainee_pin), _pin_species(opp_pin)
    assert len(trainee_species) == 6 and len(opp_pin_species) == 6

    pool_tb = Gen3Teambuilder(all_teams)
    env = Gen3Env(
        load_mappings(), battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder([trainee_pin]),
        opponent_team=pool_tb,
        account_configuration1=AccountConfiguration("OppPinFz", None),
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

    # The "exploiter target" stand-in: any decision-function Player works (the wrapper doesn't
    # care it isn't an RLPlayer); its pinned team rides exploiter_team, NOT its own team= kwarg.
    exploiter = RandomPlayer(battle_format=BATTLE_FORMAT,
                             account_configuration=AccountConfiguration("OppPinTgt", None),
                             start_listening=False)
    bots = [RandomPlayer(battle_format=BATTLE_FORMAT,
                         account_configuration=AccountConfiguration(f"OppPinBot{i}", None),
                         start_listening=False) for i in range(2)]
    wrapped = MaskableAgentWrapper(
        env, heuristic_opponents=bots, rng_seed=7,
        exploiter_player=exploiter, exploiter_keep_bots=True, exploiter_bot_fraction=0.5,
        exploiter_team=Gen3Teambuilder([opp_pin]), opponent_pool_team=pool_tb,
    )
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space

    rng = np.random.default_rng(0)
    n_pinned = n_bot = 0
    bot_rosters = []
    try:
        for ep in range(n_episodes):
            lines.clear()
            obs, _ = wrapped.reset()
            vs_target = wrapped.opponent is exploiter   # ground truth for this episode
            for _ in range(600):
                mask = np.asarray(obs["action_mask"]).astype(bool)
                legal = np.flatnonzero(mask)
                obs, _r, term, trunc, _i = wrapped.step(int(rng.choice(legal)) if legal.size else 0)
                if term or trunc:
                    break
            p1, p2 = _species(lines, "p1"), _species(lines, "p2")
            ours, theirs = (p1, p2) if p1 <= trainee_species else (p2, p1)
            assert ours <= trainee_species and ours, (
                f"ep {ep}: trainee fielded {sorted(ours)} ⊄ its pin {sorted(trainee_species)}")
            if vs_target:
                n_pinned += 1
                assert theirs <= opp_pin_species and theirs, (
                    f"PIN VIOLATION ep {ep}: exploiter episode but opponent fielded "
                    f"{sorted(theirs)} ⊄ its pin {sorted(opp_pin_species)}")
            else:
                n_bot += 1
                assert not (theirs <= opp_pin_species), (
                    f"LEAKED PIN ep {ep}: bot episode but opponent fielded the PIN "
                    f"{sorted(theirs)} — the pool builder was not restored")
                bot_rosters.append(frozenset(theirs))
        assert n_pinned >= 2 and n_bot >= 2, (
            f"selection did not exercise both branches (pinned={n_pinned}, bot={n_bot}) — "
            "raise n_episodes or change rng_seed")
    except Exception:
        traceback.print_exc()
        return 1
    print(f"PASS — {n_episodes} episodes: {n_pinned} pinned-target episodes fielded EXACTLY the "
          f"opponent pin; {n_bot} bot episodes restored the pool "
          f"({len(set(bot_rosters))} distinct rosters)")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
