"""Racing-player reproduction — proves (or refutes) the self-play opponent staleness race.

Needs a LIVE Showdown server on a private 9XXX port (NEVER 8000/8001):
    npm run showdown -- 9123          # separate shell
    python src/agents/training/racing_player_fuzz_e2e_test.py [--episodes 40] [--widen 0.003] [--port 9123]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

WHY: static analysis was inconclusive — the per-battle lock (ps_client.py:202) is held across
`_choose_move`'s await, which *should* stop `battle2` mutating mid-decision, yet training crashed
with "Switch slot N (heracross) is not a currently available switch". This drives the EXACT
training stack — `Gen3Env` + `MaskableAgentWrapper` + an `RLPlayer` opponent polled on the
training thread (`single_agent_wrapper.py:44`) — and, inside the opponent's `action_to_order`,
compares the captured legality SNAPSHOT (`ctx.legal`) against the LIVE battle at serialize time.
A divergence == the race fired. `--widen` injects a sleep between snapshot and serialize to widen
the window so a rare race becomes reliably observable (and to test the lock hypothesis: if widening
produces NO divergence, the lock really does protect the bot and the bug is elsewhere).

The opponent picks a random LEGAL action (no model needed) but runs the full real snapshot /
record / serialize machinery — the race-relevant path. Random legal moves for the trainee too.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np

from poke_env import AccountConfiguration
from poke_env.ps_client.server_configuration import localhost_server_configuration

from agents.action.mapper import StaleDecisionError
from agents.battle.live_view import LegalActions
from agents.inference.player import RLPlayer
from agents.observation.state_encoder import load_mappings
from agents.training.gen3_env import Gen3Env
from agents.training.wrappers import MaskableAgentWrapper
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"


class _RaceProbeOpponent(RLPlayer):
    """Real RLPlayer machinery, but: picks a random legal action (no model), and detects
    snapshot-vs-live divergence at serialize time with a widenable race window."""

    def __init__(self, *args, widen_s: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._widen_s = widen_s
        self.divergences: list[dict] = []

    def _predict_best_action(self, battle, **kwargs):
        # Run the REAL obs build (this is where ctx.legal is captured + tracker.record fires),
        # then pick a random legal index from the mask instead of a model forward pass.
        od = self.embed_battle(battle)
        mask = np.asarray(od["action_mask"])
        legal = [i for i in range(len(mask)) if mask[i]]
        idx = random.choice(legal) if legal else None
        return idx, None, None

    def action_to_order(self, action_idx, battle):
        ctx = self._get_tracker(battle).last_ctx
        snap_sw = tuple(ctx.legal.switch_slots)
        snap_mv = tuple(ctx.legal.move_ids)
        if self._widen_s:
            # Yields the GIL (sleep) so POKE_LOOP can run during the snapshot→serialize window.
            time.sleep(self._widen_s)
        live = LegalActions.from_battle(battle)
        live_sw, live_mv = tuple(live.switch_slots), tuple(live.move_ids)
        if live_sw != snap_sw or live_mv != snap_mv:
            self.divergences.append({
                "turn": getattr(battle, "turn", None),
                "snap_switches": snap_sw, "live_switches": live_sw,
                "snap_moves": snap_mv, "live_moves": live_mv,
            })
        try:
            return super().action_to_order(action_idx, battle)
        except StaleDecisionError:
            # Diagnostic-only: this probe measures raw snapshot-vs-live DIVERGENCE rate (recorded
            # above), so it short-circuits to a default to keep the run alive and collect every
            # occurrence. The PRODUCTION opponent instead RE-DECIDES on the now-current request
            # (RLPlayer.choose_move) — see redecide_rollback_fuzz_test.py for that path.
            return self.choose_default_move()


def _build(port: int, widen_s: float):
    mappings = load_mappings()
    teams = TeamLoader().get_sample_teams()
    sc = localhost_server_configuration(port)
    ts = int(time.time()) % 100000
    env = Gen3Env(
        mappings,
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(teams),
        server_configuration=sc,
        account_configuration1=AccountConfiguration(f"RaceTr{ts}", "password"),
    )
    opp = _RaceProbeOpponent(
        model=None,
        team=Gen3Teambuilder(teams),
        battle_format=BATTLE_FORMAT,
        server_configuration=sc,
        mappings=mappings,
        account_configuration=AccountConfiguration(f"RaceOp{ts}", "password"),
        start_listening=False,            # opponent is a brain, the env's agent2 is the connection
        widen_s=widen_s,
    )
    wrapped = MaskableAgentWrapper(env, opp)
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space
    return wrapped, opp, env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--widen", type=float, default=0.003, help="snapshot→serialize widen sleep (s)")
    ap.add_argument("--port", type=int, default=9123)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    print(f"Racing-player repro — {args.episodes} episodes — widen={args.widen*1000:.1f}ms — :{args.port}")
    wrapped, opp, env = _build(args.port, args.widen)
    rng = random.Random(args.seed)
    finished = 0
    try:
        for ep in range(args.episodes):
            obs, _ = wrapped.reset()
            done = False
            steps = 0
            while not done and steps < 400:
                mask = np.asarray(obs["action_mask"])
                legal = [i for i in range(len(mask)) if mask[i]]
                a = rng.choice(legal) if legal else 0
                obs, _r, term, trunc, _info = wrapped.step(a)
                done = bool(term or trunc)
                steps += 1
            finished += 1
            n_dec, n_def, n_stale = wrapped.opponent_default_stats()
            print(f"  ep {ep+1:>3}/{args.episodes}: {steps:>3} steps | "
                  f"divergences={len(opp.divergences):>3} | stale_defaults={n_stale:>3}/{n_dec}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    n_dec, n_def, n_stale = wrapped.opponent_default_stats()
    print(f"\n{'='*68}")
    print(f"episodes finished : {finished}/{args.episodes}")
    print(f"opponent decisions: {n_dec}")
    print(f"stale-defaults    : {n_stale}  (opponent caught its own decision going stale)")
    print(f"divergences caught: {len(opp.divergences)}  (snapshot != live at serialize time)")
    if opp.divergences:
        print("\n--- first divergences ---")
        for d in opp.divergences[:8]:
            sw = "" if d["snap_switches"] == d["live_switches"] else \
                 f" SWITCHES snap={d['snap_switches']} live={d['live_switches']}"
            mv = "" if d["snap_moves"] == d["live_moves"] else \
                 f" MOVES snap={d['snap_moves']} live={d['live_moves']}"
            print(f"  turn {d['turn']}:{sw}{mv}")
        print("\n>>> RACE REPRODUCED: the opponent's decision snapshot diverged from the live "
              "battle at serialize time. <<<")
    else:
        print("\n>>> No divergence observed at this widen/episode count. <<<")
    sys.exit(0)


if __name__ == "__main__":
    main()
