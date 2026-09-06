"""The BOT CALIBRATION arm: each side, in the same session and against the same server,
plays the shared anchor bots.

This is the check on the ELO prediction. The anchored ratings are extrapolations from ~5% loss
rates against these bots, so if the head-to-head disagrees with the ladder, the bot rows say
whether the two eras' *anchor* performance also disagrees with what the ladder recorded — i.e.
whether the extrapolation or the head-to-head is the thing that moved.

Both sides run this in their OWN process on their OWN era's code, but the bots themselves are
comparable: `diff`ing the two trees' `agents/opponents.py` shows the bot decision LOGIC is
identical, and the only change is an opt-in per-instance RNG for the staller's Protect coin
whose default is the era's own `random.random()`. So a bot row measured on the era side and one
measured on the current side are measuring the same opponent.

Unlike the head-to-head, both players live in ONE process here, so `battle_against` is used and
no cross-process ordering is involved. Teams come from the same plan file, so each side faces
each bot on the same team sequence.

Run:
    <runner>.sh bots.py --plan plan.json --model <ckpt> --label v8_14 --out <jsonl> --n-games 120
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import argparse
import asyncio
import hashlib
import json
import os
import time

from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import localhost_server_configuration

from side import SequenceTeambuilder, packed_species   # same directory


# Present with identical decision logic in BOTH trees.
BOTS = {
    "heuristic2": "Gen3HeuristicV2Player",
    "staller_v2": "Gen3StallerV2Player",
    "aggressive_v2": "Gen3AggressiveV2Player",
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True, help="account-name suffix, must be unique")
    ap.add_argument("--n-games", type=int, default=120)
    ap.add_argument("--port", type=int, default=9137)
    ap.add_argument("--format", default="gen3ou")
    ap.add_argument("--bots", default=",".join(BOTS))
    args = ap.parse_args()

    if args.port in (8000, 8001):
        raise SystemExit(f"refusing --port {args.port}")

    import agents.opponents as opponents
    from sb3_contrib import MaskablePPO
    from agents.inference.player import RLPlayer
    from agents.observation.state_encoder import load_mappings

    plan = json.load(open(args.plan))
    team_texts = []
    for t in plan["teams"]:
        text = open(os.path.join(plan["team_dir"], t["name"])).read()
        got = hashlib.sha256(text.encode()).hexdigest()
        if got != t["sha256"]:
            raise SystemExit(f"team {t['name']}: sha256 {got} != plan's {t['sha256']}")
        team_texts.append(text)
    index_of = {t["name"]: i for i, t in enumerate(plan["teams"])}
    games = plan["games"][: args.n_games]
    model_seq = [index_of[g["side_a_team"]] for g in games]
    bot_seq = [index_of[g["side_b_team"]] for g in games]

    model = MaskablePPO.load(args.model, env=None, device="cpu")
    for m in model.policy.modules():
        if hasattr(m, "_debugger"):
            m._debugger = None

    sc = localhost_server_configuration(args.port)
    mappings = load_mappings()
    rows = []

    for bot_key in args.bots.split(","):
        bot_key = bot_key.strip()
        cls_name = BOTS[bot_key]
        cls = getattr(opponents, cls_name, None)
        if cls is None:
            print(f"[{args.label}] SKIP {bot_key}: {cls_name} absent in this era", flush=True)
            continue

        # A fresh player per bot so the team sequence restarts and the accounts stay distinct.
        model_tb = SequenceTeambuilder(team_texts, model_seq)
        bot_tb = SequenceTeambuilder(team_texts, bot_seq)
        short = bot_key.replace("_", "")[:6]
        me = RLPlayer(
            model=model, team=model_tb, battle_format=args.format,
            server_configuration=sc, mappings=mappings,
            account_configuration=AccountConfiguration(f"G3b{args.tag}{short}m", "password"),
            max_concurrent_battles=1, stochastic=False, temperature=1.0,
        )
        bot = cls(
            battle_format=args.format, team=bot_tb, max_concurrent_battles=1,
            server_configuration=sc,
            account_configuration=AccountConfiguration(f"G3b{args.tag}{short}b", "password"),
        )
        t0 = time.time()
        try:
            await asyncio.wait_for(
                me.battle_against(bot, n_battles=len(games)),
                120.0 + 10.0 * len(games),
            )
            hit = False
        except asyncio.TimeoutError:
            hit = True

        finished = me.n_finished_battles
        won = me.n_won_battles
        for k, (tag, battle) in enumerate(me.battles.items()):
            rows.append({
                "label": args.label, "bot": bot_key, "battle_tag": tag,
                "order_index": k, "won": battle.won, "finished": battle.finished,
                "turns": battle.turn,
            })
        print(f"[{args.label}] {bot_key}: finished={finished}/{len(games)} won={won} "
              f"wr={(won/finished if finished else float('nan')):.3f} "
              f"timeouts={len(games)-finished} elapsed={time.time()-t0:.0f}s "
              f"deadline_hit={hit}", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{args.label}] -> {args.out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
