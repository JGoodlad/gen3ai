"""One SIDE of the cross-era head-to-head. Runs in its own process, on its own era's code.

The two generations cannot load in one process (v8 is config_version 45 / obs 2992; current
code refuses it), so each side is a separate process and they meet over a websocket Showdown
server as ordinary clients.

This file is deliberately written against the COMMON SUBSET of the two eras' APIs — the
`RLPlayer(model, team, battle_format, server_configuration, mappings, account_configuration,
max_concurrent_battles, stochastic, temperature)` constructor is present and identical at both
commits — so ONE script drives both sides and neither era gets a bespoke code path that could
differ in a way the measurement cannot see.

Run it with the matching runner so the right tree is on PYTHONPATH:
    PYTHONPATH=<current worktree>/src   python side.py --role a ...
    PYTHONPATH=/tmp/v8rep_era/src       python side.py --role b ...
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Two era-specific facts this encodes, both verified rather than assumed:

* **Every account carries a non-empty password.** The era's vendored poke-env still has the
  passwordless-login hang: the server greets each connection with ``|updateuser| Guest N``
  before ``|challstr|``, and the era's client sets ``logged_in`` on that, so a passwordless
  challenger fires ``/challenge`` at a user that does not exist yet and BOTH sides wait
  forever. Current code guards it with ``_trn_sent``; the era does not. A password makes the
  buggy branch unreachable, which is why every era eval account has one.
* **The ERA is the ACCEPTOR.** The challenger is the side that races on ``logged_in``, so the
  side with the unfixed client is given the role that does not race.

Action selection is GREEDY on both sides (``stochastic=False``), which is each era's own eval
convention — "eval = greedy yardstick" appears verbatim in both trees' eval_worker.py.
"""

import argparse
import asyncio
import hashlib
import json
import os
import time

from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import localhost_server_configuration
from utils.teambuilder import Gen3Teambuilder


# ---------------------------------------------------------------------------
# The team sequence
# ---------------------------------------------------------------------------

def packed_species(packed_team: str):
    """The species in a packed team string, lowercased and sorted.

    The packed format is ``NICKNAME|SPECIES|ITEM|ABILITY|MOVES|...`` per mon, mons joined by
    ``]``. SPECIES is empty when the nickname IS the species, which is the common case for our
    pool, so fall back to field 0.
    """
    out = []
    for mon in packed_team.split("]"):
        if not mon.strip():
            continue
        fields = mon.split("|")
        species = (fields[1] if len(fields) > 1 and fields[1] else fields[0])
        out.append("".join(ch for ch in species.lower() if ch.isalnum()))
    return sorted(out)


class SequenceTeambuilder(Gen3Teambuilder):
    """Yields a PREDETERMINED team per battle instead of drawing one.

    Subclassing the real teambuilder (rather than reimplementing it) keeps the era's own
    validation + packing in the path, so the team this side fields is packed by the same code
    that packs it in training. ``packed_teams`` is parallel to the constructor's team list in
    both eras, which the constructor asserts.
    """

    def __init__(self, team_texts, sequence):
        super().__init__(list(team_texts))
        if len(self.packed_teams) != len(team_texts):
            raise SystemExit(
                f"teambuilder packed {len(self.packed_teams)} teams from {len(team_texts)} "
                f"inputs — the index↔team correspondence this harness relies on is broken"
            )
        self._sequence = list(sequence)
        self._n_yield = 0
        self.yielded = []          # pool index per yield, in call order

    def yield_team(self):
        # Clamp rather than raise: poke-env may ask once more than it plays (a challenge that
        # is never accepted). An extra yield must not kill a finished run; the realized team is
        # verified per battle from the battle object anyway.
        i = self._sequence[min(self._n_yield, len(self._sequence) - 1)]
        self._n_yield += 1
        self.yielded.append(i)
        return self.packed_teams[i]


# ---------------------------------------------------------------------------

def build_player(args, teambuilder, server_config):
    from sb3_contrib import MaskablePPO
    from agents.inference.player import RLPlayer
    from agents.observation.state_encoder import load_mappings

    t0 = time.time()
    model = MaskablePPO.load(args.model, env=None, device="cpu")
    # The era's RLPlayer has no debugger guard, and a `--log-level periodic` checkpoint carries
    # a live ObservationDebugger that print()s a full 12-mon board on every forward.
    for m in model.policy.modules():
        if hasattr(m, "_debugger"):
            m._debugger = None
    print(f"[{args.label}] model loaded in {time.time() - t0:.1f}s", flush=True)

    return RLPlayer(
        model=model,
        team=teambuilder,
        battle_format=args.format,
        server_configuration=server_config,
        mappings=load_mappings(),
        # A NON-EMPTY password is load-bearing on the era side — see the module docstring.
        account_configuration=AccountConfiguration(args.name, "password"),
        max_concurrent_battles=1,
        stochastic=False,          # greedy: both eras' eval convention
        temperature=1.0,
    )


def collect_results(player, args, plan_games, teambuilder, team_meta):
    """One record per finished battle, joined to the plan by battle tag order.

    Games are strictly sequential (``max_concurrent_battles=1``), so the k-th battle is the
    k-th plan game. That ordering is not TRUSTED: every record carries the species actually
    fielded and the species the plan intended, and `team_match` reports whether they agree.
    """
    key = "side_a" if args.role == "a" else "side_b"
    # poke-env keeps battles in insertion order, which is the order they started.
    tags = list(player.battles.keys())
    rows = []
    for k, tag in enumerate(tags):
        battle = player.battles[tag]
        game = plan_games[k] if k < len(plan_games) else None
        played = sorted(
            "".join(ch for ch in p.species.lower() if ch.isalnum())
            for p in battle.team.values()
        )
        intended_name = game[f"{key}_team"] if game else None
        intended = team_meta[intended_name]["species"] if intended_name else None
        rows.append({
            "battle_tag": tag,
            "side": args.role,
            "label": args.label,
            "order_index": k,
            "game_index": game["game_index"] if game else None,
            "pair_index": game["pair_index"] if game else None,
            "orientation": game["orientation"] if game else None,
            "intended_team": intended_name,
            "intended_sha": game[f"{key}_sha"] if game else None,
            "played_species": played,
            "team_match": (played == intended) if intended is not None else None,
            "won": battle.won,
            "finished": battle.finished,
            "turns": battle.turn,
        })
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--role", choices=("a", "b"), required=True,
                    help="which plan role this process plays")
    ap.add_argument("--mode", choices=("challenge", "accept"), required=True,
                    help="the ERA side should be 'accept' — the challenger races on logged_in")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True, help="Showdown account name")
    ap.add_argument("--opponent", required=True, help="the other side's account name")
    ap.add_argument("--label", required=True, help="human label, e.g. v8_14 / v9_59")
    ap.add_argument("--out", required=True, help="JSONL to write")
    ap.add_argument("--port", type=int, default=9137)
    ap.add_argument("--format", default="gen3ou")
    ap.add_argument("--n-games", type=int, default=None,
                    help="default: every game in the plan")
    ap.add_argument("--deadline-s", type=float, default=None,
                    help="total wall clock; default 120s + 10s/game")
    args = ap.parse_args()

    if args.port in (8000, 8001):
        raise SystemExit(f"refusing --port {args.port}: 8000 is the dev server, "
                         f"8001 the live TRAINING server")

    plan = json.load(open(args.plan))
    games = plan["games"][: args.n_games] if args.n_games else plan["games"]

    # Team texts, in the plan's own order, read from the plan's own directory. The plan
    # records each team's sha256; re-verify it here so a side that somehow reads a different
    # file cannot silently play a different team.
    team_meta, team_texts = {}, []
    for t in plan["teams"]:
        text = open(os.path.join(plan["team_dir"], t["name"])).read()
        got = hashlib.sha256(text.encode()).hexdigest()
        if got != t["sha256"]:
            raise SystemExit(f"team {t['name']}: sha256 {got} != plan's {t['sha256']}")
        team_texts.append(text)
        team_meta[t["name"]] = {"sha256": got}
    index_of = {t["name"]: i for i, t in enumerate(plan["teams"])}

    key = "side_a" if args.role == "a" else "side_b"
    sequence = [index_of[g[f"{key}_team"]] for g in games]

    server_config = localhost_server_configuration(args.port)
    teambuilder = SequenceTeambuilder(team_texts, sequence)
    for name, i in index_of.items():
        team_meta[name]["species"] = packed_species(teambuilder.packed_teams[i])

    player = build_player(args, teambuilder, server_config)
    n = len(games)
    deadline = args.deadline_s if args.deadline_s is not None else 120.0 + 10.0 * n
    print(f"[{args.label}] role={args.role} mode={args.mode} n_games={n} "
          f"deadline={deadline:.0f}s as {args.name} vs {args.opponent}", flush=True)

    # ── FAIL FAST ON A REFUSED LOGIN ────────────────────────────────────────────────
    # `localhost_server_configuration` still authenticates against the REAL Smogon
    # action.php, so a SHORT account name that somebody has registered upstream is
    # refused ("Wrong password") even though our own server runs --no-security. Neither
    # era's `send_challenges`/`accept_challenges` is wrapped by the connect-or-raise
    # guard (only `_battle_against` is), so without this check a refused login costs the
    # WHOLE deadline in silence — measured: both sides sat for 320 s and reported 20
    # timeouts. Use a distinctive, unregistered name; this turns the mistake into a
    # 30-second error instead of a lost run.
    try:
        await asyncio.wait_for(player.ps_client.logged_in.wait(), 30.0)
    except asyncio.TimeoutError:
        raise SystemExit(
            f"[{args.label}] {args.name}: not logged in after 30s. A refused login is the "
            f"usual cause — localhost still authenticates against the real Smogon "
            f"action.php, so pick an account name nobody has registered upstream."
        )
    print(f"[{args.label}] logged in as {args.name}", flush=True)

    t0 = time.time()
    timed_out = False
    try:
        if args.mode == "challenge":
            # Give the acceptor a moment to be listening before the first /challenge.
            await asyncio.sleep(5.0)
            await asyncio.wait_for(player.send_challenges(args.opponent, n), deadline)
        else:
            await asyncio.wait_for(player.accept_challenges(args.opponent, n), deadline)
    except asyncio.TimeoutError:
        timed_out = True
        print(f"[{args.label}] DEADLINE HIT after {time.time()-t0:.0f}s — "
              f"unfinished games go to the TIMEOUT bucket", flush=True)

    elapsed = time.time() - t0
    rows = collect_results(player, args, games, teambuilder, team_meta)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    finished = sum(1 for r in rows if r["finished"])
    won = sum(1 for r in rows if r["finished"] and r["won"])
    mismatch = sum(1 for r in rows if r["team_match"] is False)
    print(f"[{args.label}] played={len(rows)} finished={finished} won={won} "
          f"planned={n} timeouts={n - finished} team_mismatch={mismatch} "
          f"elapsed={elapsed:.0f}s deadline_hit={timed_out}", flush=True)
    print(f"[{args.label}] -> {args.out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
