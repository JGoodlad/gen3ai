"""Play battles with a trained Gen3 model (or two random bots) over a WEBSOCKET.

This is the ONLY entry point that talks to a Showdown server as a client rather than
through the in-process bridge, so it is the exact code path a rated ladder game uses.
Four modes:

    selfplay   two RandomPlayers battle each other (the historical behaviour; no model)
    challenge  our model challenges a named user
    accept     our model waits for challenges from a named user (or anyone)
    ladder     our model queues for rated games with `/search <format>`

Examples::

    # local smoke on a throwaway port (never 8000/8001 — see the port guard below)
    python src/main/play.py --mode selfplay --port 9017

    # our model, laddering on the OFFICIAL server under a registered account
    PS_PASSWORD=... python src/main/play.py --mode ladder --server official \\
        --model models/<run>/final_model.zip --username Gen3AI --n-battles 20 \\
        --proxy socks5h://127.0.0.1:1080

Every mode that logs in is bounded by a CONNECT-OR-RAISE deadline (`--connect-timeout`,
default 30 s): a login the server never completes raises `ShowdownConnectionError` naming the
username and the server instead of sitting silently until the battle deadline — which is what a
username registered on the official ladder does even against a `--no-security` local server,
since `localhost_server_configuration` still authenticates against Smogon's `action.php`.

`--server official` REQUIRES `--username` and a password. Not because the server demands
it (rated play has no registration gate — verified in source, see
designs/research_state/ladder_readiness.md) but because WE do: a guest name is
server-assigned and claimable by anyone, the rating we are trying to measure needs a
stable account to accrue on, and `Config.forceregisterelo` may cut a guest off
mid-campaign.
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    ServerConfiguration,
    ShowdownServerConfiguration,
    localhost_server_configuration,
)
from utils.teambuilder import Gen3Teambuilder

# Ports this process must NEVER touch. 8001 carries the live training run (dropping it
# crashes every poke-env websocket at once) and 8000 is the shared dev server. A ladder
# client has no business on either, so the refusal is in CODE rather than in a docs
# warning — see the root CLAUDE.md § Showdown Server.
RESERVED_PORTS = {8000: "the shared DEV server", 8001: "the live TRAINING server"}

# Team from https://pokepast.es/f6229d2c867e21d6
STAR_TSS_TEAM = """
Skarmory (F) @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 8 Def / 248 SpD
Careful Nature
IVs: 0 Atk
- Spikes
- Protect
- Roar
- Toxic

Blissey @ Leftovers
Ability: Natural Cure
Shiny: Yes
EVs: 252 Def / 252 SpA / 4 Spe
Modest Nature
IVs: 0 Atk
- Soft-Boiled
- Ice Beam
- Toxic
- Fire Blast

Tyranitar (F) @ Leftovers
Ability: Sand Stream
EVs: 248 HP / 196 Atk / 12 Def / 52 SpD
Adamant Nature
- Focus Punch
- Rock Slide
- Hidden Power [Bug]
- Earthquake

Swampert (F) @ Leftovers
Ability: Torrent
EVs: 240 HP / 136 Def / 40 SpA / 48 SpD / 44 Spe
Relaxed Nature
- Earthquake
- Ice Beam
- Hydro Pump
- Protect

Gengar (F) @ Leftovers
Ability: Levitate
EVs: 168 HP / 164 SpD / 176 Spe
Timid Nature
- Will-O-Wisp
- Thunderbolt
- Ice Punch
- Explosion

Starmie @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Hydro Pump
- Ice Beam
- Thunderbolt
- Rapid Spin
"""


def resolve_server(server: str, port: int) -> ServerConfiguration:
    """Server config for `server`, refusing the two reserved local ports."""
    if server == "official":
        return ShowdownServerConfiguration
    reason = RESERVED_PORTS.get(port)
    if reason is not None:
        raise SystemExit(
            f"refusing --port {port}: that is {reason}. Start your own throwaway "
            f"server on a 9XXX port (`npm run showdown -- 9017`) and pass it here."
        )
    return localhost_server_configuration(port)


def build_teambuilder(team_file: Optional[str], pool: bool) -> Gen3Teambuilder:
    """The team(s) we play. A ladder account should be pinned to ONE team so its rating
    measures a single matchup distribution; `--team-pool` is the multi-team arm."""
    if team_file:
        with open(team_file) as f:
            return Gen3Teambuilder(f.read())
    if pool:
        from utils.team_loader import TeamLoader
        return Gen3Teambuilder(TeamLoader().get_all_teams())
    return Gen3Teambuilder(STAR_TSS_TEAM)


def build_account(username: Optional[str], password: Optional[str], server: str,
                  suffix: str = "") -> Optional[AccountConfiguration]:
    if username is None:
        if server == "official":
            raise SystemExit(
                "--server official needs --username (and a password via $PS_PASSWORD): "
                "a guest name is server-assigned and claimable, and the rating this run "
                "exists to measure needs a stable account to accrue on."
            )
        return None
    return AccountConfiguration(f"{username}{suffix}", password)


def build_model_player(args, teambuilder, server_config, account):
    """Load the checkpoint and wrap it in the same RLPlayer eval uses."""
    from sb3_contrib import MaskablePPO
    from agents.inference.player import RLPlayer
    from agents.observation.state_encoder import load_mappings

    model = MaskablePPO.load(args.model, env=None, device=args.device)
    if not args.debug_obs:
        # A checkpoint trained with --log-level periodic carries a live
        # ObservationDebugger that print()s a full 12-mon board dump on forward
        # passes. Over a ladder session that is megabytes of stdout and a real
        # per-decision cost, for nobody's benefit. Same silencing the prober does.
        for m in model.policy.modules():
            if hasattr(m, "_debugger"):
                m._debugger = None
    return RLPlayer(
        model=model,
        team=teambuilder,
        battle_format=args.format,
        server_configuration=server_config,
        mappings=load_mappings(),
        account_configuration=account,
        max_concurrent_battles=args.concurrency,
        stochastic=args.temperature > 0.0,
        temperature=max(args.temperature, 1e-6),
        avatar=args.avatar,
        proxy_url=args.proxy,
        # None ⇒ the class default (DEFAULT_CONNECT_TIMEOUT_S). The guard this feeds wraps
        # ladder / accept / challenge as well as battle_against — see Gen3Player.
        connect_timeout_s=args.connect_timeout,
    )


async def main(args) -> int:
    server_config = resolve_server(args.server, args.port)
    teambuilder = build_teambuilder(args.team, args.team_pool)

    if args.mode == "selfplay":
        account_1 = build_account(args.username, args.password, args.server, "1")
        account_2 = build_account(args.username, args.password, args.server, "2")
        p1 = RandomPlayer(battle_format=args.format, team=teambuilder,
                          max_concurrent_battles=args.concurrency,
                          server_configuration=server_config,
                          account_configuration=account_1, proxy_url=args.proxy)
        p2 = RandomPlayer(battle_format=args.format, team=teambuilder,
                          max_concurrent_battles=args.concurrency,
                          server_configuration=server_config,
                          account_configuration=account_2, proxy_url=args.proxy)
        print(f"[play] selfplay: {args.n_battles} {args.format} battle(s) "
              f"on {server_config.websocket_url}")
        await p1.battle_against(p2, n_battles=args.n_battles)
        print(f"[play] finished={p1.n_finished_battles} p1_wins={p1.n_won_battles}")
        return 0

    if not args.model:
        raise SystemExit(f"--mode {args.mode} needs --model <checkpoint.zip>")

    account = build_account(args.username, args.password, args.server)
    player = build_model_player(args, teambuilder, server_config, account)
    print(f"[play] {args.mode}: {args.n_battles} {args.format} battle(s) as "
          f"{player.username} on {server_config.websocket_url}")
    print(f"[play] connect-or-raise deadline: "
          f"{'none (waits forever)' if not player.connect_timeout_s else f'{player.connect_timeout_s:g}s'}")

    if args.mode == "ladder":
        await player.ladder(args.n_battles)
    elif args.mode == "accept":
        await player.accept_challenges(args.opponent, args.n_battles)
    else:  # challenge
        if not args.opponent:
            raise SystemExit("--mode challenge needs --opponent <username>")
        await player.send_challenges(args.opponent, args.n_battles)

    won, total = player.n_won_battles, player.n_finished_battles
    print(f"[play] finished={total} won={won} "
          f"win_rate={(won / total if total else 0.0):.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Play on Pokemon Showdown")
    p.add_argument("--mode", choices=("selfplay", "ladder", "accept", "challenge"),
                   default="selfplay", help="what to do once connected")
    p.add_argument("--server", choices=("local", "official"), default="local",
                   help="'official' = wss://sim3.psim.us (the rated public ladder)")
    p.add_argument("--port", type=int, default=9017,
                   help="localhost port for --server local (8000/8001 are REFUSED)")
    p.add_argument("--format", default="gen3ou")
    p.add_argument("--model", default=None, help="checkpoint .zip for the model player")
    p.add_argument("--device", default="cpu",
                   help="torch device for inference; keep 'cpu' so a ladder run never "
                        "contends with a training GPU")
    p.add_argument("--username", default=os.environ.get("PS_USERNAME"),
                   help="Showdown account name (default $PS_USERNAME)")
    p.add_argument("--password", default=os.environ.get("PS_PASSWORD"),
                   help="account password (default $PS_PASSWORD; never pass on the CLI "
                        "on a shared box — it lands in the process list)")
    p.add_argument("--avatar", default=None, help="Showdown avatar id")
    p.add_argument("--opponent", default=None,
                   help="username for --mode challenge/accept (accept: omit for anyone)")
    p.add_argument("--n-battles", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1,
                   help="max concurrent battles; keep 1 on the public ladder")
    p.add_argument("--team", default=None, help="path to a Showdown-export team file")
    p.add_argument("--team-pool", action="store_true",
                   help="sample from the whole data/teams pool instead of one team")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy (the measurement setting); >0 samples the policy")
    p.add_argument("--debug-obs", action="store_true",
                   help="keep the checkpoint's ObservationDebugger board dumps (off by "
                        "default — it prints a full board on every forward)")
    # Default None = "use the library's own deadline"
    # (agents.inference.player.DEFAULT_CONNECT_TIMEOUT_S, 30 s), resolved when the player is
    # built. Not read here, so `--help` and `--mode selfplay` stay free of the torch import that
    # module pulls in — the same reason `build_model_player` imports MaskablePPO lazily. The
    # effective value is printed at startup, so it is never a hidden number.
    p.add_argument("--connect-timeout", type=float, default=None, metavar="SECONDS",
                   help="how long to wait for THIS client's login before raising "
                        "(default: agents.inference.player.DEFAULT_CONNECT_TIMEOUT_S = 30s; "
                        "0 waits forever). A login the server never completes — e.g. a username "
                        "registered upstream, which even a --no-security local server refuses, "
                        "since localhost auth still goes to Smogon's action.php — otherwise "
                        "waits out the whole battle deadline in silence.")
    p.add_argument("--proxy", type=str, default=None, metavar="SOCKS5_URL",
                   help="SOCKS5 proxy URL, e.g. socks5h://127.0.0.1:1080")
    return p


if __name__ == "__main__":
    sys.exit(asyncio.run(main(build_parser().parse_args())))
