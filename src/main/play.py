"""Play battles with a trained Gen3 model (or seeded random policies) over a WEBSOCKET.

This is the entry point that talks to a Showdown server as a CLIENT rather than through the in-process bridge.
Since poke-env retirement P4 (2026-10-07) its DEFAULT client is the RUST stack (``--client rust``):
``main.live.client`` over the reader session ``main.live.reader`` — the observation is the TRAINING chain's own
row (``live_reader``, the ``sim_bridge`` core-observation chain), never poke-env's. ``--client poke-env`` keeps the
legacy ``RLPlayer`` path for ``main.anchors``' our side (until P3) and for P4's shadow gate; P6 deletes it.

Modes:

    selfplay   two clients on this server play each other (the model if --model, else seeded random policies)
    challenge  our model challenges a named user
    accept     our model waits for challenges from a named user (or, on a LOCAL server, anyone)
    ladder     REFUSED by policy: the ladder campaign is DEFERRED and we never play humans (owner 2026-10-07)

Examples::

    # local smoke on a throwaway port (never 8000/8001 — see the port guard below)
    python src/main/play.py --mode selfplay --port 9017
    python src/main/play.py --mode selfplay --port 9017 --model models/<run>/final_model.zip --n-battles 4

🚨 **ZERO contact with humans (owner 2026-10-07).** ``--mode ladder`` is refused, and ``--server official``
is refused unless ``--public-acceptance`` is given with BOTH accounts in ``$PS_OWN_ACCOUNTS`` — the gated
self-vs-self acceptance series, which needs the orchestrator's explicit go and goes through the owner's SOCKS5
proxy. The Rust client sends only a declared command set (no chat, no PM, no ``/search``).

🚨 **T28: a parse panic HALTS all live play.** An unreadable line, an encoder raise or a choice the client could
not send exits ``FATAL_LIVE_PARSE`` (7) with a durable marker (``python -m main.live.halt status``); every
mode refuses to start while it exists.

Every mode that logs in is bounded by a CONNECT-OR-RAISE deadline (`--connect-timeout`, default 30 s).
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

from agents.training.stall import StallConfig
from utils.teambuilder import Gen3Teambuilder

#: The public server's websocket (the official client's). Reached ONLY by the gated public acceptance.
OFFICIAL_URI = "wss://sim3.psim.us/showdown/websocket"
#: The env var listing OUR OWN accounts (comma-separated) — the only names a public-server game may involve.
OWN_ACCOUNTS_ENV = "PS_OWN_ACCOUNTS"

# Ports this process must NEVER touch. 8001 carries the live training run (dropping it
# crashes every poke-env websocket at once) and 8000 is the shared dev server. A ladder
# client has no business on either, so the refusal is in CODE rather than in a docs
# warning — see the root CLAUDE.md § Showdown Server.
RESERVED_PORTS = {8000: "the shared DEV server", 8001: "the live TRAINING server"}

# THE FORFEIT DEADLINE, read from the TRAINER rather than restated. `StallConfig.threshold`
# defaults to `agents.observation.constants.MAX_TURNS` (250) and is the turn at which a training
# episode's player returns a `ForfeitBattleOrder` — `gen3_deadline_clock_v1`, the same number the
# observation's turn clock normalises by. A websocket game that ran to a DIFFERENT limit would not
# be measuring the agent we train, so this entry point inherits the constant instead of owning a
# second one. Pinned by `src/main/play_forfeit_limit_test.py`.
DEFAULT_FORFEIT_TURN_LIMIT = StallConfig().threshold

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


def check_port(port: int) -> None:
    reason = RESERVED_PORTS.get(port)
    if reason is not None:
        raise SystemExit(
            f"refusing --port {port}: that is {reason}. Start your own throwaway "
            f"server on a 9XXX port (`npm run showdown -- 9017`) and pass it here."
        )


def resolve_server(server: str, port: int):
    """poke-env server config for `server` (the LEGACY client), refusing the two reserved local ports."""
    from poke_env.ps_client.server_configuration import (ShowdownServerConfiguration,
                                                         localhost_server_configuration)
    if server == "official":
        return ShowdownServerConfiguration
    check_port(port)
    return localhost_server_configuration(port)


def resolve_uri(server: str, port: int) -> str:
    """The websocket URI the RUST client dials, refusing the two reserved local ports."""
    if server == "official":
        return OFFICIAL_URI
    check_port(port)
    return f"ws://127.0.0.1:{port}/showdown/websocket"


def check_policy(args) -> None:
    """The owner's 2026-10-07 policy, in CODE: no ladder; the public server only for the gated self-vs-self
    acceptance between our own accounts."""
    if args.mode == "ladder":
        raise SystemExit(
            "refusing --mode ladder: the ladder campaign is DEFERRED and we never play humans (owner "
            "2026-10-07, designs/endstate/design_ladder_campaign.md Decision record). Local --mode "
            "challenge/accept/selfplay only; the public self-vs-self acceptance is --public-acceptance.")
    if args.server != "official":
        return
    if not args.public_acceptance:
        raise SystemExit(
            "refusing --server official: P4 validates LOCALLY only. The self-vs-self public acceptance "
            "(our own accounts, custom challenges, the owner's SOCKS5 proxy) needs the orchestrator's explicit "
            "go and --public-acceptance.")
    own = {to_id(n) for n in os.environ.get(OWN_ACCOUNTS_ENV, "").split(",") if n.strip()}
    if args.mode not in ("challenge", "accept", "selfplay"):
        raise SystemExit(f"--public-acceptance plays custom challenges only, not --mode {args.mode}")
    names = [args.username] + ([args.opponent] if args.mode != "selfplay" else [])
    if args.mode == "selfplay":
        names = [f"{args.username}1", f"{args.username}2"]
    missing = [n for n in names if not n or to_id(n) not in own]
    if missing:
        raise SystemExit(f"refusing --public-acceptance: {missing} not in ${OWN_ACCOUNTS_ENV} — a public game "
                         "involves OUR OWN accounts only (owner 2026-10-07)")
    if not args.proxy:
        raise SystemExit("refusing --public-acceptance without --proxy: the owner's SOCKS5 proxy is required")
    if args.client != "rust":
        raise SystemExit("--public-acceptance runs the Rust client only (it carries the T28 halt)")


def to_id(text) -> str:
    from utils.showdown_id import to_id_str
    return to_id_str(text or "")


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
                  suffix: str = ""):
    if username is None:
        if server == "official":
            raise SystemExit(
                "--server official needs --username (and a password via $PS_PASSWORD): "
                "a guest name is server-assigned and claimable, and the rating this run "
                "exists to measure needs a stable account to accrue on."
            )
        return None
    from poke_env.ps_client import AccountConfiguration
    return AccountConfiguration(f"{username}{suffix}", password)


def load_policy(path: str, device: str):
    """Load a checkpoint for INFERENCE — the one seam a measurement harness can replace.

    A plain load rebuilds the extractor from the zip's own ``policy_kwargs``, so it
    only works for a checkpoint whose flag set the CURRENT ``ExtractorBuild`` still accepts: a
    frozen snapshot from an older run in the same observation family dies here with
    ``ExtractorBuild.__init__() got an unexpected keyword argument ...``. That is correct for a
    LADDER session (it should play the model it was handed, or refuse), and wrong for an
    external-anchor read across our own history — which is why this is a function and not an
    inline call. ``main.anchors`` swaps in ``agents.model.snapshot.load_foreign_opponent``, which
    verifies the ``arch_signature`` instead of trusting the kwargs.
    """
    from agents.model.oracle_reveal import refuse_if_revealed
    from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict

    # a websocket client builds its observation with the PYTHON encoder, which has no reveal
    refuse_if_revealed(path, tool="main.play", reason="A websocket client builds its observation without the reveal.")
    # `historical_load_kwargs` strips the policy / extractor kwargs deleted since the checkpoint was
    # written (an ON one is REFUSED), so a current-lineage checkpoint written before a deletion
    # still plays — a ladder session plays the model it was handed or refuses it for a stated reason.
    # `load_checkpoint_strict` refuses sb3's non-strict retry too (`gen3_strict_checkpoint_load_v1`): a
    # checkpoint missing an extractor submodule's keys is a `StrictLoadError`, never a model with that
    # submodule at fresh init playing rated games.
    return load_checkpoint_strict(path, device=device, **historical_load_kwargs(path))


def build_model_player(args, teambuilder, server_config, account):
    """Load the checkpoint and wrap it in the same RLPlayer eval uses."""
    from agents.inference.player import RLPlayer
    from agents.observation.state_encoder import load_mappings

    model = load_policy(args.model, args.device)
    return RLPlayer(
        model=model,
        team=teambuilder,
        # THE FORFEIT LIMIT. `RLPlayer.choose_move` already forfeits at
        # `StallConfig.threshold`; naming it here makes the number VISIBLE and overridable
        # (`--forfeit-turn-limit`) instead of an invisible class default, which is what a
        # head-to-head against a third-party bot needs — the two clients must agree on when a
        # stalled game ends, and the answer has to be OUR trainer's number.
        stall_config=StallConfig(threshold=args.forfeit_turn_limit),
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
    # T28 (owner 2026-10-07): a parse-panic HALT marker refuses EVERY live game, self-play included.
    from main.live.halt import refuse_if_halted
    refuse_if_halted("main.play")
    check_policy(args)
    if args.client == "poke-env":
        return await main_poke_env(args)
    return await main_rust(args)


def rust_team_fn(args):
    """The packed team(s) our side plays (one per battle)."""
    import random
    if args.team:
        with open(args.team) as f:
            tb = Gen3Teambuilder(f.read())
        return tb.yield_team
    if args.team_pool:
        from utils.team_sources import packed_teams
        teams = packed_teams("pool")
        rng = random.Random(args.seed)
        return lambda: rng.choice(teams)
    return Gen3Teambuilder(STAR_TSS_TEAM).yield_team


def rust_policy(args, seed_offset: int = 0):
    from main.live.policy import ModelPolicy, RandomPolicy
    if args.model:
        return ModelPolicy(args.model, temperature=args.temperature, seed=args.seed + seed_offset,
                           device=args.device)
    return RandomPolicy(args.seed + seed_offset)


async def main_rust(args) -> int:
    """The Rust client (`main.live`): the observation is the training chain's own row."""
    from main.live.client import ClientConfig, LiveClient

    uri = resolve_uri(args.server, args.port)
    auth = "official" if args.server == "official" else "local"

    def cfg(name: str) -> ClientConfig:
        return ClientConfig(uri=uri, username=name, battle_format=args.format, password=args.password,
                            auth=auth, proxy=args.proxy,
                            connect_timeout_s=args.connect_timeout or 30.0,
                            forfeit_turn_limit=args.forfeit_turn_limit)

    print(f"[play] client: rust (main.live) on {uri}; forfeit turn limit {args.forfeit_turn_limit} "
          f"(trainer default {DEFAULT_FORFEIT_TURN_LIMIT})", flush=True)
    if args.mode == "selfplay":
        base = args.username or "p4self"
        a = LiveClient(cfg(f"{base}1"), policy=rust_policy(args, 0), team_fn=rust_team_fn(args))
        b = LiveClient(cfg(f"{base}2"), policy=rust_policy(args, 1), team_fn=rust_team_fn(args))
        acc: Optional[asyncio.Future] = None
        try:
            await b.connect()
            await a.connect()
            acc = asyncio.ensure_future(b.accept(a.name, args.n_battles))
            res = await a.challenge(b.name or f"{base}2", args.n_battles)
            await acc
        finally:
            # a halt on EITHER side stops both (T28: no other battle continues)
            if acc is not None and not acc.done():
                acc.cancel()
                await asyncio.gather(acc, return_exceptions=True)
            await a.close()
            await b.close()
    else:
        if not args.model:
            raise SystemExit(f"--mode {args.mode} needs --model <checkpoint.zip>")
        if args.mode == "challenge" and not args.opponent:
            raise SystemExit("--mode challenge needs --opponent <username>")
        if args.server == "official" and not args.opponent:
            raise SystemExit("--server official needs --opponent (one of our own accounts)")
        c = LiveClient(cfg(args.username or "p4live"), policy=rust_policy(args), team_fn=rust_team_fn(args))
        try:
            await c.connect()
            if args.mode == "accept":
                res = await c.accept(args.opponent, args.n_battles)
            else:
                res = await c.challenge(args.opponent, args.n_battles)
        finally:
            await c.close()
    won = sum(1 for r in res if r.won)
    print(f"[play] finished={len(res)} won={won} win_rate={(won / len(res) if res else 0.0):.3f}", flush=True)
    return 0


async def main_poke_env(args) -> int:
    """The LEGACY poke-env client (`--client poke-env`): `main.anchors`' our side until P3, P4's shadow."""
    server_config = resolve_server(args.server, args.port)
    teambuilder = build_teambuilder(args.team, args.team_pool)

    if args.mode == "selfplay":
        from poke_env.player import RandomPlayer
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
    print(f"[play] forfeit turn limit: {args.forfeit_turn_limit} "
          f"(trainer default {DEFAULT_FORFEIT_TURN_LIMIT})")
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
                   help="'official' = wss://sim3.psim.us — REFUSED unless --public-acceptance (owner 2026-10-07)")
    p.add_argument("--port", type=int, default=9017,
                   help="localhost port for --server local (8000/8001 are REFUSED)")
    p.add_argument("--client", choices=("rust", "poke-env"), default="rust",
                   help="rust (default): main.live over the training chain's reader; poke-env: the LEGACY "
                        "RLPlayer client (main.anchors' our side until P3, P4's shadow gate)")
    p.add_argument("--public-acceptance", action="store_true",
                   help="with --server official: the GATED self-vs-self acceptance (our own accounts in "
                        "$PS_OWN_ACCOUNTS, --proxy required; needs the orchestrator's explicit go)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed of the Rust client's team draw and policy sampling")
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
    p.add_argument("--forfeit-turn-limit", type=int, default=DEFAULT_FORFEIT_TURN_LIMIT,
                   metavar="TURNS",
                   help="forfeit a battle once it reaches this turn, the way a TRAINING episode "
                        f"does (default {DEFAULT_FORFEIT_TURN_LIMIT}, read from "
                        "agents.training.stall.StallConfig.threshold — do not restate it). Lower "
                        "it only for a deliberately shorter head-to-head; raising it above the "
                        "trainer's number measures an agent we do not train.")
    p.add_argument("--proxy", type=str, default=None, metavar="SOCKS5_URL",
                   help="SOCKS5 proxy URL, e.g. socks5h://127.0.0.1:1080")
    return p


def run(argv=None) -> int:
    from main.exit_codes import TrainExitCode
    from main.live.halt import HaltActive
    from main.live.halt import LiveParseHalt, exit_on_halt
    try:
        return asyncio.run(main(build_parser().parse_args(argv)))
    except HaltActive as exc:
        print(f"🛑 {exc}", file=sys.stderr)
        return int(TrainExitCode.FATAL_LIVE_PARSE)
    except LiveParseHalt as exc:
        exit_on_halt(exc, entry_point="main.play")


if __name__ == "__main__":
    sys.exit(run())
