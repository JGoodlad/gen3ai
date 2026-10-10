"""Play battles with a trained Gen3 model (or seeded random policies) over a WEBSOCKET.

This is the entry point that talks to a Showdown server as a CLIENT rather than through the in-process bridge.
Its client is the RUST stack (poke-env retirement P4, the only client since P6, 2026-10-08): ``main.live.client``
over the reader session ``main.live.reader`` — the observation is the TRAINING chain's own row (``live_reader``, the
``sim_bridge`` core-observation chain). The legacy poke-env ``RLPlayer`` client (``--client poke-env``) and its two
poke-env-only flags (``--avatar``, ``--concurrency``) are DELETED; a typed one is refused with its reason
(``designs/deleted_flags.md``).

Modes:

    selfplay   two clients on this server play each other (the model if --model, else seeded random policies)
    challenge  our model challenges a named user
    accept     our model waits for challenges from a named user (or, on a LOCAL server, anyone)
    ladder     our model queues on the ladder (`/search`): ONE battle at a time, behind the guards below

Examples::

    # local smoke on a throwaway port (never 8000/8001 — see the port guard below)
    python src/main/play.py --mode selfplay --port 9017
    python src/main/play.py --mode selfplay --port 9017 --model models/<run>/final_model.zip --n-battles 4

🚨 **RESPECT FOR PLAYERS (owner 2026-10-07; ladder ruling 2026-10-09: "allow mode ladder, just never for our
agents without my explicit approval, and we would only ever do it at concurrency 1").** ``--mode ladder`` is
allowed for USERS under four hard guards (``main.ladder_guard``): concurrency is FIXED at 1 (a typed ``--concurrency``
is refused); the T28 halt marker refuses to start; the drift gate (``python src/main/ladder_drift_scan.py``) must have
passed in the last ``DRIFT_MAX_AGE_DAYS`` days; and in an AI-agent session (Claude Code's ``CLAUDECODE`` /
``CLAUDE_CODE_ENTRYPOINT`` environment) it REFUSES unless the OWNER has written the approval token — agents never
create it. A ladder session prints a short respect-for-players notice at startup. Separately, ``--server official``
for anything but a ladder session is refused unless ``--public-acceptance`` is given with BOTH accounts in
``$PS_OWN_ACCOUNTS`` — the gated self-vs-self acceptance series, which needs the orchestrator's explicit go and goes
through the owner's SOCKS5 proxy. The Rust client sends only a declared command set (no chat, no PM, and `/search`
only in ladder mode).

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
from main.ladder_guard import DRIFT_MAX_AGE_DAYS
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


def resolve_uri(server: str, port: int) -> str:
    """The websocket URI the RUST client dials, refusing the two reserved local ports."""
    if server == "official":
        return OFFICIAL_URI
    check_port(port)
    return f"ws://127.0.0.1:{port}/showdown/websocket"


def check_ladder(args):
    """The owner's 2026-10-09 ladder ruling, in CODE (``main.ladder_guard`` owns the guards): returns the permit, or
    exits with the reason. The T28 halt (``HaltActive``) propagates to ``run``, which maps it to its exit code."""
    from main.ladder_guard import LadderRefused, check_ladder_policy
    if not args.model:
        raise SystemExit("refusing --mode ladder without --model <checkpoint.zip>: the ladder plays the model it "
                         "was handed, never a random policy against real people")
    if not args.username:
        raise SystemExit("--mode ladder needs --username (or $PS_USERNAME): the account that queues")
    if args.opponent or args.public_acceptance:
        raise SystemExit("--mode ladder queues for whoever the ladder matches: it never names an --opponent and is "
                         "not a --public-acceptance series")
    try:
        return check_ladder_policy(battle_format=args.format)
    except LadderRefused as exc:
        raise SystemExit(str(exc)) from exc


def check_policy(args):
    """The owner's policy, in CODE: the ladder only behind ``main.ladder_guard``'s guards (2026-10-09), and the public
    server otherwise only for the gated self-vs-self acceptance between our own accounts (2026-10-07). Returns the
    ladder permit for ``--mode ladder``, else ``None``."""
    if args.mode == "ladder":
        return check_ladder(args)
    if args.server != "official":
        return None
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
    return None


def to_id(text) -> str:
    from utils.showdown_id import to_id_str
    return to_id_str(text or "")


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

    # a websocket client's row is the live reader's, which carries no oracle reveal
    refuse_if_revealed(path, tool="main.play", reason="A websocket client builds its observation without the reveal.")
    # `historical_load_kwargs` strips the policy / extractor kwargs deleted since the checkpoint was
    # written (an ON one is REFUSED), so a current-lineage checkpoint written before a deletion
    # still plays — a ladder session plays the model it was handed or refuses it for a stated reason.
    # `load_checkpoint_strict` refuses sb3's non-strict retry too (`gen3_strict_checkpoint_load_v1`): a
    # checkpoint missing an extractor submodule's keys is a `StrictLoadError`, never a model with that
    # submodule at fresh init playing rated games.
    return load_checkpoint_strict(path, device=device, **historical_load_kwargs(path))


async def main(args) -> int:
    # T28 (owner 2026-10-07): a parse-panic HALT marker refuses EVERY live game, self-play included.
    from main.live.halt import refuse_if_halted
    refuse_if_halted("main.play")
    permit = check_policy(args)
    if permit is not None:
        from main.ladder_guard import respect_notice
        print(respect_notice(permit, forfeit_turn_limit=args.forfeit_turn_limit), flush=True)
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
                            forfeit_turn_limit=args.forfeit_turn_limit, ladder=args.mode == "ladder")

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
        if args.server == "official" and args.mode != "ladder" and not args.opponent:
            raise SystemExit("--server official needs --opponent (one of our own accounts)")
        c = LiveClient(cfg(args.username or "p4live"), policy=rust_policy(args), team_fn=rust_team_fn(args))
        try:
            await c.connect()
            if args.mode == "ladder":
                res = await c.ladder(args.n_battles)  # one battle at a time; re-checks main.ladder_guard per game
            elif args.mode == "accept":
                res = await c.accept(args.opponent, args.n_battles)
            else:
                res = await c.challenge(args.opponent, args.n_battles)
        finally:
            await c.close()
    won = sum(1 for r in res if r.won)
    print(f"[play] finished={len(res)} won={won} win_rate={(won / len(res) if res else 0.0):.3f}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Play on Pokemon Showdown")
    p.add_argument("--mode", choices=("selfplay", "ladder", "accept", "challenge"),
                   default="selfplay",
                   help="what to do once connected. 'ladder' queues on the ladder (/search) for real players: "
                        "allowed for USERS at concurrency 1 (fixed; --concurrency is refused), only with --model and "
                        "--username, while no T28 halt marker exists and python src/main/ladder_drift_scan.py has "
                        f"passed in the last {DRIFT_MAX_AGE_DAYS} days; in an AI-agent session (Claude Code's "
                        "CLAUDECODE environment) it is REFUSED unless the OWNER has written the approval token "
                        "(~/.local/state/gen3ai/ladder_owner_approval.json) - agents must never create it "
                        "(owner ruling 2026-10-09)")
    p.add_argument("--server", choices=("local", "official"), default="local",
                   help="'official' = wss://sim3.psim.us — for --mode ladder, or REFUSED unless --public-acceptance "
                        "(owner 2026-10-07)")
    p.add_argument("--port", type=int, default=9017,
                   help="localhost port for --server local (8000/8001 are REFUSED)")
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
    p.add_argument("--opponent", default=None,
                   help="username for --mode challenge/accept (accept: omit for anyone)")
    p.add_argument("--n-battles", type=int, default=1)
    p.add_argument("--team", default=None, help="path to a Showdown-export team file")
    p.add_argument("--team-pool", action="store_true",
                   help="sample from the whole data/teams pool instead of one team")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy (the measurement setting); >0 samples the policy")
    # Default None = the Rust client's own deadline (`main.live.client.ClientConfig.connect_timeout_s`, 30 s).
    p.add_argument("--connect-timeout", type=float, default=None, metavar="SECONDS",
                   help="how long to wait for THIS client's login before raising "
                        "(default: main.live.client.ClientConfig.connect_timeout_s = 30s). "
                        "A login the server never completes — e.g. a username "
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


#: Flags DELETED with the legacy poke-env client (P6 of the poke-env retirement, 2026-10-08): a typed one is refused
#: with its reason rather than silently ignored (``designs/deleted_flags.md``).
DELETED_FLAGS = {
    "--client": "main.play runs the Rust client (main.live) only; the legacy poke-env RLPlayer client was deleted "
                "in P6 of the poke-env retirement (2026-10-08)",
    "--avatar": "a poke-env client setting; the Rust client sends a declared command set only (P6, 2026-10-08)",
    "--concurrency": "the Rust client plays ONE battle at a time per account (P6 deleted the poke-env client that "
                     "took it, 2026-10-08), and --mode ladder is fixed at concurrency 1 with no option to raise it "
                     "(owner 2026-10-09: \"we would only ever do it at concurrency 1\")",
}


def refuse_deleted_flags(argv) -> None:
    for a in argv:
        flag = a.split("=", 1)[0]
        if flag in DELETED_FLAGS:
            raise SystemExit(f"{flag} was DELETED: {DELETED_FLAGS[flag]}")


def run(argv=None) -> int:
    from main.exit_codes import TrainExitCode
    from main.ladder_guard import LadderRefused
    from main.live.halt import HaltActive
    from main.live.halt import LiveParseHalt, exit_on_halt
    argv = list(sys.argv[1:] if argv is None else argv)
    refuse_deleted_flags(argv)
    try:
        return asyncio.run(main(build_parser().parse_args(argv)))
    except HaltActive as exc:
        print(f"🛑 {exc}", file=sys.stderr)
        return int(TrainExitCode.FATAL_LIVE_PARSE)
    except LadderRefused as exc:  # a guard that lapsed BETWEEN ladder games (token expired, drift record aged out)
        print(f"🛑 {exc}", file=sys.stderr)
        return 1
    except LiveParseHalt as exc:
        exit_on_halt(exc, entry_point="main.play")


if __name__ == "__main__":
    sys.exit(run())
