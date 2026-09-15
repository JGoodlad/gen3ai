"""Throughput of the websocket front end vs a real Node Showdown server, at 1/4/8 concurrency.

THE QUESTION. The front end's reason to exist is that an external opponent needs a socket, not
that it is fast — the foul-play de-risk was explicit that at a realistic search budget the
OPPONENT dominates the server by two orders of magnitude, so a shim "buys determinism, not
throughput". This benchmark exists to state that honestly with a number instead of asserting it,
and to catch the one way it could be WRONG: a front end that serialized what the server overlaps
would cap a series' concurrency and nobody would notice from a win rate.

Both arms play the identical workload — two poke-env `RandomPlayer`s, the same pool teams, the
same battle count — over a real websocket. The only difference is what is listening: this module's
front end (one `sim_bridge`/`local_sim_bridge` child per battle) or `npm run showdown -- <port>`.

🚨 **HONEST SCOPE: THIS IS A SINGLE-PROCESS HARNESS.** Both clients AND (in the front-end arm) the
server run in ONE Python process on one event loop, so above ~1 concurrent battle the ceiling is
this process's own CPU share, not the transport's. That is why the front-end arm barely moves with
concurrency while the Showdown-server arm — a separate process doing the sim work — climbs. Real
use puts the opponent in its own process (that is the whole point of the front end), so read these
rows as a FLOOR on the front end and a CEILING on the server, not as a head-to-head verdict. The
row that is a clean comparison is **concurrency 1**.

🚨 **A BENCHMARK IS A MEASUREMENT, SO ITS BOUNDS ARE NEVER STRETCHED.** The project scales test
timeouts by measured contention; a benchmark must not, because the contention IS part of what is
being measured. This script prints the contention factor and WARNS above a threshold, and never
adjusts a number for it. Run it on as quiet a box as you can get, and report the factor with the
table.

    export PYTHONPATH=$PYTHONPATH:src
    python3 src/utils/bridge/ws_frontend_throughput_benchmark.py --battles 8
    python3 src/utils/bridge/ws_frontend_throughput_benchmark.py --no-node-server   # front end only

The Node arm starts a Showdown server on ``--server-port`` (default 9690, and 8000/8001 are
refused in code) and stops it by the PID it started — never a bare ``npm run stop``, which kills
the shared dev server on :8000.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from poke_env import AccountConfiguration  # noqa: E402
from poke_env.player import RandomPlayer  # noqa: E402
from poke_env.ps_client.server_configuration import ServerConfiguration  # noqa: E402

from utils.bridge.ws_frontend import RESERVED_PORTS, ShowdownFrontEnd  # noqa: E402
from utils.contention import (cpu_contention_factor, describe_contention,  # noqa: E402
                              warn_if_contended)
from utils.paths import repo_root  # noqa: E402
from utils.team_loader import TeamLoader  # noqa: E402
from utils.teambuilder import Gen3Teambuilder  # noqa: E402

_AUTH = "https://play.pokemonshowdown.com/action.php?"
#: Above this measured contention the numbers are about the box, not the transports.
CONTENTION_WARN = 1.5


def _players(config, teams, concurrency: int, tag: str):
    return (
        RandomPlayer(battle_format="gen3ou", team=Gen3Teambuilder(teams, rng_seed=31),
                     account_configuration=AccountConfiguration(f"Bm{tag}A", None),
                     server_configuration=config, max_concurrent_battles=concurrency),
        RandomPlayer(battle_format="gen3ou", team=Gen3Teambuilder(teams, rng_seed=32),
                     account_configuration=AccountConfiguration(f"Bm{tag}B", None),
                     server_configuration=config, max_concurrent_battles=concurrency),
    )


async def _series(config, teams, battles: int, concurrency: int, tag: str) -> float:
    p1, p2 = _players(config, teams, concurrency, tag)
    start = time.perf_counter()
    try:
        await asyncio.wait_for(p1.battle_against(p2, n_battles=battles), timeout=1800)
    finally:
        elapsed = time.perf_counter() - start
        await p1.ps_client.stop_listening()
        await p2.ps_client.stop_listening()
    assert p1.n_finished_battles == battles, (
        f"{tag}: only {p1.n_finished_battles}/{battles} battles finished — the row is not a "
        "measurement")
    return elapsed


async def _front_end_row(impl: str, teams, battles: int, concurrency: int) -> float:
    from websockets.asyncio.server import serve

    front = ShowdownFrontEnd(impl=impl, validate_teams=False)
    server = await serve(front.handler, "127.0.0.1", 0, max_size=None, ping_interval=None)
    port = server.sockets[0].getsockname()[1]
    config = ServerConfiguration(f"ws://127.0.0.1:{port}/showdown/websocket", _AUTH)
    try:
        return await _series(config, teams, battles, concurrency, f"F{concurrency}")
    finally:
        await front.close()
        server.close()
        await server.wait_closed()


def _start_node_server(port: int):
    """Start `npm run showdown -- <port>` and return the Popen. Stopped by THIS pid only."""
    if port in RESERVED_PORTS:
        raise SystemExit(f"refusing --server-port {port}: that is {RESERVED_PORTS[port]}.")
    proc = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security", str(port)],
        cwd=str(repo_root() / "deps" / "pokemon-showdown"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    return proc


def _stop_node_server(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:  # pragma: no cover
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


async def run(args) -> dict:
    teams = [t.strip() for t in TeamLoader().get_all_teams()][:16]
    print(f"[bench] {describe_contention()}", flush=True)
    # The project's own benchmark posture: WARN, never scale (`warn_if_contended`'s docstring
    # carries the incident where a scaled throughput comparison came out REVERSED).
    warn_if_contended("ws_frontend throughput", threshold=CONTENTION_WARN)
    factor = cpu_contention_factor()

    rows = []
    node_proc = None
    try:
        if args.node_server:
            node_proc = _start_node_server(args.server_port)
            await asyncio.sleep(args.server_warmup)
            node_config = ServerConfiguration(
                f"ws://127.0.0.1:{args.server_port}/showdown/websocket", _AUTH)
        for concurrency in args.concurrency:
            row = {"concurrency": concurrency, "battles": args.battles}
            elapsed = await _front_end_row(args.impl, teams, args.battles, concurrency)
            row[f"ws_frontend_{args.impl}_s"] = elapsed
            row[f"ws_frontend_{args.impl}_bps"] = args.battles / elapsed
            if args.node_server:
                elapsed = await _series(node_config, teams, args.battles, concurrency,
                                        f"N{concurrency}")
                row["node_server_s"] = elapsed
                row["node_server_bps"] = args.battles / elapsed
            rows.append(row)
            print(f"[bench] concurrency {concurrency}: " + " | ".join(
                f"{k}={v:.3f}" for k, v in row.items() if isinstance(v, float)), flush=True)
    finally:
        _stop_node_server(node_proc)

    print("\nbattles/s (higher is better) — " + f"{args.battles} battles per cell")
    header = f"{'concurrency':>11} {'ws_frontend':>12} {'node server':>12} {'ratio':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        front = row[f"ws_frontend_{args.impl}_bps"]
        node = row.get("node_server_bps")
        print(f"{row['concurrency']:>11} {front:>12.3f} "
              f"{(f'{node:.3f}' if node else '—'):>12} "
              f"{(f'{front / node:.2f}x' if node else '—'):>7}")
    result = {"rows": rows, "impl": args.impl, "contention_factor": factor,
              "battles_per_cell": args.battles}
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--battles", type=int, default=8, help="battles per cell")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--impl", default="rust", choices=("rust", "node"))
    p.add_argument("--server-port", type=int, default=9690,
                   help="port for the comparison Showdown server (8000/8001 are REFUSED)")
    p.add_argument("--server-warmup", type=float, default=12.0)
    p.add_argument("--no-node-server", dest="node_server", action="store_false",
                   help="skip the Showdown-server arm (front end only)")
    p.add_argument("--out", default=None, help="write the table as JSON here")
    return p


if __name__ == "__main__":
    asyncio.run(run(build_parser().parse_args()))
