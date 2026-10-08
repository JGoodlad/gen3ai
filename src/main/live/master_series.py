"""P4 gate (c) — the Rust client against a LOCAL Node Showdown built from CURRENT MASTER.

(c) Real framing, real ``rqid``, the server's own room lines, the master simulator's protocol: our checkpoint on the
    Rust client (``main.live``) challenges scripted bots (the RUST ports of the roster bots, each a Rust-stack client
    in its own process, ``main.live.gate_peer``; poke-env clients until P6) and itself (two Rust clients — the
    self-vs-self acceptance's shape), and optionally a Metamon agent. The verdict is the T28 count: every halt is a
    finding to root-cause.

Gate (d) — the SHADOW (our checkpoint on the legacy poke-env client with a Rust reader beside it) — is BANKED (16,523
decisions, 0 differences, ``designs/research_state/measurements/pokeenv_p4_live_2026-10-07/``) and was retired with
that client in P6 of the poke-env retirement (2026-10-08).

The server is never the pinned ``deps/``: it is a master checkout (``--showdown``, default
``~/.cache/gen3ai/p4/showdown-master``) started with ``--no-security`` on a 9XXX port and stopped BY PID. The halt
marker of THIS harness is gate-local (``<out>/live_halt.json``): a halt here is the gate's finding, root-caused before
P6, and never blocks another agent's live tool (stated in the measurement README).

    python -m main.live.master_series --model <ckpt.zip> --games-per-bot 10 --self-games 20 --out <dir>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from main.live.gate_peer import BOTS

DEFAULT_SHOWDOWN = Path.home() / ".cache" / "gen3ai" / "p4" / "showdown-master"

_VALIDATE_JS = r"""
const path = process.argv[1];
const { Teams, TeamValidator } = require(path + '/dist/sim');
const teams = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const v = new TeamValidator('gen3ou');
const out = teams.map(t => { try { const e = v.validateTeam(Teams.unpack(t)); return e && e.length ? e : null; }
                             catch (x) { return ['EXCEPTION ' + x]; } });
console.log(JSON.stringify(out));
"""


def master_valid_teams(showdown: Path, teams: List[str]) -> Dict[str, Any]:
    """Validate every packed team with MASTER's own validator (the format-spec drift, measured)."""
    proc = subprocess.run(["node", "-e", _VALIDATE_JS, str(showdown)], input=json.dumps(teams),
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"master team validation failed: {proc.stderr[-2000:]}")
    errs = json.loads(proc.stdout)
    ok = [t for t, e in zip(teams, errs) if e is None]
    bad = [{"team": t[:120], "errors": e[:3]} for t, e in zip(teams, errs) if e is not None]
    return {"valid": ok, "rejected": bad}


def start_master(showdown: Path, port: int, log: Path) -> subprocess.Popen:
    from main.play import check_port
    check_port(port)
    fh = open(log, "w")
    proc = subprocess.Popen(["node", "pokemon-showdown", "start", "--no-security", str(port)], cwd=str(showdown),
                            stdout=fh, stderr=subprocess.STDOUT)
    deadline = time.time() + 120
    import socket
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the master server exited {proc.returncode}; see {log}")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                time.sleep(1.0)
                return proc
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"the master server never listened on {port}; see {log}")


def stop(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def start_peer(argv: List[str], log: Path) -> subprocess.Popen:
    from utils.paths import src_root
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root()) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    fh = open(log, "w")
    proc = subprocess.Popen([sys.executable, "-m", "main.live.gate_peer", *argv], stdout=fh, stderr=subprocess.STDOUT,
                            env=env, cwd=os.getcwd())
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the peer exited {proc.returncode}; see {log}")
        if "[peer] READY" in log.read_text(errors="replace"):
            return proc
        time.sleep(0.3)
    stop(proc)
    raise RuntimeError(f"the peer never logged in; see {log}")


def start_metamon(agent: str, port: int, our_name: str, peer_name: str, n: int, out: Path, seed: int) -> subprocess.Popen:
    """A Metamon policy as the ACCEPTOR (its own interpreter, UPSTREAM poke-env, CPU, greedy), through the anchors'
    own peer plan (every hazard workaround included). It plays from its 'competitive' team set."""
    import re

    from main.anchors import config as config_mod
    from main.anchors.peers import PEERS
    cfg = config_mod.load_config()
    plan = PEERS["metamon"].plan(cfg=cfg.opponent("metamon"), agent=agent, regime="greedy", teamset="away",
                                 battle_format="gen3ou", server_uri=f"ws://127.0.0.1:{port}/showdown/websocket",
                                 username=peer_name, opponent_username=our_name, role="acceptor", n_games=n,
                                 team_seed=seed, out_dir=out)
    fh = open(plan.log_path, "w")
    proc = subprocess.Popen(plan.argv, cwd=str(plan.cwd), env=plan.env, stdout=fh, stderr=subprocess.STDOUT)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the Metamon peer exited {proc.returncode}; see {plan.log_path}")
        if re.search(plan.ready_pattern, plan.log_path.read_text(errors="replace")):
            time.sleep(5.0)  # the banner precedes the websocket login by a moment
            return proc
        time.sleep(1.0)
    stop(proc)
    raise RuntimeError(f"the Metamon peer never became ready; see {plan.log_path}")


class Recorder:
    def __init__(self, path: Path) -> None:
        self.fh = open(path, "a")
        self.decisions = 0

    def __call__(self, d: Any) -> None:
        self.decisions += 1

    def battle(self, cell: str, res: Any) -> None:
        self.fh.write(json.dumps({"cell": cell, "battle": res.battle, "side": res.side, "won": res.won,
                                  "turns": res.turns, "decisions": res.decisions, "forfeited": res.forfeited}) + "\n")
        self.fh.flush()


async def our_games(uri: str, name: str, opponent: str, n: int, policy: Any, teams: List[str], seed: int,
                    rec: Recorder, cell: str, on_message=None) -> Dict[str, Any]:
    from main.live.client import ClientConfig, LiveClient, TeamRejected
    rng = random.Random(seed)
    c = LiveClient(ClientConfig(uri=uri, username=name),
                   policy=policy, team_fn=lambda: rng.choice(teams), on_decision=rec, on_message=on_message)
    rejected = 0
    played = 0
    try:
        await c.connect()
        while played < n:
            try:
                for r in await c.challenge(opponent, 1):
                    rec.battle(cell, r)
                    played += 1
            except TeamRejected:
                rejected += 1
                if rejected > 20:
                    raise
    finally:
        await c.close()
    return {"played": played, "team_rejections": rejected}


async def self_games(uri: str, n: int, policy: Any, teams: List[str], seed: int, rec: Recorder) -> Dict[str, Any]:
    from main.live.client import ClientConfig, LiveClient
    rng = random.Random(seed)
    a = LiveClient(ClientConfig(uri=uri, username=f"p4ma{seed % 100}"), policy=policy,
                   team_fn=lambda: rng.choice(teams), on_decision=rec)
    b = LiveClient(ClientConfig(uri=uri, username=f"p4mb{seed % 100}"), policy=policy,
                   team_fn=lambda: rng.choice(teams), on_decision=rec)
    acc = None
    try:
        await b.connect()
        await a.connect()
        acc = asyncio.ensure_future(b.accept(a.name, n))
        res = await a.challenge(b.name, n)
        await acc
        for r in res + b.results:
            rec.battle("self", r)
    finally:
        if acc is not None and not acc.done():
            acc.cancel()
            await asyncio.gather(acc, return_exceptions=True)
        await a.close()
        await b.close()
    return {"played": len(res)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.live.master_series", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--showdown", default=str(DEFAULT_SHOWDOWN))
    ap.add_argument("--port", type=int, default=9552)
    ap.add_argument("--bots", default=",".join(BOTS))
    ap.add_argument("--games-per-bot", type=int, default=5)
    ap.add_argument("--self-games", type=int, default=10)
    ap.add_argument("--metamon", default="", help="Metamon agent(s), comma-separated (e.g. SmallRL)")
    ap.add_argument("--metamon-games", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["GEN3AI_LIVE_HALT_FILE"] = str(out / "live_halt.json")  # gate-local (see the module doc)
    from main.live.halt import LiveParseHalt, read_halt, record_halt
    from main.live.policy import ModelPolicy
    from utils.team_sources import packed_teams

    showdown = Path(a.showdown)
    sha = (showdown / "MASTER_SHA").read_text().strip() if (showdown / "MASTER_SHA").exists() else "unknown"
    pool = packed_teams("pool")
    from main.play import STAR_TSS_TEAM
    from utils.teambuilder import Gen3Teambuilder
    peer_team = Gen3Teambuilder(STAR_TSS_TEAM).yield_team()
    if master_valid_teams(showdown, [peer_team])["rejected"]:
        raise SystemExit("the peers' fixed team (main.play.STAR_TSS_TEAM) is rejected by master's validator")
    val = master_valid_teams(showdown, pool)
    teams = val["valid"]
    (out / "master_team_rejections.json").write_text(json.dumps(val["rejected"], indent=1))
    print(f"[master] showdown master {sha}: {len(teams)}/{len(pool)} pool teams valid under master's validator",
          flush=True)
    policy = ModelPolicy(a.model, temperature=0.0)
    rec = Recorder(out / "battles.jsonl")
    uri = f"ws://127.0.0.1:{a.port}/showdown/websocket"
    cells: Dict[str, Any] = {}
    halts: List[Dict[str, Any]] = []
    srv = start_master(showdown, a.port, out / "master_server.log")
    try:
        jobs = [("bot", b) for b in a.bots.split(",") if b and a.games_per_bot > 0]
        jobs += [("metamon", m) for m in a.metamon.split(",") if m and a.metamon_games > 0]
        if a.self_games > 0:
            jobs.append(("self", "self"))
        for k, (kind, what) in enumerate(jobs):
            cell = f"{kind}:{what}"
            t0 = time.time()
            peer = None
            try:
                if kind == "self":
                    cells[cell] = asyncio.run(self_games(uri, a.self_games, policy, teams, a.seed * 100 + k, rec))
                elif kind == "metamon":
                    peer_name, our_name = f"p4mm{k}", f"p4ours{k}"
                    peer = start_metamon(what, a.port, our_name, peer_name, a.metamon_games, out, a.seed * 100 + k)
                    cells[cell] = asyncio.run(our_games(uri, our_name, peer_name, a.metamon_games, policy, teams,
                                                        a.seed * 100 + k, rec, cell))
                else:
                    peer_name, our_name = f"p4peer{k}", f"p4ours{k}"
                    n = a.games_per_bot
                    pargs = ["--bot", what, "--seed", str(a.seed * 100 + k), "--port", str(a.port),
                             "--username", peer_name, "--opponent", our_name, "--n", str(n)]
                    peer = start_peer(pargs, out / f"peer_{k}_{what}.log")
                    cells[cell] = asyncio.run(our_games(uri, our_name, peer_name, n, policy, teams,
                                                        a.seed * 100 + k, rec, cell))
            except LiveParseHalt as exc:
                record_halt(reason=exc.reason, entry_point=f"master_series:{cell}", battle_id=exc.battle_id,
                            offending_lines=exc.offending_lines, stream=exc.stream, exc=exc)
                halts.append({"cell": cell, "reason": exc.reason, "battle": exc.battle_id,
                              "offending": exc.offending_lines[:5]})
                cells[cell] = {"halted": exc.reason}
            except Exception as exc:  # noqa: BLE001 — a non-T28 failure is named, and the series continues
                cells[cell] = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                stop(peer)
            cells[cell] = {**cells.get(cell, {}), "seconds": round(time.time() - t0, 1)}
            print(f"[master] {cell}: {cells[cell]}", flush=True)
    finally:
        stop(srv)
    summary = {
        "gate": "P4 (c) master Node series", "showdown_master": sha, "model": a.model,
        "pool_teams": len(pool), "master_valid_teams": len(teams), "master_rejected_teams": len(val["rejected"]),
        "cells": cells, "our_decisions": rec.decisions, "t28_halts": halts,
        "marker": read_halt(),
        "verdict_c": "PASS" if not halts and all("error" not in v for v in cells.values()) else "FAIL",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("cells", "our_decisions", "t28_halts", "verdict_c")},
                     indent=1, default=str)[:6000])
    return 0 if summary["verdict_c"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
