"""P4 gate (a) — TWO ROADS, ONE ROW: the live session's rows equal the training core's, every decision.

Road 1 (LIVE): battles are played over a real websocket against OUR Rust front end (``--server rust``,
``utils.bridge.ws_frontend``, seeded, capturing) by two :class:`main.live.client.LiveClient` s; every decision's
frame is the one the reader session (``live_reader``) built from the server's messages, real framing and ``rqid``
included.

Road 2 (TRAINING): each captured battle's exact command stream (the seeded ``START`` + every ``CHOOSE`` /
``FORCELOSE`` the front end wrote to its child) is replayed into a FRESH ``sim_bridge`` with the core observation
mode ON for both sides (``"core_obs": {"sides": ["p1", "p2"]}``) — the chain training's rows are held byte-equal
to (``encoder.md`` §5a; the env core, gate ①).

The verdict, per side and per decision: the frame's row BYTES, mask, tokens, turn, stream line and frame index
``n`` all equal; the frame COUNT equal; the only field excluded is ``rqid`` (the server's injected id, which the
bridge never sees — ``ws_frontend.md`` "The rqid"; it is checked separately to be the id the client echoed).
The replay must also reproduce the captured protocol chunks byte for byte (``rqid`` stripped), so the two roads
are provably the SAME battle; a battle whose replay diverges is reported as UNREPLAYABLE, never as agreement.

Run::

    python -m main.live.two_roads --battles 1000 --pairs 8 --policy random --out <dir>
    python -m main.live.two_roads --battles 100 --pairs 4 --policy model:<ckpt.zip> --out <dir>
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_RQID = re.compile(r',"rqid":\d+\}$')


def strip_rqid(chunk: str) -> str:
    return "\n".join(_RQID.sub("}", ln) if ln.startswith("|request|") else ln for ln in chunk.split("\n"))


def replay_core_obs(capture: Dict[str, Any], bridge_argv: List[str]) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]], List[str]]:
    """Replay one front-end capture through a fresh ``sim_bridge`` with core_obs ON for both sides.

    Returns (frames per side as raw JSON text, chunks as (side, text), errors)."""
    cmds = [c for c in capture["commands"] if c != "END"]
    if not cmds or not cmds[0].startswith("START "):
        raise ValueError("a capture with no START")
    start = json.loads(cmds[0][len("START "):])
    if start.get("seed") is None:
        raise ValueError("an UNSEEDED capture is not replayable (ws_frontend.md H3)")
    start["core_obs"] = {"sides": ["p1", "p2"]}
    stdin = "START " + json.dumps(start) + "\n" + "".join(c + "\n" for c in cmds[1:]) + "END\n"
    proc = subprocess.run(bridge_argv, input=stdin, capture_output=True, text=True, timeout=300)
    frames: Dict[str, List[str]] = {"p1": [], "p2": []}
    chunks: List[Tuple[str, str]] = []
    errors: List[str] = []
    for ln in proc.stdout.splitlines():
        if ln.startswith("__OBS__ "):
            _, side, payload = ln.split(" ", 2)
            frames[side].append(payload)
        elif ln.startswith("__ERR__ "):
            errors.append(base64.b64decode(ln[len("__ERR__ "):]).decode("utf-8", "replace"))
        elif ln.startswith(("p1 ", "p2 ")):
            side, b64 = ln.split(" ", 1)
            chunks.append((side, base64.b64decode(b64).decode("utf-8")))
    return frames, chunks, errors


def frame_key(text: str) -> Dict[str, Any]:
    obj = json.loads(text)
    obj.pop("rqid", None)
    return obj


def compare_battle(live: Dict[str, List[Dict[str, Any]]], capture: Dict[str, Any],
                   bridge_argv: List[str]) -> Dict[str, Any]:
    """One battle's verdict. ``live[side]`` = the recorded decisions ({"raw", "rqid", "sent_rqid"})."""
    frames, chunks, errors = replay_core_obs(capture, bridge_argv)
    out: Dict[str, Any] = {"tag": capture["tag"], "errors": errors, "sides": {}}
    cap_chunks = [(s, strip_rqid(c)) for s, c in capture["chunks"]]
    out["same_battle"] = cap_chunks == chunks
    if not out["same_battle"]:
        first = next((i for i, (a, b) in enumerate(zip(cap_chunks, chunks)) if a != b), min(len(cap_chunks), len(chunks)))
        out["chunk_divergence"] = {"index": first, "captured": len(cap_chunks), "replayed": len(chunks)}
    for side in ("p1", "p2"):
        L = live.get(side, [])
        C = frames[side]
        diffs = []
        for i, (lv, cv) in enumerate(zip(L, C)):
            a, b = frame_key(lv["raw"]), frame_key(cv)
            if a != b:
                fields = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                diffs.append({"n": i, "fields": fields})
            if not isinstance(lv.get("rqid"), int):
                # every live request carries the server's injected rqid (the client echoes it)
                diffs.append({"n": i, "fields": ["rqid_missing"], "rqid": lv.get("rqid")})
        if len(L) != len(C):
            diffs.append({"fields": ["count"], "live": len(L), "core": len(C)})
        out["sides"][side] = {"decisions": len(L), "core_frames": len(C), "diffs": diffs}
    out["ok"] = (not errors and out["same_battle"]
                 and all(not v["diffs"] for v in out["sides"].values()))
    return out


# -- road 1: the live series ------------------------------------------------------------------

def start_frontend(port: int, seed_base: int, capture_dir: Path, log: Path) -> subprocess.Popen:
    from utils.paths import src_root
    argv = [sys.executable, "-m", "utils.bridge.ws_frontend", "--port", str(port), "--impl", "rust",
            "--seed-base", str(seed_base), "--capture-dir", str(capture_dir)]
    fh = open(log, "w")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root()) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=str(src_root()), env=env)
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the front end exited {proc.returncode}; see {log}")
        if "READY" in log.read_text(errors="replace"):
            return proc
        time.sleep(0.2)
    proc.kill()
    raise RuntimeError(f"the front end never printed READY; see {log}")


async def play_pair(uri: str, i: int, n: int, policy_spec: str, seed: int, teams: List[str],
                    record: Dict[str, Dict[str, List[Dict[str, Any]]]], shared: Any = None) -> None:
    import random

    from main.live.client import ClientConfig, LiveClient
    from main.live.policy import RandomPolicy

    def policy(k: int):
        if policy_spec == "random":
            return RandomPolicy(seed * 1000 + 2 * i + k)
        return shared  # ONE greedy model shared by every client (stateless at T = 0; loaded before any connect)

    rng = random.Random(seed * 7919 + i)

    def team():
        return rng.choice(teams)

    def on_decision(name: str):
        def cb(d) -> None:
            side_rec = record.setdefault(d.battle, {}).setdefault(d.frame.side, [])
            side_rec.append({"raw": d.frame.raw, "rqid": d.frame.rqid,
                             "index": d.index, "token": d.token, "who": name})
        return cb

    na, nb = f"p4a{seed % 1000}x{i}", f"p4b{seed % 1000}y{i}"
    a = LiveClient(ClientConfig(uri=uri, username=na), policy=policy(0), team_fn=team, on_decision=on_decision(na))
    b = LiveClient(ClientConfig(uri=uri, username=nb), policy=policy(1), team_fn=team, on_decision=on_decision(nb))
    try:
        await b.connect()
        await a.connect()
        acc = asyncio.ensure_future(b.accept(a.name, n))
        await a.challenge(b.name or nb, n)
        await acc
    finally:
        await a.close()
        await b.close()


async def play_series(uri: str, battles: int, pairs: int, policy_spec: str, seed: int) -> Dict[str, Any]:
    from utils.team_sources import packed_teams
    teams = packed_teams("pool")
    record: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    per = [battles // pairs + (1 if k < battles % pairs else 0) for k in range(pairs)]
    shared = None
    if policy_spec != "random":
        from main.live.policy import ModelPolicy
        shared = ModelPolicy(policy_spec.split(":", 1)[1], temperature=0.0)
    await asyncio.gather(*(play_pair(uri, k, per[k], policy_spec, seed, teams, record, shared)
                           for k in range(pairs) if per[k] > 0))
    return record


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.live.two_roads", description=__doc__.split("\n\n")[0])
    ap.add_argument("--battles", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--policy", default="random", help="random | model:<ckpt.zip>")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=9581)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from main.play import check_port
    from utils.bridge.sim_bridge_bin import resolve_sim_bridge_bin
    check_port(a.port)
    out = Path(a.out)
    cap = out / "captures"
    cap.mkdir(parents=True, exist_ok=True)
    bridge = [resolve_sim_bridge_bin()]
    fe = start_frontend(a.port, 7_000_000 + a.seed * 10_000, cap, out / "frontend.log")
    t0 = time.time()
    try:
        record = asyncio.run(play_series(f"ws://127.0.0.1:{a.port}/showdown/websocket", a.battles, a.pairs,
                                         a.policy, a.seed))
    finally:
        fe.terminate()
        try:
            fe.wait(timeout=20)
        except subprocess.TimeoutExpired:
            fe.kill()
            fe.wait()
    played = time.time() - t0
    verdicts = []
    for tag, sides in sorted(record.items()):
        path = cap / f"{tag}.json"
        if not path.exists():
            verdicts.append({"tag": tag, "ok": False, "errors": ["no capture"]})
            continue
        verdicts.append(compare_battle(sides, json.loads(path.read_text()), bridge))
    n_dec = sum(v["sides"][s]["decisions"] for v in verdicts if "sides" in v for s in ("p1", "p2"))
    bad = [v for v in verdicts if not v["ok"]]
    summary = {"gate": "P4 (a) two roads, one row", "battles": len(verdicts), "decisions": n_dec,
               "failing_battles": len(bad), "policy": a.policy, "seed": a.seed, "play_seconds": round(played, 1),
               "sim_bridge": bridge[0], "verdict": "PASS" if verdicts and not bad else "FAIL"}
    (out / "verdicts.jsonl").write_text("".join(json.dumps(v) + "\n" for v in verdicts))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    for v in bad[:10]:
        print("FAIL", json.dumps(v)[:600])
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
