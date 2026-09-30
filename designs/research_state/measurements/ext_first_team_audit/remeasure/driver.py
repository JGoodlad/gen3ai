#!/usr/bin/env python3
"""F-LH-13 re-measure — the population loop's manipulation checks over ALL of each specialist's teams.

Pre-registration: PREREG.md beside this file (its sha256 is stamped on every row). The loop body is
``agents.training.untaught_meter.play_cells``' per-battle body (pin 6eb9c776), with the SPECIALIST as
the pilot on one of its own pinned teams, the GENERALIST as the opponent on the pool, and BOTH sides
greedy (``stochastic=False``) — the ``ext_`` series' regime.

UNITS: (generalist, specialist, team, 10-battle chunk). ROWS: one JSON line per battle, fsynced.
RESUME: a unit skips every battle index on disk (a torn last line is truncated).

    driver.py run --workers N     # supervisor (run DETACHED)
    driver.py worker --k K --n N
    driver.py status              # progress only
    driver.py aggregate --out F   # refuses before the registered n

Environment: PYTHONPATH = the pinned tree's src, cwd = that tree, POKESIM_SIM_BRIDGE_BIN = its build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows"
PREREG = HERE / "PREREG.md"
PREREG_SHA = hashlib.sha256(PREREG.read_bytes()).hexdigest()
M = "/home/goodlad/dev/gen3ai/models/"

GENERALISTS = {
    "B1": M + "ai_v13_22_popr1_loop/snapshots/snapshot_000100000032.zip",
    "C1": M + "ai_v13_23_popr1_ctrl/snapshots/snapshot_000100000032.zip",
    "B2": M + "ai_v13_27_popr2_loop/snapshots/snapshot_000110000016.zip",
    "C2": M + "ai_v13_28_popr2_ctrl/snapshots/snapshot_000110000016.zip",
}
CONFIGS = {
    "B1": M + "ai_v13_22_popr1_loop/model_config.json",
    "C1": M + "ai_v13_23_popr1_ctrl/model_config.json",
    "B2": M + "ai_v13_27_popr2_loop/model_config.json",
    "C2": M + "ai_v13_28_popr2_ctrl/model_config.json",
}
SPECIALISTS = {
    "A": M + "ai_v13_18_teach5_offense_hidose/final_model.zip",
    "S13": M + "ai_v13_13_exploit5_offense/final_model.zip",
    "RB": M + "ai_v13_24_popr1_read_loop/final_model.zip",
}
ROUNDS = {1: (("B1", "C1"), ("A", "S13")), 2: (("B2", "C2"), ("A", "S13", "RB"))}
GAMES_PER_TEAM = 60
CHUNK = 10
SEED = 0


def _cells() -> List[Tuple[str, str]]:
    out = []
    for _, (gens, specs) in ROUNDS.items():
        for g in gens:
            for s in specs:
                out.append((g, s))
    return out


def _teams(spec: str):
    """The specialist's recorded pin, in index order, as untaught_meter TeamSlices."""
    from agents.training import untaught_meter as engine
    from agents.training.matchup_spec import read_recorded_trainee_teams
    run_dir = os.path.dirname(SPECIALISTS[spec])
    teams = read_recorded_trainee_teams(run_dir, require_teams=True)
    man = HERE / f"_manifest_{spec}.json"
    man.write_text(json.dumps({"teams": teams}))
    return engine.load_team_manifest(str(man), prefix="T")


def _units() -> List[Tuple[str, str, int, int]]:
    n_chunks = GAMES_PER_TEAM // CHUNK
    # chunk-major so every cell advances together; B and C of a round adjacent (CRN)
    return [(g, s, ti, c) for c in range(n_chunks) for ti in range(5) for (g, s) in _cells()]


def _unit_path(g: str, s: str, ti: int, c: int) -> Path:
    return ROWS / g / s / f"t{ti}" / f"c{c:02d}.jsonl"


def _read_rows(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n") + 1
        with open(path, "r+b") as fh:
            fh.truncate(cut)
        raw = raw[:cut]
    out: Dict[int, dict] = {}
    for line in raw.decode().splitlines():
        if line.strip():
            d = json.loads(line)
            out[int(d["j"])] = d
    return out


def worker(k: int, n: int) -> int:
    import asyncio

    import torch as th
    from poke_env.ps_client import AccountConfiguration
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

    from agents.inference.player import RLPlayer
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings
    from agents.training import untaught_meter as engine
    from utils.bridge.local_battle_runner import run_local_battles
    from utils.team_loader import TeamLoader

    engine.check_concurrency(1)
    th.set_num_threads(1)
    PinnedTeam, PairedPool = engine._teambuilders()
    maps = load_mappings()
    cv = current_model_version(maps)
    pool = PairedPool(TeamLoader().get_all_teams())
    n_pool = len(pool.packed_teams)
    seqs = {ti: engine.pool_sequence(SEED, ti, GAMES_PER_TEAM, n_pool) for ti in range(5)}
    teams = {s: _teams(s) for s in SPECIALISTS}
    for s, ts in teams.items():
        assert len(ts) == 5, (s, len(ts))
    mine = [u for i, u in enumerate(_units()) if i % n == k]
    models: Dict[str, object] = {}
    commit = os.environ.get("AUDIT_TREE_COMMIT", "?")

    def model(key: str, path: str, cfg):
        if key not in models:
            models[key] = engine._strip_debugger(load_foreign_opponent(
                path, current_version=cv, device="cpu", config_path=cfg)[0])
        return models[key]

    for (g, s, ti, c) in mine:
        path = _unit_path(g, s, ti, c)
        path.parent.mkdir(parents=True, exist_ok=True)
        done = _read_rows(path)
        todo = [j for j in range(c * CHUNK, (c + 1) * CHUNK) if j not in done]
        if not todo:
            continue
        team = teams[s][ti]
        spec_model = model(s, SPECIALISTS[s], None)
        gen_model = model(g, GENERALISTS[g], CONFIGS[g])
        pilot = RLPlayer(model=spec_model, team=PinnedTeam(team.path), battle_format="gen3ou",
                         server_configuration=LocalhostServerConfiguration, mappings=maps,
                         account_configuration=AccountConfiguration(f"XA{ti}a", "pw"),
                         stochastic=False, start_listening=False)
        opp = RLPlayer(model=gen_model, team=pool, battle_format="gen3ou",
                       server_configuration=LocalhostServerConfiguration, mappings=maps,
                       account_configuration=AccountConfiguration(f"XA{ti}b", "pw"),
                       stochastic=False, start_listening=False)
        pool.set_sequence(seqs[ti])
        with open(path, "a") as fh:
            for j in todo:
                t0 = time.time()
                ps, os_ = engine.policy_seeds(SEED, ti, j)
                engine._reseed_player(pilot, ps)
                engine._reseed_player(opp, os_)
                pool.at(j)
                pilot.reset_battles()
                opp.reset_battles()
                asyncio.run(run_local_battles(pilot, opp, 1, concurrency=1, impl="rust",
                                              seed=engine.sim_seed(SEED, ti, j)))
                fin = int(pilot.n_finished_battles == 1)
                row = {"gen": g, "spec": s, "team_index": ti, "team_pin_sha": team.pin_sha,
                       "team_sha": team.team_sha, "j": j, "chunk": c, "finished": fin,
                       "spec_won": int(pilot.n_won_battles) if fin else 0,
                       "tied": int(pilot.n_tied_battles) if fin else 0,
                       "opp_team": seqs[ti][j], "wall_s": round(time.time() - t0, 2),
                       "worker": k, "tree": commit, "prereg_sha": PREREG_SHA}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        print(f"[w{k}] {time.strftime('%H:%M:%S')} done {g} {s} t{ti} c{c:02d}", flush=True)
    print(f"[w{k}] ALL UNITS DONE", flush=True)
    return 0


def run(n: int) -> int:
    logs = ROWS / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for k in range(n):
        fh = open(logs / f"worker_{k}.log", "a")
        procs.append(subprocess.Popen([sys.executable, __file__, "worker", "--k", str(k), "--n", str(n)],
                                      stdout=fh, stderr=subprocess.STDOUT))
    (logs / "supervisor.pids").write_text(" ".join(str(p.pid) for p in procs)
                                          + f"\nsupervisor {os.getpid()}\n")
    rcs = [p.wait() for p in procs]
    print(f"supervisor: worker rcs {rcs}", flush=True)
    return 0 if all(rc == 0 for rc in rcs) else 1


def _collect():
    """{(gen, spec, ti): [rows]} and the count of missing battles."""
    out: Dict[Tuple[str, str, int], List[dict]] = {}
    missing = 0
    for (g, s) in _cells():
        for ti in range(5):
            rows: List[dict] = []
            for c in range(GAMES_PER_TEAM // CHUNK):
                got = _read_rows(_unit_path(g, s, ti, c))
                for j in range(c * CHUNK, (c + 1) * CHUNK):
                    if j in got:
                        rows.append(got[j])
                    else:
                        missing += 1
            out[(g, s, ti)] = rows
    return out, missing


def status() -> int:
    cells, missing = _collect()
    total = len(_cells()) * 5 * GAMES_PER_TEAM
    walls = [r["wall_s"] for rs in cells.values() for r in rs]
    to = sum(1 for rs in cells.values() for r in rs if not r["finished"])
    mean_wall = (sum(walls) / len(walls)) if walls else float("nan")
    print(f"battles on disk {total - missing}/{total}  timeouts {to}  mean wall {mean_wall:.1f}s "
          "— PROGRESS ONLY; no level is read before the registered n")
    return 0


def _gen_counts(rows: List[dict]) -> Tuple[int, int, int]:
    """(generalist wins, finished, attempted)."""
    fin = [r for r in rows if r["finished"]]
    gw = sum(1 for r in fin if not r["spec_won"] and not r["tied"])
    return gw, len(fin), len(rows)


def aggregate(out: Path) -> int:
    import numpy as np
    from agents.training.best_response_gap import newcombe_diff_ci, wilson_ci

    cells, missing = _collect()
    if missing:
        print(f"REFUSING: {missing} battle(s) not on disk — the verdict is read at the registered n "
              "only.", file=sys.stderr)
        return 2
    banked_team0 = {"B1": (88, 200), "C1": (76, 200), "B2": (146, 300), "C2": (109, 300)}
    res: Dict[str, object] = {"prereg_sha": PREREG_SHA, "games_per_team": GAMES_PER_TEAM,
                              "regime": "greedy vs greedy; specialist pilots its own team; "
                                        "generalist on the pool", "rounds": {}}
    inconclusive = []
    for (g, s, ti), rows in cells.items():
        gw, fin, att = _gen_counts(rows)
        if att and (att - fin) / att > 0.25:
            inconclusive.append([g, s, ti, att - fin, att])
    res["inconclusive_cells"] = inconclusive
    rng = np.random.default_rng(20260930)
    for rnd, ((bg, cg), specs) in ROUNDS.items():
        R: Dict[str, object] = {}

        def pooled(g, specs_, tis):
            w = f = 0
            for s in specs_:
                for ti in tis:
                    a, b, _ = _gen_counts(cells[(g, s, ti)])
                    w += a
                    f += b
            return w, f

        wb, nb = pooled(bg, specs, range(5))
        wc, nc = pooled(cg, specs, range(5))
        d, lo, hi = newcombe_diff_ci(wb, nb, wc, nc)
        # paired cluster bootstrap over (specialist, team) cells
        keys = [(s, ti) for s in specs for ti in range(5)]
        rb = np.array([_gen_counts(cells[(bg, s, ti)])[0] / max(1, _gen_counts(cells[(bg, s, ti)])[1])
                       for s, ti in keys])
        rc = np.array([_gen_counts(cells[(cg, s, ti)])[0] / max(1, _gen_counts(cells[(cg, s, ti)])[1])
                       for s, ti in keys])
        idx = rng.integers(0, len(keys), size=(20000, len(keys)))
        dd = rb[idx].mean(axis=1) - rc[idx].mean(axis=1)
        R["M_all"] = {"B": [wb, nb, wb / nb], "C": [wc, nc, wc / nc], "point": d, "ci95": [lo, hi],
                      "ABSORBED": bool(lo > 0),
                      "cluster_bootstrap_ci95": [float(np.percentile(dd, 2.5)),
                                                 float(np.percentile(dd, 97.5))]}
        per_spec = {}
        for s in specs:
            a, na = pooled(bg, [s], range(5))
            b, nb2 = pooled(cg, [s], range(5))
            dd2, l2, h2 = newcombe_diff_ci(a, na, b, nb2)
            per_spec[s] = {"B": [a, na], "C": [b, nb2], "B_minus_C": dd2, "ci95": [l2, h2]}
        R["per_specialist"] = per_spec
        per_team = {}
        for ti in range(5):
            a, na = pooled(bg, specs, [ti])
            b, nb2 = pooled(cg, specs, [ti])
            dd2, l2, h2 = newcombe_diff_ci(a, na, b, nb2)
            per_team[f"t{ti}"] = {"team_sha": cells[(bg, specs[0], ti)][0]["team_sha"],
                                  "B": [a, na], "C": [b, nb2], "B_minus_C": dd2, "ci95": [l2, h2]}
        R["per_team"] = per_team
        # representativeness: team 0 vs teams 1-4, per generalist
        rep = {}
        for g in (bg, cg):
            a, na = pooled(g, specs, [0])
            b, nb2 = pooled(g, specs, range(1, 5))
            dd2, l2, h2 = newcombe_diff_ci(a, na, b, nb2)
            rep[g] = {"team0": [a, na, a / na], "teams1_4": [b, nb2, b / nb2],
                      "team0_minus_rest": dd2, "ci95": [l2, h2]}
        R["representativeness"] = rep
        # the team-0-only M (the re-measured analogue of the banked read)
        a, na = pooled(bg, specs, [0])
        b, nb2 = pooled(cg, specs, [0])
        dd2, l2, h2 = newcombe_diff_ci(a, na, b, nb2)
        R["M_team0_remeasured"] = {"point": dd2, "ci95": [l2, h2]}
        # reproduction check vs the banked same-cycle row
        repro = {}
        for g in (bg, cg):
            a, na = pooled(g, specs, [0])
            bw, bn = banked_team0[g]
            dd2, l2, h2 = newcombe_diff_ci(a, na, bw, bn)
            repro[g] = {"remeasured_team0": [a, na], "banked_team0": [bw, bn], "diff": dd2,
                        "ci95": [l2, h2], "PASS": bool(l2 <= 0 <= h2),
                        "remeasured_wilson": list(wilson_ci(a, na))}
        R["reproduction_check"] = repro
        res["rounds"][str(rnd)] = R
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=6)
    w = sub.add_parser("worker")
    w.add_argument("--k", type=int, required=True)
    w.add_argument("--n", type=int, required=True)
    sub.add_parser("status")
    a = sub.add_parser("aggregate")
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        return run(args.workers)
    if args.cmd == "worker":
        return worker(args.k, args.n)
    if args.cmd == "status":
        return status()
    return aggregate(Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
