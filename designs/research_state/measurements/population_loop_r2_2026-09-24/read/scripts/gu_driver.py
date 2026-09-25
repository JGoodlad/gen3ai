#!/usr/bin/env python3
"""G-U — the untaught-8 KILL guard of population-loop ROUND 2, run INCREMENTALLY.

Round 1's driver (``../../../population_loop_r1_2026-09-23/read/scripts/gu_driver.py``) with only the
refs, the output names and the guard contrast changed. Registration:
``designs/research_state/population_loop_round2_2026-09-24.md`` §4.3: refs ``popr2_loop`` = B2
(ai_v13_27_popr2_loop, the round-2 loop generalist), ``popr2_ctrl`` = C2 (ai_v13_28_popr2_ctrl, the
round-2 no-exploiter control), ``plateau_b1`` = G0 (ai_v13_12_plateau) RE-RUN IN FULL as the
reproduction check. KILL iff B2 − C2 is OUTSIDE and BELOW the 3.69 pp floor. Descriptors B2 − B,
C2 − C, B2 − G0, C2 − G0 reuse round 1's BANKED rows for B / C (same tree, same seeds, same pure
function; the reproduction check is the licence). Round 1's docstring follows.

(Round 1:) The registered
cell is ONE ``main.untaught_meter`` invocation with three refs (``popr1_loop`` = B, the loop
generalist; ``popr1_ctrl`` = C, the no-exploiter control; ``plateau_b1`` = G0, ai_v13_12_plateau),
``--opponent untaught_meter_opponent --workers 8 --seed 0``, concurrency 1, 200 games per team.

WHY THIS IS NOT THAT ONE INVOCATION (a declared deviation in FORM, not in content). The owner's rule
of 2026-09-24 (ORCHESTRATOR_SOP §2, "LONG MEASUREMENTS ARE INCREMENTAL") forbids a many-hour one-shot
driver. The meter's own contract makes the split free: a cell is a PURE FUNCTION of (ref, team
index, battle index) — every battle re-seeds the sim (``sim_seed``), both players' sampling
generators (``policy_seeds``) and the opponent-team draw (``pool_sequence``, prefix-consistent), and
the tool's own sharding already relies on it (``untaught_meter_reproducibility_integration_test``).
So this driver runs the tool's ``play_cells`` loop body VERBATIM, one battle at a time, and writes
one durable row per battle. The registered ``plateau_b1`` = 60.19 pp (963/1600) reproduction check
is what licenses the split: a drift anywhere (tree, split, resume) would move it.

UNITS: (ref, team, 25-battle chunk) — 3 × 8 × 8 = 192 units of a few minutes each. ROWS: one JSON
line per battle in ``<rows>/<label>/<team_key>/c<chunk>.jsonl``, flushed + fsynced as the battle
ends. RESUME: a unit skips every battle index already on disk (a torn last line is truncated).

    gu_driver.py run --workers N        # the supervisor: spawns N workers, waits (run it DETACHED)
    gu_driver.py worker --k K --n N     # one worker: its static share of the units
    gu_driver.py status                 # progress from the rows (safe at any time)
    gu_driver.py aggregate --out DIR    # the tool's own aggregate + the registered guard rule

Run with PYTHONPATH = the PINNED tree's src (6eb9c776, the arms' own code), cwd = that tree, and
POKESIM_SIM_BRIDGE_BIN = that tree's own release build.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
READ_DIR = HERE.parent
DEFAULT_ROWS = READ_DIR / "gu_rows"

#: THE REGISTERED REF LIST (§4.3), in the registered order. Labels are the registration's.
REFS = [
    "popr2_loop=/home/goodlad/dev/gen3ai/models/ai_v13_27_popr2_loop/final_model.zip",
    "popr2_ctrl=/home/goodlad/dev/gen3ai/models/ai_v13_28_popr2_ctrl/final_model.zip",
    "plateau_b1=/home/goodlad/dev/gen3ai/models/ai_v13_12_plateau/final_model.zip",
]
TOOL_ARGS = ["--opponent", "untaught_meter_opponent", "--seed", "0", "--workers", "8"]
GAMES_PER_TEAM = 200
CHUNK = 25
SEED = 0

#: the registered reproduction check and floor
REPRO_LABEL, REPRO_WINS, REPRO_N, REPRO_PP = "plateau_b1", 963, 1600, 60.19
FLOOR_PP = 3.69
#: round 1's BANKED rows (B = popr1_loop, C = popr1_ctrl), for the descriptors only
R1_ROWS = READ_DIR.parents[1] / "population_loop_r1_2026-09-23" / "read" / "gu_rows"
R1_REFS = [
    "popr1_loop=/home/goodlad/dev/gen3ai/models/ai_v13_22_popr1_loop/final_model.zip",
    "popr1_ctrl=/home/goodlad/dev/gen3ai/models/ai_v13_23_popr1_ctrl/final_model.zip",
    "plateau_b1=/home/goodlad/dev/gen3ai/models/ai_v13_12_plateau/final_model.zip",
]


def _resolve(refs: List[str]):
    """Resolve through the TOOL's own parser and resolver, exactly as the invocation would."""
    from main import untaught_meter as cli
    args = cli.build_parser().parse_args(list(refs) + TOOL_ARGS)
    cli.apply_baseline_defaults(args)
    refs_r, baseline, controls, opponent, teams = cli.resolve_all(args)
    assert baseline is None and not controls, "the registered cell has no --baseline/--control"
    return args, refs_r, opponent, teams


def _units(refs_r, teams) -> List[Tuple[str, int, int]]:
    """(label, team index, chunk) ordered chunk-major so every ref advances together (CRN)."""
    n_chunks = GAMES_PER_TEAM // CHUNK
    return [(r.label, t.index, c) for c in range(n_chunks) for t in teams for r in refs_r]


def _unit_path(rows: Path, label: str, team_key: str, chunk: int) -> Path:
    return rows / label / team_key / f"c{chunk:02d}.jsonl"


def _read_rows(path: Path) -> Dict[int, dict]:
    """Rows on disk keyed by battle index; a torn last line (a kill mid-write) is TRUNCATED."""
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


def worker(k: int, n: int, rows: Path) -> int:
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

    _, refs_r, opponent, teams = _resolve(REFS)
    by_label = {r.label: r for r in refs_r}
    by_index = {t.index: t for t in teams}
    mine = [u for i, u in enumerate(_units(refs_r, teams)) if i % n == k]

    # ---- play_cells' setup, VERBATIM (src/agents/training/untaught_meter.py) ----
    engine.check_concurrency(1)
    th.set_num_threads(1)
    PinnedTeam, PairedPool = engine._teambuilders()
    maps = load_mappings()
    cv = current_model_version(maps)
    opp_model = engine._strip_debugger(load_foreign_opponent(
        opponent.zip_path, current_version=cv, device="cpu",
        config_path=opponent.config_path)[0])
    pool = PairedPool(TeamLoader().get_all_teams())
    n_pool = len(pool.packed_teams)
    seqs = {t.index: engine.pool_sequence(SEED, t.index, GAMES_PER_TEAM, n_pool) for t in teams}
    models: Dict[str, object] = {}
    commit = os.environ.get("GU_TREE_COMMIT", "?")

    for (label, ti, chunk) in mine:
        team = by_index[ti]
        path = _unit_path(rows, label, team.key, chunk)
        path.parent.mkdir(parents=True, exist_ok=True)
        done = _read_rows(path)
        todo = [j for j in range(chunk * CHUNK, (chunk + 1) * CHUNK) if j not in done]
        if not todo:
            continue
        if label not in models:
            ref = by_label[label]
            models[label] = engine._strip_debugger(load_foreign_opponent(
                ref.zip_path, current_version=cv, device="cpu", config_path=ref.config_path)[0])
        model = models[label]
        # ---- play_cells' per-cell body, VERBATIM but for the battle range ----
        pilot = RLPlayer(model=model, team=PinnedTeam(team.path), battle_format="gen3ou",
                         server_configuration=LocalhostServerConfiguration, mappings=maps,
                         account_configuration=AccountConfiguration(f"UM{ti}a", "pw"),
                         stochastic=True, start_listening=False)
        opp = RLPlayer(model=opp_model, team=pool, battle_format="gen3ou",
                       server_configuration=LocalhostServerConfiguration, mappings=maps,
                       account_configuration=AccountConfiguration(f"UM{ti}b", "pw"),
                       stochastic=True, start_listening=False)
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
                finished = int(pilot.n_finished_battles == 1)
                row = {"label": label, "team_key": team.key, "team_index": ti, "j": j,
                       "chunk": chunk, "finished": finished,
                       "won": int(pilot.n_won_battles) if finished else 0,
                       "tied": int(pilot.n_tied_battles) if finished else 0,
                       "opp_team": seqs[ti][j], "wall_s": round(time.time() - t0, 2),
                       "worker": k, "tree": commit}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        print(f"[w{k}] {time.strftime('%H:%M:%S')} done {label} {team.key} c{chunk:02d}", flush=True)
    print(f"[w{k}] ALL UNITS DONE", flush=True)
    return 0


def run(n: int, rows: Path) -> int:
    rows.mkdir(parents=True, exist_ok=True)
    logs = rows / "_logs"
    logs.mkdir(exist_ok=True)
    procs = []
    for k in range(n):
        fh = open(logs / f"worker_{k}.log", "a")
        procs.append(subprocess.Popen([sys.executable, __file__, "worker", "--k", str(k),
                                       "--n", str(n), "--rows", str(rows)],
                                      stdout=fh, stderr=subprocess.STDOUT))
    (rows / "_logs" / "supervisor.pids").write_text(
        " ".join(str(p.pid) for p in procs) + f"\nsupervisor {os.getpid()}\n")
    rcs = [p.wait() for p in procs]
    print(f"supervisor: worker rcs {rcs}", flush=True)
    return 0 if all(rc == 0 for rc in rcs) else 1


def _collect(rows: Path, refs=None):
    from agents.training import untaught_meter as engine
    _, refs_r, opponent, teams = _resolve(refs or REFS)
    cells: Dict[str, Dict[str, "engine.Cell"]] = {}
    missing = 0
    for r in refs_r:
        cells[r.label] = {}
        for t in teams:
            c = engine.Cell()
            for chunk in range(GAMES_PER_TEAM // CHUNK):
                got = _read_rows(_unit_path(rows, r.label, t.key, chunk))
                for j in range(chunk * CHUNK, (chunk + 1) * CHUNK):
                    d = got.get(j)
                    if d is None:
                        missing += 1
                        continue
                    c.attempted += 1
                    c.opp_teams.append(d["opp_team"])
                    if not d["finished"]:
                        continue
                    c.finished += 1
                    c.wins += d["won"]
                    c.ties += d["tied"]
                    c.losses += 1 - d["won"] - d["tied"]
            cells[r.label][t.key] = c
    return refs_r, opponent, teams, cells, missing


def status(rows: Path) -> int:
    refs_r, _, teams, cells, missing = _collect(rows)
    total = len(refs_r) * len(teams) * GAMES_PER_TEAM
    print(f"battles on disk {total - missing}/{total}  (missing {missing})  "
          "— PROGRESS ONLY; no level is read before n = 1600 per ref")
    for lab, tc in cells.items():
        att = sum(c.attempted for c in tc.values())
        to = sum(c.timeouts for c in tc.values())
        print(f"  {lab:12s} attempted {att:5d}  timeouts {to}")
    return 0


def paired(a, b, seed: int = 20260915):
    """VERBATIM from hidose_delta.py / admission_delta.py — 20,000 draws, ONE shared index set."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(20000, n))
    dd = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(a.mean() - b.mean()),
            float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))


def aggregate(rows: Path, out: Path) -> int:
    import numpy as np
    from agents.training import untaught_meter as engine
    refs_r, opponent, teams, cells, missing = _collect(rows)
    if missing:
        print(f"REFUSING: {missing} battle(s) not on disk — the verdict is read at the "
              "registered n only (SOP §2 rule 4).", file=sys.stderr)
        return 2
    keys = [t.key for t in teams]
    result = engine.aggregate(cells, keys, ref_labels=[r.label for r in refs_r],
                              baseline_label=None, control_labels=[])
    from agents.training.untaught_meter import DEFAULT_TEAMS_MANIFEST
    meta = {"mode": "play (incremental driver, per-battle rows)",
            "teams_manifest": str(DEFAULT_TEAMS_MANIFEST), "teams": [t.to_json() for t in teams],
            "opponent": opponent.to_json(), "refs": [r.to_json() for r in refs_r],
            "games_per_team": GAMES_PER_TEAM, "seed": SEED, "concurrency": 1, "impl": "rust",
            "tree": os.environ.get("GU_TREE_COMMIT", "?"),
            "argv_equivalent": REFS + TOOL_ARGS}
    doc = {"_meta": meta, "result": result}
    out.mkdir(parents=True, exist_ok=True)
    (out / "untaught_popr2.json").write_text(json.dumps(doc, indent=1))
    (out / "untaught_popr2.md").write_text(engine.render_markdown(doc))

    lv = result["levels"]
    per = {lab: np.array([lv[lab]["per_team"][k]["wins"] / lv[lab]["per_team"][k]["finished"]
                          for k in keys]) for lab in lv}
    level_pp = {lab: 100.0 * lv[lab]["wins"] / lv[lab]["finished"] for lab in lv}
    # round 1's banked B / C / G0 rows (descriptors only; same tree, seeds and pure function)
    _, _, teams1, cells1, missing1 = _collect(R1_ROWS, R1_REFS)
    assert not missing1 and [t.key for t in teams1] == keys, (missing1, [t.key for t in teams1])
    for lab, tc in cells1.items():
        per["r1_" + lab] = np.array([tc[k].wins / tc[k].finished for k in keys])
        level_pp["r1_" + lab] = 100.0 * sum(tc[k].wins for k in keys) / sum(tc[k].finished for k in keys)
    rw, rn = lv[REPRO_LABEL]["wins"], lv[REPRO_LABEL]["finished"]
    repro_ok = (rw == REPRO_WINS and rn == REPRO_N)
    r1_repro_same_rows = bool(np.array_equal(per["r1_plateau_b1"], per[REPRO_LABEL]))

    def rule(a: str, b: str) -> dict:
        d, lo, hi = paired(per[a], per[b])
        d, lo, hi = 100 * d, 100 * lo, 100 * hi
        inside = (lo <= FLOOR_PP <= hi) or (lo <= -FLOOR_PP <= hi)
        outside = abs(d) > FLOOR_PP and not inside
        return {"delta_pp": round(d, 2), "ci95_pp": [round(lo, 2), round(hi, 2)],
                "floor_pp": FLOOR_PP, "clause_a": abs(d) > FLOOR_PP,
                "clause_b_ci_excludes_floor_point": not inside,
                "OUTSIDE": outside, "OUTSIDE_BELOW": outside and d < 0,
                "per_team_delta": {k: round(float(per[a][i] - per[b][i]), 4)
                                   for i, k in enumerate(keys)}}

    guard = {
        "what": "G-U, the untaught-8 KILL guard (round-2 registration §4.3)",
        "tree": meta["tree"],
        "reproduction_check": {"label": REPRO_LABEL, "wins": rw, "finished": rn,
                               "level_pp": round(100 * rw / rn, 2), "registered_pp": REPRO_PP,
                               "registered_wins": REPRO_WINS, "PASS": repro_ok,
                               "per_team_identical_to_round1_rows": r1_repro_same_rows},
        "levels_pp": {k: round(v, 2) for k, v in level_pp.items()},
        "timeouts": result["timeouts"],
        "B2_minus_C2 (popr2_loop - popr2_ctrl)": rule("popr2_loop", "popr2_ctrl"),
        "B2_minus_B (popr2_loop - r1 popr1_loop), descriptor": rule("popr2_loop", "r1_popr1_loop"),
        "C2_minus_C (popr2_ctrl - r1 popr1_ctrl), descriptor": rule("popr2_ctrl", "r1_popr1_ctrl"),
        "B2_minus_G0 (popr2_loop - plateau_b1), descriptor": rule("popr2_loop", "plateau_b1"),
        "C2_minus_G0 (popr2_ctrl - plateau_b1), descriptor": rule("popr2_ctrl", "plateau_b1"),
        "bootstrap": "20,000 draws, ONE shared team index set, seed 20260915; the TEAM is the unit",
    }
    bc = guard["B2_minus_C2 (popr2_loop - popr2_ctrl)"]
    guard["KILL"] = bool(repro_ok and bc["OUTSIDE_BELOW"])
    guard["VOID"] = (not repro_ok) or bool(result["timeouts"]["inconclusive"])
    (out / "gu_guard.json").write_text(json.dumps(guard, indent=1))
    print(json.dumps({k: v for k, v in guard.items() if "per_team" not in str(k)}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "worker", "status", "aggregate"))
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rows", default=str(DEFAULT_ROWS))
    ap.add_argument("--out", default=str(READ_DIR))
    a = ap.parse_args()
    rows = Path(a.rows)
    if a.cmd == "run":
        return run(a.workers, rows)
    if a.cmd == "worker":
        return worker(a.k, a.n, rows)
    if a.cmd == "status":
        return status(rows)
    return aggregate(rows, Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
