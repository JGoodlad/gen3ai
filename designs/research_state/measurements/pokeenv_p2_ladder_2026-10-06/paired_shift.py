"""P2's PAIRED COMPARISON: the same frozen ladder pairs played on the PYTHON stack (the pre-P2 ladder: two poke-env
``RLPlayer`` s over the in-process bridge, the Python encoder, unmirrored, ``a`` on p1, greedy) and on the RUST stack
(``agents.training.snapshot_ladder_play``: the Rust eval engine, seat-balanced mirrored pairs, greedy), on CPU.

Durable + resumable: one JSON row per (pair, stack) appended to ``rows.jsonl`` beside this file (fsynced); a re-run
skips every (pair, stack) already recorded. Output is OUTSIDE ``models/`` (which is read only).

    # from the worktree root (the team pool is read cwd-relative), under the memory cap:
    scripts/ops/mem_cap.sh 16 timeout 3h <python> designs/research_state/measurements/pokeenv_p2_ladder_2026-10-06/paired_shift.py play --stack python
    ... play --stack rust
    ... read                      # the per-pair table + the paired shift with its CI (README.md)

Both stacks run the emission SELF-CHECK Rust builds already in the worktree (the bridge's ``sim_bridge`` and the env
core's ``selfcheck`` profile): the same game semantics as release, plus the self-checks.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("POKESIM_EMISSION_SELFCHECK", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only, never the GPU

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.jsonl"
RUN = "/home/goodlad/dev/gen3ai/models/ai_v14_01_base"     # N0, 20 snapshots, 380 banked Python edges
N_GAMES = 100
#: the pair plan: (i, i + d) over the 20 sorted pool steps, a = the NEWER node (as on a promotion), gaps 1, 3, 6, 10
GAPS = (1, 3, 6, 10)
STARTS = (0, 4, 8, 9)


def plan(steps):
    out = []
    for d in GAPS:
        for i in STARTS:
            if i + d < len(steps):
                out.append((steps[i + d], steps[i]))
    return out


def done():
    have = set()
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                have.add((r["a"], r["b"], r["stack"]))
    return have


def append(row):
    with open(ROWS, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------------------------- the PYTHON stack
def _play_pair_python(run_dir, step_a, step_b, n_games, mappings, cv, all_teams, sample_teams):
    """The pre-P2 ladder's `_play_pair` (snapshot_ladder.py at 11d27574), compile off, impl rust (what the
    in-training updater passed), concurrency 4."""
    import torch
    torch.set_num_threads(1)
    from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

    from agents.inference.player import RLPlayer
    from agents.model.snapshot import load_foreign_opponent
    from agents.training import snapshot_ladder as sl
    from utils.bridge.local_battle_runner import run_local_battles
    from utils.teambuilder import Gen3Teambuilder

    cfg = os.path.join(run_dir, "snapshots", "model_config.json")
    if not os.path.exists(cfg):
        cfg = os.path.join(run_dir, "model_config.json")

    def _player(step, tag):
        model, _ = load_foreign_opponent(sl._snapshot_zip(run_dir, step), current_version=cv, device="cpu",
                                         config_path=cfg)
        return RLPlayer(model=model, team=Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=0.1),
                        battle_format="gen3ou", server_configuration=LocalhostServerConfiguration, mappings=mappings,
                        account_configuration=AccountConfiguration(f"L{tag}", "pw"), stochastic=False,
                        start_listening=False)

    pa = _player(step_a, f"a{step_a % 100000:05d}")
    pb = _player(step_b, f"b{step_b % 100000:05d}")
    pa.reset_battles(); pb.reset_battles()
    asyncio.run(run_local_battles(pa, pb, n_games, concurrency=4, impl="rust"))
    return pa.n_won_battles, pa.n_tied_battles, pa.n_finished_battles


def play_python(pairs, have):
    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import load_mappings
    from utils.team_loader import TeamLoader

    mappings = load_mappings()
    cv = current_model_version(mappings)
    loader = TeamLoader()
    all_teams, sample = loader.get_all_teams(), loader.get_sample_teams()
    for a, b in pairs:
        if (a, b, "python") in have:
            continue
        t0 = time.time()
        w, t, f = _play_pair_python(RUN, a, b, N_GAMES, mappings, cv, all_teams, sample)
        append({"a": a, "b": b, "stack": "python", "wins_a": w, "ties": t, "finished": f, "n_games": N_GAMES,
                "wall_s": round(time.time() - t0, 1), "load1": os.getloadavg()[0], "at": time.time()})
        print(f"[python] {a} vs {b}: {w}/{f} (ties {t}) in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------- the RUST stack
def play_rust(pairs, have):
    from agents.training import snapshot_ladder_play as LP

    todo = [p for p in pairs if (p[0], p[1], "rust") not in have]
    if not todo:
        return
    with LP.open_engine(RUN, todo, n_envs=32, threads=2, torch_threads=2, front="ffi", profile="selfcheck") as eng:
        for a, b in todo:
            t0 = time.time()
            r = eng.play(a, b, N_GAMES, batch=0)
            append({"a": a, "b": b, "stack": "rust", **{k: r[k] for k in (
                "wins_a", "losses_a", "draws", "games", "games_played", "by_seat", "pair_counts", "n_pairs",
                "seat_split", "cycle_seed", "regime_id", "protocol")}, "wall_s": round(time.time() - t0, 1),
                "load1": os.getloadavg()[0], "at": time.time()})
            print(f"[rust] {a} vs {b}: {r['wins_a']}/{r['games']} decisive, {r['draws']} draws, seats "
                  f"{r['by_seat']} in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------- the read
def banked(a, b):
    """The run's own banked Python edge for (a, b) from `games.jsonl` (wins for a, games), summed over rows."""
    from agents.training import snapshot_ladder as sl
    g = sl.load_games(RUN)
    lo, hi = sl._pair_key(a, b)
    if (lo, hi) not in g:
        return None
    w_lo, n = g[(lo, hi)]
    return (w_lo if a == lo else n - w_lo), n


def read(as_json=False):
    import numpy as np

    rows = [json.loads(x) for x in ROWS.read_text().splitlines() if x.strip()]
    by = {}
    for r in rows:
        by.setdefault((r["a"], r["b"]), {})[r["stack"]] = r
    table, d_seat, d_full, d_bank, v_seat, v_full, v_bank = [], [], [], [], [], [], []
    for (a, b), s in sorted(by.items()):
        if "python" not in s or "rust" not in s:
            continue
        py, ru = s["python"], s["rust"]
        p_py = py["wins_a"] / py["finished"]                       # the old fold: a draw counts against a
        ws, ls, ds = ru["by_seat"]["a_p1"]
        p_seat = ws / (ws + ls + ds)                               # Rust, a on p1 only, the same draw fold
        p_full = ru["wins_a"] / ru["games"] if ru["games"] else float("nan")   # the Rust edge as fitted
        bk = banked(a, b)
        p_bank = bk[0] / bk[1] if bk else float("nan")
        var = lambda p, n: p * (1 - p) / n                         # noqa: E731 — binomial (conservative for mirrored)
        d_seat.append(p_seat - p_py); v_seat.append(var(p_py, py["finished"]) + var(p_seat, ws + ls + ds))
        d_full.append(p_full - p_py); v_full.append(var(p_py, py["finished"]) + var(p_full, ru["games"]))
        if bk:
            d_bank.append(p_py - p_bank); v_bank.append(var(p_py, py["finished"]) + var(p_bank, bk[1]))
        table.append({"a": a, "b": b, "py": round(p_py, 3), "rust_a_p1": round(p_seat, 3),
                      "rust_edge": round(p_full, 3), "banked_py": round(p_bank, 3), "rust_draws": ru["draws"],
                      "py_ties": py.get("ties", 0)})

    def summary(d, v, label):
        if not d:
            return {"label": label, "k": 0}
        d = np.asarray(d); k = len(d)
        m = float(d.mean())
        se_model = math.sqrt(sum(v)) / k
        se_emp = float(d.std(ddof=1) / math.sqrt(k)) if k > 1 else float("nan")
        rng = np.random.default_rng(20261006)
        boots = np.array([rng.choice(d, size=k, replace=True).mean() for _ in range(4000)])
        return {"label": label, "k": k, "mean_delta": round(m, 4),
                "ci95_model": [round(m - 1.96 * se_model, 4), round(m + 1.96 * se_model, 4)],
                "ci95_empirical_t": [round(m - 1.96 * se_emp, 4), round(m + 1.96 * se_emp, 4)],
                "ci95_bootstrap": [round(float(np.quantile(boots, .025)), 4), round(float(np.quantile(boots, .975)), 4)],
                "mean_abs_delta": round(float(np.abs(d).mean()), 4)}

    out = {"pairs": table,
           "rust_a_on_p1_minus_python": summary(d_seat, v_seat, "Rust (a on p1, draws vs a) − Python, same protocol"),
           "rust_edge_minus_python": summary(d_full, v_full, "Rust ladder edge (seat-balanced, draws out) − Python"),
           "python_now_minus_banked": summary(d_bank, v_bank, "Python now − Python banked (the same stack's floor)")}
    if as_json:
        print(json.dumps(out, indent=2))
    else:
        print(f"{'a':>12} {'b':>12} {'py':>6} {'r_p1':>6} {'r_edge':>6} {'banked':>6} {'r_dr':>4}")
        for t in table:
            print(f"{t['a']:>12} {t['b']:>12} {t['py']:>6} {t['rust_a_p1']:>6} {t['rust_edge']:>6} "
                  f"{t['banked_py']:>6} {t['rust_draws']:>4}")
        for k in ("rust_a_on_p1_minus_python", "rust_edge_minus_python", "python_now_minus_banked"):
            print(json.dumps(out[k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play"); p.add_argument("--stack", choices=("python", "rust"), required=True)
    p.add_argument("--max-pairs", type=int, default=None)
    r = sub.add_parser("read"); r.add_argument("--json", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(Path.cwd() / "src"))           # the worktree's tree (run from its root)
    if a.cmd == "read":
        read(a.json)
        return 0
    from agents.training import snapshot_ladder as sl
    pairs = plan(sl.pool_snapshot_steps(RUN))[: a.max_pairs]
    print(f"[plan] {len(pairs)} pairs on {RUN}: {pairs}", flush=True)
    (play_python if a.stack == "python" else play_rust)(pairs, done())
    return 0


if __name__ == "__main__":
    sys.exit(main())
