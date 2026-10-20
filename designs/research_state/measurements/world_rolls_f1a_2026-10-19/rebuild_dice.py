"""F1 tier A, STEP 5 (descriptive): the honest builder REBUILT on the same states with ``dice="seed"`` (today) and
``dice="rolls"`` (tier A: each damage roll's VALUE any roll that prints what the searched side saw, the stream aligned;
``main.value_search.worlds``, THE DICE). World BUILDS only: no playout value, no model, no outcome, no study arm re-run.

The states are step 1's (``screen_shadow.sample_ids``: a seeded random prefix of the stage-0 population plus a seeded
sample of the states the stored F rows EXCLUDED). The call is the studies' own: ``sample(log, at, side, J=8,
key=<decision id>, screen_cache={})``, ``max_tries`` 60. Per state and mode it keeps the teams, the per-world fact
match flags, the info audit's regate (and its count of seen mons whose EXACT HP at the root equals the truth's), and
under ``rolls`` each world's likelihood log-weight and the number of rolls it took off the stream.

    PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
    CUDA_VISIBLE_DEVICES="" POKESIM_EMISSION_SELFCHECK=1 nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 \\
        $PY -I rebuild_dice.py --dice rolls --out <dir>

One row per state appended to ``<out>/<dice>.jsonl`` (resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "world_legality_f2_2026-10-19"))

J = 8
MAX_TRIES = 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dice", choices=("seed", "rolls"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--n-excluded", type=int, default=20)
    a = ap.parse_args()

    import main as _m

    if not str(Path(_m.__file__).resolve()).startswith(str(REPO / "src")):
        raise SystemExit(f"main imported from {_m.__file__}, not this checkout: run with -I")
    import rebuild_worlds as F2
    import screen_shadow as S1
    from main.info_audit import checks as C
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search.worlds import WorldError, WorldSampler, prepare_root

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stored = F2.stored_rows()
    ids = S1.sample_ids(stored, a.n_random, a.n_excluded)
    rand = set(ids[: a.n_random])
    path = out / f"{a.dice}.jsonl"
    done = {json.loads(x)["id"] for x in path.read_text().splitlines() if x.strip()} if path.exists() else set()
    ids = [i for i in ids if i not in done]
    sampler = WorldSampler(reveal="off", max_tries=MAX_TRIES, dice=a.dice)
    banks = {pop: load_bank(bank) for pop, bank, _f, _k in F2.populations()}
    decs = {pop: {d["id"]: d for d in b.decisions} for pop, b in banks.items()}
    head = os.popen(f"git -C {REPO} rev-parse HEAD").read().strip()
    print(f"[f1a rebuild] {a.dice}: {len(ids)} states at {head[:8]}", flush=True)
    t0 = time.monotonic()
    with open(path, "a") as fo:
        for n_i, did in enumerate(ids):
            pop = stored[did]["pop"]
            bank = banks[pop]
            d = decs[pop][did]
            b = bank.battles[bank.battle_index[d["battle"]]]
            side = d["side"]
            at = cmd_index(b, side, d["n"])
            log = to_log(b)
            oi = 1 if side == "p1" else 0
            row = {"id": did, "pop": pop, "dice": a.dice, "head": head,
                   "stratum": "random" if did in rand else "excluded", "stored_ok": bool(stored[did].get("ok"))}
            w0 = time.monotonic()
            try:
                W = sampler.sample(log, at, side, J, key=did, screen_cache={})
                root = prepare_root(log, at, side, did)
                rg = C.regate(root, W)
                row.update(ok=True, teams=list(W.teams), true_team=log["teams"][oi], tries=W.tries, fails=W.fails,
                           screens=W.screens, log_w=W.log_w, widened=[len(r) for r in W.rolls],
                           facts=[[{c: f[c]["match"] for c in ("item", "ability", "spread") if c in f}
                                   for f in fl if f.get("seen")] for fl in W.facts],
                           regate=rg["pass"], seen_exact_hp_equal=rg["seen_exact_hp_equal"],
                           seen_mons=rg["seen_mons"],
                           samplers=[{k: s[k] for k in ("species", "proposals", "steps", "failed")}
                                     for s in W.samplers])
            except WorldError as exc:
                row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:400]}")
            except Exception as exc:                 # a DEFECT (the guards included): recorded and printed
                row.update(ok=False, crash=True, error=f"{type(exc).__name__}: {str(exc)[:400]}",
                           trace=traceback.format_exc()[-1500:])
                print(f"[f1a rebuild] DEFECT {did}: {row['error'][:300]}", flush=True)
            row["wall_s"] = round(time.monotonic() - w0, 2)
            fo.write(json.dumps(row, sort_keys=True) + "\n")
            fo.flush()
            print(f"[f1a rebuild] {a.dice} {n_i + 1}/{len(ids)} ok={row['ok']} {(time.monotonic() - t0) / 60:.1f} min",
                  flush=True)
    print("[f1a rebuild] done", flush=True)


if __name__ == "__main__":
    main()
