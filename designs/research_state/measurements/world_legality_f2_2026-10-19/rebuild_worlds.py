"""F2 of ``designs/endstate/design_world_sampler.md``: the honest builder (F, ``WorldSampler(reveal="off")``) REBUILT with
the legality check on the stage-0 banks' states, to read the illegal rate, the exclusion rate and the world
distribution before / after. World BUILDS only: no playout, no model, no outcome, no study arm re-run.

States: the stage-0 population (maturity interim LEAN ids; E2 end2 / prod), every state F was asked to build (its OK
and its EXCLUDED rows), read-only from the archives. The study's own build call: ``sample(log, at, side, J=8,
key=<decision id>, screen_cache={})`` with ``max_tries=60``.

Two builds per state, both at THIS checkout:
* ``on``  — the builder as shipped (Showdown's validator, ``world_legality``);
* ``off`` — the same code with an always-legal rule injected (the pre-F2 draw), built ONLY where it can differ: a
  state whose ``on`` build (every ``on`` row present) rejected a draw on legality (``illegal_*`` fails) or raised. Elsewhere ``on`` consumed
  exactly the rng ``off`` would (``new_set``'s contract), so ``off`` = ``on`` there by construction; the script
  RE-CHECKS that on a fixed sample of such states (``--control N``), a check that fails if the claim is wrong.

The stored F rows are the third column (``stored``): the worlds the studies actually ran, built at their own pins.

    PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
    CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 6 \\
        $PY -I rebuild_worlds.py --mode on --shard 0 --of 2 --out <dir>
    ... --mode off --out <dir>              (after every on shard: the states that need it, plus --control)

One row per (state, mode) appended to ``<out>/<mode>_<shard>_of_<W>.jsonl``; a rerun skips the states done (resumable).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

ARCH = Path.home() / "gen3ai_archive"
MAT = ARCH / "hidden_info_maturity"
E2 = ARCH / "hidden_info_e2"
MAT_STATES = REPO / "designs" / "research_state" / "measurements" / "hidden_info_maturity_x26_2026-10-18" / \
    "states_mat_x26.json"
J = 8
MAX_TRIES = 60


def _jsonl(path: Path) -> List[dict]:
    out = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                         # a torn last line of a file still being written (read only)
    return out


def populations():
    lean = {k for k, v in json.loads(MAT_STATES.read_text())["states"].items() if v["lean"]}
    return [("mat_lean", MAT / "bank", MAT / "worlds" / "shared" / "F" / "worlds_0_of_1.jsonl", lean),
            ("e2_end2", E2 / "end2" / "bank", E2 / "end2" / "worlds" / "F" / "worlds_0_of_1.jsonl", None),
            ("e2_prod", E2 / "prod" / "bank", E2 / "prod" / "worlds" / "F" / "worlds_0_of_1.jsonl", None)]


def stored_rows() -> Dict[str, dict]:
    """{decision id: the stored F row (first per id) + its population}: the states F was asked to build."""
    out: Dict[str, dict] = {}
    for pop, _bank, fpath, keep in populations():
        for r in _jsonl(fpath):
            did = r.get("id")
            if did is None or (keep is not None and did not in keep) or did in out:
                continue
            out[did] = dict(r, pop=pop)
    return out


class _AlwaysLegal:
    """The pre-F2 rule (no validator): every set and team legal."""

    def problems(self, teams):
        return [() for _ in teams]


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("on", "off"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", type=int, default=40, help="off: legality-clean states rebuilt as the identity check")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    import main as _m

    if not str(Path(_m.__file__).resolve()).startswith(str(REPO / "src")):
        raise SystemExit(f"main imported from {_m.__file__}, not this checkout ({REPO / 'src'}): run with -I")
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import world_legality as WL
    from main.value_search.worlds import WorldError, WorldSampler, _gender_pad

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stored = stored_rows()
    ids = sorted(stored)
    if a.mode == "off":                      # where on and off can differ, plus the control sample
        on = {}
        for p in out.glob("on_*.jsonl"):
            for r in _jsonl(p):
                on[r["id"]] = r
        need = sorted(i for i, r in on.items() if r.get("legality_touched"))
        clean = sorted(i for i, r in on.items() if not r.get("legality_touched"))
        step = max(1, len(clean) // max(1, a.control))
        ctrl = clean[::step][: a.control]
        ids = sorted(set(need) | set(ctrl))
        ctrl_set = set(ctrl)
    else:
        ctrl_set = set()
    # a SEEDED shuffle, so every prefix of the work is a uniform random sample of the states (the read may stop early
    # on a contended box and still describe the population; the rest stays resumable)
    ids = sorted(ids, key=lambda i: hashlib.sha256(f"f2_2026-10-19:{i}".encode()).hexdigest())
    ids = [i for k, i in enumerate(ids) if k % a.of == a.shard]
    path = out / f"{a.mode}_{a.shard}_of_{a.of}.jsonl"
    done = {r["id"] for r in _jsonl(path)} if path.exists() else set()
    ids = [i for i in ids if i not in done][: a.limit]
    legality = WL.Legality(validator=_AlwaysLegal(), pad=_gender_pad()) if a.mode == "off" else None
    sampler = WorldSampler(reveal="off", max_tries=MAX_TRIES, legality=legality)
    banks = {pop: load_bank(bank) for pop, bank, _f, _k in populations()}
    decs = {pop: {d["id"]: d for d in b.decisions} for pop, b in banks.items()}
    head = os.popen(f"git -C {REPO} rev-parse HEAD").read().strip()
    print(f"[f2 rebuild] {a.mode} shard {a.shard}/{a.of}: {len(ids)} states at {head[:8]}", flush=True)
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
            row = {"id": did, "pop": pop, "mode": a.mode, "head": head, "control": did in ctrl_set}
            c0, w0 = cpu_seconds(), time.monotonic()
            try:
                W = sampler.sample(log, at, side, J, key=did, screen_cache={})
                row.update({"ok": True, "teams": list(W.teams), "tries": W.tries, "fails": W.fails,
                            "true_team": log["teams"][oi]})
            except WorldError as exc:
                row.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}"})
            except Exception as exc:                 # a DEFECT (the guards included): recorded and printed, never silent
                row.update({"ok": False, "crash": True, "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                            "trace": traceback.format_exc()[-2000:]})
                print(f"[f2 rebuild] DEFECT {did}: {row['error'][:300]}", flush=True)
            fails = row.get("fails") or {}
            row["legality_touched"] = bool(any(k.startswith("illegal") for k in fails)
                                           or "illegal" in row.get("error", "") or row.get("crash"))
            row["cost"] = {"cpu_s": round(cpu_seconds() - c0, 3), "wall_s": round(time.monotonic() - w0, 3)}
            fo.write(json.dumps(row, sort_keys=True) + "\n")
            fo.flush()
            if (n_i + 1) % 10 == 0:
                print(f"[f2 rebuild] {n_i + 1}/{len(ids)} {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    print("[f2 rebuild] done", flush=True)


if __name__ == "__main__":
    main()
