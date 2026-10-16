"""E2 §2: the WORLDS of every arm at every E2 state, built at the E2 HEAD (no model, no playout, no outcome) — Tier 1's
`build_worlds.py` on this bank: `main.value_search.worlds.WorldSampler(reveal=<spec>, max_tries=60)`, honest by default,
the world consistency guard (`world_guard.check_worlds`) inside every sample. ``key`` = the decision id for every arm, so
the seen-mon screen is deterministic per (decision, constraint) and F+U takes F's seen-mon draws (one shared
``screen_cache`` per state: common random numbers).

    run_e2.sh head 1 8 24h build_e2_worlds.py --cp end2 [--arms F,F+U,...] [--shard i --of W] [--limit N] [--ids FILE]
    run_e2.sh head 1 8 2h  build_e2_worlds.py --cp end2 --validation oracle --ids FILE     # the identity check

One durable row per (state, arm) in ``<archive>/<cp>/worlds/<arm>/worlds_<shard>_of_<W>.jsonl`` in the pilot's det
format (``teams``, ``cmds``) plus the builder's provenance, its cost and ``correct`` (Tier 1's `correct_counts`). A
``WorldError`` is an EXCLUSION (``ok`` false, the reason). Any other exception — a ``WorldConsistencyError`` from the
guard included — is a DEFECT: a ``crash`` row (PREREGISTRATION §5), printed, never silent.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_common as E  # noqa: E402

E.add_paths()


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", required=True, choices=sorted(E.CPS))
    ap.add_argument("--arms", default=None, help="comma list of labels (default: every E2 arm)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="a file of decision ids (one per line) to restrict to")
    ap.add_argument("--validation", default=None, help="a VALIDATION build (not an arm): an information-lever spec "
                                                         "(e.g. oracle), written under the arm dir V_<spec>")
    a = ap.parse_args()
    if os.environ.get("HIB_E2_KIND") != "head":
        raise SystemExit("worlds are built at the E2 HEAD only (run_e2.sh head)")
    import main as _m

    if not str(Path(_m.__file__).resolve()).startswith(os.environ["HIB_E2_CHECKOUT_DIR"]):
        raise SystemExit(f"main imported from {_m.__file__}, not the E2 HEAD checkout")
    from build_worlds import correct_counts, split_prefix   # Tier 1's helpers (pure; imported, never copied)

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search.worlds import WorldError, WorldSampler

    if a.validation:
        labels = {f"V:{a.validation}": a.validation}
        want = list(labels)
        dirs = {lab: "V_" + E.arm_dir(a.validation) for lab in want}
    else:
        labels = dict(E.ARMS)
        want = [x for x in E.BUILD_ORDER if a.arms is None or x in a.arms.split(",")]
        if a.arms is not None and set(a.arms.split(",")) - set(want):
            raise SystemExit(f"unknown arms {sorted(set(a.arms.split(',')) - set(want))}")
        dirs = {lab: E.arm_dir(lab) for lab in want}
    bank = load_bank(E.bank_path(a.cp))
    doc = json.loads(E.states_path(a.cp).read_text())
    if bank.manifest["content_sha256"] != doc["bank_content_sha256"]:
        raise SystemExit("the bank is not the one states_e2_<cp>.json describes")
    ids = sorted(d["id"] for d in bank.decisions)
    if a.ids:
        keep = {x.strip() for x in Path(a.ids).read_text().splitlines() if x.strip()}
        ids = [x for x in ids if x in keep]
    ids = E.shard(ids, a.shard, a.of)[: a.limit]
    samplers = {lab: WorldSampler(reveal=labels[lab], max_tries=60) for lab in want}
    done: Dict[str, set] = {}
    files = {}
    for lab in want:
        d = E.cp_dir(a.cp) / "worlds" / dirs[lab]
        d.mkdir(parents=True, exist_ok=True)
        done[lab] = set(E.read_dir(d, "worlds_*.jsonl"))
        files[lab] = open(d / f"worlds_{a.shard}_of_{a.of}.jsonl", "a")
    sha = os.environ["HIB_E2_CHECKOUT_HEAD"]
    dec = {d["id"]: d for d in bank.decisions}
    print(f"[e2 worlds] {a.cp} shard {a.shard}/{a.of}: {len(ids)} states x {want} at {sha[:8]}", flush=True)
    t0 = time.monotonic()
    try:
        for n_i, did in enumerate(ids):
            todo = [lab for lab in want if did not in done[lab]]
            if not todo:
                continue
            d = dec[did]
            b = bank.battles[bank.battle_index[d["battle"]]]
            side = d["side"]
            at = cmd_index(b, side, d["n"])
            log = to_log(b)
            oi = 1 if side == "p1" else 0
            screen_cache: Dict = {}                      # F's seen-mon screen, shared by F+U (common random numbers)
            for lab in todo:
                c0, w0 = cpu_seconds(), time.monotonic()
                row = {"id": did, "arm": lab, "cp": a.cp, "head": sha, "at": at, "side": side}
                try:
                    W = samplers[lab].sample(log, at, side, E.J, key=did, screen_cache=screen_cache)
                    teams = [lg["teams"][oi] for lg in W.logs]
                    if teams != list(W.teams):
                        raise RuntimeError("Worlds.teams differ from the logs' opponent teams")
                    row.update({"ok": True, "teams": teams, "cmds": [split_prefix(lg, at) for lg in W.logs],
                                "tries": W.tries, "fails": W.fails, "screens": getattr(W, "screens", 0),
                                "leak": W.leak, "revealed": W.revealed, "spec": W.spec, "facts": W.facts,
                                "samplers": getattr(W, "samplers", []),
                                "correct": correct_counts(teams, log["teams"][oi], W.revealed)})
                except WorldError as exc:
                    row.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}"})
                except Exception as exc:                 # a DEFECT (the guard's included), never a silent skip
                    row.update({"ok": False, "crash": True, "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                                "trace": traceback.format_exc()[-2000:]})
                row["cost"] = {"cpu_s": round(cpu_seconds() - c0, 3), "wall_s": round(time.monotonic() - w0, 3)}
                E.write_row(files[lab], row)
                if not row["ok"]:
                    print(f"[e2 worlds] {lab} {'DEFECT' if row.get('crash') else 'EXCLUDED'} {did}: "
                          f"{row['error'][:200]}", flush=True)
            el = time.monotonic() - t0
            print(f"[e2 worlds] {n_i + 1}/{len(ids)} {el / 60:.1f} min", flush=True)
    finally:
        for f in files.values():
            f.close()


if __name__ == "__main__":
    main()
