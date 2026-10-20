"""STAGE 1, BUILDER SIDE: every arm's WORLDS for a list of bank states (no model, no playout, no outcome). Runs at the
BUILDER checkout (``run_s1.sh builder``). Per state ONE screen cache, the arms in ``s1_common.ARMS`` order (F_legal
first: every seed-dice sibling takes its seen-mon screen; F_rolls keys its own by the dice mode); ``key`` = the
decision id for every arm (the builder's draws are deterministic per (decision, arm)).

    run_s1.sh builder 1 8 6h build_s1.py --ids <id,id,...> | --smoke N

One fsync'd row per (state, arm) in ``<archive>/worlds/<arm>/worlds.jsonl`` (Tier 1's det format: ``teams``, ``cmds``,
plus ``rolls`` / ``log_w`` for F_rolls, the arm's proposal report ``prop``, the cost). A ``WorldError`` (POOL's
``PoolExcluded`` included) is an EXCLUSION (``ok`` false); any other exception is a DEFECT (``crash`` true), printed,
never silent. A state already holding a row for an arm is skipped for that arm (resumable).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import resource
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_common as E  # noqa: E402

for _p in (E.T1, E.PILOT, E.BRC):
    if str(_p) not in sys.path:
        sys.path.insert(1, str(_p))


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def check_builder() -> str:
    if os.environ.get("WS1_KIND") != "builder":
        raise SystemExit("worlds are built at the BUILDER checkout only (run_s1.sh builder)")
    import main

    if not str(Path(main.__file__).resolve()).startswith(os.environ["WS1_CHECKOUT_DIR"]):
        raise SystemExit(f"main imported from {main.__file__}, not the builder checkout")
    return os.environ["WS1_CHECKOUT_HEAD"]


def smoke_ids(n: int):
    """ENGINEERING SMOKE states, OUTSIDE the 287: the interim LEAN ids whose BANKED F build was an exclusion (they hold
    225M roots and truth rows); the first ``n`` in sha256 order. Never read as stage-1 output."""
    st = E.mat_states()
    lean = {k for k, v in st.items() if v["lean"]}
    common = set(E.common_ids())
    out = sorted(lean - common, key=lambda i: hashlib.sha256(("ws_stage1_smoke_v1:" + i).encode()).hexdigest())
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=None, help="comma list of decision ids (in the registered order)")
    ap.add_argument("--smoke", type=int, default=0, help="N engineering-smoke states outside the 287 (separate archive)")
    ap.add_argument("--arms", default=",".join(E.ARMS))
    a = ap.parse_args()
    sha = check_builder()
    import json

    from build_worlds import correct_counts, split_prefix   # Tier 1's helpers (imported, never copied)
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search.worlds import WorldError

    import samplers_s1 as SS

    if a.smoke:
        if not str(E.ARCH).endswith("_smoke"):
            raise SystemExit("the engineering smoke writes only to a *_smoke archive (WS1_ARCHIVE)")
        ids = smoke_ids(a.smoke)
    else:
        ids = [x for x in (a.ids or "").split(",") if x]
        allowed = set(E.common_ids())
        if any(x not in allowed for x in ids):
            raise SystemExit("an id outside the 287 registered states")
    arms = [x for x in a.arms.split(",") if x]
    if any(x not in E.ARMS for x in arms):
        raise SystemExit(f"unknown arm in {arms}")
    fit = json.loads((E.HERE / "alpha_fit.json").read_text())
    beliefs = {r["id"]: r for r in E.read_rows(E.MAT_ARCH / "fn_belief" / f"{E.ANCHOR}.jsonl")}
    bank = load_bank(E.MAT_ARCH / "bank")
    dec = {d["id"]: d for d in bank.decisions}
    done = {arm: E.read_by_id(E.worlds_path(arm)) for arm in arms}
    todo = [x for x in ids if any(x not in done[arm] for arm in arms)]
    print(f"[s1 build] {len(todo)} states x {len(arms)} arms at {sha[:8]} (alpha {fit['ALPHA']})", flush=True)
    if not todo:
        return
    t_load = time.monotonic()
    A = SS.Arms(alpha=fit["ALPHA"], beliefs=beliefs)
    print(f"[s1 build] loaded in {time.monotonic() - t_load:.0f} s (ladder {len(A.ladder.teams)}, forum "
          f"{len(A.forum.teams)})", flush=True)
    files = {arm: open(E.worlds_path(arm), "a") for arm in arms}
    t0 = time.monotonic()
    try:
        for n_i, did in enumerate(todo):
            d = dec[did]
            b = bank.battles[bank.battle_index[d["battle"]]]
            side = d["side"]
            at = cmd_index(b, side, d["n"])
            log = to_log(b)
            oi = 1 if side == "p1" else 0
            cache: dict = {}
            for arm in arms:
                if did in done[arm]:
                    continue
                c0, w0 = cpu_seconds(), time.monotonic()
                row = {"id": did, "arm": arm, "head": sha, "at": at, "side": side, "smoke": bool(a.smoke)}
                try:
                    W, rep = A.sample(arm, log, at, side, did, cache)
                    teams = [lg["teams"][oi] for lg in W.logs]
                    if teams != list(W.teams):
                        raise RuntimeError("Worlds.teams differ from the logs' opponent teams")
                    rolls = [list(lg.get("rolls") or []) for lg in W.logs]
                    if arm != "F_rolls" and any(rolls):
                        raise RuntimeError(f"{arm}: a seed-dice world carries rolls")
                    if arm == "F_rolls" and (W.dice != "rolls" or len(W.log_w) != E.J):
                        raise RuntimeError("F_rolls returned worlds without their weights")
                    row.update({"ok": True, "teams": teams, "cmds": [split_prefix(lg, at) for lg in W.logs],
                                "rolls": rolls, "log_w": [float(x) for x in (W.log_w or [])], "dice": W.dice,
                                "tries": W.tries, "fails": W.fails, "screens": getattr(W, "screens", 0),
                                "leak": W.leak, "revealed": W.revealed, "spec": W.spec, "facts": W.facts,
                                "prop": rep, "pool": getattr(W, "pool_meta", None),
                                "correct": correct_counts(teams, log["teams"][oi], W.revealed)})
                except WorldError as exc:
                    row.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                                "pool": getattr(exc, "meta", None)})
                except Exception as exc:                 # a DEFECT (a guard included): recorded and printed
                    row.update({"ok": False, "crash": True, "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                                "trace": traceback.format_exc()[-2000:]})
                row["cost"] = {"cpu_s": round(cpu_seconds() - c0, 3), "wall_s": round(time.monotonic() - w0, 3)}
                E.write_row(files[arm], row)
                if not row["ok"]:
                    print(f"[s1 build] {arm} {'DEFECT' if row.get('crash') else 'EXCLUDED'} {did}: "
                          f"{row['error'][:200]}", flush=True)
            print(f"[s1 build] {n_i + 1}/{len(todo)} {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    finally:
        for f in files.values():
            f.close()


if __name__ == "__main__":
    main()
