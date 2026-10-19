"""PREREGISTRATION §3.3: the WORLDS, built at the HEAD checkout (no model, no playout, no outcome): Tier 1's / E2's
builder on this bank. ``main.value_search.worlds.WorldSampler(reveal=<spec>, max_tries=60)``, honest by default, the
world consistency guard inside every sample; FN = Tier 1's ``fn_worlds.fn_sampler`` over one anchor's belief rows
(`fn_belief_mat.py`). ``key`` = the decision id for every arm, so the seen-mon screen is deterministic per (decision,
constraint).

    run_mat.sh head 1 8 12h build_mat_worlds.py --shared F[,F+M,F+U] --mode lean [--limit N] [--bank interim|plateau]
    run_mat.sh head 1 8 12h build_mat_worlds.py --fn x26_025M --mode lean [--limit N]
    run_mat.sh head 1 8 2h  build_mat_worlds.py --validation oracle                       # the identity worlds

SHARED arms (F, and FULL's F+M / F+U) are network-free: built ONCE (``worlds/shared/<arm>/``) and played at every anchor,
so the anchors' F-based IMPs differ only by the network. FN is per anchor (``worlds/<label>/FN/``). COMMON RANDOM
NUMBERS: per state ONE ``screen_cache``; F is sampled first into it whenever a sibling (F+U, F+M, FN) is built, and when
F was already built in an earlier call its re-sample must reproduce the stored F teams exactly (a determinism TOOTH: a
mismatch is a DEFECT row for the sibling, never a silent skip).

One durable row per (state, arm) in the pilot's det format (``teams``, ``cmds``) plus the builder's provenance, its cost
and ``correct``. A ``WorldError`` / ``LookupError`` is an EXCLUSION (``ok`` false, the reason); any other exception (the
guard's ``WorldConsistencyError`` included) is a DEFECT: a ``crash`` row, printed, never silent. ``--limit`` caps NEW
work (states with an arm still to build).
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mat_common as E  # noqa: E402

E.add_paths()


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", default=None, help="comma list of shared arms (F, F+M, F+U, O-M, O-U)")
    ap.add_argument("--fn", default=None, help="an anchor label: build its FN worlds")
    ap.add_argument("--validation", default=None, help="an information-lever spec (e.g. oracle) for the identity check")
    ap.add_argument("--mode", choices=tuple(E.MODE_RANK), default="lean")
    ap.add_argument("--bank", default="interim", choices=sorted(E.BANKS))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    E.set_bank(a.bank)
    if a.validation and a.bank != "interim":
        raise SystemExit("the identity worlds are built on the interim bank only (A1)")
    if sum(x is not None for x in (a.shared, a.fn, a.validation)) != 1:
        raise SystemExit("name exactly one of --shared, --fn, --validation")
    E.check_head()
    from build_worlds import correct_counts, split_prefix   # Tier 1's helpers (pure; imported, never copied)

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search.worlds import WorldError, WorldSampler

    sha = os.environ["HIB_MAT_CHECKOUT_HEAD"]
    if not E.PILOT_MODE and not sha.startswith(E.BUILDER_SHA):
        raise SystemExit(f"the builder checkout is at {sha}, not the registered builder {E.BUILDER_SHA} (§12.13.7 (f))")
    bank = load_bank(E.bank_path())
    dec = {d["id"]: d for d in bank.decisions}
    f_sampler = WorldSampler(reveal="off", max_tries=60)
    beliefs: Dict[str, dict] = {}
    if a.validation:
        want = [f"V:{a.validation}"]
        samplers = {want[0]: WorldSampler(reveal=a.validation, max_tries=60)}
        ids = sorted(dec) if E.PILOT_MODE else E.identity_ids()
        label = None
    elif a.fn:
        if a.fn not in E.registered_ckpts():
            raise SystemExit(f"{a.fn}: not a registered anchor")
        import fn_worlds                                       # Tier 1's FN sampler (HEAD side)

        p = E.d("fn_belief") / f"{a.fn}.jsonl"
        beliefs = E.read_dir(p.parent, p.name)
        want = ["FN"]
        samplers = {}
        ids = [x for x in E.mode_ids(a.mode) if x in beliefs]          # a state builds once its belief row exists
        label = a.fn
    else:
        want = [x for x in E.SHARED_BUILD_ORDER if x in a.shared.split(",")]
        if set(a.shared.split(",")) - set(want):
            raise SystemExit(f"unknown shared arms {sorted(set(a.shared.split(',')) - set(want))}")
        samplers = {lab: (f_sampler if lab == "F" else WorldSampler(reveal=E.ARMS[lab][0], max_tries=60)) for lab in want}
        ids = E.mode_ids(a.mode)
        label = None
    done = {lab: E.read_dir(E.worlds_dir(label, lab), "worlds_*.jsonl") for lab in want}
    stored_f = E.read_dir(E.worlds_dir(None, "F"), "worlds_*.jsonl")
    todo_ids = [x for x in ids if any(x not in done[lab] for lab in want)]
    todo_ids = todo_ids[: a.limit] if a.limit else todo_ids          # --limit caps NEW work only
    files = {lab: open(E.worlds_dir(label, lab) / "worlds_0_of_1.jsonl", "a") for lab in want}
    print(f"[mat worlds] {len(todo_ids)} states x {want} at {sha[:8]}", flush=True)
    t0 = time.monotonic()
    try:
        for n_i, did in enumerate(todo_ids):
            d = dec[did]
            b = bank.battles[bank.battle_index[d["battle"]]]
            side = d["side"]
            at = cmd_index(b, side, d["n"])
            log = to_log(b)
            oi = 1 if side == "p1" else 0
            cache: Dict = {}
            f_teams = None
            siblings = [lab for lab in want if lab not in ("F",) and not lab.startswith("V:")]
            if siblings and not ("F" in want and did not in done["F"]):
                # F first into the shared cache (common random numbers); a stored F must be reproduced exactly
                try:
                    Wf = f_sampler.sample(log, at, side, E.J, key=did, screen_cache=cache)
                    f_teams = list(Wf.teams)
                except WorldError:
                    f_teams = None
            for lab in [x for x in want if did not in done[x]]:
                c0, w0 = cpu_seconds(), time.monotonic()
                row = {"id": did, "arm": lab, "cp": label, "head": sha, "at": at, "side": side}
                try:
                    sf = stored_f.get(did)
                    if lab != "F" and not lab.startswith("V:") and sf is not None and sf.get("ok"):
                        if f_teams != sf["teams"]:
                            raise RuntimeError("the F re-sample does not reproduce the stored F worlds (CRN broken)")
                    if lab == "FN":
                        bel = beliefs.get(did)
                        if bel is None:
                            raise LookupError(f"{did}: no network-belief row")
                        W = fn_worlds.fn_sampler(bel, max_tries=60).sample(log, at, side, E.J, key=did,
                                                                            screen_cache=cache)
                    else:
                        W = samplers[lab].sample(log, at, side, E.J, key=did, screen_cache=cache)
                    teams = [lg["teams"][oi] for lg in W.logs]
                    if teams != list(W.teams):
                        raise RuntimeError("Worlds.teams differ from the logs' opponent teams")
                    if lab == "F":
                        f_teams = teams
                    row.update({"ok": True, "teams": teams, "cmds": [split_prefix(lg, at) for lg in W.logs],
                                "tries": W.tries, "fails": W.fails, "screens": getattr(W, "screens", 0),
                                "leak": W.leak, "revealed": W.revealed, "spec": W.spec, "facts": W.facts,
                                "samplers": getattr(W, "samplers", []),
                                "correct": correct_counts(teams, log["teams"][oi], W.revealed)})
                    extra = getattr(W, "fn", None)
                    if extra:
                        row["fn"] = extra
                except (WorldError, LookupError) as exc:
                    row.update({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:600]}"})
                except Exception as exc:                 # a DEFECT (the guard's included), never a silent skip
                    row.update({"ok": False, "crash": True, "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                                "trace": traceback.format_exc()[-2000:]})
                row["cost"] = {"cpu_s": round(cpu_seconds() - c0, 3), "wall_s": round(time.monotonic() - w0, 3)}
                E.write_row(files[lab], row)
                if not row["ok"]:
                    print(f"[mat worlds] {lab} {'DEFECT' if row.get('crash') else 'EXCLUDED'} {did}: "
                          f"{row['error'][:200]}", flush=True)
            print(f"[mat worlds] {n_i + 1}/{len(todo_ids)} {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    finally:
        for f in files.values():
            f.close()


if __name__ == "__main__":
    main()
