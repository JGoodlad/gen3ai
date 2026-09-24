"""The M3 LOSS CATALOGUE: what the native record holds, and what the frozen ``TurnDelta``, the α/β
opponent-intent LABEL and the 22-column EVENT WINDOW keep or lose — per case, per 1,000 decisions,
over the MILESTONE corpus (the 2 x 360 seeded-random + 2 x 50 production-policy battles of
``rust_core_parity.MILESTONE_*_KEYS``, the same key recipe, so the same battles).

INCREMENTAL (SOP §2): one UNIT = one battle; each writes ONE durable row to ``rows.jsonl`` (appended,
flushed) with its per-case counts; a rerun SKIPS every key already written, so a kill loses at most
the battle in flight. Detached:

    nohup python catalogue.py run --out rows.jsonl > run.log 2>&1 < /dev/null &
    python catalogue.py report rows.jsonl          # the verdict, once every key is written

Every battle's core output is also held to slice T (the core's trackers + label == the Python
path's), so the label read here IS the label training builds; a battle with a divergence is
recorded as such and fails the report.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

#: The cases, in report order: id → one line of what happened in the battle.
CASES = {
    "multi_action_side": "a side has >= 2 actions in one decision window (an action then its replacement, a pass then its switch, a switch then a drag)",
    "multi_faint": ">= 2 faints in one window (a trade, Explosion, Destiny Bond, Perish Song, a residual KO after a KO)",
    "denied_fainted_first": "a turn actor fainted before its action (outsped and KOed, Explosion, a recoil trade)",
    "denied_turn_cut": "a turn actor that did NOT faint lost its action because a faint earlier in the turn CUT it (gen-3: any faint cancels every remaining queued action; in singles that is a faster SELF-KO — Explosion / Self-Destruct, recoil, confusion, Rough Skin)",
    "refused": "a chosen action refused (`cant`: par, slp, frz, flinch, recharge, Focus Punch, Taunt, Disable) and never taken",
    "baton_pass": "a Baton Pass entry (the receiver + the boosts / volatiles passed)",
    "called_move": "a called move (Sleep Talk / Mirror Move) — caller then called",
    "pursuit_on_switch": "Pursuit striking a switching target",
    "drag": "a drag (Roar / Whirlwind): a switch that was not the player's choice",
    "replacement": "a replacement switch after a faint (the free switch)",
    "hazard_ko": "a faint to Spikes on entry",
    "residual_damage": "damage / heal from a residual source (weather, poison, burn, Leech Seed, Leftovers, Wish …)",
    "item_transfer": "an item moved or removed (Thief / Covet / Trick / Knock Off)",
    "substitute_hit": "a Substitute took a hit",
    "protect_block": "a move LOST its target to Protect / Detect",
    "charge": "a charge turn (`-prepare`)",
    "recharge": "a recharge turn",
    "multi_hit": "a multi-hit move (>= 2 hits of one move)",
    "wish_heal": "a Wish heal resolving (the wisher named)",
    "item_via_trick": "an item line caused by Trick (both mons) — the event window's ITEM_TRANSITION reads REVEALED for each, never SWAPPED",
    "item_via_thief_covet": "an item line caused by Thief / Covet (the victim's reads SWAPPED, the thief's REVEALED)",
    "hazard_cleared": "a side condition ENDED by Rapid Spin — the event window's HAZARD row is identical to a hazard SET",
    "faint_destinybond": "a faint caused by Destiny Bond — the event window / TurnDelta faint-cause vocabulary has no such cause",
    "faint_perishsong": "a faint caused by Perish Song — likewise",
    "unboost": "a stat stage DROP (`|-unboost|`: Intimidate, Curse's Speed, Overheat, Screech …) — the event window's BOOST row reads it as a RISE (MAGNITUDE sign inverted)",
    "encore_same_turn": "an Encore landing on the opponent BEFORE it moves that turn — its executed move is the encored one, whatever it chose (the label names the executed move)",
    "clock_false_progress": "the progress clock's clause (i) inputs hold — TurnDelta's `our_damaging_event` set and `opp_target_hp_delta` <= -0.03 — while OUR move dealt NO direct damage to the opponent in this window (a status / failed / self move credited with sand, poison, recoil or any other chip)",
    "clock_false_progress_reset": "... and that clause ALONE reset the clock (the counterfactual without `our_damaging_event` does not reset) — the GIGO that reaches the obs's progress-clock scalar",
}

#: Label cases: the α/β label at the decision the window closes, when the OPPONENT's part of the
#: window is one of these.
LABEL_CASES = ("opp_denied_fainted_first", "opp_denied_turn_cut", "opp_refused", "opp_baton_pass", "opp_called_move", "opp_dragged",
               "opp_replacement_only")


def classify(window: List[dict], label: Optional[dict], trk: Optional[dict] = None) -> Dict[str, int]:
    """Per-case counts for ONE viewer decision window (the core's native record); ``trk`` is the
    decision's tracker state (slice T holds it equal to training's), for the clock / TurnDelta cases."""
    c: Dict[str, int] = collections.Counter()
    per_side: Dict[str, int] = collections.Counter()
    faints = 0
    opp = collections.Counter()
    for a in window:
        k = a["kind"]
        side = None
        if k == "move":
            side = a["user"][0]
            if not a.get("called_by"):
                per_side[side] += 1
            if a.get("called_by"):
                c["called_move"] += 1
                if side == "opp":
                    opp["opp_called_move"] += 1
            if a.get("pursuit_on_switch"):
                c["pursuit_on_switch"] += 1
            hits = sum(1 for e in a["effects"] if e[1] == "damage" and e[3][0] == "direct" and e[0] and e[0][0] != side)
            if hits >= 2:
                c["multi_hit"] += 1
            if a["id"] == "batonpass" and side == "opp":
                opp["opp_baton_pass"] += 1
        elif k == "switch":
            side = a["to"][0]
            per_side[side] += 1
            entry = a["entry"]
            if isinstance(entry, dict) and "batonpass" in entry:
                c["baton_pass"] += 1
            if isinstance(entry, dict) and "replacement" in entry:
                c["replacement"] += 1
                if side == "opp":
                    opp["opp_replacement"] += 1
        elif k == "drag":
            side = a["to"][0]
            per_side[side] += 1
            c["drag"] += 1
            if side == "opp":
                opp["opp_dragged"] += 1
        elif k == "cant":
            side = a["mon"][0]
            if not a.get("then_moved"):
                per_side[side] += 1
                c["refused"] += 1
                if side == "opp":
                    opp["opp_refused"] += 1
        elif k == "denied":
            side = a["actor"][0]
            why = a.get("why", "fainted_first")
            c[f"denied_{why}"] += 1
            if why == "turn_cut":
                c[f"turn_cut_cause:{a['cause'][0]}"] += 1
            if side == "opp":
                opp[f"opp_denied_{why}"] += 1
        for e in a["effects"]:
            what, cause = e[1], e[3][0]
            if what == "faint":
                faints += 1
                if cause == "spikes":
                    c["hazard_ko"] += 1
                if cause in ("destinybond", "perishsong"):
                    c[f"faint_{cause}"] += 1
            if what == "heal" and cause == "wish":
                c["wish_heal"] += 1
            if what == "item" and e[3] == ["move", "Trick"]:
                c["item_via_trick"] += 1
            if what == "item" and e[3][0] == "move" and e[3][1] in ("Thief", "Covet"):
                c["item_via_thief_covet"] += 1
            if what == "side_end" and e[3] == ["move", "Rapid Spin"]:
                c["hazard_cleared"] += 1
            if what == "boost" and isinstance(e[2], list) and len(e[2]) == 2 and e[2][1] < 0:
                c["unboost"] += 1
            elif what in ("damage", "heal") and cause in ("weather", "status", "leechseed", "curse", "nightmare", "wish") \
                    or (what == "heal" and cause == "item"):
                c["residual_damage"] += 1
            elif what == "item" and e[2][1] and any(w in e[2][2] for w in ("Thief", "Covet", "Trick", "Knock Off")):
                c["item_transfer"] += 1
            elif what == "substitute":
                c["substitute_hit"] += 1
            elif what == "blocked":
                c["protect_block"] += 1
            elif what == "prepare":
                c["charge"] += 1
        if k == "cant" and a.get("reason") == "recharge":
            c["recharge"] += 1
    if any(n >= 2 for n in per_side.values()):
        c["multi_action_side"] += 1
    for i, a in enumerate(window):
        if a["kind"] == "move" and a["user"][0] == "ours" and any(
                e[1] == "volatile_start" and e[2] == "Encore" and e[0] and e[0][0] == "opp" for e in a["effects"]):
            later = [b for b in window[i + 1:] if b["kind"] == "move" and b["user"][0] == "opp" and not b.get("called_by")]
            if later:
                c["encore_same_turn"] += 1
                if label is not None:
                    c[f"label:opp_encored_same_turn:{label['kind']}"] += 1
                    if label["kind"] == 0 and label.get("move_id") == later[0]["id"]:
                        c["label:opp_encored_same_turn:names_the_encored_move"] += 1
    d = (trk or {}).get("delta")
    if d and d.get("our_damaging") and d.get("opp_target_hp_delta") is not None and d["opp_target_hp_delta"] <= -0.03:
        ours_hit = any(a["kind"] == "move" and a["user"][0] == "ours" and any(
            e[1] == "damage" and e[0] and e[0][0] == "opp" and e[3][0] == "direct" for e in a["effects"]) for a in window)
        if not ours_hit:
            c["clock_false_progress"] += 1
            if trk.get("clause_i_decisive"):
                c["clock_false_progress_reset"] += 1
    if faints >= 2:
        c["multi_faint"] += 1
    # the label at this decision, per opponent case (kind 0 MOVE / 1 SWITCH / 2 UNKNOWN = masked)
    if label is not None:
        only_repl = opp["opp_replacement"] and not any(opp[x] for x in ("opp_denied_fainted_first", "opp_denied_turn_cut", "opp_refused", "opp_dragged"))
        cases = [x for x in LABEL_CASES if opp[x]] + (["opp_replacement_only"] if only_repl else [])
        for x in cases:
            c[f"label:{x}:{label['kind']}"] += 1
            if x == "opp_called_move" and label["kind"] == 0:
                called = [a for a in window if a["kind"] == "move" and a["user"][0] == "opp" and a.get("called_by")]
                c[f"label:{x}:labels_the_{'called' if called and label['move_id'] == called[-1]['id'] else 'caller'}"] += 1
    return c


def units():
    """(kind, key) — the MILESTONE corpus, in its manifest order."""
    from agents.battle import rust_core_parity as P

    for r in P.MILESTONE_RANDOM_KEYS:
        for k in r:
            yield "random", k
    for r in P.MILESTONE_POLICY_KEYS:
        for k in r:
            yield "policy", k


def run_unit(kind: str, key: int, policy) -> dict:
    from agents.battle import rust_core_parity as P

    lv = P.play(key, tag="Cat", policy=policy if kind == "policy" else None)
    decisions, counts, t = census_battle(lv.recorded)
    boundary = sum(n for k, n in t.divergences.items() if k.startswith("[BOUNDARY]"))
    return {"kind": kind, "key": key, "decisions": decisions, "counts": dict(counts), "boundary_violations": boundary,
            "slice_t_divergences": sum(t.divergences.values()), "slice_t_decisions": t.decisions}


def census_battle(recorded):
    """One recorded battle through the core + slice T: (decisions, per-case counts, the census)."""
    import dataclasses

    from agents.battle import rust_core_parity as P
    from agents.battle import rust_core_parity_trackers as T
    from agents.training.progress_clock import ProgressClock

    t = T.TrackerCensus()
    counts: Dict[str, int] = collections.Counter()
    decisions = 0
    results: list = []
    # The progress clock's clause (i) — "our move dealt damage" — read as a COUNTERFACTUAL on the
    # Python path slice T drives: at every progress verdict, re-ask with `our_damaging_event`
    # removed; `decisive` = the reset happened ONLY because of clause (i). Read-only wrappers (each
    # returns the real result), restored in `finally`.
    real_is_progress = ProgressClock.__dict__["_is_progress"].__func__
    real_state = T.tracker_state
    decisive: dict = {}
    alive: list = []
    order: list = []

    def is_progress(delta, live, *a, **k):
        r = real_is_progress(delta, live, *a, **k)
        if r and getattr(delta, "our_damaging_event", None) is not None:
            decisive[id(delta)] = not real_is_progress(dataclasses.replace(delta, our_damaging_event=None), live, *a, **k)
            alive.append(delta)
        return r

    def tracker_state(tr, delta, label, battle):
        order.append(bool(delta is not None and decisive.get(id(delta))))
        return real_state(tr, delta, label, battle)

    ProgressClock._is_progress = staticmethod(is_progress)
    T.tracker_state = tracker_state
    try:
        P.check_battles([recorded], P.Census(), trackers=t, on_result=lambda _b, res: results.append(res))
    finally:
        ProgressClock._is_progress = staticmethod(real_is_progress)
        T.tracker_state = real_state
    i = 0
    for res in results:
        for viewer in res.get("trackers") or []:
            for cap in viewer:
                if "trackers" not in cap:
                    continue
                flag = order[i] if i < len(order) else None
                i += 1
                if "window" not in cap:
                    continue
                decisions += 1
                trk = dict(cap["trackers"], clause_i_decisive=flag)
                counts.update(classify(cap["window"], cap["trackers"]["label"], trk))
    if i != len(order):
        counts["[align] clause-i flags vs decisions mismatch"] += 1
    return decisions, counts, t


def cmd_run(out: str, limit: Optional[int]) -> int:
    from agents.battle import rust_core_parity as P

    logging.getLogger("poke-env").setLevel(logging.ERROR)
    P.check_manifest()
    done = set()
    if os.path.exists(out):
        for line in open(out):
            if line.strip():
                r = json.loads(line)
                done.add((r["kind"], r["key"]))
    todo = [u for u in units() if u not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(done)} units already written, {len(todo)} to run", flush=True)
    policy = P.load_production_policy() if any(k == "policy" for k, _ in todo) else None
    with open(out, "a") as fh:
        for i, (kind, key) in enumerate(todo):
            t0 = time.time()
            row = run_unit(kind, key, policy)
            row["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if i % 20 == 0:
                print(f"  {i + 1}/{len(todo)} {kind}_{key} {row['decisions']} decisions {row['seconds']} s", flush=True)
    return 0


def cmd_report(rows_path: str) -> int:
    rows = [json.loads(line) for line in open(rows_path) if line.strip()]
    want = set(units())
    have = {(r["kind"], r["key"]) for r in rows}
    missing = want - have
    dec = sum(r["decisions"] for r in rows)
    tot: Dict[str, int] = collections.Counter()
    for r in rows:
        tot.update(r["counts"])
    div = sum(r["slice_t_divergences"] for r in rows)
    bnd = sum(r.get("boundary_violations", 0) for r in rows)
    print(f"units {len(have)}/{len(want)} (missing {len(missing)}); decisions {dec}; slice-T divergences {div} "
          f"(of which information-boundary violations {bnd})")
    print("\ncase | per 1,000 decisions | count")
    for k, desc in CASES.items():
        print(f"{k:22s} {1000 * tot[k] / max(dec, 1):8.2f}  {tot[k]:7d}   {desc}")
    causes = sorted(k for k in tot if k.startswith("turn_cut_cause:"))
    print("turn-cut causes (the first faint's cause):", {k.split(":", 1)[1]: tot[k] for k in causes})
    print("\nlabel outcomes (kind 0 MOVE / 1 SWITCH / 2 UNKNOWN=masked), per opponent case:")
    enc = {kk: tot[f"label:opp_encored_same_turn:{kk}"] for kk in (0, 1, 2)}
    print(f"  {'opp_encored_same_turn':28s} {enc}  (MOVE labels naming the ENCORED move {tot['label:opp_encored_same_turn:names_the_encored_move']})")
    for x in LABEL_CASES:
        parts = {kk: tot[f"label:{x}:{kk}"] for kk in (0, 1, 2)}
        extra = ""
        if x == "opp_called_move":
            extra = f"  (MOVE labels naming the called move {tot['label:opp_called_move:labels_the_called']}, the caller {tot['label:opp_called_move:labels_the_caller']})"
        print(f"  {x:28s} {parts}{extra}")
    return 0 if (not missing and div == 0) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "rows.jsonl"))
    r.add_argument("--limit", type=int, default=None, help="run at most N units (the kill+resume proof)")
    rep = sub.add_parser("report")
    rep.add_argument("rows")
    a = ap.parse_args(argv)
    return cmd_run(a.out, a.limit) if a.cmd == "run" else cmd_report(a.rows)


if __name__ == "__main__":
    sys.exit(main())
