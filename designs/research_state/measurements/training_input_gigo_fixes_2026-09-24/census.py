"""The diff CENSUS for the training-input GIGO fixes (`gen3_event_window_semantics_fixes_v1`,
`gen3_intent_label_semantics_fixes_v1`, `gen3_progress_clock_attribution_fix_v1`) — the goldens are
PROVEN, never regenerated blind.

Two captures of the same deterministic battles, one from the BASE tree and one from the FIX tree
(``run_census.sh`` points ``PYTHONPATH`` at each; nothing else differs), then a value-aware DIFF:
every changed event-window row, α/β label, progress-clock value and obs cell is resolved to its
cause and checked against the EVENT LOG the row was built from. Anything the fixes do not explain
fails the census.

Captures (each writes one durable JSON line per battle; resumable — a re-run skips done battles):

    golden  OUT.jsonl [N_BATTLES N_TEAMS]   golden_obs_capture's fixed set (the obs vectors too)
    corpus  OUT.jsonl SOURCE KEYS…          slice T's corpora: `commit`, or `random|policy|ladder`
                                            with battle keys (`a:b[:step]` ranges)

    diff    BASE.jsonl FIX.jsonl            the census; exit 1 on any UNEXPLAINED change

What the diff accepts, and nothing else:

* event-window ROW keys — `hp_delta` of a BOOST row: sign flipped on an `|-unboost|` (W1); of a
  HAZARD row: 0 → +1 / −1 by `sidestart` / `sideend` (W5); of a MOVE row: the removed amount equals
  the bare `-damage` on its target landed while the OTHER side was moving (T1's class, on the row);
  `failed` of a MOVE row: False → True with a Protect / Detect `-activate` on the other side before
  the next move (W4); `faint_cause`: → "other" where the mon's last damage since entry left it
  above 0 HP or there was none (W2); `item_tr`: REVEALED → SWAPPED on an `|-item|` [from] a
  transfer move (W3);
* the LABEL — SWITCH → masked with `opp_dragged` (L1 / L2) or `opp_switch_is_replacement` (L3);
  MOVE x → MOVE caller with `opp_called_via` (L4); → masked with `opp_choice_overridden` (L5);
* the CLOCK — a decision whose transition CLASS (reset / stay / increment) differs must hold the T1
  condition (our damaging event, the target's net fall ≥ 3 %, our own hit < 3 %) or the T2 one
  (base outcome "hit", fix outcome "fail" against Protect / Detect / Endure); `n` may then differ
  downstream (the cascade) until the two trajectories re-join;
* OBS cells (golden only) — inside the event-window block, a cell whose row's changed key is one of
  the above (MAGNITUDE ↔ hp_delta, OUT_HIT / OUT_FAIL ↔ failed, FAINT_CAUSE, ITEM_TRANSITION); the
  clock scalar `reactive[2]` where the two `n` differ. Any other obs index is unexplained.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

EPS = 0.03
_TRANSFER = ("trick", "thief", "covet", "switcheroo")
_BLOCK = ("protect", "detect", "move: protect", "move: detect")
_STALL = ("protect", "detect", "endure")


# ------------------------------------------------------------------------------ the capture hooks

class _Cap:
    """Per EventWindowTracker: every appended row (final values), stamped with the seq of the event
    that appended it, and every event it folded (compact)."""

    by_tracker: Dict[int, dict] = {}

    @classmethod
    def of(cls, ew) -> dict:
        return cls.by_tracker.setdefault(id(ew), {"rows": [], "events": {}, "decisions": [], "cur": None})


def _compact(e) -> dict:
    v = e.value or {}
    return {"seq": e.seq, "turn": e.turn, "kind": e.kind.name, "side": e.side, "actor": e.actor_species,
            "move_id": v.get("move_id"), "from": e.from_clause, "amount": v.get("amount"),
            "hp_after": v.get("hp_after"), "effect": v.get("effect") or v.get("condition"), "op": v.get("op")}


def install_hooks() -> None:
    from agents.training.episode_tracker import EventWindowTracker
    from agents.training.progress_clock import ProgressClock

    real_update, real_append = EventWindowTracker.update, EventWindowTracker._append

    def update(self, turn, events, our_active, opp_active):
        cap = _Cap.of(self)
        for e in events or []:
            cap["events"].setdefault(e.seq, _compact(e))
            cap["cur"] = e.seq
            real_update(self, turn, [e], None, None)
        cap["cur"] = None
        real_update(self, turn, [], our_active, opp_active)

    def _append(self, rec):
        cap = _Cap.of(self)
        rec["_seq"] = cap["cur"]
        cap["rows"].append(rec)
        return real_append(self, rec)

    EventWindowTracker.update, EventWindowTracker._append = update, _append

    real_clock = ProgressClock.update

    def clock_update(self, delta, live, legal, legal_prev=None):
        real_clock(self, delta, live, legal, legal_prev)
        self._census_delta = _delta_fields(delta)

    ProgressClock.update = clock_update


def _delta_fields(d) -> dict:
    if d is None:
        return {}
    g = lambda k, dflt=None: getattr(d, k, dflt)  # noqa: E731
    return {"our_damaging": g("our_damaging_event") is not None,
            "tgt": None if g("opp_target_hp_delta") is None else float(g("opp_target_hp_delta")),
            "hit": float(g("our_move_hit_delta", 0.0) or 0.0), "outcome": g("our_move_outcome"),
            "opp_mv": g("opp_resolved_move_id"), "forced": bool(g("phase_is_forced_switch")),
            "opp_dragged": bool(g("opp_dragged", False)), "opp_repl": bool(g("opp_switch_is_replacement", False)),
            "opp_called_via": g("opp_called_via"), "opp_over": bool(g("opp_choice_overridden", False)),
            "opp_switch_to": g("opp_switch_to")}


def _row_out(r: dict) -> dict:
    return {k: (float(v) if isinstance(v, float) else v) for k, v in r.items()
            if k in ("t", "actor", "side", "target", "move_id", "hp_delta", "missed", "failed", "crit", "eff",
                     "turn", "faint_cause", "item_tr", "cant", "_seq")}


def _flush(label: str, trackers: List[Any], out_f) -> None:
    """One JSON line: per tracker (in creation order), rows / events / decisions."""
    rec = {"battle": label, "trackers": []}
    for tr in trackers:
        cap = _Cap.by_tracker.pop(id(tr.event_window), {"rows": [], "events": {}, "decisions": []})
        rec["trackers"].append({"rows": [_row_out(r) for r in cap["rows"]],
                                "events": sorted(cap["events"].values(), key=lambda e: e["seq"]),
                                "decisions": cap["decisions"]})
    out_f.write(json.dumps(rec) + "\n")
    out_f.flush()


def _done(out: str) -> set:
    if not os.path.exists(out):
        return set()
    return {json.loads(line)["battle"] for line in open(out) if line.strip()}


# ------------------------------------------------------------------------------ capture: golden

def capture_golden(out: str, n_battles: str = "", n_teams: str = "") -> None:
    import asyncio

    import numpy as np
    from poke_env import AccountConfiguration
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

    from agents.battle.gen3_battle import Gen3Battle
    from agents.observation.constants import EVENT_WINDOW_N, OFFSET_EVENT_WINDOW, OFFSET_REACTIVE
    from agents.training import golden_obs_capture as G
    from utils.bridge.local_battle_runner import run_local_battles
    from utils.team_loader import TeamLoader

    install_hooks()
    if n_battles:
        G.N_BATTLES, G.N_TEAMS = int(n_battles), int(n_teams)
    trackers: List[Any] = []
    real_choose = G._DetPlayer.choose_move

    def choose_move(self, battle):
        order = real_choose(self, battle)
        if self._record:
            tr = self._tracker
            if not trackers or trackers[-1] is not tr:
                trackers.append(tr)
            cap = _Cap.of(tr.event_window)
            vec = self.vectors[-1]
            cap["decisions"].append({
                "n": tr.progress_clock.n, "rows_total": len(cap["rows"]),
                "delta": getattr(tr.progress_clock, "_census_delta", {}),
                "clock_cell": float(vec[OFFSET_REACTIVE + 2]),
                "ew": [float(x) for x in vec[OFFSET_EVENT_WINDOW:OFFSET_EVENT_WINDOW + EVENT_WINDOW_N * 22]],
                "rest": None})
            cap["decisions"][-1]["_vec"] = vec
        return order

    G._DetPlayer.choose_move = choose_move

    async def _cap():
        from agents.observation import moves as _moves_enc
        _moves_enc._CATEGORY_VAL_CACHE.clear()
        pool = (TeamLoader().get_sample_teams() or TeamLoader().get_all_teams())[:G.N_TEAMS]
        p1 = G._DetPlayer(record=True, battle_format=G.BATTLE_FORMAT, team=G._CyclingTeambuilder(pool),
                          account_configuration=AccountConfiguration("GoldCap", "pw"),
                          server_configuration=LocalhostServerConfiguration, start_listening=False,
                          battle_class=Gen3Battle)
        p2 = G._DetPlayer(battle_format=G.BATTLE_FORMAT, team=G._CyclingTeambuilder(pool[1:] + pool[:1]),
                          account_configuration=AccountConfiguration("GoldOpp", "pw"),
                          server_configuration=LocalhostServerConfiguration, start_listening=False,
                          battle_class=Gen3Battle)
        await run_local_battles(p1, p2, G.N_BATTLES, seed=G.BRIDGE_SEED)
        return p1.vectors

    vecs = asyncio.run(_cap())
    hashes = G.vector_hashes(vecs)
    with open(out, "w") as f:
        for i, tr in enumerate(trackers):
            cap = _Cap.of(tr.event_window)
            for d in cap["decisions"]:
                v = d.pop("_vec")
                d["rest"] = [float(x) for x in np.concatenate([v[:OFFSET_EVENT_WINDOW],
                                                               v[OFFSET_EVENT_WINDOW + EVENT_WINDOW_N * 22:]])]
            _flush(f"golden.{i}", [tr], f)
        f.write(json.dumps({"battle": "__hashes__", "hashes": hashes}) + "\n")
    print(f"captured {len(vecs)} decisions over {len(trackers)} battles -> {out}")


# ------------------------------------------------------------------------------ capture: corpus

def _keys(specs: List[str]) -> List[int]:
    out: List[int] = []
    for s in specs:
        p = [int(x) for x in s.split(":")]
        out.extend(range(p[0], p[1], p[2] if len(p) > 2 else 1) if len(p) > 1 else [p[0]])
    return out


def capture_corpus(out: str, source: str, *specs: str) -> None:
    from agents.battle import rust_core_parity as P
    from agents.battle import rust_core_parity_trackers as T

    logging.getLogger("poke-env").setLevel(logging.ERROR)
    install_hooks()
    done = _done(out)
    created: List[Any] = []
    from agents.training import episode_tracker as ET
    real_init = ET.EpisodeTracker.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        created.append(self)

    ET.EpisodeTracker.__init__ = init
    real_state = T.tracker_state

    def tracker_state(tr, delta, label, battle):
        s = real_state(tr, delta, label, battle)
        cap = _Cap.of(tr.event_window)
        cap["decisions"].append({"n": tr.progress_clock.n, "rows_total": len(cap["rows"]),
                                 "delta": _delta_fields(delta),
                                 "label": [label["kind"], label["move_id"], label["switch_species"]]})
        return s

    T.tracker_state = tracker_state
    if source == "commit":
        battles = [(b.label, b) for b in P.commit_corpus()]
    else:
        pol = P.load_production_policy() if source == "policy" else None
        battles = [(f"{source}.{k}", k) for k in _keys(list(specs))]
    with open(out, "a") as f:
        for label, b in battles:
            if label in done:
                continue
            created.clear()
            if not isinstance(b, P.RecordedBattle):
                lv = P.play(b, policy=pol, source=("ladder" if source == "ladder" else None))
                b = lv.recorded
            t = T.TrackerCensus()
            P.check_battles([b], P.Census(), trackers=t)
            if t.divergences or t.refused:
                raise SystemExit(f"{label}: slice T is not clean — the census reads a tree that is "
                                 f"not in parity:\n{t.render()}")
            _flush(label, list(created), f)
    print(f"corpus {source}: {len(battles)} battles -> {out}")


# ------------------------------------------------------------------------------ the diff

def _events_by_seq(tr: dict) -> Dict[int, dict]:
    return {e["seq"]: e for e in tr["events"]}


def _explain_row(key: str, b: dict, f: dict, tr: dict, ev_list: List[dict]) -> Optional[str]:
    seq = f.get("_seq")
    ix = next((i for i, e in enumerate(ev_list) if e["seq"] == seq), None)
    e = ev_list[ix] if ix is not None else None
    if e is None:
        return None
    if key == "hp_delta" and f["t"] == 6:
        return "W1 stat drop" if e["kind"] == "UNBOOST" and f["hp_delta"] == -b["hp_delta"] < 0 else None
    if key == "hp_delta" and f["t"] == 8:
        want = -1.0 if e["op"] == "sideend" else 1.0
        return ("W5 side end" if want < 0 else "W5 side start") if b["hp_delta"] == 0.0 and f["hp_delta"] == want else None
    if key == "item_tr" and f["t"] == 7:
        ok = (e["kind"] == "ITEM" and b["item_tr"] == 1 and f["item_tr"] == 4
              and any(w in str(e["from"] or "").lower() for w in _TRANSFER))
        return "W3 transfer |-item|" if ok else None
    if key == "faint_cause" and f["t"] == 3:
        if f["faint_cause"] != "other":
            return None
        last = None
        for p in ev_list[:ix][::-1]:
            if p["side"] == e["side"] and p["kind"] in ("SWITCH", "DRAG"):
                break
            if p["side"] == e["side"] and p["kind"] == "DAMAGE":
                last = p
                break
        nonlethal = last is None or (last["hp_after"] is not None and last["hp_after"] > 0)
        return f"W2 faint without a lethal line (was {b['faint_cause']})" if nonlethal else None
    if f["t"] == 1 and key in ("failed", "hp_delta"):
        # the move's own window: events after its MOVE until the next MOVE of the SAME side
        mover, turn = e["side"], e["turn"]
        tail = ev_list[ix + 1:]
        if key == "failed":
            for p in tail:
                if p["kind"] == "MOVE" and p["side"] == mover:
                    break
                if p["kind"] == "MOVE":
                    break
                if (p["kind"] == "ACTIVATE" and p["side"] != mover and p["turn"] == turn
                        and str(p["effect"] or "").strip().lower() in _BLOCK):
                    return "W4 Protect block" if (not b["failed"] and f["failed"]) else None
            return None
        foreign, moving = 0.0, mover
        for p in tail:
            if p["turn"] != turn or (p["kind"] == "MOVE" and p["side"] == mover):
                break
            if p["kind"] == "MOVE":
                moving = p["side"]
            if (p["kind"] == "DAMAGE" and p["side"] != mover and moving != mover and not p["from"]
                    and p["actor"] == f["target"]):
                foreign += float(p["amount"] or 0.0)
        return "T1-row foreign bare damage" if foreign != 0.0 and abs((b["hp_delta"] - f["hp_delta"]) - foreign) < 1e-9 else None
    return None


def _cls(prev: int, n: int) -> str:
    if n == 0:
        return "reset/stay0"
    if n == prev:
        return "stay"
    return "inc"


def _explain_label(b, f, d) -> Optional[str]:
    kb, kf = b[0], f[0]
    if kb == 1 and kf == 2:
        if d.get("opp_dragged"):
            return "L1/L2 drag"
        if d.get("opp_repl"):
            return "L3 straddling replacement"
    if kb == 0 and kf == 0 and d.get("opp_called_via") and f[1] == d["opp_called_via"]:
        return "L4 caller"
    if kf == 2 and d.get("opp_over"):
        return "L5 Encore override"
    return None


def diff(base_path: str, fix_path: str) -> int:
    from agents.observation.constants import EVENT_TOKEN_DIM, EventCol as C

    load = lambda p: {r["battle"]: r for r in map(json.loads, open(p)) if r.get("battle")}  # noqa: E731
    base, fix = load(base_path), load(fix_path)
    common = sorted(set(base) & set(fix) - {"__hashes__"})
    missing = sorted(set(base) ^ set(fix))
    rows_n = dec_n = 0
    row_why, lab_why, clk_why, cell_why = (collections.Counter() for _ in range(4))
    unexplained: List[str] = []
    changed_decisions = set()
    branched: set = set()
    col_key = {C.MAGNITUDE: "hp_delta", C.OUT_HIT: "failed", C.OUT_FAIL: "failed",
               C.FAINT_CAUSE: "faint_cause", C.ITEM_TRANSITION: "item_tr"}
    for name in common:
        tb_all, tf_all = base[name]["trackers"], fix[name]["trackers"]
        if len(tb_all) != len(tf_all):
            unexplained.append(f"{name}: tracker count {len(tb_all)} vs {len(tf_all)}")
            continue
        for vi, (tb, tf) in enumerate(zip(tb_all, tf_all)):
            ev_list = tf["events"]
            if [e["seq"] for e in tb["events"]] != [e["seq"] for e in ev_list]:
                if name.startswith("policy."):
                    # the production POLICY reads the obs these fixes change, so its later choices
                    # may differ: a branched policy battle is EXCLUDED (counted), never compared
                    branched.add(name)
                    continue
                unexplained.append(f"{name}/v{vi}: the event streams differ — the trajectory branched")
                continue
            if len(tb["rows"]) != len(tf["rows"]):
                unexplained.append(f"{name}/v{vi}: row count {len(tb['rows'])} vs {len(tf['rows'])}")
                continue
            row_explained: Dict[int, Dict[str, str]] = {}
            for ri, (rb, rf) in enumerate(zip(tb["rows"], tf["rows"])):
                rows_n += 1
                for k in sorted(set(rb) | set(rf)):
                    if rb.get(k) == rf.get(k):
                        continue
                    why = _explain_row(k, rb, rf, tf, ev_list)
                    if why is None:
                        unexplained.append(f"{name}/v{vi} row {ri} {k}: {rb.get(k)!r} -> {rf.get(k)!r}  row={rf}")
                    else:
                        row_why[why] += 1
                        row_explained.setdefault(ri, {})[k] = why
            db, df = tb["decisions"], tf["decisions"]
            if len(db) != len(df):
                unexplained.append(f"{name}/v{vi}: decision count {len(db)} vs {len(df)}")
                continue
            diverged = False
            for di, (xb, xf) in enumerate(zip(db, df)):
                dec_n += 1
                d = xf["delta"]
                lo = max(0, xf["rows_total"] - 32)
                if any(ri in row_explained for ri in range(lo, xf["rows_total"])):
                    changed_decisions.add((name, vi, di))      # its visible window holds a changed row
                if "label" in xb and xb["label"] != xf["label"]:
                    changed_decisions.add((name, vi, di))
                    why = _explain_label(xb["label"], xf["label"], d)
                    if why is None:
                        unexplained.append(f"{name}/v{vi} dec {di} label {xb['label']} -> {xf['label']} delta={d}")
                    else:
                        lab_why[why] += 1
                pb = db[di - 1]["n"] if di else 0
                pf = df[di - 1]["n"] if di else 0
                if _cls(pb, xb["n"]) != _cls(pf, xf["n"]) or (not diverged and xb["n"] != xf["n"]):
                    t1 = d.get("our_damaging") and d.get("tgt") is not None and d["tgt"] <= -EPS and d["hit"] > -EPS
                    t2 = (xb["delta"].get("outcome") == "hit" and d.get("outcome") == "fail"
                          and d.get("opp_mv") in _STALL)
                    if t1 or t2:
                        clk_why["T1 clause (i) without our own hit" if t1 else "T2 blocked attack frozen"] += 1
                    elif diverged:
                        clk_why["cascade (n carries)"] += 1
                    else:
                        unexplained.append(f"{name}/v{vi} dec {di} clock {pb}->{xb['n']} vs {pf}->{xf['n']} delta={d}")
                diverged = xb["n"] != xf["n"]
                if xb["n"] != xf["n"]:
                    changed_decisions.add((name, vi, di))
                # obs cells (golden captures carry the vector)
                if "ew" in xf:
                    if xb["rest"] != xf["rest"]:
                        idx = [i for i, (a, c) in enumerate(zip(xb["rest"], xf["rest"])) if a != c]
                        from agents.observation.constants import OFFSET_EVENT_WINDOW, OFFSET_REACTIVE
                        for i in idx:
                            flat = i if i < OFFSET_EVENT_WINDOW else i + 32 * EVENT_TOKEN_DIM
                            if flat == OFFSET_REACTIVE + 2 and xb["n"] != xf["n"]:
                                cell_why["reactive[2] turns_since_progress"] += 1
                            else:
                                unexplained.append(f"{name} dec {di}: obs index {flat} changed outside the window/clock")
                        changed_decisions.add((name, vi, di))
                    total = xf["rows_total"]
                    n_rows = min(32, total)
                    for slot in range(32):
                        cells_b = xb["ew"][slot * EVENT_TOKEN_DIM:(slot + 1) * EVENT_TOKEN_DIM]
                        cells_f = xf["ew"][slot * EVENT_TOKEN_DIM:(slot + 1) * EVENT_TOKEN_DIM]
                        if cells_b == cells_f:
                            continue
                        changed_decisions.add((name, vi, di))
                        ordinal = total - n_rows + (slot - (32 - n_rows))
                        for c, (a, z) in enumerate(zip(cells_b, cells_f)):
                            if a == z:
                                continue
                            k = col_key.get(c)
                            why = row_explained.get(ordinal, {}).get(k) if k else None
                            if why is None:
                                unexplained.append(f"{name} dec {di} slot {slot} col {C(c).name}: {a} -> {z} "
                                                   f"(row {ordinal}: {tf['rows'][ordinal] if 0 <= ordinal < len(tf['rows']) else '?'})")
                            else:
                                cell_why[f"{C(c).name} ← {why}"] += 1
    print(f"battles compared: {len(common)}  (only in one capture: {missing[:5]}{'…' if len(missing) > 5 else ''})")
    if branched:
        print(f"EXCLUDED — a production-policy battle whose trajectory branched (the policy reads the "
              f"changed obs): {sorted(branched)}")
    print(f"event-window rows: {rows_n}   viewer decisions: {dec_n}   decisions with any change (a changed row in its window, its label, or its clock): {len(changed_decisions)}")
    per = lambda n: f"{1000.0 * n / max(dec_n, 1):7.2f} /1,000"  # noqa: E731
    for title, c in (("ROW changes", row_why), ("LABEL changes", lab_why), ("CLOCK transitions", clk_why),
                     ("OBS cells (golden)", cell_why)):
        print(f"\n{title}:")
        for k, n in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"  {k:60s} {n:8d}  {per(n)}")
    if "__hashes__" in fix:
        bh, fh = base.get("__hashes__", {}).get("hashes", []), fix["__hashes__"]["hashes"]
        print(f"\nobs vectors: {len(fh)} decisions, {sum(a != b for a, b in zip(bh, fh))} changed")
    if unexplained:
        print(f"\n❌ {len(unexplained)} UNEXPLAINED (first 30):")
        for u in unexplained[:30]:
            print("   " + u)
        return 1
    print("\n✅ every change is explained by the three fixes")
    return 0


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "golden":
        capture_golden(*args)
    elif mode == "corpus":
        capture_corpus(*args)
    elif mode == "diff":
        sys.exit(diff(*args))
    else:
        raise SystemExit(__doc__)
