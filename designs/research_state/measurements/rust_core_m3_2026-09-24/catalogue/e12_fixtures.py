"""The E12 mechanic catalogue on CONSTRUCTED battles: for each of the native record's fixtures
(``src/rust_sim/tests/window_record_test.rs``, whose inputs ``export_fixtures`` writes to
``e12_fixtures.jsonl``), replay it through ``core_events --trackers`` with slice T on, and print —
per viewer, per decision — side by side:

* the NATIVE record's window (what happened, in order, with attribution),
* the frozen ``TurnDelta`` the Python path folds for that decision (every non-empty field),
* the α/β opponent-intent LABEL (``kind`` 0 MOVE / 1 SWITCH / 2 masked),
* the 22-column EVENT WINDOW rows the decision APPENDED (the obs's route for the history).

Slice T holds the core's trackers equal to the Python path's at every one of these decisions, so the
label and the rows printed ARE what training builds. The output is the evidence behind the README's
§5 per-mechanic "keeps / loses" table.

    (cd src/rust_sim && WINDOW_FIXTURES_OUT=$PWD/../../<this dir>/e12_fixtures.jsonl \\
        cargo test --profile selfcheck --features emission-selfcheck --test window_record_test \\
        -- --ignored export_fixtures)
    python e12_fixtures.py e12_fixtures.jsonl > e12_fixtures.txt
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys

EW_KEYS = ("t", "actor", "side", "target", "move_id", "hp_delta", "missed", "failed", "crit", "eff", "we_first",
           "status", "cant", "faint_cause", "item_tr", "forced_window")


def _delta_fields(d) -> dict:
    if d is None:
        return {}
    out = {}
    for f in dataclasses.fields(d):
        v = getattr(d, f.name)
        if hasattr(v, "tolist"):
            v = [round(float(x), 3) for x in v.tolist()]
            if not any(v):
                continue
        elif v is None or v is False or (isinstance(v, (int, float, str, tuple, list)) and not v):
            continue
        elif isinstance(v, float):
            v = round(v, 3)
        elif hasattr(v, "name"):
            v = v.name
        elif not isinstance(v, (int, str, bool, list, tuple, dict)):
            v = type(v).__name__
        out[f.name] = v
    r = d.opp_resolved_move_id
    if r:
        out["opp_resolved_move_id (property)"] = r
    return out


def _act(a: dict) -> str:
    k = a["kind"]
    keep = {x: y for x, y in a.items() if x not in ("kind", "effects", "turn", "scope") and y not in (None, False, [])}
    eff = ["/".join(str(z) for z in (e[1], e[0][1] if e[0] else None, e[3][0], e[3][1]) if z) for e in a.get("effects", [])]
    return f"{k} {json.dumps(keep, separators=(',', ':'))}" + (f"  effects: {eff}" if eff else "")


def main(path: str) -> int:
    from agents.battle import rust_core_parity as P
    from agents.battle import rust_core_parity_trackers as T

    logging.getLogger("poke-env").setLevel(logging.ERROR)
    fixtures = [json.loads(line) for line in open(path) if line.strip()]
    battles = [P.RecordedBattle(label=f"e12.{f['name']}", format_id="gen3customgame", seed=f["seed"],
                                p1={"name": "P1", "team": f["p1"]}, p2={"name": "P2", "team": f["p2"]},
                                commands=[[f"p{s + 1}", tok, "if_open"] for s, tok in f["script"]])
               for f in fixtures]
    # the full TurnDelta of each decision, in check_trackers' order (p1's decisions, then p2's)
    deltas: list = []
    real_proj = T.delta_projection

    def capture(d):
        deltas.append(d)
        return real_proj(d)

    T.delta_projection = capture
    # the fixtures are `gen3customgame` (fixed levels and sets, no team validation); slice T's
    # production scope is gen3ou only — widen it for these constructed battles, and say so
    real_scope = T.SCOPE_FORMATS
    T.SCOPE_FORMATS = frozenset(real_scope | {"gen3customgame"})
    fails = 0
    try:
        for b in battles:
            t = T.TrackerCensus()
            got: dict = {}
            deltas.clear()
            P.check_battles([b], P.Census(), trackers=t, on_result=lambda _b, res: got.update(res))
            print(f"\n######## {b.label} — slice T: {sum(t.divergences.values())} divergences over {t.decisions} decisions")
            if t.divergences or t.refused or not t.decisions:
                fails += 1
                print(t.render())
            di = 0
            for vi, viewer in enumerate(got.get("trackers") or []):
                seen_rows = 0
                for cap in viewer:
                    if "trackers" not in cap:
                        continue
                    d = deltas[di] if di < len(deltas) else None
                    di += 1
                    rows = cap["trackers"]["window"]["rows"]
                    new = rows[seen_rows:] if len(rows) >= seen_rows else rows
                    seen_rows = len(rows)
                    print(f"--- p{vi + 1} decision after chunk {cap['after']}")
                    for a in cap.get("window") or []:
                        print("    RECORD ", _act(a))
                    print("    DELTA  ", json.dumps(_delta_fields(d), separators=(",", ":"), default=lambda o: getattr(o, "name", str(o))))
                    print("    LABEL  ", json.dumps(cap["trackers"]["label"], separators=(",", ":")))
                    print("    CLOCK  ", json.dumps({"n": cap["trackers"]["clock"]["n"], "value": cap["trackers"]["clock_value"]}))
                    for r in new:
                        print("    EW-ROW ", json.dumps({k: r.get(k) for k in EW_KEYS if r.get(k) not in (None, False, 0, 0.0)},
                                                        separators=(",", ":")))
    finally:
        T.delta_projection = real_proj
        T.SCOPE_FORMATS = real_scope
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "e12_fixtures.jsonl"))
