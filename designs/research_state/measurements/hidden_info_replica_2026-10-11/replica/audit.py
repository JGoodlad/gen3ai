"""Audits of the replication's own instrument (it holds itself to §5's A1 / A4 and the teeth rule).

    python -m replica.audit --cp end2 --out <dir> [--rows <rows dir>]

- ID   BUILDER IDENTITY: every registered state's TRUE opponent team pushed through this builder (explicit genders,
       the aligned seed, the rendering, the pending-choice drop) must pass the gate (lines AND rows). 0 failures.
- TEETH each planted defect must FAIL the gate on every state where it is observable:
       T1 a seen mon's REVEALED item swapped for another legal item; T2 a switch command rendered to the wrong
       slot; T3 a move the searched side saw used (``|move|``) removed from the set; T4 the aligned seed one draw
       off.
- PARSE every fact the parser calls revealed, checked against the true team (an audit-only read of the truth):
       a revealed move / item / ability that the true set does not hold is a PARSER ERROR.
- O    with ``--rows``: the oracle arm's outcomes on dice seeds 0-7 against the Phase 1 rows' seeds 0-7 (the same
       battle, seeds and continuation: the harness must reproduce them).
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    from replica import env
    env.setup(a.cp)
    from main.policy_spectrum.bank import load_bank

    from replica import builder as B
    from replica import sampling as S

    bank = load_bank(env.ARCHIVE / a.cp / "bank")
    dec = {d["id"]: d for d in bank.decisions}
    ids = list(env.registered_states(a.cp))
    if a.limit:
        ids = ids[:a.limit]
    res: dict = {"cp": a.cp, "n": len(ids), "ID": Counter(), "teeth": {t: Counter() for t in ("T1", "T2", "T3", "T4")},
                 "parse": Counter(), "parse_errors": [], "pending": 0, "prep_errors": [], "id_failures": [],
                 "truth_legality": Counter()}
    for sid in ids:
        prep = B.prepare(bank, dec, sid)
        if prep.error:
            res["prep_errors"].append([sid, prep.error])
            continue
        res["pending"] += prep.pending is not None
        for p in S.legality(prep.true_opp):
            res["truth_legality"][p.split(":")[0]] += 1
        w = B.assemble(prep, "ID", B.rng_for("audit", sid))
        g = B.gate_many(prep, [w], rows=True)[0]
        res["ID"]["pass" if g.ok else "FAIL"] += 1
        if not g.ok and len(res["id_failures"]) < 10:
            res["id_failures"].append([sid, g.first_bad_line, g.error])
        # ---- parse audit
        for name, m in prep.seen.mons.items():
            t = prep.true_opp[prep.seen_slot[name]]
            for mv in m.moves:
                ok = mv in t.moves or (mv == "hiddenpower" and any(S.is_hp(x) for x in t.moves))
                res["parse"]["move_ok" if ok else "move_ERR"] += 1
                if not ok and len(res["parse_errors"]) < 20:
                    res["parse_errors"].append([sid, name, "move", mv, t.moves])
            if m.item is not None:
                ok = m.item == S.to_id_safe(t.item)
                res["parse"]["item_ok" if ok else "item_ERR"] += 1
                if not ok and len(res["parse_errors"]) < 20:
                    res["parse_errors"].append([sid, name, "item", m.item, t.item])
            if m.ability is not None:
                ok = m.ability == S.to_id_safe(t.ability)
                res["parse"]["ability_ok" if ok else "ability_ERR"] += 1
                if not ok and len(res["parse_errors"]) < 20:
                    res["parse_errors"].append([sid, name, "ability", m.ability, t.ability])
        # ---- teeth
        planted = []
        rev_items = [n for n, m in prep.seen.mons.items() if m.item]
        if rev_items:
            w1 = [x.copy() for x in w]
            x = w1[prep.seen_slot[rev_items[0]]]
            x.item = "leftovers" if x.item != "leftovers" else "lumberry"
            planted.append(("T1", w1, None))
        opp_sw = [j for j, c in enumerate(prep.prefix) if c[0] == prep.opp and c[1].startswith("switch ")
                  and j != prep.pending]
        if opp_sw:
            planted.append(("T2", w, opp_sw[0]))
        used = None
        for name, m in prep.seen.mons.items():
            t = prep.true_opp[prep.seen_slot[name]]
            for mv in m.moves:
                if mv in t.moves and any(ln.startswith(f"|move|{prep.opp}a: {name}|") for ln in prep.truth_lines):
                    used = (name, mv)
                    break
            if used:
                break
        if used:
            w3 = [x.copy() for x in w]
            x = w3[prep.seen_slot[used[0]]]
            alt = next(m for m in ("protect", "toxic", "substitute", "rest", "earthquake") if m not in x.moves)
            x.moves = [alt if mv == used[1] else mv for mv in x.moves]
            planted.append(("T3", w3, None))
        for tag, ww, sw_ix in planted:
            if tag == "T2":
                cmds, _ = B.render(prep, ww)
                # move the first opponent switch to another slot's mon
                k = [i for i, c in enumerate(cmds) if c[0] == prep.opp and c[1].startswith("switch ")][0]
                cur = cmds[k][1].split(" ", 1)[1]
                other = next(m.name for m in ww if m.name != cur)
                cmds[k] = [cmds[k][0], f"switch {other}"]
                from agents.battle.core_replay import run_core
                r = run_core([B._recorded(prep.battle, prep.opp, S.pack_team(ww), cmds, prep.own_team,
                                          prep.world_seed)], trackers=True, obs=True)[0]
                failed = (not r.get("ok")) or B.lines_of(r, prep.v) != prep.truth_lines or \
                    B.rows_of(r, prep.v) != prep.truth_rows
            else:
                failed = not B.gate_many(prep, [ww], rows=True)[0].ok
            res["teeth"][tag]["caught" if failed else "MISSED"] += 1
        # T4: the seed one draw off
        saved = prep.world_seed
        prep.world_seed = B.lcg_advance(prep.battle.seed, prep.gender_draws + 1)
        res["teeth"]["T4"]["caught" if not B.gate_many(prep, [w], rows=True)[0].ok else "MISSED"] += 1
        prep.world_seed = saved
    if a.rows:
        ph1 = {}
        for f in glob.glob(str(env.ARCHIVE / a.cp / "rows" / "*.jsonl")):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    ph1[r["id"]] = r["out"]
        c = Counter()
        for f in glob.glob(str(Path(a.rows) / a.cp / "O" / "*.jsonl")):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("ok"):
                    c["not_ok"] += 1
                    continue
                for act, vals in r["outcomes"].items():
                    mine = "".join({1.0: "+", -1.0: "-", 0.0: "0"}[v] for v in vals)
                    c["agree" if mine == ph1[r["id"]][act][:8] else "DIFFER"] += 1
        res["O_vs_phase1_seeds0to7"] = dict(c)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in res.items()}
    res["teeth"] = {k: dict(v) for k, v in res["teeth"].items()}
    (out / f"audit_{a.cp}.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("parse_errors",)}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
