"""Q19 stage 0a (a TOOL, OUTCOME-BLIND): why F's battles stop searching. For each given cell (default: the sample's
forum72 cells), the battle's FIRST exclusion is found in its banked decision log (classed as
``../inputs/timer_replay.py``'s ``excl_class``), I at that decision is rebuilt from the capture's honest log
(``screen.battle_plan``, every identity check of the screen; ``info_set.extract`` on the cut), and the segments the
carry had not matched — I's segments after the previous SEARCHED decision's I (all of them when none searched before)
— are tagged from I's lines alone. Never a game result, a summary or a referee field.

THE TAGS (each a rule on I, none on the truth; a battle may carry several; ``other`` iff none):

* ``crit``: a ``|-crit|`` line in an unmatched segment;
* ``sleep_onset``: a ``|-status|<mon>|slp`` line not from Rest (``honest_inv._last_onset``'s rule) in one;
* ``beat_up``: a ``|move|<mon>|Beat Up`` line in one;
* ``hidden_hp_chain``: at an unmatched segment, ``info_set.hp_last`` lists >= 2 earlier damage rolls on one opponent
  mon since its exact HP was last known (a chain of percent-only hits the matcher must re-roll);
* ``outside_prior_<move|item|ability|species>``: a reveal NEW since the previous searched I (``honest_worlds.Scaffold``
  of each I: the seen mons' moves, item, ability, species) that the builder's own Smogon prior
  (``world_draws.Prior``: ``moves`` / ``items`` / ``ability`` keys, ``pool``) gives no mass; a bare Hidden Power counts
  as inside iff the prior holds any Hidden Power;
* ``scaffold_error``: I's scaffold cannot be read (its WorldError, named).

THE STAGE-0a INPUT (from the banked logs alone): the share of F's STUCK decisions whose class is
``stuck_probe_revised_only``; ``--stuck-denominator`` (REQUIRED: the draft does not define "stuck") names the
denominator, ``probe`` (the decisions classed ``stuck_probe_*``) or ``excluded`` (every excluded decision), over
``--stuck-scope`` (REQUIRED) ``sample`` (the sample's forum72 cells) or ``all`` (every complete forum72 cell). Both
denominators and scopes are written; the named one is ``decided``.

    $PY diagnose_f.py --sample sample.json --out DIR --stuck-denominator probe --stuck-scope all [--cells A,B]
        [--max-battles N]
"""
import argparse
import collections
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "inputs"))
import screen as S
from timer_replay import excl_class

FIRST_CLASSES = ("warm_none_built", "cold_failed", "warm_some_built")


def _onset(ln):
    p = ln.split("|")
    return ln.startswith("|-status|") and p[3:4] == ["slp"] and "Rest" not in ln


def tags_of(info, prev, prior):
    from main.value_search import honest_worlds as HW
    from main.value_search import info_set as IS
    from main.value_search.world_draws import WorldError, is_hp

    segs = info.segments()
    k = len(prev.segments()) if prev is not None else 0
    if prev is not None and tuple(info.lines[:len(prev.lines)]) != tuple(prev.lines):
        return ["not_prefix"], {"k_prev": k, "segments": len(segs)}
    un = segs[k:]
    lines = [ln for s in un for ln in s]
    tags = set()
    if any(ln.startswith("|-crit|") for ln in lines):
        tags.add("crit")
    if any(_onset(ln) for ln in lines):
        tags.add("sleep_onset")
    if any(ln.startswith("|move|") and ln.split("|")[3:4] == ["Beat Up"] for ln in lines):
        tags.add("beat_up")
    hl = IS.hp_last(info)
    if any(len(e[1]) >= 2 for s in range(k, len(segs)) for e in hl[s]):
        tags.add("hidden_hp_chain")
    new = []
    try:
        cur = {m.nick: m for m in HW.Scaffold.of(info).seen}
        old = {m.nick: m for m in HW.Scaffold.of(prev).seen} if prev is not None else {}
    except WorldError as exc:
        tags.add("scaffold_error")
        return sorted(tags), {"k_prev": k, "segments": len(segs), "scaffold_error": str(exc)[:200]}
    for nick, m in cur.items():
        o = old.get(nick)
        if m.sid not in prior.pool:
            if o is None:
                new.append(("species", m.sid))
            continue
        moves = prior.moves(m.sid)
        for mv in m.moves:
            if o is not None and mv in o.moves:
                continue
            inside = any(is_hp(x) for x in moves) if is_hp(mv) else mv in moves
            if not inside:
                new.append(("move", f"{m.sid}:{mv}"))
        if m.item and (o is None or o.item != m.item) and m.item not in prior.items(m.sid):
            new.append(("item", f"{m.sid}:{m.item}"))
        if m.ability and (o is None or o.ability != m.ability) and m.ability not in prior.ability(m.sid):
            new.append(("ability", f"{m.sid}:{m.ability}"))
    for what, _x in new:
        tags.add(f"outside_prior_{what}")
    return sorted(tags) or ["other"], {"k_prev": k, "segments": len(segs), "outside_prior": new}


def stuck_share(cells_dir, names):
    c = collections.Counter()
    for n in names:
        for r in S._loads(Path(cells_dir) / n / "search_decisions.jsonl"):
            if r.get("why") == "excluded":
                c[excl_class(r.get("error") or "")] += 1
    probe = c["stuck_probe_revised_only"] + c["stuck_probe_other"]
    exc = sum(c.values())
    return {"battles": len(names), "classes": dict(sorted(c.items())),
            "probe": [c["stuck_probe_revised_only"], probe],
            "excluded": [c["stuck_probe_revised_only"], exc],
            "share_probe": float(Fraction(c["stuck_probe_revised_only"], probe)) if probe else None,
            "share_excluded": float(Fraction(c["stuck_probe_revised_only"], exc)) if exc else None}


def main(a):
    from main.value_search import info_set as IS
    from main.value_search.world_draws import Prior

    smp = json.loads(Path(a.sample).read_text())
    cells = Path(smp["cells_dir"])
    names = a.cells.split(",") if a.cells else list(smp["panels"]["forum72"])
    if a.max_battles:
        names = names[:a.max_battles]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prior = Prior()
    rows = []
    for name in names:
        row = {"cell": name}
        try:
            cap, recs = S.cell_inputs(cells / name)
            xs = [i for i, r in enumerate(recs) if r.get("why") == "excluded"]
            if not xs:
                row["first_class"] = None
                rows.append(row)
                continue
            i = xs[0]
            r = recs[i]
            row.update(first_class=excl_class(r.get("error") or ""), turn=int(r["turn"]), n=int(r["n"]),
                       exclusions=len(xs), path=sum(1 for x in recs if S.is_path(x)))
            prev_i = next((j for j in range(i - 1, -1, -1) if recs[j].get("searched")), None)
            plan = S.battle_plan(cap, recs)
            x = plan["decisions"][i]
            info, _row = IS.extract(x["log"], x["at"], plan["side"])
            prev = None
            if prev_i is not None:
                y = plan["decisions"][prev_i]
                prev, _r = IS.extract(y["log"], y["at"], plan["side"])
                row["prev_turn"] = int(recs[prev_i]["turn"])
            row["tags"], row["detail"] = tags_of(info, prev, prior)
        except (S.ReconError, ValueError, KeyError) as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("cell", "first_class", "turn", "tags", "error")}), flush=True)
    with open(out / "diagnose_rows.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    table = collections.defaultdict(collections.Counter)
    for row in rows:
        if row.get("first_class") is None:
            continue
        table[row["first_class"]]["battles"] += 1
        for t in row.get("tags") or ["error"]:
            table[row["first_class"]][t] += 1
    all_f = S.complete_cells(cells)["forum72"]
    stuck = {"sample": stuck_share(cells, smp["panels"]["forum72"]), "all": stuck_share(cells, all_f)}
    pick = stuck[a.stuck_scope]
    doc = {"schema": "q19_stage0a_diagnose_v1", "cells": len(names),
           "with_exclusion": sum(1 for r in rows if r.get("first_class")),
           "errors": [r for r in rows if r.get("error")],
           "tags_by_first_class": {c: dict(sorted(v.items())) for c, v in sorted(table.items())},
           "first_classes_of_the_draft": list(FIRST_CLASSES),
           "stuck_share": stuck,
           "decided": {"scope": a.stuck_scope, "denominator": a.stuck_denominator,
                       "revised_only_of_stuck": pick[a.stuck_denominator],
                       "share": pick[f"share_{a.stuck_denominator}"]}}
    (out / "diagnose.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: doc[k] for k in ("tags_by_first_class", "decided")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default="", help="comma-separated cell names (default: the sample's forum72 cells)")
    ap.add_argument("--max-battles", type=int, default=0)
    ap.add_argument("--stuck-denominator", choices=("probe", "excluded"), required=True)
    ap.add_argument("--stuck-scope", choices=("sample", "all"), required=True)
    main(ap.parse_args())
