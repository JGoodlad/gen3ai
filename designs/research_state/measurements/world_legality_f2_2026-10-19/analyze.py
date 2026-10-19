"""F2's read (descriptive, post hoc; no playout, no outcome): the stored F worlds (``stored``, built at each study's pin)
against the rebuild at this checkout with the legality rule (``on``) and, where they can differ, without it (``off``).
Reads ``rebuild_worlds.py``'s rows; writes ``result_f2.json``.

The analysed states are the DONE PREFIX of each shard's seeded order (``rebuild_worlds.py``): a uniform random sample
of the stage-0 population when the rebuild stops early.

    PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
    CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 $PY -I analyze.py --out-dir <rebuild dir> [--of 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rebuild_worlds as RB  # noqa: E402

PAIRS = [("tyranitar", "dragondance", "pursuit"), ("blissey", "wish", "aromatherapy"),
         ("blissey", "aromatherapy", "healbell"), ("skarmory", "whirlwind", "drillpeck"),
         ("gengar", "willowisp", "perishsong"), ("salamence", "dragondance", "wish")]


def rows(d: Path, mode: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in sorted(d.glob(f"{mode}_*_of_*.jsonl")):
        for r in RB._jsonl(p):
            out[r["id"]] = r
    return out


def prefix(ids: List[str], done: set, of: int) -> List[str]:
    order = sorted(ids, key=lambda i: hashlib.sha256(f"f2_2026-10-19:{i}".encode()).hexdigest())
    out = []
    for s in range(of):
        sh = [i for k, i in enumerate(order) if k % of == s]
        n = next((k for k, i in enumerate(sh) if i not in done), len(sh))
        out += sh[:n]
    return sorted(out)


def sets_of(team: str):
    from main.value_search.worlds import parse_set, species_id, typed_moves

    for e in team.split("]"):
        f = parse_set(e)
        yield species_id(f), set(typed_moves(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--of", type=int, default=2)
    ap.add_argument("--write", default=str(HERE / "result_f2.json"))
    a = ap.parse_args()
    from main.value_search import world_legality as WL
    from main.value_search.worlds import _gender_pad

    d = Path(a.out_dir)
    stored = RB.stored_rows()
    on, off = rows(d, "on"), rows(d, "off")
    ids = prefix(sorted(stored), set(on), a.of)
    L = WL.Legality(pad=_gender_pad())

    def illegal(teams: List[str]) -> List[bool]:
        return [bool(p) for p in L.team_problems(teams)]

    rep: dict = {"schema": "world_legality_f2_v1", "population": len(stored), "analysed": len(ids),
                 "by_pop": dict(Counter(stored[i]["pop"] for i in ids))}
    for name, src in (("stored", stored), ("on", on)):
        ok = [i for i in ids if src[i].get("ok")]
        teams = [t for i in ok for t in src[i]["teams"]]
        bad = illegal(teams)
        per_state = defaultdict(int)
        k = 0
        for i in ok:
            for _ in src[i]["teams"]:
                per_state[i] += bad[k]
                k += 1
        rep[name] = {"states": len(ids), "ok": len(ok), "excluded": sum(1 for i in ids if not src[i].get("ok")
                                                                        and not src[i].get("crash")),
                     "defects": sum(1 for i in ids if src[i].get("crash")),
                     "worlds": len(teams), "illegal_worlds": sum(bad),
                     "states_with_an_illegal_world": sum(1 for v in per_state.values() if v)}
    fails = Counter()
    for i in ids:
        for k_, v in (on[i].get("fails") or {}).items():
            if isinstance(v, int):
                fails[k_] += v
    touched = [i for i in ids if on[i].get("legality_touched")]
    rep["on"]["fails"] = dict(fails)
    rep["on"]["states_touched"] = len(touched)
    rep["on"]["defect_examples"] = [on[i]["error"][:300] for i in ids if on[i].get("crash")][:5]
    rep["on"]["excluded_reasons"] = dict(Counter(("illegal" if "illegal" in on[i].get("error", "") else "other")
                                                 for i in ids if not on[i].get("ok") and not on[i].get("crash")))
    # ---- on vs off where they can differ (the legality rule's own effect, at this checkout)
    both = [i for i in touched if i in off]
    ctrl = [i for i in ids if i in off and off[i].get("control")]
    rep["on_vs_off"] = {
        "touched_rebuilt_off": len(both), "touched_missing_off": len(touched) - len(both),
        "excluded_on": sum(1 for i in both if not on[i].get("ok")),
        "excluded_off": sum(1 for i in both if not off[i].get("ok")),
        "newly_excluded_by_the_rule": sorted(i for i in both if off[i].get("ok") and not on[i].get("ok")),
        "rescued_by_the_rule": sorted(i for i in both if on[i].get("ok") and not off[i].get("ok")),
        "control_states": len(ctrl),
        "control_identical": sum(1 for i in ctrl if off[i].get("teams") == on[i].get("teams")
                                 and off[i].get("ok") == on[i].get("ok")),
        "off_illegal_worlds": sum(illegal([t for i in both if off[i].get("ok") for t in off[i]["teams"]])),
        "off_worlds": sum(len(off[i]["teams"]) for i in both if off[i].get("ok")),
    }
    # ---- off (this checkout, no rule) vs stored (the study pins): what ELSE changed in the builder since
    reb = [i for i in ids if i in off]
    rep["off_vs_stored"] = {
        "states": len(reb),
        "same_verdict": sum(1 for i in reb if bool(off[i].get("ok")) == bool(stored[i].get("ok"))),
        "identical_worlds": sum(1 for i in reb if off[i].get("ok") and off[i].get("teams") == stored[i].get("teams")),
    }
    # ---- the world distribution: species and move marginals over the worlds of the states OK in both
    common = [i for i in ids if stored[i].get("ok") and on[i].get("ok")]
    marg = {}
    for name, src in (("stored", stored), ("on", on)):
        sp, mv, n_sp = Counter(), Counter(), Counter()
        pairs = Counter()
        for i in common:
            for t in src[i]["teams"]:
                for sid, moves in sets_of(t):
                    sp[sid] += 1
                    for m in moves:
                        mv[(sid, m)] += 1
                    for s_, x, y in PAIRS:
                        if sid == s_:
                            n_sp[s_] += 1
                            pairs[(s_, x, y)] += int(x in moves and y in moves)
        marg[name] = (sp, mv, n_sp, pairs)
    (sp0, mv0, n0, pr0), (sp1, mv1, n1, pr1) = marg["stored"], marg["on"]
    tot0, tot1 = sum(sp0.values()), sum(sp1.values())
    tvd_sp = 0.5 * sum(abs(sp0[s] / tot0 - sp1[s] / tot1) for s in set(sp0) | set(sp1))
    shifts = []
    for s in sorted(set(sp0) & set(sp1)):
        if min(sp0[s], sp1[s]) < 40:
            continue
        for m in sorted({m for (s_, m) in list(mv0) + list(mv1) if s_ == s}):
            d_ = mv1[(s, m)] / sp1[s] - mv0[(s, m)] / sp0[s]
            shifts.append({"species": s, "move": m, "stored": round(mv0[(s, m)] / sp0[s], 4),
                           "on": round(mv1[(s, m)] / sp1[s], 4), "delta": round(d_, 4), "n_stored": sp0[s],
                           "n_on": sp1[s]})
    shifts.sort(key=lambda r: -abs(r["delta"]))
    rep["marginals"] = {
        "common_ok_states": len(common), "species_sets": {"stored": tot0, "on": tot1},
        "species_tvd": round(tvd_sp, 4),
        "top_move_shifts": shifts[:20],
        "illegal_pairs": [{"species": s, "pair": [x, y], "stored": f"{pr0[(s, x, y)]}/{n0[s]}",
                           "on": f"{pr1[(s, x, y)]}/{n1[s]}"} for s, x, y in PAIRS],
    }
    Path(a.write).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "marginals"}, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in rep["marginals"].items() if k != "top_move_shifts"}, indent=1))
    for r in rep["marginals"]["top_move_shifts"][:12]:
        print(r)


if __name__ == "__main__":
    main()
