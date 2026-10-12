"""DESCRIPTIVE (no outcome, no model): how often do the honest worlds guess the opponent's hidden MOVES, by sampler?
Per accepted, PLAYED world (the first 8) of each world directory (``det`` = Phase 2's Smogon prior, ``det_b`` = 2b's
belief species + Smogon sets, ``det_c`` = 2c's belief species + the NETWORK's move belief):

- SEEN mons (species revealed at the root), the move slots the searched side never saw used:
  ``slot_hit`` = the world's move in that slot equals the true move in that slot (`world_accuracy_read.py`'s metric,
  the Phase 2 "16 %"); ``set_recall`` = the share of the true unseen moves that the world's set contains anywhere (the
  game-relevant one: a move's slot order matters only to the choice token);
- NEVER-SEEN slots: ``species_hit`` = the world's species is a true hidden species; ``new_set_recall`` = on those
  hits, the share of the true species' four moves the world's set contains;
- ``det_c`` only, BEFORE the gate: ``pi_expected_recall`` = Σ_{true unseen moves} π_m / Σ_{candidates} π_m per seen mon
  (the network's expected share, from the root's stored ``move_w``).

    python -I move_accuracy_read.py <cp> [<cp> ...]   →  move_accuracy_read.json
"""
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = Path.home() / "gen3ai_archive" / "value_disc_pilot"
DIRS = ("det", "det_b", "det_c")


def toid(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def parse(e):
    f = e.split("|")
    return f + [""] * (12 - len(f))


def seen_hp(tm, seen):
    return tm in seen or (tm.startswith("hiddenpower") and "hiddenpower" in seen)


def read_dir(cp, dname, battles, decs, ids=None):
    acc = {k: [0, 0] for k in ("slot_hit", "set_recall", "species_hit", "new_set_recall", "pi_expected_recall")}
    spnum = json.loads((HERE.parents[3] / "data" / "pokemon" / "gen3_species.json").read_text())
    n_states = 0
    for p in sorted((ARCH / cp / dname).glob("worlds_*.jsonl")):
        for line in p.read_text().splitlines():
            w = json.loads(line)
            if not w.get("ok") or any(t is None for t in w["teams"][:8]):
                continue
            if ids is not None and w["id"] not in ids:
                continue
            n_states += 1
            d = decs[w["id"]]
            b = battles[d["battle"]]
            opp = b["p2"] if d["side"] == "p1" else b["p1"]
            true = [parse(e) for e in opp["team"].split("]")]
            kept = w["revealed"]
            tsets = {toid(f[1] or f[0]): f[4].split(",") for f in true}
            hidden = set(tsets) - set(kept)
            for team in w["teams"][:8]:
                ws = [parse(e) for e in team.split("]")]
                for tf, wf in zip(true, ws):
                    tid, wid = toid(tf[1] or tf[0]), toid(wf[1] or wf[0])
                    if tid in kept:
                        seen = set(kept[tid])
                        tm_un = [m for m in tf[4].split(",") if not seen_hp(m, seen)]
                        wset = set(wf[4].split(","))
                        for tm, wm in zip(tf[4].split(","), wf[4].split(",")):
                            if seen_hp(tm, seen):
                                continue
                            acc["slot_hit"][0] += tm == wm
                            acc["slot_hit"][1] += 1
                        acc["set_recall"][0] += sum(m in wset for m in tm_un)
                        acc["set_recall"][1] += len(tm_un)
                    else:
                        acc["species_hit"][0] += wid in hidden
                        acc["species_hit"][1] += 1
                        if wid in hidden:
                            wset = set(wf[4].split(","))
                            acc["new_set_recall"][0] += sum(m in wset for m in tsets[wid])
                            acc["new_set_recall"][1] += len(tsets[wid])
            if dname == "det_c" and "move_w" in w:
                for tid, seenl in kept.items():
                    num = str(int(spnum.get(tid, {}).get("num") or 0))
                    mw = w["move_w"]["seen"].get(num)
                    if not mw:
                        continue
                    seen = set(seenl)
                    hp_known = "hiddenpower" in seen
                    cand = {k: v for k, v in mw.items() if not (hp_known and k.startswith("hiddenpower"))}
                    tm_un = [m for m in tsets[tid] if not seen_hp(m, seen)]
                    tot = sum(cand.values())
                    if tm_un and tot > 0:
                        acc["pi_expected_recall"][0] += sum(cand.get(m, 0.0) for m in tm_un) / tot * len(tm_un)
                        acc["pi_expected_recall"][1] += len(tm_un)
    out = {k: (round(a / b, 4) if b else None) for k, (a, b) in acc.items()}
    out.update({f"n_{k}": b for k, (a, b) in acc.items()})
    out["states"] = n_states
    return out


def main():
    res = {}
    for cp in sys.argv[1:]:
        with gzip.open(ARCH / cp / "bank" / "battles.jsonl.gz", "rt") as fh:
            battles = {b["battle_id"]: b for b in (json.loads(x) for x in fh)}
        with gzip.open(ARCH / cp / "bank" / "decisions.jsonl.gz", "rt") as fh:
            decs = {d["id"]: d for d in (json.loads(x) for x in fh)}
        # the states every arm built a full set of 8 worlds for (the common states)
        sets = []
        for dn in DIRS:
            s = set()
            for p in sorted((ARCH / cp / dn).glob("worlds_*.jsonl")):
                for line in p.read_text().splitlines():
                    w = json.loads(line)
                    if w.get("ok") and all(t is not None for t in w["teams"][:8]):
                        s.add(w["id"])
            sets.append(s)
        common = set.intersection(*sets)
        res[cp] = {"common_states": len(common),
                   **{dn: read_dir(cp, dn, battles, decs, common) for dn in DIRS}}
        print(cp, json.dumps(res[cp]))
    (HERE / "move_accuracy_read.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
