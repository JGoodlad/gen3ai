"""DESCRIPTIVE: how close are the determinized worlds to the truth? Per accepted world (the 8 played), the share
of the opponent's never-seen slots whose sampled SPECIES is one of the true hidden species, and the share of
re-drawn MOVE slots (seen mons) whose sampled move equals the true move in that slot. Reads the archive's
``<cp>/det/worlds_*.jsonl`` and ``<cp>/bank``; no model, no outcome.

    python -I world_accuracy_read.py <cp> [<cp> ...]   →  world_accuracy_read.json
"""
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = Path.home() / "gen3ai_archive" / "value_disc_pilot"


def toid(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def parse(e):
    f = e.split("|")
    return f + [""] * (12 - len(f))


def main():
    out = {}
    for cp in sys.argv[1:]:
        with gzip.open(ARCH / cp / "bank" / "battles.jsonl.gz", "rt") as fh:
            battles = {b["battle_id"]: b for b in (json.loads(x) for x in fh)}
        with gzip.open(ARCH / cp / "bank" / "decisions.jsonl.gz", "rt") as fh:
            decs = {d["id"]: d for d in (json.loads(x) for x in fh)}
        sp_hit = sp_n = mv_hit = mv_n = 0
        for p in sorted((ARCH / cp / "det").glob("worlds_*.jsonl")):
            for line in p.read_text().splitlines():
                w = json.loads(line)
                if not w.get("ok"):
                    continue
                d = decs[w["id"]]
                b = battles[d["battle"]]
                opp = b["p2"] if d["side"] == "p1" else b["p1"]
                true = [parse(e) for e in opp["team"].split("]")]
                kept = set(w["revealed"])
                hidden = {toid(f[1] or f[0]) for f in true} - kept
                for team in w["teams"][:8]:
                    if team is None:
                        continue
                    ws = [parse(e) for e in team.split("]")]
                    for tf, wf in zip(true, ws):
                        tid, wid = toid(tf[1] or tf[0]), toid(wf[1] or wf[0])
                        if tid in kept:
                            seen = set(w["revealed"][tid])
                            for tm, wm in zip(tf[4].split(","), wf[4].split(",")):
                                if tm in seen or (tm.startswith("hiddenpower") and "hiddenpower" in seen):
                                    continue
                                mv_n += 1
                                mv_hit += tm == wm
                        else:
                            sp_n += 1
                            sp_hit += wid in hidden
        out[cp] = {"hidden_slot_species_in_true_hidden_set": round(sp_hit / max(1, sp_n), 4), "hidden_slots": sp_n,
                   "redrawn_move_slot_equals_true": round(mv_hit / max(1, mv_n), 4), "redrawn_move_slots": mv_n}
        print(cp, out[cp])
    (HERE / "world_accuracy_read.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
