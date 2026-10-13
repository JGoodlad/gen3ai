"""STAGE FACTS for the by-reveal-count read (README of this folder): per registered state, from the SEARCHED side's own
protocol stream up to its root decision (the TRUE prefix replayed at the checkpoint's pin, trackers on; the same replay
`root_facts.py` does), the MATERIAL still on the board: how many of each side's mons have fainted, and each side's
remaining-HP fraction (sum of its mons' HP fractions / team size). No model, no playout, no outcome.

    HP is tracked from the lines that carry it: `switch` / `drag` / `replace` (field 4), `-damage` / `-heal` / `-sethp`
    (field 3), `faint` (0). An UNSEEN opponent mon, or one never damaged, counts as full HP (it IS full in the truth: a
    mon that never entered has taken nothing). The opponent's HP arrives as a percent, ours as exact points; both are
    turned into fractions. Internal checks (reported, not assumed): every fainted mon ends at 0, no fraction leaves
    [0, 1], and the HP-tracked number of dead mons equals the `faint` count.

    run_at_pin.sh <cp> 1 6 30m stage_facts.py   →  stage_facts_<cp>.json (this folder)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "value_disc_pilot_2026-10-11"
sys.path.insert(0, str(PILOT))
import common as C  # noqa: E402


def frac(field: str):
    """'331/331 par' -> 1.0; '45/100' -> 0.45; '0 fnt' -> 0.0; anything else -> None."""
    if not field:
        return None
    parts = field.split(" ")
    tok = parts[0]
    if "fnt" in parts[1:] or tok == "0":
        return 0.0
    if "/" in tok:
        a, _, b = tok.partition("/")
        try:
            return max(0.0, min(1.0, int(a) / int(b)))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def parse_stage(res, viewer: int, n: int, n_team: dict):
    ent = [e for e in res["trackers"][viewer] if "obs" in e]
    after = int(ent[n]["after"])
    me, op = f"p{viewer + 1}", f"p{2 - viewer}"
    hp = {me: {}, op: {}}                     # side -> {mon label -> fraction}
    fainted = {me: set(), op: set()}
    bad = 0
    for side, lines in res["chunks"][: after + 1]:
        if side != viewer:
            continue
        for line in lines:
            p = line.split("|")
            if len(p) < 4:
                continue
            tag, who = p[1], p[2][:2]
            if who not in hp:
                continue
            label = p[2]
            if tag == "faint":
                fainted[who].add(label)
                hp[who][label] = 0.0
            elif tag in ("switch", "drag", "replace") and len(p) > 4:
                f = frac(p[4])
                bad += f is None
                hp[who][label] = 1.0 if f is None else f
            elif tag in ("-damage", "-heal", "-sethp"):
                f = frac(p[3])
                if f is not None:
                    hp[who][label] = f
                    if f == 0.0:
                        fainted[who].add(label)
                else:
                    bad += 1
    out = {"parse_unreadable_hp": int(bad)}
    for who, tag in ((me, "ours"), (op, "theirs")):
        nt = n_team[who]
        dead_hp = sum(1 for v in hp[who].values() if v == 0.0)
        seen_hp = sum(hp[who].values())
        out[f"fainted_{tag}"] = len(fainted[who])
        out[f"alive_{tag}"] = nt - len(fainted[who])
        out[f"hp_frac_{tag}"] = (seen_hp + (nt - len(hp[who]))) / nt
        out[f"n_team_{tag}"] = nt
        out[f"hp_dead_check_{tag}"] = int(dead_hp == len(fainted[who]))
    return out


def main():
    c = C.cp()
    C.check_pin(c)
    from agents.battle.core_replay import RecordedBattle, run_core
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index

    bank = load_bank(C.out_dir(c) / "bank")
    decs = sorted(bank.decisions, key=lambda x: x["id"])
    recs, meta = [], []
    for d in decs:
        b = bank.battles[bank.battle_index[d["battle"]]]
        at = cmd_index(b, d["side"], d["n"])
        recs.append(RecordedBattle(label=d["id"], format_id=b.format_id, seed=b.seed, p1=dict(b.p1), p2=dict(b.p2),
                                   commands=[list(x) for x in b.commands[:at]]))
        meta.append((d, b))
    res = []
    for i in range(0, len(recs), 32):
        res.extend(run_core(recs[i:i + 32], trackers=True, obs=True))
    out, bad = {}, []
    for (d, b), r in zip(meta, res):
        if not r["ok"]:
            out[d["id"]] = {"ok": False, "error": str(r.get("error"))[:300]}
            bad.append(d["id"])
            continue
        viewer = 0 if d["side"] == "p1" else 1
        n_team = {who: len([e for e in (b.p1 if who == "p1" else b.p2)["team"].split("]") if e.strip()])
                  for who in ("p1", "p2")}
        out[d["id"]] = {"ok": True, **parse_stage(r, viewer, d["n"], n_team)}
    ok = [v for v in out.values() if v.get("ok")]
    summary = {"cp": c, "states": len(out), "refused": len(bad),
               "unreadable_hp_fields": int(sum(v["parse_unreadable_hp"] for v in ok)),
               "hp_dead_mismatch": int(sum(2 - v["hp_dead_check_ours"] - v["hp_dead_check_theirs"] for v in ok)),
               "team_sizes": sorted({(v["n_team_ours"], v["n_team_theirs"]) for v in ok})}
    (HERE / f"stage_facts_{c}.json").write_text(json.dumps({"summary": summary, "states": out}, sort_keys=True) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
