"""DESCRIPTIVE (no outcome, no model): the gate's CONSTRUCTION-DRAW leak, v1 vs v2 builder (Amendment 8).

The engine draws a turn-0 gender for every mon of a gender-RATIO species whose packed set names none, so the
pre-root dice depend on how many such mons the opponent's never-seen slots hold. v1 accepted only worlds whose
never-seen mons cost the TRUE number of draws (the hidden-info study's A6: 98 % vs 53 % ungated); v2 places the
world's never-seen mons so the draws fall on the true draw slots and needs only m ≥ r (m = the world's ratio'd
never-seen mons, r = the true never-seen draws).

Per Smogon-prior world directory (``det`` = v1, ``det2`` = v2; the first 8 worlds of every state with 8 built), against
the UNGATED proposal (`determinize.sample_world`'s species draw, v1 path, 8 per state, a fixed seed):
``P(m == r)``, ``P(m >= r)``, ``E[m − r]``, and the share of never-seen world mons whose species is a TRUE hidden
species (the leak pulls it toward the truth). v2's prediction: accepted P(m == r) ≈ the proposal's
P(m == r | m ≥ r).

    run_at_pin.sh <cp> 1 4 30m gender_leak_read.py   →  gender_leak_read_<cp>.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import determinize as D  # noqa: E402


def main():
    c = C.cp()
    C.check_pin(c)
    from main.policy_spectrum.bank import load_bank

    od = C.out_dir(c)
    bank = load_bank(od / "bank")
    dmap = {d["id"]: d for d in bank.decisions}
    P = D.Priors(Path(os.environ["VDP_PIN"]) / "data" / "pokemon")

    def ratio(sid):
        return D.species_draws_gender(P, sid)

    out = {"cp": c}
    rng = np.random.default_rng(20261012)
    for dn in ("det", "det2"):
        acc = {"eq": 0, "ge": 0, "diff": [], "n": 0, "hit": 0, "hit_n": 0}
        prop = {"eq": 0, "ge": 0, "eq_given_ge": 0, "n": 0}
        for p in sorted((od / dn).glob("worlds_*.jsonl")):
            for w in C.read_rows(p):
                if not w.get("ok") or len(w["teams"]) < 8 or any(t is None for t in w["teams"][:8]):
                    continue
                d = dmap[w["id"]]
                b = bank.battles[bank.battle_index[d["battle"]]]
                team = (b.p2 if d["side"] == "p1" else b.p1)["team"]
                ents = team.split("]")
                tids = [D.toid(D.parse_set(e)[1] or D.parse_set(e)[0]) for e in ents]
                kept = w["revealed"]
                ns = [i for i, s in enumerate(tids) if s not in kept]
                if not ns:
                    continue
                r = len(D.gender_draw_slots(P, team) & set(ns))
                hidden = {tids[i] for i in ns}
                for t in w["teams"][:8]:
                    we = t.split("]")
                    sp = [D.toid(D.parse_set(we[i])[1] or D.parse_set(we[i])[0]) for i in ns]
                    m = sum(ratio(s) for s in sp)
                    acc["eq"] += m == r
                    acc["ge"] += m >= r
                    acc["diff"].append(m - r)
                    acc["n"] += 1
                    acc["hit"] += sum(s in hidden for s in sp)
                    acc["hit_n"] += len(sp)
                if dn == "det":
                    rv = {k: set(x) for k, x in kept.items()}
                    for _ in range(8):
                        t, _st = D.sample_world(rng, P, team, rv)
                        we = t.split("]")
                        m = sum(ratio(D.toid(D.parse_set(we[i])[1] or D.parse_set(we[i])[0])) for i in ns)
                        prop["eq"] += m == r
                        prop["ge"] += m >= r
                        prop["eq_given_ge"] += (m == r) and (m >= r)
                        prop["n"] += 1
        out[dn] = {"worlds": acc["n"], "P(m==r)": round(acc["eq"] / max(1, acc["n"]), 4),
                   "P(m>=r)": round(acc["ge"] / max(1, acc["n"]), 4),
                   "E[m-r]": round(float(np.mean(acc["diff"])) if acc["diff"] else 0.0, 4),
                   "never_seen_species_hit": round(acc["hit"] / max(1, acc["hit_n"]), 4)}
        if dn == "det":
            out["ungated_proposal"] = {"worlds": prop["n"], "P(m==r)": round(prop["eq"] / max(1, prop["n"]), 4),
                                       "P(m>=r)": round(prop["ge"] / max(1, prop["n"]), 4),
                                       "P(m==r | m>=r)": round(prop["eq_given_ge"] / max(1, prop["ge"]), 4)}
    (C.HERE / f"gender_leak_read_{c}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
