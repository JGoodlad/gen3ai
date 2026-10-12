"""A4 LEGALITY of every PLAYED world team (the first 8 worlds of every state with an ok det playout row, det/ and
det_b/), against the pin's own data, with the TRUE opponent teams of the same states as the control:

  species_clause       two sets of the same national-dex num
  multi_hp             > 1 Hidden Power in a set
  banned_engine_*      a move / item in ``determinize.BANNED_MOVES`` / ``BANNED_ITEMS`` (the pin engine's FAIL-LOUD list)
  not_learnable        a move the species cannot learn (``agents.gen3_data.learnset.is_legal``)
  hp_iv_mismatch       a typed Hidden Power whose type is not the one the set's IVs give
  ability_not_species  an ability outside the species' ability list (``gen3_species.json``, when it lists them)
  dup_move / n_moves   repeated moves / a set without 4 moves
  ev_total             EVs > 510 in total or > 255 in one stat
  format_spec:<rule>   ``agents.gen3_data.team_legality.validate_team`` (the gen3ou bans + clauses at the pin)

A world violation is attributed to the SAMPLED parts when the TRUE team of the same state does not carry the same
(rule, slot, detail) violation, else to the KEPT parts (the true team's own content). The Node team validator is NOT
run: the pin checkouts' ``deps/pokemon-showdown`` is not initialised, and running it would read the main checkout's
build (said in the output). PASS = zero violations in sampled parts.

    run_at_pin.sh <cp> 1 4 1h audit_a4_legality.py
→ here ``audit_a4_legality_<cp>.json``.
"""
import collections
import json
import os
from pathlib import Path

import audit_lib as A

C, D = A.C, A.D
J = 8
KNOWN = set()


def check_team(team, P, sp_json, learn, tl):
    """[(rule, slot, detail)] for one packed team."""
    out = []
    sets = [D.parse_set(e) for e in team.split("]")]
    nums = collections.defaultdict(list)
    for t, f in enumerate(sets):
        sid = D.toid(f[1] or f[0])
        mv = [m for m in f[4].split(",") if m]
        nums[int(P.species.get(sid, {}).get("num") or -1)].append(t)
        if sum(m.startswith("hiddenpower") for m in mv) > 1:
            out.append(("multi_hp", t, f"{sid}: {mv}"))
        for m in mv:
            if m in D.BANNED_MOVES:
                out.append(("banned_engine_move", t, f"{sid}:{m}"))
            base = "hiddenpower" if m.startswith("hiddenpower") else m
            if not learn.is_legal(sid, base):
                # the pin's learnset lacks pre-evolution / egg moves (the TRUE teams fail it too): a pair is
                # only counted when no real (bank) team runs it on that species either
                out.append(("not_in_pin_learnset" if (sid, base) in KNOWN else "not_learnable_unreferenced", t, f"{sid}:{m}"))
            if m.startswith("hiddenpower") and m != "hiddenpower":
                ty = D.hp_type_of(D.set_ivs(f))
                if m != f"hiddenpower{ty}":
                    out.append(("hp_iv_mismatch", t, f"{sid}:{m} ivs give {ty} ({f[8]!r})"))
        if len(set(mv)) != len(mv):
            out.append(("dup_move", t, f"{sid}: {mv}"))
        if len(mv) != 4:
            out.append(("n_moves", t, f"{sid}: {len(mv)} moves"))
        if D.toid(f[2]) in D.BANNED_ITEMS:
            out.append(("banned_engine_item", t, f"{sid}:{f[2]}"))
        abl = sp_json.get(sid, {}).get("abilities")
        if abl and f[3]:
            names = {D.toid(x) for x in (abl.values() if isinstance(abl, dict) else abl)}
            if D.toid(f[3]) not in names:
                out.append(("ability_not_species", t, f"{sid}:{f[3]} not in {sorted(names)}"))
        evs = [int(x) if x else 0 for x in (f[6].split(",") if f[6] else [])]
        if sum(evs) > 510 or any(e > 255 for e in evs):
            out.append(("ev_total", t, f"{sid}: {evs}"))
    for n, ts in nums.items():
        if len(ts) > 1:
            out.append(("species_clause", ts[1], f"num {n} in slots {ts}"))
    for p in tl.validate_team(tl.parse_packed(team)):
        out.append((f"format_spec:{p.rule}{'(ladder_only)' if p.ladder_only else ''}", p.slot, p.message[:160]))
    return out


def main():
    c = A.cp()
    from agents.gen3_data import learnset as learn
    from agents.gen3_data import team_legality as tl

    bk = A.bank(c)
    P = A.priors()
    for cc in ("end2", "prod"):           # every (species, move) a REAL team of either bank runs: known legal
        for bt in A.bank(cc).battles:
            for side in (bt.p1, bt.p2):
                for e in side["team"].split("]"):
                    f = D.parse_set(e)
                    for m in f[4].split(","):
                        if m:
                            KNOWN.add((D.toid(f[1] or f[0]), "hiddenpower" if m.startswith("hiddenpower") else m))
    sp_json = json.loads((Path(os.environ["VDP_PIN"]) / "data" / "pokemon" / "gen3_species.json").read_text())
    res = {"audit": "A4 legality of played world teams", "cp": c,
           "node_validator": "SKIPPED: the pin checkouts' deps/pokemon-showdown is not initialised (empty); running "
                             "validate_team.js would read the main checkout's build",
           "species_with_ability_lists": sum(1 for v in sp_json.values() if v.get("abilities")),
           "learnset_species": len(learn.raw()), "by_dir": {}}
    total_sampled = 0
    for det_dir in A.AUDIT_DIRS:
        W = A.world_rows(c, det_dir)
        ids = sorted(A.played_ids(c, det_dir))
        cnt, cnt_kept, ctrl = collections.Counter(), collections.Counter(), collections.Counter()
        ex = collections.defaultdict(list)
        n_teams = 0
        no_learnset = collections.Counter()
        for did in ids:
            d, b = A.decision(bk, did)
            opp = "p2" if d["side"] == "p1" else "p1"
            true_team = getattr(b, opp)["team"]
            truth = set(check_team(true_team, P, sp_json, learn, tl))
            for p in truth:
                ctrl[p[0]] += 1
            for j in range(J):
                team = W[did]["teams"][j]
                n_teams += 1
                for f in (D.parse_set(e) for e in team.split("]")):
                    sid = D.toid(f[1] or f[0])
                    if learn.get_legal_moves(sid) is None:
                        no_learnset[sid] += 1
                for p in check_team(team, P, sp_json, learn, tl):
                    sampled = p not in truth
                    (cnt if sampled else cnt_kept)[p[0]] += 1
                    key = f"{p[0]}|{'sampled' if sampled else 'kept'}"
                    if len(ex[key]) < 10:
                        ex[key].append({"id": did, "world": j, "slot": p[1], "detail": p[2]})
        total_sampled += sum(n for k, n in cnt.items() if k != "not_in_pin_learnset")
        res["by_dir"][det_dir] = {"states": len(ids), "world_teams": n_teams,
                                  "violations_in_SAMPLED_parts": dict(cnt), "violations_in_KEPT_parts": dict(cnt_kept),
                                  "control_true_teams_violations_per_state": dict(ctrl),
                                  "species_without_learnset_entry (unchecked)": dict(no_learnset), "examples": dict(ex)}
    res["violations_in_sampled_total"] = total_sampled
    res["pass"] = total_sampled == 0
    res["pass_rule"] = "zero sampled-part violations, not counting not_in_pin_learnset (a pair some real bank team runs)"
    A.write_json(f"audit_a4_legality_{c}.json", res)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in res["by_dir"].items()}, indent=1))


if __name__ == "__main__":
    main()
