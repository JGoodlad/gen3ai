"""STEP 0, THE STAGE CENSUS (§12.13.2): k, turn and material for EVERY decision of the checkpoint's eligible pool.
Replay only: no model, no playout, and no outcome is read (pool.npz is opened for its `ids` alone).

    run_pin.sh <end2|prod> 2 8 2h census.py [--workers 2]

The pool is re-derived exactly as the pilot's `make_states.py` derives it (`t1_supp` docstring), streaming 32-battle
chunks through `core_events` (`--workers` processes at a time; the results are kept in pick order). Per pool decision:

  k       = distinct opponent species the searched side has seen switch / be dragged in up to the decision
            (`make_states.revealed_counts`, = 6 - the pilot's `unrevealed`);
  turn    = the stamped turn (the latest `|turn|N` the searched side has seen; gated against the obs clock);
  material = alive_ours + alive_theirs (`stage_facts.parse_stage` on the searched side's own chunks up to its root).

TEETH (must hold with 0 mismatches, else the census is refused and nothing is written): the re-derived pool's ids
equal pool.npz's ids IN ORDER; gate ① and the clock-decode gate reproduce `states_<cp>.json`'s counts; on the
registered 600, k = 6 - unrevealed, turn = the registered turn, alive_ours / alive_theirs = `stage_facts_<cp>.json`'s.

Writes ~/gen3ai_archive/hidden_info_tier1/<cp>/census.npz and census_<cp>.json (this folder).
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import t1_supp as S

T = S.T


def counts(keys, order):
    c = Counter(keys)
    return {b: int(c.get(b, 0)) for b in order}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    c = S.cp()
    S.check_plain_pin(c)
    t_wall = time.monotonic()
    import common as C  # the pilot's
    import make_states as MS
    import stage_facts as SF

    from agents.battle.core_replay import run_core
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.replay import _viewer_decisions

    S.patch_categories()
    cycles, picks, _ = S.eval_picks(c)
    print(f"[census] {c}: {len(picks)} traces from {[x.name for x in cycles]}", flush=True)
    off = C.clock_offset()
    gate = {"battles_checked": 0, "decisions_checked": 0, "failures": [], "refused": []}
    turn_gate = {"decisions": 0, "agree": 0}
    cols = {k: [] for k in ("ids", "battle", "k", "turn", "alive_ours", "alive_theirs", "material", "n_legal",
                            "req_alive_ours", "opp_alive_stamp")}
    n_opp_seen = Counter()
    parts = [picks[i:i + 32] for i in range(0, len(picks), 32)]

    def load_and_run(part):
        loaded = [B.load_battle(T.MODELS, src, opp, p) for src, opp, p in part]
        res = run_core([b.recorded() for b, _ in loaded], trackers=True, obs=True)
        return loaded, res

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for g in range(0, len(parts), a.workers):
            for loaded, res in ex.map(load_and_run, parts[g:g + a.workers]):
                for (b, arr), r in zip(loaded, res):
                    if not r["ok"]:
                        gate["refused"].append(f"{b.battle_id}: {r.get('error')}")
                        continue
                    ti = 0 if b.traced_side == "p1" else 1
                    decs = [d for d in _viewer_decisions(r, ti) if d.choice is not None]
                    try:
                        gate["decisions_checked"] += B.check_recorded(b, decs, arr)["decisions"]
                        gate["battles_checked"] += 1
                    except B.GateOneFailure as exc:
                        gate["failures"].append(str(exc)[:300])
                        continue
                    rev = MS.revealed_counts(r, ti)
                    n_opp = len(B.team_species((b.p2 if b.banked_side == "p1" else b.p1)["team"]))
                    nt = S.n_team(b)
                    if decs:
                        dt = C.decode_turn(np.stack([d.row for d in decs]), off)
                        turn_gate["decisions"] += len(decs)
                        turn_gate["agree"] += int((np.array([d.turn for d in decs]) == dt).sum())
                    for d in decs:
                        row = B.stamp(b, d, arr)
                        if not S.eligible(row):
                            continue
                        n_opp_seen[n_opp] += 1
                        st = SF.parse_stage(r, ti, d.n, nt)
                        cols["ids"].append(row["id"])
                        cols["battle"].append(row["battle"])
                        cols["k"].append(int(rev[d.n]))
                        cols["turn"].append(int(row["turn"]))
                        cols["alive_ours"].append(int(st["alive_ours"]))
                        cols["alive_theirs"].append(int(st["alive_theirs"]))
                        cols["material"].append(int(st["alive_ours"] + st["alive_theirs"]))
                        cols["n_legal"].append(int(row["n_legal"]))
                        cols["req_alive_ours"].append(int(row["our_alive"]))
                        cols["opp_alive_stamp"].append(int(row["opp_alive"]))
            print(f"[census] {min(len(parts), g + a.workers)}/{len(parts)} chunks, pool {len(cols['ids'])}, "
                  f"{(time.monotonic() - t_wall) / 60:.1f} min", flush=True)
    cpu_s = C.cpu_seconds()

    # ---------------- TEETH ----------------
    reg_doc = json.loads((T.PILOT / f"states_{c}.json").read_text())
    reg = reg_doc["states"]
    sf = json.loads((T.BRC / f"stage_facts_{c}.json").read_text())["states"]
    pool_ids = np.load(T.PILOT_ARCH / c / "pool.npz")["ids"]          # ids ONLY (no outcome is read)
    ids = np.array(cols["ids"])
    fails = []
    if len(gate["failures"]) or len(gate["refused"]):
        fails.append(f"gate ①: {len(gate['failures'])} failures, {len(gate['refused'])} refused")
    for key in ("battles_checked", "decisions_checked"):
        if gate[key] != reg_doc["gate1"][key]:
            fails.append(f"gate ① {key} {gate[key]} != registered {reg_doc['gate1'][key]}")
    if turn_gate != reg_doc["turn_decode_gate"]:
        fails.append(f"clock-decode gate {turn_gate} != registered {reg_doc['turn_decode_gate']}")
    pool_identity = bool(len(ids) == len(pool_ids) and (ids == pool_ids).all())
    if not pool_identity:
        fails.append(f"pool ids differ from pool.npz ({len(ids)} vs {len(pool_ids)}; "
                     f"{len(set(ids) ^ set(pool_ids.tolist()))} in the symmetric difference)")
    if set(n_opp_seen) != {6}:
        fails.append(f"opponent team sizes {dict(n_opp_seen)} (k = 6 - unrevealed needs 6)")
    pos = {x: i for i, x in enumerate(cols["ids"])}
    mism = Counter()
    examples = []
    for did, e in reg.items():
        i = pos.get(did)
        if i is None:
            mism["missing"] += 1
            continue
        s = sf[did]
        checks = {"k": (cols["k"][i], 6 - e["unrevealed"]), "turn": (cols["turn"][i], e["turn"]),
                  "alive_ours": (cols["alive_ours"][i], s["alive_ours"]),
                  "alive_theirs": (cols["alive_theirs"][i], s["alive_theirs"]),
                  "material": (cols["material"][i], s["alive_ours"] + s["alive_theirs"])}
        for name, (got, want) in checks.items():
            if got != want:
                mism[name] += 1
                if len(examples) < 10:
                    examples.append({"id": did, "axis": name, "census": got, "registered": want})
    teeth = {"registered_states": len(reg), "mismatches": {k: int(mism.get(k, 0)) for k in
                                                           ("missing", "k", "turn", "alive_ours", "alive_theirs",
                                                            "material")},
             "examples": examples, "pool_ids_equal_in_order": pool_identity,
             "gate1": {k: (len(v) if isinstance(v, list) else v) for k, v in gate.items()},
             "turn_decode_gate": turn_gate}
    if sum(mism.values()):
        fails.append(f"registered-600 teeth: {dict(mism)}")
    teeth["pass"] = not fails
    if fails:
        print(json.dumps(teeth, indent=1), flush=True)
        raise SystemExit("CENSUS REFUSED: " + "; ".join(fails))

    # ---------------- write ----------------
    out = S.census_path(c)
    arrays = {k: np.array(v) for k, v in cols.items()}
    np.savez_compressed(out, **arrays)
    sha = S.sha256_file(out)
    k = arrays["k"]
    is_reg = np.array([x in reg for x in cols["ids"]])

    def table(mask):
        return {"n": int(mask.sum()),
                "k_bin": counts([T.k_bin(x) for x in k[mask]], T.K_BINS),
                "k_raw": {str(v): int((k[mask] == v).sum()) for v in range(7)},
                "turn_bin": counts([T.turn_bin(x) for x in arrays["turn"][mask]], T.TURN_BINS),
                "mat_bin": counts([T.mat_bin(x) for x in arrays["material"][mask]], T.MAT_BINS)}

    doc = {"cp": c, "run": T.CPS[c]["run"], "pin": T.CPS[c]["pin"], "cycles": [x.name for x in cycles],
           "traces": len(picks), "census_npz": str(out), "census_sha256": sha,
           "fields": sorted(arrays), "eligible_pool": table(np.ones(len(k), bool)), "registered": table(is_reg),
           "not_registered": table(~is_reg), "teeth": teeth,
           "cross_checks_descriptive": {
               "alive_ours_vs_request": int((arrays["alive_ours"] == arrays["req_alive_ours"]).sum()),
               "alive_theirs_vs_stamp_opp_alive": int((arrays["alive_theirs"] == arrays["opp_alive_stamp"]).sum()),
               "of": len(k)},
           "cost": {"cpu_s_incl_children": round(cpu_s, 1), "wall_s": round(time.monotonic() - t_wall, 1),
                    "workers": a.workers}}
    (S.HERE / f"census_{c}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({x: doc[x] for x in ("census_sha256", "eligible_pool", "registered", "cost")}), flush=True)
    print(json.dumps({"teeth": {k_: teeth[k_] for k_ in ("pass", "mismatches", "pool_ids_equal_in_order")}}),
          flush=True)


if __name__ == "__main__":
    main()
