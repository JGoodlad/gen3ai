"""THE SUPPLEMENT BANK (§12.13.2): the supplement's ids as a `main.policy_spectrum.bank` directory, written exactly as
the pilot's `make_states.py` writes the registered bank, plus `states_supp_<cp>.json` (the per-state extras
`t1_common.supplement_states` reads). Model forward at the ROOT only; no playout; no outcome of any branch is read
(each trace's own recorded outcome is stamped, as make_states stamps it, for the re-weighting).

    run_pin.sh <end2|prod> 2 8 1h make_supp_bank.py

Steps: the supplement's battles are re-loaded from the eval traces and replayed (`core_events --trackers --obs`);
gate ① is checked on EVERY decision of each kept battle (make_states' `check_recorded`); every decision is stamped
(`bank.stamp`) and the supplement's rows are re-checked eligible, and their k / turn / material re-derived and
asserted equal to the census. The root forward is make_states' (`logits_and_values`, batch 1024, `inference_globals`);
the gap and V are compared with pool.npz's (reported; a difference above 1e-4 refuses), and a top-2 tie (gap < 1e-6)
refuses. TEETH: the same forward over the REGISTERED bank reproduces `states_<cp>.json`'s logits (6-decimal rounding).

Writes ~/gen3ai_archive/hidden_info_tier1/<cp>/bank_supp/ and states_supp_<cp>.json (this folder).
"""
import hashlib
import json
import time
from collections import Counter

import numpy as np
import t1_supp as S

T = S.T
FWD_TOL = 1e-4


def main():
    c = S.cp()
    S.check_plain_pin(c)
    t_wall = time.monotonic()
    import common as C  # the pilot's
    import make_states as MS
    import stage_facts as SF

    from agents.battle.core_replay import run_core
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.replay import _viewer_decisions, core_events_identity

    S.patch_categories()
    sup_path = S.supplement_path(c)
    sup = json.loads(sup_path.read_text())
    want_ids = {x: b for b, xs in sup["ids_per_bin"].items() for x in xs}
    cen_path = S.census_path(c)
    if S.sha256_file(cen_path) != sup["census_sha256"]:
        raise SystemExit("census.npz changed since the supplement was selected")
    cen = np.load(cen_path)
    cpos = {str(x): i for i, x in enumerate(cen["ids"])}
    pool = np.load(T.PILOT_ARCH / c / "pool.npz")
    ppos = {str(x): i for i, x in enumerate(pool["ids"])}
    pgap, pv = pool["gap"], pool["v"]                        # no outcome is read
    need_battles = {x.rsplit("#", 2)[0] for x in want_ids}

    cycles, picks, capture = S.eval_picks(c)
    reg_doc = json.loads((T.PILOT / f"states_{c}.json").read_text())
    if capture != reg_doc["capture"] or [x.name for x in cycles] != reg_doc["cycles"]:
        raise SystemExit("the capture block / cycles differ from states_<cp>.json's")
    picks = [(src, opp, p) for src, opp, p in picks if f"{src.run}/{src.step_dir}/{opp}/{p}" in need_battles]
    if len(picks) != len(need_battles):
        raise SystemExit(f"{len(need_battles)} supplement battles, {len(picks)} found in the traces")
    loaded = [B.load_battle(T.MODELS, src, opp, p) for src, opp, p in picks]
    raw = []
    for i in range(0, len(loaded), 32):
        raw.extend(run_core([b.recorded() for b, _ in loaded[i:i + 32]], trackers=True, obs=True))
    gate = {"battles_checked": 0, "decisions_checked": 0, "failures": [], "refused": []}
    off = C.clock_offset()
    turn_gate = {"decisions": 0, "agree": 0}
    chosen = []                                   # (battle index, ReplayDecision, stamped row)
    for bi, ((b, arr), res) in enumerate(zip(loaded, raw)):
        if not res["ok"]:
            gate["refused"].append(f"{b.battle_id}: {res.get('error')}")
            continue
        ti = 0 if b.traced_side == "p1" else 1
        decs = [d for d in _viewer_decisions(res, ti) if d.choice is not None]
        try:
            gate["decisions_checked"] += B.check_recorded(b, decs, arr)["decisions"]
            gate["battles_checked"] += 1
        except B.GateOneFailure as exc:
            gate["failures"].append(str(exc)[:300])
            continue
        dt = C.decode_turn(np.stack([d.row for d in decs]), off)
        turn_gate["decisions"] += len(decs)
        turn_gate["agree"] += int((np.array([d.turn for d in decs]) == dt).sum())
        rev = MS.revealed_counts(res, ti)
        nt = S.n_team(b)
        for d in decs:
            row = B.stamp(b, d, arr)
            if row["id"] not in want_ids:
                continue
            if not S.eligible(row):
                raise SystemExit(f"{row['id']}: not eligible on re-derivation")
            st = SF.parse_stage(res, ti, d.n, nt)
            ci = cpos[row["id"]]
            got = (int(rev[d.n]), int(row["turn"]), int(st["alive_ours"]), int(st["alive_theirs"]))
            exp = tuple(int(cen[f][ci]) for f in ("k", "turn", "alive_ours", "alive_theirs"))
            if got != exp:
                raise SystemExit(f"{row['id']}: k/turn/alive {got} != census {exp}")
            chosen.append((bi, d, row, st))
    if gate["failures"] or gate["refused"] or turn_gate["agree"] != turn_gate["decisions"]:
        raise SystemExit(f"gate ① / clock gate failed: {gate['failures'][:3]} {gate['refused'][:3]} {turn_gate}")
    if sorted(r["id"] for _, _, r, _ in chosen) != sorted(want_ids):
        raise SystemExit("the re-derived supplement rows are not the selected ids")
    chosen.sort(key=lambda t: t[2]["id"])

    rows = np.stack([d.row for _, d, _, _ in chosen]).astype(np.float32)
    masks = np.stack([d.mask for _, d, _, _ in chosen]).astype(np.uint8)
    reg_bank = B.load_bank(T.bank_path(c, "registered"))
    with inference_globals(C.threads()):
        model = C.load_model(c)
        out = logits_and_values(model, rows, masks, batch=1024)
        # TEETH: the registered bank's roots through the same forward reproduce states_<cp>.json's logits
        reg_rows, reg_masks, reg_ids = _registered_roots(reg_bank, run_core, _viewer_decisions)
        reg_out = logits_and_values(model, reg_rows, reg_masks, batch=1024)
    reg = reg_doc["states"]
    reg_dev = max(float(np.max(np.abs(np.round(reg_out[i, :11].astype(np.float64), 6) - np.array(reg[x]["logits"]))))
                  for i, x in enumerate(reg_ids))
    reg_v_dev = max(abs(round(float(reg_out[i, 11]), 6) - reg[x]["v_root"]) for i, x in enumerate(reg_ids))
    if len(reg_ids) != len(reg) or reg_dev > FWD_TOL or reg_v_dev > FWD_TOL:
        raise SystemExit(f"forward teeth: {len(reg_ids)} registered roots, max |logit dev| {reg_dev}, V {reg_v_dev}")
    lg = np.where(masks.astype(bool), out[:, :11].astype(np.float64), -np.inf)
    srt = -np.sort(-lg, axis=1)
    gap = srt[:, 0] - srt[:, 1]
    v = out[:, 11].astype(np.float64)
    pi = [ppos[r["id"]] for _, _, r, _ in chosen]
    gap_dev = float(np.max(np.abs(gap - pgap[pi])))
    v_dev = float(np.max(np.abs(v - pv[pi])))
    if gap_dev > FWD_TOL or v_dev > FWD_TOL:
        raise SystemExit(f"the root forward disagrees with pool.npz: gap {gap_dev}, V {v_dev}")
    if (gap < C.EDGE_EPS).any():
        raise SystemExit("a top-2 tie among the supplement (the selection excludes them)")

    extra, dec_rows = {}, []
    for i, (bi, d, r, st) in enumerate(chosen):
        r = dict(r)
        did = r["id"]
        lgi = out[i, :11].astype(np.float64)
        top = [int(x) for x in np.argsort(-np.where(masks[i].astype(bool), lgi, -np.inf), kind="stable")[:3]]
        k = int(cen["k"][cpos[did]])
        mat = int(st["alive_ours"] + st["alive_theirs"])
        b = loaded[bi][0]
        extra[did] = {"quintile": None, "gap": round(float(gap[i]), 6), "v_root": round(float(v[i]), 6),
                      "logits": [round(float(x), 6) for x in out[i, :11]], "top": top,
                      "unrevealed": 6 - k, "turn": r["turn"], "outcome": r["outcome"],
                      "cycle_opp": f"{b.source['step_dir']}/{b.source['opponent']}",
                      "n_legal": r["n_legal"], "battle": r["battle"],
                      "k": k, "k_bin": T.k_bin(k), "material": mat, "alive_ours": int(st["alive_ours"]),
                      "alive_theirs": int(st["alive_theirs"]), "turn_bin": T.turn_bin(r["turn"]),
                      "mat_bin": T.mat_bin(mat)}
        if extra[did]["k_bin"] != want_ids[did]:
            raise SystemExit(f"{did}: k bin {extra[did]['k_bin']} != the selection's {want_ids[did]}")
        dec_rows.append(r)
    keep_b = sorted({bi for bi, _, _, _ in chosen})
    kept = [loaded[i][0] for i in keep_b]
    manifest = {"schema": B.BANK_SCHEMA, "built_at_commit": T.CPS[c]["pin"],
                "study": "hidden_info_tier1_2026-10-13 (k-stratified supplement)",
                "selection_rule": sup["rule"], "supplement_sha256": S.sha256_file(sup_path),
                "gate1": {**{k_: v_ for k_, v_ in gate.items() if k_ != "failures"}, "failures": len(gate["failures"]),
                          "encoder": core_events_identity()},
                "counts": B.summarize(dec_rows), "content_sha256": B.content_sha(kept, dec_rows)}
    bank_dir = T.bank_path(c, "supp")
    B.write_bank(bank_dir, manifest, kept, dec_rows)
    B.load_bank(bank_dir)                                       # verifies the content sha on read
    doc = {"cp": c, "run": T.CPS[c]["run"], "pin": T.CPS[c]["pin"], "cycles": [x.name for x in cycles],
           "bank": str(bank_dir), "bank_content_sha256": manifest["content_sha256"], "gate1": manifest["gate1"],
           "turn_decode_gate": turn_gate, "census_sha256": sup["census_sha256"],
           "supplement_sha256": manifest["supplement_sha256"], "sampled": len(extra), "battles": len(kept),
           "per_k_bin": {b: sum(1 for e in extra.values() if e["k_bin"] == b) for b in T.K_BINS},
           "per_turn_bin": {b: sum(1 for e in extra.values() if e["turn_bin"] == b) for b in T.TURN_BINS},
           "per_mat_bin": {b: sum(1 for e in extra.values() if e["mat_bin"] == b) for b in T.MAT_BINS},
           "forward_check": {"vs_pool_max_abs_gap_dev": gap_dev, "vs_pool_max_abs_v_dev": v_dev,
                             "registered_roots": len(reg_ids), "registered_max_abs_logit_dev": reg_dev,
                             "registered_max_abs_v_dev": reg_v_dev, "tol": FWD_TOL},
           "outcome_counts_trace": dict(Counter(e["outcome"] for e in extra.values())),
           "cost": {"cpu_s_incl_children": round(C.cpu_seconds(), 1), "wall_s": round(time.monotonic() - t_wall, 1)},
           "capture": capture, "states": extra}
    (S.HERE / f"states_supp_{c}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k_: doc[k_] for k_ in ("sampled", "battles", "per_k_bin", "gate1", "forward_check",
                                             "bank_content_sha256", "cost")}), flush=True)


def _registered_roots(bank, run_core, viewer_decisions):
    """The registered bank's root rows / masks, re-derived by replaying its battles (the registered bank stores inputs,
    never obs). Returns (rows, masks, ids) in id order."""
    want = {d["id"]: d for d in bank.decisions}
    res = []
    for i in range(0, len(bank.battles), 32):
        res.extend(run_core([b.recorded() for b in bank.battles[i:i + 32]], trackers=True, obs=True))
    got = {}
    for b, r in zip(bank.battles, res):
        if not r["ok"]:
            raise SystemExit(f"registered battle {b.battle_id} refused: {r.get('error')}")
        ti = 0 if b.traced_side == "p1" else 1
        for d in viewer_decisions(r, ti):
            did = f"{b.battle_id}#{b.banked_side}#{d.n}"
            if did in want:
                if "".join(str(int(x)) for x in d.mask) != want[did]["mask"]:
                    raise SystemExit(f"{did}: registered mask differs on replay")
                got[did] = (d.row.copy(), d.mask.copy())
    ids = sorted(got)
    if hashlib.sha256("".join(ids).encode()).hexdigest() != hashlib.sha256("".join(sorted(want)).encode()).hexdigest():
        raise SystemExit("the registered roots were not all re-derived")
    return (np.stack([got[x][0] for x in ids]).astype(np.float32), np.stack([got[x][1] for x in ids]).astype(np.uint8),
            ids)


if __name__ == "__main__":
    main()
