"""PHASE 3, step 1 (the design doc's §7.10 Amendment 7): the Q head's TRAINING states. Every eval-trace cycle
of the checkpoint's run (all seven, each recorded at the run's pin), every trace, MINUS every battle of the
registered evaluation bank (the 600 Phase-1 states' battles): a re-encoded, gate-① checked pool of eligible
decisions (Phase 1's filter: a free move round, turn 2–40, ≥ 3 legal actions), at most ``PER_BATTLE_Q`` per
battle in ``sha256("vdp_qstates_v1:" + id)`` order. No playout and no outcome is read here.

    run_at_pin.sh end2 4 24 2h make_qstates.py [--workers 4]

Writes the archive ``<cp>/qbank/`` (a ``main.policy_spectrum.bank`` directory) and ``<cp>/qstates.json``
(per state: battle, root logits, V, top, turn, unrevealed count; the counts), and this folder's compact
``qstates_<cp>.json`` (counts and the disjointness check only).
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from make_states import MIN_LEGAL, TURN_MAX, TURN_MIN, revealed_counts  # noqa: E402

PER_BATTLE_Q = 10
SALT_Q = "vdp_qstates_v1:"


def key_q(did: str) -> str:
    return hashlib.sha256((SALT_Q + did).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    c = C.cp()
    C.check_pin(c)
    from agents.battle.core_replay import run_core
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.replay import _viewer_decisions, core_events_identity

    import main.policy_spectrum.categories as CAT

    CAT.token_category = lambda t: t.split()[0]
    CAT.token_subtype = lambda t: None
    od = C.out_dir(c)
    evalbank = B.load_bank(od / "bank")
    eval_battles = {b.battle_id for b in evalbank.battles}
    run = C.CPS[c]["run"]
    tr = C.MODELS / run / "eval_traces"
    cycles = sorted((p for p in tr.glob("step_*") if p.is_dir()), key=lambda p: int(p.name[5:]))
    picks = []
    for cyc in cycles:
        man = json.loads((cyc / "eval_manifest.json").read_text())
        if man.get("git" + "_hash") != C.CPS[c]["pin"]:
            raise SystemExit(f"{cyc}: recorded at another commit, not the pin")
        src = B.Source(run, cyc.name, f"{c}@{cyc.name}")
        for opp in man["opponents"]:
            if opp.startswith("ext_"):
                continue
            B.opponent_class(opp)
            for p in B.trace_prefixes(cyc / opp):
                picks.append((src, opp, p))
    battles, arrays = [], []
    dropped_eval = 0
    for src, opp, p in picks:
        b, arr = B.load_battle(C.MODELS, src, opp, p)
        if b.battle_id in eval_battles:
            dropped_eval += 1
            continue
        battles.append(b)
        arrays.append(arr)
    print(f"[qstates] {c}: {len(picks)} traces from {len(cycles)} cycles; {dropped_eval} are evaluation battles "
          f"(dropped); {len(battles)} kept", flush=True)
    recs = [b.recorded() for b in battles]
    parts = [recs[i:i + 32] for i in range(0, len(recs), 32)]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        raw = [r for part in ex.map(lambda q: run_core(q, trackers=True, obs=True), parts) for r in part]
    gate = {"battles_checked": 0, "decisions_checked": 0, "failures": [], "refused": []}
    pool = []
    for bi, (b, arr, res) in enumerate(zip(battles, arrays, raw)):
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
        rev = revealed_counts(res, ti)
        n_opp = len(B.team_species((b.p2 if b.banked_side == "p1" else b.p1)["team"]))
        for d in decs:
            row = B.stamp(b, d, arr)
            if row["kind"] != "free" or not (TURN_MIN <= d.turn <= TURN_MAX) or row["n_legal"] < MIN_LEGAL:
                continue
            row["unrevealed"] = int(n_opp - rev[d.n])
            pool.append((bi, d, row))
    print(f"[qstates] gate ①: {gate['decisions_checked']} decisions, {len(gate['failures'])} failures, "
          f"{len(gate['refused'])} refused; eligible {len(pool)}", flush=True)
    if len(gate["failures"]) > 0.01 * max(1, len(battles)):
        raise SystemExit(f"gate ① failed on {len(gate['failures'])} battles: {gate['failures'][:3]}")
    order = sorted(range(len(pool)), key=lambda i: key_q(pool[i][2]["id"]))
    per_b, chosen = Counter(), []
    for i in order:
        bid = pool[i][2]["battle"]
        if per_b[bid] >= PER_BATTLE_Q:
            continue
        per_b[bid] += 1
        chosen.append(i)
    chosen.sort(key=lambda i: pool[i][2]["id"])
    rows = np.stack([pool[i][1].row for i in chosen]).astype(np.float32)
    masks = np.stack([pool[i][1].mask for i in chosen]).astype(np.uint8)
    with inference_globals(C.threads()):
        model = C.load_model(c)
        out = logits_and_values(model, rows, masks, batch=1024)
    keep_b = sorted({pool[i][0] for i in chosen})
    dec_rows, extra = [], {}
    for j, i in enumerate(chosen):
        r = dict(pool[i][2])
        lgi = out[j, :11].astype(np.float64)
        top = [int(x) for x in np.argsort(-np.where(masks[j].astype(bool), lgi, -np.inf), kind="stable")[:3]]
        extra[r["id"]] = {"v_root": round(float(out[j, 11]), 6), "logits": [round(float(x), 6) for x in lgi],
                          "top": top, "unrevealed": r.pop("unrevealed"), "turn": r["turn"], "outcome": r["outcome"],
                          "n_legal": r["n_legal"], "battle": r["battle"],
                          "cycle_opp": f"{battles[pool[i][0]].source['step_dir']}/{battles[pool[i][0]].source['opponent']}"}
        dec_rows.append(r)
    kept = [battles[i] for i in keep_b]
    if {b.battle_id for b in kept} & eval_battles:
        raise SystemExit("a training battle is an evaluation battle")
    manifest = {"schema": B.BANK_SCHEMA, "built_at_commit": C.CPS[c]["pin"], "pilot": "value_disc_pilot_2026-10-11",
                "selection_rule": "Amendment 7 (Phase 3 training states: every trace of every cycle minus the "
                                  "evaluation bank's battles; eligible free turns 2-40, >= 3 legal; <= 10 per battle "
                                  "in sha256 order)",
                "gate1": {**{k: v_ for k, v_ in gate.items() if k != "failures"}, "failures": len(gate["failures"]),
                          "encoder": core_events_identity()},
                "counts": B.summarize(dec_rows), "content_sha256": B.content_sha(kept, dec_rows)}
    B.write_bank(od / "qbank", manifest, kept, dec_rows)
    doc = {"cp": c, "run": run, "pin": C.CPS[c]["pin"], "cycles": [x.name for x in cycles], "traces": len(picks),
           "eval_battles_dropped": dropped_eval, "gate1": manifest["gate1"], "eligible_pool": len(pool),
           "sampled": len(chosen), "battles": len(keep_b), "bank_content_sha256": manifest["content_sha256"],
           "disjoint_from_eval_bank": True}
    (od / "qstates.json").write_text(json.dumps({**doc, "states": extra}, sort_keys=True) + "\n")
    (C.HERE / f"qstates_{c}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps(doc), flush=True)


if __name__ == "__main__":
    main()
