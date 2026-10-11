"""STATES (README §3): CP's own eval traces → a re-encoded, gate-① checked pool of eligible decisions →
CP's root forward (top-2 gap, V) → gap QUINTILES → 120 states per quintile in sha256 order, ≤ 2 per
battle. No playout, no outcome of any branch is read here (each trace's own recorded outcome is
stamped, for the re-weighting).

    run_at_pin.sh <cp> 4 24 2h make_states.py [--workers 4]

Writes the archive ``<cp>/bank/`` (a ``main.policy_spectrum.bank`` directory holding only the
sampled battles' sampled decisions) and ``<cp>/pool.npz`` (every eligible decision's gap / V /
quintile), and this folder's ``states_<cp>.json`` (committed: the edges, the counts, the ids).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

#: the run's LAST THREE eval-trace cycles (recorded at the run's own pin, by its 10M / 12M / 14M
#: snapshots); every trace of every roster opponent (the per-outcome quota is stamped, not re-drawn)
N_CYCLES = 3
TURN_MIN, TURN_MAX, MIN_LEGAL = 2, 40, 3


def revealed_counts(res: dict, viewer: int) -> list:
    """Per obs-bearing tracker entry (decision order): the DISTINCT opponent species the viewer has
    seen switch / drag in, up to that entry's chunk."""
    opp = f"p{2 - viewer}a: "
    chunks = res["chunks"]
    seen, out, k = set(), [], 0
    for entry in res["trackers"][viewer]:
        if "obs" not in entry:
            continue
        after = int(entry["after"])
        while k <= after:
            side, lines = chunks[k]
            if side == viewer:
                for line in lines:
                    parts = line.split("|")
                    if len(parts) > 3 and parts[1] in ("switch", "drag") and parts[2].startswith(opp):
                        seen.add(parts[3].split(",")[0].strip().lower())
            k += 1
        out.append(len(seen))
    return out


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

    # The core's choice tokens are SLOT-based at these pins ("move 1", "switch 3"), so the Lane S
    # move-category stamp (keyed by move id) cannot read them: the category is the token's verb here.
    import main.policy_spectrum.categories as CAT

    CAT.token_category = lambda t: t.split()[0]
    CAT.token_subtype = lambda t: None
    run = C.CPS[c]["run"]
    tr = C.MODELS / run / "eval_traces"
    cycles = sorted((p for p in tr.glob("step_*") if p.is_dir()), key=lambda p: int(p.name[5:]))[-N_CYCLES:]
    picks, capture = [], {}
    for cyc in cycles:
        man = json.loads((cyc / "eval_manifest.json").read_text())
        if man.get("git_hash") != C.CPS[c]["pin"]:
            raise SystemExit(f"{cyc}: recorded at {man.get('git_hash')}, not the pin")
        sel = man.get("selection", {}).get("opponents", {})
        src = B.Source(run, cyc.name, f"{c}@{cyc.name}")
        for opp in man["opponents"]:
            if opp.startswith("ext_"):
                continue
            B.opponent_class(opp)
            s = sel.get(opp, {})
            capture[f"{cyc.name}/{opp}"] = {k: s.get(k) for k in (
                "battles_played", "battles_won", "battles_drawn", "capture_rate_win", "capture_rate_loss",
                "capture_rate_draw", "traces_written", "traces_won", "traces_drawn")}
            for p in B.trace_prefixes(cyc / opp):
                picks.append((src, opp, p))
    print(f"[states] {c}: {len(picks)} traces from {[x.name for x in cycles]}", flush=True)
    battles, arrays = [], []
    for src, opp, p in picks:
        b, arr = B.load_battle(C.MODELS, src, opp, p)
        battles.append(b)
        arrays.append(arr)
    recs = [b.recorded() for b in battles]
    parts = [recs[i:i + 32] for i in range(0, len(recs), 32)]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        raw = [r for part in ex.map(lambda q: run_core(q, trackers=True, obs=True), parts) for r in part]
    gate = {"battles_checked": 0, "decisions_checked": 0, "failures": [], "refused": []}
    pool = []                      # (battle index, ReplayDecision, stamped row, revealed)
    turn_check = []
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
            turn_check.append((d.turn, d.row))
            if row["kind"] != "free" or not (TURN_MIN <= d.turn <= TURN_MAX) or row["n_legal"] < MIN_LEGAL:
                continue
            row["unrevealed"] = int(n_opp - rev[d.n])
            pool.append((bi, d, row))
    if gate["failures"] or gate["refused"]:
        print(f"[states] gate ① failures {len(gate['failures'])}, refused {len(gate['refused'])}", flush=True)
    if len(gate["failures"]) > 0.01 * max(1, len(battles)):
        raise SystemExit(f"gate ① failed on {len(gate['failures'])} battles: {gate['failures'][:3]}")
    off = C.clock_offset()
    tt = np.array([t for t, _ in turn_check])
    dt = C.decode_turn(np.stack([r for _, r in turn_check]), off)
    turn_gate = {"decisions": int(len(tt)), "agree": int((tt == dt).sum())}
    if turn_gate["agree"] != turn_gate["decisions"]:
        raise SystemExit(f"the clock decode disagrees with |turn|: {turn_gate}")
    rows = np.stack([d.row for _, d, _ in pool]).astype(np.float32)
    masks = np.stack([d.mask for _, d, _ in pool]).astype(np.uint8)
    with inference_globals(C.threads()):
        model = C.load_model(c)
        out = logits_and_values(model, rows, masks, batch=1024)
    lg = np.where(masks.astype(bool), out[:, :11].astype(np.float64), -np.inf)
    srt = -np.sort(-lg, axis=1)
    gap = srt[:, 0] - srt[:, 1]
    v = out[:, 11].astype(np.float64)
    edges = np.quantile(gap, [0.2, 0.4, 0.6, 0.8])
    near_edge = np.min(np.abs(gap[:, None] - edges[None, :]), axis=1) < C.EDGE_EPS
    tie = gap < C.EDGE_EPS
    quint = np.searchsorted(edges, gap, side="right")         # 0..4; strict < edge = lower quintile
    excl = near_edge | tie
    order = sorted(range(len(pool)), key=lambda i: C.sample_key(pool[i][2]["id"]))
    per_q, per_b, chosen = Counter(), Counter(), []
    for i in order:
        if excl[i]:
            continue
        q = int(quint[i])
        bid = pool[i][2]["battle"]
        if per_q[q] >= C.PER_QUINTILE or per_b[bid] >= C.PER_BATTLE:
            continue
        per_q[q] += 1
        per_b[bid] += 1
        chosen.append(i)
    chosen.sort(key=lambda i: pool[i][2]["id"])
    keep_b = sorted({pool[i][0] for i in chosen})
    dec_rows = []
    extra = {}
    for i in chosen:
        r = dict(pool[i][2])
        did = r["id"]
        lgi = out[i, :11].astype(np.float64)
        top = [int(x) for x in np.argsort(-np.where(masks[i].astype(bool), lgi, -np.inf), kind="stable")[:3]]
        extra[did] = {"quintile": int(quint[i]) + 1, "gap": round(float(gap[i]), 6), "v_root": round(float(v[i]), 6),
                      "logits": [round(float(x), 6) for x in out[i, :11]], "top": top,
                      "unrevealed": r.pop("unrevealed"), "turn": r["turn"], "outcome": r["outcome"],
                      "cycle_opp": f"{battles[pool[i][0]].source['step_dir']}/{battles[pool[i][0]].source['opponent']}",
                      "n_legal": r["n_legal"], "battle": r["battle"]}
        dec_rows.append(r)
    for _, _, r in pool:
        r.pop("unrevealed", None)
    od = C.out_dir(c)
    kept = [battles[i] for i in keep_b]
    manifest = {"schema": B.BANK_SCHEMA, "built_at_commit": C.CPS[c]["pin"], "pilot": "value_disc_pilot_2026-10-11",
                "selection_rule": "README §3 (every trace of the last three cycles; eligible free turns 2-40, "
                                  ">= 3 legal; gap quintiles; 120 per quintile in sha256 order, <= 2 per battle)",
                "gate1": {**{k: v_ for k, v_ in gate.items() if k != "failures"}, "failures": len(gate["failures"]),
                          "encoder": core_events_identity()},
                "counts": B.summarize(dec_rows), "content_sha256": B.content_sha(kept, dec_rows)}
    B.write_bank(od / "bank", manifest, kept, dec_rows)
    np.savez_compressed(od / "pool.npz", ids=np.array([r["id"] for _, _, r in pool]), gap=gap, v=v,
                        quintile=quint + 1, excluded=excl, outcome=np.array([r["outcome"] for _, _, r in pool]))
    by_q = defaultdict(list)
    for did, e in extra.items():
        by_q[e["quintile"]].append(did)
    doc = {"cp": c, "run": run, "pin": C.CPS[c]["pin"], "cycles": [x.name for x in cycles],
           "traces": len(picks), "gate1": manifest["gate1"], "turn_decode_gate": turn_gate,
           "eligible_pool": len(pool), "excluded_rule8": int(excl.sum()), "excluded_ties": int(tie.sum()),
           "quintile_edges_logits": [round(float(e), 6) for e in edges],
           "pool_per_quintile": [int(((quint == q) & ~excl).sum()) for q in range(5)],
           "sampled_per_quintile": [per_q[q] for q in range(5)], "sampled": len(chosen),
           "battles": len(keep_b), "bank_content_sha256": manifest["content_sha256"],
           "capture": capture, "states": extra}
    (C.HERE / f"states_{c}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: doc[k] for k in ("eligible_pool", "excluded_rule8", "quintile_edges_logits",
                                          "pool_per_quintile", "sampled_per_quintile", "battles",
                                          "turn_decode_gate")}), flush=True)


if __name__ == "__main__":
    main()
