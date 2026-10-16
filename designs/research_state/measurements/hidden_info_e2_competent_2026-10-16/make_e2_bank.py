"""E2 §1.2-§1.3: THE BANK. The generated games' CAPTURE, the eligible POOL and the DRAW, exactly as the pilot's
`make_states.py` draws the registered bank except for the source and the salt. No playout; no outcome of any branch is
read (each trace's own recorded outcome is stamped and, for the capture, read from the trace's name, as the live quota
reads it).

    run_e2.sh <end2|prod> 1 14 3h make_e2_bank.py [--extension]

Steps (PREREGISTRATION §1.2-§1.3):
  1. CAPTURE: per generated cycle (`gen_games.sh`; its manifest must be COMPLETE, at the pin, on the registered seed, from
     the live cycle's own snapshot) and sentinel, the battles ordered by sha256("hib_e2_capture_v1:" + battle id), the
     first 24 losses, 12 draws, 12 wins (`--extension`: the union with the extension batch, caps 48 / 24 / 24).
  2. POOL: every captured battle replayed (`core_events --trackers --obs`); gate ① per battle (> 1 % failing refuses); the
     clock-decode turn gate on every decision; eligible = a free move round, turn 2-40, >= 3 legal.
  3. ROOT FORWARD of the searched network (`final_model.zip`, `logits_and_values`, batch 1024): the top-2 logit gap; the
     quintile edges = the pool's own 20/40/60/80 % quantiles; rule-8 exclusion |gap - edge| < 1e-6 or gap < 1e-6.
  4. DRAW: 120 per quintile in sha256("hib_e2_states_v1:" + id) order, <= 2 per battle. A quintile that cannot be filled
     STOPS (exit 4; the extension rule then applies).
  TEETH: the same forward over the REGISTERED bank's 600 roots reproduces `states_<cp>.json`'s logits and V within 1e-4;
  no E2 battle or state id is a Tier-1 one.

Writes ~/gen3ai_archive/hidden_info_e2/<cp>/bank/ and pool.npz, and this folder's states_e2_<cp>.json.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_common as E  # noqa: E402

E.add_paths()
FWD_TOL = 1e-4


def check_plain_pin(c: str) -> None:
    kind, head, pdir = (os.environ.get(k, "") for k in ("HIB_E2_KIND", "HIB_E2_CHECKOUT_HEAD", "HIB_E2_CHECKOUT_DIR"))
    if kind != "pin" or head != E.CPS[c]["pin"] or pdir != E.CPS[c]["pin_dir"]:
        raise SystemExit(f"not the plain pin of {c}: kind={kind!r} head={head!r} dir={pdir!r}")
    import main

    if not str(Path(main.__file__).resolve()).startswith(pdir):
        raise SystemExit(f"main imported from {main.__file__}, not {pdir}")


def patch_categories() -> None:
    """make_states' patch: the core's choice tokens are SLOT-based at these pins, so the category is the token's verb."""
    import main.policy_spectrum.categories as CAT

    CAT.token_category = lambda t: t.split()[0]
    CAT.token_subtype = lambda t: None


def cycles(c: str, extension: bool):
    """[(shadow run name, step, cycle dir, manifest)] of the generated cycles (the extension batch's too when asked)."""
    out = []
    for step, seed in zip(E.CPS[c]["steps"], E.CPS[c]["seeds"]):
        for ext in ((False, True) if extension else (False,)):
            name = E.gen_run(c, step) + ("_ext" if ext else "")
            want_seed = seed + (E.EXTENSION_SEED_OFFSET if ext else 0)
            cyc = E.gen_root(c) / name / "eval_traces" / f"step_{step}"
            man = json.loads((cyc / "eval_manifest.json").read_text())
            g = man.get("generated_by") or {}
            live_snap = E.MODELS / E.CPS[c]["run"] / "eval_traces" / f"step_{step}" / "snapshot.zip"
            probs = []
            if g.get("complete") is not True:
                probs.append(f"incomplete ({g.get('battles_played')} of {g.get('battles_expected')})")
            if man.get("git_hash") != E.CPS[c]["pin"]:
                probs.append(f"recorded at {man.get('git_hash')}")
            if int(g.get("seed", -1)) != want_seed:
                probs.append(f"seed {g.get('seed')} != {want_seed}")
            if Path(g.get("checkpoint", "")).resolve() != live_snap.resolve():
                probs.append(f"checkpoint {g.get('checkpoint')} != the live cycle's snapshot")
            if g.get("eval_sentinel_greedy") is not True or g.get("games_per_opponent") != E.GAMES:
                probs.append(f"regime / games {g.get('eval_sentinel_greedy')} / {g.get('games_per_opponent')}")
            live_sent = sorted(int(s) for s in g.get("sentinel_steps", []))
            if not live_sent or max(live_sent) >= step:
                probs.append(f"sentinel steps {live_sent}")
            if probs:
                raise SystemExit(f"{cyc}: " + "; ".join(probs))
            out.append((name, step, cyc, man))
    return out


def capture(c: str, cyc_list, extension: bool):
    """(picks [(Source, opponent, prefix)], the capture table) under §1.2."""
    from main.policy_spectrum import bank as B

    mult = 2 if extension else 1
    caps = {k: v * mult for k, v in E.CAPTURE_CAPS.items()}
    cells = defaultdict(list)                  # (step, sentinel label) -> [(key, battle id, Source, opp, prefix, outcome)]
    for name, step, cyc, man in cyc_list:
        src = B.Source(name, cyc.name, f"{c}@e2/{name}")
        for opp in man["opponents"]:
            if not opp.startswith("sentinel_"):
                continue                       # the `random` games are a by-product, never drawn (§1.1)
            if B.opponent_class(opp) != "pool_snapshot":
                raise SystemExit(f"{opp}: not a pool snapshot")
            for p in B.trace_prefixes(cyc / opp):
                out = B.outcome_of(p)
                bid = f"{name}/{cyc.name}/{opp}/{p}"
                cells[(step, opp)].append((E.capture_key(bid), bid, src, opp, p, out))
    picks, table = [], {}
    for (step, opp), xs in sorted(cells.items()):
        xs.sort(key=lambda t: t[0])
        took = Counter()
        avail = Counter(t[5] for t in xs)
        for key, bid, src, o, p, out in xs:
            if took[out] < caps[out]:
                took[out] += 1
                picks.append((src, o, p))
        table[f"step_{step}/{opp}"] = {"available": dict(avail), "captured": dict(took)}
    return picks, table, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--extension", action="store_true")
    a = ap.parse_args()
    c = os.environ.get("VDP_CP")
    if c not in E.CPS:
        raise SystemExit("run through run_e2.sh <end2|prod> (VDP_CP unset)")
    check_plain_pin(c)
    t_wall = time.monotonic()
    import common as C  # the pilot's
    import make_states as MS
    import stage_facts as SF

    from agents.battle.core_replay import run_core
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.replay import _viewer_decisions, core_events_identity

    patch_categories()
    cyc_list = cycles(c, a.extension)
    picks, cap_table, caps = capture(c, cyc_list, a.extension)
    print(f"[e2 bank] {c}: {len(picks)} captured battles from {[n for n, *_ in cyc_list]}", flush=True)
    t1_battles = set()
    for st in (json.loads((E.PILOT / f"states_{c}.json").read_text())["states"],
               json.loads((E.T1 / f"states_supp_{c}.json").read_text())["states"]):
        t1_battles |= {e["battle"] for e in st.values()}

    battles, arrays = [], []
    for src, opp, p in picks:
        b, arr = B.load_battle(E.gen_root(c), src, opp, p)
        if b.battle_id in t1_battles:
            raise SystemExit(f"{b.battle_id}: a Tier-1 battle")
        battles.append(b)
        arrays.append(arr)
    raw = []
    for i in range(0, len(battles), 32):
        raw.extend(run_core([b.recorded() for b in battles[i:i + 32]], trackers=True, obs=True))
        print(f"[e2 bank] replayed {min(len(battles), i + 32)}/{len(battles)}", flush=True)
    gate = {"battles_checked": 0, "decisions_checked": 0, "failures": [], "refused": []}
    pool = []                                  # (battle index, decision, stamped row, k, stage)
    turn_check = []
    n_opp_sizes = Counter()
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
        rev = MS.revealed_counts(res, ti)
        n_opp = len(B.team_species((b.p2 if b.banked_side == "p1" else b.p1)["team"]))
        n_opp_sizes[n_opp] += 1
        nt = {who: len([e for e in (b.p1 if who == "p1" else b.p2)["team"].split("]") if e.strip()])
              for who in ("p1", "p2")}
        for d in decs:
            row = B.stamp(b, d, arr)
            turn_check.append((d.turn, d.row))
            if row["kind"] != "free" or not (MS.TURN_MIN <= d.turn <= MS.TURN_MAX) or row["n_legal"] < MS.MIN_LEGAL:
                continue
            st = SF.parse_stage(res, ti, d.n, nt)
            pool.append((bi, d, row, int(rev[d.n]), st))
    n_fail = len(gate["failures"]) + len(gate["refused"])
    if n_fail:
        print(f"[e2 bank] gate ① failures {len(gate['failures'])}, refused {len(gate['refused'])}", flush=True)
    if n_fail > 0.01 * max(1, len(battles)):
        raise SystemExit(f"gate ① failed on {n_fail} of {len(battles)} battles: {(gate['failures'] + gate['refused'])[:3]}")
    if set(n_opp_sizes) != {6}:
        raise SystemExit(f"opponent team sizes {dict(n_opp_sizes)} (k needs 6)")
    off = C.clock_offset()
    tt = np.array([t for t, _ in turn_check])
    dt = C.decode_turn(np.stack([r for _, r in turn_check]), off)
    turn_gate = {"decisions": int(len(tt)), "agree": int((tt == dt).sum())}
    if turn_gate["agree"] != turn_gate["decisions"]:
        raise SystemExit(f"the clock decode disagrees with |turn|: {turn_gate}")

    rows = np.stack([d.row for _, d, _, _, _ in pool]).astype(np.float32)
    masks = np.stack([d.mask for _, d, _, _, _ in pool]).astype(np.uint8)
    reg_doc = json.loads((E.PILOT / f"states_{c}.json").read_text())
    with inference_globals(C.threads()):
        model = C.load_model(c)
        out = logits_and_values(model, rows, masks, batch=1024)
        # TEETH: the registered bank's 600 roots through the same forward reproduce states_<cp>.json
        sys.path.insert(1, str(E.T1))
        import make_supp_bank as MSB

        reg_bank = B.load_bank(E.PILOT_ARCH / c / "bank")
        reg_rows, reg_masks, reg_ids = MSB._registered_roots(reg_bank, run_core, _viewer_decisions)
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
    edges = np.quantile(gap, [0.2, 0.4, 0.6, 0.8])
    near_edge = np.min(np.abs(gap[:, None] - edges[None, :]), axis=1) < E.EDGE_EPS
    tie = gap < E.EDGE_EPS
    quint = np.searchsorted(edges, gap, side="right")          # 0..4; strictly below an edge = the lower quintile
    excl = near_edge | tie
    order = sorted(range(len(pool)), key=lambda i: E.states_key(pool[i][2]["id"]))
    per_q, per_b, chosen = Counter(), Counter(), []
    for i in order:
        if excl[i]:
            continue
        q = int(quint[i])
        bid = pool[i][2]["battle"]
        if per_q[q] >= E.PER_QUINTILE or per_b[bid] >= E.PER_BATTLE:
            continue
        per_q[q] += 1
        per_b[bid] += 1
        chosen.append(i)
    short = {q + 1: per_q[q] for q in range(5) if per_q[q] < E.PER_QUINTILE}
    pool_per_q = [int(((quint == q) & ~excl).sum()) for q in range(5)]
    if short:
        print(json.dumps({"short_quintiles": short, "pool_per_quintile": pool_per_q, "battles": len(battles)}), flush=True)
        raise SystemExit(4)
    chosen.sort(key=lambda i: pool[i][2]["id"])
    reg_ids_all = set(reg) | set(json.loads((E.T1 / f"states_supp_{c}.json").read_text())["states"])
    extra, dec_rows = {}, []
    for i in chosen:
        bi, d, r, k, st = pool[i]
        r = dict(r)
        did = r["id"]
        if did in reg_ids_all:
            raise SystemExit(f"{did}: a Tier-1 state")
        lgi = out[i, :11].astype(np.float64)
        top = [int(x) for x in np.argsort(-np.where(masks[i].astype(bool), lgi, -np.inf), kind="stable")[:3]]
        b = battles[bi]
        mat = int(st["alive_ours"] + st["alive_theirs"])
        opp = b.source["opponent"]
        sent_steps = next(m for n, _, _, m in cyc_list if n == b.source["run"])["generated_by"]["sentinel_steps"]
        extra[did] = {"quintile": int(quint[i]) + 1, "gap": round(float(gap[i]), 6), "v_root": round(float(v[i]), 6),
                      "logits": [round(float(x), 6) for x in out[i, :11]], "top": top,
                      "unrevealed": 6 - k, "k": k, "turn": r["turn"], "outcome": r["outcome"],
                      "material": mat, "alive_ours": int(st["alive_ours"]), "alive_theirs": int(st["alive_theirs"]),
                      "cycle_opp": f"{b.source['step_dir']}/{opp}", "gen_run": b.source["run"],
                      "sentinel_step": int(sent_steps[int(opp.split('_')[1])]),
                      "n_legal": r["n_legal"], "battle": r["battle"]}
        dec_rows.append(r)
    keep_b = sorted({pool[i][0] for i in chosen})
    kept = [battles[i] for i in keep_b]
    rule = ("PREREGISTRATION §1.2-§1.3 (5881ee83): the generated cycles' sentinel battles, per (step, sentinel) the first "
            f"{caps['loss']} losses / {caps['draw']} draws / {caps['win']} wins in sha256('{E.CAPTURE_SALT}' + battle id) "
            "order; eligible free turns 2-40, >= 3 legal; gap quintiles of this pool; 120 per quintile in "
            f"sha256('{E.STATES_SALT}' + id) order, <= 2 per battle")
    manifest = {"schema": B.BANK_SCHEMA, "built_at_commit": E.CPS[c]["pin"], "study": "hidden_info_e2_competent_2026-10-16",
                "selection_rule": rule,
                "gate1": {**{k_: v_ for k_, v_ in gate.items() if k_ != "failures"}, "failures": len(gate["failures"]),
                          "encoder": core_events_identity()},
                "counts": B.summarize(dec_rows), "content_sha256": B.content_sha(kept, dec_rows)}
    bank_dir = E.bank_path(c)
    B.write_bank(bank_dir, manifest, kept, dec_rows)
    B.load_bank(bank_dir)                                    # verifies the content sha on read
    np.savez_compressed(E.cp_dir(c) / "pool.npz", ids=np.array([r["id"] for _, _, r, _, _ in pool]), gap=gap, v=v,
                        quintile=quint + 1, excluded=excl, k=np.array([k for _, _, _, k, _ in pool]))
    gen = {n: {"seed": m["generated_by"]["seed"], "sentinel_steps": m["generated_by"]["sentinel_steps"],
               "battles_played": m["generated_by"]["battles_played"], "checkpoint": m["generated_by"]["checkpoint"],
               "checkpoint_sha": m["generated_by"]["checkpoint_sha"], "near_ties": m["generated_by"].get("near_ties"),
               "wall_seconds": m["generated_by"].get("wall_seconds")} for n, _, _, m in cyc_list}
    by = lambda key: dict(sorted(Counter(str(e[key]) for e in extra.values()).items()))  # noqa: E731
    doc = {"cp": c, "run": E.CPS[c]["run"], "pin": E.CPS[c]["pin"], "rule": rule, "extension": a.extension,
           "generated": gen, "capture": cap_table, "captured_battles": len(battles), "gate1": manifest["gate1"],
           "turn_decode_gate": turn_gate, "eligible_pool": len(pool), "excluded_rule8": int(excl.sum()),
           "excluded_ties": int(tie.sum()), "quintile_edges_logits": [round(float(x), 6) for x in edges],
           "pool_per_quintile": pool_per_q, "sampled_per_quintile": [per_q[q] for q in range(5)],
           "sampled": len(chosen), "battles": len(kept), "bank": str(bank_dir),
           "bank_content_sha256": manifest["content_sha256"],
           "forward_teeth": {"registered_roots": len(reg_ids), "max_abs_logit_dev": reg_dev, "max_abs_v_dev": reg_v_dev,
                             "tol": FWD_TOL},
           "descriptive": {"k": by("k"), "outcome": by("outcome"), "sentinel_step": by("sentinel_step"),
                           "gen_run": by("gen_run")},
           "cost": {"cpu_s_incl_children": round(C.cpu_seconds(), 1), "wall_s": round(time.monotonic() - t_wall, 1)},
           "states_sha256": hashlib.sha256("".join(sorted(extra)).encode()).hexdigest(), "states": extra}
    E.states_path(c).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k_: doc[k_] for k_ in ("captured_battles", "eligible_pool", "pool_per_quintile", "sampled",
                                             "battles", "forward_teeth", "descriptive", "cost")}), flush=True)


if __name__ == "__main__":
    main()
