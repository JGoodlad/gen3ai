"""THE BANK (``PREREGISTRATION.md`` §4): the maturity battery's draw (``maturity_bank.draw``, the rule of
``hidden_info_maturity_x26_2026-10-18/make_mat_bank.py``) over the generated off-pool cycles. NO model and NO playout:
the draw is k- and opponent-stratified, never by a network's logits.

    python -I make_bank.py --code <8fe4fb89 checkout> [--extension] [--smoke]

Steps (each a hard stop on failure):
  1. POOL: every trace of the registered cycles (``offpool_common.CYCLES``, both parts; ``--extension`` adds
     ``EXTENSION``). OFF-POOL GATE per battle: the OPPONENT's packed team is one of the 72 heldout72_hpfix teams as the
     eval core packed them, and the traced side's is a pool team (the trainee builder's own packing). One failing
     battle STOPS the build.
  2. Replay through ``core_events --trackers --obs`` at the code commit; gate ① per battle (> 1 % failing refuses); the
     clock-decode turn gate on every decision; eligible = a free move round, turn 2-40, >= 3 legal.
  3. CENSUS: k (opponent species seen), turn, material, outcome, the STRATUM (``sentinel_*`` -> snapshot; a roster bot
     -> bot) and the opponent's team id.
  4. DRAW: ``maturity_bank.draw`` with ``PER_CELL`` = 32 per (k bin, stratum) cell, salt ``STATES_SALT``, <= 1 state
     per battle per k bin, <= 3 per battle. A short cell exits 4 (the extension is ``--extension``).
Writes ``<archive>/bank/``, ``<archive>/census.npz`` and this folder's ``states_offpool.json``. ``--smoke`` reads the
engineering smoke's cycles instead and writes NOTHING but a printed census.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offpool_common as E  # noqa: E402

RULE = ("PREREGISTRATION §4 (the maturity battery's draw, make_mat_bank.py): the eligible pool of the generated "
        "off-pool cycles {cycles} (both parts, every trace; free move round, turn 2-40, >= 3 legal, gate ① passed); per k "
        "bin in the order 1-2, 3, 4, 5, 6 and within it per stratum (snapshot, then bot), candidates in sha256('{salt}' + "
        "id) order, at most 1 state per battle per k bin and 3 per battle in all, until the cell holds {n}")


def cycle_dirs(names, smoke: bool):
    """Every part of every named cycle, each COMPLETE (every opponent played its registered game count) or a stop."""
    root = E.ARCH / ("smoke" if smoke else "traces")
    out = []
    for name in names:
        for part in (("bots", "sentinels") if smoke else tuple(E.PARTS)):
            run = f"{name}_{part}"
            d = root / run / "eval_traces" / f"step_{E.PSTAR_STEP}"
            if not (d / "eval_manifest.json").exists():
                raise SystemExit(f"{d}: no generated cycle")
            man = json.loads((d / "eval_manifest.json").read_text())
            want = man["generated_by"]["games_per_opponent"] if smoke else E.GAMES[part]
            played = {o: v.get("battles_played") for o, v in man["selection"]["opponents"].items()}
            if man["generated_by"].get("seed") != (E.SMOKE_SEED if smoke else E.SEEDS[(name, part)]) or \
                    any(v != want for v in played.values()) or len(played) != len(man["opponents"]):
                raise SystemExit(f"{d}: INCOMPLETE or not the registered part: {played}, seed "
                                 f"{man['generated_by'].get('seed')}")
            out.append((root, run, d, name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--extension", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    code = E.use_code(Path(a.code))
    meas = code / "designs" / "research_state" / "measurements"
    for p in (meas / "value_disc_pilot_2026-10-11", meas / "hidden_info_by_reveal_count_2026-10-12"):
        sys.path.insert(1, str(p))
    if E.states_path(a.smoke).exists():
        raise SystemExit(f"{E.states_path()} exists: a frozen bank is never redrawn")
    t_wall = time.monotonic()
    import common as C  # the pilot's (clock decode, cpu seconds)
    import make_states as MS
    import stage_facts as SF

    from agents.battle.core_replay import run_core
    from main.hidden_info_study import maturity_bank as MB
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.replay import _viewer_decisions, core_events_identity
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder

    import main.policy_spectrum.categories as CAT

    CAT.token_category = lambda t: t.split()[0]       # make_mat_bank's patch: SLOT-based choice tokens
    CAT.token_subtype = lambda t: None

    texts = E.heldout_texts()
    order = sorted(texts)
    held_packed = list(Gen3Teambuilder([texts[t] for t in order]).packed_teams)
    held_id = {p: tid for p, tid in zip(held_packed, order)}
    if len(held_id) != 72:
        raise SystemExit("the heldout packing is not 72 distinct teams")
    pool_packed = set(Gen3Teambuilder(TeamLoader().get_all_teams()).packed_teams)
    if pool_packed & set(held_id):
        raise SystemExit("a heldout team IS a pool team (packed)")

    names = list(E.CYCLES) + ([E.EXTENSION] if a.extension else [])
    if a.smoke:
        names = ["smoke2"]
    dirs = cycle_dirs(names, a.smoke)
    battles, arrays, strata, meta = [], [], [], []
    offpool_gate = {"battles": 0, "opp_heldout": 0, "trainee_pool": 0}
    for root, run, d, cyc in dirs:
        man = json.loads((d / "eval_manifest.json").read_text())
        gen = man.get("generated_by") or {}
        for opp in man["opponents"]:
            src = B.Source(run, f"step_{E.PSTAR_STEP}", f"offpool@{run}")
            for p in B.trace_prefixes(d / opp):
                b, arr = B.load_battle(root, src, opp, p)
                summ = json.loads((d / opp / f"{p}_summary.json").read_text())["meta"]
                tr, ot = (b.p1, b.p2) if b.traced_side == "p1" else (b.p2, b.p1)
                offpool_gate["battles"] += 1
                if ot["team"] not in held_id:
                    raise SystemExit(f"OFF-POOL GATE: {b.battle_id}: the opponent's team is not a heldout72_hpfix team")
                offpool_gate["opp_heldout"] += 1
                if tr["team"] not in pool_packed:
                    raise SystemExit(f"OFF-POOL GATE: {b.battle_id}: the traced side's team is not a pool team")
                offpool_gate["trainee_pool"] += 1
                battles.append(b)
                arrays.append(arr)
                strata.append(E.stratum(opp))
                opp_step = (gen["sentinel_steps"][int(opp.split("_")[1])] if opp.startswith("sentinel_") else None)
                meta.append({"cycle": cyc, "part": run.split("_", 1)[1], "seed": gen.get("seed"),
                             "game": summ["trace_source"]["cycle"]["game"], "opp_step": opp_step,
                             "opp_team": held_id[ot["team"]], "result": summ["result"]})
    print(f"[offpool bank] {len(battles)} traces from {[r for _, r, _, _ in dirs]}; off-pool gate {offpool_gate}",
          flush=True)
    raw = []
    for i in range(0, len(battles), 32):
        raw.extend(run_core([b.recorded() for b in battles[i:i + 32]], trackers=True, obs=True))
        print(f"[offpool bank] replayed {min(len(battles), i + 32)}/{len(battles)}", flush=True)
    gate = {"battles_checked": 0, "decisions_checked": 0, "failures": [], "refused": []}
    pool, turn_check = [], []
    n_opp_sizes = Counter()
    for bi, (b, arr, res) in enumerate(zip(battles, arrays, raw)):
        if not res["ok"]:
            gate["refused"].append(f"{b.battle_id}: {res.get('error')}")
            continue
        ti = 0 if b.traced_side == "p1" else 1
        decs = [x for x in _viewer_decisions(res, ti) if x.choice is not None]
        try:
            gate["decisions_checked"] += B.check_recorded(b, decs, arr)["decisions"]
            gate["battles_checked"] += 1
        except B.GateOneFailure as exc:
            gate["failures"].append(str(exc)[:300])
            continue
        rev = MS.revealed_counts(res, ti)
        n_opp_sizes[len(B.team_species((b.p2 if b.banked_side == "p1" else b.p1)["team"]))] += 1
        nt = {who: len([e for e in (b.p1 if who == "p1" else b.p2)["team"].split("]") if e.strip()])
              for who in ("p1", "p2")}
        for x in decs:
            row = B.stamp(b, x, arr)
            turn_check.append((x.turn, x.row))
            if row["kind"] != "free" or not (MS.TURN_MIN <= x.turn <= MS.TURN_MAX) or row["n_legal"] < MS.MIN_LEGAL:
                continue
            pool.append((bi, row, int(rev[x.n]), SF.parse_stage(res, ti, x.n, nt)))
    n_fail = len(gate["failures"]) + len(gate["refused"])
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

    cands = [MB.Candidate(id=r["id"], battle=r["battle"], k=k, stratum=strata[bi]) for bi, r, k, _st in pool]
    avail = Counter((MB.k_bin(c.k), c.stratum) for c in cands)
    avail_battles = {cell: len({c.battle for c in cands if (MB.k_bin(c.k), c.stratum) == cell}) for cell in avail}
    census = {"eligible_pool": len(pool), "traces": len(battles),
              "candidates_per_cell": {f"{k}|{s}": avail[(k, s)] for k in MB.K_BINS for s in E.STRATA},
              "battles_per_cell": {f"{k}|{s}": avail_battles.get((k, s), 0) for k in MB.K_BINS for s in E.STRATA},
              "outcomes": {s: dict(Counter(m["result"] for m, st in zip(meta, strata) if st == s)) for s in E.STRATA}}
    if a.smoke:
        print(json.dumps({"SMOKE (not a bank)": census, "gate1": {k: (len(v) if isinstance(v, list) else v)
                                                                  for k, v in gate.items()}}, indent=1), flush=True)
    per_cell = 2 if a.smoke else E.PER_CELL                     # the smoke's engineering bank: 20 states
    targets = {(kb, s): per_cell for kb in MB.K_BINS for s in E.STRATA}
    try:
        sel = MB.draw(cands, targets, E.STATES_SALT, E.STRATA, E.PER_BATTLE_PER_BIN, E.PER_BATTLE)
    except MB.BankShort as exc:
        print(json.dumps({"short": str(exc), **census}), flush=True)
        raise SystemExit(4)
    pos = {r["id"]: i for i, (_bi, r, _k, _st) in enumerate(pool)}
    extra, dec_rows = {}, []
    for (kb, s), ids in sel.items():
        for rank, did in enumerate(ids):
            bi, r, k, st = pool[pos[did]]
            b = battles[bi]
            mat = int(st["alive_ours"] + st["alive_theirs"])
            extra[did] = {"k": k, "k_bin": kb, "turn": int(r["turn"]), "turn_bin": MB.turn_bin(int(r["turn"])),
                          "material": mat, "mat_bin": MB.mat_bin(mat), "alive_ours": int(st["alive_ours"]),
                          "alive_theirs": int(st["alive_theirs"]), "stratum": s, "rank_in_cell": rank,
                          "battle": r["battle"], "outcome": r["outcome"], "n_legal": r["n_legal"],
                          "cycle_opp": f"{b.source['run']}/{b.source['opponent']}", **meta[bi]}
            dec_rows.append(dict(r))
    dec_rows.sort(key=lambda r: r["id"])
    kept = [battles[i] for i in sorted({pool[pos[x]][0] for x in extra})]
    rule = RULE.format(cycles=names, salt=E.STATES_SALT, n=per_cell)
    manifest = {"schema": B.BANK_SCHEMA, "built_at_commit": E.CODE_SHA, "study": "search_offpool_bank_2026-10-20",
                "selection_rule": rule,
                "gate1": {**{k_: v_ for k_, v_ in gate.items() if k_ != "failures"}, "failures": len(gate["failures"]),
                          "encoder": core_events_identity()},
                "counts": B.summarize(dec_rows), "content_sha256": B.content_sha(kept, dec_rows)}
    bank_dir = E.ARCH / ("smoke" if a.smoke else "") / "bank"
    B.write_bank(bank_dir, manifest, kept, dec_rows)
    B.load_bank(bank_dir)                                       # verifies the content sha on read
    cen = {"ids": np.array([r["id"] for _, r, _, _ in pool]), "battle": np.array([r["battle"] for _, r, _, _ in pool]),
           "k": np.array([k for _, _, k, _ in pool]), "turn": np.array([int(r["turn"]) for _, r, _, _ in pool]),
           "material": np.array([int(st["alive_ours"] + st["alive_theirs"]) for _, _, _, st in pool]),
           "stratum": np.array([strata[bi] for bi, _, _, _ in pool])}
    cen_path = E.ARCH / ("smoke" if a.smoke else "") / "census.npz"
    np.savez_compressed(cen_path, **cen)
    by = lambda key: dict(sorted(Counter(str(extra[x][key]) for x in extra).items()))  # noqa: E731
    doc = {"rule": rule, "salt": E.STATES_SALT, "cycles": names, "extension": a.extension, "code": E.CODE_SHA,
           "offpool_gate": offpool_gate, "gate1": manifest["gate1"], "turn_decode_gate": turn_gate, **census,
           "per_cell": per_cell, "sampled": len(extra), "battles": len(kept),
           "opp_teams": len({extra[x]["opp_team"] for x in extra}),
           "bank": str(bank_dir), "bank_content_sha256": manifest["content_sha256"],
           "census_sha256": E.sha256_file(cen_path), "states_sha256": MB.ids_sha(extra),
           "descriptive": {f: by(f) for f in ("stratum", "k_bin", "turn_bin", "mat_bin", "outcome", "cycle_opp")},
           "cost": {"cpu_s_incl_children": round(C.cpu_seconds(), 1), "wall_s": round(time.monotonic() - t_wall, 1)},
           "states": extra}
    E.states_path(a.smoke).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k_: doc[k_] for k_ in ("eligible_pool", "candidates_per_cell", "battles_per_cell", "sampled",
                                             "battles", "opp_teams", "states_sha256", "bank_content_sha256",
                                             "descriptive", "cost")}), flush=True)


if __name__ == "__main__":
    main()
