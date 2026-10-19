"""PREREGISTRATION §2: THE FROZEN BANK of the maturity battery. The pool, the census and the draw. NO model and NO
playout: the draw is k- and opponent-stratified (``maturity_bank.draw``), never by a network's logits, so the bank is the
same object for every checkpoint it is scored at. No outcome of any branch is read (each trace's own recorded outcome
is stamped, as the pilot's `make_states.py` stamps it).

    run_mat.sh pin 1 14 3h make_mat_bank.py [--bank interim|plateau] [--extension] [--pilot N]

``--bank plateau`` (amendment A1; PREREGISTRATION §2.5): the SAME rule on P*'s three newest cycles
(``mat_common.BANKS["plateau"]``) with its own salt, every battle of the interim bank EXCLUDED before the draw, written
to ``<archive>/plateau/`` and ``states_mat_x26_plateau.json``.

Steps:
  1. POOL: every trace of every roster opponent (bots and own-snapshot sentinels; never ``ext_``) of the COPIED eval
     cycles (`trigger_mat.py traces`; §2.1), replayed through the pin's ``core_events --trackers --obs``; gate ① per
     battle (> 1 % failing refuses); the clock-decode turn gate on every decision; eligible = a free move round, turn
     2-40, >= 3 legal (the pilot's `make_states.TURN_MIN / TURN_MAX / MIN_LEGAL`).
  2. CENSUS: k (`make_states.revealed_counts`), turn, material (`stage_facts.parse_stage`), and the STRATUM
     (``snapshot`` = a ``sentinel_*`` opponent, ``bot`` = a roster bot) for every pool decision.
  3. DRAW: ``maturity_bank.draw`` with 80 states per (k bin, stratum) cell (160 per k bin, 800 in all), salt
     ``hib_maturity_x26_v1:``, <= 1 per battle per k bin, <= 3 per battle; LEAN = the first 32 of each cell in
     selection order (320). A short cell STOPS (exit 4): the pre-registered extension (§2.4) is ``--extension``.
  ``--pilot N``: ALSO N pilot states that are NOT in the bank (the snapshot stratum, round-robin over the k bins in
     sha256('hib_maturity_pilot_v1:' + id) order, <= 1 per battle), written to ``<archive>/pilot_bank`` only.

Writes ``<archive>/bank/``, ``<archive>/census.npz`` and this folder's ``states_mat_x26.json``.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mat_common as E  # noqa: E402

E.add_paths()
MB = E.src_module("maturity_bank")
RULE = ("PREREGISTRATION §2 (design_hidden_info_binding.md §12.13.2 / §12.13.7): the eligible pool of X26's copied eval "
        "cycles {cycles} (every non-ext opponent; free move round, turn 2-40, >= 3 legal, gate ① passed); per k bin in the "
        "order 1-2, 3, 4, 5, 6 and within it per stratum (snapshot, then bot), candidates in sha256('{salt}' + id) order, "
        "at most 1 state per battle per k bin and 3 per battle in all, until the cell holds {full}; LEAN = the first "
        "{lean} of each cell in selection order{exclusion}")


def patch_categories() -> None:
    """make_states' patch: the core's choice tokens are SLOT-based at the pin, so the category is the token's verb."""
    import main.policy_spectrum.categories as CAT

    CAT.token_category = lambda t: t.split()[0]
    CAT.token_subtype = lambda t: None


def picks(cycles):
    from main.policy_spectrum import bank as B

    run = E.series().run
    out, capture = [], {}
    for cyc in cycles:
        d = E.traces_root() / run / "eval_traces" / cyc
        man = json.loads((d / "eval_manifest.json").read_text())
        if man.get("git_hash") != E.series().pin:
            raise SystemExit(f"{d}: recorded at {man.get('git_hash')}, not the pin")
        sel = man.get("selection", {}).get("opponents", {})
        src = B.Source(run, cyc, f"{E.SERIES_NAME}@{cyc}")
        for opp in man["opponents"]:
            if opp.startswith("ext_"):
                continue
            B.opponent_class(opp)
            s = sel.get(opp, {})
            capture[f"{cyc}/{opp}"] = {k: s.get(k) for k in ("battles_played", "battles_won", "battles_drawn",
                                                             "traces_written", "traces_won", "traces_drawn")}
            for p in B.trace_prefixes(d / opp):
                out.append((src, opp, p))
    return out, capture


def stratum(opp: str) -> str:
    from main.policy_spectrum import bank as B

    c = B.opponent_class(opp)
    if c == "pool_snapshot":
        return "snapshot"
    if c == "bot":
        return "bot"
    raise SystemExit(f"{opp}: opponent class {c} is neither stratum")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extension", action="store_true")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--bank", default="interim", choices=sorted(E.BANKS))
    a = ap.parse_args()
    bk = E.set_bank(a.bank)
    if a.pilot and bk.name != "interim":
        raise SystemExit("the pilot draws beside the interim bank only")
    if E.states_path().exists():
        raise SystemExit(f"{E.states_path()} exists: a frozen bank is never redrawn")
    E.check_pin()
    t_wall = time.monotonic()
    import common as C  # the pilot's (clock decode, cpu seconds)
    import make_states as MS
    import stage_facts as SF

    from agents.battle.core_replay import run_core
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.replay import _viewer_decisions, core_events_identity

    patch_categories()
    cycles = bk.cycles + ((bk.extension,) if a.extension else ())
    excluded_battles = set()
    for other in bk.exclude:                    # §2.5: every battle of an earlier bank is excluded
        excluded_battles |= {e["battle"] for e in json.loads(E.states_path(other).read_text())["states"].values()}
    pk, capture = picks(cycles)
    print(f"[mat bank] {len(pk)} traces from {list(cycles)}", flush=True)
    battles, arrays, strata = [], [], []
    for src, opp, p in pk:
        b, arr = B.load_battle(E.traces_root(), src, opp, p)
        battles.append(b)
        arrays.append(arr)
        strata.append(stratum(opp))
    raw = []
    for i in range(0, len(battles), 32):
        raw.extend(run_core([b.recorded() for b in battles[i:i + 32]], trackers=True, obs=True))
        print(f"[mat bank] replayed {min(len(battles), i + 32)}/{len(battles)}", flush=True)
    gate = {"battles_checked": 0, "decisions_checked": 0, "failures": [], "refused": []}
    pool = []                                   # (battle index, stamped row, k, stage)
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
        n_opp_sizes[len(B.team_species((b.p2 if b.banked_side == "p1" else b.p1)["team"]))] += 1
        nt = {who: len([e for e in (b.p1 if who == "p1" else b.p2)["team"].split("]") if e.strip()])
              for who in ("p1", "p2")}
        for d in decs:
            row = B.stamp(b, d, arr)
            turn_check.append((d.turn, d.row))
            if row["kind"] != "free" or not (MS.TURN_MIN <= d.turn <= MS.TURN_MAX) or row["n_legal"] < MS.MIN_LEGAL:
                continue
            pool.append((bi, row, int(rev[d.n]), SF.parse_stage(res, ti, d.n, nt)))
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
    targets = {(kb, s): E.FULL_PER_CELL for kb in MB.K_BINS for s in E.STRATA}
    n_excluded = sum(1 for c in cands if c.battle in excluded_battles)
    try:
        full = MB.draw(cands, targets, bk.salt, E.STRATA, E.PER_BATTLE_PER_BIN, E.PER_BATTLE,
                       exclude_battles=excluded_battles)
    except MB.BankShort as exc:
        avail = Counter((MB.k_bin(c.k), c.stratum) for c in cands)
        print(json.dumps({"short": str(exc), "candidates_per_cell": {f"{k}|{s}": n for (k, s), n in avail.items()}}),
              flush=True)
        raise SystemExit(4)
    lean = MB.nested_subset(full, E.LEAN_PER_CELL)
    lean_ids = set(MB.flat(lean))
    pos = {r["id"]: i for i, (_bi, r, _k, _st) in enumerate(pool)}
    extra, dec_rows = {}, []
    for (kb, s), ids in full.items():
        for rank, did in enumerate(ids):
            bi, r, k, st = pool[pos[did]]
            b = battles[bi]
            mat = int(st["alive_ours"] + st["alive_theirs"])
            extra[did] = {"k": k, "k_bin": kb, "turn": int(r["turn"]), "turn_bin": MB.turn_bin(int(r["turn"])),
                          "material": mat, "mat_bin": MB.mat_bin(mat), "alive_ours": int(st["alive_ours"]),
                          "alive_theirs": int(st["alive_theirs"]), "stratum": s, "rank_in_cell": rank,
                          "lean": did in lean_ids, "battle": r["battle"], "outcome": r["outcome"],
                          "n_legal": r["n_legal"], "cycle_opp": f"{b.source['step_dir']}/{b.source['opponent']}"}
            dec_rows.append(dict(r))
    dec_rows.sort(key=lambda r: r["id"])
    keep_b = sorted({pool[pos[x]][0] for x in extra})
    kept = [battles[i] for i in keep_b]
    excl = (f"; every battle of the {', '.join(bk.exclude)} bank ({len(excluded_battles)} battles) excluded before the "
            "draw" if bk.exclude else "")
    rule = RULE.format(cycles=list(cycles), salt=bk.salt, full=E.FULL_PER_CELL, lean=E.LEAN_PER_CELL, exclusion=excl)
    if set(extra) & {x for o in bk.exclude for x in json.loads(E.states_path(o).read_text())["states"]} or \
            {extra[x]["battle"] for x in extra} & excluded_battles:
        raise SystemExit("the drawn bank shares a state or a battle with an excluded bank")
    manifest = {"schema": B.BANK_SCHEMA, "built_at_commit": E.series().pin,
                "study": "hidden_info_maturity_x26_2026-10-18" + ("" if bk.name == "interim" else f" ({bk.name} bank)"), "selection_rule": rule,
                "gate1": {**{k_: v_ for k_, v_ in gate.items() if k_ != "failures"}, "failures": len(gate["failures"]),
                          "encoder": core_events_identity()},
                "counts": B.summarize(dec_rows), "content_sha256": B.content_sha(kept, dec_rows)}
    B.write_bank(E.bank_path(), manifest, kept, dec_rows)
    B.load_bank(E.bank_path())                                  # verifies the content sha on read
    cen = {"ids": np.array([r["id"] for _, r, _, _ in pool]), "battle": np.array([r["battle"] for _, r, _, _ in pool]),
           "k": np.array([k for _, _, k, _ in pool]), "turn": np.array([int(r["turn"]) for _, r, _, _ in pool]),
           "material": np.array([int(st["alive_ours"] + st["alive_theirs"]) for _, _, _, st in pool]),
           "stratum": np.array([strata[bi] for bi, _, _, _ in pool])}
    np.savez_compressed(E.census_path(), **cen)
    avail = Counter((MB.k_bin(c.k), c.stratum) for c in cands)
    by = lambda key, ids=None: dict(sorted(Counter(str(extra[x][key]) for x in (ids or extra)).items()))  # noqa: E731
    doc = {"series": E.SERIES_NAME, "run": E.series().run, "pin": E.series().pin, "rule": rule, "bank_name": bk.name,
           "salt": bk.salt, "excluded_banks": list(bk.exclude), "excluded_battles": len(excluded_battles),
           "excluded_candidates": n_excluded,
           "extension": a.extension, "cycles": list(cycles), "capture": capture, "traces": len(pk),
           "gate1": manifest["gate1"], "turn_decode_gate": turn_gate, "eligible_pool": len(pool),
           "candidates_per_cell": {f"{k}|{s}": avail[(k, s)] for k in MB.K_BINS for s in E.STRATA},
           "per_cell_full": E.FULL_PER_CELL, "per_cell_lean": E.LEAN_PER_CELL,
           "sampled": len(extra), "lean": len(lean_ids), "battles": len(kept),
           "bank": str(E.bank_path()), "bank_content_sha256": manifest["content_sha256"],
           "census_sha256": E.src_module("maturity_series").sha256_file(E.census_path()),
           "states_sha256": MB.ids_sha(extra), "lean_states_sha256": MB.ids_sha(lean_ids),
           "descriptive": {"full": {f: by(f) for f in ("stratum", "k_bin", "turn_bin", "mat_bin", "outcome")},
                           "lean": {f: by(f, sorted(lean_ids)) for f in ("stratum", "k_bin", "turn_bin", "mat_bin",
                                                                          "outcome")}},
           "cost": {"cpu_s_incl_children": round(C.cpu_seconds(), 1), "wall_s": round(time.monotonic() - t_wall, 1)},
           "states": extra}
    E.states_path().write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k_: doc[k_] for k_ in ("eligible_pool", "candidates_per_cell", "sampled", "lean", "battles",
                                             "states_sha256", "lean_states_sha256", "bank_content_sha256",
                                             "descriptive", "cost")}), flush=True)
    if a.pilot:
        pilot(a.pilot, pool, battles, strata, set(extra), B)


def pilot(n: int, pool, battles, strata, bank_ids: set, B) -> None:
    """N engineering states OUTSIDE the bank (never scored as trajectory output): the snapshot stratum, round-robin over
    the k bins in sha256(PILOT_SALT + id) order, <= 1 per battle."""
    by_bin = {kb: [] for kb in MB.K_BINS}
    for bi, r, k, _st in pool:
        if r["id"] not in bank_ids and strata[bi] == "snapshot":
            by_bin[MB.k_bin(k)].append((MB.order_key(E.PILOT_SALT, r["id"]), bi, r))
    for v in by_bin.values():
        v.sort(key=lambda t: t[0])
    used, chosen = set(), []
    while len(chosen) < n and any(by_bin.values()):
        for kb in MB.K_BINS:
            while by_bin[kb] and len(chosen) < n:
                _key, bi, r = by_bin[kb].pop(0)
                if r["battle"] in used:
                    continue
                used.add(r["battle"])
                chosen.append((bi, r))
                break
    rows = sorted((dict(r) for _bi, r in chosen), key=lambda r: r["id"])
    kept = [battles[i] for i in sorted({bi for bi, _ in chosen})]
    man = {"schema": B.BANK_SCHEMA, "built_at_commit": E.series().pin, "study": "maturity PILOT (not the bank)",
           "selection_rule": "pilot", "gate1": {}, "counts": B.summarize(rows), "content_sha256": B.content_sha(kept, rows)}
    out = E.ARCH / "pilot_bank"
    B.write_bank(out, man, kept, rows)
    (E.ARCH / "pilot_ids.txt").write_text("\n".join(r["id"] for r in rows) + "\n")
    print(f"[mat bank] pilot: {len(rows)} states outside the bank -> {out}", flush=True)
    assert not ({r["id"] for r in rows} & bank_ids)


if __name__ == "__main__":
    main()
