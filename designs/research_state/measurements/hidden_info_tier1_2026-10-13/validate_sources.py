"""VALIDATION of the three world SOURCES (FN, POOLSETS, POOL) on a k-spread subset of the registered states (HEAD).

    run_head.sh 6 3h validate_sources.py <cp> [--n 12] [--fresh 4] [--arms F,FN,FNfresh,POOLSETS,POOL]

Per state (resumable; one row per state in ``~/gen3ai_archive/hidden_info_tier1/<cp>/validate_sources.jsonl``):
F built first on a fresh cache; FN on F's cache (the harness's shared case); on the first ``--fresh`` states FN AGAIN on
a FRESH cache (F in another process): its seen-mon screen must equal F's by determinism (the seen mons' item, ability,
nature, EVs, IVs of every world equal the shared-cache FN's, and every sampler's samples equal F's); POOLSETS on its
own cache; POOL. Every built world set gets the audit suite's ``regate``, ``legality``, ``genders``, ``playout_open``
(``main.info_audit.checks``); POOL's strict gate is checked against the builder's own strict verdicts on every team F
gated (pass AND fail). Descriptive truth-match rates (the pilot's `move_accuracy_read.py` definitions): never-seen
``species_hit``, seen mons' unseen-move ``set_recall``, ``new_set_recall`` on species hits; POOL: the true team drawn.
"""
from __future__ import annotations

import argparse
import hashlib
import resource
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_common as T  # noqa: E402


def cpu() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def pick_states(beliefs: dict, n: int) -> list:
    """A deterministic k-spread subset: by sha256 order within each k bin (k from the belief row's seen mons)."""
    want = {"1-2": 3, "3": 2, "4": 2, "5": 2, "6": 3}
    scale = n / sum(want.values())
    by: dict = {}
    for did, r in beliefs.items():
        k = len(r["moves"]["seen"])
        if 1 <= k <= 6:
            by.setdefault(T.k_bin(k), []).append(did)
    out = []
    for b, m in want.items():
        ids = sorted(by.get(b, []), key=lambda x: hashlib.sha256(("hib_t1_validate:" + x).encode()).hexdigest())
        out += ids[: max(1, round(m * scale))]
    return out


def seen_hp(tm: str, seen: set) -> bool:
    return tm in seen or (tm.startswith("hiddenpower") and "hiddenpower" in seen)


def truth_match(root, teams) -> dict:
    from main.value_search import worlds as W

    acc = {k: [0, 0] for k in ("species_hit", "set_recall", "new_set_recall")}
    true = [W.parse_set(e) for e in root.team.split("]")]
    tsets = {W.toid(f[1] or f[0]): W.typed_moves(f) for f in true}
    kept = {s: set(m) for s, m in root.kept.items()}
    hidden = set(tsets) - set(kept)
    for team in teams:
        ws = {W.toid(f[1] or f[0]): f for f in (W.parse_set(e) for e in team.split("]"))}
        for sid, f in ws.items():
            wset = set(W.typed_moves(f))
            if sid in kept:
                un = [m for m in tsets[sid] if not seen_hp(m, kept[sid])]
                acc["set_recall"][0] += sum(m in wset for m in un)
                acc["set_recall"][1] += len(un)
            else:
                acc["species_hit"][0] += sid in hidden
                acc["species_hit"][1] += 1
                if sid in hidden:
                    acc["new_set_recall"][0] += sum(m in wset for m in tsets[sid])
                    acc["new_set_recall"][1] += len(tsets[sid])
    return acc


def audits(root, w, ws) -> dict:
    from main.info_audit import checks as C

    out = {}
    for v in (C.regate(root, w), C.legality(root, w), C.genders(root, w), C.playout_open(root, w, ws)):
        out[v["check"]] = {"pass": v["pass"], **{k: v[k] for k in v if k not in ("check", "pass", "examples")},
                           "examples": v.get("examples", [])[:2]}
    return out


def seen_nonmove(root, teams) -> list:
    from main.value_search import worlds as W

    out = []
    for t in teams:
        row = {}
        for e in t.split("]"):
            f = W.parse_set(e)
            sid = W.toid(f[1] or f[0])
            if sid in root.kept:
                row[sid] = [f[2], f[3], f[5], f[6], f[8]]
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cp")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--fresh", type=int, default=4)
    ap.add_argument("--arms", default="F,FN,FNfresh,POOLSETS,POOL")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--tag", default="", help="output validate_sources<tag>.jsonl (a re-run of some arms)")
    a = ap.parse_args()
    arms = a.arms.split(",")
    from agents.battle.core_replay import run_core
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import worlds as W
    from main.value_search.world_draws import Prior

    import fn_worlds as FNW
    import pool_teams as PT
    import pool_worlds as PW
    import poolsets_worlds as PSW

    beliefs = FNW.load_beliefs(a.cp, "registered")
    ids = a.ids.split(",") if a.ids else pick_states(beliefs, a.n)
    bank = load_bank(T.bank_path(a.cp, "registered"))
    decs = {d["id"]: d for d in bank.decisions}
    out = T.cp_dir(a.cp) / f"validate_sources{a.tag}.jsonl"
    done = {r["id"] for r in T.read_rows(out)}
    base = Prior()
    wsF = W.WorldSampler(prior=base, max_tries=60, reveal="off")
    pool = PT.load_pool()
    PS = PSW.poolsets_sampler(max_tries=60, base=base, pool=pool)
    POOL = PW.PoolSampler(max_tries=60, pool=pool, ws=wsF)
    print(f"[validate] {a.cp}: {len(ids)} states ({len(done)} done); pool {len(pool.teams)} teams sha {pool.sha256[:12]}",
          flush=True)
    with open(out, "a") as fh:
        for i, did in enumerate(ids):
            if did in done:
                continue
            d = decs[did]
            b = bank.battles[bank.battle_index[d["battle"]]]
            at = cmd_index(b, d["side"], d["n"])
            log = to_log(b)
            root = W.prepare_root(log, at, d["side"], did)
            row: dict = {"id": did, "k": len(root.kept), "k_net": len(beliefs[did]["moves"]["seen"]), "arms": {}}
            cacheF: dict = {}
            got: dict = {}
            for arm in arms:
                if arm == "FNfresh" and i >= a.fresh:
                    continue
                c0, t0 = cpu(), time.time()
                rec: dict = {}
                try:
                    if arm == "F":
                        w = wsF.sample(log, at, d["side"], T.J, key=did, screen_cache=cacheF)
                        sw = wsF
                    elif arm == "FN":
                        S = FNW.fn_sampler(beliefs[did], max_tries=60, base=base)
                        w = S.sample(log, at, d["side"], T.J, key=did, screen_cache=cacheF)
                        sw = S.ws
                        rec["fn"] = {k: w.fn[k] for k in ("counts", "draw_counts", "row_check")}
                        rec["fn"]["screen_rebuilt"] = len(w.fn["screen"]["rebuilt_on_F_parameters"])
                    elif arm == "FNfresh":
                        S = FNW.fn_sampler(beliefs[did], max_tries=60, base=base)
                        fresh: dict = {}
                        w = S.sample(log, at, d["side"], T.J, key=did, screen_cache=fresh)
                        sw = S.ws
                        rb = w.fn["screen"]["rebuilt_on_F_parameters"]
                        rec["screen_rebuilt"] = len(rb)
                        rec["hp_any_reset"] = sum(abs(x["hp_any_network"] - x["hp_any_smogon"]) > 1e-12 for x in rb)
                        # every sampler's samples equal F's (the common prefix), and the worlds' seen facts equal FN's
                        sf, sn = cacheF.get(("samplers", did), {}), fresh.get(("samplers", did), {})
                        eq = ne = 0
                        for k_, m in sn.items():
                            if k_ in sf:
                                L = min(len(m.samples), len(sf[k_].samples))
                                eq += m.samples[:L] == sf[k_].samples[:L]
                                ne += m.samples[:L] != sf[k_].samples[:L]
                        rec["samplers_equal_F"] = eq
                        rec["samplers_differ_F"] = ne
                        rec["samplers_missing_in_F"] = sum(1 for k_ in sn if k_ not in sf)
                        if "FN" in got:
                            rec["seen_facts_equal_FN"] = seen_nonmove(root, w.teams) == seen_nonmove(root, got["FN"].teams)
                            rec["teams_equal_FN"] = w.teams == got["FN"].teams
                    elif arm == "POOLSETS":
                        w = PS.sample(log, at, d["side"], T.J, key=did, screen_cache={})
                        sw = PS.ws
                        rec["poolsets"] = {"counts": w.poolsets["counts"],
                                           "fallback_reads": len(w.poolsets["fallback_reads"])}
                    elif arm == "POOL":
                        w = POOL.sample(log, at, d["side"], T.J, key=did)
                        sw = POOL.ws
                        m = w.pool_meta
                        rec["pool"] = {k: m[k] for k in ("candidates", "distinct_candidates", "gated", "passed",
                                                         "passed_distinct_teams", "true_in_pool", "true_in_candidates",
                                                         "true_gated", "true_in_passed", "fails")}
                        rec["pool"]["true_drawn"] = sum(m["true_drawn"])
                    else:
                        raise SystemExit(f"unknown arm {arm}")
                    got[arm] = w
                    rec.update({"ok": True, "tries": w.tries, "fails": {k: v for k, v in w.fails.items()
                                                                        if isinstance(v, int)},
                                "screens": w.screens, "truth": truth_match(root, w.teams),
                                "audit": audits(root, w, sw), "teams": w.teams})
                except W.WorldError as e:
                    rec.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[-400:]}",
                                "reason": getattr(e, "reason", None), "meta": getattr(e, "meta", None)})
                except Exception as e:                         # counted, never dropped
                    rec.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[-400:]}",
                                "trace": traceback.format_exc()[-1500:]})
                rec["cpu_s"] = round(cpu() - c0, 2)
                rec["wall_s"] = round(time.time() - t0, 2)
                row["arms"][arm] = rec
                print(f"[validate] {did} k={row['k']} {arm}: ok={rec['ok']} cpu {rec['cpu_s']}s "
                      f"{rec.get('error', '')[:200]}", flush=True)
            # POOL's strict gate == the builder's strict gate on every team F judged strictly
            vF = cacheF.get(("verdict", did), {})
            strict = [(k_[: -len("##strict")], v[0]) for k_, v in vF.items() if k_.endswith("##strict")]
            if strict:
                res = run_core([root.rec(t, f"{did}#eq{j}", W.render(root.tmpl, t)) for j, (t, _) in enumerate(strict)],
                               trackers=True, obs=True)
                mine = [PW.strict_verdict(root, r) for r in res]
                row["gate_equality"] = {"teams": len(strict), "builder_pass": sum(v for _, v in strict),
                                        "agree": sum(m == v for m, (_, v) in zip(mine, strict))}
            T.write_row(fh, row)


if __name__ == "__main__":
    main()
