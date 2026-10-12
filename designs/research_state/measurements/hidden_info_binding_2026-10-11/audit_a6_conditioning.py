"""A6 THE GATE'S CONDITIONING (descriptive; no outcome is read).

(1) For SEEN opponent mons, the moves drawn into their re-drawn slots in the PLAYED worlds (first 8 of each state
    with an ok det playout row; det/ and det_b/, which re-draw seen mons' moves from the same Smogon prior) against
    the UNGATED proposal: ``determinize.sample_moves`` with the same kept slots and Hidden Power type, drawn
    ``--mc`` times per (state, mon) with a fixed seed. Per species: observed inclusion counts vs the expected
    (proposal inclusion probability x worlds); the pooled KL(observed || proposal) over moves, per species and
    weighted; the 20 moves most over- and under-drawn (by observed - expected count, with the ratio). The proposal
    IS the Smogon prior renormalized over the eligible moves (minus kept, banned, the HP rule) and drawn without
    replacement, so the gap is the gate's conditioning (and sampling noise: ~1/sqrt(n)).
(2) Why states failed to build 8 worlds (the ~3.5 % excluded): their ``fails`` (refused / rows_differ / short), the
    first-diff decision index (as a fraction of the root's row count), and the stratum (unrevealed opponent slots,
    turn) vs the included states.

    run_at_pin.sh <cp> 1 4 1h audit_a6_conditioning.py [--mc 400]
→ here ``audit_a6_conditioning_<cp>.json``.
"""
import argparse
import collections
import math

import numpy as np

import audit_lib as A

C, D = A.C, A.D
J = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=400)
    a = ap.parse_args()
    c = A.cp()
    bk = A.bank(c)
    P = A.priors()
    res = {"audit": "A6 the gate's conditioning (descriptive)", "cp": c, "mc_per_mon": a.mc, "by_dir": {}}
    for det_dir in ("det", "det_b"):
        W = A.world_rows(c, det_dir)
        played = A.played_ids(c, det_dir)
        rng = np.random.default_rng(20261011)
        obs = collections.defaultdict(collections.Counter)      # species -> move -> inclusions in played worlds
        exp = collections.defaultdict(collections.Counter)      # species -> move -> expected inclusions (proposal)
        n_mon_worlds = collections.Counter()
        null = collections.defaultdict(collections.Counter)     # the matched-noise control: J UNGATED draws
        for did in sorted(played):
            d, b = A.decision(bk, did)
            opp = "p2" if d["side"] == "p1" else "p1"
            true_team = getattr(b, opp)["team"]
            wr = W[did]
            revealed = {k: set(v) for k, v in wr["revealed"].items()}
            t_sets = [D.parse_set(e) for e in true_team.split("]")]
            for t, f in enumerate(t_sets):
                sid = D.toid(f[1] or f[0])
                if sid not in revealed or sid not in P.moves:
                    continue
                mv = f[4].split(",")
                keep = {i: m for i, m in enumerate(mv)
                        if m in revealed[sid] or (m.startswith("hiddenpower") and "hiddenpower" in revealed[sid])}
                free = [i for i in range(len(mv)) if i not in keep]
                if not free:
                    continue
                hp_t = D.hp_type_of(D.set_ivs(f))
                for j in range(J):
                    wm = D.parse_set(wr["teams"][j].split("]")[t])[4].split(",")
                    for i in free:
                        obs[sid][wm[i]] += 1
                    n_mon_worlds[sid] += 1
                for _ in range(J):
                    s = D.sample_moves(rng, P, sid, keep, hp_t)
                    if s is not None:
                        for i in free:
                            null[sid][s[i]] += 1
                inc = collections.Counter()
                ok = 0
                for _ in range(a.mc):
                    s = D.sample_moves(rng, P, sid, keep, hp_t)
                    if s is None:
                        continue
                    ok += 1
                    for i in free:
                        inc[s[i]] += 1
                for m, n in inc.items():
                    exp[sid][m] += J * n / max(ok, 1)
        def pooled_kl(o_c):
            num_, den_ = 0.0, 0
            for sp_ in o_c:
                no_, ne_ = sum(o_c[sp_].values()), sum(exp[sp_].values())
                k_ = sum((n / no_) * math.log((n / no_) / (exp[sp_][m] / ne_)) for m, n in o_c[sp_].items() if exp[sp_].get(m, 0) > 0)
                num_ += k_ * no_
                den_ += no_
            return round(num_ / max(den_, 1), 4)

        per_sp, tot_kl_w, tot_w = [], 0.0, 0
        diffs = []
        for sid in obs:
            no, ne = sum(obs[sid].values()), sum(exp[sid].values())
            kl = 0.0
            for m, n in obs[sid].items():
                p = n / no
                q = exp[sid].get(m, 0.0) / ne if ne else 0.0
                kl += p * math.log(p / q) if q > 0 else float("inf")
            per_sp.append({"species": sid, "draws": no, "mon_worlds": n_mon_worlds[sid], "kl_obs_vs_proposal": round(kl, 4),
                           "outside_proposal": sum(n for m, n in obs[sid].items() if exp[sid].get(m, 0) == 0)})
            if math.isfinite(kl):
                tot_kl_w += kl * no
                tot_w += no
            for m in set(obs[sid]) | set(exp[sid]):
                o_, e_ = obs[sid].get(m, 0), exp[sid].get(m, 0.0)
                diffs.append({"species": sid, "move": m, "observed": o_, "expected": round(e_, 2),
                              "obs_minus_exp": round(o_ - e_, 2), "ratio": round(o_ / e_, 3) if e_ > 0 else None,
                              "z": round((o_ - e_) / math.sqrt(e_), 2) if e_ > 0 else None})
        diffs.sort(key=lambda x: x["obs_minus_exp"])
        per_sp.sort(key=lambda x: -x["draws"])
        # (3) the CONSTRUCTION-DRAW conditioning: the turn-0 gender draws (one PRNG draw per mon of a species with no
        # fixed gender) shift the whole dice stream, so the gate tends to accept only worlds whose never-seen slots cost
        # the TRUE number of draws -- information about the true team. Accepted (played) vs the UNGATED proposal
        # (``sample_world`` itself, Smogon weights; det/ only -- det_b's belief weights are not archived).
        def gdraws(team):
            return sum(1 for e in team.split("]") if not D.parse_set(e)[7]
                       and P.species.get(D.toid(D.parse_set(e)[1] or D.parse_set(e)[0]), {}).get("gender") is None)

        acc_eq, acc_n, prop_eq, prop_n = 0, 0, 0, 0
        for did in sorted(played):
            d, b = A.decision(bk, did)
            true_team = getattr(b, "p2" if d["side"] == "p1" else "p1")["team"]
            g0 = gdraws(true_team)
            wr = W[did]
            for j in range(J):
                acc_eq += gdraws(wr["teams"][j]) == g0
                acc_n += 1
            if det_dir == "det":
                rv = {k: set(x) for k, x in wr["revealed"].items()}
                for _ in range(J):
                    tm, _st = D.sample_world(rng, P, true_team, rv)
                    prop_eq += gdraws(tm) == g0
                    prop_n += 1
        gender_cond = {"accepted_worlds_with_true_gender_draw_count": round(acc_eq / max(acc_n, 1), 4), "accepted_n": acc_n,
                       "ungated_proposal_with_true_gender_draw_count": round(prop_eq / prop_n, 4) if prop_n else None,
                       "proposal_n": prop_n}
        # (2) the excluded states
        rows = [W[k] for k in sorted(W)]
        inc_st, exc_st = [], []
        for r in rows:
            d, _ = A.decision(bk, r["id"])
            full8 = r.get("ok") and all(t is not None for t in r["teams"][:J])
            unrev = 6 - len(r.get("revealed", {})) if r.get("ok") else None
            (inc_st if full8 else exc_st).append((r, d["turn"], unrev))
        fail_tot = collections.Counter()
        first = []
        for r, _, _ in exc_st:
            for k, v in (r.get("fails") or {}).items():
                if isinstance(v, int):
                    fail_tot[k] += v
            nr = r.get("n_rows") or 1
            first += [x / nr for x in (r.get("fails") or {}).get("first_diff_at", [])]
        fail_inc = collections.Counter()
        first_inc = []
        for r, _, _ in inc_st:
            for k, v in (r.get("fails") or {}).items():
                if isinstance(v, int):
                    fail_inc[k] += v
            nr = r.get("n_rows") or 1
            first_inc += [x / nr for x in (r.get("fails") or {}).get("first_diff_at", [])]

        def strat(xs):
            u = collections.Counter(str(x[2]) for x in xs)
            tb = collections.Counter("<=5" if x[1] <= 5 else "6-15" if x[1] <= 15 else "16-30" if x[1] <= 30 else ">30" for x in xs)
            return {"unrevealed_slots": dict(sorted(u.items())), "turn": dict(tb)}

        q = lambda xs: [round(float(v), 3) for v in np.quantile(xs, [0.1, 0.25, 0.5, 0.75, 0.9])] if xs else None  # noqa: E731
        res["by_dir"][det_dir] = {
            "redrawn_slot_draws": sum(sum(v.values()) for v in obs.values()),
            "species": len(obs),
            "pooled_kl_obs_vs_proposal (draw-weighted, finite)": round(tot_kl_w / max(tot_w, 1), 4),
            "pooled_kl_obs_vs_proposal (same estimator as the null)": pooled_kl(obs),
            "draws_outside_proposal_support": sum(x["outside_proposal"] for x in per_sp),
            "null_pooled_kl (J ungated proposal draws per mon-world: the matched-noise floor)": pooled_kl(null),
            "per_species_top30": per_sp[:30],
            "most_OVER_drawn_20": diffs[::-1][:20], "most_UNDER_drawn_20": diffs[:20],
            "construction_gender_draw_conditioning": gender_cond,
            "excluded_states": {"n": len(exc_st), "fails_total": dict(fail_tot),
                                "first_diff_fraction_quantiles_10_25_50_75_90": q(first),
                                "strata": strat(exc_st), "ids_errors": [r["id"] + (f" ERROR {r.get('error')}" if not r.get("ok") else "")
                                                                       for r, _, _ in exc_st][:30]},
            "included_states": {"n": len(inc_st), "fails_total": dict(fail_inc),
                                "first_diff_fraction_quantiles_10_25_50_75_90": q(first_inc), "strata": strat(inc_st)},
        }
    A.write_json(f"audit_a6_conditioning_{c}.json", res)
    for k, v in res["by_dir"].items():
        print(k, {kk: v[kk] for kk in ("redrawn_slot_draws", "species", "pooled_kl_obs_vs_proposal (draw-weighted, finite)",
                                       "draws_outside_proposal_support")}, v["excluded_states"]["n"], flush=True)


if __name__ == "__main__":
    main()
