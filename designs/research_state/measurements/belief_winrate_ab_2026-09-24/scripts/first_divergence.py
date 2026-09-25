"""POST-REGISTRATION E: the win-rate effect stratified by k at the FIRST DIVERGENT decision.

    python first_divergence.py <rows_dir> <out_dir>      # -> out_dir/first_divergence.json

Why this stratifier is valid where K_final (post_registration.py B) is not. Greedy vs greedy with a
fixed sim seed: the LEARNED and switched (PRIOR-ONLY / MOVE) trajectories of one index are identical
up to the first decision at which the switch changes the greedy action. The LEARNED row's side read
names that decision (the ALL / MOVE change flag re-forwards the SAME obs with the switch ON), and
its k (opponent species revealed) is a function of the SHARED prefix. So stratum membership is the
same under both arms and is fixed before the treatment has touched the trajectory. It is not a
selection on either arm's outcome. The premise is CHECKED here, not assumed: an index with no
divergence must replay identically (winner, turns, n_dec) under the switched arm.

What the stratum effect means: the effect of turning the switch ON for the whole battle, in battles
whose first switch-changed decision happens with k opponent species revealed. It is NOT the effect
of the switch "at k only"; every later change is included.

Same bootstrap recipe as analyze.py: 10,000 resamples of battle indices, percentile CIs; each
stratum is resampled within itself (the POOL and LADDER strata are different battles, so a DiD's
two halves are resampled independently). No correlation across battles is computed.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze import PRIMARY, SECONDARY, B, SEED, load, score, ci, tag  # noqa: E402

NAMED = (3, 5)
FLAG = {"prior": 2, "move": 3}   # side entry = [k, off_mismatch, all_changed, move_changed]


def first_div(side, j):
    for d, e in enumerate(side):
        if e[j]:
            return d, e[0]
    return None, None


def boot_mean(y, rng):
    n = len(y)
    if n == 0:
        return None
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
    return w @ y / n


def main(rows_dir, out_dir):
    rows = load(rows_dir)
    idx = sorted({k[2] for k in rows})
    cells = PRIMARY + SECONDARY
    ok = [i for i in idx if all(rows.get((c, b, i), {}).get("status") == "ok" for c, b in cells)]
    rng = np.random.default_rng(SEED + 3)
    out = {"n_indices": len(ok), "named_strata": list(NAMED), "arms": {}}
    for arm, j in FLAG.items():
        A = {}
        samp_named = {}
        for col in ("pool", "ladder"):
            strata = {}
            premise_bad = []
            for i in ok:
                L, S = rows[(col, "learned", i)], rows[(col, arm, i)]
                d, k = first_div(L["side"], j)
                e = score(S) - score(L)
                if d is None:
                    if (S["winner"], S["turns"], S["n_dec"]) != (L["winner"], L["turns"], L["n_dec"]):
                        premise_bad.append(i)
                    key = "none"
                else:
                    if S["n_dec"] <= d:
                        premise_bad.append(i)
                    key = str(k)
                strata.setdefault(key, []).append((e, d))
            per = {}
            for key in sorted(strata, key=lambda s: (s == "none", s)):
                y = np.array([e for e, _ in strata[key]])
                s = boot_mean(y, rng)
                c_ = ci(s)
                per[key] = {"n": len(y), "E_pp": 100 * float(y.mean()), "ci_pp": [100 * c_[0], 100 * c_[1]],
                            "read": tag(y.mean(), c_),
                            "median_first_div_decision": (None if key == "none"
                                                          else float(np.median([d for _, d in strata[key]])))}
                if key in {str(k) for k in NAMED}:
                    samp_named[(col, key)] = s
            A[col] = {"premise_violations": premise_bad, "strata": per}
        did = {}
        for k in NAMED:
            a, b = samp_named.get(("ladder", str(k))), samp_named.get(("pool", str(k)))
            if a is None or b is None:
                continue
            p = A["ladder"]["strata"][str(k)]["E_pp"] - A["pool"]["strata"][str(k)]["E_pp"]
            c_ = ci(100 * (a - b))
            did[str(k)] = {"DiD_pp": p, "ci_pp": c_, "read": tag(p, c_)}
        A["DiD_by_stratum"] = did
        out["arms"][arm] = A
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "first_divergence.json"), "w") as f:
        json.dump(out, f, indent=1)
    for arm in out["arms"]:
        print(f"== {arm}")
        for col in ("pool", "ladder"):
            a = out["arms"][arm][col]
            print(f"  {col}: premise violations {len(a['premise_violations'])} {a['premise_violations'][:10]}")
            for key, v in a["strata"].items():
                print(f"    k_div={key:>4} n={v['n']:5d} E={v['E_pp']:+6.2f} pp "
                      f"[{v['ci_pp'][0]:+6.2f}, {v['ci_pp'][1]:+6.2f}] {v['read']:12s} "
                      f"median d={v['median_first_div_decision']}")
        for k, v in out["arms"][arm]["DiD_by_stratum"].items():
            print(f"  DiD k_div={k}: {v['DiD_pp']:+.2f} [{v['ci_pp'][0]:+.2f}, {v['ci_pp'][1]:+.2f}] {v['read']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
