"""X66 PREREG §8.2: how much of S'h's departure from P* is reproducible by S'h itself (zero CPU: stage 1b's rows).
Stage 1b (../sprime_honest_inc_stage1b_2026-11-06/) re-ran the incremental builder r0s4_honest_inc@10 under another key
tag (salt 'B') at 162 of its 444 targets; 157 pairs have both searches' q and log_pi. Per pair: each run's pick
(argmax of log pi + q / 0.015, the piKL mode = what S'h plays), P*'s recorded greedy pi_argmax. Wilson 95 % intervals.
    python3 noise_ceiling.py > noise_ceiling.json
"""
import gzip
import json
import math
from pathlib import Path

S1B = Path(__file__).resolve().parent.parent / "sprime_honest_inc_stage1b_2026-11-06"
LAM = 0.015


def load(name):
    out = {}
    for line in gzip.open(S1B / name, "rt"):
        r = json.loads(line)
        if r.get("target") in (True, "True") and r.get("key") and isinstance(r.get("res"), dict):
            if r["res"].get("q") and r["res"].get("log_pi"):
                out[r["key"]] = r
    return out


def pick(res):
    s = {a: float(res["log_pi"][a]) + float(res["q"][a]) / LAM for a in res["q"]}
    return max(sorted(s), key=s.get)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]


a, b = load("inc_rows.jsonl.gz"), load("inc_rep_rows.jsonl.gz")
keys = sorted(set(a) & set(b))
n = same = dep_a = dep_b = both_same = either = 0
for k in keys:
    pa, pb, base = pick(a[k]["res"]), pick(b[k]["res"]), str(a[k]["pi_argmax"])
    n += 1
    same += pa == pb
    dep_a += pa != base
    dep_b += pb != base
    both_same += pa != base and pa == pb
    either += pa != base or pb != base
print(json.dumps({
    "source": "sprime_honest_inc_stage1b_2026-11-06 inc_rows vs inc_rep_rows (salt B), targets with q in both",
    "pairs": n, "self_agreement": wilson(same, n), "departure_rate_run_a": wilson(dep_a, n),
    "departure_rate_run_b": wilson(dep_b, n), "both_depart_to_the_same_action": wilson(both_same, n),
    "reproduced_share_of_run_a_departures": wilson(both_same, dep_a),
    "counts": {"same": same, "dep_a": dep_a, "dep_b": dep_b, "both_same": both_same, "either": either}}, indent=1))
