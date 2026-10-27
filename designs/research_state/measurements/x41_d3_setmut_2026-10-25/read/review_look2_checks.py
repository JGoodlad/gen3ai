"""REVIEW_look2.md's statistics, from the REGISTERED rule's own code (``main.x35.d1``, byte-identical at the play pin
c4a5c37b6 and at main on 2026-10-27).

Run (CPU, about 3 min; numbers quoted in REVIEW_look2.md are from seed 20261027, 4,000 / 20,000 reps):
    PYTHONPATH=src CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 python \
        designs/research_state/measurements/x41_d3_setmut_2026-10-25/read/review_look2_checks.py [reps]

1. NULL reachability: the look-2 Newcombe half-width at 700 games per panel per arm, against the +-3.0 pp band.
2. The look-2 class probabilities with both arms binomial (one look; with a 1.5 pp seed SD; the two-look procedure).
3. Conditional power: C0 s3501-s3505 are the bank (known), D3 s3501-s3503 are look 1's counts (the E1 re-play moves
   about 1 % of games), only D3 s3504-s3505 are random.
"""
import json
import sys
from pathlib import Path

import numpy as np

from main.x35 import d1 as D

HERE = Path(__file__).resolve().parent
BANK = HERE.parents[1] / "x35_breakers_stage_2026-10-17" / "d1_read" / "read_look2_input.json"
LOOK1 = HERE / "inputs.json"
SEEDS = ("3501", "3502", "3503", "3504", "3505")
P = D.PANELS


def counts(doc, side):
    return {s: {k: list(v) for k, v in c.items()} for s, c in doc["h"][side].items()}


def main(reps):
    rng = np.random.default_rng(20261027)
    off = counts(json.loads(BANK.read_text()), "off")
    on1 = counts(json.loads(LOOK1.read_text()), "on")
    c0 = {k: sum(off[s][k][0] for s in off) / sum(off[s][k][1] for s in off) for k in P}
    out = {"c0_bank_per_panel": c0}

    # 1. NULL reachability
    h0 = D.d1_primary(2, off, off)
    out["delta0_at_bank_rates"] = {"class": h0.cls, "ci_pp": h0.ci_pp, "phi": h0.od.phi}
    hw = {}
    for p in (0.40, 0.44, 0.45, 0.48, 0.50):
        s = D.strat_score({k: [round(p * 700), 700] for k in P})
        hw[p] = 100 * D.newcombe_strat(s, s)[2]
    out["unwidened_half_width_pp_by_rate"] = hw

    # 2. class probabilities, both arms binomial
    def arm(shift, tau):
        cells = {}
        for s in SEEDS:
            u = rng.normal(0, tau) if tau else 0.0
            cells[s] = {k: [int(rng.binomial(140, min(max(c0[k] + shift + u, 0.01), 0.99))), 140] for k in P}
        return cells

    def sim(delta_pp, tau_pp=0.0, two_look=False):
        tally = {}
        for _ in range(reps):
            on, of = arm(delta_pp / 100, tau_pp / 100), arm(0.0, tau_pp / 100)
            key = None
            if two_look:
                l1 = D.d1_primary(1, {s: on[s] for s in SEEDS[:3]}, {s: of[s] for s in SEEDS[:3]})
                if l1.outcome != D.CONTINUE:
                    key = f"L1:{l1.outcome}"
            if key is None:
                key = ("L2:" if two_look else "") + str(D.d1_primary(2, on, of).cls)
            tally[key] = tally.get(key, 0) + 1
        return {k: v / reps for k, v in sorted(tally.items())}

    out["one_look"] = {d: sim(d) for d in (0.0, 1.27, 1.5, 2.0, 3.0, 4.0)}
    out["one_look_seed_sd_1p5"] = {d: sim(d, tau_pp=1.5) for d in (0.0, 1.5, 3.0)}
    out["two_look"] = {d: sim(d, two_look=True) for d in (0.0, 1.5, 3.0)}

    # 3. conditional power
    def cond(p_new, n):
        tally = {}
        for _ in range(n):
            on = dict(on1)
            for s in SEEDS[3:]:
                on[s] = {k: [int(rng.binomial(140, p_new[k])), 140] for k in P}
            c = D.d1_primary(2, on, off).cls
            tally[c] = tally.get(c, 0) + 1
        return {k: v / n for k, v in sorted(tally.items())}

    d3 = {k: sum(on1[s][k][0] for s in on1) / sum(on1[s][k][1] for s in on1) for k in P}
    out["conditional"] = {f"C0+{x}pp": cond({k: c0[k] + x / 100 for k in P}, 5 * reps) for x in (0, 1.27, 2, 3, 5)}
    out["conditional"]["d3_look1_rates"] = cond(d3, 5 * reps)
    for w in range(70, 141):
        on = dict(on1)
        on.update({s: {k: [w, 140] for k in P} for s in SEEDS[3:]})
        h = D.d1_primary(2, on, off)
        if h.cls == D.GAIN:
            out["gain_needs_new_seeds_at"] = {"wins_of_140_every_panel": w, "delta_pp": h.delta_pp, "ci_pp": h.ci_pp}
            break
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
