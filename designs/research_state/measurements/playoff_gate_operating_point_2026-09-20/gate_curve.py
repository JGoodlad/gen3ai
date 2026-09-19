"""THE PLAYOFF GATE'S OPERATING CURVE — swept on the BANKED dice, so it costs nothing.

    python3 gate_curve.py --ext <rerollx dir> --banked <reroll dir> --out gate_curve.json

WHAT THIS IS. The 2026-09-19 K-curve settled that a rollout leaf out-ranks the 75M-step win-prob
critic on the `top1|top2` column, and that the `playoff` arm buys nothing in games because its own
acting rule -- |mean(d)| >= SE_MULTIPLE * SE over R common-random-number paired rollouts, with at
least MIN_PAIRS pairs -- resolves 4.5 % of pairs at R = 4. That read priced ONE point of the gate.
This sweeps the gate.

THE DICE ARE THE SAME DICE. For re-roll k the three branches of a banked fork share ONE sim seed,
so `d_k = score(top1, k) - score(top2, k)` is exactly the statistic `PlayoffRunner.adjudicate`
forms. The gate reads the FRESH re-rolls 8..8+R-1; the K'=8 LABEL is the banked re-rolls 0..7. The
two sets are DISJOINT, so nothing here selects on the quantity it is scored against.

NOTHING IS RE-IMPLEMENTED. `playoff.paired_stats`, `playoff.is_conclusive` and `playoff.decide` are
imported from the production module, exactly as `playoff_rule_offline.py` did. A second hand-written
copy of a gate is how a gate's constant gets quietly forked.

THE FOUR COLUMNS, and why the EV one is the one to steer by.

* RESOLVE RATE -- the fraction of forks the gate would act on. Coverage.
* AGREEMENT vs the K'=8 LABEL -- of the forks it resolves, how often its pick is the branch the
  label ranks higher. ** COMPRESSED TOWARD 0.5 BY LABEL NOISE ** (the 2026-09-19 measured ceiling on
  `top1|top2` is 0.6709), so it is a sanity column, never the objective.
* AGREEMENT vs the 16-ROLLOUT estimate -- the same, against the fresh dice's own full-K mean. 🚨 This
  one is BIASED UPWARD BY CONSTRUCTION at every R: the gate's R draws are a SUBSET of the 16, so the
  scorer contains the scored. Reported because the instruction asks for it, and labelled every time.
* EXPECTED VALUE GAIN -- mean over RESOLVED forks of
  `label_value(gate's pick) - label_value(policy's top-1)`, in win-prob units.
  ** THIS IS THE UNBIASED ONE. ** The K'=8 label is a mean of 8 Bernoulli draws and is therefore an
  unbiased estimator of the branch's true value, so the EXPECTATION of the difference is the true
  expected gain whatever the label's variance -- unlike an accuracy, which label noise drags toward
  0.5. The gate keeps the policy's action on a positive mean, which contributes EXACTLY 0.0; only an
  overrule can move anything, which is why the overrule rate is carried beside it.

THE ORACLE CEILING, AND WHY EVERY EV NUMBER IS QUOTED BESIDE IT. An acting rule cannot gain more
than an oracle that always picks the label-better branch, which gains `E[max(lab2 - lab1, 0)]` per
decision -- about half of E|label gap|, minus whatever head start the policy's top-1 already has.
On these forks E|gap| is small and nearly half the pairs are label-TIED, so the ceiling is a couple
of win-prob points and a rule reaching a third of it is doing well. A raw EV in win-prob units says
nothing on its own; the FRACTION OF THE ORACLE is what compares two rules.

THE NO-GATE COMPARATOR. `ALWAYS` takes whichever branch has the higher R-rollout mean, with no bar
at all (ties keep the policy's action). It resolves ~100 % by construction, and its per-decision EV
gain is the number the gate has to beat to be worth having.

MIN_PAIRS IS INERT AT R >= 4 AND THAT IS REGISTERED (PREDICTION.md S1): `is_conclusive` bars on
`n < min_pairs`, and offline every fork has all R pairs, so n = R. The sweep therefore also runs
R in {2, 3}, where the axis bites. If the two MIN_PAIRS columns differ at any R >= 4 the harness is
reading something other than the production rule and the read is REFUSED.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

BRANCHES = ("top1", "top2", "rand")
#: R >= 4 is the registered grid; 2 and 3 are the EXTENSION that gives MIN_PAIRS something to do.
RS = (2, 3, 4, 8, 16)
KS = (0.5, 1.0, 1.5, 2.0)
MPS = (2, 4)
MAXK = 16
LABEL_K = 8
BOOT = 2000


def read(d: str, pat: str) -> dict:
    o = {}
    for p in sorted(glob.glob(os.path.join(d, pat))):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break          # a torn last line of a live shard -- stop, never guess
            o[(r["shard"], int(r["fork_line"]))] = r
    return o


def join(ext_dir: str, banked_dir: str):
    """Fork rows that carry FULL fresh K = 16 and a FULL banked K' = 8, joined on the salt.

    The salt equality is the join's own assertion: `battle:inv:date` identifies the fork, and two
    rows that disagree on it are not the same decision point however well their indices line up.
    """
    ext, lab = read(ext_dir, "rerollx_s*.jsonl"), read(banked_dir, "reroll_s*.jsonl")
    forks, dropped = [], {"unjoined": 0, "salt": 0, "short_ext": 0, "short_label": 0}
    for k in sorted(ext):
        e, b = ext[k], lab.get(k)
        if b is None:
            dropped["unjoined"] += 1
            continue
        if e["seed_salt"] != b["seed_salt"]:
            dropped["salt"] += 1
            continue
        if any(len(e["scores"][x]) < MAXK for x in BRANCHES):
            dropped["short_ext"] += 1
            continue
        if any(len(b["scores"][x]) < LABEL_K for x in BRANCHES):
            dropped["short_label"] += 1
            continue
        forks.append((e, b))
    return forks, dropped


def build(forks):
    """Per-fork arrays: the fresh paired differences, the label values, the full-K fresh means.

    `d[i]` is `fresh_top1[i] - fresh_top2[i]` under the SAME sim seed and the same torch seed -- the
    common-random-number difference the live runner forms. The label pair is the banked K'=8 means.
    """
    d = np.asarray([[float(e["scores"]["top1"][i]) - float(e["scores"]["top2"][i])
                     for i in range(MAXK)] for e, _ in forks], dtype=float)
    lab1 = np.asarray([float(np.mean(b["scores"]["top1"][:LABEL_K])) for _, b in forks])
    lab2 = np.asarray([float(np.mean(b["scores"]["top2"][:LABEL_K])) for _, b in forks])
    m16_1 = np.asarray([float(np.mean(e["scores"]["top1"][:MAXK])) for e, _ in forks])
    m16_2 = np.asarray([float(np.mean(e["scores"]["top2"][:MAXK])) for e, _ in forks])
    return d, lab1, lab2, m16_1, m16_2


def cell(mask_resolved, pick_top1, lab1, lab2, m16_1, m16_2, rng, boot=BOOT):
    """One (rule, R) cell's five numbers, with a fork-level bootstrap on the two headline ones.

    `pick_top1` is only meaningful where `mask_resolved`; a gate that declines plays the policy's
    action (which IS top1 here, by construction of the banked tree), so the unresolved forks
    contribute EXACTLY 0.0 to the per-decision gain -- carried explicitly rather than implied.
    """
    n = len(mask_resolved)
    gain_all = np.where(mask_resolved & ~pick_top1, lab2 - lab1, 0.0)
    res = mask_resolved.astype(float)
    nres = int(res.sum())
    agree_lab = (pick_top1 == (lab1 > lab2))[mask_resolved & (lab1 != lab2)]
    agree_16 = (pick_top1 == (m16_1 > m16_2))[mask_resolved & (m16_1 != m16_2)]
    out = {
        "resolve_rate": float(res.mean()),
        "n_resolved": nres,
        "overrule_rate_of_resolved": float((~pick_top1)[mask_resolved].mean()) if nres else None,
        "overrule_rate_of_all": float((mask_resolved & ~pick_top1).mean()),
        "agree_label_when_resolved": float(agree_lab.mean()) if agree_lab.size else None,
        "n_agree_label": int(agree_lab.size),
        "agree_r16_when_resolved": float(agree_16.mean()) if agree_16.size else None,
        "n_agree_r16": int(agree_16.size),
        "ev_gain_per_resolved": float(gain_all.sum() / nres) if nres else None,
        "ev_gain_per_decision": float(gain_all.mean()),
    }
    idx = np.arange(n)
    bg, br = [], []
    for _ in range(boot):
        s = rng.choice(idx, n, True)
        bg.append(float(gain_all[s].mean()))
        br.append(float(res[s].mean()))
    out["ev_gain_per_decision_ci"] = [float(np.quantile(bg, 0.025)), float(np.quantile(bg, 0.975))]
    out["resolve_rate_ci"] = [float(np.quantile(br, 0.025)), float(np.quantile(br, 0.975))]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", required=True)
    ap.add_argument("--banked", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args(argv)

    from main.search_dividend.playoff import (MIN_PAIRS, SE_MULTIPLE, decide, is_conclusive,
                                              paired_stats)

    forks, dropped = join(args.ext, args.banked)
    if not forks:
        raise SystemExit("REFUSED: no fork joined")
    d, lab1, lab2, m16_1, m16_2 = build(forks)
    n = len(forks)
    rng = np.random.default_rng(7)

    oracle = float(np.mean(np.maximum(lab2 - lab1, 0.0)))
    b_or = [float(np.maximum(lab2 - lab1, 0.0)[rng.choice(np.arange(n), n, True)].mean())
            for _ in range(args.boot)]

    out = {
        "n_forks": n,
        "dropped": dropped,
        "production_se_multiple": SE_MULTIPLE,
        "production_min_pairs": MIN_PAIRS,
        "rule_source": "main.search_dividend.playoff (imported, not re-implemented)",
        "label": f"banked K'={LABEL_K} mean (re-rolls 0..{LABEL_K - 1}), DISJOINT from the gate's "
                 f"fresh re-rolls 8..23",
        "e_abs_label_gap": float(np.mean(np.abs(lab1 - lab2))),
        "e_signed_label_gap_top1_minus_top2": float(np.mean(lab1 - lab2)),
        "frac_label_tied": float(np.mean(lab1 == lab2)),
        # THE CEILING: the per-decision gain of an oracle that always takes the label-better
        # branch. Every `ev_gain_per_decision` below is also reported as a fraction of this.
        "oracle_gain_per_decision": oracle,
        "oracle_gain_per_decision_ci": [float(np.quantile(b_or, 0.025)),
                                        float(np.quantile(b_or, 0.975))],
        "grid": {"R": list(RS), "se_multiple": list(KS), "min_pairs": list(MPS)},
        "gated": {}, "always": {},
    }

    # --- the GATED grid ---------------------------------------------------------------------
    for R, k, mp in itertools.product(RS, KS, MPS):
        stats = [paired_stats(d[i, :R]) for i in range(n)]
        resolved = np.asarray([is_conclusive(m, se, nn, k=k, min_pairs=mp)
                               for m, se, nn in stats], dtype=bool)
        # `decide` returns a1 on a positive mean and a2 on a negative one; a1 IS the policy's
        # action in the banked tree, so `pick_top1` is exactly `mean > 0` -- asserted against the
        # production function rather than assumed.
        picks = np.asarray([decide(1, 2, m, se, nn, 1, k=k, min_pairs=mp)[0] == 1
                            for m, se, nn in stats], dtype=bool)
        assert np.array_equal(picks[resolved], np.asarray([s[0] for s in stats])[resolved] > 0)
        out["gated"][f"R{R}_k{k}_mp{mp}"] = dict(
            R=R, se_multiple=k, min_pairs=mp,
            **cell(resolved, picks, lab1, lab2, m16_1, m16_2, rng, args.boot))

    # --- the NO-GATE comparator -------------------------------------------------------------
    for R in RS:
        means = d[:, :R].mean(axis=1)
        resolved = means != 0.0            # a tie keeps the policy's action; it is not an action
        picks = means > 0.0
        out["always"][f"R{R}"] = dict(
            R=R, **cell(resolved, picks, lab1, lab2, m16_1, m16_2, rng, args.boot))

    # --- the registered MIN_PAIRS inertness check (PREDICTION.md S1/A10) ---------------------
    inert = {}
    for R, k in itertools.product(RS, KS):
        a, b = out["gated"][f"R{R}_k{k}_mp2"], out["gated"][f"R{R}_k{k}_mp4"]
        inert[f"R{R}_k{k}"] = bool(a["resolve_rate"] == b["resolve_rate"]
                                   and a["ev_gain_per_decision"] == b["ev_gain_per_decision"])
    for group in ("gated", "always"):
        for c in out[group].values():
            c["frac_of_oracle"] = (float(c["ev_gain_per_decision"] / oracle) if oracle else None)
            lo, hi = c["ev_gain_per_decision_ci"]
            c["ev_detected"] = bool(lo > 0.0 or hi < 0.0)

    out["min_pairs_identical"] = inert
    bad = [key for key, same in inert.items() if not same and int(key[1:].split("_")[0]) >= 4]
    if bad:
        raise SystemExit(f"REFUSED: MIN_PAIRS is not inert at R>=4 ({bad}) — the harness is not "
                         f"reading the production rule")

    json.dump(out, open(args.out, "w"), indent=1)
    for R in RS:
        for k in KS:
            c = out["gated"][f"R{R}_k{k}_mp4"]
            print(f"R={R:2d} k={k:.1f}  resolve {c['resolve_rate']:.4f}  "
                  f"agree_lab {c['agree_label_when_resolved']}  "
                  f"ev/res {c['ev_gain_per_resolved']}  "
                  f"ev/dec {c['ev_gain_per_decision']:+.5f} "
                  f"[{c['ev_gain_per_decision_ci'][0]:+.5f}, {c['ev_gain_per_decision_ci'][1]:+.5f}]"
                  f" {'DET' if c['ev_detected'] else 'ND '} {c['frac_of_oracle']:.3f}x oracle",
                  flush=True)
        a = out["always"][f"R{R}"]
        print(f"R={R:2d} ALWAYS resolve {a['resolve_rate']:.4f}  "
              f"agree_lab {a['agree_label_when_resolved']}  "
              f"ev/dec {a['ev_gain_per_decision']:+.5f} "
              f"[{a['ev_gain_per_decision_ci'][0]:+.5f}, {a['ev_gain_per_decision_ci'][1]:+.5f}]")
    print(f"-> {args.out}  (n={n} forks, E|label gap|={out['e_abs_label_gap']:.4f}, "
          f"label-tied {out['frac_label_tied']:.3f}, ORACLE ceiling "
          f"{oracle:+.5f}/decision)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
