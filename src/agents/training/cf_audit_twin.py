"""cf_audit_twin — the TWIN-HEAD paired read + the SHADOW critic (``gen3_cf_twin_heads_v1``).

The audit's three win-prob heads score the SAME labelled states, so the comparison available here
is a *paired* one: per-row proper-score differences with a battle-clustered CI, the per-head spread
binned by each head's own prediction, and the shadow critic's signed gap against the live one.
:func:`attach_twin_heads` is the one impure function — it forwards a checkpoint over the labelled
states and writes the columns the three readouts consume.

Extracted verbatim from ``cf_audit.py`` (2026-09-06, the file-size ratchet's second cut of the
1,000-2,000 band) — it was already its own banner section there. ``cf_audit`` re-imports every name
below, so ``from agents.training.cf_audit import paired_head_read`` still resolves, and the
arithmetic is unchanged: ``cf_audit_test.py``'s extraction-parity golden was captured BEFORE the
move and reproduces byte-for-byte after it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from agents.training.stats import MIN_CELL_N, cluster_bootstrap_ci, sd_true_excess


#: The three win-prob heads, in the order the factorial reads them. The KEY is the label-dict field
#: `attach_twin_heads` writes; head A's is the one the bias map already carries.
TWIN_HEADS = (("A", "win_prob"), ("B", "twin_b_pred"), ("C", "twin_c_pred"))

#: The pre-registered paired CONTRASTS and what each isolates. Declared here rather than assembled
#: at the call site so a reader (and a test) sees the whole factorial in one place, and so the
#: interpretation travels with the arithmetic.
TWIN_CONTRASTS = (
    ("B_minus_A", "B", "A", "COVERAGE — the same loss form on the cf-labelled states"),
    ("C_minus_B", "C", "B", "PRECISION — the same states, a tight-MC target instead of one draw"),
    ("C_minus_A", "C", "A", "the TOTAL effect (coverage + precision), i.e. the original R1 claim"),
)


def paired_head_read(labels: Sequence[dict], *, draws: int = 2000, seed: int = 13) -> dict:
    """THE PRIMARY meter of the twin-heads amendment: paired proper-score differences.

    For every labelled state the audit holds three predictions of the SAME quantity — heads A, B
    and C — and one tight-MC measurement of it. So the comparison is a paired proper-scoring read:
    per row, ``brier_X = (pred_X − mc)²`` and ``abs_err_X = |pred_X − mc|``, differenced ACROSS
    HEADS ON THE SAME ROW, with a battle-clustered bootstrap CI on the difference.

    WHY THIS IS THE PRIMARY AND `sd_true_excess` IS NOT, in this arm. G0's amended meter exists
    because there was only ever ONE head, so the only available question was "how much true spread
    does this head fail to resolve". With three heads on identical rows there is a strictly sharper
    question — "which label stream produced the head that is CLOSER to ground truth" — and it has
    two properties the spread meter cannot match here:

    * **The hidden-information floor cancels EXACTLY**, not merely in expectation. The floor is a
      property of the STATE (`tmp/hidden_info_floor_report.md`: 39% of the meter's variance, and
      concentrated — 49% of states carry almost none, the top 10% carry half). Every head scores
      the same states, so it is the same additive constant in every arm and it vanishes in a
      per-row difference. The amended §2 argued the floor cancels in an arm-vs-control difference
      at matched step; within-run twins strengthen that to matched STATE.
    * **No stratification, so no selection correction is owed.** The eval capture quota
      over-samples losses, which is why the bias map recombines its cells at population shares.
      A paired difference over identical rows carries that bias identically in both terms.

    ⚠️ **A NEAR-ZERO DIFFERENCE WITH A NEAR-ZERO |B−C| IS NOT A NULL RESULT — it is a coverage or
    dosage reading.** If the twins never diverged (check `mean_abs_pred_diff` and the live
    `cf/twin_b_vs_c_abs`), the label streams did not separate the heads and there is nothing to
    decompose yet. The kill in the runbook applies to a divergence that HAPPENED and bought nothing.
    """
    # FILTER the rows, never test `all` over the unfiltered list. `attach_twin_heads` skips any
    # label whose `states.npz` is missing or whose `inv` is out of range (routine after
    # `prober.groom`), so ONE unscored row out of a thousand would otherwise delete heads B and C
    # from `present` and report "nothing to pair" on a run that scored 99.9% — a null that is
    # entirely an accounting artifact, on the arm's PRIMARY meter.
    keys = [k for _n, k in TWIN_HEADS]
    labels = [r for r in labels if all(r.get(k) is not None for k in keys[1:])
              or all(r.get(k) is None for k in keys[1:])]
    scored = [r for r in labels if r.get("twin_b_pred") is not None]
    if scored:
        labels = scored
    present = [(name, key) for name, key in TWIN_HEADS
               if labels and all(r.get(key) is not None for r in labels)]
    if len(present) < 2 or not labels:
        return {"heads_present": [n for n, _k in present], "n_labels": len(labels),
                "contrasts": [], "note": "fewer than two heads scored — nothing to pair"}
    battles = [r["battle"] for r in labels]
    mc = np.asarray([r["mc"] for r in labels], dtype=float)
    per_head = {}
    for name, key in present:
        pred = np.asarray([r[key] for r in labels], dtype=float)
        per_head[name] = {"pred": pred,
                          "brier": (pred - mc) ** 2,
                          "abs_err": np.abs(pred - mc)}
    out: dict = {
        "heads_present": [n for n, _k in present],
        "n_labels": len(labels),
        "n_battles": len(set(battles)),
        "by_head": {n: {"mean_pred": float(v["pred"].mean()),
                        "brier": float(v["brier"].mean()),
                        "abs_err": float(v["abs_err"].mean())}
                    for n, v in per_head.items()},
        "contrasts": [],
    }
    for label, hi, lo, meaning in TWIN_CONTRASTS:
        if hi not in per_head or lo not in per_head:
            continue
        row = {"contrast": label, "isolates": meaning}
        for metric in ("brier", "abs_err"):
            # SIGN CONVENTION, stated because it is the thing a reader gets wrong: these are ERROR
            # scores, so a NEGATIVE difference means the first-named head is BETTER.
            diff = per_head[hi][metric] - per_head[lo][metric]
            ci_lo, ci_hi = cluster_bootstrap_ci(diff.tolist(), battles, draws=draws, seed=seed)
            row[metric] = float(diff.mean())
            row[f"{metric}_ci"] = [ci_lo, ci_hi]
        row["mean_abs_pred_diff"] = float(
            np.abs(per_head[hi]["pred"] - per_head[lo]["pred"]).mean())
        out["contrasts"].append(row)
    return out


def twin_resolution_read(labels: Sequence[dict], n_rollouts: int) -> dict:
    """Each head's own `sd_true_excess`, binned by ITS OWN prediction — the G0 continuity link.

    Reported BESIDE `paired_head_read`, never instead of it, and with one caveat stated in the
    output itself: these cells are **UNWEIGHTED**. The bias map recombines its decile cells at the
    eval frame's population (decile, outcome) shares to undo the capture quota's loss
    over-sampling — and that correction is unavailable for heads B and C, because the frame carries
    only head A's predictions, so B's and C's decile membership over the whole frame is unknown.
    Re-using A's masses for B's bins would be a number that looks population-weighted and is not.

    So: absolute levels here are NOT comparable with the bias map's
    `population_weighted_sd_true_excess`, and the field is named to make that impossible to miss.
    What they are good for is the SHAPE — whether a head's blur moved in the deciles the arm
    predicted it would.
    """
    out: dict = {"weighting": "UNWEIGHTED — not comparable with the bias map's "
                              "population_weighted_sd_true_excess; see the docstring",
                 "by_head": {}}
    for name, key in TWIN_HEADS:
        # Per-head row filtering, for `paired_head_read`'s reason: a single unscored row must not
        # delete a whole head's block.
        rows = [r for r in labels if r.get(key) is not None]
        if not rows:
            continue
        cells = []
        for dec in range(10):
            sub = [r for r in rows if min(9, int(float(r[key]) * 10)) == dec]
            if len(sub) < MIN_CELL_N:
                continue
            st = sd_true_excess([r["mc"] for r in sub], n_rollouts)
            st["decile"] = dec
            st["mean_predicted"] = float(np.mean([float(r[key]) for r in sub]))
            cells.append(st)
        tot = sum(c["n"] for c in cells) or 1
        out["by_head"][name] = {
            "cells": cells,
            "sample_weighted_sd_true_excess": float(
                sum(c["n"] * (c["sd_true_excess"] or 0.0) for c in cells) / tot),
        }
    return out


def shadow_read(labels: Sequence[dict], *, draws: int = 2000, seed: int = 17) -> Optional[dict]:
    """The SHADOW critic's audit block, or None when the checkpoint carries no shadow head.

    Two readings, and they answer different questions:

    * **`shadow_vs_live_v`** — the signed mean of (shadow V − live V) in real shaped-return units
      on the same states, battle-clustered. This is the staged-promotion meter: a shadow sitting
      systematically BELOW the live critic says the live critic is optimistic about the states the
      factory samples, measured against a ground-truth target rather than argued from a curve.
    * **`shadow_vs_live_v_abs`** — the same gap unsigned, so a large signed value can be told from
      a large scatter that happens to average near zero.

    ⚠️ **THE STANDING CAVEAT, and there is no field that removes it here.** This block compares two
    FITTED heads. A divergence is evidence about the PAIR; it does not say which of them moved, and
    this function computes no external anchor that would. Reading it as "the live critic is
    optimistic" needs the trainer-side `cf/shadow_live_v_vs_label` (live V against the MC label
    itself) beside it — that one has a ground-truth arm. An anchor column here was considered and
    NOT built: the audit's MC is a WIN PROBABILITY and the shadow's units are shaped return, so any
    mapping between them would be a crude direction check dressed as a calibration.

    Reported with `n` and `n_battles` on every line. The shadow is trained off ``mc_return`` rows,
    and a run whose producer never shipped one carries a randomly-initialised head whose divergence
    from the live critic is pure noise — the trainer-side `cf/shadow_coverage` is the scalar that
    says so, since a checkpoint cannot report what it was trained on.
    """
    rows = [r for r in labels if r.get("shadow_value") is not None]
    if not rows:
        return None
    battles = [r["battle"] for r in rows]
    shadow = np.asarray([r["shadow_value"] for r in rows], dtype=float)
    out: dict = {"n": len(rows), "n_battles": len(set(battles)),
                 "shadow_mean": float(shadow.mean())}
    live = [r.get("live_v") for r in rows]
    if all(v is not None for v in live):
        lv = np.asarray(live, dtype=float)
        diff = shadow - lv
        lo, hi = cluster_bootstrap_ci(diff.tolist(), battles, draws=draws, seed=seed)
        out.update(live_v_mean=float(lv.mean()),
                   shadow_vs_live_v=float(diff.mean()),
                   shadow_vs_live_v_ci=[lo, hi],
                   shadow_vs_live_v_abs=float(np.abs(diff).mean()))
    return out


def attach_twin_heads(session, labels: List[dict], npz_cache: "Dict[str, np.ndarray]") -> int:
    """Score every labelled state through the twin heads and the shadow critic, in place.

    Adds ``twin_b_pred`` / ``twin_c_pred`` (P(win) from heads B and C) and ``shadow_value`` /
    ``live_v`` (real-unit values) where the checkpoint carries them; returns how many rows were
    scored. **Zero is a first-class outcome** — a checkpoint without the heads leaves the fields
    ABSENT and every reader downstream omits its columns rather than filling them with a number
    that would read as a measurement.

    BEST-EFFORT, exactly like `attach_evidential`: the audit's products are the labels and the bias
    map, and a model that will not load (architecture drift — 79 of 79 archived runs cannot be
    re-loaded at any given HEAD) must cost the run these columns and nothing else.
    """
    if not labels or not hasattr(session, "probe_model"):
        return 0
    try:
        model, _choice = session.probe_model(labels[0]["battle"] + "_summary.json")
    except Exception as exc:                                            # noqa: BLE001
        print(f"  twin heads: no model ({type(exc).__name__}: {str(exc)[:120]}) — "
              f"columns omitted", flush=True)
        return 0
    if not hasattr(model, "cf_twin_batch"):
        return 0
    obs_rows, targets = [], []
    for r in labels:
        arr = npz_cache.get(r["battle"] + "_states.npz")
        if arr is None or not (0 <= r["inv"] < len(arr)):
            continue
        obs_rows.append(arr[r["inv"]])
        targets.append(r)
    if not obs_rows:
        return 0
    try:
        out = model.cf_twin_batch(np.stack(obs_rows))
    except Exception as exc:                                            # noqa: BLE001
        print(f"  twin heads: forward failed ({type(exc).__name__}: {str(exc)[:120]}) — "
              f"columns omitted", flush=True)
        return 0
    if not out:
        print("  twin heads: this checkpoint carries neither cf_twin_head_b nor cf_shadow_head — "
              "columns omitted", flush=True)
        return 0
    for col, values in out.items():        # `col`, not `field`: `field` is the dataclasses import
        for r, v in zip(targets, values):
            r[col] = float(v)
    print(f"  twin heads: scored {len(targets)} states ({', '.join(sorted(out))})", flush=True)
    return len(targets)
