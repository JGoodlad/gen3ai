#!/usr/bin/env python3
"""PROBE M — census: reconstruct the ONE live BIAS term (`no_progress_tax`) from recorded obs.

Model-free, exact. The reward's anti-stall tax is a FLAT ``-no_progress_penalty`` charged by
``ProgressClock.update()`` on a "charged NO_OP" window, and the SAME clock counter is written into
the observation (``gen3_markovian_progress_v1`` — obs board scalar ``turns_since_progress``; the
whole point of the Markovian design is that obs and reward key on ONE number). So the tax is
reconstructible from two consecutive recorded observations plus the recorded action mask, with no
model and no re-simulation.

THE TIMING, ESTABLISHED EMPIRICALLY (not assumed)
-------------------------------------------------
``gen3_env.embed_battle`` runs ``record -> update_progress_clock -> encode`` and poke-env runs
``embed_battle`` BEFORE ``calc_reward``. So the fold that produces ``obs[t+1]`` is the fold whose
``last_penalty`` lands in the reward for action ``a_t``. The charge therefore belongs to
**decision t**, and is read off the transition ``n(t) -> n(t+1)``.

But the fold's ``curr_ctx`` and its ``legal`` are built from the battle AS IT STANDS AT EMBED TIME,
i.e. the request for decision **t+1**. So both clock gates — ``phase_is_forced_switch`` (sit out)
and "was a switch legal" (charge suppression) — read decision **t+1**'s request, not decision t's.
That is measured, not inferred: over 12,314 windows the sit-out coincides with *decision t+1* being
a forced switch **0/1513** times in violation, against **1216/1512** for *decision t*.

CLASSES (per decision t, the action whose reward carries the charge)
--------------------------------------------------------------------
  TAXED     n rose, a switch was legal          -> `no_progress_tax` fired: -no_progress_penalty
  TRAPPED   n rose, no switch legal             -> a no-op the tax deliberately EXEMPTS
  FROZEN    n held > 0                          -> DENIED: exogenous RNG denial, or an in-grace heal
  PROGRESS  n reset to 0 from > 0               -> the clock's "progress" verdict
  NEUTRAL   n held at 0                         -> progress-or-freeze at zero (untaxed either way)
  SITOUT    the fold's request is a forced swap -> the clock sits out entirely
  CAP       n(t) == 10                          -> an increment is INVISIBLE at the clamp; excluded

Run:
  nice -n 15 python bias_tax_head_alignment_census.py --run models/ai_v9_29_rev1_0823 \
      --out /tmp/probeM_census.jsonl
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

# --- the obs column, verified rather than trusted --------------------------------------------
# `agents/observation/reactive.py:218-224` puts `turns_since_progress` at the reactive block's
# `vec[2]`; the absolute index is asserted below against the LATTICE and the counter dynamics, per
# run, so a layout drift FAILS rather than silently reading a neighbouring feature.
CLOCK_COL_DEFAULT = 1602
CLOCK_CAP = 10
_LAT = np.array([math.log(1.0 + k) / math.log(1.0 + CLOCK_CAP) for k in range(CLOCK_CAP + 1)])

SWITCH_END = 6          # actions 0-5 are switches (agents/action/constants.py)
MOVE_END = 11           # actions 6-9 moves, 10 struggle


def decode_clock(col: np.ndarray) -> np.ndarray:
    """Exact inverse of `ProgressClock.value()`; raises if any value is off the 11-point lattice."""
    d = np.abs(col[:, None] - _LAT[None, :])
    n = d.argmin(axis=1)
    off = d.min(axis=1).max()
    if off > 1e-5:
        raise ValueError(f"obs column is not the progress clock (max lattice distance {off:.3g})")
    return n.astype(np.int16)


def validate_column(files, col: int) -> dict:
    """Hard gate on the decode AND on the fold->window alignment.

    1. **lattice + counter dynamics** — every step-to-step move is +1, hold, or a reset to 0. A
       wrong column, or a `has_state` gap, shows up here as a >+1 jump.
    2. **the opening value** — every battle must open on the same n (`reset()` also runs one
       degenerate pre-first-decision fold, so the measured value is 1, not 0; it is not a charged
       window and the tax reads a DIFFERENCE, so it is inert).
    3. **the sit-out alignment** — reported for BOTH candidate alignments. The gate requires the
       t+1 alignment to be violation-free; the t alignment's rate is carried into the record as the
       evidence that the choice was measured.
    """
    n_batt = n_step = bad_dyn = 0
    opens: dict = {}
    fa = fa_moved = fb = fb_moved = 0
    for f in files:
        z = np.load(f)
        hs = z["has_state"].astype(bool)
        if hs.sum() < 3:
            continue
        n = decode_clock(z["obs"][hs][:, col].astype(np.float64))
        mask = z["action_mask"][hs].astype(bool)
        n_batt += 1
        n_step += len(n)
        opens[int(n[0])] = opens.get(int(n[0]), 0) + 1
        d = np.diff(n)
        bad_dyn += int(((d > 1) | ((d < 0) & (n[1:] != 0))).sum())
        forced = ~mask[:, SWITCH_END:MOVE_END].any(axis=1)
        fa += int(forced[:-1].sum()); fa_moved += int((forced[:-1] & (d != 0)).sum())
        fb += int(forced[1:].sum());  fb_moved += int((forced[1:] & (d != 0)).sum())
    return {"battles": n_batt, "steps": n_step, "illegal_transitions": bad_dyn,
            "opening_values": opens,
            "sitout_align_decision_t": {"n": fa, "violations": fa_moved},
            "sitout_align_decision_t_plus_1": {"n": fb, "violations": fb_moved}}


def classify(n_t: int, n_next: int, switch_legal_next: bool, sitout: bool) -> str:
    if sitout:
        return "SITOUT"
    if n_t >= CLOCK_CAP:
        return "CAP"
    if n_next == n_t + 1:
        return "TAXED" if switch_legal_next else "TRAPPED"
    if n_next == n_t:
        return "FROZEN" if n_t > 0 else "NEUTRAL"
    if n_next == 0:
        return "PROGRESS"
    return "ANOMALY"


def battle_rows(npz_path: str, col: int, run: str, step: str, opponent: str) -> list:
    sm_path = npz_path.replace("_states.npz", "_summary.json")
    if not os.path.exists(sm_path):
        return []
    with open(sm_path) as fh:
        summary = json.load(fh)
    inv = summary.get("invocations", [])
    meta = summary.get("meta", {})
    z = np.load(npz_path)
    obs, hs = z["obs"], z["has_state"].astype(bool)
    wp, vv, mask, act = z["win_probs"], z["values"], z["action_mask"], z["actions"]
    if len(inv) != len(hs):
        return []                                    # trace/summary desync -> drop the battle
    n_all = np.full(len(hs), -1, dtype=np.int16)
    n_all[hs] = decode_clock(obs[hs][:, col].astype(np.float64))

    result = str(meta.get("result", "?"))
    n_turns = int(meta.get("turns", 0) or 0)
    bid = os.path.basename(npz_path)[: -len("_states.npz")]
    rows = []
    for t in range(len(hs) - 1):
        if not (hs[t] and hs[t + 1]):
            continue
        m, mn = mask[t].astype(bool), mask[t + 1].astype(bool)
        sitout = not bool(mn[SWITCH_END:MOVE_END].any())          # fold's request is a forced swap
        cls = classify(int(n_all[t]), int(n_all[t + 1]),
                       bool(mn[:SWITCH_END].any()), sitout)
        rec = inv[t] if t < len(inv) else {}
        rows.append({
            "run": run, "step": step, "opponent": opponent, "battle": bid, "result": result,
            "i": t, "turn": rec.get("turn"), "phase": rec.get("phase"),
            "chosen": rec.get("chosen"), "action": int(act[t]),
            "n_t": int(n_all[t]), "n_next": int(n_all[t + 1]), "cls": cls,
            # decision t's OWN request (what the agent actually faced)
            "self_forced": not bool(m[SWITCH_END:MOVE_END].any()),
            "self_switch_legal": bool(m[:SWITCH_END].any()),
            "n_legal": int(m.sum()),
            "phi": float(wp[t]), "phi_next": float(wp[t + 1]),
            "phi_prev": float(wp[t - 1]) if (t > 0 and hs[t - 1]) else None,
            "v": float(vv[t]), "v_next": float(vv[t + 1]),
            "n_steps": int(len(hs)), "battle_turns": n_turns,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--col", type=int, default=CLOCK_COL_DEFAULT)
    ap.add_argument("--validate-json", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run, "eval_traces", "*", "*", "*_states.npz")))
    if not files:
        print(f"no traces under {args.run}", file=sys.stderr)
        return 1
    v = validate_column(files[:400], args.col)
    print(f"[validate] col={args.col} {v}", file=sys.stderr)
    if (v["illegal_transitions"] or len(v["opening_values"]) != 1
            or v["sitout_align_decision_t_plus_1"]["violations"]):
        print("FATAL: column is not the episode-scoped progress clock, or the fold->window "
              "alignment does not hold", file=sys.stderr)
        return 2
    if args.validate_json:
        with open(args.validate_json, "w") as fh:
            json.dump({"run": args.run, "col": args.col, **v}, fh, indent=1)

    run_name = os.path.basename(args.run.rstrip("/"))
    n_rows = n_batt = 0
    with open(args.out, "w") as out:
        for f in files:
            parts = f.split(os.sep)
            step, opponent = parts[-3], parts[-2]
            try:
                rows = battle_rows(f, args.col, run_name, step, opponent)
            except ValueError as exc:
                print(f"[skip] {f}: {exc}", file=sys.stderr)
                continue
            if rows:
                n_batt += 1
            for r in rows:
                out.write(json.dumps(r) + "\n")
                n_rows += 1
    print(f"[census] {run_name}: {n_batt} battles, {n_rows} decision rows -> {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
