"""THE AMORTIZATION RESIDUAL PROBE — the Q head against the simulator it is amortizing.

``gen3_q_winprob_head_v1``, E5 step 5 (MEASURE; ledger 5edbd05). The `QWinProbHead` claims to
replace an eleven-way simulator sweep with one forward. This script is the claim's meter: at
recorded decisions it reads the head's per-action ``P(win|s,a)`` and compares it against the
prober's own one-ply sweep — the SAME `lookahead` machinery that re-rolls each legal action through
the real simulator, materializes the successor, and scores it with the loaded model.

**The headline is the AMORTIZATION RESIDUAL**: how far the head's per-action row sits from the
sweep's, and how well the two RANK the same actions. The ledger's reading of it:

  * **shrinking** ⇒ the AlphaZero ratchet — one-ply search's value has migrated into the net, and
    search must go deeper to add anything;
  * **stubbornly large on a class of states** ⇒ those are the states that genuinely need live
    search, which is a triage signal for the time manager, not a defect.

🚨 **IT IS A PREDICTIVE METER, NOT A BEHAVIORAL ONE.** A small residual says the head reproduces
the sweep; it says NOTHING about whether the policy plays better for having it. Iteration 2's
lesson is that those two must be kept as separate numbers, so this script deliberately reports no
win rate and no policy change.

⚠️ **THE GROUND TRUTH IS ITSELF A MODEL READ.** `lookahead` scores each re-rolled successor with
the loaded checkpoint's own critic / win head, so the comparison is "does the Q head reproduce what
this network would have said after a simulator step", not "does it reproduce the truth". That is
the right question for an amortization meter — the thing being amortized IS the sweep — but it is
not a calibration result, and a run whose critic is badly calibrated will show a small residual
against a wrong target. The `--metric win_prob` mode compares like with like (P(win) vs P(win));
`--metric value` compares RANKINGS only, because V(s') and P(win|s,a) are not in the same units.

Two modes:

    # (1) the INIT-STATE SANITY — no traces, no checkpoint, ~2 s. Verifies the zero-init contract:
    #     an untrained head emits P = 0.5 for every action, i.e. an uninformative uniform ranking.
    python -m main.q_amortization --self-check

    # (2) the REAL comparison, over a run's bridge-eval traces
    python -m main.q_amortization models/run_<ts> [--battles 20] [--worst 3] [--metric win_prob]

Mode (2) needs a run at the CURRENT architecture whose traces carry a ``*_reconstruction.json``
sibling (bridge-eval only) and whose checkpoint was trained with ``--q-winprob-mode read_only``. It
reports, per decision and pooled: Spearman rank correlation between the two per-action rows,
top-1 agreement, and the mean absolute residual.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Spearman's rho over two equal-length rows, or ``None`` when it is undefined.

    Undefined means fewer than 3 points, or either row constant (a zero-variance rank vector has no
    correlation with anything). Returning ``None`` rather than 0.0 is load-bearing here: an
    UNTRAINED Q head emits a constant row by construction, and reporting that as rho = 0 would put
    "the head has learned nothing" and "the head has learned something uncorrelated" in the same
    bucket — which is exactly the distinction this probe exists to make.

    Ties are handled by average ranking, so a head that ties two actions is not silently ordered.
    """
    if len(a) != len(b) or len(a) < 3:
        return None
    ra, rb = _avg_rank(a), _avg_rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _avg_rank(xs: Sequence[float]) -> np.ndarray:
    """Average ranks (1-based), ties shared — the standard Spearman pre-transform."""
    arr = np.asarray(xs, dtype=float)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def compare_rows(q_row: Dict[int, float], truth_row: Dict[int, float]) -> Dict[str, object]:
    """One decision's comparison: the Q head's row vs the one-ply sweep's, over the SHARED actions.

    The intersection is taken on purpose. The Q head scores all eleven slots (the extractor forward
    reads no mask), while the sweep covers only the actions that were legal AND whose re-roll
    produced a scorable successor — a terminal or stuck arm has no V(s′) at all. Scoring the head
    on a slot the sweep could not evaluate would be comparing it against nothing.
    """
    actions = sorted(set(q_row) & set(truth_row))
    if len(actions) < 2:
        return {"n_actions": len(actions), "rho": None, "top1_agree": None, "residual": None}
    q = [q_row[a] for a in actions]
    t = [truth_row[a] for a in actions]
    top1 = int(actions[int(np.argmax(q))] == actions[int(np.argmax(t))])
    return {
        "n_actions": len(actions),
        "actions": actions,
        "rho": spearman(q, t),
        "top1_agree": top1,
        # Only meaningful when both rows are probabilities; the caller sets it to None under
        # `--metric value`, where V(s') and P(win|s,a) are not in the same units and a difference
        # would be a number with no interpretation.
        "residual": float(np.mean(np.abs(np.asarray(q) - np.asarray(t)))),
    }


def self_check() -> int:
    """The INIT-STATE SANITY: a zero-init Q head is uniform, hence uninformative. Returns an exit code.

    This is the half of the probe that ships GATED rather than merely runnable — it needs no
    checkpoint, no traces and no simulator, so it can assert the head's cold-start contract on any
    machine. Three claims, each one a way the head could be silently broken at birth:

      1. every action's P(win|s,a) is EXACTLY 0.5 (the shared scorer is zero-init in weight AND
         bias, so every logit is exactly 0);
      2. the resulting ranking is therefore a total tie — `spearman` returns ``None`` rather than a
         number, which is what "uninformative" has to look like to a consumer;
      3. the row is [ACTION_SPACE_SIZE]-wide and its columns are the action space.
    """
    import torch

    from agents.action.constants import ACTION_SPACE_SIZE
    from agents.model.arch_constants import D_MODEL
    from agents.model.q_winprob_head import QWinProbHead

    torch.manual_seed(0)
    head = QWinProbHead(move_token_dim=D_MODEL, d_model=D_MODEL, ctx_dim=D_MODEL,
                        move_cell_dim=3, switch_cell_dim=5)
    B = 4
    logits = head(torch.randn(B, D_MODEL), torch.randn(B, 4, D_MODEL), torch.ones(B, 4),
                  torch.randn(B, 6, D_MODEL), torch.randn(B, 4, 3), torch.randn(B, 6, 5))
    probs = torch.sigmoid(logits)
    ok = True
    if tuple(logits.shape) != (B, ACTION_SPACE_SIZE):
        print(f"FAIL shape: {tuple(logits.shape)} != {(B, ACTION_SPACE_SIZE)}")
        ok = False
    if not torch.equal(probs, torch.full_like(probs, 0.5)):
        print(f"FAIL uniform: max|P - 0.5| = {float((probs - 0.5).abs().max()):.3e} "
              f"(the zero-init scorer must make every logit EXACTLY 0)")
        ok = False
    if spearman(probs[0].tolist(), list(range(ACTION_SPACE_SIZE))) is not None:
        print("FAIL uninformative: a constant row must produce rho=None, not a number")
        ok = False
    print(f"{'PASS' if ok else 'FAIL'}: zero-init QWinProbHead emits P(win|s,a) = 0.5 for all "
          f"{ACTION_SPACE_SIZE} actions ⇒ a total tie ⇒ an honest uninformative cold start")
    return 0 if ok else 1


def probe_run(run_dir: str, *, battles: int, worst: int, metric: str,
              impl: str) -> Dict[str, object]:
    """Compare the Q head against the one-ply sweep over a run's traces. Returns the report dict."""
    from main.prober.session import ProbeSession

    session = ProbeSession(run_dir, impl=impl)
    decisions: List[Dict[str, object]] = []
    errors: List[str] = []
    for b in session.battles()[:battles]:
        bid = str(b.get("id") or b.get("battle") or b.get("tag"))
        try:
            la = session.lookahead(bid, worst=worst)
        except Exception as exc:                       # a trace without a reconstruction sibling,
            errors.append(f"{bid}: {exc}")             # a drifted arch, a replay desync — all
            continue                                   # per-battle and none fatal to the sweep
        for dec in la.get("decisions", []):
            try:
                got = _one_decision(session, bid, dec, metric)
            except Exception as exc:
                errors.append(f"{bid}#{dec.get('inv')}: {exc}")
                continue
            if got is not None:
                decisions.append(got)
    return _summarize(decisions, errors, metric)


def _one_decision(session, battle_id: str, dec: dict, metric: str) -> Optional[Dict[str, object]]:
    """One lookahead decision → its comparison row, or ``None`` when the sweep covered too little.

    It reaches through `ProbeSession`'s private `_battle` / `_npz` / `_model_for` rather than
    re-opening the trace tree, deliberately: those three are what `lookahead` itself used to
    produce ``dec``, so the obs this scores the Q head on is bit-for-bit the obs the sweep
    branched from. Re-deriving it would be a second decode of the same file, i.e. a second thing
    that can drift.
    """
    inv = dec.get("inv")
    if inv is None:
        return None
    key = "win_prob_crn" if metric == "win_prob" else "value_crn"
    truth = {int(c["action"]): float(c[key])
             for c in dec.get("candidates", []) if c.get(key) is not None}
    if len(truth) < 2:
        return None
    trace = session._battle(battle_id)
    model, _ = session._model_for(trace)
    npz = session._npz(trace)
    obs = np.asarray(npz["obs"][int(inv)], dtype=np.float32)
    mask = np.asarray(npz["action_mask"][int(inv)], dtype=np.float32)
    q = model.q_winprob_at(obs, mask)
    if q is None:
        raise RuntimeError(
            "the loaded checkpoint has no q_winprob_head — this probe needs a run trained with "
            "--q-winprob-mode read_only. (Everything else about the run can be fine; the head is "
            "a state_dict delta, so it cannot be added after the fact.)")
    q_row = {a: float(q[a]) for a in truth}
    out = compare_rows(q_row, truth)
    if metric != "win_prob":
        # V(s′) and P(win|s,a) are not in the same units, so only the RANKING transfers.
        out["residual"] = None
    out["inv"] = int(inv)
    out["turn"] = dec.get("turn")
    out["battle"] = battle_id
    return out


def _summarize(decisions: List[Dict[str, object]], errors: List[str],
               metric: str) -> Dict[str, object]:
    """Pool the per-decision rows. Every aggregate skips its own undefined entries, never zeroes them."""
    rhos = [d["rho"] for d in decisions if d["rho"] is not None]
    tops = [d["top1_agree"] for d in decisions if d["top1_agree"] is not None]
    res = [d["residual"] for d in decisions if d["residual"] is not None]
    return {
        "metric": metric,
        "n_decisions": len(decisions),
        # A row with rho=None is an UNINFORMATIVE Q head on that decision (a constant row), which
        # is a finding, not a missing measurement — so it is counted rather than dropped silently.
        "n_rho_undefined": len(decisions) - len(rhos),
        "spearman_mean": float(np.mean(rhos)) if rhos else None,
        "spearman_median": float(np.median(rhos)) if rhos else None,
        "top1_agreement": float(np.mean(tops)) if tops else None,
        # THE HEADLINE (win_prob metric only): mean |Q_head − one-ply sweep| per labelled action.
        "amortization_residual": float(np.mean(res)) if res else None,
        "errors": errors[:10],
        "n_errors": len(errors),
        "decisions": decisions,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m main.q_amortization",
        description="The Q head's AMORTIZATION RESIDUAL against the prober's one-ply sweep.")
    ap.add_argument("run_dir", nargs="?", help="models/run_<timestamp> (omit with --self-check)")
    ap.add_argument("--self-check", action="store_true",
                    help="verify the zero-init contract only — no checkpoint, no traces, no sim")
    ap.add_argument("--battles", type=int, default=10, help="how many battles to sweep")
    ap.add_argument("--worst", type=int, default=3,
                    help="decisions per battle (the lookahead's delta-craters)")
    ap.add_argument("--metric", choices=("win_prob", "value"), default="win_prob",
                    help="win_prob compares LIKE with like and yields a residual; value compares "
                         "RANKINGS only (V(s') is not in P(win) units)")
    ap.add_argument("--impl", choices=("node", "rust"), default="node",
                    help="the offline re-roll driver")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = ap.parse_args(argv)

    if args.self_check:
        return self_check()
    if not args.run_dir:
        ap.error("run_dir is required unless --self-check is given")
    report = probe_run(args.run_dir, battles=args.battles, worst=args.worst,
                       metric=args.metric, impl=args.impl)
    if args.json:
        print(json.dumps({k: v for k, v in report.items()}, indent=2, default=str))
        return 0
    print(f"metric={report['metric']}  decisions={report['n_decisions']}  "
          f"errors={report['n_errors']}")
    print(f"  spearman(mean/median) : {_fmt(report['spearman_mean'])} / "
          f"{_fmt(report['spearman_median'])}   (undefined on "
          f"{report['n_rho_undefined']} decisions — a CONSTANT Q row, i.e. an uninformative head)")
    print(f"  top-1 agreement       : {_fmt(report['top1_agreement'])}")
    print(f"  AMORTIZATION RESIDUAL : {_fmt(report['amortization_residual'])}"
          f"{'' if report['metric'] == 'win_prob' else '   (n/a under --metric value)'}")
    for e in report["errors"]:
        print(f"  ! {e}")
    return 0


def _fmt(x: object) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.4f}"


if __name__ == "__main__":                                        # pragma: no cover
    sys.exit(main())
