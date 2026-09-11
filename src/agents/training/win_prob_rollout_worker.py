"""The CHILD process that plays the continuations for `--win-prob-rollout-target`.

    python -m agents.training.win_prob_rollout_worker --job <job.json> --out <out.json>

One short-lived process per slice of a rollout's labelling batch. It loads ONE weights snapshot,
plays every assigned state's ``R`` continuations, writes one JSON result file and exits.

**Why a subprocess and not a thread.** The same three reasons eval, the search teacher and the
counterfactual producer all use children: poke-env drives every battle on a single global
``POKE_LOOP`` thread, so a rollout inside the trainer would serialise against nothing and contend
with everything; the live policy trunk MUTATES during ``train()`` while these forwards must see one
fixed snapshot; and a wedged bridge child then kills a worker instead of the run.

**Why short-lived.** A persistent pool would amortise the model load, but it would have to be fed
fresh weights every iteration anyway (the label must be measured under the CURRENT policy, and the
policy moves every rollout), and a pool that survives a ``train()`` is state the trainer has to own
across its own restarts. A spawn costs one model load; at production a PPO iteration is minutes, so
the load is a small constant beside the continuations it enables. The cost is MEASURED and reported
(``load_seconds`` in the result) rather than assumed.

**``torch.compile`` is decided by ARITHMETIC, not taste** (:data:`COMPILE_BREAK_EVEN_ARMS`). A
compile costs ~40 s once per process and buys 6.4x on every decision (26.3 ms eager -> 4.1 ms, B=1
CPU, measured 2026-08-23); a continuation is ~104 decisions, so it pays for itself at ~18
continuations and loses badly below that. The worker compiles iff its own assignment clears the
bar, and says which it did.

THE LABEL THIS FILE PRODUCES. For each assigned ``(record, turn)``: replay the recorded episode to
the START of ``turn``, then play BOTH sides live from there with the snapshot's own weights,
sampling at temperature 1.0, on fresh post-divergence dice — ``R`` times, on ``R`` different dice.
``wins / n`` is the state's Monte-Carlo win probability. A line that reaches the 250-turn
stall-forfeit cap scores **0.5** (`gen3_cf_draw_at_cap_v1`: at the cap both sides forfeit and the
winner is decided by which ``FORCELOSE`` the sim processes first — on a training record, always
p1's, i.e. always the trainee's), and so does a tie; the scorer is `cf_producer.rollout_outcome_score`
verbatim, so a rollout label from here and a label from the counterfactual factory are the same
quantity measured the same way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

#: Continuations at or above which this worker compiles its extractor. 40 s of compile against
#: ~104 decisions x (26.3 - 4.1) ms saved per continuation => break-even at ~17.3; 18 rounds up.
COMPILE_BREAK_EVEN_ARMS = 18


def _live_decisions(res: Dict[str, Any], divergence_turn: int) -> float:
    """An ESTIMATE of the live policy decisions one continuation cost, for the cost meter.

    ``(final_turn - divergence_turn + 1) x 2`` — both sides decide once per turn from the handoff.
    An estimate and named as one: forced switches add rounds and a battle that ends mid-turn
    subtracts one, so it is right to within a few percent and it is the only number available
    without instrumenting `poke-env`'s choose_move on a path that is already the hot loop.
    """
    turns = float(res.get("turns") or 0.0)
    return max(0.0, (turns - float(divergence_turn) + 1.0) * 2.0)


def run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Play every state in ``job`` and return the result dict written to ``--out``."""
    from agents.training.cf_producer import (_trainee_side, rollout_outcome_score,
                                             load_snapshot)
    from main.prober.falsifier import fresh_seeds
    from utils.bridge.counterfactual import replay_counterfactual
    from utils.bridge.reconstruction import ReconstructionRecord
    import dataclasses

    states: List[Dict[str, Any]] = list(job.get("states") or ())
    R = int(job.get("rollouts") or 8)
    impl = str(job.get("impl") or "rust")
    n_arms = len(states) * R
    t0 = time.perf_counter()
    snap = load_snapshot(str(job["weights"]), None, device="cpu",
                         compile_extractor=n_arms >= COMPILE_BREAK_EVEN_ARMS)
    load_seconds = time.perf_counter() - t0

    results: List[Dict[str, Any]] = []
    for st in states:
        out: Dict[str, Any] = {"id": int(st["id"]), "wins": 0.0, "n": 0, "capped": 0,
                               "decisions": 0.0, "error": None}
        try:
            record = ReconstructionRecord.load(str(st["record"]))
            side = _trainee_side(record)
            rec = dataclasses.replace(record, trainee_username=record.username(side))
            seed = record.start_options().get("seed")
            turn = int(st["turn"])
            other = "p2" if side == "p1" else "p1"
            # The R dice, derived from the STATE's own provenance so a rerun of the same buffer
            # draws the same continuations — the `cf_q_labels.q_arm_seeds` derivation, and for its
            # reason: `fresh_seeds(4)` is a strict prefix of `fresh_seeds(8)`, so a smaller R is a
            # sub-sample of the same dice rather than a different experiment.
            seeds = fresh_seeds(R, salt=f"{st.get('salt') or st['record']}:{turn}")
            for ps in seeds:
                try:
                    res = replay_counterfactual(
                        rec,
                        trainee=snap.make_player(rec, side, role="Wa"),
                        opponent=snap.make_player(rec, other, role="Wb"),
                        # substitute_choice=None => our side plays LIVE at the divergence turn.
                        # That is what makes the label V(s) — the value of the STATE — rather than
                        # Q(s, a_recorded), which is what `cf_producer` measures by passing the
                        # recorded choice here. The distinction is the whole point: the win-prob
                        # head's target is a state value.
                        divergence_turn=turn, substitute_choice=None,
                        seed=seed, post_t_seed=ps, impl=impl)
                except Exception as exc:                                   # noqa: BLE001
                    out["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                    continue
                out["n"] += 1
                out["wins"] += float(rollout_outcome_score(res))
                out["capped"] += int(bool(res.get("capped")))
                out["decisions"] += _live_decisions(res, turn)
        except Exception as exc:                                           # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        results.append(out)
    return {"results": results, "load_seconds": load_seconds, "compiled": bool(snap.compiled),
            "pid": os.getpid()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", required=True, help="path to the job JSON written by the parent")
    ap.add_argument("--out", required=True, help="where to write the result JSON")
    args = ap.parse_args(argv)
    with open(args.job) as f:
        job = json.load(f)
    try:
        payload = run_job(job)
    except Exception as exc:                                               # noqa: BLE001
        # A worker that dies must say so in its RESULT FILE, not only in an exit code: the parent
        # treats a missing label as "this state keeps its terminal bit" and a fatal as "report it",
        # and those are different facts about the run.
        payload = {"results": [], "fatal": f"{type(exc).__name__}: {str(exc)[:300]}"}
    tmp = str(args.out) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
