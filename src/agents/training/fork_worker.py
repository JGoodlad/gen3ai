"""The CHILD process that PLAYS the branches of a fork (`--fork-fraction`, `gen3_fork_v1`).

    python -m agents.training.fork_worker --job <job.json> --out <out.json>

One short-lived process per slice of a rollout's fork batch. It loads ONE weights snapshot, plays
every assigned fork's branches to a terminal, writes the branch TRANSITIONS to an ``.npz`` beside
its result JSON, and exits.

**Why a subprocess and not a thread** — the three reasons eval, the search teacher, the
counterfactual producer and the win-prob rollout labeller all use children: poke-env drives every
battle on a single global ``POKE_LOOP`` thread, so a continuation inside the trainer would
serialise against nothing and contend with everything; the live policy trunk MUTATES during
``train()`` while these forwards must see one fixed snapshot; and a wedged bridge child then kills
a worker instead of the run.

WHAT ONE BRANCH IS
------------------
Replay the recorded episode to the START of the fork turn on the record's OWN resolved seed, send
``action`` there instead of what the trainee actually sent, then play the rest LIVE — the snapshot
on both sides, sampling at temperature 1.0 — to a terminal. `utils.bridge.counterfactual` owns all
of that; the only thing added here is the CAPTURE.

THE CAPTURE, AND THE TWO OFF-BY-ONES IT CLOSES
----------------------------------------------
The rows this worker returns are ordinary PPO transitions and they must line up with the branch
exactly:

* **row 0 is the FORK STEP** — the state the decision was made FROM, with the BRANCH's action. Its
  obs comes from `install_scripted_prefix`'s ``on_scripted_decision`` hook, which fires at that
  decision with the obs the scripted path built, so row 0 describes the fork state and not the
  state after it. (The same hook exists for the same reason in the shadow critic's `mc_return`
  labels: a tracker hooked only into the LIVE ``choose_move`` would begin at turn T+1.)
* **row 1 is the SUCCESSOR** ``s'`` — the first LIVE decision, i.e. the state a one-ply search
  would score and the state the arm's whole pairwise endpoint is about. A branch that ENDS at the
  fork turn has no successor, and is reported with ``n_rows == 1`` rather than with a fabricated
  one.

The substitute is handed over as a **callable**, not a string: a buffer row knows its action as an
INDEX, and the index→choice mapping is a function of the LIVE legal set. Resolving it through the
player's own ``action_to_order`` at the divergence decision is the only way to be sure the branch
sends what the mask said it could.

A live decision's action index is recovered by recording what ``_predict_best_action`` returned,
and committing only the LAST attempt of a ``choose_move`` call — `RLPlayer` RE-DECIDES on a stale
decision race and restores its tracker, so a superseded attempt describes a state the battle left.

WHAT IS *NOT* HERE
------------------
No values, no log-probs, no advantages. Those are computed in the PARENT, by the live policy, in
one batched forward — and that is not an optimisation. The parent's weights ARE the weights that
collected the buffer, so ``old_log_prob`` computed there is exactly the collecting policy's and
the PPO ratio is 1.0 at the first epoch, as the objective requires. A log-prob computed in a child
from a re-loaded snapshot would be the same number only up to serialisation, and "only up to" is
not a property an importance ratio survives.

No rewards either: under ``--critic winprob`` the reward stream is the TERMINAL WIN INDICATOR alone
(`combination_checks.winprob_critic_needs_the_indicator_terminal`), so a branch's entire reward
sequence is reconstructible from its outcome bit. That is why the arm REFUSES any other critic —
not taste, but the fact that a shaped reward cannot be recovered outside the env that computed it.

A branch that reaches the 250-turn stall-forfeit cap is a DRAW AT CAP whose win/loss is decided by
SEAT rather than by the position (`counterfactual._battle_outcome`), so it is reported as
``capped`` and the parent DROPS it — never scores it 0.5, and never lets it into a sibling pair.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

#: Branch continuations at or above which this worker compiles its extractor. The
#: `win_prob_rollout_worker` arithmetic, verbatim: a compile costs ~40 s once per process and buys
#: 6.4x on every decision (26.3 ms eager -> 4.1 ms, B=1 CPU, measured 2026-08-23); a continuation
#: is ~104 decisions, so it pays for itself at ~18 continuations and loses badly below that.
COMPILE_BREAK_EVEN_BRANCHES = 18


class BranchCapture:
    """Record ``(obs, action_mask, action)`` for every decision OUR side commits in one branch.

    Wraps the player's ``_predict_best_action`` (which returns the chosen index) to stage a row,
    and ``choose_move`` to COMMIT the last staged row of that call. The two-level shape is what
    handles `RLPlayer`'s stale-decision RE-DECIDE: each attempt stages a row and only the attempt
    whose order was accepted survives. A call that ends in ``choose_default_move`` stages an index
    of ``None`` and commits nothing.
    """

    def __init__(self, player) -> None:
        self.rows: List[Dict[str, Any]] = []
        self._staged: List[Dict[str, Any]] = []
        self._obs = None
        orig_embed = player.embed_battle
        orig_predict = player._predict_best_action
        orig_choose = player.choose_move

        def embed(battle):
            # THE OBS IS TAKEN HERE AND NOWHERE ELSE. `BattleContext` (the tracker's ``last_ctx``)
            # carries the mask but not the observation vector, and re-running `embed_battle` to
            # get one would advance the tracker's turn history a second time — a phantom turn in
            # the very obs the row is supposed to hold. So the live build is stashed as it happens.
            out = orig_embed(battle)
            self._obs = out
            return out

        def predict(battle, **kw):
            self._obs = None
            out = orig_predict(battle, **kw)
            idx = out[0] if isinstance(out, tuple) else out
            if idx is not None and self._obs is not None:
                self._staged.append({
                    "obs": np.asarray(self._obs["observation"], dtype=np.float32).reshape(-1),
                    "mask": np.asarray(self._obs["action_mask"], dtype=np.int8).reshape(-1),
                    "action": int(idx),
                })
            return out

        def choose(battle):
            self._staged = []
            order = orig_choose(battle)
            if self._staged:
                # The LAST attempt is the committed one: `RLPlayer` RE-DECIDES on a stale-decision
                # race and RESTORES its tracker, so every earlier attempt of this call describes a
                # state the battle has already left.
                self.rows.append(self._staged[-1])
            self._staged = []
            return order

        player.embed_battle = embed
        player._predict_best_action = predict
        player.choose_move = choose

    def fork_row(self, obs_dict, action: int) -> None:
        """Commit ROW 0 — the fork step, whose decision is SCRIPTED and therefore invisible to the
        live ``choose_move`` wrapper above."""
        self.rows.insert(0, {
            "obs": np.asarray(obs_dict["observation"], dtype=np.float32).reshape(-1),
            "mask": np.asarray(obs_dict["action_mask"], dtype=np.int8).reshape(-1),
            "action": int(action),
        })


def _live_decisions(res: Dict[str, Any], divergence_turn: int) -> float:
    """An ESTIMATE of the live policy decisions one branch cost, for the cost meter.

    ``(final_turn - divergence_turn + 1) x 2`` — both sides decide once per turn from the handoff.
    `win_prob_rollout_worker._live_decisions` verbatim, and named an estimate for its reasons:
    forced switches add rounds and a battle that ends mid-turn subtracts one.
    """
    turns = float(res.get("turns") or 0.0)
    return max(0.0, (turns - float(divergence_turn) + 1.0) * 2.0)


def run_job(job: Dict[str, Any], npz_path: str) -> Dict[str, Any]:
    """Play every fork in ``job``, write its rows to ``npz_path``, return the result dict."""
    import dataclasses

    from agents.training.cf_producer import _trainee_side, load_snapshot
    from agents.training.fork_crn import player_seeds
    from utils.bridge.counterfactual import replay_counterfactual
    from utils.bridge.reconstruction import ReconstructionRecord

    forks: List[Dict[str, Any]] = list(job.get("forks") or ())
    impl = str(job.get("impl") or "rust")
    crn = str(job.get("crn") or "dice_and_draws")
    n_branches = sum(len(f.get("actions") or {}) for f in forks)

    t0 = time.perf_counter()
    snap = load_snapshot(str(job["weights"]), None, device="cpu",
                         compile_extractor=n_branches >= COMPILE_BREAK_EVEN_BRANCHES)
    load_seconds = time.perf_counter() - t0

    obs_rows: List[np.ndarray] = []
    mask_rows: List[np.ndarray] = []
    act_rows: List[int] = []
    results: List[Dict[str, Any]] = []

    for fk in forks:
        out: Dict[str, Any] = {"id": int(fk["id"]), "branches": {}, "error": None}
        try:
            record = ReconstructionRecord.load(str(fk["record"]))
            side = _trainee_side(record)
            rec = dataclasses.replace(record, trainee_username=record.username(side))
            other = "p2" if side == "p1" else "p1"
            seed = record.start_options().get("seed")
            turn = int(fk["turn"])
            salt = str(fk.get("salt") or f"{fk['record']}:{turn}")
            t_seed, o_seed = player_seeds(salt, crn)
            for name, action in sorted((fk.get("actions") or {}).items()):
                row: Dict[str, Any] = {"action": int(action), "outcome": None, "capped": False,
                                       "turns": 0, "decisions": 0.0, "row_start": -1,
                                       "n_rows": 0, "error": None}
                try:
                    trainee = snap.make_player(rec, side, role="Fa")
                    opponent = snap.make_player(rec, other, role="Fb")
                    # COMMON RANDOM NUMBERS on the DRAWS: the same seed for every branch of this
                    # fork, a different one per side. See `agents.training.fork_crn`.
                    trainee._policy_seed = t_seed
                    trainee._policy_gens = {}
                    opponent._policy_seed = o_seed
                    opponent._policy_gens = {}
                    cap = BranchCapture(trainee)
                    seen = {"done": False, "at_fork": False}

                    def _sub(player, battle, _a=int(action), _seen=seen):
                        # Resolved against the LIVE legal set through the player's own mapper —
                        # the only mapping that is guaranteed to agree with the action_mask the
                        # buffer row carries. A raise here is deliberately NOT caught by
                        # `install_scripted_prefix`; it surfaces as this branch's error.
                        #
                        # It also ARMS the row-0 hook. The hook fires on every scripted decision,
                        # and a turn comparison would be a guess (a forced switch can share the
                        # fork's turn number); this is called from the substitute branch ONLY, one
                        # statement before the hook, so the arming is exact rather than heuristic.
                        _seen["at_fork"] = True
                        return player.action_to_order(_a, battle).message[len("/choose "):]

                    def _hook(battle, obs_dict, _choice, _cap=cap, _seen=seen, _a=int(action)):
                        # ROW 0 — the fork step's own obs, which the SCRIPTED path built and the
                        # live `choose_move` wrapper therefore never sees.
                        if _seen.get("at_fork") and not _seen["done"] and obs_dict is not None:
                            _seen["done"] = True
                            _cap.fork_row(obs_dict, _a)

                    res = replay_counterfactual(
                        rec, trainee=trainee, opponent=opponent, divergence_turn=turn,
                        substitute_choice=_sub, seed=seed,
                        # COMMON RANDOM NUMBERS on the DICE: ONE stream for the whole line, which
                        # is `replay_counterfactual`'s default and the pairing the offline fork
                        # dataset used. Never a post-divergence reseed — that is the knob that
                        # UNPAIRS the branches.
                        post_t_seed=None,
                        trainee_decision_hook=_hook, impl=impl)
                    row["outcome"] = {"win": 1.0, "loss": 0.0}.get(str(res.get("outcome")))
                    row["capped"] = bool(res.get("capped"))
                    row["turns"] = int(res.get("turns") or 0)
                    row["decisions"] = _live_decisions(res, turn)
                    row["substitute"] = res.get("substitute")
                    row["script_exhausted"] = list(res.get("script_exhausted") or ())
                    if cap.rows and not row["capped"] and row["outcome"] is not None:
                        row["row_start"] = len(obs_rows)
                        row["n_rows"] = len(cap.rows)
                        for r in cap.rows:
                            obs_rows.append(r["obs"])
                            mask_rows.append(r["mask"])
                            act_rows.append(int(r["action"]))
                except Exception as exc:                                    # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                out["branches"][name] = row
        except Exception as exc:                                            # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        results.append(out)

    if obs_rows:
        np.savez(npz_path, obs=np.asarray(obs_rows, dtype=np.float32),
                 mask=np.asarray(mask_rows, dtype=np.int8),
                 action=np.asarray(act_rows, dtype=np.int64))
    return {"results": results, "load_seconds": load_seconds, "compiled": bool(snap.compiled),
            "rows": len(obs_rows), "npz": npz_path if obs_rows else None, "pid": os.getpid()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", required=True, help="path to the job JSON written by the parent")
    ap.add_argument("--out", required=True, help="where to write the result JSON")
    args = ap.parse_args(argv)
    with open(args.job) as f:
        job = json.load(f)
    npz = str(args.out) + ".npz"
    try:
        payload = run_job(job, npz)
    except Exception as exc:                                                # noqa: BLE001
        # A worker that dies must say so in its RESULT FILE, not only in an exit code: the parent
        # treats a missing fork as "this fork produced no rows" and a fatal as "report it", and
        # those are different facts about the run.
        payload = {"results": [], "fatal": f"{type(exc).__name__}: {str(exc)[:300]}"}
    tmp = str(args.out) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
