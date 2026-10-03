"""THE FORK ARM'S FILL TABLE (`--fork-fraction`, `gen3_fork_v1`) — what an injected branch row holds.

A branch row is not played inside a `Gen3Env`, so the capture yields ``observation`` and
``action_mask`` and nothing else — while the rollout buffer's obs Dict may carry a dozen flag-gated
LABEL keys. :data:`FILL` declares, per key, what an injected row honestly holds; a key that is not
in the table makes the arm REFUSE AT SETUP (:func:`unfillable_keys`, :func:`refusal_text`), naming
the key and the flag. That is deliberate: the alternative is a heuristic ("anything ending in
``_mask`` is zero"), and a heuristic would keep working, silently and wrongly, the first time someone
adds a key it happens to match.

The Rust fork pass (`agents/training/rust_rollout/fork.py`) is the only consumer. The PYTHON arm's
other halves that lived here — the branch-row builder, the GAE, and the ``ForkRolloutBuffer`` that
grew the SB3 buffer — were deleted with the arm (deletion pass L5); the Rust arm's rows ride the
collector's completed-game FIFO instead (`designs/training/forks.md` §14). THE MASK RULE (the fork
step is out of the POLICY term for EVERY branch, the top-2 included, and fully in the value terms)
is pinned by `rust_rollout/fork_test.py::test_the_fill_hands_the_pg_mask_to_the_learner_and_the_term_renormalises`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from agents.training.fork_arm import PG_MASK_KEY
from agents.training.opp_intent_labels import KIND_UNKNOWN, SWITCH_SLOT_NONE

#: Injected rows are played against a SELF-LIKE opponent (the current snapshot), which is the
#: `MaskableAgentWrapper.OPP_CLASS_POOL` population. Naming it POOL rather than the parent
#: episode's real class is the honest label: the class tags WHO THE ROW WAS PLAYED AGAINST, and
#: these rows were not played against the parent's opponent.
_OPP_CLASS_POOL = 1


def _const_block(value, shape, dtype):
    return lambda n, ctx, _v=value, _s=shape, _d=dtype: np.full((n,) + tuple(_s), _v, dtype=_d)


#: What an INJECTED row holds for every obs key other than ``observation`` / ``action_mask``.
#: Each entry is ``(builder, why)``; the builder takes ``(n_rows, ctx)`` where ``ctx`` carries
#: ``outcome`` (the branch's win bit) and ``pg_mask`` (the per-row policy-term multiplier) and
#: returns an array whose leading axis is the row axis.
#:
#: 🚨 **THE RULE FOR A LABEL THE BRANCH CANNOT SUPPLY IS "NOT SCORED", NEVER A GUESS.** Every
#: privileged belief label is a fact about the OPPONENT'S TEAM read from ``battle2`` inside the
#: env; a branch has no env, so those rows are masked out of their losses instead of being filled
#: with a plausible number. The cost is supervision on ~1 row in 3; the alternative cost is a
#: belief head trained on fiction.
FILL: Dict[str, Tuple[Any, str]] = {
    # ── the two the arm actually MEANS ──────────────────────────────────────────────────────
    "win_target": (lambda n, ctx: np.full((n, 1), float(ctx["outcome"]), dtype=np.float32),
                   "the BRANCH's own terminal outcome — the label the whole arm exists to add"),
    "win_mask": (_const_block(1.0, (1,), np.float32),
                 "scored: a branch always ran to a real terminal (a CAPPED one is dropped before "
                 "it gets here)"),
    PG_MASK_KEY: (lambda n, ctx: np.asarray(ctx["pg_mask"], dtype=np.float32).reshape(n, 1),
                  "0.0 on the FORK STEP, 1.0 after — see THE MASK RULE"),
    # ── honest present-state values ─────────────────────────────────────────────────────────
    "opp_class": (_const_block(_OPP_CLASS_POOL, (1,), np.int64),
                  "the branch WAS played against a self-like (pool) opponent — the ecology "
                  "approximation, labelled rather than hidden"),
    # ── not reconstructible outside the env ⇒ NOT SCORED ────────────────────────────────────
    "win_margin": (_const_block(0.0, (1,), np.float32),
                   "Phi_mat lives in the env's reward manager and a branch has none. 0.0 is the "
                   "CONTESTED stratum, which is where a contested fork's rows belong anyway — but "
                   "it is a FILL, so `--win-prob-strata-weight` is REFUSED alongside this arm"),
    "belief_species": (_const_block(-1, (6,), np.int64), "PAD/not-scored sentinel"),
    "belief_moves": (_const_block(-1, (6, 4), np.int64), "PAD/not-scored sentinel"),
    "known_moves": (_const_block(-1, (6, 4), np.int64), "PAD/not-scored sentinel"),
    "belief_spread": (_const_block(0.0, (6, 5), np.float32), "masked off below"),
    "belief_spread_mask": (_const_block(0.0, (6,), np.float32), "NOT SCORED"),
    "belief_nature": (_const_block(0, (6,), np.int64), "masked off below"),
    "belief_nature_mask": (_const_block(0.0, (6,), np.float32), "NOT SCORED"),
    "belief_ev": (_const_block(0.0, (6, 5), np.float32), "masked off below"),
    "belief_ev_mask": (_const_block(0.0, (6,), np.float32), "NOT SCORED"),
    "hp_type_label": (_const_block(-1, (6,), np.int64), "PAD/not-scored sentinel"),
    "hp_type_mask": (_const_block(0.0, (6,), np.float32), "NOT SCORED"),
    "item_label": (_const_block(-1, (6,), np.int64), "PAD/not-scored sentinel"),
    "item_mask": (_const_block(0.0, (6,), np.float32), "NOT SCORED"),
    "opp_action_kind": (_const_block(KIND_UNKNOWN, (1,), np.int64),
                        "UNKNOWN is the intent labels' own MASK value — 'there was no choice we "
                        "can read', which is exactly true of a row whose delta we never folded"),
    "opp_action_num": (_const_block(0, (1,), np.int64), "unread under KIND_UNKNOWN"),
    "opp_switch_slot": (_const_block(SWITCH_SLOT_NONE, (1,), np.int64),
                        "the labels' own 'cannot be tied to a slot' sentinel"),
    "opp_switch_species": (_const_block(0, (1,), np.int64), "unread under KIND_UNKNOWN"),
}

def unfillable_keys(obs_keys: Sequence[str]) -> List[str]:
    """Obs keys this module cannot honestly fill for an injected row, in declaration order.

    The RETURN of this function is the arm's setup gate: a non-empty list is a refusal, and the
    caller prints :func:`refusal_text`. ``observation`` and ``action_mask`` come from the capture;
    everything else must be in :data:`FILL`.
    """
    bad = []
    for k in obs_keys:
        if k in ("observation", "action_mask") or k in FILL:
            continue
        bad.append(str(k))
    return bad


def refusal_text(keys: Sequence[str]) -> str:
    """One message per unfillable key."""
    lines = []
    for k in keys:
        why = (f"the obs Dict carries `{k}`, which `agents.training.fork_buffer.FILL` does not "
               f"declare a value for. An injected row is not played inside a Gen3Env, so every "
               f"env-computed key needs an explicit decision: a real value or a NOT-SCORED "
               f"sentinel. Add the row (with its reason) or turn the flag off.")
        lines.append(f"  - {k}: {why}")
    return ("🚨 --fork-fraction REFUSED: an injected branch row cannot be built for "
            f"{len(keys)} obs key(s).\n" + "\n".join(lines))
