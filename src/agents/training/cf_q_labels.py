"""cf_q_labels — the PER-ACTION half of the counterfactual label factory (``gen3_cf_q_labels_v1``).

`cf_producer` labels a decision with ONE number: the tight-MC win probability of the action the
recorded policy actually took. The Q head shipped at v107 (`agents/model/q_winprob_head.py`) reads
ELEVEN, one per legal action — and it landed as a trained consumer of a stream nobody wrote. This
module is the arithmetic of the supply side: which sibling actions get rolled out, on WHICH DICE,
and how the results become the ``q_labels`` wire objects `cf_label_buffer` already parses.

Everything here is PURE — no model, no bridge, no record. The producer owns the rollouts; this
owns the decisions that make a sweep of them comparable to each other, so they can be tested
without a simulator and cannot drift into the one file that is expensive to exercise.

THE PAIRING IS THE POINT (common random numbers)
------------------------------------------------
A sweep's whole job is a RANKING: "is Rock Slide better than Earthquake here?". Rolled out on
independent dice, the difference between two sibling arms carries the variance of BOTH — at the
producer's default R=8 the per-arm standard error is ~0.18, so a 0.1 gap between two actions is
invisible. Rolled out on the SAME dice, the shared dice cancel in the difference and what survives
is the effect of the action, which is the quantity the head is being asked to learn.

So every sibling action at a decision draws :func:`q_arm_seeds` — a list that depends on the
BATTLE and the DECISION and the producer's ``--seed``, and **not on the action**. That is not a
convention to be remembered; :func:`assert_paired_dice` is called at the seam with the seeds every
arm actually received, and a producer that ever derives them per-action fails loudly rather than
shipping a sweep whose ranking is noise. (`gen3_cf_q_labels_v1`.)

⚠️ **The pairing is over the SIM DICE, and that is not the only randomness in a rollout.** Both
sides are played by a stochastic snapshot at temperature 1.0 (§ *THE ECOLOGY DECISION* in
`cf_producer`), and `torch.distributions.Categorical.sample` draws from torch's global RNG, which
no seed here reaches. Two sibling arms therefore share their dice exactly and their POLICY draws
not at all. That residual is stated rather than hidden: it is a variance term the pairing does not
remove, it biases nothing (both arms draw from the same policy), and closing it is not possible
by seeding alone — the arms diverge immediately, so after the first live decision they are not
even drawing the same NUMBER of samples.

The recorded action is FREE, and its label is an IDENTITY
---------------------------------------------------------
The per-state ``label`` the producer already writes IS the recorded action's counterfactual label,
computed from :func:`q_arm_seeds` with the same salt. So the recorded action's arm is not rolled
out twice: :func:`recorded_arm_is_reusable` says when the existing result can be lifted verbatim,
and `cf_producer` then ships ``q_labels[recorded] == label`` **exactly**. That identity is pinned
by a test rather than asserted in prose — it is also the cheapest possible check that the sweep is
anchored to the same measurement the row's own label came from.

The selection rule (``cf_q_sweep_v1``) — declared, versioned, written into every row
-------------------------------------------------------------------------------------
Cost multiplies by the number of arms, so a budget knob is unavoidable; WHICH actions it drops is
a distribution decision and therefore gets a version string, for the same reason
`cf_producer`'s ``SAMPLER_VERSION`` does.

* The RECORDED action is always first (it is free, and it anchors the sweep — see above).
* The remaining legal actions are ordered by a **deterministic shuffle** keyed to the decision,
  and the cap takes a prefix of that.

The shuffle is the load-bearing half. The two obvious orders are both wrong here:

* by DESCENDING policy probability — this is exactly the on-policy starvation the Q head exists to
  escape. Probe L measured the policy sampling its own better-ranked alternative at a median
  p=0.002, so a probability-ordered cap would spend the budget on the actions already covered and
  leave the head untrained precisely where it will be consulted.
* by ACTION INDEX — the action space is ``[switch x6, move x4, struggle]``, so a prefix of it is a
  systematic preference for SWITCHES. A capped sweep would then teach the head about switching and
  almost nothing about attacking.

A decision-keyed shuffle is unbiased across the action space, reproducible across processes (the
key is the battle tag, the decision index and ``--seed`` — the same three inputs the dice take),
and costs nothing. ``max_actions=0`` (the default) sweeps every legal action, where the question
does not arise at all.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.action.constants import ACTION_SPACE_SIZE

#: Written into every row that carries a ``q_labels`` block. Bump it when the SELECTION rule or the
#: dice derivation changes — a reader comparing two runs' per-action labels needs to know they were
#: drawn the same way, exactly as ``SAMPLER_VERSION`` promises for the per-state stream.
Q_SWEEP_VERSION = "cf_q_sweep_v1"


def q_arm_salt(*, tag: str, decision_index: int, producer_seed: int) -> str:
    """The dice salt for one decision's arms — **no action term, deliberately**.

    This function is the single definition of "the same dice". :func:`q_arm_seeds` and
    `cf_producer._rollout` both derive from it, so the recorded action's already-paid-for arm and
    every sibling arm are drawing from one list rather than from two that happen to agree.
    """
    return f"{tag}:{int(decision_index)}:cfp{int(producer_seed)}"


def q_arm_seeds(*, tag: str, decision_index: int, producer_seed: int, n: int) -> List[str]:
    """``n`` post-divergence dice seeds for one decision, IDENTICAL for every sibling action.

    Delegates to :func:`main.prober.falsifier.fresh_seeds`, which derives seed ``i`` from
    ``sha256(f"{salt}:{i}")``. Two consequences are used elsewhere and are worth naming: the list
    for ``n=4`` is a strict PREFIX of the list for ``n=8`` (so a smaller ``--q-rollouts`` is a
    sub-sample of the same dice, not a different experiment), and the whole list is reproducible
    from the row's own provenance fields.
    """
    from main.prober.falsifier import fresh_seeds
    return fresh_seeds(int(n), salt=q_arm_salt(
        tag=tag, decision_index=decision_index, producer_seed=producer_seed))


def assert_paired_dice(per_action_seeds: "Dict[int, Sequence[str]]") -> None:
    """Raise unless every sibling action was rolled out on the SAME dice. Called at the seam.

    ``per_action_seeds`` maps an action index to the seed list that action's arms actually
    received — collected from the rollout calls themselves, never re-derived, because a check that
    recomputes what it is checking proves only that one function is deterministic.

    An empty or single-action sweep is trivially paired and passes.
    """
    distinct = {tuple(v) for v in per_action_seeds.values()}
    if len(distinct) <= 1:
        return
    detail = ", ".join(
        f"{a}: {list(s)[:2]}{'…' if len(list(s)) > 2 else ''}"
        for a, s in sorted(per_action_seeds.items()))
    raise RuntimeError(
        "cf q-sweep DICE ARE NOT PAIRED: sibling actions at one decision were rolled out on "
        f"{len(distinct)} different seed lists ({detail}). The sweep's product is a RANKING, and "
        "an unpaired ranking at R=8 is noise — the per-arm standard error (~0.18) is larger than "
        "the differences the Q head is being taught. Every arm must take `q_arm_seeds(...)`, "
        "whose salt has no action term.")


def select_q_actions(mask, recorded_action: int, *, max_actions: int = 0,
                     tag: str = "", decision_index: int = 0,
                     producer_seed: int = 0) -> "List[int]":
    """The actions one decision's sweep will roll out — ``cf_q_sweep_v1`` (see the module docstring).

    The recorded action leads (free, and the sweep's anchor); the rest follow in a deterministic
    decision-keyed shuffle, truncated to ``max_actions``. ``max_actions <= 0`` means every legal
    action, which is the default and the only setting with no selection bias to declare.

    ``recorded_action`` is included even if the mask somehow does not offer it — that cannot happen
    on a real record (the index was RECOVERED by inverting the committed choice against this very
    mask), and preferring the recorded action over a defensive filter keeps the ``q_labels[recorded]
    == label`` identity total rather than conditional.
    """
    legal = [int(a) for a in np.flatnonzero(np.asarray(mask))
             if 0 <= int(a) < ACTION_SPACE_SIZE]
    rec = int(recorded_action)
    others = [a for a in legal if a != rec]
    if others:
        # A decision-keyed permutation: same three inputs as the dice, so a row's sweep is
        # reproducible from its own provenance without the producer's process being alive.
        key = hashlib.sha256(
            q_arm_salt(tag=tag, decision_index=decision_index,
                       producer_seed=producer_seed).encode()).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(key, "big"))
        others = [others[int(i)] for i in rng.permutation(len(others))]
    ordered = ([rec] if 0 <= rec < ACTION_SPACE_SIZE else []) + others
    if max_actions and max_actions > 0:
        ordered = ordered[:int(max_actions)]
    return ordered


def recorded_arm_is_reusable(*, q_rollouts: int, rollouts: int) -> bool:
    """True when the per-state ``label`` IS the recorded action's q-label and needs no re-roll.

    The per-state label is ``R`` arms at :func:`q_arm_seeds`, substituting the recorded choice. A
    sweep arm for the recorded action at ``q_rollouts == rollouts`` is byte-for-byte that same
    experiment, so re-running it would spend one arm's worth of rollouts to reproduce a number
    already in hand.

    ⚠️ It is deliberately NOT reused when the counts differ, even though the shorter seed list is a
    strict prefix of the longer one. The per-state label would then be a mean over ``R`` arms while
    every sibling is a mean over ``q_rollouts`` — an anchor with more evidence than the actions it
    anchors, which makes ``q_labels[recorded] - q_labels[other]`` a comparison between two
    different sample sizes. One re-rolled arm is cheaper than that footnote.
    """
    return int(q_rollouts) == int(rollouts)


def q_label_entry(action: int, *, wins: float, n: int) -> dict:
    """One ``q_labels`` wire object: ``{"action": int, "label": float in [0,1], "n_rollouts": int}``.

    A LIST OF OBJECTS is the schema (`cf_label_buffer`'s docstring states why): three same-length
    parallel arrays can be written in the wrong order by a producer and read as valid by a
    consumer, and this tree treats an order-mismatch bug as drop-everything. The action index each
    entry names is the SAME index the policy's logits and the action mask use, so the consumer's
    scatter into column ``a`` cannot land anywhere else.

    ``wins`` is fractional for the same reason the per-state label's is: a draw — including the
    250-turn cap, which is a draw by forfeit-ordering artifact rather than by play — scores 0.5.
    """
    n = int(n)
    label = (float(wins) / float(n)) if n else 0.0
    return {"action": int(action), "label": round(float(label), 6), "n_rollouts": n}


def q_labels_block(results: "Sequence[Tuple[int, float, int]]") -> "List[dict]":
    """``[(action, wins, n), ...]`` → the wire list, dropping arms where every rollout failed.

    An action whose arms all failed carries NO evidence, so it is omitted rather than shipped at
    ``n_rollouts: 0``: the consumer's mask is built from PRESENCE, and a zero-evidence entry would
    mask ON a cell whose target is the ``label = 0.0`` fallback — a confident loss for an action
    nobody measured. Omission leaves that cell unsupervised, which is what it is.
    """
    return [q_label_entry(a, wins=w, n=n) for (a, w, n) in results if int(n) > 0]


def q_provenance(*, actions: "Sequence[int]", rollouts: int, capped: int,
                 reused_recorded: bool, max_actions: int,
                 wall_seconds: Optional[float] = None) -> dict:
    """The additive, buffer-ignored audit block a swept row carries beside ``q_labels``.

    The buffer reads a fixed key set, so none of this is a schema change — it is the difference
    between a per-action label you can stratify a year later and a bag of numbers. ``arms`` is the
    measured **~n_legal multiplier** for this row, which is the number an operator sizing a
    producer needs and the one thing a reader cannot re-derive from ``q_labels`` alone once
    zero-evidence arms have been dropped.
    """
    block = {
        "version": Q_SWEEP_VERSION,
        "arms": len(list(actions)),
        "rollouts_per_arm": int(rollouts),
        "n_capped": int(capped),
        "recorded_arm_reused": bool(reused_recorded),
        "max_actions": int(max_actions),
    }
    if wall_seconds is not None:
        block["wall_seconds"] = round(float(wall_seconds), 3)
    return block
