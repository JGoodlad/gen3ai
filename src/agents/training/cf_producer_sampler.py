"""cf_producer_sampler — the DECLARED, VERSIONED priority sampler, and nothing else.

`cf_producer` is a SAMPLER, not a sweep: it labels the top ``--top-n`` decisions of each record
and skips the rest. Which ones those are is therefore a distribution, and a silent change to it is
a distribution-shift confound for every downstream readout — so the weights, the version string
and the candidate filter live together here, pure NumPy in and floats out, testable without a
model or a battle.

Per candidate decision:

``critic_surprise`` = ``|P(win|s) - realized outcome|`` — highest first. This is the **conviction
region**: the states where the head was sure and the game disagreed. G0 measured that class at
+0.23 predicted-minus-MC, and measured that a single realized outcome cannot say whether the head
was wrong or the dice were (53% of that class was genuinely winning). Tight-MC labels are the only
instrument that separates them, so they are spent here first.

``policy_entropy`` = the masked action distribution's entropy / ``log(n_legal)`` — the decisions
the policy has not made up its mind about, where a ground-truth value is worth most.

``score = 1.00*critic_surprise + 0.35*policy_entropy``. Surprise dominates deliberately; entropy
is a tie-break that keeps the sample off the degenerate "one legal move" states.

A checkpoint with no win-prob head (``--win-prob-mode none``) has no surprise term at all; the
producer says so once and ranks on entropy alone rather than pretending the term was zero.

Extracted verbatim from ``cf_producer.py`` (2026-09-06, the file-size ratchet's third cut of the
1,000-2,000 band) — it was already its own banner section there, and the two constants had to
travel with it because ``ProducerState`` and ``label_row`` both stamp them. ``cf_producer``
re-imports every name below, so ``from agents.training.cf_producer import priority_score`` still
resolves, and the arithmetic is unchanged: ``cf_producer_test.py``'s extraction-parity golden was
captured BEFORE the move and reproduces byte-for-byte after it.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from agents.action.constants import MOVE_START


#: Written into the state file and every label row. Bump it when the weights or the candidate
#: filter below change — a downstream reader comparing two runs' labels needs to know they were
#: drawn from the same distribution.
SAMPLER_VERSION = "cf_producer_priority_v1"

#: The declared priority weights (§ *The sampler*).
PRIORITY_WEIGHTS = {"critic_surprise": 1.00, "policy_entropy": 0.35}

#: Skipped and COUNTED, never silently dropped (`cf_audit` declares the same bound).
#:
#: ⚠️ Its ORIGINAL reason is GONE: "the offline replay driver cannot open turn 1" was a rust
#: `search_driver` defect, fixed 2026-08-23 (`gen3_search_turn1_open_v1`), and both impls now
#: open turn 1. The second reason it carried — "a turn-1 divergence has no prefix to be faithful
#: about" — does not survive inspection either: an EMPTY prefix is trivially faithful, which
#: makes turn 1 the easiest case rather than an excluded one.
#:
#: It is left at 2 deliberately, as a SAMPLER choice rather than a capability limit. Lowering it
#: widens the declared candidate distribution by ~3.35% of move decisions, and this module's own
#: rule is that missing a label is free while silently re-weighting the sampler is not — so that
#: is its own change, with its own before/after on the label mix, not a rider on the driver fix.
MIN_LABELABLE_TURN = 2


def normalized_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy of a masked action distribution, divided by ``log(n_legal)``.

    Normalizing by the support size is what makes the number comparable ACROSS decisions: a
    3-legal-action decision and a 9-legal-action one have different maximum entropies, and an
    un-normalized entropy would rank "many options, all equal" above "two options, genuinely
    50/50" purely on the option count. Returns 0.0 when there is nothing to be undecided about
    (0 or 1 legal actions).
    """
    p = np.asarray(list(probs), dtype=float)
    p = p[p > 0.0]
    if p.size <= 1:
        return 0.0
    h = float(-(p * np.log(p)).sum())
    return max(0.0, min(1.0, h / math.log(p.size)))


def critic_surprise(win_prob: Optional[float], outcome: float) -> float:
    """``|P(win|s) − realized outcome|`` — 0.0 when the checkpoint has no win-prob head.

    ``outcome`` is 1.0 for a won battle, 0.0 for a lost one and 0.5 for a tie (the turn cap), so a
    tie is maximally uninformative about conviction rather than counted as a loss."""
    if win_prob is None or not np.isfinite(win_prob):
        return 0.0
    return float(abs(float(win_prob) - float(outcome)))


def priority_score(surprise: float, entropy: float,
                   weights: "Optional[dict]" = None) -> float:
    w = weights or PRIORITY_WEIGHTS
    return (w["critic_surprise"] * float(surprise)
            + w["policy_entropy"] * float(entropy))


def is_move_round(mask) -> bool:
    """A start-of-turn MOVE round (as opposed to a mid-turn forced switch).

    The divergence a counterfactual anchors at is a move round, so a forced-switch decision — whose
    mask offers only switches — is not labelable by this path. Read off the mask rather than a
    phase string because a ring record has no invocation metadata to carry one."""
    m = np.asarray(mask)
    return bool(m[MOVE_START:].sum() > 0)
