"""COMMON RANDOM NUMBERS for the fork arm (`--fork-crn`, `gen3_fork_v1`).

**THE FINDING THIS MODULE EXISTS FOR.** `paired_refit_discrimination_2026-09-14` banked a concrete,
testable account of the `cf_q_labels` null: *"`cf_q_labels` pairs only the SIM DICE and leaves both
sides sampling at temperature 1.0 (its own recorded caveat), while these forks are greedy on both
sides — which is the only reason 104 determinism re-runs came back identical. A label factory that
pairs the dice but not the policy draws may be teaching the head noise."*

The fork arm cannot take the offline dataset's way out. Its branches must be played in the ECOLOGY
the training actor plays in — temperature 1.0 on both sides — because the transitions go into the
PPO buffer, and a greedy continuation would be a different policy's trajectory. So the draws are
PAIRED instead of removed.

HOW, and why it is one line of existing machinery
-------------------------------------------------
`RLPlayer` already carries `gen3_policy_sample_rng_v1`: pass ``policy_seed`` and its
``Categorical`` sample is drawn from a PER-INSTANCE ``torch.Generator`` instead of torch's
process-wide default. Give every branch of one fork players seeded with the SAME value and the
k-th live decision of branch A and the k-th of branch B consume the SAME uniform. The branches
then differ in exactly one thing: the action at the fork.

Three properties make that true rather than approximately true, and each is closed rather than
noted:

* **the prefix consumes nothing.** `counterfactual.install_scripted_prefix` returns a passthrough
  ``BattleOrder`` for every scripted decision and never reaches ``_predict_best_action``, so the
  generator is untouched until the handoff. Two branches therefore start their streams aligned at
  the fork, not at turn 1.
* **one uniform per decision, whatever the state.** ``torch.multinomial(cat.probs, 1, True, gen)``
  draws exactly one sample from an 11-way categorical every call, so the streams cannot slip
  relative to each other by a state-dependent amount.
* **the two SIDES get different seeds.** Both sides of one branch sharing a stream would couple
  the trainee's coin to the opponent's — a correlation that is not in the training ecology and
  would be a second confound smuggled in under a fix for the first.

``dice`` (the comparator) passes no seed at all, which is byte-identical to `cf_producer`'s own
rollouts: the shared default generator, the `cf_q_labels` regime.

**The pairing is VERIFIED, not assumed** — `fork_crn_sim_test` replays two branches of a real
bridge battle with IDENTICAL substitute actions and asserts the two protocol streams are
byte-identical, which can only hold if both the dice and every policy draw were shared.
"""

from __future__ import annotations

import hashlib

#: Torch's generator seeds are 64-bit; keep the derived value inside it with room to spare.
_SEED_MASK = (1 << 62) - 1


def branch_policy_seed(salt: str, side: str) -> int:
    """The sampling-stream seed for ``side`` of EVERY branch of the fork identified by ``salt``.

    Depends on the fork and the side and **not on the branch**, which is the whole pairing. Derived
    by sha256 rather than by arithmetic on the salt so two adjacent forks cannot end up with
    adjacent (and therefore correlated) Mersenne streams.
    """
    h = hashlib.sha256(f"gen3_fork_v1|{salt}|{side}".encode()).hexdigest()
    return int(h[:16], 16) & _SEED_MASK


def player_seeds(salt: str, crn: str) -> "tuple[int | None, int | None]":
    """``(trainee_seed, opponent_seed)`` for one branch, or ``(None, None)`` under ``dice``.

    ``None`` means "use torch's shared default generator", which is `RLPlayer`'s documented
    unseeded behaviour and exactly what `cf_producer`'s rollouts do today — so the ``dice`` setting
    is the pre-existing regime and not a new one.
    """
    if str(crn) != "dice_and_draws":
        return None, None
    return branch_policy_seed(salt, "trainee"), branch_policy_seed(salt, "opponent")
