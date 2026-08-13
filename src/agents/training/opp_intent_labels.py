"""Privileged labels for `α`/`β` — what the opponent ACTUALLY did (design_opponent_intent.md §7 step 2).

Built from the `TurnDelta` the event log already folds, which is the authoritative per-decision
record (poke-env's `last_move` is a stale-field footgun the event-sourced layer exists to replace —
use `opp_resolved_move_id`, never `opp_move_id`).

## The distinction this module exists to make: DECISION vs CONSEQUENCE

`α`/`β` model what the opponent CHOSE. Several things in a `TurnDelta` look like a switch and are
not choices at all, and supervising on them would teach the heads to predict our own Roar:

| situation | delta shape | label |
|---|---|---|
| they clicked a move | `opp_resolved_move_id`, no switch | **MOVE** |
| they voluntarily pivoted | `opp_switch_to`, no faint, no move | **SWITCH** — the only case `β` learns from |
| we PHAZED them (Roar/Whirlwind) | `opp_switch_to` **and** a move resolved | **UNKNOWN** — our choice, not theirs |
| their mon fainted and was replaced | `opp_switch_to` **and** `opp_fainted` | **UNKNOWN** — forced by the rules |
| the window closes on a forced switch | `phase_is_forced_switch` | **UNKNOWN** — there was no choice to make |

Everything not a genuine choice is MASKED, never guessed. A mask costs supervision on that row; a
wrong label costs the head's meaning, and `α`'s accuracy stops being readable as "how well do we
predict the opponent".

## Timing — the label is for the PREVIOUS decision

The delta available when we build the observation for decision *t* describes the window that just
closed, i.e. the opponent's action at decision *t-1*. So the key emitted at *t* is a label for
*t-1*, and the consumer must shift by one within the rollout (and drop the pair across an episode
boundary). That shift lives in the PPO loss, not here — this module only reports what the delta
says, and `LABEL_IS_FOR_PREVIOUS_DECISION` is the flag that keeps the contract visible at both ends.
"""
from __future__ import annotations

from typing import Optional, Tuple

# Kinds, matching `agents.model.opp_intent.match_seats_to_move_num`'s `kind` argument.
KIND_MOVE, KIND_SWITCH, KIND_UNKNOWN = 0, 1, 2

# Consumer-facing contract marker: the emitted label describes decision t-1, not t.
LABEL_IS_FOR_PREVIOUS_DECISION = True

# `β`'s target when the switch-in cannot be tied to a team slot.
SWITCH_SLOT_NONE = -100


def build_opp_intent_label(
    delta,
    move_num_of,
    opp_slot_of_species,
    species_num_of=None,
) -> Tuple[int, int, int, int]:
    """`(kind, move_num, switch_slot, switch_species)` for the decision the given delta closes.

    TWO switch targets, because a REVEALED mon and a BELIEVED mon are addressed differently:

      * `switch_slot` — the encoder slot, valid ONLY when the switch-in was already REVEALED at the
        decision. Exact: that slot holds that mon's real encoding.
      * `switch_species` — the species NUM, always emitted for a switch. This is the
        CONTENT-ADDRESSED key for a switch-in that was still HIDDEN, resolved at loss time against
        the model's OWN believed-slot species posterior.

    Why the hidden case cannot use a slot index: the believed slots are ANONYMOUS DETR-style
    queries, and the species loss matches them to the true hidden mons by HUNGARIAN assignment —
    which discards the label's ordering entirely. `assign_hidden_to_slots` canonicalises by
    Pokedex number purely to make the label deterministic, so "the 4th-lowest-num hidden mon" and
    "the slot the model learned to put that mon in" are unrelated. Supervising `β` on the
    canonical index would train it toward a slot whose learned content is a DIFFERENT mon —
    label noise dressed as signal. Worse, it is unlearnable in principle: nothing in the forward
    pass tells the model which anonymous slot the canonicalisation picked.

    `move_num_of(move_id) -> int|None` and `opp_slot_of_species(species) -> int|None` are injected
    so this stays pure and unit-testable; the env supplies the real lookups.

    `move_num` is 0 unless `kind == KIND_MOVE`. `switch_slot` is `SWITCH_SLOT_NONE` unless the
    switch-in was REVEALED; `switch_species` is 0 unless `kind == KIND_SWITCH`.
    """
    if delta is None:
        return KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0

    # A forced-switch window contains no opponent decision at all.
    if getattr(delta, "phase_is_forced_switch", False):
        return KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0

    move_id = delta.opp_resolved_move_id
    switch_to = delta.opp_switch_to

    if switch_to:
        # A switch that co-occurs with a resolved move is a PHAZE (our Roar/Whirlwind moved them),
        # and one that co-occurs with a faint is a forced replacement. Neither is their choice.
        if move_id or getattr(delta, "opp_fainted", False):
            return KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0
        slot = opp_slot_of_species(switch_to)
        sp = None if species_num_of is None else species_num_of(switch_to)
        return (KIND_SWITCH, 0,
                (SWITCH_SLOT_NONE if slot is None else int(slot)),
                (0 if sp is None else int(sp)))

    if move_id:
        num = move_num_of(move_id)
        if num is None:
            return KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0   # unnameable ⇒ masked, never guessed
        return KIND_MOVE, int(num), SWITCH_SLOT_NONE, 0

    return KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0


def align_labels_to_predictions(labels, episode_starts, fill):
    """Shift a `[n_steps, n_envs]` label block back one row so it lines up with the PREDICTIONS.

    The env emits, at buffer row *i*, a label describing decision *i-1* (see the timing note above).
    The head predicts, at row *i*, the opponent's action at *i*. So the label belonging to row *i*
    is stored at row *i+1* of the SAME env column, and the aligned block is `labels[i+1]`.

    Two rows must be dropped, and dropping them is not optional:

      * the LAST row has no successor in this rollout — its label lives in the next one;
      * any row whose successor STARTS AN EPISODE, because that successor's label describes the
        first decision of a NEW battle, not the last decision of the old one. Keeping it would
        splice one game's outcome onto another's board — a silent cross-episode GIGO bug that no
        shape check and no loss curve would reveal.

    Pure and array-library-agnostic (numpy or torch), so it is testable without a rollout.
    `fill` is the masked-out value (`INTENT_IGNORE` / `SWITCH_SLOT_NONE`).
    """
    out = labels.copy() if hasattr(labels, "copy") else labels.clone()
    out[:-1] = labels[1:]
    out[-1] = fill
    # `episode_starts[i+1] == 1` ⇒ row i+1 opens a new episode ⇒ row i's pair is invalid.
    starts_next = episode_starts[1:]
    if hasattr(out, "shape"):
        sel = starts_next > 0.5
        out[:-1][sel] = fill
    return out


def zero_opp_intent_label() -> Tuple[int, int, int, int]:
    """The pre-battle / no-delta value — fully masked, so a reset never trains on a fabricated row."""
    return KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0
