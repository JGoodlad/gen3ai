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
| they clicked a move and THEN switched in this window (fainted and replaced, phazed by us, Baton Pass) | `opp_move_before_switch` | **MOVE** — the move was the choice, the switch its consequence (`gen3_intent_label_move_then_switch_v1`) |
| they voluntarily pivoted | `opp_switch_to`, no faint, no move | **SWITCH** — the only case `β` learns from |
| a switch and a move with the switch FIRST (two choices in one window), or our phaze with no move of theirs | `opp_switch_to` **and** a move, not `opp_move_before_switch` | **UNKNOWN** |
| we DRAGGED them after their own switch, or while they were asleep / paralysed / frozen / loafing | `opp_dragged` | **UNKNOWN** — the dragged mon is one they never chose |
| their mon fainted and was replaced | `opp_switch_to` **and** `opp_fainted` | **UNKNOWN** — forced by the rules |
| the replacement lands in the NEXT window (the faint lay in the previous one) | `opp_switch_is_replacement` | **UNKNOWN** — the same forced switch |
| the window closes on OUR forced switch | `phase_is_forced_switch` | **labelled as above** — it holds their genuine choice of the turn before, usually the move that KOed us (the old UNKNOWN mask dropped every KO turn's attack: 2026-10-26 GIGO audit D-1, `gen3_intent_label_move_then_switch_v1`) |
| they called a move (Sleep Talk → Rest, Mirror Move, Metronome, Assist, Nature Power, Magic Coat, Snatch) | `opp_called_via` | **MOVE the CALLER** — they clicked Sleep Talk, not Rest |
| our Encore landed before they moved, so the encored move ran in place of their pick | `opp_choice_overridden` | **UNKNOWN** — the executed move is not their choice |

Everything not a genuine choice is MASKED, never guessed (the drag, straddling-replacement, caller and
Encore rows are `gen3_intent_label_semantics_fixes_v1`, found by the Rust core M3 loss catalogue:
`designs/research_state/measurements/rust_core_m3_2026-09-24/` §5.3). A mask costs supervision on that row; a
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

from typing import Tuple

# Kinds, matching `agents.model.opp_intent.match_seats_to_move_num`'s `kind` argument.
KIND_MOVE, KIND_SWITCH, KIND_UNKNOWN = 0, 1, 2

# Consumer-facing contract marker: the emitted label describes decision t-1, not t.
LABEL_IS_FOR_PREVIOUS_DECISION = True

# `β`'s target when the switch-in cannot be tied to a team slot.
SWITCH_SLOT_NONE = -100


class IntentLabelError(RuntimeError):
    """An intent-label lookup that cannot be made (F-X5-3) — never a silent UNKNOWN / species 0."""


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

    A move or switch-in species the injected lookup cannot name RAISES `IntentLabelError` (F-X5-3,
    `gen3_label_lookup_guard_v1`): it was a silent UNKNOWN label / species 0. The Rust writer
    (`rust_env/src/labels/intent.rs`) is the production one and raises the same way.
    """
    if delta is None:
        return KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0

    # gen3_intent_label_move_then_switch_v1: a window that closes on OUR forced switch is NOT masked — it holds their
    # choice of the turn before (the window after it holds no opponent action and reads UNKNOWN by itself).
    # Our Encore overrode the move they picked: what executed is not what they chose.
    if getattr(delta, "opp_choice_overridden", False):
        return KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0

    # A CALLED move (Sleep Talk → Rest) is the caller's doing: the choice was the CALLER.
    move_id = getattr(delta, "opp_called_via", None) or delta.opp_resolved_move_id
    switch_to = delta.opp_switch_to
    if move_id and getattr(delta, "opp_move_before_switch", False):
        switch_to = None      # the move came first: it is the choice, the switch-in its consequence

    if switch_to:
        # A switch that co-occurs with a resolved move is a PHAZE (our Roar/Whirlwind moved them),
        # one that co-occurs with a faint is a forced replacement, a DRAGGED entrant is one they
        # never chose (even when they had switched first, or were asleep), and a replacement for a
        # faint in the PREVIOUS window is the same forced switch. None of them is their choice.
        if (move_id or getattr(delta, "opp_fainted", False)
                or getattr(delta, "opp_dragged", False)
                or getattr(delta, "opp_switch_is_replacement", False)):
            return KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0
        slot = opp_slot_of_species(switch_to)
        sp = 0
        if species_num_of is not None:
            sp = species_num_of(switch_to)
            if sp is None:
                raise IntentLabelError(f"switch-in species {switch_to!r} has no num (F-X5-3)")
        return (KIND_SWITCH, 0,
                (SWITCH_SLOT_NONE if slot is None else int(slot)),
                int(sp))

    if move_id:
        num = move_num_of(move_id)
        if num is None:
            raise IntentLabelError(f"move {move_id!r} has no num (F-X5-3: it was a silent UNKNOWN label)")
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
    `fill` is the masked-out value (`INTENT_IGNORE` / `SWITCH_SLOT_NONE`), a scalar, or an ARRAY of
    `labels`' shape whose own row is written at every dropped row — the form a per-EPISODE key takes
    (`opp_class`: a dropped row keeps its own class; a scalar fill would be a real class code,
    `OPP_CLASS_BOT` = 0, stamped on every game's last row — `gigo_audit_datapath_2026-10-26`).
    """
    out = labels.copy() if hasattr(labels, "copy") else labels.clone()
    out[:-1] = labels[1:]
    per_row = hasattr(fill, "shape") and tuple(fill.shape) == tuple(labels.shape)
    out[-1] = fill[-1] if per_row else fill
    # `episode_starts[i+1] == 1` ⇒ row i+1 opens a new episode ⇒ row i's pair is invalid.
    starts_next = episode_starts[1:]
    if hasattr(out, "shape"):
        sel = starts_next > 0.5
        out[:-1][sel] = fill[:-1][sel] if per_row else fill
    return out


class EpisodeKeyNotConstant(ValueError):
    """A per-EPISODE key (``opp_class``) changed between two rows of one game in a buffer column: the
    rows of a column are not one game's consecutive decisions, so every one-ahead label shift is wrong."""


def check_constant_within_episodes(values, episode_starts, key: str) -> None:
    """FAIL-CLOSED producer/consumer check of the one-ahead shift's premise: inside every ``[n_steps,
    n_envs]`` column, a per-episode key is the same on row i and row i + 1 unless row i + 1 starts an
    episode. Raises `EpisodeKeyNotConstant` naming the first offending cell."""
    import numpy as _np

    v = _np.asarray(values).reshape(_np.asarray(values).shape[0], _np.asarray(values).shape[1], -1)
    s = _np.asarray(episode_starts).reshape(v.shape[0], v.shape[1])
    same_game = s[1:] <= 0.5
    differs = (v[1:] != v[:-1]).any(axis=-1) & same_game
    if bool(differs.any()):
        i, j = (int(x) for x in _np.argwhere(differs)[0])
        raise EpisodeKeyNotConstant(
            f"{key}: buffer column {j} changes from {v[i, j].tolist()} to {v[i + 1, j].tolist()} between rows {i} "
            f"and {i + 1} with no episode start — the column is not one game's consecutive rows "
            f"({int(differs.sum())} such pairs)")


def zero_opp_intent_label() -> Tuple[int, int, int, int]:
    """The pre-battle / no-delta value — fully masked, so a reset never trains on a fabricated row."""
    return KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0
