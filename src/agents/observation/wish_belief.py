"""gen3_wish_wired_v1 — the per-side "wish floating" signal.

Wish (gen3) sets a one-turn-delayed, SLOT-keyed heal: used on turn N, it heals whatever mon
occupies the user's slot at the END of turn N+1. poke-env does NOT track the pending wish anywhere
in battle state (the only wish/slotCondition reference is the static `Move.slot_condition` dex
property; there is no Side/slotConditions analog, and the `|-heal|...|[from] move: Wish` resolve falls
through poke-env's heal handler with zero state recorded), so we reconstruct it from OUR event log.

Verified gen3 mechanics (adversarial workflow over `deps/pokemon-showdown` — gen3 INHERITS the **gen4**
Wish condition, NOT the base):

* **Heal amount = floor(RECIPIENT.maxhp / 2)** — the mon in the slot at RESOLUTION, computed at resolve
  time (`data/mods/gen4/moves.ts:1505` `this.heal(target.baseMaxhp / 2, target, target)`). NOT the
  wisher's HP, NOT captured at cast. So the heal as a fraction of the recipient's own max HP is ALWAYS
  ≈0.5 — which is why this feature needs **no max-HP read and cannot suffer HP-stat GIGO**: we encode the
  flat heal fraction `WISH_HEAL_FRACTION = 0.5` when a wish is pending, else 0.0.
* **Duration 2** (`gen4/moves.ts:1500`), decremented each end-of-turn residual → resolves at the END of
  the turn AFTER the cast. So at any decision on turn D, a wish is pending iff this side cast a wish on
  turn D-1 (it resolves at the end of THIS turn). Deterministic single-turn horizon.
* **Slot-keyed (by position)** — persists through the wisher fainting, being phazed out (Roar/Whirlwind),
  voluntarily switching (wish-passing), or self-KOing; the heal lands on whoever is in the slot at
  resolution (in gen3 a fainted mon's replacement enters MID-TURN, before residuals, so it gets healed).
  Therefore the pending signal is per-SIDE, independent of which mon is currently active.
* **Double-Wish FAILS** — a second Wish while one is already pending on the slot is rejected (no
  `onRestart`), the existing one is untouched. A wish cast on turn T is "successful" only if this side did
  NOT successfully cast one on turn T-1 (which would still occupy the slot during T).
* **Full-HP resolve is SILENT** (no `-heal` line) but the slot condition is still consumed — so the
  pending state is cleared purely by timing (turn arithmetic), not by waiting for a heal event.
"""
from __future__ import annotations

from typing import Any, Dict

from agents.battle.battle_event import EventKind, OURS, OPP

# The gen3 (gen4-inherited) Wish heal as a fraction of the RECIPIENT's max HP: floor(maxhp/2)/maxhp ≈ 0.5
# for every real HP value. Constant by construction — no per-mon max-HP read needed (GIGO-proof).
WISH_HEAL_FRACTION = 0.5
_WISH = "wish"


def build_wish_pending(battle: Any) -> Dict[str, bool]:
    """Fold the whole-battle event log into `{OURS: bool, OPP: bool}` — is a Wish PENDING for each side
    at the current decision (i.e. it will heal the slot mon at the end of THIS turn). A wish is pending
    for a side iff that side SUCCESSFULLY cast Wish on the previous turn (`current_turn - 1`); a cast is
    successful only if the side did not also have a successful cast the turn before that (double-Wish on
    an occupied slot fails). One linear pass over `battle.events`; keyed by SIDE (slot), so it is correct
    through faints / phazes / switches (the heal follows the slot, not the wisher)."""
    events = getattr(battle, "events", None)
    if not events:
        return {OURS: False, OPP: False}
    cur_turn = int(getattr(battle, "turn", 0) or 0)
    # per side: the turn of the most recent SUCCESSFUL wish cast (for the double-Wish guard) + the SET
    # of all successful cast turns, so the pending check is a membership test (a later cast can't mask
    # an earlier one's window). The double-Wish `last_ok` guard assumes turn-ordered events — which the
    # event log always is (`_record` appends in parse order; `event.turn` only rises on a `|turn|N`).
    last_ok: Dict[str, int] = {OURS: -10, OPP: -10}
    ok_turns: Dict[str, set] = {OURS: set(), OPP: set()}
    for e in events:
        if e.kind is EventKind.MOVE and e.move_id == _WISH and e.side in (OURS, OPP):
            t = int(e.turn)
            # double-Wish guard: a wish at turn t FAILS if a successful wish from this side was cast at
            # t-1 (it still occupies the slot during t). Events arrive in turn order, so the most-recent
            # prior successful cast is last_ok.
            if last_ok[e.side] != t - 1:
                last_ok[e.side] = t
                ok_turns[e.side].add(t)
    # pending now iff this side successfully cast a wish LAST turn (resolves end of this turn)
    target = cur_turn - 1
    return {OURS: target in ok_turns[OURS], OPP: target in ok_turns[OPP]}


def wish_floating_value(pending: bool) -> float:
    """The per-side reserved-dim value: the gen3 wish heal fraction (≈0.5 of the slot mon's max HP)
    when a wish is pending and will resolve at the end of this turn, else 0.0."""
    return WISH_HEAL_FRACTION if pending else 0.0
