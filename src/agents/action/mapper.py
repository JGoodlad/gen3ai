"""Action ↔ order mapping for the Gen 3 OU 11-action discrete space.

The decode step is a **pure function** over a captured
:class:`~agents.battle.live_view.LegalActions` snapshot — it touches no poke-env objects
and is fully testable with a ``LegalActions`` stub:

    ``Gen3ActionMapper.action_to_choice(action_idx, legal) -> Choice``

The only poke-env touch — turning a :class:`~agents.action.choice.Choice` into the
``BattleOrder`` the client sends — lives in :mod:`agents.action.serialize`. The env /
player compose the two: validate the decision is current, decode to a ``Choice``, then
serialize. There is no longer any ``battle._gen3_decision_context`` stash: the immutable
``LegalActions`` captured at observation time IS the per-decision snapshot, carried on the
:class:`~agents.training.battle_snapshot.BattleContext`.

Action space:
  0–5:  Switches (team slots 0–5)
  6–9:  Moves (request slots 0–3)
  10:   Struggle (single-sourced via ``legal.struggle`` — never a move slot)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from agents.action.choice import Choice
from agents.action.constants import SWITCH_END, MOVE_START, MOVE_END, STRUGGLE
from agents.action.serialize import choice_to_order, order_to_action
from agents.battle.live_view import LegalActions
from utils import race_trace  # debug ring buffer (GEN3_RACE_TRACE); no-op when off

if TYPE_CHECKING:
    from poke_env.player.battle_order import BattleOrder


class StaleDecisionError(RuntimeError):
    """The latched decision context no longer matches the live battle — POKE_LOOP parsed a
    background message (typically an in-flight turn-resolution during the model forward) and the
    request/turn shifted under us between embed_battle() and action_to_order(). Acting on the
    stale snapshot would send the wrong/obsolete order. Two handlers, by who owns the decision:

    - **Self-play OPPONENT** (``RLPlayer.choose_move``) catches it and RE-DECIDES on the
      now-current request. The opponent's decision is internal to ``SingleAgentWrapper.step`` —
      SB3 never sees it — so it must always return a valid order (SB3 has no failed-step path).
    - **TRAINEE** (``gen3_env``) lets it propagate and crash (crash-over-corruption). Its action
      comes from SB3 and can't be re-run inside the step, so acting on a stale snapshot would
      corrupt its ``(obs, action) → (reward, next_obs)`` transition.

    Raised by ``assert_decision_current`` / ``action_to_order``."""


class Gen3ActionMapper:
    """Maps the 11-action discrete index to/from poke-env battle orders, through the
    server-authoritative :class:`LegalActions` snapshot."""

    # ------------------------------------------------------------------ #
    # Decode (PURE) — action index → Choice, using only the legal snapshot
    # ------------------------------------------------------------------ #
    @staticmethod
    def action_to_choice(action_idx: int, legal: LegalActions) -> Choice:
        """Resolve a discrete action index against ``legal`` into a tagged
        :class:`Choice`. Pure: no battle, no poke-env objects. Raises ``ValueError``
        when the index is illegal for this decision (the mask should already prevent
        that — this is the defensive backstop)."""
        # 0–5: switch to the bench mon at that team slot.
        if action_idx < SWITCH_END:
            for sw in legal.switches:
                if sw.slot == action_idx:
                    return Choice.switch(sw.species, sw.slot)
            raise ValueError(
                f"Switch action {action_idx} is not a legal switch this decision "
                f"(legal switch slots: {list(legal.switch_slots)})."
            )

        # 6–9: use the move in request slot (action - 6).
        if action_idx < MOVE_END:
            slot = action_idx - MOVE_START
            if slot < len(legal.move_slots):
                m = legal.move_slots[slot]
                if m.disabled:
                    raise ValueError(
                        f"Move slot {slot} ('{m.id}') is disabled this decision."
                    )
                return Choice.move(m.id, slot)
            raise ValueError(
                f"Move slot {slot} out of range "
                f"({len(legal.move_slots)} legal move slots)."
            )

        # 10: struggle — single-sourced via the legal flag, never a move slot.
        if action_idx == STRUGGLE:
            if not legal.struggle:
                raise ValueError(
                    "Struggle (action 10) pressed but the server is not forcing "
                    "Struggle this decision."
                )
            return Choice.struggle()

        raise ValueError(f"Unhandled action index: {action_idx}")

    # ------------------------------------------------------------------ #
    # Convenience: action index → poke-env order (decode + serialize)
    # ------------------------------------------------------------------ #
    @staticmethod
    def action_to_order(
        action_idx: int,
        battle,
        legal: Optional[LegalActions] = None,
        mask=None,
    ) -> "BattleOrder":
        """Decode ``action_idx`` to a :class:`Choice` and serialize it to a poke-env
        order. The env / player pass the captured ``legal`` snapshot from the decision
        context (the protected path); back-to-back callers (fuzz / opponent) may omit it
        and a fresh snapshot is taken from ``battle``. When ``mask`` is given the action
        is checked against it first (crash-over-corruption).

        End-to-end **conformance** is enforced fail-loud across the whole send chain —
        offered (mask) → picked (action) → sent (order):
          1. ``mask[action] != 0`` — the model only picks what the mask OFFERED;
          2. ``action_to_choice`` raises if the action doesn't conform to ``legal``;
          3. ``choice_to_order`` raises if the choice can't resolve to a real move/switch;
          4. the serialized order is **round-tripped** back to an action index and must
             equal what the model picked — proving the move/switch we're about to send
             Showdown is the very one the action selected (catches a serialization drift,
             e.g. a duplicate-id or Hidden-Power mis-resolution)."""
        snapshot_provided = legal is not None
        if legal is None:
            legal = LegalActions.from_battle(battle)
        if mask is not None and mask[action_idx] == 0:
            raise ValueError(
                f"Illegal action {action_idx}: the offered mask does not permit it. "
                f"Mask: {mask}"
            )
        # Decode is PURE over the captured snapshot. A failure here means the action is
        # illegal in the snapshot ITSELF (a mask/legal inconsistency = real corruption),
        # so it stays a fatal ValueError — never reclassified as staleness.
        choice = Gen3ActionMapper.action_to_choice(action_idx, legal)
        # Serialization + the round-trip read the LIVE battle. When a snapshot was
        # supplied (the protected opponent/trainee path), action_to_choice already proved
        # the action conforms to that snapshot — so ANY failure reifying it against the
        # live battle means the board diverged under us (POKE_LOOP applied a background
        # message: a switch target fainted, a move got disabled, a force-switch began…).
        # That is STALENESS, not corruption: raise StaleDecisionError so it surfaces as a
        # mid-decision divergence — the self-play opponent RE-DECIDES on it, the trainee crashes
        # (crash-over-corruption) — rather than sending a wrong/obsolete order. This is the
        # catch-all for every live-divergence axis, not just switches. With a fresh snapshot
        # (legal=None: fuzz / back-to-back) there is nothing to be stale against, so failures
        # propagate as-is.
        try:
            order = choice_to_order(choice, battle)
            round_trip = order_to_action(order, battle, legal=legal)
        except StaleDecisionError:
            raise
        except (ValueError, RuntimeError) as e:
            if snapshot_provided:
                raise StaleDecisionError(
                    f"Live battle diverged from the decision snapshot while serializing "
                    f"action {action_idx} ({choice}): {e}"
                ) from e
            raise
        if round_trip != action_idx:
            msg = (
                f"Action conformance violation: the model picked action {action_idx}, "
                f"but the order we serialized back-maps to action {round_trip} "
                f"(choice={choice}, order={order})."
            )
            if snapshot_provided:
                raise StaleDecisionError(
                    msg + " The live battle diverged from the decision snapshot."
                )
            raise RuntimeError(
                msg + " We would send Showdown a different move/switch than the model "
                "selected — refusing rather than mis-acting."
            )
        return order

    # ------------------------------------------------------------------ #
    # Staleness / mid-decision guard (replaces the old context latch checks)
    # ------------------------------------------------------------------ #
    @staticmethod
    def assert_decision_current(ctx, battle) -> None:
        """Crash-over-corruption guard run before acting: the decision context must
        exist, belong to the current turn, and its captured ``legal`` snapshot must still
        match the server's current move ordering AND switch availability. A divergence
        means poke-env processed a background message and the request shifted under us —
        acting on the stale snapshot would send the wrong move/switch, so we raise
        ``StaleDecisionError`` instead (the self-play opponent re-decides on it; the trainee
        crashes — see the StaleDecisionError docstring). ``action_to_order`` is the backstop for
        the residual window between this check and serialization: any live-vs-snapshot failure
        there is reclassified as staleness too."""
        tag = getattr(battle, "battle_tag", "")
        race_trace.trace(
            tag,
            f"ASSERT ctx.turn={getattr(ctx, 'turn', None) if ctx is not None else None} "
            f"battle.turn={getattr(battle, 'turn', '?')}",
        )
        if ctx is None:
            raise RuntimeError(
                f"Decision context is missing at turn {getattr(battle, 'turn', '?')}. "
                "get_mask() / embed_battle() must run before action_to_order()."
                + race_trace.dump(tag)
            )
        if ctx.turn != battle.turn:
            raise StaleDecisionError(
                f"Decision context is from turn {ctx.turn} but current turn is "
                f"{battle.turn}. get_mask() / embed_battle() must run before "
                "action_to_order()." + race_trace.dump(tag)
            )
        legal = getattr(ctx, "legal", None)
        if legal is not None:
            current = LegalActions.from_battle(battle)
            # Compare EVERY action-relevant axis of the captured snapshot against the live
            # request. Any divergence means POKE_LOOP applied a background message between the
            # snapshot and now — a faint flipping us to force-switch (the move-face: moves→[]),
            # a switch target leaving the bench or changing identity (the heracross switch-face),
            # a move getting disabled, or the request going to wait / struggle / trapped. Acting
            # on the stale snapshot would send the wrong order, so we raise — the self-play
            # opponent RE-DECIDES on the now-current request (RLPlayer.choose_move), the trainee
            # crashes (gen3_env). This is the detector; the opponent's re-decide is the resolver.
            # Moves carry (id, disabled)
            # and switches carry (slot, species), so a same-length set whose CONTENTS changed is
            # caught too. action_to_order is the final backstop for the residual window between
            # this check and serialization.
            axes = (
                ("moves",
                 tuple((m.id, m.disabled) for m in legal.move_slots),
                 tuple((m.id, m.disabled) for m in current.move_slots)),
                ("switches",
                 tuple((s.slot, s.species) for s in legal.switches),
                 tuple((s.slot, s.species) for s in current.switches)),
                ("force_switch", legal.force_switch, current.force_switch),
                ("trapped", legal.trapped, current.trapped),
                ("maybe_trapped", legal.maybe_trapped, current.maybe_trapped),
                ("wait", legal.wait, current.wait),
                ("struggle", legal.struggle, current.struggle),
            )
            for axis, latched, live in axes:
                if latched != live:
                    raise StaleDecisionError(
                        f"Mid-decision state change at turn {battle.turn} [{axis}]: "
                        f"latched={latched!r}, server={live!r}" + race_trace.dump(tag)
                    )

    # ------------------------------------------------------------------ #
    # Reverse map (diagnostics / opponent labelling) — delegated to the adapter
    # ------------------------------------------------------------------ #
    @staticmethod
    def order_to_action(order: "BattleOrder", battle) -> int:
        """Reverse map a poke-env order to its action index (best-effort, returns 0 on
        anything unrecognized). Lives in :mod:`serialize` as the order boundary; this is
        a thin pass-through so existing call sites keep working."""
        return order_to_action(order, battle)
