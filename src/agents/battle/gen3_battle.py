"""``Gen3Battle`` — an event-aware Gen 3 OU singles battle.

Architecture (as built: ``designs/ai_v4/impl_step8_strict_battle_api_and_turndelta_fold.md``):

    AbstractBattle  (poke-env fork base — state engine + dispatch)
       └── Battle   (classic singles — unchanged behaviour, upstream-clean)
             └── Gen3Battle   (current state + revealed-order event log)

**Enrich, don't reimplement.** ``Gen3Battle`` keeps poke-env's state tracker *verbatim*
— every line still flows through ``super().parse_message``, so HP / status / boosts /
items / hazards are derived by the same battle-tested code. We only **add** a second
view: an ordered :class:`~agents.battle.battle_event.BattleEvent` log captured in the
same parse pass, with attribution resolved *before* the line mutates state.

Because the override calls ``super()`` for every line, current-state behaviour is
identical to ``Battle`` by construction (the §7.2 state-equivalence guarantee is
structural, not merely tested). The only new control-flow is:
  * classify each line via :data:`~agents.battle.battle_event.MESSAGE_POLICY`
    (raising on a non-gen3 / unclassified keyword — a deliberate tripwire), and
  * append a ``BattleEvent`` for ``EVENT``-policy lines.

Injected via the ``battle_class`` seam on ``poke_env.player.Player``, the event log
propagates to every consumer (encoder, ``BattleContext``, action mask, replay
recorder) for free — nothing else needs to know it changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from poke_env.battle.battle import Battle
from poke_env.data.normalize import to_id_str

from agents.battle.battle_event import (
    EVENT_KIND,
    OPP,
    OURS,
    BattleEvent,
    EventKind,
    Policy,
    classify,
    from_clause_move_source,
)

if TYPE_CHECKING:
    # These three would be CIRCULAR at import time — `live_view` and `strict_view` both
    # import `Gen3Battle`. `live_view()` / `strict_view()` / `live_weather()` therefore
    # import them INSIDE the method body and annotate the return with a string. This block
    # is what makes those strings resolvable to a type checker (and to ruff, which
    # otherwise reports the annotations as undefined names — F821).
    from agents.battle.live_view import LiveView, LiveWeather
    from agents.battle.strict_view import StrictBattleView


# Tokens that look like a player identifier prefix ("p1", "p2"). Move/item name
# normalisation reuses poke-env's `to_id_str` so event ids match the encoder's.
def _is_ident(token: str) -> bool:
    return len(token) >= 2 and token[0] == "p" and token[1] in ("1", "2")


class Gen3Battle(Battle):
    """A :class:`Battle` that also records the revealed-order event log."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._events: List[BattleEvent] = []
        self._seq: int = 0
        # turn -> index of the first event belonging to that turn (O(1) slicing)
        self._turn_start: Dict[int, int] = {}
        # conservation bookkeeping
        self._policy_counts: Dict[Policy, int] = {p: 0 for p in Policy}
        self._lines_seen: int = 0
        # Running weather state, folded incrementally as WEATHER events arrive (see
        # _update_weather). Lets live_view() read weather in O(1) instead of re-scanning
        # the whole event log per call (which was O(turns²) over a battle).
        self._weather_id: Optional[str] = None
        self._weather_permanent: bool = False
        self._weather_start_turn: int = 0

    # ------------------------------------------------------------------ #
    # Public read API (consumed by TurnView / TurnDelta / reward / replay) #
    # ------------------------------------------------------------------ #
    def live_view(self) -> "LiveView":
        """Immutable snapshot of the **current board** (no past-turn state).

        This is the single source of truth for "what is true right now" — HP, status,
        boosts, revealed moves/item/ability, volatiles, hazards. For "what happened,
        in order" use :meth:`events_for_turn` / :class:`TurnView` instead. The two
        surfaces are deliberately disjoint so neither can drift from the other.
        """
        from agents.battle.live_view import LiveView

        return LiveView.from_battle(self)

    def strict_view(self) -> "StrictBattleView":
        """The strict boundary object our non-``battle/`` code should read through.

        Wraps this battle so consumers reach only the vetted surfaces — ``.live``
        (:class:`LiveView`), ``.legal`` (:class:`LegalActions`), ``.turn(n)`` /
        ``.history`` (:class:`TurnView`), and the event window — never the raw
        ``Battle`` / ``Pokemon`` objects. See :class:`StrictBattleView`.
        """
        from agents.battle.strict_view import StrictBattleView

        return StrictBattleView(self)

    @property
    def events(self) -> Sequence[BattleEvent]:
        """The whole-battle event log, in revealed order."""
        return self._events

    def events_for_turn(self, turn: int) -> List[BattleEvent]:
        """All events stamped with ``turn`` (the actions that resolved on that turn).

        Uses the turn-boundary index for an O(1) start; events also carry ``turn`` so
        the slice stays correct across forced-switch sub-turns.
        """
        start = self._turn_start.get(turn)
        if start is None:
            return [e for e in self._events if e.turn == turn]
        end = len(self._events)
        later = [t for t in self._turn_start if t > turn]
        if later:
            end = self._turn_start[min(later)]
        return self._events[start:end]

    @property
    def event_cursor(self) -> int:
        """The seq of the *next* event to be recorded (== number of events so far).

        A consumer captures this when it is asked for an input, then later calls
        :meth:`events_since` with the saved value to get exactly the events that were
        revealed in the window since — the right granularity for the per-decision
        ``TurnDelta`` window (which is "since I was last asked to act", NOT a protocol
        ``|turn|N`` boundary; a forced-switch can split a turn into two windows).
        """
        return self._seq

    def events_since(self, cursor: int) -> List[BattleEvent]:
        """All events revealed since ``cursor`` (a value previously read from
        :attr:`event_cursor`), in order. ``events_since(0)`` is the whole log.

        ``seq`` is assigned densely in append order, so this is an O(1) slice.
        """
        if cursor <= 0:
            return list(self._events)
        return self._events[cursor:]

    def events_between(self, start: int, end: Optional[int] = None) -> List[BattleEvent]:
        """Events with seq in ``[start, end)`` (``end=None`` ⇒ to the current end of the
        log). Unlike :meth:`events_since`, this UPPER-BOUNDS the window.

        The per-decision window for a PAST turn is ``[cursor_k, cursor_{k+1})``; for the
        most-recent turn pass ``end=None``. The turn-history feature folds each past slot
        over its OWN window with this — ``events_since(start)`` would run that turn through
        *now*, which both costs O(N²) across the N history slots and (because the fold
        takes the last move) makes an older slot report the latest turn.
        """
        lo = max(0, start)
        return self._events[lo:end]

    def live_weather(self) -> "LiveWeather":
        """The current :class:`LiveWeather`, from the incrementally-folded weather state
        (O(1)). Identical to ``_fold_weather`` over the whole log, computed on append
        instead of on read."""
        from agents.battle.live_view import LiveWeather

        if self._weather_id is None:
            return LiveWeather(weather=None, is_permanent=False, turns_active=0)
        return LiveWeather(
            weather=self._weather_id,
            is_permanent=self._weather_permanent,
            turns_active=max(0, self.turn - self._weather_start_turn),
        )

    # ------------------------------------------------------------------ #
    # Parse-pass override: classify -> mutate (super) -> record event     #
    # ------------------------------------------------------------------ #
    def parse_message(self, split_message: List[str]) -> None:
        keyword = split_message[1] if len(split_message) > 1 else ""
        policy, _reason = classify(keyword)  # raises on UNSUPPORTED / UNKNOWN
        self._lines_seen += 1
        self._policy_counts[policy] += 1

        if policy is not Policy.EVENT:
            super().parse_message(split_message)
            if keyword == "turn":
                # |turn|N has just set self.turn = N via end_turn(); mark the slice.
                self._turn_start.setdefault(self._turn, len(self._events))
            return

        # EVENT: snapshot the facts the line is about to destroy, mutate, then emit.
        pre = self._capture_pre(keyword, split_message)
        super().parse_message(split_message)
        ev = self._build_event(keyword, split_message, pre)
        if ev is not None:
            self._record(ev)
        if keyword == "move":
            for extra in self._move_suffix_events(split_message):
                self._record(extra)

    def record_choice_rejected(self, split_message: Sequence[str]) -> None:
        """Append a :class:`~agents.battle.battle_event.EventKind.CHOICE_REJECTED` event.

        Called OUT-OF-BAND from ``Player._handle_battle_message`` when the server returns
        ``|error|[Unavailable choice]`` — i.e. it refused the action we just chose. That line
        is intercepted by poke-env *before* ``parse_message`` (to set ``_trying_again`` and
        re-prompt), so it never flows through the normal parse pass; this is the explicit hook
        that makes the rejection a first-class, ordered event so the per-decision ``TurnDelta``
        window can fold it into the trap-reveal history signal.

        In gen3ou an unavailable choice means a switch we attempted while trapped (Arena Trap /
        Shadow Tag / Magnet Pull / Mean Look). The trapped mon is still active (the board has
        not advanced), so it is the actor; the attempted target slot is not on the wire and is
        recovered at fold time from the action index.

        Conservation is unaffected: this does NOT touch ``_lines_seen`` / ``_policy_counts``
        (the line never reached ``parse_message`` and is absent from ``_replay_data``), so the
        ``assert_conservation`` balance — lines_seen == replay_lines − terminals, bucket_sum ==
        lines_seen — still holds. (``len(events)`` already exceeds the EVENT bucket via the
        move-suffix synthetics, so no invariant ties them.)
        """
        reason = split_message[2] if len(split_message) > 2 else ""
        ev = self._new(
            EventKind.CHOICE_REJECTED,
            tuple(split_message),
            side=OURS,
            actor=self._active_species(OURS),
            value={"reason": reason},
        )
        self._record(ev)

    def _record(self, ev: BattleEvent) -> None:
        self._events.append(ev)
        self._seq += 1
        self._turn_start.setdefault(ev.turn, len(self._events) - 1)
        if ev.kind is EventKind.WEATHER:
            self._update_weather(ev)

    def _update_weather(self, ev: BattleEvent) -> None:
        """Fold one WEATHER event into the running weather state. Mirrors the per-event
        branch logic of ``live_view._fold_weather`` exactly so the result is identical:
        a bare/cause-tagged set updates the weather (ability cause ⇒ permanent) and resets
        the start turn; ``[upkeep]`` continues without resetting; ``none`` clears."""
        wid = ev.value.get("weather")
        if wid in (None, "none"):
            self._weather_id = None
            self._weather_permanent = False
            return
        if "[upkeep]" in ev.raw:
            if self._weather_id is None:  # upkeep seen without a prior set (mid-battle join)
                self._weather_id = wid
                self._weather_start_turn = ev.turn
            return
        self._weather_id = wid
        self._weather_permanent = str(ev.value.get("from", "")).startswith("ability")
        self._weather_start_turn = ev.turn

    # ------------------------------------------------------------------ #
    # Conservation invariant (design §4.3)                                #
    # ------------------------------------------------------------------ #
    def conservation_report(self) -> Dict[str, int]:
        """Per-bucket counts plus the totals used by :meth:`assert_conservation`."""
        # _replay_data also holds the terminal ["", "win"/"tie"] sentinel(s) appended
        # by won_by()/tied() outside parse_message — exclude them from the balance.
        terminal = sum(
            1 for e in self._replay_data if len(e) > 1 and e[1] in ("win", "tie")
        )
        return {
            **{p.name.lower(): self._policy_counts[p] for p in Policy},
            "events_recorded": len(self._events),
            "lines_seen": self._lines_seen,
            "replay_lines": len(self._replay_data),
            "terminal_sentinels": terminal,
        }

    def assert_conservation(self) -> Dict[str, int]:
        """Prove no protocol line was silently dropped (design §4.3).

        Every line poke-env appended to ``_replay_data`` (minus terminal win/tie
        sentinels) must have passed through our classifier exactly once, landing in
        exactly one :class:`Policy` bucket. ``UNSUPPORTED`` / ``UNKNOWN`` never appear
        — they raise — so a balanced battle has zero of them.
        """
        r = self.conservation_report()
        bucket_sum = sum(r[p.name.lower()] for p in Policy)
        if bucket_sum != r["lines_seen"]:
            raise AssertionError(
                f"policy buckets ({bucket_sum}) != lines seen ({r['lines_seen']}) — "
                f"a line was classified more than once or not at all: {r}"
            )
        expected = r["replay_lines"] - r["terminal_sentinels"]
        if r["lines_seen"] != expected:
            raise AssertionError(
                f"lines seen ({r['lines_seen']}) != replay lines minus terminals "
                f"({expected}) — a protocol line bypassed parse_message: {r}"
            )
        return r

    # ------------------------------------------------------------------ #
    # Attribution helpers                                                 #
    # ------------------------------------------------------------------ #
    def _side_of(self, ident: Optional[str]) -> Optional[str]:
        if not ident or not _is_ident(ident):
            return None
        return OURS if ident[:2] == self._player_role else OPP

    def _species_of(self, ident: Optional[str]) -> Optional[str]:
        if not ident or not _is_ident(ident):
            return None
        mon = self.get_pokemon(ident)
        return mon.species if mon else None

    def _active_species(self, side: Optional[str]) -> Optional[str]:
        if side is None:
            return None
        mon = self.active_pokemon if side == OURS else self.opponent_active_pokemon
        return mon.species if mon else None

    def _hp_fraction(self, ident: str) -> float:
        mon = self.get_pokemon(ident)
        return float(mon.current_hp_fraction) if mon else 0.0

    @staticmethod
    def _parse_from(split_message: Sequence[str]) -> Dict[str, str]:
        """Extract ``[from] …`` cause and ``[of] …`` source tokens, if present."""
        out: Dict[str, str] = {}
        for tok in split_message[3:]:
            t = tok.strip()
            if t.startswith("[from]"):
                out["from"] = t[len("[from]"):].strip()
            elif t.startswith("[of]"):
                out["of"] = t[len("[of]"):].strip()
        return out

    def _move_suffix_events(self, sm: Sequence[str]) -> List[BattleEvent]:
        """Legacy protocol variant: ``|move|…|[miss]`` / ``[notarget]`` carries the
        outcome on the move line instead of a standalone ``|-miss|`` / ``|-notarget|``.
        Emit the matching outcome event so the log is complete for both formats (the
        modern live server uses standalone lines, so this is a no-op there). poke-env
        mirrors both into its move-outcome trackers; we mirror both into the log."""
        out: List[BattleEvent] = []
        toks = set(sm[3:])
        if "[miss]" in toks:
            out.append(self._from_ident(
                EventKind.MISS, sm, sm[2], value={"op": "miss", "from": "move-suffix"}))
        if "[notarget]" in toks:
            out.append(self._from_ident(
                EventKind.FAIL, sm, sm[2], value={"op": "fail", "from": "move-suffix"}))
        return out

    @staticmethod
    def _delegated_from(split_message: Sequence[str], executed_id: str) -> Optional[str]:
        """Return the move that CALLED this one (Sleep Talk / Metronome / Snatch / …), or None for
        a free selection.

        Built on the shared :func:`~agents.battle.battle_event.from_clause_move_source` wire parser,
        so the bundled-gen3 BARE ``[from] <MoveName>`` form and the modern ``[from]move: <MoveName>``
        form are both handled in ONE place (they previously diverged — only the modern form was
        recognised, so gen3 Sleep Talk delegation was silently dropped). A ``[from]`` naming the
        SAME move (Pursuit-on-switch's self-tag ``[from] Pursuit``) or the ``lockedmove``
        continuation marker (Outrage / two-turn releases) is the mon running its own move, NOT a
        delegation."""
        src = from_clause_move_source(split_message[3:])
        if src is None or src == executed_id or src == "lockedmove":
            return None
        return src

    # ------------------------------------------------------------------ #
    # Pre-mutation capture (only the facts the line is about to overwrite) #
    # ------------------------------------------------------------------ #
    def _capture_pre(self, keyword: str, split_message: List[str]) -> Dict[str, Any]:
        pre: Dict[str, Any] = {}
        if keyword == "move":
            actor = split_message[2]
            side = self._side_of(actor)
            # Resolve the target: explicit identifier if present, else the other
            # side's active (singles). Status is sampled NOW — a frozen Flash Fire
            # holder thaws in the same hit, so post-resolution status would lie.
            if len(split_message) > 4 and _is_ident(split_message[4]):
                tmon = self.get_pokemon(split_message[4])
            else:
                tmon = (
                    self.opponent_active_pokemon
                    if side == OURS
                    else self.active_pokemon
                )
            pre["target_species"] = tmon.species if tmon else None
            pre["target_status"] = tmon.status.name if tmon and tmon.status else None
        elif keyword in ("-damage", "-heal", "-sethp"):
            pre["hp_before"] = self._hp_fraction(split_message[2])
        elif keyword in ("switch", "drag"):
            side = self._side_of(split_message[2])
            pre["prev_active"] = self._active_species(side) if side else None
        return pre

    # ------------------------------------------------------------------ #
    # Event construction (post-mutation — state is now settled)           #
    # ------------------------------------------------------------------ #
    def _new(
        self,
        kind: EventKind,
        split_message: Sequence[str],
        *,
        side: Optional[str] = None,
        actor: Optional[str] = None,
        target: Optional[str] = None,
        value: Optional[Dict[str, Any]] = None,
    ) -> BattleEvent:
        return BattleEvent(
            seq=self._seq,
            turn=self._turn,
            kind=kind,
            side=side,
            actor_species=actor,
            target_species=target,
            value=value or {},
            raw=tuple(split_message),
        )

    def _from_ident(
        self,
        kind: EventKind,
        split_message: Sequence[str],
        ident: Optional[str],
        *,
        target: Optional[str] = None,
        value: Optional[Dict[str, Any]] = None,
    ) -> BattleEvent:
        """Build an event whose side + actor come from a mon identifier (``p1a: …``)."""
        return self._new(
            kind,
            split_message,
            side=self._side_of(ident),
            actor=self._species_of(ident),
            target=target,
            value=value,
        )

    def _build_event(
        self, keyword: str, split_message: List[str], pre: Dict[str, Any]
    ) -> Optional[BattleEvent]:
        kind = EVENT_KIND[keyword]
        sm = split_message
        cause = self._parse_from(sm)

        # ---- moves ----
        if kind is EventKind.MOVE:
            actor_ident = sm[2]
            # The move id comes straight from THIS line (`|move|user|MoveName|…`).
            # We deliberately do NOT read `mon.last_move.id`: that is poke-env's
            # mutable "current state", and across a delegation sequence
            #   |move|Snorlax|Sleep Talk
            #   |move|Snorlax|Earthquake|[from] Sleep Talk   (bundled-gen3 BARE form, no `move:`)
            # it can still point at the delegating move (Sleep Talk) when the second
            # line resolves — so reading it mislabels the executed move. Each |move|
            # line names its own move, so sm[3] is both protocol-true and free of any
            # dependence on poke-env's state (the divergence the event log exists to
            # kill). Hidden Power stays the protocol's untyped "hiddenpower" (matching
            # the old DamagingMoveEvent); the resolved HP *type* lives on the mon's
            # moveset, not on the event.
            #
            # Delegation is captured explicitly: the called move emits its own MOVE
            # event carrying `from_move` (the delegator), and TurnView treats the
            # last MOVE of the turn as the one that actually fired.
            move_id = to_id_str(sm[3])
            value: Dict[str, Any] = {
                "move_id": move_id,
                "target_status": pre.get("target_status"),
            }
            delegated = self._delegated_from(sm, move_id)
            if delegated:
                value["from_move"] = delegated
            return self._from_ident(
                kind, sm, actor_ident, target=pre.get("target_species"), value=value
            )

        # ---- switches ----
        if kind in (EventKind.SWITCH, EventKind.DRAG):
            return self._from_ident(
                kind, sm, sm[2],
                value={
                    "prev_active": pre.get("prev_active"),
                    "details": sm[3] if len(sm) > 3 else "",
                },
            )

        # ---- faint ----
        if kind is EventKind.FAINT:
            return self._from_ident(kind, sm, sm[2])

        # ---- damage / heal ----
        if kind in (EventKind.DAMAGE, EventKind.HEAL):
            ident = sm[2]
            hp_after = self._hp_fraction(ident)
            hp_before = pre.get("hp_before", hp_after)
            value = {
                "amount": hp_after - hp_before,
                "hp_before": hp_before,
                "hp_after": hp_after,
            }
            if "from" in cause:
                value["reason"] = cause["from"]
            return self._from_ident(kind, sm, ident, value=value)

        # ---- sethp (Pain Split): absolute HP set, evented so the log is HP-complete ----
        if kind is EventKind.SETHP:
            ident = sm[2]
            hp_after = self._hp_fraction(ident)
            hp_before = pre.get("hp_before", hp_after)
            value = {
                "hp": hp_after,
                "hp_before": hp_before,
                "amount": hp_after - hp_before,  # signed delta, like DAMAGE/HEAL
            }
            if "from" in cause:
                value["reason"] = cause["from"]
            return self._from_ident(kind, sm, ident, value=value)

        # ---- boosts ----
        if kind in (EventKind.BOOST, EventKind.UNBOOST, EventKind.SETBOOST):
            stat = sm[3] if len(sm) > 3 else None
            raw_amt = int(sm[4]) if len(sm) > 4 and sm[4].lstrip("-").isdigit() else 0
            signed = -raw_amt if kind is EventKind.UNBOOST else raw_amt
            return self._from_ident(
                kind, sm, sm[2], value={"stat": stat, "amount": signed}
            )
        if kind is EventKind.CLEARBOOST:
            ident = sm[2] if len(sm) > 2 and _is_ident(sm[2]) else None
            return self._from_ident(
                kind, sm, ident, value={"op": keyword.lstrip("-")}
            )

        # ---- status ----
        if kind is EventKind.STATUS:
            value = {"status": sm[3] if len(sm) > 3 else None}
            if "from" in cause:
                value["reason"] = cause["from"]
            return self._from_ident(kind, sm, sm[2], value=value)
        if kind is EventKind.CURESTATUS:
            return self._from_ident(
                kind, sm, sm[2],
                value={"status": sm[3] if len(sm) > 3 else None, "op": keyword.lstrip("-")},
            )

        # ---- cant ----
        if kind is EventKind.CANT:
            # gen3_damp_cant_v1: Showdown puts an ABILITY-sourced cant on the ability HOLDER with
            # the BLOCKED move as its argument, and names the mon that actually lost its turn in
            # `[of]`:  |cant|p1a: Quagsire|ability: Damp|Self-Destruct|[of] p2a: Snorlax
            # Taken at face value that row says Quagsire could not use Self-Destruct — a move it
            # never had — while the side that really lost its turn goes unmentioned. Resolve the
            # `[of]` mon HERE, at emission, where the ident is still resolvable; the fold stays a
            # pure function of the log.
            _of = cause.get("of")
            return self._from_ident(
                kind, sm, sm[2],
                value={
                    "reason": sm[3] if len(sm) > 3 else None,
                    "move": to_id_str(sm[4]) if len(sm) > 4 else None,
                    "of": _of,
                    "of_side": self._side_of(_of) if _of else None,
                    "of_actor": self._species_of(_of) if _of else None,
                },
            )

        # ---- move outcome (crit/miss/fail): attach to the resolving mover ----
        if kind in (EventKind.CRIT, EventKind.MISS, EventKind.FAIL):
            mover = self._current_move_user_side  # "ours"/"opp"/None
            target_ident = sm[2] if len(sm) > 2 and _is_ident(sm[2]) else None
            return self._new(
                kind, sm, side=mover,
                actor=self._active_species(mover),
                target=self._species_of(target_ident),
                value={"op": keyword.lstrip("-")},
            )

        # ---- effectiveness: attach to the resolving mover ----
        if kind in (EventKind.IMMUNE, EventKind.RESISTED, EventKind.SUPEREFFECTIVE):
            mover = self._current_move_user_side
            defender_ident = sm[2] if len(sm) > 2 and _is_ident(sm[2]) else None
            if mover is None and defender_ident is not None:
                # Fall back: mover is the side opposite the named defender.
                d_side = self._side_of(defender_ident)
                mover = OPP if d_side == OURS else OURS if d_side == OPP else None
            mult = {
                EventKind.IMMUNE: 0.0,
                EventKind.RESISTED: 0.5,
                EventKind.SUPEREFFECTIVE: 2.0,
            }[kind]
            return self._new(
                kind, sm, side=mover,
                actor=self._active_species(mover),
                target=self._species_of(defender_ident),
                value={"multiplier": mult},
            )

        # ---- items ----
        if kind in (EventKind.ITEM, EventKind.ENDITEM):
            value = {"item": to_id_str(sm[3]) if len(sm) > 3 else None, **cause}
            return self._from_ident(kind, sm, sm[2], value=value)

        # ---- abilities ----
        if kind is EventKind.ABILITY:
            value = {
                "ability": to_id_str(sm[3]) if len(sm) > 3 else None,
                "op": "end" if keyword == "-endability" else "reveal",
                **cause,
            }
            return self._from_ident(kind, sm, sm[2], value=value)

        # ---- weather (no mon) ----
        if kind is EventKind.WEATHER:
            value = {"weather": to_id_str(sm[2]) if len(sm) > 2 else None, **cause}
            return self._new(kind, sm, value=value)

        # ---- field / side conditions ----
        if kind is EventKind.FIELD:
            return self._new(
                kind, sm,
                value={"effect": sm[2] if len(sm) > 2 else None, "op": keyword.lstrip("-")},
            )
        if kind is EventKind.SIDE:
            side_token = sm[2] if len(sm) > 2 else ""
            return self._new(
                kind, sm, side=self._side_of(side_token),
                value={
                    "condition": sm[3] if len(sm) > 3 else side_token,
                    "op": keyword.lstrip("-"),
                },
            )

        # ---- volatiles ----
        if kind in (EventKind.VOLATILE_START, EventKind.VOLATILE_END):
            value = {"effect": sm[3] if len(sm) > 3 else None, "op": keyword.lstrip("-"), **cause}
            return self._from_ident(kind, sm, sm[2], value=value)
        if kind is EventKind.ACTIVATE:
            value = {"effect": sm[3] if len(sm) > 3 else None, **cause}
            return self._from_ident(kind, sm, sm[2], value=value)

        # ---- prepare / mustrecharge ----
        if kind is EventKind.PREPARE:
            return self._from_ident(
                kind, sm, sm[2],
                value={"move": to_id_str(sm[3]) if len(sm) > 3 else None},
            )
        if kind is EventKind.MUSTRECHARGE:
            return self._from_ident(kind, sm, sm[2])

        # ---- transform / forme change / swap ----
        if kind is EventKind.TRANSFORM:
            target_ident = sm[3] if len(sm) > 3 and _is_ident(sm[3]) else None
            return self._from_ident(
                kind, sm, sm[2], target=self._species_of(target_ident)
            )
        if kind is EventKind.FORMECHANGE:
            return self._from_ident(
                kind, sm, sm[2],
                value={"details": sm[3] if len(sm) > 3 else "", "op": keyword.lstrip("-")},
            )
        if kind is EventKind.SWAP:
            return self._from_ident(
                kind, sm, sm[2], value={"detail": sm[3] if len(sm) > 3 else ""}
            )

        # Should be unreachable: EVENT policy without a builder branch.
        raise AssertionError(f"no event builder for keyword {keyword!r} (kind {kind!r})")
