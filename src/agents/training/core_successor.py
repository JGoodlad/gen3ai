"""``gen3_core_search_v1`` — a search successor's FULL observation from the Rust core's VERSION.

The Rust Core Program's M2 search adoption (``designs/endstate/program_rust_core.md`` §2 M2, §6).
On ``materializer=core`` the search driver's tree is a tree of ``BattleVersion``s: an arm's
successor IS a version, and the driver hands back, per arm, that version's ``present()`` view (the
side's own stream folded into poke-env's reading — every presentation rule applied in Rust), its
legality, its raw request, and the READINGS of the ply's events. What is left for Python is exactly
what M3 has not crossed yet: the per-decision TRACKERS (recency, pair history, the event window,
the progress clock, the Hidden-Power belief) and the ENCODER (M4).

So a successor here is the view road's successor (:mod:`agents.training.view_successor`) with its
two Python re-derivations deleted — ``view_adapter``'s presentation rules and
``ViewEventFolder``'s re-parse of the ply's protocol — and nothing else changed: the same tracker
fork (the pinned-pickle thaw), the same ``record_context`` / ``advance_window`` cadence, the same
``view_context`` and the same encoder, so an obs byte that differs from the other roads is a
difference in the VERSION, which is what the three parity gates measure.

The driver folds every successor from the side's TEXT (``parse(render)``, the one observation path;
the typed shortcut and its integrity mode are deleted, program §4 M4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from agents.battle.battle_event import BattleEvent, EventKind
from agents.battle.core_view import legal_actions_from_core, live_view_from_core
from agents.battle.view_adapter import ViewBattle
from agents.training.view_successor import LazyTokens, ViewSuccessorFactory, _choice_map, view_context


def events_from_readings(readings: Sequence[Mapping[str, Any]]) -> List[BattleEvent]:
    """The ply's ``BattleEvent``s from the core's readings — the same eight fields
    ``Gen3Battle`` builds (slice E holds them equal, type-strict), with no field derived here."""
    return [BattleEvent(seq=r["seq"], turn=r["turn"], kind=EventKind[r["kind"]], side=r["side"],
                        actor_species=r["actor"], target_species=r["target"],
                        value=dict(r["value"]), raw=tuple(r["raw"]))
            for r in readings]


@dataclass(frozen=True)
class CoreSuccessor:
    """One arm's leaf — the same contract as :class:`~agents.training.view_successor.ViewSuccessor`."""

    obs: np.ndarray
    mask: np.ndarray
    action_choices: "Mapping[int, str]"
    _fork: Any

    def child(self, encoder) -> "CoreSuccessorFactory":
        tracker, events, tag = self._fork
        return CoreSuccessorFactory(tracker, encoder, events, tag)


class CoreSuccessorFactory(ViewSuccessorFactory):
    """The fork point every arm of one ply branches from: the ROOT's tracker (frozen once, thawed
    per arm) and the whole-battle event log to this fork. No board — the board is the version the
    driver holds."""

    __slots__ = ()

    def __init__(self, tracker, encoder, prior_events, battle_tag: str) -> None:
        super().__init__(tracker, None, encoder, prior_events, battle_tag)

    @classmethod
    def at_fork(cls, tracker, battle, encoder) -> "CoreSuccessorFactory":  # type: ignore[override]
        """The ROOT fork, from the player the shared prefix replay left at the branch decision."""
        return cls(tracker, encoder, battle.events_since(0), battle.battle_tag)

    def _encode(self, core: Mapping[str, Any], view: Mapping[str, Any], action: int, events):
        from agents.action.mask_generator import Gen3ActionMasker

        live = live_view_from_core(view, battle_tag=self._battle_tag)
        legal = legal_actions_from_core(core.get("legal"), core.get("request"))
        vbattle = ViewBattle(live, legal, view, events=self._prior_events + events)
        if legal is None or not legal.last_request:
            return None
        mask = Gen3ActionMasker.get_mask(vbattle, legal=legal, live=live).astype(np.int8)
        if int(mask.sum()) == 0 or live.finished:
            return None
        tr = self._clone_tracker()
        tr.advance(int(action))
        ctx = view_context(live, legal, mask, tr._our_slots, tr._opp_slots, events)
        tr.record_context(ctx, live)
        tr.advance_window(live, legal, events, events)
        obs = self._encoder.encode(
            vbattle, hp_tracker=tr.hidden_power_tracker, legal=legal,
            progress_clock=tr.progress_clock, recency=tr.recency,
            pair_history=tr.pair_history, event_window=tr.event_window,
        )
        return np.asarray(obs, dtype=np.float32), mask, tr, vbattle, legal

    def successor(self, core: Mapping[str, Any], action: int,  # type: ignore[override]
                  where: str = "") -> "Optional[CoreSuccessor]":
        """ONE arm from its ``core_pN`` payload, or ``None`` when the leaf opened no decision (an
        all-zero mask — a DEFERRAL in the live player, never a row). The cadence is the live one:
        advance the tracker with the action that was pressed, then record → advance → encode."""
        events = events_from_readings(core.get("events") or ())
        got = self._encode(core, core["view"], action, events)
        if got is None:
            return None
        obs, mask, tr, vbattle, legal = got
        return CoreSuccessor(
            obs=obs, mask=mask,
            action_choices=LazyTokens(lambda: _choice_map(vbattle, mask, legal)),
            _fork=(tr, self._prior_events + events, self._battle_tag))
