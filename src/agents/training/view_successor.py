"""``gen3_view_successor_v1`` — a search successor's FULL observation, trackers included, built
from the port's one-sided view instead of by replaying the ply's protocol through poke-env.

This is ``designs/rust_sim/one_sided_view.md`` **DEFERRAL D5**, which is the thing that stood
between the view payload and a ``materialize_from_view()``. The board half was already solved
(:mod:`agents.battle.view_adapter`); the missing half is the per-decision TRACKERS — recency, the
pair history, the 32-row event window, the progress clock and the Hidden-Power belief — every one
of which folds the EVENT LOG, which a board is not.

The design, and why this one
----------------------------
Two were available. **Emit the events from the port** was rejected: a ``BattleEvent`` is a fold
over poke-env's protocol with poke-env's own attribution rules on it (the Damp ``[of]``
redirection, Sleep-Talk delegation, the move-suffix ``[miss]`` synthetics), and §2 of the contract
doc is the standing finding that a rule which is really poke-env behaviour must stay on the Python
side of the wall. **Carry the ROOT's trackers forward and apply the branch's events** is what is
built here, with the events folded from the arm's OWN one-sided protocol — bytes ``expand_many``
already returns — by :class:`~agents.battle.event_fold.ViewEventFolder`, which keeps
``Gen3Battle._build_event`` verbatim by inheritance and replaces only the five board reads it
makes.

**The semantics are identical to the live path because the code is the same code.** The tracker
advance runs through ``EpisodeTracker.record_context`` / ``EpisodeTracker.advance_window`` — the
bodies of ``record`` and ``update_progress_clock``, split out rather than copied — in the same
order the live player uses (record → advance → encode). What this module supplies is the two
inputs those bodies would otherwise have taken off a poke-env ``Battle``: the ``BattleContext``
and the event window.

What a successor's context is, and is NOT
-----------------------------------------
``BattleContext`` carries 30-odd fields, of which the OBSERVATION path reads exactly fifteen (the
set is ``_FED`` below, and it was read out of the code rather than guessed:
``turn_delta.build_from_events``, ``progress_clock``, ``state_encoder.encode`` and
``episode_tracker``'s two HP hooks are the only consumers a successor reaches). Every one of the
fifteen is a function of the ``LiveView`` + the ``LegalActions`` + the ply's events, so all fifteen
are FED. The rest are the REWARD's and the replay recorder's, they have no source on this path,
and :class:`ViewContext` **raises on reading one** rather than serving a plausible default — the
same discipline ``view_adapter._ViewStrict`` applies to the board.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.battle.event_fold import ViewEventFolder
from agents.battle.live_view import LegalActions, LiveView
from agents.battle.view_adapter import read_models_from_payload
from agents.gen3_mechanics import boosts_array
from agents.training.battle_snapshot import BattleContext
from agents.training.clone_pins import (_pickle_pinned, _pin_shared,
                                        _unpickle_pinned)
from agents.training.slot_registry import SlotRegistry

#: The ``BattleContext`` fields a SUCCESSOR's observation path actually reads. Sourced by
#: grepping every consumer a successor reaches, not by inspection: ``TurnDelta.build_from_events``
#: (both the prev and the curr context), ``ProgressClock.update``, ``state_encoder.encode`` and
#: ``EpisodeTracker``'s ``_maybe_observe_hidden_power``. ``mask`` and the three cheap board-derived
#: sets are fed as well because they cost nothing and a diagnostic that raises is worse than one
#: that prints.
_FED = frozenset({
    "turn", "phase", "mask", "our_slot_map", "opp_slot_map", "our_hp", "opp_hp",
    "our_active", "opp_active", "our_fainted_count", "opp_fainted_count",
    "our_fainted_species", "opp_fainted_species", "our_team_status",
    "active_move_ids", "our_boosts", "opp_boosts", "our_team_order", "legal",
    "our_last_damaging_event", "opp_last_damaging_event",
    "opp_active_revealed_moves", "we_moved_first",
})


class ViewContext(BattleContext):
    """A :class:`BattleContext` built from the one-sided view, LOUD about what it does not carry.

    🚨 The unfed fields are not ``None`` by accident and must never be read as if they were: on
    the protocol road ``opp_last_move_id`` / the cant reasons / the six move-outcome flags /
    ``our_last_effectiveness`` are poke-env's own turn-gated state, and this road has no poke-env.
    They are REWARD and recorder inputs, and a search successor computes no reward. Reading one
    raises with the pointer, which is how a future consumer that starts needing it finds out
    at the source instead of by measuring something subtly wrong."""

    __slots__ = ()

    def __getattribute__(self, name: str) -> Any:
        if name in _UNFED:
            raise AttributeError(
                f"`{name}` is not carried by a one-sided VIEW successor's context "
                f"(gen3_view_successor_v1). It is poke-env turn-gated state with no source on "
                f"this road; the observation path does not read it. If a consumer now needs it, "
                f"see designs/rust_sim/one_sided_view.md — adding it means finding its fact in "
                f"the ply's event fold, not defaulting it here.")
        return object.__getattribute__(self, name)


#: Everything on ``BattleContext`` that :data:`_FED` does not name. Computed once, from the
#: dataclass itself, so a field ADDED to ``BattleContext`` is unfed-and-loud by default rather
#: than silently zero.
_UNFED = frozenset(f.name for f in BattleContext.__dataclass_fields__.values()) - _FED


def _hp_by_slot(mons, slots: SlotRegistry) -> np.ndarray:
    out = np.zeros(6, dtype=np.float32)
    for m in mons:
        slot = slots.get(m.species)
        if slot is not None:
            out[slot] = m.hp_fraction
    return out


def view_context(live: LiveView, legal: Optional[LegalActions], mask: np.ndarray,
                 our_slots: SlotRegistry, opp_slots: SlotRegistry,
                 events: Sequence[Any]) -> ViewContext:
    """The successor's :class:`BattleContext`, field for field against ``BattleContext.from_battle``.

    Each line below is the same expression that method runs, with its poke-env read replaced by
    the read-model's: ``battle.team.values()`` → ``live.ours.mons`` (the payload emits roster
    order, which IS what the slot registry and the obs layout key on),
    ``mon.current_hp_fraction`` → ``LivePokemon.hp_fraction``, ``boosts_array(active)`` → the same
    helper over the read-model's boosts dict. The slot registries are MUTATED here exactly as
    ``from_battle`` mutates them, because a slot id is assigned on first sight and must not depend
    on which road saw the mon first."""
    from agents.battle.turn_view import TurnView
    from agents.enums import Status
    from poke_env.battle.abstract_battle import DamagingMoveEvent

    for m in live.ours.mons:
        our_slots.assign(m.species)
    for m in live.opp.mons:
        opp_slots.assign(m.species)

    our_active_mon = live.ours.active
    opp_active_mon = live.opp.active
    # `from_battle` guards OUR active on `not fainted` and the opponent's on nothing at all.
    our_active = (our_active_mon.species
                  if (our_active_mon is not None and not our_active_mon.fainted) else "NONE")
    opp_active = opp_active_mon.species if opp_active_mon is not None else "NONE"

    active_move_ids = (list(legal.move_ids) if legal is not None else []) + [None] * 4
    active_move_ids = active_move_ids[:4]

    # The two DAMAGING-MOVE events, folded from this ply's own window by the SAME
    # `TurnView` → `DamagingMoveEvent` conversion `TurnDelta.build_from_events` runs. The
    # Hidden-Power belief reads the opponent's one (`_maybe_observe_hidden_power`), and it is
    # turn-gated on the protocol road to exactly the window folded here.
    tv = TurnView.from_events(list(events))

    def _to_dme(st):
        """poke-env's PROMOTION rule, not ``TurnView``'s looser one — and the difference is
        MEASURED, not stylistic.

        ``AbstractBattle`` captures a pending ``DamagingMoveEvent`` at every ``|move|`` of a
        Physical/Special move, and promotes it into ``*_last_damaging_move`` **only when an
        effectiveness emission lands in the same turn** (``|-resisted|`` / ``|-supereffective|``
        / ``|-immune|``); with no emission the pending event silently expires. ``TurnView``
        additionally fills ``damaging_move`` whenever the move DEALT damage, with a synthesised
        1.0 — right for the turn-delta consumer that asked for it, wrong here. Reading the
        looser one fed the Hidden-Power belief observations poke-env never made: 8 successor
        vectors over 14 fresh battles differed in the 17-dim HP block and nowhere else.

        The TURN GATE is the second half of the same property (``_last_turn_gated``): the slot
        reads only while it names the turn that has just resolved."""
        if st.effectiveness is None or st.damaging_move is None:
            return None
        if int(tv.turn) != int(live.turn) - 1:
            return None
        dm = st.damaging_move
        ts = dm.target_status
        return DamagingMoveEvent(
            user_species=dm.user_species or "",
            target_species=dm.target_species or "",
            target_status=Status.__members__.get(ts) if ts else None,
            move_id=dm.move_id or "",
            effectiveness=dm.effectiveness if dm.effectiveness is not None else 1.0,
        )

    return ViewContext(
        turn=int(live.turn),
        phase="forced_switch" if (legal is not None and legal.force_switch) else "move_selection",
        mask=mask,
        our_slot_map=our_slots.snapshot(),
        opp_slot_map=opp_slots.snapshot(),
        our_hp=_hp_by_slot(live.ours.mons, our_slots),
        opp_hp=_hp_by_slot(live.opp.mons, opp_slots),
        our_active=our_active,
        opp_active=opp_active,
        our_fainted_count=sum(1 for m in live.ours.mons if m.fainted),
        opp_fainted_count=sum(1 for m in live.opp.mons if m.fainted),
        our_fainted_species=frozenset(m.species for m in live.ours.mons if m.fainted),
        opp_fainted_species=frozenset(m.species for m in live.opp.mons if m.fainted),
        our_team_status={m.species: m.status for m in live.ours.mons},
        active_move_ids=active_move_ids,
        our_boosts=boosts_array(our_active_mon),
        opp_boosts=boosts_array(opp_active_mon),
        our_team_order=tuple(m.species for m in live.ours.mons),
        legal=legal,
        opp_active_revealed_moves=frozenset(
            opp_active_mon.move_ids if opp_active_mon is not None else ()),
        # 🚨 The six below have NO default on the dataclass, so they must be PASSED — and each
        # is in `_UNFED`, so reading one still raises. Passing `None` here is not a value: the
        # frozen `__init__` only writes them, and `ViewContext.__getattribute__` is what any
        # reader meets. A silent `None` would be the plausible-default failure this class exists
        # to refuse.
        opp_last_move_id=None,
        opp_all_last_move_ids=None,
        our_cant_reason=None,
        opp_cant_reason=None,
        our_last_effectiveness=None,
        opp_last_effectiveness=None,
        our_last_damaging_event=_to_dme(tv.ours),
        opp_last_damaging_event=_to_dme(tv.opp),
        we_moved_first=tv.we_moved_first,
    )


def intermediate_decisions(chunks: Sequence[str]) -> int:
    """How many DECISIONS the port resolved inside this ply before the one its view describes.

    🚨 **This is the one place the two roads describe different STATES, and it is measured, not
    feared** (``designs/rust_sim/one_sided_view.md`` D10). When an arm's ply KOs one of our mons,
    the replacement round is a second request inside the same ``expand_many`` arm. The port
    answers it through its own follow-up policy and hands back the board AFTER it, while
    ``materialize_branches`` stops at the FIRST request its action list cannot answer — so the
    protocol road's successor row is the replacement decision and the view's board is the turn
    beyond it. A leaf scored at the wrong one of those is not a rounding difference.

    The count is read off the arm's own protocol: every ``|request|`` line except the LAST, that
    a replay player would have turned into a decision row (non-empty, not a ``wait``, and
    carrying an ``active`` or a ``forceSwitch`` block — the rule ``_ReplayObsPlayer`` applies).
    Verified against the protocol road's realized row count on 104 arms over 6 fresh battles:
    the two agree exactly, 98 arms at 0 and 5 at 1 (the 6th was a read-model residual).

    A caller that must reproduce ``materialize_branches``' row therefore FALLS BACK to it when
    this is non-zero, and counts the fallback rather than hiding it."""
    import json

    reqs = [line[len("|request|"):]
            for chunk in chunks for line in str(chunk).split("\n")
            if line.startswith("|request|")]
    n = 0
    for raw in reqs[:-1]:
        if not raw.strip():
            continue
        try:
            req = json.loads(raw)
        except ValueError:                                  # noqa: PERF203 - not a request we can read
            continue
        if not req.get("wait") and (req.get("active") or req.get("forceSwitch")):
            n += 1
    return n


#: Announced ONCE per process: a pickle fall-back is a ~20x per-arm slowdown that nothing else
#: would mention, and an invisible perf regression is the failure mode this project has eaten
#: most often.
_FREEZE_WARNED = False


def _freeze(tracker, pins: Mapping[int, Any]) -> "Optional[bytes]":
    """The tracker graph as ONE pinned pickle, or ``None`` if it will not pickle."""
    global _FREEZE_WARNED
    pin_idx = {id(o): i for i, o in enumerate(pins.values())}
    try:
        return _pickle_pinned(tracker, pin_idx)
    except Exception as e:                                      # noqa: BLE001
        if not _FREEZE_WARNED:
            _FREEZE_WARNED = True
            print(f"⚠️ [view_successor] the tracker fork fell back to deepcopy (~20x slower per "
                  f"arm) — {type(e).__name__}: {str(e)[:160]}", file=sys.stderr, flush=True)
        return None


@dataclass(frozen=True)
class ViewSuccessor:
    """One arm's leaf: what ``materialize_branches`` returns for that arm, and nothing more.

    ``obs`` and ``mask`` are the pair a leaf is scored from; ``action_choices`` is the child's own
    legal surface from the REAL mapper, which is what makes a DEEPER ply possible. ``child`` is
    the fork THIS successor is, so a ply d+1 needs no poke-env battle either."""

    obs: np.ndarray
    mask: np.ndarray
    action_choices: Dict[int, str]
    _fork: "Tuple[Any, Any, list, str]"

    def child(self, encoder) -> "ViewSuccessorFactory":
        """The factory a DEEPER ply forks from. Built on demand — most plies are depth 1, and the
        pin walk it needs is a graph traversal worth skipping when nobody deepens."""
        tracker, board, events, tag = self._fork
        return ViewSuccessorFactory(tracker, board, encoder, events, tag)


class ViewSuccessorFactory:
    """The fork point every arm of one ply branches from, reusable across the whole arm set.

    Two things are carried and they are exactly the two the ply's own bytes cannot supply: the
    ``EpisodeTracker`` (deep-copied per arm, because each arm advances it differently) and the
    BOARD the event fold starts from (a :class:`~agents.battle.event_fold.ViewEventFolder`,
    ``branch()``-ed per arm — twelve small objects, against the pickled poke-env battle graph the
    protocol road restores).

    Build the ROOT one with :meth:`at_fork` from the prefix replay's player; build a deeper one
    with :meth:`ViewSuccessor.child`, which needs no battle at all."""

    __slots__ = ("_tracker", "_board", "_encoder", "_battle_tag", "_pins", "_pin_list",
                 "_blob", "_prior_events")

    def __init__(self, tracker, board, encoder, prior_events, battle_tag: str) -> None:
        self._tracker = tracker
        self._board = board
        self._encoder = encoder
        self._battle_tag = battle_tag
        #: Every event from the start of the battle to this fork. The encoder's two whole-log
        #: folds (the pending-Wish pair, D3, and the sleep-wake belief, D4) read it; a branch
        #: only APPENDS, so one list is shared by every arm.
        self._prior_events: list = list(prior_events)
        # The clone pins, walked ONCE here for the same reason `_PlayerSnapshot` walks them once:
        # a `MappingProxyType` (`LegalActions.last_request`) and a `logging.Logger` are both
        # reachable from the tracker's history and neither may be copied. `clone_pins` is the ONE
        # definition of that set — see its module docstring for why it is not two.
        self._pins: Dict[int, Any] = _pin_shared(tracker, {})
        self._pin_list: list = list(self._pins.values())
        # 🚨 FREEZE ONCE, THAW PER ARM — the same 9x `_PlayerSnapshot._freeze` documents, and
        # this road needs it for the same reason: a `deepcopy` re-walks and re-dispatches every
        # node of the tracker graph every time, while pickle walks it once here and the per-arm
        # cost collapses to a C-level rebuild from a flat buffer. MEASURED on a live fork with a
        # 253-event root: **10.5 ms per arm by deepcopy against 0.5 ms by thaw**, which was the
        # whole of the view road's per-arm cost and made it SLOWER than the protocol road it
        # replaces (0.84x at B=33) until this landed.
        self._blob: Optional[bytes] = _freeze(tracker, self._pins)

    def _clone_tracker(self):
        if self._blob is None:
            return copy.deepcopy(self._tracker, dict(self._pins))
        return _unpickle_pinned(self._blob, self._pin_list)

    @classmethod
    def at_fork(cls, tracker, battle, encoder) -> "ViewSuccessorFactory":
        """The ROOT fork, from the player the shared prefix replay left standing at the branch
        decision — the same state ``materialize_branches`` snapshots, and the only place a
        poke-env battle is still needed."""
        return cls(tracker, ViewEventFolder.seed_from(battle), encoder,
                   battle.events_since(0), battle.battle_tag)

    def successor(self, payload: Mapping[str, Any], chunks: Sequence[str],
                  action: int) -> "Optional[ViewSuccessor]":
        """ONE arm, or ``None`` when the arm opened no decision (an all-zero mask, which the live
        player treats as a DEFERRAL and never as a row — the rule ``_ReplayObsPlayer`` applies).

        🚨 The caller must have checked :func:`intermediate_decisions` first: an arm whose ply
        resolved a replacement round describes a different STATE on the two roads, and this
        method would answer for the port's, not for the materializer's.

        The cadence is the live one and the order is load-bearing: advance the tracker with the
        action that was pressed, fold the ply, build the read-models, then record → advance →
        encode."""
        from agents.action.mask_generator import Gen3ActionMasker

        board = self._board.branch()
        events = board.fold(chunks)
        live, legal, vbattle = read_models_from_payload(
            payload, battle_tag=self._battle_tag, events=self._prior_events + events)
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
        return ViewSuccessor(
            obs=np.asarray(obs, dtype=np.float32), mask=mask,
            action_choices=_choice_map(vbattle, mask, legal),
            _fork=(tr, board, self._prior_events + events, self._battle_tag))


def _choice_map(battle, mask: np.ndarray, legal: LegalActions) -> Dict[int, str]:
    """``{action index: sim choice string}`` for the successor's legal set, through the REAL
    mapper — the same primitive ``_ReplayObsPlayer._choice_map`` uses, so a deeper ply branches
    on exactly the tokens the protocol road would have offered."""
    from agents.action.mapper import Gen3ActionMapper

    out: Dict[int, str] = {}
    for idx in np.flatnonzero(np.asarray(mask)):
        try:
            out[int(idx)] = Gen3ActionMapper.action_to_order(
                int(idx), battle, legal=legal).message[len("/choose "):]
        except Exception:                                    # noqa: BLE001 — see _choice_map
            continue
    return out
