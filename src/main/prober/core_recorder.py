"""The prober's legacy SUMMARY of a core trace, built from a Rust-core WALK (``gen3_core_walk_v1``).

The shape is the live eval recorder's (:class:`~agents.training.battle_recorder.BattleRecorder`:
``teams`` + one ``invocations`` entry per decision, each with its ``outcome``), and so are the rules —
this module re-states ``record`` / ``_fill_pending_outcome`` / ``finalize`` / ``to_summary`` over the
CORE's read-models instead of a poke-env battle:

* the board at a decision is the core's ``present()`` view (``LiveView``) and its legality
  (``LegalActions``) — the reading training's rows are encoded from;
* the per-decision CONTEXT (per-slot HP, the actives, the phase, the fainted sets) is built from that
  view and the core trackers' SLOT REGISTRIES, exactly as ``BattleContext.from_battle`` builds it;
* "what happened" is the core's frozen ``TurnDelta`` projection (``trackers/delta.rs``), the window
  each decision closed — the recorder reads only its switch / faint / resolved-move / HP-delta fields;
* every label is :mod:`agents.training.trace_labels` (shared with the live recorder), the reward
  breakdown is :func:`agents.training.reward_config.terminal_breakdown` (shared with the reward
  manager) and the result is :func:`agents.training.trace_result.classify_result`.

Float32 is kept wherever the live recorder computed in float32 (the context's HP arrays, the
``TurnDelta``'s HP deltas), so a ``%.0f`` label rounds the same way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.gen3_mechanics import boosts_str
from agents.training import trace_labels as L
from agents.training.reward_config import RewardConfig, terminal_breakdown
from agents.training.reward_weights import _TIMEOUT_TURN_CAP
from agents.training.trace_result import (DRAW, DRAW_KIND_KEY, RESULT_VOCABULARY, VOCABULARY_KEY,
                                          check_result, classify_result)
from main.prober.core_walk import CoreWalk, WalkDecision

#: poke-env's ``Pokemon._boosts`` key order — the order the live view's boosts dict (non-zero only)
#: carries them, and so the order ``boosts_str`` renders them in.
_BOOST_ORDER = ("accuracy", "atk", "def", "evasion", "spa", "spd", "spe")


class _Mon:
    """A ``LivePokemon`` stand-in whose boosts follow poke-env's key order (``boosts_str`` input)."""

    def __init__(self, mon):
        b = mon.boosts
        self.boosts = {k: b[k] for k in _BOOST_ORDER if k in b}
        unknown = set(b) - set(_BOOST_ORDER)
        if unknown:
            raise ValueError(f"unknown boost stats {sorted(unknown)}")


def _boosts(mon) -> Optional[str]:
    return None if mon is None else boosts_str(_Mon(mon))


@dataclass(frozen=True)
class _Ctx:
    """The ``BattleContext`` fields the recorder reads."""

    turn: int
    phase: str
    our_slot_map: Dict[str, int]
    opp_slot_map: Dict[str, int]
    our_hp: np.ndarray
    opp_hp: np.ndarray
    our_active: str
    opp_active: str
    our_fainted_count: int
    opp_fainted_count: int
    #: in BOARD order (not a set): a turn that fainted two mons names them in the board's order,
    #: deterministically — the live recorder's frozenset iterates in hash order (FINDING F-P5-1)
    our_fainted_species: Tuple[str, ...]
    opp_fainted_species: Tuple[str, ...]


def _hp(mons, slots: Dict[str, int]) -> np.ndarray:
    out = np.zeros(6, dtype=np.float32)
    for m in mons:
        s = slots.get(m.species)
        if s is not None:
            out[s] = m.hp_fraction
    return out


def _ctx(view, legal, our_slots: Sequence[str], opp_slots: Sequence[str]) -> _Ctx:
    our = {sp: i for i, sp in enumerate(our_slots)}
    opp = {sp: i for i, sp in enumerate(opp_slots)}
    oa, pa = view.ours.active, view.opp.active
    return _Ctx(
        turn=view.turn,
        phase="forced_switch" if (legal is not None and legal.force_switch) else "move_selection",
        our_slot_map=our, opp_slot_map=opp,
        our_hp=_hp(view.ours.mons, our), opp_hp=_hp(view.opp.mons, opp),
        our_active=oa.species if (oa is not None and not oa.fainted) else "NONE",
        opp_active=pa.species if pa is not None else "NONE",
        our_fainted_count=sum(1 for m in view.ours.mons if m.fainted),
        opp_fainted_count=sum(1 for m in view.opp.mons if m.fainted),
        our_fainted_species=tuple(m.species for m in view.ours.mons if m.fainted),
        opp_fainted_species=tuple(m.species for m in view.opp.mons if m.fainted),
    )


def _assign(slots: List[str], mons) -> None:
    """``SlotRegistry.assign`` over a side's mons, in order (the registries a terminal context grows)."""
    for m in mons:
        if m.species not in slots:
            slots.append(m.species)


def _hp_pct_ours(ctx: _Ctx) -> str:
    if ctx.our_active == "NONE":
        return "0%"
    return f"{ctx.our_hp[ctx.our_slot_map.get(ctx.our_active, 0)] * 100:.0f}%"


def _hp_pct_opp(ctx: _Ctx) -> str:
    if ctx.opp_active == "NONE":
        return "?%"
    slot = ctx.opp_slot_map.get(ctx.opp_active)
    return f"{ctx.opp_hp[slot] * 100:.0f}%" if slot is not None else "?%"


def _side(ctx_species: str, hp: str, mon, bench: str) -> dict:
    out: dict = {"species": ctx_species, "hp": hp}
    status = L.mon_display_status(mon)
    if status:
        out["status"] = status
    b = _boosts(mon)
    if b:
        out["boosts"] = b
    out["bench"] = bench
    return out


def _f32(xs: Sequence[float]) -> np.ndarray:
    return np.asarray(xs, dtype=np.float32)


def _reward(cfg: RewardConfig, view) -> dict:
    return terminal_breakdown(cfg, won=bool(view.won), lost=bool(view.lost), finished=bool(view.finished),
                              turn=view.turn).to_dict()


class CoreRecorder:
    """The summary of ONE walked side over its first ``n_rows`` decisions (the ``states.npz`` rows)."""

    def __init__(self, walk: CoreWalk, reward_config: RewardConfig):
        self.walk = walk
        self.cfg = reward_config
        self._invocations: List[dict] = []
        self._status_seen: Optional[dict] = None

    def summary(self, *, n_rows: int, actions: np.ndarray, probs: Sequence[np.ndarray],
                masks: Sequence[np.ndarray], step: int, battle_tag: str) -> dict:
        w = self.walk
        decs = w.decisions[:n_rows]
        if w.terminal_view is None:
            raise ValueError("the walk has no terminal view — the battle did not end")
        ctxs = [_ctx(d.view, d.legal, d.our_slots, d.opp_slots) for d in decs]
        for i, d in enumerate(decs):
            a = int(actions[i])
            entry = self._entry(i, d, ctxs[i], a, probs[i], masks[i])
            if i + 1 < len(decs):
                nxt = decs[i + 1]
                self._outcome(entry, ctxs[i], ctxs[i + 1], nxt.delta, nxt.view)
            else:
                # the stall-forfeit decision (a walk decision past the rows) closes the same window
                # the terminal does: its delta is the last recorded decision's outcome delta
                tdelta = w.decisions[n_rows].delta if len(w.decisions) > n_rows else w.terminal_delta
                self._final(entry, ctxs[i], d, tdelta)
            self._invocations.append(entry)
        return self._to_summary(step, battle_tag)

    # ---- one decision ------------------------------------------------------------------------

    def _entry(self, i: int, d: WalkDecision, ctx: _Ctx, a: int, probs: np.ndarray, mask: np.ndarray) -> dict:
        live, legal = d.view, d.legal
        if legal is None:
            raise ValueError(f"decision {i} has no legality")
        if self._status_seen is None:
            self._status_seen = L.snapshot_statuses(live)
        our_mon, opp_mon = live.ours.active, live.opp.active
        return {
            "i": i + 1,
            "turn": ctx.turn,
            "phase": ctx.phase,
            "chosen": L.action_label(a, live, legal),
            "our": _side(ctx.our_active, _hp_pct_ours(ctx), our_mon, L.bench_summary(live.ours.active, live.ours.mons)),
            "opp": _side(ctx.opp_active, _hp_pct_opp(ctx), opp_mon, L.bench_summary(live.opp.active, live.opp.mons)),
            "outcome": None,
            "actions": L.all_action_labels(live, probs, mask, legal),
        }

    def _outcome(self, entry: dict, prev: _Ctx, curr: _Ctx, delta: Mapping[str, Any], live) -> None:
        """``BattleRecorder._fill_pending_outcome``."""
        if delta["our_switch_to"]:
            we_action = f"switched_to:{delta['our_switch_to']}"
        elif prev.phase == "forced_switch" and curr.our_active not in ("NONE", prev.our_active):
            we_action = f"forced_switch_to:{curr.our_active}"
        else:
            we_action = entry["chosen"]
        opp_move_id = delta["opp_resolved_move_id"]
        if prev.phase == "forced_switch":
            they_action = "none"
        elif delta["opp_switch_to"]:
            if delta["opp_fainted"] and opp_move_id:
                they_action = f"{opp_move_id} → {delta['opp_switch_to']}_sent_in"
            elif delta["opp_fainted"]:
                they_action = f"{delta['opp_switch_to']}_sent_in"
            elif opp_move_id:
                they_action = f"{opp_move_id} → phazed_to:{delta['opp_switch_to']}"
            else:
                they_action = f"switched_to:{delta['opp_switch_to']}"
        elif opp_move_id is not None:
            they_action = opp_move_id
        else:
            they_action = "unknown"
        our_fainted = (L.newly_fainted(prev.our_fainted_species, curr.our_fainted_species, prev.our_active)
                       if delta["we_fainted"] else [])
        opp_fainted = (L.newly_fainted(prev.opp_fainted_species, curr.opp_fainted_species, prev.opp_active)
                       if delta["opp_fainted"] else [])
        our_ref = (our_fainted[0] if our_fainted else None) or (delta["our_switch_to"] or prev.our_active)
        opp_ref = (opp_fainted[0] if opp_fainted else None) or (delta["opp_switch_to"] or prev.opp_active)
        our_delta = _f32(delta["our_hp_delta"])[prev.our_slot_map.get(our_ref, 0)] * 100
        opp_delta = _f32(delta["opp_hp_delta"])[prev.opp_slot_map.get(opp_ref, 0)] * 100
        events = [f"our:{sp}:fainted" for sp in our_fainted] + [f"opp:{sp}:fainted" for sp in opp_fainted]
        st, self._status_seen = L.status_events(self._status_seen, live)
        events += st
        entry["outcome"] = {
            "our": {"action": we_action, "hp_delta": f"{our_delta:+.0f}%"},
            "opp": {"action": they_action, "hp_delta": f"{opp_delta:+.0f}%"},
            "reward": _reward(self.cfg, live),
            "events": events,
        }

    def _final(self, entry: dict, prev: _Ctx, last: WalkDecision, delta: Optional[Mapping[str, Any]]) -> None:
        """``BattleRecorder.finalize``: the last decision's outcome, read off the TERMINAL board."""
        if delta is None:
            raise ValueError("the walk has no terminal delta")
        live = self.walk.terminal_view
        our_hp = _hp(live.ours.mons, prev.our_slot_map)
        opp_hp = _hp(live.opp.mons, prev.opp_slot_map)
        final_our = tuple(m.species for m in live.ours.mons if m.fainted)
        final_opp = tuple(m.species for m in live.opp.mons if m.fainted)
        our_fainted = L.newly_fainted(prev.our_fainted_species, final_our, prev.our_active)
        opp_fainted = L.newly_fainted(prev.opp_fainted_species, final_opp, prev.opp_active)
        our_fainted_species = our_fainted[0] if our_fainted else prev.our_active
        opp_fainted_species = opp_fainted[0] if opp_fainted else prev.opp_active
        we_new = len(final_our) > prev.our_fainted_count
        opp_new = len(final_opp) > prev.opp_fainted_count
        our_ref = our_fainted_species if we_new else (delta["our_switch_to"] or prev.our_active)
        opp_ref = opp_fainted_species if opp_new else (delta["opp_switch_to"] or prev.opp_active)
        our_slot = prev.our_slot_map.get(our_ref, 0)
        opp_slot = prev.opp_slot_map.get(opp_ref, 0)
        our_delta = (our_hp[our_slot] - prev.our_hp[our_slot]) * 100
        opp_delta = (opp_hp[opp_slot] - prev.opp_hp[opp_slot]) * 100
        events = []
        if we_new:
            events += [f"our:{sp}:fainted" for sp in our_fainted]
        if opp_new:
            events += [f"opp:{sp}:fainted" for sp in opp_fainted]
        st, self._status_seen = L.status_events(self._status_seen, live)
        events += st
        res, kind = classify_result(won=live.won, lost=live.lost, finished=live.finished, turn=live.turn,
                                    turn_cap=_TIMEOUT_TURN_CAP)
        events.append(f"result:{res.lower()}" + (f":{kind}" if kind else ""))
        entry["outcome"] = {
            "our": {"action": entry["chosen"], "hp_delta": f"{our_delta:+.0f}%"},
            "opp": {"action": "unknown", "hp_delta": f"{opp_delta:+.0f}%"},
            "reward": _reward(self.cfg, live),
            "events": events,
        }

    # ---- the battle ----------------------------------------------------------------------------

    def _to_summary(self, step: int, battle_tag: str) -> dict:
        live = self.walk.terminal_view
        result, draw_kind = classify_result(won=live.won, lost=live.lost, finished=live.finished,
                                            turn=live.turn, turn_cap=_TIMEOUT_TURN_CAP)
        check_result(result)
        meta = {"step": step, "battle_id": battle_tag, "result": result, "turns": live.turn,
                "invocations": len(self._invocations), VOCABULARY_KEY: RESULT_VOCABULARY}
        if result == DRAW:
            meta[DRAW_KIND_KEY] = draw_kind
        return {
            "meta": meta,
            "teams": {
                "ours": [{"species": m.species, "item": m.item or "none",
                          "final_hp": f"{m.hp_fraction * 100:.0f}%", "fainted": m.fainted}
                         for m in live.ours.mons],
                "opponent": [{"species": m.species, "final_hp": f"{m.hp_fraction * 100:.0f}%",
                              "fainted": m.fainted} for m in live.opp.mons],
            },
            "invocations": self._invocations,
        }
