"""The Rust Core parity harness — slice T, the TRACKERS (``gen3_core_parity_trackers_v1``, the Rust
Core Program's M3).

At EVERY decision of a recorded battle, for BOTH viewers, the tracker state TRAINING builds — the
real :class:`~agents.training.episode_tracker.EpisodeTracker` driven exactly as ``Gen3Env.embed_battle``
drives it (``record(battle, mask, legal)`` then ``update_progress_clock(battle, legal)``) over a
``Gen3Battle`` fed the viewer's per-side text — against the core's (``core_events --trackers``, the
trackers folded on the version from the same stream). Compared TYPE-strict, **no allowlist**:

* the slot registries, the Hidden-Power belief (every (16,) float32 vector, the ruled-out set, the
  infeasible count, the revision), the progress clock (``n`` and every piece of state ``n`` depends
  on, and ``value()``), recency, pair history, and the 32-row EVENT WINDOW (every row, every column
  the encoder reads, plus the open-move/forced/active state that decides the next rows);
* the whole-log wish and sleep folds (``build_wish_pending`` / ``build_sleep_sources``);
* the α/β opponent-intent LABEL for the decision the window closes (kind, move id, switch species,
  switch slot — the num lookups are the encoder's tables and cross at M4);
* the frozen ``TurnDelta``'s fields the two live consumers read (the clock, the label) — the
  projection, not the layout (the layout is not ported; program M3);
* the REWARD: the win indicator (``Gen3RewardManager`` under ``terminal_indicator``, ``victory_value``
  1.0, no shaping — the win-prob era's reward), at every decision and at the battle's end.

The core's NATIVE record (the ordered per-action window) is not compared here — it has no Python
twin; it is gated by its own fixtures (``src/rust_sim/tests/window_record_test.rs``) and measured by
the loss catalogue (``designs/research_state/measurements/rust_core_m3_2026-09-24/``).
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.battle.offline_feed import new_battle, player_names
from agents.battle.rust_core_parity_views import decision_points

#: The reward config every win-prob-era run trains on (`1 TERMINAL + 0 PBRS + 0 BIAS`).
_WIN_INDICATOR = dict(hand_shaping=False, terminal_indicator=True, victory_value=1.0, draw_penalty=0.0)


def _rank(side: str) -> int:
    return 0 if side == "ours" else 1


def _mon_map(d: dict) -> list:
    """``{(side, species): v}`` → ``[[side, species, v], …]`` in the core's order."""
    return [[s, sp, v] for (s, sp), v in sorted(d.items(), key=lambda kv: (_rank(kv[0][0]), kv[0][1]))]


def _pair_map(d: dict) -> list:
    return [[a, b, v] for (a, b), v in sorted(d.items())]


def _status_name(st) -> Optional[str]:
    return None if st is None else st.name.lower()


def _row(r: dict) -> dict:
    out = {k: r[k] for k in ("t", "actor", "side", "target", "move_id", "hp_delta", "missed", "failed",
                            "crit", "eff", "we_first", "status", "turn", "forced_window")}
    out["hp_delta"] = float(out["hp_delta"])
    out["eff"] = int(out["eff"])
    out["we_first"] = bool(out["we_first"])
    for k in ("faint_cause", "item_tr", "cant"):
        if k in r:
            out[k] = r[k]
    return out


def delta_projection(d) -> dict:
    """The frozen ``TurnDelta``'s fields the clock and the label read."""
    return {
        "v": 1,
        "our_move_id": d.our_move_id, "our_switch_to": d.our_switch_to, "our_prev_active": d.our_prev_active,
        "opp_move_id": d.opp_move_id, "opp_switch_to": d.opp_switch_to, "opp_prev_active": d.opp_prev_active,
        "opp_resolved_move_id": d.opp_resolved_move_id, "our_move_outcome": d.our_move_outcome,
        "our_status_applied": _status_name(d.our_status_applied),
        "our_status_cured": _status_name(d.our_status_cured),
        "opp_status_applied": _status_name(d.opp_status_applied),
        "we_fainted": bool(d.we_fainted), "opp_fainted": bool(d.opp_fainted),
        "our_failed_to_move": bool(d.our_failed_to_move),
        "phase_is_forced_switch": bool(d.phase_is_forced_switch),
        "decision_was_forced_switch": bool(d.decision_was_forced_switch),
        "our_damaging": d.our_damaging_event is not None, "opp_damaging": d.opp_damaging_event is not None,
        "opp_target_hp_delta": None if d.opp_target_hp_delta is None else float(np.float32(d.opp_target_hp_delta)),
        "our_hp_delta": [float(x) for x in np.asarray(d.our_hp_delta, dtype=np.float32)],
        "opp_hp_delta": [float(x) for x in np.asarray(d.opp_hp_delta, dtype=np.float32)],
    }


class _Nums:
    """A bijective id ↔ int registry, so ``build_opp_intent_label``'s ``int(num)`` runs on ids."""

    def __init__(self):
        self.to: Dict[str, int] = {}
        self.back: Dict[int, str] = {}

    def num(self, key: Optional[str]) -> Optional[int]:
        if not key:
            return None
        if key not in self.to:
            n = len(self.to) + 1
            self.to[key] = n
            self.back[n] = key
        return self.to[key]


def intent_label(delta, prev_frame: Sequence[str]) -> dict:
    """``build_opp_intent_label`` with id-valued lookups, decoded back to ids."""
    from poke_env.data.normalize import to_id_str

    from agents.training.opp_intent_labels import KIND_SWITCH, build_opp_intent_label

    moves, species = _Nums(), _Nums()
    frame = {to_id_str(s): i for i, s in enumerate(prev_frame)}
    kind, num, slot, sp = build_opp_intent_label(
        delta, moves.num, lambda s: frame.get(to_id_str(s) if s else ""), species.num)
    return {"kind": int(kind), "move_id": moves.back.get(num) if num else None,
            "switch_species": species.back.get(sp) if kind == KIND_SWITCH and sp else None,
            "switch_slot": None if slot < 0 else int(slot),
            # the mon that made the decision — meaningful (and compared) on a MOVE label only
            "attacker": delta.opp_prev_active if (delta is not None and kind == 0) else None}


def tracker_state(tr, delta, label: dict, battle) -> dict:
    """The Python tracker state in the core's ``SideTrackers::json`` shape."""
    from agents.battle.battle_event import OPP, OURS
    from agents.observation.sleep_belief import build_sleep_sources
    from agents.observation.wish_belief import build_wish_pending

    hp = tr.hidden_power_tracker
    clk = tr.progress_clock
    rec = tr.recency
    ph = tr.pair_history
    ew = tr.event_window
    wish = build_wish_pending(battle)
    last = {}
    for side in (OURS, OPP):
        la = ph._last.get(side)
        if la is not None:
            last[side] = {"move_id": la["move_id"], "was_switch": bool(la["was_switch"]), "missed": bool(la["missed"]),
                          "failed": bool(la["failed"]), "crit": bool(la["crit"]), "turn": int(la["turn"])}
    return {
        # decisions RECORDED — the history deque is capped (history_cap), the transition count is not
        "decisions": (tr._n_transitions + 1) if tr._history else 0,
        "our_slots": [s for s, _ in sorted(tr._our_slots._slots.items(), key=lambda kv: kv[1])],
        "opp_slots": [s for s, _ in sorted(tr._opp_slots._slots.items(), key=lambda kv: kv[1])],
        "hp": {"state": [[sp, [float(x) for x in v]] for sp, v in sorted(hp._state.items())],
               "ruled_out": sorted(hp._ruled_out), "infeasible": int(hp._infeasible_observations),
               "revision": int(hp._revision)},
        "clock": {"n": int(clk.n), "prev_spikes": int(clk._prev_spikes), "prev_our_spikes": int(clk._prev_our_spikes),
                  "prev_our_boost_sum": int(clk._prev_our_boost_sum), "prev_our_has_sub": bool(clk._prev_our_has_sub),
                  "heal_streak": int(clk._heal_streak), "is_rest_loop": bool(clk._is_rest_loop),
                  "rested_species": sorted(clk._rested_species)},
        "clock_value": float(clk.value()),
        "recency": {"turn": int(rec._turn), "seen": _mon_map(rec._seen), "acted": _mon_map(rec._acted),
                    "hit": _mon_map(rec._hit)},
        "pair": {"max_seq": int(ph._max_seq), "turn": int(ph._turn), "our_active": ph._our_active,
                 "opp_active": ph._opp_active, "last": last,
                 "switch_ins": _pair_map(ph._switch_ins), "attacks": _pair_map(ph._attacks),
                 "status_clicks": _pair_map(ph._status_clicks), "shared_count": _pair_map(ph._shared_count),
                 "shared_last_turn": _pair_map(ph._shared_last_turn)},
        "window": {"turn": int(ew._turn), "max_seq": int(ew._max_seq), "our_active": ew._our_active,
                   "opp_active": ew._opp_active,
                   "forced": [bool(ew._forced.get("our")), bool(ew._forced.get("opp"))],
                   "rows": [_row(r) for r in ew.window()]},
        "wish": [bool(wish[OURS]), bool(wish[OPP])],
        "sleep": [[s, sp, [bool(a), bool(b)]] for (s, sp), (a, b) in
                  sorted(build_sleep_sources(battle).items(), key=lambda kv: (_rank(kv[0][0]), kv[0][1]))],
        "delta": delta_projection(delta) if delta is not None else None,
        "label": label,
    }


def _same(x: Any, y: Any) -> bool:
    """TYPE-strict (an int is not a float; a float compares by value, -0.0 == 0.0)."""
    if isinstance(x, bool) or isinstance(y, bool):
        return type(x) is type(y) and x == y
    if type(x) is not type(y):
        return False
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    if isinstance(x, list):
        return len(x) == len(y) and all(_same(a, b) for a, b in zip(x, y))
    return x == y


def differences(x: Any, y: Any, path: str = "", out: Optional[list] = None, cap: int = 50) -> list:
    """EVERY path where ``x`` and ``y`` differ (type-strict), up to ``cap`` — one per leaf, so one
    divergent field never hides another."""
    out = [] if out is None else out
    if len(out) >= cap:
        return out
    if isinstance(x, dict) and isinstance(y, dict):
        for k in list(x) + [k for k in y if k not in x]:
            if k not in x or k not in y:
                out.append((f"{path}.{k}", x.get(k, "<absent>"), y.get(k, "<absent>")))
            else:
                differences(x[k], y[k], f"{path}.{k}", out, cap)
        return out
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append((f"{path}[len]", len(x), len(y)))
        for i, (a, b) in enumerate(zip(x, y)):
            differences(a, b, f"{path}[{i}]", out, cap)
        return out
    if not _same(x, y):
        out.append((path, x, y))
    return out


def first_difference(x: Any, y: Any, path: str = "") -> Optional[Tuple[str, Any, Any]]:
    """The first path where ``x`` and ``y`` differ (type-strict), or ``None``."""
    if isinstance(x, dict) and isinstance(y, dict):
        for k in list(x) + [k for k in y if k not in x]:
            if k not in x or k not in y:
                return (f"{path}.{k}", x.get(k, "<absent>"), y.get(k, "<absent>"))
            d = first_difference(x[k], y[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return (f"{path}[len]", len(x), len(y))
        for i, (a, b) in enumerate(zip(x, y)):
            d = first_difference(a, b, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if _same(x, y) else (path, x, y)


@dataclass
class TrackerCensus:
    battles: int = 0
    viewers: int = 0
    decisions: int = 0
    rewards: int = 0
    terminal_rewards: int = 0
    labels: collections.Counter = field(default_factory=collections.Counter)
    window_rows: int = 0
    divergences: collections.Counter = field(default_factory=collections.Counter)
    examples: Dict[str, Any] = field(default_factory=dict)
    refused: List[str] = field(default_factory=list)
    out_of_scope: collections.Counter = field(default_factory=collections.Counter)

    def diverge(self, key: str, example: Any) -> None:
        self.divergences[key] += 1
        self.examples.setdefault(key, example)

    def render(self) -> str:
        head = (f"{self.battles} battles, {self.viewers} viewers, {self.decisions} decisions, "
                f"{self.window_rows} event-window rows, labels {dict(self.labels)}, "
                f"{self.rewards} rewards ({self.terminal_rewards} terminal)")
        if self.out_of_scope:
            head += f"; OUT OF SCOPE (not gen3ou): {dict(self.out_of_scope)}"
        if not self.divergences and not self.refused:
            return f"✅ slice T: 0 divergences — {head}"
        out = [f"❌ slice T: {sum(self.divergences.values())} divergences in {len(self.divergences)} classes, "
               f"{len(self.refused)} refused — {head}"]
        for k, n in self.divergences.most_common():
            out.append(f"   {n:7d}  {k}\n            e.g. {self.examples[k]}")
        for r in self.refused[:10]:
            out.append(f"   REFUSED {r}")
        return "\n".join(out)


#: The formats slice T audits — the training path is gen3ou-only (slice V's scope).
SCOPE_FORMATS = frozenset({"gen3ou"})


def _field_of(path: str) -> str:
    """A divergence's CLASS: the first two path components (``.window.rows``, ``.clock.n``)."""
    parts = [p for p in path.replace("[", ".[").split(".") if p and not p.startswith("[")]
    return ".".join(parts[:2])


def boundary_violations(window: Sequence[Mapping]) -> List[Tuple[int, Any]]:
    """The INFORMATION BOUNDARY on one viewer's native record: a denied or refused opponent action
    carries ``"choice": "opp"`` and nothing else (the viewer never saw what the opponent chose); one
    of our own carries ``{"own": …}``. Returns ``(action index, action)`` for every violation."""
    out = []
    for i, a in enumerate(window):
        if a.get("kind") not in ("denied", "cant"):
            continue
        side = (a.get("actor") if a["kind"] == "denied" else a.get("mon"))[0]
        ch = a.get("choice")
        ok = ch == "opp" if side == "opp" else (isinstance(ch, dict) and set(ch) == {"own"})
        if not ok:
            out.append((i, a))
    return out


def check_trackers(label: str, chunks: Sequence[Tuple[str, str]], core_viewers: Sequence[Sequence[Mapping]],
                   census: TrackerCensus, teams: Optional[Mapping[str, str]] = None,
                   format_id: str = "gen3ou", ended: Optional[Mapping[str, Any]] = None,
                   obs: Optional[Any] = None) -> None:
    """Every decision of both viewers of one battle: the Python tracker state vs the core's.
    ``core_viewers[i]`` is ``core_events --trackers``' list of per-decision records for viewer i
    (``{"after", "trackers", "reward"}``) plus a final ``{"terminal": …}`` entry. ``obs`` (a
    :class:`agents.battle.rust_core_parity_obs.ObsCensus`) also runs slice O at every decision:
    the row ``Gen3Env.embed_battle`` encodes after this fold against the core's (``--obs``)."""
    from agents.action.mask_generator import Gen3ActionMasker
    from agents.training.episode_tracker import EpisodeTracker
    from agents.training.reward_config import RewardConfig
    from agents.training.reward_manager import Gen3RewardManager

    if format_id not in SCOPE_FORMATS:
        census.out_of_scope[format_id] += 1
        return
    census.battles += 1
    if obs is not None:
        obs.battles += 1
    lines_all = [ln for _, c in chunks for ln in c.split("\n")]
    names = player_names(lines_all)
    for vi, viewer in enumerate(("p1", "p2")):
        census.viewers += 1
        if obs is not None:
            obs.viewers += 1
        caps = [c for c in core_viewers[vi] if "trackers" in c]
        terminal = next((c for c in core_viewers[vi] if "terminal" in c), None)
        battle = new_battle(viewer, names, packed_team=(teams or {}).get(viewer))
        tr = EpisodeTracker(history_cap=1)
        mgr = Gen3RewardManager(config=RewardConfig(**_WIN_INDICATOR), progress_clock=tr.progress_clock)
        frame: List[str] = []
        k = 0
        for i, b in decision_points(chunks, viewer, battle):
            sv = b.strict_view()
            legal = sv.legal
            mask = Gen3ActionMasker.get_mask(b, legal=legal).astype(np.int8)
            if int(mask.sum()) == 0:
                continue
            where = f"{label}/{viewer}@t{b.turn}#c{i}"
            if k >= len(caps):
                census.diverge("[ALIGN] the reading decided where the core recorded no decision", (where,))
                continue
            cap = caps[k]
            k += 1
            if cap["after"] != i:
                census.diverge("[ALIGN] the core decided at a different chunk", (where, cap["after"]))
                continue
            census.decisions += 1
            for j, act in boundary_violations(cap.get("window") or ()):
                census.diverge("[BOUNDARY] a denial's choice crosses the information boundary", (where, j, act))
            tr.advance(0)
            tr.record(b, mask, legal=legal)
            delta = tr.update_progress_clock(b, legal)
            if obs is not None:
                from agents.battle.rust_core_parity_obs import compare_row, python_row

                compare_row(where, cap, python_row(b, tr, legal), mask, obs)
            lab = intent_label(delta, frame)
            census.labels[lab["kind"]] += 1
            from agents.observation.base import ObservationEncoder
            frame = [m.species for m in ObservationEncoder.get_team_list(b, is_opponent=True) if m is not None]
            want = tracker_state(tr, delta, lab, b)
            census.window_rows += len(want["window"]["rows"])
            seen = set()
            for path, a, b2 in differences(want, cap["trackers"]):
                key = f"[TRACKER] {_field_of(path)}"
                if key in seen:
                    continue
                seen.add(key)
                census.diverge(key, (where, path, "python", a, "core", b2))
            # the reward of the transition INTO this decision
            reward = float(mgr.process_turn_reward(b, delta))
            census.rewards += 1
            if not _same(reward, cap["reward"]):
                census.diverge("[REWARD] per-decision", (where, "python", reward, "core", cap["reward"]))
        if k != len(caps):
            census.diverge("[ALIGN] the core recorded a decision the reading never took",
                           (label, viewer, len(caps) - k))
        # the TERMINAL reward, once the whole stream has been read
        if terminal is not None and battle.finished:
            census.terminal_rewards += 1
            live = battle.live_view()
            reward = float(mgr.config.victory_value) if live.won else 0.0
            if not _same(reward, terminal["terminal"]):
                census.diverge("[REWARD] terminal", (label, viewer, "python", reward, "core", terminal["terminal"]))
