from typing import Callable, Optional
import numpy as np

from agents.battle.live_view import LivePokemon, LiveView
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.training.reward_function import RewardFunction
from agents.training.reward_tracker import RewardTracker
from agents.training.slot_registry import SlotRegistry
from agents.training.reward_manager import Gen3RewardManager
from agents.gen3_mechanics import boosts_str as _boosts_str_fn


class BattleRecorder:
    """
    Records every model invocation for a single battle and exports a
    human-readable summary JSON.

    Call record() once per choose_move(), finalize() at battle end,
    then to_summary() to get the exportable dict.

    Each invocation entry is self-contained — action labels are inlined,
    not referenced from a legend — so a human can read any turn in isolation.

    Reward computation is delegated to RewardTracker, which shares the same
    SlotRegistries so BattleContext slot lookups stay consistent.
    """

    def __init__(self, battle_tag: str, reward_fn_factory: Callable[[], RewardFunction] = Gen3RewardManager,
                 gamma: float = 0.99):
        self.battle_tag = battle_tag
        self._our_slots = SlotRegistry()
        self._opp_slots = SlotRegistry()
        self._invocations: list[dict] = []
        self._pending_entry: Optional[dict] = None
        self._tracker = RewardTracker(reward_fn_factory, self._our_slots, self._opp_slots)
        # Optional raw model I/O per invocation, parallel to _invocations
        # (obs/logits/value the model saw). Populated only when record() gets them;
        # exported via states_arrays() for offline forensic replay.
        self._states: list[dict] = []
        # The chosen action index per invocation (agent-side, always available).
        # Exported in states_arrays() so an offline replay can re-advance the
        # turn-history tracker with the exact actions the live player took —
        # required for bit-faithful obs materialization (obs_materializer.py).
        self._actions_taken: list[int] = []
        # Per-decision TD residual δ(t) = r(t) + γ·V(s_{t+1}) − V(s_t) — the SAME formula the
        # prober uses (main/prober/session.py::_td), the single source of truth. Computed live at
        # the NEXT record() (when complete_pending finalizes reward(t) and the current state
        # carries V(s_{t+1})), so the last decision yields no δ (no next state — matches the
        # prober leaving td_residual(last)=None). The left tail of these is the "critic got
        # blindsided" signal (#4) the eval cycle folds into eval/td_resid_tail_*.
        self._gamma = float(gamma)
        self._td_residuals: list[float] = []
        self._prev_value: Optional[float] = None  # V(s_t) carried across decisions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, battle, action_idx: int, probs: np.ndarray, mask: np.ndarray,
               state: Optional[dict] = None) -> None:
        """Record a model invocation. Call from choose_move() after prediction.

        `state` (optional): {"obs", "logits", "value"} as captured by
        RLPlayer._predict_best_action — the raw model I/O for this turn. Stored
        parallel to the invocation so a saved battle can be replayed exactly.
        """
        # Build the strict view ONCE; the recorder reads all current-board state
        # through its LiveView (and legality through .legal) — never the raw battle.
        # The raw `battle` is still passed to BattleContext/RewardTracker below, which
        # own their poke-env reads behind their own boundaries.
        view = battle.strict_view()
        live = view.live
        legal = view.legal
        curr_ctx = self._build_ctx(battle, mask, legal)
        self._states.append(state or {})
        self._actions_taken.append(int(action_idx))

        if self._pending_entry is not None:
            prev_ctx = self._tracker.pending_ctx
            delta, reward = self._tracker.complete_pending(curr_ctx, battle)
            self._fill_pending_outcome(prev_ctx, curr_ctx, delta, reward, live)
            # δ(prev) = r(prev) + γ·V(s_now) − V(s_prev): the critic's surprise on the just-closed
            # transition. `reward` is the scalar total for it (== outcome.reward.total the prober
            # reads); `_prev_value` is V(s_prev) stashed last call; this call's value is V(s_now).
            v_next = (state or {}).get("value")
            if self._prev_value is not None and v_next is not None and reward is not None:
                self._td_residuals.append(reward + self._gamma * float(v_next) - self._prev_value)

        chosen = self._action_label(action_idx, live, legal)
        our_mon = live.ours.active
        opp_mon = live.opp.active
        our_status = self._mon_display_status(our_mon)
        opp_status = self._mon_display_status(opp_mon)
        our_boosts = _boosts_str_fn(our_mon)
        opp_boosts = _boosts_str_fn(opp_mon)

        our_section: dict = {"species": curr_ctx.our_active, "hp": self._our_hp_pct(curr_ctx)}
        if our_status:
            our_section["status"] = our_status
        if our_boosts:
            our_section["boosts"] = our_boosts
        our_section["bench"] = self._our_bench_summary(live)

        opp_section: dict = {"species": curr_ctx.opp_active, "hp": self._opp_hp_pct(curr_ctx)}
        if opp_status:
            opp_section["status"] = opp_status
        if opp_boosts:
            opp_section["boosts"] = opp_boosts
        opp_section["bench"] = self._opp_bench_summary(live)

        # The model's top-k species guess for each still-hidden opp slot (only when the hidden-opponent
        # belief is on AND a slot is unrevealed; see RLPlayer._decode_belief). Sits right after `opp`
        # — "the board, then what we believe is still hidden" — and is omitted entirely otherwise.
        belief = (state or {}).get("belief")
        entry = {
            "i": len(self._invocations) + 1,
            "turn": curr_ctx.turn,
            "phase": curr_ctx.phase,
            "chosen": chosen,
            "our": our_section,
            "opp": opp_section,
            **({"belief": belief} if belief else {}),
            "outcome": None,
            "actions": self._all_action_labels(live, probs, mask, legal),
        }

        self._tracker.begin_turn(curr_ctx, action_idx, view.event_cursor)
        self._pending_entry = entry
        # Carry V(s_now) so the NEXT record() can close δ for this decision.
        self._prev_value = (state or {}).get("value")

    def td_residuals(self) -> list[float]:
        """The per-decision TD residuals δ closed so far (one per non-terminal decision).

        Empty when no value was captured (the cheap fast path) or fewer than two decisions ran.
        The eval cycle pools these across an opponent's captured battles → a tail statistic
        (``eval/td_resid_tail_*``). Read it BEFORE the recorder is discarded at battle end."""
        return list(self._td_residuals)

    def states_arrays(self) -> dict:
        """Stack the per-invocation raw model I/O into npz-ready arrays, aligned
        index-for-index with to_summary()['invocations']. Returns {} if states
        were never captured. `has_state` flags which turns carry real data.
        `actions` is the chosen action index per invocation (always real data —
        it's the agent-side input an offline obs replay needs to re-advance the
        turn-history tracker; see obs_materializer.py)."""
        if not any(self._states):
            return {}
        obs_dim = next(len(s["obs"]) for s in self._states if s)
        n_act = next(len(s["logits"]) for s in self._states if s)
        T = len(self._states)
        obs = np.zeros((T, obs_dim), dtype=np.float32)
        logits = np.zeros((T, n_act), dtype=np.float32)
        values = np.zeros(T, dtype=np.float32)
        # P(win) from the win-probability head, parallel to `values`. NaN = no head (--win-prob-mode
        # none) / not captured, so the prober can distinguish "unavailable" from a real P(win)=0.0.
        win_probs = np.full(T, np.nan, dtype=np.float32)
        # Distributional value head's per-atom return distribution [T, bins], parallel to `values`. The
        # key is OMITTED entirely when the head is off (no state carried a distribution) so the prober's
        # KeyError guard reads "unavailable"; a captured-but-headless row stays all-NaN.
        vd_bins = next((len(s["value_dist"]) for s in self._states
                        if s and s.get("value_dist") is not None), 0)
        value_dist = np.full((T, vd_bins), np.nan, dtype=np.float32) if vd_bins else None
        has_state = np.zeros(T, dtype=np.int8)
        for i, s in enumerate(self._states):
            if not s:
                continue
            obs[i] = s["obs"]
            logits[i] = s["logits"]
            values[i] = s.get("value", 0.0)
            wp = s.get("win_prob")
            if wp is not None:
                win_probs[i] = float(wp)
            if value_dist is not None:
                vd = s.get("value_dist")
                if vd is not None:
                    value_dist[i] = np.asarray(vd, dtype=np.float32)
            has_state[i] = 1
        actions = np.asarray(self._actions_taken, dtype=np.int16)
        out = {"obs": obs, "logits": logits, "values": values, "win_probs": win_probs,
               "has_state": has_state, "actions": actions}
        if value_dist is not None:
            out["value_dist"] = value_dist
        return out

    def finalize(self, battle) -> None:
        """Complete the last pending invocation at battle end."""
        if self._pending_entry is None:
            return

        view = battle.strict_view()
        live = view.live
        prev_ctx = self._tracker.pending_ctx
        our_hp, opp_hp = self._terminal_hp(live)
        terminal_ctx, delta, reward = self._tracker.finalize(battle)

        # HP delta: for faint turns use the fainted mon's slot, not the forced switch-in.
        our_ref = prev_ctx.our_active if delta.we_fainted else (delta.our_switch_to or prev_ctx.our_active)
        opp_ref = prev_ctx.opp_active if delta.opp_fainted else (delta.opp_switch_to or prev_ctx.opp_active)
        our_slot = prev_ctx.our_slot_map.get(our_ref, 0)
        opp_slot = prev_ctx.opp_slot_map.get(opp_ref, 0)
        our_delta = (our_hp[our_slot] - prev_ctx.our_hp[our_slot]) * 100
        opp_delta = (opp_hp[opp_slot] - prev_ctx.opp_hp[opp_slot]) * 100

        events = []
        final_our_fainted = sum(1 for m in live.ours.mons if m.fainted)
        final_opp_fainted = sum(1 for m in live.opp.mons if m.fainted)
        if final_our_fainted > prev_ctx.our_fainted_count:
            events.append(f"our:{prev_ctx.our_active}:fainted")
        if final_opp_fainted > prev_ctx.opp_fainted_count:
            events.append(f"opp:{prev_ctx.opp_active}:fainted")

        self._append_status_events(events, prev_ctx, delta, live)

        if view.won:
            events.append("result:win")
        elif view.lost:
            events.append("result:loss")
        else:
            events.append("result:tie")

        breakdown = getattr(self._tracker._reward_fn, "_last_breakdown", None)
        self._pending_entry["outcome"] = {
            "our": {"action": self._pending_entry["chosen"], "hp_delta": f"{our_delta:+.0f}%"},
            "opp": {"action": "unknown",                     "hp_delta": f"{opp_delta:+.0f}%"},
            "reward": breakdown.to_dict() if breakdown is not None else round(reward, 3),
            "events": events,
        }
        self._invocations.append(self._pending_entry)
        self._pending_entry = None

    def to_summary(self, battle, step: int) -> dict:
        """Export the full battle summary as a JSON-serializable dict."""
        view = battle.strict_view()
        live = view.live
        if view.won:
            result = "WIN"
        elif view.lost:
            result = "LOSS"
        else:
            result = "TIE"

        return {
            "meta": {
                "step": step,
                "battle_id": self.battle_tag,
                "result": result,
                "turns": view.turn,
                "invocations": len(self._invocations),
            },
            "teams": {
                "ours": [
                    {
                        "species": m.species,
                        "item": m.item or "none",
                        "final_hp": f"{m.hp_fraction * 100:.0f}%",
                        "fainted": m.fainted,
                    }
                    for m in live.ours.mons
                ],
                "opponent": [
                    {
                        "species": m.species,
                        "final_hp": f"{m.hp_fraction * 100:.0f}%",
                        "fainted": m.fainted,
                    }
                    for m in live.opp.mons
                ],
            },
            "invocations": self._invocations,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ctx(self, battle, mask: np.ndarray, legal=None) -> BattleContext:
        return BattleContext.from_battle(battle, mask, self._our_slots, self._opp_slots, legal)

    def _action_label(self, action_idx: int, live: LiveView, legal) -> str:
        team_list = live.ours.mons
        move_ids = list(legal.move_ids)

        if action_idx < 6:
            return f"switch:{team_list[action_idx].species}" if action_idx < len(team_list) else f"switch:slot{action_idx}"
        elif action_idx < 10:
            m = action_idx - 6
            return move_ids[m] if m < len(move_ids) else f"move{m}"
        return "struggle"

    def _all_action_labels(self, live: LiveView, probs: np.ndarray, mask: np.ndarray, legal) -> dict:
        team_list = live.ours.mons
        move_ids = list(legal.move_ids)

        result = {}
        for i in range(11):
            if i < 6:
                label = f"switch:{team_list[i].species}" if i < len(team_list) else f"switch:slot{i}"
            elif i < 10:
                m = i - 6
                label = move_ids[m] if m < len(move_ids) else f"move{m}"
            else:
                label = "struggle"
            result[label] = {"prob": f"{probs[i] * 100:.1f}%", "valid": bool(mask[i])}
        return result

    @staticmethod
    def _mon_display_status(mon: LivePokemon | None) -> str | None:
        """Rich status string including counters and volatiles, read from a
        :class:`LivePokemon` (id-form status + ``{volatile_id: counter}``).

        Examples: "SLP(3)", "TOX(5)", "BRN", "PAR|TAUNT", "PERISH(2)|CONF"
        """
        if mon is None:
            return None
        parts = []
        status = mon.status  # id form: 'slp'/'tox'/'brn'/'par'/'frz'/'psn'/'fnt' or None
        ctr = mon.status_counter or 0
        vol = mon.volatiles  # {volatile_id: counter}
        if status == "slp":
            parts.append(f"SLP({ctr})" if ctr else "SLP")
        elif status == "tox":
            parts.append(f"TOX({ctr})" if ctr else "TOX")
        elif status is not None:
            name_map = {"brn": "BRN", "par": "PAR", "frz": "FRZ", "psn": "PSN"}
            if name := name_map.get(status):
                parts.append(name)
        effect_names = {
            "taunt": "TAUNT", "confusion": "CONF", "encore": "ENCORE",
            "attract": "ATTRACT", "disable": "DISABLE", "substitute": "SUB",
        }
        for vid, name in effect_names.items():
            if vid in vol:
                parts.append(name)
        for n, vid in [(3, "perish3"), (2, "perish2"), (1, "perish1"), (0, "perish0")]:
            if vid in vol:
                parts.append(f"PERISH({n})")
                break
        return "|".join(parts) if parts else None

    @staticmethod
    def _status_key(status_str: str | None) -> str | None:
        """Normalize for change detection — strips counter values, sorts parts."""
        if not status_str:
            return None
        return "|".join(sorted(p.split("(")[0] for p in status_str.split("|")))

    def _our_bench_summary(self, live: LiveView) -> str:
        return self._bench_summary(live.ours.active, live.ours.mons)

    def _opp_bench_summary(self, live: LiveView) -> str:
        return self._bench_summary(live.opp.active, live.opp.mons)

    def _bench_summary(self, active: LivePokemon | None, mons) -> str:
        parts = []
        for mon in mons:
            if active and mon.species == active.species:
                continue
            if mon.fainted:
                parts.append(f"{mon.species}(faint)")
            else:
                pct = f"{mon.hp_fraction * 100:.0f}%"
                status = self._mon_display_status(mon)
                parts.append(f"{mon.species}({pct},{status})" if status else f"{mon.species}({pct})")
        return ", ".join(parts)

    def _our_hp_pct(self, ctx: BattleContext) -> str:
        if ctx.our_active == "NONE":
            return "0%"
        return f"{ctx.our_hp[ctx.our_slot_map.get(ctx.our_active, 0)] * 100:.0f}%"

    def _opp_hp_pct(self, ctx: BattleContext) -> str:
        if ctx.opp_active == "NONE":
            return "?%"
        slot = ctx.opp_slot_map.get(ctx.opp_active)
        return f"{ctx.opp_hp[slot] * 100:.0f}%" if slot is not None else "?%"

    def _append_status_events(self, events: list, prev_ctx: BattleContext,
                              delta: TurnDelta, live: LiveView) -> None:
        """Append events for any status conditions newly applied this turn."""
        prev_our_status = self._pending_entry["our"].get("status")
        prev_opp_status = self._pending_entry["opp"].get("status")

        # Only track the mon that was active at decision time; skip if they fainted.
        if not delta.we_fainted:
            our_mon = live.ours.get(prev_ctx.our_active)
            if our_mon:
                new_status = self._mon_display_status(our_mon)
                if self._status_key(new_status) != self._status_key(prev_our_status):
                    if new_status:
                        events.append(f"our:{prev_ctx.our_active}:{new_status}")

        if not delta.opp_fainted:
            opp_mon = live.opp.get(prev_ctx.opp_active)
            if opp_mon:
                new_status = self._mon_display_status(opp_mon)
                if self._status_key(new_status) != self._status_key(prev_opp_status):
                    if new_status:
                        events.append(f"opp:{prev_ctx.opp_active}:{new_status}")

    def _fill_pending_outcome(self, prev_ctx: BattleContext, curr_ctx: BattleContext,
                              delta: TurnDelta, reward: float, live: LiveView) -> None:
        """Build and commit the JSON outcome for the pending entry using already-computed delta/reward."""
        # Our action
        if delta.our_switch_to:
            we_action = f"switched_to:{delta.our_switch_to}"
        elif prev_ctx.phase == "forced_switch" and curr_ctx.our_active not in ("NONE", prev_ctx.our_active):
            we_action = f"forced_switch_to:{curr_ctx.our_active}"
        else:
            we_action = self._pending_entry["chosen"]

        # Their action — distinguish: voluntary switch / faint+forced-switch / phaze / moved.
        # `delta.opp_resolved_move_id` prefers the protocol-confirmed event when
        # a damaging move resolved (protects against the stale-last_move bug
        # surfaced in step 3 smoke tests), and falls back to the inferred
        # opp_move_id for non-damaging moves (status, Roar, BP).
        opp_move_id = delta.opp_resolved_move_id
        if prev_ctx.phase == "forced_switch":
            they_action = "none"
        elif delta.opp_switch_to:
            if delta.opp_fainted and opp_move_id:
                they_action = f"{opp_move_id} → {delta.opp_switch_to}_sent_in"
            elif delta.opp_fainted:
                they_action = f"{delta.opp_switch_to}_sent_in"
            elif opp_move_id:
                they_action = f"{opp_move_id} → phazed_to:{delta.opp_switch_to}"
            else:
                they_action = f"switched_to:{delta.opp_switch_to}"
        elif opp_move_id is not None:
            they_action = opp_move_id
        else:
            # No protocol-confirmed (opp_resolved_move_id) or inferred (opp_move_id)
            # move this window. The old fallback read poke-env's stale `last_move` —
            # exactly the past-turn-field footgun this migration removes. History now
            # comes from the event-log TurnDelta, so an empty window is just "unknown".
            they_action = "unknown"

        # HP delta: for faint turns use the fainted mon's slot, not the forced switch-in.
        our_ref = prev_ctx.our_active if delta.we_fainted else (delta.our_switch_to or prev_ctx.our_active)
        opp_ref = prev_ctx.opp_active if delta.opp_fainted else (delta.opp_switch_to or prev_ctx.opp_active)
        our_slot = prev_ctx.our_slot_map.get(our_ref, 0)
        opp_slot = prev_ctx.opp_slot_map.get(opp_ref, 0)
        our_delta = delta.our_hp_delta[our_slot] * 100
        opp_delta = delta.opp_hp_delta[opp_slot] * 100

        events = []
        if delta.we_fainted:
            events.append(f"our:{prev_ctx.our_active}:fainted")
        if delta.opp_fainted:
            events.append(f"opp:{prev_ctx.opp_active}:fainted")

        self._append_status_events(events, prev_ctx, delta, live)

        breakdown = getattr(self._tracker._reward_fn, "_last_breakdown", None)
        self._pending_entry["outcome"] = {
            "our": {"action": we_action,   "hp_delta": f"{our_delta:+.0f}%"},
            "opp": {"action": they_action, "hp_delta": f"{opp_delta:+.0f}%"},
            "reward": breakdown.to_dict() if breakdown is not None else round(reward, 3),
            "events": events,
        }
        self._invocations.append(self._pending_entry)
        self._pending_entry = None

    def _terminal_hp(self, live: LiveView) -> tuple[np.ndarray, np.ndarray]:
        our_hp = np.zeros(6, dtype=np.float32)
        for mon in live.ours.mons:
            slot = self._our_slots.get(mon.species)
            if slot is not None:
                our_hp[slot] = mon.hp_fraction
        opp_hp = np.zeros(6, dtype=np.float32)
        for mon in live.opp.mons:
            slot = self._opp_slots.get(mon.species)
            if slot is not None:
                opp_hp[slot] = mon.hp_fraction
        return our_hp, opp_hp


def write_battle_record(out_prefix: str, recorder: "BattleRecorder", battle, step: int) -> None:
    """Finalize a recorder and write its forensic trace to disk.

    Writes three co-located artifacts for the same battle:

    * ``<out_prefix>_summary.json`` — the human-readable per-invocation summary.
    * ``<out_prefix>_states.npz`` — obs/logits/values aligned with the summary's
      invocations (only when raw model I/O was captured), for ``probe_replay.py``.
    * ``<out_prefix>_replay.html`` — a self-contained, **browser-watchable** Showdown
      replay of the battle. The two files above are prober-only forensic dumps; this
      one lets a human (no checkout, no prober) just open the game in a browser.

    Shared by the eval forensic capture so the on-disk format stays identical.
    """
    import os
    import re
    import json
    import sys

    recorder.finalize(battle)
    summary = recorder.to_summary(battle, step)

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    with open(f"{out_prefix}_summary.json", "w") as f:
        text = json.dumps(summary, indent=2)
        # Collapse the tiny leaf objects onto one line so a turn reads top-to-bottom.
        text = re.sub(
            r'\{\s*"prob":\s*"([^"]+)",\s*"valid":\s*(true|false)\s*\}',
            r'{"prob": "\1", "valid": \2}', text,
        )
        text = re.sub(
            r'\{\s*"species":\s*"([^"]+)",\s*"hp":\s*"([^"]+)"\s*\}',
            r'{"species": "\1", "hp": "\2"}', text,
        )
        text = re.sub(
            r'\{\s*"action":\s*"([^"]+)",\s*"hp_delta":\s*"([^"]+)"\s*\}',
            r'{"action": "\1", "hp_delta": "\2"}', text,
        )
        # Belief leaves: one species-guess per line ({"species": "tyranitar", "prob": "41.2%"}).
        text = re.sub(
            r'\{\s*"species":\s*"([^"]+)",\s*"prob":\s*"([^"]+)"\s*\}',
            r'{"species": "\1", "prob": "\2"}', text,
        )
        f.write(text)

    states = recorder.states_arrays()
    if states:
        np.savez_compressed(f"{out_prefix}_states.npz", **states)

    # Human-watchable replay alongside the forensic dump. `save_replay` is a poke-env
    # method (renders the accumulated protocol stream — `_replay_data`, populated on every
    # parsed line regardless of the fast/heavy path — into a standalone HTML page), NOT
    # battle state, so it's an allowed raw seam, same as stall.py. Guarded so a replay
    # failure never costs us the forensic trace we already wrote.
    try:
        battle.save_replay(f"{out_prefix}_replay.html")
    except Exception as e:
        sys.stderr.write(f"Failed to save replay HTML for {out_prefix}: {e}\n")
        sys.stderr.flush()
