from typing import Callable, Optional
import numpy as np

from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.reward_function import RewardFunction
from agents.training.reward_tracker import RewardTracker
from agents.training.slot_registry import SlotRegistry
from agents.training.reward_manager import Gen3RewardManager
from agents.gen3_mechanics import boosts_str as _boosts_str_fn
from poke_env.battle.status import Status
from poke_env.battle.effect import Effect


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

    def __init__(self, battle_tag: str, reward_fn_factory: Callable[[], RewardFunction] = Gen3RewardManager):
        self.battle_tag = battle_tag
        self._our_slots = SlotRegistry()
        self._opp_slots = SlotRegistry()
        self._invocations: list[dict] = []
        self._pending_entry: Optional[dict] = None
        self._tracker = RewardTracker(reward_fn_factory, self._our_slots, self._opp_slots)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, battle, action_idx: int, probs: np.ndarray, mask: np.ndarray) -> None:
        """Record a model invocation. Call from choose_move() after prediction."""
        curr_ctx = self._build_ctx(battle, mask)

        if self._pending_entry is not None:
            prev_ctx = self._tracker.pending_ctx
            delta, reward = self._tracker.complete_pending(curr_ctx, battle)
            self._fill_pending_outcome(prev_ctx, curr_ctx, delta, reward, battle)

        chosen = self._action_label(action_idx, battle)
        our_mon = battle.active_pokemon
        opp_mon = battle.opponent_active_pokemon
        our_status = self._mon_display_status(our_mon)
        opp_status = self._mon_display_status(opp_mon)
        our_boosts = _boosts_str_fn(our_mon)
        opp_boosts = _boosts_str_fn(opp_mon)

        our_section: dict = {"species": curr_ctx.our_active, "hp": self._our_hp_pct(curr_ctx)}
        if our_status:
            our_section["status"] = our_status
        if our_boosts:
            our_section["boosts"] = our_boosts
        our_section["bench"] = self._our_bench_summary(battle)

        opp_section: dict = {"species": curr_ctx.opp_active, "hp": self._opp_hp_pct(curr_ctx)}
        if opp_status:
            opp_section["status"] = opp_status
        if opp_boosts:
            opp_section["boosts"] = opp_boosts
        opp_section["bench"] = self._opp_bench_summary(battle)

        entry = {
            "i": len(self._invocations) + 1,
            "turn": curr_ctx.turn,
            "phase": curr_ctx.phase,
            "chosen": chosen,
            "our": our_section,
            "opp": opp_section,
            "outcome": None,
            "actions": self._all_action_labels(battle, probs, mask),
        }

        self._tracker.begin_turn(curr_ctx, action_idx)
        self._pending_entry = entry

    def finalize(self, battle) -> None:
        """Complete the last pending invocation at battle end."""
        if self._pending_entry is None:
            return

        prev_ctx = self._tracker.pending_ctx
        our_hp, opp_hp = self._terminal_hp(battle)
        terminal_ctx, delta, reward = self._tracker.finalize(battle)

        # HP delta: for faint turns use the fainted mon's slot, not the forced switch-in.
        our_ref = prev_ctx.our_active if delta.we_fainted else (delta.our_switch_to or prev_ctx.our_active)
        opp_ref = prev_ctx.opp_active if delta.opp_fainted else (delta.opp_switch_to or prev_ctx.opp_active)
        our_slot = prev_ctx.our_slot_map.get(our_ref, 0)
        opp_slot = prev_ctx.opp_slot_map.get(opp_ref, 0)
        our_delta = (our_hp[our_slot] - prev_ctx.our_hp[our_slot]) * 100
        opp_delta = (opp_hp[opp_slot] - prev_ctx.opp_hp[opp_slot]) * 100

        events = []
        final_our_fainted = sum(1 for m in battle.team.values() if m.fainted)
        final_opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
        if final_our_fainted > prev_ctx.our_fainted_count:
            events.append(f"our:{prev_ctx.our_active}:fainted")
        if final_opp_fainted > prev_ctx.opp_fainted_count:
            events.append(f"opp:{prev_ctx.opp_active}:fainted")

        self._append_status_events(events, prev_ctx, delta, battle)

        if battle.won:
            events.append("result:win")
        elif battle.lost:
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
        if battle.won:
            result = "WIN"
        elif battle.lost:
            result = "LOSS"
        else:
            result = "TIE"

        return {
            "meta": {
                "step": step,
                "battle_id": self.battle_tag,
                "result": result,
                "turns": battle.turn,
                "invocations": len(self._invocations),
            },
            "teams": {
                "ours": [
                    {
                        "species": m.species,
                        "item": m.item or "none",
                        "final_hp": f"{m.current_hp_fraction * 100:.0f}%",
                        "fainted": m.fainted,
                    }
                    for m in battle.team.values()
                ],
                "opponent": [
                    {
                        "species": m.species,
                        "final_hp": f"{m.current_hp_fraction * 100:.0f}%",
                        "fainted": m.fainted,
                    }
                    for m in battle.opponent_team.values()
                ],
            },
            "invocations": self._invocations,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ctx(self, battle, mask: np.ndarray) -> BattleContext:
        return BattleContext.from_battle(battle, mask, self._our_slots, self._opp_slots)

    def _latched(self, battle) -> dict:
        """Return the decision context latched by Gen3ActionMasker, or empty dict."""
        return getattr(battle, "_gen3_decision_context", {})

    def _action_label(self, action_idx: int, battle) -> str:
        ctx = self._latched(battle)
        team_list = ctx.get("team_objects", list(battle.team.values()))
        move_ids = ctx.get("move_ids", [])

        if action_idx < 6:
            return f"switch:{team_list[action_idx].species}" if action_idx < len(team_list) else f"switch:slot{action_idx}"
        elif action_idx < 10:
            m = action_idx - 6
            return move_ids[m] if m < len(move_ids) else f"move{m}"
        return "struggle"

    def _all_action_labels(self, battle, probs: np.ndarray, mask: np.ndarray) -> dict:
        ctx = self._latched(battle)
        team_list = ctx.get("team_objects", list(battle.team.values()))
        move_ids = ctx.get("move_ids", [])

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
    def _mon_display_status(mon) -> str | None:
        """Rich status string including counters and volatiles.

        Examples: "SLP(3)", "TOX(5)", "BRN", "PAR|TAUNT", "PERISH(2)|CONF"
        """
        if mon is None:
            return None
        parts = []
        status = getattr(mon, "status", None)
        ctr = getattr(mon, "status_counter", 0) or 0
        effects = getattr(mon, "effects", {})
        if status == Status.SLP:
            parts.append(f"SLP({ctr})" if ctr else "SLP")
        elif status == Status.TOX:
            parts.append(f"TOX({ctr})" if ctr else "TOX")
        elif status is not None:
            name_map = {Status.BRN: "BRN", Status.PAR: "PAR", Status.FRZ: "FRZ", Status.PSN: "PSN"}
            if name := name_map.get(status):
                parts.append(name)
        effect_names = {
            Effect.TAUNT: "TAUNT", Effect.CONFUSION: "CONF", Effect.ENCORE: "ENCORE",
            Effect.ATTRACT: "ATTRACT", Effect.DISABLE: "DISABLE", Effect.SUBSTITUTE: "SUB",
        }
        for eff, name in effect_names.items():
            if eff in effects:
                parts.append(name)
        for n, e in [(3, Effect.PERISH3), (2, Effect.PERISH2), (1, Effect.PERISH1), (0, Effect.PERISH0)]:
            if e in effects:
                parts.append(f"PERISH({n})")
                break
        return "|".join(parts) if parts else None

    @staticmethod
    def _status_key(status_str: str | None) -> str | None:
        """Normalize for change detection — strips counter values, sorts parts."""
        if not status_str:
            return None
        return "|".join(sorted(p.split("(")[0] for p in status_str.split("|")))

    def _our_bench_summary(self, battle) -> str:
        active = battle.active_pokemon
        parts = []
        for mon in battle.team.values():
            if active and mon.species == active.species:
                continue
            if mon.fainted:
                parts.append(f"{mon.species}(faint)")
            else:
                pct = f"{mon.current_hp_fraction * 100:.0f}%"
                status = self._mon_display_status(mon)
                parts.append(f"{mon.species}({pct},{status})" if status else f"{mon.species}({pct})")
        return ", ".join(parts)

    def _opp_bench_summary(self, battle) -> str:
        active = battle.opponent_active_pokemon
        parts = []
        for mon in battle.opponent_team.values():
            if active and mon.species == active.species:
                continue
            if mon.fainted:
                parts.append(f"{mon.species}(faint)")
            else:
                pct = f"{mon.current_hp_fraction * 100:.0f}%"
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
                              delta: TurnDelta, battle) -> None:
        """Append events for any status conditions newly applied this turn."""
        prev_our_status = self._pending_entry["our"].get("status")
        prev_opp_status = self._pending_entry["opp"].get("status")

        # Only track the mon that was active at decision time; skip if they fainted.
        if not delta.we_fainted:
            our_mon = next(
                (m for m in battle.team.values() if m.species == prev_ctx.our_active), None
            )
            if our_mon:
                new_status = self._mon_display_status(our_mon)
                if self._status_key(new_status) != self._status_key(prev_our_status):
                    if new_status:
                        events.append(f"our:{prev_ctx.our_active}:{new_status}")

        if not delta.opp_fainted:
            opp_mon = next(
                (m for m in battle.opponent_team.values() if m.species == prev_ctx.opp_active), None
            )
            if opp_mon:
                new_status = self._mon_display_status(opp_mon)
                if self._status_key(new_status) != self._status_key(prev_opp_status):
                    if new_status:
                        events.append(f"opp:{prev_ctx.opp_active}:{new_status}")

    def _fill_pending_outcome(self, prev_ctx: BattleContext, curr_ctx: BattleContext,
                              delta: TurnDelta, reward: float, battle) -> None:
        """Build and commit the JSON outcome for the pending entry using already-computed delta/reward."""
        # Our action
        if delta.our_switch_to:
            we_action = f"switched_to:{delta.our_switch_to}"
        elif prev_ctx.phase == "forced_switch" and curr_ctx.our_active not in ("NONE", prev_ctx.our_active):
            we_action = f"forced_switch_to:{curr_ctx.our_active}"
        else:
            we_action = self._pending_entry["chosen"]

        # Their action — distinguish: voluntary switch / faint+forced-switch / phaze / moved
        if prev_ctx.phase == "forced_switch":
            they_action = "none"
        elif delta.opp_switch_to:
            if delta.opp_fainted and delta.opp_move_id:
                they_action = f"{delta.opp_move_id} → {delta.opp_switch_to}_sent_in"
            elif delta.opp_fainted:
                they_action = f"{delta.opp_switch_to}_sent_in"
            elif delta.opp_move_id:
                they_action = f"{delta.opp_move_id} → phazed_to:{delta.opp_switch_to}"
            else:
                they_action = f"switched_to:{delta.opp_switch_to}"
        else:
            opp_mon = battle.opponent_active_pokemon
            last_move = getattr(opp_mon, "last_move", None) if opp_mon else None
            they_action = last_move.id if (last_move and hasattr(last_move, "id")) else "unknown"

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

        self._append_status_events(events, prev_ctx, delta, battle)

        breakdown = getattr(self._tracker._reward_fn, "_last_breakdown", None)
        self._pending_entry["outcome"] = {
            "our": {"action": we_action,   "hp_delta": f"{our_delta:+.0f}%"},
            "opp": {"action": they_action, "hp_delta": f"{opp_delta:+.0f}%"},
            "reward": breakdown.to_dict() if breakdown is not None else round(reward, 3),
            "events": events,
        }
        self._invocations.append(self._pending_entry)
        self._pending_entry = None

    def _terminal_hp(self, battle) -> tuple[np.ndarray, np.ndarray]:
        our_hp = np.zeros(6, dtype=np.float32)
        for mon in battle.team.values():
            slot = self._our_slots.get(mon.species)
            if slot is not None:
                our_hp[slot] = mon.current_hp_fraction
        opp_hp = np.zeros(6, dtype=np.float32)
        for mon in battle.opponent_team.values():
            slot = self._opp_slots.get(mon.species)
            if slot is not None:
                opp_hp[slot] = mon.current_hp_fraction
        return our_hp, opp_hp
