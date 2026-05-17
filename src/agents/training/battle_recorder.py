from typing import Callable, Optional
import numpy as np

from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.reward_function import RewardFunction
from agents.training.slot_registry import SlotRegistry
from agents.training.reward_manager import Gen3RewardManager


class BattleRecorder:
    """
    Records every model invocation for a single battle and exports a
    human-readable summary JSON.

    Call record() once per choose_move(), finalize() at battle end,
    then to_summary() to get the exportable dict.

    Each invocation entry is self-contained — action labels are inlined,
    not referenced from a legend — so a human can read any turn in isolation.
    """

    def __init__(self, battle_tag: str, reward_fn_factory: Callable[[], RewardFunction] = Gen3RewardManager):
        self.battle_tag = battle_tag
        self._our_slots = SlotRegistry()
        self._opp_slots = SlotRegistry()
        self._invocations: list[dict] = []
        self._pending_ctx: Optional[BattleContext] = None
        self._pending_action: int = -1
        self._pending_entry: Optional[dict] = None
        self._reward_fn = reward_fn_factory()
        self._reward_fn.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, battle, action_idx: int, probs: np.ndarray, mask: np.ndarray) -> None:
        """Record a model invocation. Call from choose_move() after prediction."""
        ctx = self._build_ctx(battle, mask)

        if self._pending_entry is not None:
            self._complete_pending(ctx, battle)

        chosen = self._action_label(action_idx, battle)
        entry = {
            "i": len(self._invocations) + 1,
            "turn": ctx.turn,
            "phase": ctx.phase,
            "chosen": chosen,
            "our": {"species": ctx.our_active, "hp": self._our_hp_pct(ctx), "bench": self._our_bench_summary(battle)},
            "opp": {"species": ctx.opp_active, "hp": self._opp_hp_pct(ctx), "bench": self._opp_bench_summary(battle)},
            "outcome": None,
            "actions": self._all_action_labels(battle, probs, mask),
        }

        self._pending_ctx = ctx
        self._pending_action = action_idx
        self._pending_entry = entry

    def finalize(self, battle) -> None:
        """Complete the last pending invocation at battle end."""
        if self._pending_entry is None:
            return

        prev_ctx = self._pending_ctx
        our_hp, opp_hp = self._terminal_hp(battle)

        terminal_ctx = BattleContext.from_battle(
            battle, np.zeros(11, dtype=np.float32), np.zeros(1, dtype=np.float32),
            self._our_slots, self._opp_slots,
        )
        delta = TurnDelta.build(prev_ctx, terminal_ctx, self._pending_action)
        self._reward_fn.record_action(prev_ctx, self._pending_action)
        reward = self._reward_fn.process_turn_reward(battle, delta)

        # When a switch occurred, damage landed on the incoming mon — track that slot.
        our_slot = prev_ctx.our_slot_map.get(delta.our_switch_to or prev_ctx.our_active, 0)
        opp_slot = prev_ctx.opp_slot_map.get(delta.opp_switch_to or prev_ctx.opp_active, 0)
        our_delta = (our_hp[our_slot] - prev_ctx.our_hp[our_slot]) * 100
        opp_delta = (opp_hp[opp_slot] - prev_ctx.opp_hp[opp_slot]) * 100

        events = []
        final_our_fainted = sum(1 for m in battle.team.values() if m.fainted)
        final_opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
        if final_our_fainted > prev_ctx.our_fainted_count:
            events.append(f"our:{prev_ctx.our_active}:fainted")
        if final_opp_fainted > prev_ctx.opp_fainted_count:
            events.append(f"opp:{prev_ctx.opp_active}:fainted")

        if battle.won:
            events.append("result:win")
        elif battle.lost:
            events.append("result:loss")
        else:
            events.append("result:tie")

        breakdown = getattr(self._reward_fn, "_last_breakdown", None)
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
        return BattleContext.from_battle(
            battle, mask, np.zeros(1, dtype=np.float32), self._our_slots, self._opp_slots
        )

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

    def _our_bench_summary(self, battle) -> str:
        active = battle.active_pokemon
        parts = []
        for mon in battle.team.values():
            if active and mon.species == active.species:
                continue
            if mon.fainted:
                parts.append(f"{mon.species}(faint)")
            else:
                parts.append(f"{mon.species}({mon.current_hp_fraction * 100:.0f}%)")
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
                pct = mon.current_hp_fraction * 100
                parts.append(f"{mon.species}({pct:.0f}%)")
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

    def _complete_pending(self, curr_ctx: BattleContext, battle) -> None:
        prev_ctx = self._pending_ctx
        self._reward_fn.record_action(prev_ctx, self._pending_action)
        delta = TurnDelta.build(prev_ctx, curr_ctx, self._pending_action)
        reward = self._reward_fn.process_turn_reward(battle, delta)

        # Our action
        if delta.our_switch_to:
            we_action = f"switched_to:{delta.our_switch_to}"
        elif prev_ctx.phase == "forced_switch" and curr_ctx.our_active not in ("NONE", prev_ctx.our_active):
            we_action = f"forced_switch_to:{curr_ctx.our_active}"
        else:
            we_action = self._pending_entry["chosen"]

        # Their action — prefer TurnDelta switch detection, then poke-env last_move
        if prev_ctx.phase == "forced_switch":
            # Opponent doesn't act on forced-switch turns — we're just picking a replacement
            they_action = "none"
        elif delta.opp_switch_to:
            if delta.opp_move_id:
                # Phaze (Roar/Whirlwind): they moved first, then were forced out
                they_action = f"{delta.opp_move_id} → phazed_to:{delta.opp_switch_to}"
            else:
                they_action = f"switched_to:{delta.opp_switch_to}"
        else:
            opp_mon = battle.opponent_active_pokemon
            last_move = getattr(opp_mon, "last_move", None) if opp_mon else None
            they_action = last_move.id if (last_move and hasattr(last_move, "id")) else "unknown"

        # When a switch occurred, damage landed on the incoming mon — track that slot.
        our_slot = prev_ctx.our_slot_map.get(delta.our_switch_to or prev_ctx.our_active, 0)
        opp_slot = prev_ctx.opp_slot_map.get(delta.opp_switch_to or prev_ctx.opp_active, 0)
        our_delta = delta.our_hp_delta[our_slot] * 100
        opp_delta = delta.opp_hp_delta[opp_slot] * 100

        events = []
        if delta.we_fainted:
            events.append(f"our:{prev_ctx.our_active}:fainted")
        if delta.opp_fainted:
            events.append(f"opp:{prev_ctx.opp_active}:fainted")

        breakdown = getattr(self._reward_fn, "_last_breakdown", None)
        self._pending_entry["outcome"] = {
            "our": {"action": we_action,   "hp_delta": f"{our_delta:+.0f}%"},
            "opp": {"action": they_action, "hp_delta": f"{opp_delta:+.0f}%"},
            "reward": breakdown.to_dict() if breakdown is not None else round(reward, 3),
            "events": events,
        }
        self._invocations.append(self._pending_entry)
        self._pending_entry = None
        self._pending_ctx = None

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
