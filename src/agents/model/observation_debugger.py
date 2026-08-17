"""Diagnostic trace collector for the Gen3 feature extractor.

Moved out of Gen3FeaturesExtractor to keep the forward path clean.
Only instantiated when log_level >= LogLevel.PERIODIC; None in production.
"""
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch

from utils.logging.rate_limiter import RateLimitedLogger

if TYPE_CHECKING:
    from agents.observation.state_encoder import Gen3ObservationEncoder


class ObservationDebugger:
    """Collects raw observation snapshots and prints decoded battle state periodically."""

    def __init__(self, mappings: dict, interval_seconds: int = 30):
        self._mappings = mappings
        self._encoder: Optional["Gen3ObservationEncoder"] = None  # lazy: only built when a print is triggered
        self._trace_buffer: List[torch.Tensor] = []
        self._trace_collect_remaining: int = 0
        self._logger = RateLimitedLogger(interval_seconds=interval_seconds)

    def _get_encoder(self) -> "Gen3ObservationEncoder":
        if self._encoder is None:
            from agents.observation.state_encoder import Gen3ObservationEncoder
            self._encoder = Gen3ObservationEncoder(self._mappings)
        return self._encoder

    def on_forward(self, obs_tensor: torch.Tensor) -> None:
        """Call from forward() with obs["observation"] — shape [B, obs_dim]."""
        if self._logger.should_log():
            self._trace_buffer = []
            self._trace_collect_remaining = 3
        if self._trace_collect_remaining > 0:
            self._trace_buffer.append(obs_tensor[0].detach().clone())
            self._trace_collect_remaining -= 1
            if self._trace_collect_remaining == 0:
                self._print_deep_trace()

    def _print_one_turn(self, obs_np: np.ndarray, label: str) -> None:
        """Print one turn's decoded state from a full obs array."""
        encoder = self._get_encoder()
        desc = encoder.describe_vector(obs_np)
        world = desc.get('world', {})
        NAME_W, ACTV_W, TYPE_W = 12, 6, 18
        TAB = "    "

        print(f"\n{'─' * 60}")
        print(f"  {label}   |  Turn: {world.get('turn', '?')} | Weather: {world.get('weather', 'NONE')} | Spikes: {world.get('our_spikes', 0)}/{world.get('opp_spikes', 0)}")
        print(f"{'─' * 60}")

        ctx = desc.get('our_active', {})
        print(f"Active ctx — Boosts: {ctx.get('boosts', {})} | Volatiles: {ctx.get('volatiles', [])}")

        print("\n--- TEAMS ---")
        for i, mon in enumerate(desc['our_team']):
            active_str = "[actv]" if mon.get('active') else "      "
            s = mon['stats']
            print(f"[OUR {i}] {mon['species'].lower():{NAME_W}} {active_str:{ACTV_W}}  {mon['types'].lower():{TYPE_W}}  hp: {mon['hp']:>6}  status: {mon['status'].lower():7}  {s['hp']}/{s['atk']}/{s['def']}/{s['spa']}/{s['spd']}/{s['spe']}")
            print(f"{TAB}item: {mon['item'].lower():17}  ably: {mon['ability'].lower():16}  moves: {mon.get('moves', [])}")
        print("-" * 30)
        for i, mon in enumerate(desc['opp_team']):
            active_str = "[actv]" if mon.get('active') else "      "
            s = mon['stats']
            print(f"[OPP {i}] {mon['species'].lower():{NAME_W}} {active_str:{ACTV_W}}  {mon['types'].lower():{TYPE_W}}  hp: {mon['hp']:>6}  status: {mon['status'].lower():7}  {s['hp']}/{s['atk']}/{s['def']}/{s['spa']}/{s['spd']}/{s['spe']}")
            print(f"{TAB}item: {mon['item'].lower():17}  ably: {mon['ability'].lower():16}  moves: {mon.get('moves', [])}")

        momentum = desc.get('momentum', {})
        print(f"\nFainted: {momentum.get('fainted_our', 0)} (Us) / {momentum.get('fainted_opp', 0)} (Them)")

        td = desc.get("turn_delta")
        if td is not None:
            def _action_str(switched: Any, failed: Any, cant: Any,
                            move: Optional[Dict[str, Any]]) -> str:
                move_str = None
                if move and move.get("move_id", 0) > 0:
                    name = move.get("move_name") or f"#{move['move_id']}"
                    type_ = (move.get("move_type") or "").title()
                    pwr = move["power"]
                    meta = []
                    if type_: meta.append(type_)
                    if pwr > 0: meta.append(f"{pwr}bp")
                    if move["secondary"]: meta.append("+eff")
                    if move["recoil"]: meta.append("recoil")
                    suffix = f" [{', '.join(meta)}]" if meta else ""
                    move_str = f"{name}{suffix}"
                if switched:
                    return f"{move_str} → phazed" if move_str else "switch"
                if failed:
                    return f"✗ {cant or '?'}"
                if move_str:
                    return move_str
                return "(first turn)"

            W = 36
            opp_known = "" if td["opp_move_known"] else "  [unconfirmed]"
            faint_parts = (["us"] if td["we_fainted"] else []) + (["opp"] if td["opp_fainted"] else [])
            faint_str = f"   💀 fainted: {'/'.join(faint_parts)}" if faint_parts else ""
            print("--- Last Turn ---")
            print(f"  Us:  {_action_str(td['our_switched'], td['our_failed'], td['our_cant'], td['our_move']):{W}}")
            print(f"  Opp: {_action_str(td['opp_switched'], td['opp_failed'], td['opp_cant'], td['opp_move']) + opp_known:{W}}")
            print(f"  ΔHP  us={td['our_hp_delta']:+.2f}  opp={td['opp_hp_delta']:+.2f}{faint_str}")

        warnings, is_critical = encoder.integrity_check(obs_np)
        if warnings:
            print("\n⚠️ [INTEGRITY CHECK WARNINGS]")
            for w in warnings:
                print(f"  - {w}")
        if is_critical:
            raise ValueError(f"CRITICAL INTEGRITY FAILURE: {warnings}")

    def _print_deep_trace(self) -> None:
        """Print the last buffered turns as a succession."""
        if not self._trace_buffer:
            return
        n = len(self._trace_buffer)
        print("\n" + "🧬" * 30)
        print(f"🧬 [DEEP TRACE — {n} turns ending at {time.strftime('%H:%M:%S')}]")
        print("=" * 60)
        labels = ["turn -2 (oldest)", "turn -1", "turn 0 (current)"][-n:]
        for obs_t, label in zip(self._trace_buffer, labels):
            self._print_one_turn(obs_t.cpu().numpy(), label)
        print("=" * 60 + "\n")
