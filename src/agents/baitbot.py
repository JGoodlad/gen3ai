"""BaitBot — a scripted opponent that PIVOTS INTO IMMUNITY on a dial.

Why it exists (bait_programme_habit_verdict.json, 2026-08-21): the model's bait failure is
**exploration starvation at a saturated action** — it fires an attack the arrival cannot take at
p~0.97, so the alternatives at p~0.01-0.03 are never sampled and their advantage is never realized.
Four instruments agree nothing is missing from its INFORMATION (alpha/beta know the switch, the
critic already ranks an alternative above the whiff in 21/23 loop decisions). The one signal
starvation cannot seal off is **how often the habit is punished**, and only a scripted opponent puts
that frequency on a controlled dial — a gate needs a controlled variable.

The predicate is GROUND TRUTH, not a table: `effective_multiplier_by_types` already resolves the
gen-3 ability immunities (Levitate / Water Absorb / Volt Absorb / Flash Fire) and the type chart
from `data/`, verified 2026-08-22 including the negative control (Charizard WITHOUT Flash Fire
reads 0.5x, not 0). A hand-copied immunity table would drift from `data/`; this cannot.
"""
from __future__ import annotations

import random
from typing import Optional

from poke_env.battle.battle import Battle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from agents.gen3_mechanics import effective_multiplier
from agents.opponents import _best_damage_move_v2, _best_switch_v2

#: Default pivot probability. ~0.6 is the training dial; the gate's held-out generalization read
#: uses a DIFFERENT value, so the arm cannot pass by memorising this one.
DEFAULT_P_BAIT = 0.6


def blocks(move: Move, defender: Pokemon) -> bool:
    """True iff `defender` takes ZERO damage from `move` — type chart OR ability.

    Status moves are NOT baits: the pathology is firing an ATTACK into an arrival that cannot take
    it, and a status move landing on an immune switch-in is a different (and much cheaper) error.
    """
    if move.base_power is None or move.base_power <= 0:
        return False
    return effective_multiplier(move.type, defender) == 0.0


def known_attacks(mon: Optional[Pokemon]) -> list[Move]:
    """The damaging moves we have actually SEEN this mon use.

    Deliberately not its full movepool: BaitBot must bait on revealed information like any
    opponent, or the gate would be measuring an oracle rather than a scripted strategy.
    """
    if mon is None:
        return []
    return [m for m in mon.moves.values() if m.base_power and m.base_power > 0]


def bait_targets(battle: Battle) -> list[Pokemon]:
    """Alive BENCH mons that are immune to EVERY revealed attack of the opposing active.

    Empty when the opponent has revealed no attacks — with nothing known, nothing is baitable, and
    guessing would make the trigger rate depend on the movepool rather than on the dial.
    """
    attacks = known_attacks(battle.opponent_active_pokemon)
    if not attacks:
        return []
    return [s for s in battle.available_switches
            if not s.fainted and all(blocks(m, s) for m in attacks)]


class Gen3BaitBotPlayer(Player):
    """Pivots into an immune bench mon with probability ``p_bait``; else plays the v2 heuristic.

    ``p_bait`` is the controlled variable. ``seed`` makes a run reproducible for tests; production
    leaves it None so the arm sees independent draws.
    """

    def __init__(self, *args, p_bait: float = DEFAULT_P_BAIT, seed: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= p_bait <= 1.0:
            raise ValueError(f"p_bait must be in [0, 1], got {p_bait}")
        self.p_bait = p_bait
        self._rng = random.Random(seed)
        #: Instrument counters — the realized rate is what validates the dial against the loops
        #: detector; a bot whose declared p_bait and realized rate disagree is not a controlled
        #: variable, it is a story.
        self.n_bait_opportunities = 0
        self.n_baits_taken = 0

    def choose_move(self, battle: Battle) -> BattleOrder:
        if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
            return self.choose_random_move(battle)

        # A forced switch is not a bait DECISION — no opportunity is counted.
        if not battle.available_moves and battle.available_switches:
            best = _best_switch_v2(battle)
            return Player.create_order(best or self._rng.choice(battle.available_switches))

        targets = bait_targets(battle)
        if targets and battle.available_switches:
            self.n_bait_opportunities += 1
            if self._rng.random() < self.p_bait:
                self.n_baits_taken += 1
                # Prefer the healthiest immune wall — a fainted-next-turn bait teaches nothing.
                return Player.create_order(max(targets, key=lambda s: s.current_hp_fraction))

        best = _best_damage_move_v2(battle)
        if best is not None:
            return Player.create_order(best)
        if battle.available_moves:
            return Player.create_order(max(battle.available_moves, key=lambda m: m.base_power or 0))
        return self.choose_random_move(battle)

    @property
    def realized_bait_rate(self) -> float:
        """Baits taken / opportunities — must track ``p_bait`` or the dial is not a dial."""
        return self.n_baits_taken / self.n_bait_opportunities if self.n_bait_opportunities else 0.0
