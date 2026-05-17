"""
Gen 3 OU heuristic opponents for training diversity.

Three distinct playstyles to complement RandomPlayer and SimpleHeuristicsPlayer:
  - Gen3StallerPlayer:      Status infliction → recovery → hazard clear → damage
  - Gen3AggressivePlayer:   Maximum damage every turn, never switches voluntarily
  - Gen3SetupSweepPlayer:   Accumulate setup boosts when winning, then attack
"""
import random

from poke_env.battle.battle import Battle
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from poke_env.player.battle_order import BattleOrder
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.player.player import Player
from agents.type_utils import effective_multiplier


# ---------------------------------------------------------------------------
# Helpers shared across players
# ---------------------------------------------------------------------------

def _effective_damage_score(move: Move, active: Pokemon, opponent: Pokemon) -> float:
    """Weighted damage score: base_power × STAB × type effectiveness × accuracy."""
    if move.base_power == 0:
        return 0.0
    stab = 1.5 if move.type in active.types else 1.0
    eff = effective_multiplier(move.type, opponent)
    return move.base_power * stab * eff * move.accuracy


def _best_damage_move(battle: Battle):
    """Return the move with the highest effective damage score, or None."""
    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if not battle.available_moves or active is None or opponent is None:
        return None
    return max(battle.available_moves, key=lambda m: _effective_damage_score(m, active, opponent))


def _best_switch(battle: Battle):
    """Return the switch-in with the best type matchup, or None."""
    opponent = battle.opponent_active_pokemon
    if not battle.available_switches or opponent is None:
        return None
    return max(
        battle.available_switches,
        key=lambda s: SimpleHeuristicsPlayer._estimate_matchup(s, opponent),
    )


# ---------------------------------------------------------------------------
# Gen3StallerPlayer
# ---------------------------------------------------------------------------

_STATUS_MOVES = {"toxic", "willowisp", "thunderwave", "stunspore", "sleeppowder", "spore", "glare", "toxic"}
_RECOVERY_MOVES = {"recover", "softboiled", "moonlight", "morningsun", "synthesis", "rest", "wish", "slackoff", "milkdrink"}
_HAZARD_CLEAR = {"rapidspin"}
_PROTECT_MOVES = {"protect", "detect"}

_RECOVERY_HP_THRESHOLD = 0.50
_PROTECT_PROBABILITY = 0.6  # use protect ~60% of turns when opponent is toxiced


class Gen3StallerPlayer(Player):
    """
    Status-oriented stall player.

    Priority each turn:
      1. Inflict a status condition on an unstatused opponent.
      2. Stall with Protect/Detect ~60% of turns when opponent is toxiced.
      3. Use recovery if our active mon is below the HP threshold.
      4. Clear entry hazards with Rapid Spin.
      5. Attack with the most effective damage move.
      6. Switch to the best available matchup.
    """

    def choose_move(self, battle: Battle) -> BattleOrder:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        if battle.available_moves:
            # 1. Status if opponent is clean
            if opponent.status is None:
                for move in battle.available_moves:
                    if move.id in _STATUS_MOVES and move.base_power == 0:
                        return Player.create_order(move)
                    # Secondary-effect status (e.g. Body Slam paralysis) — skip,
                    # prefer a dedicated status move.

            # 2. Protect to rack up toxic damage
            if opponent.status == Status.TOX and random.random() < _PROTECT_PROBABILITY:
                for move in battle.available_moves:
                    if move.id in _PROTECT_MOVES:
                        return Player.create_order(move)

            # 3. Recovery when low
            if active.current_hp_fraction < _RECOVERY_HP_THRESHOLD:
                for move in battle.available_moves:
                    if move.id in _RECOVERY_MOVES:
                        return Player.create_order(move)

            # 4. Hazard clear
            if battle.side_conditions:
                for move in battle.available_moves:
                    if move.id in _HAZARD_CLEAR:
                        return Player.create_order(move)

            # 4. Best damage move
            best = _best_damage_move(battle)
            if best is not None:
                return Player.create_order(best)

        # 5. Switch
        switch = _best_switch(battle)
        if switch is not None:
            return Player.create_order(switch)

        return self.choose_random_move(battle)


# ---------------------------------------------------------------------------
# Gen3AggressivePlayer
# ---------------------------------------------------------------------------

class Gen3AggressivePlayer(Player):
    """
    Pure-offense player that maximises damage every turn.

    - Always picks the highest effective-damage move.
    - Never switches voluntarily (only switches on forced switches).
    - Ignores status, setup, and utility moves entirely.
    """

    def choose_move(self, battle: Battle) -> BattleOrder:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        if battle.available_moves:
            damaging = [m for m in battle.available_moves if m.base_power > 0]
            if damaging:
                best = max(damaging, key=lambda m: _effective_damage_score(m, active, opponent))
                return Player.create_order(best)
            # All available moves are non-damaging — pick the highest base_power
            # (even if 0) to avoid forfeit.
            best = max(battle.available_moves, key=lambda m: m.base_power)
            return Player.create_order(best)

        # Forced switch — pick the hardest hitter by Attack/SpA
        if battle.available_switches:
            best_switch = max(
                battle.available_switches,
                key=lambda s: max(s.base_stats.get("atk", 0), s.base_stats.get("spa", 0)),
            )
            return Player.create_order(best_switch)

        return self.choose_random_move(battle)


# ---------------------------------------------------------------------------
# Gen3SetupSweepPlayer
# ---------------------------------------------------------------------------

_SETUP_MOVES = {
    "swordsdance", "calmmind", "dragondance", "nastyplot",
    "bulkup", "curse", "meditate", "sharpen",
}
_SETUP_STATS = {"atk", "spa", "spe"}  # offensive boosts worth stacking
_SETUP_HP_THRESHOLD = 0.75
_SETUP_BOOST_CAP = 4   # stop boosting once we hit this total offensive boost


class Gen3SetupSweepPlayer(Player):
    """
    Setup-then-sweep player.

    Priority each turn:
      1. Use a setup move if: healthy, winning the matchup, and not yet at cap.
      2. Attack with the most effective damage move.
      3. Switch to a better matchup if the current one is clearly losing.
    """

    def choose_move(self, battle: Battle) -> BattleOrder:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        matchup = SimpleHeuristicsPlayer._estimate_matchup(active, opponent)
        total_offensive_boosts = sum(
            active.boosts.get(s, 0) for s in _SETUP_STATS
        )

        # Switch out first if the matchup is clearly losing and a better switch exists.
        should_switch = (
            matchup < SimpleHeuristicsPlayer.SWITCH_OUT_MATCHUP_THRESHOLD
            and bool(battle.available_switches)
        )

        if battle.available_moves and not should_switch:
            # 1. Setup when healthy, winning, and not capped
            if (
                active.current_hp_fraction >= _SETUP_HP_THRESHOLD
                and matchup > 0
                and total_offensive_boosts < _SETUP_BOOST_CAP
            ):
                for move in battle.available_moves:
                    if move.id in _SETUP_MOVES and move.target == "self":
                        boosted_stat_not_capped = any(
                            active.boosts.get(s, 0) < 6
                            for s, v in (move.boosts or {}).items()
                            if v > 0 and s in _SETUP_STATS
                        )
                        if boosted_stat_not_capped:
                            return Player.create_order(move)

            # 2. Attack
            best = _best_damage_move(battle)
            if best is not None:
                return Player.create_order(best)

        # 3. Switch to a better matchup
        if battle.available_switches:
            switch = _best_switch(battle)
            if switch is not None:
                return Player.create_order(switch)

        if battle.available_moves:
            return Player.create_order(battle.available_moves[0])

        return self.choose_random_move(battle)
