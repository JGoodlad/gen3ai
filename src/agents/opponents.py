"""
Gen 3 OU heuristic opponents for training diversity.

Three distinct playstyles to complement RandomPlayer and SimpleHeuristicsPlayer:
  - Gen3StallerPlayer:      Status infliction → recovery → hazard clear → damage
  - Gen3AggressivePlayer:   Maximum damage every turn, never switches voluntarily
  - Gen3SetupSweepPlayer:   Accumulate setup boosts when winning, then attack

V2 variants keep each identity but play slightly sharper (physical/special-aware
damage, status-immunity checks, no Protect spam, wall-escape switching, speed-aware
setup, boost preservation):
  - Gen3StallerV2Player, Gen3AggressiveV2Player, Gen3SetupSweepV2Player

Plus one smarter all-around generalist that improves on SimpleHeuristicsPlayer with
KO awareness, Gen-3-ability-aware damage, and opportunistic status/recovery:
  - Gen3HeuristicV2Player  (display name "Heuristic2")

NOTE: the V2 classes are defined here but not yet wired into the training/eval
rotation (OPPONENT_CLASSES / eval lists in train_rl_agent.py). Their display names
are registered in eval_callback._OPPONENT_NAMES so the TUI/TensorBoard label them
correctly once they are added.
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
from agents.gen3_mechanics import (
    effective_multiplier,
    is_status_move_immune,
    STATUS_MOVES as _STATUS_MOVES,
    RECOVERY_MOVES as _RECOVERY_MOVES,
    HAZARD_CLEAR_MOVES as _HAZARD_CLEAR,
    INVULNERABLE_MOVES as _PROTECT_MOVES,
    SETUP_MOVES as _SETUP_MOVES,
)


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

_RECOVERY_HP_THRESHOLD = 0.50
_PROTECT_PROBABILITY = 0.6  # use protect ~60% of turns when opponent is toxiced


# ---------------------------------------------------------------------------
# V2 helpers — physical/special-aware damage scoring (shared by all V2 bots)
# ---------------------------------------------------------------------------

def _damage_ratios(active: Pokemon, opponent: Pokemon) -> tuple[float, float]:
    """(physical_ratio, special_ratio): our offensive stat over their defensive stat.

    Reuses SimpleHeuristicsPlayer._stat_estimation (boost-aware, fixed-IV) so the
    score reflects whether *this* attacker hits harder physically or specially —
    something the V1 `_effective_damage_score` ignores entirely.
    """
    se = SimpleHeuristicsPlayer._stat_estimation
    physical_ratio = se(active, "atk") / se(opponent, "def")
    special_ratio = se(active, "spa") / se(opponent, "spd")
    return physical_ratio, special_ratio


def _damage_score_v2(
    move: Move,
    active: Pokemon,
    opponent: Pokemon,
    physical_ratio: float,
    special_ratio: float,
) -> float:
    """Damage score: base_power × STAB × stat-ratio × accuracy × expected_hits × type.

    Improves on V1's `_effective_damage_score` by (a) weighting the physical vs
    special stat ratio, (b) counting multi-hit moves via `expected_hits`, and
    (c) using `effective_multiplier` so Gen-3 ability immunities (Levitate, Volt/
    Water Absorb, Flash Fire, Wonder Guard) are respected. Status / 0-power moves
    score 0 so they never win a damage comparison.
    """
    if move.base_power == 0:
        return 0.0
    stab = 1.5 if move.type in active.types else 1.0
    ratio = physical_ratio if move.category == MoveCategory.PHYSICAL else special_ratio
    eff = effective_multiplier(move.type, opponent)
    return move.base_power * stab * ratio * move.accuracy * move.expected_hits * eff


def _best_damage_move_v2(battle: Battle):
    """Highest `_damage_score_v2` move, or None when no moves/active/opponent."""
    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon
    if not battle.available_moves or active is None or opponent is None:
        return None
    pr, sr = _damage_ratios(active, opponent)
    return max(
        battle.available_moves,
        key=lambda m: _damage_score_v2(m, active, opponent, pr, sr),
    )


def _used_protect_last_turn(active: Pokemon) -> bool:
    """True if the active mon's last move was Protect/Detect/Endure.

    Stateless (reads battle-derived `Pokemon.last_move`) so it is safe even when
    the env reuses a single opponent instance across many battles. Protect fails
    when used on consecutive turns, so callers use this to avoid spamming it.
    """
    last = getattr(active, "last_move", None)
    return last is not None and last.id in _PROTECT_MOVES


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


# ===========================================================================
# V2 opponents — same playstyle identities, slightly sharper patterns
# ===========================================================================


class Gen3StallerV2Player(Player):
    """
    Staller V2 — same status/stall identity, three fixes over V1:

      * Heals *before* throwing a status when about to die (V1 could fish for
        Toxic at 10% HP and faint instead).
      * Skips status moves the opponent is immune to (`is_status_move_immune`):
        no more Toxic into a Steel/Poison wall or Thunder Wave into a Ground type.
      * Won't spam Protect on consecutive turns (Protect fails when repeated).

    Priority: emergency heal → status → Protect-stall → hazard clear → chip → switch.
    """

    def choose_move(self, battle: Battle) -> BattleOrder:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        if battle.available_moves:
            # 1. Emergency recovery — survive before doing anything cute.
            if active.current_hp_fraction < _RECOVERY_HP_THRESHOLD:
                for move in battle.available_moves:
                    if move.id in _RECOVERY_MOVES:
                        return Player.create_order(move)

            # 2. Status a clean opponent — only if it isn't immune.
            if opponent.status is None:
                for move in battle.available_moves:
                    if (
                        move.id in _STATUS_MOVES
                        and move.base_power == 0
                        and not is_status_move_immune(move.id, opponent)
                    ):
                        return Player.create_order(move)

            # 3. Protect to rack up toxic damage — but not two turns running.
            if (
                opponent.status == Status.TOX
                and random.random() < _PROTECT_PROBABILITY
                and not _used_protect_last_turn(active)
            ):
                for move in battle.available_moves:
                    if move.id in _PROTECT_MOVES:
                        return Player.create_order(move)

            # 4. Hazard clear
            if battle.side_conditions:
                for move in battle.available_moves:
                    if move.id in _HAZARD_CLEAR:
                        return Player.create_order(move)

            # 5. Best chip damage (physical/special-aware)
            best = _best_damage_move_v2(battle)
            if best is not None:
                return Player.create_order(best)

        # 6. Switch to the best matchup
        switch = _best_switch(battle)
        if switch is not None:
            return Player.create_order(switch)

        return self.choose_random_move(battle)


class Gen3AggressiveV2Player(Player):
    """
    Aggressive V2 — still hyper-offense, two fixes over V1:

      * Escapes when genuinely walled: if the best move does 0× (immune target)
        or the matchup is badly lost and a better switch-in exists, it pivots
        instead of mindlessly clicking a useless attack.
      * On a forced switch it picks the best *matchup*, not the raw highest
        Atk/SpA base stat (V1 would send a strong-but-countered mon to its death).

    Otherwise it attacks every turn with the strongest move.
    """

    def choose_move(self, battle: Battle) -> BattleOrder:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        if battle.available_moves:
            damaging = [m for m in battle.available_moves if m.base_power > 0]
            if damaging:
                best = _best_damage_move_v2(battle)

                # Wall escape — only when a real switch is available.
                if battle.available_switches:
                    immune = effective_multiplier(best.type, opponent) == 0.0
                    matchup = SimpleHeuristicsPlayer._estimate_matchup(active, opponent)
                    badly_outmatched = (
                        matchup < SimpleHeuristicsPlayer.SWITCH_OUT_MATCHUP_THRESHOLD
                        and any(
                            SimpleHeuristicsPlayer._estimate_matchup(s, opponent) > matchup
                            for s in battle.available_switches
                        )
                    )
                    if immune or badly_outmatched:
                        switch = _best_switch(battle)
                        if switch is not None:
                            return Player.create_order(switch)

                return Player.create_order(best)

            # All available moves are non-damaging — pick the highest base_power
            # (even if 0) to avoid forfeit.
            best = max(battle.available_moves, key=lambda m: m.base_power)
            return Player.create_order(best)

        # Forced switch — pick the best matchup, not the biggest raw attacker.
        if battle.available_switches:
            switch = _best_switch(battle)
            if switch is not None:
                return Player.create_order(switch)

        return self.choose_random_move(battle)


class Gen3SetupSweepV2Player(Player):
    """
    Setup-sweep V2 — same setup-then-sweep identity, two fixes over V1:

      * Won't set up in front of a faster attacker unless at full HP — boosting
        while a faster threat gets a free hit (and may KO mid-setup) is bad.
      * Never voluntarily switches a *boosted* mon out — that would throw away
        the very boosts it spent turns accumulating.

    The boost cap stays at ~2 setup moves (`_SETUP_BOOST_CAP = 4` total offensive
    stages ≈ two Dragon Dances / two Swords Dances), then it sweeps.
    """

    def choose_move(self, battle: Battle) -> BattleOrder:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        matchup = SimpleHeuristicsPlayer._estimate_matchup(active, opponent)
        total_offensive_boosts = sum(active.boosts.get(s, 0) for s in _SETUP_STATS)

        # A faster opponent gets a free turn while we boost — only safe at full HP.
        opponent_faster = opponent.base_stats["spe"] > active.base_stats["spe"]
        unsafe_to_setup = opponent_faster and active.current_hp_fraction < 1.0

        # Switch out only when losing AND unboosted — keep boosts once we have them.
        should_switch = (
            matchup < SimpleHeuristicsPlayer.SWITCH_OUT_MATCHUP_THRESHOLD
            and bool(battle.available_switches)
            and total_offensive_boosts == 0
        )

        if battle.available_moves and not should_switch:
            # 1. Setup when healthy, winning, not capped, and not in front of a
            #    faster threat.
            if (
                active.current_hp_fraction >= _SETUP_HP_THRESHOLD
                and matchup > 0
                and total_offensive_boosts < _SETUP_BOOST_CAP
                and not unsafe_to_setup
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

            # 2. Attack (physical/special-aware)
            best = _best_damage_move_v2(battle)
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


# ---------------------------------------------------------------------------
# Gen3HeuristicV2Player — the smarter all-around generalist ("Heuristic2")
# ---------------------------------------------------------------------------

_KO_HP_THRESHOLD = 0.55       # treat a super-effective hit into <=55% HP as a likely KO
_KO_FINISH_THRESHOLD = 0.35   # at <=35% HP a strong hit likely finishes even if we're slower
_STATUS_WALL_HP_THRESHOLD = 0.70  # only chip a healthy opponent with status


class Gen3HeuristicV2Player(Player):
    """
    A well-rounded generalist that improves on SimpleHeuristicsPlayer.

    Unlike the single-axis playstyle bots it does whatever the position rewards,
    layering Gen-3-aware play on the proven SimpleHeuristics structure:

      * KO awareness — never switches/sets up/statuses away a kill it can take
        this turn (super-effective into a softened target while faster, or any
        strong hit into a very low opponent).
      * Gen-3-ability-aware damage (`_damage_score_v2` → `effective_multiplier`),
        so it doesn't click Earthquake into a Levitator or Surf into Water Absorb.
      * Smarter switching via `SimpleHeuristicsPlayer._should_switch_out`.
      * Self-preservation recovery when low and not about to be KO'd through it.
      * Entry-hazard setting/removal and full-HP setup, reused from SimpleHeuristics.
      * Opportunistic, immunity-checked status on a healthy wall it can't break.

    Decision order: KO → switch-out → recovery → hazards → setup → status → best
    damage → best switch → random.
    """

    def choose_move(self, battle: Battle) -> BattleOrder:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        best = _best_damage_move_v2(battle) if battle.available_moves else None
        we_faster = active.base_stats["spe"] > opponent.base_stats["spe"]
        matchup = SimpleHeuristicsPlayer._estimate_matchup(active, opponent)

        # 1. Take the kill if we have one.
        if best is not None and best.base_power > 0:
            best_eff = effective_multiplier(best.type, opponent)
            likely_ko = (
                best_eff >= 2.0 and opponent.current_hp_fraction <= _KO_HP_THRESHOLD
            ) or opponent.current_hp_fraction <= _KO_FINISH_THRESHOLD
            if likely_ko and (we_faster or opponent.current_hp_fraction <= _KO_FINISH_THRESHOLD):
                return Player.create_order(best)

        # 2. Bail out of a losing matchup (SimpleHeuristics' own switch trigger).
        if battle.available_switches and SimpleHeuristicsPlayer._should_switch_out(battle):
            switch = _best_switch(battle)
            if switch is not None:
                return Player.create_order(switch)

        if battle.available_moves:
            n_remaining = sum(1 for m in battle.team.values() if not m.fainted)
            n_opp_remaining = 6 - sum(
                1 for m in battle.opponent_team.values() if m.fainted
            )

            # 3. Recovery when low and not obviously about to be KO'd through it.
            if active.current_hp_fraction < _RECOVERY_HP_THRESHOLD and (
                we_faster or matchup > 0
            ):
                for move in battle.available_moves:
                    if move.id in _RECOVERY_MOVES:
                        return Player.create_order(move)

            # 4. Entry hazards — set when the opponent still has a deep bench,
            #    remove when our own side is hazard-clogged.
            for move in battle.available_moves:
                if (
                    n_opp_remaining >= 3
                    and move.id in SimpleHeuristicsPlayer.ENTRY_HAZARDS
                    and SimpleHeuristicsPlayer.ENTRY_HAZARDS[move.id]
                    not in battle.opponent_side_conditions
                ):
                    return Player.create_order(move)
                if (
                    battle.side_conditions
                    and move.id in SimpleHeuristicsPlayer.ANTI_HAZARDS_MOVES
                    and n_remaining >= 2
                ):
                    return Player.create_order(move)

            # 5. Setup at full HP into a winning matchup.
            if active.current_hp_fraction == 1.0 and matchup > 0:
                for move in battle.available_moves:
                    if (
                        move.boosts
                        and sum(move.boosts.values()) >= 2
                        and move.target == "self"
                        and min(
                            active.boosts.get(s, 0)
                            for s, v in move.boosts.items()
                            if v > 0
                        )
                        < 6
                    ):
                        return Player.create_order(move)

            # 6. Chip a healthy wall we can't break with a non-immune status.
            best_eff = (
                effective_multiplier(best.type, opponent)
                if best is not None and best.base_power > 0
                else 0.0
            )
            if (
                opponent.status is None
                and matchup <= 0
                and best_eff < 2.0
                and opponent.current_hp_fraction > _STATUS_WALL_HP_THRESHOLD
            ):
                for move in battle.available_moves:
                    if (
                        move.id in _STATUS_MOVES
                        and move.base_power == 0
                        and not is_status_move_immune(move.id, opponent)
                    ):
                        return Player.create_order(move)

            # 7. Otherwise just hit it as hard as possible.
            if best is not None:
                return Player.create_order(best)

        # 8. No moves — switch or fall back.
        switch = _best_switch(battle)
        if switch is not None:
            return Player.create_order(switch)

        return self.choose_random_move(battle)
