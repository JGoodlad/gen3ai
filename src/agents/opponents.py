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
  - Gen3HeuristicV2Player  (display name "heuristic2")

NOTE: the V2 classes are defined here but not yet wired into the training/eval
rotation (OPPONENT_CLASSES / eval lists in train_rl_agent.py). Their display names
are registered in eval_callback._OPPONENT_NAMES so the TUI/TensorBoard label them
correctly once they are added.
"""
import random

from poke_env.battle.battle import Battle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder
from poke_env.player.baselines import SimpleHeuristicsPlayer

from agents.enums import MoveCategory, Status
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


# ---------------------------------------------------------------------------
# Damage estimation — the real edge SimpleHeuristics lacks. A rough Gen 3
# level-100 damage calc, expressed as a fraction of the defender's CURRENT HP,
# so a bot can tell "this move KOs, click it" from "this barely dents it".
# ---------------------------------------------------------------------------

_AVG_DAMAGE_ROLL = 0.925   # midpoint of Gen 3's 0.85–1.0 damage spread
_KO_FRACTION = 1.0         # est. damage ≥ current HP → likely KO
_WALL_FRACTION = 0.34      # best move can't even 3HKO → we can't break it


def _estimate_max_hp(mon: Pokemon) -> float:
    """Rough level-100 max HP with mid EV investment.

    Gen 3 HP at L100/31 IV is 2*base + 110 (0 EV) … 2*base + 173 (252 EV);
    2*base + 160 is a defensible middle that leans tanky, so KO calls stay
    conservative (we under-claim KOs rather than commit to a non-KO)."""
    return 2.0 * mon.base_stats["hp"] + 160.0


def _estimate_damage_fraction(move: Move, attacker: Pokemon, defender: Pokemon) -> float:
    """Approx Gen 3 damage of `move` as a fraction of the defender's CURRENT HP.

    Uses `_stat_estimation` for the relevant attack/defense stats (boost-aware),
    the average damage roll, STAB, ability-aware type effectiveness, and
    `expected_hits`. Accuracy is intentionally excluded — this answers "if it
    hits, does it KO", not expected value (move *ranking* keeps accuracy via
    `_damage_score_v2`). Returns 0.0 for status / immune moves.
    """
    if move.base_power == 0:
        return 0.0
    eff = effective_multiplier(move.type, defender)
    if eff == 0.0:
        return 0.0
    se = SimpleHeuristicsPlayer._stat_estimation
    if move.category == MoveCategory.PHYSICAL:
        atk, dfn = se(attacker, "atk"), se(defender, "def")
    else:
        atk, dfn = se(attacker, "spa"), se(defender, "spd")
    stab = 1.5 if move.type in attacker.types else 1.0
    base = (42.0 * move.base_power * atk / dfn) / 50.0 + 2.0
    dmg = base * stab * eff * move.expected_hits * _AVG_DAMAGE_ROLL
    cur_hp = _estimate_max_hp(defender) * max(defender.current_hp_fraction, 1e-6)
    return dmg / cur_hp if cur_hp > 0 else 0.0


def _best_move_and_ko_fraction(battle: Battle):
    """(best damaging move by score, its est. damage fraction vs the opp active).

    Returns (None, 0.0) when there is no usable damaging move.
    """
    best = _best_damage_move_v2(battle)
    if best is None or best.base_power == 0:
        return best, 0.0
    return best, _estimate_damage_fraction(best, battle.active_pokemon, battle.opponent_active_pokemon)


def _opp_known_max_damage_fraction(battle: Battle) -> float:
    """Max damage fraction the opponent's *revealed* damaging moves would do to us.

    0.0 when the opponent has shown no damaging move yet (can't tell → assume
    safe). Lets a setup bot hold back only against a concrete fast threat."""
    opp = battle.opponent_active_pokemon
    active = battle.active_pokemon
    if opp is None or active is None:
        return 0.0
    moves = [m for m in opp.moves.values() if m.base_power > 0]
    if not moves:
        return 0.0
    return max(_estimate_damage_fraction(m, opp, active) for m in moves)


def _best_switch_v2(battle: Battle):
    """Damage-aware switch selector — the big shared upgrade over `_best_switch`.

    Scores each bench mon by: type matchup (as before) + how hard it hits the
    opponent with its own moveset (we know our full movesets) − 1.5× the worst
    hit it would take from the opponent's *revealed* moves. So it prefers a
    switch-in that both threatens back and actually survives, instead of one
    that merely has a nice type chart but gets OHKO'd. Falls back gracefully to
    pure matchup when the opponent has revealed nothing.
    """
    opp = battle.opponent_active_pokemon
    switches = battle.available_switches
    if not switches or opp is None:
        return None
    opp_moves = [m for m in opp.moves.values() if m.base_power > 0]

    def score(s: Pokemon) -> float:
        incoming = max((_estimate_damage_fraction(m, opp, s) for m in opp_moves), default=0.0)
        own_moves = [m for m in s.moves.values() if m.base_power > 0]
        outgoing = max((_estimate_damage_fraction(m, s, opp) for m in own_moves), default=0.0)
        return SimpleHeuristicsPlayer._estimate_matchup(s, opp) + outgoing - 1.5 * incoming

    return max(switches, key=score)


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

    Priority: defensive pivot (vs likely OHKO) → emergency heal → status →
    Protect-stall → hazard clear → chip → switch. Switching is damage-aware
    (`_best_switch_v2`) so it lands on a real wall, not just a type chart.
    """

    def choose_move(self, battle: Battle) -> BattleOrder:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        # 0. Defensive pivot: if the opponent's revealed moves likely OHKO the
        #    active mon, slide to the sturdiest switch-in — preserving walls is
        #    the staller's whole game, and it dodges the agent's removal attempt.
        if battle.available_switches and _opp_known_max_damage_fraction(battle) >= 0.85:
            switch = _best_switch_v2(battle)
            if switch is not None:
                return Player.create_order(switch)

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
        switch = _best_switch_v2(battle)
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
                # Lock in a guaranteed KO if we have one (most reliable: weight by
                # accuracy × hits); otherwise just maximise damage.
                ko_moves = [
                    m for m in damaging
                    if _estimate_damage_fraction(m, active, opponent) >= _KO_FRACTION
                ]
                best = (
                    max(ko_moves, key=lambda m: m.accuracy * m.expected_hits)
                    if ko_moves else _best_damage_move_v2(battle)
                )

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
                        switch = _best_switch_v2(battle)
                        if switch is not None:
                            return Player.create_order(switch)

                return Player.create_order(best)

            # All available moves are non-damaging — pick the highest base_power
            # (even if 0) to avoid forfeit.
            best = max(battle.available_moves, key=lambda m: m.base_power)
            return Player.create_order(best)

        # Forced switch — pick the best matchup, not the biggest raw attacker.
        if battle.available_switches:
            switch = _best_switch_v2(battle)
            if switch is not None:
                return Player.create_order(switch)

        return self.choose_random_move(battle)


class Gen3SetupSweepV2Player(Player):
    """
    Setup-sweep V2 — same setup-then-sweep identity, sharpened:

      * Closes games out: if the damage calc says the best move already KOs, it
        attacks instead of wasting a turn on another boost.
      * Holds back from setup only against a *concrete* fast threat — an opponent
        that both outspeeds us AND whose revealed moves can ~2HKO us. V1's blanket
        "never set up below full HP vs a faster mon" made it far too passive (it
        is a *setup sweeper* — aggression is its threat), which is why agents beat
        the over-cautious first cut more easily.
      * Never voluntarily switches a *boosted* mon out — that would throw away
        the boosts it spent turns accumulating.

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
        best, best_frac = _best_move_and_ko_fraction(battle)

        # 0. If we can already KO, take it — don't burn a turn boosting.
        if battle.available_moves and best is not None and best_frac >= _KO_FRACTION:
            return Player.create_order(best)

        # Only boost when we're offensively favoured (matchup > 0) — V1's key
        # strength: it sets up to sweep, not in front of a wall it can't break.
        # The damage check is an extra *safety* gate on top, not a replacement:
        # don't boost if the opponent's revealed moves can roughly 2HKO us.
        safe_to_setup = _opp_known_max_damage_fraction(battle) < 0.55

        # Switch out only when losing AND unboosted — keep boosts once we have them.
        should_switch = (
            matchup < SimpleHeuristicsPlayer.SWITCH_OUT_MATCHUP_THRESHOLD
            and bool(battle.available_switches)
            and total_offensive_boosts == 0
        )

        if battle.available_moves and not should_switch:
            # 1. Setup when healthy, offensively favoured, not capped, and not
            #    facing a move that can roughly 2HKO us.
            if (
                active.current_hp_fraction >= _SETUP_HP_THRESHOLD
                and matchup > 0
                and safe_to_setup
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

            # 2. Attack (physical/special-aware) — reuse the move from step 0.
            if best is not None:
                return Player.create_order(best)

        # 3. Switch to a better matchup
        if battle.available_switches:
            switch = _best_switch_v2(battle)
            if switch is not None:
                return Player.create_order(switch)

        if battle.available_moves:
            return Player.create_order(battle.available_moves[0])

        return self.choose_random_move(battle)


# ---------------------------------------------------------------------------
# Gen3HeuristicV2Player — the smarter all-around generalist ("heuristic2")
# ---------------------------------------------------------------------------

_KO_FINISH_THRESHOLD = 0.35   # at <=35% HP a strong hit likely finishes even if we're slower
_STATUS_WALL_HP_THRESHOLD = 0.70  # only chip a healthy opponent with status


class Gen3HeuristicV2Player(Player):
    """
    A well-rounded generalist that improves on SimpleHeuristicsPlayer.

    Unlike the single-axis playstyle bots it does whatever the position rewards,
    layering Gen-3-aware play on the proven SimpleHeuristics structure:

      * KO awareness via a real Gen-3 damage calc (`_estimate_damage_fraction`):
        if the best move's estimated damage ≥ the opponent's current HP and we
        strike first, it takes the kill — never switching/setting up/statusing
        away a winning hit.
      * Gen-3-ability-aware damage (`_damage_score_v2` → `effective_multiplier`),
        so it doesn't click Earthquake into a Levitator or Surf into Water Absorb.
      * Smarter switching via `SimpleHeuristicsPlayer._should_switch_out`.
      * Self-preservation recovery when low and not about to be KO'd through it.
      * Entry-hazard setting/removal and full-HP setup, reused from SimpleHeuristics.
      * Status only on a wall the damage calc says it genuinely can't break
        (best move can't even 3HKO), and only if not immune.

    Decision order: KO → defensive switch → switch-out → recovery → hazards →
    setup → status → best damage → best switch → random.
    """

    def choose_move(self, battle: Battle) -> BattleOrder:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        best, best_frac = (
            _best_move_and_ko_fraction(battle) if battle.available_moves else (None, 0.0)
        )
        we_faster = active.base_stats["spe"] > opponent.base_stats["spe"]
        matchup = SimpleHeuristicsPlayer._estimate_matchup(active, opponent)

        # 1. Take the kill if the damage calc says we have one (and we strike
        #    first, or the opponent is already low enough that the trade is fine).
        if best is not None and best_frac >= _KO_FRACTION and (
            we_faster or opponent.current_hp_fraction <= _KO_FINISH_THRESHOLD
        ):
            return Player.create_order(best)

        # 2. Defensive pivot: if we're slower and the opponent's revealed moves
        #    likely KO us this turn while we can't KO back, switch to a check
        #    rather than dying in place.
        if battle.available_switches and not we_faster and best_frac < _KO_FRACTION:
            if _opp_known_max_damage_fraction(battle) >= 0.85:
                switch = _best_switch_v2(battle)
                if switch is not None:
                    return Player.create_order(switch)

        # 3. Bail out of a losing matchup (SimpleHeuristics' own switch trigger).
        if battle.available_switches and SimpleHeuristicsPlayer._should_switch_out(battle):
            switch = _best_switch_v2(battle)
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

            # 6. Chip a healthy wall we genuinely can't break (best move can't
            #    even 3HKO it) with a non-immune status.
            if (
                opponent.status is None
                and matchup <= 0
                and best_frac < _WALL_FRACTION
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
        switch = _best_switch_v2(battle)
        if switch is not None:
            return Player.create_order(switch)

        return self.choose_random_move(battle)
