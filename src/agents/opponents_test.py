import pytest
from unittest.mock import MagicMock, patch

from poke_env.battle.status import Status
from poke_env.player.battle_order import BattleOrder

from agents.opponents import (
    Gen3StallerPlayer,
    Gen3AggressivePlayer,
    Gen3SetupSweepPlayer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_move(id_, base_power=80, type_=None, accuracy=1.0, target="normal", boosts=None):
    move = MagicMock()
    move.id = id_
    move.base_power = base_power
    move.type = type_ or MagicMock()
    move.accuracy = accuracy
    move.target = target
    move.boosts = boosts
    return move


def _make_mon(types=None, base_stats=None, boosts=None, hp_fraction=1.0, status=None):
    mon = MagicMock()
    mon.types = types or [MagicMock(), MagicMock()]
    mon.base_stats = base_stats or {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80}
    mon.boosts = boosts or {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "acc": 0, "eva": 0}
    mon.current_hp_fraction = hp_fraction
    mon.status = status
    mon.fainted = False
    mon.damage_multiplier = MagicMock(return_value=1.0)
    return mon


def _make_battle(moves=(), switches=(), side_conditions=None, active=None, opponent=None):
    battle = MagicMock()
    battle.available_moves = list(moves)
    battle.available_switches = list(switches)
    battle.side_conditions = side_conditions or {}
    battle.active_pokemon = active or _make_mon()
    battle.opponent_active_pokemon = opponent or _make_mon()
    battle.force_switch = False
    return battle


def _make_player(cls):
    """Instantiate a player without connecting to a server."""
    with patch.object(cls, "__init__", lambda self, **kw: None):
        p = cls.__new__(cls)
    p.choose_random_move = MagicMock(return_value=MagicMock(spec=BattleOrder))
    return p


# ---------------------------------------------------------------------------
# Gen3StallerPlayer
# ---------------------------------------------------------------------------

class TestGen3StallerPlayer:
    def setup_method(self):
        self.player = _make_player(Gen3StallerPlayer)

    def _order_id(self, battle):
        order = self.player.choose_move(battle)
        return getattr(order.order, "id", None)

    def test_uses_status_move_against_clean_opponent(self):
        status_move = _make_move("toxic", base_power=0)
        attack = _make_move("earthquake", base_power=100)
        opponent = _make_mon(status=None)
        battle = _make_battle(moves=[status_move, attack], opponent=opponent)

        assert self._order_id(battle) == "toxic"

    def test_skips_status_move_when_opponent_already_statused(self):
        status_move = _make_move("toxic", base_power=0)
        attack = _make_move("earthquake", base_power=100)
        opponent = _make_mon(status=Status.TOX)
        battle = _make_battle(moves=[status_move, attack], opponent=opponent)

        # Should fall through to the damage move
        assert self._order_id(battle) == "earthquake"

    def test_uses_recovery_when_low_hp(self):
        recover = _make_move("recover", base_power=0)
        attack = _make_move("earthquake", base_power=100)
        active = _make_mon(hp_fraction=0.3)
        opponent = _make_mon(status=Status.TOX)  # already statused so no status move
        battle = _make_battle(moves=[recover, attack], active=active, opponent=opponent)

        assert self._order_id(battle) == "recover"

    def test_no_recovery_when_hp_is_fine(self):
        recover = _make_move("recover", base_power=0)
        attack = _make_move("earthquake", base_power=100)
        active = _make_mon(hp_fraction=0.8)
        opponent = _make_mon(status=Status.TOX)
        battle = _make_battle(moves=[recover, attack], active=active, opponent=opponent)

        # HP is above threshold — should attack instead
        assert self._order_id(battle) == "earthquake"

    def test_clears_hazards_with_rapidspin(self):
        spin = _make_move("rapidspin", base_power=20)
        attack = _make_move("earthquake", base_power=100)
        active = _make_mon(hp_fraction=0.8)
        opponent = _make_mon(status=Status.TOX)
        from poke_env.battle.side_condition import SideCondition
        battle = _make_battle(
            moves=[spin, attack],
            side_conditions={SideCondition.SPIKES: 1},
            active=active,
            opponent=opponent,
        )

        assert self._order_id(battle) == "rapidspin"

    def test_no_hazard_clear_without_side_conditions(self):
        spin = _make_move("rapidspin", base_power=20)
        attack = _make_move("earthquake", base_power=100)
        active = _make_mon(hp_fraction=0.8)
        opponent = _make_mon(status=Status.TOX)
        battle = _make_battle(moves=[spin, attack], side_conditions={}, active=active, opponent=opponent)

        assert self._order_id(battle) == "earthquake"

    def test_falls_back_to_damage_move(self):
        attack = _make_move("earthquake", base_power=100)
        opponent = _make_mon(status=Status.BRN)
        battle = _make_battle(moves=[attack], opponent=opponent)

        assert self._order_id(battle) == "earthquake"

    def test_switches_when_no_moves(self):
        switch_target = _make_mon()
        opponent = _make_mon(status=None)
        # No moves available but there is a switch
        battle = _make_battle(moves=[], switches=[switch_target], opponent=opponent)
        # _best_switch returns the best matchup; with one choice it returns that mon
        order = self.player.choose_move(battle)
        assert order is not None

    def test_uses_protect_when_opponent_is_toxiced(self):
        protect = _make_move("protect", base_power=0)
        attack = _make_move("earthquake", base_power=100)
        opponent = _make_mon(status=Status.TOX)
        battle = _make_battle(moves=[protect, attack], opponent=opponent)

        with patch("agents.opponents.random.random", return_value=0.0):  # always protect
            assert self._order_id(battle) == "protect"

    def test_skips_protect_when_opponent_not_toxiced(self):
        protect = _make_move("protect", base_power=0)
        attack = _make_move("earthquake", base_power=100)
        opponent = _make_mon(status=Status.PAR)  # paralysed, not toxiced
        battle = _make_battle(moves=[protect, attack], opponent=opponent)

        with patch("agents.opponents.random.random", return_value=0.0):
            assert self._order_id(battle) == "earthquake"

    def test_protect_skipped_by_probability(self):
        protect = _make_move("protect", base_power=0)
        attack = _make_move("earthquake", base_power=100)
        opponent = _make_mon(status=Status.TOX)
        battle = _make_battle(moves=[protect, attack], opponent=opponent)

        with patch("agents.opponents.random.random", return_value=0.99):  # above threshold
            assert self._order_id(battle) == "earthquake"

    def test_recovery_takes_priority_over_hazard_clear_when_low_hp(self):
        recover = _make_move("recover", base_power=0)
        spin = _make_move("rapidspin", base_power=20)
        active = _make_mon(hp_fraction=0.3)
        opponent = _make_mon(status=Status.TOX)
        from poke_env.battle.side_condition import SideCondition
        battle = _make_battle(
            moves=[recover, spin],
            side_conditions={SideCondition.SPIKES: 1},
            active=active,
            opponent=opponent,
        )
        assert self._order_id(battle) == "recover"


# ---------------------------------------------------------------------------
# Gen3AggressivePlayer
# ---------------------------------------------------------------------------

class TestGen3AggressivePlayer:
    def setup_method(self):
        self.player = _make_player(Gen3AggressivePlayer)

    def _order_id(self, battle):
        order = self.player.choose_move(battle)
        return getattr(order.order, "id", None)

    def test_picks_highest_damage_move(self):
        weak = _make_move("tackle", base_power=40)
        strong = _make_move("closecombat", base_power=120)
        active = _make_mon()
        opponent = _make_mon()
        opponent.damage_multiplier.return_value = 1.0
        # Neither move shares active's type so no STAB modifier difference
        active.types = [MagicMock()]
        weak.type = MagicMock()   # different from active's type
        strong.type = MagicMock()
        battle = _make_battle(moves=[weak, strong], active=active, opponent=opponent)

        assert self._order_id(battle) == "closecombat"

    def test_prefers_stab_move(self):
        shared_type = MagicMock()
        stab_move = _make_move("waterfall", base_power=80)
        stab_move.type = shared_type
        no_stab = _make_move("earthquake", base_power=100)
        no_stab.type = MagicMock()

        active = _make_mon(types=[shared_type])
        opponent = _make_mon()
        opponent.damage_multiplier.return_value = 1.0

        battle = _make_battle(moves=[stab_move, no_stab], active=active, opponent=opponent)

        # waterfall: 80 * 1.5 * 1.0 = 120 > earthquake: 100 * 1.0 * 1.0 = 100
        assert self._order_id(battle) == "waterfall"

    def test_prefers_super_effective_move(self):
        normal_move = _make_move("tackle", base_power=80)
        super_eff = _make_move("icebeam", base_power=90)

        active = _make_mon(types=[MagicMock()])
        normal_move.type = MagicMock()
        super_eff.type = MagicMock()

        opponent = _make_mon()
        def dmg_mult(move_type):
            if move_type is super_eff.type:
                return 2.0
            return 1.0
        opponent.damage_multiplier.side_effect = dmg_mult

        battle = _make_battle(moves=[normal_move, super_eff], active=active, opponent=opponent)

        # ice beam: 90 * 1.0 * 2.0 = 180 > tackle: 80 * 1.0 * 1.0 = 80
        assert self._order_id(battle) == "icebeam"

    def test_ignores_status_only_moves_when_damaging_available(self):
        status_move = _make_move("toxic", base_power=0)
        attack = _make_move("earthquake", base_power=100)
        active = _make_mon()
        opponent = _make_mon(status=None)
        opponent.damage_multiplier.return_value = 1.0
        battle = _make_battle(moves=[status_move, attack], active=active, opponent=opponent)

        assert self._order_id(battle) == "earthquake"

    def test_forced_switch_picks_best_attacker(self):
        sweeper = _make_mon(base_stats={"hp": 80, "atk": 130, "def": 70, "spa": 60, "spd": 70, "spe": 110})
        wall = _make_mon(base_stats={"hp": 100, "atk": 50, "def": 150, "spa": 40, "spd": 140, "spe": 30})
        battle = _make_battle(moves=[], switches=[sweeper, wall])

        order = self.player.choose_move(battle)
        assert order.order is sweeper

    def test_does_not_voluntarily_switch_when_moves_available(self):
        attack = _make_move("earthquake", base_power=100)
        switch_target = _make_mon()
        active = _make_mon()
        opponent = _make_mon()
        opponent.damage_multiplier.return_value = 1.0
        battle = _make_battle(moves=[attack], switches=[switch_target], active=active, opponent=opponent)

        order = self.player.choose_move(battle)
        # Should be a Move order, not a switch
        assert isinstance(order.order, type(attack))


# ---------------------------------------------------------------------------
# Gen3SetupSweepPlayer
# ---------------------------------------------------------------------------

class TestGen3SetupSweepPlayer:
    def setup_method(self):
        self.player = _make_player(Gen3SetupSweepPlayer)

    def _order_id(self, battle):
        order = self.player.choose_move(battle)
        return getattr(order.order, "id", None)

    def _winning_matchup(self, active, opponent):
        """Configure mocks so _estimate_matchup returns a positive score."""
        # opponent takes 2x from active's type, active takes 0.5x from opponent's type
        opponent.damage_multiplier.return_value = 2.0
        active.damage_multiplier.return_value = 0.5

    def _losing_matchup(self, active, opponent):
        opponent.damage_multiplier.return_value = 0.5
        active.damage_multiplier.return_value = 2.0

    def test_uses_setup_when_healthy_and_winning(self):
        sd = _make_move("swordsdance", base_power=0, target="self", boosts={"atk": 2})
        attack = _make_move("earthquake", base_power=100)
        active = _make_mon(hp_fraction=1.0, boosts={"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0})
        opponent = _make_mon()
        self._winning_matchup(active, opponent)
        battle = _make_battle(moves=[sd, attack], active=active, opponent=opponent)

        assert self._order_id(battle) == "swordsdance"

    def test_skips_setup_when_hp_is_low(self):
        sd = _make_move("swordsdance", base_power=0, target="self", boosts={"atk": 2})
        attack = _make_move("earthquake", base_power=100)
        active = _make_mon(hp_fraction=0.5, boosts={"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0})
        opponent = _make_mon()
        self._winning_matchup(active, opponent)
        battle = _make_battle(moves=[sd, attack], active=active, opponent=opponent)

        assert self._order_id(battle) == "earthquake"

    def test_skips_setup_when_matchup_is_losing(self):
        sd = _make_move("swordsdance", base_power=0, target="self", boosts={"atk": 2})
        attack = _make_move("earthquake", base_power=100)
        active = _make_mon(hp_fraction=1.0, boosts={"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0})
        opponent = _make_mon()
        self._losing_matchup(active, opponent)
        battle = _make_battle(moves=[sd, attack], active=active, opponent=opponent)

        assert self._order_id(battle) == "earthquake"

    def test_skips_setup_when_boost_cap_reached(self):
        sd = _make_move("swordsdance", base_power=0, target="self", boosts={"atk": 2})
        attack = _make_move("earthquake", base_power=100)
        # atk already at +4 — total offensive boosts == cap
        active = _make_mon(hp_fraction=1.0, boosts={"atk": 4, "def": 0, "spa": 0, "spd": 0, "spe": 0})
        opponent = _make_mon()
        self._winning_matchup(active, opponent)
        battle = _make_battle(moves=[sd, attack], active=active, opponent=opponent)

        assert self._order_id(battle) == "earthquake"

    def test_skips_setup_when_stat_already_maxed(self):
        sd = _make_move("swordsdance", base_power=0, target="self", boosts={"atk": 2})
        attack = _make_move("earthquake", base_power=100)
        # atk is at +6 max — no room left
        active = _make_mon(hp_fraction=1.0, boosts={"atk": 6, "def": 0, "spa": 0, "spd": 0, "spe": 0})
        opponent = _make_mon()
        self._winning_matchup(active, opponent)
        battle = _make_battle(moves=[sd, attack], active=active, opponent=opponent)

        assert self._order_id(battle) == "earthquake"

    def test_attacks_when_no_setup_available(self):
        attack = _make_move("earthquake", base_power=100)
        active = _make_mon(hp_fraction=1.0, boosts={"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0})
        opponent = _make_mon()
        opponent.damage_multiplier.return_value = 1.0
        battle = _make_battle(moves=[attack], active=active, opponent=opponent)

        assert self._order_id(battle) == "earthquake"

    def test_switches_on_poor_matchup(self):
        attack = _make_move("tackle", base_power=40)
        switch_target = _make_mon()
        active = _make_mon(hp_fraction=1.0, boosts={"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0})
        opponent = _make_mon()
        # 4x damage taken: score = 0.5 - 4.0 = -3.5, below SWITCH_OUT_MATCHUP_THRESHOLD (-2)
        opponent.damage_multiplier.return_value = 0.5
        active.damage_multiplier.return_value = 4.0
        # Make switch target look good against opponent
        switch_target.damage_multiplier.return_value = 2.0
        switch_target.types = [MagicMock()]
        battle = _make_battle(moves=[attack], switches=[switch_target], active=active, opponent=opponent)

        order = self.player.choose_move(battle)
        assert order.order is switch_target


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
