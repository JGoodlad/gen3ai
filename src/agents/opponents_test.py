import os
import random

import pytest
from unittest.mock import MagicMock, patch

from poke_env.battle.status import Status
from poke_env.player.battle_order import BattleOrder

from agents import opponents as _opponents
from agents.opponents import (
    Gen3StallerPlayer,
    Gen3StallerV2Player,
    Gen3AggressivePlayer,
    Gen3SetupSweepPlayer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_move(id_, base_power=80, type_=None, accuracy=1.0, target="normal", boosts=None, effectiveness=1.0):
    move = MagicMock()
    move.id = id_
    move.base_power = base_power
    move.accuracy = accuracy
    move.target = target
    move.boosts = boosts
    if type_ is not None:
        move.type = type_
    else:
        move.type = MagicMock()
        move.type.damage_multiplier.return_value = effectiveness
    return move


def _make_mon(types=None, base_stats=None, boosts=None, hp_fraction=1.0, status=None):
    mon = MagicMock()
    mon.types = types or [MagicMock(), MagicMock()]
    mon.type_1 = None
    mon.type_2 = None
    mon.ability = None
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


def _seeded_player(cls, seed):
    """…the same, but with the staller's own Protect RNG installed (what `protect_seed=` does)."""
    p = _make_player(cls)
    p._protect_rng = random.Random(seed)
    return p


@pytest.fixture(autouse=True)
def _inject_mock_effectiveness(monkeypatch):
    """`_make_move` injects a move's type effectiveness via
    `move.type.damage_multiplier.return_value` (a MagicMock convention from when the real
    `effective_multiplier` *called* that method). It now reads a precomputed type chart and
    ignores the mock — and a MagicMock type isn't a chart key — so route the opponents
    module's `effective_multiplier` back to the injected value. These are heuristic-SELECTION
    tests; the chart math itself is covered exhaustively in gen3_mechanics_test.py.
    """
    def _eff(move_type, opp):
        rv = getattr(getattr(move_type, "damage_multiplier", None), "return_value", None)
        return rv if isinstance(rv, (int, float)) and not isinstance(rv, bool) else 1.0
    monkeypatch.setattr("agents.opponents.effective_multiplier", _eff)


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
        weak = _make_move("tackle", base_power=40, effectiveness=1.0)
        strong = _make_move("closecombat", base_power=120, effectiveness=1.0)
        active = _make_mon()
        # Neither move shares active's type so no STAB modifier difference
        active.types = [MagicMock()]
        opponent = _make_mon()
        battle = _make_battle(moves=[weak, strong], active=active, opponent=opponent)

        assert self._order_id(battle) == "closecombat"

    def test_prefers_stab_move(self):
        shared_type = MagicMock()
        shared_type.damage_multiplier.return_value = 1.0
        stab_move = _make_move("waterfall", base_power=80)
        stab_move.type = shared_type
        no_stab = _make_move("earthquake", base_power=100, effectiveness=1.0)

        active = _make_mon(types=[shared_type])
        opponent = _make_mon()

        battle = _make_battle(moves=[stab_move, no_stab], active=active, opponent=opponent)

        # waterfall: 80 * 1.5 * 1.0 = 120 > earthquake: 100 * 1.0 * 1.0 = 100
        assert self._order_id(battle) == "waterfall"

    def test_prefers_super_effective_move(self):
        normal_move = _make_move("tackle", base_power=80, effectiveness=1.0)
        super_eff = _make_move("icebeam", base_power=90, effectiveness=2.0)

        active = _make_mon(types=[MagicMock()])
        opponent = _make_mon()
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


# ---------------------------------------------------------------------------
# The staller Protect coin — per-instance RNG (gen3_staller_protect_rng_v1)
# ---------------------------------------------------------------------------
# Motivation is a MEASUREMENT, not tidiness. The transfer-coefficient cell
# (`designs/research_state/measurements/transfer_coefficient_cell_2026-08-29.md` §4) ran a paired
# falsifier whose zero-overrule units MUST be the same battle in both arms: it passed EXACTLY on
# the seven deterministic bots (2693 pairs, delta 0.0000, zero divergences) and failed on exactly
# these two stallers (755 pairs, 4 divergences). The two arms interleave `choose_move` differently,
# so a coin drawn from the process-wide `random` module lands differently with no treatment
# involved. Unbiased noise — but it widens every paired interval for free.

class TestStallerProtectRng:
    """Two claims: the DEFAULT is the shared global stream (byte-identical), and a SEEDED staller
    is decision-identical across arms whose interleaving differs."""

    @staticmethod
    def _toxic_battle():
        protect = _make_move("protect", base_power=0)
        attack = _make_move("earthquake", base_power=100)
        return _make_battle(moves=[protect, attack], opponent=_make_mon(status=Status.TOX))

    def _flips(self, player, n=40):
        """The bot's Protect/attack decision sequence — the observable the paired design compares."""
        battle = self._toxic_battle()
        out = []
        for _ in range(n):
            order = player.choose_move(battle)
            out.append(getattr(order.order, "id", None))
        return out

    def test_default_is_the_shared_global_stream(self):
        """No seed anywhere ⇒ the coin is `random.random` itself, so a default run is unchanged.
        (The three tests above already patch `agents.opponents.random.random` and pass — this states
        the property those inherit rather than leaving it implicit.)"""
        p = _make_player(Gen3StallerPlayer)
        assert p._protect_rng is _opponents.random

    def test_an_env_hook_seeds_every_staller_in_the_process(self):
        """The hook a paired-arm harness needs when it does not own the construction site (the bots
        are built deep inside `env_factory` / `eval_worker`)."""
        with patch.dict(os.environ, {"GEN3AI_STALLER_SEED": "7"}):
            rng = _opponents._resolve_protect_rng(None)
        assert isinstance(rng, random.Random)
        assert rng.random() == random.Random(7).random()

    def test_an_unparseable_env_seed_raises_rather_than_falling_back(self):
        """A seed that was meant to be set and silently was not would make an arm LOOK reproducible
        while it is not — the failure mode this whole fix exists to remove."""
        with patch.dict(os.environ, {"GEN3AI_STALLER_SEED": "not-an-int"}):
            with pytest.raises(ValueError, match="GEN3AI_STALLER_SEED"):
                _opponents._resolve_protect_rng(None)

    def test_no_seed_and_no_env_returns_the_module_itself(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _opponents._resolve_protect_rng(None) is _opponents.random

    def test_two_paired_arms_with_seeded_stallers_are_decision_identical(self):
        """THE regression test. Arm A and arm B share a seed but are INTERLEAVED with unrelated
        global-`random` traffic between decisions — the exact asymmetry the searched arm introduces
        (it awaits an executor; the control runs inline). Seeded, the two decision sequences are
        identical; the arms differ only in what the global stream did around them."""
        arm_a = _seeded_player(Gen3StallerPlayer, 1234)
        arm_b = _seeded_player(Gen3StallerPlayer, 1234)

        random.seed(0)
        flips_a = self._flips(arm_a)
        # Arm B burns a DIFFERENT, unpredictable amount of the global stream between its decisions.
        random.seed(999)
        battle, flips_b = self._toxic_battle(), []
        for _ in range(40):
            for _ in range(random.randint(1, 5)):
                random.random()
            flips_b.append(getattr(arm_b.choose_move(battle).order, "id", None))

        assert flips_a == flips_b
        assert {"protect", "earthquake"} <= set(flips_a)   # the coin really did both

    @pytest.mark.parametrize("cls", [Gen3StallerPlayer, Gen3StallerV2Player])
    def test_the_coin_itself_is_per_instance_for_BOTH_stallers(self, cls):
        """The two bots reach the coin through different priority ladders (V2 heals and checks
        status-immunity first), so the decision-sequence test above is driven through V1's simpler
        mocks. This one asserts the property at the shared seam both call — and includes its own
        revert arm: unseeded, the same interleaving pulls the two apart."""
        seeded_a, seeded_b = _seeded_player(cls, 42), _seeded_player(cls, 42)
        bare_a, bare_b = _make_player(cls), _make_player(cls)

        def draws(player, jitter_seed):
            random.seed(jitter_seed)
            out = []
            for _ in range(30):
                for _ in range(random.randint(1, 5)):
                    random.random()      # the other arm's unrelated global traffic
                out.append(player._protect_roll())
            return out

        assert draws(seeded_a, 0) == draws(seeded_b, 999)
        assert draws(bare_a, 0) != draws(bare_b, 999)

    def test_the_unseeded_default_is_the_one_that_couples(self):
        """REVERT-VERIFICATION for the fix: run the same interleaving WITHOUT seeds and the two
        arms diverge. If this ever passes, the per-instance RNG has stopped being the difference."""
        arm_a, arm_b = _make_player(Gen3StallerPlayer), _make_player(Gen3StallerPlayer)
        random.seed(0)
        flips_a = self._flips(arm_a)
        random.seed(999)
        battle, flips_b = self._toxic_battle(), []
        for _ in range(40):
            for _ in range(random.randint(1, 5)):
                random.random()
            flips_b.append(getattr(arm_b.choose_move(battle).order, "id", None))
        assert flips_a != flips_b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
