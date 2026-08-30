"""The GLOBAL-RANDOM COUPLING regression suite.

One genre, three seams. A drawer that reaches into the process-wide ``random`` module couples
every other drawer in the process to it: two players interleave their ``choose_move`` calls inside
one battle, and two paired ARMS interleave them differently (one awaits an executor, the other
runs inline), so a decision that consumed the shared stream lands differently in the two arms with
no treatment involved.

The founding specimen was ``Gen3StallerPlayer``'s Protect coin, fixed at 4437c85 — and it was
found by a FAILED integrity check, not a review: the transfer-coefficient cell
(``designs/research_state/measurements/transfer_coefficient_cell_2026-08-29.md`` §4) ran a paired
falsifier whose zero-overrule units MUST be the same battle in both arms, and it came back exactly
0.0000 on the deterministic bots and non-zero on exactly the two stallers. Its own regression
tests live in ``opponents_test.py::TestStallerProtectRng``.

This file covers the three seams the follow-up census found
(``designs/research_state/measurements/global_random_sweep_2026-08-30.md``):

* ``poke_env.player.player`` — ``choose_random_*`` and the ``DEFAULT_CHOICE_CHANCE`` coin. The
  WIDEST of the three: it is ``RandomPlayer``'s entire policy (one draw per decision, not a
  conditional one), the fallback of all sixteen scripted bots, and it fires inside the RL players
  too. The transfer cell could not have caught it — its falsifier conditions on zero-overrule
  units and the overrule rate against ``random`` is 1.00, so that bot contributed no units.
* ``utils.teambuilder.Gen3Teambuilder`` — the per-battle team DRAW. Coupling here is not noise on
  a decision, it is a different game.
* ``agents.training.snapshot_pool.SnapshotPool`` — the self-play pool draw, which sat inside a
  caller that already owned a per-env seeded RNG for the bucket choice.
* ``agents.inference.player.RLPlayer`` — the stochastic ACTION SAMPLE. Same genre, different RNG:
  ``Categorical.sample()`` draws from TORCH's process-wide default generator, and self-play puts
  two of these players in one battle.

Every seam gets the same five claims, and the fifth is the one that keeps the suite honest:
**a revert arm**. If "unseeded, the two arms diverge" ever passes, the per-instance RNG has
stopped being the difference and the other four tests are asserting nothing.
"""
import os
import random
from unittest.mock import patch

import pytest
import torch

from agents.inference.player import RLPlayer, _resolve_policy_seed

from poke_env.battle.battle import Battle
from poke_env.player import player as _player_mod
from poke_env.player.player import Player, _resolve_player_rng

from utils import teambuilder as _tb_mod
from utils.teambuilder import Gen3Teambuilder, _resolve_team_rng

from agents.training import snapshot_pool as _pool_mod
from agents.training.snapshot_pool import SnapshotEntry, SnapshotPool, _resolve_pool_rng


# ---------------------------------------------------------------------------
# The shared shape of the fix — asserted once, over all three seams
# ---------------------------------------------------------------------------
# (module, resolver, env var). Every seam resolves its RNG the same way on purpose: an unseeded
# call returns the `random` MODULE (so the default is byte-identical), a seeded one returns a
# private `random.Random`, and an unparseable env seed RAISES. That last one is not fussiness —
# a seed that was meant to be set and silently was not makes a paired arm LOOK reproducible while
# it is not, which is the exact failure mode the whole genre is about.
_SEAMS = [
    pytest.param(_player_mod, _resolve_player_rng, "GEN3AI_PLAYER_SEED", id="player"),
    pytest.param(_tb_mod, _resolve_team_rng, "GEN3AI_TEAM_SEED", id="teambuilder"),
    pytest.param(_pool_mod, _resolve_pool_rng, "GEN3AI_POOL_SEED", id="snapshot_pool"),
]


@pytest.mark.parametrize("mod, resolve, env_var", _SEAMS)
class TestTheResolverContract:

    def test_no_seed_anywhere_is_the_shared_global_module(self, mod, resolve, env_var):
        """THE default-unchanged claim. Not `isinstance(..., Random)` — the module ITSELF, so the
        unseeded call site makes the same call on the same stream it always did."""
        with patch.dict(os.environ, {}, clear=True):
            assert resolve(None) is random

    def test_an_explicit_seed_gives_a_private_stream(self, mod, resolve, env_var):
        with patch.dict(os.environ, {}, clear=True):
            rng = resolve(11)
        assert isinstance(rng, random.Random)
        assert rng.random() == random.Random(11).random()

    def test_the_env_hook_seeds_every_instance_in_the_process(self, mod, resolve, env_var):
        """The hook a paired-arm harness needs when it does not own the construction site — the
        players are built deep inside `env_factory` / `eval_worker`, the teambuilders inside
        `matchup_spec`, the pool inside the env worker."""
        with patch.dict(os.environ, {env_var: "7"}):
            rng = resolve(None)
        assert isinstance(rng, random.Random)
        assert rng.random() == random.Random(7).random()

    def test_an_explicit_seed_beats_the_env(self, mod, resolve, env_var):
        with patch.dict(os.environ, {env_var: "7"}):
            assert resolve(3).random() == random.Random(3).random()

    def test_an_empty_env_value_is_not_a_seed(self, mod, resolve, env_var):
        with patch.dict(os.environ, {env_var: ""}):
            assert resolve(None) is random

    def test_an_unparseable_env_seed_raises_rather_than_falling_back(self, mod, resolve, env_var):
        with patch.dict(os.environ, {env_var: "not-an-int"}):
            with pytest.raises(ValueError, match=env_var):
                resolve(None)


# ---------------------------------------------------------------------------
# Seam 1 — the player choice coin (gen3_player_choice_rng_v1)
# ---------------------------------------------------------------------------

class _FakeBattle(Battle):
    """A `Battle` only insofar as `choose_random_move`'s isinstance dispatch needs one. It reads
    exactly `valid_orders`, so nothing else has to exist."""

    def __init__(self, orders):          # deliberately does NOT call super()
        self._orders = list(orders)

    @property
    def valid_orders(self):
        return self._orders


class _Bot(Player):
    """A player that does nothing but expose the inherited random-choice path.

    It skips the whole websocket/asyncio construction but installs its RNG through
    `_install_player_rng` — the SAME function `Player.__init__` calls, and the only writer of
    `_choice_rng` — so this harness cannot pass while the real constructor is broken."""

    def __init__(self, rng_seed=None):
        _player_mod._install_player_rng(self, rng_seed)

    def choose_move(self, battle):
        return self.choose_random_move(battle)


def _battle():
    return _FakeBattle([f"order{i}" for i in range(8)])


def _decisions(bot, jitter_seed, n=40):
    """The bot's decision sequence, with the OTHER arm's unrelated global traffic interleaved
    between decisions — the exact asymmetry a searched arm introduces (it awaits an executor; the
    control runs inline)."""
    battle = _battle()
    random.seed(jitter_seed)
    out = []
    for _ in range(n):
        for _ in range(random.randint(1, 5)):
            random.random()
        out.append(bot.choose_move(battle))
    return out


class TestPlayerChoiceRng:
    """The widest seam: `choose_random_move` is `RandomPlayer`'s whole policy and every scripted
    bot's fallback."""

    def test_the_class_attribute_is_the_module_so_an_uninited_player_is_unchanged(self):
        """Several unit suites build bots with `cls.__new__(cls)`. Without a class-level default
        those would raise instead of behaving exactly as they always have."""
        assert Player._choice_rng is random
        assert _Bot.__new__(_Bot)._choice_rng is random

    def test_an_unseeded_player_does_not_even_carry_the_attribute(self):
        """Stronger than "behaves the same": the default instance's ``__dict__`` is byte-identical
        to the pre-fix one, so nothing that inspects or pickles a player can tell the difference."""
        with patch.dict(os.environ, {}, clear=True):
            assert "_choice_rng" not in _Bot().__dict__

    def test_the_class_call_form_still_works_and_still_uses_the_global_stream(self):
        """`Player.choose_random_singles_move(battle)` is called from `singles_env`, `doubles_env`
        and `baselines`; `self.choose_random_move(battle)` from ~50 bot and fuzz sites. The
        descriptor has to keep BOTH spellings, and the class form has no player to be
        per-instance about, so it stays on the shared stream."""
        battle = _battle()
        random.seed(5)
        via_class = [Player.choose_random_singles_move(battle) for _ in range(10)]
        random.seed(5)
        expected = [battle.valid_orders[int(random.random() * 8)] for _ in range(10)]
        assert via_class == expected

    def test_the_unseeded_instance_form_is_byte_identical_to_the_old_static_call(self):
        """THE default-unchanged claim at the call site: same stream, same arithmetic, same
        consumption count."""
        bot, battle = _Bot(), _battle()
        random.seed(99)
        got = [bot.choose_move(battle) for _ in range(20)]
        random.seed(99)
        expected = [battle.valid_orders[int(random.random() * 8)] for _ in range(20)]
        assert got == expected

    def test_two_paired_arms_with_seeded_players_are_decision_identical(self):
        """THE regression test. Arm A and arm B share a seed but burn DIFFERENT, unpredictable
        amounts of the global stream between their decisions. Seeded, the two decision sequences
        are identical; the arms differ only in what the global stream did around them."""
        arm_a, arm_b = _Bot(rng_seed=1234), _Bot(rng_seed=1234)
        assert _decisions(arm_a, 0) == _decisions(arm_b, 999)

    def test_the_unseeded_default_is_the_one_that_couples(self):
        """REVERT-VERIFICATION. If this ever passes, the per-instance RNG has stopped being the
        difference and the test above is asserting nothing."""
        assert _decisions(_Bot(), 0) != _decisions(_Bot(), 999)

    def test_the_env_hook_reaches_a_player_built_without_a_seed_kwarg(self):
        with patch.dict(os.environ, {"GEN3AI_PLAYER_SEED": "31"}):
            arm_a, arm_b = _Bot(), _Bot()
        assert _decisions(arm_a, 0) == _decisions(arm_b, 999)

    def test_the_default_choice_chance_coin_moves_to_the_same_stream(self):
        """`DEFAULT_CHOICE_CHANCE` fires at 1/1000 per rejected request in EVERY player, the RL
        ones included, so it is a shared-stream consumer even in a roster of deterministic bots.
        It reads `self._choice_rng`, so an unseeded player still draws from the module."""
        bot = _Bot()
        random.seed(3)
        got = [bot._choice_rng.random() for _ in range(5)]
        random.seed(3)
        assert got == [random.random() for _ in range(5)]
        assert _Bot(rng_seed=8)._choice_rng.random() == random.Random(8).random()

    def test_the_baseline_bots_fallbacks_reach_the_instance_stream_too(self):
        """`SimpleHeuristicsPlayer` is in BOTH the eval and training rosters and reaches
        `choose_random_singles_move` on two real decision-path branches; `MaxBasePowerPlayer` on
        one. Those calls were written in the CLASS form (`Player.choose_random_singles_move(
        battle)`), which by construction cannot carry a per-instance stream — so seeding the
        player would have left exactly these branches coupled. Both methods now take the same
        descriptor."""
        from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer

        for cls in (SimpleHeuristicsPlayer, MaxBasePowerPlayer):
            raw = cls.__dict__["choose_singles_move"]
            assert isinstance(raw, _player_mod._rng_aware_static), cls.__name__
            # Bound through an instance it carries THAT instance's stream ...
            bot = cls.__new__(cls)
            bot._choice_rng = random.Random(3)
            assert bot.choose_singles_move.keywords["rng"] is bot._choice_rng
            # ... and through the class it stays on the shared module, as it always was.
            assert cls.choose_singles_move.keywords["rng"] is random

    def test_an_instance_attribute_still_shadows_the_descriptor(self):
        """`opponents_test.py` does `p.choose_random_move = MagicMock(...)`. A DATA descriptor
        would have silently ignored that and quietly changed what those tests measure."""
        bot = _Bot()
        bot.choose_random_move = lambda battle: "mocked"
        assert bot.choose_move(_battle()) == "mocked"


# ---------------------------------------------------------------------------
# Seam 2 — the team draw (gen3_team_draw_rng_v1)
# ---------------------------------------------------------------------------

def _builder(seed=None, n=12, bias=None, bias_prob=0.0, pfsp="off"):
    """A `Gen3Teambuilder` past its constructor's team VALIDATION (which shells out to the node
    validator and is not what this file is about). Only the draw path is populated — and the RNG
    goes in through `_install_team_rng`, the same and only writer `__init__` uses."""
    tb = Gen3Teambuilder.__new__(Gen3Teambuilder)
    tb.packed_teams = [f"team{i}" for i in range(n)]
    tb._pool_index_by_packed = {t: i for i, t in enumerate(tb.packed_teams)}
    tb.bias_packed_teams = list(bias) if bias else []
    tb.bias_prob = bias_prob
    tb._team_pfsp = pfsp
    tb._tp_weights = None
    tb._last_pool_idx = None
    tb._block_episodes = 1
    tb._block_cached = None
    tb._block_cached_idx = None
    tb._block_left = 0
    _tb_mod._install_team_rng(tb, seed)
    return tb


def _draws(tb, jitter_seed, n=40):
    random.seed(jitter_seed)
    out = []
    for _ in range(n):
        for _ in range(random.randint(1, 5)):
            random.random()
        out.append(tb.yield_team())
    return out


class TestTeamDrawRng:
    """Coupling here is not noise on one decision — the two arms play different GAMES."""

    def test_the_class_attribute_is_the_module(self):
        assert Gen3Teambuilder._rng is random
        assert "_rng" not in _builder().__dict__

    def test_the_unseeded_uniform_draw_is_byte_identical(self):
        """The `team_pfsp="off"` branch is the documented byte-identity baseline: exactly one RNG
        call, the same `random.choice`, on the same stream."""
        tb = _builder()
        random.seed(4)
        got = [tb.yield_team() for _ in range(25)]
        random.seed(4)
        assert got == [random.choice(tb.packed_teams) for _ in range(25)]

    def test_the_bias_branch_consumes_the_same_two_calls_in_the_same_order(self):
        """The bias branch spends a coin AND a choice; swapping the stream must not reorder or
        drop either, or the `bias_prob` a run was configured with stops being the rate it gets."""
        tb = _builder(bias=["biasA", "biasB"], bias_prob=0.5)
        random.seed(17)
        got = [tb.yield_team() for _ in range(30)]
        random.seed(17)
        expected = []
        for _ in range(30):
            if random.random() < 0.5:
                expected.append(random.choice(tb.bias_packed_teams))
            else:
                expected.append(random.choice(tb.packed_teams))
        assert got == expected
        assert {"biasA", "biasB"} & set(got)      # the bias branch really fired

    def test_two_paired_arms_with_seeded_builders_draw_the_same_teams(self):
        assert _draws(_builder(seed=1234), 0) == _draws(_builder(seed=1234), 999)

    def test_the_unseeded_default_is_the_one_that_couples(self):
        """REVERT-VERIFICATION — and the sharpest statement of the stake: unseeded, the two arms
        are handed different teams."""
        assert _draws(_builder(), 0) != _draws(_builder(), 999)

    def test_the_pfsp_weighted_branch_is_seeded_too(self):
        arm_a, arm_b = _builder(seed=5, pfsp="var"), _builder(seed=5, pfsp="var")
        arm_a._tp_weights = arm_b._tp_weights = [1.0 + i for i in range(12)]
        assert _draws(arm_a, 0) == _draws(arm_b, 999)

    def test_the_env_hook_reaches_a_builder_built_without_a_seed_kwarg(self):
        with patch.dict(os.environ, {"GEN3AI_TEAM_SEED": "44"}):
            arm_a, arm_b = _builder(), _builder()
        assert _draws(arm_a, 0) == _draws(arm_b, 999)


# ---------------------------------------------------------------------------
# Seam 3 — the self-play pool draw (gen3_pool_sample_rng_v1)
# ---------------------------------------------------------------------------

def _pool(seed=None, n=6):
    """A `SnapshotPool` past its directory scan — `sample()` reads only these three fields. The
    RNG goes in through `_install_pool_rng`, the same and only writer `__init__` uses."""
    p = SnapshotPool.__new__(SnapshotPool)
    p._entries = [SnapshotEntry(path=None, step=1000 * (i + 1)) for i in range(n)]
    p.recency_weight = 0.3
    p.pfsp_scale = 0.0
    _pool_mod._install_pool_rng(p, seed)
    return p


def _samples(pool, jitter_seed, n=40):
    random.seed(jitter_seed)
    out = []
    for _ in range(n):
        for _ in range(random.randint(1, 5)):
            random.random()
        out.append(pool.sample().step)
    return out


class TestSnapshotPoolRng:
    """An INTERNAL INCONSISTENCY as much as a coupling: the only caller of `sample()` is
    `MaskableAgentWrapper`, which already owns a per-env `random.Random(rng_seed)` for *which
    bucket* it picks — and then reached into the global module for *which snapshot*. So a wrapper
    that looks seeded was not reproducible."""

    def test_the_class_attribute_is_the_module(self):
        assert SnapshotPool._rng is random
        assert "_rng" not in _pool().__dict__

    def test_the_unseeded_draw_is_byte_identical(self):
        pool = _pool()
        weights = [pool.entry_weight(e) for e in pool._entries]
        random.seed(6)
        got = [pool.sample().step for _ in range(25)]
        random.seed(6)
        expected = [random.choices(pool._entries, weights=weights, k=1)[0].step
                    for _ in range(25)]
        assert got == expected

    def test_two_paired_arms_with_seeded_pools_face_the_same_selves(self):
        assert _samples(_pool(seed=1234), 0) == _samples(_pool(seed=1234), 999)

    def test_the_unseeded_default_is_the_one_that_couples(self):
        """REVERT-VERIFICATION."""
        assert _samples(_pool(), 0) != _samples(_pool(), 999)

    def test_the_env_hook_reaches_a_pool_built_without_a_seed_kwarg(self):
        with patch.dict(os.environ, {"GEN3AI_POOL_SEED": "21"}):
            arm_a, arm_b = _pool(), _pool()
        assert _samples(arm_a, 0) == _samples(arm_b, 999)


# ---------------------------------------------------------------------------
# Seam 4 — the policy's action sample (gen3_policy_sample_rng_v1)
# ---------------------------------------------------------------------------
# Same genre, different RNG: `Categorical.sample()` draws from TORCH's process-wide default
# generator. Self-play puts two `RLPlayer`s in ONE battle, interleaved by the bridge, and
# `stochastic=True` is the default for the pool opponents and the stable cross-run opponents —
# so this is not a conditional coin, it is every decision on both sides.

class TestPolicySampleRng:

    @staticmethod
    def _logits():
        torch.manual_seed(0)
        return torch.randn(1, 11)

    @staticmethod
    def _player(policy_seed=None):
        """Deliberately does NOT set `_policy_gens` — this is the `cls.__new__` shape, the harder
        case, and it exercises the class-level default plus the `setdefault` guard that keeps that
        default from becoming shared state."""
        p = RLPlayer.__new__(RLPlayer)
        p._policy_seed = _resolve_policy_seed(policy_seed)
        return p

    def test_no_seed_means_use_the_shared_default_generator(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _resolve_policy_seed(None) is None
            assert self._player()._policy_generator("cpu") is None

    def test_the_env_hook_and_the_explicit_kwarg_both_seed(self):
        with patch.dict(os.environ, {"GEN3AI_POLICY_SEED": "12"}):
            assert _resolve_policy_seed(None) == 12
            assert _resolve_policy_seed(4) == 4          # explicit beats the env
        with patch.dict(os.environ, {"GEN3AI_POLICY_SEED": ""}):
            assert _resolve_policy_seed(None) is None

    def test_an_unparseable_env_seed_raises_rather_than_falling_back(self):
        with patch.dict(os.environ, {"GEN3AI_POLICY_SEED": "nope"}):
            with pytest.raises(ValueError, match="GEN3AI_POLICY_SEED"):
                _resolve_policy_seed(None)

    def test_the_two_sampling_branches_are_the_same_draw(self):
        """THE default-unchanged claim, and the reason the fix samples from ``cat.probs`` rather
        than re-deriving a softmax: under one seed the seeded branch's call and the untouched
        ``Categorical.sample()`` produce the same index, so switching generators is the ONLY
        difference between them — no last-bit drift in the probabilities."""
        cat = torch.distributions.Categorical(logits=self._logits())
        for seed in (0, 1, 7, 99):
            torch.manual_seed(seed)
            a = cat.sample().item()
            torch.manual_seed(seed)
            b = torch.multinomial(cat.probs, 1, True).item()
            assert a == b

    def test_two_paired_arms_with_seeded_players_sample_identically(self):
        """THE regression test, in torch: the two arms burn DIFFERENT amounts of the shared torch
        stream between decisions (the searched arm's extra forwards do exactly this)."""
        cat = torch.distributions.Categorical(logits=self._logits())

        def draws(player, jitter):
            torch.manual_seed(jitter)
            out = []
            for _ in range(30):
                torch.randn(int(torch.randint(1, 6, (1,)).item()))   # the other arm's traffic
                gen = player._policy_generator(cat.probs.device)
                out.append(torch.multinomial(cat.probs, 1, True, generator=gen).item()
                           if gen is not None else cat.sample().item())
            return out

        assert draws(self._player(1234), 0) == draws(self._player(1234), 999)

    def test_the_unseeded_default_is_the_one_that_couples(self):
        """REVERT-VERIFICATION."""
        cat = torch.distributions.Categorical(logits=self._logits())

        def draws(jitter):
            torch.manual_seed(jitter)
            out = []
            for _ in range(30):
                torch.randn(int(torch.randint(1, 6, (1,)).item()))
                out.append(cat.sample().item())
            return out

        assert draws(0) != draws(999)

    def test_the_generator_is_lazy_and_per_device(self):
        """An unseeded player must allocate nothing (this runs in every env worker), and a seeded
        one needs a generator per device — an opponent can be built on cpu while a trainee samples
        on cuda, and a generator must live on its tensor's device."""
        assert self._player()._policy_gens == {}
        p = self._player(5)
        assert p._policy_gens == {}
        g1, g2 = p._policy_generator("cpu"), p._policy_generator("cpu")
        assert g1 is g2 and set(p._policy_gens) == {"cpu"}
        # ... and the class-level empty default was never mutated into shared state.
        assert RLPlayer._policy_gens == {}
        assert self._player(6)._policy_gens == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
