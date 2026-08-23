"""Unit tests for gen3_bait_entropy_v1 — the state-conditioned entropy boost on BAIT-opportunity
decisions: the env's `_bait_opportunity` predicate, the shared boost-anneal schedule, and the loss-side
weighting (including its composition with the defensive boost) on the REAL `train()` path.

The probe this serves (`ledger.md` 2026-08-23, the E4 verdict): the bait verdict's stated mechanism is
*exploration starvation at a saturated action* — the whiff sits at p≈0.97, so the alternatives at
p≈0.01-0.03 are never sampled. This flag + boost is the sampling-side test of exactly that claim, so the
tests that matter are (a) does the predicate fire on the boards the offline detector calls baits (the
cross-check lives in `bait_opportunity_integration_test.py`, which needs real battles), and (b) is the
weighting exactly what it says it is — OFF is off, and ON scales only the flagged rows.

The immunity table itself is NOT re-tested here: the predicate defers to `baitbot.blocks` →
`gen3_mechanics.effective_multiplier` → `data/`, pinned against ground truth (with negative controls) in
`agents/baitbot_test.py`. What IS tested here is everything the env adds on top of it — WHICH move is
the candidate, WHICH opponent mons are eligible, and that the hot path never raises.
"""
import types

import numpy as np
import pytest
import torch as th

from agents.enums import PokemonType as T
from agents.training.gen3_env import Gen3Env, _bait_candidate_attack
from agents.training.instrumented_ppo import InstrumentedMaskablePPO as PPO


# ----------------------------------------------------------------- the env predicate
def _move(mid, bp=100, mtype=T.GROUND):
    return types.SimpleNamespace(id=mid, base_power=bp, type=mtype)


def _mon(t1, t2=None, ability=None, fainted=False, active=False):
    """The surface `effective_multiplier` + the predicate read off a real poke-env Pokemon."""
    return types.SimpleNamespace(type_1=t1, type_2=t2, ability=ability, status=None,
                                 fainted=fainted, active=active)


def _flag(moves, bench, *, last_move=None, opp_active=None):
    active = types.SimpleNamespace(last_move=last_move)
    opp_active = opp_active if opp_active is not None else _mon(T.NORMAL, active=True)
    team = {f"p2: m{i}": m for i, m in enumerate([opp_active, *bench])}
    b = types.SimpleNamespace(active_pokemon=active, available_moves=list(moves),
                              opponent_team=team, opponent_active_pokemon=opp_active)
    return Gen3Env._bait_opportunity(types.SimpleNamespace(battle1=b))


def test_type_immunity_on_the_bench_fires():
    """The canonical gen-15 loop: Earthquake, with a Salamence sitting on their bench."""
    salamence = _mon(T.DRAGON, T.FLYING)
    assert _flag([_move("earthquake", 100, T.GROUND)], [salamence]) == 1.0
    # A bench mon that TAKES the move is not a bait, however bulky.
    assert _flag([_move("earthquake", 100, T.GROUND)], [_mon(T.STEEL)]) == 0.0


@pytest.mark.parametrize("ability,move_type,types", [
    ("levitate",    T.GROUND,   (T.DRAGON, T.GROUND)),   # Flygon — Ground/Ground, so only the ability blocks
    ("waterabsorb", T.WATER,    (T.WATER, None)),
    ("voltabsorb",  T.ELECTRIC, (T.ELECTRIC, None)),
    ("flashfire",   T.FIRE,     (T.FIRE, T.FLYING)),
])
def test_each_gen3_ability_immunity_fires_once_revealed(ability, move_type, types):
    """The four gen-3 absorb abilities, through the env predicate. Each pair is chosen so the TYPE chart
    alone does NOT give a zero — the same mon without the ability is the negative control below, so
    "everything is immune" cannot pass."""
    t1, t2 = types
    assert _flag([_move("m", 100, move_type)], [_mon(t1, t2, ability=ability)]) == 1.0
    # Ability UNREVEALED (poke-env leaves `ability` None until it is) → NOT a flagged bait. This is the
    # documented scope: the flag uses the same public information the policy holds.
    assert _flag([_move("m", 100, move_type)], [_mon(t1, t2, ability=None)]) == 0.0


def test_only_ALIVE_BENCH_opponents_count():
    salamence = _mon(T.DRAGON, T.FLYING)
    eq = [_move("earthquake", 100, T.GROUND)]
    assert _flag(eq, [salamence]) == 1.0
    # Fainted → it can never arrive, so it is not an opportunity.
    assert _flag(eq, [_mon(T.DRAGON, T.FLYING, fainted=True)]) == 0.0
    # ACTIVE, not bench: the whiff is already on the board — a different (and already-punished)
    # decision. The bait predicate is about the pivot that has not happened yet.
    assert _flag(eq, [], opp_active=_mon(T.DRAGON, T.FLYING, active=True)) == 0.0
    # The `active` FLAG is honoured too, not just identity with opponent_active_pokemon.
    assert _flag(eq, [_mon(T.DRAGON, T.FLYING, active=True)]) == 0.0


def test_no_attack_no_flag():
    salamence = _mon(T.DRAGON, T.FLYING)
    assert _flag([], [salamence]) == 0.0                                  # forced switch → no moves
    assert _flag([_move("toxic", 0, T.POISON)], [salamence]) == 0.0        # status moves are not baits
    assert _flag([_move("roar", None, T.NORMAL)], [salamence]) == 0.0


def test_the_candidate_is_the_re_click_when_the_last_move_is_still_legal():
    """The re-click is the sharpest form of the pathology, so a still-legal last damaging move wins over
    the bigger attack — and that changes the answer in BOTH directions."""
    eq = _move("earthquake", 100, T.GROUND)
    surf = _move("surf", 95, T.WATER)
    flyer = _mon(T.DRAGON, T.FLYING)          # immune to Ground, not to Water
    # Last move = the blocked one ⇒ flagged even though the max-BP move happens to be the same here.
    assert _flag([eq, surf], [flyer], last_move=eq) == 1.0
    # Last move = the move that WORKS ⇒ not flagged, even though a blocked attack is also legal.
    assert _flag([eq, surf], [flyer], last_move=surf) == 0.0
    # A last move that is no longer legal (Disable / 0 PP) falls back to the best legal attack.
    assert _flag([eq, surf], [flyer], last_move=_move("hiddenpowerice", 70, T.ICE)) == 1.0


def test_candidate_selection_units():
    eq = _move("earthquake", 100, T.GROUND)
    hp = _move("hiddenpowerice", 70, T.ICE)
    status = _move("toxic", 0, T.POISON)
    active = types.SimpleNamespace(last_move=None)
    assert _bait_candidate_attack(active, [hp, eq]) is eq                 # highest base power
    assert _bait_candidate_attack(active, [status]) is None               # no damaging move
    assert _bait_candidate_attack(active, []) is None
    # A last move with no base power (Recover) is not a candidate — fall through to the best attack.
    assert _bait_candidate_attack(types.SimpleNamespace(last_move=status), [hp, eq]) is eq
    assert _bait_candidate_attack(types.SimpleNamespace(last_move=hp), [hp, eq]) is hp


def test_flag_never_raises_on_garbage():
    """Hot path: a malformed battle must yield 0.0, never crash."""
    assert Gen3Env._bait_opportunity(types.SimpleNamespace(battle1=None)) == 0.0
    bad = types.SimpleNamespace(active_pokemon=object(), available_moves=[object()],
                                opponent_team={"a": object()}, opponent_active_pokemon=None)
    assert Gen3Env._bait_opportunity(types.SimpleNamespace(battle1=bad)) == 0.0
    # A revealed mon with no type surface at all (a poke-env shape we do not expect) still yields 0.0.
    half = types.SimpleNamespace(active_pokemon=types.SimpleNamespace(last_move=None),
                                 available_moves=[_move("earthquake")],
                                 opponent_team={"a": object()}, opponent_active_pokemon=None)
    assert Gen3Env._bait_opportunity(types.SimpleNamespace(battle1=half)) == 0.0


# ----------------------------------------------------------------- the boost-anneal schedule
def _beff(B, af, progress_remaining):
    fs = types.SimpleNamespace(bait_entropy_boost=B, bait_entropy_anneal_frac=af,
                               _current_progress_remaining=progress_remaining)
    return PPO._bait_entropy_boost_eff(fs)


def test_boost_constant_when_off_or_no_anneal():
    assert _beff(1.0, 0.0, 1.0) == 1.0          # OFF
    assert _beff(1.0, 0.5, 0.5) == 1.0          # OFF stays off regardless of anneal
    assert _beff(3.0, 0.0, 0.5) == 3.0          # constant boost (no anneal)


def test_boost_anneals_linearly_to_one():
    # B=3, anneal over the first 50% of training. progress_remaining 1.0→0.0 ⇒ done 0.0→1.0.
    assert abs(_beff(3.0, 0.5, 1.00) - 3.0) < 1e-6     # start (done 0.0)
    assert abs(_beff(3.0, 0.5, 0.75) - 2.0) < 1e-6     # done 0.25 (halfway through the anneal)
    assert abs(_beff(3.0, 0.5, 0.50) - 1.0) < 1e-6     # done 0.50 (anneal complete)
    assert abs(_beff(3.0, 0.5, 0.20) - 1.0) < 1e-6     # past the anneal → stays 1.0


def test_the_two_boosts_share_ONE_schedule():
    """Defensive and bait must anneal identically — one function, so they cannot drift apart."""
    fs = types.SimpleNamespace(bait_entropy_boost=4.0, bait_entropy_anneal_frac=0.8,
                               defensive_entropy_boost=4.0, defensive_entropy_anneal_frac=0.8,
                               _current_progress_remaining=0.6)
    assert PPO._bait_entropy_boost_eff(fs) == PPO._defensive_entropy_boost_eff(fs)


# ----------------------------------------------------------------- the weighting invariant
def test_weighting_is_identity_off_flag_and_off_boost():
    """The exact loss-side expression: a weight of 1 off-flag (and everywhere at boost 1) ⇒ unweighted."""
    ent = th.rand(16)
    flag = (th.arange(16) % 2).float()
    unweighted = -th.mean(ent)
    assert th.allclose(-th.mean((1.0 + (1.0 - 1.0) * flag) * ent), unweighted)          # boost 1 = off
    assert th.allclose(-th.mean((1.0 + (3.0 - 1.0) * th.zeros(16)) * ent), unweighted)  # no flagged rows
    assert float(-(-th.mean((1.0 + (3.0 - 1.0) * flag) * ent))) > float(-unweighted)    # boost 3 ⇒ more bonus


def test_the_two_boosts_compose_multiplicatively_and_each_is_inert_alone():
    """Overlap semantics, stated as arithmetic: the weights MULTIPLY, and each factor is exactly 1 off
    its own flag — so turning one on cannot change the weight the other assigns."""
    d_flag = th.tensor([1.0, 1.0, 0.0, 0.0])
    b_flag = th.tensor([1.0, 0.0, 1.0, 0.0])
    w = (1.0 + (3.0 - 1.0) * d_flag) * (1.0 + (5.0 - 1.0) * b_flag)
    assert th.allclose(w, th.tensor([15.0, 3.0, 5.0, 1.0]))     # both / defensive / bait / neither
    # Bait OFF ⇒ the composed weight is exactly the defensive one.
    assert th.allclose((1.0 + (3.0 - 1.0) * d_flag) * (1.0 + (1.0 - 1.0) * b_flag),
                       1.0 + (3.0 - 1.0) * d_flag)


# ----------------------------------------------------------------- the real train() path
class _BaitDictEnv(__import__("gymnasium").Env):
    """Tiny Dict-obs maskable env carrying the training-only `bait_opportunity` key at a CONSTANT value,
    so a minibatch is either all-flagged or none-flagged and the weighting is exactly predictable.
    Mirrors `instrumented_ppo_test._CounterDictEnv`; defined here so this file is self-contained."""

    def __init__(self, flag=1.0, ep_len=1000):
        super().__init__()
        from gymnasium import spaces
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1e4, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(2,), dtype=np.int8),
            "bait_opportunity": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(2)
        self._flag, self._ep_len, self._t = float(flag), ep_len, 0

    def action_masks(self):
        return np.ones(2, dtype=np.int8)

    def _obs(self):
        return {"observation": np.array([float(self._t % 17)], dtype=np.float32),
                "action_mask": np.ones(2, dtype=np.int8),
                "bait_opportunity": np.array([self._flag], dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        return self._obs(), float((self._t * 7) % 5), self._t >= self._ep_len, False, {}


def _bait_ppo(flag, ent_coef, n_steps=8, n_envs=4):
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([(lambda: _BaitDictEnv(flag=flag)) for _ in range(n_envs)])
    return PPO("MultiInputPolicy", venv, n_steps=n_steps, batch_size=4, n_epochs=1,
               normalize_advantage=False,   # per-micro-batch adv-norm is the one non-identity
               ent_coef=ent_coef, vf_coef=0.5, device="cpu", seed=0)


def _arm(*, flag, boost, ent_coef=0.05, seed=123):
    """One FRESH model, one rollout, one `train()` → a snapshot of every parameter.

    Deliberately not "reset one model and re-train it": `train()` mutates the rollout buffer in place
    (PopArt rescales the returns), so a second call on the same buffer is NOT a repeat of the first and
    an arm-vs-arm comparison built that way measures the mutation. Two fresh models seeded identically
    collect a bit-identical rollout, which is the comparison this file needs."""
    model = _bait_ppo(flag=flag, ent_coef=ent_coef)
    model.bait_entropy_boost = boost
    np.random.seed(seed)          # the rollout buffer's get() permutation — identical across arms
    th.manual_seed(seed)
    model.learn(total_timesteps=8 * 4)
    np.random.seed(seed)
    th.manual_seed(seed)
    model.train()
    return model, {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


def _same(a, b):
    return all(th.equal(a[k], b[k]) for k in a)


def test_off_is_byte_identical_even_with_every_row_flagged():
    """B=1.0 on a buffer where EVERY decision carries `bait_opportunity`=1 ⇒ the unweighted entropy term,
    bit for bit. The populated-flag case is the one that matters: a guard keyed on the KEY's presence
    instead of the COEFFICIENT would pass an all-zero test."""
    _, off = _arm(flag=1.0, boost=1.0)
    _, off2 = _arm(flag=1.0, boost=1.0)
    assert _same(off, off2)                       # the harness itself is deterministic
    _, on = _arm(flag=1.0, boost=3.0)
    assert not _same(off, on)                     # …and the boost is not a no-op when it is on


def test_the_boost_cannot_touch_unflagged_decisions():
    """Every row `bait_opportunity`=0 ⇒ B=3 is byte-identical to B=1. This is the claim that the weight is
    `1 + (B-1)·flag` and not, say, `B` applied to the whole minibatch."""
    _, one = _arm(flag=0.0, boost=1.0)
    _, three = _arm(flag=0.0, boost=3.0)
    assert _same(one, three)


def test_an_all_flagged_boost_equals_scaling_ent_coef():
    """With every row flagged the weight is the constant B, so `(ent_coef=c, boost=B)` must produce the
    EXACT update of `(ent_coef=B·c, boost=1)`. An exact identity, so it pins the formula rather than its
    direction — a `B·flag` or a `(1+B)·flag` weight both fail it."""
    _, got = _arm(flag=1.0, boost=3.0, ent_coef=0.05)
    _, want = _arm(flag=1.0, boost=1.0, ent_coef=0.15)
    assert _same(got, want)


def test_metrics_are_emitted_only_when_the_boost_is_on():
    off_model, _ = _arm(flag=1.0, boost=1.0)
    assert not [k for k in off_model.logger.name_to_value if k.startswith("baitent/")]

    model, _ = _arm(flag=1.0, boost=3.0)
    rec = model.logger.name_to_value
    assert rec["baitent/flagged_frac"] == pytest.approx(1.0)
    assert rec["baitent/boost_eff"] == pytest.approx(3.0)
    assert "baitent/entropy_flagged" in rec
    # Every row is flagged here, so the unflagged bucket must be ABSENT rather than reported as 0.
    assert "baitent/entropy_unflagged" not in rec
    # The standard entropy metric stays UNWEIGHTED (the template's rule) — it must not carry the boost.
    assert "train/entropy_loss" in rec
