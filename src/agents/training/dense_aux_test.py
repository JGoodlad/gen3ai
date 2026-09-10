"""The DENSE AUXILIARY targets, masks and loss (`gen3_dense_aux_v1`, `--win-prob-dense-aux`).

The model half is `agents/model/dense_aux_head_test.py`. This file is about the LABELS: where the
25 numbers come from, which of them are scored, and the back-fill that carries them to every state.

The claim that needs pinning hardest is the MASK. A target with no feature correspondence is the
exact defect this arm exists to remove, so re-introducing one — by fabricating "alive at full HP"
for an opponent mon that was never revealed, or by scoring slot 4 at turn 1 when the state's own
observation has nothing in slot 4 — would make the whole build a more elaborate version of the
problem.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.model.dense_aux_head import (
    DENSE_AUX_DIM_OUT, DENSE_AUX_HP, DENSE_AUX_MAX_TURNS, DENSE_AUX_SURVIVAL, DENSE_AUX_TURNS,
)
from agents.training.dense_aux import (
    AUX_MASK_KEY, AUX_TARGET_KEY, AUX_TURN_KEY, ko_counts, scale_turns_left, state_visibility,
    terminal_facts,
)
from agents.training.dense_aux_callback import (
    INFO_MASK, INFO_TARGET, INFO_TURN, DenseAuxLabelCallback, DenseAuxScratch,
    _scale_turns_left_array,
)
from agents.training.instrumented_ppo.value_terms import ValueTerms

S0, H0, T0 = DENSE_AUX_SURVIVAL[0], DENSE_AUX_HP[0], DENSE_AUX_TURNS[0]


# ── toy battle ───────────────────────────────────────────────────────────────────────────────────

class _Mon:
    def __init__(self, hp: float):
        self._hp = float(hp)

    @property
    def fainted(self) -> bool:
        return self._hp <= 0.0

    @property
    def current_hp_fraction(self) -> float:
        return self._hp


class _Battle:
    """The two dicts `ObservationEncoder.get_team_list` reads, and the turn."""

    def __init__(self, ours, theirs, turn=0):
        self.team = {f"o{i}": m for i, m in enumerate(ours)}
        self.opponent_team = {f"t{i}": m for i, m in enumerate(theirs)}
        self.opponent_active_pokemon = theirs[0] if theirs else None
        self.turn = turn


def _six(*hps):
    return [_Mon(h) for h in hps]


# ── terminal facts ───────────────────────────────────────────────────────────────────────────────

def test_terminal_facts_read_survival_and_hp_in_the_observations_own_slot_order():
    b = _Battle(_six(1.0, 0.0, 0.42, 0.0, 0.9, 0.0),
                _six(0.0, 0.0, 0.31), turn=37)
    tgt, mask, turn = terminal_facts(b)
    assert turn == 37.0
    np.testing.assert_allclose(tgt[S0:S0 + 6], [1, 0, 1, 0, 1, 0])
    np.testing.assert_allclose(tgt[H0:H0 + 6], [1.0, 0.0, 0.42, 0.0, 0.9, 0.0], rtol=1e-6)
    np.testing.assert_allclose(tgt[S0 + 6:S0 + 12], [0, 0, 1, 0, 0, 0])
    np.testing.assert_allclose(tgt[H0 + 6:H0 + 12], [0.0, 0.0, 0.31, 0.0, 0.0, 0.0], rtol=1e-6)


def test_an_opponent_slot_that_was_never_revealed_is_MASKED_not_fabricated():
    """The load-bearing one. Three of their six never appeared; those slots have no observed
    fainted flag and no observed HP, and 'alive at full HP' would be a label that is wrong in a
    DIRECTION (the mons a beaten team never got to send out)."""
    b = _Battle(_six(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), _six(0.0, 0.0, 0.5), turn=20)
    _tgt, mask, _turn = terminal_facts(b)
    assert list(mask[S0:S0 + 6]) == [1.0] * 6
    assert list(mask[S0 + 6:S0 + 12]) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert list(mask[H0 + 6:H0 + 12]) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert mask[T0] == 1.0, "the terminal turn is an episode-level fact and is always available"


def test_a_fainted_mon_reports_zero_hp_from_the_same_read():
    """Both facts come from the survival branch, so a mid-faint accessor disagreement cannot make
    the two columns contradict each other."""
    class _Weird(_Mon):
        @property
        def current_hp_fraction(self):    # a fainted mon claiming HP
            return 0.7
    b = _Battle([_Weird(0.0)], [_Mon(1.0)], turn=3)
    tgt, _m, _t = terminal_facts(b)
    assert tgt[S0] == 0.0 and tgt[H0] == 0.0


def test_no_battle_returns_None_rather_than_a_zero_block():
    assert terminal_facts(None) is None


# ── per-state visibility ─────────────────────────────────────────────────────────────────────────

def test_visibility_follows_the_opponents_REVEAL_prefix():
    """Opponent slot order IS reveal order, so slot i at turn t is the same mon as slot i at
    termination for every i below the reveal count — and nothing at all above it."""
    early = state_visibility(_Battle(_six(*[1.0] * 6), _six(1.0), turn=1))
    assert list(early[S0 + 6:S0 + 12]) == [1.0, 0, 0, 0, 0, 0]
    assert list(early[S0:S0 + 6]) == [1.0] * 6
    later = state_visibility(_Battle(_six(*[1.0] * 6), _six(1.0, 1.0, 1.0), turn=9))
    assert list(later[S0 + 6:S0 + 12]) == [1.0, 1.0, 1.0, 0, 0, 0]
    # A PREFIX, always: an earlier state's visible set is contained in a later one's.
    assert np.all(later >= early)


def test_the_turns_output_is_visible_even_with_no_battle():
    assert state_visibility(None)[T0] == 1.0
    assert float(state_visibility(None)[:T0].sum()) == 0.0


# ── the turns-left scale ─────────────────────────────────────────────────────────────────────────

def test_the_two_spellings_of_the_turn_scale_agree():
    """The callback uses a vectorised copy over ~130k rows; two spellings of one scale is exactly
    how a target and its documentation drift apart."""
    xs = np.array([0.0, 1.0, 7.0, 60.0, 249.0, 250.0, 900.0], dtype=np.float32)
    np.testing.assert_allclose(
        _scale_turns_left_array(xs), [scale_turns_left(float(x)) for x in xs], rtol=1e-6)


def test_the_scale_is_monotone_bounded_and_zero_at_the_terminal_state():
    assert scale_turns_left(0.0) == 0.0
    assert scale_turns_left(-5.0) == 0.0
    assert scale_turns_left(DENSE_AUX_MAX_TURNS) == pytest.approx(1.0)
    assert scale_turns_left(10_000.0) == 1.0
    assert scale_turns_left(3.0) < scale_turns_left(30.0) < scale_turns_left(120.0)


# ── the back-fill ────────────────────────────────────────────────────────────────────────────────

class _Buf:
    def __init__(self, n_steps, n_envs, episode_starts):
        self.episode_starts = np.asarray(episode_starts, dtype=np.float32)
        self.observations = {
            AUX_TARGET_KEY: np.zeros((n_steps, n_envs, DENSE_AUX_DIM_OUT), dtype=np.float32),
            AUX_MASK_KEY: np.ones((n_steps, n_envs, DENSE_AUX_DIM_OUT), dtype=np.float32),
            AUX_TURN_KEY: np.zeros((n_steps, n_envs, 1), dtype=np.float32),
        }


class _Model:
    def __init__(self, buf, n_steps, n_envs):
        self.rollout_buffer = buf
        self.n_steps, self.n_envs = n_steps, n_envs
        self._dense_aux_scratch = DenseAuxScratch(n_steps, n_envs)


def _run_backfill(model):
    cb = DenseAuxLabelCallback()
    cb.model = model
    cb._on_rollout_end()
    return model.rollout_buffer.observations


def test_the_backfill_matches_the_win_bits_backfill_and_masks_the_trailing_episode():
    """Two episodes in one env: rows 0-2 end at row 2, rows 3-5 are still running. The finished
    episode's rows all carry ITS facts; the trailing rows carry nothing and are masked — never
    trained toward a fabricated label, exactly as `win_mask` does it."""
    n_steps, n_envs = 6, 1
    es = np.zeros((n_steps, n_envs), dtype=np.float32)
    es[3, 0] = 1.0                                   # a new episode begins at row 3
    m = _Model(_Buf(n_steps, n_envs, es), n_steps, n_envs)
    for t in range(n_steps):
        m.rollout_buffer.observations[AUX_TURN_KEY][t, 0, 0] = float(t)
    facts = terminal_facts(_Battle(_six(1.0, 0.0, 0.5, 1.0, 1.0, 1.0),
                                   _six(0.0, 0.0, 0.0, 0.25), turn=2))
    m._dense_aux_scratch.record(2, 0, {INFO_TARGET: facts[0], INFO_MASK: facts[1],
                                       INFO_TURN: facts[2]})
    obs = _run_backfill(m)
    at, am = obs[AUX_TARGET_KEY], obs[AUX_MASK_KEY]
    for t in (0, 1, 2):
        np.testing.assert_allclose(at[t, 0, S0:S0 + 12], facts[0][S0:S0 + 12])
        assert am[t, 0, S0] == 1.0
    for t in (3, 4, 5):
        assert float(am[t, 0].max()) == 0.0, "the trailing in-progress episode must carry no label"


def test_turns_left_is_per_state_and_zero_at_the_terminal_row():
    n_steps, n_envs = 4, 1
    m = _Model(_Buf(n_steps, n_envs, np.zeros((n_steps, n_envs), np.float32)), n_steps, n_envs)
    for t in range(n_steps):
        m.rollout_buffer.observations[AUX_TURN_KEY][t, 0, 0] = float(t * 3)   # turns 0, 3, 6, 9
    facts = terminal_facts(_Battle(_six(*[1.0] * 6), _six(1.0), turn=9))
    m._dense_aux_scratch.record(3, 0, {INFO_TARGET: facts[0], INFO_MASK: facts[1],
                                       INFO_TURN: facts[2]})
    at = _run_backfill(m)[AUX_TARGET_KEY]
    np.testing.assert_allclose(
        at[:, 0, T0],
        [scale_turns_left(9.0), scale_turns_left(6.0), scale_turns_left(3.0), 0.0], rtol=1e-6)


def test_the_mask_is_the_THREE_WAY_and():
    """Known-episode AND terminal-availability AND per-state visibility. Row 0 sees only one
    opponent mon; the battle ends with four revealed. Slot 1 has a terminal fact but is invisible
    at row 0, so it is not scored THERE and is scored at row 1 — which is the whole point."""
    n_steps, n_envs = 2, 1
    buf = _Buf(n_steps, n_envs, np.zeros((n_steps, n_envs), np.float32))
    buf.observations[AUX_MASK_KEY][0, 0] = state_visibility(
        _Battle(_six(*[1.0] * 6), _six(1.0)))
    buf.observations[AUX_MASK_KEY][1, 0] = state_visibility(
        _Battle(_six(*[1.0] * 6), _six(1.0, 1.0)))
    m = _Model(buf, n_steps, n_envs)
    facts = terminal_facts(_Battle(_six(*[1.0] * 6), _six(0.0, 0.0, 0.0, 0.4), turn=11))
    m._dense_aux_scratch.record(1, 0, {INFO_TARGET: facts[0], INFO_MASK: facts[1],
                                       INFO_TURN: facts[2]})
    am = _run_backfill(m)[AUX_MASK_KEY]
    assert list(am[0, 0, S0 + 6:S0 + 12]) == [1.0, 0, 0, 0, 0, 0]      # visibility binds
    assert list(am[1, 0, S0 + 6:S0 + 12]) == [1.0, 1.0, 0, 0, 0, 0]
    # slot 4/5 have NO terminal fact at all, so they are unscored on every row of the episode.
    assert float(am[:, 0, S0 + 10:S0 + 12].max()) == 0.0


def test_the_plumbing_meters_are_published_even_when_nothing_was_labelled():
    """An absent `win_prob/aux_*` family must mean "the flag is off", and nothing else — the
    lesson the strata build paid one smoke to learn."""
    n_steps, n_envs = 3, 2
    m = _Model(_Buf(n_steps, n_envs, np.zeros((n_steps, n_envs), np.float32)), n_steps, n_envs)
    _run_backfill(m)
    met = m._dense_aux_metrics
    assert met["aux_active"] == 1.0 and met["aux_coverage"] == 0.0
    assert met["aux_masked_frac"] == 1.0


# ── the λ precedence ─────────────────────────────────────────────────────────────────────────────

def test_the_lambda_recursion_touches_no_aux_key():
    """These are TERMINAL FACTS, not returns: there is no recorded per-state estimate of 'slot 4's
    final HP' for a λ recursion to blend, and blending them would make the target a mixture of an
    outcome and a prediction of a different quantity. The precedence is STRUCTURAL — the recursion
    writes only `win_target`/`win_mask` — and this is the source scan that keeps it so."""
    import inspect

    from agents.training import win_prob_callback as wpc
    src = inspect.getsource(wpc)
    for key in (AUX_TARGET_KEY, AUX_MASK_KEY, AUX_TURN_KEY):
        assert key not in src, (
            f"the win-prob/λ callback names {key!r}; the dense-aux targets are end-of-battle "
            "FACTS and must not be λ-blended")


# ── the loss ─────────────────────────────────────────────────────────────────────────────────────

def _rows(n=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, DENSE_AUX_DIM_OUT, generator=g)
    t[:, S0:S0 + 12] = (t[:, S0:S0 + 12] > 0.5).float()
    return t


def test_a_fully_masked_batch_scores_nothing_rather_than_NaN_poisoning_the_loss():
    assert ValueTerms._dense_aux_loss(
        torch.zeros(8, DENSE_AUX_DIM_OUT), _rows(8), torch.zeros(8, DENSE_AUX_DIM_OUT)) is None
    assert ValueTerms._dense_aux_loss(None, None, None) is None


def test_masked_slots_contribute_NOTHING_to_the_loss():
    """The proof that the mask is applied and not merely carried: poisoning the target under a
    masked slot must leave the loss bit-identical."""
    t = _rows()
    mask = torch.ones_like(t)
    mask[:, S0 + 9:S0 + 12] = 0.0
    mask[:, H0 + 9:H0 + 12] = 0.0
    logits = torch.randn(t.shape[0], DENSE_AUX_DIM_OUT, generator=torch.Generator().manual_seed(2))
    a, _ = ValueTerms._dense_aux_loss(logits, t, mask)
    poisoned = t.clone()
    poisoned[:, S0 + 9:S0 + 12] = 1.0 - poisoned[:, S0 + 9:S0 + 12]
    poisoned[:, H0 + 9:H0 + 12] = 0.123
    b, _ = ValueTerms._dense_aux_loss(logits, poisoned, mask)
    assert torch.equal(a, b)


def test_the_total_is_the_MEAN_of_the_three_terms():
    """Not a pooled mean over 25 columns: twelve survival outputs must not outvote the one
    turns-left output twelve to one."""
    t = _rows()
    loss, m = ValueTerms._dense_aux_loss(
        torch.randn(t.shape[0], DENSE_AUX_DIM_OUT, generator=torch.Generator().manual_seed(5)),
        t, torch.ones_like(t))
    assert float(loss) == pytest.approx(
        (m["aux_survival_loss"] + m["aux_hp_loss"] + m["aux_turns_loss"]) / 3.0, rel=1e-5)


def test_a_block_with_no_unmasked_slot_is_DROPPED_not_scored_as_zero():
    t = _rows()
    mask = torch.ones_like(t)
    mask[:, T0] = 0.0
    _loss, m = ValueTerms._dense_aux_loss(torch.zeros_like(t), t, mask)
    assert "aux_turns_loss" not in m and "aux_survival_loss" in m


def test_the_survival_AUC_is_reported_per_SIDE_and_reads_one_half_at_zero_init():
    """A pooled AUC over all twelve slots would hide exactly the own-vs-opponent asymmetry the arm
    is built to move. Ties take their average rank, so an untrained head reads exactly 0.5."""
    t = _rows()
    _loss, m = ValueTerms._dense_aux_loss(torch.zeros_like(t), t, torch.ones_like(t))
    assert m["aux_auc_own"] == 0.5 and m["aux_auc_opp"] == 0.5
    # A head that predicts survival perfectly on OUR side only separates the two meters.
    logits = torch.zeros_like(t)
    logits[:, S0:S0 + 6] = (t[:, S0:S0 + 6] - 0.5) * 20.0
    _loss, m = ValueTerms._dense_aux_loss(logits, t, torch.ones_like(t))
    assert m["aux_auc_own"] == 1.0 and m["aux_auc_opp"] == 0.5


def test_an_AUC_with_one_class_absent_is_OMITTED_rather_than_logged():
    """A real state (an early rollout in which nothing of ours ever fainted). TensorBoard would
    draw a fabricated number as a measurement."""
    t = _rows()
    t[:, S0:S0 + 6] = 1.0
    _loss, m = ValueTerms._dense_aux_loss(torch.zeros_like(t), t, torch.ones_like(t))
    assert "aux_auc_own" not in m and "aux_auc_opp" in m


def test_the_ko_counts_are_derived_and_honour_the_mask():
    surv = np.array([1, 0, 0, 1, 1, 1] + [0, 0, 1, 1, 1, 1], dtype=np.float32)
    mask = np.ones(12, dtype=np.float32)
    assert ko_counts(surv, mask) == (2.0, 2.0)
    mask[6:9] = 0.0                       # three opponent slots unscored
    assert ko_counts(surv, mask) == (2.0, 0.0)


def test_the_hp_and_turns_MAE_are_published_beside_their_BCE():
    """A BCE against a SOFT target has a non-zero floor (the target's own entropy), so the loss
    numbers alone cannot say whether the head is good — the MAEs are the interpretable read."""
    t = _rows()
    _loss, m = ValueTerms._dense_aux_loss(torch.zeros_like(t), t, torch.ones_like(t))
    assert 0.0 < m["aux_hp_mae"] < 1.0 and 0.0 < m["aux_turns_mae"] < 1.0
    assert m["aux_ko_mae_own"] >= 0.0 and m["aux_ko_mae_opp"] >= 0.0
    assert m["aux_scored_frac"] == 1.0


# ── the CLI surface ──────────────────────────────────────────────────────────────────────────────

def test_the_flag_is_REFUSED_without_the_winprob_critic():
    """Under `--critic shaped` the win-prob head is a diagnostic and the value function is the
    scalar net, so the dense targets would enrich a readout and change nothing about V."""
    from main.train.combination_checks import COMBINATION_CHECKS
    row = next(c for c in COMBINATION_CHECKS if c.name == "dense_aux_needs_the_winprob_critic")

    class _A:
        win_prob_dense_aux = 1.0
        critic = "shaped"
    assert row.predicate(_A())
    _A.critic = "winprob"
    assert not row.predicate(_A())
    _A.critic, _A.win_prob_dense_aux = "shaped", 0.0
    assert not row.predicate(_A())


def test_the_coefficient_is_recorded_at_BOTH_model_version_construction_sites():
    """`lifecycle._run_roundtrip_test` and `model_build.build_and_train` each build a ModelVersion
    by hand. A coefficient added to one and not the other reads as 0.0 in half the run's records —
    the drift class the flag registry exists to end, in the one place it cannot reach."""
    import inspect

    from main.train import lifecycle, model_build
    assert "win_prob_dense_aux" in inspect.getsource(lifecycle._run_roundtrip_test)
    assert inspect.getsource(model_build).count("win_prob_dense_aux=args.win_prob_dense_aux") == 2


def test_the_dense_aux_keys_are_NOT_in_the_synthetic_obs_registry():
    """`extra_obs_keys` declares the keys the EXTRACTOR forward reads, so every synthetic-obs site
    must supply them. These three are LABEL keys the network never reads — putting them there
    would make five call sites build 25-float blocks for a consumer that does not exist."""
    from agents.model.extra_obs_keys import EXTRA_OBS_KEYS
    names = {row.key for row in EXTRA_OBS_KEYS}
    assert not names & {AUX_TARGET_KEY, AUX_MASK_KEY, AUX_TURN_KEY}
