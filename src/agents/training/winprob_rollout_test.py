"""Gates for `--win-prob-rollout-target` (`gen3_winprob_rollout_target_v1`, v118, critic arm 10).

THE LEVER, and why each gate below exists. Under `--critic winprob` the value loss is a BCE against
ONE terminal bit copied to every state of the episode: one bit about the GAME, none about the
STATE. `--win-prob-lambda` (arm 8) moved the network's own later estimates backward; this flag BUYS
NEW BITS — it replays a sampled state out of the `cf_records` ring, plays `R` continuations forward
with the current policy, and makes `wins / R` that state's target. Every failure mode of a target
rewrite is SILENT, and this one adds two more that are also EXPENSIVE (it spawns processes and it
blocks the training loop), so each is made unrepresentable rather than merely unlikely:

  * OFF must be BIT-identical — not approximately equal — or every arm run without the flag is a
    different experiment from the one recorded. 0.0 skips the whole path, including the handle
    capture, so an unflagged run does not even allocate the scratch.
  * The applied target must be EXACTLY `wins / R` on the sampled rows and EXACTLY the terminal bit
    everywhere else. A labeller that leaked a label onto a neighbouring row would be invisible.
  * The subsample must be SEEDED and reproducible, and at most ONE row per EPISODE SLICE — R
    rollouts from two states of one game are far more correlated than R from two games, which is
    the whole bits-per-state argument applied to the sample itself.
  * The per-rollout CAP must bind, because the cost is linear in the state count and it is paid as
    a stall on the training loop.
  * A state that produced NO finished continuation must keep its terminal bit. A fabricated label
    in the value objective is the one thing this subsystem must never do.
  * It must REFUSE without `--critic winprob` (the BCE is a diagnostic there) and without
    `--cf-records` (there is no replayable episode, so it would label zero states in silence).
  * λ PRECEDENCE: a labelled row is an ANCHOR the recursion terminates on, at outcome-weight 1.0 —
    a measured win fraction is a measurement OF that state, not a bootstrap off the network.
  * The values must be RECORDED and re-read on a flagless resume, or a launcher restart converts
    the arm back into its own control under the same run name (the v100 defect) — and here it
    would do it while the run got FASTER, which reads as a win rather than as a lost treatment.
  * The COST IDENTITY must be arithmetic a test can check, because it is the number that sets the
    arm's registered fraction.
"""
import numpy as np
import pytest
import torch

from agents.model.model_version.migrations import _migrate_config
from agents.training.win_prob_callback import WinProbLabelCallback, lambda_return_targets
from agents.training.win_prob_rollout import (BANKED_CONTINUATION_DECISIONS, DEFAULT_ROLLOUT_R,
                                              MAX_STATES_PER_ROLLOUT, ROLLOUT_MODES, ROLLOUT_OFF,
                                              SELECTOR_VERSION, apply_rollout_labels,
                                              budget_multiple, eligible_mask,
                                              fraction_for_unit_budget, index_records,
                                              n_states_for, record_key, rollout_metrics,
                                              select_rows, slice_ids)

MOVE_ROW = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]      # switches 0-5, moves 6-9, struggle 10
SWITCH_ROW = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]    # a forced switch: no move is legal


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE COST IDENTITY — the arithmetic that sets the arm's registered fraction.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_budget_multiple_IS_fraction_times_R_times_the_continuation_length():
    """`labelling decisions / collection decisions` = `fraction x R x mean_continuation`.

    Pinned because this identity is the whole cost story: it is what makes 1/32 at R = 8 cost ~26x
    the run's own simulation budget, and it is what the registered fraction is derived FROM.
    """
    n_steps, n_envs, R, arm = 2048, 48, 8, 104.0
    frac = 1.0 / 32.0
    n = int(frac * n_steps * n_envs)
    assert budget_multiple(n, R, arm, n_steps, n_envs) == pytest.approx(frac * R * arm, rel=1e-9)
    assert budget_multiple(n, R, arm, n_steps, n_envs) == pytest.approx(26.0, rel=0.01)


def test_the_unit_budget_fraction_is_ONE_over_R_times_the_continuation_length():
    """The fraction that costs exactly 1x — independent of the buffer shape, because both sides
    of the ratio scale with it. This is the number an arm registers."""
    assert fraction_for_unit_budget(8, 104.0) == pytest.approx(1.0 / 832.0)
    got = fraction_for_unit_budget(8, BANKED_CONTINUATION_DECISIONS)
    assert budget_multiple(int(got * 2048 * 48), 8, BANKED_CONTINUATION_DECISIONS,
                           2048, 48) == pytest.approx(1.0, abs=0.01)


def test_the_state_count_is_a_fraction_of_the_WHOLE_buffer_and_is_CAPPED():
    assert n_states_for(0.0, 2048, 48) == 0
    assert n_states_for(1.0 / 1024.0, 2048, 48) == 96
    # The cap binds long before a careless fraction can: 1/32 of a production buffer asks for 3,072.
    assert n_states_for(1.0 / 32.0, 2048, 48) == MAX_STATES_PER_ROLLOUT
    assert n_states_for(0.5, 2048, 48, cap=7) == 7


# ──────────────────────────────────────────────────────────────────────────────────────────────
# ELIGIBILITY + SELECTION (`winprob_rollout_select_v1`)
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _grid(n_steps=6, n_envs=2, move=True):
    wm = np.ones((n_steps, n_envs), dtype=np.float32)
    turns = np.full((n_steps, n_envs), 5, dtype=np.int64)
    handles = np.ones((n_steps, n_envs), dtype=bool)
    am = np.tile(np.asarray(MOVE_ROW if move else SWITCH_ROW, dtype=np.float32),
                 (n_steps, n_envs, 1))
    return wm, turns, handles, am


def test_the_four_eligibility_filters_each_EXCLUDE_on_their_own():
    wm, turns, handles, am = _grid()
    assert eligible_mask(wm, turns, handles, am).all()
    # 1. the episode did not terminate inside the buffer => no record on disk yet
    wm2 = wm.copy(); wm2[0, 0] = 0.0
    assert not eligible_mask(wm2, turns, handles, am)[0, 0]
    # 2. below cf_producer's MIN_LABELABLE_TURN
    t2 = turns.copy(); t2[1, 0] = 1
    assert not eligible_mask(wm, t2, handles, am)[1, 0]
    # 3. no reconstruction handle was captured for the row
    h2 = handles.copy(); h2[2, 0] = False
    assert not eligible_mask(wm, turns, h2, am)[2, 0]
    # 4. a mid-turn FORCED SWITCH: a counterfactual replay cuts at a TURN boundary, so labelling
    #    one would label that turn's move decision instead.
    a2 = am.copy(); a2[3, 0] = np.asarray(SWITCH_ROW, dtype=np.float32)
    assert not eligible_mask(wm, turns, h2, a2)[3, 0]


def test_a_forced_switch_only_buffer_yields_NOTHING_eligible():
    wm, turns, handles, am = _grid(move=False)
    assert not eligible_mask(wm, turns, handles, am).any()


def test_the_slice_ids_separate_two_episodes_of_one_env():
    es = np.asarray([[1], [0], [0], [1], [0], [0]], dtype=np.float32)
    assert slice_ids(es).ravel().tolist() == [1, 1, 1, 2, 2, 2]


def test_the_selection_takes_AT_MOST_ONE_row_per_episode_slice():
    """R rollouts from two states of one game are far more correlated than R from two games."""
    es = np.zeros((6, 2), dtype=np.float32)
    es[0, :] = 1.0
    es[3, :] = 1.0                                     # 2 envs x 2 slices = 4 slices
    el = np.ones((6, 2), dtype=bool)
    picks = select_rows(el, es, 12, np.random.default_rng(0))
    assert len(picks) == 4, "12 asked for, but only 4 slices exist"
    sids = slice_ids(es)
    assert len({(e, int(sids[t, e])) for t, e in picks}) == len(picks)


def test_the_selection_is_SEEDED_and_reproducible_and_moves_with_the_seed():
    es = np.zeros((20, 3), dtype=np.float32)
    es[0, :] = 1.0
    es[10, :] = 1.0
    el = np.ones((20, 3), dtype=bool)
    a = select_rows(el, es, 4, np.random.default_rng([7, 1]))
    b = select_rows(el, es, 4, np.random.default_rng([7, 1]))
    c = select_rows(el, es, 4, np.random.default_rng([7, 2]))
    assert a == b, "the same (run seed, rollout index) must replay the same sample"
    assert a != c, "a different rollout index must not draw the same stream"
    assert SELECTOR_VERSION == "winprob_rollout_select_v1"


def test_the_selection_NEVER_returns_an_ineligible_row():
    es = np.zeros((8, 2), dtype=np.float32)
    es[0, :] = 1.0
    el = np.zeros((8, 2), dtype=bool)
    el[3, 1] = True
    assert select_rows(el, es, 5, np.random.default_rng(0)) == [(3, 1)]
    assert select_rows(np.zeros((8, 2), dtype=bool), es, 5, np.random.default_rng(0)) == []
    assert select_rows(el, es, 0, np.random.default_rng(0)) == []


# ──────────────────────────────────────────────────────────────────────────────────────────────
# APPLYING THE LABEL
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _targets(shape=(4, 2)):
    wt = np.zeros(shape + (1,), dtype=np.float32)
    wm = np.ones(shape + (1,), dtype=np.float32)
    y = np.asarray([[1.0, 0.0]] * shape[0], dtype=np.float64)
    wt[:, :, 0] = y
    return wt, wm, y


def test_replace_gives_the_sampled_rows_EXACTLY_wins_over_R_and_changes_NOTHING_else():
    wt, wm, y = _targets()
    am, av, applied = apply_rollout_labels(wt, wm, [(1, 0), (2, 1)], [5 / 8, 3 / 8], "replace", y)
    assert wt[1, 0, 0] == pytest.approx(0.625) and wt[2, 1, 0] == pytest.approx(0.375)
    untouched = [(t, e) for t in range(4) for e in range(2) if (t, e) not in {(1, 0), (2, 1)}]
    for t, e in untouched:
        assert wt[t, e, 0] == pytest.approx(y[t, e]), "a non-sampled row keeps its terminal bit"
    assert am[1, 0] and am[2, 1] and am.sum() == 2
    assert av[1, 0] == pytest.approx(0.625)
    assert [a[:2] for a in applied] == [(1, 0), (2, 1)]


def test_blend_is_the_MEAN_of_the_rollout_fraction_and_the_terminal_bit():
    wt, wm, y = _targets()
    apply_rollout_labels(wt, wm, [(1, 0), (2, 1)], [0.25, 0.25], "blend", y)
    assert wt[1, 0, 0] == pytest.approx(0.625)       # (0.25 + 1) / 2, a WIN episode
    assert wt[2, 1, 0] == pytest.approx(0.125)       # (0.25 + 0) / 2, a LOSS episode


def test_a_state_with_NO_finished_continuation_keeps_its_terminal_bit():
    """The one thing this subsystem must never do is put a fabricated number in the objective."""
    wt, wm, y = _targets()
    am, _av, applied = apply_rollout_labels(wt, wm, [(1, 0), (2, 1)], [None, float("nan")],
                                            "replace", y)
    assert wt[1, 0, 0] == pytest.approx(1.0) and wt[2, 1, 0] == pytest.approx(0.0)
    assert applied == [] and not am.any()


def test_a_labelled_row_is_UNMASKED_into_the_loss():
    wt, wm, y = _targets()
    wm[:] = 0.0
    apply_rollout_labels(wt, wm, [(1, 0)], [0.5], "replace", y)
    assert wm[1, 0, 0] == 1.0


def test_an_unknown_mode_RAISES_rather_than_falling_back():
    wt, wm, y = _targets()
    with pytest.raises(ValueError):
        apply_rollout_labels(wt, wm, [(1, 0)], [0.5], "average", y)
    assert ROLLOUT_MODES == ("replace", "blend")


# ──────────────────────────────────────────────────────────────────────────────────────────────
# λ PRECEDENCE — a labelled row is an ANCHOR the recursion terminates on.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _col(a):
    return np.asarray(a, dtype=np.float64).reshape(-1, 1)


def test_NO_anchors_is_byte_identical_to_the_pre_flag_recursion():
    v, y = _col([.4, .5, .6, .7]), _col([1, 1, 1, 1])
    m, es = _col([1, 1, 1, 1]), _col([1, 0, 0, 0])
    a = lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([1.0]), 0.5)
    b = lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([1.0]), 0.5,
                              anchor_mask=np.zeros((4, 1), bool), anchor_value=np.zeros((4, 1)))
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()


def test_an_ANCHOR_terminates_the_recursion_and_carries_outcome_weight_ONE():
    """With the anchor at row 2 the states before it blend toward the MEASURED probability:
        G2 = 0.2 (the anchor, exactly)
        G1 = .5*V2 + .5*G2 = .30 + .10 = 0.40
        G0 = .5*V1 + .5*G1 = .25 + .20 = 0.45
    and the outcome weight is 1.0 at the anchor because a Monte-Carlo win fraction is a
    MEASUREMENT of that state, not a bootstrap off the network."""
    v, y = _col([.4, .5, .6, .7]), _col([1, 1, 1, 1])
    m, es = _col([1, 1, 1, 1]), _col([1, 0, 0, 0])
    anchors = np.zeros((4, 1), bool); anchors[2, 0] = True
    vals = np.zeros((4, 1)); vals[2, 0] = 0.2
    g, w, _nm, _nu = lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([1.0]), 0.5,
                                           anchor_mask=anchors, anchor_value=vals)
    assert np.allclose(g.ravel(), [0.45, 0.40, 0.20, 1.0])
    assert w[2, 0] == 1.0 and w[1, 0] == pytest.approx(0.5) and w[0, 0] == pytest.approx(0.25)


def test_a_MISSHAPEN_anchor_array_RAISES():
    v, y = _col([.4, .5]), _col([1, 1])
    m, es = _col([1, 1]), _col([1, 0])
    with pytest.raises(ValueError):
        lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([1.0]), 0.5,
                              anchor_mask=np.zeros((3, 1), bool), anchor_value=np.zeros((3, 1)))


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE HANDLE <-> RECORD JOIN
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_handle_and_the_ring_filename_use_the_SAME_sanitiser(tmp_path):
    """Two spellings of the sanitiser would make the join silently miss on exactly the tags that
    needed sanitising."""
    from agents.training.cf_records import CfRecordRing
    ring = CfRecordRing(tmp_path)
    tag = "battle-gen3ou-7/odd"
    path = ring.write_record(tag, {"log": []})
    assert path is not None
    idx = index_records(tmp_path)
    import os
    assert idx[record_key(os.getpid(), tag)] == str(path)


def test_the_newest_record_WINS_for_a_reused_key(tmp_path):
    from utils.bridge.reconstruction import RECON_SUFFIX
    for ns in ("0000000000000000001", "0000000000000000002"):
        (tmp_path / f"{ns}_99_battle-x{RECON_SUFFIX}").write_text("{}")
    idx = index_records(tmp_path)
    assert idx["99_battle-x"].endswith(f"0000000000000000002_99_battle-x{RECON_SUFFIX}")


def test_a_missing_records_dir_is_an_EMPTY_index_not_a_crash(tmp_path):
    assert index_records(tmp_path / "nope") == {}


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE CALLBACK, end to end against a STUB labeller.
# ──────────────────────────────────────────────────────────────────────────────────────────────

class _Policy:
    _critic_mode = "winprob"

    def predict_values(self, obs):
        return torch.zeros(1, 1)


class _Buf:
    def __init__(self, n_steps, n_envs):
        self.values = np.full((n_steps, n_envs), 0.5, dtype=np.float32)
        self.episode_starts = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.episode_starts[0, :] = 1.0
        self.observations = {
            "win_target": np.zeros((n_steps, n_envs, 1), dtype=np.float32),
            "win_mask": np.zeros((n_steps, n_envs, 1), dtype=np.float32),
            "action_mask": np.tile(np.asarray(MOVE_ROW, dtype=np.float32), (n_steps, n_envs, 1)),
            "opp_class": np.zeros((n_steps, n_envs, 1), dtype=np.int64),
        }


class _Model:
    def __init__(self, n_steps, n_envs, **kw):
        self.rollout_buffer = _Buf(n_steps, n_envs)
        self.n_steps, self.n_envs = n_steps, n_envs
        self.policy = _Policy()
        self.device = "cpu"
        self.seed = 11
        self._last_obs = None
        self._last_episode_starts = np.ones(n_envs, dtype=np.float32)
        self._win_terminal_scratch = np.full((n_steps, n_envs), np.nan, np.float32)
        self._win_prob_lambda_metrics = None
        self._win_prob_rollout_metrics = None
        for k, v in kw.items():
            setattr(self, k, v)
        # gen3_winprob_rollout_weight_v1: the env declares `win_row_w` under EXACTLY this
        # predicate (`env_factory`), so the fake buffer does too — a harness that carried the key
        # unconditionally would hide the "flag on, key absent" path the callback must survive.
        if (float(getattr(self, "win_prob_rollout_weight", 1.0) or 1.0) > 1.0
                and float(getattr(self, "win_prob_rollout_target", 0.0) or 0.0) > 0.0):
            self.rollout_buffer.observations["win_row_w"] = np.ones(
                (n_steps, n_envs, 1), dtype=np.float32)

    def save(self, path):
        with open(path, "w") as f:
            f.write("weights")


def _run(monkeypatch, tmp_path, *, stub=None, n_steps=6, n_envs=2, records=True,
         write_records=True, **kw):
    """One rollout on a synthetic buffer whose single episode per env ENDS at the last row."""
    from utils.bridge.reconstruction import RECON_SUFFIX
    import agents.training.win_prob_rollout_labeller as lab

    model = _Model(n_steps, n_envs, **kw)
    cb = WinProbLabelCallback(records_dir=str(tmp_path) if records else None)
    cb.model = model
    cb._on_rollout_start()                                   # clears the scratch — fill it AFTER
    model._win_terminal_scratch[n_steps - 1, 0] = 1.0        # env 0 WON
    model._win_terminal_scratch[n_steps - 1, 1] = 0.0        # env 1 LOST
    keys, turns = cb._handle_scratch() if cb._rollout_on() else (None, None)
    if keys is not None:
        for t in range(n_steps):
            for e in range(n_envs):
                keys[t, e] = f"77_battle-{e}"
                turns[t, e] = 3 + t
        if write_records:
            for e in range(n_envs):
                (tmp_path / f"000000000000000000{e}_77_battle-{e}{RECON_SUFFIX}").write_text("{}")
    calls = {}

    def _stub(*, model, states, rollouts, impl, **_kw):
        calls["states"] = list(states)
        calls["rollouts"] = rollouts
        labels = stub(states) if stub else [0.25] * len(states)
        return labels, {"seconds": 1.5, "arms": len(states) * rollouts, "arms_capped": 0,
                        "arm_decisions": 40.0}
    monkeypatch.setattr(lab, "label_states", _stub)
    cb._on_rollout_end()
    obs = model.rollout_buffer.observations
    return model, obs["win_target"], obs["win_mask"], calls


def test_OFF_leaves_the_terminal_bit_target_BIT_identical(monkeypatch, tmp_path):
    _m0, wt0, wm0, c0 = _run(monkeypatch, tmp_path)                       # no attribute at all
    _m1, wt1, wm1, c1 = _run(monkeypatch, tmp_path, win_prob_rollout_target=0.0)
    assert wt0.tobytes() == wt1.tobytes() and wm0.tobytes() == wm1.tobytes()
    assert c0 == {} and c1 == {}, "OFF must not even ASK the labeller"
    assert np.allclose(wt0[:, 0, 0], 1.0) and np.allclose(wt0[:, 1, 0], 0.0)


def test_OFF_publishes_NO_rollout_family(monkeypatch, tmp_path):
    model, _wt, _wm, _c = _run(monkeypatch, tmp_path, win_prob_rollout_target=0.0)
    assert model._win_prob_rollout_metrics is None, \
        "an absent win_prob/rollout_* family must mean the fraction is 0.0 and nothing else"


def test_ON_the_sampled_rows_get_EXACTLY_wins_over_R_and_the_rest_keep_the_bit(monkeypatch,
                                                                               tmp_path):
    model, wt, wm, calls = _run(monkeypatch, tmp_path, win_prob_rollout_target=0.5,
                                win_prob_rollout_r=4)
    assert calls["rollouts"] == 4
    labelled = {(t, e) for t in range(6) for e in range(2) if wt[t, e, 0] == np.float32(0.25)}
    assert labelled, "at least one state must carry the measured target"
    for t in range(6):
        for e in range(2):
            if (t, e) in labelled:
                continue
            assert wt[t, e, 0] == np.float32(1.0 if e == 0 else 0.0)
    m = model._win_prob_rollout_metrics
    assert m["rollout_states"] == len(labelled) and m["rollout_r"] == 4.0
    assert m["rollout_seconds"] == 1.5


def test_ON_labels_AT_MOST_ONE_state_per_episode(monkeypatch, tmp_path):
    """Two envs, one episode each => at most two labels, whatever the fraction asks for."""
    _model, wt, _wm, calls = _run(monkeypatch, tmp_path, win_prob_rollout_target=1.0)
    assert len(calls["states"]) <= 2


def test_the_dose_meter_is_the_mean_distance_from_the_bit_it_REPLACED(monkeypatch, tmp_path):
    model, _wt, _wm, _c = _run(monkeypatch, tmp_path, win_prob_rollout_target=1.0,
                               stub=lambda s: [0.25] * len(s))
    m = model._win_prob_rollout_metrics
    assert m["rollout_states"] > 0.0
    # env 0 won (|0.25-1| = 0.75), env 1 lost (|0.25-0| = 0.25); with one label per env the mean
    # of the labelled rows is 0.5.
    assert m["rollout_shift"] == pytest.approx(0.5)
    assert m["rollout_win_rate"] == pytest.approx(0.25)
    assert 0.0 <= m["rollout_mass"] <= 1.0


def test_a_FAILED_state_is_counted_and_keeps_its_bit(monkeypatch, tmp_path):
    model, wt, _wm, _c = _run(monkeypatch, tmp_path, win_prob_rollout_target=1.0,
                              stub=lambda s: [None] * len(s))
    m = model._win_prob_rollout_metrics
    assert m["rollout_states"] == 0.0 and m["rollout_failed"] > 0.0
    assert np.allclose(wt[:, 0, 0], 1.0) and np.allclose(wt[:, 1, 0], 0.0)


def test_a_state_whose_RECORD_VANISHED_is_counted_and_never_sent(monkeypatch, tmp_path):
    """The ring is pruned by the env workers while this runs, so a handle CAN outlive its file.
    That is a counted benign skip, never an exception and never a label."""
    model, wt, _wm, calls = _run(monkeypatch, tmp_path, win_prob_rollout_target=1.0,
                                 records=True, write_records=False)
    m = model._win_prob_rollout_metrics
    assert m["rollout_records_missing"] > 0.0 and m["rollout_states"] == 0.0
    assert calls == {}, "a state with no record must never reach the labeller"
    assert np.allclose(wt[:, 0, 0], 1.0) and np.allclose(wt[:, 1, 0], 0.0)


def test_it_is_INERT_under_a_critic_that_is_not_winprob(monkeypatch, tmp_path):
    """Belt-and-braces behind the combination check: under `shaped` the win-prob BCE is an
    auxiliary readout, so a measured target there would re-aim a diagnostic and still pay for
    every continuation."""
    import agents.training.win_prob_rollout_labeller as lab
    model = _Model(6, 2, win_prob_rollout_target=1.0)
    model.policy = _Policy()
    model.policy._critic_mode = "shaped"
    cb = WinProbLabelCallback(records_dir=str(tmp_path))
    cb.model = model
    called = []
    monkeypatch.setattr(lab, "label_states", lambda **kw: called.append(kw) or ([], {}))
    cb._on_rollout_start()
    model._win_terminal_scratch[5, :] = 1.0
    cb._on_rollout_end()
    assert called == [] and model._win_prob_rollout_metrics is None


def test_it_REFUSES_with_no_cf_records_ring_and_SAYS_SO_ONCE(monkeypatch, tmp_path, capsys):
    import agents.training.win_prob_rollout_labeller as lab
    called = []
    monkeypatch.setattr(lab, "label_states", lambda **kw: called.append(kw) or ([], {}))
    model = _Model(6, 2, win_prob_rollout_target=1.0)
    cb = WinProbLabelCallback(records_dir=None)
    cb.model = model
    for _ in range(3):
        cb._on_rollout_start()
        model._win_terminal_scratch[5, :] = 1.0
        cb._on_rollout_end()
    out = capsys.readouterr().out
    assert called == []
    assert out.count("no cf_records ring") == 1, "said once, not once per rollout"


def test_the_handle_scratch_is_NOT_allocated_on_an_unflagged_run(monkeypatch, tmp_path):
    model, _wt, _wm, _c = _run(monkeypatch, tmp_path)
    assert getattr(model, "_win_handle_keys", None) is None


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE METRIC FAMILY
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_metric_family_names_the_cost_and_the_ecology(monkeypatch, tmp_path):
    model, _wt, _wm, _c = _run(monkeypatch, tmp_path, win_prob_rollout_target=1.0)
    m = model._win_prob_rollout_metrics
    for k in ("rollout_target", "rollout_r", "rollout_mode_blend", "rollout_eligible",
              "rollout_requested", "rollout_states", "rollout_failed", "rollout_seconds",
              "rollout_arms", "rollout_capped_frac", "rollout_mass", "rollout_shift",
              "rollout_win_rate", "rollout_bot_share", "rollout_budget_multiple",
              "rollout_unit_budget_fraction", "rollout_records_missing"):
        assert k in m, k
    assert m["rollout_bot_share"] == 1.0, "the synthetic buffer is all opp_class 0 (bot)"


def test_a_zero_state_rollout_still_prices_itself_with_the_BANKED_continuation_length():
    """A cost meter that reads 'free' because nothing was measured is worse than a banked one."""
    m = rollout_metrics(applied=[], picks=[], labels=[], eligible=np.zeros((4, 1), bool),
                        terminal_y=np.zeros((4, 1)), new_mask=np.ones((4, 1)),
                        opp_class=np.zeros((4, 1), dtype=np.int64), fraction=0.01, rollouts=8,
                        mode="replace", seconds=0.0, arms_played=0, arms_capped=0,
                        arm_decisions=BANKED_CONTINUATION_DECISIONS, records_missing=0,
                        n_steps=4, n_envs=1)
    assert m["rollout_unit_budget_fraction"] == pytest.approx(1.0 / 832.0)


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE RECORD: model_config + the migration
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_BOTH_ModelVersion_sites_carry_the_three_fields_at_config_118():
    """The FIELD (what a config holds and what a flagless resume reads back) and the
    CONSTRUCTION (what a save writes) are two separate files, and a field declared in one but not
    threaded through the other records a default forever — silently."""
    import inspect

    from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersionFields
    from agents.model.model_version.construct import ModelVersionConstruction

    import dataclasses

    assert MODEL_CONFIG_VERSION >= 118
    defaults = {f.name: f.default for f in dataclasses.fields(ModelVersionFields)}
    assert defaults["win_prob_rollout_target"] == 0.0
    assert defaults["win_prob_rollout_r"] == DEFAULT_ROLLOUT_R
    assert defaults["win_prob_rollout_mode"] == "replace"
    sig = inspect.signature(ModelVersionConstruction.from_layout_and_policy_kwargs)
    src = inspect.getsource(ModelVersionConstruction.from_layout_and_policy_kwargs)
    for name in ("win_prob_rollout_target", "win_prob_rollout_r", "win_prob_rollout_mode"):
        assert name in sig.parameters, f"{name} is not accepted by the construction site"
        assert f"{name}=" in src, f"{name} is accepted but never written into the config"


def test_a_pre_v118_config_MIGRATES_to_the_terminal_bit_target():
    """0.0 is a RECORD, not a guess: the terminal bit IS what every pre-v118 run trained against."""
    out = _migrate_config({"config_version": 117})
    assert out["win_prob_rollout_target"] == 0.0
    assert out["win_prob_rollout_r"] == DEFAULT_ROLLOUT_R
    assert out["win_prob_rollout_mode"] == "replace"
    assert out["config_version"] >= 118


def test_the_fraction_is_declared_in_the_coefficient_module_table():
    """A fraction above 0 with no win-prob head pays for thousands of continuations and re-aims a
    loss that is not being computed — the most expensive INERT the table can show."""
    from agents.model.arch_tables import _COEF_MODULE
    assert _COEF_MODULE["win_prob_rollout_target"] == "win_head"


def test_the_three_flags_are_INHERITED_on_a_flagless_resume():
    import inspect
    import main.train.config as cfg
    src = inspect.getsource(cfg.resolve_config)
    for name in ("win_prob_rollout_target", "win_prob_rollout_r", "win_prob_rollout_mode"):
        assert f'_resolve("{name}"' in src, (
            f"{name} must be re-read from the checkpoint's model_config on a flagless resume — a "
            f"launcher restart otherwise converts the arm into its own control, and here it does "
            f"it while the run gets FASTER, which reads as a win")


def test_the_defaults_are_the_OFF_position():
    assert ROLLOUT_OFF == 0.0 and DEFAULT_ROLLOUT_R == 8 and ROLLOUT_MODES[0] == "replace"


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE HANDLE IS CAPTURED AT DECISION TIME — the off-by-one that would label the wrong state.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_wrapper_publishes_the_handle_only_when_the_flag_threaded_it_on():
    """A run without `--win-prob-rollout-target` must not even build the tuple."""
    import inspect

    from agents.training.wrappers import MaskableAgentWrapper

    src = inspect.getsource(MaskableAgentWrapper.step)
    assert '_emit_wp_rollout_handle' in src
    # The capture must happen BEFORE the step, or the handle names the turn the step LANDED on and
    # the label describes a state one turn downstream of the buffer row.
    i_capture = src.index("_wp_handle = record_key")
    i_step = src.index("obs, reward, term, trunc, info = super().step(action)")
    assert i_capture < i_step, (
        "the reconstruction handle must be read BEFORE super().step() — the buffer row holds the "
        "observation the decision was made FROM, so the handle must name the turn we were ASKED "
        "at, not the turn the step landed on")
    i_publish = src.index('info["wp_handle"]')
    assert i_publish > i_step, "it is published onto the info the step returned"


def test_the_async_collector_records_the_handle_on_EVERY_row_not_only_a_done_one():
    """The async collector owns the per-env buffer row, so it records inline — and unlike
    `win_outcome`, a handle exists at every decision, not only at a terminal."""
    import inspect

    from agents.training import async_vec_env

    src = inspect.getsource(async_vec_env)
    i_handle = src.index('_wp_keys[t, i] = str(info["wp_handle"])')
    i_done = src.index('_win_scr[t, i] = float(info["win_outcome"])')
    assert i_handle < i_done, (
        "the handle capture must sit OUTSIDE the `if done:` block the outcome capture is inside")
