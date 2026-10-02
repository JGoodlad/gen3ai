"""K9(b) on the python env core and K9(c) fail-closed non-finite — pins (``learner_gates`` module docs).

All on the learner golden's production-surface learner and its committed real buffer (``learner_golden``),
through the REAL ``train()``:

* the normal path passes the behaviour gate (the stored log-probs are the seeded learner's own, from the
  Rust collector's eager inference at a different batch composition: |Δ| at float rounding);
* STALE weights (the parameters moved between the rollout and the update) FATAL before any optimizer step;
* a buffer that carries per-row policy versions takes Lane G's pre-loop probe instead, never both;
* a NaN win-prob LABEL / an Inf return is refused at the BUFFER, before PopArt's advance or any forward;
* a non-finite TERM (a poisoned win-prob head) → a typed FATAL before its backward, naming it;
* a NaN input row is refused by the action distribution's own validation, before any step;
* a one-row final micro-batch (NaN advantage std) trains finite (SB3 PPO's ``len > 1`` guard);
* a finite loss whose BACKWARD yields a NaN gradient → a typed FATAL naming the parameter, untouched.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch as th

from agents.training import learner_golden as L
from agents.training.instrumented_ppo import learner_gates as G
from agents.training.rust_rollout.consistency import BehaviourMismatch


def _learner(check: str = "fatal"):
    model = L.build_learner()
    L.load_buffer_into(model)
    model.behaviour_check = check
    return model


def _params(model):
    return th.cat([p.detach().reshape(-1).clone() for p in model.policy.parameters()])


def _train(model):
    with L._one_thread():
        np.random.seed(L.UPDATE_SEED)
        th.manual_seed(L.UPDATE_SEED)
        model.train()


def test_normal_python_path_passes_the_behaviour_gate_on_the_first_micro_batch():
    model = _learner("fatal")
    assert G.behaviour_gate_mode(model) == "in_loop"
    _train(model)
    worst = model.logger.name_to_value["behaviour/max_abs_dlogp_current"]
    assert worst < 1e-5, worst          # measured ~1e-6 (batch-composition rounding), 100x under the bar
    assert model.logger.name_to_value["behaviour/rows_current"] == L.GOLDEN_OVERRIDES["batch_size"]
    from agents.training.rust_rollout import consistency as K   # fp32: the margins came from a real forward
    excl = model.logger.name_to_value["behaviour/excluded_frac"]
    assert 0.0 <= excl < K.FP32_EXCLUDED_CEILING
    assert model.logger.name_to_value["behaviour/rows_judged"] == round((1 - excl) * L.GOLDEN_OVERRIDES["batch_size"])


@pytest.mark.parametrize("precision", ["highest", "high"])
def test_rollout_weights_one_optimizer_step_stale_are_fatal_before_any_step(precision):
    """The fault K9(b) exists for, at its smallest real size: the learner holds weights ONE optimizer step
    (the recipe lr, 2.8e-5) newer than the rollout's. FATAL under both precision-keyed gates, before any
    optimizer step of the update. (On CUDA the same fault's smallest micro-batch statistic is 200x the
    fp32 bar and 3.1x the TF32 p99 bar — `designs/training/learner_gates.md`.)"""
    prev = th.get_float32_matmul_precision()
    th.set_float32_matmul_precision(precision)
    try:
        model = _learner("off")
        model.n_epochs, model.batch_size, model.grad_accum_steps = 1, L.N_STEPS * L.N_ENVS, 1
        _train(model)                                     # exactly ONE optimizer step on the rows
        L.load_buffer_into(model)                         # the rollout's (now stale) behaviour log-probs
        model.n_epochs, model.batch_size, model.grad_accum_steps = L.GOLDEN_OVERRIDES["n_epochs"], L.GOLDEN_OVERRIDES["batch_size"], 1
        model.behaviour_check = "fatal"
        before = _params(model)
        with pytest.raises(BehaviourMismatch, match=f"first micro-batch.*{precision}"):
            _train(model)
        assert th.equal(before, _params(model)), "the gate must fire before any optimizer step"
    finally:
        th.set_float32_matmul_precision(prev)


def test_warn_reports_and_off_skips():
    model = _learner("off")
    assert G.behaviour_gate_mode(model) == "off"
    model.behaviour_check = "warn"
    model.rollout_buffer.log_probs += 1e-3
    worst = G.check_behaviour_first_micro(
        model, th.as_tensor(model.rollout_buffer.log_probs[0] - 1e-3), th.as_tensor(model.rollout_buffer.log_probs[0]),
        margins=_no_tie(model.rollout_buffer.log_probs[0].size))
    assert worst == pytest.approx(1e-3, rel=1e-3)


def test_a_versioned_buffer_takes_lane_g_probe_not_the_in_loop_gate():
    model = _learner("fatal")
    model._rust_row_versions = np.zeros((L.N_STEPS, L.N_ENVS), np.int64)
    assert G.behaviour_gate_mode(model) == "probe"


def test_a_nan_label_is_refused_at_the_buffer_before_popart_or_any_forward():
    model = _learner("off")
    model.rollout_buffer.observations["win_target"][3, 1] = np.nan      # ONE poisoned label row
    before = _params(model)
    with pytest.raises(G.NonFiniteLearnerError, match=r"NON-FINITE ROLLOUT BUFFER.*win_target.*1 values"):
        _train(model)
    assert th.equal(before, _params(model))
    model.rollout_buffer.observations["win_target"][3, 1] = 1.0
    model.rollout_buffer.returns[0, 0] = np.inf
    with pytest.raises(G.NonFiniteLearnerError, match=r"returns \(1 rows\)"):
        _train(model)


def test_a_nan_input_row_is_refused_before_any_step():
    """The flat observation is not scanned (a policy INPUT, ~0.3 s at production size): a NaN there
    reaches the action logits, and torch's Categorical argument validation raises before any loss —
    fail-closed already (untyped). Pinned so that a `validate_args=False` somewhere cannot open it."""
    model = _learner("off")
    ob = model.rollout_buffer.observations["observation"]
    # a CONTINUOUS feature (fractional values somewhere — an id channel would index-error instead)
    col = int(np.flatnonzero((np.abs(ob - np.round(ob)) > 1e-3).reshape(-1, ob.shape[-1]).any(0))[0])
    ob[:, :, col] = np.nan
    before = _params(model)
    with pytest.raises((ValueError, G.NonFiniteLearnerError)):
        _train(model)
    assert th.equal(before, _params(model))


def test_a_non_finite_term_is_fatal_before_its_backward_naming_it():
    """A term whose NaN does not pass through the action distribution — the win-prob critic, from a
    poisoned head weight — is caught by the per-micro-batch loss check and NAMED."""
    model = _learner("off")
    p = next(q for n, q in model.policy.named_parameters() if n.startswith("features_extractor.win_head"))
    with th.no_grad():
        p.view(-1)[0] = float("nan")
    before = _params(model)
    with pytest.raises(G.NonFiniteLearnerError, match=r"NON-FINITE LOSS .* epoch 0 \(micro-batch 0 of this update\).*win_prob"):
        _train(model)
    assert th.allclose(before, _params(model), rtol=0.0, atol=0.0, equal_nan=True)


def test_a_one_row_final_micro_batch_trains_finite():
    """64 rows at micro 21 → 21, 21, 21, 1: the one-row micro-batch's advantage std is NaN, which
    silently poisoned every weight before K9(c). SB3 PPO's `len > 1` guard skips normalising it."""
    model = _learner("off")
    model.batch_size = 21
    _train(model)
    assert bool(th.isfinite(_params(model)).all())


def test_kl_check_and_the_set_valued_beta_loss_keep_a_nan_visible():
    G.check_kl_finite(0.01, epoch=0)
    with pytest.raises(G.NonFiniteLearnerError, match="approx-KL inf"):
        G.check_kl_finite(math.inf, epoch=2)
    from agents.model.opp_intent import set_valued_switch_loss

    logits = th.tensor([[0.0, float("-inf"), float("nan"), 0.0, 0.0, 0.0]])
    believed = th.tensor([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    # slot 1 is the deliberate -inf (unreachable); slot 2 is a NaN: the row is SCORED, the NaN shows
    out = set_valued_switch_loss(logits, believed, th.tensor([True]))
    assert out is not None and bool(th.isnan(out))
    believed_ok = th.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    assert set_valued_switch_loss(logits, believed_ok, th.tensor([True])) is None   # -inf only: dropped


def test_a_nan_gradient_from_a_finite_loss_is_fatal_naming_the_parameter():
    model = _learner("off")
    name, p = next((n, q) for n, q in model.policy.named_parameters() if n.startswith("pointer_head"))
    p.register_hook(lambda grad: grad * float("nan"))
    before = _params(model)
    with pytest.raises(G.NonFiniteLearnerError, match=r"NON-FINITE GRADIENT.*" + name.replace(".", r"\.")):
        _train(model)
    assert th.equal(before, _params(model))


def test_check_loss_finite_names_every_non_finite_term():
    good = th.tensor(1.0)
    G.check_loss_finite(good, {"policy": good}, epoch=0, micro=0)
    with pytest.raises(G.NonFiniteLearnerError, match="item_belief, win_prob"):
        G.check_loss_finite(th.tensor(math.inf), {"policy": good, "item_belief": th.tensor(math.nan),
                                                  "win_prob": th.tensor(math.inf), "value": 0.0},
                            epoch=1, micro=3)
    with pytest.raises(G.NonFiniteLearnerError, match="none of the named terms"):
        G.check_loss_finite(th.tensor(math.nan), {"policy": good}, epoch=0, micro=0)


def test_the_gate_is_keyed_by_the_matmul_precision_the_run_uses():
    """ONE table (`consistency.BEHAVIOUR_GATES`), ONE enforcement (`enforce_behaviour`), read by BOTH
    implementations: fp32 = DETERMINISTIC — max < 1e-4 over the rows not at a tie, single-shot, and the
    excluded share under its ceiling; TF32 = p99 < 3.6e-3 single-shot AND max < 0.071,
    FATAL only on `TF32_MAX_PERSISTENCE` (4) consecutive updates; an undeclared precision is refused."""
    from agents.training.rust_rollout import consistency as K

    C = K.GateCondition
    assert K.behaviour_gate("highest") == (C("max", 1e-4, 1, K.FP32_TIE_EPS),
                                           C("excluded_frac", K.FP32_EXCLUDED_CEILING, 1, K.FP32_TIE_EPS))
    assert K.BEHAVIOUR_BAR == 1e-4
    assert K.TF32_MAX_PERSISTENCE == 4
    assert K.behaviour_gate("high") == (C("p99", 3.6e-3, 1), C("max", 0.071, K.TF32_MAX_PERSISTENCE))
    with pytest.raises(K.UndeclaredPrecision, match="medium"):
        K.behaviour_gate("medium")
    model = _learner("fatal")
    mid = th.full((2048,), 1e-3, dtype=th.float64)          # above the fp32 bar, under both TF32 bars
    with _precision("high"):
        G.check_behaviour_first_micro(model, mid, th.zeros(2048))
        assert model.logger.name_to_value["behaviour/bar_p99"] == 3.6e-3
        assert model.logger.name_to_value["behaviour/bar_max"] == 0.071
    with pytest.raises(BehaviourMismatch, match="'highest'"):
        G.check_behaviour_first_micro(model, mid, th.zeros(2048), margins=_no_tie(2048))   # fp32: single-shot


class _precision:
    def __init__(self, p):
        self.p = p

    def __enter__(self):
        self.prev = th.get_float32_matmul_precision()
        th.set_float32_matmul_precision(self.p)

    def __exit__(self, *a):
        th.set_float32_matmul_precision(self.prev)


def _localized(seed: int, bad_rows: int = 10, value: float = 0.3):
    """A 2,048-row |Δ| micro-batch: TF32-sized healthy noise, plus ``bad_rows`` localized rows at
    ``value`` (the measured localized signals: 0.24 – 1e8, `designs/training/learner_gates.md`)."""
    g = th.Generator().manual_seed(seed)
    d = th.rand(2048, generator=g, dtype=th.float64) * 1e-3
    if bad_rows:
        d[th.randperm(2048, generator=g)[:bad_rows]] = value
    return d


def _fp32(bad_rows: int, value: float = 1e-3):
    """A 2,048-row |Δ| micro-batch at fp32's healthy noise (~1e-6) plus ``bad_rows`` rows over its bar."""
    g = th.Generator().manual_seed(3)
    d = th.rand(2048, generator=g, dtype=th.float64) * 2e-6
    d[th.randperm(2048, generator=g)[:bad_rows]] = value
    return d


def _no_tie(n, tied=()):
    """`behaviour_margins_first_micro`'s shape: every row far from a cutoff, except ``tied`` (exact ties)."""
    m = np.ones(n)
    m[list(tied)] = 0.0
    return m, ["test" for _ in range(n)]


def _step(model, d, margins=None):
    G.check_behaviour_first_micro(model, d, th.zeros(d.numel()),
                                  actions=th.arange(d.numel()) % 11, masks=th.ones(d.numel(), 11),
                                  margins=margins if margins is not None else _no_tie(d.numel()))


def test_one_isolated_tf32_max_violation_warns_loudly_dumps_the_rows_and_does_not_fatal(tmp_path, capsys):
    from agents.training.rust_rollout import consistency as K

    model = _learner("fatal")
    model.behaviour_dump_dir = str(tmp_path)
    with _precision("high"):
        _step(model, _localized(1))                                   # ONE violation: no raise
        assert model.logger.name_to_value["behaviour/streak_max"] == 1.0
        assert model.logger.name_to_value["behaviour/streak_p99"] == 0.0
    out = capsys.readouterr().out
    assert "🚨" in out and "max 0.3 NOT < 0.071" in out and "FATAL at 4" in out and "NOT fatal YET" in out
    import json
    rec = json.loads((tmp_path / K.VIOLATION_DUMP).read_text().splitlines()[0])
    assert rec["precision"] == "high" and rec["p99"] < 3.6e-3 and rec["rows_judged"] == 2048
    top = rec["rows"][:10]
    assert all(r["abs_dlogp"] == 0.3 for r in top) and {"index", "action", "mask"} <= set(top[0])
    assert top[0]["mask"] == "1" * 11


def test_a_tf32_max_violation_on_k_consecutive_updates_is_fatal_and_not_before():
    from agents.training.rust_rollout import consistency as K

    model = _learner("fatal")
    k = K.TF32_MAX_PERSISTENCE
    with _precision("high"):
        for i in range(k - 1):
            _step(model, _localized(i + 1))                             # warnings, no raise
        assert model.logger.name_to_value["behaviour/streak_max"] == float(k - 1)
        with pytest.raises(BehaviourMismatch, match=f"max: violated {k} consecutive update.*FATAL at {k}"):
            _step(model, _localized(99))


def test_a_clean_update_resets_the_streak_so_interleaved_violations_never_fatal():
    from agents.training.rust_rollout import consistency as K

    model = _learner("fatal")
    k = K.TF32_MAX_PERSISTENCE
    with _precision("high"):
        for rnd in range(3):                                           # (k-1 violations, 1 clean) x 3
            for i in range(k - 1):
                _step(model, _localized(10 * rnd + i + 1))
            _step(model, _localized(10 * rnd + 9, bad_rows=0))         # a clean update resets the streak
            assert model.logger.name_to_value["behaviour/streak_max"] == 0.0
        _step(model, _localized(77))                                   # violated again: streak 1, a warning
        assert model.logger.name_to_value["behaviour/streak_max"] == 1.0


def test_the_tf32_p99_is_single_shot_the_fp32_rule_is_deterministic_and_a_nan_is_always_fatal():
    """fp32 on the Python path (`gen3_behaviour_tie_exclusion_v1`): ONE judged row over 1e-4 is FATAL on the
    FIRST update; the same row AT a tie is excluded (no violation); too many tied rows FATAL; a NaN — even
    on an excluded row — is never rounding; and fp32 without margins is refused, never judged blind."""
    from agents.training.rust_rollout import consistency as K
    from agents.training.rust_rollout.tie_margins import TieMarginError

    model = _learner("fatal")
    with _precision("high"):
        with pytest.raises(BehaviourMismatch, match="p99: violated 1 consecutive"):
            _step(model, th.full((2048,), 5e-3, dtype=th.float64))     # a GLOBAL fault: first update
        model._behaviour_streaks = {}
        nan = _localized(1, bad_rows=0)
        nan[7] = float("nan")
        with pytest.raises(BehaviourMismatch):
            _step(model, nan)                                          # never rounding: no persistence
    model._behaviour_streaks = {}
    d = _fp32(bad_rows=1)
    bad = int(th.argmax(d))
    with pytest.raises(BehaviourMismatch, match=r"max 0\.001 NOT < 0\.0001.*FATAL at 1"):
        _step(model, d)                                                # not at a tie: FATAL at once
    model._behaviour_streaks = {}
    _step(model, d, margins=_no_tie(2048, tied=[bad]))                 # the same row AT a tie: excluded
    assert model.logger.name_to_value["behaviour/rows_excluded"] == 1.0
    assert model.logger.name_to_value["behaviour/streak_max"] == 0.0
    nan = _fp32(bad_rows=0)
    nan[5] = float("nan")
    with pytest.raises(BehaviourMismatch, match="NON-FINITE"):
        _step(model, nan, margins=_no_tie(2048, tied=[5]))             # a NaN on an EXCLUDED row
    model._behaviour_streaks = {}
    many = list(range(int(K.FP32_EXCLUDED_CEILING * 2048) + 1))
    with pytest.raises(BehaviourMismatch, match="TOO MANY rows sit at a tie"):
        _step(model, _fp32(bad_rows=0), margins=_no_tie(2048, tied=many))
    with pytest.raises(TieMarginError):
        G.check_behaviour_first_micro(model, _fp32(bad_rows=0), th.zeros(2048))


def _wide_learner(n_steps: int = 64):
    """The golden learner at a 256-row buffer (the 64 real rows tiled), one 256-row micro-batch, lr 0
    so every update sees the same weights — a "fresh rollout" each update with the SAME fault."""
    from stable_baselines3.common.logger import configure

    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.parity import _unset_to_class_defaults
    from agents.training.rust_vec_env import RustVecEnv
    from main.rust_core_cutover.envs import production_args
    from main.train.model_build import apply_training_hparams

    _a, obs, act = TK.production_spaces()
    env = RustVecEnv(n_envs=L.N_ENVS, observation_space=obs, action_space=act, build=lambda m: None)
    m = TK.fresh_model(env, n_steps=n_steps, batch_size=n_steps * L.N_ENVS, n_epochs=1, seed=L.MODEL_SEED,
                       perturb_seed=L.PERTURB_SEED, learning_rate=0.0)
    apply_training_hparams(m, production_args(), mappings=None, attach_cf_labels=lambda _m: None)
    _unset_to_class_defaults(m)
    m.grad_accum_steps, m.behaviour_check = 1, "fatal"
    m._logger = configure(None, [])
    with np.load(L.BUFFER_PATH) as z:
        data = {k: z[k] for k in z.files}
    reps = n_steps // L.N_STEPS
    rows = {k: np.concatenate([v] * reps, axis=0) for k, v in data.items()}
    return m, rows


def _fill(m, rows, corrupt):
    """Fresh buffer from ``rows``: behaviour log-probs from THIS policy (eval, the rollout's mode), then
    the corrupted rows' actions pointed at their LEAST likely other legal action (a wrong action index)."""
    from stable_baselines3.common.utils import obs_as_tensor

    rb = m.rollout_buffer
    rb.reset()
    for k in rb.observations:
        rb.observations[k][...] = rows["obs:" + k]
    for f in ("actions", "rewards", "episode_starts", "values", "advantages", "returns", "action_masks"):
        getattr(rb, f)[...] = rows[f]
    n = rb.buffer_size * rb.n_envs
    obs = {k: rb.observations[k].reshape(n, *rb.observations[k].shape[2:]) for k in rb.observations}
    masks = rb.action_masks.reshape(n, -1)
    acts = rb.actions.reshape(n).astype(np.int64)
    m.policy.set_training_mode(False)
    with th.no_grad():
        dist = m.policy.get_distribution(obs_as_tensor(obs, m.device), action_masks=masks)
        lp = dist.log_prob(th.as_tensor(acts)).numpy()
        probs = dist.distribution.probs.numpy()
    for i in corrupt:
        legal = np.flatnonzero(masks[i] > 0.5)
        others = legal[legal != acts[i]]
        acts[i] = others[np.argmin(probs[i, others])]
    rb.actions[...] = acts.reshape(rb.actions.shape)
    rb.log_probs[...] = lp.reshape(rb.log_probs.shape)          # μ of the ORIGINAL actions
    rb.full, rb.pos = True, rb.buffer_size
    m._current_progress_remaining = 1.0


def test_a_persistent_localized_fault_on_the_real_buffer_is_fatal_on_its_kth_update(capsys):
    """Real rows through the REAL `train()`, TF32 gate: 2 of 256 rows (0.8 %, under the p99's 1 %) point their action at the
    least likely other legal action on EVERY update (a systematic fault). Updates 1..k-1: the p99 stays
    clean, the max fires, a loud warning and no FATAL. Update k (the same fault): FATAL, before any step."""
    from agents.training.rust_rollout import consistency as K

    m, rows = _wide_learner()
    n = 256
    masks = rows["action_masks"].reshape(n, -1)
    corrupt = [i for i in range(0, n, 37) if (masks[i] > 0.5).sum() >= 2][:2]
    assert len(corrupt) == 2
    k = K.TF32_MAX_PERSISTENCE
    with _precision("high"):
        for i in range(k - 1):
            _fill(m, rows, corrupt)
            np.random.seed(i)
            m.train()
            assert m.logger.name_to_value["behaviour/p99_abs_dlogp_current"] < 3.6e-3      # (a) passes
            assert m.logger.name_to_value["behaviour/max_abs_dlogp_current"] >= 0.071      # (b) fires
            assert m.logger.name_to_value["behaviour/streak_max"] == float(i + 1)
        assert "NOT fatal YET" in capsys.readouterr().out
        _fill(m, rows, corrupt)
        before = _params(m)
        with pytest.raises(BehaviourMismatch, match=f"{k} consecutive"):
            m.train()
        assert th.equal(before, _params(m))


_BATTERY = "designs/research_state/measurements/learner_battery_2026-09-26/"
_PAUSED_ARGVS = ("argv_L95.txt", "scripts/argv_T32_STANDIN.txt", "validation/reruns_2026-09-28/argv_Cfix.txt",
                 "validation/reruns_2026-09-28/argv_T32b_sentinel.txt")


@pytest.mark.parametrize("name", _PAUSED_ARGVS)
def test_the_paused_battery_argvs_parse_unchanged_under_k9(name):
    """The four paused learner-battery arms (two of them `--matmul-precision high`, none typing
    `--behaviour-check`) must still launch: K9(b) resolves to `fatal` at the TF32 bar, and no K9 rule
    refuses them. The launcher-only flags are stripped exactly as the launcher strips them, and so
    are the flags DELETED since the battery was recorded (deletion pass L2: the recorded argvs carry
    the entropy-boost / true-team defaults; a pinned launch is judged by its OWN commit's parser, so
    HEAD's parser never sees them — what K9 does with the REST of the argv is what this pins)."""
    import shlex

    from main.train.combination_checks import failing_checks
    from main.train.rust_env_setup import resolve_env_core_args
    from main.train_rl_agent import build_parser
    from utils.paths import repo_path

    toks = shlex.split(repo_path(*(_BATTERY + name).split("/")).read_text())
    argv, i = [], 0
    dead_valued = {"--defensive-entropy-boost", "--defensive-entropy-anneal-frac",
                   "--bait-entropy-boost", "--bait-entropy-anneal-frac"}
    dead_bool = {"--value-true-team", "--no-value-true-team"}
    while i < len(toks):
        if toks[i] in ("--restart-interval-hours", "--pin-commit") or toks[i] in dead_valued:
            i += 2                                                    # launcher-only / deleted
            continue
        if toks[i] in dead_bool:
            i += 1
            continue
        argv.append(toks[i])
        i += 1
    args = build_parser().parse_args(argv)
    assert args.behaviour_check is None
    # Each arm is a `--model` FORK of a python-era checkpoint, so an untyped `--env-core` INHERITS
    # python at launch (`rust_env_setup.resolve_env_core_default`), not the bare parser's `rust`.
    assert "--model" in argv and "--env-core" not in argv
    args.env_core = "python"
    k9 = [c.name for c in failing_checks(args) if {"behaviour_check", "matmul_precision"} & set(c.dests)]
    assert not k9, k9
    resolve_env_core_args(args)
    from agents.training.rust_rollout.consistency import behaviour_gate

    assert args.behaviour_check == "fatal" and behaviour_gate(args.matmul_precision)


def test_every_non_finite_fatal_is_the_launchers_class_and_carries_the_tag():
    """K9(c) raises `main.exit_codes.NonFiniteLearnerError` (cutover-prep's class — the trainer exits
    FATAL_NONFINITE = 4 on it and the launcher does not restart), every message tagged once."""
    from main.exit_codes import NonFiniteLearnerError, TrainExitCode, exit_code_for

    assert G.NonFiniteLearnerError is NonFiniteLearnerError
    assert G.LEARNER_FATAL_TAG == "[Learner] FATAL"
    with pytest.raises(NonFiniteLearnerError) as ei:
        G.check_kl_finite(math.nan, epoch=0)
    assert str(ei.value).startswith("[Learner] FATAL [K9(c)]")
    assert exit_code_for(ei.value) == int(TrainExitCode.FATAL_NONFINITE) == 4
    assert str(G.nonfinite("[Learner] FATAL x")) == "[Learner] FATAL x"
