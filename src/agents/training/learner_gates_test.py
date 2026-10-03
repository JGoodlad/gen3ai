"""K9(b) through ``train()`` and K9(c) fail-closed non-finite — pins (``learner_gates`` module docs).

All on the learner golden's production-surface learner and its committed real buffer (``learner_golden``),
through the REAL ``train()``:

* the normal path passes the behaviour probe (the stored log-probs are the seeded learner's own, from the
  Rust collector's eager inference at a different batch composition: |Δ| at float rounding) — the golden
  buffer carries no version record, so the probe judges every row as current;
* STALE weights (the parameters moved between the rollout and the update) FATAL before any optimizer step;
* every check but ``off`` is Lane G's probe — the ONE implementation (the python core's in-loop variant
  was deleted, U4); the gate's table and its enforcement (``consistency.enforce_behaviour``) are pinned
  directly;
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


def test_the_normal_path_passes_the_behaviour_probe():
    model = _learner("fatal")
    assert G.behaviour_gate_mode(model) == "probe"
    _train(model)
    worst = model.logger.name_to_value["behaviour/max_abs_dlogp_current"]
    assert worst < 1e-5, worst          # measured ~1e-6 (batch-composition rounding), 100x under the bar
    rows = min(L.GOLDEN_OVERRIDES["batch_size"], L.N_STEPS * L.N_ENVS)
    assert model.logger.name_to_value["behaviour/rows_current"] == rows
    from agents.training.rust_rollout import consistency as K   # fp32: the margins came from a real forward
    excl = model.logger.name_to_value["behaviour/excluded_frac"]
    assert 0.0 <= excl < K.FP32_EXCLUDED_CEILING
    assert model.logger.name_to_value["behaviour/rows_judged"] == round((1 - excl) * rows)


def test_rollout_weights_one_optimizer_step_stale_are_fatal_before_any_step():
    """The fault K9(b) exists for, at its smallest real size: the learner holds weights ONE optimizer step
    (the recipe lr, 2.8e-5) newer than the rollout's. FATAL before any optimizer step of the update. (On
    CUDA the same fault's smallest micro-batch statistic is 200x the fp32 bar —
    `designs/training/learner_gates.md`.)"""
    model = _learner("off")
    model.n_epochs, model.batch_size, model.grad_accum_steps = 1, L.N_STEPS * L.N_ENVS, 1
    _train(model)                                         # exactly ONE optimizer step on the rows
    L.load_buffer_into(model)                             # the rollout's (now stale) behaviour log-probs
    model.n_epochs, model.batch_size, model.grad_accum_steps = L.GOLDEN_OVERRIDES["n_epochs"], L.GOLDEN_OVERRIDES["batch_size"], 1
    model.behaviour_check = "fatal"
    before = _params(model)
    with pytest.raises(BehaviourMismatch, match=r"CURRENT policy version.*fp32"):
        _train(model)
    assert th.equal(before, _params(model)), "the gate must fire before any optimizer step"


def test_warn_reports_and_off_skips():
    model = _learner("off")
    assert G.behaviour_gate_mode(model) == "off"
    model.behaviour_check = "warn"
    n = model.rollout_buffer.log_probs[0].size
    out = _judge(model, th.full((n,), 1e-3, dtype=th.float64))       # over the bar: warns, never raises
    assert out["behaviour/max_abs_dlogp_judged"] == pytest.approx(1e-3, rel=1e-6)


def test_every_check_but_off_is_the_probe_with_or_without_a_version_record():
    model = _learner("fatal")
    assert G.behaviour_gate_mode(model) == "probe"
    model._rust_row_versions = np.zeros((L.N_STEPS, L.N_ENVS), np.int64)
    assert G.behaviour_gate_mode(model) == "probe"
    assert not hasattr(G, "check_behaviour_first_micro") and not hasattr(G, "behaviour_margins_first_micro")


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


def test_the_gate_is_ONE_fp32_table_and_a_process_at_any_other_precision_is_refused():
    """ONE table (`consistency.BEHAVIOUR_GATE`), ONE enforcement (`enforce_behaviour`): fp32 = DETERMINISTIC — max < 1e-4 over the rows not at a tie, single-shot, and the
    excluded share under its ceiling. fp32 'highest' is the only precision (TF32 retired, K2): a process
    at any other — 'high' included — is refused, never judged at a bar measured elsewhere."""
    from agents.training.rust_rollout import consistency as K

    C = K.GateCondition
    assert K.behaviour_gate() == (C("max", 1e-4, K.FP32_TIE_EPS),
                                  C("excluded_frac", K.FP32_EXCLUDED_CEILING, K.FP32_TIE_EPS))
    assert K.BEHAVIOUR_BAR == 1e-4
    for gone in ("BEHAVIOUR_GATES", "TF32_MAX_PERSISTENCE", "TF32_P99_MULTIPLE", "TF32_MAX_MULTIPLE"):
        assert not hasattr(K, gone), gone
    model = _learner("fatal")
    mid = th.full((2048,), 1e-3, dtype=th.float64)          # above the fp32 bar
    for other in ("high", "medium"):
        with _precision(other):
            with pytest.raises(K.UndeclaredPrecision, match=f"'{other}'"):
                K.behaviour_gate()
            with pytest.raises(K.UndeclaredPrecision, match=f"'{other}'"):
                _judge(model, mid)
    with pytest.raises(BehaviourMismatch, match="fp32"):
        _judge(model, mid)                                                                  # single-shot


class _precision:
    def __init__(self, p):
        self.p = p

    def __enter__(self):
        self.prev = th.get_float32_matmul_precision()
        th.set_float32_matmul_precision(self.p)

    def __exit__(self, *a):
        th.set_float32_matmul_precision(self.prev)


def _fp32(bad_rows: int, value: float = 1e-3):
    """A 2,048-row |Δ| micro-batch at fp32's healthy noise (~1e-6) plus ``bad_rows`` rows over its bar."""
    g = th.Generator().manual_seed(3)
    d = th.rand(2048, generator=g, dtype=th.float64) * 2e-6
    d[th.randperm(2048, generator=g)[:bad_rows]] = value
    return d


def _no_tie(n, tied=()):
    """Tie margins (`tie_margins`) of ``n`` rows: every row far from a cutoff, except ``tied`` (exact ties)."""
    m = np.ones(n)
    m[list(tied)] = 0.0
    return m, ["test" for _ in range(n)]


def _judge(model, d, margins=None, *, with_margins=True):
    """The ONE enforcement (`consistency.enforce_behaviour`) on per-row |Δ| ``d`` — what the probe calls on
    its current rows. Returns its metrics."""
    from agents.training.rust_rollout import consistency as K

    m = (margins if margins is not None else _no_tie(d.numel())) if with_margins else (None, None)
    return K.enforce_behaviour(model, d.detach().abs().double().numpy(), where="the test's rows",
                               actions=th.arange(d.numel()) % 11, masks=th.ones(d.numel(), 11),
                               margins=m[0], sites=m[1])


def _step(model, d, margins=None):
    return _judge(model, d, margins)


def test_a_violation_is_fatal_at_once_and_dumps_the_offending_rows_first(tmp_path, capsys):
    """Single-shot: the FIRST violation raises (no persistence, no warning grace), and the rows are
    dumped BEFORE the raise. Revert the dump-before-raise order ⇒ no `behaviour_violations.jsonl`."""
    from agents.training.rust_rollout import consistency as K

    model = _learner("fatal")
    model.behaviour_dump_dir = str(tmp_path)
    d = _fp32(bad_rows=10, value=0.3)
    with pytest.raises(BehaviourMismatch, match=r"max 0\.3 NOT < 0\.0001"):
        _step(model, d)
    out = capsys.readouterr().out
    assert "🛑" in out and "single-shot, FATAL at once" in out
    import json
    rec = json.loads((tmp_path / K.VIOLATION_DUMP).read_text().splitlines()[0])
    assert rec["precision"] == "highest" and rec["rows_judged"] == 2048
    assert [c["statistic"] for c in rec["conditions"]] == ["max", "excluded_frac"]
    top = rec["rows"][:10]
    assert all(r["abs_dlogp"] == 0.3 for r in top) and {"index", "action", "mask"} <= set(top[0])
    assert top[0]["mask"] == "1" * 11


def test_the_fp32_rule_is_deterministic_and_a_nan_is_always_fatal():
    """fp32 (`gen3_behaviour_tie_exclusion_v1`): ONE judged row over 1e-4 is FATAL on the
    FIRST update; the same row AT a tie is excluded (no violation); too many tied rows FATAL; a NaN — even
    on an excluded row — is never rounding; and fp32 without margins is refused, never judged blind."""
    from agents.training.rust_rollout import consistency as K
    from agents.training.rust_rollout.tie_margins import TieMarginError

    model = _learner("fatal")
    with pytest.raises(BehaviourMismatch, match=r"max 0\.005 NOT < 0\.0001"):
        _step(model, th.full((2048,), 5e-3, dtype=th.float64))         # a GLOBAL fault: first update
    d = _fp32(bad_rows=1)
    bad = int(th.argmax(d))
    with pytest.raises(BehaviourMismatch, match=r"max 0\.001 NOT < 0\.0001"):
        _step(model, d)                                                # not at a tie: FATAL at once
    out = _step(model, d, margins=_no_tie(2048, tied=[bad]))           # the same row AT a tie: excluded
    assert out["behaviour/rows_excluded"] == 1.0
    nan = _fp32(bad_rows=0)
    nan[5] = float("nan")
    with pytest.raises(BehaviourMismatch, match="NON-FINITE"):
        _step(model, nan, margins=_no_tie(2048, tied=[5]))             # a NaN on an EXCLUDED row
    many = list(range(int(K.FP32_EXCLUDED_CEILING * 2048) + 1))
    with pytest.raises(BehaviourMismatch, match="TOO MANY rows sit at a tie"):
        _step(model, _fp32(bad_rows=0), margins=_no_tie(2048, tied=many))
    with pytest.raises(TieMarginError):
        _judge(model, _fp32(bad_rows=0), with_margins=False)


def _wide_learner(n_steps: int = 64):
    """The golden learner at a 256-row buffer (the 64 real rows tiled), one 256-row micro-batch, lr 0
    so every update sees the same weights — a "fresh rollout" each update with the SAME fault."""
    from agents.training.rust_rollout import testkit as TK
    from agents.training.train_logger import configure
    from agents.training.rust_vec_env import RustVecEnv
    from main.train.production_args import production_args
    from main.train.model_build import apply_training_hparams

    _a, obs, act = TK.production_spaces()
    env = RustVecEnv(n_envs=L.N_ENVS, observation_space=obs, action_space=act, build=lambda m: None)
    m = TK.fresh_model(env, n_steps=n_steps, batch_size=n_steps * L.N_ENVS, n_epochs=1, seed=L.MODEL_SEED,
                       perturb_seed=L.PERTURB_SEED, learning_rate=0.0)
    apply_training_hparams(m, production_args(), mappings=None)
    TK.unset_to_class_defaults(m)
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
        dist = m.policy.get_distribution({k: th.as_tensor(v, device=m.device) for k, v in obs.items()},
                                         action_masks=masks)
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


def test_a_localized_fault_on_the_real_buffer_is_fatal_on_the_first_update_before_any_step():
    """Real rows through the REAL `train()`: 2 of 256 rows (0.8 %) point their action at the least likely
    other legal action (a wrong action index — the fault a percentile statistic cannot see under 1 %).
    The fp32 gate is on the max over the judged rows, single-shot: FATAL on the first update, before any
    optimizer step. The clean buffer (control) trains through the same gate."""
    m, rows = _wide_learner()
    n = 256
    masks = rows["action_masks"].reshape(n, -1)
    corrupt = [i for i in range(0, n, 37) if (masks[i] > 0.5).sum() >= 2][:2]
    assert len(corrupt) == 2
    _fill(m, rows, [])
    np.random.seed(0)
    m.train()                                                          # control: no raise
    assert m.logger.name_to_value["behaviour/max_abs_dlogp_current"] < 1e-4
    _fill(m, rows, corrupt)
    before = _params(m)
    with pytest.raises(BehaviourMismatch, match=r"max .* NOT < 0\.0001"):
        m.train()
    assert th.equal(before, _params(m))


_BATTERY = "designs/research_state/measurements/learner_battery_2026-09-26/"
_PAUSED_ARGVS = ("argv_L95.txt", "scripts/argv_T32_STANDIN.txt", "validation/reruns_2026-09-28/argv_Cfix.txt",
                 "validation/reruns_2026-09-28/argv_T32b_sentinel.txt")


@pytest.mark.parametrize("name", _PAUSED_ARGVS)
def test_the_paused_battery_argvs_parse_unchanged_under_k9(name):
    """The four paused learner-battery arms (none typing `--behaviour-check`) must still launch: K9(b)
    resolves to `fatal`, and no K9 rule refuses them. The launcher-only flags are stripped exactly as
    the launcher strips them, and so are the flags DELETED since the battery was recorded (every deletion
    pass since L2: the recorded argvs carry the entropy-boost / true-team / distillation / search-teacher
    defaults, `--matmul-precision high`, `--eval-battles 100`, `--self-play-use-cpu`, ... each at its OFF
    value; a pinned launch is judged by its OWN commit's parser, so HEAD's parser never sees them — what
    K9 does with the REST of the argv is what this pins)."""
    import shlex

    from main.train.combination_checks import failing_checks
    from main.train.rust_env_setup import resolve_env_core_args
    from main.train_rl_agent import build_parser
    from utils.paths import repo_path

    from agents.training.learner_benchmark import _unknown_to_the_trainer, split_flags

    toks = shlex.split(repo_path(*(_BATTERY + name).split("/")).read_text())
    # Every flag the trainer's parser no longer defines is dropped WITH its value (a recorded argv types the
    # whole family of levers every deletion unit removes, each at its OFF value) — the flag census (P11) made
    # a per-batch strip list a standing chore, so this is the same class fix `learner_benchmark` already uses.
    unknown = _unknown_to_the_trainer()
    argv = []
    for flag, vals in split_flags(toks):
        if flag in ("--restart-interval-hours", "--pin-commit") or unknown(flag):
            continue                                                  # launcher-only / deleted
        argv.append(flag)
        argv.extend(vals)
    args = build_parser().parse_args(argv)
    assert args.behaviour_check is None
    # Each arm is a `--model` FORK of a python-era checkpoint: it moves onto the Rust core at launch (announced
    # as a CORE SWITCH, `rust_env_setup.env_core_switch_line`, D4 — the Python core was deleted; `--env-core`
    # itself is deleted, so the strip above drops it from a recorded argv).
    assert "--model" in argv
    k9 = [c.name for c in failing_checks(args) if "behaviour_check" in set(c.dests)]
    assert not k9, k9
    resolve_env_core_args(args)
    from agents.training.rust_rollout.consistency import behaviour_gate

    assert args.behaviour_check == "fatal" and behaviour_gate()


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
