"""`gen3_metric_producer_guard_v1` — every pending metric key has ONE producer per dump interval.

THE CLASS. `Logger.record` replaces a pending key, so two call sites writing one key between two dumps lose the
earlier site's value SILENTLY. The metric-key collision fix (`109c4256`, 2026-10-15) found three of them by reading the
code; this file pins each fix on the REAL producers, under a STRICT logger (`train_logger.Logger.strict_producers`,
which pytest turns on by itself) so a revert raises `DuplicateProducerError` naming both call sites:

* `fork/*` — `rust_rollout/fork.py` recorded it at collection, and `train()` recorded the same stash again;
* `train/effective_batch` — the dose log and the adaptive-batch controller (and, with the controller on,
  `train/dose_rate` used the OLD K);
* `lifecycle/cuda_streams_after_freeze` (and `reserved_mib`, `device_free_mib`, `reserved_after_freeze_mib`) — a
  per-sample reading overwritten by the closed window's;
* the second half of the class — a registered metric that is NEVER produced: `grad/species_belief_*`
  (the bank registered the hidden-team term as `species_belief_set`; the probe looked up `species_belief`).
"""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch as th

from agents.training.train_logger import DuplicateProducerError, Logger


def _strict() -> Logger:
    return Logger(None, [], strict_producers=True)


# ── fork/* ──────────────────────────────────────────────────────────────────────────────────────
def test_the_fork_family_is_recorded_once_by_the_collection_pass_and_train_adds_none():
    """`RustForkPass.record` (collection) is THE producer; `train()` used to record the stash again, so every
    `fork/*` key had two producers and the later write won. Before the fix this raises `DuplicateProducerError`
    at `train()`'s first `fork/` record."""
    from agents.training.instrumented_ppo_test import _build_tiny_ppo
    from agents.training.rust_rollout.fork import RustForkPass

    model, _venv = _build_tiny_ppo(n_steps=8, n_envs=4)
    model.learn(total_timesteps=8 * 4)
    model._logger = _strict()
    metrics = {"rate": 0.25, "own_rows": 32.0, "opp_substituted": float("nan")}
    stub = types.SimpleNamespace(metrics=lambda rep: dict(metrics), guard=types.SimpleNamespace(streak=2))
    RustForkPass.record(stub, model, rep=None)
    assert model._fork_metrics == metrics                      # the stash stays (the benchmark and tests read it)
    model.train()                                              # a strict logger: a second `fork/` producer raises
    pending = dict(model.logger.name_to_value)
    assert pending["fork/rate"] == 0.25 and pending["fork/own_rows"] == 32.0
    assert pending["supply/fork_dry_streak"] == 2.0
    assert "fork/opp_substituted" not in pending, "a NaN is an empty slice: omitted, never logged"
    assert any(k.startswith("train/") for k in pending), "the update really ran"


# ── train/effective_batch + train/dose_rate ─────────────────────────────────────────────────────
def _dose_model(logger: Logger, *, k: int = 2) -> types.SimpleNamespace:
    m = types.SimpleNamespace(
        batch_size=1024, grad_accum_steps=k, n_epochs=4, num_timesteps=0, logger=logger, n_steps=2048, n_envs=1,
        policy=types.SimpleNamespace(optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-4}])),
        _dose_lr_flag=None, _fork_lr_pin=None, _dose_kl=None)
    m.noise_ratio_sample = lambda kind, b_big: (6.0, 500)       # noise-limited: the controller doubles K
    return m


def test_the_dose_log_is_registered_AFTER_the_adaptive_batch_controller(tmp_path, run_archive):
    """The order is the contract: the dose log reads the K the next `train()` runs, so it must follow the
    controller's decision (`main.train.callbacks.build_callbacks`)."""
    from main.train.callbacks import build_callbacks
    from main.train.config import resolve_config
    from main.train_rl_agent import build_parser

    def names(extra):
        p = build_parser()
        args = p.parse_args(["--steps", "15000000", "--n-envs", "48", "--allow-nonproduction-recipe", *extra])
        resolve_config(args, p)
        bundle = build_callbacks(
            args=args, model_dir=str(tmp_path / f"m{len(extra)}"), annealing_mode=False, _pool=None,
            _fixed_opponents=None, _bot_weight_vec=None, OPPONENT_NAMES=(), _specialist_team_str=None,
            _promote_threshold=0.6, _heuristic_floor=0.0, _sp_start_wr=0.5, _sp_full_wr=0.9)
        return [type(c).__name__ for c in bundle.callbacks]

    on, off = names(["--adaptive-batch", "policy"]), names([])
    assert on.index("AdaptiveBatchCallback") < on.index("DoseLogCallback"), on
    assert "AdaptiveBatchCallback" not in off and off.count("DoseLogCallback") == 1, off


def test_effective_batch_and_dose_rate_have_one_producer_and_read_the_k_the_controller_decided():
    """Both callbacks, in the order `build_callbacks` registers them, on a STRICT real logger. Before the fix the
    controller re-recorded `train/effective_batch` (a second producer: `DuplicateProducerError`) and
    `train/dose_rate` was computed from the OLD K (2) while the update ran at the new one (4)."""
    from agents.training.adaptive_batch_callback import AdaptiveBatchCallback, AdaptiveBatchController
    from agents.training.dose import dose_block
    from main.train.run_io import DoseLogCallback

    logger = _strict()
    model = _dose_model(logger, k=2)
    adaptive = AdaptiveBatchCallback(AdaptiveBatchController(mode="policy", every=1, max_accum=64), verbose=0)
    dose = DoseLogCallback()
    for cb in (adaptive, dose):                                  # the registration order
        cb.init_callback(model)
    adaptive.on_training_start()
    for cb in (adaptive, dose):
        cb.on_rollout_end()
    assert model.grad_accum_steps == 4, "the controller must have moved K (the test is vacuous otherwise)"
    pending = dict(logger.name_to_value)
    assert pending["train/effective_batch"] == 1024 * 4
    assert pending["train/dose_rate"] == pytest.approx(dose_block(model)["dose_rate_now"])
    assert pending["train/grad_accum_steps"] == 4


# ── lifecycle/cuda_* ────────────────────────────────────────────────────────────────────────────
def test_the_cuda_memory_watch_has_one_producer_per_tag_across_window_closes():
    """40 updates cross several window closes with no dump between them (the worst case). Before the fix the
    window close re-wrote `reserved_mib`, `device_free_mib`, `reserved_after_freeze_mib` and
    `streams_after_freeze` from the closed window and raised."""
    from agents.training import cuda_memory_trend as cmt
    from agents.training import learner_lifecycle as LL
    from agents.training.learner_lifecycle_test import _mem_sample

    logger = _strict()
    model = types.SimpleNamespace(_logger=logger, device="cpu", policy=None)

    def sampler(device, *, update, phase):
        s = _mem_sample(update, phase, 3000, reserved_mib=7814 + update)
        return cmt.MemorySample(**{**s.as_row(), "streams": 11 + (update % 5)})

    watch = LL.CudaMemoryWatch(model, emit=lambda m: None, say=lambda m: None, device="cuda:0", sampler=sampler)
    watch.start()
    for _ in range(40):
        watch.observe("post_rollout")
        watch.observe("post_update")
    assert watch.trend.windows, "the run must close at least one window"
    keys = set(logger.name_to_value)
    assert "lifecycle/cuda_streams_after_freeze" in keys and "lifecycle/cuda_streams_after_freeze_max" in keys
    assert logger.name_to_value["lifecycle/cuda_streams_after_freeze_max"] >= \
        logger.name_to_value["lifecycle/cuda_streams_after_freeze"]


# ── a registered gradient-probe term that is never produced ─────────────────────────────────────
def test_every_belief_bank_probe_name_is_a_looked_up_term():
    """`select_probe_terms` reads `_R1_PROBE_TERMS`, which is DERIVED from `belief_bank.ROWS`: a row registered
    under any name is found (this was `species_belief_set` vs `species_belief`)."""
    from agents.training.belief_bank import ROWS
    from agents.training.instrumented_ppo.ppo import _R1_PROBE_TERMS, select_probe_terms

    assert _R1_PROBE_TERMS == tuple(r.probe for r in ROWS) + ("win_prob",)
    terms = {name: th.ones(()) for name in _R1_PROBE_TERMS}
    assert list(select_probe_terms(terms, {n: True for n in terms})) == list(_R1_PROBE_TERMS)
    assert select_probe_terms(terms, {}) == {}, "a term the minibatch did not SCORE is not probed"
    assert select_probe_terms({}, {n: True for n in terms}) == {}


def test_a_real_update_produces_the_grad_family_of_every_registered_probe_term():
    """The production network's golden buffer holds every X5 case, so ONE `train()` scores every bank row. Each
    registered probe term must publish its `grad/<term>_{share,norm_shared,policy_cosine}` — and the names the
    launcher's panel reads for the hidden-team term (`grad/species_belief_*`) must be among them. (`win_prob` is
    de-duplicated into `value` under the win-prob critic by design: `grad_balance.grad_balance_metrics`.)"""
    from agents.training import learner_golden as L
    from agents.training.belief_bank import ROWS
    from main.launcher import format as launcher_format
    from utils.torch_state_guard import single_thread_build

    with single_thread_build():
        model = L.build_learner()
        model.logger.strict_producers = True                     # the whole production update, one producer per key
        L.load_buffer_into(model, L.BUFFER_PATH)
        np.random.seed(L.UPDATE_SEED)
        th.manual_seed(L.UPDATE_SEED)
        model.train()
        logged = set(model.logger.name_to_value)
    for term in [r.probe for r in ROWS] + ["opp_intent"]:
        for suffix in ("share", "norm_shared", "policy_cosine"):
            assert f"grad/{term}_{suffix}" in logged, f"grad/{term}_{suffix} is registered but never produced"
    panel = {k for k in launcher_format._METRIC_LABELS if k.startswith("grad/species_belief")}
    assert len(panel) == 3, panel                          # the panel names the family: the check has teeth
    assert panel <= logged, sorted(panel - logged)
    assert not model.logger.duplicate_producers


def test_a_duplicate_producer_error_is_the_fatal_config_class():
    """It is a `MetricKeyCollisionError` (FATAL_CONFIG): a smoke that finds one stops, with its reason."""
    from main.exit_codes import TrainExitCode, exit_code_for
    assert exit_code_for(DuplicateProducerError("x")) == int(TrainExitCode.FATAL_CONFIG)
