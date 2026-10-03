"""Pins for the update trigger (M5 Lane G, order constraint 6): the declared target / quantum and
today's rollout size as the default."""
from __future__ import annotations

import pytest

from agents.training.rust_rollout.trigger import (SampleTrigger, TriggerError, ragged_accumulation, trigger_for,
                                                  update_rows)


def test_the_default_target_is_todays_rollout_size_on_the_lcm_quantum():
    t = trigger_for(n_envs=48, n_steps=2048, micro_batch=2048)
    assert isinstance(t, SampleTrigger) and t.target == 98_304 and t.quantum == 6144
    assert not t.ready(98_303) and t.ready(98_304) and t.ready(120_000) and t.take() == 98_304


def test_update_rows_is_the_ONE_definition_of_the_update_size_and_the_trigger_agrees_with_it():
    """The rows one update trains on: the complete-game target (0 = n_steps x n_envs).
    `compile_trainer.check_shape_stability`'s callers judge THIS (P10-E, F9); the trigger the collector
    builds takes the same number, so the startup check and the runtime refusal
    (`RustCollector._ensure_buffer`) cannot disagree."""
    assert update_rows(n_envs=48, n_steps=1000) == 48_000                # 0 = n_steps x n_envs
    assert update_rows(n_envs=48, n_steps=1000, target=98_304) == 98_304
    for tgt in (0, 98_304):
        t = trigger_for(n_envs=48, n_steps=2048, micro_batch=2048, target=tgt)
        assert t.take() == update_rows(n_envs=48, n_steps=2048, target=tgt)


def test_the_live_shape_takes_a_ragged_accumulation_step_and_it_is_reported_not_refused():
    assert ragged_accumulation(98_304, 2048, 32)          # 1.5 x 65,536 (the recipe review's §3.3)
    assert not ragged_accumulation(131_072, 2048, 32)


def test_a_declaration_off_the_quantum_is_refused():
    with pytest.raises(TriggerError, match="multiple"):
        SampleTrigger(target=1000, quantum=48)
    with pytest.raises(TriggerError, match="multiple"):
        SampleTrigger(target=0, quantum=48)


@pytest.mark.parametrize("n_envs,micro,accum", [(48, 2048, 32), (48, 4096, 16), (64, 2048, 8), (5, 3, 4), (7, 6, 1)])
def test_no_legal_update_ever_hands_the_learner_a_ragged_micro_batch(n_envs, micro, accum):
    """The collector's answer to a ragged final micro-batch is that it CANNOT occur: every legal target
    (the default) is a multiple of lcm(micro, N), so every micro-batch is full — no padding rows, no
    dropped rows, one compiled learner shape. What CAN be short is the last ACCUMULATION group (fewer
    full micros), which the learner rescales to its real rows
    (``instrumented_ppo_test.test_grad_accum_matches_full_batch[4-3-12]``)."""
    t = trigger_for(n_envs=n_envs, n_steps=micro, micro_batch=micro)
    assert t.target % micro == 0 and t.target % n_envs == 0
    for off in (1, micro // 2 or 1, n_envs):
        if (t.target + off) % t.quantum:
            with pytest.raises(TriggerError, match="multiple"):
                SampleTrigger(target=t.target + off, quantum=t.quantum)


def test_a_learner_micro_batch_that_moved_under_a_built_collector_is_refused_not_padded():
    from types import SimpleNamespace

    from agents.training.rust_rollout.collector import RustCollector

    t = trigger_for(n_envs=48, n_steps=2048, micro_batch=2048)
    host = SimpleNamespace(cfg=SimpleNamespace(trigger=t), n=48)
    with pytest.raises(TriggerError, match="does not divide"):
        RustCollector._ensure_buffer(host, SimpleNamespace(batch_size=3000))
