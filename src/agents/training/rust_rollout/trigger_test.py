"""Pins for the update trigger (M5 Lane G, order constraint 6): the declared target / band / quantum,
the adaptive-batch hook's refusals, and today's rollout size as the default."""
from __future__ import annotations

import pytest

from agents.training.rust_rollout.trigger import (SampleTrigger, TriggerError, WindowTrigger, ragged_accumulation,
                                                  trigger_for)


def test_the_default_target_is_todays_rollout_size_on_the_lcm_quantum():
    t = trigger_for("complete_game", n_envs=48, n_steps=2048, micro_batch=2048)
    assert isinstance(t, SampleTrigger) and t.target == 98_304 and t.quantum == 6144
    assert (t.lo, t.hi) == (98_304, 98_304)
    assert not t.ready(98_303) and t.ready(98_304) and t.ready(120_000) and t.take() == 98_304
    assert isinstance(trigger_for("window", n_envs=48, n_steps=2048, micro_batch=2048), WindowTrigger)


def test_the_live_shape_takes_a_ragged_accumulation_step_and_it_is_reported_not_refused():
    assert ragged_accumulation(98_304, 2048, 32)          # 1.5 x 65,536 (the recipe review's §3.3)
    assert not ragged_accumulation(131_072, 2048, 32)


def test_the_adaptive_hook_moves_the_target_inside_the_band_on_the_quantum_only():
    t = trigger_for("complete_game", n_envs=48, n_steps=2048, micro_batch=2048, target=98_304,
                    band_lo=49_152, band_hi=196_608)
    assert t.set_target(147_456) == 147_456 and t.moves == 1 and t.history == [98_304, 147_456]
    assert t.set_target(147_456) == 147_456 and t.moves == 1
    with pytest.raises(TriggerError, match="multiple"):
        t.set_target(147_457)
    with pytest.raises(TriggerError, match="band"):
        t.set_target(202_752)
    assert t.target == 147_456


def test_a_declaration_off_the_quantum_or_outside_its_band_is_refused():
    with pytest.raises(TriggerError, match="multiple"):
        SampleTrigger(target=1000, quantum=48, lo=1000, hi=1000)
    with pytest.raises(TriggerError, match="band"):
        SampleTrigger(target=96, quantum=48, lo=144, hi=192)
    with pytest.raises(TriggerError, match="unknown"):
        trigger_for("async", n_envs=4, n_steps=8, micro_batch=8)
