"""`--debug` is SAFE BY CONSTRUCTION (`main.train.debug_shape`): a fresh smoke never declares an update
its one env cannot fill. Each test fails on a revert of the override (the namespace keeps the production
recipe's 98,304-row target, which one CPU env needs hours to complete — three agents stalled, 2026-10-07).
The real-process twin is `debug_shape_smoke_integration_test.py` (slow)."""
from __future__ import annotations

import contextlib
import io
from math import gcd

from main.train import debug_shape as D


def _ns(*argv: str):
    from main.checkargs import resolve_against_parent
    with contextlib.redirect_stdout(io.StringIO()):
        res = resolve_against_parent(list(argv))
    assert res is not None and res["ns"] is not None, res
    return res["ns"]


def _runtime_ok(ns) -> None:
    """The parse-time quantum (lcm(batch, n_envs)) AND the run-time one (ONE env: the batch)."""
    t, b, n = int(ns.rollout_target_samples), int(ns.batch_size), int(ns.n_envs)
    assert t % (b * n // gcd(b, n)) == 0 and t % b == 0, (t, b, n)
    assert t <= D.DEBUG_MAX_UPDATE_ROWS


def test_arch_production_debug_gets_the_smoke_shape():
    ns = _ns("--arch", "production", "--debug", "--steps", "10000")
    assert ns.rollout_target_samples == D.DEBUG_TARGET_ROWS == 2304
    assert ns.batch_size == D.DEBUG_BATCH_SIZE == 384
    assert ns.n_epochs == D.DEBUG_N_EPOCHS == 1
    assert set(getattr(ns, D.OVERRIDE_ATTR)) == {"rollout_target_samples", "batch_size", "n_epochs"}
    _runtime_ok(ns)


def test_a_typed_recipe_knob_still_wins_and_the_target_follows_its_quantum():
    ns = _ns("--arch", "production", "--debug", "--batch-size", "512", "--n-epochs", "3")
    assert ns.batch_size == 512 and ns.n_epochs == 3
    assert ns.rollout_target_samples == 2560            # ceil(2304 / lcm(512, 256)) x 512
    assert set(getattr(ns, D.OVERRIDE_ATTR)) == {"rollout_target_samples"}
    _runtime_ok(ns)


def test_a_typed_target_is_never_overridden():
    ns = _ns("--arch", "production", "--debug", "--rollout-target-samples", "98304")
    assert ns.rollout_target_samples == 98304
    assert not getattr(ns, D.OVERRIDE_ATTR, None)


def test_no_debug_keeps_the_production_recipe():
    ns = _ns("--arch", "production", "--steps", "10000")
    assert ns.rollout_target_samples == 98304 and ns.batch_size == 2048 and ns.n_epochs == 10
    assert not getattr(ns, D.OVERRIDE_ATTR, None)


def test_the_default_debug_smoke_is_unchanged():
    ns = _ns("--debug", "--steps", "10000")
    assert not ns.rollout_target_samples
    assert ns.batch_size == 4096 and ns.n_epochs == 5
    assert not getattr(ns, D.OVERRIDE_ATTR, None)


def test_the_recipe_surface_reports_the_override_as_debug_never_a_silent_drift():
    from main.checkargs import check
    rep = check(["--arch", "production", "--debug", "--steps", "10000"])["recipe"]
    debug = {d.dest for d in rep.diffs if d.source == "debug"}
    assert debug == {"rollout_target_samples", "batch_size", "n_epochs"}, rep.diffs
    assert not rep.silent and not rep.refuses
