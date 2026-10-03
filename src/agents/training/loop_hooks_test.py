"""`gen3_declared_loop_hooks_v1` — the PPO loop's DECLARED hook table (`agents/training/loop_hooks.py`).

The table's rules (an undeclared owner or point, a duplicate, a registration after the table froze —
each the typed FATAL_CONFIG error), its ORDER (outermost first by `HOOK_OWNERS`, whatever order the
owners registered in), and the production learner's use of it: `OwnedLoop.learn` opens learn > collect
| update in that nesting, the two real owners (`learner_lifecycle.attach`, `CompileControl.attach`)
register INTO the table instead of reassigning bound methods. (The `sb3_reference` seam, whose arm this
test also covered, was deleted in PPO stage 3 — deletion pass U4.)
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator, List

import numpy as np
import pytest

from agents.training import learner_golden as LG
from agents.training import loop_hooks as H
from main.exit_codes import FatalConfigError


def _cm(log: List[str], tag: str) -> Any:
    @contextlib.contextmanager
    def f() -> Iterator[None]:
        log.append(f"{tag}>")
        yield
        log.append(f"<{tag}")
    return f


def test_rules_each_refusal_is_the_typed_fatal() -> None:
    t = H.LoopHooks()
    with pytest.raises(H.LoopHookError, match="undeclared hook owner"):
        t.register("profiler", "update", _cm([], "x"))
    with pytest.raises(H.LoopHookError, match="unknown hook point"):
        t.register("compile_sentinel", "dump", _cm([], "x"))
    t.register("compile_sentinel", "update", _cm([], "x"))
    with pytest.raises(H.LoopHookError, match="already hooks"):
        t.register("compile_sentinel", "update", _cm([], "x"))
    t.freeze("training start")
    with pytest.raises(H.LoopHookError, match="after the table froze"):
        t.register("learner_freeze", "update", _cm([], "x"))
    assert issubclass(H.LoopHookError, FatalConfigError)


def test_order_is_the_table_not_the_registration() -> None:
    log: List[str] = []
    t = H.LoopHooks()
    t.register("compile_sentinel", "collect", _cm(log, "compile"))     # registered FIRST
    t.register("learner_freeze", "collect", _cm(log, "freeze"))
    with t.around("collect"):
        log.append("body")
    assert log == ["freeze>", "compile>", "body", "<compile", "<freeze"]
    assert H.HOOK_OWNERS == ("learner_freeze", "compile_sentinel")
    assert H.HOOK_POINTS == ("learn", "collect", "update")


def test_after_yield_runs_only_on_a_normal_return() -> None:
    log: List[str] = []
    t = H.LoopHooks()
    t.register("learner_freeze", "update", _cm(log, "freeze"))
    with pytest.raises(RuntimeError):
        with t.around("update"):
            raise RuntimeError("the update died")
    assert log == ["freeze>"]


def test_a_duck_typed_model_gets_the_same_bodies_as_wrappers() -> None:
    log: List[str] = []

    class _M:
        def collect_rollouts(self) -> bool:
            log.append("collect")
            return True

    m = _M()
    H.install(m, "compile_sentinel", {"collect": _cm(log, "compile")})
    H.install(m, "learner_freeze", {"collect": _cm(log, "freeze")})   # attached last = outermost
    assert m.collect_rollouts() is True
    assert log == ["freeze>", "compile>", "collect", "<compile", "<freeze"]


# --------------------------------------------------------------------------------- the real learner
def _learner() -> Any:
    from agents.training.rust_rollout.testkit import ScriptedVecEnv, attach_vec_collector

    _, obs_space, act_space = LG._spaces()
    with np.load(LG.BUFFER_PATH) as z:
        data = {k: z[k] for k in z.files}
    return attach_vec_collector(LG.build_learner(ScriptedVecEnv(obs_space, act_space, data, LG.N_ENVS)))


@pytest.fixture(scope="module")
def learner() -> Any:
    with LG._one_thread():
        return _learner()


def test_learn_opens_the_declared_points_in_their_nesting() -> None:
    with LG._one_thread():
        model = _learner()
        log: List[str] = []
        for owner in H.HOOK_OWNERS:
            for point in H.HOOK_POINTS:
                model._loop_hooks.register(owner, point, _cm(log, f"{owner[0]}:{point}"))
        model.learn(total_timesteps=2 * LG.N_STEPS * LG.N_ENVS)
    one_iter = ["l:collect>", "c:collect>", "<c:collect", "<l:collect",
                "l:update>", "c:update>", "<c:update", "<l:update"]
    assert log == ["l:learn>", "c:learn>", *one_iter, *one_iter, "<c:learn", "<l:learn"], log
    assert model._loop_hooks.frozen_at is not None


def test_the_real_freeze_guard_registers_into_the_table(learner: Any) -> None:
    from agents.training import learner_lifecycle as LL

    LL.declare_learner_startup(learner)
    fz = LL.attach(learner)
    assert not {"collect_rollouts", "train", "learn"} & set(vars(learner)), \
        "the freeze guard reassigned a bound method instead of registering a loop hook"
    assert all("learner_freeze" in learner._loop_hooks.owners(p) for p in H.HOOK_POINTS)
    with LG._one_thread():
        learner.learn(total_timesteps=2 * LG.N_STEPS * LG.N_ENVS)
    assert fz.checks == 4 and fz.frozen is None         # two rollouts + two updates checked, released
    with pytest.raises(H.LoopHookError, match="froze"):
        LL.attach(learner)                                # after training start: refused


def test_the_compile_sentinel_registers_into_the_table() -> None:
    from agents.model.compile_control import CompileControl

    model = _learner()
    CompileControl().attach(model)
    assert not {"collect_rollouts", "train", "learn"} & set(vars(model))
    assert all("compile_sentinel" in model._loop_hooks.owners(p) for p in H.HOOK_POINTS)
    assert model._compile_control is not None


def test_the_per_update_region_call_window_is_reset_by_the_update_hook() -> None:
    """gen3_no_silent_eager_v1's contract on the hook table: `CompileControl.record` — which TAKES the
    per-update route window — runs once per update through the `update` hook, after `train()`. Here
    every update counts ONE R1 call; dropped from the hook, the window would accumulate across
    updates and outlive learn()."""
    from agents.model import region_calls
    from agents.model.compile_control import CompileControl

    with LG._one_thread():
        model = _learner()
        CompileControl().attach(model)
        region_calls.take()
        orig_train = model.train

        def train_with_one_counted_call(*a: Any, **k: Any) -> Any:
            region_calls.count("R1_compiled")
            return orig_train(*a, **k)

        model.train = train_with_one_counted_call
        from agents.training.rust_rollout.testkit import record_dumps
        dumps = record_dumps(model)
        model.learn(total_timesteps=2 * LG.N_STEPS * LG.N_ENVS)
    assert region_calls.peek() == {}, "the last update's route window was not taken"
    # the LAST update's value, written by learn()'s final dump (P3; one counted call per update)
    assert dumps[-1].get("lifecycle/compiled_region_calls") == 1.0, \
        "the window accumulated across updates — the per-update take() did not run"
