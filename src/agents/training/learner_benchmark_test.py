"""Pins for the learner-only PPO update benchmark and the phase hook it installs in `train()`.

The hook's contract is "zero cost and zero effect when off": production never installs one, every
call site in `train()` is a single guarded `is not None` test, and installing one changes no
number the fold computes. The behavioural half runs a tiny real `InstrumentedMaskablePPO` on CPU.
"""
import inspect
import re

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from agents.training import learner_benchmark as lb
from agents.training.instrumented_ppo import InstrumentedMaskablePPO, phase_hook


# ---------------------------------------------------------------- the hook: off, guarded, inert

def test_phase_hook_is_off_by_default():
    assert phase_hook.PHASE_HOOK is None
    assert phase_hook.current() is None


def test_installed_restores_the_previous_hook():
    seen = []
    with phase_hook.installed(seen.append):
        assert phase_hook.current() is not None
    assert phase_hook.current() is None


def test_every_mark_in_train_is_guarded_and_named():
    src = inspect.getsource(InstrumentedMaskablePPO.train)
    assert src.count("_ph = _current_phase_hook()") == 1, "the hook must be read ONCE per call"
    assert len(re.findall(r"\b_ph\s*=", src)) == 1, "_ph is rebound somewhere in train()"
    calls = [ln.strip() for ln in src.splitlines() if "_ph(" in ln]
    assert calls, "no phase marks found in train()"
    names = []
    for ln in calls:
        m = re.fullmatch(r'if _ph is not None: _ph\("([a-z_]+)"\)', ln)
        assert m, f"an unguarded or non-literal phase mark: {ln!r}"
        names.append(m.group(1))
    assert set(names) <= set(phase_hook.PHASES)
    assert set(phase_hook.PHASES) <= set(names), "a documented phase is never marked"
    assert names[0] == "start" and names[-1] == "logging"


class _CounterDictEnv(gym.Env):
    """Tiny Dict-obs maskable env (the `instrumented_ppo_test` fixture's shape)."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1e4, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(2,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(2)
        self._t = 0

    def action_masks(self):
        return np.ones(2, dtype=np.int8)

    def _obs(self):
        return {"observation": np.array([float(self._t % 17)], dtype=np.float32),
                "action_mask": np.ones(2, dtype=np.int8)}

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        return self._obs(), float((self._t * 7) % 5), self._t >= 1000, False, {}


@pytest.fixture
def tiny_model():
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([(lambda: _CounterDictEnv()) for _ in range(4)])
    model = InstrumentedMaskablePPO("MultiInputPolicy", venv, n_steps=8, batch_size=4,
                                    n_epochs=2, ent_coef=0.0, vf_coef=0.5, device="cpu", seed=0)
    from agents.training.rust_rollout.testkit import attach_vec_collector
    attach_vec_collector(model)         # the rollout comes from the toy VecEnv (the Rust collector is production's)
    model.grad_accum_steps = 2          # takes the accumulation AND the noise-probe branches
    # Stop learn() at its first train(): the buffer is then exactly as a real update would see it.
    orig = InstrumentedMaskablePPO.train
    InstrumentedMaskablePPO.train = lambda self: None
    try:
        model.learn(total_timesteps=8 * 4)
    finally:
        InstrumentedMaskablePPO.train = orig
    return model


def _params(model):
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


def _train_from(model, pristine, state0, seed=7):
    lb.restore_buffer_state(model.rollout_buffer, pristine)
    lb.restore_model_state(model, state0)
    lb.seed_all(seed)
    model.train()
    return _params(model)


def test_hook_changes_no_number_and_restore_repeats_the_same_work(tiny_model):
    m = tiny_model
    pristine = lb.capture_buffer_state(m.rollout_buffer)
    state0 = lb.capture_model_state(m)
    off = _train_from(m, pristine, state0)
    loss_off = m.logger.name_to_value["train/loss"]
    # Restoring really restores: a second hook-off call is bit-identical to the first.
    again = _train_from(m, pristine, state0)
    for k in off:
        assert th.equal(off[k], again[k]), f"restore did not reproduce the update at {k}"
    # And the update really moved the weights (so the equality above is not vacuous).
    assert any(not th.equal(off[k], state0["policy"][k]) for k in off)

    timer = lb.PhaseTimer(sync=None)
    with phase_hook.installed(timer):
        on = _train_from(m, pristine, state0)
    for k in off:
        assert th.equal(off[k], on[k]), f"installing the hook changed parameter {k}"
    assert m.logger.name_to_value["train/loss"] == loss_off

    seq = timer.sequence
    n_micro, n_epochs = lb._n_micro(m), int(m.n_epochs)
    assert seq[:4] == ["start", "setup", "batch", "forward"] and seq[-1] == "logging"
    for name in ("batch", "forward", "loss", "kl", "noise_probe", "backward", "optim"):
        assert seq.count(name) == n_micro * n_epochs, name
    assert seq.count("epoch_end") == n_epochs
    assert timer.counts["probes"] == 2 * n_micro * n_epochs
    assert set(timer.seconds) == set(phase_hook.PHASES) - {"start"}


def test_hook_off_path_does_not_call_anything(tiny_model):
    """With no hook installed nothing is called — a hook installed AFTER the call's read is ignored
    too, because `train()` reads it once, at the top."""
    m = tiny_model
    pristine = lb.capture_buffer_state(m.rollout_buffer)
    state0 = lb.capture_model_state(m)
    _train_from(m, pristine, state0)          # would raise below if anything called the hook
    assert phase_hook.current() is None


# ---------------------------------------------------------------- pure helpers

def test_phase_timer_books_each_segment_to_the_closing_mark():
    clock = iter([10.0, 10.5, 11.5, 11.75, 20.0, 20.25]).__next__
    t = lb.PhaseTimer(sync=None, clock=clock)
    for name in ("start", "batch", "forward", "batch", "start", "logging"):
        t(name)
    assert t.seconds == {"batch": pytest.approx(0.5 + 0.25), "forward": pytest.approx(1.0),
                         "logging": pytest.approx(0.25)}
    assert t.counts == {"batch": 2, "forward": 1, "logging": 1}


def test_phase_timer_syncs_before_reading_the_clock():
    order = []
    t = lb.PhaseTimer(sync=lambda: order.append("sync"), clock=lambda: order.append("clock") or 0.0)
    t("start")
    assert order == ["sync", "clock"]


def test_interval_union_counts_overlap_once():
    assert lb.interval_union([]) == 0.0
    assert lb.interval_union([(0, 10), (5, 15), (20, 25), (24, 24)]) == 20.0
    assert lb.interval_union([(3, 4), (0, 1)]) == 2.0


_CMD = ("/x/src/main/launcher/__main__.py --device cuda --ent-coef 0.05 --compile-opponents "
        "--compile-opponents-strict --compile-trainer --n-envs 48 --device cuda "
        "--restart-interval-hours 3 --steps 83005952 --pin-commit abc --run-name ai_x "
        "--model models/p/final_model.zip --fork-lr 2.8e-05 --fork-lr-freeze "
        "--matmul-precision highest --tb-inherit")


def _flags(argv):
    return [t for t in argv if t.startswith("--")]


def test_trainer_argv_repoints_the_recorded_command_as_a_fork():
    from main.checkargs import LAUNCHER_ONLY
    a = lb.build_trainer_argv(_CMD, model_zip="/m/final_model.zip", run_dir="/arch/run",
                              steps=99, device="cuda", launcher_only=LAUNCHER_ONLY)
    f = _flags(a)
    assert "--restart-interval-hours" not in f and "--pin-commit" not in f
    assert "--run-name" not in f and "--tb-inherit" not in f
    for flag in ("--model", "--run-dir", "--steps", "--device", "--eval-freq", "--behaviour-check"):
        assert f.count(flag) == 1, flag
    # `--matmul-precision` (typed in C's recorded command) is DELETED with TF32 (deletion pass K2): it
    # is stripped and never re-supplied, or the trainer's parser would refuse the argv
    assert "--matmul-precision" not in f and "--matmul_precision" not in f
    assert "highest" not in a
    # the pinned buffer's behaviour log-probs are another program's: K9(b) warns, never FATALs
    assert a[a.index("--behaviour-check") + 1] == "warn"
    assert a[a.index("--model") + 1] == "/m/final_model.zip"
    assert a[a.index("--run-dir") + 1] == "/arch/run"
    assert a[a.index("--steps") + 1] == "99"
    assert a[a.index("--device") + 1] == "cuda"
    # C's training flags survive verbatim, the compile included — except the `--compile-opponents` family,
    # deleted with the Python env core (deletion pass U3): the unknown-flag filter drops it.
    for flag in ("--ent-coef", "--compile-trainer", "--fork-lr", "--fork-lr-freeze", "--n-envs"):
        assert flag in f, flag
    assert "--compile-opponents" not in f and "--compile-opponents-strict" not in f
    assert "--no-tb-inherit" in f
    assert not any("train_rl_" + "agent" in t for t in a)


def test_trainer_argv_drops_flags_the_trainers_parser_no_longer_knows():
    """C's recorded command types flags later deleted (TF32's `--matmul-precision`, L2's entropy boosts and
    true-team toggle, U3's `--compile-opponents` family): one unknown flag is an argparse exit that kills the worker before it measures.
    Revert ⇒ the argv carries them and the REAL trainer parser refuses it."""
    from main.checkargs import LAUNCHER_ONLY
    from main.train.parser import build_parser

    cmd = _CMD + " --defensive-entropy-boost 1.0 --defensive-entropy-anneal-frac 0.0 --no-value-true-team"
    a = lb.build_trainer_argv(cmd, model_zip="/m.zip", run_dir="/r", steps=1, device="cuda",
                              launcher_only=LAUNCHER_ONLY)
    f = _flags(a)
    for gone in ("--defensive-entropy-boost", "--defensive-entropy-anneal-frac", "--no-value-true-team",
                 "--matmul-precision"):
        assert gone not in f, gone
    assert "1.0" not in a and "highest" not in a, "a dropped flag's VALUE goes with it"
    assert a[a.index("--ent-coef") + 1] == "0.05" and "--fork-lr" in f
    build_parser().parse_args(a)                     # the real parser accepts what is left


def test_trainer_argv_cpu_and_tiny_swap_the_compile_and_the_env_count():
    a = lb.build_trainer_argv(_CMD, model_zip="/m.zip", run_dir="/r", steps=1, device="cpu",
                              tiny=True)
    f = _flags(a)
    assert "--compile-trainer" not in f and "--compile-opponents" not in f
    assert "--no-compile-trainer" in f and "--no-compile-opponents" not in f
    assert a[a.index("--n-envs") + 1] == str(lb.TINY["n_envs"]) and f.count("--n-envs") == 1


def test_trainer_argv_refuses_the_trainer_module_name():
    with pytest.raises(ValueError):
        lb.build_trainer_argv(_CMD + " --critic /x/train_rl_" + "agent", model_zip="/m.zip",
                              run_dir="/r", steps=1, device="cuda")


def test_recorded_steps():
    assert lb.recorded_steps(_CMD) == 83005952
    assert lb.recorded_steps("--ent-coef 1") is None


def test_gpu_busy_reason():
    assert lb.gpu_busy_reason("", {1: "bash", 2: "python x.py"}, own_pid=2) is None
    why = lb.gpu_busy_reason("123, python3, 7000 MiB\n", {}, own_pid=1)
    assert why and "1 CUDA compute" in why
    why = lb.gpu_busy_reason("", {7: "python /t/src/main/train_rl_" + "agent.py --x"}, own_pid=1)
    assert why and "7" in why
    # Our own process never counts against us.
    assert lb.gpu_busy_reason("", {1: "train_rl_" + "agent"}, own_pid=1) is None


def test_buffer_state_round_trip_survives_in_place_edits():
    class _Buf:
        pass
    b = _Buf()
    b.observations = {"obs": np.arange(6, dtype=np.float32).reshape(3, 2, 1)}
    b.actions = np.zeros((3, 2, 1))
    b.generator_ready, b.buffer_size, b.device = False, 3, "cpu"
    st = lb.capture_buffer_state(b)
    assert "device" not in st
    b.observations["obs"][:] = -1          # what `get()`'s flatten / the label shift do in place
    b.generator_ready = True
    lb.restore_buffer_state(b, st)
    assert b.observations["obs"].sum() == 15 and b.generator_ready is False
    b.observations["obs"][:] = -1
    lb.restore_buffer_state(b, st)         # the pristine copy itself was never aliased
    assert b.observations["obs"].sum() == 15
    assert lb.buffer_fingerprint(st)["observations"] == {"obs": [[3, 2, 1], "float32"]}


def test_is_plain():
    assert lb._is_plain({"a": [1, 2.0, None], "b": np.float32(1)})
    assert not lb._is_plain({"a": th.zeros(1)})
    assert not lb._is_plain(lambda: 0)


def test_summarise_calls_skips_warmup_and_reports_phases():
    calls = [
        {"config": "baseline", "warmup": True, "train_ms": 9e9, "wall_s": 1.0, "work": {}},
        {"config": "baseline", "warmup": False, "train_ms": 1000.0, "wall_s": 1.0,
         "phases_s": {"forward": 0.4, "backward": 0.5}, "phase_counts": {"forward": 2},
         "work": {"train/loss": 1.0, "param_abs_sum": 3.0}},
        {"config": "baseline", "warmup": False, "train_ms": 3000.0, "wall_s": 3.0,
         "phases_s": {"forward": 0.6, "backward": 0.7}, "phase_counts": {"forward": 2},
         "work": {"train/loss": 1.0, "param_abs_sum": 3.0}},
    ]
    s = lb.summarise_calls(calls)
    assert s["n"] == 2 and s["train_ms"]["median"] == 2000.0 and s["train_ms"]["max"] == 3000.0
    assert s["phases"]["forward"]["s_per_update"] == pytest.approx(0.5)
    assert s["phases"]["forward"]["pct_of_bracketed_train_ms"] == pytest.approx(25.0)
    assert s["work_spread"]["train/loss"] == {"min": 1.0, "max": 1.0}


# ------------------------------------- the worker's hooks: the STARTUP update is production's own

@pytest.fixture
def clean_worker_state():
    """`_WORKER` is process-global (the worker owns the process); a test starts and leaves it empty."""
    saved = dict(lb._WORKER)
    lb._WORKER.clear()
    try:
        yield lb._WORKER
    finally:
        lb._WORKER.clear()
        lb._WORKER.update(saved)


def test_the_tool_never_takes_the_trainers_startup_update(clean_worker_state):
    """K2 (2026-10-02): the trainer runs a real `train()` BEFORE `learn()` — the CUDA fit check's dry
    update (`update_fit.dry_update`, here run for real on the learner golden's CPU learner) — and a
    worker that replaced `train` for the whole process took that call as its measurement: the time stage
    timed a 4,096-row fixture (0.32 s) and exited before the pinned buffer was ever restored. The
    startup update must run the ORIGINAL `train`; the tool's takes over at the learn loop's first
    collection. Fails on revert: with `train` replaced unconditionally the dry update lands on the tool."""
    import agents.training.update_fit as UF
    from agents.training import learner_golden as LG

    m = LG.build_learner()
    LG.load_buffer_into(m)
    cls = type(m)
    orig_train, orig_collect = cls.train, cls.collect_rollouts
    seen = {"tool": 0, "orig": 0}

    def counting_orig(self):
        seen["orig"] += 1
        return orig_train(self)

    def tool(self):
        seen["tool"] += 1

    cls.train = counting_orig
    try:
        lb.install_worker_hooks(cls, tool_train=tool, buffer_collect=lambda self, *a, **k: True)
        UF.dry_update(m)                                   # the trainer's STARTUP update
        assert seen == {"tool": 0, "orig": 1}, seen
        assert not lb._WORKER.get("learning")
        assert cls.collect_rollouts(m, None, None, None, 0) is True      # the learn loop's first collect
        assert lb._WORKER["learning"] is True
        m.train()
        assert seen == {"tool": 1, "orig": 1}, seen         # now the tool's, and only now
        assert lb._WORKER["_orig_train"] is counting_orig
    finally:
        cls.train, cls.collect_rollouts = orig_train, orig_collect


def test_the_hooks_wrap_the_classes_own_collect_when_no_buffer_is_reused(clean_worker_state):
    """The benchmark's FRESH-collection mode keeps the real `collect_rollouts` — marked, not replaced."""
    class _Cls:
        calls = 0

        def train(self):
            return "orig"

        def collect_rollouts(self, *a, **k):
            type(self).calls += 1
            return "real"

    lb.install_worker_hooks(_Cls, tool_train=lambda self: "tool")
    o = _Cls()
    assert o.train() == "orig"                              # before the loop: production's
    assert o.collect_rollouts() == "real" and _Cls.calls == 1
    assert o.train() == "tool"


def test_bench_collect_records_that_the_pinned_buffer_was_restored(tiny_model, tmp_path, clean_worker_state):
    m = tiny_model
    state = lb.capture_buffer_state(m.rollout_buffer)
    path = tmp_path / "rollout_buffer.pkl"
    lb.save_buffer(path, state, {}, {"n_steps": 8, "n_envs": 4})
    lb._WORKER["buffer_in"] = str(path)
    assert not lb._WORKER.get("buffer_restored")
    assert lb._bench_collect(m, None, None, m.rollout_buffer, 8) is True
    assert lb._WORKER["buffer_restored"] is True
