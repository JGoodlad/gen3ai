"""E1 — the LOCKSTEP DIFFERENTIAL: the owned PPO loop IS upstream's (`design_own_ppo_loop.md` §4).

Two production-surface learners are built from the same seeds (the K9 golden's builder) on the same
SCRIPTED VecEnv, and each runs three `learn()` iterations (two epochs, a ragged accumulation group)
plus a save -> load -> resume leg — one under upstream sb3's loop (the `GEN3AI_PPO_LOOP=sb3_reference`
seam, which defers EVERY vendored method to its upstream original), one under `OwnedLoop`. Then
EXACT equality on everything the loop touches:

* the HOOK TRACE — every callback call in order, with `num_timesteps`, `n_calls`, the `self.locals`
  KEY SET (callbacks read the collect's local names) and `rollout_buffer.pos` at `on_step` (the
  win-prob labeller writes the row ABOUT to be added);
* the BUFFER at every `on_rollout_end` (every obs key and every flat array, byte for byte);
* every logger DUMP: its step and every key/value (wall clocks excluded);
* the parameters after every update, `_n_updates`, `_current_progress_remaining`, `ep_info_buffer`;
* the train()/collect_rollouts() INSTANCE WRAPPERS still intercept (K6 and the compile sentinel wrap
  them that way) — each sees every call under both loops.

The scripted env exercises the branches a real one would: episode ends with ``info["episode"]``,
win outcomes, and a `TimeLimit.truncated` end with a ``terminal_observation`` (the bootstrap branch).
The callbacks are the REAL ones that need no live env — `WinProbLabelCallback` (the `buf.pos` /
locals consumer), `AdaptivePPOCallback` (the KL controller reading `train/approx_kl` off the logger,
cooldown 0 so it moves the LR), `SignalMetricsCallback` — plus an EVAL STAND-IN that dumps the logger
mid-rollout at an earlier step, exactly as the eval callbacks do: the owned loop preserves that
defect (design §2.1) bit-for-bit, and this test is what says so.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from agents.training import learner_golden as LG
from agents.training.instrumented_ppo import loop as L

ITERATIONS = 3
RESUME_ITERATIONS = 1
_WALL = ("time/fps", "time/time_elapsed")


def _is_wall(k: str) -> bool:
    return k in _WALL or k.endswith("_ms") or k.endswith("_s")


class ScriptedVecEnv(VecEnv):
    """A deterministic VecEnv over the production spaces, serving the golden buffer's real rows."""

    def __init__(self, obs_space: Any, act_space: Any, data: Dict[str, np.ndarray]) -> None:
        super().__init__(LG.N_ENVS, obs_space, act_space)
        self.data = data
        self.keys = sorted(obs_space.spaces)
        self.t = 0
        self._actions = None

    def _row(self) -> int:
        return self.t % LG.N_STEPS

    def _obs(self) -> Dict[str, np.ndarray]:
        r = self._row()
        return {k: self.data["obs:" + k][r].copy() for k in self.keys}

    def reset(self) -> Dict[str, np.ndarray]:
        self.t = 0
        return self._obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions).copy()

    def step_wait(self) -> Tuple[Any, np.ndarray, np.ndarray, List[dict]]:
        prev = self._obs()
        self.t += 1
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.array([(self.t + e) % 7 == 0 for e in range(self.num_envs)])
        infos: List[dict] = []
        for e in range(self.num_envs):
            info: dict = {}
            if dones[e]:
                won = float((self.t + e) % 2)
                rewards[e] = won
                info["episode"] = {"r": won, "l": 7, "t": 0.0}
                info["win_outcome"] = won
                if e == 1 and (self.t // 7) % 2 == 1:      # a truncated end: the bootstrap branch
                    info["TimeLimit.truncated"] = True
                    info["terminal_observation"] = {k: v[e] for k, v in prev.items()}
            infos.append(info)
        return self._obs(), rewards, dones, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: Any = None) -> List[Any]:
        if attr_name == "render_mode":
            return [None] * self.num_envs
        raise AttributeError(attr_name)

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        raise AttributeError(attr_name)

    def has_attr(self, attr_name: str) -> bool:
        return attr_name == "action_masks"

    def env_method(self, method_name: str, *args: Any, indices: Any = None, **kwargs: Any) -> List[Any]:
        if method_name == "action_masks":
            r = self._row()
            return [self.data["action_masks"][r, e].astype(bool) for e in range(self.num_envs)]
        raise AttributeError(method_name)

    def env_is_wrapped(self, wrapper_class: Any, indices: Any = None) -> List[bool]:
        return [False] * self.num_envs


class _DumpRecorder(KVWriter):
    def __init__(self) -> None:
        self.dumps: List[Tuple[int, Dict[str, Any]]] = []

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Any], step: int = 0) -> None:
        self.dumps.append((int(step), {k: _plain(v) for k, v in key_values.items() if not _is_wall(k)}))

    def close(self) -> None:
        pass


def _plain(v: Any) -> Any:
    if isinstance(v, th.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return ("nd", v.dtype.str, v.shape, v.tobytes())
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return ("nan",) if f != f else f
    return v


class _Recorder(BaseCallback):
    """Every hook, with what the loop has handed the callbacks at that moment."""

    def __init__(self, trace: List[Any], buffers: List[Any]) -> None:
        super().__init__()
        self.trace, self.buffers = trace, buffers

    def _rec(self, hook: str, **extra: Any) -> None:
        self.trace.append((hook, int(self.model.num_timesteps), int(self.n_calls),
                           tuple(sorted(self.locals)), extra))

    def _on_training_start(self) -> None:
        self._rec("training_start")

    def _on_rollout_start(self) -> None:
        self._rec("rollout_start")

    def _on_step(self) -> bool:
        buf = self.locals.get("rollout_buffer")
        dones = self.locals.get("dones")
        self._rec("step", pos=None if buf is None else int(buf.pos),
                  dones=None if dones is None else tuple(bool(d) for d in dones))
        return True

    def _on_rollout_end(self) -> None:
        self._rec("rollout_end", progress=float(self.model._current_progress_remaining),
                  lr=float(self.model.policy.optimizer.param_groups[0]["lr"]))
        rb = self.model.rollout_buffer
        snap = {"obs:" + k: v.tobytes() for k, v in rb.observations.items()}
        for f in ("actions", "rewards", "episode_starts", "values", "log_probs", "advantages", "returns",
                  "action_masks"):
            snap[f] = getattr(rb, f).tobytes()
        self.buffers.append(snap)

    def _on_training_end(self) -> None:
        self._rec("training_end")


class _EvalDumpStandIn(BaseCallback):
    """The eval callbacks' `logger.dump(step)` from inside `_on_step`, mid-rollout, at an EARLIER step."""

    def __init__(self, at_call: int) -> None:
        super().__init__()
        self.at_call = at_call

    def _on_step(self) -> bool:
        if self.n_calls == self.at_call:
            self.logger.record("eval/stand_in", 1.0)
            self.logger.dump(max(0, int(self.num_timesteps) - 5))
        return True


def _params_sha(model: Any) -> str:
    return LG.params_sha256(model)


def _wrap_counting(model: Any, counts: Dict[str, int]) -> None:
    """Instance-attribute wrappers, the way K6 and the compile sentinel attach."""
    orig_collect, orig_train = model.collect_rollouts, model.train

    def collect_rollouts(*a: Any, **k: Any) -> Any:
        counts["collect"] += 1
        return orig_collect(*a, **k)

    def train(*a: Any, **k: Any) -> Any:
        counts["train"] += 1
        out = orig_train(*a, **k)
        counts.setdefault("post_update_params", []).append(_params_sha(model))  # type: ignore[union-attr]
        return out

    model.collect_rollouts = collect_rollouts
    model.train = train


def _callbacks(trace: List[Any], buffers: List[Any], lr: float) -> List[BaseCallback]:
    from agents.training.adaptive_lr_callback import AdaptivePPOCallback
    from agents.training.signal_callback import SignalMetricsCallback
    from agents.training.win_prob_callback import WinProbLabelCallback

    return [WinProbLabelCallback(emit=lambda _m: None),
            AdaptivePPOCallback(initial_lr=lr, target_kl=1e-6, cooldown_rollouts=0, verbose=0),
            SignalMetricsCallback(),
            _EvalDumpStandIn(at_call=LG.N_STEPS + 5),          # mid-rollout, iteration 2
            _Recorder(trace, buffers)]


def _run(mode: str, tmp_path: Any) -> Dict[str, Any]:
    _, obs_space, act_space = LG._spaces()
    with np.load(LG.BUFFER_PATH) as z:
        data = {k: z[k] for k in z.files}
    os.environ[L.LOOP_ENV] = mode
    try:
        with LG._one_thread():
            env = ScriptedVecEnv(obs_space, act_space, data)
            model = LG.build_learner(env)
            rec = _DumpRecorder()
            model.set_logger(Logger(folder=None, output_formats=[rec]))
            counts: Dict[str, Any] = {"collect": 0, "train": 0}
            _wrap_counting(model, counts)
            trace: List[Any] = []
            buffers: List[Any] = []
            lr = float(model.policy.optimizer.param_groups[0]["lr"])
            np.random.seed(LG.UPDATE_SEED)
            th.manual_seed(LG.UPDATE_SEED)
            model.learn(total_timesteps=ITERATIONS * LG.N_STEPS * LG.N_ENVS,
                        callback=_callbacks(trace, buffers, lr), reset_num_timesteps=True)
            out: Dict[str, Any] = {
                "trace": trace, "buffers": buffers, "dumps": copy.deepcopy(rec.dumps),
                "counts": {k: v for k, v in counts.items()},
                "n_updates": int(model._n_updates), "num_timesteps": int(model.num_timesteps),
                "progress": float(model._current_progress_remaining),
                "ep_info": [dict(e) for e in model.ep_info_buffer],
                "params": _params_sha(model), "mode": model._ppo_loop_mode,
            }
            # RESUME leg: save -> load -> learn(reset_num_timesteps=False)
            path = tmp_path / f"m_{mode}.zip"
            model.save(path)
            from agents.training.instrumented_ppo import InstrumentedMaskablePPO
            m2 = InstrumentedMaskablePPO.load(path, env=env, device="cpu")
            m2.behaviour_check = "off"
            rec2 = _DumpRecorder()
            m2.set_logger(Logger(folder=None, output_formats=[rec2]))
            trace2: List[Any] = []
            buffers2: List[Any] = []
            np.random.seed(LG.UPDATE_SEED + 1)
            th.manual_seed(LG.UPDATE_SEED + 1)
            m2.learn(total_timesteps=RESUME_ITERATIONS * LG.N_STEPS * LG.N_ENVS,
                     callback=_callbacks(trace2, buffers2, lr), reset_num_timesteps=False)
            out.update({"resume_trace": trace2, "resume_buffers": buffers2, "resume_dumps": rec2.dumps,
                        "resume_params": _params_sha(m2), "resume_num_timesteps": int(m2.num_timesteps),
                        "resume_n_updates": int(m2._n_updates)})
            return out
    finally:
        os.environ.pop(L.LOOP_ENV, None)


@pytest.fixture(scope="module")
def both(tmp_path_factory: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tmp = tmp_path_factory.mktemp("own_loop")
    return _run(L.LOOP_REFERENCE, tmp), _run(L.LOOP_OWNED, tmp)


def test_the_arms_ran_the_loop_they_name(both: Any) -> None:
    ref, own = both
    assert ref["mode"] == L.LOOP_REFERENCE and own["mode"] == L.LOOP_OWNED
    assert ref["n_updates"] == own["n_updates"] == ITERATIONS * int(LG.golden_recipe()["n_epochs"])


def test_hook_trace_is_identical(both: Any) -> None:
    ref, own = both
    assert len(ref["trace"]) == len(own["trace"]) > ITERATIONS * LG.N_STEPS
    for i, (a, b) in enumerate(zip(ref["trace"], own["trace"])):
        assert a == b, f"hook #{i} differs:\n  sb3:   {a}\n  owned: {b}"
    # the precondition the win-prob labeller rests on: on_step BEFORE add (pos = the row to come)
    steps = [t for t in own["trace"] if t[0] == "step"]
    assert [t[4]["pos"] for t in steps[:LG.N_STEPS]] == list(range(LG.N_STEPS))


def test_buffers_are_byte_identical(both: Any) -> None:
    ref, own = both
    assert len(ref["buffers"]) == len(own["buffers"]) == ITERATIONS
    for i, (a, b) in enumerate(zip(ref["buffers"], own["buffers"])):
        bad = sorted(k for k in a if a[k] != b.get(k))
        assert not bad and a.keys() == b.keys(), f"rollout {i}: fields differ {bad}"


def test_logger_dumps_are_identical(both: Any) -> None:
    ref, own = both
    assert [s for s, _ in ref["dumps"]] == [s for s, _ in own["dumps"]]
    for (s, a), (_, b) in zip(ref["dumps"], own["dumps"]):
        assert a == b, f"dump at step {s}: keys only-sb3 {sorted(set(a) - set(b))}, only-owned " \
                       f"{sorted(set(b) - set(a))}, differing {sorted(k for k in a if k in b and a[k] != b[k])}"


def test_the_eval_dump_defect_is_preserved(both: Any) -> None:
    """Design §2.1: the mid-rollout dump writes update 1's train/* at the EARLIER eval step and clears
    it, so the KL controller skips — PRESERVED by stage 1 (identity). Stage 2 flips this test."""
    _, own = both
    stand_in = [(s, d) for s, d in own["dumps"] if "eval/stand_in" in d]
    assert len(stand_in) == 1
    step, d = stand_in[0]
    assert "train/approx_kl" in d, "the stand-in dump no longer carries the previous update's train/*"
    after = [dd for s, dd in own["dumps"] if s > step]
    assert after and "train/approx_kl" not in after[0], (
        "the next dump carries train/approx_kl again — the defect was fixed; that is stage 2's "
        "labelled change, not stage 1's")


def test_params_and_counters_are_identical(both: Any) -> None:
    ref, own = both
    for k in ("params", "n_updates", "num_timesteps", "progress", "ep_info"):
        assert ref[k] == own[k], k
    assert ref["counts"]["post_update_params"] == own["counts"]["post_update_params"]


def test_instance_wrappers_still_intercept(both: Any) -> None:
    for arm in both:
        assert arm["counts"]["collect"] == ITERATIONS
        assert arm["counts"]["train"] == ITERATIONS


def test_resume_leg_is_identical(both: Any) -> None:
    ref, own = both
    assert ref["resume_trace"] == own["resume_trace"]
    assert ref["resume_buffers"] == own["resume_buffers"]
    assert ref["resume_dumps"] == own["resume_dumps"]
    for k in ("resume_params", "resume_num_timesteps", "resume_n_updates"):
        assert ref[k] == own[k], k
    # the resume target is RELATIVE: it ran exactly RESUME_ITERATIONS more rollouts
    assert own["resume_num_timesteps"] == own["num_timesteps"] + RESUME_ITERATIONS * LG.N_STEPS * LG.N_ENVS


def test_an_unknown_loop_value_refuses(monkeypatch: Any) -> None:
    monkeypatch.setenv(L.LOOP_ENV, "sb3")
    with pytest.raises(ValueError, match="sb3_reference"):
        L.loop_mode()
    monkeypatch.delenv(L.LOOP_ENV)
    assert L.loop_mode() == L.LOOP_OWNED


def test_loop_phases_are_the_declared_order() -> None:
    """`learn()`'s body is `LOOP_PHASES` written out: the phase comments appear in that order."""
    import inspect
    src = inspect.getsource(L.OwnedLoop.learn)
    pos = [src.index(f"# {p}") for p in L.LOOP_PHASES]
    assert pos == sorted(pos), L.LOOP_PHASES
    assert L.LOOP_ITERATION == ("collect", "progress", "dump", "update")
    # the dump precedes the update (the TB step contract), and both go through the attribute
    assert src.index("self.dump_logs(") < src.index("self.train()")
    assert "self.collect_rollouts(" in src and "self.train()" in src


def test_upstream_pins_hold() -> None:
    L.verify_upstream_loop_unchanged()
