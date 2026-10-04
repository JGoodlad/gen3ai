"""Shared scaffolding for the collector's tests and gates (M5 Lane G) — never imported by training.

* ``build_selfcheck()`` — THIS checkout's self-check cdylib + process binary (the build every pytest
  session runs; ``cargo`` into ``src/rust_env/target``, never main's);
* ``production_spaces()`` — the trainee's observation / action space at the PRODUCTION
  surface (``production_args()``), i.e. every production label key;
* ``fresh_model(...)`` — an ``InstrumentedMaskablePPO`` at the production policy kwargs over a
  ``RustVecEnv``, its weights a SEEDED PERTURBATION of the fresh init (a fresh pointer head is exactly
  uniform, which would make every sampled-action comparison vacuous);
* ``collector_for(model, env, ...)`` — the collector on seeded pool teams, p1 = the trainee through T2
  (eager, CPU), p2 on an EXTERNAL route answered by a seeded random policy the harness records;
* ``ToyVecEnv`` + ``VecEnvCollector`` / ``attach_vec_collector`` — a toy learner's env and rollout (the
  ``train()``-fold unit tests).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.training.trainer_env import TrainerVecEnv

FEATURES = ("--profile", "selfcheck", "--features", "emission-selfcheck")


def cargo() -> str:
    c = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(c):
        raise RuntimeError("cargo is not installed — the Rust collector's tests cannot run (install rustup)")
    return c


def build_selfcheck() -> None:
    from utils.paths import src_path
    from utils.rust_env import proc

    crate = src_path("rust_env")
    env = dict(os.environ, CARGO_TARGET_DIR=str(crate / "target"))
    r = subprocess.run([cargo(), "build", "--lib", "--bin", proc.BIN_NAME, *FEATURES, "--manifest-path",
                        str(crate / "Cargo.toml")], env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"building the rust env (self-check) failed:\n{r.stderr[-4000:]}")


def production_spaces() -> Tuple[Any, Any, Any]:
    """``(args, obs_space, action_space)`` at the production surface."""
    from agents.training.rust_rollout.build import trainee_spaces
    from main.train.production_args import production_args

    args = production_args()
    obs, act = trainee_spaces(args)
    return args, obs, act


def unset_to_class_defaults(model: Any) -> None:
    """``production_args()`` is the parser's namespace + the production config, NOT ``resolve_config``'s
    (which refuses a namespace whose every key reads as typed): the flags it leaves ``None`` are the ones
    ``_resolve`` would default. A learner built from it takes the class default for exactly those."""
    from main.train.model_build import _TRAINING_HPARAMS

    for name, _how in _TRAINING_HPARAMS:
        if getattr(model, name, None) is None and getattr(type(model), name, None) is not None:
            setattr(model, name, getattr(type(model), name))


def fresh_model(env: Any, *, n_steps: int, batch_size: int, n_epochs: int = 1, seed: int = 0,
                perturb_seed: int = 1234, gamma: float = 1.0, gae_lambda: float = 0.8,
                policy_args: Any = None, perturb_keyed: bool = False, **kw: Any) -> Any:
    import torch

    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from main.fresh_checkpoint import _production_policy_kwargs
    from utils.torch_state_guard import single_thread_build

    _args, _layout, pk = _production_policy_kwargs(policy_args)
    torch.manual_seed(seed)
    with single_thread_build():      # the init follows the thread count (F-X5-4): same seed => same bytes
        model = InstrumentedMaskablePPO(Gen3DualHeadMaskablePolicy, env, n_steps=n_steps, batch_size=batch_size,
                                        n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, device="cpu",
                                        seed=seed, policy_kwargs=pk, verbose=0, **kw)
    # ``perturb_keyed``: name-keyed noise (`parity_probe._keyed_noise`) — the K9 golden's non-blob arms,
    # whose parameter list differs from blob's, so a shared parameter keeps its noise across arms.
    perturb_(model.policy, seed=perturb_seed, scale=PERTURB_SCALE, keyed=perturb_keyed)
    model.ep_info_buffer = deque(maxlen=100)
    model.ep_success_buffer = deque(maxlen=100)
    return model


class ToyVecEnv(TrainerVecEnv):
    """A TEST env for a toy learner: ``n`` gym envs stepped in lockstep (what sb3's ``DummyVecEnv`` did for
    these tests until deletion pass U4): a finished env is reset at once, its last observation in
    ``info["terminal_observation"]`` and ``info["TimeLimit.truncated"]`` set; ``reset`` passes each env
    the seed ``seed(s)`` recorded (``s + i``), once. ``action_masks()`` stacks every env's masks. Never
    imported by training."""

    def __init__(self, env_fns: Sequence[Callable[[], Any]]):
        self.envs = [fn() for fn in env_fns]
        e0 = self.envs[0]
        super().__init__(len(self.envs), e0.observation_space, e0.action_space)
        self._keys = list(self.observation_space.spaces)

    def _stack(self, obs: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {k: np.stack([np.asarray(o[k], dtype=self.observation_space[k].dtype) for o in obs])
                for k in self._keys}

    def reset(self) -> Dict[str, np.ndarray]:
        obs = [env.reset(seed=s)[0] for env, s in zip(self.envs, self._seeds)]
        self._seeds = [None] * self.num_envs
        return self._stack(obs)

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[dict]]:
        obs, rews, dones, infos = [], np.zeros(self.num_envs, np.float32), np.zeros(self.num_envs, bool), []
        for i, env in enumerate(self.envs):
            o, r, terminated, truncated, info = env.step(actions[i])
            info = dict(info)
            rews[i], dones[i] = r, terminated or truncated
            info["TimeLimit.truncated"] = truncated and not terminated
            if dones[i]:
                info["terminal_observation"] = o
                o, _ = env.reset()
            obs.append(o)
            infos.append(info)
        return self._stack(obs), rews, dones, infos

    def action_masks(self) -> np.ndarray:
        return np.stack([np.asarray(env.action_masks()) for env in self.envs])

    def env_method(self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any) -> List[Any]:
        return [getattr(self.envs[i], method_name)(*method_args, **method_kwargs) for i in self._indices(indices)]


class ScriptedVecEnv(TrainerVecEnv):
    """A deterministic TEST env over the production spaces that serves the K9 golden buffer's real rows
    (``data`` = the ``np.load`` of ``learner_golden.BUFFER_PATH``), row ``t % n_rows`` at step ``t``; games
    end on a fixed schedule (env ``e`` at every ``t`` with ``(t + e) % 7 == 0``), each with
    ``info["episode"]`` and a win outcome, env 1's every other end a truncation. For loop-level tests
    (the hook nesting, the dump order) that need a production-surface learner and a real rollout."""

    def __init__(self, obs_space: Any, act_space: Any, data: Dict[str, np.ndarray], n_envs: int):
        super().__init__(n_envs, obs_space, act_space)
        self.data = data
        self.keys = sorted(obs_space.spaces)
        self.n_rows = int(data["actions"].shape[0])
        self.t = 0

    def _obs(self) -> Dict[str, np.ndarray]:
        r = self.t % self.n_rows
        return {k: self.data["obs:" + k][r].copy() for k in self.keys}

    def reset(self) -> Dict[str, np.ndarray]:
        self.t = 0
        return self._obs()

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[dict]]:
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
                if e == 1 and (self.t // 7) % 2 == 1:
                    info["TimeLimit.truncated"] = True
                    info["terminal_observation"] = {k: v[e] for k, v in prev.items()}
            infos.append(info)
        return self._obs(), rewards, dones, infos

    def action_masks(self) -> np.ndarray:
        r = self.t % self.n_rows
        return np.stack([self.data["action_masks"][r, e].astype(bool) for e in range(self.num_envs)])

    def env_method(self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any) -> List[Any]:
        raise AttributeError(f"ScriptedVecEnv serves no env_method ({method_name!r})")


class VecEnvCollector:
    """A TEST double for the Rust collector: fills the learner's rollout buffer from the model's own
    `ToyVecEnv`, written from the requirements rather than copied from sb3 (the vendored copy of
    upstream's collect that used to serve this was deleted with the Python env core, deletion pass U3).

    The production rollout is the Rust collector's: ``RolloutProbes.collect_rollouts`` refuses a learner
    without a ``_rust_collector``. The ``train()``-fold unit tests (accumulation, the diagnostics, the
    ride-along heads, the loop's hook trace …) still need SOME rollout from a toy env, so they attach this
    through :func:`attach_vec_collector`, the seam ``_collect_rust`` calls (``collect`` / ``after_update`` /
    ``seen_updates``).

    What it does, and nothing more: ``n_steps`` vector steps, the policy acting under the env's action
    masks; the ``step`` event fires BEFORE the row is added, with the declared step locals
    (``loop_callbacks.STEP_LOCALS``); then the buffer's own window GAE. It models no truncation bootstrap
    and no complete-game window — those are the Rust collector's business and are pinned by
    ``rust_rollout/store_test.py``. Never imported by training."""

    seen_updates = None

    def after_update(self, model: Any) -> None:       # no inference service to refresh
        return None

    def collect(self, model: Any, callback: Any, rollout_buffer: Any) -> bool:
        import torch as th

        env = model.env
        assert model._last_obs is not None, "the learner was not set up (learn() runs _setup_learn first)"

        def as_tensor(obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
            return {k: th.as_tensor(v, device=model.device) for k, v in obs.items()}

        model.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        for _ in range(model.n_steps):
            with th.no_grad():
                masks = env.action_masks()
                actions, values, log_probs = model.policy(as_tensor(model._last_obs), action_masks=masks)
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            model.num_timesteps += env.num_envs
            callback.update_locals({"infos": infos, "dones": dones})
            if not callback.on_step():
                return False
            model._update_info_buffer(infos, dones)
            rollout_buffer.add(model._last_obs, actions.reshape(-1, 1), rewards, model._last_episode_starts,
                               values, log_probs, action_masks=masks)
            model._last_obs, model._last_episode_starts = new_obs, dones
        with th.no_grad():
            last_values = model.policy.predict_values(as_tensor(model._last_obs))
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=model._last_episode_starts)
        callback.on_rollout_end()
        return True


def attach_vec_collector(model: Any) -> Any:
    """Make ``model`` collect from its own VecEnv through :class:`VecEnvCollector` (a test double for the
    Rust collector); returns the model. A loaded model lost its collector (it is never saved): re-attach."""
    model._rust_collector = VecEnvCollector()
    return model


def record_dumps(model: Any) -> List[Dict[str, Any]]:
    """Give ``model`` a logger (`train_logger.Logger`) whose every dump is appended, as a ``dict`` with its
    step under ``"_step"``, to the returned list. ``learn()``'s FINAL dump (the last update's scalars, P3)
    clears the pending values, so a test reads what an update logged from here, not from
    ``logger.name_to_value`` after ``learn()``."""
    from agents.training.train_logger import Logger

    out: List[Dict[str, Any]] = []

    class _Recorder:
        def write(self, kv: Dict[str, Any], _excluded: Dict[str, Any], step: int = 0) -> None:
            out.append({**kv, "_step": int(step)})

        def write_sequence(self, _seq: Any) -> None:
            return None

        def close(self) -> None:
            return None

    model.set_logger(Logger(None, [_Recorder()]))
    return out


class RandomP2:
    """p2 on an EXTERNAL route: a seeded uniform-random legal action, recorded per (env, episode) in
    play order — the scripted opponent a replay plays back."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.log: Dict[Tuple[int, int], List[int]] = {}

    def __call__(self, cols: Any, envs: np.ndarray) -> np.ndarray:
        out = []
        for e in envs.tolist():
            a = int(self.rng.choice(np.flatnonzero(cols["mask"][e, 1])))
            self.log.setdefault((e, int(cols["episode"][e])), []).append(a)
            out.append(a)
        return np.asarray(out, dtype=np.int32)


def pool_builder(n_teams: int = 24, offset: int = 0) -> Any:
    from utils.team_sources import team_list
    from utils.teambuilder import Gen3Teambuilder

    raw = team_list("pool")
    return Gen3Teambuilder([raw[(offset + 37 * i) % len(raw)] for i in range(n_teams)])


def collector_for(model: Any, obs_space: Any, *, decl: Any, p2: Optional[Callable[..., np.ndarray]] = None,
                  builder: Any = None, opp_builder: Any = None) -> Any:
    from agents.training import rust_env_opponents as E
    from agents.training.rust_rollout.build import OpponentSources, build_collector

    tb = builder if builder is not None else pool_builder()
    ob = opp_builder if opp_builder is not None else tb
    plan = E.OpponentPlan(bots=("random",))
    return build_collector(decl, obs_space=obs_space, trainee_policy=model.policy, plan=plan,
                           sources=OpponentSources(), trainee_builder=tb, opponent_builder=ob,
                           external_p2=p2 if p2 is not None else RandomP2(7), emit=lambda _m: None)


class NullCallback:
    """The minimum of the callback protocol ``collect`` drives (``loop_callbacks``; counts the step events)."""

    def __init__(self) -> None:
        self.steps = 0
        self.infos: List[dict] = []
        self.rollouts = 0

    def on_rollout_start(self) -> None:
        self.rollouts += 1

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        self.infos.extend(locals_.get("infos") or [])

    def on_step(self) -> bool:
        self.steps += 1
        return True

    def on_rollout_end(self) -> None:
        pass


def perturb_weights(model: Any, seed: int) -> None:
    """A stand-in for one update: move every parameter a little (seeded)."""
    import torch

    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for p in model.policy.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 1e-2)
    model._n_updates += 1


def spaces_rows(buf: Any) -> int:
    return int(buf.buffer_size) * int(buf.n_envs)


def ages_of(model: Any) -> Sequence[int]:
    v = np.asarray(model._rust_row_versions).reshape(-1)
    return (int(model._rust_version) - v).tolist()
