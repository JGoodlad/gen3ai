"""Shared scaffolding for the collector's tests and gates (M5 Lane G) — never imported by training.

* ``build_selfcheck()`` — THIS checkout's self-check cdylib + process binary (the build every pytest
  session runs; ``cargo`` into ``src/rust_env/target``, never main's);
* ``production_spaces()`` — the trainee ``Gen3Env``'s observation / action space at the PRODUCTION
  surface (``production_args()``), i.e. every production label key;
* ``fresh_model(...)`` — an ``InstrumentedMaskablePPO`` at the production policy kwargs over a
  ``RustVecEnv``, its weights a SEEDED PERTURBATION of the fresh init (a fresh pointer head is exactly
  uniform, which would make every sampled-action comparison vacuous);
* ``collector_for(model, env, ...)`` — the collector on seeded pool teams, p1 = the trainee through T2
  (eager, CPU), p2 on an EXTERNAL route answered by a seeded random policy the harness records.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

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
    from main.rust_core_cutover.envs import production_args

    args = production_args()
    obs, act = trainee_spaces(args)
    return args, obs, act


def fresh_model(env: Any, *, n_steps: int, batch_size: int, n_epochs: int = 1, seed: int = 0,
                perturb_seed: int = 1234, gamma: float = 1.0, gae_lambda: float = 0.8,
                policy_args: Any = None, **kw: Any) -> Any:
    import torch

    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from main.fresh_checkpoint import _production_policy_kwargs

    _args, _layout, pk = _production_policy_kwargs(policy_args)
    torch.manual_seed(seed)
    model = InstrumentedMaskablePPO(Gen3DualHeadMaskablePolicy, env, n_steps=n_steps, batch_size=batch_size,
                                    n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, device="cpu", seed=seed,
                                    policy_kwargs=pk, verbose=0, **kw)
    perturb_(model.policy, seed=perturb_seed, scale=PERTURB_SCALE)
    model.ep_info_buffer = deque(maxlen=100)
    model.ep_success_buffer = deque(maxlen=100)
    return model


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
    """The minimum of SB3's callback contract ``collect`` drives (counts the vec-step calls)."""

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
