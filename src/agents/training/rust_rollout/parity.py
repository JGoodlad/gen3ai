"""THE ROLLOUT-LEVEL SLICE N and the LEARNER-LEVEL CHECK (M5 Lane G's gate; package docstring).

**What is compared.** The BUFFER THE LEARNER SEES. The Rust collector, in WINDOW mode (today's
schedule, so the new collector is proven before it changes the schedule), fills the model's
``[n_steps, n_envs]`` buffer; TODAY'S PYTHON PATH — ``InstrumentedMaskablePPO.collect_rollouts``
(sb3's loop) over a ``DummyVecEnv`` of production-surface ``Gen3Env``s on the rust bridge, each in
``Monitor(MaskableAgentWrapper(...))`` exactly as ``env_factory`` wraps a worker, with
``WinProbLabelCallback`` registered — plays THE SAME GAMES and fills its own buffer. Then, window by
window, every field: every observation key (the row, the mask, every label, ``opp_class``,
``win_target`` / ``win_mask``), ``actions``, ``action_masks``, ``rewards``, ``episode_starts``,
``values``, ``log_probs``, ``advantages``, ``returns``.

**The same games.** RECORD in Rust, REPLAY in Python (Lanes C / D / E's direction): per env, the
core's episode sequence — its two packed teams and battle seed (the stager's, into the startup table),
and p2's actions (an EXTERNAL route answered by a seeded random policy, played back through the real
mapper). The trainee is NOT scripted: the replay's policy forward computes its own log-probs and draws
its action with the KEYED DRAW from its own key (env, episode ordinal, decision index) — the one
harness substitution into sb3's loop (``_KeyedForward``: ``Gen3DualHeadMaskablePolicy.forward`` line
for line, the sample replaced), so "the replay played the recorded action" is itself a check.

**The bars (declared).** EXACT (bytes) on every observation key, the actions, masks, rewards, episode
starts and the win labels. ``values`` / ``log_probs`` within ``FLOAT_BAR`` = 1e-5 absolute, and
``advantages`` / ``returns`` within ``GAE_BAR`` = 1e-4: the two paths forward the SAME weights on the
SAME rows in DIFFERENT batch compositions (T2 pads the trainee's rows to its bucket; sb3 forwards all
N envs each step), and CPU matmul rounding depends on the batch shape — measured ~6e-7 on log-probs
(``consistency_test``) — which GAE accumulates over up to ``n_steps`` rows. A disagreeing ACTION is
never tolerated; if one occurs at a keyed-draw margin below ``NEAR_BAR`` it is reported as a NEAR-TIE
(the rounding moved u across a CDF boundary) and the gate still FAILS unless the caller allows it.

**The learner-level check.** Two identical production learners (same fresh seed, same perturbation,
the production training hparams), ONE UPDATE each — one epoch, the whole buffer in ``ACCUM`` = 4
micro-batches accumulated into ONE optimizer step (production's accumulation path) — one on the Rust
buffer, one on the Python buffer of the same window (the one with the most win-labelled rows, so the
win-prob BCE is live),
numpy / torch RNG re-seeded identically before each — then the parameters (max |Δ| ≤
``LEARNER_PARAM_BAR`` = 1e-5, while the update itself moves them ~1e-3) and every logged scalar the
update records — the PPO losses, every belief / intent / win-prob term, the diagnostics; wall-clock
tags excluded — (|Δ| ≤ ``LEARNER_LOSS_ABS`` + ``LEARNER_LOSS_BAR`` × |value| = 1e-6 + 1e-4 × |value|). Declared, not zero: the buffers already
differ at float rounding (the bars above), which one update carries into the weights. WHY ONE
OPTIMIZER STEP (measured 2026-09-30, 8 envs x 128 x 3 windows): with 8 optimizer steps in the update
the later minibatches' PPO ratios, clip gates and Adam's normalisation turned a <= 5e-7 log-prob
difference into a 3.4e-4 parameter difference (0.15 of the update's own movement) — a property of
iterated PPO steps on inputs that differ at rounding, not of either collector (the same learner fed
the SAME buffer twice is bit-identical). One step from identical weights isolates what the gate is
about: that the Rust buffer drives the update the Python buffer drives.

    python -m agents.training.rust_rollout.parity --n-envs 4 --n-steps 32 --windows 2 --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FLOAT_BAR = 1e-5
GAE_BAR = 1e-4
NEAR_BAR = 1e-5
LEARNER_PARAM_BAR = 1e-5      # max |Δ param| after one update (absolute)
ACCUM = 4                     # the learner check's micro-batches per optimizer step (the production path)
LEARNER_LOSS_BAR = 1e-4       # relative allowance of any logged scalar after the update ...
LEARNER_LOSS_ABS = 1e-6       # ... plus this absolute one (scalars whose value is ~0)
EXACT_FIELDS = ("actions", "action_masks", "rewards", "episode_starts")
FLOAT_FIELDS = (("values", FLOAT_BAR), ("log_probs", FLOAT_BAR), ("advantages", GAE_BAR), ("returns", GAE_BAR))


@dataclass
class EnvScript:
    """One env's recorded episode sequence (play order)."""

    teams: List[Tuple[str, str]] = field(default_factory=list)
    seeds: List[List[int]] = field(default_factory=list)
    p2: List[List[int]] = field(default_factory=list)


class _EpisodeRecorder:
    """Collector hook: each env's episodes as the core started them (teams, seed)."""

    def __init__(self, n: int):
        self.scripts = [EnvScript() for _ in range(n)]

    def episode_started(self, col: Any, moved: np.ndarray) -> None:
        from agents.training.rust_rollout.teams import battle_seed

        table = col.stager.table.teams
        for e in moved.tolist():
            k = int(col.cols["episode"][e])
            s = self.scripts[e]
            if k != len(s.teams):
                raise AssertionError(f"env {e}: episode {k} started after {len(s.teams)} recorded")
            p1, p2 = (int(x) for x in col.stager.cur_teams[e])
            s.teams.append((table[p1], table[p2]))
            s.seeds.append(battle_seed(col.cfg.run_seed, e, k))

    def p2(self, col: Any) -> None:
        pass

    def trainee(self, *a: Any) -> None:
        pass


def _snapshot(buf: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {f"obs:{k}": np.array(v, copy=True) for k, v in buf.observations.items()}
    for f in EXACT_FIELDS + tuple(x for x, _ in FLOAT_FIELDS):
        out[f] = np.array(getattr(buf, f), copy=True)
    return out


# ------------------------------------------------------------------------------------------ record

def record(*, n_envs: int, n_steps: int, windows: int, run_seed: int = 17, p2_seed: int = 5,
           model_seed: int = 0, perturb_seed: int = 1234, teams: str = "pool", n_teams: int = 24,
           team_offset: int = 0, front: str = "ffi") -> Dict[str, Any]:
    """Play ``windows`` WINDOW-mode rollouts on the Rust collector (a fixed policy — no update between
    them). Returns the buffers, the episode scripts and the setup a replay needs."""
    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.build import RustEnvDecl
    from agents.training.rust_vec_env import RustVecEnv

    _args, obs, act = TK.production_spaces()
    builder = _builder(teams, n_teams, team_offset)
    decl = RustEnvDecl(n_envs=n_envs, threads=2, front=front, profile="selfcheck", trigger="window",
                       n_steps=n_steps, micro_batch=n_steps, device="cpu", backend="eager", run_seed=run_seed,
                       gamma=1.0, gae_lambda=0.8)
    p2 = TK.RandomP2(p2_seed)
    env = RustVecEnv(n_envs=n_envs, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl, p2=p2, builder=builder))
    model = TK.fresh_model(env, n_steps=n_steps, batch_size=n_steps, seed=model_seed, perturb_seed=perturb_seed)
    col = env.startup(model)
    rec = _EpisodeRecorder(n_envs)
    col.hooks.append(rec)
    # the first episode of every env started inside `startup` (before the hook existed)
    rec.episode_started(col, np.arange(n_envs))
    cb = TK.NullCallback()
    bufs = []
    t0 = time.perf_counter()
    try:
        for _w in range(windows):
            if not col.collect(model, cb, model.rollout_buffer):
                raise AssertionError("the collector stopped")
            bufs.append(_snapshot(model.rollout_buffer))
        life = col.check_lifecycle()
        near = int(col.stats.near_boundary)
    finally:
        env.close()
    for e, s in enumerate(rec.scripts):
        s.p2 = [list(p2.log.get((e, k), [])) for k in range(len(s.teams))]
    return {"buffers": bufs, "scripts": rec.scripts, "run_seed": run_seed, "n_envs": n_envs, "n_steps": n_steps,
            "windows": windows, "model_seed": model_seed, "perturb_seed": perturb_seed, "lifecycle": life,
            "near_boundary_rust": near, "record_s": time.perf_counter() - t0}


def _builder(teams: str, n_teams: int, offset: int) -> Any:
    from agents.training.rust_rollout import testkit as TK

    if teams == "pool":
        return TK.pool_builder(n_teams, offset)
    from main.rust_core_cutover.envs import packed_teams
    from utils.teambuilder import Gen3Teambuilder

    class _Packed(Gen3Teambuilder):
        """A builder over already-PACKED ladder teams (the core's table form)."""

        def __init__(self, packed: List[str]):
            self.packed_teams = list(packed)
            self._pool_keys = [str(i) for i in range(len(packed))]
            self._pool_index_by_packed = {p: i for i, p in enumerate(packed)}
            self.bias_packed_teams = []
            self.bias_prob = 0.0
            self._team_pfsp = "off"
            self._tp_weights = None
            self._last_pool_idx = None
            self._twr_wins, self._twr_games = {}, {}
            self._block_episodes, self._block_cached, self._block_cached_idx, self._block_left = 1, None, None, 0

    pool = packed_teams(teams)
    return _Packed([pool[(offset + 7919 * i) % len(pool)] for i in range(n_teams)])


# ------------------------------------------------------------------------------------------ replay

def _script_player_cls() -> Any:
    from poke_env.player.battle_order import DefaultBattleOrder
    from poke_env.player.player import Player

    from agents.action.mapper import Gen3ActionMapper
    from agents.action.mask_generator import Gen3ActionMasker
    from agents.battle.live_view import LegalActions

    class ScriptP2(Player):
        """p2's recorded indices through the real mapper, consumed only when agent2's order is sent."""

        def __init__(self, **kw: Any):
            super().__init__(**kw)
            self.actions: List[int] = []
            self.j = 0
            self.env: Any = None

        def choose_move(self, battle: Any) -> Any:
            if self.env is None or not self.env.agent2_to_move:
                return DefaultBattleOrder()
            legal = LegalActions.from_battle(battle)
            mask = Gen3ActionMasker.get_mask(battle, legal=legal)
            if self.j >= len(self.actions):
                raise AssertionError(f"p2 decision {self.j} past the recorded {len(self.actions)}")
            a = self.actions[self.j]
            self.j += 1
            if not mask[a]:
                raise AssertionError(f"p2 decision {self.j - 1}: recorded index {a} illegal in Python")
            return Gen3ActionMapper.action_to_order(a, battle, legal=legal, mask=mask)

    return ScriptP2


def _replay_env_factory(i: int, script: EnvScript, tag: str) -> Any:
    def _init() -> Any:
        from poke_env import AccountConfiguration
        from stable_baselines3.common.monitor import Monitor

        from agents.model.critic_mode import CRITIC_DEFAULT  # noqa: F401 — the wrapper's own default
        from agents.observation.state_encoder import load_mappings
        from agents.training.gen3_env import Gen3Env
        from agents.training.reward_config import RewardConfig
        from agents.training.reward_manager import Gen3RewardManager
        from main.rust_core_cutover.envs import SequenceTeambuilder, production_args
        from main.train.env_factory import trainee_env_kwargs
        from utils.bridge.bridge_session import attach_bridge_transport

        args = production_args()
        kw = trainee_env_kwargs(args)
        env = Gen3Env(load_mappings(), battle_format="gen3ou",
                      team=SequenceTeambuilder([t[0] for t in script.teams]),
                      opponent_team=SequenceTeambuilder([t[1] for t in script.teams]),
                      reward_fn=Gen3RewardManager(config=RewardConfig.from_args(args)),
                      account_configuration1=AccountConfiguration(f"{tag}e{i}"[:18], None),
                      start_listening=False, **kw)
        session = attach_bridge_transport(env, battle_format="gen3ou", persistent=True, impl="rust", core_obs=True)
        opp = _script_player_cls()(battle_format="gen3ou", account_configuration=AccountConfiguration(f"{tag}o{i}"[:18], None),
                                   start_listening=False)
        opp.env = env
        wrapped = _ScriptedWrapper(env, heuristic_opponents=[opp], rng_seed=i, critic=args.critic,
                                   team_wr_tracking=False)
        wrapped.bind(session, opp, script)
        wrapped.action_space = env.action_space
        wrapped.observation_space = env.observation_space
        return Monitor(wrapped)

    return _init


def _scripted_wrapper_base() -> Any:
    from agents.training.wrappers import MaskableAgentWrapper

    return MaskableAgentWrapper


class _ScriptedWrapper(_scripted_wrapper_base()):
    """``MaskableAgentWrapper`` (the production worker's wrapper: wait absorption, the outcome infos,
    ``resolve_episode_end``) that, at each reset, sets the recorded battle seed and p2's recorded
    actions, and counts the trainee's (episode, decision) — the keyed draw's key."""

    def bind(self, session: Any, opp: Any, script: EnvScript) -> None:
        self._session, self._opp, self._script = session, opp, script
        self.episode = -1
        self.decision = 0

    def reset(self, *, seed: Any = None, options: Any = None) -> Any:
        self.episode += 1
        k = self.episode
        if k >= len(self._script.teams):
            raise AssertionError(f"replay reached episode {k}; the core recorded {len(self._script.teams)}")
        self._session.seed = list(self._script.seeds[k])
        self._opp.actions, self._opp.j = list(self._script.p2[k]), 0
        self.decision = 0
        return super().reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:
        out = super().step(action)
        self.decision += 1
        return out


class _KeyedForward:
    """``Gen3DualHeadMaskablePolicy.forward`` line for line, with the SAMPLE replaced by the keyed draw
    keyed by each env's (episode, decision) (module docs). Installed on the policy INSTANCE."""

    def __init__(self, policy: Any, venv: Any, run_seed: int):
        self.policy, self.venv, self.run_seed = policy, venv, int(run_seed)
        self.near = 0
        self.margins: List[float] = []

    def __call__(self, obs: Any, deterministic: bool = False, action_masks: Any = None) -> Any:
        import torch as th

        from agents.training import keyed_draw as KD

        p = self.policy
        pi_features, vf_features = p.extract_features(obs)
        latent_pi = p.mlp_extractor.forward_actor(pi_features)
        latent_vf = p.mlp_extractor.forward_critic(vf_features)
        values = p._critic_value(latent_vf)
        distribution = p._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        mask = th.as_tensor(action_masks).bool()
        logits = distribution.distribution.logits
        logp = th.where(mask, logits, th.full_like(logits, float("-inf"))).detach().cpu().numpy()
        wrappers = [e.env for e in self.venv.envs]        # Monitor -> _ScriptedWrapper
        env = np.arange(len(wrappers))
        ep = np.array([w.episode for w in wrappers])
        dec = np.array([w.decision for w in wrappers])
        u = KD.keyed_uniforms(self.run_seed, KD.STREAM_TRAINEE, env, ep, dec)
        a, margin = KD.keyed_actions(logp, u)
        self.margins.extend(margin.tolist())
        self.near += int((margin < NEAR_BAR).sum())
        actions = th.as_tensor(a.astype(np.int64))
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *p.action_space.shape))
        return actions, values, log_prob


def replay(rec: Dict[str, Any], *, tag: str = "LG") -> Dict[str, Any]:
    """Today's Python path over the recorded games (module docs). Returns its buffers."""
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.training.rust_rollout import testkit as TK
    from agents.training.win_prob_callback import WinProbLabelCallback

    n, n_steps = int(rec["n_envs"]), int(rec["n_steps"])
    venv = DummyVecEnv([_replay_env_factory(i, s, tag) for i, s in enumerate(rec["scripts"])])
    model = TK.fresh_model(venv, n_steps=n_steps, batch_size=n_steps, seed=int(rec["model_seed"]),
                           perturb_seed=int(rec["perturb_seed"]))
    kf = _KeyedForward(model.policy, venv, int(rec["run_seed"]))
    model.policy.forward = kf
    cb = CallbackList([WinProbLabelCallback()])
    t0 = time.perf_counter()
    bufs = []
    try:
        _total, cb2 = model._setup_learn(10 ** 12, cb, reset_num_timesteps=True)
        cb2.on_training_start(locals(), globals())
        for _w in range(int(rec["windows"])):
            if not model.collect_rollouts(model.env, cb2, model.rollout_buffer, n_steps):
                raise AssertionError("the Python collect_rollouts stopped")
            bufs.append(_snapshot(model.rollout_buffer))
    finally:
        _close(venv)
    return {"buffers": bufs, "near_boundary_python": kf.near, "replay_s": time.perf_counter() - t0}


def _close(venv: Any) -> None:
    """Close every replay env. A window ends MID-BATTLE by design, and poke-env refuses to reset a player
    whose battle is still running — AFTER the bridge session (the child) has already been closed by the
    wrapped ``close`` (``bridge_session._wrap_close``). That one refusal is the expected end state."""
    for e in venv.envs:
        try:
            e.close()
        except OSError as exc:
            if "still running" not in str(exc):
                raise


# ------------------------------------------------------------------------------------------ compare

def compare(rust: List[Dict[str, Any]], py: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Field by field, window by window (module docs). ``div`` names every field outside its bar."""
    div: Dict[str, Any] = {}
    worst: Dict[str, float] = {}
    for w, (a, b) in enumerate(zip(rust, py)):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                div[f"w{w}:{k}"] = "missing on one side"
                continue
            x, y = a[k], b[k]
            if x.shape != y.shape:
                div[f"w{w}:{k}"] = f"shape {x.shape} vs {y.shape}"
                continue
            bar = dict(FLOAT_FIELDS).get(k)
            if bar is None:
                if x.dtype != y.dtype or x.tobytes() != y.tobytes():
                    bad = np.argwhere(np.asarray(x != y).reshape(x.shape[0], x.shape[1], -1).any(-1))
                    div[f"w{w}:{k}"] = {"cells": int(bad.shape[0]), "first": bad[:3].tolist(),
                                        "dtypes": (str(x.dtype), str(y.dtype))}
            else:
                d = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.size else 0.0
                worst[k] = max(worst.get(k, 0.0), d)
                if not d <= bar:
                    div[f"w{w}:{k}"] = {"max_abs": d, "bar": bar}
    return {"divergences": div, "max_abs": worst, "windows": min(len(rust), len(py))}


# ------------------------------------------------------------------------------------------ learner

def learner_check(rec: Dict[str, Any], rust_buf: Dict[str, Any], py_buf: Dict[str, Any], *,
                  n_epochs: int = 1) -> Dict[str, Any]:
    """One production ``train()`` on each buffer from identical learners (module docs)."""
    import torch as th
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: F401 — the learner's env shape

    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_vec_env import RustVecEnv
    from main.rust_core_cutover.envs import production_args
    from main.train.model_build import apply_training_hparams

    args = production_args()
    _a, obs, act = TK.production_spaces()
    n, n_steps = int(rec["n_envs"]), int(rec["n_steps"])
    results = []
    rows = n * n_steps
    micro = rows // ACCUM
    for buf in (rust_buf, py_buf):
        venv = RustVecEnv(n_envs=n, observation_space=obs, action_space=act, build=lambda m: None)
        model = TK.fresh_model(venv, n_steps=n_steps, batch_size=micro, n_epochs=n_epochs,
                               seed=int(rec["model_seed"]), perturb_seed=int(rec["perturb_seed"]))
        apply_training_hparams(model, args, mappings=None, attach_cf_labels=lambda _m: None)
        _unset_to_class_defaults(model)
        model.n_epochs = n_epochs
        model.batch_size = micro
        model.grad_accum_steps = ACCUM          # ONE optimizer step over the whole buffer (module docs)
        model.behaviour_check = "off"
        model._logger = configure(None, [])
        rb = model.rollout_buffer
        rb.reset()
        for k, v in buf.items():
            if k.startswith("obs:"):
                rb.observations[k[4:]][...] = v
            else:
                getattr(rb, k)[...] = v
        rb.full, rb.pos = True, n_steps
        model._current_progress_remaining = 1.0
        np.random.seed(123)
        th.manual_seed(123)
        model.train()
        vals = {k: float(v) for k, v in model.logger.name_to_value.items()
                if isinstance(v, (int, float)) and np.isfinite(v) and not k.startswith("time/")
                and not k.endswith("_ms") and not k.endswith("_s")}
        params = th.cat([p.detach().reshape(-1).double() for p in model.policy.parameters()])
        results.append((params, vals))
    (pa, la), (pb, lb) = results
    dparam = float((pa - pb).abs().max())
    moved = float((pa - _fresh_params(rec, obs, act)).abs().max())
    keys = sorted(set(la) & set(lb))
    dloss = {k: abs(la[k] - lb[k]) for k in keys}
    # |Δ| against its allowance LEARNER_LOSS_ABS + LEARNER_LOSS_BAR x |value| (a scalar whose true value
    # is ~0 — the first step's policy-gradient loss on normalised advantages — has no relative scale)
    excess = {k: dloss[k] / (LEARNER_LOSS_ABS + LEARNER_LOSS_BAR * abs(la[k])) for k in keys}
    worst = max(excess, key=excess.get) if excess else None
    return {"max_abs_dparam": dparam, "param_moved_by_update": moved, "losses_compared": len(keys),
            "losses": keys, "worst_loss": worst, "worst_loss_abs": (dloss[worst] if worst else 0.0),
            "worst_loss_value": (la[worst] if worst else 0.0),
            "worst_loss_over_allowance": (excess[worst] if worst else 0.0),
            "pass": bool(moved > 0 and dparam <= LEARNER_PARAM_BAR and (not excess or max(excess.values()) <= 1.0))}


def _unset_to_class_defaults(model: Any) -> None:
    """``production_args()`` is the parser's namespace + the production config, NOT ``resolve_config``'s
    (which refuses a namespace whose every key reads as typed): the flags it leaves ``None`` are the ones
    ``_resolve`` would default. The learner check takes the class default for exactly those."""
    from main.train.model_build import _TRAINING_HPARAMS

    for name, _how in _TRAINING_HPARAMS:
        if getattr(model, name, None) is None and getattr(type(model), name, None) is not None:
            setattr(model, name, getattr(type(model), name))


def _fresh_params(rec: Dict[str, Any], obs: Any, act: Any) -> Any:
    import torch as th

    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_vec_env import RustVecEnv

    venv = RustVecEnv(n_envs=int(rec["n_envs"]), observation_space=obs, action_space=act, build=lambda m: None)
    m = TK.fresh_model(venv, n_steps=int(rec["n_steps"]), batch_size=int(rec["n_steps"]),
                       seed=int(rec["model_seed"]), perturb_seed=int(rec["perturb_seed"]))
    return th.cat([p.detach().reshape(-1).double() for p in m.policy.parameters()])


def run(*, n_envs: int = 3, n_steps: int = 24, windows: int = 2, learner: bool = True, **kw: Any) -> Dict[str, Any]:
    rec = record(n_envs=n_envs, n_steps=n_steps, windows=windows, **kw)
    rep = replay(rec)
    cmp = compare(rec["buffers"], rep["buffers"])
    out = {"n_envs": n_envs, "n_steps": n_steps, "windows": windows, "rows_compared": windows * n_envs * n_steps,
           "episodes": sum(len(s.teams) for s in rec["scripts"]), **cmp,
           "near_boundary": {"rust": rec["near_boundary_rust"], "python": rep["near_boundary_python"]},
           "lifecycle": rec["lifecycle"], "record_s": rec["record_s"], "replay_s": rep["replay_s"],
           "win_mask_rows": int(sum(float(b["obs:win_mask"].sum()) for b in rec["buffers"])),
           "episode_starts": int(sum(float(b["episode_starts"].sum()) for b in rec["buffers"]))}
    if learner:
        w = int(np.argmax([float(b["obs:win_mask"].sum()) for b in rec["buffers"]]))
        out["learner"] = {"window": w, **learner_check(rec, rec["buffers"][w], rep["buffers"][w])}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-envs", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=24)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--teams", default="pool", choices=("pool", "ladder"))
    ap.add_argument("--team-offset", type=int, default=0)
    ap.add_argument("--run-seed", type=int, default=17)
    ap.add_argument("--no-learner", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = run(n_envs=a.n_envs, n_steps=a.n_steps, windows=a.windows, learner=not a.no_learner, teams=a.teams,
              team_offset=a.team_offset, run_seed=a.run_seed)
    text = json.dumps(out, indent=1, default=str)
    print(text)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text)
    return 0 if not out["divergences"] and out.get("learner", {}).get("pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
