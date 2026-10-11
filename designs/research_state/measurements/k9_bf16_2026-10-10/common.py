"""Shared machinery for the K9(b)-under-bf16 measurement (T25 lever A; `README.md`).

Builds an ``--arch endstate`` learner exactly as a launch resolves it (`main.train.parser.build_parser` +
`main.train.config.resolve_config`, the perf phase's `update_bench.resolved_args`), on a seeded real
Rust-collector rollout (the `k9_tie_identity` / `k9_early_probe` harness: self-check build, T2 on the
declared device + backend, p2 a seeded random policy, seeded pool teams). Every forward helper here is a
READ: nothing in ``src/`` is changed, and the bf16 region is the production lever
(`agents.model.trunk_precision.set_trunk_precision`).
"""
from __future__ import annotations

import collections
import contextlib
import copy
import io
import os
import sys
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "designs/research_state/measurements/perf_phase_2026-10-10/scripts"))

BAR = 1e-4                       # K9(b)'s fp32 bar (consistency.BEHAVIOUR_BAR)
CEILING = 0.15                   # consistency.FP32_EXCLUDED_CEILING
CLIP = 0.15                      # --arch endstate's clip_range (resolved)
STALE_LRS = (2.8e-5, 3e-4)       # the 2026-09-30 fault table's lr, and --arch endstate's recipe lr
JITTER = 3e-7                    # the k9_behaviour_exclusion variant's few-ulp weight jitter


def round_up_125(x: float) -> float:
    """The smallest 1-2-5 decade value >= x (k9_behaviour_exclusion/derive.py's rule)."""
    if x <= 0:
        return 0.0
    e = np.floor(np.log10(x))
    for m in (1.0, 2.0, 5.0, 10.0):
        v = m * 10.0 ** e
        if v >= x * (1 - 1e-12):
            return float(v)
    return float(10.0 ** (e + 1))


def endstate_args(device: str, arch: str = "endstate") -> Any:
    from update_bench import resolved_args
    return resolved_args(["--arch", arch], device=device)


def _state_dict(path: str) -> Dict[str, th.Tensor]:
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    return th.load(path, map_location="cpu", weights_only=True)


def build(*, state: str, weights_from: Optional[str], seed: int, n_envs: int, n_steps: int, run_seed: int,
          device: str, t2_backend: str, trunk: str, train_hparams: bool = False, arch: str = "endstate"):
    """``(model, env, args)``: the endstate learner at ``state`` with T2 started (its replicas deep-copied
    from the policy AFTER the trunk precision ``trunk`` is set, so T2 serves that precision).

    ``state``: ``fresh`` (seeded init, no perturbation), ``perturb05`` / ``perturb10`` (+ the parity
    probe's name-keyed ladder noise at 0.05 / 0.1, seed 1234), ``weights`` (``weights_from``)."""
    from agents.model.parity_probe import perturb_
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.model.trunk_precision import set_trunk_precision
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.build import RustEnvDecl, trainee_spaces
    from agents.training.rust_vec_env import RustVecEnv
    from agents.training.train_logger import configure
    from main.fresh_checkpoint import _production_policy_kwargs
    from main.train.model_build import apply_training_hparams
    from utils.torch_state_guard import single_thread_build

    TK.build_selfcheck()
    args = endstate_args(device, arch)
    obs, act = trainee_spaces(args)
    decl = RustEnvDecl(n_envs=n_envs, threads=2, front="ffi", profile="selfcheck", n_steps=n_steps,
                       micro_batch=n_steps, device=device, backend=t2_backend, run_seed=run_seed, gamma=1.0,
                       gae_lambda=0.8, oracle_reveal=getattr(args, "oracle_reveal", "off") or "off")
    p2 = TK.RandomP2(5)
    env = RustVecEnv(n_envs=n_envs, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl, p2=p2, builder=TK.pool_builder()))
    _a, _l, pk = _production_policy_kwargs(args)
    th.manual_seed(seed)
    with single_thread_build():
        model = InstrumentedMaskablePPO(Gen3DualHeadMaskablePolicy, env, n_steps=n_steps, batch_size=n_steps * n_envs,
                                        n_epochs=1, gamma=1.0, gae_lambda=0.8, device=device, seed=seed,
                                        policy_kwargs=pk, verbose=0)
    model.ep_info_buffer = collections.deque(maxlen=100)
    model.ep_success_buffer = collections.deque(maxlen=100)
    if train_hparams:
        apply_training_hparams(model, args, mappings=None)
        TK.unset_to_class_defaults(model)
        model.grad_accum_steps = 1
        model.batch_size = n_steps * n_envs
        # the constructor's epochs / clip are sb3's (1 / 0.2 here); the resolved recipe's are 10 / 0.15
        model.n_epochs = int(args.n_epochs)
        clip = float(args.clip_range)
        model.clip_range = lambda _p, c=clip: c
    model._logger = configure(None, [])
    if state in ("perturb05", "perturb10"):
        perturb_(model.policy, seed=1234, scale=0.05 if state == "perturb05" else 0.1, keyed=True)
    elif state == "weights":
        sd = _state_dict(weights_from)
        own = model.policy.state_dict()
        bad = [k for k in own if k not in sd or tuple(sd[k].shape) != tuple(own[k].shape)]
        if bad:
            raise SystemExit(f"weights do not fit --arch {arch}: {bad[:8]}")
        model.policy.load_state_dict({k: sd[k] for k in own}, strict=True)
    elif state != "fresh":
        raise SystemExit(f"unknown state {state!r}")
    set_trunk_precision(model.policy, trunk)
    return model, env, args


def collect(model: Any, env: Any) -> Any:
    from agents.training.rust_rollout import testkit as TK
    col = env.startup(model)
    t0 = time.time()
    if not col.collect(model, TK.NullCallback(), model.rollout_buffer):
        raise RuntimeError("collector stopped")
    print(f"[k9-bf16] collected {model.rollout_buffer.log_probs.size} rows in {time.time() - t0:.0f}s", flush=True)
    return col


def rows(model: Any) -> Dict[str, Any]:
    """Every buffer row, flat, in (t, e) order."""
    buf = model.rollout_buffer
    n_envs = int(buf.n_envs)
    n = int(buf.log_probs.size)
    f = np.arange(n)
    t, e = f // n_envs, f % n_envs
    return {"obs": {k: v[t, e] for k, v in buf.observations.items()},
            "acts": th.as_tensor(buf.actions[t, e].reshape(-1)).long(),
            "masks": th.as_tensor(buf.action_masks[t, e]),
            "mu": buf.log_probs[t, e].astype(np.float64), "n": n}


@contextlib.contextmanager
def trunk(policy: Any, precision: str):
    """Set the trunk precision for the block and restore what it was (a class swap; `trunk_precision`)."""
    from agents.model.trunk_precision import set_trunk_precision, trunk_precision_of
    was = trunk_precision_of(policy)
    set_trunk_precision(policy, precision)
    try:
        yield
    finally:
        set_trunk_precision(policy, was)


def forward(model: Any, R: Dict[str, Any], *, chunk: int, train: bool, grad: bool, mode_factory=None,
            idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """``(log pi of the taken actions [n] f64, full masked log-probs [n, A] f64, the modes)`` of the
    policy on rows ``idx`` (all by default) in chunks of ``chunk``. ``train`` / ``grad`` select the
    learner-like probe forward (`consistency.probe_forward`: train mode, grad on, nothing saved) or a
    T2-like one (eval, no grad). ``mode_factory(rows)`` builds a TorchFunctionMode per chunk."""
    from stable_baselines3.common.utils import obs_as_tensor

    from agents.training.rust_rollout import consistency as K

    pol = model.policy
    idx = np.arange(R["n"]) if idx is None else np.asarray(idx)
    out = np.empty(idx.size)
    full: List[np.ndarray] = []
    modes: List[Any] = []
    for s in range(0, idx.size, chunk):
        sl = idx[s:s + chunk]
        obs = {k: v[sl] for k, v in R["obs"].items()}
        acts, masks = R["acts"][sl].to(model.device), R["masks"][sl].to(model.device)
        md = mode_factory(sl.size) if mode_factory is not None else None
        if train and grad:
            lp, fl = K.probe_forward(model, obs, acts, masks, md)
        else:
            was = pol.training
            pol.set_training_mode(train)
            try:
                ctx = th.enable_grad() if grad else th.no_grad()
                with ctx, (md if md is not None else contextlib.nullcontext()):
                    _v, logp, _e = pol.evaluate_actions(obs_as_tensor(obs, model.device), acts, action_masks=masks)
                lp = logp.detach().float().cpu().numpy().astype(np.float64)
                fl = K._stashed_logp(pol)
            finally:
                pol.set_training_mode(was)
            K.release_autograd_stashes(pol)
        out[s:s + sl.size] = lp
        full.append(np.asarray(fl, dtype=np.float64) if fl is not None else np.full((sl.size, 11), np.nan))
        modes.append(md)
    return out, np.concatenate(full, axis=0), modes


def margins_of(modes: List[Any], n: int) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray], np.ndarray]:
    """Merge per-chunk `TieMargins` (in order) into ``(margin [n], site [n], per-site margin {site: [n]},
    flippable [n])``. A chunk's ``check()`` is called (an undeclared op raises)."""
    margin = np.full(n, np.inf)
    site: List[str] = []
    per_site: Dict[str, np.ndarray] = {}
    flip = np.zeros(n, dtype=bool)
    o = 0
    for md in modes:
        md.check()
        k = md.rows
        margin[o:o + k] = md.margin
        site += list(md.site)
        for s, per in md.calls or []:
            arr = per_site.setdefault(s, np.full(n, np.inf))
            arr[o:o + k] = np.minimum(arr[o:o + k], per)
        if md.near_eps > 0:
            flip[o:o + k] = md.flippable()
        o += k
    return margin, site, per_site, flip


@contextlib.contextmanager
def jittered(policy: Any, rel: float, seed: int):
    """Every float parameter multiplied by (1 + rel * N(0, 1)) for the block, restored bit-exactly."""
    saved = {k: p.detach().clone() for k, p in policy.named_parameters()}
    g = th.Generator(device="cpu").manual_seed(int(seed))
    with th.no_grad():
        for p in policy.parameters():
            if p.is_floating_point():
                p.mul_(1 + rel * th.randn(p.shape, generator=g).to(p.device))
    try:
        yield
    finally:
        with th.no_grad():
            for k, p in policy.named_parameters():
                p.copy_(saved[k])


@contextlib.contextmanager
def stale_step(policy: Any, lr: float, seed: int = 5):
    """The 2026-09-30 / k9_behaviour_exclusion one-optimizer-step-stale fault: every float parameter moved
    by ``lr * sign(N(0, 1))`` (an Adam first step's per-weight size), seeded; restored bit-exactly."""
    saved = {k: p.detach().clone() for k, p in policy.named_parameters()}
    g = th.Generator(device="cpu").manual_seed(int(seed))
    with th.no_grad():
        for p in policy.parameters():
            if p.is_floating_point():
                p.add_(lr * th.sign(th.randn(p.shape, generator=g)).to(p.device))
    try:
        yield
    finally:
        with th.no_grad():
            for k, p in policy.named_parameters():
                p.copy_(saved[k])


def dist(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)] if x.size else x
    if not x.size:
        return {"n": 0}
    q = np.quantile(x, [0.5, 0.99, 0.999])
    return {"n": int(x.size), "p50": float(q[0]), "p99": float(q[1]), "p999": float(q[2]), "max": float(x.max()),
            "zero_frac": float((x == 0).mean())}


def per_block_max(x: np.ndarray, judged: np.ndarray, block: int) -> List[float]:
    out = []
    for s in range(0, x.size, block):
        j = judged[s:s + block]
        out.append(float(np.max(x[s:s + block][j])) if j.any() else float("nan"))
    return out


def fp64_copy(policy: Any) -> Any:
    """A float64 copy of ``policy`` at the fp32 trunk (take it BEFORE any grad-mode forward: a stash holding
    a graph tensor refuses `copy.deepcopy`)."""
    from agents.model.trunk_precision import set_trunk_precision
    K_release(policy)
    p64 = copy.deepcopy(policy)
    set_trunk_precision(p64, "fp32")
    p64 = p64.double()
    p64.set_training_mode(False)
    return p64


def fp64_logp(p64: Any, model: Any, R: Dict[str, Any], idx: np.ndarray, chunk: int = 256) -> np.ndarray:
    """log pi of the taken actions on rows ``idx`` from the float64 copy ``p64`` (`fp64_copy`) under
    `learner_golden_fp64.Fp64Mode` (every fp32 operand promoted), eval / no grad."""
    from stable_baselines3.common.utils import obs_as_tensor

    from agents.training.learner_golden_fp64 import Fp64Mode, _default_dtype

    out = np.empty(idx.size)
    for s in range(0, idx.size, chunk):
        sl = idx[s:s + chunk]
        obs = obs_as_tensor({k: v[sl] for k, v in R["obs"].items()}, model.device)
        with th.no_grad(), _default_dtype(th.float64), Fp64Mode():
            _v, lp, _e = p64.evaluate_actions(obs, R["acts"][sl].to(model.device),
                                              action_masks=R["masks"][sl].to(model.device))
        out[s:s + sl.size] = lp.detach().cpu().double().numpy()
    K_release(p64)
    return out


def K_release(pol: Any) -> None:
    from agents.training.rust_rollout import consistency as K
    K.release_autograd_stashes(pol)
