"""K9(b) EARLY-PROBE exclusion (2026-10-06): which cutoff excludes each row at the first probe AFTER the
selection-free update, and can resolving it the other way move log pi?

CPU only (no GPU lease). The `k9_tie_identity_2026-10-05/measure.py` harness (a seeded complete-game Rust
collector rollout, the learner's probe forward under the tie-margin recorder, then FLIP forwards that
resolve every near-tied unit the other way and compare the full masked log-probs bit for bit), with ONE
change: the policy's weights are LOADED from a file (``--weights-from``: a ``behaviour_violation_u<n>_policy.pt``
state dict a K9(b) violation dumped, or a checkpoint's ``policy.pth``) before the rollout, so the rows
are played and judged at those weights. The source file is only READ. The rule BEFORE is now the rule
before BOTH `gen3_behaviour_tie_identity_v1` and `gen3_behaviour_tie_consumed_v1` (no payload identity,
every adjacent pair of the X5 sort head); ``new_rule`` is the production recorder at this commit.

    PYTHONPATH=src python designs/research_state/measurements/k9_early_probe_2026-10-06/measure.py \
        --arm fixed_mass --weights-from models/rb_x5ab_fm_s1006/behaviour_violation_u10_policy.pt --out …
"""
from __future__ import annotations

import importlib.util
import io
import os
import sys
import time
import zipfile

import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))
_OLD = os.path.join(os.path.dirname(_HERE), "k9_tie_identity_2026-10-05", "measure.py")
_spec = importlib.util.spec_from_file_location("k9_tie_identity_measure", _OLD)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)   # type: ignore[union-attr]

_WEIGHTS_FROM = None


def _capturing_site_margin(rule, name, args, kwargs, payload=None, out=None, consumed=None):
    """The old harness's capture, passing the production recorder's ``consumed`` count through (the flip
    forwards call `site_margin` without one: the rule BEFORE, every pair of the head)."""
    g = old._site_margin(rule, name, args, kwargs, payload, out, consumed)
    if old._CAPTURE is not None and g is not None:
        old._CAPTURE.append((rule, name, args, kwargs))
    return g


old.TM.site_margin = _capturing_site_margin


@old.contextlib.contextmanager
def _rule_before():
    """The rule BEFORE both `gen3_behaviour_tie_identity_v1` (no payload identity) and
    `gen3_behaviour_tie_consumed_v1` (no consumption declaration: every pair of the sort head)."""
    SS = old.SS
    saved = dict(SS.MARGIN)
    for k, r in list(SS.MARGIN.items()):
        if r.payload or r.consumed:
            SS.MARGIN[k] = r._replace(payload=(), consumed="")
    SS.line_map.cache_clear()
    try:
        yield
    finally:
        SS.MARGIN.clear()
        SS.MARGIN.update(saved)
        SS.line_map.cache_clear()


old.old_rule = _rule_before


def _state_dict(path: str):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    return th.load(path, map_location="cpu", weights_only=True)


def build(arm, weights, seed, n_envs, n_steps, run_seed):
    from agents.training.learner_golden import _one_thread
    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.build import RustEnvDecl, trainee_spaces
    from agents.training.rust_vec_env import RustVecEnv
    from main.train.production_args import production_args

    TK.build_selfcheck()
    args = production_args()
    for k, v in old.ARMS[arm].items():
        setattr(args, k, v)
    obs, act = trainee_spaces(args)
    decl = RustEnvDecl(n_envs=n_envs, threads=2, front="ffi", profile="selfcheck", n_steps=n_steps,
                       micro_batch=n_steps, device="cpu", backend="eager", run_seed=run_seed, gamma=1.0,
                       gae_lambda=0.8, oracle_reveal=old.ARMS[arm].get("oracle_reveal", "off"))
    p2 = TK.RandomP2(5)
    env = RustVecEnv(n_envs=n_envs, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl, p2=p2, builder=TK.pool_builder()))
    with _one_thread():
        model = TK.fresh_model(env, n_steps=n_steps, batch_size=n_steps, seed=seed, perturb_seed=1234,
                               policy_args=args, perturb_keyed=arm != "blob")
        sd = _state_dict(_WEIGHTS_FROM)
        own = model.policy.state_dict()
        missing = sorted(set(own) - set(sd))
        extra = sorted(set(sd) - set(own))
        bad = [k for k in set(own) & set(sd) if tuple(own[k].shape) != tuple(sd[k].shape)]
        if missing or bad:
            raise SystemExit(f"weights do not fit this arm: missing {missing[:8]} shape {bad[:8]}")
        model.policy.load_state_dict({k: sd[k] for k in own}, strict=True)
        print(f"loaded {len(own)} tensors from {_WEIGHTS_FROM} (ignored {len(extra)}: {extra[:6]})", flush=True)
    col = env.startup(model)
    t0 = time.time()
    if not col.collect(model, TK.NullCallback(), model.rollout_buffer):
        raise RuntimeError("collector stopped")
    print(f"collected in {time.time() - t0:.0f}s", flush=True)
    return model, env


def main() -> int:
    """``--weights-from <file>`` plays and judges at those weights; without it the 2026-10-05 harness's own
    ``--weights fresh|perturbed`` build runs (the before / after comparison at those states)."""
    global _WEIGHTS_FROM
    argv = list(sys.argv[1:])
    if "--weights-from" in argv:
        i = argv.index("--weights-from")
        _WEIGHTS_FROM = argv[i + 1]
        del argv[i:i + 2]
        old.build = build
    return old.main(argv)


if __name__ == "__main__":
    sys.exit(main())
