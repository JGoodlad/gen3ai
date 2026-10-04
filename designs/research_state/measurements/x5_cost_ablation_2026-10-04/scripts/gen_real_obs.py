"""REAL observation rows for the X5 parity check, from the Rust collector (2026-10-04).

    python gen_real_obs.py --out real_obs.npz [--envs 64] [--steps 64]

Same recipe as `agents.training.learner_golden rebuild_buffer` (a real complete-game rollout, the
production observation space, the SAME seeded perturbed learner as the behaviour policy, a random
external opponent) at a larger size, and WITHOUT touching the committed buffer. The observation space
does not depend on `--belief-tokens`, so a blob model collects and the fixed_mass model reads the rows.
CPU only. Writes `obs:<key>` arrays flattened to [envs*steps, ...] plus the action mask.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    from agents.training import learner_golden as LG
    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.build import RustEnvDecl
    from agents.training.rust_vec_env import RustVecEnv

    _a, obs, act = LG._spaces()
    decl = RustEnvDecl(n_envs=a.envs, threads=a.threads, front="proc", profile="release",
                       n_steps=a.steps, micro_batch=a.steps, device="cpu", backend="eager",
                       run_seed=LG.RECORD_RUN_SEED, gamma=1.0, gae_lambda=0.8)
    p2 = TK.RandomP2(LG.RECORD_P2_SEED)
    env = RustVecEnv(n_envs=a.envs, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl, p2=p2, builder=TK.pool_builder()))
    t0 = time.time()
    try:
        with LG._one_thread():
            model = TK.fresh_model(env, n_steps=a.steps, batch_size=a.steps, seed=LG.MODEL_SEED,
                                   perturb_seed=LG.PERTURB_SEED)
        col = env.startup(model)
        assert col.collect(model, TK.NullCallback(), model.rollout_buffer), "the collector stopped"
        rb = model.rollout_buffer
        out = {f"obs:{k}": np.ascontiguousarray(np.asarray(v).reshape(-1, *np.asarray(v).shape[2:]))
               for k, v in rb.observations.items()}
        out["action_masks"] = np.ascontiguousarray(np.asarray(rb.action_masks).reshape(-1, *np.asarray(rb.action_masks).shape[2:]))
    finally:
        env.close()
    np.savez_compressed(a.out, **out)
    n = next(iter(out.values())).shape[0]
    print(f"wrote {a.out}: {n} rows, {len(out)} arrays, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
