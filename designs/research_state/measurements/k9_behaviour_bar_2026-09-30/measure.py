"""K9(b) — the healthy |Δ log π| distribution per fp32 matmul precision, and the injected faults' signal.

WHAT. The python env core's behaviour gate compares the ROLLOUT's log π(a|s) (eval mode, no grad, a
batch of n_envs rows) with the LEARNER's recomputation (train mode, grad on, a micro-batch) at the SAME
weights. This measures that difference on real rows, on CUDA, under `highest` (fp32) and `high` (TF32),
eager and `--compile-trainer`, across several seeds and several REAL updates each; then the signal of
two faults under the same conditions:

* STALE-1STEP — the learner holds weights ONE optimizer step newer than the rollout's (one real
  PPO step on the rows, at the recipe lr 2.8e-5 and at 3e-4);
* STALE-1UPDATE — one full update newer (2 epochs x 3 micro-batches);
* MODE — eval/train mode ALONE at the same batch shape (the policy has no dropout / batch norm, so any
  difference is kernel choice: e.g. nn.TransformerEncoderLayer's eval fast path, checkpointing).

ROWS. Per seed a real complete-game rollout from the Rust collector at the production surface (48 envs
x 128 steps = 6,144 rows; CPU, eager T2), learner = the learner golden's production-surface learner with
a per-seed perturbation. The gate statistic is the MAX over one micro-batch (2,048 rows, the production micro-batch), so both
the per-row distribution and the per-micro-batch max are reported.

INCREMENTAL (the GPU-lock rule, 2026-09-30): the driver runs OUTSIDE the lock; every ARM (one precision x
compile cell, ~15 min on the 11.6 GB card) is its own ``scripts/ops/gpu_lock.sh timeout 1500 ...``
acquisition in a fresh process, and writes a durable ``parts/<arm>.json`` — a re-run skips every arm
already written, so an interrupted pass resumes. The first pass (2026-09-30) ran all four arms under ONE
acquisition (~70 min), before the rule.

    python designs/research_state/measurements/k9_behaviour_bar_2026-09-30/measure.py \
        --out designs/research_state/measurements/k9_behaviour_bar_2026-09-30/result.json
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch as th

N_ENVS, N_STEPS, MICRO = 48, 128, 2048   # the production micro-batch; 3 per update per seed
SEEDS = (0, 1, 2, 3)
UPDATES = 3
LR_RECIPE, LR_HIGH = 2.8e-5, 3e-4
DEVICE = "cuda"


def collect_rows(seed: int):
    from agents.training import learner_golden as L
    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.build import RustEnvDecl
    from agents.training.rust_vec_env import RustVecEnv

    _a, obs, act = TK.production_spaces()
    decl = RustEnvDecl(n_envs=N_ENVS, threads=4, front="ffi", profile="selfcheck", trigger="complete_game",
                       n_steps=N_STEPS, micro_batch=N_STEPS, device="cpu", backend="eager",
                       run_seed=17 + seed, gamma=1.0, gae_lambda=0.8)
    p2 = TK.RandomP2(5 + seed)
    env = RustVecEnv(n_envs=N_ENVS, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl, p2=p2,
                                                      builder=TK.pool_builder(offset=seed)))
    try:
        model = TK.fresh_model(env, n_steps=N_STEPS, batch_size=N_STEPS, seed=L.MODEL_SEED,
                               perturb_seed=L.PERTURB_SEED + seed)
        col = env.startup(model)
        assert col.collect(model, TK.NullCallback(), model.rollout_buffer)
        rb = model.rollout_buffer
        out = {"obs": {k: np.asarray(v).copy() for k, v in rb.observations.items()}}
        for f in ("actions", "rewards", "episode_starts", "values", "log_probs", "advantages", "returns",
                  "action_masks"):
            out[f] = np.asarray(getattr(rb, f)).copy()
    finally:
        env.close()
    return out


def learner(seed: int, rows, compiled: bool):
    from stable_baselines3.common.logger import configure

    from agents.model.compile_trainer import compile_trainer_extractor
    from agents.training import learner_golden as L
    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.parity import _unset_to_class_defaults
    from agents.training.rust_vec_env import RustVecEnv
    from main.rust_core_cutover.envs import production_args
    from main.train.model_build import apply_training_hparams

    _a, obs, act = TK.production_spaces()
    env = RustVecEnv(n_envs=N_ENVS, observation_space=obs, action_space=act, build=lambda m: None)
    m = TK.fresh_model(env, n_steps=N_STEPS, batch_size=MICRO, n_epochs=2, seed=L.MODEL_SEED,
                       perturb_seed=L.PERTURB_SEED + seed, learning_rate=LR_RECIPE,
                       ent_coef=L.RECIPE["ent_coef"], clip_range=L.RECIPE["clip_range"])
    apply_training_hparams(m, production_args(), mappings=None, attach_cf_labels=lambda _m: None)
    _unset_to_class_defaults(m)
    m.grad_accum_steps, m.behaviour_check = 1, "off"
    m._logger = configure(None, [])
    m.policy.to(DEVICE)
    m.device = th.device(DEVICE)
    m.rollout_buffer.device = th.device(DEVICE)
    fill(m, rows, rows["log_probs"].reshape(-1))
    m._current_progress_remaining = 1.0
    if compiled:
        compile_trainer_extractor(m, True)
    return m


def fill(m, rows, mu: np.ndarray) -> None:
    """(Re)fill the buffer in its [n_steps, n_envs] layout (a previous `train()` flattened it)."""
    rb = m.rollout_buffer
    rb.reset()
    for k in rb.observations:
        rb.observations[k][...] = rows["obs"][k]
    for f in ("actions", "rewards", "episode_starts", "values", "advantages", "returns", "action_masks"):
        getattr(rb, f)[...] = rows[f]
    rb.log_probs[...] = mu.reshape(N_STEPS, N_ENVS)
    rb.full, rb.pos = True, N_STEPS


def _flat(rows):
    obs = {k: np.asarray(v).reshape(-1, *np.asarray(v).shape[2:]) for k, v in rows["obs"].items()}
    return obs, rows["actions"].reshape(-1).astype(np.int64), rows["action_masks"].reshape(-1, 11)


def logp(m, rows, *, batch: int, train_mode: bool) -> np.ndarray:
    from stable_baselines3.common.utils import obs_as_tensor

    obs, acts, masks = _flat(rows)
    m.policy.set_training_mode(train_mode)
    out = []
    ctx = th.enable_grad() if train_mode else th.no_grad()
    with ctx:
        for s in range(0, len(acts), batch):
            o = obs_as_tensor({k: v[s:s + batch] for k, v in obs.items()}, m.device)
            _v, lp, _e = m.policy.evaluate_actions(o, th.as_tensor(acts[s:s + batch], device=m.device),
                                                   action_masks=th.as_tensor(masks[s:s + batch], device=m.device))
            out.append(lp.detach().double().cpu().numpy())
    m.policy.set_training_mode(False)
    return np.concatenate(out)


def micro_maxes(d: np.ndarray) -> np.ndarray:
    return np.array([np.abs(d[s:s + MICRO]).max() for s in range(0, len(d), MICRO)])


def micro_stats(d: np.ndarray) -> list:
    """Per micro-batch: [max, p99, mean] of |Δ| — the gate statistic and two robust alternatives."""
    out = []
    for s in range(0, len(d), MICRO):
        a = np.abs(d[s:s + MICRO])
        out.append([float(a.max()), float(np.quantile(a, 0.99)), float(a.mean())])
    return out


def update(m, rows, mu: np.ndarray, *, n_epochs: int, batch: int, lr: float, seed: int, accum: int = 1) -> None:
    fill(m, rows, mu)
    m.n_epochs, m.batch_size, m.grad_accum_steps = n_epochs, batch, accum
    for g in m.policy.optimizer.param_groups:
        g["lr"] = lr
    m.lr_schedule = lambda _p: lr
    np.random.seed(seed)
    th.manual_seed(seed)
    m.train()


def summarise(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"n": int(x.size), "max": float(x.max()), "p99": float(np.quantile(x, 0.99)),
            "p99_9": float(np.quantile(x, 0.999)), "median": float(np.median(x))}


ARMS = [(p, c) for p in ("highest", "high") for c in (False, True)]


def run_arm(prec: str, compiled: bool, rows) -> dict:
    healthy, healthy_ms, mode, shape = [], [], [], []
    faults = {"stale_1step_lr_recipe": [], "stale_1step_lr_3e-4": [], "stale_1update": []}
    th.set_float32_matmul_precision(prec)
    for s in SEEDS:
        m = learner(s, rows[s], compiled)
        mu_prev = None
        for k in range(UPDATES):
            mu = logp(m, rows[s], batch=N_ENVS, train_mode=False)       # the rollout
            pi = logp(m, rows[s], batch=MICRO, train_mode=True)         # the learner
            d = pi - mu
            healthy.append(np.abs(d))
            healthy_ms += micro_stats(d)
            mode.append(np.abs(pi - logp(m, rows[s], batch=MICRO, train_mode=False)))
            shape.append(np.abs(mu - logp(m, rows[s], batch=MICRO, train_mode=False)))
            if mu_prev is not None:                                     # one update stale
                faults["stale_1update"] += micro_stats(pi - mu_prev)
            state = copy.deepcopy(m.policy.state_dict())
            opt = copy.deepcopy(m.policy.optimizer.state_dict())
            for name, lr in (("stale_1step_lr_recipe", LR_RECIPE), ("stale_1step_lr_3e-4", LR_HIGH)):
                update(m, rows[s], mu, n_epochs=1, batch=1024, accum=N_ENVS * N_STEPS // 1024,
                       lr=lr, seed=100 + k)                                 # ONE optimizer step (accumulated)
                faults[name] += micro_stats(logp(m, rows[s], batch=MICRO, train_mode=True) - mu)
                m.policy.load_state_dict(state)
                m.policy.optimizer.load_state_dict(opt)
            update(m, rows[s], mu, n_epochs=2, batch=MICRO, lr=LR_RECIPE, seed=200 + k)       # a real update
            mu_prev = mu
        del m
        if DEVICE == "cuda":
            th.cuda.empty_cache()
    return {"healthy_per_row": summarise(np.concatenate(healthy)),
            "healthy_micro_max": summarise([r[0] for r in healthy_ms]),
            "mode_only_per_row": summarise(np.concatenate(mode)),
            "batch_shape_only_per_row": summarise(np.concatenate(shape)),
            "healthy_micro_stats": healthy_ms,
            "faults_micro_stats": faults}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="CPU, one seed, two updates, eager fp32 (a code check)")
    ap.add_argument("--arm", default=None, help="internal: run ONE arm in this process (a fresh dynamo cache)")
    a = ap.parse_args()
    global SEEDS, UPDATES, DEVICE
    arms = ARMS
    if a.smoke:
        SEEDS, UPDATES, DEVICE, arms = (0,), 2, "cpu", [("highest", False)]
    if a.arm is not None or a.smoke:
        prec, comp = (a.arm or "highest/eager").split("/")
        rows = {s: collect_rows(s) for s in SEEDS}
        Path(a.out).write_text(json.dumps(run_arm(prec, comp == "compiled", rows)) + "\n")
        return 0
    import sys

    t0 = time.time()
    res = {"setup": {"n_envs": N_ENVS, "n_steps": N_STEPS, "rows_per_seed": N_ENVS * N_STEPS,
                     "micro": MICRO, "seeds": list(SEEDS), "updates_per_seed": UPDATES,
                     "lr_recipe": LR_RECIPE, "lr_high": LR_HIGH,
                     "torch": th.__version__, "micro_stats_columns": ["max", "p99", "mean"],
                     "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                              text=True).stdout.strip(),
                     "note": "each arm in its own process (a fresh dynamo cache per precision)"},
           "arms": {}}
    parts = Path(a.out).parent / "parts"
    parts.mkdir(exist_ok=True)
    lock = str(Path(__file__).resolve().parents[4] / "scripts" / "ops" / "gpu_lock.sh")
    for prec, comp in arms:
        arm = f"{prec}/{'compiled' if comp else 'eager'}"
        part = parts / (arm.replace("/", "_") + ".json")
        if not part.exists():                       # durable per arm: a re-run resumes
            tmp = part.with_suffix(".partial")
            r = subprocess.run([lock, "timeout", "1500", sys.executable, __file__, "--out", str(tmp), "--arm", arm])
            if r.returncode != 0:
                raise SystemExit(f"arm {arm} failed ({r.returncode}); the arms already written are kept")
            tmp.rename(part)
        res["arms"][arm] = json.loads(part.read_text())
        print(arm, json.dumps({k: v for k, v in res["arms"][arm].items() if "stats" not in k}), flush=True)
    res["wall_s"] = time.time() - t0
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
