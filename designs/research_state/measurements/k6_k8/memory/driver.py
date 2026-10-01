"""K6 memory calibration DRIVER — a healthy production-shape learner's CUDA memory, update by update.

    export PYTHONPATH=$PYTHONPATH:src
    scripts/ops/gpu_lock.sh timeout 1200 scripts/ops/mem_cap.sh 40 \\
        python3 designs/research_state/measurements/k6_k8/memory/driver.py run \\
        --arm default|expandable --minutes 17 [--n-epochs 2] [--rollout-forwards 256]

One UNIT = one fresh worker process (a training process's lifetime in miniature):

* the PRODUCTION-SURFACE learner (`testkit.fresh_model` at `production_args()` + `recipe.fresh` of
  `designs/production_config.json`: 48 envs x 2048 steps, micro-batch 2048, grad-accum 32,
  `--diagnostics-every 10`, the win-prob critic), seeded and perturbed — weights never change what
  is allocated, shapes and branches do;
* `--compile-trainer` exactly as a launch arms it: the real-obs parity gate
  (`compile_trainer_extractor`), then `arm_compile_sentinel` (reset, prewarm every production
  signature, lock after the first update) — so the allocator sees the compiled program;
* the REAL 98,304-row buffer arm C's learner benchmark pickled (`~/gen3ai_archive/learner_bench/
  20260928_135948_cuda/rollout_buffer.pkl`; its keys and shapes are checked against this learner's);
* then, until the deadline, CYCLES of: restore the pristine buffer (host) -> a ROLLOUT BLOCK on the
  card (`--rollout-forwards` eval/no-grad `policy(obs, action_masks)` calls at 48 rows — the
  rollout's own signature — then the behaviour log-probs of all 98,304 rows recomputed in 48
  train/no-grad chunks of 2048, the rank probe's declared signature, so K9's behaviour gate holds) ->
  sample -> `train()` (a fresh minibatch permutation each update, like fresh data) -> sample; every
  `--save-every` updates a checkpoint-style `model.save`.

Every sample is one JSON line in `<out>/<unit>/samples.jsonl`, flushed and fsynced, so a killed unit
keeps every row it took; `<out>/units.jsonl` gets one row per finished unit. Units are independent:
a restarted session just runs more. `summarize` reads them all.

THE COMPILE CACHE: one root per (torch version, commit) under `<out>/compile_cache/`, declared by
this driver (as `agents.model.compile_cache_benchmark` declares its WARM root) and reused only by
this driver's own units — the run-own-restart semantics. A cached kernel is the same kernel, so the
cache cannot change what is allocated; it only spares each unit the ~11 min cold compile.

Benchmarks WARN, they never stretch: the box load is recorded per unit; a busy GPU is REFUSED.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_BUFFER = Path.home() / "gen3ai_archive/learner_bench/20260928_135948_cuda/rollout_buffer.pkl"
DEFAULT_OUT = Path.home() / "gen3ai_archive/k6_k8_memory"
ARMS = {"default": None, "expandable": "expandable_segments:True"}
N_ENVS, N_STEPS, MICRO = 48, 2048, 2048


def _append(path: Path, row: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load() -> Dict[str, Any]:
    try:
        la = os.getloadavg()
    except OSError:
        la = (-1.0, -1.0, -1.0)
    return {"load1": la[0], "load5": la[1], "load15": la[2], "cpus": os.cpu_count(),
            "uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()}


def _gpu_apps() -> str:
    return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout


# ------------------------------------------------------------------------------------------------
# worker
# ------------------------------------------------------------------------------------------------

def _build(cfg: Dict[str, Any]) -> Any:
    import torch as th
    from stable_baselines3.common.logger import configure

    from agents.training.baselines import production_recipe_block
    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_vec_env import RustVecEnv
    from main.rust_core_cutover.envs import production_args
    from agents.training.rust_rollout.parity import _unset_to_class_defaults
    from main.train.model_build import apply_training_hparams

    args = production_args()
    recipe = production_recipe_block()["fresh"]
    for k, v in recipe.items():
        if k != "kl_controller" and hasattr(args, k):
            setattr(args, k, v)
    args.diagnostics_every = 10            # the fresh default (gen3_diagnostics_cadence_v1)
    args.behaviour_check = "warn"          # K9(b) runs and logs; a mismatch cannot kill a unit
    _a, obs, act = TK.production_spaces()
    env = RustVecEnv(n_envs=N_ENVS, observation_space=obs, action_space=act, build=lambda m: None)
    model = TK.fresh_model(env, n_steps=N_STEPS, batch_size=MICRO, n_epochs=int(cfg["n_epochs"]),
                           seed=0, gamma=float(recipe["gamma"]), gae_lambda=0.95,
                           learning_rate=float(recipe["lr"]), ent_coef=float(recipe["ent_coef"]),
                           clip_range=float(recipe["clip_range"]), max_grad_norm=0.5, target_kl=None)
    apply_training_hparams(model, args, mappings=None, attach_cf_labels=lambda _m: None)
    _unset_to_class_defaults(model)        # as learner_golden: an unresolved None -> the class default
    model.grad_accum_steps = int(recipe["grad_accum_steps"])
    model.gae_lambda = float(args.policy_gae_lambda)
    dev = th.device(cfg.get("device", "cuda"))
    model.policy.to(dev)
    model.device = dev
    model.rollout_buffer.device = dev
    model._logger = configure(None, [])
    model._current_progress_remaining = 1.0
    th.set_float32_matmul_precision(str(args.matmul_precision or "highest"))
    return model, args


def _flat(a: Any) -> Any:
    """[n_steps, n_envs, ...] -> [n_envs * n_steps, ...] in SB3's `swap_and_flatten` order."""
    import numpy as np
    return np.ascontiguousarray(a.swapaxes(0, 1).reshape(-1, *a.shape[2:]))


def worker(cfg_path: str) -> None:
    cfg = json.loads(Path(cfg_path).read_text())
    from agents.model import compile_cache as CC
    CC._export(cfg["cache_root"])                       # before torch is imported
    out = Path(cfg["unit_dir"])
    rows = out / "samples.jsonl"
    deadline = time.monotonic() + float(cfg["minutes"]) * 60.0
    t_start = time.monotonic()

    import numpy as np
    import torch as th
    from stable_baselines3.common.utils import obs_as_tensor

    from agents.model.compile_trainer import arm_compile_sentinel, compile_trainer_extractor
    from agents.training import learner_benchmark as LB
    from agents.training.cuda_memory_trend import MemoryTrend, sample_cuda

    dev = th.device(cfg.get("device", "cuda"))
    cuda = dev.type == "cuda"
    if cuda:
        th.cuda.init()
    head = {"kind": "header", "torch": th.__version__, "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "cfg": cfg, "device_name": th.cuda.get_device_name(0) if cuda else "cpu", **_load()}
    _append(rows, head)
    model, args = _build(cfg)
    trend = MemoryTrend()

    def sync() -> None:
        if cuda:
            th.cuda.synchronize()

    def sample(update: int, phase: str, extra: Dict[str, Any]) -> None:
        if not cuda:
            _append(rows, {"kind": "smoke", "update": update, "phase": phase, **extra})
            return
        s = sample_cuda(dev, update=update, phase=phase)
        v = trend.observe(s)
        _append(rows, {"kind": "sample", "t": round(time.monotonic() - t_start, 3), **s.as_row(),
                       "verdict": v.level, "reasons": list(v.reasons), **extra})

    sample(0, "post_build", {})
    t0 = time.perf_counter()
    if cuda:
        compile_trainer_extractor(model, True, emit=None)
    gate_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    if cuda:
        arm_compile_sentinel(model, n_envs=N_ENVS, batch_size=MICRO, critic=args.critic, emit=None)
    sync()
    sample(0, "post_startup", {"gate_s": round(gate_s, 1), "prewarm_s": round(time.perf_counter() - t1, 1)})

    saved = LB.load_buffer(Path(cfg["buffer"]))
    buf = model.rollout_buffer
    live = LB.buffer_fingerprint(LB.capture_buffer_state(buf))
    want = LB.buffer_fingerprint(saved["buffer"])
    if live.get("observations") != want.get("observations"):
        raise SystemExit(f"buffer obs layout differs from the production learner's: "
                         f"{sorted(set(live.get('observations', {})) ^ set(want.get('observations', {})))}")
    pristine = saved["buffer"]
    for k, v in saved["stash"].items():
        setattr(model, k, v)
    obs_steps = pristine["observations"]                 # [n_steps, n_envs, ...] per key
    masks_steps = pristine["action_masks"]
    flat_obs = {k: _flat(v) for k, v in obs_steps.items()}
    flat_act = _flat(pristine["actions"]).reshape(-1)
    flat_mask = _flat(masks_steps)
    n_rows = flat_act.shape[0]
    pol = model.policy
    upd = 0
    rf = int(cfg["rollout_forwards"])
    save_dir = out / "save"
    save_dir.mkdir(exist_ok=True)
    while True:
        cyc0 = time.perf_counter()
        LB.restore_buffer_state(buf, pristine)
        # -- the ROLLOUT BLOCK: the rollout's own signature, then the behaviour log-probs
        pol.set_training_mode(False)
        with th.no_grad():
            for i in range(rf):
                t = (upd * 37 + i) % N_STEPS
                o = obs_as_tensor({k: v[t] for k, v in obs_steps.items()}, dev)
                pol(o, action_masks=masks_steps[t])
        sync()
        roll_s = time.perf_counter() - cyc0
        lp = np.empty(n_rows, dtype=np.float32)
        pol.set_training_mode(True)
        with th.no_grad():
            for a in range(0, n_rows, MICRO):
                sl = slice(a, a + MICRO)
                o = obs_as_tensor({k: v[sl] for k, v in flat_obs.items()}, dev)
                _v, logp, _e = pol.evaluate_actions(o, th.as_tensor(flat_act[sl], device=dev).long(),
                                                    action_masks=th.as_tensor(flat_mask[sl], device=dev))
                lp[sl] = logp.float().cpu().numpy()
        buf.log_probs[...] = lp.reshape(N_ENVS, N_STEPS).T
        sync()
        relabel_s = time.perf_counter() - cyc0 - roll_s
        sample(upd, "post_rollout", {"rollout_s": round(roll_s, 3), "relabel_s": round(relabel_s, 3)})
        # -- the UPDATE. `num_timesteps` advances by one rollout, as `collect_rollouts` does: the
        # diagnostics cadence's phase is the ROLLOUT INDEX (num_timesteps // rollout rows).
        model.num_timesteps += N_STEPS * N_ENVS
        np.random.seed(1000 + upd)
        th.manual_seed(1000 + upd)
        model.logger.name_to_value.clear()
        t2 = time.perf_counter()
        model.train()
        sync()
        train_s = time.perf_counter() - t2
        upd += 1
        ntv = model.logger.name_to_value
        extra = {"train_s": round(train_s, 3),
                 "train_ms": float(ntv["train/train_ms"]) if "train/train_ms" in ntv else None,
                 "loss": float(ntv["train/loss"]) if "train/loss" in ntv else None,
                 "diag": bool("train/noise_per_term_ms" in ntv),
                 "behaviour_max": float(ntv["behaviour/max_abs_dlogp_current"])
                 if "behaviour/max_abs_dlogp_current" in ntv else None}
        if upd % int(cfg["save_every"]) == 0:
            t3 = time.perf_counter()
            try:
                model.save(str(save_dir / "ckpt"))
                extra["save"] = "model.save"
            except Exception as e:  # noqa: BLE001 — the memory read does not depend on the file
                th.save(pol.state_dict(), str(save_dir / "ckpt_state.pt"))
                extra["save"] = f"state_dict ({type(e).__name__})"
            extra["save_s"] = round(time.perf_counter() - t3, 3)
        sample(upd, "post_update", extra)
        cycle = time.perf_counter() - cyc0
        if time.monotonic() + 1.3 * cycle > deadline:
            break
    _append(rows, {"kind": "footer", "updates": upd, "wall_s": round(time.monotonic() - t_start, 1),
                   **_load()})
    sys.stdout.flush()
    os._exit(0)


def t2probe(cfg_path: str) -> None:
    """Does T2's CUDA-graph backend start (compile, capture every slot x bucket, its parity gate) under
    this process's allocator config? The trainer hosts T2 under `--env-core rust`, so an allocator
    setting the learner wants must not break the inference service in the same process."""
    cfg = json.loads(Path(cfg_path).read_text())
    from agents.model import compile_cache as CC
    CC._export(cfg["cache_root"])
    out = Path(cfg["unit_dir"])
    import torch as th

    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.inference.service.fixtures import perturbed_fresh_policy
    from agents.training.cuda_memory_trend import sample_cuda
    row: Dict[str, Any] = {"kind": "t2probe", "torch": th.__version__,
                           "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), **_load()}
    try:
        policy = perturbed_fresh_policy(0)
        spec = ServiceSpec(groups=(SlotGroupSpec("trainee", 1, policy), SlotGroupSpec("pool", 4, policy)),
                           device="cuda", backend="graph", buckets=(8, 48), lanes=4,
                           max_rows_per_flush=max(1024, 5 * 48))
        t0 = time.perf_counter()
        svc = InferenceService(spec).startup()
        row["startup_s"] = round(time.perf_counter() - t0, 1)
        row["counters"] = {k: v for k, v in dict(svc.counters).items() if isinstance(v, (int, float))}
        row["parity_paths"] = dict(getattr(svc, "parity_paths", {}))
        row["ok"] = True
    except Exception as e:  # noqa: BLE001 — the probe's answer IS whether this raises
        import traceback
        row["ok"] = False
        row["error"] = repr(e)[:2000]
        row["traceback"] = traceback.format_exc()[-4000:]
    try:
        row["memory"] = sample_cuda("cuda", update=0, phase="t2_after_startup").as_row()
    except Exception as e:  # noqa: BLE001
        row["memory_error"] = repr(e)
    _append(out / "samples.jsonl", row)
    sys.stdout.flush()
    os._exit(0)


# ------------------------------------------------------------------------------------------------
# parent
# ------------------------------------------------------------------------------------------------

def run(a: argparse.Namespace) -> int:
    root = Path(a.out).expanduser().resolve()
    if "/models/" in str(root) or str(root).endswith("/models"):
        print("REFUSED: never write under models/")
        return 2
    apps = _gpu_apps() if a.device == "cuda" else ""
    if apps.strip():
        print(f"REFUSED: the GPU holds compute processes: {apps.strip()}")
        return 2
    import torch  # noqa: F401 — version only; no CUDA init in the parent
    tv = torch.__version__.split("+")[0]
    sha = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent)).stdout.strip()
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    unit = (f"{stamp}_{a.arm}_t{tv}_" + ("t2probe" if a.t2probe else f"e{a.n_epochs}")
            + ("_cpusmoke" if a.device == "cpu" else ""))
    unit_dir = root / "units" / unit
    unit_dir.mkdir(parents=True, exist_ok=False)
    cache_root = root / "compile_cache" / f"torch{tv}_{sha}"
    cache_root.mkdir(parents=True, exist_ok=True)
    cfg = {"unit": unit, "unit_dir": str(unit_dir), "cache_root": str(cache_root), "arm": a.arm,
           "minutes": a.minutes, "n_epochs": a.n_epochs, "rollout_forwards": a.rollout_forwards,
           "save_every": a.save_every, "device": a.device, "buffer": str(Path(a.buffer).expanduser()), "git": sha}
    (unit_dir / "cfg.json").write_text(json.dumps(cfg, indent=1))
    env = dict(os.environ)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if a.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if ARMS[a.arm]:
        env["PYTORCH_CUDA_ALLOC_CONF"] = ARMS[a.arm]
    t0 = time.monotonic()
    before = _load()
    with open(unit_dir / "worker.log", "wb") as lf:
        rc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__),
                             "_t2probe" if a.t2probe else "_worker",
                             str(unit_dir / "cfg.json")], stdout=lf, stderr=subprocess.STDOUT,
                            env=env).returncode
    _append(root / "units.jsonl", {"unit": unit, "arm": a.arm, "torch": tv, "git": sha, "rc": rc,
                                   "wall_s": round(time.monotonic() - t0, 1), "load_before": before,
                                   "load_after": _load(), "cfg": cfg})
    print(f"[k6mem] unit {unit}: exit {rc}, {time.monotonic() - t0:.0f}s — {unit_dir}")
    return rc


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=sorted(ARMS), required=True)
    r.add_argument("--minutes", type=float, default=17.0, help="worker wall budget incl. startup")
    r.add_argument("--n-epochs", type=int, default=2)
    r.add_argument("--rollout-forwards", type=int, default=256)
    r.add_argument("--save-every", type=int, default=10)
    r.add_argument("--buffer", default=str(DEFAULT_BUFFER))
    r.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="cpu = a code-path SMOKE (no compile, no memory rows; one cycle at --minutes 0)")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--t2probe", action="store_true",
                   help="instead of the learner: start T2's graph backend under the arm's allocator config")
    for name in ("_worker", "_t2probe"):
        w = sub.add_parser(name)
        w.add_argument("cfg")
    a = p.parse_args(argv)
    if a.cmd == "_worker":
        worker(a.cfg)
        return 0
    if a.cmd == "_t2probe":
        t2probe(a.cfg)
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
