"""THE COLLECTION CELL of the frozen GPU benchmark suite (`gpu_bench_suite`, T13) — rollout throughput through the
REAL trainer: the Rust env core at the production n-envs, the trainee AND the self-play opponents served by the
batched inference service (T2), timed per rollout, with the collector's own phase breakdown.

WHY THE REAL TRAINER. Collection is ~27 s of X26's ~75 s cycle (`rollout/collect_ms`, 2026-10-18), and the
collector's own read says where it goes: per host step ~37 ms waiting on the GPU (`rust_env/gpu_wait_ms_per_host_step`)
and ~16 ms in the Rust core (`core_ms`), over ~416 host steps. So a faster card shortens collection too — but only
the GPU-wait part, and only under the opponent mix production runs: X26 plays 90 % of its episodes against its
20-snapshot pool (`train/selfplay_fraction` 0.90; `main.ops.tb_read` warns a throughput number without that fraction is
not interpretable: 0.0 -> 0.9 took a rollout from 107 s to 163 s on an older run). A harness that played the trainee
against bots would measure a different program.

WHAT IT RUNS (one process, from FIXTURES only — no run data, no `models/`):

1. a SCRATCH run directory (never under `models/`) whose self-play pool is pre-built through the pool's own API
   (`SnapshotPool.add`): ``n_snapshots`` copies of ONE seeded fresh production-surface model, each with its own seeded
   perturbation (distinct weights, as a real pool's), and a pool summary at ``win_rate_vs_bots`` (default 0.94, X26's
   0.936 rounded) — so the trainer's own startup computes the production self-play fraction (0.90) from it;
2. the trainer IN PROCESS (`learner_benchmark`'s worker pattern: `asyncio.run(main())` with a FRESH argv = the
   suite's production argv + `--run-dir <scratch> --steps <enough> --eval-freq <never> --no-compile-trainer`);
3. hooks: `collect_rollouts` is timed per call with the collector's stats snapshot around it (`rust_env/*`
   per-phase seconds); the LEARN-LOOP `train()` is a STUB that only advances the update counter, so after every
   rollout the collector reloads the (unchanged) weights into T2 exactly as production does after an update. The
   startup update-fit check (`update_fit.check_update_fits`) is skipped and the learner is not compiled: neither
   touches the collection path (T2 holds its own copies of the policy, built before any learner compile —
   `designs/training/rust_collector.md`), and both cost minutes of rented time;
4. ``warm`` untimed rollouts (T2 graph capture, pool loads, the first games' cold start), then ``timed`` rollouts;
   the result is written and the worker leaves by explicit PID (`learner_benchmark._exit_worker`).

CAVEATS (each recorded in the result): the policies are UNTRAINED fixtures, so the game mix (episode length, switch
rate, the 250-turn cap) is not a trained run's — the per-decision GPU cost is weight-independent (static shapes), the
per-host-step core cost is not; the rented host's CPU is not ours (the core part of the breakdown reads the host, the
GPU-wait part reads mostly the card, and NOT purely: a small-batch inference is partly launch-bound on the host).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: The collector's per-phase timers (`rust_rollout.collector`'s `stats.seconds`), read per host step.
COLLECTOR_PHASES = ("submit", "flush", "gpu_wait", "opp_draw", "draw", "write", "core", "post")
#: An eval cadence that cannot fire inside the measurement (an eval cycle spawns its own games).
NO_EVAL_FREQ = "1000000000000"
_TRAINER_LITERAL = "train_rl_" + "agent"   # never spelled whole in an argv (the live-run watchers grep for it)

_STATE: Dict[str, Any] = {}


def build_pool(run_dir: Path, args: Any, *, n_snapshots: int, win_rate: float, seed: int = 0,
               perturb_scale: float = 1e-3) -> Dict[str, Any]:
    """Pre-build ``<run_dir>/snapshots`` through `SnapshotPool`'s own API: ``n_snapshots`` seeded, perturbed copies of
    one seeded learner built on ``args`` (the resolved trainer namespace) plus the summary the trainer's startup reads
    its self-play fraction from. Deterministic: same args + seed => same bytes.

    The learner is `learner_golden.build_learner` — the trainee's OWN class over the trainee's spaces (every label key),
    exactly what a real pool's `add(model)` saves. A `main.fresh_checkpoint` model (observation + mask only) is REFUSED
    by T2 at the first opponent load: its forward fingerprint differs on the observation space (`SlotArchMismatch`,
    found by the CPU code path 2026-10-18)."""
    import torch

    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import load_mappings
    from agents.training.pool_seed import pool_dir_for
    from agents.training.snapshot_pool import SnapshotPool, heuristic_fraction
    from agents.training import learner_golden as LG
    from main.train.run_io import _run_arch_toggles

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    model = LG.build_learner(args=args)
    cv = current_model_version(load_mappings(), **_run_arch_toggles(args))
    pool = SnapshotPool(pool_dir=Path(pool_dir_for(str(run_dir))), current_version=cv, device="cpu",
                        max_snapshots=max(20, int(n_snapshots)), owns_dir=True)
    base = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    for i in range(int(n_snapshots)):
        g = torch.Generator().manual_seed(10_000 + i)
        with torch.no_grad():
            for name, p in model.policy.named_parameters():
                p.copy_(base[name] + perturb_scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
        pool.add(model, step=1_000_000 * (i + 1))
    pool.persist_summary(win_rate_vs_bots=float(win_rate), seeded=True, pool_generation=1,
                         note="gpu_bench_suite fixture pool (seeded fresh-init copies; not a training run)")
    return {"pool_dir": str(pool.pool_dir), "n_snapshots": len(pool), "win_rate_vs_bots": float(win_rate),
            "expected_self_play_fraction": 1.0 - heuristic_fraction(float(win_rate)),
            "perturb_scale": perturb_scale, "build_s": time.perf_counter() - t0}


def trainer_argv(base_argv: Sequence[str], *, run_dir: str, steps: int, device: str, seed: int,
                 extra: Sequence[str] = ()) -> List[str]:
    """The fresh argv the worker hands the trainer (pure). Never carries the trainer's module name."""
    argv = list(base_argv) + list(extra) + [
        "--device", device, "--no-compile-trainer", "--run-dir", run_dir, "--steps", str(int(steps)),
        "--eval-freq", NO_EVAL_FREQ, "--checkpoint-every-steps", str(10 * int(steps)), "--seed", str(int(seed))]
    bad = [t for t in argv if _TRAINER_LITERAL in t]
    if bad:
        raise ValueError(f"trainer argv would carry the trainer's module name: {bad}")
    return argv


def stats_delta(after: Dict[str, float], before: Dict[str, float]) -> Dict[str, float]:
    """One rollout's collector read from two `_snapshot()`s (pure): totals, decisions/s, ms per host step."""
    d = {k: float(after.get(k, 0.0)) - float(before.get(k, 0.0)) for k in after}
    steps = max(1.0, d.get("host_steps", 0.0))
    out = {"host_steps": d.get("host_steps", 0.0), "trainee_decisions": d.get("trainee_decisions", 0.0),
           "p2_policy_decisions": d.get("p2_policy_decisions", 0.0),
           "p2_external_decisions": d.get("p2_external_decisions", 0.0),
           "games_ended": d.get("games_ended", 0.0), "games_cut": d.get("games_cut", 0.0),
           "trainee_rows_per_host_step": d.get("trainee_decisions", 0.0) / steps,
           "fill_ms": 1000.0 * d.get("s_fill", 0.0)}
    for k in COLLECTOR_PHASES:
        out[f"{k}_ms_per_host_step"] = 1000.0 * d.get(f"s_{k}", 0.0) / steps
    return out


def summarise(rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Medians over the TIMED rollouts (pure)."""
    import statistics

    timed = [r for r in rollouts if r.get("timed")]
    if not timed:
        return {}

    def med(key: str) -> float:
        return float(statistics.median(float(r[key]) for r in timed))
    out: Dict[str, Any] = {"n_timed": len(timed), "wall_s_median": med("wall_s"),
                           "env_steps_per_s_median": med("env_steps_per_s"),
                           "trainee_decisions_median": med("trainee_decisions"),
                           "host_steps_median": med("host_steps"),
                           "p2_policy_share_median": med("p2_policy_share")}
    for k in COLLECTOR_PHASES:
        out[f"{k}_ms_per_host_step_median"] = med(f"{k}_ms_per_host_step")
    out["gpu_wait_s_per_rollout_median"] = out["gpu_wait_ms_per_host_step_median"] * out["host_steps_median"] / 1e3
    walls = [float(r["wall_s"]) for r in timed]
    out["wall_spread_rel"] = (max(walls) - min(walls)) / out["wall_s_median"] if len(walls) > 1 else 0.0
    return out


# --------------------------------------------------------------------------------------------- the worker

def _write(res: Dict[str, Any]) -> None:
    path = Path(_STATE["out"])
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=str))
    tmp.replace(path)


def _finish(code: int) -> None:
    from agents.training.learner_benchmark import _exit_worker
    _exit_worker(code)


def _peak_mib() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return torch.cuda.max_memory_reserved() / 2**20
    except Exception:  # noqa: BLE001
        pass
    return None


def _collect_hook(orig: Any) -> Any:
    def collect_rollouts(self: Any, *a: Any, **k: Any) -> Any:
        res = _STATE["res"]
        rc = getattr(self, "_rust_collector", None)
        before = rc._snapshot() if rc is not None else {}
        n0 = int(self.num_timesteps)
        t = time.perf_counter()
        ok = orig(self, *a, **k)
        wall = time.perf_counter() - t
        rows = int(self.num_timesteps) - n0
        rec: Dict[str, Any] = {"index": len(res["rollouts"]), "timed": len(res["rollouts"]) >= int(_STATE["warm"]),
                               "wall_s": wall, "env_steps": rows, "env_steps_per_s": rows / max(wall, 1e-9)}
        if rc is not None:
            rec.update(stats_delta(rc._snapshot(), before))
            rec["p2_policy_share"] = rec["p2_policy_decisions"] / max(1.0, rec["trainee_decisions"])
        else:
            rec["p2_policy_share"] = float("nan")
        rec["peak_reserved_mib"] = _peak_mib()
        eps = list(getattr(self, "ep_info_buffer", None) or [])
        rec["ep_len_mean_window"] = (sum(float(e["l"]) for e in eps) / len(eps)) if eps else None
        res["rollouts"].append(rec)
        if res.get("buffer_shape") is None:
            rb = getattr(self, "rollout_buffer", None)
            res["buffer_shape"] = {"n_steps": int(getattr(rb, "buffer_size", 0) or 0),
                                   "n_envs": int(getattr(rb, "n_envs", 0) or 0),
                                   "rows": int(getattr(rb, "buffer_size", 0) or 0) * int(getattr(rb, "n_envs", 0) or 0),
                                   "n_envs_model": int(getattr(self, "n_envs", 0) or 0)}
        print(f"[gpu_bench collect] rollout {rec['index']} ({'timed' if rec['timed'] else 'warm'}): {wall:.2f}s, "
              f"{rows} env steps, {rec['env_steps_per_s']:.0f}/s, gpu_wait "
              f"{rec.get('gpu_wait_ms_per_host_step', float('nan')):.2f} ms/host step", flush=True)
        _write(res)
        if len(res["rollouts"]) >= int(_STATE["warm"]) + int(_STATE["timed"]):
            res["summary"] = summarise(res["rollouts"])
            res["status"] = "ok"
            res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _write(res)
            _finish(0)
        return ok
    return collect_rollouts


def _stub_train(self: Any) -> None:
    """The learn loop's update, STUBBED: advance the update counter only, so the collector's `after_update` reloads the
    weights into T2 before the next rollout exactly as it does after a real update. Nothing is learned."""
    self._n_updates = int(getattr(self, "_n_updates", 0)) + 1


def worker_main(cfg_path: str) -> None:
    cfg = json.loads(Path(cfg_path).read_text())
    _STATE.update(out=cfg["out"], warm=cfg["warm"], timed=cfg["timed"])
    res: Dict[str, Any] = {"status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "trainer_argv": cfg["trainer_argv"], "pool": cfg.get("pool"), "rollouts": [],
                           "buffer_shape": None,
                           "stubbed": ["learn-loop train() -> counter only", "update_fit.check_update_fits",
                                       "learner compile (--no-compile-trainer)"]}
    _STATE["res"] = res
    _write(res)
    try:
        import asyncio
        import importlib

        from agents.training import update_fit
        from agents.training.instrumented_ppo import InstrumentedMaskablePPO
        from agents.training.learner_benchmark import install_worker_hooks

        update_fit.check_update_fits = lambda model, run_dir=None: None   # noqa: E731 — the startup gate (see docstring)
        install_worker_hooks(InstrumentedMaskablePPO, tool_train=_stub_train)
        InstrumentedMaskablePPO.collect_rollouts = _collect_hook(InstrumentedMaskablePPO.collect_rollouts)
        trainer = importlib.import_module("main." + _TRAINER_LITERAL)
        sys.argv = ["gpu_bench_collect[worker]"] + list(cfg["trainer_argv"])
        asyncio.run(trainer.main())
        res["status"] = "error"
        res["error"] = "the trainer returned before the timed rollouts completed"
        _write(res)
        _finish(4)
    except SystemExit as e:
        res["status"] = "error"
        res["error"] = f"the trainer exited ({e.code!r}) before the timed rollouts completed"
        _write(res)
        _finish(5)
    except BaseException as e:  # noqa: BLE001 — record, then leave by explicit PID
        res["status"] = "error"
        res["error"] = repr(e)[:3000]
        res["traceback"] = traceback.format_exc()[-6000:]
        _write(res)
        print(res["traceback"], flush=True)
        _finish(3)
