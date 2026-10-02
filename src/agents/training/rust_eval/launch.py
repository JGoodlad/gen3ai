"""The eval callbacks' seam onto the Rust eval core (M5 Lane H).

Under ``--env-core rust`` both eval callbacks (``PerOpponentEvalCallback``, ``SelfPlayCallback``) write
the cycle's plan and manifest exactly as today and then call :func:`run_rust_eval_cycle` INSTEAD of
spawning ``main.eval_worker`` processes. The cycle runs IN THE TRAINER'S PROCESS, BLOCKING, on the
eval core and the T2 slots declared at startup (``rust_eval.build``); it publishes one
``ShardResult`` per unit into the run dir, so the callback's collect / record / promotion code is
unchanged. It runs between two host steps of the collector (the callbacks' ``on_step``), when no
T2 result of the rollout is outstanding.

A missing evaluator under ``--env-core rust`` is REFUSED (``RustEvalUnavailable``) — never a silent
fall-back to the Python workers.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional


class RustEvalUnavailable(RuntimeError):
    """``--env-core rust`` but the model carries no declared eval core."""


def cycle_seed(run_seed: int, step: int) -> int:
    """The cycle's per-game seed base (``rust_eval.seeds``): a hash of the collector's run seed and the
    eval step — a restart's re-eval at the same step replays the same games only within one segment."""
    d = hashlib.blake2b(f"gen3_eval_cycle_seed_v1:{int(run_seed)}:{int(step)}".encode(), digest_size=8).digest()
    return int.from_bytes(d, "big") & ((1 << 62) - 1)


def evaluator_of(model: Any) -> Any:
    rc = getattr(model, "_rust_collector", None)
    ev = getattr(rc, "evaluator", None)
    if ev is None:
        raise RustEvalUnavailable(
            "--env-core rust: the model's Rust collector declares no eval core (rust_env_setup.eval_decl) — "
            "eval cannot run on the Python workers under the Rust env core")
    return ev


def load_sentinels(pool: Any, model: Any) -> Dict[str, Any]:
    """Each SENTINEL item's snapshot, loaded as the eval worker loads it (``load_opponent_snapshot`` gated
    against THIS run's architecture), on CPU — T2 copies it into its declared slot."""
    from agents.model.snapshot import arch_toggles_from_model, current_model_version, load_opponent_snapshot
    from agents.observation.state_encoder import load_mappings
    from agents.training.eval_sharding import SENTINEL

    items = [it for it in pool.items if it.kind == SENTINEL]
    if not items:
        return {}
    version = current_model_version(load_mappings(), **arch_toggles_from_model(model))
    out = {}
    for it in items:
        m = load_opponent_snapshot(it.path, current_version=version, device="cpu")
        out[it.key] = m.policy.eval()
    return out


def run_rust_eval_cycle(cb: Any, *, pool: Any, run_dir: str, step: int,
                        game_log: Optional[List[Dict[str, Any]]] = None, seed: Optional[int] = None,
                        forensic: bool = True, record: bool = True) -> Dict[str, Any]:
    """Play the plan in ``run_dir`` on the Rust eval core (module docs). Returns the cycle's stats.

    ``seed`` overrides the cycle's per-game seed base (the SPRT promotion's batches play on their OWN
    namespace, ``sprt_promotion.sprt_seed``, so no selection game is ever replayed as a decision game);
    ``forensic=False`` captures no trace and ``record=False`` writes no ``rust_eval/*`` scalar (an SPRT
    batch is neither a trace sample nor the cycle's cost)."""
    model = cb.model
    ev = evaluator_of(model)
    rc = model._rust_collector
    model_dir = getattr(cb, "_model_dir", None)
    forensic_root = (os.path.join(model_dir, "eval_traces", f"step_{step}")
                     if (model_dir and forensic) else None)
    if seed is None:
        seed = cycle_seed(int(getattr(getattr(rc, "cfg", None), "run_seed", 0) or 0), step)
    was_training = bool(model.policy.training)
    model.policy.eval()
    try:
        st = ev.run_cycle(pool, run_dir, step=step, trainee_policy=model.policy,
                          sentinel_policies=load_sentinels(pool, model), forensic_root=forensic_root,
                          quota=getattr(cb, "_forensic_quota", None), gamma=float(model.gamma),
                          sentinel_greedy=bool(getattr(cb, "_eval_sentinel_greedy", False)),
                          self_play_temp=float(getattr(cb, "_self_play_temp", 1.0)), cycle_seed=seed,
                          game_log=game_log)
    finally:
        if was_training:
            model.policy.train()
    out = st.as_dict()
    out["cycle_seed"] = seed
    logger = getattr(cb, "logger", None) if record else None
    if logger is not None:
        s = out["seconds"]
        logger.record("rust_eval/cycle_wall_s", float(s["total"]))
        logger.record("rust_eval/games", float(out["games"]))
        logger.record("rust_eval/trainee_decisions_per_s", out["trainee_decisions"] / max(1e-9, s["total"]))
        logger.record("rust_eval/host_steps", float(out["host_steps"]))
        logger.record("rust_eval/traces", float(out["traces"]))
        logger.record("rust_eval/near_ties", float(out["near_ties"]))
        for k in ("load", "drain", "core", "trace"):
            logger.record(f"rust_eval/{k}_s", float(s[k]))
    return out
