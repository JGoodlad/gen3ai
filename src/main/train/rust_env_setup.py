"""Phase 3, ``--env-core rust`` (M5 Lane G): the trainer's ``RustVecEnv`` from the resolved args.

``env_factory.create_training_env_random`` builds today's per-worker ``Gen3Env`` closures; this module
is its twin for the Rust env core, fed the SAME inputs (the matchup's teambuilders, the floor roster,
``--bot-weights``, the stable opponents, the exploiter, the self-play pool and its starting fraction),
so the two env cores differ in WHERE a battle runs, not in what the run declared.

Two phases (the declared lifecycle; ``rust_vec_env`` module docs): ``build_rust_vec_env`` fixes the
spaces and N before the model exists; ``RustVecEnv.startup(model)`` — called by ``model_build`` right
after the model is built or loaded, BEFORE ``--compile-trainer`` touches the extractor — builds the
core, T2 and the arena from the model's OWN policy and hyperparameters (on a resume SB3 restores the
checkpoint's n_steps / batch_size / gamma / gae_lambda, so they are read off the model, never the argv).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from main.launcher.ipc import emit as _emit


def resolve_env_core_args(args: Any) -> None:
    """Fill every collector flag left ``None`` (untyped) with the default its help states. Idempotent."""
    rust = getattr(args, "env_core", "python") == "rust"
    defaults = {
        "rollout_trigger": "complete_game", "rollout_target_samples": 0, "rust_env_front": "proc",
        "rust_env_threads": 8, "rust_env_profile": "release", "rust_env_refusal_budget": 64,
        "rust_env_respawn_budget": 2, "version_pinning": "off", "t2_lanes": 0, "opponent_sampling": "keyed",
    }
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if getattr(args, "trainee_slots", None) is None:
        args.trainee_slots = 3 if args.version_pinning == "per_game" else 1
    if getattr(args, "behaviour_check", None) is None:
        args.behaviour_check = "fatal" if rust else "off"


def _bot_names(opponent_classes: Sequence[Any]) -> List[str]:
    """The floor roster's classes as Lane F's bot names (``bot_inventory``); BaitBot's per-run subclass
    (``make_baitbot_class``) maps to ``baitbot``."""
    from agents.baitbot import Gen3BaitBotPlayer
    from utils.rust_env import bot_inventory as BI

    by_cls = BI.by_class()
    out = []
    for cls in opponent_classes:
        if isinstance(cls, type) and issubclass(cls, Gen3BaitBotPlayer):
            out.append("baitbot")
            continue
        key = f"{cls.__module__}.{cls.__qualname__}"
        row = by_cls.get(key)
        if row is None or not row.rust:
            raise RuntimeError(f"--env-core rust: the floor roster's {key} has no ported Rust bot (Lane F)")
        out.append(row.name)
    return out


def segment_seed(seed: int, num_timesteps: int) -> int:
    """The collector's run seed for THIS process: a hash of the run's ``--seed`` and ``num_timesteps`` at
    startup — monotonic across a launcher restart, so a restart never replays the first segment's teams,
    battle seeds and keyed draws (the `win_prob_rollout` precedent)."""
    from agents.training import keyed_draw as KD

    return int(KD.draw_keys(int(seed or 0) & 0x7FFFFFFFFFFFFFFF, 9, 0, int(num_timesteps), 0)) & 0x7FFFFFFFFFFF


def build_rust_vec_env(args: Any, *, mappings: Any, trainee_teambuilder: Any, opponent_teambuilder: Any,
                       opponent_classes: Sequence[Any], bot_weights: Optional[Sequence[float]],
                       fixed_opponents: Sequence[Any], exploiter_entry: Any, snapshot_dir: Optional[str],
                       opponent_version: Any, self_play_fraction: float, n_envs: int,
                       emit: Callable[[str], None] = _emit) -> Any:
    """The ``RustVecEnv`` for this run (module docs)."""
    from agents.training import rust_env_opponents as E
    from agents.training.rust_rollout.build import OpponentSources, RustEnvDecl, build_collector, trainee_spaces
    from agents.training.rust_rollout.teams import pinned_builder
    from agents.training.rust_vec_env import RustVecEnv
    from utils.rust_env import episode as EP

    resolve_env_core_args(args)
    obs_space, act_space = trainee_spaces(args, mappings)
    plan = E.OpponentPlan.from_args(args, bot_names=_bot_names(opponent_classes),
                                    stable_entries=list(fixed_opponents or ()), exploiter_entry=exploiter_entry,
                                    heuristic_weights=bot_weights, bot_seed=0)
    band = getattr(args, "rollout_target_band", None)
    band_lo, band_hi = (int(x) for x in band.split(",")) if band else (0, 0)
    buckets = tuple(int(x) for x in args.t2_buckets.split(",")) if getattr(args, "t2_buckets", None) else ()
    device = str(args.device if args.device != "auto" else ("cuda" if _cuda() else "cpu"))
    backend = args.t2_backend or ("graph" if device.startswith("cuda") else "eager")

    def build(model: Any) -> Any:
        from agents.model.snapshot import load_foreign_opponent
        from agents.training.reward_config import RewardConfig
        from agents.training.snapshot_pool import SnapshotPool

        run_seed = segment_seed(getattr(args, "seed", 0), int(model.num_timesteps))
        decl = RustEnvDecl(
            n_envs=int(n_envs), threads=int(args.rust_env_threads), front=args.rust_env_front,
            profile=args.rust_env_profile, trigger=args.rollout_trigger, n_steps=int(model.n_steps),
            micro_batch=int(model.batch_size), target=int(args.rollout_target_samples or 0), band_lo=band_lo,
            band_hi=band_hi, gamma=float(model.gamma), gae_lambda=float(model.gae_lambda), run_seed=run_seed,
            turn_limit=EP.stall_threshold(), terminal=EP.terminal_from_reward_config(RewardConfig.from_args(args)),
            decision_tense=bool(getattr(args, "progress_decision_tense", False)),
            switch_freeze=bool(getattr(args, "progress_switch_freeze", False)),
            refusal_budget=int(args.rust_env_refusal_budget), respawn_budget=int(args.rust_env_respawn_budget),
            device=device, backend=backend, buckets=buckets, lanes=int(args.t2_lanes or 0),
            version_pinning=args.version_pinning == "per_game", trainee_slots=int(args.trainee_slots),
            opponent_sampling=args.opponent_sampling, policy_seed=run_seed)
        sources = OpponentSources(self_play_fraction=float(self_play_fraction))
        if plan.pool_slots and snapshot_dir is not None:
            sources.pool = SnapshotPool(pool_dir=snapshot_dir, current_version=opponent_version, device=device,
                                        pfsp_scale=getattr(args, "pfsp_scale", 0.0),
                                        pool_spread=getattr(args, "pool_spread", False))
        route_builders: Dict[int, Any] = {}
        for r in plan.routes():
            if r.team_strs:
                route_builders[r.index] = pinned_builder(r.team_strs)
        for e in (fixed_opponents or ()):
            m, _ = load_foreign_opponent(e.zip_path, current_version=opponent_version, device=device,
                                         config_path=e.config_path)
            sources.stable[e.label] = m.policy.eval()
        if exploiter_entry is not None:
            m, _ = load_foreign_opponent(exploiter_entry.zip_path, current_version=opponent_version, device=device,
                                         config_path=exploiter_entry.config_path)
            sources.exploiter = m.policy.eval()
        model.behaviour_check = args.behaviour_check
        col = build_collector(decl, obs_space=obs_space, trainee_policy=model.policy, plan=plan, sources=sources,
                              trainee_builder=trainee_teambuilder, opponent_builder=opponent_teambuilder,
                              route_builders=route_builders,
                              team_wr_tracking=bool(getattr(args, "team_wr_tracking", True)), emit=emit)
        model._env_core_stamp = env_core_stamp(decl, col)
        emit(f"🦀 [ENV CORE] rust — {model._env_core_stamp['summary']}")
        return col

    desc = (f"rust env core: {n_envs} envs, {args.rust_env_front} front end, {args.rollout_trigger} trigger, "
            f"T2 {backend} on {device}")
    return RustVecEnv(n_envs=int(n_envs), observation_space=obs_space, action_space=act_space, build=build,
                      describe=desc)


def env_core_stamp(decl: Any, col: Any) -> Dict[str, Any]:
    """What ``metadata.json`` records about the env core this process ran (``env_core``)."""
    stamp = getattr(col.core, "stamp", "") or ""
    if not stamp:
        try:
            stamp = col.core.lib.rust_env_stamp().decode()
        except Exception:
            stamp = ""
    trig = decl.trigger
    return {
        "env_core": "rust", "front": decl.front, "profile": decl.profile, "core_stamp": stamp,
        "n_envs": decl.n_envs, "threads": decl.threads, "trigger": trig,
        "target_samples": int(getattr(col.cfg.trigger, "target", 0) or 0),
        "version_pinning": decl.version_pinning, "trainee_slots": decl.trainee_slots,
        "t2": {"backend": decl.backend, "device": decl.device, "buckets": list(decl.resolved_buckets)},
        "opponent_sampling": decl.opponent_sampling, "keyed_draw": "gen3_keyed_draw_v1",
        "run_seed": decl.run_seed,
        "summary": (f"{decl.n_envs} envs x {decl.threads} threads, {decl.front}/{decl.profile}, "
                    f"{col.cfg.trigger.describe()}, T2 {decl.backend} buckets {decl.resolved_buckets}"),
    }


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
