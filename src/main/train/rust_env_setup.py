"""Phase 3, the Rust env core (M5 Lane G): the trainer's ``RustVecEnv`` from the resolved args.

The ONLY env core since deletion pass U3 (the Python core's per-worker ``Gen3Env`` factory,
``env_factory.create_training_env_random``, was deleted). It is fed the run's declarations: the matchup's
teambuilders, the floor roster, ``--bot-weights``, the stable opponents, the exploiter, the self-play pool
and its starting fraction.

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
    defaults = {
        "rollout_trigger": "complete_game", "rollout_target_samples": 0, "rust_env_front": "proc",
        "rust_env_threads": 8, "rust_env_profile": "release", "rust_env_refusal_budget": 64,
        "rust_env_respawn_budget": 2, "version_pinning": "off", "t2_lanes": 0, "t2_opponent_bucket_cap": 64,
        "opponent_sampling": "keyed",
        "rust_eval_envs": 64,
    }
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if getattr(args, "trainee_slots", None) is None:
        args.trainee_slots = 3 if args.version_pinning == "per_game" else 1
    if getattr(args, "behaviour_check", None) is None:
        # K9(b) is FATAL on both env cores (M5 Lane K): under python it reads the first micro-batch's
        # own forward (`instrumented_ppo/learner_gates.py`) — one host read per update, no forward.
        args.behaviour_check = "fatal"


def recorded_env_core(model_path: Optional[str]) -> Optional[str]:
    """The env core a checkpoint was PRODUCED on (``"python"`` / ``"rust"``), or None when unrecorded.

    Its sidecar (``<ckpt>.json``) first, else the run's ``metadata.json`` — both carry ``env_core`` on every
    save (``run_io._model_hparams``). JSON only; no ``.zip`` is opened."""
    import json
    import os

    if not model_path:
        return None
    try:
        from agents.training.lineage import resolve_model_path, run_dir_of

        path = resolve_model_path(model_path)
        cands = [os.path.splitext(path)[0] + ".json"]
        run_dir = run_dir_of(model_path)
        if run_dir:
            cands.append(os.path.join(run_dir, "metadata.json"))
    except Exception:
        return None
    for c in cands:
        try:
            with open(c) as f:
                ec = json.load(f).get("env_core")
        except (OSError, ValueError):
            continue
        if isinstance(ec, dict) and ec.get("env_core") in ("python", "rust"):
            return str(ec["env_core"])
    return None


class PythonEraShapedCheckpoint(ValueError):
    """Deletion pass D4: a ``--model`` checkpoint that trained the SHAPED critic cannot resume or fork on
    this code — the Rust core refuses the shaped critic and the Python core that served it was deleted
    (U3). Re-running cannot change it (``FATAL_CONFIG``); the way out is to run it PINNED to a commit that
    has the shaped path (the launcher pins a resume to its checkpoint's commit by default)."""


def recorded_critic(model_path: Optional[str], saved_ver: Any = None) -> Optional[str]:
    """The critic a checkpoint TRAINED (``"shaped"`` / ``"winprob"``), or None when no record can be read.

    The parsed ``ModelVersion`` when the caller has it, else the RAW ``model_config.json`` (JSON only). A
    config that never carried the key predates ``--critic`` → ``CRITIC_UNRECORDED`` (shaped)."""
    import json

    from agents.model.critic_mode import CRITIC_UNRECORDED

    if saved_ver is not None:
        return str(getattr(saved_ver, "critic", CRITIC_UNRECORDED))
    if not model_path:
        return None
    from agents.model.model_version.shaped_reward import saved_config_path
    path = saved_config_path(model_path)
    if not path:
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    return str(raw.get("critic", CRITIC_UNRECORDED)) if isinstance(raw, dict) else None


def python_era_refusal(model_path: str, critic: str) -> str:
    return (f"⛔ [ENV CORE] FATAL (deletion pass D4): {model_path} trained the {critic!r} critic, which no "
            "env core on this code serves — the Rust core refuses it and the Python core was deleted (U3). "
            "Run it PINNED to its own commit (the launcher's default for a resume; never --no-pin / "
            "--sync-to-main), or start a fresh run. Not a silent switch: a shaped critic cannot be carried "
            "onto the win-prob objective (designs/ops/deletion_pass_manifest.md §0 D4).")


#: The announcement a python-era checkpoint's move onto the Rust core carries (D4).
D4_CORE_SWITCH = ("a PYTHON-ERA checkpoint moves onto the Rust core — a CORE SWITCH (deletion pass D4): "
                  "the data stream changes (keyed opponent draws, the complete-game collector); the "
                  "checkpoint's weights and recorded config carry across, as the M5 switch's did")


def refuse_python_era_checkpoint(model: Optional[str], saved_ver: Any = None) -> None:
    """Deletion pass D4, the one refusal that outlived ``--env-core``: a ``--model`` checkpoint that trained
    the SHAPED critic cannot resume or fork on this code. Raises :class:`PythonEraShapedCheckpoint`; a no-op
    for a fresh launch (``model`` falsy), for a winprob checkpoint, and when no record can be read.

    Keyed on the RECORD (``recorded_critic``: the parsed ``ModelVersion`` when the caller has it, else the
    raw ``model_config.json``), never on a flag — there is no ``--env-core`` or ``--critic`` to type any
    more (deletion pass P11b). A python-era checkpoint that DID train winprob (produced on ``python``, or
    recorded before the env core was stamped) is not refused: it moves onto the Rust core ANNOUNCED, by
    :func:`env_core_switch_line` (D4; the ledger's M5 switch entry carries checkpoints across)."""
    from agents.model.critic_mode import is_winprob
    if not model:
        return
    critic = recorded_critic(model, saved_ver)
    if critic is not None and not is_winprob(critic):
        raise PythonEraShapedCheckpoint(python_era_refusal(model, critic))


def env_core_switch_line(args: Any) -> Optional[str]:
    """A LOUD line when a resume runs on another env core than its checkpoint was produced on, else None.

    A python-era checkpoint moves onto the Rust core (deletion pass D4) — a change of data stream, announced
    as a CORE SWITCH. It is not a refusal, and it may not be silent. Keyed on the checkpoint's RECORDED core
    (``recorded_env_core``), since there is no ``--env-core`` flag any more."""
    model = getattr(args, "model", None)
    if not model:
        return None
    rec = recorded_env_core(model) or "python"     # recorded before the env core was stamped = python
    if rec == "rust":
        return None
    return f"🔀 [ENV CORE] CORE SWITCH — {D4_CORE_SWITCH}"


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
            raise RuntimeError(f"the Rust env core: the floor roster's {key} has no ported Rust bot (Lane F)")
        out.append(row.name)
    return out


def segment_seed(seed: int, num_timesteps: int) -> int:
    """The collector's run seed for THIS process: a hash of the run's ``--seed`` and ``num_timesteps`` at
    startup — monotonic across a launcher restart, so a restart never replays the first segment's teams,
    battle seeds and keyed draws."""
    from agents.training import keyed_draw as KD

    return int(KD.draw_keys(int(seed or 0) & 0x7FFFFFFFFFFFFFFF, 9, 0, int(num_timesteps), 0)) & 0x7FFFFFFFFFFF


def build_rust_vec_env(args: Any, *, mappings: Any, trainee_teambuilder: Any, opponent_teambuilder: Any,
                       opponent_classes: Sequence[Any], bot_weights: Optional[Sequence[float]],
                       fixed_opponents: Sequence[Any], exploiter_entry: Any, snapshot_dir: Optional[str],
                       opponent_version: Any, self_play_fraction: float, n_envs: int,
                       eval_trainee_team_str: Any = None,
                       emit: Callable[[str], None] = _emit) -> Any:
    """The ``RustVecEnv`` for this run (module docs)."""
    from agents.training import rust_env_opponents as E
    from agents.training.rust_rollout.build import OpponentSources, RustEnvDecl, build_collector, trainee_spaces
    from agents.training.rust_rollout.teams import pinned_builder
    from agents.training.rust_vec_env import RustVecEnv
    from utils.rust_env import episode as EP

    resolve_env_core_args(args)
    # F-LG-6: build THIS checkout's env core now (startup), so a launcher PIN — a fresh worktree with
    # no src/rust_env/target — runs; incremental, a no-op when current (`utils.rust_env.build`).
    from utils.rust_env.build import ensure_built

    ensure_built(args.rust_env_profile, emit=emit)
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
        from agents.training.rust_rollout.fork import fork_decl_from_args
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
            opponent_bucket_cap=int(args.t2_opponent_bucket_cap),
            version_pinning=args.version_pinning == "per_game", trainee_slots=int(args.trainee_slots),
            opponent_sampling=args.opponent_sampling, policy_seed=run_seed, fork=fork_decl_from_args(args))
        sources = OpponentSources(self_play_fraction=float(self_play_fraction))
        if plan.pool_slots and snapshot_dir is not None:
            # device="cpu" (gen3_declared_slot_load_v1): a pool snapshot is only a WEIGHT SOURCE that T2 copies
            # into its declared slot; loaded on the card, each promotion's snapshot stayed there in the pool's
            # LRU (+~33 MiB of quiescent floor per promotion, measured on sizing arm A, 2026-10-01).
            sources.pool = SnapshotPool(pool_dir=snapshot_dir, current_version=opponent_version, device="cpu",
                                        pfsp_scale=getattr(args, "pfsp_scale", 0.0),
                                        pool_spread=getattr(args, "pool_spread", False))
        route_builders: Dict[int, Any] = {}
        for r in plan.routes():
            if r.team_strs:
                route_builders[r.index] = pinned_builder(r.team_strs)
        # device="cpu" (gen3_declared_slot_load_v1, P10 follow-up F1): a stable / exploiter opponent is only a
        # WEIGHT SOURCE — T2 copies it into its declared slot (a group template, then `svc.load`) and nothing
        # reads the policy again. Loaded on the card it stayed there for the run, a duplicate of its slot
        # (`OpponentSources.policy_for` refuses a source on the card, as it does the pool's).
        for e in (fixed_opponents or ()):
            m, _ = load_foreign_opponent(e.zip_path, current_version=opponent_version, device="cpu",
                                         config_path=e.config_path)
            sources.stable[e.label] = m.policy.eval()
        if exploiter_entry is not None:
            m, _ = load_foreign_opponent(exploiter_entry.zip_path, current_version=opponent_version, device="cpu",
                                         config_path=exploiter_entry.config_path)
            sources.exploiter = m.policy.eval()
        model.behaviour_check = args.behaviour_check
        # M5 Lane H: eval on the core — its slots are part of the ONE T2 declaration (startup only).
        edecl = eval_decl(args, plan, fixed_opponents)
        extra = ()
        if edecl is not None:
            from agents.training.rust_eval.build import eval_extra_slots

            extra = tuple(eval_extra_slots(edecl, model.policy, sources.stable))
        col = build_collector(decl, obs_space=obs_space, trainee_policy=model.policy, plan=plan, sources=sources,
                              trainee_builder=trainee_teambuilder, opponent_builder=opponent_teambuilder,
                              route_builders=route_builders,
                              team_wr_tracking=bool(getattr(args, "team_wr_tracking", True)), emit=emit,
                              extra_slots=extra)
        col.evaluator = None
        if edecl is not None:
            from agents.training.rust_eval.build import build_eval_core, eval_builders

            etb, eopp, efixed = eval_builders(eval_trainee_team_str, list(fixed_opponents or ()))
            col.evaluator = build_eval_core(edecl, collector_decl=decl, svc=col.svc, extra_ids=col.extra_slots,
                                            trainee_builder=etb, opp_builder=eopp, fixed_builders=efixed,
                                            turn_limit=decl.turn_limit, terminal=decl.terminal,
                                            fixed_policies=sources.stable, emit=emit)
        model._env_core_stamp = env_core_stamp(decl, col)
        emit(f"🦀 [ENV CORE] rust — {model._env_core_stamp['summary']}")
        return col

    desc = (f"rust env core: {n_envs} envs, {args.rust_env_front} front end, {args.rollout_trigger} trigger, "
            f"T2 {backend} on {device}")
    return RustVecEnv(n_envs=int(n_envs), observation_space=obs_space, action_space=act_space, build=build,
                      describe=desc)


def eval_decl(args: Any, plan: Any, fixed_opponents: Sequence[Any]) -> Any:
    """The Rust eval declaration (M5 Lane H) — None when this run evaluates nothing (the callbacks'
    ``_run_eval``: a ``--debug`` smoke without ``--debug-eval``)."""
    from agents.training.rust_eval.build import EvalDecl

    if bool(getattr(args, "debug", False)) and not bool(getattr(args, "debug_eval", False)):
        return None
    fixed = tuple(e.label for e in (fixed_opponents or ()))
    training_slots = {r.family.split(":", 1)[1]: int(r.slot) for r in plan.routes()
                      if r.kind == "policy" and r.family.startswith("stable:")}
    return EvalDecl(n_envs=int(getattr(args, "rust_eval_envs", 64) or 64),
                    n_sentinels=int(getattr(args, "n_sentinels", 5)) if bool(getattr(args, "self_play", False)) else 0,
                    fixed_labels=fixed,
                    reused_fixed=tuple((lab, training_slots[lab]) for lab in fixed if lab in training_slots))


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
