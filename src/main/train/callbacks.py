"""Phase 4 — CALLBACK ASSEMBLY: everything that runs DURING `learn()`.

The LR controller (adaptive, or the two-phase KL->cosine schedule), the checkpointer, the
exploiter temperature curriculum, the label back-fillers, the search teacher, and the one
non-blocking eval callback — `SelfPlayCallback` under `--self-play`, `PerOpponentEvalCallback`
otherwise, neither under a plain `--debug` smoke.

Each optional callback is registered ONLY when its flag is on, so an off run adds no callback and
makes no `env_method` push — the byte-identical property several of these flags claim.
"""
import dataclasses
import os
import sys
from typing import Any, List, Optional

from agents.model.snapshot import read_checkpoint_metadata
from agents.training.adaptive_lr_callback import AdaptivePPOCallback, TwoPhaseLRCallback
from agents.training.eval_callback import PerOpponentEvalCallback
from agents.training.graceful_restart_callback import GracefulRestartCallback
from agents.training.metrics_exporter_callback import MetricsExporterCallback
from agents.training.signal_callback import SignalMetricsCallback
from agents.training.selfplay_callback import SelfPlayCallback
from main.train.constants import (
    DEFAULT_EVAL_BATTLES, SMOKE_EVAL_BATTLES, SMOKE_STEPS, checkpoint_save_freq_vec_calls,
)
from main.train.run_io import DoseLogCallback, _HparamLogCallback, _TrackingCheckpointCallback


@dataclasses.dataclass
class CallbackBundle:
    """The callback list, plus the individual handles later phases still have to reach."""

    callbacks: List[Any]
    eval_callback: Optional[Any]
    lr_callback: Any
    adaptive_ppo_callback: Any
    graceful_restart_callback: Any
    effective_max_lr: float
    run_eval: bool


def build_callbacks(*, args, model_dir, server_config, annealing_mode, _pool,
                    _fixed_opponents, _bot_weight_vec, OPPONENT_CLASSES,
                    _specialist_team_str, _promote_threshold,
                    _heuristic_floor, _sp_start_wr, _sp_full_wr) -> CallbackBundle:
    """Build every `learn()`-time callback this run's flags ask for."""
    # --- Callback Setup (Shared) ---
    # Periodic checkpoints land in <run>/checkpoints/ (SB3 makedirs it); the callback
    # keeps latest.txt + metadata.json at the run root (derived from save_path).
    #
    # 🚨 `save_freq` IS IN VEC-ENV CALLS, NOT ENV STEPS — one `_on_step` per `vec_env.step()`, which
    # advances `n_envs` envs at once, so the real interval is `save_freq * n_envs`. This was a bare
    # hardcoded `50000` and was read as "50k steps" by everyone including the counterfactual R1
    # design; at `--n-envs 48` it is 2,400,000 env steps, which starved the label producer by 16x
    # its own staleness bound (`constants.checkpoint_save_freq_vec_calls` carries the measurement).
    # The conversion lives in `main.train.constants` because `config`'s duty-cycle refusal must
    # agree with it to the step, and phase 1 cannot import phase 4.
    #
    # A run that passes no `--checkpoint-every-steps` gets DEFAULT_CHECKPOINT_SAVE_FREQ_VEC_CALLS
    # back verbatim, so its checkpointer is byte-identical to the pre-flag one.
    _n_envs = 1 if args.debug else int(args.n_envs)      # --debug is DummyVecEnv: one env, always
    _save_freq = checkpoint_save_freq_vec_calls(
        getattr(args, "checkpoint_every_steps", None), _n_envs)
    checkpoint_callback = _TrackingCheckpointCallback(
        save_freq=_save_freq,
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="checkpoint",
    )

    # --lr must lie within [--min-lr, --max-lr]. This is the user-facing contract
    # for both pure-adaptive runs and TwoPhaseLR Phase 1 — KL adaptation reads
    # args.lr as the seed for fresh runs and as the cap for resumes, so it has
    # to be a valid in-band LR. Enforced before any callback is constructed.
    _effective_max_lr = args.max_lr if args.max_lr is not None else args.lr * 2.0
    if not (args.min_lr <= args.lr <= _effective_max_lr):
        print(f"[AdaptiveLR] ERROR: --lr {args.lr:.2e} is outside "
              f"[--min-lr {args.min_lr:.2e}, --max-lr {_effective_max_lr:.2e}]")
        sys.exit(1)

    # If resuming and the checkpoint has already crossed into the cosine phase,
    # read the persisted handoff_lr so the new callback can pick up the same
    # cosine starting point. None means "still in Phase 1, KL-driven."
    resumed_handoff_lr: float | None = None
    if annealing_mode and args.model and os.path.exists(args.model):
        try:
            _meta = read_checkpoint_metadata(args.model)
            _h = _meta.get("handoff_lr")
            if isinstance(_h, (int, float)):
                resumed_handoff_lr = float(_h)
        except Exception as e:
            print(f"[TwoPhaseLR] WARNING: failed to read handoff_lr from {args.model}: {e}")

    if annealing_mode:
        lr_callback = TwoPhaseLRCallback(
            initial_lr=args.lr,
            total_steps=args.steps,
            anneal_start_steps=args.anneal_lr_start_steps,
            anneal_min_lr=args.anneal_min_lr,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            handoff_lr=resumed_handoff_lr,
        )
    else:
        lr_callback = AdaptivePPOCallback(
            initial_lr=args.lr,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
        )
    # Keep alias so references below still resolve during the resume path.
    adaptive_ppo_callback = lr_callback
    # These two used to close over `main()`'s `model`, which did not exist yet at this point in
    # the function. They read the callback's OWN `self.model` instead — SB3 binds it in
    # `init_callback()` before any `_on_step`, and `_on_step` is the only consumer of both.
    checkpoint_callback._current_lr_fn = (
        lambda: checkpoint_callback.model.policy.optimizer.param_groups[0]["lr"])
    checkpoint_callback._current_epochs_fn = lambda: checkpoint_callback.model.n_epochs
    # Only TwoPhaseLRCallback exposes a handoff_lr; AdaptivePPOCallback does not.
    checkpoint_callback._handoff_lr_fn = (
        (lambda: lr_callback.handoff_lr) if isinstance(lr_callback, TwoPhaseLRCallback) else None
    )
    graceful_restart_callback = GracefulRestartCallback()
    # SIGNAL METRICS (gen3_signal_rate_metrics_v1): the `signal/outcome_entropy*` half of the
    # signal-rate group — rolling p(1−p) over the episode outcomes the training loop ALREADY sees
    # (`info["win_outcome"]` / `info["opponent_class"]`), split by opponent kind. ALWAYS ON and
    # flagless: it plays no battles, touches no env, and costs a handful of numpy means over ≤200-
    # element deques per rollout. Its partner `signal/adv_*` is recorded inside `train()`; the two
    # are only readable together (see agents/training/CLAUDE.md → the `signal/` group).
    signal_callback = SignalMetricsCallback()
    callbacks = [checkpoint_callback, lr_callback, MetricsExporterCallback(), _HparamLogCallback(args.ent_coef), DoseLogCallback(), graceful_restart_callback, signal_callback]
    # RANK TRIPWIRE (gen3_distill_target_gate_v1, design_advantage_gated_distillation.md §4.1):
    # watchdog over the EXISTING rank/policy_pr probe — EMA vs the run's own early baseline, with
    # a persistence rule. Default "warn" (no fold runs blind again); pure diagnostic bookkeeping —
    # no loss, no grad, no forward — except that "abort" stops learn() cleanly on a confirmed
    # collapse. "off" registers nothing.
    if getattr(args, "rank_tripwire", "warn") != "off":
        from agents.training.rank_tripwire import RankTripwireCallback
        callbacks.append(RankTripwireCallback(mode=args.rank_tripwire,
                                              drop=args.rank_tripwire_drop))
    # THE OFF-SLICE DISTILL ANCHOR (gen3_distill_offslice_anchor_v1) — the frozen fold PARENT that
    # `instrumented_ppo/distill_anchor.py` regularises toward, plus the live collateral meters.
    # Registered only when the coefficient is live OR --distill-anchor-monitor is on, so an ordinary
    # fold attaches nothing and stays byte-identical.
    #
    # A CALLBACK rather than an `apply_training_hparams` row on purpose: `_on_training_start` runs
    # on EVERY launch, which is the cadence at which the parent must be re-read from the ORIGINAL
    # fork-parent path (an idempotent fork's `--model` is swapped to the fork's own latest
    # checkpoint on each restart — anchoring to that would let the trust region drift with the
    # student). The resolution happens HERE, in phase 4, so an unresolvable parent refuses BEFORE
    # the model is built rather than mid-`learn()`. The loader is injected so the agents-layer
    # callback needs no `mappings` and no `main.train` import.
    if (getattr(args, "distill_anchor_coef", 0.0) or 0.0) > 0 or getattr(args, "distill_anchor_monitor", False):
        from agents.training.distill_anchor_callback import (
            DistillAnchorCallback, resolve_anchor_parent)
        _anchor_path, _anchor_route = resolve_anchor_parent(
            explicit=getattr(args, "distill_anchor_parent", None),
            run_dir=model_dir, cli_model=args.model)
        if not _anchor_path:
            from main.exit_codes import TrainExitCode
            print("\n[DistillAnchor] FATAL:--distill-anchor-coef / --distill-anchor-monitor is on "
                  "but no fold parent could be resolved — no --distill-anchor-parent, no --model, "
                  f"and {model_dir}/metadata.json records no `original_command` with a --model. "
                  "The anchor has nothing to anchor to; refusing rather than training without it.")
            sys.exit(int(TrainExitCode.FATAL_CONFIG))

        def _load_anchor_parent(path, _args=args):
            """Load the frozen parent the way a stable opponent / distill teacher is loaded."""
            from agents.model.snapshot import current_model_version, load_foreign_opponent
            from agents.observation.state_encoder import load_mappings
            from agents.training.fixed_opponent_pool import _resolve_zip_and_config
            from main.train.run_io import _run_arch_toggles
            _zip, _cfg, _ = _resolve_zip_and_config(path, None)
            _model, _ = load_foreign_opponent(
                _zip, current_version=current_model_version(load_mappings(),
                                                            **_run_arch_toggles(_args)),
                device=str(_args.device), config_path=_cfg)
            _model.policy.set_training_mode(False)
            return _model

        callbacks.append(DistillAnchorCallback(
            parent_path=_anchor_path, route=_anchor_route,
            coef=float(getattr(args, "distill_anchor_coef", 0.0) or 0.0),
            mode=str(getattr(args, "distill_anchor_mode", "off_slice") or "off_slice"),
            monitor=bool(getattr(args, "distill_anchor_monitor", False)),
            load_parent=_load_anchor_parent))
    # gen3_exploiter_temp_anneal_v1: control the EXPLOITER target's sampling temperature over training
    # (a difficulty curriculum via opponent stochasticity — hot/weak early → true strength later),
    # pushed to every env's exploiter RLPlayer via env_method each rollout. Registered ONLY when
    # --exploiter-temp-start is set → an off run makes no push (byte-identical). Training-only.
    # 'fixed' = linear time schedule; 'ratchet' = dynamic win-rate-driven one-way ratchet.
    if args.exploiter and args.exploiter_temp_start is not None:
        from agents.training.exploiter_temp_callback import (
            ExploiterTempAnnealCallback, ExploiterTempRatchetCallback)
        if args.exploiter_temp_mode == "ratchet":
            callbacks.append(ExploiterTempRatchetCallback(
                temp_start=args.exploiter_temp_start, temp_end=args.exploiter_temp_end,
                threshold=args.exploiter_temp_ratchet_wr, factor=args.exploiter_temp_ratchet_factor,
                min_games=args.exploiter_temp_ratchet_games, run_dir=model_dir))
        else:
            callbacks.append(ExploiterTempAnnealCallback(
                temp_start=args.exploiter_temp_start, temp_end=args.exploiter_temp_end,
                anneal_frac=args.exploiter_temp_anneal_frac))
    # gen3_exploiter_pool_ladder_v1: the OTHER difficulty axis — swap the exploiter target's WEIGHTS
    # up a ladder of frozen opponents (weakest → the --exploiter target) as the trainee's training
    # win-rate vs the LIVE rung clears the gate. Orthogonal to the temperature curriculum above (that
    # one varies stochasticity, this one varies strength) and composable with it. The rungs were
    # resolved + arch-gated in phase 2 (main.train.matchup_setup); registered ONLY when they exist →
    # an off run adds no callback and makes no env_method call (byte-identical). Training-only.
    _ladder_rungs = getattr(args, "_exploiter_ladder_rungs", None)
    if _ladder_rungs:
        from agents.training.exploiter_ladder import ExploiterLadderCallback
        callbacks.append(ExploiterLadderCallback(
            rungs=_ladder_rungs, gate=args.exploiter_ladder_gate,
            window=args.exploiter_ladder_window, run_dir=model_dir))
    # Win-probability head: captures each episode's win/loss outcome during collection + back-fills the
    # rollout buffer's MC label before train() (only when the head is on → a default run pays nothing).
    if args.win_prob_mode != "none":
        from agents.training.win_prob_callback import WinProbLabelCallback
        callbacks.append(WinProbLabelCallback())
    # Team-side PFSP: variance-weighted TEAM sampling by self-play win-rate. Registered ONLY when on
    # → an off run adds no callback and makes no env_method calls (byte-identical). Training-only.
    if args.team_pfsp != "off":
        from agents.training.team_pfsp_callback import TeamPFSPCallback
        callbacks.append(TeamPFSPCallback(cap=args.team_pfsp_cap, floor=args.team_pfsp_floor,
                                          mode=args.team_pfsp, persist_dir=model_dir))
    # PER-TEAM WIN-RATE TRACKING (default ON): instrumentation only — sparse TB summaries + a
    # restart-safe <run>/team_win_rates.json full table. Independent of --team-pfsp (different key,
    # different opponent scope, separate counter table); the two only share the builder's draw index.
    if getattr(args, "team_wr_tracking", True):
        from agents.training.team_winrate_callback import TeamWinRateCallback
        callbacks.append(TeamWinRateCallback(run_dir=model_dir))
    # SEARCH-TEACHER: each cycle, search + confirm the worst loss craters and distil verified-better
    # corrections into model._correction_buffer (the AWR aux loss samples it). Non-blocking subprocess
    # workers; off by default (the buffer fills nothing → coef-0 loss is byte-identical regardless).
    if args.search_teacher:
        from agents.training.teacher.callback import SearchTeacherCallback
        callbacks.append(SearchTeacherCallback(
            run_dir=model_dir,
            freq_steps=(args.teacher_search_freq if args.teacher_search_freq > 0 else 2_000_000),
            budget=args.teacher_search_budget, n_workers=args.teacher_search_workers,
            confirm_rollouts=args.teacher_confirm_rollouts,
            persistent=args.teacher_persistent, refresh_steps=args.teacher_refresh_steps,
            n_battles=args.teacher_gen_battles,
            # OPD: when --opd-coef>0 the workers ALSO build the improved distribution π' (the KL target).
            opd_build_pi_target=bool(args.opd_coef and args.opd_coef > 0), opd_beta=args.opd_beta,
            # ai_v12 routes 2+3: which TEACHER produces the corrections. "crater" (the default) is
            # the behaviour that existed before the flag; "winprob_oneply" swaps the selection +
            # production halves and leaves everything downstream of the CorrectionBuffer alone.
            mode=args.search_teacher_mode, wp_band=args.winprob_teacher_band,
            wp_margin=args.winprob_teacher_margin,
            # The workers' sim engine follows the run's --use-bridge impl (no separate flag);
            # "node" when the bridge is off, which is the historical behaviour.
            impl=args.bridge_impl,
            verbose=1))
    eval_callback = None
    # A --debug smoke run skips ALL eval by default — the periodic eval callback below AND the
    # final win-rate eval — so it needs no eval opponents / Showdown eval connection and stays
    # light on CPU. --debug-eval opts back in. Real (non-debug) runs are unaffected (always True).
    _run_eval = (not args.debug) or args.debug_eval
    # gen3_smoke_eval_scale_v1: resolve --eval-battles. An explicit value ALWAYS wins; otherwise a
    # short run gets the smoke count. Keyed on --steps rather than on a new --smoke flag so it needs
    # nothing remembered at the call site, and it cannot quietly weaken a real run: at 15M steps the
    # condition is false and the default stays 100.
    if args.eval_battles is None:
        _smoke = int(getattr(args, "steps", 0) or 0) < SMOKE_STEPS
        args.eval_battles = SMOKE_EVAL_BATTLES if _smoke else DEFAULT_EVAL_BATTLES
        if _smoke and _run_eval:
            print(f"[SmokeEval] --steps {args.steps:,} < {SMOKE_STEPS:,}: final eval scaled to "
                  f"{SMOKE_EVAL_BATTLES} battles/opponent (pass --eval-battles N to override). "
                  f"These win rates are NOT a measurement — they only prove the eval path runs.")

    # On resume, the last eval lives in the resumed checkpoint's metadata.json (a different
    # dir from this fresh run) — point the eval callback at it so the TUI shows the most
    # recent eval immediately instead of a blank panel until the next cycle.
    _resume_meta = None
    if args.model:
        _ckpt_dir = args.model if os.path.isdir(args.model) else os.path.dirname(args.model)
        # metadata.json is run-LEVEL (at the run root); a relocated checkpoint lives in
        # <run>/checkpoints/, so strip a trailing checkpoints/ to find it.
        if _ckpt_dir and os.path.basename(os.path.normpath(_ckpt_dir)) == "checkpoints":
            _ckpt_dir = os.path.dirname(_ckpt_dir)
        if _ckpt_dir:
            _resume_meta = os.path.join(_ckpt_dir, "metadata.json")

    if args.self_play and _pool is not None and _run_eval:
        # Self-play eval mirrors the bot-eval frozen-snapshot SUBPROCESS pattern
        # (non-blocking): the workers work-steal the bot roster AND up to 5 pool sentinels,
        # play a frozen snapshot, and the parent collects + promotes on a later poll. The
        # worker rebuilds opponents / teambuilders / mappings itself from the data dir, so
        # nothing live is constructed here. Under --debug it runs only with --debug-eval
        # (fast eval cadence), so `--self-play --debug --debug-eval` against a 9XXX server
        # exercises seed → pool eval → promotion; a plain --debug smoke skips it.
        eval_callback = SelfPlayCallback(
            pool=_pool,
            eval_games=args.eval_games,
            eval_freq=args.eval_freq,
            snapshot_ladder_games=args.snapshot_ladder_games,
            model_dir=model_dir,
            server_config=server_config,
            showdown_port=args.showdown_port,
            use_showdown_bridge=args.use_showdown_bridge,
            compile_extractor=args.compile_opponents,
            bridge_impl=args.bridge_impl,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            promote_threshold=_promote_threshold,
            self_play_temp=args.self_play_temp,
            # Greedy-vs-greedy pool eval (best-vs-best signal); default off keeps the live run's
            # win_rate_vs_pool / ELO continuous until opted in.
            eval_sentinel_greedy=args.eval_sentinel_greedy,
            # Curriculum knobs (#2): the live per-episode fraction the callback pushes each eval
            # uses these, matching the env's initial fraction computed above.
            heuristic_floor=_heuristic_floor,
            self_play_start_wr=_sp_start_wr,
            self_play_full_wr=_sp_full_wr,
            # Self-play eval is ~2x the inference of bot eval — the sentinel matchups
            # (--n-sentinels) run the model for BOTH players (trainee + sentinel), vs bot matchups
            # where only the trainee infers. So double the work-stealing pool to keep wall-clock
            # comparable (5 bot-eval workers → 10 here); raise --eval-workers too if --n-sentinels
            # is pushed high so the extra sentinel shards still drain promptly.
            n_workers=args.eval_workers * 2,
            eval_device=args.eval_device,
            eval_concurrency=args.eval_concurrency_per_worker,
            eval_shard_games=args.eval_shard_games,
            keep_eval_snapshots=args.keep_eval_snapshots,
            keep_eval_trace_steps=args.keep_eval_trace_steps,
            keep_stalls=args.keep_stalls,
            keep_crashes=args.keep_crashes,
            resume_eval_metadata=_resume_meta,
            fixed_opponents=_fixed_opponents,
            stable_opponent_mastered_wr=args.stable_opponent_mastered_wr,
            # Reporting-only: lets the callback REPORT the exact per-episode opponent-mix fractions
            # (train/selfplay_fraction = pool, train/stable_fraction, train/nonbot_fraction) the env
            # wrapper's selection implies — no change to selection. The capped stable challenge share,
            # the bot-weight vector, and the floor bot-roster size all live only in the wrapper / here.
            stable_challenge_share=args.stable_opponent_selfplay_share,
            stable_pfsp=args.stable_opponent_pfsp,
            bot_weight_vec=_bot_weight_vec,
            floor_roster_count=len(OPPONENT_CLASSES),
            # PFSP: when >0 the callback EMA-smooths the per-sentinel win-rates each eval and pushes
            # them to the env pools so sampling oversamples the selves we're losing to (0.0 = off).
            pfsp_scale=args.pfsp_scale,
            n_sentinels=args.n_sentinels,
            debug=args.debug,
            # --trainee-team pin → eval measures the trainee ON ITS OWN TEAM (None = default pool).
            trainee_team_str=_specialist_team_str,
        )
        callbacks.append(eval_callback)
    elif _run_eval:
        # Bot eval runs in a frozen-snapshot subprocess (non-blocking, CPU). The
        # worker rebuilds opponents/teambuilders/mappings itself from the data
        # dir, so nothing live is constructed here.
        eval_callback = PerOpponentEvalCallback(
            model_dir=model_dir,
            eval_games=args.eval_games,
            eval_freq=args.eval_freq,
            server_config=server_config,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            n_workers=args.eval_workers,
            eval_device=args.eval_device,
            eval_concurrency=args.eval_concurrency_per_worker,
            eval_shard_games=args.eval_shard_games,
            showdown_port=args.showdown_port,
            use_showdown_bridge=args.use_showdown_bridge,
            compile_extractor=args.compile_opponents,
            bridge_impl=args.bridge_impl,
            resume_eval_metadata=_resume_meta,
            keep_eval_snapshots=args.keep_eval_snapshots,
            keep_eval_trace_steps=args.keep_eval_trace_steps,
            keep_stalls=args.keep_stalls,
            keep_crashes=args.keep_crashes,
            fixed_opponents=_fixed_opponents,
            # --trainee-team pin → eval measures the trainee ON ITS OWN TEAM (None = default pool).
            trainee_team_str=_specialist_team_str,
        )
        callbacks.append(eval_callback)

    return CallbackBundle(
        callbacks=callbacks, eval_callback=eval_callback, lr_callback=lr_callback,
        adaptive_ppo_callback=adaptive_ppo_callback,
        graceful_restart_callback=graceful_restart_callback,
        effective_max_lr=_effective_max_lr, run_eval=_run_eval)
