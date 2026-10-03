"""Phase 4 — CALLBACK ASSEMBLY: everything that runs DURING `learn()`.

The LR controller (adaptive, or the two-phase KL->cosine schedule), the checkpointer, the
exploiter temperature curriculum, the label back-fillers, and the one
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
from agents.model.critic_mode import CRITIC_DEFAULT, is_winprob
from agents.training.eval_callback import PerOpponentEvalCallback, ForensicQuota
from agents.training.graceful_restart_callback import GracefulRestartCallback
from agents.training.metrics_exporter_callback import MetricsExporterCallback
from agents.training.signal_callback import SignalMetricsCallback
from agents.training.selfplay_callback import SelfPlayCallback
from agents.training.lever_supply import starve_cycles_for
from main.train.constants import checkpoint_interval_env_steps
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



def _value_sidecar_on(args) -> bool:
    """Is the training-side value sidecar on for this run? (`gen3_value_sidecar_v1`.)

    `auto` — the default — means ON under `--critic winprob` and OFF otherwise, and the asymmetry
    is deliberate rather than a convenience. Under `winprob` the buffer's `values` ARE `P(win|s)`
    and `win_target` is the objective's own label, so a row is a calibration measurement of the
    thing being optimised. Under `shaped` the value is a PopArt-normalised shaped return whose
    scale moves over the run and whose target is not an outcome at all; the same file would carry a
    column named `v` in a different currency every rollout, which is the four-currencies confusion
    the training leaf warns about. `on` overrides for a shaped run that wants the raw pairs anyway
    — the header records the mode either way, so a consumer is never guessing.
    """
    mode = getattr(args, "value_sidecar", "auto")
    if mode == "off":
        return False
    if mode == "on":
        return True
    return is_winprob(getattr(args, "critic", None) or CRITIC_DEFAULT)


def build_callbacks(*, args, model_dir, annealing_mode, _pool,
                    _fixed_opponents, _bot_weight_vec, OPPONENT_CLASSES,
                    _specialist_team_str, _promote_threshold,
                    _heuristic_floor, _sp_start_wr, _sp_full_wr) -> CallbackBundle:
    """Build every `learn()`-time callback this run's flags ask for."""
    # --- Callback Setup (Shared) ---
    # Periodic checkpoints land in <run>/checkpoints/ (SB3 makedirs it); the callback
    # keeps latest.txt + metadata.json at the run root (derived from save_path).
    #
    # 🚨 THE CADENCE IS A TOTAL-ENV-STEP BOUNDARY, never a call count: a save lands at the first
    # callback call whose `num_timesteps` reaches the next multiple of the interval (`run_io.
    # _TrackingCheckpointCallback`, `constants.checkpoint_due`), so it means the same thing at any
    # `--n-envs` and under every collector — sync, Rust, `--async-rollout` waves. Counting calls
    # cost twice: 50 000 calls read as "50k steps" at N = 48 (2.4M env steps), and was ~102M env
    # steps at N = 2048 (F-SZ-3). The interval lives in `main.train.constants` because phase 1
    # cannot import phase 4. Unset `--checkpoint-every-steps` = DEFAULT_CHECKPOINT_EVERY_ENV_STEPS.
    checkpoint_callback = _TrackingCheckpointCallback(
        interval_env_steps=checkpoint_interval_env_steps(
            getattr(args, "checkpoint_every_steps", None)),
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
    # REWARD TERM EXPORT (gen3_reward_term_export_v1): the `reward/` group — every ACTIVE reward
    # term's per-decision mean and its |·|-weighted share of the reward stream's movement, pulled
    # from the env workers by `env_method` once per rollout. ALWAYS ON and flagless for the same
    # reason as the signal group: no battles, no env state, one small dict per worker per rollout.
    # The `{term -> class}` map is derived from THIS run's `reward_class_composition` census, so
    # the exported grouping and the startup composition line read one declaration.
    from agents.training.reward_manager import (
        RewardConfig as _RewardConfig, reward_class_composition as _rcc)
    from agents.training.reward_term_callback import RewardTermMetricsCallback
    from agents.training.reward_term_stats import term_class_map as _tcm
    reward_term_callback = RewardTermMetricsCallback(
        term_class=_tcm(_rcc(_RewardConfig.from_args(args))))
    callbacks = [checkpoint_callback, lr_callback, MetricsExporterCallback(), _HparamLogCallback(args.ent_coef), DoseLogCallback(), graceful_restart_callback, signal_callback, reward_term_callback]
    # ADAPTIVE BATCH (gen3_adaptive_batch_v1): the second controller — it holds a gradient-noise-scale
    # ratio near a target by moving `--grad-accum-steps` K, the one batch lever with no shape change
    # and no memory cost. Registered only when the flag is on, so an `off` run adds no callback and
    # records no series (byte-identical). K PERSISTS through the EXISTING checkpointer: `_model_hparams`
    # already writes `grad_accum_steps` off the model attribute this callback owns, so a restart reads
    # it back from the same sidecar `handoff_lr` rides in — no new key, no checkpoint-path edit.
    if getattr(args, "adaptive_batch", "off") != "off":
        from agents.training.adaptive_batch_callback import (
            AdaptiveBatchCallback, AdaptiveBatchController,
        )
        _resumed_accum: int | None = None
        if args.model and os.path.exists(args.model):
            try:
                _k = read_checkpoint_metadata(args.model).get("grad_accum_steps")
                if isinstance(_k, int) and _k >= 1:
                    _resumed_accum = int(_k)
            except Exception as e:
                print(f"[AdaptiveBatch] WARNING: failed to read grad_accum_steps from "
                      f"{args.model}: {e} — starting from --grad-accum-steps instead.")
        callbacks.append(AdaptiveBatchCallback(
            AdaptiveBatchController(
                mode=args.adaptive_batch,
                target=args.adaptive_batch_target,
                band=args.adaptive_batch_band,
                min_accum=args.adaptive_batch_min_accum,
                max_accum=args.adaptive_batch_max_accum,
                every=args.adaptive_batch_every,
            ),
            resume_accum=_resumed_accum,
        ))
    # RANK TRIPWIRE (gen3_distill_target_gate_v1):
    # watchdog over the EXISTING rank/policy_pr probe — EMA vs the run's own early baseline, with
    # a persistence rule. Default "warn" (no fold runs blind again); pure diagnostic bookkeeping —
    # no loss, no grad, no forward — except that "abort" stops learn() cleanly on a confirmed
    # collapse. "off" registers nothing.
    if getattr(args, "rank_tripwire", "warn") != "off":
        from agents.training.rank_tripwire import RankTripwireCallback
        callbacks.append(RankTripwireCallback(mode=args.rank_tripwire,
                                              drop=args.rank_tripwire_drop))
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
    # THE TRAINING-SIDE VALUE SIDECAR (gen3_value_sidecar_v1). The Rust COLLECTOR fills `win_target` /
    # `win_mask` itself (complete games: every row its own outcome; the window fill:
    # `win_prob_callback.backfill_terminal_labels`) BEFORE `on_rollout_end`, so the sidecar reads the
    # real labels; an all-zero mask is REPORTED rather than written, because a plausible wrong number
    # is worse than a gap. (The `WinProbLabelCallback` this used to be ordered after was the Python
    # collect's per-step capture; it was deleted with that core — deletion pass U3.)
    if _value_sidecar_on(args):
        from agents.training.value_sidecar import ValueSidecarCallback
        callbacks.append(ValueSidecarCallback(
            model_dir, fraction=args.value_sidecar_fraction,
            seed=args.value_sidecar_seed, critic_mode=(args.critic or CRITIC_DEFAULT)))
    # PER-TEAM WIN-RATE TRACKING (default ON): instrumentation only — sparse TB summaries + a
    # restart-safe <run>/team_win_rates.json full table.
    if getattr(args, "team_wr_tracking", True):
        from agents.training.team_winrate_callback import TeamWinRateCallback
        callbacks.append(TeamWinRateCallback(run_dir=model_dir))
    eval_callback = None
    # A --debug smoke run skips ALL eval by default — the periodic eval callback below — so it
    # needs no eval opponents / Showdown eval connection and stays light on CPU. --debug-eval
    # opts back in. Real (non-debug) runs are unaffected (always True).
    _run_eval = (not args.debug) or args.debug_eval
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
            eval_shard_games=args.eval_shard_games,
            forensic_quota=ForensicQuota(win=args.forensic_win_quota,
                                         loss=args.forensic_loss_quota,
                                         draw=args.forensic_draw_quota),
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
            env_core=getattr(args, "env_core", "rust"),
            # gen3_supply_guard_v2: the pool and PFSP supply floors (lever_supply.LEVERS).
            pool_starve_cycles=starve_cycles_for(args, "self_play_pool"),
            pfsp_starve_cycles=starve_cycles_for(args, "pfsp"),
            # T17 mirrored team pairs (resolved: argv, else the run's recorded regime, else OFF).
            eval_mirrored_pairs=bool(getattr(args, "eval_mirrored_pairs", False)),
            # T6 SPRT promotion (resolved: argv, else the run's recorded regime, else OFF).
            promotion_sprt=bool(getattr(args, "promotion_sprt", False)),
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
            best_model_save_path=os.path.join(model_dir, "best_model"),
            eval_shard_games=args.eval_shard_games,
            forensic_quota=ForensicQuota(win=args.forensic_win_quota,
                                         loss=args.forensic_loss_quota,
                                         draw=args.forensic_draw_quota),
            resume_eval_metadata=_resume_meta,
            keep_eval_snapshots=args.keep_eval_snapshots,
            keep_eval_trace_steps=args.keep_eval_trace_steps,
            keep_stalls=args.keep_stalls,
            keep_crashes=args.keep_crashes,
            fixed_opponents=_fixed_opponents,
            # --trainee-team pin → eval measures the trainee ON ITS OWN TEAM (None = default pool).
            trainee_team_str=_specialist_team_str,
            env_core=getattr(args, "env_core", "rust"),
            # T17 mirrored team pairs (resolved: argv, else the run's recorded regime, else OFF).
            eval_mirrored_pairs=bool(getattr(args, "eval_mirrored_pairs", False)),
        )
        callbacks.append(eval_callback)

    return CallbackBundle(
        callbacks=callbacks, eval_callback=eval_callback, lr_callback=lr_callback,
        adaptive_ppo_callback=adaptive_ppo_callback,
        graceful_restart_callback=graceful_restart_callback,
        effective_max_lr=_effective_max_lr)
