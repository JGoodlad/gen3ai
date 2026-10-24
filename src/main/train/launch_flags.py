"""EVERY CLI FLAG, CLASSED FOR A `--model` LAUNCH — "an argv is not a config" (root CLAUDE.md), as data.

**THE CONTRACT.** With `--model`, every flag the argv does not TYPE is carried over from the run that wrote the
checkpoint (the run itself on a restart; the PARENT on a fork). Each trainer flag (`main.train.parser`) and each
launcher flag (`main.launcher.run.build_launcher_parser`) is in EXACTLY ONE of these classes, and
`launch_flags_test.py` fails when a flag is in none or in two, so a new flag cannot fall through:

  1. a **recipe row** (`recipe_surface.ROWS`, `FORK_ROWS`): the production recipe; routes in `recipe_surface`;
  2. a **recorded field** (`ModelVersion`, a tri-state flag): `config.resolve_config`'s `_resolve` sweep inherits it
     from the checkpoint's `model_config.json` (a fork: the parent's); `FIELD_SPECIAL` names the two that
     `resolve_eval_sentinel_regime` resolves instead;
  3. a **RECORDED row** (`RECORDED_ROWS`, below): a run-defining flag that is NOT in the production recipe — the K9(b)
     mode, the save and eval cadence, retention, telemetry, the self-play schedule: restored from the source run's
     `metadata.json:cli_args` on a restart AND a fork;
  4. a **RESTART-ONLY row** (`RESTART_ONLY_ROWS`): the run's identity and its matchup levers, restored on a SAME-RUN
     restart and NEVER on a fork — a fork is a new experiment that declares its own seed and matchup (the
     `--team-mix` rule, `matchup_setup.inherit_team_mix`, which keeps its own route: `OWN_ROUTE`);
  5. **PER-LAUNCH** (`PER_LAUNCH`): deliberately not carried, each with its reason.

The resolver is `recipe_surface.inherit_recorded_recipe` (rows 1, 3, 4); the launcher passes what it restored to the
child TYPED (`materialize_recorded_recipe`), because a child pinned to an older commit resolves an untyped flag to its
own parser default. A recorded flag the source run's `cli_args` lacks predates the flag and stays untyped (its value
resolves to the parser's); `cli_args` records the LATEST process's resolved namespace, so a recorded value is what the
run was last using.

`designs/launcher/flags.md` renders this module; `launch_flags_test.py` pins the classes against the live parsers.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, NamedTuple, Tuple

from main.train.recipe_surface import RecipeRow


def _row(dest: str, note: str) -> RecipeRow:
    """A RECORDED / RESTART-ONLY row: `unset` is unused (these rows are never compared with the production mirror)."""
    return RecipeRow(dest, "--" + dest.replace("_", "-"), None, note)


#: Run-defining flags outside the production recipe, carried by a restart AND a fork.
RECORDED_ROWS: Tuple[RecipeRow, ...] = (
    # --- safety: a mode or a budget that decides when the run STOPS or what it REFUSES
    _row("behaviour_check", "K9(b) mode: fatal / warn / off (the 2026-10-24 defect: a resume ran FATAL under a warn arm)"),
    _row("supply_starve_cycles", "the in-flight supply floors; key=0 disables a lever's FATAL_SUPPLY"),
    _row("rust_env_refusal_budget", "quarantined battles before the batch fails"),
    _row("rust_env_respawn_budget", "process-front-end child deaths before FATAL"),
    # --- cadence: when the run saves and evaluates (a different cadence is a different record)
    _row("checkpoint_every_steps", "env steps between periodic checkpoints (the 2026-10-24 defect: 2.4M planned for 1M)"),
    _row("eval_freq", "env steps between eval cycles"),
    _row("eval_games", "games per opponent per eval cycle"),
    _row("eval_shard_games", "games per eval work-steal shard"),
    _row("n_sentinels", "pool snapshots eval'd as sentinels each cycle"),
    _row("snapshot_ladder_games", "games per pair of the per-promotion snapshot ladder"),
    # --- retention: what stays on disk (a dropped cap fills the disk)
    _row("keep_eval_snapshots", "eval weight snapshots retained"),
    _row("keep_eval_trace_steps", "eval trace step dirs retained (0 = all)"),
    _row("keep_stalls", "stall replays retained"),
    _row("keep_crashes", "launcher crash logs retained"),
    _row("forensic_win_quota", "forensic traces kept per opponent per cycle: wins"),
    _row("forensic_loss_quota", "forensic traces kept per opponent per cycle: losses"),
    _row("forensic_draw_quota", "forensic traces kept per opponent per cycle: draws"),
    # --- telemetry: the series a run keeps logging
    _row("value_sidecar", "the critic-vs-its-own-target sidecar"),
    _row("value_sidecar_fraction", "share of rollout states the sidecar samples"),
    _row("value_sidecar_seed", "the sidecar sampler's seed"),
    _row("team_wr_tracking", "the per-team win-rate table"),
    # --- learner: partners of recipe rows
    _row("anneal_min_lr", "the cosine floor; `anneal_start_needs_min_lr` refuses a restored `--anneal-lr-start-steps` without it"),
    _row("adaptive_batch", "the gradient-noise-scale loop on K"),
    _row("adaptive_batch_target", "the loop's ratio target"),
    _row("adaptive_batch_band", "the loop's no-op band"),
    _row("adaptive_batch_min_accum", "the loop's lower clamp on K"),
    _row("adaptive_batch_max_accum", "the loop's upper clamp on K"),
    _row("adaptive_batch_every", "rollouts between K moves"),
    _row("opp_belief_moves_weight", "moves BCE weight inside the belief aux term"),
    # --- the self-play schedule (the recipe's `--self-play` ROW is the on/off; these shape the ramp)
    _row("self_play_temp", "sampling temperature of self-play training opponents"),
    _row("self_play_start_wr", "win_rate_vs_bots at which self-play ramps in"),
    _row("self_play_full_wr", "win_rate_vs_bots at which self-play reaches its floor"),
    _row("heuristic_floor", "minimum share of episodes vs real bots under saturated self-play"),
    _row("pfsp_scale", "PFSP hardness weighting of pool sampling"),
    _row("pool_spread", "pool retention: temporally diverse ladder"),
    _row("team_block_episodes", "consecutive episodes a drawn trainee team is held"),
)

#: The run's identity and matchup levers: a same-run restart restores them, a FORK never does.
RESTART_ONLY_ROWS: Tuple[RecipeRow, ...] = (
    _row("seed", "the run's replication identity (`rb_x35_*_s3504`): a manual resume at the default 42 would also correlate sibling arms"),
    _row("trainee_team", "specialist pin: the one trainee team"),
    _row("trainee_teams", "specialist pin: the fixed trainee team set"),
    _row("bot_weights", "bias of the heuristic-opponent pick"),
    _row("stable_opponents", "foreign models used as fixed opponents"),
    _row("stable_opponent_temp", "play temperature of the stable opponents"),
    _row("stable_opponent_mastered_wr", "win rate at which a stable opponent counts as mastered"),
    _row("stable_opponent_selfplay_share", "share of self-play episodes spent vs stable opponents"),
    _row("stable_opponent_pfsp", "dynamic stable-opponent selection"),
)

#: Every row `inherit_recorded_recipe` restores beyond `recipe_surface.ROWS`, in report order.
INHERITED_ROWS: Tuple[RecipeRow, ...] = RECORDED_ROWS + RESTART_ONLY_ROWS
RECORDED_DESTS: FrozenSet[str] = frozenset(r.dest for r in RECORDED_ROWS)
RESTART_ONLY_DESTS: FrozenSet[str] = frozenset(r.dest for r in RESTART_ONLY_ROWS)
INHERITED_DESTS: FrozenSet[str] = RECORDED_DESTS | RESTART_ONLY_DESTS

#: A flag with its OWN inheritance route (not `inherit_recorded_recipe`): dest -> where.
OWN_ROUTE: Dict[str, str] = {
    "team_mix": "matchup_setup.inherit_team_mix: a same-run restart restores it, a fork never does",
    "team_mix_sides": "matchup_setup.inherit_team_mix (with --team-mix)",
}

#: Tri-state `ModelVersion` flags that `config.resolve_eval_sentinel_regime` resolves (not a plain `_resolve` line).
FIELD_SPECIAL: Dict[str, str] = {
    "promote_threshold": "config.resolve_eval_sentinel_regime: re-derived from a TYPED regime, else inherited",
    "eval_sentinel_greedy": "config.resolve_eval_sentinel_regime: inherited from the checkpoint's config",
}


class PerLaunch(NamedTuple):
    group: str
    reason: str


_ID, _PROC, _FRESH, _CONSENT, _FORK, _HOST, _SESSION = (
    "identity", "process", "fresh-only", "consent", "fork-act", "host", "launcher-session")

#: THE ONE TABLE of flags a `--model` launch deliberately does NOT carry over: dest -> (group, reason).
PER_LAUNCH: Dict[str, PerLaunch] = {
    # --- this launch's identity and dose
    "model": PerLaunch(_ID, "the checkpoint being launched IS the launch"),
    "run_dir": PerLaunch(_ID, "set by the launcher from where the checkpoint lives (it decides restart vs fork)"),
    "run_name": PerLaunch(_ID, "naming a run makes the launch a FORK (`resolve_launch_run_dir`); inheriting it would turn a resume into one"),
    "steps": PerLaunch(_ID, "this launch's target: the dose being asked for now"),
    # --- the process, not the run
    "debug": PerLaunch(_PROC, "the smoke shape; inherited it would turn a resume into a smoke"),
    "debug_eval": PerLaunch(_PROC, "a smoke-run eval switch"),
    "device": PerLaunch(_PROC, "where THIS process runs (the host decides)"),
    "log_level": PerLaunch(_PROC, "this process's verbosity"),
    # --- fresh-only sugar: refused beside --model, or desugared into fields that DO inherit
    "arch": PerLaunch(_FRESH, "fresh-only umbrella, refused beside --model (`fresh_only_flags`); its values are recorded fields"),
    "unified_moves": PerLaunch(_FRESH, "umbrella sugar, deliberately not desugared on a --model launch (`desugar_umbrella_flags`); its components are recorded fields"),
    "unified_damage": PerLaunch(_FRESH, "umbrella sugar (see --unified-moves)"),
    "damage_matrices": PerLaunch(_FRESH, "umbrella sugar for two recorded fields (`damage_matrices_outgoing` / `_incoming`)"),
    # --- consents: one launch's decision, never a standing one
    "allow_nonproduction_arch": PerLaunch(_CONSENT, "consent for a FRESH run; a resume has no production comparison to consent to"),
    "allow_nonproduction_recipe": PerLaunch(_CONSENT, "consent for a FRESH run (see --allow-nonproduction-arch)"),
    "allow_desktop_gpu": PerLaunch(_CONSENT, "consent for THIS host's GPU state, recorded per launch"),
    "allow_low_disk": PerLaunch(_CONSENT, "consent for THIS host's free disk, recorded per launch"),
    "allow_belief_grad_mode_change": PerLaunch(_CONSENT, "consent to one intentional migration on this resume"),
    "allow_inherited_fork_lr": PerLaunch(_CONSENT, "consent to one fork inheriting a frozen parent's LR"),
    "allow_empty_pool": PerLaunch(_CONSENT, "consent to one fork running self-play on an empty pool"),
    "allow_untaught_teacher": PerLaunch(_CONSENT, "research override for one specialist launch"),
    "diagnostic_fork_consent": PerLaunch(_CONSENT, "consent for one diagnostic fork (`diagnostic_fork`)"),
    # --- acts of forking: done once, at fork time
    "fork_lr": PerLaunch(_FORK, "a one-time pin at the moment of forking; a restart must not re-apply it (`fork_lr.resolve_fork_lr`)"),
    "fork_lr_freeze": PerLaunch(_FORK, "persists on its own: re-read from `metadata.json` `dose.fork_lr_pin` on every restart"),
    "grow": PerLaunch(_FORK, "a fork's declared growth; a fork of a grown fork must not grow again (`grow_fork.decide`)"),
    "fork_pool_seed": PerLaunch(_FORK, "seeds a fork's empty self-play pool once, at fork time"),
    "tb_inherit": PerLaunch(_FORK, "copies the parent's TensorBoard curve once, at fork time"),
    "exploiter": PerLaunch(_FORK, "the flag IS the fork signal (`resolve_launch_run_dir`); an exploiter run is relaunched with its own argv"),
    "exploiter_keep_bots": PerLaunch(_FORK, "exploiter-mode companion (see --exploiter)"),
    "exploiter_bot_fraction": PerLaunch(_FORK, "exploiter-mode companion (see --exploiter)"),
    # --- host-bound performance: decided per start from the device and box; bit-identical or parity-gated
    "compile_trainer": PerLaunch(_HOST, "auto from the device at every start (cuda only; a typed value on cpu is refused)"),
    "learner_cudagraphs": PerLaunch(_HOST, "runtime perf lever, cuda only, parity-gated; not recorded in model_config.json"),
    "device_batch": PerLaunch(_HOST, "how micro-batches reach the card; bit-identical serving, memory and time only"),
    "grad_checkpointing": PerLaunch(_HOST, "bit-exact memory-for-time trade, sized to this card's VRAM"),
    "t2_backend": PerLaunch(_HOST, "inference backend: graph on cuda, eager on cpu"),
    "t2_lead_trainee": PerLaunch(_HOST, "T2 serving shape; decisions equal at fp32 tolerance (gated at startup)"),
    "t2_grouped_opponents": PerLaunch(_HOST, "T2 serving shape; decisions equal at fp32 tolerance (gated at startup)"),
    "t2_opponent_bucket_cap": PerLaunch(_HOST, "T2 capture size, sized to this card's memory"),
    "rust_env_front": PerLaunch(_HOST, "process topology of the env core"),
    "rust_env_profile": PerLaunch(_HOST, "which build of this checkout's env core loads"),
    "rust_env_threads": PerLaunch(_HOST, "worker threads, sized to this box's cores"),
    "rust_eval_envs": PerLaunch(_HOST, "eval-core envs, sized to this box"),
    # --- the launcher's own session (stripped before the child; `_strip_launcher_args`)
    "restart_interval_hours": PerLaunch(_SESSION, "this launcher session's restart schedule (owner default 6 h)"),
    "restart_grace_minutes": PerLaunch(_SESSION, "this session's force-kill window"),
    "max_crash_restarts": PerLaunch(_SESSION, "this session's crash circuit-breaker"),
    "nice": PerLaunch(_SESSION, "this session's scheduling niceness"),
    "no_pin": PerLaunch(_SESSION, "this session's pin policy"),
    "dry_run": PerLaunch(_SESSION, "this invocation only"),
    "sync_to_main": PerLaunch(_SESSION, "this session's pin source"),
    "pin_commit": PerLaunch(_SESSION, "this session's pin source (refused on a restart that would move the pin)"),
    "allow_torch_switch": PerLaunch(_SESSION, "consent for one interpreter switch"),
    "gpu_window": PerLaunch(_SESSION, "whether THIS launcher honours GPU-window requests (`designs/launcher/gpu_window.md`)"),
    "help": PerLaunch(_SESSION, "argparse's own help action, not a run setting"),
}


def inherits_on(dest: str, role: str) -> bool:
    """Does a `--model` launch in ``role`` (``"restart"`` / ``"fork"``) restore ``dest`` through
    `recipe_surface.inherit_recorded_recipe`'s record route?"""
    if dest in RECORDED_DESTS:
        return True
    return dest in RESTART_ONLY_DESTS and role == "restart"
