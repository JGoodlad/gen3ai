"""The bot-roster EVAL CALLBACK (``PerOpponentEvalCallback``) + the force-an-eval-now channel.

Split along its seams on 2026-10-01 (TECH_DEBT §1, the file was 1,907 lines against the 2,000-line
bar). The shared machinery BOTH eval callbacks use now lives beside it:

* ``eval_roster``  — the bot roster, schedule constants, player builders;
* ``eval_player``  — ``EvalRLPlayer`` + the forensic capture quota;
* ``eval_launch``  — the cycle manifest, nonce, hung-cycle bound, Rust-core cycle, worker spawn/kill;
* ``eval_collect`` — shard merge, trace selection, snapshot / trace retention, best-model copies, wall;
* ``eval_record``  — per-opponent records, eval blocks, the live ELO, the resume republish.

Every public name of those modules is RE-EXPORTED here, so ``from agents.training.eval_callback import X``
keeps resolving. 🚨 A test that STUBS a name must patch the module that READS it — re-exporting a name
here does NOT route another module's call through this namespace (``src/test_stub_vacuity_gate_test.py``).
"""
import os
import time
import shutil
import threading
import subprocess
from typing import Any  # noqa: F401 — kept for the historical import surface

from agents.training.loop_callbacks import BaseCallback
from poke_env.ps_client import LocalhostServerConfiguration

from agents.model.snapshot import record_eval_results, arch_toggles_from_model
from agents.training.logger_scope import isolated_dump
from agents.training.fixed_opponent_pool import is_external
from utils.contention import describe_contention
from agents.training.artifact_retention import (
    prune_run_artifacts, KEEP_STALLS_DEFAULT, KEEP_CRASHES_DEFAULT,
    KEEP_EVAL_TRACE_STEPS_DEFAULT,
)
from main.launcher.ipc import emit, send_metrics, send_event

# ---- the re-exported surface (see the module docstring) -------------------------------------------
from agents.training.eval_roster import (  # noqa: F401
    BATTLE_FORMAT, _EVAL_CONCURRENCY, _EVAL_SUBPROCESS_CONCURRENCY, EVAL_FREQ_STEPS, EVAL_GAMES,
    EVAL_SHARD_GAMES, _OPPONENT_NAMES, opponent_name, RANDOM_OPPONENT_NAME, _EVAL_OPPONENT_SPECS,
    _EVAL_ROSTER, eval_opponent_names, eval_opponent_class, build_eval_opponents, build_eval_players,
)
from agents.training.eval_player import (  # noqa: F401
    trace_filename_stem, _FORENSIC_LOSS_QUOTA, _FORENSIC_WIN_QUOTA, _FORENSIC_DRAW_QUOTA,
    ForensicQuota, forensic_selection_rule, _rule_for, episode_length_sum, EvalRLPlayer,
)
from agents.training.eval_launch import (  # noqa: F401
    EVAL_MANIFEST_NAME, EVAL_SNAPSHOT_NAME, _read_run_identity, opponent_pins_of,
    write_eval_manifest, _EVAL_CYCLE_TIMEOUT_SEC, eval_cycle_timeout, _b36, eval_run_nonce,
    launch_rust_eval_cycle, kill_eval_workers, spawn_eval_workers, mirrored_eval_games,
    eval_cycle_seed, mirrored_worker_cfg,
)
from agents.training.eval_collect import (  # noqa: F401
    merge_eval_results, prune_eval_traces, prune_eval_snapshots, record_eval_selection,
    persist_eval_snapshot, copy_run_config_to_best_model, write_best_model_sidecar,
    record_cycle_wall,
)
from agents.training.eval_record import (  # noqa: F401
    read_latest_eval_block, replay_last_eval_to_tui, latest_recorded_eval_step, bot_mean,
    record_per_opponent, external_aggregate, external_elo, record_external_elos,
    build_externals_block, build_bot_eval_block, _record_opponent_elos, record_elo,
    mirrored_pairs_block, record_pair_scores,
)

# TD-residual tail metric (#4): the left tail of per-decision critic surprise
# δ = r + γ·V(s') − V(s) (BattleRecorder, the prober's formula) pooled over an eval cycle's
# CAPTURED battles. The headline scalar is the mean of the worst `TD_TAIL_FRAC` fraction (CVaR),
# which isolates the value-cliffs the loss analysis flags (a mean over all turns washes them out;
# a raw min is one-freak-turn noisy). Below `TD_TAIL_MIN_SAMPLES` residuals, report the single
# most-negative one. Sign-meaningful in reward units: more negative = critic more often blindsided,
# so a successful critic-coverage obs change should pull eval/td_resid_tail_mean UP toward 0.
#
# `td_tail` + its constants now live in `eval_sharding.results` (the aggregation layer that owns
# the pooled computation under battle-level work-stealing); re-exported here so the recorder, the
# worker, the prober-parity test, and this module's callers keep their existing import site.
from agents.training.eval_sharding import (  # noqa: E402,F401
    td_tail, TD_TAIL_FRAC, TD_TAIL_MIN_SAMPLES, ShardedEvalPool, PLAN_NAME,
    EvalItem, BOT, FIXED,
)


# ── Force-an-eval-now request channel (the launcher "force eval" button) ──────
# The launcher TUI's `f` key (confirm → SIGUSR2) lets an operator trigger an
# off-cadence eval cycle. The SIGUSR2 handler (train_rl_agent._setup_signal_handlers)
# runs in signal context, so it does the minimum — flip this process-global Event;
# whichever eval callback is active CONSUMES it once on its next `_on_step`. A request
# that arrives while a cycle is already in flight is REJECTED (reported to the launcher
# Events panel), mirroring the normal cadence's skip-while-running rule.
_force_eval_event = threading.Event()


def request_forced_eval() -> None:
    """Ask the active eval callback to launch an eval cycle at its next training step.

    Async-signal-safe (``Event.set`` only briefly takes an internal lock and never
    allocates), so it is callable straight from the SIGUSR2 handler. Idempotent:
    stacking N requests before the next ``_on_step`` collapses to a single launch —
    we never want a queue of forced cycles."""
    _force_eval_event.set()


def consume_forced_eval_request() -> bool:
    """Check-and-clear the forced-eval request; ``True`` iff one was pending.

    The only setter is the signal handler and the only consumer is the single training
    thread, so the check-then-clear is race-free in practice (a signal landing in the
    gap is simply caught on the next step)."""
    if _force_eval_event.is_set():
        _force_eval_event.clear()
        return True
    return False


class _ForcedEvalMixin:
    """Shared 'force an eval cycle now' handling for BOTH eval callbacks.

    Mixed into ``PerOpponentEvalCallback`` and ``SelfPlayCallback`` (which otherwise
    share only ``BaseCallback``) so the force path can't drift between them. Each calls
    ``_maybe_force_eval`` from its ``_on_step`` and ``_restore_last_eval_step`` from its
    ``_init_callback``; the mixin reads the duck-typed eval state both classes expose
    (``_eval_root`` / ``_pending`` / ``_last_eval_step`` / ``_model_dir`` /
    ``_resume_eval_metadata`` / ``num_timesteps``) and drives the subclass's own
    ``_launch_eval``."""

    def _restore_last_eval_step(self) -> None:
        """Restore the eval-cadence anchor from metadata, **CLAMPED to the current step**.

        ``_last_eval_step`` is in-memory and resets to 0 each process, so a resume restores it
        from metadata — otherwise the resumed step is far past a cadence boundary and a fresh 0
        would fire an eval on the very first step. That is right for a launcher RESTART, where the
        recorded eval and the loaded weights are the same point in the run.

        It is WRONG for a **FORK**. ``resume_eval_metadata`` is the SOURCE run's run-level
        ``metadata.json``, whose ``latest_eval.step`` is where that run last evaluated — not the
        step of the older checkpoint being forked from. Fork gen-17's 9.08M checkpoint out of a run
        that went on to 25M and the anchor restores to 24,000,000 against a ``num_timesteps`` of
        9,084,672; the cadence test ``(now // freq) > (anchor // freq)`` is then false until the
        fork itself reaches 26M, so a fork that trains a few million steps runs **ZERO eval
        cycles** — no ``win_rate_vs_*``, no ``eval_results.jsonl`` row, no ELO. An A/B or
        exploiter gate whose verdict is an eval metric silently produces nothing to read.

        Clamping to ``num_timesteps`` restores the intended meaning ("we are considered evaluated
        as of here"), so the next boundary after the fork point fires normally. A restart is
        unaffected (its recorded step is at or behind the current one, so the clamp never bites).
        Same family as ``SelfPlayCallback._warn_if_fork_pool_empty`` — a fork inherits the base's
        weights but none of its run-directory state, and the silent failures live in that gap.
        """
        recorded = latest_recorded_eval_step(
            getattr(self, "_model_dir", None), getattr(self, "_resume_eval_metadata", None))
        # Read the MODEL's counter, not ``BaseCallback.num_timesteps``: the latter is a mirror SB3
        # only syncs inside ``_on_step``, so at ``_init_callback`` time it is still 0 even on a
        # resume at 9M — reading it would clamp every restart to 0 and re-eval on step 1. Fall back
        # to the mirror when no model is attached yet (the attribute is typed ``int``; anything
        # else is not a live model).
        raw = getattr(getattr(self, "model", None), "num_timesteps", None)
        now = int(raw) if isinstance(raw, int) else int(self.num_timesteps)
        if recorded > now:
            # State the FACT, not a cause: this is a FORK off an older checkpoint most of the time,
            # but a crash-restart that rewound past a completed eval reads identically.
            msg = (f"⚠️  [EVAL] eval-cadence anchor is AHEAD of this model — the resumed metadata "
                   f"records an eval at step {recorded:,} but the model is at {now:,} (a FORK off "
                   f"an older checkpoint, or a restart that rewound past a completed eval). "
                   f"Clamping the anchor to {now:,}; unclamped, this run would launch NO eval "
                   f"cycle until step {recorded:,}.")
            try:
                emit(msg)      # prints when standalone, goes to the launcher event stream otherwise
            except Exception:  # noqa: BLE001 — a warning must never break a run
                print(msg, flush=True)
            recorded = now
        self._last_eval_step = recorded

    def _maybe_force_eval(self) -> None:
        """If a forced-eval request is pending, launch an off-cadence cycle now — or
        REJECT it (reported to the launcher Events panel) when one is already running."""
        if not consume_forced_eval_request():
            return
        if getattr(self, "_eval_root", None) is None:
            send_event("⚡ Force-eval ignored — eval is disabled (no run dir)")
            return
        if self._pending is not None:
            send_event(
                "⏳ Force-eval rejected — an eval cycle is already running "
                f"(step {self._pending['step']:,})"
            )
            return
        send_event(f"⚡ Force-eval — launching an eval cycle now (step {self.num_timesteps:,})")
        # Consume the current cadence bucket too, so the regular schedule check below
        # can't double-launch this same step (the next boundary still fires normally).
        self._last_eval_step = self.num_timesteps
        self._launch_eval()



class PerOpponentEvalCallback(_ForcedEvalMixin, BaseCallback):
    """
    Evaluates the trained agent against the full bot roster on a flat schedule
    (`EVAL_FREQ_STEPS` / `EVAL_GAMES`), but does so in a **separate subprocess** with
    a frozen weight snapshot, NON-BLOCKING:

    - On a trigger step, snapshot the live model to disk (`model.save`) and spawn
      `n_workers` eval-worker processes that **work-steal** opponents from a shared
      pool (atomic O_EXCL claim files) — a worker that finishes an opponent grabs
      the next unclaimed one, so uneven per-opponent cost self-balances. Training
      continues immediately; the workers run on their own CPU(s).
    - On later steps, poll; when all workers finish, merge their per-opponent result
      JSONs and record win-rate / reward / ep-len to TensorBoard + the TUI, append
      to metadata.json, and promote the snapshot to best_model if it won.
    - If a trigger fires while the previous cycle is still running, skip it
      (logged) — on CPU an eval can outlast its interval; cadence just goes sparser.
    - On graceful shutdown (training end OR a SIGTERM-driven restart) the callback
      DRAINS the in-flight cycle so its results land before exit, never orphaned.
    - On startup it re-publishes the most recent eval from metadata.json to the TUI,
      so a resumed run shows the last known eval instead of a blank panel.

    The frozen snapshot is what makes parallel eval correct: a worker can't read
    the trainer's mutating in-memory weights, so each cycle evaluates the model
    exactly as it was at the snapshot step. Running in a fresh process also returns
    all eval memory to the OS on exit, avoiding fragmentation in the trainer.
    """

    # Wait for in-flight workers to finish on a graceful shutdown (10 min) — long
    # enough to let a full CPU eval complete; the restart path's grace window
    # (--restart-grace-minutes, 20 min default) comfortably covers it.
    _DRAIN_TIMEOUT_SEC = 600

    def __init__(
        self,
        model_dir: str | None,
        server_config=LocalhostServerConfiguration,
        *,
        best_model_save_path: str | None = None,
        n_workers: int = 3,
        eval_device: str = "cpu",
        eval_concurrency: int = _EVAL_SUBPROCESS_CONCURRENCY,
        eval_shard_games: int = EVAL_SHARD_GAMES,
        eval_games: int | None = None,
        eval_freq: int | None = None,
        showdown_port: int | None = None,
        use_showdown_bridge: bool = False,
        compile_extractor: bool = False,
        bridge_impl: str = "node",
        resume_eval_metadata: str | None = None,
        forensic_quota: "ForensicQuota | dict | None" = None,
        keep_eval_snapshots: int = 10,
        keep_eval_trace_steps: int = KEEP_EVAL_TRACE_STEPS_DEFAULT,
        keep_stalls: int = KEEP_STALLS_DEFAULT,
        keep_crashes: int = KEEP_CRASHES_DEFAULT,
        fixed_opponents: "list | None" = None,
        trainee_team_str: "str | list[str] | None" = None,
        env_core: str = "python",
        eval_mirrored_pairs: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        # MIRRORED TEAM PAIRS (T17, `gen3_mirrored_pairs_v1`, `--eval-mirrored-pairs`, default OFF): every
        # team pairing is played from BOTH sides on ONE battle seed, the per-opponent count is EVEN, and the
        # PAIR is the statistical unit of every interval recorded. A REGIME — recorded per run and per row.
        self._mirrored = bool(eval_mirrored_pairs)
        # M5 Lane H: under --env-core rust the cycle plays IN PROCESS on the declared eval core
        # (`rust_eval.launch`), never on Python workers.
        self._env_core = env_core
        # Per-opponent games per eval cycle (--eval-games; None → the module default EVAL_GAMES).
        # n=100 → ±0.098 (95% CI) per cell; n=200 → ±0.069 — the owner opted into 200 (2026-07-21)
        # for tighter sentinel cells at ~2× eval cost (work-stolen, off the training path).
        self._eval_games = int(eval_games) if eval_games else EVAL_GAMES
        if self._mirrored:
            self._eval_games = mirrored_eval_games(self._eval_games)
        # gen3_eval_freq_flag_v1: the cadence is a KNOB, not a constant. A short gate arm at the
        # 2M default gets only 1-2 cycles inside a 3M budget, which cannot satisfy a >=4-cycle
        # discipline. None => EVAL_FREQ_STEPS, so every pre-existing command is byte-identical.
        self._eval_freq = int(eval_freq) if eval_freq else EVAL_FREQ_STEPS
        self._model_dir = model_dir
        self._server_config = server_config
        # SPECIALIST eval alignment (--trainee-team): the raw Showdown-export team string the
        # TRAINEE is pinned to, threaded into every eval-worker cfg so eval measures the model
        # piloting the team it actually trains — None = the default pool builder. Without this the
        # worker's hardcoded default measured specialists on random teams (pure OOD).
        self._trainee_team_str = trainee_team_str
        # Stable cross-run opponents (FixedOpponentEntry list) — played as an extra ext_ eval
        # matchup each cycle, kept out of win_rate_vs_bots / the ELO fit.
        self._fixed_opponents = list(fixed_opponents or [])
        self.best_model_save_path = best_model_save_path
        self._n_workers = max(1, n_workers)
        self._eval_device = eval_device
        self._eval_concurrency = eval_concurrency
        # Games per work-steal shard unit (battle-level work-stealing); see EVAL_SHARD_GAMES.
        self._eval_shard_games = max(1, eval_shard_games)
        self._forensic_quota = ForensicQuota.coerce(forensic_quota).clamped()
        self._showdown_port = showdown_port
        # Bridge eval: workers play in-process via run_local_battles (no server connection).
        self._use_showdown_bridge = use_showdown_bridge
        # Runtime perf knob, threaded to the eval workers exactly like the bridge flag: they
        # run the SAME frozen extractor on CPU at B=1, so they get the same ~6.5x.
        self._compile_extractor = compile_extractor
        # Which bridge child the workers spawn when use_showdown_bridge: "node" | "rust".
        self._bridge_impl = bridge_impl
        # >0: persist the eval weight snapshot into eval_traces/step_<N>/snapshot.zip
        # (keeping only the N most-recent) so the prober can reload the bit-exact model
        # that produced a cycle's traces. 0 disables (traces still carry the manifest).
        self._keep_eval_snapshots = max(0, keep_eval_snapshots)
        # The trainer writes the forensic traces, so it grooms them: after each cycle
        # keep only the N most-recent eval step dirs (0 = keep all). Older dirs are
        # removed whole. `python -m main.prober.groom` is the manual fallback.
        self._keep_eval_trace_steps = max(0, keep_eval_trace_steps)
        # Bound the per-run debug-artifact dirs each cycle too: keep the N most-recent
        # stalls/*.html + crashes/*.txt (0 = keep all). See artifact_retention.py.
        self._keep_stalls = max(0, keep_stalls)
        self._keep_crashes = max(0, keep_crashes)
        # metadata.json of the checkpoint being resumed (if any) — read at startup so
        # the TUI shows the last eval immediately after a restart.
        self._resume_eval_metadata = resume_eval_metadata
        self._last_eval_step = 0
        self._best_aggregate_win_rate = -1.0
        # The in-flight eval cycle, or None. dict: step, names, procs[], snapshot, run_dir,
        # launched_at.
        self._pending: dict | None = None
        self._eval_root: str | None = None
        # Per-PROCESS account-name nonce (+ a per-cycle counter) so two launcher restarts
        # never reuse Showdown account names — the step-derived tag collided across restarts
        # and hung a worker on a lingering challenge.
        self._eval_run_nonce = eval_run_nonce()
        self._eval_cycle = 0
        # Set by train_rl_agent after signal handlers are wired (kept for parity with
        # the old fail-fast path; the subprocess design logs-and-continues instead).
        self.abort_fn = None
        # The run's `DeferredAbort.safe_point` (wired with abort_fn; P10-A2): the in-process Rust eval
        # cycle calls it every host step, so a stop signal / forced checkpoint is honoured mid-cycle.
        self.safe_point_fn = None

    def _schedule(self) -> tuple[int, int]:
        return self._eval_freq, self._eval_games

    def _init_callback(self) -> None:
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self._model_dir is not None:
            self._eval_root = os.path.join(self._model_dir, ".eval_runs")
            os.makedirs(self._eval_root, exist_ok=True)
        # Resumed run: re-publish the last eval so the TUI panel isn't blank until
        # the next cycle (which can be millions of steps away).
        self._replay_last_eval_to_tui()
        # Restore the last eval step so a resume doesn't re-eval the same checkpoint
        # immediately (it waits for the next cadence boundary instead). CLAMPED to the current
        # step — see _restore_last_eval_step: an unclamped FORK anchor starves eval entirely.
        self._restore_last_eval_step()

    def _on_step(self) -> bool:
        if self._pending is not None:
            now = time.monotonic()
            if self._all_done(self._pending):
                self._collect_pending()
            elif now - self._pending.get("launched_at", now) > eval_cycle_timeout():
                self._abort_pending_cycle()   # hung worker → don't wedge eval forever
        if self.num_timesteps == 0:
            return True
        self._maybe_force_eval()   # launcher "force eval" button (SIGUSR2); rejects if running
        freq, _ = self._schedule()
        if (self.num_timesteps // freq) > (self._last_eval_step // freq):
            self._last_eval_step = self.num_timesteps
            if self._pending is not None:
                print(f"[EVAL] step {self.num_timesteps:,}: previous eval "
                      f"(step {self._pending['step']:,}) still running — skipping this cycle")
            else:
                self._launch_eval()
        return True

    # ------------------------------------------------------------------ launch

    def _launch_eval(self) -> None:
        if self._eval_root is None:
            return  # no model_dir → nowhere to snapshot/collect; eval disabled
        t_launch = time.monotonic()   # gen3_eval_wall_sec_v1: the cycle's wall clock starts HERE
        _, n_games = self._schedule()
        step = self.num_timesteps
        run_dir = os.path.join(self._eval_root, f"step_{step}")
        # Clear any crash-leftover from a prior run at this step (the same step re-evals on resume),
        # so no stale plan/shard/lock files from an aborted cycle are mistaken for this one's.
        shutil.rmtree(run_dir, ignore_errors=True)
        claim_dir = os.path.join(run_dir, "claims")
        os.makedirs(claim_dir, exist_ok=True)

        snapshot_base = os.path.join(run_dir, "snapshot")
        self.model.save(snapshot_base)  # SB3 appends .zip
        snapshot_zip = snapshot_base + ".zip"

        # Full work-steal universe = the bot roster + any stable cross-run opponents (ext_<label>),
        # each an EvalItem the pool splits into shard units. The plan.json (written below) is the
        # single source of truth for the items + shard split — workers and collect read it.
        fixed_cfgs = [e.to_cfg() for e in self._fixed_opponents]
        items = [EvalItem(name, BOT, n_games) for name in eval_opponent_names()]
        items += [EvalItem.fixed_from_cfg(f, n_games) for f in fixed_cfgs]
        names = [it.key for it in items]
        pool = ShardedEvalPool(items, self._eval_shard_games, step=step, mirrored=self._mirrored)
        pool.write_plan(run_dir)
        # Record exactly which model produced this cycle's traces (the prober reads
        # this to reload the right model). snapshot=None now; _persist_snapshot patches
        # it to the retained filename on success when --keep-eval-snapshots is set.
        write_eval_manifest(self._model_dir, step, opponents=names, n_games=n_games,
                            trainee_team_str=self._trainee_team_str,
                            opponent_pins=opponent_pins_of(self._fixed_opponents),
                            quota=self._forensic_quota, mirrored_pairs=self._mirrored)
        # Process-unique account tag (per-process nonce + per-cycle counter), NOT the step:
        # the resume re-eval fires at the same step every restart, so a step tag collided
        # across restarts and hung a worker on a lingering challenge.
        self._eval_cycle += 1
        cycle_tag = f"{self._eval_run_nonce}{_b36(self._eval_cycle, 1)}"
        # Cap by UNITS, not opponents — sharding gives many more units than opponents, so the full
        # worker pool can help drain the tail (the whole point).
        n_workers = max(1, min(self._n_workers, pool.n_units))
        base_cfg = {
            "snapshot": snapshot_zip,
            "port": self._showdown_port,
            "use_showdown_bridge": self._use_showdown_bridge,
            "compile_extractor": self._compile_extractor,
            "bridge_impl": self._bridge_impl,
            "model_dir": self._model_dir,
            "step": step,
            "claim_dir": claim_dir,
            "result_dir": run_dir,         # plan.json lives here; workers write shard__<unit>.json
            "concurrency": self._eval_concurrency,
            "device": self._eval_device,
            "cycle_tag": cycle_tag,
            # Run's discount → the recorder's δ uses the real γ (not the 0.99 fallback), so the
            # live td-residual tail matches the prober's offline _td at the same γ.
            "gamma": float(self.model.gamma),
            # The forensic capture quota crosses the process boundary HERE, so the worker
            # enforces the run's quota rather than the module default.
            "forensic_quota": self._forensic_quota._asdict(),
            # This run's arch toggles → the worker's current_model_version gates sentinel/foreign
            # snapshots against the RUN's real arch (belief-ON / …), not a toggle-OFF default
            # that would FATAL on the run's own belief-ON sentinels.
            "arch_toggles": arch_toggles_from_model(self.model),
            # --trainee-team pin (None = default pool): eval measures the trainee ON ITS OWN TEAM.
            "trainee_team_str": self._trainee_team_str,
            # MIRRORED pairs: every game seeded by the per-GAME rule (a pair shares one battle seed).
            **(mirrored_worker_cfg(self.model, step) if self._mirrored else {}),
        }
        if self._env_core == "rust":
            procs = []
            launch_rust_eval_cycle(self, pool, run_dir, step)
        else:
            procs = spawn_eval_workers(run_dir, base_cfg, n_workers)

        self._pending = {"step": step, "names": names, "procs": procs,
                         "snapshot": snapshot_zip, "run_dir": run_dir, "n_games": n_games,
                         "launched_at": time.monotonic(), "t_launch": t_launch}
        if self._env_core == "rust":
            self._collect_pending()      # the cycle already played (blocking, in process)
            return
        print(f"[EVAL] step {step:,}: spawned {n_workers} work-stealing worker(s) on "
              f"{self._eval_device} ({len(names)} opponents, {pool.n_units} shard units, "
              f"conc {self._eval_concurrency}) — non-blocking")
        send_event(f"🧪 Eval @ {step:,}: started "
                   f"({len(names)} opponents, {pool.n_units} units, {n_workers} worker(s))")

    def _abort_pending_cycle(self) -> None:
        """A cycle overran `_EVAL_CYCLE_TIMEOUT_SEC` → presumed hung. Kill its workers,
        then collect whatever results landed (clears `_pending` so eval can resume)."""
        pending = self._pending
        elapsed = time.monotonic() - pending["launched_at"]
        print(f"⚠️ [EVAL] step {pending['step']:,}: eval cycle hung "
              f"({elapsed:.0f}s > {eval_cycle_timeout():.0f}s) — killing workers, "
              f"collecting partial results. {describe_contention()}")
        send_event(f"⚠️ Eval @ {pending['step']:,}: hung — killed after {elapsed:.0f}s, partial")
        kill_eval_workers(pending["procs"])
        self._collect_pending()

    # ------------------------------------------------------------------ collect

    @staticmethod
    def _all_done(pending: dict) -> bool:
        return all(w["proc"].poll() is not None for w in pending["procs"])

    @isolated_dump   # gen3_eval_dump_isolation_v1: this cycle's dump must not take the last update's train/*
    def _collect_pending(self) -> None:
        pending = self._pending
        self._pending = None
        step = pending["step"]
        run_dir = pending["run_dir"]

        for w in pending["procs"]:
            w["log"].close()
        bad_exits = [w for w in pending["procs"] if w["proc"].returncode not in (0, None)]

        # Battle-level work stealing writes one shard__<unit_id>.json per played shard; the parent
        # pools an opponent's shards back exactly. Missing = a name with ZERO shards (a worker died
        # holding every one of its claims) — log and carry on; partial coverage is warned inside.
        merged, missing = merge_eval_results(run_dir, pending["names"])

        if missing:
            print(f"⚠️ [EVAL] step {step:,}: missing results for {missing} "
                  f"(worker crash mid-opponent?) — see {run_dir}/worker_*.log")
        for w in bad_exits:
            print(f"⚠️ [EVAL] worker exited {w['proc'].returncode}; see {w['log_path']}")

        if not merged["win_rates"]:
            print(f"⚠️ [EVAL] step {step:,}: no results (all workers failed); skipping record")
            send_event(f"⚠️ Eval @ {step:,}: failed (no results)")
            self._cleanup(pending, keep_logs=True)
            return

        self._record(step, merged, pending["n_games"], n_workers=len(pending["procs"]))
        self._maybe_save_best(step, pending, merged["win_rates"])
        # State the trace SELECTION in the cycle's own manifest, BEFORE pruning: a pruned step dir
        # takes its manifest with it, and a cycle whose traces survive must carry its record.
        record_eval_selection(self._model_dir, step, merged,
                                    quota=self._forensic_quota)
        self._persist_snapshot(pending)
        self._prune_eval_traces()   # trainer grooms the traces it writes
        self._prune_run_artifacts()  # …and bounds its stalls/ + crashes/ dirs
        self._cleanup(pending, keep_logs=bool(missing or bad_exits))
        record_cycle_wall(self, pending, merged, tag="EVAL")

    def _record(self, step: int, merged: dict, n_games: int = EVAL_GAMES,
                n_workers: int = 1) -> None:
        win_rates = merged["win_rates"]
        reward_means = merged["reward_means"]
        ep_lens = merged["ep_lens"]
        td_tails = merged.get("td_resid_tails", {})
        # Stable cross-run opponents (ext_<label>) are a SEPARATE yardstick: per-opponent metrics
        # are recorded (the loop below), but they are kept OUT of the bot aggregate / best-model
        # signal / ELO fit so they never move the curriculum (bot_mean already excludes them).
        bot_wr = {k: v for k, v in win_rates.items() if not is_external(k)}
        ext_wr = {k: v for k, v in win_rates.items() if is_external(k)}
        bot_td_tails = {k: v for k, v in td_tails.items() if not is_external(k)}
        aggregate = sum(bot_wr.values()) / len(bot_wr) if bot_wr else 0.0
        bot_rewards = {k: v for k, v in reward_means.items() if not is_external(k)}
        aggregate_reward = sum(bot_rewards.values()) / len(bot_rewards) if bot_rewards else 0.0
        wr_bots = bot_mean(win_rates)
        rew_bots = bot_mean(reward_means)
        eplen_bots = bot_mean(ep_lens)
        wr_external = external_aggregate(ext_wr)
        total_dur = sum(merged["durations_sec"].values())

        tui: dict[str, float] = {}
        # Per-opponent win/reward/ep_len for every opponent incl. stable ones (shared recorder).
        record_per_opponent(self.logger, tui, win_rates, win_rates, reward_means, ep_lens)
        # TD-residual tail is a BOT/sentinel critic-coverage diagnostic — NOT emitted for stable
        # opponents (a display-only yardstick; uniform with the self-play path, which omits it too).
        for name in bot_td_tails:
            self.logger.record(f"eval/td_resid_tail_vs_{name}", bot_td_tails[name])
            tui[f"eval/td_resid_tail_vs_{name}"] = bot_td_tails[name]
        self.logger.record("eval/win_rate_mean", aggregate)
        self.logger.record("eval/win_rate_vs_bots", wr_bots)
        self.logger.record("eval/mean_reward_mean", aggregate_reward)
        self.logger.record("eval/mean_reward_vs_bots", rew_bots)
        self.logger.record("eval/mean_ep_len_vs_bots", eplen_bots)
        if wr_external is not None:
            self.logger.record("eval/win_rate_vs_external", wr_external)
            tui["eval/win_rate_vs_external"] = wr_external
        self.logger.record("eval/duration_sec", total_dur)
        # TD-residual tail headline (#4): mean of the per-opponent tails (a mean-of-CVaRs). Only
        # recorded when captured battles produced residuals — lower/more-negative = critic more
        # often blindsided; the leading indicator for the critic-coverage obs work.
        td_tail_mean = sum(bot_td_tails.values()) / len(bot_td_tails) if bot_td_tails else None
        if td_tail_mean is not None:
            self.logger.record("eval/td_resid_tail_mean", td_tail_mean)
            tui["eval/td_resid_tail_mean"] = td_tail_mean
        # Anchored-BT ELO from the accumulated results (appends this cycle's row first).
        # bot_wr carries every BOT incl. random (a valid anchor); ext_ opponents are kept OUT of the
        # fit (no ladder distortion); no sentinels on the bot-only path.
        bot_counts = {k: merged["counts"][k] for k in bot_wr if k in merged.get("counts", {})}
        # The vs-target record (e.g. the exploiter VERDICT) rides the append-only jsonl too — it
        # used to live only in the overwritten latest_eval block + TensorBoard.
        ext_block = {k: {"win_rate": v, "counts": merged.get("counts", {}).get(k)}
                     for k, v in ext_wr.items()}
        # MIRRORED pairs: the PAIR-level score over the bots (Random excluded, like win_rate_vs_bots) and
        # each opponent's pentanomial on the row — the regime stamp the live ELO fit filters by.
        mp_block = None
        if self._mirrored:
            record_pair_scores(self.logger, tui, merged, [k for k in bot_wr if k != RANDOM_OPPONENT_NAME],
                               "bots")
            mp_block = mirrored_pairs_block(merged, bots=list(bot_wr), externals=list(ext_wr))
        elo_result = record_elo(self._model_dir, step, bot_wr, [], n_games,
                                self.logger, tui, bot_td_tails=bot_td_tails, bot_counts=bot_counts,
                                externals=ext_block or None, mirrored_pairs=mp_block)
        # ELO for each stable opponent (display-only) → fills the eval table's elo column for the
        # ext_ rows: its OWN recorded ELO when available, else a ballpark from the trainee's rating.
        if ext_wr:
            record_external_elos(self.logger, tui, elo_result[0] if elo_result else None, ext_wr,
                                 {e.label: e.source_elo for e in self._fixed_opponents})
        # Record at the SNAPSHOT step so the eval curve aligns to when the model was
        # frozen, not the (later) step at which the worker happened to finish.
        self.logger.dump(step)

        tui.update({
            "eval/win_rate_mean": aggregate, "eval/win_rate_vs_bots": wr_bots,
            "eval/mean_reward_mean": aggregate_reward, "eval/mean_reward_vs_bots": rew_bots,
            "eval/mean_ep_len_vs_bots": eplen_bots, "eval/duration_sec": total_dur,
            # Worker count for the TUI. duration_sec is the SUMMED UNIT TIME, never wall time:
            # that is eval/wall_sec (record_cycle_wall at the end of the collection).
            "eval/n_workers": float(max(1, n_workers)),
            "_step": step,
        })
        send_metrics(tui)

        # _maybe_save_best() updates _best_aggregate_win_rate AFTER this, so the value
        # here is the prior best — surface a new best explicitly rather than printing a
        # stale "best" that the current rate already beats. (-1.0 = no eval yet.)
        prev_best = self._best_aggregate_win_rate
        pct = aggregate * 100
        if prev_best < 0.0:
            summary = f"{pct:.1f}% (first eval)"
        elif aggregate > prev_best:
            summary = f"{pct:.1f}% 🏆 new best (+{(aggregate - prev_best) * 100:.1f}pts)"
        else:
            summary = (f"{pct:.1f}% (best {prev_best * 100:.1f}%, "
                       f"-{(prev_best - aggregate) * 100:.1f}pts)")
        print(f"[EVAL] step {step:,}: aggregate {summary}")
        send_event(f"🧪 Eval @ {step:,}: {summary}")

        if self._model_dir:
            # Bot-only dicts so the block's win_rate_mean / mean_reward_mean agree with each other
            # and with the live TB values (ext_ is recorded separately in `externals` below).
            block = build_bot_eval_block(bot_wr, bot_rewards, ep_lens, bot_td_tails)
            if elo_result:
                block["elo"], block["elo_ci"] = elo_result
            if ext_wr:
                # Stable cross-run opponents — recorded as a separate yardstick (display-only).
                block["externals"] = build_externals_block(ext_wr, win_rates, reward_means, ep_lens)
                if wr_external is not None:  # only the multi-opponent aggregate
                    block["win_rate_vs_external"] = wr_external
            if mp_block is not None:
                block["mirrored_pairs"] = mp_block      # the cycle's regime + per-opponent pentanomials
            record_eval_results(self._model_dir, step, block)

    def _maybe_save_best(self, step: int, pending: dict, win_rates: dict) -> None:
        # Best-model is chosen on the BOTS (+ random), not the stable cross-run opponents — an
        # ext_ yardstick must never drive checkpoint selection.
        bot_wr = {k: v for k, v in win_rates.items() if not is_external(k)}
        if not bot_wr:
            return
        aggregate = sum(bot_wr.values()) / len(bot_wr)
        if aggregate <= self._best_aggregate_win_rate:
            return
        self._best_aggregate_win_rate = aggregate
        if self.best_model_save_path is not None:
            # The frozen snapshot IS the best model — copy it rather than re-saving.
            dst = os.path.join(self.best_model_save_path, "best_model.zip")
            shutil.copy2(pending["snapshot"], dst)
            copy_run_config_to_best_model(self._model_dir, self.best_model_save_path)  # model_config.json
            write_best_model_sidecar(self._model_dir, dst, self.model)                 # best_model.json (+ELO)
            print(f"[EVAL] new best ({aggregate * 100:.1f}%) saved to {dst}")

    def _persist_snapshot(self, pending: dict) -> None:
        persist_eval_snapshot(self._model_dir, pending["step"], pending["snapshot"],
                              self._keep_eval_snapshots)

    def _prune_eval_snapshots(self) -> None:
        prune_eval_snapshots(self._model_dir, self._keep_eval_snapshots)

    def _prune_eval_traces(self) -> None:
        prune_eval_traces(self._model_dir, self._keep_eval_trace_steps)

    def _prune_run_artifacts(self) -> None:
        prune_run_artifacts(self._model_dir, self._keep_stalls, self._keep_crashes)

    def _cleanup(self, pending: dict, keep_logs: bool) -> None:
        # Always drop the (large) transient run-dir snapshot; _persist_snapshot has
        # already copied it into eval_traces/ when retention is on. Keep the run dir
        # only if a worker failed, so its log survives for debugging.
        try:
            if os.path.exists(pending["snapshot"]):
                os.remove(pending["snapshot"])
        except OSError:
            pass
        if not keep_logs:
            shutil.rmtree(pending["run_dir"], ignore_errors=True)

    def _on_training_end(self) -> None:
        self.drain()

    def drain(self, timeout: float | None = None) -> None:
        """Block (up to `timeout` TOTAL seconds) for the in-flight eval cycle, then record it.

        Called on graceful shutdown so an eval in flight is never orphaned and its
        results land in metadata.json + the TUI. Both the normal training end and the
        (self-initiated) scheduled restart wait the full `_DRAIN_TIMEOUT_SEC` (10 min)
        so a CPU eval can finish — the restart's grace window (--restart-grace-minutes,
        20 min) covers it, and the checkpoint is already saved first regardless.

        `timeout` is a total budget across all workers (not per-worker), so several
        hung workers can't stack past the deadline. Idempotent (no-op if nothing pending).
        """
        if self._pending is None:
            return
        budget = self._DRAIN_TIMEOUT_SEC if timeout is None else timeout
        deadline = time.monotonic() + budget
        print(f"[EVAL] graceful shutdown — waiting up to {budget:.0f}s for eval worker(s)...")
        for w in self._pending["procs"]:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                w["proc"].wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                w["proc"].kill()
        self._collect_pending()

    # ------------------------------------------------------------- TUI resume

    def _replay_last_eval_to_tui(self) -> None:
        replay_last_eval_to_tui(self._model_dir, self._resume_eval_metadata)
