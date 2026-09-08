# `scripts/ops/` — the OPERATIONAL instruments for a live training run

The watchers and reads `designs/ops/TRAINING_RUN_SOP.md` names. They were built by the Training
Run session over 2026-08/09 inside a session-scoped temporary directory, where every one of them
had an arm name, a `/home/...` model directory and an interpreter path written in as literals,
and where they would have vanished with the session that wrote them. Promoted 2026-09-07.

**Two halves, and the split is the point.** The shell scripts here are OS processes: they run
under `nohup`, survive a dead Claude session, and are the record of an unattended period. The
Python instruments live in [`src/main/ops/`](../../src/main/ops/) so they are importable and
testable — a shell script cannot have a unit test on a synthetic events file, and the TB reader
needed one.

**Every one of them refuses rather than defaults.** No argument does not mean "the usual run";
a missing scalar is not a zero; an absent archive is not an empty one. `src/main/ops/ops_scripts_test.py`
holds that line for the shell layer (`bash -n`, `--help`, bare-invocation refusal, no `/home` literal).

---

## The shell layer

| script | what it measures / does |
|---|---|
| [`watch_run.sh`](watch_run.sh) | **SOP §2 layer 1.** Polls a run's step progress, bounds the idle gap (35-min wedge limit), matches FAILURE WORDS (`OutOfMemory`, `FATAL`, `Traceback`), checks the arm-INVALIDATING `--sync-to-main` line in the launcher log, and appends one status line per tick carrying MARGINAL fps from checkpoint mtimes. Survives a dead session. |
| [`marginal_fps.sh`](marginal_fps.sh) | **The standard throughput meter** (owner, 2026-08-23). `(steps_N − steps_{N−1}) / (mtime_N − mtime_{N−1})` per consecutive checkpoint pair — never SB3's cumulative `time/fps`, never the integer-`fps` derivative. Prints `train/selfplay_fraction` at every artifact beside it, because an fps figure on a self-play arm is a function of it. |
| [`famine_read.sh`](famine_read.sh) | **The deciding famine read.** Runs `main.critic_gate` with the parent and famine comparator given BY REGISTRY NAME plus the control runs, then the noise-aware kill-bar table. Prints the ladder-backfill command rather than running it (a backfill concurrent with the run's own updater reports the previous node's verdict). |
| [`restart_read.sh`](restart_read.sh) | **The registered first-restart read.** The vf_coef statistic and its verdict, `main.sidecar_audit` (the pin must be unchanged across the restart), the restart evidence, then `killbar` · `vf_framings` · `restart_startup`. |
| [`_common.sh`](_common.sh) | Sourced, not executed. Repo root, MAIN checkout, `models/` (mirroring `utils.paths.main_models_dir()`, `$GEN3AI_MODELS_DIR` authoritative), run resolution and the interpreter. |

## The Python layer — `src/main/ops/`

| module | what it measures |
|---|---|
| `tb_read` | A run's registered scalars from the **TENSORBOARD EVENTS**, not the child log's rendered table. Last-N series, median, min/max, the vf_coef verdict and the calibration-cost ratio. A missing tag WARNS and exits non-zero. |
| `killbar` | The kill bars as an AND-gate, with clause 2 a FIXED EFFECT SIZE (+3.0 turns) evaluated only INSIDE ONE OPPONENT REGIME, against a reference frozen once per regime. |
| `vf_framings` | `grad/value_policy_logratio` under all three framings with the REGISTERED one named, per-1M buckets tagged by regime, and the ratio's COMPONENTS so a walk in the denominator cannot look like the critic sharpening. |
| `restart_startup` | A restart's startup cost: the TB wall-time gap across the boundary minus the steady cadence, against a pre-registered reading and a baseline file. |
| `plateau_signal` | The two-clause ladder plateau REPORT trigger (second consecutive below-predecessor add AND pooled h2h below 0.50 with the Wilson interval excluding it). Never a kill. |
| `g7_report` | The G7 rows QUOTED from `main.critic_gate`, plus the arm's own re-referenced `ep_bots` bar. "untouched" and "clean" are not available as outputs. |
| `stall_exhibit` | `ep_len` and `draw_rate` per 1M bucket on the SAME rows — the table that separates a stall from a competence sawtooth. Every direction word derived, never a literal. |
| `calib_trend` | G1 resolution and G4 skill per checkpoint, BOT and POOL kept apart, parsed from a `critic_gate` text report. |
| `perbot_r` | Pearson `r(base, skill)` across opponents with a battle-clustered bootstrap — the cap account predicts a NEGATIVE sign. |
| `perbot_rank` | The same claim as an ORDERING: Spearman rho, bootstrap clustered by OPPONENT. |
| `negskill_null` | The null for a count of negative-skill cells: a perfectly calibrated critic with the head's OWN forecasts scores negatives by chance at these n. |
| `run_ref` | Shared: a run NAME or DIRECTORY -> the directory, refusing when there is no archive. |

Every module is `python -m main.ops.<name>`; every one prints its contract on `--help`.

## Two carried-over notes

- **`killbar`'s frozen references now live beside the RUN** (`<run>/ops_frozen_refs.json`), not
  beside the module — a repo-level file lets two arms overwrite each other's, and the contract is
  that a frozen entry is never rewritten. They are written only under an explicit `--freeze`.
  The reference the Training Run session froze for `ai_v12_02_winprob_critic` is
  `selfplay_0.90 = 32.61095210484096`, n=21, window `4,000,032..6,000,032`, frozen 2026-09-06 —
  transcribe it rather than re-freezing if that arm's bar is ever read again.
- **`restart_read.sh` step 1 reads the CHILD LOG** while `tb_read` exists precisely because that
  source is a rendering over a ring buffer. It is left as it was — changing a registered reading
  is not this promotion's call — and the discrepancy is recorded in the script's header and in
  the ledger entry for 2026-09-07.
