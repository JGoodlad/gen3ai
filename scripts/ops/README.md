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
| [`restart_read.sh`](restart_read.sh) | **The registered first-restart read.** The vf_coef statistic and its verdict **from the TB EVENTS** (`main.ops.tb_read`, verdict bar imported from `vf_framings`), with the child log's table demoted to a labelled CROSS-CHECK that prints both medians and their difference and warns above 0.10 log10 without ever deciding; then `main.sidecar_audit` (the pin must be unchanged across the restart), the restart evidence, `killbar` · `vf_framings` · `restart_startup`. |
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
| `critic_read` | **The CRITIC LADDER's registered read, as ONE command.** identity (`cf_audit`, no `--checkpoint`) + G1-G4 (`main.critic_gate`) + the `corr(turn,V) - corr(turn,MC)` contrast + the **CONDITIONING** section, every quantity as **ARM - CONTROL** with a battle-clustered difference of INDEPENDENT bootstraps and the registered DETECTED / WITHIN FLOOR / NOT DETECTED label. `--step N` PINS the cycle (pin it whenever the arm's launcher may still be alive — `--on-live skip-newest` silently drops the newest cycle, and the report now prints WHICH cycle and WHY). Refuses on a missing manifest, an uncollected cycle, npz without `win_probs`, an anchor rate under the label-trust gate, or a draw/timeout share over 25%. 🚨 **QUOTA-MATCHED by default** (2026-09-09): the two cycles' REALIZED per-opponent capture profiles are printed in the header, and where they differ the richer side is subsampled to the poorer side's cap before any FRAME-SENSITIVE conditioning row is read — `--no-quota-match` opts out and then those rows print `UNMATCHED — not a reading` INSTEAD of a label. |
| `conditioning_meters` | Shared: **does the win-prob critic know WHO it is playing and WHOSE TEAM it holds?** The between-opponent SPREAD IDENTITY of `V` against the outcome (`E[V\|opp] = E[y\|opp]` for any calibrated critic, so the ratio's target is **1.0**), turn-1-3 and all-states, noise-corrected AND unclamped; the bias-on-opponent-Elo slope per 100 Elo; the own-team **leave-one-battle-out** win-rate R² of `V` at turn 1 and over all turns; and the turn-1 opponent-CLASS AUC of `V`. Computed on ONE cycle from the **RECORDED** `win_probs` (no model forward), HT-reweighted by that cycle's capture rates, every interval a bootstrap that resamples battles WITHIN their opponent cell. Promoted 2026-09-09 from `winprob_mixture_diagnostic_2026-09-09/` and `winprob_probe_read_2026-09-09/`; those measurement dirs can import it. REFUSES a cycle with no `selection` block and an UNANCHORED ladder refit. |
| `value_sidecar_read` | **The TRAINING-SIDE calibration read** — the critic against its OWN training target (`win_target`), from `<run>/value_sidecar/rows.jsonl`. Mean V vs mean target, Murphy (reliability / resolution / uncertainty) and skill, sliced by turn bucket, opponent class, outcome and 1M step bucket, each with an EPISODE-clustered bootstrap CI. The counterpart to `critic_read`, which reads EVAL battles — neither supersedes the other, and a disagreement is a GENERALISATION finding. Refuses a run with no sidecar, a headerless sidecar (the critic mode, and so whether V is a probability, would be unknown), a `shaped` sidecar without `--allow-shaped`, and a wholly unlabelled one (which would mean a callback-ORDER defect). |
| `critic_readouts` | Shared: the ladder's statistics — the three selection weightings (sampler, capture-rate/rule-17, none), the battle-clustered bootstraps, Murphy with the base-rate cap, the delta label. Promoted from the 75M read's measurement directory. |
| `quota_match` | Shared: **equalises two arms' TRACE FRAMES before a fitted conditioning row is compared between them.** The realized per-opponent capture profile (the manifest's `selection` block CROSS-CHECKED against the battles on disk — the disk wins, and a nominal 5/10/5 lands as **8/12** under shard rounding); the cap plan (elementwise minimum, so a richer CONTROL is the side that gets cut — the replicate-floor read carries the same asymmetry); an **IN-MEMORY** subsample with the capture rates RECOMPUTED per draw so rule 17 holds on the view actually read; and a SEARCHED decoder-matched rung, because `MIN_TEAM_BATTLES` makes the fitted frame a nonlinear function of team diversity. Nothing is copied, symlinked or written. ~8 s per pair. Built for the 2026-09-09 RETRACTION: a decoder's out-of-fold score rises with the frame it was fit on, which HT reweighting does not touch. |
| `critic_read_render` | Shared: `critic_read`'s report — the markdown and the one-line ledger QUOTE. Pure; reads nothing, writes nothing. |
| `run_ref` | Shared: a run NAME or DIRECTORY -> the directory, refusing when there is no archive. |

Every ENTRY POINT is `python -m main.ops.<name>` and prints its contract on `--help`. `run_ref`, `critic_readouts`, `critic_read_render`, `conditioning_meters` and `quota_match` are LIBRARY modules — imported, never invoked — and have no `--help` of their own; their table rows say so.

## Two carried-over notes

- **`killbar`'s frozen references now live beside the RUN** (`<run>/ops_frozen_refs.json`), not
  beside the module — a repo-level file lets two arms overwrite each other's, and the contract is
  that a frozen entry is never rewritten. They are written only under an explicit `--freeze`.
  The reference the Training Run session froze for `ai_v12_02_winprob_critic` is
  `selfplay_0.90 = 32.61095210484096`, n=21, window `4,000,032..6,000,032`, frozen 2026-09-06 —
  transcribe it rather than re-freezing if that arm's bar is ever read again.
- **`restart_read.sh` computes the registered vf_coef statistic ONCE, from the EVENTS** (fixed
  2026-09-07). It used to compute it twice from two sources — step 1 from the child log's rendered
  table, step 5 from the events via `vf_framings` — and said nothing about which to believe. The
  child log is a rendering over a ~1 MiB ring buffer, so its "last 20" is the last 20 rows that
  still FIT; the events carry every rollout with its step, and the SOP's rule is *validate at the
  source*. The child-log read survives only as a **CROSS-CHECK**: it prints both medians and their
  difference and WARNS when they differ by more than **0.10 log10** — one fifth of the narrowest
  verdict band (|med| ≤ 0.5), so a difference at or below it cannot move the call. It never
  produces a verdict, never overrides one, and an empty cross-check (the ring buffer trimmed) says
  nothing about the events read. The verdict bar itself is imported from `vf_framings.verdict`, so
  there is one definition of it in the tree.
