# The road to X26: readiness inventory, the toy controls, the exit check and the GPU schedule

🚨 **ALWAYS-CURRENT (an `endstate/` doc).** A decision or build that differs from this plan updates it in the
same commit, saying what changed and why. It ends with a **Decision record**.

**Status: PLAN 2026-10-15 (orchestrator brief; inventory read at `9ada3604`).** Nothing has been built or launched
from this plan yet. The owner's roadmap (2026-10-09, after the closing test and the perf phase) has three steps
left: **the toy controls, then the exit check, then X26** (the deep run to plateau). This doc covers all three:
- §1 is the inventory: every prerequisite of X26, with its state and evidence.
- §2 is the X26 recipe this plan proposes.
- §3 is the design for the toy controls.
- §4 is the exit check as a checklist.
- §5 is the GPU schedule from tonight to the X26 launch, plus the parallel build units.
- §6 lists the risks and the owner's questions.
- §7 lists the findings.

Units and terms:
- **pp** is a percentage point of head-to-head win rate.
- **h2h** is `main.h2h`, the mirrored head-to-head engine.
- **Tier 1** is the plateau GSPRT (a sequential probability-ratio test) of the newest checkpoint against the one W
  steps back, `main.plateau` (`design_evaluation.md` §8.1).
- **EB** is the bf16 end-state arm of the bf16 A/B, seeds `rb_bf_eb_s2001`–`s2008`.
- **GPU-h** is hours of exclusive use of the local RTX 3080 Ti.

## 0. One screen

- **X26 is BLOCKED by the legacy deletion's U3.** The manifest
  ([`../ops/legacy_deletion_manifest_2026-10-12.md`](../ops/legacy_deletion_manifest_2026-10-12.md)) has D-LD1–D-LD7
  ADOPTED (the orchestrator's default, 2026-10-15); **U0 and U1 have LANDED** (2026-10-15; U1 = config v156, the eight
  op levers flagless). Lane **U3** changes the fresh-seed random stream, so it must land before any X26 seed. The
  chain is U0 → U1 → U3, landed one unit at a time with green gates.
- **The plateau meter is partly built.** Tier 1 is BUILT and run offline (`main.plateau`, `131f3412`). It is
  VALIDATED on a real run (T33, 2026-10-15, on the T-ENT toy), and its cost is measured (§2.6). On the agent cores,
  an X26 near-plateau check costs 1.5–5 h today. The proposal (§2.6) is that X26 ticks at the registered Δ 10M,
  the meter's batches get pipelined, and no in-process eval is built for X26. The cycle monitor (U5), the outside panel and the two-tier decision (U9) are NOT
  STARTED, and U9 waits on the owner's Q8. Every meter component runs OFFLINE over checkpoints, so the deep run
  needs only the stop RULE registered and Tier 1 validated at launch (§2.4, owner Q2).
- **The ride-along heads still attach** to the end state. They have already ridden on 32 end-state runs. Their
  registration is stale (its literal argv fails `checkargs`) and needs an Amendment 2.
- **The toy controls need no build for the first toy.** T-ENT (entity tokens and attention only) launches today
  from flags (§3.2; `checkargs` exit 0, verified for this doc before U1 — since U1 the argv runs PINNED, §3.2). T-MLP (a flat MLP) needs a build of about 1.5
  agent-days.
- **The first GPU job tonight is T-ENT seed 4001**, launched by the Training session after its checkargs /
  dry-run validation (§5.1).
- **The X26 launch estimate is about Mon 10-19 to Tue 10-20.** It is set by the deletion chain and the exit check,
  not by the GPU. The GPU runs the toys and a production scout until then (about 40–55 GPU-h).

## 1. INVENTORY — every prerequisite of X26

State: **DONE** · **PARTIAL** · **IN FLIGHT** · **NOT STARTED** · **DROP?** (proposed to drop or defer past launch).
"Needed at launch" says whether X26 cannot start without the item (**L**), or the item must exist before a check
that X26 reaches later (**later**).

| # | prerequisite | state | evidence | needed at launch? | what is missing |
|---|---|---|---|---|---|
| I1 | **The end state is production** (static tokens, trunk depth 3, X5 tokens, move resolution, recovery and fact levers, no critic threat injection) | DONE | ledger 2026-10-12 BUILT (MAJOR); closing test PASS Δ̂ +0.60 pp, 90 % [−0.72, +1.92] | L | — |
| I2 | **bf16 trunk in `--arch production`** | DONE | `10b3a6b6`; `production_config.json` `recipe.fresh.trunk_precision = "bf16"`; A/B PASS Δ̂ +1.06 pp, 90 % [−1.19, +3.31] | L | — |
| I3 | **Perf phase** (host syncs, bf16, compile cache) | DONE for X26's purposes | ledger 2026-10-10 T25 windows D/E; bf16 `train_ms` 42.1 s vs 54.3 s fp32 | — | T26 (overlap) and CUDA graphs stay future arms with their own A/B |
| I4 | **Torch build: cu126 vs cu128** | cu128 env BUILT, NOT the default; the local cu126-vs-cu128 baseline is PENDING | `designs/ops/cloud_gpu_benchmark.md` L9; the dry-run resolves torch 2.8.0+cu126 | L (as a DECISION) | **Decision (§2): X26 runs on cu126** (`gen3ai_torch28`). The launcher refuses a cross-build resume, and the default switches only at an experiment boundary |
| I5 | **Legacy deletion U0** (the refusal machinery, `RetiredLever` rows, the `pre-legacy-deletion` tag, `--arch static_recovery` deleted) | **SHIPPED 2026-10-15** (D-LD1–D-LD7 adopted as the orchestrator's default; the vacuity rule built and tested with a real toy build) | the manifest's §7 / §8 | L (U3 depends on it) | the owner's D-LD1–D-LD7; the unit itself (~0.5 agent-day) |
| I6 | **Legacy deletion U1** (the op judgments D1–D6) | **LANDED 2026-10-15** (config v156: `--status-facts`, `--ko-ramp`, `--g-ledger`, `--op-reduction`, `--speed-physics`, `--eot-residual`, `--switch-hazard-cost`, `--effective-stats` deleted; adopted behaviour unconditional behind `damage_op`; K9 golden byte-identical) | as I5 | L (U3 depends on it) | — |
| I7 | **Legacy deletion U3** (`--move-resolution off` and the seven per-action blocks; `--value-threat-inject`). **It changes the fresh-init random stream (D-LD7)** | NOT STARTED | manifest §0 D-LD7, §3 | **L** | ~1.5 agent-days; the K9 golden re-recorded once; the ledger line "fresh-init stream changed at `<U3 sha>`" |
| I8 | Legacy deletion U2, U4 (observation levers; the legacy token encoding) | NOT STARTED | as I5 | no, **if** each proves production's fresh init is byte-identical before and after | either lands before the exit check with that proof, or lands after X26 |
| I9 | Legacy deletion U5 (optional Rust progress clock), U6 (close-out) | NOT STARTED | manifest §3 | no (U6 waits for X26's argv, by its own text) | — |
| I10 | **The ride-along heads attach to the end state** | DONE | `checkargs --argv "--arch production … --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all"` exit 0; the launcher dry-run from main "would launch"; B rebased onto the flat pointer (`350fb83b`, refused without it at `ridealong_heads.py:280-310`); RND reads `obs_dim` at build (2845); all 16 bf16 A/B runs and the 8 `rb_es_end2_*` typed these flags (`metadata.json`); `ridealong/disabled` max 0.0 in `rb_bf_eb_s2005`'s TB | L | — |
| I11 | **The ride-along registration** ([`PREREGISTRATION.md`](../research_state/measurements/ridealong_baseline/PREREGISTRATION.md)) | STALE | its literal argv fails `checkargs` (exit 1: `--critic winprob`, `--terminal-indicator`, `--victory-value 1.0`, `--draw-penalty 0`, `--env-core rust` are deleted constants); restart interval 3 h (now 6 h); the K10(a) recipe clause; "FIRST GPU run on the M5 core"; read points 10M / 25M / FINAL; the overhead basis 67.05 s; the checkpoint cadence "50,000 vec-calls × 48 envs"; B on α's support; the 2761-dim memory estimate; `feat`'s drift prediction from fp32 legacy features | L | **Amendment 2** (unit T32), committed before the launch |
| I12 | **Head bit-identity under bf16** (heads ON vs OFF learn bit-identically) | UNVERIFIED | `ridealong_update_test` proves it at fp32 only | L | one test under bf16 autocast (part of T32) |
| I13 | **Plateau meter, Tier 1** (newest vs W-back GSPRT, H0 0.50 / H1 0.52, Δ 10M, W 50M, cap 6,000 pairs) | **VALIDATED on a real run (T33, 2026-10-15)**: 10 checks (25M–70M) on `rb_toy_ent_s4001`, all GAIN, with no CONFLICT or INCONCLUSIVE; status CLIMBING (the run died at 72M on a CompileCanary FATAL, F17) | `src/agents/training/plateau_t1.py`, `src/main/plateau.py` (`131f3412`); the `plateau_t1:rb_toy_ent_s4001:*` requests and decisions in `models/_ledger`; §2.6 | L (validated) | its CPU cost is MEASURED (§2.6): production net 0.62 games/s on the agent cores today, so a near-plateau check takes 87–137 min against a 53-min Δ 5M cadence. **Use the registered Δ 10M for X26 and pipeline the 40-pair batches (T33-2)** |
| I14 | **Plateau meter, the cycle monitor** (U5) | NOT STARTED | no `src/main/cycle_monitor*`; `monitor_row` exists only as a ledger kind | later (before the first Tier-1 FLAT can stop X26) | ~12–16 agent-hours, offline only (TASK_BACKLOG T20) |
| I15 | **Plateau meter, the outside panel + the two-tier decision** (U9) | NOT STARTED; the owner's Q8 is unanswered | `design_league_decisions.md` L110 unchecked; `design_evaluation.md` Decision record 2026-10-03 "Q8 … provisional" | later | Q8 (owner Q2 below); ~8 agent-hours |
| I16 | **U4 scheduler** (the in-window eval lane) | NOT STARTED / PAUSED | `design_evaluation.md` §7; `design_x5_belief_tokens.md` "eval U4 is paused" | **no — DROP? from X26's prerequisites.** X26 is pinned, so its checks run through the offline driver anyway (F-ED-19) | — |
| I17 | **The X26 recipe** (§2) | PARTIAL: production resolves; the run-specific choices are this doc's proposal | dry-run: FRESH, restarts every 6 h, 3 crash restarts | L | the owner's Q1 (seeds) and Q4 (promotion), then Amendment 2 |
| I18 | **The checkpoint cadence for Tier 1** | DECISION | the default `--checkpoint-every-steps` is 2.4M (`hyperparameters.py:79`). Tier 1's node must fall in [grid, grid + 500k), so a 2.4M cadence misses most grid points (`MISSING_NODE`) | L | type `--checkpoint-every-steps 1000000` (as the closing test and the bf16 A/B did) |
| I19 | **The K9(b) tie-share margin** (the behaviour-check's excluded share; ceiling `FP32_EXCLUDED_CEILING` 0.15, single-shot FATAL under the default `--behaviour-check fatal`) | MARGIN OK for the MEAN; the SPIKE risk is unmeasured | §1.1 below | L (as a declared handling) | declare the closing test's Amendment 1 / 2 handling in X26's registration; log the share on EVERY update (TB keeps ~1 in 10) — T32 |
| I20 | **Disk** | OK | `df`: 341 GB free of 1003 GB; an end-state 15M run is 4.2–4.8 GB, of which 3.2–3.3 GB is a fixed compile cache and 55 MB per checkpoint | L (≥ 100 GB free at launch) | no pruner exists for long rb_ runs (none needed: §2.3) |
| I21 | **The plateau battery's harness** (the per-stage VOI + belief-quality dose re-run on X26's checkpoints, `design_hidden_info_binding.md` §12.13.7) | PARTIAL: the world builder, FN beliefs, the dose family and blind scoring are BUILT; the battery exists only as measurement scripts | `t1_common.py` CHECKPOINTS = {end2, prod}; `run_pin.sh` takes only those; no plateau-drawn bank sampler; no snapshot trigger; the R-MATURITY rules are not code | later (first read at ≥ 25M; it reads retained checkpoints offline, so it can land during the run) | T35 (~1.5 agent-days); cost FULL ≈ 47 CPU-h / LEAN ≈ 14 CPU-h per checkpoint, no GPU |
| I22 | **Own PPO loop stage 2** ("lands before the X26 ride-along baseline") | DONE | `design_own_ppo_loop.md` status: stage 2 and stage 3 BUILT | — | — |
| I23 | **The `--unified-moves` umbrella silently overrides a typed `--damage-op false`** (and `move_prior_fusion`, `damage_outgoing`) | **DONE 2026-10-15 (T31)**: the conflict is REFUSED (`config.UmbrellaOverride`, FATAL_CONFIG; `checkargs` prints the same refusal and exits 3); `--unified-damage` now defaults to None so a typed `off` is seen; a typed `--damage-topk 0` beside an incoming matrix is refused too. The toy argv's `checkargs` report and resolved namespace are byte-identical before / after; so is `--arch production`'s | `src/main/train/umbrella_override_test.py` (fails on revert) | before T-MLP and any later ablation; not X26 (production sets them on) | T31 (~2–3 h) |
| I24 | **The `data/` freeze** (standing rule 6: a pin isolates code, not data) | DECISION | root `CLAUDE.md` rule 6; memory `feedback_pin_isolates_code_not_data` | L | **No `data/` change for X26's whole run (≈ 3–4 days at 500M).** Announce it at launch |
| I25 | **The training-input changes since the reference seeds**: the learnset (legal movepool) fix `gen3_learnset_prevo_chain_v1` (2026-10-14: priors and X5 candidates move, the observation does not) and the fork-recipe fix (2026-10-14) | DONE at HEAD | ledger 2026-10-14 GIGO FIX ×2 | — | none for a FRESH X26. They are why X26 is not a continuation of an EB seed (§2.1) |
| I26 | `--promotion-sprt` and `--eval-mirrored-pairs` "flip at X26" (2026-10-01) | DECISION OPEN | `design_model_management.md:38`, `design_league_decisions.md:32`, TASK_BACKLOG T6 / T17 | L (as a decision) | owner Q4 |
| I27 | The KL early stop for X26 | DECIDED OFF (2026-10-03); `era_plan_post_m5.md` row 0 said "KL early stop ON" (STALE; corrected in this commit) | `EXPERIMENT_BACKLOG.md` X5 decisions | — | — (finding F7) |
| I28 | **The metric-key collision crash**: every production run at a commit ≥ `65b71f6a` died at its FIRST post-update dump (step ~202k on the GPU; the training session's fresh `--arch production` CUDA smoke at `6f6b7f90`): sb3's stdout table cut `cell/move_resolution_restored_move_{weight,grad}_norm` (and the `restored_switch` / `status_switch` pairs) to one label and raised | **FIXED `109c4256`** (`gen3_metric_key_collision_v1`, ledger 2026-10-15 CRASH FIX): a collision-proof table + a STARTUP key check (`MetricKeyCollisionError`, FATAL_CONFIG) | `src/agents/training/train_logger.py`, `metric_keys.py`; the CPU production smoke reaches its dumps clean | L | the Training session's CUDA re-smoke in a GPU window |
| I29 | **Silent double producers and a never-produced metric** (found by the I28 fix's survey): `fork/*`, `train/effective_batch` (+ `train/dose_rate` one decision stale under `--adaptive-batch`) and `lifecycle/cuda_streams_after_freeze` each had TWO producers (the later write won silently); `grad/species_belief_*` was never produced (the bank registered `species_belief_set`, the probe looked up `species_belief`) | **FIXED** (`gen3_metric_producer_guard_v1`, ledger 2026-10-15 GIGO FIX): one producer per quantity + a logger guard (`DuplicateProducerError` under pytest / `--debug`; counted + reported in production) + `_R1_PROBE_TERMS` derived from the bank | `src/agents/training/train_logger.py`, `metric_producers_test.py` | L | none |

### 1.1 The K9(b) tie-share margin, read from the end-state runs

**What K9(b) is.** K9(b) is the behaviour-consistency probe (`agents/training/rust_rollout/consistency.py`). Once
per update, before any optimizer step, the learner recomputes log π on rows played at the current policy version
and compares it with the stored behaviour log-probability.
- A row whose forward pass has a selection or threshold within a relative 2e-4 of its cutoff is a near tie, and is
  **excluded** from the comparison.
- The **excluded share** is capped at 0.15, so that a fault which pushes many rows onto ties cannot hide.
- One update over the cap stops the run under `--behaviour-check fatal`, the default, which every `rb_bf_*` and
  `rb_es_end2_*` run used.
- It is logged as the TB scalar `behaviour/excluded_frac`. TB keeps about 152 of ~1,490 updates per 15M run.

**The share over steps** (window mean ±0.5M around each step, mean [range] over 8 seeds per arm):

| arm | ~1M | ~5M | ~10M | ~15M | slope / 10M, late half (7.5–15M) | highest single read |
|---|---|---|---|---|---|---|
| `rb_bf_eb` (production today) | .020 [.015, .024] | .028 [.023, .036] | .028 [.024, .035] | .032 [.028, .035] | +.004 [−.007, +.012] | .077 |
| `rb_bf_e` (fp32) | .018 | .028 | .030 | .032 | +.005 | .065 |
| `rb_es_end2` (closing test, end state) | .019 | .028 | .030 | .029 | +.003 | .068 |
| `rb_es_prod` (pre-adoption production) | .024 | .029 | .031 | .034 | +.003 | .064 |

The curve is concave: it rises fast to about 6M and then nearly flattens, with per-seed late slopes on both sides
of zero. A straight line from 2M would cross 0.15 at about 110–120M. That line is NOT defensible, because the late
slope is a third of the early one. Fits to the late half reach 0.15 at about 230–450M, and a log fit gives about
0.06 at 300M.

**The knock-in forks.** Revealing moves is a step change, not a drift. The per-update table of each fork's own log
(`bank_hib.py`, re-read 2026-10-15; 29–31 updates per fork over +3M steps):

| arm (seeds 3101 / 3102 / 3103) | median excluded share | per-update max | live K9(b) messages > 0.15 |
|---|---|---|---|
| `moves` (`moves:seen@0.95`) | **0.159 / 0.160 / 0.158** | 0.198 / 0.212 / 0.223 | 22 / 22 / 19 |
| `items` | 0.030 / 0.029 / 0.032 | ≤ 0.058 | 0 |
| `species` (`species:unseen@0.95`) | 0.029 / 0.027 / 0.027 | ≤ 0.065 | 0 |
| `spreads` | 0.006 / 0.005 / 0.005 | ≤ 0.019 | 0 |
| `ctrl` (no reveal) | 0.036 / 0.031 / 0.028 | ≤ 0.057 | 0 |

- **Every fork passes strict-by-read.** The live max |d log π| is ≤ 3.95e-5 on all 15, so every moves stop was a
  tie-share false alarm. `rb_hib_moves_s3101` stopped twice under `fatal` (0.165 and 0.151) before the uniform
  switch to `warn` (decision (f)).
- **The spreads reveal LOWERS the share** below the control's (0.005 vs 0.03). The mechanism is UNVERIFIED: the exact
  stats may remove near-ties at a threshold or `topk` cutoff.
- **The moves step's mechanism is UNVERIFIED too.** A candidate is more exact or near ties among the told seen mons'
  move candidates at the damage op's `topk`.
- The same forks' strength read: none of the reveals bought strength (`design_hidden_info_binding.md` §8.7).

**Verdict:**
- The MEAN is unlikely to threaten the ceiling before about 200M steps, and may never.
- The real risk for X26 is a single-update SPIKE. The static-token screen stopped on single updates of 0.17–0.25
  at 2–8M, before the flip-judge fix. The highest single read in production so far is 0.077.
- X26 therefore declares the closing test's Amendment 1 / 2 handling up front. A stop caused only by the tie share,
  with max |d log π| < 1e-4, is a false alarm: the run resumes, and once the crash budget is spent it continues
  under `warn`.
- X26 also logs the share on EVERY update (T32), so a spike is visible rather than inferred.
- A production change toward a move reveal (or a moves-like belief) would sit at the ceiling from step 1. That is
  a re-measure-the-ceiling trigger, not an X26 risk.

Scratch read: `/tmp/k9trend/` (not committed; the numbers above are copied from it).

## 2. The X26 recipe (proposed)

### 2.1 Fresh, not a continuation

The 2026-10-03 plan was "X26 continues from the winning arm's seed". This plan **supersedes** that: X26 is a
**FRESH seed at the exit-check commit**. The reasons:
1. U3 changes the fresh-init stream, and it must land before X26 (D-LD7). A continuation of a pre-U3 seed would
   carry a network no post-deletion seed can reproduce.
2. Every existing end-state seed trained on the pre-fix learnset table (I25), which read 1,500 legal (species,
   move) pairs as illegal. A continuation would change its priors mid-run, and the 10-09 rule is that a pin
   isolates code, not data.
3. A continuation is a FORK, with the fork-recipe class of hazards (2026-10-14 GIGO fix) and an inherited pool.
4. A fresh run costs only ~2.6 GPU-h more than a continuation over the first 15M of a 300–500M run.

### 2.2 The argv's contents (the Training session builds and validates the actual argv)

- `--arch production`. bf16 is included. The ride-along heads are typed: `--ridealong-ensemble 5 --ridealong-rnd
  --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all`. Their cost is in the bf16 A/B's measured
  throughput.
- `--checkpoint-every-steps 1000000` (I18).
- `--steps 500000000`, a CAP and not a target. The stop is the rule in §2.4.
- `--snapshot-ladder-games` at its default of 100, NOT 0, so `ladder.json` and the Hodge stand-in exist until the
  cycle monitor lands. Every reference run typed 0 for its speed rule. X26 has no speed rule.
- The restart interval is the default 6 h. T24's question (whether a restart buys anything on the Rust core)
  rides along as telemetry.
- `--compile-trainer` is automatic for cuda.
- `gen3ai_torch28` (cu126).
- KL early stop OFF.
- `--promotion-sprt` and `--eval-mirrored-pairs`: **OFF in this proposal, pending owner Q4**, so X26's plateau is
  production's recipe at depth.
- Run name `rb_x26_s2601`. A second seed, if Q1 says so, is `rb_x26_s2602`.

### 2.3 Length, cost and disk

| quantity | value | source |
|---|---|---|
| throughput | ~1,600–1,630 env steps/s, ~10.3 min per 1M steps, heads on, bf16 | `rb_bf_eb_s2005`: 15.05M in 9,222 s |
| GPU-h to 100M / 200M / 300M / 500M | ≈ 17 / 34 / 52 / 86 | that rate, exclusive GPU |
| disk at a 1M cadence: 200M / 500M | ≈ 20 GB / ≈ 36 GB | 55 MB per checkpoint + 3.3 GB compile cache + ~2 GB traces + ~2.4 GB value sidecar |
| free disk | 341 GB | `df` 2026-10-15 |

### 2.4 The stop rule, and what "the meter before the deep run" needs

Every plateau component reads retained checkpoints OFFLINE (`design_evaluation.md` F-ED-19). So the meter does not
have to exist inside X26. What X26 needs at launch:
- **(a)** the stop rule written into Amendment 2, and
- **(b)** Tier 1 shown to work on real end-state checkpoints (T33).

Until U5 and U9 land, the INTERIM rule can only say CONTINUE early, never stop early:

> **X26 stops only when ALL of the following hold:**
> - Tier 1 reads `TIER1_PLATEAU` (`python -m main.plateau tick` then `status`, registered Δ 10M / W 50M, so 70M at
>   the earliest);
> - the Hodge cyclic width (`python -m main.elo <run> --hodge-bootstrap 200`) is within its bootstrap noise floor;
> - the bot rates in the in-loop `eval_results.jsonl` and a Metamon SmallRL greedy anchor read are flat over the
>   same 50M window.
>
> Any disagreement means CONTINUE. Once U5 and U9 land, the two-tier rule replaces the last two bullets for every
> later check. The cap is 500M.

The cost of the interim rule is at most running X26 somewhat past its plateau, which is cheap. It never stops the
run on a false plateau.

### 2.5 What rides on X26's checkpoints (all offline, all under rule 5b while a run is live)

- The ride-along reads R1–R5, at the final checkpoint and at every 25M.
- The maturity battery (`design_hidden_info_binding.md` §12.13.7) at ≥ 25M, every 25M, and at the plateau.
- The X31 team-gradient re-check.
- The toy controls' plateau read (§3.5).

### 2.6 Evaluation cost and the idle budget (T33, measured 2026-10-15)

**The owner's question:** "Do we need to invest in evals running on the main process and letting them use the spare
cycles?" **Answer: not for X26.** The offline CPU lane keeps up with X26 if two cheap things are done:
- tick at the REGISTERED Δ 10M, not 5M;
- pipeline the meter's 40-pair batches. This is an offline fix, not a trainer change.

An in-process eval (c) pays only past the threshold below. An out-of-process filler (b) does not pay. X14's overlap
(d) is worth roughly 5–20× more wall time than any eval placement.

Evidence: `../research_state/measurements/t33_plateau_cost_2026-10-15/` (README, scripts, raw tick and sampler logs).

**The first real Tier-1 ticks** (`rb_toy_ent_s4001`, Δ 5M, lag 4, W 20M; ledger `models/_ledger`, `plateau_t1`
requests; CPU on the agent cores under rule 5b, `--threads 4 --torch-threads 2 --n-envs 64`):

| check | newest vs W-back node | verdict | pairs (games) | LLR | newest's score | wall | CPU-s | status after |
|---|---|---|---|---|---|---|---|---|
| 25M | 25.00M vs 5.00M | **GAIN** | 80 (160) | +3.044 | 91.9 % | 66 s | 100 | CLIMBING |
| 30M | 30.00M vs 10.00M | **GAIN** | 120 (240) | +4.060 | 77.5 % | 92 s | 110 | CLIMBING |
| 35M | 35.00M vs 15.00M | **GAIN** | 120 (240) | +3.424 | 69.2 % | 96 s | 121 | CLIMBING |
| 40M | 40.00M vs 20.00M | **GAIN** | 120 (240) | +3.130 | 66.2 % | 79 s | 117 | CLIMBING |
| 45M | 45.00M vs 25.00M | **GAIN** | 200 (400) | +3.070 | 59.5 % | 66 s | 118 | CLIMBING |
| 50M | 50.00M vs 30.00M | **GAIN** | 320 (640) | +3.362 | 57.3 % | 123 s | 224 | CLIMBING |
| 55M | 55.00M vs 35.00M | **GAIN** | 200 (400) | +2.977 | 59.0 % | 299 s | 324 | CLIMBING |
| 60M | 60.00M vs 40.00M | **GAIN** | 320 (640) | +3.134 | 56.5 % | 397 s | 449 | CLIMBING |
| 65M † | 65.00M vs 45.00M | **GAIN** | 400 (800) | +3.721 | 56.1 % | 146 s | 252 | CLIMBING |
| 70M † | 70.00M vs 50.00M | **GAIN** | 200 (400) | +3.079 | 60.8 % | 195 s | 218 | CLIMBING |

- **Tier-1 status: CLIMBING as of the 70M check. There is no FLAT yet, so there is no toy plateau.**
- The newest node's score against the node 20M back falls from 92 % to 56–61 %. The pairs the GSPRT needs grow from 80
  to 320–400, and a FLAT check would take about 1,600 pairs (§8.1's OC table).
- The run did not reach its cap. **It died at 08:24 on a `CompileCanary` FATAL after update 700**: the compiled bf16
  learner's gradients on `pointer_head.move_proj` disagreed with eager (FATAL_CONFIG, not restarted). The safe
  rollback point is 62.0M, and the last passing canary was at 62.83M.
- **† The 65M and 70M checks used nodes that trained after the last passing canary**, so they are on weights that
  trained on the suspect graph. Treat those two verdicts as void until the training session rules on the rollback
  (FINDING F17).
- The meter loop was stopped by its operator at 08:39, because no further check was due. Re-run
  `python -m main.plateau tick rb_toy_ent_s4001 --delta-steps 5000000 --lag 4` after any resume.

- Every check decided, with no CONFLICT and no INCONCLUSIVE.
- The 20M check reads `MISSING_NODE` (its W-back node is step 0, and no run saves one). This is expected, and the
  first playable check is 25M, as §3.4 says.

**The cost of one tick, measured beside the live toy.** The agent cores ran at load ≈ 11–15, shared with two other
agents' jobs.

| network | check | games | wall | CPU-s | rate (per 80-game batch) |
|---|---|---|---|---|---|
| toy (T-ENT) | 25M–70M, GAIN at 80–400 pairs | 160–800 | 66–397 s (all 10: 26 min) | 100–449 (all 10: 34 CPU-min) | 1.3–6.1 games/s per check; ≈ 0.35–1.1 CPU-s per game |
| **production** (`rb_bf_eb_s2001` 15M vs 5M, scratch ledger) | GAIN at 120 pairs | 240 | **410 s** | **650** | **0.57–0.67 games/s**; ≈ 2.6 CPU-s per game |
| production, ONE 160-pair batch (`main.h2h play`, 15M vs 10M) | — | 320 | 296 s | 510 | **1.16 games/s**; ≈ 1.5 CPU-s per game |

Engine start-up costs 10–22 s per check.
- **The meter is tail-bound.** A 40-pair batch is 80 games on 64 envs, so every batch waits for its slowest game.
  On the production net that wastes about 45 % of the lane, as the 1.16 vs 0.62 games/s rows show (FINDING T33-2).
- §8.1's design figure of 4.8 games/s does not hold for the production net on contended agent cores. The measured
  rate is 0.62 today, or 1.16 when pipelined.

**Extrapolated to X26** (production net, ≈ 1,570 steps/s ⇒ Δ 5M = 53 min, the registered Δ 10M = 106 min). The
games per check come from §8.1's OC table (`design_evaluation.md`):

| check | games | lane today (0.62/s): wall · CPU-h | lane pipelined (1.16/s): wall · CPU-h | in-process GPU (c) at 17–37 games/s |
|---|---|---|---|---|
| a GAIN check (climbing) | 240–640 | 6–17 min · 0.2–0.5 | 3–9 min · 0.1–0.3 | 7–38 s |
| true p 0.50 (mean, FLAT) | 3,240 | **87 min** · 2.3 | 47 min · 1.35 | 1.5–3.2 min |
| true p 0.51 (mean, hardest) | 5,100 | **137 min** · 3.7 | 73 min · 2.1 | 2.3–5 min |
| the cap | 12,000 | **5.4 h** · 8.7 | 2.9 h · 5.0 | 5.4–12 min |

The (c) column's rate is a range:
- 33–42 games/s (37 used here) is the trainer's own blocking eval cycle on production nets (1,300–1,400 games in 31–43 s; `rb_bf_eb_s2001` and
  `rb_es_end2_s2001` logs). Those are model-vs-bot games.
- An h2h runs two networks, so 17 games/s is the pessimistic half of that.

**Can 3 crowded agent cores keep up?** A check keeps up when its games ≤ rate × cadence.
- Today's limit is 1,940 games per check at Δ 5M and 3,880 at Δ 10M. With pipelined batches it is 3,620 and 7,250.
- So the lane keeps up at every cadence while X26 is CLIMBING (GAIN checks take 6–17 min).
- Near the plateau, today's lane at Δ 5M FALLS BEHIND on every check. A FLAT check takes 87–137 min against a
  53-min cadence, so the stop decision comes about 2–4 h late. That is 2–4 GPU-h of training past the plateau, per seed.
- At the registered Δ 10M with pipelined batches, the lane keeps up through the hardest mean case (73 min against
  106). It lags only at the 90th percentile or at the cap, by about 1 h.
- Each near-plateau check also takes 1.4–3.7 CPU-h from the lane that every agent shares.

**The trainer's idle budget, from records** (the per-update `rollout/collect_ms`, `train/train_ms` and
`rust_env/*_ms_per_host_step` of each run's `launcher_child.log`, medians over the last ¾ of each log; script
`phase_stats.py`). The toy's phases were also sampled read-only at 0.5 s for 175 s (`/proc/stat` plus
`nvidia-smi` queries; no GPU work; script `sampler.py` / `phase_util.py`). The trainer is pinned to CPUs 0–4 and 8–12,
which are 10 logical CPUs.

| run | collect | update | env-core CPUs idle (the update share) | GPU idle in collect (≥) |
|---|---|---|---|---|
| `rb_bf_eb_s2001` (production, bf16) | 19.7 s | 42.1 s | **68 %** of the 61.8 s cycle | 12.1 s (inference waits are 7.6 s of 19.7) ⇒ **≥ 20 %** of the cycle |
| `rb_es_end2_s2001` (production, fp32) | 13.9 s | 54.4 s | 80 % | ≥ 6 s ⇒ ≥ 9 % |
| `rb_hib_ctrl_s3103` (production fork) | 16.5 s | 41.9 s | 72 % | ≥ 7.7 s ⇒ ≥ 13 % |
| `rb_toy_ent_s4001` (toy), SAMPLED | 9.9 s | 9.2 s | 48 % | measured: GPU util 29.5 % in collect, 81.5 % in update |

In the toy's sample, the trainer's 10 CPUs were 44 % busy in collect and 16 % busy in the update (the Python trainer
used 1.1 cores; the env core was idle).
- The production per-phase CPU and GPU utilization is NOT in any record. The GPU column above is a lower bound on idle
  time derived from the inference waits.
- The procedure that measures it needs no GPU window and no lease, because it only reads `/proc` and queries
  `nvidia-smi`. While any production run is live, run `sampler.py <out> 300 <trainer pid> <rust_env_proc pid(s)>`
  beside `nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv,noheader -lms 500`, both for 5 min under
  `taskset -c 5-7,13-15 nice -n 19`. Then run `phase_util.py`. That is ≤ 10 min in total.

**The four placements, compared** (X26, production net):

| | where the games run | cost to training | can keep X26's cadence? | build | risk |
|---|---|---|---|---|---|
| **(a)** offline on the agent cores (today) | 6 crowded logical CPUs | none | climbing: yes. Near plateau: NO at Δ 5M today; YES at Δ 10M with pipelined batches | the pipelining fix (~0.5 agent-day, offline code, unpinned) | late stops: 2–4 GPU-h per seed at Δ 5M today |
| **(b)** out-of-process filler at SCHED_IDLE | ≈ 8 idle logical CPUs during the update (68 % of the cycle), so about 5.5 core-equivalents, uncrowded | unknown. Rule 5b's record: sharing physical cores cost a live seed 25–30 % even at nice 19. It needs §4.5's harm A/B (≈ 8.6 h of a production run) | yes, roughly 2× the lane | a scheduler-aware worker plus the harm A/B (~2–3 agent-days) | a trainer slowdown. The GPU half is NOT feasible: CUDA does not preempt between processes, and the GPU's idle time falls in collect, when the CPUs are busy |
| **(c)** in-process blocking filler (the U4 lane, Tier 1 only) | the trainer's GPU and its eval core, between two host steps (like today's eval cycle) | ≈ 0.1–1 % of wall while climbing; 3–6 % of wall per near-plateau check (1.5–5 min of every 53); about 0.5–1 GPU-h over a 500M run | yes, with large margin | U4 (~1.5–3 agent-days). It must be in P_X26 before launch, because a PINNED run cannot gain it later (F-ED-19), so it would land on X26's critical path | stalls training for each check |
| **(d)** X14 rollout/update overlap | — (spends the same idle time on SPEED) | a GAIN: the ceiling is 61.8 → 42.1 s per cycle (1.47×); the survey estimate is 18–30 % (UNVERIFIED) | — | M, plus its own strength-per-GPU-h A/B | one-version-stale rows |

(b) and (d) compete for the same idle update-phase CPUs. (c) composes with (d), because it is a short pause.

**The threshold at which (b) or (c) pays.** Let G be a near-plateau check's games, r the lane's rate and Δt the
cadence.
- **(c) pays when G / r − Δt exceeds about 1 h per near-plateau check.** At that point the GPU-h saved by stopping on time beats
  (c)'s ≈ 0.5–1 GPU-h per run plus its build.
  - For X26 at the registered Δ 10M with pipelined batches, G / r − Δt is ≤ 0 at the mean and about 1 h only at the
    90th percentile or the cap. So **(c) does not pay for X26**.
  - It starts to pay for a net about 2× slower on CPU than today's production net, for a cadence ≤ 5M, or when
    several seeds must each stop on their own plateau.
  - It also needs an unpinned run, or a P_X26 that already carries it.
- **(b) pays only if** its harm A/B shows ≤ 1 % on `train_ms` and on collection, AND (c) is unavailable (a pinned run),
  AND X14 has not claimed the update-phase CPUs. None of these hold today.
- **(d) dominates both on wall time.** Its 18–30 % is 5–20× the 1–6 % that any eval placement costs or saves.
  Decide X14 on its own A/B (EXPERIMENT_BACKLOG #12), and treat eval as a short blocking pause (c) inside it if a
  future run needs one.

## 3. The TOY CONTROLS

### 3.1 The question

The owner, 2026-10-06: *"I am open to a trivial model … I don't want to recreate our end state architecture, so
I'm happy to have a toy model … to understand what all the complexity is allegedly buying us."*

A toy is a **control trained to ITS OWN plateau**, never a candidate end state. The architecture audit's F24 named
three controls ([`design_arch_audit.md`](design_arch_audit.md) §3.3):
- **C-ENT**, entity tokens and the trunk only;
- **C-MLP**, a flat MLP;
- **C-NOCELL**, production minus the seven per-action blocks. C-NOCELL is **MOOT**: move resolution retired those
  blocks in the end state.

### 3.2 The two toys

| toy | human description | how it is declared | learner params | MFLOP / row | build |
|---|---|---|---|---|---|
| **T-ENT** (`rb_toy_ent_s40xx`) | **"Toy: entity tokens + attention only."** The end state's static per-Pokémon tokens, TWO attention rounds, the pointer action head and the win-probability critic. It has NO damage operator, NO belief heads or priors, NO edges or seats, NO per-action cells, NO move resolution and NO turn-history events. The four observation-FACT levers (`obs_facts`, `move_actor_state`, `move_target_state`, `drop_progress_clock`) stay at production, so the toy differs from production in MACHINERY only, never in facts handed to it | `--arch production --allow-nonproduction-arch` + the 44-key ablation set below. **`checkargs` exit 0 at `9ada3604`** ("44 of 57 ARCH-surface keys differ … EXPLICIT", "7 of 29 RECIPE knobs differ", all typed, "✓ this command still launches", trunk bf16) | ≈ 1.42M (production 2.61M) | ≈ 17.4 (production 95.7) | none |
| **T-MLP** (`rb_toy_mlp_s41xx`) | **"Toy: flat MLP."** The flat 2845-dim observation through an MLP trunk (2845 → 512 → 512). The MLP projects to 10 action tokens (4 moves, 6 switches, `Linear(512, 10 × 128)`), which go into the SAME pointer head and critic contract. No entities and no attention | a new extractor mode (unit T34) | ≈ 3.0M (estimate) | ≈ 6 (estimate) | ~10–16 agent-hours (opus-high), plus a 20-minute GPU check under a lease |

**The T-ENT ablation set**, verified by `checkargs` for this doc (`/tmp/x26_rd/toy_argv.txt`). It is a
requirement list, not a command to paste:

```
--unified-moves off --no-damage-op --no-damage-outgoing --damage-matrices off --damage-topk 0
--move-belief-mode off --no-move-prior-fusion --no-move-latent --opp-belief-cls-k 0 --no-item-belief
--no-spread-belief --no-spread-belief-nature --no-t0-species-prior --no-species-prior-fusion
--entity-topk-seats 0 --no-entity-tail-seats --edge-bias-families off --no-history-events
--no-value-entity-pool --no-value-entity-pool-full --move-resolution off --speed-physics off --op-reduction max
--mon-hazard-cost off --trunk-layers 2 --switch-hazard-cost off --eot-residual off --move-resolution-facts off
--status-facts off --ko-ramp ramp --g-ledger coarse --effective-stats off --move-set-closure off
--no-intent-move-cell --no-intent-threshold --no-intent-conditional --no-pair-outcome-cell
--no-pair-outcome-switch --no-switch-branch-cell --no-conditional-threat-cell --no-op-drop-renders
--no-op-believed-lean
--opp-belief-aux-coef 0 --opp-intent-coef 0 --move-belief-coef 0 --move-belief-latent-coef 0
--spread-belief-coef 0 --item-belief-coef 0 --hp-type-belief-coef 0
```

Notes on the set:
- **`--unified-moves off` is load-bearing.** Without it, the umbrella silently turns `damage_op`,
  `move_prior_fusion` and `damage_outgoing` back on (I23).
- **The seven zero coefficients are required.** Without them the belief losses are refused, or the recipe
  re-derives `opp_belief_slots` and `opp_intent`.
- **Rejected: a one-line `--arch toy_ent` entry.** `ArmDecl` has no recipe overlay, and the umbrella reverts five
  keys (a simulated arm was refused). A clean named arm costs ~2–4 agent-hours and buys only a shorter argv, so the
  typed set with `--allow-nonproduction-arch` is used.
- 🚨 **Since legacy deletion U1 (2026-10-15) this argv NO LONGER PARSES at HEAD** — EXPECTED (D-LD4): `--speed-physics`,
  `--op-reduction`, `--switch-hazard-cost`, `--eot-residual`, `--status-facts`, `--ko-ramp`, `--g-ledger` and
  `--effective-stats` are deleted flags. A T-ENT seed launched before U1 resumes PINNED to its own launch commit (a
  pinned argv is judged by the pin's parser); a NEW T-ENT seed launches pinned at `pre-legacy-deletion` or with those
  eight flags dropped (they are vacuous without the damage op). A T-ENT checkpoint still LOADS at HEAD: the toy built
  no damage op, so every deleted lever is VACUOUS there (`model_version/adopted_levers.py`, §3.5's requirement).
- **Rejected: depth below 2 and a no-trunk mode.** `TRUNK_LAYERS_CHOICES` is (2, 3, 4), so no network below two
  rounds exists without a build. T-MLP is that build.

### 3.3 The recipe and the seeds

- **Recipe: production's fresh recipe, unchanged.** That is N 256, n_steps 384, 10 epochs, lr 3e-4 under the KL
  controller, and the bf16 trunk; only the zeroed coefficients of the absent heads differ.
  - The recipe was tuned for production, which may handicap a toy. This is DECLARED, and it is the question as the
    owner posed it: what the complexity buys at our recipe.
- **No ride-along heads.** They are detached and would not change learning, but they are compute the toy does not
  need.
- `--checkpoint-every-steps 1000000` (the Tier-1 nodes), `--snapshot-ladder-games 0`, default eval.
- **Seeds:** T-ENT `s4001`–`s4003`, T-MLP `s4101`–`s4103`. Three per toy is the audit's replicate design.
- **Where it runs:** T-ENT was launched at HEAD before the deletion. Its levers' OFF values existed only before U1 /
  U3 (U1 has landed: the eight op flags are gone), and D-LD4 says any later ablation of an adopted lever runs PINNED at
  `pre-legacy-deletion`. T-MLP runs at HEAD
  after its build. As a new mode, it survives the deletion.
- **Data:** T-ENT reads no prior, belief or X5 candidate, so the 10-14 learnset fix cannot reach it. The
  observation did not move under that fix (991 / 991).

### 3.4 The toy PLATEAU STOP RULE

The toy is cheap and should plateau early, so it uses a SCALED Tier 1:

> **Toy plateau:** run `python -m main.plateau tick <run> --delta-steps 5000000 --lag 4`, so Δ = 5M and
> W = 20M; the first check is due at 25M.
> - STOP at `TIER1_PLATEAU`, which is two consecutive FLAT checks (GSPRT H0 0.50 / H1 0.52, α = β = 0.05, cap 6,000
>   pairs). So 30M at the earliest.
> - The toy's **plateau checkpoint** is the newest node of the second FLAT check.
> - **Hard cap: 100M steps.** A cap stop is recorded as `CAP`, not as a plateau.
> - An `INCONCLUSIVE` or `CONFLICT` check is a finding and does not stop the run; the run continues to the next
>   check.

Running this rule on the toys is also **T33, the first real validation of Tier 1**. It exercises the full
`main.plateau` path on current-architecture checkpoints, with its CPU cost timed beside a live run, before X26
depends on it.

**K9(b) on the toys:** keep `--behaviour-check fatal`, with the closing test's Amendment 1 / 2 handling for a stop
caused only by the tie share. A toy's tie share is unmeasured.

### 3.5 What the comparison reads (PRE-REGISTERED here, before any toy seed)

**The production references:**
- at 5M and 15M: the 8 EB seeds' `final_model_stage1_5M.zip` and `final_model.zip`;
- from 15M to 60M: the R-60 scout (§5.1, G3);
- at plateau: X26's plateau checkpoint.

The estimator is the closing test's `cross_stat`: the two-way mirrored cross, Δ̂ = toy − production, with a
two-sided 90 % interval and the rule-8 band of 1e-9.

| read | cells | pairs | what it gives |
|---|---|---|---|
| **R-T1, matched steps (PRIMARY):** each toy seed at 15M against each EB 15M final | 3 × 8 | 500 mirrored pairs per cell | **G_m** = production − toy at matched 15M |
| R-T1b, early: the toy 5M against EB 5M | 3 × 8 | 250 per cell | descriptive |
| R-T1c, matched steps beyond 15M: the toy at 30M and 60M against R-60 at the same steps | 3 × 1 | 1,000 per cell | descriptive; one production seed |
| **R-T2, plateau against plateau (PRIMARY):** each toy plateau checkpoint against X26's plateau checkpoint | 3 × 1 (3 × 2 if Q1 = two seeds) | 1,000 per cell | **G_p** = production − toy at each one's plateau. With one X26 seed, production's seed variance is NOT in the interval (declared) |
| R-T3, external units: each toy plateau against Metamon SmallRL, greedy vs greedy, `main.anchors` | 3 | 200 games | descriptive; the regime stated with every number |
| R-T4, cost: GPU-h to plateau, params, FLOPs / row, measured fps | — | — | the price side |

**Loadability across the deletion (requirement on T30-U0).** R-T2 plays a pre-deletion T-ENT checkpoint against a
post-deletion X26 checkpoint. Under D-LD1, a recorded OFF value of a deleted lever is refused at HEAD.
- **Requirement:** U0's `RetiredLever.active` treats a lever as INACTIVE when the recorded config lacks the lever's
  prerequisites. That is D-LD2's "vacuous otherwise" rule, applied to loading. T-ENT has no damage operator, so
  every op lever it records is vacuous.
- **If U0 cannot meet it:** R-T2 runs through the EB BRIDGE instead. Toy plateau vs the 8 EB finals is played at
  the `pre-legacy-deletion` tag, and X26 plateau vs the 8 EB finals at HEAD (EB records the full adopted surface,
  so it loads at both), on the same schedule seeds. Then G_p = WR(X26 vs EB) − WR(toy vs EB), a difference of two
  estimates, so it needs about 2× the pairs.

### 3.6 What "the complexity buys", in one sentence the owner can act on

> **"At matched steps the end state's machinery is worth G_m pp over a model with only entity tokens and attention
> (G_m′ over a flat MLP); at each one's plateau it is worth G_p pp; it costs k× the compute per step. So the
> complexity buys {CEILING | SPEED ONLY | NOTHING MEASURABLE}."**

The classification is registered here, with the ±2.0 pp bar the closing test and the bf16 A/B used:
- **CEILING**: G_p's 90 % interval lies wholly above +2.0 pp. The machinery raises how strong the model gets.
  *Action:* keep it, and judge future machinery at plateau.
- **SPEED ONLY**: G_m's interval lies wholly above +2.0 pp, AND G_p's interval lies inside [−2.0, +2.0]. The
  machinery makes training faster, not the model stronger.
  - *Action (an owner decision):* weigh it per GPU-hour. A simpler network can reach the same plateau, so
    simplification becomes a registered challenger.
- **NOTHING MEASURABLE**: both intervals lie inside [−2.0, +2.0].
  - *Action:* the strongest cheap result this project can produce (audit §3.3). A SIMPLIFICATION screen goes to
    the owner as the next arm.
- **Otherwise UNDECIDED.** Report it descriptively and take no action. A wide interval is never read as a null
  (equivalence needs the delta's interval inside the bar).

For T-MLP the same classes apply with G_m′ / G_p′, and the question becomes "what do entities and attention buy".

### 3.7 GPU-h estimate for the toys

- **Measured:** production bf16 runs at 1,631 fps, 2.56 h per 15M.
- **Estimated (UNVERIFIED; T-ENT s4001's first hour measures it):** T-ENT at ~3,000–5,000 fps. The learner update
  dominates and shrinks 3–5×, but a small model on the GPU is limited by kernel launches, and the Rust core stays
  at ~5.5 s per cycle. That gives about 0.8–1.4 GPU-h per 15M.
- **Per seed:** 30–60M to a plateau ≈ 2–6 GPU-h; to the 100M cap ≤ 9 GPU-h.
- **Three T-ENT seeds:** ≈ 8–18 GPU-h expected, ≤ 27 GPU-h at the cap. T-MLP is about the same.
- **The crosses:** R-T1 is 12,000 pairs. Its GPU time is UNVERIFIED; the Training session schedules it in a
  window.

## 4. The EXIT CHECK

The exit check is PASS only when every row passes. X26 launches only after it. The commit at which it passes is
**P_X26**, and X26 is pinned there.

| # | item | PASS criterion | evidence to record |
|---|---|---|---|
| E1 | **The bounded list** (the 2026-10-06 exact refactors, behaviour screens and perf) | every item landed or explicitly DROPPED with a reason in this doc's Decision record. Already done: refactors (version break 2026-10-07), screens (end state ADOPTED 2026-10-12), perf (bf16 ADOPTED 2026-10-14). Explicitly deferred past X26, each a future arm with its own A/B: T26 overlap and CUDA graphs | this table |
| E2 | **Legacy deletion U0, U1, U3 landed** | each a separate `/gen3ai-ship` with green gates; the K9 golden re-recorded exactly once (at U3); the ledger line "fresh-init stream changed at `<U3 sha>`" | commits + ledger |
| E3 | **U2 / U4 either landed with an init-identity proof or deferred** | for each unit that lands before P_X26, a test that FAILS on revert: production's fresh state_dict at a fixed seed is byte-identical before and after the unit | the test name per unit |
| E4 | **The umbrella GIGO fixed (T31)** | a typed key the umbrella would override is REFUSED, typed, with a test that fails on revert; production's resolution unchanged (`checkargs --arch production` byte-identical report) | commit |
| E5 | **The routine gate and the slow-tier status** at P_X26 | routine gate green (or the targeted set + static tier, scope named, per the testing rules); `slow_tier_status.json` has no recorded FAIL | gate log |
| E6 | **Cost re-measured at P_X26** | a production run's quiet-window `train_ms` and fps over ≥ 20 updates within ±7 % of `rb_bf_eb`'s 42.07 s / 1,631 fps. A regression beyond that is investigated before launch (it changes nothing about strength but sizes the run) | the numbers + their run |
| E7 | **The sanity screen at P_X26** | TWO fresh production seeds (`rb_xs_s2701`, `s2702`) to 5M (~1.8 GPU-h), crossed against the 8 EB `final_model_stage1_5M.zip`, 500 mirrored pairs per cell (2 × 8). **PASS iff the pooled Δ̂ (sanity − EB) ≥ −3.0 pp AND neither seed's own Δ̂ < −6.0 pp**, with no crash, no NaN, and no K9(b) stop that is not a tie-share false alarm. A value within 1e-9 of a bound is not a crossing (rule 8). **FAIL ⇒ STOP and bisect the deletion units**: it is a screen for breakage, not an equivalence test. (Seed-to-seed SD at 5M is ~3 pp, from the stage-1 cross's s2_cols 9.7 pp²) | `verdict.json` from a coded rule committed before the cross |
| E8 | **Plateau meter at launch** | (a) the stop rule of §2.4 written into Amendment 2; (b) T33 PASSED: at least one `main.plateau tick` decided, with no CONFLICT, on a real current-architecture run (a T-ENT seed or R-60), with its CPU minutes per check recorded | the ledger `plateau_t1` rows |
| E9 | **The ride-along registration** | Amendment 2 committed (T32): every stale clause of I11 fixed, read points every 25M + plateau, bf16 head-identity test green, the K9(b) handling declared, per-update excluded-share logging live | commit |
| E10 | **The argv** | `python -m main.checkargs --argv "<X26 argv>"` exit 0 at P_X26; `python -m main.launcher --dry-run …` "would launch" (from the MAIN checkout); torch 2.8.0+cu126 | both outputs in the launch note |
| E11 | **Disk and data** | ≥ 100 GB free on `models/`; the `data/` freeze announced to every session for X26's duration | `df`; the announcement |
| E12 | **The owner's answers** | Q1 (seeds), Q2 (meter scope), Q3 (D-LD decisions), Q4 (promotion) recorded in this doc's Decision record, or the recommendation adopted under the 15-minute standing rule | Decision record |

## 5. The GPU SCHEDULE from tonight to the X26 launch

The GPU lease is held by the Training session ('Training Oct', expiring 2026-10-19 01:36). Every job below runs
through it. The last hidden-info knock-in fork (`rb_hib_ctrl_s3103`, started 01:35) ends at about 02:15. The
estimates assume an exclusive GPU.

**Preemption:** any GPU need of the P0.1 hidden-info study (for example its crosses) goes BEFORE every job below,
per the owner's P0.1 quota priority.

### 5.1 The order

| # | job | needs a build? | GPU-h | what it buys |
|---|---|---|---|---|
| **G1** | **T-ENT seed 4001**, fresh at HEAD, to its toy plateau (§3.4). Its first ~20 minutes are the toy's GPU check: the compiled path plus bf16 on a flag combination never compiled before | **no** | 2–6 expected, ≤ 9 | the first toy; the measured toy fps; the first real Tier-1 ticks (T33) |
| G2 | T-ENT seed 4002, then seed 4003 | no | 4–12, ≤ 18 | the 3 replicates |
| G3 | **R-60, the production scout:** continue `rb_bf_eb_s2001` (pre-declared as the lowest seed index, not chosen by strength) from 15M to 60M in its own lineage, at its recorded pin (`f45d0c27`) and recipe | no | ≈ 7.7 | the only production curve past 15M before X26: matched-step references at 30M / 60M (R-T1c); the first production Tier-1 check (60M vs the 10M W-back); the K9(b) share at 4× the longest horizon so far; restart telemetry for T24. **It is a scout, never a comparator for X26** (pre-fix learnset, pre-U3 init) |
| G4 | The R-T1 / R-T1b crosses (toy 15M and 5M against EB), in a GPU window | no | UNVERIFIED (≤ ~2) | G_m |
| G5 | **T-MLP** GPU check (20 min), then seeds 4101–4103 to their plateau | **yes** (T34, after T31) | ~0.3 + 8–18 | G_m′, G_p′ |
| G6 | **The exit-check sanity screen** (E7): 2 fresh production seeds × 5M at P_X26, then the 2 × 8 cross | yes (P_X26 = after T30-U3) | ≈ 1.8 + the cross | E7 |
| G7 | **X26 launch** (`rb_x26_s2601`) | — | 17 per 100M; ≤ 86 at the cap | the deep run |

**The expected timeline.** G1–G4 fill Thu 10-15 to Sat 10-17 (about 20–35 GPU-h). T34 should land by about
Sat 10-17, so G5 runs Sat to Sun. G6 runs when U3 lands, about Sun 10-18 to Mon 10-19. X26 launches about Mon 10-19
to Tue 10-20. If the deletion chain slips, the GPU is never idle: more T-ENT or T-MLP seeds, or extend R-60 toward
100M (each 10M ≈ 1.7 GPU-h). The rented-card cloud benchmark (weekend of 10-17) does not use the local GPU.

### 5.2 The FIRST job tonight: launch requirements for T-ENT seed 4001

This is a requirement list for the Training session to turn into an argv and validate. It is NOT a command to
paste.
1. **Lease:** the Training session's own lease. Launch from the MAIN checkout (deps built) at origin/main HEAD
   (≥ `9ada3604`, or the commit that lands this doc). No pin flag is needed: a fresh run records its commit.
2. **Surface:** `--arch production --allow-nonproduction-arch`, plus EXACTLY the §3.2 ablation set (44 ARCH keys,
   7 RECIPE coefficients), plus `--steps 100000000 --seed 4001 --checkpoint-every-steps 1000000
   --snapshot-ladder-games 0 --device cuda --run-name rb_toy_ent_s4001`. No ride-along flags.
3. **Validate before launch:**
   - `checkargs` exit 0, reporting 44 ARCH keys differ and 7 recipe knobs differ, all typed. `damage_op False`,
     `trunk_layers 2` and `move_resolution 'off'` must be listed. `obs_facts`, `move_actor_state`,
     `move_target_state` and `drop_progress_clock` must NOT be listed (they stay at production). The trunk line
     must read bf16.
   - `--dry-run` reports "would launch".
4. **After the first save (1M):** diff `model_config.json`'s arch keys against the 44 intended values. This catches
   the I23 class at the source: a silently re-enabled key is a STOP.
5. **Watch:**
   - the first compile (never compiled before; a failure is a finding, not a retry loop);
   - fps over the first hour (it replaces §3.7's UNVERIFIED estimate);
   - `behaviour/excluded_frac`;
   - the first 6 h restart: `--dry-run` of the resume must show the 7 zero coefficients INHERITED. If not, stop
     before the restart and report.
6. **Stop:** the toy plateau rule (§3.4), ticked by T33's operator at every 5M from 25M. Cap 100M.

### 5.3 The CPU / agent BUILD UNITS (in parallel; rule 5b while the GPU is leased)

Each is a dispatchable brief. All agent CPU work runs `nice -n 19 taskset -c 5-7,13-15`, at most 2 heavy jobs at
once. They are deduped against `../ops/TASK_BACKLOG.md`, where they are added as T30–T35.

| unit | scope | files | tier | est. | depends on | dedupe |
|---|---|---|---|---|---|---|
| **T30** | **LEGACY DELETION, the X26 critical path: U0 → U1 → U3**, one `/gen3ai-ship` each. Plus this doc's U0 requirement: `RetiredLever.active` is FALSE when the recorded config lacks the lever's prerequisites (D-LD2 vacuity applied to loading), with a test that a T-ENT checkpoint loads at HEAD. U2 / U4 only with E3's init-identity proof | the manifest's rows (`agents/model/`, `main/train/config.py`, `model_version/`, `flag_registry.py`, `arch_arms.py`) | sonnet-xhigh for U0 / U1, opus-high for U3 (init stream, golden re-record) | ~3 agent-days sequential, plus gate time on 3 cores | owner Q3 (D-LD1–D-LD7) | NEW row. The manifest has no backlog row (T22 is the older, Rustboro-boundary manifest) |
| **T31** | **The umbrella GIGO:** a typed `--damage-op false` (or `--no-move-prior-fusion`, `--no-damage-outgoing`) under `--unified-moves both` is REFUSED, typed, not silently overridden. Audit every umbrella in `config.py` for the same class; a test that fails on revert | `main/train/config.py:257-260`, its test | opus-medium | 2–3 h | — | NEW (found by this unit) |
| **T32** | **X26 Amendment 2 + its two small builds:** (a) fix every stale clause of I11, write in §2's recipe, the read points (every 25M + plateau) and the §2.4 stop rule, and declare the K9(b) handling; (b) a test: heads ON vs OFF learn bit-identically under bf16 autocast; (c) the K9(b) excluded share on EVERY update (one jsonl line per update, beside TB) | `measurements/ridealong_baseline/PREREGISTRATION.md`, `ridealong_update_test.py`, `rust_rollout/consistency.py` | opus-high | 4–6 h | owner Q1, Q4 for the recipe lines | NEW. X26's EXPERIMENT_BACKLOG row points here |
| **T33** | **Tier-1 validation (P-A, U9a's deferred item):** operate `main.plateau tick` / `status` on T-ENT (Δ 5M, lag 4) and on R-60 at 60M; time a CPU tick beside a live run; record the CPU minutes per check; report any CONFLICT / INCONCLUSIVE as a finding. Operating plus a ledger entry, not a build | `main/plateau.py` (read only unless a bug) | opus-medium | ~2 h of work, spread over the toy runs | G1 at 25M | NEW (U9a's deferred validation had no row) |
| **T34** | **The T-MLP build** (C-MLP): a new extractor mode, flat obs → MLP → 10 action tokens → the same pointer and critic contract; the registry row and the five synced surfaces; a `MODEL_CONFIG_VERSION` bump and migration; `tier_contract`; guard the ~30 sites that assume `team_transformer` / `pokemon_encoder` exist (trunk precision, rank metrics, grad balance, capacity probes, the prober, compile inventory); ARCHITECTURE / CHANGELOG; tests. Production byte-identical (K9 golden unchanged) | `agents/model/extractor_*`, `flag_registry.py`, `model_version/`, the telemetry sites | opus-high | 10–16 agent-hours, plus a 20-min leased GPU check | T31; rebase onto each landed T30 unit (same files); if the conflict churn grows, wait for U3 | NEW (audit F24's C-MLP) |
| **T35** | **The maturity-battery harness, generalised:** a generic "checkpoint at its pin + pad" entry (no hardcoded {end2, prod}); the plateau-drawn 800-state bank sampler (frozen by id and hash); the R-MATURITY trajectory rules as code; a trigger that queues the battery at every 25M checkpoint of a named run, under rule 5b | `designs/research_state/measurements/hidden_info_tier1_2026-10-13/` → a `src/main/` module | opus-high | ~1.5 agent-days | — (needed by X26's 25M read, which is offline) | NEW (`design_hidden_info_binding.md` §12.13.7 has no backlog row) |
| (T20 / U5 + U9) | **The cycle monitor (offline only) and the two-tier plateau decision** | `design_evaluation.md` §10 rows 9–10 | opus-high | ~2.5–3 agent-days | owner Q2 (Q8) | EXISTING T20 (+ U9 in `design_evaluation.md`): no new row; pulled forward to land during X26's first ~100M |

## 6. Risks and the owner's questions

### 6.1 Risks (operational; owned by the orchestrator)

- **R1. The deletion chain slips.** U0 → U1 → U3 is ~3 agent-days on paper, and a routine gate on the 3 agent cores
  takes hours. *Mitigation:* the targeted set + static tier per unit; the GPU fills with toys and R-60 meanwhile.
- **R2. Cross-deletion loadability (§3.5).** *Mitigation:* the U0 vacuity requirement, else the EB bridge.
- **R3. A K9(b) spike stops X26.** *Mitigation:* the declared Amendment 1 / 2 handling, and per-update logging.
- **R4. The 4-day `data/` freeze** collides with the info-levers or hidden-info work. *Mitigation:* announce it;
  those sessions stage `data/` changes on a branch.
- **R5. Tier 1 misbehaves on its first real use.** *Mitigation:* T33 runs it on the toys first.
- **R6. A toy needs the pre-deletion code for a later re-run.** *Mitigation:* the `pre-legacy-deletion` tag (U0).

### 6.2 Questions for the owner (strategic only)

- **Q1. One X26 seed or two?** At ≈ 50–90 GPU-h each, two seeds add about 3–4 days of GPU time before X26's
  plateau can be read. Your rule (2026-10-11) is that a research-path claim needs replication on a PLATEAUED
  model, and §3.5's plateau read has no production seed variance with one seed.
  - *Recommendation:* launch seed 1. Start seed 2 at seed 1's first Tier-1 FLAT, not at its stop, so the
    replicate is already running when the claims come due.
- **Q2. Does "the plateau meter before the deep run" mean the full two-tier meter, or the rule plus a validated
  Tier 1?** The full meter is U5 + U9, about 3 agent-days more, and U9 needs your still-open answer to Q8 (the
  two-tier test, `design_evaluation.md` §11.2).
  - *Recommendation:* launch with the rule plus a validated Tier 1. The interim rule (§2.4) can only CONTINUE until
    every signal agrees, so the only cost is running X26 a little past its plateau. U5 and U9 land during its
    first ~100M. Answer Q8 as orchestrated (two-tier, Tier 1 primary).
- **Q3. The legacy deletion manifest's D-LD1–D-LD7 are still PROPOSED** and gate X26 (U3).
  - *Recommendation:* adopt the manifest's recommendations as written: per-lever refusal (no floor raise),
    DERIVED-on, delete `static_recovery` in U0, no bisection, model-only `turns_since_progress`, keep
    `--trunk-layers`, accept the init-stream break once in U3. Add this doc's vacuity rule to D-LD1.
- **Q4. Should `--promotion-sprt` and the mirrored-pair eval flip ON at X26, as decided on 2026-10-01?** Both
  change how the self-play pool is filled, so X26 would no longer be production's recipe at depth.
  - *Recommendation:* keep both OFF for X26, so its plateau is production's plateau. SPRT promotion becomes a
    registered challenger that must beat that plateau, consistent with the funnel.

## 7. FINDINGS (standing rule 7)

- **F1. GIGO: the `--unified-moves both` umbrella silently overrides a typed `--damage-op false`** (and
  `move_prior_fusion`, `damage_outgoing`), while `checkargs` reports "every ARCH-surface key matches"
  (`main/train/config.py:257-260`). The "explicit flags win" guarantee is false for those keys → T31.
  **FIXED 2026-10-15 (T31):** refused, typed — the Decision record row says why a refusal and not "typed wins".
- **F2. The ride-along registration's literal argv fails `checkargs`** (5 deleted constants), and nine of its
  clauses are stale (I11) → T32.
- **F3. Tier 1 has never run on a real run**, and its CPU cost beside a live run is unmeasured → T33.
  **RESOLVED 2026-10-15 (T33):** Tier 1 ran on `rb_toy_ent_s4001` with every check decided and no CONFLICT. Its cost
  is in §2.6, and its FINDINGS are F13–F17.
- **F4. The legacy deletion has not started**, and its owner decisions are open, though the closing test's
  adoption ordered it on 2026-10-12. It is now X26's critical path → T30, Q3.
- **F5. The K9(b) excluded share is logged on ~1 update in 10**, so a per-update maximum cannot be read after the
  fact → T32(c).
- **F6. No head bit-identity proof under bf16** (`ridealong_update_test` is fp32) → T32(b).
- **F7. Stale docs:**
  - `era_plan_post_m5.md` row 0 says X26 has "KL early stop ON" (decided OFF on 2026-10-03).
  - The 10-06 roadmap's "eval-design U4 scheduler" is not a prerequisite of a pinned X26 (F-ED-19).
  - "X26 continues from the winning arm's seed" is superseded by §2.1.
  - All three are corrected in this commit: `era_plan_post_m5.md` row 0 now reads KL early stop OFF (with a
    Decision-record row), and the second and third are in this doc's Decision record.
- **F8. Snapshot-ladder coverage:** only 9 of ~90 rb_ runs have `snapshot_ladder/`, because every reference run
  typed `--snapshot-ladder-games 0`. X26 must not, or the interim Hodge read does not exist (§2.2).
- **F9. No automatic checkpoint pruning exists for rb_ runs.** It is not needed for X26 (≈ 36 GB at 500M), and
  nothing pins a Tier-1 node while a request is open (U4b), so no pruner may be introduced while X26 runs.
- **F10. The toy throughput is an estimate** (3,000–5,000 fps, UNVERIFIED). G1 measures it.
- **F11. A restart's inheritance of a toy's typed zero coefficients is UNVERIFIED** → §5.2 item 5.
- **F12. The launcher dry-run refuses from an agent worktree without `deps/pokemon-showdown`** (an environment
  refusal, not an architecture one). Validate X26's argv from the main checkout.
- **F13 (T33-1). The production net plays 0.57–0.67 games/s on the contended agent cores, not §8.1's 4.8.** So a
  near-plateau Tier-1 check takes 1.5–5 h there, not the design's ~11–42 min (§2.6).
- **F14 (T33-2). The meter is tail-bound.** `main.plateau` plays one 40-pair batch (80 games on 64 envs) per engine
  call and waits for its slowest game. One 160-pair batch ran at 1.16 games/s against 0.62, and at 1.5 CPU-s per game
  against 2.6. Fix: keep several batches in flight and judge them in batch order. The sequential rule is unchanged,
  and an overshoot costs only games. This is offline code (~0.5 agent-day), not yet a backlog row.
- **F15 (T33-3). The trainer's blocking eval cycle costs ≈ 3.5 % of production wall, not ≈ 1.5 %.**
  `rb_bf_eb_s2001` shows a 43–47 s cycle every 2M steps (≈ 1,270 s). The toy shows 16 s every ≈ 380 s, which is 4.2 %.
  `designs/training/eval_and_rating.md` quoted only O9's ~1.5 % at N = 256. It now gives both figures, corrected in T33's commit.
- **F16 (T33-4). The per-phase CPU and GPU utilization of a PRODUCTION run is in no record.** §2.6 bounds the GPU's
  idle time from the inference waits, and sampled only the toy. The procedure is read-only, needs no GPU window and
  takes ≤ 10 min (§2.6).
- **F17 (T33-5). `rb_toy_ent_s4001` died at 08:24 on a `CompileCanary` FATAL after update 700**, at about 72M. The
  compiled bf16 TRAIN graph's gradient on `pointer_head.move_proj.{weight,bias}` was 0.40 from the fp32 reference,
  against eager's 0.028, which is beyond the 12× bar. It was confirmed on an independent slice. This is FATAL_CONFIG,
  and the run was not restarted.
  - The safe rollback point is 62.0M (the last passing canary was at 62.83M).
  - Tier-1's 65M and 70M checks used post-canary nodes, so their GAIN verdicts are VOID pending the training
    session's ruling.
  - A resume IN PLACE from 62M would write new 63M+ checkpoints into the same `checkpoints/` folder. The decided 65M
    and 70M requests froze their nodes' sha256, so `main.plateau` would then refuse the check, or plan a different
    node. **Resume into a new run directory**, or the toy's later checks need `--lineage-run`.
  - The meter played on the agent cores (CPUs 5–7 and 13–15, nice 19), and the trainer is pinned to 0–4 and 8–12.
    The canary is a numerical check, so contention cannot produce it.

## 8. Decision record

| date | decision | rejected | evidence |
|---|---|---|---|
| 2026-10-15 | **X26 is a FRESH seed at the exit-check commit P_X26**, not a continuation of an A/B seed (supersedes 2026-10-03 "X26 continues from the winning arm's seed-1001 run") | continuing an EB seed | §2.1: U3's init-stream change (D-LD7); the 10-14 learnset fix changed priors; the fork hazards; +2.6 GPU-h only |
| 2026-10-15 | **X26 on cu126** (`gen3ai_torch28`); cu128 waits for a clean experiment boundary after X26 | switching for X26 | the local cu126-vs-cu128 baseline is PENDING; the launcher refuses a cross-build resume |
| 2026-10-15 | **The plateau meter's launch requirement = the stop rule registered + Tier 1 validated on real checkpoints**; U5 + U9 land during the run; the U4 scheduler is DROPPED from X26's prerequisites (proposal; owner Q2) | holding X26 for the full two-tier meter | every component is offline over retained checkpoints (F-ED-19); the interim rule can only CONTINUE |
| 2026-10-15 | **Checkpoint cadence 1M, snapshot ladder ON (default 100), cap 500M** for X26 | the 2.4M default; ladder 0 | Tier-1 nodes need a checkpoint in [grid, grid + 500k); the Hodge stand-in needs `snapshot_ladder/` |
| 2026-10-15 | **The toys: T-ENT (flags, no build) now; T-MLP built (T34); C-NOCELL dropped** | a one-line `--arch toy_ent` entry; trunk depth < 2 | the umbrella reverts 5 keys under a bare arm; `TRUNK_LAYERS_CHOICES` (2, 3, 4); move resolution retired the seven blocks |
| 2026-10-15 | **T-ENT keeps the four observation-FACT levers at production** (differs in machinery only) | the full `toy_min` (facts off) | the fact-vs-judgment test: facts save capacity and stay; the deletion makes those four unconditional anyway |
| 2026-10-15 | **X26's Tier 1 ticks at the REGISTERED Δ 10M / W 50M on the offline CPU lane, and no in-process or SCHED_IDLE eval is built for X26** (PROPOSAL, T33 → the orchestrator; owner question "evals on the main process, using the spare cycles?") | an in-process blocking Tier-1 lane (c) in P_X26; an out-of-process SCHED_IDLE filler (b); Δ 5M for X26 | §2.6: at Δ 10M with pipelined batches the lane keeps up through the hardest mean case (73 vs 106 min). (c) saves ≲ 1 GPU-h per seed and would sit on the critical path, since a pinned run cannot gain it later. (b) risks the trainer (rule 5b's −25–30 %), has no GPU path, and competes with X14 for the same idle CPUs |
| 2026-10-15 | **The toy stop rule:** Tier 1 at Δ 5M / lag 4, two consecutive FLAT, min 30M, cap 100M | a fixed 15M budget (the audit's draft) | the owner wants each one's OWN plateau; the scaled Tier 1 doubles as T33's validation |
| 2026-10-15 | **The comparison is pre-registered (§3.5–§3.6)** before any toy seed: G_m at matched 15M against the 8 EB seeds, G_p at plateau against X26, the CEILING / SPEED ONLY / NOTHING MEASURABLE classes at the ±2.0 pp bar | — | the closing test's estimator and bar |
| 2026-10-15 | **The exit-check sanity screen (E7):** 2 seeds × 5M at P_X26 against the 8 EB 5M finals, PASS iff pooled Δ̂ ≥ −3.0 pp and neither seed < −6.0 pp | a single seed; a full 8 × 8 equivalence test | a breakage screen after an init-stream change, not an adoption; seed SD ~3 pp at 5M |
| 2026-10-15 | **R-60, the production scout**, is pre-declared as `rb_bf_eb_s2001` (the lowest seed index) | choosing the strongest EB seed | a selection by strength would bias the matched-step references |
| 2026-10-15 | **T31: an umbrella that would overwrite a sub-flag already set to a DIFFERENT value REFUSES (`UmbrellaOverride`, FATAL_CONFIG), it never resolves silently.** Audit of every umbrella in `main/train/config.py`: `--unified-damage` → `damage_op` / `move_prior_fusion` / `damage_outgoing` (the bug), `--unified-moves` → a typed `--unified-damage off` (the same class, hidden by an `"off"` default, now None), `--damage-matrices incoming` → a typed `--damage-topk 0` (silently raised to 5); every other umbrella write was already "only if None" (`move_latent`, `move_belief_latent_coef`, `damage_topk_k`, `move_belief_mode`, `--arch`'s surface, the recipe, `--debug`'s shape). The coefficient auto-zeroes in `resolve_config` (`hp_type_belief_coef`, `item_belief_coef`) are printed and zero an INERT dose, not an umbrella | "the typed value wins with a note": the umbrella's other writes (`--damage-topk`, the outgoing block, prior fusion) then form a combination `combination_checks` refuses under a message about a DIFFERENT flag, and an ambiguous intent should not be guessed | `src/main/train/umbrella_override_test.py` (14 tests, 9 fail on revert); toy argv `checkargs` byte-identical |
| 2026-10-15 | **T30-U0 shipped; D-LD1–D-LD7 adopted at their recommendations** (orchestrator default under Q3's 15-minute rule; owner may override before U3 lands). The U0 vacuity requirement (§3.5) is MET: R-T2 can play a pre-deletion T-ENT checkpoint at post-deletion HEAD, so the EB bridge fallback is not needed | the EB bridge | `adopted_levers_test.py` (a real toy build passes the config and zip judges with all four units in force) |
| 2026-10-15 | **§1.1's knock-in-fork data point re-read per seed** (the hidden-info study's §8.7 read) | the per-fork table (median / max excluded share, messages over 0.15) from each fork's own log; every fork strict-by-read PASS; the spreads reveal lowers the share; both mechanisms UNVERIFIED | the one-line range alone (it said 0.006–0.036 for the other arms; the spreads medians read 0.005) | `bank_hib.py` on the 15 `rb_hib_*` forks; `design_hidden_info_binding.md` §8.7 |
| 2026-10-15 | **T30-U1 landed (config v156): the eight op levers are flagless; the T-ENT argv no longer parses at HEAD (expected, D-LD4), and T-ENT checkpoints keep loading through the vacuity rule.** An absent deleted field in a config written at v156+ reads ADOPTED (`adopted_levers.absent_value`, a fix made in U1: U0's rule would have refused the production mirror itself) | keeping the eight flags until the toys finish | `adopted_levers_test.py`; the manifest §1.4 D1–D6 |
| 2026-10-15 | **I28: the metric-key collision CLASS is closed in the logger, not by renaming keys** — the stdout table's column is 48 (every production row prints in full) and a longer row's label is `<head>~<digest>~<tail>` (collision-proof; a residual clash prints the full key, never raises) and a colliding key set is refused at STARTUP; TensorBoard tags stay verbatim | renaming the six `cell/move_resolution_*` keys (dashboards read TB tags by name, and the next long key would crash again); a wider column alone (still a mid-run raise past it) | ledger 2026-10-15 CRASH FIX; `designs/training/telemetry_scalars.md` "Metric keys stay DISTINCT" |
| 2026-10-15 | **I29: the one-producer class is closed in the logger's record path, strict where a stop is free** — a second call site writing a pending key raises `DuplicateProducerError` under pytest and `--debug`, and is counted (`telemetry/duplicate_producer_keys`) + reported once per key in a production run, which never stops on it | raising in production (one wrong number is cheaper than a run); an exemption list (every entry is a hidden bug) | ledger 2026-10-15 GIGO FIX; `designs/training/telemetry_scalars.md` "ONE producer per key" |
