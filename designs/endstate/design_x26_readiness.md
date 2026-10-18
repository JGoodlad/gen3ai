# The road to X26: readiness inventory, the toy controls, the exit check and the GPU schedule

🚨 **ALWAYS-CURRENT (an `endstate/` doc).** A decision or build that differs from this plan updates it in the
same commit, saying what changed and why. It ends with a **Decision record**.

**Status: EXIT CHECK ASSEMBLED 2026-10-15 15:45 PDT at `87e493c3` (§4's status column, the launch plan §4.1, the
bf16 headroom watch §4.2).** Built since this plan was written (inventory read at `9ada3604`): the legacy deletion's
X26 path (U0 / U1 / U3), T31, T32 (Amendment 2), T33 (Tier 1 validated), T36 (the canary's clip-margin fixture,
CONFIRMED on CUDA at `87e493c3`), the I28 / I29 fixes and the post-U3 test sweep. T-ENT: s4001 died at 72.66M (I30),
s4002 reached its 100M cap, s4003 is training (to about 20:45 PDT). **Seven of twelve exit rows PASS; E3 is
DEFERRED; E5, E6, E7 and E11's data-freeze announcement are OPEN.** Every open row needs P_X26 to exist (§4.1).
X26 has not launched. The owner's roadmap (2026-10-09, after the closing test and the perf phase) has three steps
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

- **The legacy deletion's X26 critical path has LANDED.** The manifest
  ([`../ops/legacy_deletion_manifest_2026-10-12.md`](../ops/legacy_deletion_manifest_2026-10-12.md)) has D-LD1–D-LD7
  ADOPTED (the orchestrator's default, 2026-10-15); **U0, U1 and U3 have LANDED** (2026-10-15; U1 = config v156, the
  eight op levers flagless; U3 = config v157, move resolution derived, the seven per-action blocks and the critic's
  threat injection deleted). 🚨 U3 CHANGED the fresh-seed random stream (D-LD7), so every X26 seed starts at or after
  it, and no pre-U3 seed is "the same seed" at HEAD.
- **The plateau meter is partly built.** Tier 1 is BUILT and run offline (`main.plateau`, `131f3412`). It is
  VALIDATED on a real run (T33, 2026-10-15, on the T-ENT toy), and its cost is measured (§2.6). On the agent cores,
  an X26 near-plateau check costs 1.5–5 h today. The proposal (§2.6) is that X26 ticks at the registered Δ 10M,
  the meter's batches get pipelined (DONE, T38 2026-10-16: `--inflight-batches`, default 4; 2.0× the production
  net's game rate, the rule unchanged; §2.6), and no in-process eval is built for X26. The cycle monitor (U5), the outside panel and the two-tier decision (U9) are NOT
  STARTED, and U9 waits on the owner's Q8. Every meter component runs OFFLINE over checkpoints, so the deep run
  needs only the stop RULE registered and Tier 1 validated at launch (§2.4, owner Q2).
- **The ride-along heads still attach** to the end state. They have already ridden on 32 end-state runs. Their
  registration's **Amendment 2 is written** (T32, 2026-10-15): X26's argv passes `checkargs` and the launcher dry-run
  at HEAD, the heads are proven bit-identical-neutral under the bf16 trunk, and the K9(b) handling is declared. Two
  registered reads (R1×R2, R5) still need reader code before X26's FINAL read (offline; not a launch blocker).
- **The toy controls need no build for the first toy.** T-ENT (entity tokens and attention only) launches today
  from flags (§3.2; `checkargs` exit 0, verified for this doc before U1 — since U1 the argv runs PINNED, §3.2). T-MLP (a flat MLP) needs a build of about 1.5
  agent-days.
- **The toy does NOT plateau by its 100M cap** (§3.8): T-ENT s4002's 16 Tier-1 checks, 25M–100M, all read GAIN
  (52.6–53.8 % against the node 20M back from 80M on), so it is a `CAP` stop, and s4001 died at 72.66M still
  climbing. §3.5's "plateau against plateau" becomes production at plateau against a toy at its cap (F32, F33).
- **The T-ENT seeds:**
  - s4001 died at 72.66M on the canary FATAL (I30).
  - s4002 reached its 100M cap at 14:44 PDT 10-15 (about 5.7 GPU-h, ≈ 4,900 steps/s).
  - s4003 started at 15:01 PDT. All three are pinned at `3899277a`, which predates T36 (FINDING F25).
  - T-MLP (T34) is not built, and it no longer goes before X26 (§4.1).
- **BLOCKER I30, CLOSED 2026-10-15: the bf16 canary FATAL on the toy `rb_toy_ent_s4001` at 72.66M.** Diagnosed on
  CPU as a false alarm built into the canary itself: the PPO clip on its fixture rows is a decision boundary, so
  two healthy arms can disagree 30×. Not shown to be a miscompile. **The fixture fix T36 is BUILT**
  (`gen3_r1_clip_margin_v1`, 2026-10-15: boundary rows relocated on every rung, the margin checked, the confirmation
  on two disjoint halves; at the FATAL's weights healthy pairs above K fall 21 % → 0 of 992 on CPU). **CONFIRMED on
  CUDA at `87e493c3` (2026-10-15):** `verify_compile_gate_cuda` PASS 4/4 and `diag_bf16_canary` on `rb_toy_ent_s4001`
  with all five criteria met (I30). I30 is CLOSED. Its remaining exposure, the compiled arm's log-prob shift growing
  with training, is WATCHED in-run (§4.2).
- **The X26 launch estimate is now Fri 10-16, about 00:00–01:00 PDT, at the EARLIEST** (§4.1). That assumes P_X26
  lands before toy s4003 ends, at about 20:45 Thu. After that, the GPU needs about 2.75 GPU-h at P_X26 (E7 with
  E6 riding on it, plus the crosses). The production scout R-60 (G3) is DROPPED (§4.1). The deletion chain no longer
  sets the date; P_X26 does.

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
| I7 | **Legacy deletion U3** (`--move-resolution off` and the seven per-action blocks; `--value-threat-inject`). **It changes the fresh-init random stream (D-LD7)** | **LANDED 2026-10-15** (config v157: `--move-resolution`, `--move-resolution-facts`, `--value-threat-inject` and the seven block flags deleted; move resolution DERIVED from its prerequisites; the trained network's shape unchanged; the fresh-init stream CHANGED) | manifest §0 D-LD7, §3, §8; the ledger's U3 entry; `research_state/measurements/legacy_deletion_u3_init_2026-10-15/` | **L** | the K9 golden re-recorded once |
| I8 | Legacy deletion U2, U4 (observation levers; the legacy token encoding) | NOT STARTED | as I5 | no, **if** each proves production's fresh init is byte-identical before and after | **DEFERRED past P_X26 (2026-10-15).** They may land DURING X26, which is pinned and never runs them, but only under §4.1's four conditions. Every offline reader of X26's checkpoints runs at HEAD, so the conditions add a FORWARD-identity proof on a v157 production checkpoint |
| I9 | Legacy deletion U5 (optional Rust progress clock), U6 (close-out) | NOT STARTED | manifest §3 | no (U6 waits for X26's argv, by its own text) | — |
| I10 | **The ride-along heads attach to the end state** | DONE | `checkargs --argv "--arch production … --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all"` exit 0; the launcher dry-run from main "would launch"; B rebased onto the flat pointer (`350fb83b`, refused without it at `ridealong_heads.py:280-310`); RND reads `obs_dim` at build (2845); all 16 bf16 A/B runs and the 8 `rb_es_end2_*` typed these flags (`metadata.json`); `ridealong/disabled` max 0.0 in `rb_bf_eb_s2005`'s TB | L | — |
| I11 | **The ride-along registration** ([`PREREGISTRATION.md`](../research_state/measurements/ridealong_baseline/PREREGISTRATION.md)) | **DONE 2026-10-15 (T32, `ed2b9f82`): Amendment 2** — every stale clause below fixed (its §A2.1 table), X26's argv with every non-default flag and why, `checkargs` exit 0 + launcher `--dry-run` "would launch" at `264a35e3` pasted, R1–R5 + Amendment 1 restated on the end state, read points 10M / every 25M / plateau / FINAL, the §2.4 stop rule, the K9(b) handling, the Q1 / Q2 / Q4 defaults. Was: STALE | its literal argv fails `checkargs` (exit 1: `--critic winprob`, `--terminal-indicator`, `--victory-value 1.0`, `--draw-penalty 0`, `--env-core rust` are deleted constants); restart interval 3 h (now 6 h); the K10(a) recipe clause; "FIRST GPU run on the M5 core"; read points 10M / 25M / FINAL; the overhead basis 67.05 s; the checkpoint cadence "50,000 vec-calls × 48 envs"; B on α's support; the 2761-dim memory estimate; `feat`'s drift prediction from fp32 legacy features | L | **Amendment 2** (unit T32), committed before the launch |
| I12 | **Head bit-identity under bf16** (heads ON vs OFF learn bit-identically) | **DONE 2026-10-15 (T32, `ed2b9f82`)** | `ridealong_update_test::test_BF16_one_update_is_BIT_IDENTICAL_with_the_heads_on_and_off`: the learner at bf16 (rollout fp32), heads OFF vs ON with every variant, bit-identical params / PPO optimizer / scalars / RNG; non-vacuous (bf16 classes; the bf16 update differs from the fp32 one). Mutation-checked: dropping `RideAlongBatch.detached`'s `.detach()` fails it, and so does a ride-along step that leaves the trunk at fp32 (invisible to the fp32 test) | L | — |
| I13 | **Plateau meter, Tier 1** (newest vs W-back GSPRT, H0 0.50 / H1 0.52, Δ 10M, W 50M, cap 6,000 pairs) | **VALIDATED on a real run (T33, 2026-10-15)**: 10 checks (25M–70M) on `rb_toy_ent_s4001`, all GAIN, with no CONFLICT or INCONCLUSIVE; status CLIMBING (the run died at 72M on a CompileCanary FATAL, F17) | `src/agents/training/plateau_t1.py`, `src/main/plateau.py` (`131f3412`); the `plateau_t1:rb_toy_ent_s4001:*` requests and decisions in `models/_ledger`; §2.6 | L (validated) | its CPU cost is MEASURED (§2.6): production net 0.62 games/s on the agent cores today, so a near-plateau check takes 87–137 min against a 53-min Δ 5M cadence. **Use the registered Δ 10M for X26 and pipeline the 40-pair batches (T33-2)** — pipelined: **T38 DONE 2026-10-16** (`main.plateau tick --inflight-batches`, default 4: production 1.03 → 2.07 games/s on the same 640 games, §2.6) |
| I14 | **Plateau meter, the cycle monitor** (U5) | NOT STARTED | no `src/main/cycle_monitor*`; `monitor_row` exists only as a ledger kind | later (before the first Tier-1 FLAT can stop X26) | ~12–16 agent-hours, offline only (TASK_BACKLOG T20) |
| I15 | **Plateau meter, the outside panel + the two-tier decision** (U9) | NOT STARTED; the owner's Q8 is unanswered | `design_league_decisions.md` L110 unchecked; `design_evaluation.md` Decision record 2026-10-03 "Q8 … provisional" | later | Q8 (owner Q2 below); ~8 agent-hours |
| I16 | **U4 scheduler** (the in-window eval lane) | NOT STARTED / PAUSED | `design_evaluation.md` §7; `design_x5_belief_tokens.md` "eval U4 is paused" | **no — DROP? from X26's prerequisites.** X26 is pinned, so its checks run through the offline driver anyway (F-ED-19) | — |
| I17 | **The X26 recipe** (§2) | PARTIAL: production resolves; the run-specific choices are this doc's proposal | dry-run: FRESH, restarts every 6 h, 3 crash restarts | L | the owner's Q1 (seeds) and Q4 (promotion), then Amendment 2 |
| I18 | **The checkpoint cadence for Tier 1** | DECISION | the default `--checkpoint-every-steps` is 2.4M (`hyperparameters.py:79`). Tier 1's node must fall in [grid, grid + 500k), so a 2.4M cadence misses most grid points (`MISSING_NODE`) | L | type `--checkpoint-every-steps 1000000` (as the closing test and the bf16 A/B did) |
| I19 | **The K9(b) tie-share margin** (the behaviour-check's excluded share; ceiling `FP32_EXCLUDED_CEILING` 0.15, single-shot FATAL under the default `--behaviour-check fatal`) | MARGIN OK for the MEAN; **the handling DECLARED and the per-update record PINNED — DONE 2026-10-15 (T32, `ed2b9f82`)** | §1.1 below; Amendment 2 §A2.7 (the closing test's Amendment 1 / 2, adapted: a tie-only stop resumes without limit, then `warn` once the crash budget is spent, any judged \|d log π\| ≥ 1e-4 is a GIGO stop). The share was ALREADY on every update in TensorBoard (F5 was wrong: 153 of 153 updates in `rb_bf_eb_s2005`'s raw events); `diagnostics_cadence_test::test_the_K9b_excluded_share_reaches_TENSORBOARD_on_EVERY_update_not_on_the_diagnostics_cadence` now pins it; read it with `main.ops.tb_read` (the TB web UI keeps 1,000 points per tag, ≈ 1 in 5 at X26's length) | L (as a declared handling) | — |
| I20 | **Disk** | OK | `df`: 341 GB free of 1003 GB (330 GB at the exit check, 2026-10-15 15:10 PDT); an end-state 15M run is 4.2–4.8 GB, of which 3.2–3.3 GB is a fixed compile cache and 55 MB per checkpoint | L (≥ 100 GB free at launch) | no pruner exists for long rb_ runs (none needed: §2.3) |
| I21 | **The plateau battery's harness** (the per-stage VOI + belief-quality dose re-run on X26's checkpoints, `design_hidden_info_binding.md` §12.13.7) | **BUILT 2026-10-18 (T35)**: a generic series / pin (+ pad) entry, the 25M trigger and the read-only checkpoint copy (`maturity_series.py`), the frozen bank sampler by id and hash (`maturity_bank.py`), R-MATURITY as code (`maturity_rules.py`), LEAN / FULL modes, a detached resumable queue | was: `t1_common.py` CHECKPOINTS = {end2, prod}; no bank sampler, trigger or rules as code. Remaining: the PLATEAU bank (drawn when X26 plateaus, §12.13.7's registration) | later (first read at ≥ 25M; it reads retained checkpoints offline, so it can land during the run) | T35 (~1.5 agent-days); cost FULL ≈ 47 CPU-h / LEAN ≈ 14 CPU-h per checkpoint, no GPU |
| I22 | **Own PPO loop stage 2** ("lands before the X26 ride-along baseline") | DONE | `design_own_ppo_loop.md` status: stage 2 and stage 3 BUILT | — | — |
| I23 | **The `--unified-moves` umbrella silently overrides a typed `--damage-op false`** (and `move_prior_fusion`, `damage_outgoing`) | **DONE 2026-10-15 (T31)**: the conflict is REFUSED (`config.UmbrellaOverride`, FATAL_CONFIG; `checkargs` prints the same refusal and exits 3); `--unified-damage` now defaults to None so a typed `off` is seen; a typed `--damage-topk 0` beside an incoming matrix is refused too. The toy argv's `checkargs` report and resolved namespace are byte-identical before / after; so is `--arch production`'s | `src/main/train/umbrella_override_test.py` (fails on revert) | before T-MLP and any later ablation; not X26 (production sets them on) | T31 (~2–3 h) |
| I24 | **The `data/` freeze** (standing rule 6: a pin isolates code, not data) | DECISION | root `CLAUDE.md` rule 6; memory `feedback_pin_isolates_code_not_data` | L | **No `data/` change for X26's whole run (≈ 3–4 days at 500M).** Announce it at launch |
| I25 | **The training-input changes since the reference seeds**: the learnset (legal movepool) fix `gen3_learnset_prevo_chain_v1` (2026-10-14: priors and X5 candidates move, the observation does not) and the fork-recipe fix (2026-10-14) | DONE at HEAD | ledger 2026-10-14 GIGO FIX ×2 | — | none for a FRESH X26. They are why X26 is not a continuation of an EB seed (§2.1) |
| I26 | `--promotion-sprt` and `--eval-mirrored-pairs` "flip at X26" (2026-10-01) | DECISION OPEN | `design_model_management.md:38`, `design_league_decisions.md:32`, TASK_BACKLOG T6 / T17 | L (as a decision) | owner Q4 |
| I27 | The KL early stop for X26 | DECIDED OFF (2026-10-03); `era_plan_post_m5.md` row 0 said "KL early stop ON" (STALE; corrected in this commit) | `EXPERIMENT_BACKLOG.md` X5 decisions | — | — (finding F7) |
| I28 | **The metric-key collision crash**: every production run at a commit ≥ `65b71f6a` died at its FIRST post-update dump (step ~202k on the GPU; the training session's fresh `--arch production` CUDA smoke at `6f6b7f90`): sb3's stdout table cut `cell/move_resolution_restored_move_{weight,grad}_norm` (and the `restored_switch` / `status_switch` pairs) to one label and raised | **FIXED `109c4256`** (`gen3_metric_key_collision_v1`, ledger 2026-10-15 CRASH FIX): a collision-proof table + a STARTUP key check (`MetricKeyCollisionError`, FATAL_CONFIG) | `src/agents/training/train_logger.py`, `metric_keys.py`; the CPU production smoke reaches its dumps clean | L | DONE: the Training session's production CUDA smoke PASSED at `ffe2961b` |
| I29 | **Silent double producers and a never-produced metric** (found by the I28 fix's survey): `fork/*`, `train/effective_batch` (+ `train/dose_rate` one decision stale under `--adaptive-batch`) and `lifecycle/cuda_streams_after_freeze` each had TWO producers (the later write won silently); `grad/species_belief_*` was never produced (the bank registered `species_belief_set`, the probe looked up `species_belief`) | **FIXED `af6bc0eb`** (`gen3_metric_producer_guard_v1`, ledger 2026-10-15 GIGO FIX): one producer per quantity + a logger guard (`DuplicateProducerError` under pytest / `--debug`; counted + reported in production) + `_R1_PROBE_TERMS` derived from the bank | `src/agents/training/train_logger.py`, `metric_producers_test.py` | L | none |
| I30 | **🚨 BLOCKER: the bf16 CANARY FATAL on trained weights** — the toy `rb_toy_ent_s4001` ended FATAL at 72,658,984 steps (update 700): the compiled bf16 learner's `pointer_head.move_proj.weight` gradient read 4.03e-1 from the fp32 reference vs eager bf16's 2.82e-2 (14× > K = 12), confirmed on both slices. A canary or startup-gate FATAL is FATAL_CONFIG (never restarted), so X26 (thousands of updates, a startup gate at every 6-hourly restart) cannot launch on a check that fires this way | **FIXED (T36, `gen3_r1_clip_margin_v1`, 2026-10-15) and **CONFIRMED on CUDA at `87e493c3`** (`verify_compile_gate_cuda` PASS 4/4; `diag_bf16_canary` on `rb_toy_ent_s4001`: the replay passes at the FATAL's weights (relocated ratio 0.655), the relocated null reads 0 for the Ebf pairs and for Cbf at all 8 checkpoints, no arm switches a relocated row (`relocated.branch_flips` 0 everywhere), `fp32_rule.over_bar` is empty, C32 sits within 4e-6 of E32, and Cbf's taken-action \|Δ log π\| peaks at 0.110 ≤ 0.18; `../research_state/measurements/x26_exit_check_2026-10-15/gpu_t36_87e493c3/`). **CLOSED**; the shift's growth is watched in-run (§4.2).** The fix: before judging, every rung of the startup gate and the canary moves each fixture row within `R1_CLIP_MARGIN` = 0.72 (log-prob; 4 × the largest measured arm shift 0.18, PROVISIONAL until the compiled arm's CUDA shift is read) of its switching clip bound to 2× the margin inside its active side, refuses by name an arm that moves a log-prob past the margin, and the canary confirms on two DISJOINT halves; CPU at the FATAL's weights: healthy pairs above K 212 / 992 → 0 / 992 (max 3.65×). THE DIAGNOSIS: the canary's per-parameter statistic contains a DECISION BOUNDARY: the PPO clip on the fixture's 64 OFF-POLICY rows. A row whose ratio sits within rounding of the clip boundary that switches its gradient carries its whole policy gradient in one arm and none in the other; at the FATAL's weights two rows sit 0.005–0.007 (log-prob) from theirs, eager bf16 moves log-probs by up to 0.08, and two HEALTHY eager arms read a ratio above 12 in 18 % of 992 dithered pairs. With the branch held fixed the error never exceeds 0.020; eager bf16 itself flips at 66M (0.29). Census: a boundary row (within 0.01) at 14 of the toy's 62 pre-suspect checkpoints and at 2 of 10 production-surface states (the bf16 A/B's seeds), eager flips at 4 / 62 and 1 / 10 — X26 can draw it, and the startup gate (every restart) is SINGLE-SHOT. The production baseline's CUDA PASS (ratio 1.55) is a state with no boundary row (nearest 0.074). Not shown to be a miscompile | [`../training/compile_flags.md`](../training/compile_flags.md) "The bf16 canary FATAL on trained weights"; `src/main/ops/diag_bf16_canary.py` (CPU reads); ledger 2026-10-15 FINDING | **L** | — (CLOSED 2026-10-15). Was: (1) ~~**T36**: the canary's fixture made rule-8-clean~~ BUILT 2026-10-15; (2) ~~the GPU window AT THE T36 COMMIT~~ DONE at `87e493c3` (the Training session under its lease): `scripts/ops/verify_compile_gate_cuda.sh` (all four CUDA tests PASS on the relocated rows) and `scripts/ops/diag_bf16_canary.sh rb_toy_ent_s4001` (the canary's own replay PASSES at the FATAL's weights; relocated null P(ratio > K) = 0 for Ebf pairs and Cbf; Cbf switches no relocated row; Cbf's max \|Δ log π\| ≤ 0.18; the fp32 rule clean). Fallback if the GPU contradicts: the fp32 trunk for X26 (+29 % `train_ms`) |
| I31 | **The X26 launch's `UpdateWontFit` (2026-10-15, `rb_xs_s2701` at `71351ff5`)**: the bf16 gate PASSED, then the update-fit check refused — demand 9,790 MiB reserved, headroom 978 < 1,024 — while the dry update peaked at 6,579 allocated. Cause (GPU-attributed): T36's relocation forward (fp32, grad mode, never backpropagated) left its whole autograd graph, 4,486 MiB, alive through the policy's stashes under the compiled arm (gate peak 5,428 → 8,244), and the allocator kept that peak cached into the reading; the mid-run canary carried the same stack (8,223 MiB at update 2) | **FIXED (this change; tested at `20c07e0b`, `gen3_gate_releases_memory_v1` + `gen3_update_fit_v2`, ledger 2026-10-16 CRASH FIX)**: the relocation keeps nothing for a backward, the gate hands its cache back after the lock, and the demand is the peak reserved over the dry update AND the canary's check from an emptied cache. GPU, the X26 argv: gate peak 5,480; demand 8,324 → **headroom 2,444 MiB**; the canary at update 2 PASSED at 5,458 allocated, reserved 8,324 = the run's reserved at every update (the prediction); 4 updates, no OOM; K9 golden IDENTICAL | `designs/training/compile_flags.md` "The gate releases its memory"; `learner_lifecycle.md` "The update fit check" | L | none — P_X26 moves to the landed sha |

### 1.1 The K9(b) tie-share margin, read from the end-state runs

**What K9(b) is.** K9(b) is the behaviour-consistency probe (`agents/training/rust_rollout/consistency.py`). Once
per update, before any optimizer step, the learner recomputes log π on rows played at the current policy version
and compares it with the stored behaviour log-probability.
- A row whose forward pass has a selection or threshold within a relative 2e-4 of its cutoff is a near tie, and is
  **excluded** from the comparison.
- The **excluded share** is capped at 0.15, so that a fault which pushes many rows onto ties cannot hide.
- One update over the cap stops the run under `--behaviour-check fatal`, the default, which every `rb_bf_*` and
  `rb_es_end2_*` run used.
- It is logged as the TB scalar `behaviour/excluded_frac` on EVERY update (independent of `--diagnostics-every`). An
  update is 98,304 rows, so a 15M run has ≈ 153 updates, and the raw events hold all of them (`rb_bf_eb_s2005`: 153
  of 153). *(Corrected 2026-10-15, T32: this line read "TB keeps about 152 of ~1,490 updates", a 10× error in the
  update count.)* The TensorBoard web UI keeps at most 1,000 points per tag, so at X26's ≈ 5,100 updates read the
  per-update series with `python -m main.ops.tb_read <run> --tag behaviour/excluded_frac`, never off the UI.

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
- The share is in the record on EVERY update (verified and pinned by T32), so a spike is visible rather than
  inferred.
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

**The pipelining, built and measured (T38, 2026-10-16;
`../research_state/measurements/t38_parallel_batches_2026-10-16/`).** `main.plateau tick --inflight-batches K`
(default 4) plays K 40-pair batches in ONE cycle; each batch keeps its own seed block and row, and the envs pull
single pairs across the wave. Agent cores under rule 5b (`taskset -c 5-7,13-15`, `nice 19`, `mem_cap`), load ≈ 5–6,
quieter than T33's 11–15, so the RATIO is the result, not the absolute rate:

| network | the same games at each K | K = 1 | K = 2 | K = 4 | K = 8 |
|---|---|---|---|---|---|
| **production** (`rb_bf_eb_s2001` 15M vs 10M, 8 batches = 640 games, one engine) | games/s · CPU-s per game (the Python process) | 1.03 · 1.88 | — | **2.07 · 0.94 (2.0×)** | 2.62 · 0.75 (2.5×) |
| toy (`rb_toy_ent_s4002` 65M vs 45M, the recorded check: GAIN at 17 batches) | games/s per wave · wall to the decision | 8.2–9.0 · 133 s ‡ | 12.3–15.2 · 97 s | **18.1–23.3 · 79 s (1.7×)** | 26.4–29.1 · 76 s |

‡ K = 1's 146 s less its one-time 13 s env-core build. Every K re-played the SAME games: on production every batch's
outcome digest and pentanomial were identical across K; on the toy every K re-played all 17 RECORDED batches with the
recorded digests and decided GAIN at 680 pairs, LLR +3.071067199, as on record (K > 1 adds K − 1 or fewer surplus
batches the rule never reads).
- **Why 4 and not 8:** a wave overshoots the stop by up to K − 1 batches. On a 3-batch GAIN check (production, early
  in a run) K = 4 plays 320 games at 2.07/s (155 s) against K = 1's 240 at 1.03/s (233 s); K = 8 plays 640 at 2.62/s
  (244 s) — slower than K = 1. Near the plateau (≈ 80 batches) K = 8 saves a further ≈ 20 % of the wall. Pass
  `--inflight-batches 8` for a check known to be long.
- **Applied to the table below** (an inference, UNVERIFIED at T33's contention): 2.0× on T33's 0.62 games/s gives
  ≈ 1.24 games/s, a little above the 1.16 column's assumption, so its conclusions hold.
- The pipelining also halves the CPU per game (1.88 → 0.94 CPU-s): the one-batch loop spent half its CPU waiting on
  each batch's tail, which rule 5b's shared cores pay for.

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
  blocks in the end state, and legacy deletion U3 deleted them.

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
  U3 (both have landed: the eight op flags, `--move-resolution`, `--move-resolution-facts` and the seven block flags
  are gone; the argv above parses only at its pin), and D-LD4 says any later ablation of an adopted lever runs PINNED at
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

**Measured (§3.8): neither seed plateaued under this rule; s4002 is a `CAP` stop at 100M, still reading GAIN.**

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
| **R-T2, plateau against plateau (PRIMARY):** each toy's **100M CAP checkpoint** (no toy plateaued; the amendment below) against X26's plateau checkpoint | 3 × 1 (3 × 2 if Q1 = two seeds) | 1,000 per cell | **G_p(cap)** = production at plateau − toy at its cap (biased toward production). With one X26 seed, production's seed variance is NOT in the interval (declared) |
| R-T3, external units: each toy plateau against Metamon SmallRL, greedy vs greedy, `main.anchors` | 3 | 200 games | descriptive; the regime stated with every number |
| R-T4, cost: GPU-h to plateau, params, FLOPs / row, measured fps | — | — | the price side |

> **Dated note, 2026-10-15 (the exit check), before any R-T read was played.** Two references changed. No rule,
> estimator or bar changed.
> - **R-T1b's EB 5M references:** only s2001–s2004 carry `final_model_stage1_5M.zip`. Seeds s2005–s2008 ran fresh
>   to 15M, so their 5M reference is `checkpoints/checkpoint_5000*_steps.zip`. The same substitution applies to E7.
>   The budget changes no schedule at P_bf16 (`design_bf16_strength_ab.md` §5), so a 5M checkpoint of a 15M run and
>   a 5M final are the same kind of state.
> - **R-T1c's production reference is X26 at 30M and 60M, not R-60.** R-60 is dropped (§4.1). X26 is the
>   post-deletion, post-learnset-fix production recipe, so it is a better matched-step reference than a pre-U3
>   continuation. It is still one production seed, and R-T1c stays descriptive.
> - **Load check at `87e493c3`** (strict loader, CPU, 2026-10-15; a SAMPLE, not all of them). These four load:
>   `rb_bf_eb_s2001/final_model_stage1_5M.zip`, `rb_bf_eb_s2008/checkpoints/checkpoint_5000214_steps.zip`, and
>   `rb_toy_ent_s4002`'s 15M checkpoint and its final model. The cells plan must load every reference before any
>   cell is played.

> **Pre-registration amendment, 2026-10-16: G_p(cap) (closes F32 and F33). Registered BEFORE any R-T2 read: no R-T2 cell has been played, and no R-T2 number exists.**
> The cause is s4002's Tier-1 ticks (`32589f9a`, §3.8): no toy seed plateaued by its 100M cap, so §3.5's "each toy plateau checkpoint" does not exist. This amendment changes
> WHICH toy checkpoint R-T2 plays and HOW its result may be worded; the estimator (`cross_stat`), the cells' pair count, the rule-8 band and the ±2.0 pp bar are unchanged.
> - **The toy's R-T2 reference is each seed's 100M CAP checkpoint:** `rb_toy_ent_s4001b/final_model.zip`, `rb_toy_ent_s4002/final_model.zip`, `rb_toy_ent_s4003/final_model.zip`.
> - **G_p(cap) = production at its plateau (X26's plateau checkpoint) minus the toy at its 100M cap.** It is a BIASED estimate, in production's favour, because the toy is still
>   climbing at the cap (§3.8: about +18 to +27 Elo per 20M from 80M on): G_p(cap) over-states G_p at each one's plateau by the toy's unrealised gain. The size of that bias is not measured.
> - **Reading rules.** (1) "CEILING" can only be stated as **"production's plateau beats the toy at 100M steps"**; it is never worded "at each one's plateau" and never as
>   "the machinery raises how strong the model gets". (2) "SPEED ONLY" and "NOTHING MEASURABLE" stay safe and keep their wording, because the bias can only widen the gap, so a
>   G_p(cap) interval inside [−2.0, +2.0] would hold a fortiori for the true G_p. (3) A G_p(cap) interval that is neither wholly above +2.0 nor inside [−2.0, +2.0] stays UNDECIDED. (4) The cap
>   checkpoint is used whatever the later reading of the toy's curve; the cap is not raised before R-T2.
> - **§3.6's owner sentence is re-worded accordingly** (below).
>
> **Descriptive record, 2026-10-16: as observed on the X26 road's W4 (`98cd814b`, `../research_state/measurements/x26_exit_check_2026-10-15/road/w4/result/`). NOT a rule, and NOT an R-T read of
> the registered kind.** The toy seeds' 15M and 5M checkpoints were played against the 8 EB seeds in the same window as E7's cross (mirrored pairs; the toy's mean win rate minus 50 pp,
> pooled over its 3 × 8 cells): **R-T1, toy@15M vs EB 15M ≈ −14.1 pp** (the three toy seeds −16.1 / −14.2 / −12.0); **R-T1b, toy@5M vs EB 5M ≈ −25.5 pp** (−25.1 / −26.9 / −24.6).
> **No interval was computed:** `cross_stat` refuses a non-square matrix (3 × 8; `CrossInputError: needs an n × n matrix with n ≥ 2`), so these are point means with no 90 % interval and no verdict.
> Both are many times the ±2.0 pp bar, but a number without an interval is not a classification. A rectangular-cross statistic is TASK_BACKLOG T40; it is needed before R-T1 can be read as G_m.
>
> **Descriptive record, 2026-10-16: R-T5 (EQUAL COMPUTE) and R-T6 (TOY AT CAP vs EB 15M).** These were pre-registered at `72350102`
> ([`../research_state/measurements/toy_crosses_2026-10-16/PREREG.md`](../research_state/measurements/toy_crosses_2026-10-16/PREREG.md)) before any game, and the results are in
> [`RESULTS.md`](../research_state/measurements/toy_crosses_2026-10-16/RESULTS.md) there.
> - **They are DESCRIPTIVE, not R-T reads of the registered kind.** No §3.6 class is read from them. They are not G_m, and R-T6 is NOT R-T2: its reference is production at
>   15M, not at plateau.
> - **The design:** 3 toy seeds × the 8 EB 15M finals, 250 mirrored pairs per cell, on CPU at main (`66e66237`'s `src/`), where all 14 references load.
> - **The equal-compute checkpoint** is the toy's 1M checkpoint nearest the step at which its own effective `learn()` wall clock equals EB's mean to 15M. That mean is
>   9,066.8 s (SD 170.2 s), and the checkpoints are s4001 48M, s4002 46M, s4003 47M. The toy is 3.33–3.44× faster per step at 15M, and slower later as its games lengthen.
> - **The interval** is a declared seeds-as-units rectangular statistic (`rect_stat.py`, Welch df). It is **not T40**. The W4 rows below pass through the same
>   statistic retrospectively.
>
> | read | toy checkpoint | toy − 50, pooled | 90 % interval | per seed (s4001 / s4002 / s4003) |
> |---|---|---|---|---|
> | R-T1b (W4) | 5M | −25.51 pp | [−28.04, −22.98] | −25.1 / −26.9 / −24.6 |
> | R-T1 (W4) | 15M | −14.10 pp | [−17.16, −11.04] | −16.1 / −14.2 / −12.0 |
> | **R-T5, equal compute** | 48M / 46M / 47M | **+6.48 pp** | **[+4.54, +8.43]** | +5.9 / +6.3 / +7.2 |
> | **R-T6, toy at 100M cap** | `s4001b` / `s4002` / `s4003` final | **+10.20 pp** | **[+7.50, +12.91]** | +8.1 / +11.2 / +11.3 |
>
> **What it says.**
> - **At the GPU wall clock production spends to reach 15M (≈ 2.5 h), the toy is STRONGER than production's 15M finals**, and all 48 cells are above 50 %. The machinery's
>   per-step advantage (−14.1 pp for the toy at matched steps) is smaller than the toy's ≈ 3.3× step throughput, at this one budget.
> - **What it does not say:** that the toy would still lead at a larger equal budget (production keeps climbing past 15M), or anything about the ceiling. The ceiling
>   remains R-T2's.
> - **Caveat:** `s4001b`'s R-T6 row inherits the post-canary steps (F36).
>
> **Descriptive record, 2026-10-16: INTERIM X26 vs the toy (NOT R-T2).** X26's LIVE 70M checkpoint (`rb_x26_s2601`, `checkpoint_70000008_steps.zip`) against each toy's 100M cap, 500 mirrored
> pairs per cell: production scores **55.3 / 52.4 / 55.5 %** (vs s4001b / s4002 / s4003), pooled **+4.40 pp**, seeds-as-units 90 % [+1.48, +7.32] (df 2; one production seed, so its seed variance is not in
> the interval). It is registered descriptive and interim, before any game; R-T2 stays reserved for X26's plateau. No §3.6 class is read.
> Record: [`../research_state/measurements/x26_vs_toy_interim_2026-10-16/RESULTS.md`](../research_state/measurements/x26_vs_toy_interim_2026-10-16/RESULTS.md).

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
> (G_m′ over a flat MLP); production at its plateau is worth G_p(cap) pp over that model at its 100M-step cap (the toy
> had not plateaued, so G_p(cap) is biased in production's favour); it costs k× the compute per step. So the
> complexity buys {CEILING | SPEED ONLY | NOTHING MEASURABLE}, where CEILING means only that production's plateau beats
> the toy at 100M steps."**

*(Re-worded 2026-10-16 from "at each one's plateau it is worth G_p pp", before any R-T2 read; see the pre-registration amendment in §3.5.)*

The classification is registered here, with the ±2.0 pp bar the closing test and the bf16 A/B used:
- **CEILING**: G_p(cap)'s 90 % interval lies wholly above +2.0 pp. Stated ONLY as "production's plateau beats the toy at 100M steps" (the toy was still climbing, §3.5's amendment); read as "the machinery raises how strong the model gets" only as far as that bias allows.
  *Action:* keep it, and judge future machinery at plateau.
- **SPEED ONLY**: G_m's interval lies wholly above +2.0 pp, AND G_p's interval lies inside [−2.0, +2.0]. The
  machinery makes training faster, not the model stronger.
  - *Action (an owner decision):* weigh it per GPU-hour. A simpler network can reach the same plateau, so
    simplification becomes a registered challenger.
- **NOTHING MEASURABLE**: G_m's and G_p(cap)'s intervals both lie inside [−2.0, +2.0] (safe for the same reason).
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
- **Per seed:** 30–60M to a plateau ≈ 2–6 GPU-h; to the 100M cap ≤ 9 GPU-h. **MEASURED (§3.8): no seed plateaued
  by 70M / 100M, so the cap binds; s4002 took ≈ 5.7 h wall to 100M.**
- **Three T-ENT seeds:** ≈ 8–18 GPU-h expected, ≤ 27 GPU-h at the cap. T-MLP is about the same.
- **The crosses:** R-T1 is 12,000 pairs. Its GPU time is UNVERIFIED; the Training session schedules it in a
  window.

### 3.8 The toy seeds' plateau status (the §3.4 rule, measured 2026-10-15)

Tier 1 at the §3.4 toy plan (Δ 5M, lag 4, W 20M), by T33's procedure and scripts on the agent cores under rule 5b.
Every verdict is in `models/_ledger` (`plateau_t1` requests). Evidence: s4001 in
`../research_state/measurements/t33_plateau_cost_2026-10-15/` (§2.6), s4002 in its `s4002/` subfolder (run
retrospectively on the finished run; 16 checks in 29.5 min wall, 58 CPU-min). The score is the newest node's
(W + D/2) / games against the node 20M back. It is descriptive; the verdict is the registered quantity.

| check | s4001 verdict (pairs) | s4001 score | s4002 verdict (pairs) | s4002 score |
|---|---|---|---|---|
| 25M | GAIN (80) | 91.9 % | GAIN (80) | 88.4 % |
| 30M | GAIN (120) | 77.5 % | GAIN (120) | 81.7 % |
| 35M | GAIN (120) | 69.2 % | GAIN (160) | 65.0 % |
| 40M | GAIN (120) | 66.2 % | GAIN (160) | 63.6 % |
| 45M | GAIN (200) | 59.5 % | GAIN (200) | 61.1 % |
| 50M | GAIN (320) | 57.3 % | GAIN (320) | 58.5 % |
| 55M | GAIN (200) | 59.0 % | GAIN (400) | 55.0 % |
| 60M | GAIN (320) | 56.5 % | GAIN (280) | 56.9 % |
| 65M | GAIN (400) † | 56.1 % | GAIN (680) | 53.6 % |
| 70M | GAIN (200) † | 60.8 % | GAIN (400) | 55.2 % |
| 75M | — (died ≈ 72.66M) | — | GAIN (440) | 55.3 % |
| 80M | — | — | GAIN (880) | 52.9 % |
| 85M | — | — | GAIN (720) | 53.6 % |
| 90M | — | — | GAIN (1,200) | 52.6 % |
| 95M | — | — | GAIN (760) | 53.4 % |
| 100M | — | — | GAIN (600) | 53.8 % |
| **status** | **CLIMBING at 70M; no cap reached** (F17) | | **CLIMBING at 100M → `CAP`** | |

† post-canary nodes, VOID pending the training session's ruling (F17).

- **Neither seed plateaued.** Every one of 26 checks is GAIN; there is no FLAT, CONFLICT or INCONCLUSIVE. s4002, the
  only seed to reach the cap, is a **`CAP` stop at 100M**, not a plateau, under §3.4.
- **The two seeds agree where both exist** (25M–60M): the same verdicts, and scores within 0.4–4.2 pp of each other
  on the same downward path.
- **The toy is still improving at 100M, slowly.** From 80M on, s4002's newest node scores 52.6–53.8 % against the
  node 20M back, about +18 to +27 Elo per 20M. The GSPRT's H1 is 0.52, so a FLAT needs the true score at about 0.51 or
  below. A score that flattens at ≈ 53 % reads GAIN at every check, and the 100M cap binds.
- **What the registered rule implies for §3.5.** The cap binds, so the toy's checkpoint for R-T2 ("plateau against
  plateau") is its **100M `CAP` checkpoint**. §3.4 records a cap stop "as `CAP`, not as a plateau", and §3.5 names only
  "each toy plateau checkpoint", so that substitution is IMPLIED, not registered (F33). G_p is then **production at
  plateau against a toy still climbing**, which biases G_p toward production by the toy's unrealised gain. Only
  CEILING is exposed: it would need the reading "production's machinery beats the toy at 100M steps", not "at each
  one's plateau". SPEED ONLY and NOTHING MEASURABLE stay conservative, because the bias can only widen the gap (F32).
- **The §3.7 estimate "30–60M to a plateau" is REFUTED** by both seeds: no plateau by 70M (s4001) or 100M (s4002).
  The per-seed cost is therefore the cap's: s4002 took ≈ 5.7 h wall to 100M (from its canary timestamps: 97.3M steps
  in 5.52 h between the update-10 and update-1000 canaries).

## 4. The EXIT CHECK

The exit check is PASS only when every row passes. X26 launches only after it. The commit at which it passes is
**P_X26**, and X26 is pinned there.

| # | item | PASS criterion | evidence to record | **status at `87e493c3` (2026-10-15 15:45 PDT)** |
|---|---|---|---|---|
| E1 | **The bounded list** (the 2026-10-06 exact refactors, behaviour screens and perf) | every item landed or explicitly DROPPED with a reason in this doc's Decision record. Already done: refactors (version break 2026-10-07), screens (end state ADOPTED 2026-10-12), perf (bf16 ADOPTED 2026-10-14). Explicitly deferred past X26, each a future arm with its own A/B: T26 overlap and CUDA graphs | this table | **PASS.** Every item has landed or is deferred with its reason (T26 overlap and CUDA graphs: Decision record 2026-10-15) |
| E2 | **Legacy deletion U0, U1, U3 landed** | each a separate `/gen3ai-ship` with green gates; the K9 golden re-recorded exactly once (at U3); the ledger line "fresh-init stream changed at `<U3 sha>`" | commits + ledger | **PASS.** U0 `6f6b7f90`, U1 `38a21c0f`, U3 `882a3f86` (the K9 golden re-recorded ONCE, at `33a1c3b9`; docs `264a35e3`). The ledger has the line: 2026-10-15 BUILD (MAJOR), "the FRESH-INIT STREAM changed at `882a3f86`". The `pre-legacy-deletion` tag is `0a0d9e1d` |
| E3 | **U2 / U4 either landed with an init-identity proof or deferred** | for each unit that lands before P_X26, a test that FAILS on revert: production's fresh state_dict at a fixed seed is byte-identical before and after the unit | the test name per unit | **DEFERRED** past P_X26. Neither unit has started, and neither is needed by X26. They may land DURING X26 only under §4.1's conditions (init identity + FORWARD identity + X26's checkpoints still load + no `data/` change) |
| E4 | **The umbrella GIGO fixed (T31)** | a typed key the umbrella would override is REFUSED, typed, with a test that fails on revert; production's resolution unchanged (`checkargs --arch production` byte-identical report) | commit | **PASS.** `0a0d9e1d` (`umbrella_override_test.py`, 14 tests, 9 fail on revert; the toy argv's and `--arch production`'s `checkargs` reports byte-identical) |
| E5 | **The routine gate and the slow-tier status** at P_X26 | routine gate green (or the targeted set + static tier, scope named, per the testing rules); `slow_tier_status.json` has no recorded FAIL | gate log | **OPEN.** `slow_tier_status.json` (last written 2026-10-15 21:32Z) has 119 rows: 114 pass, 5 skip, **no recorded FAIL**. 🚨 **But it HIDES a red:** `compiled_first_update_test` (CPU) fails at every commit since `ed58a3ad` (an Inductor C++ codegen error; the post-U3 sweep `87e493c3`). Its row still reads the 2026-10-09 pass (`73408d02`), because a recorded FAIL would fail the static tier for every agent. The fix is in flight in another worktree. The CUDA twin's last recorded pass is 2026-10-04 (`174f5e62`), before the end state became production (W1 in §4.1). The 3 compile-gate CUDA rows read `skip` (an agent run at `19d840cc` without a GPU), while `verify_compile_gate_cuda` PASSED 4/4 at `87e493c3`. That script does not write this file (F26). PASS needs the fix landed, its row re-recorded PASS, and the targeted set + static tier at P_X26 |
| E6 | **Cost re-measured at P_X26** | a production run's quiet-window `train_ms` and fps over ≥ 20 updates within ±7 % of the **window-matched reference**: `rb_bf_eb`'s 42.07 s `train_ms`, and the fps of a reference run (`rb_bf_eb_s2001` / `s2005`) read over the SAME update window as the run under test (for the 5M seed, updates 10–50). *(Corrected 2026-10-16, see the dated note below: the first wording named a flat 1,631 fps, which is EB's late-window figure; games are longer later in a run, so fps falls as a run ages and a window of early updates cannot be compared with it.)* A regression beyond the band is investigated before launch (it changes nothing about strength but sizes the run) | the numbers + their run | **PASS 2026-10-16 (like-for-like).** Updates 10–50 of `rb_xs_s2701` (steps 1.09–5.02M, `launcher_child.full.log`): `train_ms` 42.00 s (−0.2 % of 42.07 s), fps 1,882, against 1,865 (`rb_bf_eb_s2001`) and 1,841 (`rb_bf_eb_s2005`) over the same window: fps +0.9 % / +2.2 %. E7's argv matches EB's surface (heads typed, `--snapshot-ladder-games 0`), so the comparison is like for like. No separate GPU time. The earlier read against the flat 1,631 fps (21 of 30 tables, median fps 1,853, +13.6 %) said OUTSIDE and stays in `road/status.txt` (01:36:18) as written; the ruling and the like-for-like line are the 01:37:24 entry (`../research_state/measurements/x26_exit_check_2026-10-15/road/status.txt`, `road/e6_read.py`) |
| E7 | **The sanity screen at P_X26** | TWO fresh production seeds (`rb_xs_s2701`, `s2702`) to 5M (~1.8 GPU-h), crossed against the 8 EB 5M models (*2026-10-15: s2001–s2004 `final_model_stage1_5M.zip`, s2005–s2008 `checkpoints/checkpoint_5000*_steps.zip`; those four have no stage-1 file, §3.5's note*), 500 mirrored pairs per cell (2 × 8). **PASS iff the pooled Δ̂ (sanity − EB) ≥ −3.0 pp AND neither seed's own Δ̂ < −6.0 pp**, with no crash, no NaN, and no K9(b) stop that is not a tie-share false alarm. A value within 1e-9 of a bound is not a crossing (rule 8). **FAIL ⇒ STOP and bisect the deletion units**: it is a screen for breakage, not an equivalence test. (Seed-to-seed SD at 5M is ~3 pp, from the stage-1 cross's s2_cols 9.7 pp²) | `verdict.json` from a coded rule committed before the cross | **OPEN** (it needs P_X26). **The coded rule is COMMITTED with this status, before any seed:** [`sanity_rule.py`](../research_state/measurements/x26_exit_check_2026-10-15/sanity_rule.py) (`decide`; `self-check` OK on seven constructed cases, including both rule-8 ties and a dirty run gate) |
| E8 | **Plateau meter at launch** | (a) the stop rule of §2.4 written into Amendment 2; (b) T33 PASSED: at least one `main.plateau tick` decided, with no CONFLICT, on a real current-architecture run (a T-ENT seed or R-60), with its CPU minutes per check recorded | the ledger `plateau_t1` rows | **PASS.** (a) Amendment 2 §A2.6 (`ed2b9f82`). (b) T33 `8ea4592d`: 10 checks on `rb_toy_ent_s4001`, all decided GAIN, no CONFLICT, CPU cost in §2.6 (the 65M / 70M verdicts are void, F17; the eight before them stand) |
| E9 | **The ride-along registration** | Amendment 2 committed (T32): every stale clause of I11 fixed, read points every 25M + plateau, bf16 head-identity test green, the K9(b) handling declared, per-update excluded-share logging live. **MET 2026-10-15 at `ed2b9f82`** (re-check the two tests green at P_X26) | commit | **PASS at `87e493c3`:** both launch-gate tests green (2 passed in 52 s, CPU, rule 5b; `x26_exit_check_2026-10-15/e9_tests_87e493c3.log`). Re-run them at P_X26 if P_X26 ≠ `87e493c3` |
| E10 | **The argv** | `python -m main.checkargs --argv "<X26 argv>"` exit 0 at P_X26; `python -m main.launcher --dry-run …` "would launch" (from the MAIN checkout); torch 2.8.0+cu126 | both outputs in the launch note | **PASS at `87e493c3`** (agent worktree, `GEN3AI_MODELS_DIR` scratch, CPU): `checkargs` exit 0, and the report is identical to Amendment 2 §A2.3's; dry-run exit 0, "would launch", FRESH, pin `87e493c3`, torch 2.8.0+cu126, restarts every 6 h (`x26_exit_check_2026-10-15/e10_*`). **Re-run both at P_X26 from the MAIN checkout** before the launch (F22: a scratch dir sizes checkpoints at 0) |
| E11 | **Disk and data** | ≥ 100 GB free on `models/`; the `data/` freeze announced to every session for X26's duration | `df`; the announcement | **Disk PASS:** 330 GB free of 1003 GB (`df`, 15:10 PDT). X26 at 500M on the 1M cadence needs ≈ 36–60 GB (§2.3, F22). **The `data/` freeze announcement is OPEN**: it is made AT the launch |
| E12 | **The owner's answers** | Q1 (seeds), Q2 (meter scope), Q3 (D-LD decisions), Q4 (promotion) recorded in this doc's Decision record, or the recommendation adopted under the 15-minute standing rule | Decision record | **PASS (the orchestrator's defaults, recorded under the 15-minute rule; no owner answer has arrived).** Q1: one seed, plus `rb_x26_s2602` at seed 1's first Tier-1 FLAT, pinned at P_X26. Q2: the stop rule plus a validated Tier 1. Q4: `--promotion-sprt` / `--eval-mirrored-pairs` OFF. All three are in Amendment 2 §A2.8 and the Decision record's T32 row. Q3: D-LD1–D-LD7 adopted (the Decision record's T30-U0 row). An owner answer before the launch supersedes them through an Amendment 3 |


> **Dated note, 2026-10-16 (E6's reference window-matched; the orchestrator's ruling on the X26 road, `98cd814b`).** E6 is a COST-REGRESSION check: it asks whether P_X26 costs more per update than the production the run was sized on. The registered 1,631 fps was `rb_bf_eb`'s whole-run figure, dominated by late updates, when games run longer and fps is lower. `rb_xs_s2701`'s 5M run reads only its early updates, where the same code is faster (1,841–1,882 fps), so the flat figure made a 13.6 % "excess" that no regression caused. The reference is now the EB runs' fps over the SAME update window, and `train_ms` (42.07 s, window-insensitive: it is the learner update) is compared as registered. The ±7 % band, the 21-of-30 update floor and every research rule are UNCHANGED; the E7 and R-T rules are untouched. The first read's 01:36:18 line is kept in `status.txt`.

**Tally at `87e493c3`:** 7 PASS (E1, E2, E4, E8, E9, E10, E12), 1 DEFERRED (E3), 4 OPEN (E5, E6, E7, plus E11's
announcement). Every OPEN row is closed by §4.1's GPU window at P_X26, except E5, which needs the CPU-Inductor fix to land.

### 4.1 Launch plan (2026-10-15, the exit check)

**P_X26 = current `main` once the CPU-Inductor fix for `compiled_first_update_test` has landed** (proposed by the
exit check; the orchestrator fixes the sha). The fix is in flight in another worktree. P_X26 is FIXED when E7's
first seed launches. Anything that lands on `main` after that is outside X26 and outside the screen.
- **Why not `87e493c3` now?** The failing test is in the compiled learner's area, the code X26 runs every update.
  - If the fix changes production code (a compiled region, the micro-step), X26 must run that code and E7 must
    screen it, so the fix has to be inside P_X26.
  - If the fix is test-only, `87e493c3`'s code would do. P_X26 still takes the later commit, so that E5 holds a
    RECORDED green row rather than the stale 2026-10-09 one.
- **Why not wait for more?** Nothing else that X26 runs is pending. Every other open unit is offline (below).

**What must land BEFORE P_X26:**
1. **The CPU-Inductor fix**, with its `slow_tier_status.json` row re-recorded PASS (E5).
2. **BUILT (T37, 2026-10-15, `--canary-shift-max`): a check-only override of the canary's margin constant** that a
   same-run resume may change. This is the delivery mechanism for §4.2's RED action. Without it, raising
   `R1_LOGPROB_SHIFT_MAX` mid-run would mean a FORK, because the launcher REFUSES to move a run's pin at a restart
   (`worktree.resolve_pin`), and `--sync-to-main` would walk X26 onto all of HEAD.
   - What it is: a typed trainer flag (`--canary-shift-max V`, `ModelVersion.canary_shift_max`, config v158), recorded in
     `model_config.json` and inherited like `--trunk-precision` (a typed value wins and is what the next save records; an
     untyped launcher restart, flagless resume or fork inherits it; a fresh run is None = the code's 0.18). It is printed
     on every R1 rule line (`M 0.720 = 4 x shift_max 0.18 [default]`) and in every `canary_verdicts.jsonl` record
     (`canary_shift_max` {value, source, margin}; `watch.*.clip_margin.shift_max`).
   - It never touches training math (the K9 golden's one update is bit-identical at None, 0.72 and 25.0) and it can only
     RAISE the constant: a value below 0.18 is refused, typed. A typed value on a launch with no compiled CUDA learner is
     refused too (it would be silently inert).
   - Tests (each fails on revert): `src/main/train/canary_shift_max_test.py` (resolution through a fresh launch, a same-run
     resume, a launcher restart, a fork; the parser's refusals; the bit-identity), `src/agents/model/canary_shift_max_gate_test.py`
     (the margin follows the model; a raised shift turns the default's refusal of a healthy-large shift into a pass; the rule
     lines and the records carry the value). Detail: `../training/compile_flags.md` "The check-only margin override".
   - It lands BEFORE P_X26 (the pinned tree must contain it: a pinned parser that lacks the flag refuses a resume that
     types it, `✗ NOT IN PINNED TREE`).
3. This commit: §4's status, the E7 coded rule (`sanity_rule.py`, committed before any seed) and §4.2. A pinned run
   does not need documentation inside its pin, but E7's rule must exist before the cross.
4. **The memory fix (I31, 2026-10-16)**: `rb_xs_s2701` at `71351ff5` was refused by `UpdateWontFit` (headroom 978 <
   1,024). The fix (the gate releases its memory; the update-fit demand is what training needs, `gen3_update_fit_v2`)
   reads headroom **2,444 MiB** on the X26 argv, the canary at 5,458 MiB allocated. **P_X26 moves to the landed sha**
   of that change; `71351ff5` cannot launch X26.

**What may land DURING X26 (offline, or code the pin never runs):**
- **The R1×R2 and R5 reader code** (A2-F2 / F20). These are offline readers of X26's checkpoints at HEAD. They must
  land before the FINAL read, not before the launch.
- **The plateau meter's parallel batches (F14 / T33-2). LANDED 2026-10-16 (T38), at X26 ≈ 2M steps with no
  `plateau_t1` request of X26's open** (its first check is 50M). This is offline `main.plateau` code.
  - It must keep the GSPRT rule unchanged (batches judged in batch order), with a test that replays a RECORDED
    check (a T-ENT ledger request) to the same decision.
  - Land it BETWEEN checks, never while a `plateau_t1` request is open. Land it before X26 nears 70M, its first
    possible `TIER1_PLATEAU` (about 12 GPU-h after launch). Before that every check is a cheap GAIN check.
- **U5 / U9, T35 (the maturity battery), T20.** These are offline, as already planned (§2.4, §5.3).
- **Legacy deletion U2 / U4: yes, they may land during X26. They need not wait for its end, but four conditions
  apply.** X26 and its seed 2 run at P_X26 and never execute U2 / U4. But every OFFLINE consumer of X26's
  checkpoints runs at HEAD: Tier 1's h2h, the ride-along reader, the maturity battery, the R-T1c / R-T2 toy crosses
  and the prober. A deletion that changed how HEAD plays a v157 production checkpoint would silently change what
  every one of those reads measures. So each unit lands only with:
  - **(a)** E3's init-identity test, which fails on revert;
  - **(b)** a FORWARD-identity test on a real v157 production checkpoint: the same logits and value, byte for byte,
    before and after the unit at HEAD. Use `rb_xs_s2701`'s 5M final, or an X26 checkpoint once one exists;
  - **(c)** proof that X26's latest checkpoint still LOADS at HEAD (the adoption rule);
  - **(d)** no `data/` change (standing rule 6).
  If (b) cannot hold, that unit waits for X26's FINAL. U6 (close-out) waits for X26's argv by its own text, which is
  fixed at launch.
- **Never during X26:** a `data/` change (rule 6, I24), or a checkpoint pruner (F9).

**THE GPU ORDER from now to the launch** (every job runs under the Training session's lease; any P0.1 hidden-info GPU
need preempts all of it):

| # | job | GPU-h | when (PDT, earliest) | what it buys |
|---|---|---|---|---|
| W0 | **T-ENT s4003** (running since 15:01 10-15, pin `3899277a`) to its toy plateau, or more likely its 100M cap (s4001 and s4002 never read FLAT) | ≤ 5.7 (≈ 5.6 left) | → ≈ 20:45 Thu | the third T-ENT replicate. Not preempted: X26 gains ~5 h at the cost of a lost replicate, and the date is set by P_X26 anyway |
| W1 | **The GPU checks of what P_X26 changed:** `compiled_first_update_test`'s CUDA variant (last recorded pass 2026-10-04, pre-end-state); plus `scripts/ops/verify_compile_gate_cuda.sh` IF the CPU-Inductor fix or the margin override touches `src/agents/model/compile*` or the micro-step | ≤ 0.25 | 20:50 | E5 on CUDA; the gate at P_X26 |
| W2 | **E7 seed `rb_xs_s2701`** at P_X26 to 5M; **E6 is read from its updates 10–50** | ≈ 1.0 | 21:05 | E7, E6 |
| W3 | **E7 seed `rb_xs_s2702`** at P_X26 to 5M | ≈ 1.0 | 22:10 | E7 |
| W4 | **ONE h2h window:** E7's cross (2 × 8 × 500 pairs) + R-T1 (the 3 toys at 15M vs the 8 EB 15M finals, 3 × 8 × 500) + R-T1b (5M vs 5M, 3 × 8 × 250) = 26,000 pairs. The bf16 A/B's 8 × 8 cross (64,000 pairs) took ≈ 1 GPU-h at 75 games/s per cell | ≈ 0.4–0.5 | 23:15 | E7's verdict, G_m (the toys' primary matched-step read), R-T1b |
| — | CPU, beside W4: E7's `decide`, E6's `phase_stats.py`, E10 re-run at P_X26 from the MAIN checkout, the `data/` freeze announcement | 0 | 23:45 | E5–E7, E10, E11 closed |
| W5 | **X26 launch** (`rb_x26_s2601`) | 17 per 100M; ≤ 86 at the cap | **≈ Fri 10-16 00:00–01:00** | the deep run |

- **E7 and E6 argv:** X26's argv (Amendment 2 §A2.2) with `--steps 5000000 --seed 2701` (then 2702),
  `--run-name rb_xs_s2701` / `rb_xs_s2702`, `--snapshot-ladder-games 0` (EB's recorded value, so E6 compares like for
  like) and `--pin-commit <P_X26>`. Run the two seeds one at a time, never two on the GPU at once (E6 needs a quiet
  window). The Training session builds and validates it (`checkargs`, `--dry-run`). A design-doc line is not a
  launch command.
- **The R-T1 / R-T1b cells play at ONE commit, named in their cells plan before any cell.** Use either P_X26, where
  the toy loads through the vacuity rule and EB through the adoption rule (checked by sample, §3.5's note), or
  `pre-legacy-deletion` (`0a0d9e1d`). The Training session decides and records which.
- **G3, R-60 (the production scout, 7.7 GPU-h), is DROPPED from the pre-launch order.** Today's canary findings
  REMOVE the case for it.
  - **It cannot run the long-run bf16 exposure test it would now be wanted for.** Its pin `f45d0c27` predates T36
    and the canary watch record, so its canary records no shift and still carries the clip-boundary false FATAL.
    The production-surface census put a boundary row in 2 of 10 states. A 45M continuation (≈ 7.7 h) draws about 5 canaries
    and 1–2 restart gates, so a false FATAL could end the scout before it reads anything.
  - **X26 is that exposure test,** on the current code, with the watch of §4.2.
  - **Every other thing R-60 bought, X26 produces within about 10 GPU-h of launch, on current code and data:** the
    matched-step references (R-T1c now reads X26, §3.5's note), the first production Tier-1 check, the K9(b) share
    past 15M, and T24's restart telemetry.
  - 7.7 GPU-h is about 45M of X26.
- **T-MLP (G5) does not go before X26.** T34 is not built (≈ 1.5 agent-days), and its seeds cost 8–18 GPU-h. They run
  after X26's plateau, or in GPU windows the orchestrator opens during X26 (the file-handoff PAUSE / GO protocol).
  This departs from the order of the owner's 10-09 roadmap ("toy controls, then the exit check, then X26"), so it
  goes to the owner as FINDING F27. T-ENT's three seeds do finish before X26.
- **Earliest realistic launch: Fri 10-16, about 00:00–01:00 PDT,** IF P_X26 has landed by about 20:45 Thu. Otherwise
  the launch is about 3.2 h of wall time after whichever comes later, P_X26 landing or s4003 finishing (W1–W4 ≈ 2.75 GPU-h, then the
  reads). An E7 FAIL stops everything: bisect U0 / U1 / U3 on CPU first (the K9 golden, the init records), and on
  the GPU only with the orchestrator's go.

### 4.2 The bf16 HEADROOM WATCH (from the T36 GPU confirmation, 2026-10-15)

**What grew.** On `rb_toy_ent_s4001`'s checkpoints, the GPU diagnostic at `87e493c3` read how far the COMPILED bf16
learner (Cbf) moves a fixture row's log-probability from the fp32 eager reference. The canary's margin rule
judges exactly this number (`compile_regions._clip_margin_rule`).

| checkpoint | 15M | 30M | 45M | 60M | 66M | 70M | 72M | FATAL (72.66M) |
|---|---|---|---|---|---|---|---|---|
| Cbf, TAKEN action (all 64 rows) | 0.050 | 0.060 | 0.051 | 0.068 | 0.059 | **0.107** | 0.086 | **0.110** |
| Ebf (eager bf16), taken action | 0.043 | 0.055 | 0.064 | 0.049 | 0.107 | 0.086 | 0.076 | 0.082 |
| Cbf, max over EVERY legal action | 0.059 | 0.082 | 0.078 | 0.072 | 0.076 | 0.117 | **0.136** | 0.110 |

Source: `../research_state/measurements/x26_exit_check_2026-10-15/canary_shift_87e493c3.json`, extracted by
`canary_shift_extract.py` from `~/.cache/gen3ai/diag_bf16_canary/87e493c35e6a/gpu_*.json`, slice 0 = all 64 rows.
C32 (compiled fp32) reads ≤ 4e-6 everywhere.

**How to read it.**
- **The quantity the rule judges is the TAKEN action's shift.** The PPO clip reads only the taken action's ratio.
  So the headroom today is **0.110 = 61 % of `R1_LOGPROB_SHIFT_MAX` 0.18**, not 76 %. The 0.136 figure is the
  all-legal-actions maximum, a superset (F24).
- **0.18 is not the refusal.** The rule refuses an arm only past the MARGIN M = 4 × 0.18 = **0.72**, or on any
  clip-branch switch. 0.18 is the largest healthy shift measured so far, and the 4× safety factor rests on it. Today's
  read sits 6.5× below the refusal.
- **The growth follows the WEIGHTS, not the compiler.** Eager bf16 grows with it (0.043 → 0.107 at 66M), and
  Cbf / Ebf stays between 0.5× and 1.4× on the 64 rows. The pre-activation p99 grew from 6.3 to 11 over the same
  checkpoints (the diag's `pre-act p99`).
- **This is the toy. Production's compiled shift past 15M has never been read.** The CPU eager census on 10
  production states read ≤ 0.055.
- A straight line through the toy's taken-action reads (≈ +0.001 per 1M steps) would cross 0.18 at about 140M and
  0.72 at about 650M. X26's cap is 500M. That line is an ILLUSTRATION (UNVERIFIED for production), not a forecast.
  It says a RED read during X26 is plausible, and a refusal is possible but not expected.

**The field, and the reader** (the canary record `gen3_canary_watch_v1`, at P_X26 since T36). Each canary, at
update 10 and then every 100 updates (≈ 9.8M steps ≈ 1.7 h of X26), appends one line to
`models/rb_x26_s2601/canary_verdicts.jsonl`:
- `.watch.scheduled.clip_margin.max_abs_dlogp.compiled` is THE WATCHED NUMBER, the compiled learner's taken-action
  shift on all 64 rows at the live weights.
- Beside it: `.eager`, the eager bf16 yardstick; `."fp32 reference"`, ≈ 0; and `.margin`, 0.72.
- An unconfirmed or FATAL record also carries `.watch.confirm_slice1` and `.watch.confirm_slice2`.

The startup gate at every 6-hourly restart prints the same numbers on its R1 rule line in `launcher_child.log`.

```
jq -r 'select(.watch.scheduled.clip_margin != null) | [.update, .num_timesteps, .verdict,
  .watch.scheduled.clip_margin.max_abs_dlogp.compiled, .watch.scheduled.clip_margin.max_abs_dlogp.eager,
  .watch.scheduled.clip_margin.margin] | @tsv' models/rb_x26_s2601/canary_verdicts.jsonl
grep -o 'max |dlogp| compiled [^,]*, eager [^,]*' models/rb_x26_s2601/launcher_child.log    # the restart gates
```

The `jq` line was validated on records written through the REAL writer (`CompileCanary._record` +
`GradWatch.record` + `clip_relocation`, 2026-10-15). No live run carries the record yet: the toys and R-60 are
pinned before it.

**Who reads it:** the X26 Training session, at every 25M read point, before every planned 6-hourly restart, and on any
`[CompileCanary]` warning. It is one `jq` line. Write s for the largest `compiled` value read so far.

| band | s (compiled, taken action) | action |
|---|---|---|
| **GREEN** | s ≤ 0.144 (80 % of 0.18) | none |
| **AMBER** | 0.144 < s ≤ 0.18 | the orchestrator STAGES the raise: the value v = s rounded UP to 2 significant digits with headroom (≥ 1.25 s), so M = 4 v. It is a NUMBER for the run to be resumed with (`--canary-shift-max v`, §4.1 item 2, T37), not a code commit: a commit that raises `R1_LOGPROB_SHIFT_MAX` serves only FUTURE pins. Queue `scripts/ops/diag_bf16_canary.sh rb_x26_s2601` (≤ 20 min, GPU) for the next restart window: it confirms the shift is shared by eager (Ebf of the same order) and that C32 is still ≈ 0, i.e. weight growth, not compiler drift |
| **RED** | 0.18 < s ≤ 0.72 | the 4× premise is broken, but the run is still categorically safe (no row can switch while s ≤ M). **Do not stop the run.** At the NEXT planned 6-h restart window, **resume the SAME run with `--canary-shift-max <v>`** (v as in AMBER) and run the diag in the same window: stop the run at its restart, then resume `--model <its latest checkpoint>` at the SAME pin with the flag typed, verified by `--dry-run` (it must show the pin unchanged and the flag in the child argv). The value is recorded in `model_config.json` and inherited, so every later restart keeps it; the startup gate's R1 line and every canary record then read `shift_max <v> [--canary-shift-max]`. No fork, no code commit, no pin move. (Only if the flag were unavailable in the pinned tree, the fallback is a fork of X26's latest checkpoint into a new run dir at P_X26 + the one-constant commit, with the fork hazards — pool auto-seeded, `--lineage-run` for Tier 1, read points continue on the fork — banked as a deviation in Amendment 2's notes.) **Never `--sync-to-main`** (it walks X26 onto all of HEAD) |
| **REFUSED** | s > 0.72, or any clip-branch switch, or any canary FATAL / startup-gate refusal | FATAL_CONFIG, not restarted. Follow the documented procedure, `../training/compile_flags.md` "The bf16 canary FATAL on trained weights": the diag on the run's own checkpoints in a GPU window. A healthy-but-large shift (Ebf similar, C32 ≈ 0) means raise the constant, re-run the gate, and resume from the latest checkpoint. Cbf off the reference with NO branch flip, or C32 off at the fp32 bar, means the fixture was not the whole story: X26 continues on the fp32 trunk through an **Amendment 3** (never silently), from the SAFE ROLLBACK POINT the FATAL prints |

**A second, descriptive trigger:** if `compiled` exceeds 2 × `eager` on two consecutive canaries, run the diag in the
next restart window. The ratio on the toy's 64 rows stayed ≤ 1.4, so 2× would point at the compiler, not the weights.

**The RED raise has a COST on the loss drift** (measured 2026-10-17 on X26's own checkpoints, CPU, UNVERIFIED on CUDA;
`../training/compile_flags.md` "bf16 drift on a sharpening policy", F37, R8). X26's pinned gate reads the bf16 loss drift
on the RELOCATED rows, and a raised margin moves a relocated A < 0 row to ρ = 0.85 e^(2M). At `--canary-shift-max 0.28`
the per-restart odds of a loss-drift refusal on the 114M–123M states rise from ≈ 1–2 % to ≈ 4 % (CPU null: 7 of 170
draws over 0.014 against 2 at 0.18). The 118M checkpoint reads 0.01397, at the bar. So weigh the raise before typing it:
today's compiled shift (0.218 at the 121M restart; the canary 10 updates later read 0.1285) is 3.3× below the refusal
(M = 0.72), and the 4× premise is a calibration margin, not a refusal. A loss-only refusal is recoverable at the same pin
(R8's rollback).

**STARTUP-GATE REFUSALS during X26: the operational procedure** (orchestrator, 2026-10-17; an OPS procedure, not a
research rule; it changes no registered read).

- **The LOSS line only** (not grad, per-param or margin) ⇒ resume the same run at the same pin from the newest periodic
  checkpoint whose CPU pre-flight reads relocated-row drift ≤ 0.007, half the bar. The pre-flight is
  `designs/research_state/measurements/x26_bf16_drift_2026-10-17/scripts/newrule.py <ckpts> 64`, ~10 s per checkpoint.
  Cost ≤ 1–2M steps.
- **The PER-PARAM line, confined to `move_resolution_cell.*`** ⇒ X26 is down anyway. Run the GPU `canary_replay` on the
  newest ~4 checkpoints, then resume from the newest one that PASSES both slices (~5 min).
  - The replay is `python -m main.ops.diag_bf16_canary gpu --arms E32,Ebf,Cbf --dither 0 --dither-rest 0` under
    `gpu_lock.sh` + `mem_cap.sh 20`. The wrapper `diag_bf16_canary.sh` does not pass `--arms`, and all 7 arms compile
    for ~19 min, past its 1,200 s wall.
  - Read `canary_replay`, the gate's own relocated `_check`, not `canary_verdict`, which reads the RAW rows.
- **ANY other parameter, or any other line** ⇒ HOLD and tell the orchestrator.
- **Evidence** (2026-10-17, X26 resume from the u13900 K9(b) stop):
  - The startup gate refused `final_model_exception.zip` (136.97M) on `move_resolution_cell.switch_proj.bias` at
    19.3 × (cached kernels, SHARED-HIT `5d99a269`). A FRESH compile reproduced it at 21 × on slice 0, the gate's rows.
  - Slice 1 of the same weights passed (3.53). The other 7 checkpoints from 110M to 136.0M passed both slices (worst
    6.02). Every refusal and near-bar raw read sat in `move_resolution_cell` (switch_proj / move_proj).
  - So the refusal is WEIGHTS-dependent, not a cached-kernel artifact. X26 resumed from `checkpoint_136000153`,
    losing ≈ 975k steps.
- **Why mid-run is safe:** the in-run canary confirms on BOTH disjoint halves, so this single-slice lottery cannot
  FATAL it alone. The single-shot startup gate is the exposure, and X26's 24 h restart interval (Amendment 3 §A3.3)
  minimises it.
- **Queued for FUTURE runs:** the startup gate confirms on disjoint halves as the canary does, and an investigation of
  `move_resolution_cell`'s gradient under Inductor bf16.

## 5. The GPU SCHEDULE from tonight to the X26 launch

The GPU lease is held by the Training session ('Training Oct', expiring 2026-10-19 01:36). Every job below runs
through it. The last hidden-info knock-in fork (`rb_hib_ctrl_s3103`, started 01:35) ends at about 02:15. The
estimates assume an exclusive GPU.

**Preemption:** any GPU need of the P0.1 hidden-info study (for example its crosses) goes BEFORE every job below,
per the owner's P0.1 quota priority.

### 5.1 The order

> **SUPERSEDED 2026-10-15 (the exit check) by §4.1's W0–W5.** The deletion chain landed three days early, so
> the order is now: G1 and G2 DONE or running; G3 DROPPED; G4 rides in W4; G5 goes after X26; G6 becomes W1–W4.
> The table below is the plan as written, kept for its reasons.

| # | job | needs a build? | GPU-h | what it buys |
|---|---|---|---|---|
| **G1** ✅ (died 72.66M, I30) | **T-ENT seed 4001**, fresh at HEAD, to its toy plateau (§3.4). Its first ~20 minutes are the toy's GPU check: the compiled path plus bf16 on a flag combination never compiled before | **no** | 2–6 expected, ≤ 9 | the first toy; the measured toy fps; the first real Tier-1 ticks (T33) |
| G2 (s4002 ✅ 100M cap; s4003 running) | T-ENT seed 4002, then seed 4003 | no | 4–12, ≤ 18 | the 3 replicates |
| G3 ❌ DROPPED 2026-10-15 (§4.1) | **R-60, the production scout:** continue `rb_bf_eb_s2001` (pre-declared as the lowest seed index, not chosen by strength) from 15M to 60M in its own lineage, at its recorded pin (`f45d0c27`) and recipe | no | ≈ 7.7 | the only production curve past 15M before X26: matched-step references at 30M / 60M (R-T1c); the first production Tier-1 check (60M vs the 10M W-back); the K9(b) share at 4× the longest horizon so far; restart telemetry for T24. **It is a scout, never a comparator for X26** (pre-fix learnset, pre-U3 init) |
| G4 → W4 | The R-T1 / R-T1b crosses (toy 15M and 5M against EB), in a GPU window | no | UNVERIFIED (≤ ~2) | G_m |
| G5 → after X26 (F27) | **T-MLP** GPU check (20 min), then seeds 4101–4103 to their plateau | **yes** (T34, after T31) | ~0.3 + 8–18 | G_m′, G_p′ |
| G6 → W1–W4 | **The exit-check sanity screen** (E7): 2 fresh production seeds × 5M at P_X26, then the 2 × 8 cross | yes (P_X26 = after T30-U3) | ≈ 1.8 + the cross | E7 |
| G7 | **X26 launch** (`rb_x26_s2601`) | — | 17 per 100M; ≤ 86 at the cap | the deep run |

**The expected timeline (as planned at `3899277a`; superseded by §4.1).** G1–G4 fill Thu 10-15 to Sat 10-17 (about 20–35 GPU-h). T34 should land by about
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
| **T32** ✅ DONE 2026-10-15 (`ed2b9f82`; F5 was wrong, so (c) is a regression test, not a jsonl writer) | **X26 Amendment 2 + its two small builds:** (a) fix every stale clause of I11, write in §2's recipe, the read points (every 25M + plateau) and the §2.4 stop rule, and declare the K9(b) handling; (b) a test: heads ON vs OFF learn bit-identically under bf16 autocast; (c) the K9(b) excluded share on EVERY update (one jsonl line per update, beside TB) | `measurements/ridealong_baseline/PREREGISTRATION.md`, `ridealong_update_test.py`, `rust_rollout/consistency.py` | opus-high | 4–6 h | owner Q1, Q4 for the recipe lines | NEW. X26's EXPERIMENT_BACKLOG row points here |
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
- **R7. The compiled bf16 learner's log-prob shift grows with training until the canary refuses** (the toy: 0.050 → 0.110
  over 15M → 72.66M; the refusal is at 0.72). *Mitigation:* §4.2's watch, with its bands and actions, and T37's
  check-only override (`--canary-shift-max`, BUILT) so a RED read is fixed at a restart without a fork.
- **R8. The startup gate's bf16 LOSS drift refuses a restart** (measured 2026-10-17, F37). It is a per-state lottery, not a
  trend. From ≈ 114M one heavily weighted, ill-conditioned golden row (row 9, ρ|A| 11.6 after the relocation) carries
  88–100 % of the drift. CPU puts it at ≈ 1–2 % per restart over 0.014 (≈ 4 % after a 0.28 T37 raise), and at ≈ 15–20 %
  for at least one refusal over X26's remaining ≈ 13 restarts. All of this is UNVERIFIED on CUDA. The refusal is
  FATAL_CONFIG, and the launcher does not restart it. *Mitigation (no code; X26's pin `9c033489` cannot take the fix):* on
  a refusal that names ONLY the bf16 LOSS (the `DRIFTED … loss` or `COMPILED … loss:` lines; never a gradient,
  per-parameter or margin refusal), resume the SAME run at the SAME pin from the newest periodic checkpoint whose CPU
  pre-flight reads ≤ half the bar. The pre-flight is `research_state/measurements/x26_bf16_drift_2026-10-17/scripts/newrule.py
  <ckpts> 64`; its "relocated-row drift" column is what the pinned gate judges. Adjacent checkpoints differ by up to 10×,
  so this costs ≤ 1–2M steps. Future pins read the loss on the golden's own rows (`gen3_r1_loss_on_golden_rows_v1`), which
  removes the amplification.

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
  clauses are stale (I11) → T32. **RESOLVED 2026-10-15 (T32): Amendment 2.** Its own findings A2-F1–A2-F4 (§A2.9
  there) are F19–F22 below.
- **F3. Tier 1 has never run on a real run**, and its CPU cost beside a live run is unmeasured → T33.
  **RESOLVED 2026-10-15 (T33):** Tier 1 ran on `rb_toy_ent_s4001` with every check decided and no CONFLICT. Its cost
  is in §2.6, and its FINDINGS are F13–F17.
- **F4. The legacy deletion has not started**, and its owner decisions are open, though the closing test's
  adoption ordered it on 2026-10-12. It is now X26's critical path → T30, Q3.
- **F5. The K9(b) excluded share is logged on ~1 update in 10**, so a per-update maximum cannot be read after the
  fact → T32(c). **WRONG (T32, 2026-10-15):** it is logged on every update (153 of 153 in `rb_bf_eb_s2005`); the
  "~1,490 updates per 15M run" was a 10× miscount (an update is 98,304 rows). T32(c) became a regression test of
  that guarantee plus the declared reader (`main.ops.tb_read`); no jsonl writer was built (F19).
- **F6. No head bit-identity proof under bf16** (`ridealong_update_test` is fp32) → T32(b). **RESOLVED 2026-10-15
  (T32):** `test_BF16_one_update_is_BIT_IDENTICAL_with_the_heads_on_and_off`.
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
  and an overshoot costs only games. **RESOLVED 2026-10-16 (T38, TASK_BACKLOG X26-8)**: `--inflight-batches K`
  (default 4) plays K batches per cycle, each on its own seed block and its own row; the rule is unchanged and judged
  in batch order. Measured: 2.0× the production game rate at K = 4, 2.5× at K = 8 (§2.6).
- **F15 (T33-3). The trainer's blocking eval cycle costs ≈ 3.5 % of production wall, not ≈ 1.5 %.**
  `rb_bf_eb_s2001` shows a 43–47 s cycle every 2M steps (≈ 1,270 s). The toy shows 16 s every ≈ 380 s, which is 4.2 %.
  `designs/training/eval_and_rating.md` quoted only O9's ~1.5 % at N = 256. It now gives both figures, corrected in T33's commit.
- **F16 (T33-4). The per-phase CPU and GPU utilization of a PRODUCTION run is in no record.** §2.6 bounds the GPU's
  idle time from the inference waits, and sampled only the toy. The procedure is read-only, needs no GPU window and
  takes ≤ 10 min (§2.6).
- **F17 (T33-5). `rb_toy_ent_s4001` died at 08:24 on a `CompileCanary` FATAL after update 700**, at about 72M. The
  compiled bf16 TRAIN graph's gradient on `pointer_head.move_proj.{weight,bias}` was 0.40 from the fp32 reference,
  against eager's 0.028, which is beyond the 12× bar. It was confirmed on the second slice (NOT independent: a subset of the first — F18). This is FATAL_CONFIG,
  and the run was not restarted.
  - The safe rollback point is 62.0M (the last passing canary was at 62.83M).
  - Tier-1's 65M and 70M checks used post-canary nodes, so their GAIN verdicts are VOID pending the training
    session's ruling.
  - A resume IN PLACE from 62M would write new 63M+ checkpoints into the same `checkpoints/` folder. The decided 65M
    and 70M requests froze their nodes' sha256, so `main.plateau` would then refuse the check, or plan a different
    node. **Resume into a new run directory**, or the toy's later checks need `--lineage-run`.
  - The meter played on the agent cores (CPUs 5–7 and 13–15, nice 19), and the trainer is pinned to 0–4 and 8–12.
    The canary is a numerical check, so contention cannot produce it.
- **F18. The R1 canary's (and the startup gate's) per-parameter bf16 statistic has a decision boundary in it** — the PPO clip on the K9 golden's 64 off-policy rows — so it can FATAL a healthy compiled learner (I30; F17's event), and the confirmation's "independent" slice 1 is a SUBSET of slice 0 (rows 32–63 of all 64), so a boundary row there confirms on both. The same latent boundary sits in the fp32 rule at ~10^4× smaller odds → T36, **FIXED on CPU 2026-10-15** (both rules, the gate and the canary; the confirmation is now two disjoint halves), the GPU confirmation pending (I30).
- **F19 (T32, A2-F1). F5's premise was wrong**, see F5. The real downsampling is the TensorBoard WEB UI's 1,000
  points per tag, which bites at X26's ≈ 5,100 updates; Amendment 2 §A2.7 names `main.ops.tb_read` as the reader.
- **F20 (T32, A2-F2). Two registered ride-along reads have NO reader code:** R1×R2's paired AUROC difference and
  R5's Brier(V) − Brier(Q_played) on the flat pointer (`main.ridealong_read` emits only B's plumbing count). R5 also
  needs `bank_v1` to carry the one-ahead opponent label in flat-pointer terms (UNVERIFIED). Offline, ~0.5 agent-day
  (est.), must land before X26's FINAL read; not a launch blocker. A plumbing smoke at HEAD on `rb_bf_eb_s2001`'s 9M
  checkpoint shows the rest of the reader runs on the end state
  (`research_state/measurements/ridealong_baseline/smoke_endstate_2026-10-15/`).
- **F21 (T32, A2-F3). The heads' overhead on the end-state network is UNVERIFIED**: ≈ 2.3 % of the bf16 update if
  the 2026-09-30 absolute cost (0.974 s) carried over. It is inside every measured end-state throughput, so §2.3
  holds either way.
- **F22 (T32, A2-F4). The launcher dry-run under a scratch `GEN3AI_MODELS_DIR` sizes checkpoints at 0** (no sized
  run there): REQUIRED 32.65 GiB vs ≈ 60 GB real at 500M on the 1M cadence. E10's dry-run runs from the main checkout.
- **F23. The slow-tier record HIDES a known red.** `compiled_first_update_test` (CPU) fails at every commit since
  `ed58a3ad` (an Inductor C++ codegen error; the post-U3 sweep `87e493c3`). Its row still reads the 2026-10-09 pass,
  because a recorded FAIL fails the static tier for every agent. So "no recorded FAIL" (E5) is not "no known red"
  today. The class: the file has no state for "known red, fix in flight", visible but not blocking others. That is
  TECH DEBT, owner-gated, and is not dispatched by this doc.
- **F24. The brief's headroom figure read the wrong column.** "0.136, 76 % of 0.18" is the compiled arm's maximum over
  EVERY legal action (`legal_logp_max_abs`). The margin rule judges the TAKEN action (`log_prob_max_abs`): 0.110 at
  most, which is 61 %. And 0.18 is not the refusal: the refusal is at M = 0.72, 6.5× above today's read (§4.2).
- **F25. All three T-ENT seeds are pinned at `3899277a`, BEFORE T36.** Their canary is the pre-fix,
  boundary-sensitive one: s4001 died of it, and s4002 survived to its cap. Their canaries record no shift (no watch
  record). If s4003 draws a canary FATAL, read it as I30's class first: run the diag on its checkpoints, then resume
  into a NEW run dir from the printed rollback point (F17). It is not evidence against the compiler until the diag
  says so.
- **F26. The T36 CUDA confirmation lived only in `~/.cache`.** `verify_compile_gate_cuda.sh` does not write
  `slow_tier_status.json`, so the three compile-gate CUDA rows still read `skip` (`19d840cc`) beside a 4/4 PASS at
  `87e493c3`. The diag's `VERDICT.md` checklist was left unfilled. This commit copies the summaries into
  `../research_state/measurements/x26_exit_check_2026-10-15/gpu_t36_87e493c3/`. The five criteria were re-verified from
  the `gpu_*.json` (I30's row), and `compile_flags.md` and the ledger now record the confirmation.
- **F27. T-MLP goes after X26: a departure from the owner's 10-09 roadmap order** ("toy controls, then the exit check,
  then X26"). T34 is unbuilt (≈ 1.5 agent-days), and its seeds cost 8–18 GPU-h, so holding X26 for it costs about
  2–3 days of the deep run. T-ENT's three seeds finish first. **For the owner** (the 15-minute rule; the
  recommendation is X26 first).
- **F28. A live run's pin cannot move at a restart** (`worktree.resolve_pin` refuses it, by design), and
  `--sync-to-main` moves it to all of HEAD. So any mid-run change to a CHECK constant (the canary margin) needs either a
  check-only override flag or a fork. **RESOLVED 2026-10-15 (T37): the flag exists** (`--canary-shift-max`, config v158;
  §4.1 item 2) and lands before P_X26, so the RED action is a resume, not a fork.
- **F29. Amendment 2 §A2.5's "~25 launcher restarts at 6 h (≈ 3–4 days)" is off by about 2×.** 500M at ≈ 10.3 min
  per 1M is ≈ 86 h, about 14 restarts. It is a declared-limit sentence, not a rule, so the registration is not edited.
  Its claim (Adam state resets at each restart) is unaffected.
- **F30. The E7 and R-T1b references "the 8 EB `final_model_stage1_5M.zip`" exist for s2001–s2004 only.**
  s2005–s2008 use their 5M checkpoints (§3.5's dated note), recorded before any cell.
- **F31. The shared compile cache sits at 18.56 of its 20 GiB cap** (the E10 dry-run). It prunes itself by design
  (`compile_cache_shared.prune`), so X26's first compile may evict other runs' entries and cost them a cold compile.
  Informational.
- **F32 (T33 s4002). The toy is still climbing at its 100M cap, so §3.5's G_p is biased toward production.** Both
  seeds read GAIN at every Tier-1 check (§3.8), and s4002 ends as a `CAP` stop with its newest node at 53.8 % against
  the node 20M back. R-T2 then compares production at plateau with a toy that has not plateaued, so G_p over-states
  what the machinery buys "at plateau" by the toy's unrealised gain. A CEILING verdict can no longer be read as "the
  machinery raises how strong the model gets"; it reads "production's plateau beats the toy at 100M steps". SPEED
  ONLY and NOTHING MEASURABLE stay conservative (the bias only widens the gap). **RESOLVED 2026-10-16** (§3.5's pre-registration amendment, before any R-T2 read): §3.6's sentence is re-worded for a capped toy, and the cap is not raised.
- **F33 (T33 s4002). The substitution of a `CAP` checkpoint for the toy's plateau checkpoint is IMPLIED, not
  registered.** §3.4 records a cap stop "as `CAP`, not as a plateau"; §3.5 names only "each toy plateau checkpoint".
  Close it before R-T2 is read: register the 100M `CAP` checkpoint as R-T2's toy node, labelled G_p(cap). **CLOSED 2026-10-16** (§3.5's pre-registration amendment).
- **F34 (T33 s4002). Tier 1's FLAT has a small dead zone, so a slowly-climbing run can reach any cap.** H0 0.50 / H1
  0.52 reads GAIN whenever the true score against the node W back is above ≈ 0.51, which is about +7 Elo per W.
  s4002 read GAIN at 52.6–53.8 % (≈ +18 to +27 Elo per 20M) with 600–1,200 pairs per check. X26's registered plan has
  the same H1 over W 50M, so it plateaus only once its gain per 50M falls to about the same size. **UNVERIFIED** whether
  the production net flattens faster than the toy; the 500M cap may bind for X26 too.
- **F35 (T33 s4002). `rb_toy_ent_s4001` has neither a plateau nor a cap checkpoint.** It died at 72.66M (F17) while
  CLIMBING, so under §3.4 it has no R-T2 node. It needs the training session's ruling: resume it to 100M into a new run
  directory (F17's sha256 hazard), or declare it out of R-T2 and record why.
- **F36 (toy crosses, 2026-10-16). `rb_toy_ent_s4001b`, s4001's continuation to its 100M cap, forked from
  `checkpoint_72000094`.** That checkpoint is ≈ 9.2M steps past s4001's last passing compile canary (62.83M; F17), so
  every read of its cap checkpoint inherits those steps.
  - In R-T6 (§3.5's 2026-10-16 descriptive record) it is the lowest row: +8.1 pp against +11.2 / +11.3 for the other
    two seeds. Without it, the pooled point is ≈ +11.3 pp.
  - Whether the canary-era steps cost it anything is UNVERIFIED.
  - R-T2 uses the same checkpoint, so report its row separately there too.
- **F37 (bf16 drift, 2026-10-17). X26's startup-gate loss drift of 6.68e-3 (48 % of 0.014) at the 121M restart is ONE
  ROW, made heavy by the T36 relocation, not bf16 losing the model.** CPU, on X26's checkpoints
  (`../research_state/measurements/x26_bf16_drift_2026-10-17/`):
  - The statistic is flat and far below the bar through 113M.
  - From ≈ 114M golden row 9 (A = −3.23, moved to ρ = 3.59) carries 88–100 % of it. Its log-prob is ill-conditioned in the
    fp32 FUNCTION ITSELF: a half-ulp weight dither moves it 0.04–0.11, smoothly, with no near-tie.
  - No trunk round dominates, and the activations are flat.
  - The bars were derived on the golden's own rows before the relocation existed. Read there, the same draws stay
    ≤ 55 % of the bar.
  - The canary's 4.86e-4, 10 updates later, read the same rows on different weights.
  - Fix for future pins: `gen3_r1_loss_on_golden_rows_v1`. X26 itself: R8's rollback; weigh the RED raise (§4.2).

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
| 2026-10-15 | **I30: the canary FATAL is treated as a FIXTURE defect, not a compiler one, pending the GPU read; X26 keeps the production bf16 compiled learner, and the canary's fixture is fixed (T36) rather than its bar recalibrated.** The fixture fix relocates each row within M (log-prob; R1_BAR_K × the largest arm difference read, ≈ 0.65) of its switching clip boundary to 2M inside the active side, by the fp32 reference's own ratio, and makes the confirmation slice disjoint | recalibrating K on trained states (a bimodal statistic has no bar with teeth: healthy flips read up to 31×); fp32 for the pointer head (already fp32 by design); `emulate_precision_casts` (eager bf16 itself flips); an eager learner; the fp32 trunk (kept as the FALLBACK if the GPU shows a branch-free disagreement or a compiled-fp32 one) | CPU, the run's own checkpoints: the bimodal null (no-flip max 0.030, flip median 0.294), the frozen-branch error ≤ 0.020 on all 32 seeds, eager's own flip at 66M, the relocated null 0 of 992 above K; `compile_flags.md` "The bf16 canary FATAL on trained weights"; ledger 2026-10-15 FINDING |
| 2026-10-15 | **T30-U3 landed (config v157): move resolution DERIVED, the seven per-action blocks and the critic's threat injection DELETED; the fresh-init stream CHANGED (D-LD7).** I7 is DONE; the T30 critical path (U0 → U1 → U3) is complete. Every X26 seed (and G6's exit-check seeds) starts at or after U3; a pre-U3 seed replicated at HEAD is not "same pin + seed". T-ENT checkpoints keep loading (vacuous: no op, no move resolution) | an RNG burn that replays the deleted modules' draws (rejected in D-LD7) | the ledger's U3 entry; `research_state/measurements/legacy_deletion_u3_init_2026-10-15/` |
| 2026-10-15 | **T32: X26's ride-along registration is AMENDED (Amendment 2) before any X26 data; the K9(b) share's every-update record is PINNED by a test, not duplicated into a jsonl** — the closing test's tie-share handling adopted for X26 (adapted: a real mismatch is a GIGO stop, not a replaced seed); read points 10M / every 25M / plateau / FINAL; R1's USABLE threshold never looser than registered (max of 0.56 and the same-checkpoint floor); (d)'s offline-prediction label DESCRIPTIVE; Q1 one seed + a second at the first FLAT, Q4 SPRT / mirrored OFF, Q2 rule + validated Tier 1 recorded as the orchestrator's defaults | a per-update jsonl beside TB (F5's premise was wrong: TB already holds every update, and a second writer of one quantity is a second producer to keep in sync); re-measuring the heads' overhead before launch (inside every measured end-state throughput) | `ridealong_baseline/PREREGISTRATION.md` Amendment 2; `ridealong_update_test.py`, `diagnostics_cadence_test.py`; `smoke_endstate_2026-10-15/`; ledger 2026-10-15 REGISTRATION AMENDMENT |
| 2026-10-15 | **T36 BUILT, two refinements of the row above.** (a) **M = 0.72, not ≈ 0.65**: the 32-seed dithered null at the FATAL's weights read an eager bf16 shift of 0.173, above the census max 0.161, so `R1_LOGPROB_SHIFT_MAX` = 0.18 (rounded up to 2 significant digits, as the bf16 bars are) and M = 4 × 0.18; PROVISIONAL until the compiled arm's CUDA shift is read. (b) **The confirmation is the two DISJOINT HALVES (rows 0–31 and 32–63), and the scheduled check and the startup gate keep all 64 rows** — "slice 0 = rows 0–31" would have cut the scheduled check's coverage (a half leaves the moves head below the per-parameter floor). Added: the margin is a CHECKED precondition on every rung (an arm whose \|Δ log π\| exceeds M, or that switches any row, is refused by name), and every rule line / canary record carries the compiled arm's shift | moving rows only to M (no headroom for a provisional margin); EXCLUDING boundary rows (changes the advantage normalisation and drops their policy gradient from the check); halving the scheduled check | `compile_regions_clip_margin_test.py`, `compile_regions_clip_margin_fatal_test.py` (212 / 992 → 0 / 992 at B = 64, fails on revert), `compile_canary_test.py` (a one-row disagreement no longer confirms itself, fails on revert); ledger 2026-10-15 GIGO FIX |
| 2026-10-15 | **THE EXIT CHECK ASSEMBLED at `87e493c3`** (§4's status column): E1, E2, E4, E8, E9, E10, E12 PASS; E3 DEFERRED; E5, E6, E7, E11 (the freeze announcement) OPEN. E9 re-checked (both launch-gate tests green at `87e493c3`), E10 re-run (`checkargs` exit 0, the dry-run "would launch", the report identical to §A2.3's) | — | `../research_state/measurements/x26_exit_check_2026-10-15/` (`e9_*`, `e10_*`) |
| 2026-10-15 | **I30 CLOSED: the T36 fixture fix CONFIRMED on CUDA at `87e493c3`**; X26 keeps the production bf16 compiled learner; `R1_LOGPROB_SHIFT_MAX` stays 0.18 (the compiled arm's taken-action shift peaks at 0.110 on the toy) | the fp32-trunk fallback (its trigger, a branch-free compiled disagreement or a C32 error, did not occur: C32 ≤ 4e-6, 0 relocated flips in every arm) | `gpu_t36_87e493c3/` (the summaries), `canary_shift_87e493c3.json`; ledger 2026-10-15 MEASUREMENT |
| 2026-10-15 | **P_X26 = `main` once the CPU-Inductor fix of `compiled_first_update_test` lands** (proposed; the orchestrator fixes the sha), FIXED when E7's first seed launches. T37 (the check-only margin override) is RECOMMENDED inside it | `87e493c3` now: the fix is in the compiled learner's area, so if it touches production code, E7 must screen it, and either way E5 needs a recorded green row. Waiting for U2 / U4, T38 or T39: X26 never runs them, or they are offline | §4.1 |
| 2026-10-15 | **U2 / U4 may land DURING X26**, each with (a) E3's init identity, (b) a FORWARD identity on a v157 production checkpoint at HEAD, (c) X26's latest checkpoint still loading at HEAD, and (d) no `data/` change; else after X26's FINAL | holding them for X26's end (unneeded: the pin never runs them); landing them with the init proof alone (every offline reader of X26 runs at HEAD, so a forward change would silently move Tier 1, the ride-along reads and the toy crosses) | §4.1 |
| 2026-10-15 | **G3 (R-60) DROPPED; R-T1c reads X26 at 30M / 60M instead** | spending 7.7 GPU-h on it before X26 | R-60's pin `f45d0c27` predates T36 and the canary watch record: it cannot measure the compiled shift (the bf16 exposure question) and carries the false-FATAL fixture. X26 produces every other product within ~10 GPU-h on current code and data |
| 2026-10-15 | **The GPU order W0–W5** (§4.1): s4003 finishes; then the CUDA checks of what P_X26 changed; E7's two seeds, one at a time, with **E6 read from seed 2701's updates 10–50** (no separate GPU time); ONE h2h window for E7's cross + R-T1 + R-T1b; then X26. Earliest launch Fri 10-16 ≈ 00:00–01:00 PDT | preempting s4003 (gains ~5 h, loses the third replicate); a separate E6 run; T-MLP before X26 (F27, to the owner) | §4.1; the bf16 A/B's 8 × 8 cross ≈ 1 GPU-h for 64,000 pairs |
| 2026-10-15 | **E7's coded rule committed before any seed** (`sanity_rule.py`: PASS iff pooled Δ̂ ≥ −3.0 and each seed's ≥ −6.0, with clean run gates; rule 8: within 1e-9 of a bound is not a crossing; INCONCLUSIVE on a missing cell or gate); E7's argv = X26's with `--steps 5000000`, its own seed / name, `--snapshot-ladder-games 0`, `--pin-commit P_X26`; the s2005–s2008 references are their 5M checkpoints | `cross_stat` (it needs a square matrix; E7 is 2 × 8 and its rule is a point screen, not an interval test) | `sanity_rule.py self-check` OK (7 cases) |
| 2026-10-15 | **THE bf16 HEADROOM WATCH** (§4.2): the watched number is `canary_verdicts.jsonl` `.watch.scheduled.clip_margin.max_abs_dlogp.compiled` (the TAKEN action's shift, the one the margin rule judges), read with one `jq` line at every 25M read point and before every planned restart. GREEN ≤ 0.144; AMBER ≤ 0.18 → stage the raised constant + the diag; RED ≤ 0.72 → deliver it at the next planned restart (the T37 override, else a fork), never stopping the run for it; REFUSED → the documented FATAL procedure, the fp32 trunk only through an Amendment 3 | "> 0.18 ⇒ re-pin at the next restart" as briefed (the launcher refuses a pin move at a restart, F28); watching the all-legal-actions maximum (not what the rule judges, F24); stopping X26 at 0.18 (the run is categorically safe below M = 0.72) | `compile_regions._clip_margin_rule`, `GradWatch.fill_margin`; the reader validated on records written by the real writer |
| 2026-10-15 | **T37 BUILT, before P_X26: `--canary-shift-max V`, a check-only RAISE of the canary's margin constant, recorded in `model_config.json` (config v158) and inherited; the headroom watch's AMBER stages a NUMBER and its RED action is a same-run RESUME with the flag, not a fork** (resolves F28) | a code-only raise of `R1_LOGPROB_SHIFT_MAX` delivered by a fork of X26's latest checkpoint (new run dir, pool auto-seeded, Tier-1 lineage, a banked deviation); `--sync-to-main` (walks X26 onto all of HEAD); a flag that may LOWER the constant (no measured basis: 0.18 is the largest healthy shift measured); a `runtime`-class flag that is never recorded (a launcher restart would drop the raise, and a fork would not carry it) | `src/main/train/canary_shift_max_test.py` (fresh / same-run resume / launcher restart / fork; raise-only; one K9-golden update bit-identical at None, 0.72 and 25.0, with a control that moves), `src/agents/model/canary_shift_max_gate_test.py` (a raised shift turns the default margin's refusal of a healthy-large shift into a pass; the rule lines and every record carry the value); `../training/compile_flags.md` "The check-only margin override" |
| 2026-10-16 | **I31 FIXED before P_X26: the startup gate releases its memory and the update-fit DEMAND is what training needs** (`gen3_gate_releases_memory_v1`, `gen3_update_fit_v2`): the relocation forward keeps nothing for a backward, the gate empties its cache after the lock, demand = peak reserved over the dry update + the canary's check from an emptied cache; headroom 978 (FATAL) → 2,444 MiB on the X26 argv; P_X26 moves to the landed sha | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (changes the allocator against the bf16 A/B reference; not measured); lowering the 1,024 MiB bar; a smaller micro-batch (a recipe change) | GPU reads in `../training/learner_lifecycle.md` "The update fit check" and `../training/compile_flags.md` "The gate releases its memory"; K9 golden IDENTICAL |
| 2026-10-16 | **G_p(cap) REGISTERED before any R-T2 read** (closes F32, F33): R-T2's toy node is each seed's 100M `CAP` checkpoint (`rb_toy_ent_s4001b` / `s4002` / `s4003` `final_model.zip`); G_p(cap) is biased toward production; CEILING is stated only as "production's plateau beats the toy at 100M steps", SPEED ONLY / NOTHING MEASURABLE unchanged; §3.6's sentence re-worded. W4's R-T1 ≈ −14.1 pp and R-T1b ≈ −25.5 pp recorded descriptively (no interval: `cross_stat` is square-only; T40) | the cap raised before R-T2; waiting for a plateau that the 100M cap says may not come | §3.5 amendment, §3.8, F32 / F33; `32589f9a` |
| 2026-10-16 | **E6's reference is WINDOW-MATCHED** (42.07 s `train_ms`; fps of a reference run over the same update window); ruled PASS like for like (−0.2 % `train_ms`, fps +0.9 / +2.2 %). No research rule changed | the flat 1,631 fps (EB's late-window figure) | §4 E6 and its dated note; `road/status.txt` |
| 2026-10-16 | **T38 BUILT (F14 / T33-2): the plateau meter's parallel batches, `main.plateau tick --inflight-batches K`, default 4** — K batches in ONE eval cycle on one engine in one process (the executor's opt-in `seed_blocks`, `gen3_eval_seed_blocks_v1`), each batch on its own seed block, row and claim; the GSPRT rule unchanged and folded in batch order (a batch past the stop is a surplus row). Landed at X26 ≈ 2M steps with no X26 `plateau_t1` request open | more processes (rule 5b's ≤ 2 heavy jobs, a second model copy); judging a wave as a whole (peeks at later batches); K = 8 as default (slower than K = 1 on a short GAIN check) | §2.6 "The pipelining, built and measured"; `main/plateau_parallel_test.py` (three RECORDED T-ENT checks replayed at 1–150 in flight with adversarial surplus: the recorded verdict, stop batch, pairs and LLR); `main/h2h/plateau_integration_test.py` (the real engine: the same games and decision at 1 and 3 in flight) |
| 2026-10-17 | **The bf16 rule's LOSS comparisons read the golden's OWN rows** (`gen3_r1_loss_on_golden_rows_v1`, F37, R8): the loss is continuous in the ratio, the relocation only multiplied a moved row's weight in it, and the bars were derived on the golden's own rows. The gradient comparisons stay relocated. **X26 keeps its pin**: on a loss-only refusal, roll back to a pre-flighted checkpoint at the same pin. The staged RED raise is weighed against its measured cost (§4.2) | a check-only raise-only override of the loss / grad bars (T37-style: X26 cannot take new code either, and it loosens the teeth with no basis); a sensitivity-normalised policy drift against the shift bar (a new unmeasured bar, ~10× looser on logit faults); fp32 for one trunk round (no round dominates; a training change); the fp32 trunk (+29 % `train_ms`; the defect is the statistic) | `../training/compile_flags.md` "bf16 drift on a sharpening policy"; `src/agents/model/compile_regions_loss_rows_test.py`; `../research_state/measurements/x26_bf16_drift_2026-10-17/` |
| 2026-10-17 | **X26 resumes after its K9(b) stop under `--behaviour-check warn` with fixed GIGO ceilings** (PREREGISTRATION Amendment 3, registered before the resume): stop on ANY update with judged max ≥ 1e-3, p99 ≥ 1e-4 or a non-finite |Δ|, read by hand via `main.ops.tb_read` at every tick, every restart and daily; resume from `final_model_exception.zip` (after update 13,899) at the same pin; restart interval 24 h (the owner may revert), with a VmRSS growth guard (> 2 GB ⇒ a manual restart) | (A) resume `fatal` (≈ 110 predicted rounding stops by 500M); `off` (no detector); an fp32 trunk (the rounding is in the fp32 forward on both sides); `checkpoint_136000153` (discards ≈ 975k clean steps) | the u13900 stop diagnosed NUMERICAL, no checkpoint void: `177f61c0`, `designs/research_state/measurements/x26_k9b_stop_2026-10-17/` |
| 2026-10-17 | **X26's startup-gate refusals get an OPS procedure** (§4.2 "STARTUP-GATE REFUSALS during X26"): loss line only ⇒ resume from the newest checkpoint pre-flighting ≤ 0.007; per-param line confined to `move_resolution_cell.*` ⇒ GPU `canary_replay` (E32/Ebf/Cbf) on the newest ~4 checkpoints, resume from the newest passing both slices; anything else ⇒ HOLD. Applied once: X26 resumed from `checkpoint_136000153` (≈ 975k steps lost) after `final_model_exception.zip` refused at 19.3× (21× in a fresh compile) | a fresh-compile retry of the refused weights (reproduces); `--no-compile-trainer` (~1.75× slower, different numerics) | the GPU replay of 8 checkpoints, 2026-10-17 (§4.2) |
| 2026-10-17 | **Amendment 3's K9(b) forecast is REFRESHED at 211.9M; its margin argument no longer holds under the best-fitting growth form, so a pre-registered WATCH TRIGGER is RECOMMENDED (not registered here; the ceilings are unchanged).** Mechanism: every dump that carries weights (u19300 / u20490 / u20630, plus the u13900 control) is the 177f61c0 NUMERICAL class: the violator ranks 1st or 2nd of 33 rows on fp32 error and one-ulp dither spread, with a gate gap of 3.2–3.7 dither σ (inside the other rows' range). u21290 / u21440 have no weights (3 weight dumps per process) and pass the record-level checks only. Forecast: the event COUNT matches 177f61c0's table (5 after 137M vs 5.4 expected), but the SCALE grows faster than linear. With the exponential form (best AIC; the data cannot yet exclude linear) the chance of crossing BEFORE 500M is ≈ 0.66 for the p99 ceiling (1e-4; first crossing 5 % / 50 % at 439M / 492M) and ≈ 0.25 for the max ceiling (1e-3; 5 % at 409M). Under linear or power growth both chances are ≤ 0.04. The cap margin for p99 is 1.4× (exponential) or 3.2× (linear), against A3.1's "≥ 4×". The proposed trigger: W1, a closed 10M window with median p99 ≥ 3.0e-5 (≈ 341M under the exponential form, ≈ 482M under linear); W2, any probe with p99 ≥ 5e-5 or judged max ≥ 5e-4. Either one ⇒ the A3-F2 re-diagnosis (replay + refreshed forecast + a decision before the forecast's 5 % crossing), not a stop. Refresh the forecast at every 25M read point. Also found: A3.3's "resumes from `final_model_exception.zip`" is stale (the live child resumed from `checkpoint_136000153`, §4.2) | (a) no change (A3.1's ≥ 4× margin rests on the linear form, which the hold-out shows under-predicts by up to 26 %); (c) a re-diagnosis now (the mechanism holds, and both statistics sit 5–6× below their ceilings) | `designs/research_state/measurements/x26_k9b_tail_refresh_2026-10-17/` (README, `replay_dumps.py`, `forecast.py`) |
| 2026-10-17 | **The K9(b) WATCH TRIGGER is REGISTERED (PREREGISTRATION Amendment 4)** before any trigger fired: W1 a closed 10M window with median p99 ≥ 3.0e-5, W2 any probe with p99 ≥ 5e-5 or judged max ≥ 5e-4 ⇒ the A3-F2 RE-DIAGNOSIS (`replay_dumps.py` + `forecast.py`, pinned code, CPU), NOT a stop; NUMERICAL ⇒ continue, NOT ⇒ HOLD per §A2.7(3). The Amendment-3 ceilings are unchanged; `forecast.py` re-runs at every 25M read. Also corrects §A3.3: the live child resumed from `checkpoint_136000153` (A4-F2). | Orchestrator, under the owner's 2026-10-17 full-discretion grant | the tail refresh `d29f781c`: P(p99 ≥ 1e-4 before 500M) 0.66 under the best-AIC exponential form, ≤ 0.04 linear; A3's ≥ 4× margin is 1.4× / 3.2× |
| 2026-10-18 | **I21 BUILT (T35): the maturity-battery harness, generalised** | I21's row now reads BUILT: a series registry with the pin (+ pad) entry, the 25M trigger and the read-only sha256 checkpoint copy, the frozen bank sampler, R-MATURITY as tested code, LEAN / FULL modes and a detached resumable queue (`src/main/hidden_info_study/maturity_*.py`; drivers in `measurements/hidden_info_maturity_x26_2026-10-18/`). It reads retained checkpoints OFFLINE on CPU beside the live run (rule 5b), so it touches no X26 process or file | extending the measurement scripts' hardcoded {end2, prod} once more | `design_hidden_info_binding.md` §12.13.7 (the harness paragraph) |
