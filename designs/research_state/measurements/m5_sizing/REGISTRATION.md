# M5 SIZING STUDY — REGISTRATION (2026-10-01)

**Status: REGISTERED before any read of an arm.** It is written and committed before the
throughput sweep's first timed block and before any learning arm's first step. Spec:
`designs/endstate/program_rust_core.md` order constraint 5 ("SIZING — decided by measurement, not
picked") and the Decision-record rows it hands questions to: overlap, grouped forward, fewer active
snapshots, T2 buckets, fresh-run epochs and where eval runs. The recipe side is
`designs/endstate/design_learner_recipe.md` §2, §3.1–3.5 and §3.22. Owner: M5 Lane J (worktree
`m5-sizing`). Progress and resume point: [`PROGRESS.md`](PROGRESS.md).

**Read before registering (declared, so nothing hides behind it):**

1. **The lineage's noise-scale history** (§6, a DESCRIPTOR of existing TensorBoard data):
   `ai_v14_01_base` (N0), `ai_v14_06_lbat_ctrl_fix` (C_fix) and `ai_v14_07_g0p_k2` (K2). Its numbers
   are quoted in §6.
2. **An instrument smoke of the throughput harness at N = 256.** It checks that the harness runs at
   N > 48 at all. Its numbers are DISCARDED, not quoted.
3. **An operating-range pilot of the two strength meters on N0's own early checkpoints**, which
   are not an arm. It asks whether a fresh run at ~8M is off the meters' floor.
   - Untaught-8, 25 games per team: N0 @4.8M **24.0 %** (48/200); N0 @9.48M **45.0 %** (90/200).
   - The SmallRL guard on N0 @9.48M, away teams, 100 games, greedy (regime verified): **38 %**.

   So the untaught meter is far off its floor at 8M and moves at ~4.5 pp per M there. The
   orchestrator's fallback (secondary meters if untaught is on its floor) is not triggered.

Nothing else was looked at. No arm of this registration has run.

**Codes.** **N** = envs in the Rust env core. **D** = rows per update (the complete-game trigger's
target, `--rollout-target-samples`). **U** = the untaught-8 meter. **G-A** = the SmallRL guard
(greedy vs greedy). **E10 / E5** = 10 / 5 PPO epochs. **N\*** = the N this study picks.
**A, A′, B, C** = the learning arms (§5). **T2** = the inference service.

---

## 0. The outputs this study owes, and which part decides each

| # | output | decided by | form |
|---|---|---|---|
| O1 | the production **N** | Part T (throughput + staleness) with Part L's GUARD | a number + the rule's inputs |
| O2 | the **n_steps maximum** (= D_max / N) | §6 rule from the measured B_noise range | a number |
| O3 | the **adaptive-batch target and K bounds** | §6 rule | numbers; NOT adopted (arm A1 adopts, `design_learner_recipe.md` §5) |
| O4 | **slots, buckets, lanes** at N\* | §3 rules from the measured rows-per-slot distribution | numbers |
| O5 | the **overlap** verdict at N\* | Part T, arm vs arm | ADOPT / NOT ADOPTED |
| O6 | the **grouped-forward** rule (opponent inference > 30 % of the step?) | Part T split | TRIGGERED / NOT TRIGGERED |
| O7 | the **fewer-active-snapshots** gain at N\* | Part T `_p8` / `_p1` arms | a number for the OWNER; never adopted here |
| O8 | **fresh-run epochs** (is 10 at 3e-4 still right?) | Part L, C vs B | ADOPT E5 / KEEP E10 |
| O9 | **where eval runs**: the blocking eval's cost at N\* | Part L descriptor | a share; a rule for the background-filler build |

---

## 1. The regime (every part)

- **Interpreter:** torch 2.8 (`gen3ai_torch28`) only.
- **Precision:** fp32 `highest`, the production default (`matmul_precision` is not in the recipe;
  TF32 is K5's undecided question).
- **Core:** release builds of THIS worktree's Rust env core (`rust_env_proc`, process front end),
  T = 8 core threads (the trainer default). T is not swept; a CPU-bound env share at large N is
  reported as a FINDING if it shows up.
- **Production mix:** 95 % self-play / 5 % the 8 training-floor bots (in the core), keyed opponent
  sampling, T2 `graph` backend, lanes = min(slots, 8).
- **Ride-along heads, RND variants and every other detached head:** OFF.
- **Box rules:**
  - GPU only under `scripts/ops/gpu_lock.sh`, with the timeout inside the lock: ≤ 20 min per
    throughput hold, ≤ 120 min per learning arm (orchestrator, 2026-10-01: one hold per arm, the
    lock released between arms).
  - Every heavy job under `scripts/ops/mem_cap.sh`.
  - Logs and scratch in `~/.cache/gen3ai/tmp/sizing/`. Durable meter rows in
    `/home/goodlad/dev/gen3ai-reads/m5_sizing/`.

---

## 2. Part T — the throughput sweep (decides O1's throughput half, O4, O5, O6, O7)

### 2.1 Arms

**N ∈ {48, 128, 256, 512, 1024, 2048}.** The program doc's four, plus 128 and 512 to locate the
knee (declared addition). D = 98,304 at every N (the production rollout). It divides lcm(2,048, N)
at every N in the set, so n_steps = D / N = 2,048 / 768 / 384 / 192 / 96 / 48.

Per N, two processes, each its own T2 build, each one GPU hold of ≤ 20 min.

**(T-a) The A/B, `python -m main.rust_core_m5 throughput`:**
- The production arms `rust_serial_keyed` and `rust_overlap_keyed`, built once and timed in 6
  rounds of interleaved 20 s blocks (rotating order).
- 120 untimed warm-up host steps per arm (≥ 2 game lengths, so every env is mid-game at the first
  block).
- Trainee = `ai_v14_06_lbat_ctrl_fix/final_model.zip` (the learner's compiled sampling forward on the
  Python side, T2 on Rust). Pool = its last 20 snapshots. Both are the same as Lane G's N = 48 read,
  for continuity.
- **At N\* only, one more process (T-a′, its own ≤ 20 min hold):** `rust_serial_keyed` against
  `rust_serial_keyed_p8` (the fewer-active-snapshots candidate, 8 of 20) and `rust_serial_keyed_p1`
  (the ceiling of any fan-out fix), all on one service, same blocks.

**(T-b) The component split, `python -m main.rust_core_m5.fanout --real-flushes 24`** (this
study's extension, shipped in `a73a8feb` before this registration):
- One serial production collector at the same N, buckets and mix.
- 300 warm-up host steps record the rows-per-slot histogram and the distinct slots per flush.
- 24 consecutive REAL flushes are then harvested (each step's trainee rows per trainee slot, and
  p2's policy rows per opponent slot, copied).
- Each flush is replayed whole and as its trainee part alone, interleaved in seeded random order,
  6 rounds × 5 replays.
- **Opponent inference** = whole − trainee part, per flush. Its share of the flush comes with a
  95 % bootstrap CI over flushes.

**(T-c) The update cost, once (it does not depend on N):** `python -m main.compile_inventory run
--stage time --device cuda --matmul-precision highest --keep-prewarm --unbracketed`.
- Same pinned buffer and checkpoint as the K8 acceptance read.
- Shape: 98,304 rows, micro 2,048, K 32, 10 epochs.
- One hold, two units. **U_E10** = their mean unbracketed update wall.
- U_E5 is taken from arm C (§5), not projected.

### 2.2 Meters (per arm, per N)

- Trainee decisions/s [95 % CI over blocks]; ms per host step; CPU µs per decision (whole process
  tree); GPU utilisation.
- The collector's components per host step: core, submit, flush call, GPU wait, draws, arena write,
  post, complete-game fill. These are the harness's existing timers.
- From (T-b): trainee inference ms, opponent inference ms and the opponent share; rows-per-slot
  histogram; distinct opponent slots per flush.
- **Staleness, predicted:** the share of an update's rows played by the previous policy version is
  ≈ N · E[L²] / (2 · E[L] · D), where L is a game's trainee length. E[L] and E[L²] are read from
  the arms' own finished games. **Measured** in Part L (`staleness/*`, Lane G's probe). At N = 48
  Lane G measured 1.1–1.4 %.
- Every `*_after_freeze` counter (must be 0), and quarantines.

### 2.3 Decision rules (Part T)

- **Collection rate** r(N) = the better of serial and overlapped decisions/s at N (each arm's
  bootstrap mean).
- **End-to-end selection rate:** E2E(N) = D / (D / r(N) + U_E10). Eval is excluded: it runs on its
  own eval core at the same cost at every N.
- **O1, the throughput half.** **N\* = the SMALLEST N in the sweep with E2E(N) ≥ 0.95 ·
  max_N E2E(N), and predicted staleness ≤ 25 %.**
  - The smallest wins, because staleness, startup, memory and the bucket set all grow with N, and
    nothing in Part T pays for them.
  - A tie inside the null band (F-LG-10: identical arms differ by up to ~3 %) resolves to the
    smaller N.
- **O5, overlap.** ADOPTED at N\* iff overlapped / serial ≥ 1.03 there, with that ratio's 95 % CI
  lower bound > 1.0. Otherwise NOT ADOPTED. The ratio at every N is reported (the crossover).
- **O6, grouped forward.** TRIGGERED iff (T-b)'s opponent inference is > 30 % of the serial host
  step at N\*. The step is the collector's measured ms per host step in (T-a), not the flush. The
  CI is reported. Not triggered ⇒ the deferral stands.
- **O7, fewer active snapshots.** Report `_p8 / serial` and `_p1 / serial` at N\* from (T-a′), with CIs. It is a
  decision for the OWNER. It is never adopted here, and the arms run the full-pool rule.
  **Construction check (descriptor, CPU):** a simulation of the exact-PPS rotating subset (the
  Decision-record construction: iterative capping, a fixed-size systematic PPS design, within-subset
  pick ∝ p_i/π_i).
  - Weightings: uniform 20; a PFSP-skewed 20 with one capped opponent; and recency weights.
  - 10⁶ episode draws each. Reported: max |realised − p_i| against its binomial SE, and the
    analytic marginal computed exactly over the design.
  - It shows whether the construction realises p. It is not a training run.

---

## 3. Slots, buckets, lanes (O4) — rules, applied at every N; the numbers are recorded at N\*

- **Slots** are DERIVED, not tuned. They are whatever `OpponentPlan` declares for the production
  argv at N\*: the self-play window (20) + stable opponents + exploiter targets + the trainee
  (version pinning OFF: 1) + the eval core's extra slots. They are read from the arm's
  `[RUST ENV] T2 up …` line.
- **Lanes** = min(slots, 8), unchanged (T2's 1 / 2 / 4 / 8-lane read).
- **Buckets** = (8, b_opp, b_tr):
  - **b_tr = min(N, 512).** The trainee's N rows are served by T2's packing in chunks of the
    largest bucket. 512 is a DECLARED cap: per-row cost has flattened by 128 (23 µs/row, T2 read),
    and the graph pool's memory for a 2,048-row bucket is unmeasured beside the learner on a 12 GB
    card. `b_tr = N` at N = 2048 is NOT run; that is declared, and the cap stays UNVERIFIED as a
    speed choice.
  - **b_opp = the p95 of rows per opponent slot per flush at N with the full 20-snapshot pool**,
    from (T-b)'s histogram, rounded up to a multiple of 16. There is no b_opp when the p95 is ≤ 8
    (the 8 bucket holds it) or when b_opp ≥ b_tr.
  - **Bound (TIME, not rows):** the share of the opponent chunks' replay time that padding costs must
    be ≤ 15 %.
    - It is computed from the histogram under T2's packing (largest bucket first, then the smallest
      that holds the remainder) and T2's replay cost model: ms = 1.119 + 0.01465 · rows. That is the
      least-squares line through T2's per-bucket replay times, 1.06 / 1.23 / 1.97 / 2.94 ms at
      B = 2 / 8 / 48 / 128 (`static_buckets_graph.jsonl`).
    - It is a time bound, because padding a small bucket is nearly free (an overhead-bound replay)
      and padding a large one is not. A row count would reject today's (8, 48) for padding 1–3 rows
      to 8.
    - A violation adds ONE bucket: the multiple of 16 that minimises the waste. At most four
      buckets.
  - **Two-pass rule.** (T-b) runs first at N with provisional buckets (8, b_opp from
    1.5 × the predicted mean rows per slot, b_tr). The histogram does not depend on the buckets.
    (T-a) runs with the rule's buckets. If they differ from the provisional set, (T-b) is re-run with
    them, and only the re-run's split is quoted.
  - The rule is code: `scripts/bucket_rule.py` (`provisional N`, `rule <split json>`).
- At N = 48 the rule reproduces today's (8, 48), which is a check on it.

---

## 4. Order and stopping (Part T)

- (T-c) once, then for N = 48, 128, 256, 512, 1024, 2048: (T-b) then (T-a). Each N is one or two
  ≤ 20 min holds.
- A hold that times out is re-run once at the same config. A second timeout makes that N
  INCONCLUSIVE, with the reason recorded.
- No early stop on a "clear" N: the curve is the deliverable.
- **Failure at large N** (OOM, a refused T2 declaration, F-LG-4's arena overflow) is a FINDING. That
  N is reported as NOT FEASIBLE with its cause, and the fix is a separate commit before any re-run.

---

## 5. Part L — learning per sample (the GUARD on N\*, and O8)

### 5.1 Arms — FRESH runs (orchestrator-approved deviation from "one short fork each")

Why fresh:
- The harm from stale rows scales with how far the policy moves per update.
- A fork at the frozen fork LR (5.6e-5) moves approx-KL ≈ 0.003 per update (battery C: 0.0039).
- N0's fresh run at a KL-controlled 3e-4 moved **0.010–0.020 per update, clip fraction ~0.2**
  (N0 TensorBoard, 0.5–75M).
- So a fork would hide exactly the effect N could have. And the epoch question (O8) is a
  fresh-run question.

| arm | run name (`models/`) | N | epochs | seed | lever vs A |
|---|---|---|---|---|---|
| **A** | `sizing_A_n48_e10_s1001` | 48 | 10 | 1001 | — (`recipe.fresh` exactly, on the Rust core) |
| **A′** | `sizing_Ap_n48_e10_s1002` | 48 | 10 | 1002 | seed only: the REPLICATE |
| **B** | `sizing_B_n<N*>_e10_s1001` | N\* | 10 | 1001 | N |
| **C** | `sizing_C_n<N*>_e5_s1001` | N\* | 5 | 1001 | epochs (vs B) |

If N\* = 48, B is A, and C runs at 48 against A.

**Every arm:**
- `python -m main.launcher` from the main checkout, `--pin-commit <this registration's shipped
  commit>`, `--arch production --env-core rust --n-envs N --n-steps D/N --steps 8060928`
  (= 82 updates × 98,304), `--checkpoint-every-steps 2000000`, `--device cuda`,
  `--t2-buckets <§3 at N>`.
- C adds `--n-epochs 5`. `lr` stays `recipe.fresh`'s 3e-4 seed under the KL controller, so the
  controller may move it, and that is part of the lever.
- **Every other knob is the resolved `recipe.fresh` and the production defaults (eval on the core,
  every 2M).** `checkargs` must report only the typed levers: `n_envs`, `n_steps`, `n_epochs`,
  `seed`, buckets and the checkpoint cadence.
- One GPU hold per arm, ≤ 120 min inside the lock, released between arms. The launcher's restart
  stays at 3 h, so a normal arm never restarts. A kill resumes from the last checkpoint (≤ 2M lost).
- Order: **A, B, C, A′**. A′ is last so that a box drift lands on the replicate, not on a lever.

**Pre-flight (not a read):** for each N of the arms, a launcher `--dry-run` plus checkargs, then
one real launch to the first completed update. That launch goes to `~/gen3ai_archive/
m5_sizing_preflight/`, never `models/`, and is deleted after. It is "the first two minutes of a
real launch" at that N.

### 5.2 Meters (at the arm's final checkpoint, 8,060,928 steps; matched samples by construction)

**Primary — U:**
- Untaught-8, **600 games per team (4,800)**, `--seed 0`, concurrency 1.
- Opponent: N0 @24M (`untaught_meter_opponent_v14`, its file).
- The battery's per-battle body: `n0_endofrun_2026-09-27/scripts/gu_unit.py`, as fixed on main in
  `bfb8e7e2` (F-SZ-1: models used as loaded; pinned by `src/main/untaught_unit_script_test.py`).
- Durable fsynced rows, resumable 25-game units.
- Statistic: the meter's own `bootstrap_index` / `cluster_ci` over the 8 teams (20,000 draws, seed
  20260915), paired by team.

**Guard — G-A:**
- `python -m main.anchors --opponent metamon:SmallRL --server rust --regime greedy`.
- `--teamset away` and `home`, 6 seeds each × 100 games: 1,200 games.
- Δ with a Newcombe CI.

**Descriptors per arm (never ruled on):**
- `staleness/*`: the share of rows one version old, the ratio outside the clip band, approx-KL on
  age-1 rows.
- approx-KL and clip fraction per update.
- `train_ms` per update (U_E10 from A and B; U_E5 from C).
- `noise_scale_policy` / `noise_scale_ratio_policy`.
- The eval cycle's blocking wall.
- `rust_env/trainee_decisions_per_s`.
- Episode length.

**Infrastructure checks, recorded per arm** (orchestrator: these are the first real training runs on
torch 2.8 under K6/K8):
- every `lifecycle/cuda_*` tag;
- the canary results;
- K9(b)'s behaviour check;
- every `*_after_freeze`.

**A K6 FATAL, a canary disagreement or a CUDA memory-trend WARN is a MAJOR FINDING,** reported to
the orchestrator at once.

### 5.3 Decision rules (Part L) — honest about power

One seed per lever arm and ONE replicate pair (A, A′) cannot show EQUIVALENCE. The delta's CI from
games and teams does not contain run-to-run variance, and one replicate pair estimates that
variance with one degree of freedom. So:

- **N (the guard on O1).** Learning per sample VETOES N\* only on a DETECTED LOSS beyond the bar.
  - **VETO iff ALL of:**
    - U(B) − U(A) < −3.69 pp (the lineage's replicate floor, `new_lineage_2026-09-26.md` §3.1);
    - its 95 % CI upper bound < 0;
    - |U(B) − U(A)| > |U(A′) − U(A)| (the loss exceeds the observed replicate spread).
  - Or a G-A loss with its CI upper bound < 0 and |Δ| > 11.0 pp (the battery's G-A floor).
  - **Anything else reads "no loss detected", which is NOT "equivalent".** N\* stands on Part T's
    throughput and staleness.
  - On a VETO, N\* falls to the next smaller N of the sweep that clears Part T's 0.95 rule. ONE
    more B arm is run there. If that is vetoed too, N\* = 48.
- **O8, epochs.** C vs B (same N, same seed). **E5 is ADOPTED into `recipe.fresh` iff ALL of:**
  - U(C) − U(B) has its 95 % CI lower bound > −3.69 (the battery's non-inferiority rule, L21451);
  - |U(C) − U(B)| ≤ |U(A′) − U(A)| or the delta is positive (the delta is not larger than the
    replicate spread in the losing direction);
  - G-A's CI lower bound > −11.0;
  - C's update is faster (its `train_ms` median < B's, by the per-update CI).

  Otherwise E10 STAYS.
  - E5 is a recipe change, so it needs positive evidence. A "not detected" outcome keeps
    production and goes to the orchestrator as a decision.
  - The KL controller's response (C's lr trajectory, approx-KL, clip fraction) is reported beside
    the verdict (the Adam-overshoot risk, L05595).
- **The replicate itself.** |U(A′) − U(A)| > 3.69 at 8M means a fresh-run floor larger than the
  bar. Both verdicts are then flagged **"run floor exceeds bar — n = 1 is not decisive"** and go to
  the orchestrator. They are not re-run or extended here.
- Ruled at the registered n only. Interim looks are progress, never verdicts.

### 5.4 The G-A operating range (pilot, pre-registration)

`main.anchors` on N0 @9.48M, away teams, 100 games, read before this registration: **38 / 100**
(regime verified per decision). **G-A is OFF its floor at ~8M and stays the guard.** The fallback
below was written for a floor reading, and it is not triggered:
- **If the pilot had read < 5 %**, G-A at 8M would have been on its floor. It would then have been demoted to a
  DESCRIPTOR, and the secondary guard would have been the last eval cycle's vs-bots win rate (the
  run's own training-time series, greedy vs greedy).
- That substitution is decided by the pilot, never by an arm's result.

### 5.5 Stopping (Part L)

- Fixed length: 8,060,928 steps.
- No early stop on a meter.
- A crash resumes from the last checkpoint through the launcher.
- A fault that could change learning (a learner FATAL whose fix changes the update) restarts that
  arm fresh, and every earlier arm is re-checked for the same fault. Declared in PROGRESS.
- At most one extra B arm (§5.3).

---

## 6. The noise-scale meter, the adaptive batch and D_max (O2, O3)

**The descriptor read** (TensorBoard, `train/noise_scale_policy`, the EMA-smoothed B_simple of the
policy term; read 2026-10-01, before this registration):

| run | ~4M | ~10M | ~20M | ~40M | ~60M | end |
|---|---|---|---|---|---|---|
| N0 (fresh, 3e-4 KL-ctl) | 8.6k | 13.5k | 13.3k | 20.5k | 24.8k | 29.0k @75M |
| C_fix (N0 + 8M) | | | | | | 38.8k @83M |
| K2 (C_fix + 8M) | | | | | | 49.4k @91M |

Below ~3M the EMA is warming up (20 folds) and the reads (0.6k–3.7k) are not quoted as B_noise.
Each part of the lineage reads the policy over-batched at today's B_eff = 65,536: ratio 0.13 →
0.75. **B_noise moved ~6× from 4M to 91M**, under the ~10× at which `design_learner_recipe.md`
§2.1 says the √B_noise schedule would be worth building. The design effect c (§2.3) is UNMEASURED:
the split-half-by-game telemetry (Stage 0.5) is not built. So every B_noise here may read LOW by
up to a factor c. That is a FINDING carried into O3.

**Rules (applied in the verdict; Part L's arms add their 0–8M reads):**
- **O3:**
  - Target `noise_scale_ratio_policy` = 1.0, band 2.0. This is `design_learner_recipe.md` §3.4's
    PROPOSED setting; the owner's 0.91 (1.1× critical) lies inside the band.
  - **K_min = 2** (the estimator's floor).
  - **K_max = the smallest power of two with micro · K_max ≥ 1.25 · max measured B_noise_policy**
    over the lineage plus the arms.

  Not adopted: arm A1 adopts.
- **O2: D_max = the smallest multiple of lcm(2,048, N\*) ≥ m · micro · K_max with m = 4** (at
  least 4 optimizer steps per epoch at the largest batch, §2.2). n_steps_max = D_max / N\*. D_max
  sizes the row arena at startup (`--rollout-target-band`'s HI). The default target stays 98,304
  until arm A2.
  - **Memory check** (pre-flight): the collector's RSS with the arena at D_max at N\* must stay under
    the run's mem cap.
  - Otherwise D_max is lowered to what fits, and K_max follows from m.

---

## 7. Where eval runs (O9)

- Descriptor: every arm's eval cycles (blocking, in process, on the eval core: Lane H). Reported:
  the median cycle wall and the eval share of wall time at N\* (cycles × wall / the arm's wall).
- **Rule:** an eval share > 10 % at N\* puts the background-FILLER eval build (Lane H's deferred
  end state) on TASK_BACKLOG with that number. At or below 10 % the blocking design stands.

---

## 8. What is recorded where

- **Throughput JSONs:** `results/` here (`throughput_n<N>.json`, `split_n<N>.json`,
  `update_e10.json`).
- **Meter rows:** `/home/goodlad/dev/gen3ai-reads/m5_sizing/` (append-only). Aggregates are copied
  here when a read completes.
- **Verdicts:** PROGRESS.md, then the Decision records (`program_rust_core.md` and
  `design_learner_recipe.md`), and `designs/production_config.json`'s `recipe.fresh` iff O1 or O8
  changes a production value. The recipe doc gate and checkargs stay green in the same commit.
- **The ledger:** one entry at the verdict.
