# The NEW LINEAGE (clean inputs, Rust-core observation) — PRE-REGISTRATION (2026-09-26)

**Status: REGISTERED before any run of the lineage exists.** Written 2026-09-26 at `origin/main`
`8d07051a`, when `models/ai_v14_01_base` did not exist and no `ai_v14_*` directory existed. This session
trained nothing, touched no GPU and wrote nothing under `models/`. The launch kit is
[`measurements/new_lineage_2026-09-26/`](measurements/new_lineage_2026-09-26/README.md) (PREPARED BUT
NEVER EXECUTED). The Training Run session launches the base run after the orchestrator's go.

**Method template:** population loop round 2 ([registration](population_loop_round2_2026-09-24.md),
[read](measurements/population_loop_r2_2026-09-24/read/README.md)). Every rule of evidence there carries over
unless this file says otherwise, with the reason.

**What the new era is (owner decisions, 2026-09-24/25/26).**
- **Clean inputs.** The training-input boundary is **`b0a28b5b`**, the observation-architecture batch
  (`gen3_event_record_v2`: obs 2501 → 2761, model config v121, the E10 Smogon hidden-slot move mixture, the
  E12 event rows, the Mud/Water Sport slots). Every GIGO fix of the week is folded in (ledger 2026-09-24
  *TRAINING-INPUT BOUNDARY `c97358e8`*, *the Rust core M3 loss catalogue's GIGO is fixed*; 2026-09-26 *THE
  OBSERVATION-ARCHITECTURE BATCH LANDS*). `MIGRATION_FLOOR` is 121, so **no existing checkpoint loads on
  HEAD, and this is a FRESH lineage from scratch.**
- **Infrastructure.** The Rust core builds the trainee's observation (`--obs-source core`, the production
  default on the rust bridge since `ac0b6469`). `--arch production` applies the production architecture surface.
- **Opponent mix: the POOL only.** The trainer draws teams from `data/teams/{sample,others}`
  (`utils/team_loader/loader.py`); no Metamon ladder team is in either, and none is passed. The ladder teams
  enter only after their Hidden Power fields are repaired (next week), as a SEPARATE EXPERIMENT ARM that
  reproduces this lineage's recipe with them added (§7).
- **The population loop carries into this lineage** with BOTH power levers registered up front: readers at
  `--eval-battles 200`, and a POOLED multi-round read declared here, before round 1 (§5).

**Codes, each described once here and again wherever they recur:** **N0** = `ai_v14_01_base`, the fresh
base run. **K1…K3** = continuation blocks of N0 at the frozen generalist dose (§3.1). **G0′** = the
lineage's plateau parent (the first continuation block that passes the plateau test). **A′** / **A2′** =
round-0 offense exploiters of G0′ (seed 1001 / 1002). **B_r** / **C_r** = the round-r loop generalist /
no-exploiter control. **RB_r** / **RC_r** = the fresh offense reader (best responder) trained against B_r /
C_r. **G-U** = the untaught-8 KILL guard. **G-A** = the SmallRL external-anchor KILL guard.

---

## 1. The base run N0 — the argv, and every departure from the recipe

**The recipe of record** is the last production lineage's fresh root, `ai_v13_02_flywheel_winprob` (the root
of `ai_v13_09_wcont` → `ai_v13_12_plateau` (G0, the old plateau parent) → every population-loop arm and the
core-obs burn-in; `python -m main.lineage ai_v13_33_core_burnin`). Its recorded `original_command` is copied
verbatim to `measurements/new_lineage_2026-09-26/scripts/template_ai_v13_02_flywheel_winprob.txt`;
`scripts/build_argv.py` applies the registered moves and REFUSES any other:

| flag | recipe | N0 | why |
|---|---|---|---|
| `--run-name` | `ai_v13_02_flywheel_winprob` | **`ai_v14_01_base`** | new lineage; the `ai_v14` prefix marks the era (owner may rename, §8) |
| `--pin-commit` | `6eb9c776` | **`8d07051a`** (origin/main HEAD at registration; ≥ `b0a28b5b`; its only change over `b0a28b5b` is the ledger/backlog text) | the clean-input boundary |
| `--arch production` | absent | **added** | the launch guard (`gen3_arch_surface_guard_v1`): the production surface applied for every flag the argv leaves unset. It changes nothing here (checkargs: every ARCH-surface key already matches) and makes the surface an assertion rather than a coincidence |
| `--obs-source core` | absent (did not exist) | **added** | already the default on the rust bridge; named so the argv says it |

**Everything else is the recipe, token-exact (235 tokens):** `--steps 75000000` (lands **75,005,952**,
763 rollouts of 48 × 2,048 = 98,304), `--n-envs 48`, `--n-steps 2048`, `--batch-size 2048
--grad-accum-steps 32` (effective 65,536; the full-surface shape on the 12 GB card), `--n-epochs 10`, `--lr
0.0003 --min-lr 1e-05` (the KL controller anneals it), `--ent-coef 0.05`, `--clip-range 0.15`, `--seed 1001`,
`--self-play --self-play-temp 1.0 --self-play-use-cpu`, `--eval-battles 100 --eval-sentinel-greedy`,
`--snapshot-ladder-games 100`, `--restart-interval-hours 3`, `--use-bridge rust`, `--compile-trainer
--compile-opponents --compile-opponents-strict`, `--no-value-true-team`, `--team-pfsp off
--team-block-episodes 1`, and the whole production architecture flag family.

**Budget and cadence, justified.** 75M is the recipe's own budget and the budget of BOTH win-prob-era fresh
roots (`ai_v12_02_winprob_critic`, `ai_v13_02_flywheel_winprob`, each ending at 75,005,952). Keeping it makes
N0's endpoint the matched-depth comparator for the one cross-lineage descriptor worth having (§3.3). Eval
every 2M on absolute steps, restarts every 3 h and checkpoint cadence are the recipe's defaults, unchanged.
`--eval-battles` stays 100 on N0 and on every generalist (B_r, C_r): the manipulation check it feeds was
ABSORBED in both old rounds at n = 600 per arm; the power problem is the readers' (§5).

**What the critic and the reward ARE (read from the resolved config, not assumed).** `--critic winprob`:
V(s) = sigmoid(win-head logit) = P(win | s); the value loss is that head's BCE against the terminal outcome,
weighted by `--vf-coef 0.5`; the reward is the TERMINAL WIN INDICATOR alone (`--no-hand-shaping
--terminal-indicator --victory-value 1.0 --draw-penalty 0`; implied `--win-prob-mode shaping`, γ = 1.0,
PopArt off). This equals `designs/production_config.json` (`critic: winprob`, `hand_shaping: false`,
`terminal_indicator: true`, `win_prob_mode: shaping`, `no_progress_tax_armed: false`). The shaped-reward flags
in the argv (`--mat-alive-weight`, `--no-progress-penalty`, …) are INERT under this critic (the old root's
`model_config.json` lists them in `inert_reward_flags`). **Stall rate and mean episode length are therefore
PRIMARY endpoints of N0, not monitored ones** (runbook §*WHICH readout is the critic*): nothing in the
objective prices a 250-turn timeout below a loss.

**"It is the experiment" (SOP §1 items 4–6).** checkargs and the dry-run print `✓ every ARCH-surface key
matches the production mirror` (39 of 51 registry toggles). The 12 keys outside the surface (the critic
readouts and supervision doses) are typed in the argv or at a default equal to production; the old root's
RESOLVED `model_config.json` differs from `production_config.json` ONLY in the keys the obs batch moved
(`active_context_dim` 58 → 60, `total_dim`, `config_version`, `arch_signature`) plus run-level keys the mirror
does not carry. **Still owed at launch (child-only):** the Training Run pastes the diff of N0's own
`model_config.json` against `production_config.json` into the launch entry; the expected diff is the
run-level keys only. **Measured against:** no registry baseline loads on HEAD (every entry is
`era_checkout_only`), so N0 is read against its OWN snapshots, fixed external anchors, and (descriptor only)
the old root at matched depth.

---

## 2. What gets read, and when

| when | read | by | kind |
|---|---|---|---|
| first 2 min | the STOP-list in `launch_base.sh` (the preload layer's only test) | Training Run | STOP on any miss |
| hours 0–6 | the watch list, §4 | Training Run | NOTIFY / STOP per §4 |
| every eval (2M) | `win_rate_vs_bots`, greedy sentinels (`eval_sentinel_greedy` true, symmetric teams), `eval/mean_ep_len_*` | SOP layers | monitored |
| 24,000,000 | `snapshots/snapshot_000024000000.zip` exists → registered as G-U's fixed opponent (§3.2) | orchestrator | a baseline registration, no read |
| 75,005,952 (N0 ends) | **N0 base read (descriptors):** untaught-8 level vs the new opponent; SmallRL 1,200 greedy games; `snapshot_ladder/ladder.json` (fit at run END only); stall and episode length over the run | reader session | descriptor |
| each continuation block | the plateau test (§3.1) | reader session | gate |
| round 0 | F′ = the reader floor (§5.2); A′, A2′ become round 1's specialists | reader session | instrument |
| each round r | manipulation check M_r; G-U and G-A on B_r − C_r (cumulative); per-round Δ_r (DESCRIPTOR) | reader session | guards + descriptor |
| after round R = 4 | **THE POOLED READ** (§5.3) | reader session | the verdict |

---

## 3. From N0 to the loop's parent

### 3.1 The continuation blocks and the plateau test (stage 1 of the ladder campaign)

The old lineage reached its plateau parent as 75M root → +12M continuation (+15.50 pp untaught, replicated
+12.94 on seed 1002) → +8M block (−1.50 pp [−3.75, +0.62], WITHIN the 3.69 floor) (ledger 2026-09-20 *THE
PLATEAU IS REACHED AT BLOCK 1*; 2026-09-23 *THE WIN-PROB CONTINUATION'S UNTAUGHT GAIN REPLICATES*). The new
lineage repeats that construction with a fixed block size:

- **Block K_k** = a FORK of the previous block's `final_model.zip` (K1 forks N0's), **+8,000,000 requested ⇒
  +8,060,928** (82 rollouts; every fork step here is a multiple of 98,304), the recipe's argv with `--model`,
  `--run-name`, `--steps` and **`--fork-lr D_g --fork-lr-freeze`** moved, `--checkpoint-every-steps 500000`
  added (as every old-lineage fork carried).
- **D_g, the generalist dose, is fixed by a RULE, not a number:** N0's own final learning rate
  (`python -m main.dose ai_v14_01_base`), two significant figures, frozen. This is the old lineage's
  construction (its 2.8e-5 was the root's annealed rate, 0.20× the v8 dose at this batch shape). If N0's
  rate is more than 2× away from 2.8e-5, that is a FINDING reported before K1 launches. **Every later
  generalist (B_r, C_r) uses the same D_g.**
- **PLATEAU iff** the block's untaught-8 Δ (block end − block start, paired over the 8 teams, the meter's
  own bootstrap) has its point estimate inside ±3.69 pp AND its CI does not exclude +3.69 from above
  (i.e. it is not OUTSIDE ABOVE). **G0′ = the END of the first block that passes.** Cap: **3 blocks
  (+24.2M)**. At the cap without a pass, G0′ = K3's end, labelled NOT PLATEAUED, and the loop proceeds (its
  control arm C_r absorbs any residual learning by construction); the label travels with every read.

### 3.2 The instruments that must be RE-BUILT for v121 (none loads across the era wall)

- **G-U's fixed opponent.** The old opponent (`untaught_meter_opponent` = `ai_v9_29_rev1_0823@24M`, config
  v101) cannot load on HEAD. **Registered replacement: N0's own `snapshot_000024000000.zip`**, the same depth
  of a fresh run as the old opponent, entered in `designs/baselines.json` under a NEW name
  (`untaught_meter_opponent_v14`, via `python -m main.baselines set … --reason`), meter flags `--opponent
  untaught_meter_opponent_v14 --config auto`. **A new opponent is a re-measurement:** no untaught level of the
  new lineage is comparable with any old-lineage level.
- **The G-U floor, 3.69 pp, is CARRIED PROVISIONALLY.** It is the old continuation read's replicate floor
  against the old opponent. Re-measuring it needs two generalist replicates (≈ 9 GPU-h), not bought here (§8).
- **The G-U reproduction check** (old: `plateau_b1` 963/1600 exactly) becomes: **G0′'s untaught read is
  re-run IN FULL in every round and must reproduce its first read EXACTLY** (a cell is a pure function of
  ref, team, battle), or that round's guard is VOID.
- **G-A's floor, 0.110, carries** (the SOP's three-seed run-level floor for SmallRL; an external opponent).
  G-A runs at the lineage pin (v121 loads nowhere older), `--server rust`.

### 3.3 One cross-lineage descriptor (never a verdict)

N0 at 75,005,952 vs the old root `ai_v13_02_flywheel_winprob` at the same step, on SmallRL greedy (1,200
games each; the old model plays from a tree ≤ `56837827`). Reads "did clean inputs move the external level at
matched depth". Confounded by the tree (anchor code) and by every GIGO fix at once; it attributes nothing to
any one fix.

---

## 4. The watch list — the first hours of N0 (Training Run)

### 4.1 Minutes 0–2: the STOP-list (printed by `launch_base.sh`)
Role FRESH, pin `8d07051a`, obs source core, transport rust; `--arch production` applied with the ✓ line;
obs dimension 2761 and signature `gen3_event_record_v2`; no `[ModelVersion] FATAL` (the "Round-trip smoke
test PASSED" line is NOT printed by a real launch, P-2 of round 2); no `🐴 [STABLE]`, no `[ForkLR]`; the
first rollout lands.

### 4.2 Hours 0–6
| item | how it is read | reference | rule |
|---|---|---|---|
| **`__OBS__` / core-obs refusal** | `CoreObsMismatch` (`agents/battle/core_obs.py`: no frame, wrong battle / decision / turn, NaN cell, mask disagreement) or `__OBS__ frame nobody asked for` (`utils/bridge/bridge_session.py`) in the child log or `crashes/` | the burn-in: none | **ONE occurrence ⇒ STOP the launcher (do not let it auto-restart) and escalate.** It is a CUTOVER-class divergence (`program_rust_core.md` §2 M6); the default reverts to `python` until fixed. Whether N0 then resumes on `--obs-source python` (byte-identical by construction) or waits for the fix is the orchestrator's call (§8) |
| **throughput** | `python -m main.ops.tb_read ai_v14_01_base --tag time/fps --tag train/selfplay_fraction` (fps is uninterpretable without the self-play regime) | the burn-in `ai_v13_33_core_burnin`: median **553.5** fps over its last 10 rollouts (min 548, max 562; self-play fraction 0.90), read 2026-09-26 05:55 PT | median over a 1M window after compile settles **< 440 (−20 %) ⇒ NOTIFY** with the profile; **< 275 (−50 %) ⇒ NOTIFY + the orchestrator decides** (it doubles the lineage's wall time). New cost not yet measured: the model forward on the 2761-dim obs with 30-column event rows and a third attention link |
| **the 250-turn stall rate** | **count of `[STALL LOGGED]` lines in `launcher_child.full.log` per 1M trainee steps** (every env logs a stall; `train_rl_agent.py` hands one `StallConfig` to all 48). Convert to a per-episode rate with `rollout/ep_len_mean`: episodes ≈ Δsteps / ep_len | the burn-in: **1,496 stall lines over 111,378,432 → 140,771,328 (29.39M steps) = 50.9 per 1M steps**; with ep_len ≈ 49 that is ≈ 600k episodes, **≈ 0.25 % of episodes** | RECORD from the first 1M on. 🚨 **Do not divide by the child log's `🏁 Episode Finished` lines:** only env 0 prints them, and only when the episode reward is non-zero, i.e. **WINS ONLY under the win-indicator reward**. The burn-in's 1,496 / 4,326 = 34.6 % is that cross-population ratio (FINDING S-1). Rule: > 3× the burn-in's per-1M rate in two consecutive 1M buckets after 10M ⇒ NOTIFY (a primary endpoint of a winprob arm) |
| mean episode length | `rollout/ep_len_mean` via `tb_read`; `python -m main.ops.stall_exhibit ai_v14_01_base` for the stall-vs-sawtooth table | burn-in median 47.4 (a 111M+ model: NOT a target for a fresh run) | descriptor; read beside the stall rate, never alone |
| crashes | `crashes/restart_err_*.txt`; the launcher allows 3 crash restarts | — | a second crash with the same signature ⇒ STOP and escalate |

---

## 5. The population loop in this lineage

### 5.1 Round structure (the old loop, carried, with one registered change)

- **Round 0.** A′ = the offense exploiter of G0′ at arm A's recipe (`ai_v13_18_teach5_offense_hidose`'s
  recorded `original_command`, token-exact except `--run-name`, `--model`/`--exploiter` → G0′'s final, `--steps`
  → G0′'s step + 8,000,000, `--pin-commit` → the lineage pin, and **`--eval-battles 100 → 200`**): the five
  offense teams, `--exploiter-keep-bots --exploiter-bot-fraction 0.5`, frozen `--fork-lr 2.5e-4
  --fork-lr-freeze` (1.78× v8), seed 1001, budget 8,060,928. **A2′** = the same at `--seed 1002`.
- **Round r ≥ 1.** B_r = B_{r−1} + 8,060,928 at D_g against stable set S_r at `--stable-opponent-selfplay-share
  0.40`, `--stable-opponent-pfsp`, retirement off (`--stable-opponent-mastered-wr 1.01`), no distillation,
  seed 1001; C_r = C_{r−1} + 8,060,928, the same argv at share **0.0** (S_r loaded and evaluated, never trained
  against). B_0 = C_0 = G0′. RB_r / RC_r = fresh readers at A′'s recipe (`--eval-battles 200`) targeting B_r's /
  C_r's final. B_r and C_r run ADJACENT, and so do RB_r and RC_r (matched contention).
- **The stable set — the ONE change from the old loop (P-1 of round 2):** **S_1 = {A′, A2′}; S_r = {A′, A2′,
  RB_{r−1}} for r ≥ 2** — a window holding the most recent reader, not an accumulating set. At share 0.40 with
  self-play fraction 0.90 the per-specialist exposure is 0.18 in round 1 and **0.12 in every later round**,
  where accumulation would fall to 0.12 → 0.09 → 0.072 and dilute the treatment the pooled read averages over.
  Round 2 of the old loop ABSORBED at 0.12, +17.0 pp [+7.3, +26.3] against its new specialist. (Owner may
  keep accumulation instead, §8.)
- **R = 4 rounds**, fixed now (§5.4).

### 5.2 The bar

**F′ = |gap(A′) − gap(A2′)|** (pooled, 800 games each), the reader's run-level floor at a fixed target on the
new lineage (rule 3: max pairwise |Δ| over replicates). **bar = max(F′, 5.0 pp).** The old F = 2.00 is NOT
carried: it is a property of the old instrument (100 games a cycle, the old target). If F′ > 5.0 the bar rises
and §5.4's power falls; that is reported, not renegotiated.

### 5.3 The primary: the POOLED read

- **Per round:** d_r = gap(RB_r) − gap(RC_r) = rate(RB_r) − rate(RC_r), each rate pooled over the reader's
  post-fork eval cycles against its target (greedy vs greedy; 4 cycles × 200 = **800 games per reader**; 5
  cycles if the fork step's residue mod 2,000,000 exceeds 1,939,072, identical for the round's two readers).
- **Estimand:** **D = (1/R) Σ_r d_r**, the equal-weight mean over the R = 4 registered rounds (every round has
  the same registered n). **CI:** fixed-effect Wald, Var(D) = Σ_r Var(d_r) / R², Var(d_r) = p_B(1−p_B)/n_B +
  p_C(1−p_C)/n_C, 95 %. It covers GAME noise within these runs; run-level (seed) noise enters through the bar.
- **Rule (the old one, on D):** OUTSIDE iff |D| > bar AND the CI excludes the bar point on that side;
  EQUIVALENT iff the whole CI lies inside [−bar, +bar]; otherwise NOT DETECTED, which is never "no effect".
- **Mechanised:** `measurements/new_lineage_2026-09-26/scripts/pooled_rule.py --pair RB1:RC1 … --pair RB4:RC4
  --bar <bar> --json <out>`. It reads every reader through the meter's own `read_exploiter`, runs the meter's
  `check_matched` on ALL eight readers together (one budget / dose / regime mismatch ⇒ VOID), refuses a round
  whose readers share a target, refuses cycle sizes other than 200 and a round count other than 4, and writes
  nothing unless `--json` is given (round 2's K-1). **Validated on the old rounds as a STAND-IN (a mechanics
  check, not a read):** it reproduces round 1's −10.00 [−16.60, −3.27] and round 2's −8.25 [−15.01, −1.38]
  exactly, and pools them to −9.13 [−13.91, −4.34], NOT DETECTED at 400 games per reader.
- **Descriptors beside it, never folded:** each d_r with its Newcombe CI; Cochran's Q; each reader's curve and
  its convergence (cycles 3–4 minus 1–2, ≥ 5 pp on one reader and not the other is a named caveat); the two
  CHAINS round over round (`main.best_response_gap` with `--rounds` on A′ → RB_1 → … and A′ → RC_1 → …, F2).
- **Secondary (confirmatory):** only if D is OUTSIDE — `python -m main.best_response_gap <reader> --play 400
  --greedy`, seed 0, concurrency 1, on all eight readers; its pooled sign must agree for a detection to stand.

### 5.4 Power, stated before any number (`pooled_rule.py --power`, validation/power_table.txt)

At reader rates near 0.60 (old readers ran 0.52–0.69):

| design | games per reader | half-width | detection needs D below | P(detect \| true −8) | P(detect \| true −10) |
|---|---:|---:|---:|---:|---:|
| old, one round | 400 | 6.79 pp | −11.79 | 0.14 | 0.30 |
| one round at 200/cycle | 800 | 4.80 | −9.80 | 0.23 | 0.53 |
| pooled R = 3 | 800 | 2.77 | −7.77 | 0.56 | 0.94 |
| **pooled R = 4 (REGISTERED)** | **800** | **2.40** | **−7.40** | **0.69** | **0.98** |
| pooled R = 5 | 800 | 2.15 | −7.15 | 0.78 | 1.00 |
| pooled R = 6 | 800 | 1.96 | −6.96 | 0.85 | 1.00 |

**The registered design detects an OBSERVED D of about −7.4 or lower**, so an effect the size of the old
lineage's two rounds (−10.00, −8.25; mean −9.13) would be detected with probability ≈ 0.90 or more, and a true
−8 with ≈ 0.69. R = 5 buys 0.78 for ≈ +18 GPU-h (§8). EQUIVALENCE needs a CI inside ±5.0, i.e. |D| ≲ 2.6 pp at
R = 4: reachable now, unlike the old rounds. Assumptions: the per-round effect is roughly constant (Q is
printed to check), and F′ ≤ 5.0.

### 5.5 KILL guards (each round, on B_r − C_r, cumulative since G0′)

Read incrementally (owner rule 2026-09-24): minutes-long units, durable rows, resumable, detached, at the
lineage pin.
- **G-U, the untaught 8.** `python -m main.untaught_meter` with refs B_r, C_r and G0′ (the reproduction
  check, §3.2), `--opponent untaught_meter_opponent_v14 --config auto`, 200 games per team, `--seed 0`,
  concurrency 1, units of (ref × team × 25 battles); paired bootstrap over the 8 TEAMS (20,000 draws, seed
  20260915). **KILL iff B_r − C_r is OUTSIDE and BELOW the 3.69 pp floor** (|Δ| > 3.69 and the CI excludes −3.69).
- **G-A, SmallRL.** `python -m main.anchors --model <final_model.zip> --opponent metamon:SmallRL --regime
  greedy --teamset {away,home} --games 100 --team-seed S --seed-base S --device cpu --server rust`,
  `OMP_NUM_THREADS=1`, S ∈ {0, 10, 20, 30, 40, 50}, the same S for B_r and C_r: 24 units = 1,200 games per
  model; every row `regime_verified_decisions` true and `argmax_match_rate` 1.0 (H-3's complete-half rule
  carries). **KILL iff B_r − C_r (pooled 1,200, Newcombe) is OUTSIDE and BELOW the 0.110 floor.**

### 5.6 Manipulation check (each round, read FIRST)

M_r = [B_r's pooled greedy rate over its LAST TWO eval cycles against every specialist in S_r] − [C_r's same],
Newcombe 95 %, n = 2 × 100 × |S_r| per arm. **ABSORBED iff the lower bound > 0.** Per specialist beside it;
the one that matters is the specialist B_r has not seen before (RB_{r−1}).

### 5.7 Stopping rule and branches (registered now; no interim look at Δ is a verdict)

Per-round d_r are DESCRIPTORS. There is **no early stop for efficacy and none for futility on d**, and no
extension beyond R = 4 whatever D reads (an extension chosen after seeing D is a forking path).

| | condition | reading | next |
|---|---|---|---|
| **V** | a `check_matched` mismatch; a STOP-list miss (B_r's TB `train/stable_fraction` ≠ 0.36 or C_r's ≠ 0.00 after its first eval; B_r / C_r `matchup_hash` differ; a reader not landing at its registered step); the G0′ reproduction check fails | the piece is VOID | re-run the failed piece; counts nothing |
| **K** | G-U or G-A fires in ANY round | the loop bought exploitability with generality | **the loop STOPS at that round**; the rounds done are reported as descriptors; no pooled verdict |
| **M1** | round 1 NOT ABSORBED | the loop did not engage at this configuration | **STOP and re-register** with share 0.60 (0.27 per specialist); nothing to pool |
| **M_r, r ≥ 2** | NOT ABSORBED | the treatment weakened in that round | continue; named caveat on D |
| after R = 4: **T** | D OUTSIDE BELOW, the secondary agrees in sign, every guard clean | **THE LOOP TURNS on this lineage — a CANDIDATE** (one lineage, one seed; rule 22) | a seed-1002 replicate of the loop (B′/C′ chains + readers) is registered before any promotion |
| **R** | D OUTSIDE ABOVE | the loop makes the generalist MORE exploitable | stop the loop at this configuration |
| **E** | D EQUIVALENT | a positive null: the loop moves the gap by less than the bar, with the evidence to say so | stop the loop at this configuration; the next question is a different lever |
| **N** | D NOT DETECTED | not measurable at the registered power | **the loop question closes on this lineage at this configuration**; report D and its CI; no fifth round |

---

## 6. GPU queue and ETA (single GPU; the Training Run owns every launch)

At ≈ 550 fps (the burn-in; N0's own rate is unmeasured): **N0 ≈ 38 h**. Each continuation block ≈ 4.5 h (1–3
blocks). Round 0 ≈ 9 h (two readers; `--eval-battles 200` doubles every eval's games, so readers run longer
than the old 3.5–4.5 h, unmeasured). Each round ≈ 18 h (B_r, C_r, RB_r, RC_r). **Total ≈ 38 + 4.5–13.5 + 9 + 72
≈ 125–135 GPU-h, ≈ 5.5 days**, before CPU reads (G-U ≈ 1.5 h, G-A ≈ 2–3 h per round, overlapping the next
round's GPU work). N0 cannot start while the burn-in `ai_v13_33_core_burnin` holds the GPU (§8).

---

## 7. The ladder-teams arm (named now, registered later)

When the Metamon ladder teams' Hidden Power fields are repaired, a SEPARATE arm reproduces N0's argv with the
ladder teams added to the opponent-team draw (the mechanism and its share are that arm's registration). It is
compared with N0 at matched snapshot COUNT. 🚨 **Data hazard:** pinned runs read `data/` from the MAIN
checkout (cwd), not the pin; the repair must not change any file N0's lineage reads while an arm is live or
queued (the Metamon team files are not under `data/` today; verify where the repair lands before it lands).

---

## 8. Decisions that are the OWNER's (recommendation beside each; none is decided here)

> **RESOLVED by the owner, 2026-09-26 ~07:00 PT (added after registration, before launch):** (1) R = 4
> rounds; (2) the WINDOW stable set; (3) the untaught meter's new fixed opponent = N0's 24M snapshot
> under a new baseline name; (4) carry the 3.69 floor provisionally; (5) stop the burn-in at launch;
> (6) on a Rust-core `__OBS__` refusal: save the repro, then EITHER resume on `--obs-source python`
> (byte-identical inputs by construction) OR leave the GPU idle while the core is fixed — the
> orchestrator's call per incident (the registration's "wait" recommendation was withdrawn);
> (7) the `ai_v14` prefix.

1. **R = 4 vs R = 5** (power at a true −8: 0.69 vs 0.78; +18 GPU-h). *Recommend R = 4:* it detects the old
   rounds' size (≈ −9) at ≥ 0.90 and keeps the loop under a week.
2. **The stable-set WINDOW (§5.1) vs the old ACCUMULATING set.** *Recommend the window:* constant 0.12
   exposure keeps the pooled treatment homogeneous.
3. **G-U's new opponent = N0's 24M snapshot**, under a new registry name. *Recommend yes.*
4. **Carry the G-U floor (3.69) provisionally vs re-measure it** (≈ 9 GPU-h, two generalist replicates).
   *Recommend carry*, with the label on every G-U read.
5. **Ending the burn-in** to free the GPU for N0. *Recommend* stopping it at its next checkpoint on the go;
   it has run 29.4M clean core-obs steps.
6. **On a core-obs refusal:** resume N0 on `--obs-source python` or wait for the fix. *Recommend* wait for the
   fix (the lineage's point is the core path), `python` only if the fix exceeds a day.
7. **The `ai_v14` run prefix.** *Recommend yes* (the era is new; the lineage reads it).

---

## 9. Hazards and findings (each is a finding, not a footnote)

- **S-1 — the "31–33 % stall rate" could not be reproduced from the burn-in's files.** The one ratio near it
  that the files give is 1,496 `[STALL LOGGED]` lines / 4,326 `🏁 Episode Finished` lines = 34.6 %, and that
  ratio divides a 48-env count by env 0's WINS (`env_factory.py` sets env 0 alone to log; `report_episode`
  returns early on a zero episode reward, i.e. every loss and stall under the win indicator). The implied
  per-episode rate is ≈ 0.25 %. §4.2 registers an estimator with a stated denominator; the source of the
  31–33 % figure is asked of the orchestrator.
- **S-2 — `signal/stall_rate` is named in `src/agents/training/CLAUDE.md` and in `wrappers.py` as "the rate
  that watches the cap", but no code emits it** (`tb_read` on the burn-in: NO SCALAR FOUND). A doc/code
  finding for the training owner.
- **S-3 — the SOP's item 3 (a 60-s `--debug` CPU smoke of the actual argv) was NOT run:** a `--debug` run
  writes a run directory under `models/`, which this registration may not create. The Training Run runs it
  (or accepts the first two minutes of the real launch as the test) before launch.
- **S-4 — the runbook's fresh-run command block says `--ent-coef 0.02 --n-envs 64 --batch-size 16384`;** the
  recipe of record (and both win-prob-era roots' shapes) is 0.05 / 48 / 2048 × 32. A design block is not a
  launch command; N0 follows the recipe.
- **S-5 — every old-lineage instrument that loads a checkpoint is behind the era wall** (MIGRATION_FLOOR 121):
  the untaught meter's opponent, every named baseline, `main.ops.g7_report`'s `--parent` /
  `--famine-comparator`. §3.2 rebuilds the one the loop needs; G7 on N0 waits for a v121 baseline.
- **Carried from round 2:** F1 (no tool VERDICT on a one-archetype round; the rule decides), F2 (siblings
  need `--rounds`), K-1 (`main.best_response_gap` writes into cwd without `--json`), H-3 (Metamon's
  post-game `RecursionError` in either half), H-9 (a pin worktree needs its submodule and the `dist` /
  `node_modules` links before G-U).
