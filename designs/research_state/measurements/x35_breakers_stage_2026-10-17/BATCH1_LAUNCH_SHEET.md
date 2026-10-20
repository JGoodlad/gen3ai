# X35 batch 1 + X47: the LAUNCH SHEET (staged 2026-10-19, nothing launched)

**Status: STAGED, every argv validated at the candidate pin `803a13e3`; nothing launched; every launch NEEDS THE
ORCHESTRATOR'S GO.** This sheet is what the Training session runs, in order, once X34 has stopped and the GPU window
(the battery, then the X14 A/B) is over. The design is [`design_compute_efficiency.md`](../../../endstate/design_compute_efficiency.md)
§5.6 (X35) and [`design_bot_ideas_screen.md`](../../../endstate/design_bot_ideas_screen.md) §2.1 (X47); the kit is
[`STAGED_PLAN.md`](STAGED_PLAN.md) §6. A design block is not a launch command: the lines below are the launch commands
only after GO, with `<PIN>` and (toy side) the T\* values filled in.

**Words used here.**
- **P\*** = X26's plateau checkpoint, `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`, sha256
  `814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed` (R's frozen call).
- **T\*** = X34's plateau checkpoint: the newest node of its second FLAT Tier-1 check. X34 (`rb_toy_ent_s4002_deep`)
  read FLAT_ONCE at 200M; a FLAT at the 210M check makes T\* the **210M node**. That file does not exist yet, so every
  toy line here was validated on the **200M node as a STAND-IN** (`checkpoint_200000244_steps.zip`, sha256
  `bb6271a3…0de057`). The stand-in is never a launch value.
- **C0** = the plain continuation control (the same fork with no lever). An arm is read only against the C0 of its own
  start checkpoint AND its own pin.
- **b-ON / b-OFF** = the arm with its lever / its C0, at the same seed.

---

## 1. The arms, in GPU order

One arm × seed holds the GPU at a time. The production block is the kit's `BATCH1` order (`main.x35.kit`); the toy
block's order is open choice C2 (§5), shown here in its recommended form. D1 keeps rank 2 (before A2): its demotion
needed X37a to read POPULATION / KAKUNA-DRIVEN **and** X37b to read TRANSFERS (`design_diversity_teams_opponents.md`
§6.1), and X37a's stage 2 read UNSEEN-DRIVEN.

**GPU-h basis (MEASURED from checkpoint mtimes, the SOP's meter):**
- **Production:** X26 250M → 280M took 22,501 s (1,333 steps/s); 270M → 280M took 7,365 s (1,358 steps/s). A 20M fork
  is therefore ≈ 4.1 h, plus ≈ 0.2 h for start and compile, ≈ **4.3 GPU-h**. A1 (+8 evaluated archive opponents) and
  A2 (+5 sentinels) pay ≈ 2 % more at equal steps, declared and not corrected (§5.6.1): ≈ 4.4. D1's per-step cost on
  forum teams is not measured: ≈ 4.3, **UNVERIFIED**.
- **Toy:** X34 150M → 200M took 10,595 s (4,719 steps/s). A 70M fork is ≈ 4.1 h + 0.2 ≈ **4.3 GPU-h**. X47's ρ (its
  wall time per step over C0's) is ESTIMATED at 1.3–1.4 from FLOPs (`design_bot_ideas_screen.md` F-BI-7): ≈ 5.6–6.0
  GPU-h per seed, measured on its first 1M.

| # | arm × seed | in plain words | parent | GPU-h | stop | matched control |
|---|---|---|---|---|---|---|
| 1 | **A1 s3501** | the ARCHIVE POOL: half the self-play games go against 8 frozen X26 nodes spread over 30–260M | P\* | 4.4 | fixed length: P\* + 20M = 300,000,170 | C0 s3501 |
| 2 | C0 s3501 | production control: P\* trained on, unchanged | P\* | 4.3 | 300,000,170 | — |
| 3 | C0 s3502 | control | P\* | 4.3 | 300,000,170 | — |
| 4 | A1 s3502 | archive pool | P\* | 4.4 | 300,000,170 | C0 s3502 |
| 5 | A1 s3503 | archive pool | P\* | 4.4 | 300,000,170 | C0 s3503 |
| 6 | C0 s3503 | control | P\* | 4.3 | 300,000,170 | — |
| 7–9 | **D1 s3501–s3503** (X38) | WIDEN THE TEAM POOL: half of each side's teams come from 2,424 human-built Smogon forum teams; eval stays on the pool | P\* | 3 × 4.3 | 300,000,170 | C0 s3501–s3503 (D1's own rule, `main.x35.d1`) |
| 10–12 | **A2 s3501–s3503** | PFSP: recent selves the trainee loses to are drawn more often (+5 sentinels feed it win rates) | P\* | 3 × 4.4 | 300,000,170 | C0 s3501–s3503 |
| | **production subtotal** | | | **≈ 52** | | |
| 13 | toy A1 s3601 | the archive pool on the toy (8 nodes of T\*'s own lineage, `rb_toy_ent_s4002` + `_deep`) | T\* | 4.4 | T\* + 70M | toy C0 s3601 |
| 14 | toy C0 s3601 | toy control | T\* | 4.3 | T\* + 70M | — |
| 15 | **X47 s3601** | DEPTH GROWTH: the toy's 2 attention rounds become 4; the 2 new rounds start as exact identities, so the network starts as T\*'s function bit for bit | T\* | 5.6–6.0 | T\* + 70M | toy C0 s3601 (PRIMARY at equal effective learn time, node s_eq) |
| 16 | X47 s3602 | depth growth | T\* | 5.6–6.0 | T\* + 70M | toy C0 s3602 |
| 17 | toy C0 s3602 | toy control | T\* | 4.3 | T\* + 70M | — |
| 18 | toy A1 s3602 | archive pool | T\* | 4.4 | T\* + 70M | toy C0 s3602 |
| 19 | toy A1 s3603 | archive pool | T\* | 4.4 | T\* + 70M | toy C0 s3603 |
| 20 | toy C0 s3603 | toy control | T\* | 4.3 | T\* + 70M | — |
| 21 | X47 s3603 | depth growth | T\* | 5.6–6.0 | T\* + 70M | toy C0 s3603 |
| 22–24 | toy A2 s3601–s3603 | PFSP on the toy (the calibration pair's plausible null) | T\* | 3 × 4.4 | T\* + 70M | toy C0 s3601–s3603 |
| | **toy subtotal** | | | **≈ 56** | | |
| | **batch 1 + X47** | | | **≈ 108 GPU-h ≈ 4.5 days of exclusive GPU** | | |

**The stop rule, every arm.** Each arm runs to its fixed `--steps` (P\* + 20M, T\* + 70M) and ends with the launcher's
`Training complete`. No arm stops on a score: the length is what makes it comparable with C0 (§5.6.1; X47's PRIMARY
reads the node s_eq inside its run, its SECONDARY the final, so X47 also runs to +70M). An arm ends early only on:
- **K9(b)** (the behaviour-policy check): production arms run `warn` with X26's Amendment-3 ceilings read by hand via
  `main.ops.tb_read` at every restart and daily (STOP on any update with judged max ≥ 1e-3, p99 ≥ 1e-4, or a
  non-finite |Δ|; rule 8 at 1e-9 relative). Toy arms: open choice C3 (§5).
- a `FATAL_*` exit (not restarted), or a 4th crash (the launcher restarts 3).
- the post-spawn pool-seed check of `launch_x35.sh` (T54), which SIGTERMs a fork whose seeded pool leaks post-P\*
  snapshots.

**Not on the GPU list.** R0 (the best responder of P\*, ≈ 1.1 GPU-h) has no registered argv (F27): choice C4. The R1
reads (≈ 0.3 GPU-h each in a lease window, or ≈ 12 h of CPU) and D1's Kakuna panels (CPU) go in the gaps between
arms, as §5.6.4 plans. Look 2 (seeds 3504–3505 / 3604–3605) is a later batch at the SAME pin.

---

## 2. The pre-launch checks (in this order; any failure is a HOLD)

`launch_x35.sh` runs checks 4–9 itself for the kit's arms (exit 7 = HOLD, reason in `~/.claude/jobs/x35/status.txt`).
X47 is not a kit arm (F-LS-1), so its checks are run by hand from the list below.

1. **X34 has STOPPED by its registered rule** (two consecutive FLAT Tier-1 checks, 200M and 210M), its launcher PID is
   gone (`ps -p`), and the stop is recorded. T\* is the 210M node: record its path, step and sha256 in the call
   (`sha256sum models/rb_toy_ent_s4002_deep/checkpoints/checkpoint_210*_steps.zip`).
2. **The GPU is clear**: the battery and the X14 A/B have finished, `nvidia-smi --query-compute-apps=pid --format=csv`
   lists nothing, and `scripts/ops/gpu_lease.sh status` reads FREE or the Training session's lease.
   `python -m main.x35.kit live` exits 0 (no training run live).
3. **The battery's T62 verdict is read** (`read_battery.py`): ADOPT, FP32_ALT, INCONCLUSIVE or INCOMPLETE. It picks the
   pin variant (§3).
4. **The pin** carries every commit in [`x35_required_commits.txt`](x35_required_commits.txt): T54 `cdab9cfe`, B-D2
   `c1b0c1f0`, the forum-corpus lift `f9360870`, F-BI-2 `4df10e65`, B-GROW `24616919`, B-GROW × F-BI-2 `aa5feaf5`, and
   T62 `bd3126e0` (all added 2026-10-19 except the first two; `launch_x35.sh` pre-check 2 enforces the file).
5. **P\*'s sha256** is `814823c5…91ed`, and `python -m main.x35.refpop verify reference_population_production.json`
   prints OK (it re-hashes P\*, the 8 archive nodes and R's 6 members, all in X26's run dir: never thin those 14 files,
   F-R3).
6. **The toy side's R is frozen** before the first toy arm: `refpop freeze --side toy` on T\* (§4.3). Not needed for
   the production block.
7. **No `models/rb_x35*` directory exists** for the arm (a restart RESUMES through the launcher, never this sheet).
8. **`data/` is FROZEN** from the first arm's launch until the last X35 arm ends: every X35 arm is a pinned run, and a
   pin isolates code, not data. The forum corpus D1 reads lives in `src/utils/team_corpora/` (pin-safe), not `data/`.
9. **The disk.** Today 288 GiB are free. The dry-run asks 10.97 GiB per production fork and 13.05 GiB per toy fork
   (each with a 25 % margin and a 4 GiB reserve, so the sum over-counts). The kit's pre-check 8 counts its 21 `BATCH1`
   forks: 230 GiB, a PASS at today's free space. Adding X47's 3 forks gives 269 GiB, which still fits, but with
   ≈ 20 GiB to spare: thin finished arms to their 10M nodes + final as they end (F22), and re-read `df` before each
   toy launch.

---

## 3. The pin: two variants

The battery's T62 gate ([`../gpu_window_battery_2026-10-19/README.md`](../gpu_window_battery_2026-10-19/README.md)):
adopt the float64 τ solve iff it costs ≤ 3 % of a training cycle (`cost_hi ≤ 3 %`); **FP32_ALT** iff `cost_lo > 3 %`;
INCONCLUSIVE in between (the orchestrator decides).

- **Variant T62 (ADOPT): `<PIN>` = main HEAD at launch**, a descendant of `803a13e3`, which is the commit every line
  here was validated at. Main HEAD carries T62 by default.
- **Variant FP32_ALT: NO SUCH COMMIT EXISTS (F-LS-2).** FP32_ALT is not a flag and not a revert. The battery README
  defines it as "the fp32 complementary-constraint alternative" from the T62 decision row in
  [`design_x5_belief_tokens.md`](../../../endstate/design_x5_belief_tokens.md): solve the COMPLEMENTARY constraint
  when the mass saturates, "a second code path and a second case split". That is a build: the code, tests that fail
  on revert, and re-recording the learner golden (`learner_golden.json`) and the h2h reveal file's `OFF_BITS` that
  `bd3126e0` / `803a13e3` re-recorded under fp64. A plain revert of `bd3126e0` is NOT FP32_ALT: it brings back the
  K9(b) τ-class divergence (X26's A3 / A6 stops) that T62 fixed, and it would also need `803a13e3` undone.
  - When such a commit lands on main, it is a descendant of `bd3126e0`, so `x35_required_commits.txt` holds in both
    variants. The argvs are unchanged: only `--pin-commit <FP32_ALT_PIN>` differs, and each line must be dry-run again
    at that pin (`validate_batch1.sh --pin <FP32_ALT_PIN>`).
  - **The toy is NOT affected by T62.** The toy has no belief tokens (`move_belief_mode off`, `opp_belief_cls_k 0`,
    no species prior). MEASURED here: the toy network's forward on its 64 recorded rows (`toy_network_identity_test`)
    calls `hypothesis_set.fixed_size_tau` **0 times**, against 20 calls in the positive control (the
    `hypothesis_set_test` presence cases), with a counting patch. Only the CPU forward was checked; the learner path
    has no belief loss on the toy (its belief coefficients are 0), which was not executed here.

What to do when the verdict is not ADOPT is open choice C1 (§5).

---

## 4. The launch lines

**Launcher flags on every arm:** `--restart-interval-hours 24 --pin-commit <PIN>` (24 h as X26 / X34). Every line is
run from the main checkout (`/home/goodlad/dev/gen3ai`) with `export PYTHONPATH=$PWD/src`, by the Training session that
holds the GPU lease. `launch_x35.sh` without `GO` is a dry run of every pre-check; `GO` launches and needs the
orchestrator's go.

### 4.1 Production (validated at `803a13e3` on the REAL P\*)

For each row of §1 (arms `a1`, `c0`, `d1`, `a2`, seeds 3501–3503), in order:

```bash
# NEEDS GO. <PIN> = variant T62: main HEAD (≥ 803a13e3); variant FP32_ALT: the FP32_ALT commit (does not exist yet, §3).
S=designs/research_state/measurements/x35_breakers_stage_2026-10-17
$S/launch_x35.sh a1 3501 --pin <PIN> --lease-owner <owner>          # dry run: every pre-check, prints the line
$S/launch_x35.sh a1 3501 --pin <PIN> --lease-owner <owner> GO       # only on the orchestrator's GO
```

The argv the kit builds is, flag for flag, the validated file `batch1_validation_803a13e3/argv_<arm>_s<seed>.txt`.
The launcher line it runs is:

```bash
python -m main.launcher --restart-interval-hours 24 --pin-commit <PIN> \
  $(cat $S/batch1_validation_803a13e3/argv_<arm>_s<seed>.txt)
```

| arm | the argv after `--model P* --behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda` | run name |
|---|---|---|
| C0 | (nothing) | `rb_x35_c0_s<seed>` |
| A1 | `--stable-opponents <N_8 … N_1 as zip:label, 260M … 30M> --stable-opponent-selfplay-share 0.5` | `rb_x35_a1arch_s<seed>` |
| D1 | `--team-mix w_forum_v1@ABC:0.5 --team-mix-sides both` | `rb_x35_d1wide_s<seed>` |
| A2 | `--pfsp-scale 2.5 --n-sentinels 10` | `rb_x35_a2pfsp_s<seed>` |

Each is followed by `--seed <seed> --run-name <run name>`.

### 4.2 Toy (validated at `803a13e3` on the 200M STAND-IN for T\*)

**At the call (CPU, rule 5b, after check 1 of §2):**
1. Plan the toy R on the real T\*:
   `python -m main.x35.refpop plan --side toy --pstar <T*> --archive-dir models/rb_toy_ent_s4002/checkpoints --archive-dir models/rb_toy_ent_s4002_deep/checkpoints --x pstar_x26=<P*>`.
   By the rule, T\* = 210M gives the archive N_1…N_8 = 20, 50, 70, 90, 120, 140, 170, 190M (95 and 142.5 resolve by
   "nearest, ties down") and the A-block 210, 200, 60, 100, 150M. That is hand-computed; the tool's output is the record.
2. Play the band cell T\* vs P\* (500 mirrored pairs, greedy vs greedy; the `plan` prints the `main.h2h play-many`
   line; ≈ 15–30 min on one CPU lane at the production band cell's 1.23 games/s), then
   `refpop freeze --side toy … --out $S/reference_population_toy.json`. P\* stays in the toy's X-block only if T\*'s score
   against it lies in [0.20, 0.80]: an empty X-block is possible and is a FINDING for the orchestrator, not a stop.
3. Build and validate every toy argv at the pin:
   `$S/validate_batch1.sh --pin <PIN> --toy-call $S/reference_population_toy.json --out /tmp/x35_toy_val toy_a1:3601 toy_c0:3601 x47:3601 x47:3602 toy_c0:3602 toy_a1:3602 toy_a1:3603 toy_c0:3603 x47:3603 toy_a2:3601 toy_a2:3602 toy_a2:3603`.
   Every line must read `checkargs=0 dryrun=0 would_launch=1 delta=[+70,000,000 steps]`, `role=[FORK of …rb_toy_ent_s4002_deep]`,
   and the X47 lines `grow=[DECLARED trunk_layers 2 -> 4 …]`.

**The kit's toy arms** (`toy_a1`, `toy_c0`, `toy_a2`) then launch exactly as §4.1: `$S/launch_x35.sh toy_c0 3601 --pin <PIN> …`.

**X47 is launched by hand** (not a kit arm, F-LS-1): the toy C0 argv plus ONE lever, `--grow trunk_layers=4`, and its own
run name. With the stand-in it reads (validated, `argv_x47_s3601.txt`):

```bash
# NEEDS GO. Stand-in values: T* = 200,000,244. At the call: <T*> and --steps = T*.step + 70,000,000, from the toy R file.
python -m main.launcher --restart-interval-hours 24 --pin-commit <PIN> \
  --model models/rb_toy_ent_s4002_deep/checkpoints/<T* zip> --behaviour-check fatal \
  --steps <T*.step + 70000000> --checkpoint-every-steps 1000000 --snapshot-ladder-games 0 \
  --supply-starve-cycles self_play_pool=0 --device cuda --grow trunk_layers=4 --seed 3601 --run-name rb_x35t_x47d4_s3601
```

Use the line `validate_batch1.sh` wrote to `argv_x47_s<seed>.txt` (it is built from the toy R file, so it cannot drift
from the call). `--behaviour-check` follows choice C3. Its hand checks before GO: §2 checks 1–9, and the dry-run's
`grow        : DECLARED trunk_layers 2 -> 4` line. After the spawn, read in the first minutes of the child log: the
fork-time growth gate (the grown network's logits and values bit-identical to T\*'s on real rows; a mismatch is
`FATAL_CONFIG`), the seeded pool GROWN the same way, and T54's pool-seed record (run
`python -m main.x35.kit pool-seed-check --arm toy_c0 --run-dir models/rb_x35t_x47d4_s<seed> --call $S/reference_population_toy.json --log models/rb_x35t_x47d4_s<seed>/launcher_child.full.log`).
Then start the SOP's four layers (watcher, chain, Monitor, 55-min cron) by hand, as `launch_x35.sh` does for the kit arms.

### 4.3 What the dry-runs showed (24 of 24 PASS at `803a13e3`)

[`batch1_validation_803a13e3/`](batch1_validation_803a13e3/) holds every argv, the one-line summary of all 24
([`summary.txt`](batch1_validation_803a13e3/summary.txt)) and the full `checkargs` and `--dry-run` transcripts for
seed 1 of each of the 8 arms. All ran on CPU (`CUDA_VISIBLE_DEVICES=""`, `nice -n 19 taskset -c 5-7,13-15`), with
`GEN3AI_MODELS_DIR` on a scratch dir, which stayed empty; nothing was written under `models/`.

| arms | `checkargs` | `--dry-run` |
|---|---|---|
| A1, C0, D1, A2 × s3501–s3503 (12) | exit 0, the current tree's parser (`--pin` = HEAD) | "would launch", FORK of `rb_x26_s2601`, 280,000,170 → 300,000,170 (+20,000,000); 12 recipe knobs RESTORED from X26's record and typed (`n_envs=256 … n_epochs=10 … ent_coef=0.05 self_play=True`); K9(b) `warn`; bf16 trunk; torch `2.8.0+cu126` carried; ✓ fork arch compatible (the trainer's own check); 10.97 GiB REQUIRED |
| toy A1, C0, A2 × s3601–s3603 (9) | exit 0 | "would launch", FORK of `rb_toy_ent_s4002_deep`, 200,000,244 → 270,000,244 (+70,000,000, STAND-IN); K9(b) `fatal`; 13.05 GiB REQUIRED |
| X47 × s3601–s3603 (3) | exit 0, `declared growth (--grow) : trunk_layers 2 -> 4` | as the toy rows, plus **`grow : DECLARED trunk_layers 2 -> 4 (parent from its model_config.json) — the parent loads strictly, identity-init parameters are appended, its AdamW state carries, new parameters start fresh`**, the seeded pool grown the same way, and ✓ fork arch compatible |

The toy stand-in R used to build the toy argvs is
[`batch1_validation_803a13e3/reference_population_toy_STANDIN_200m.json`](batch1_validation_803a13e3/reference_population_toy_STANDIN_200m.json),
made by `refpop.freeze` with no X candidate into a scratch path. Its file name keeps `launch_x35.sh` from ever reading it.

---

## 5. The open choices (the orchestrator's; a recommendation for each)

- **C1. The battery does not read ADOPT.** The registered rule says production then uses FP32_ALT, which does not exist
  (§3).
  - (a) Launch production at main HEAD anyway (T62), recording the measured cost. T62 is a correctness fix (one τ for the
    compiled rollout and the eager learner), and its cost is paid by b-ON and C0 alike, so it biases no A/B; it only
    raises the GPU-h.
  - (b) HOLD production until an FP32_ALT build lands and is validated (days), and use the GPU meanwhile for the toy
    block (τ-free, §3).
  - (c) INCONCLUSIVE only: a collect-phase A/B first (the README's suggestion).
  - **Recommendation: (a) for FP32_ALT and INCONCLUSIVE alike.** It changes the registered rule, so it needs the
    orchestrator's amendment, recorded before the first launch.
- **C2. Where X47 sits in the toy block.** (a) After toy A2, as the bot screen ranks it (calibration pair first);
  this leaves X47 un-interleaved with its C0, against the screen's "ABBA with the toy C0". (b) Interleaved with toy A1
  and toy C0, A2 last (§1's order). **Recommendation: (b)**: X47's read is against C0, so box drift should not land on
  one side; A2 is the plausible null and the cheapest to defer; the calibration pair still finishes at the same pin.
- **C3. The toy arms' K9(b) mode.** The kit types `fatal` "as X34 does" (§5.6.1, written 2026-10-17), but X34 itself
  has run `warn` with Amendment-3-style ceilings since its §7 amendment (2026-10-19). (a) Keep `fatal`: the stricter
  check, but a stop leaves an arm short. (b) `warn` with X34's §7 ceilings, the regime T\* was trained under, which
  means a `kit.py` change (the toy `Arm`s' `behaviour`) and its test. **Recommendation: (b)**, decided before the toy
  freeze. Whether X34 logged any K9(b) warning was not read here.
- **C4. R0** (the best responder of P\*, queue #0 of STAGED_PLAN §4) has no argv (F27). **Recommendation: launch batch 1
  without it**: A1 / C0 / D1 / A2 do not read it. It is A6's trigger and the R2 baseline, so stage its argv before the
  A3 / A6 batch.
- **C5. The production R's X-block** (refpop README F-R1): it has one other-lineage member (`production`), because T\*
  did not exist. T\* will exist BEFORE the first arm launches. (a) Keep R as frozen. (b) Re-freeze with T\* as a
  second X member BEFORE the first production arm: the old file moves aside (R is write-once; `launch_x35.sh` reads the
  canonical name), and one more 500-pair band cell (P\* vs T\*) is played on CPU. Whether `refpop freeze` can read the
  toy freeze's T\*-vs-P\* cell in the other orientation is **UNVERIFIED**; if not, play it twice. **Recommendation:
  (b) if the band cell can be done before the launch window (≈ 30 min of CPU after 18:15 fits); otherwise (a)**,
  because R cannot change once an arm exists (pre-check 7), and F-R2's own-lineage lean is the read's main weakness.
- **C6. X44 between the blocks.** The bot screen ranks X44 (≈ 10.4 GPU-h) above the toy calibration and X47. It is its
  own unit. Running it between the production and toy blocks delays the toy block, but changes no batch-1 pin (the toy
  block has its own C0). Not decided here.

---

## 6. Failure handling

- **A `launch_x35.sh` HOLD (exit 7).** Read `~/.claude/jobs/x35/status.txt` and the named transcript. Fix the cause and
  re-run the dry run. Never edit a pre-check to pass, and never launch by hand around a kit HOLD.
- **The launcher dies within 180 s** (a HOLD): read `~/.claude/jobs/x35/<run>.launcher.log`. A `FATAL_CONFIG` at start
  (for X47: the growth gate's bit-identity refusal) is NOT retried in another form: report it.
- **The post-spawn pool-seed check fails** (the script SIGTERMs the launcher by its PID and HOLDs): verify the PID is gone
  and the GPU is free (SOP §4.2), move the run dir aside as `models/<run>.POOLSEED_FAIL` (never delete it), bank it, and
  relaunch under the same name only after the cause is fixed.
- **A crash mid-run:** the launcher restarts it (≤ 3) from the latest checkpoint at the same pin. A 4th crash, or any
  `FATAL_*`, ends the arm: report to the orchestrator before anything else runs. A relaunch under the same name and pin
  is allowed for an operational fix; a scientific change is a new arm with a new registration (SOP §4.4).
- **A K9(b) ceiling crossed (production, warn):** stop the arm at its next safe point (SIGTERM by PID through the
  identity gate `kill_if.sh`), keep the GPU idle, and page the orchestrator. Under T62 the τ class should not recur;
  X26's Amendment 6 (the τ class STOP at 1e-2) is NOT carried into X35.
- **A `--sync-to-main` line, or a `Pinned to` sha that is not `<PIN>`:** the arm is VOID; stop it.
- **A HOLD between arms:** the GPU stays idle until the orchestrator decides (TRAINING_RUN_SOP §3, rule 3); the next arm
  never starts over a failed one.

---

## 7. FINDINGS (standing rule 7)

- **F-LS-1 (gap): X47 is not a kit arm.** `main.x35.kit` has no X47 entry, so `launch_x35.sh` cannot dry-run or launch it,
  `kit disk` does not count it, and its pre-checks and four SOP layers are by hand (§4.2). The fix is small: an
  `Arm("toy_x47", "toy", L_TOY, TOY_SEEDS, "rb_x35t_x47d4_s{seed}", …)` whose lever is `--grow trunk_layers=4`, a place
  in `BATCH1` and a `kit_test` case. It was out of this unit's scope; the toy block is days away, so it can land before
  the toy pin.
- **F-LS-2 (blocker for the FP32_ALT variant): no FP32_ALT commit exists** (§3). The registered T62 rule cannot be
  executed as written if the battery does not read ADOPT. Choice C1.
- **F-LS-3: the toy K9(b) mode in the kit predates X34's §7 amendment** (choice C3).
- **F-LS-4: every toy line is a 200M STAND-IN.** Its `--steps` and A1's archive change at the real T\*: rebuild them
  from the toy R file with `validate_batch1.sh`, never by editing the stand-in files.
- **F-LS-5: the disk margin for the toy block is ≈ 20 GiB** (§2 check 9), before X34's own growth past 200M and any
  read output. Thin finished arms as they end.
- **F-LS-6: the dry-runs ran at `803a13e3`, but the launch pin is main HEAD at launch time.** Any commit between them is
  unvalidated: `launch_x35.sh` re-runs `checkargs` and the dry-run at `<PIN>` before every arm, and X47's lines must be
  re-run with `validate_batch1.sh --pin <PIN>`.
- **F-LS-7: D1's GPU-h on forum teams is not measured**, and neither is X47's ρ (ESTIMATED 1.3–1.4). Both are read on the
  first 1M.
- **F-LS-8: "T62 does not touch the toy" was verified on the CPU eager forward only** (§3); the compiled and learner
  paths were not run.
