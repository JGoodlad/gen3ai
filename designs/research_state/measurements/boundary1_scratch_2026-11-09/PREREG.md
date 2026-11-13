# BOUNDARY-1 — the from-scratch 2 × 2, width × team breadth: PRE-REGISTRATION (2026-11-09)

**Status: REGISTERED (tier FULL) before any launch, any game and any read; AMENDED 2026-11-10 (Amendment 1, §18:
three runs a cell, twelve runs) and 2026-11-12 (Amendment 2, §18: every run types `--ent-coef 0.0125
--behaviour-check warn`, the look judges K9(b), the memory fit at width 256) and again 2026-11-12 (Amendment 3,
§18: a startup fit refusal at any start of a run is a start failure, retried from the same checkpoint), all also
before any launch, game or read; and once more 2026-11-12 (Amendment 4, §18: the early check and every look are
read from the kit at `1d592e6cd`, while every run still trains at the launch pin), after [1]'s first launch attempt
and before any look or game. LAUNCHED 2026-11-12 17:18 PST** ([1] `rb_b1c_s5101`; that first attempt was stopped
by a kit bug at its early check and is discarded whole, Amendment 4). **The LAUNCH pin is
`15136b61634b667bb9da5ccc6647ff9265238829` (§18's re-pin record, 2026-11-12); the registered pin launches
nothing. The KIT commit is `1d592e6cd5ff2b6b9f634e67573a98e3cdaede44` (Amendment 4): it trains nothing and plays nothing.**
**§13's GPU items 1 to 7 are green at the launch pin (2026-11-12; §18, slots S1 to S7): ρ = 1.392, the EQ step is
71,000,000, §12 is rewritten from ρ.** The launch is the orchestrator's GO through the Training session, after
CE-3d's three forks finish; §13 items 8 and 9 are checked at the launch (slot S8).
Nothing below changes after the first run launches except by a dated amendment (§18), written before the look
it touches is played, that says why. **Once the first look has been played, no amendment touches §5, §6, §8 or §9**
except by the owner, and it is then recorded as written after data.

**The question.** Trained from scratch to 100M steps on the production recipe, does (a) a trunk twice as wide, and
(b) training half of each side's team draws on 12,000 procedurally built teams, make a stronger agent — each read
with the other present and absent, and do the two help each other?

Abbreviations. pp = percentage points of win rate (1 pp ≈ 7 Elo near 50 %). SD = standard deviation; df = degrees of
freedom; lo / hi = the ends of a two-sided 90 % interval. Seed SD = the SD, across the seeds of one arm, of a seed's
mean score, with game noise removed. Panels (72 teams each, `../../heldout_panels.json`): **home72** = the
training pool's own teams; **H** = `heldout72_hpfix`; **F** = `forum72`; **P72** = `p72`; **U** = the equal-weight
mean of H, F and P72 (the unfamiliar teams). Kakuna = Metamon's strongest public agent, our external anchor. P\* =
baseline `pstar_plateau`. A mirrored pair = two games on one deal with the sides swapped. GPU-h = hours of the one
training card. The design that led here: [`DRAFT_DESIGN.md`](DRAFT_DESIGN.md) (explicit-only; where it and this
file disagree, this file is the registration).

## 1. What was adopted, and what changed since the draft

- **The orchestrator adopted the draft's 2 × 2 variant** (design §7) on 2026-11-09: eight fresh runs, two a cell.
  **Amendment 1 (2026-11-10, §18) makes it twelve, three a cell**: at two runs the registered answer to the
  likeliest unclear outcome (UNDECIDED) was a third seed anyway, read as a second look.
- **Breadth is decided on the in-house cross, not on Kakuna** (orchestrator ruling, 2026-11-09, "Option A"). The
  draft read breadth on Kakuna on U. At two seeds a cell Kakuna finds, 80 % of the time, only a gain of ≈ 4–6 pp
  on U and ≈ 5–7 pp on forum72, and the fork evidence for breadth there is +0.55 pp on F, so that rule could not
  have fired. Kakuna stays as the CONFIRMING meter for both factors. At registration twelve runs were declined as
  +104 GPU-h for a Kakuna rule that would still be unpowered. Amendment 1 buys the third seed for the in-house
  meter, not for Kakuna, whose 80 % resolution it improves to ≈ 3–4 pp on U and ≈ 4–5 pp on forum72 (§11.4): still
  far above the fork evidence, so the ruling stands.
- **The base is today's production**, which since the draft gained: per-run trunk width (`0220f8408`, config v169;
  production stays 128 and byte-identical), E13, their max HP as a belief (`3b1733f50`, v168), and the mechanics
  round 2 (`566d878b2`).
- **No critic or policy-tower change.** The architecture diagnosis (`a8265f36b`) found the tanh tower 0.897
  saturated but not dead, and the critic head's hidden layer unused; it recommends no change for this base. The
  draft's "critic is linear on 128" was wrong: the head has a hidden layer.
- **`ent_coef` is 0.0125 in every run (Amendment 2, §18)**: the owner's yes of 2026-11-11, after X67's PASS, for
  this experiment only. Every fresh argv types `--ent-coef 0.0125`; `designs/production_config.json` stays at 0.05
  and the owner's 2026-09-29 ruling stands for production. No arm trains at 0.05, so the experiment says nothing
  about the coefficient (§16).
- **Not in this experiment:** the MF-3 off-pool belief work, human teams (X60-T read NO-TRANSFER on 2026-11-11, so
  §10's HOLD did not fire), E9, any search.
- **The team is the unit against Kakuna** (`96adb152c`, `UNDERSTANDING.md` §8 rule 10): §6's interval is clustered
  by team.

## 2. The arms

Every run is a FRESH `--arch production` launch: the architecture and the recipe both come from
`designs/production_config.json` (ten PPO epochs), plus **the base every run types (Amendment 2, §18): `--ent-coef
0.0125 --behaviour-check warn`** (production's defaults are 0.05 and `fatal`; neither default moves). An arm types
the base and its lever and nothing else; the control types the base alone.

| arm | kit key | what differs from the control | lever typed | learner parameters | runs (seed) |
|---|---|---|---|---|---|
| **C** control | `c` | nothing | — (the base alone) | 2,609,183 | `rb_b1c_s5101`, `rb_b1c_s5102`, `rb_b1c_s5110` |
| **W** wide | `w` | the trunk at 256 with its feed-forward at 512 (production: 128 / 256) | `--d-model 256 --trunk-ffn-dim 512 --allow-nonproduction-arch` | 6,258,079 | `rb_b1w_s5103`, `rb_b1w_s5104`, `rb_b1w_s5111` |
| **Br** breadth | `br` | on EACH side, every fresh team draw is an `n1_proc_v1` team with probability 0.5, else the usual pool draw | `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both` | 2,609,183 | `rb_b1br_s5105`, `rb_b1br_s5106`, `rb_b1br_s5112` |
| **WB** wide + breadth | `wb` | both | both levers | 6,258,079 | `rb_b1wb_s5107`, `rb_b1wb_s5108`, `rb_b1wb_s5113` |

- **The two factors.** Width ON = {W, WB}, OFF = {C, Br}. Breadth ON = {Br, WB}, OFF = {C, W}.
- **The shape of every fresh argv:** `--restart-interval-hours 6 --pin-commit <PIN> --arch production --ent-coef
  0.0125 --behaviour-check warn [lever] --checkpoint-every-steps 1000000 --steps <target> --seed <seed> --device
  cuda --log-level periodic --run-name <run>`. The 1M checkpoint series is typed because the looks and the
  equal-compute read need it. A phase-2 resume types neither base knob: it inherits both from the run's record.
- **`--allow-nonproduction-arch`** is the consent the launcher demands for a width other than production's; it is
  part of the wide lever, not a second lever.
- **The corpus** is `n1_proc_v1`, tier `ALL`, 12,000 teams, data sha256 `99e3425a…0bad6`, as it verifies at the pin.
  Evaluation never draws from it: every look and the Kakuna read play the registered panels.
- **The argvs are code**, `src/main/b1/arms.py` (`python -m main.b1.arms describe`); the committed transcripts are
  what it prints at the LAUNCH pin (§3), written by §18's re-pin record: `launch_block.txt` (twelve fresh launches,
  then twelve resumes) and `argv_<run>.txt`, `checkargs_<run>.txt`, `dryrun_<run>.txt` for the first seed of each
  arm. The eight-run block of the registered pin is not kept past that record (history: `507d0c119`). `python -m
  main.checkargs --argv` accepts each with 0 unrecognised flags; the control resolves the base as typed
  (`ent_coef` 0.0125, `behaviour_check` `warn`) and each resolved config differs from the control's in its lever's
  keys alone (`d_model`, `trunk_ffn_dim`, `allow_nonproduction_arch`; `team_mix`, `team_mix_sides`), at both step
  targets. `python -m main.launcher --dry-run` on CPU, from an agent's nice-19 shell, refuses each for TWO reasons:
  the CUDA preflight (`CUDA_VISIBLE_DEVICES=""` with `--device cuda`) and the inherited-nice gate
  (`src/main/launcher/nice_gate.py`: a GPU launch from a shell above nice 10); every other preflight passes. The
  dry run with the card visible, from a shell at nice ≤ 10, is §13 item 2.
- **Parameter counts** are a CPU build of the four learners at the pin (`python -m main.b1.arms check --params`).

## 3. The pin, the seeds, and the checks before launch

- **ONE pin for everything: the LAUNCH pin, `15136b61634b667bb9da5ccc6647ff9265238829`** (set once, by §18's re-pin
  record of 2026-11-12, which lists what main gained since the registered pin). All runs train at it, every look plays
  at it, and the Kakuna read plays at it. **The look's KIT alone is read from a later commit, `1d592e6cd`** (Amendment 4,
  §18: the early check, the arm-end check, and each look's plan, read and team bootstrap), which differs from the
  pin under `src/` and `data/` in `src/main/b1/look.py` and its test and in nothing else.
- **Registered: `507d0c11961da8d4243a4c8877bcc04ef220058c`** (main at registration, the commit that carried the
  kit, the cross statistic and the Kakuna rule). It is `0220f8408` plus the nine commits main gained while this was written
  (`c4a842d12` to `c8c212214`: the light-screen tier's docs, the heavy slots admitted by memory, the core split and
  its owner, the off-pool belief meter with X68's records, three gate-reliability fixes; none touches a trainer,
  launcher, model or head-to-head file) and this unit's tooling; `designs/production_config.json` and
  `src/agents/model/model_version/` are unchanged between `0220f8408` and the pin, and `checkargs` at the pin
  resolves the same production surface (`production_config@51496e4209fc`).
- **The registered pin launches nothing (Amendment 1).** Its kit holds two seeds an arm and no base knob; the
  three-seed kit, the early check, the team bootstrap, Amendment 2's base and K9(b) reads, and the memory-fit unit
  reached main after it. What this file says of "the pin" holds for the LAUNCH pin.
- **The pin is an ARGUMENT of every command (`--pin`), never a constant in the code.** A commit cannot name
  itself, so the kit, the look and its driver carry no default pin and refuse without one. This file and the
  committed transcripts are the only places the pin is written.
- **Seeds 5101–5108 and 5110–5113, one per run; 5109 is the spare.** C 5101 / 5102 / 5110, W 5103 / 5104 / 5111, Br
  5105 / 5106 / 5112, WB 5107 / 5108 / 5113 (the first two of each arm are the registration's, the third is
  Amendment 1's; no registered seed or name moved). No two runs share a seed, so no cell of the cross is a matched
  pair and the statistic treats every run as independent.
- **Disjointness, checked 2026-11-09:** `python -m main.b1.arms check --tree <a checkout of the pin>` scanned 640
  `metadata.json` files and 3,888 directory names under the main checkout's `models/` and `~/gen3ai_archive`: no
  recorded seed, no run name and no `_s51NN` directory name collides with 5101–5109 or the twelve run names. **Again
  2026-11-10, with the amended kit, for Amendment 1's seeds:** 646 `metadata.json` files and 3,922 directory names,
  no collision with 5101–5113 or the sixteen run names (the twelve runs and the spare under each arm). **Again at
  the launch pin, 2026-11-12:** 664 `metadata.json` files and 4,017 directory names, no collision (§18's re-pin
  record). The same
  command verifies the corpus identity and that each arm differs from the control by its lever alone. **It is re-run
  immediately before the first launch** (§13 item 1): a hit refuses the launch. Run again later, it lists the
  experiment's own folders as `own` (a folder whose immutable `original_command` and seed are the registered run's
  at the pin); a foreign run under one of these names or seeds is still a hit.

## 4. The schedule

- **One run at a time on the one card, in this order within each phase:** C 5101, W 5103, Br 5105, WB 5107, C 5102,
  W 5104, Br 5106, WB 5108, C 5110, W 5111, Br 5112, WB 5113 (one seed of every arm before any arm's next, so an
  early stop leaves a balanced set).
- **Phase 1:** each run fresh to 50,000,000 steps. **The 50M look** (§8) is read. **Phase 2:** each surviving run is
  RESUMED in its own folder to 100,000,000 (`--model models/<run>/final_model.zip --steps 100000000`, no
  `--run-name`, no `--arch`: a same-run resume, never a fork).
- **Why stopping at 50M is the same run.** The claim is NOT bit equality with a run that never
  stopped, and none is needed: a stop at the 50M target is ONE MORE RESTART of the kind every run takes every six
  hours, at a chosen step, applied to all twelve runs alike. Measured by the kit's two-segment smoke on CPU
  (`python -m main.b1.resume_check smoke`, arm WB, a phased run against a run-through of the same seed, 2026-11-09;
  repeated at the launch pin with the base typed, §13 item 6, in §18's re-pin record). Its coverage is that and no more: one arm (WB), on the CPU, a `--debug
  --no-pin` run with a typed rollout shape; the card and the other three arms are covered by item 6's static
  check alone:
  - the phase-2 command resolves to a same-run RESTART of the run's own folder, and its child's namespace equals the
    launcher's own six-hour-restart child's but for three keys that are inert on a resume (`batch_size`,
    `allow_nonproduction_arch`, `run_name`);
  - the optimizer and weights phase 2 LOADS are tensor-equal to what the stop SAVED; the learning rate resumes at the
    saved value; the entropy coefficient, clip range, matchup specification and team mix are the same before and
    after; the pin record stays one entry from step 0; lineage and `original_command` do not change; every
    checkpoint written before the stop is byte-identical afterwards;
  - what DOES change, as at every restart: the environment's seeds are a function of `--seed` and the step the
    process started at, so the games after the stop differ from the ones a run-through would have played; and a
    stop at a target trains on its last rollout, where a six-hour restart saves without training on it.
  - the checkpoint the 50M look reads (§5.1) is written BEFORE the run stops, during phase 1's last rollout: it holds
    the policy one update before `final_model.zip`, in the phased block and the fallback alike.
- **The fixed fallback.** If §13 item 6 (the stop-and-resume check at the pin) is not green at launch, the twelve
  runs launch straight to 100M with `python -m main.b1.arms describe --block fallback --pin <PIN>`. **Under the
  fallback the 50M look is REPORTED and stops nothing**: it needs all twelve runs at 50M, which first holds midway
  through the twelfth run, when a stop could save at most half a run; and it can then be played only on CPU (≈ 27
  h), because the card is training. Which block launched is recorded in §18. The looks' checkpoint rule (§5.1) is
  the same under both blocks.
- **Restarts.** The launcher restarts each child every 6 h, as for every production run.
- **No run of the twelve is thinned before §9's reading is closed.** The looks read the 1M checkpoints (the
  equal-compute look at a step fixed only by the pilot, §13), and phase 2 resumes with the self-play pool the stop
  left; the standard thin plan would cut both. Disk: ≈ 100 checkpoints a run, ≈ 55 MiB each at width 128
  (`rb_x60t_s3501`'s) and 73 MiB at width 256 (the smoke's WB zip), so ≈ 75 GiB for the twelve runs; the launcher's
  dry run sizes the wide arms from a 128-wide checkpoint and understates them.

## 5. Meter 1, which decides: the head-to-head cross

### 5.1 The cells

- **54 cells a look:** every run against every run of another arm, once (4 arms, 6 arm pairs, 3 × 3 runs each).
  The arm later in (C, W, Br, WB) is the player; the other side's score is 100 − score.
- **Each cell:** 1,000 mirrored pairs on home72 and 500 on each of H, F and P72; both sides greedy (the deployed
  policy); one engine, the pin's; `--seed 0`; oracle reveal off. 270,000 games a look.
- **Which file a run plays at a look:** the first `checkpoints/checkpoint_<N>_steps.zip` with N at or after the
  look's step and below the look's step + 1,000,000. Never `final_model.zip`, so the phased and the fallback block
  read the same bytes.
- **Views.** A cell's score on a view is a fixed weighted mean of its panel rates: **N** = ½ home72 + ⅙ H + ⅙ F +
  ⅙ P72 (the owner's exchange rate r = 1 between home and unfamiliar teams, `../../../STATUS.md` §6.4); **F** =
  forum72 alone; **HF** = ½ home72 + ½ forum72; home72, U, H, P72 alone are reported.

### 5.2 A factor's statistic

For a factor and a view, rows = the factor's six ON runs, columns = its six OFF runs, 36 cells.

- **Δ̂** = the mean of the 36 cell scores − 50.
- **Variance.** A run's row mean is its mean score against all six OFF runs. Within one ON arm the other factor is
  constant, so the sample variance of that arm's three row means estimates the seed variance free of the other
  factor's effect. s²_rows = the mean over the two ON arms of that within-arm variance, **each floored at 1.76²**
  (the measured seed SD, §11.1: a lucky small variance from three seeds must not pass). s²_cols likewise over the
  two OFF arms. **V̂ = s²_rows / 6 + s²_cols / 6**, on **8 df** (two per arm), Student's t = 1.860. (In general, at
  n runs an arm: V̂ = s²_rows / 2n + s²_cols / 2n on 4(n − 1) df; the registration's n = 2 gave / 4, 4 df, 2.132.)
- **The interval** is Δ̂ ± 1.860 · √V̂, two-sided 90 %. Its half-width is never below 1.89 pp (all four variances at
  the floor; 2.653 at two runs); its simulated median is 2.14 pp on N and 2.19 pp on F (§11.2).
- A bound within 0.005 pp of the threshold it is compared with FAILS that comparison and is named `at_boundary`
  (root rule 8: the conservative class, deterministically).
- **The assumption.** The 8-df t is exact when the four arms share one seed variance. If one arm's seed SD is
  several times the others', each 5 % tail grows (to ≈ 5–7 % in §11.2's cases); nothing here tests for it, and the
  four within-arm variances are printed with every statistic.
- Code: `main.b1.cross.factor_stat`, which takes any equal number of runs an arm. `power_matrix_n3.py` classes
  every simulated draw twice, with the rule code and with its own transcription of the formula above (0 mismatches
  in 448,800 at three runs; the registration's `power_matrix.py`, 0 in 624,800). That checks the transcription; it
  is not an independent validation of the formula.

### 5.3 The 100M decision, per factor

Classes, tested in this order on the factor's DECIDING view:

| class | condition | reads as |
|---|---|---|
| **WIN** | lo > 0 | the factor makes a stronger agent at 100M |
| **LOSS** | hi < 0 | it makes a weaker one |
| **NO-USEFUL-GAIN** | hi < +4.0 | any gain is below the +4 pp that would pay for the arm |
| **UNDECIDED** | otherwise | the interval admits both a useful gain and none |

- **Width is decided on N.** Its ON and OFF sides each hold one breadth-trained arm, so the overlap between
  `n1_proc_v1` and the H and P72 panels (hazard M4) cancels in width's contrast.
- **Breadth is decided on F alone.** 62 % of H's and 56.5 % of P72's full team sets are in `n1_proc_v1`, so a
  breadth-trained run meets its own training teams there; forum72 is the one unfamiliar panel it has not seen.
  - **GAIN** = breadth's class on F is WIN.
  - **The net rule** (the owner's r = 1): a breadth WIN stands only if **Δ̂_home72 + Δ̂_F > 0** (the same statistic's
    point estimates on the two panels; a sum within 0.005 pp of zero fails). A WIN that fails it is
    **WIN-NET-NEGATIVE** and is not adopted.
  - **Written before any outcome: the in-house home72 meter alone never vetoes a generalisation lever.** Breadth's
    home72 statistic is reported as its cost; no class is read from its interval; a negative home72 estimate, however
    clear, changes nothing unless it makes the sum above negative. The net rule is the ONLY home-team cost this
    registration recognises.
- LOSS, NO-USEFUL-GAIN and UNDECIDED for breadth are read on F exactly as for width on N.

### 5.4 The interaction (REPORTED; it decides nothing)

- With m(A, B) the mean of the nine cells of arm A against arm B: **Î = ½ [(m(WB, Br) − m(W, C)) + (m(WB, W) −
  m(Br, C))]**: how much more width is worth with breadth on than off, estimated from both ways of pairing the four
  arms. Each run enters through ONE per-run quantity u: its mean score against the two arms that differ from its
  own in exactly one factor (for a WB or a C run: against W and against Br; for a W or a Br run: against C and
  against WB), which a run's own strength moves one for one. V̂ = the sum over the four arms of the within-arm
  variance of u (not of §5.2's row means), each floored at 1.76², divided by 3 (the runs an arm); on 8 df.
- Classes: SYNERGY (lo > 0), ANTAGONISM (hi < 0), NOT-SHOWN, **read on N**; the estimate and interval are also
  printed on F, without a class.
- **It is resolved only when large:** its half-width is never below 3.78 pp and its median is 4.26 pp (§11.3;
  5.31 and 6.09 at two runs). It is a point estimate with an honest interval; a NOT-SHOWN is not evidence of
  additivity.
- The six simple effects (W, Br, WB each against C; WB against Br and against W; Br against W) are reported with
  4-df intervals (half-width at least 3.06 pp).

### 5.5 The team-cluster bootstrap (REPORTED; it decides nothing; Amendment 1)

§5.2's interval treats the RUNS as the sample and a panel's 72 teams as fixed: it answers "would other seeds
agree?". Whether another 72 teams would agree is reported beside every full look, and changes no class.

- **Tool:** `python -m main.b1.team_bootstrap --work <WORK_DIR>` → `<WORK_DIR>/result/team_bootstrap.json`. The
  look's driver runs it after the read; its exit never changes the driver's.
- **What is resampled.** The ledger banks no per-game row: each batch of a cell banks a per-TEAM tally (each side's
  games and wins by the team it piloted). A cell's score is exactly 50 + 50 · Σ d_t / Σ g_t over the panel's teams,
  with g_t the games the player piloted team t and d_t the player's wins piloting t minus the opponent's wins
  piloting t. The tool refuses a cell whose tallies do not reproduce the ledger's score and a factor whose tallies
  do not reproduce the registered estimate. One draw resamples each panel's teams with replacement: ONE draw shared
  by every cell of the panel, the panels drawn independently, the runs held fixed. 10,000 draws, seed 20261110.
- **What it prints**, per factor on the view the look reads it on: the team SE and percentile interval; the
  registered interval and class; a COMBINED interval, Δ̂ ± t · √(V̂ + V_team) at the registered df, with the class
  it WOULD give and whether that differs (`class_moves`); for breadth, the share of draws whose net stays above
  zero; and the team SE on every view.
- **Its limits.** The combined variance counts the game noise twice, so it is an upper bound on "seeds and teams
  both resampled", never a second verdict. Clustering is one-way, by the team the WINNER piloted, not by the pair of
  teams a game was played on. A team that sits on two panels is drawn on each independently. A survivor look is
  refused (not built).
- **A `class_moves` goes to the owner beside the outcome and acts on nothing**: §5.3's class stands.

## 6. Meter 2, which confirms: Kakuna, clustered by team

- **Games.** Each run's 100M final plays Kakuna on H, F and P72, 720 games a panel (unit seeds 0–35), 25,920
  games. No Kakuna games on home72. Spec: [`read_spec_kakuna.json`](read_spec_kakuna.json) (read
  `boundary1_kakuna_2026-11-09`, rule `b1_kakuna`, `src/main/b1/kakuna_rule.py`). The committed spec names the
  twelve runs; its `pin` is filled by §18's re-pin record (without one it is reviewed with `check-spec
  --pin-pending` and plans nothing). The rule takes any
  number of runs an arm (the seed variance pooled on Σ(n − 1) df) and is not changed.
- **Contrasts.** CONFIRMING: **`width_U`** = ½(W + WB) − ½(C + Br) on U, and **`breadth_F`** = ½(Br + WB) − ½(C +
  W) on F. REPORTED: `breadth_U`, `width_F`, and `interaction_U` = WB − W − Br + C on U (the full difference of
  differences, the same scale as §5.4's Î).
- **The interval** is analytic and two-way, computed once for each of the two team clusterings of record
  (`96adb152c`'s addendum): by OUR team, and dyadic (a team is one cluster in either role). With V_S the pooled
  within-arm seed variance of the contrast, V_T the team-cluster sandwich variance across the twelve runs and V_TS
  the same sandwich with every cluster split by run: **V = max(V_S, V_TS) + max(0, V_T − V_TS)**. Satterthwaite
  df; two-sided 90 %; **the WIDER of the two schemes' intervals decides.** A tied game counts as a non-win. The
  lane deals teams with replacement, so a panel is not balanced by team (the record reports each panel's design).
  Classes: POSITIVE (lo > 0), NEGATIVE (hi < 0), SPANS-ZERO, with the same 0.005 pp boundary rule; INCONCLUSIVE if
  any run has fewer than 95 % of its games on a panel or more than a quarter of a panel's games ended by timeout.
- **REPORTED beside it:** two team × seed bootstraps on one shared index set (10,000 draws, seed 20261109). The
  draft named the bootstrap as the statistic; a seed bootstrap has three distinct resamples an arm at two seeds
  and ten at three, and it still understates a large seed spread (§11.4), so the analytic interval is the
  registered one (deviation, §17).
- **What it does to a 100M outcome** (`main.b1.cross.confirm`). It never changes the class.

| in-house outcome | Kakuna contrast | label | goes to the owner |
|---|---|---|---|
| WIN | POSITIVE | CONFIRMED | no |
| WIN | SPANS-ZERO | UNCONFIRMED | no |
| WIN | NEGATIVE | IN-HOUSE-ONLY | **yes** |
| anything else | POSITIVE | KAKUNA-ONLY | **yes** |
| anything else | NEGATIVE or SPANS-ZERO | NO-CONTRADICTION | no |
| any | INCONCLUSIVE (games unplayed, short or timed out) | **UNREAD** | **yes**, if it cannot be finished |

- **UNREAD is not a label to act on.** An unfinished Kakuna read is finished (the lanes resume until no unit is
  unplayed) or replayed; §9 acts on nothing meanwhile. Only a read that cannot be finished goes to the owner.
- **The files differ by design:** the Kakuna read plays each run's `final_model.zip` (at or after 100M, settled),
  the cross its first checkpoint at or after 100M. Both are the run within one rollout of 100M.
- **The finals are copied before anything continues.** `python -m main.reads plan` copies the twelve finals into
  the read's state and every unit re-hashes its copy. It is run when phase 2 ends; no run is resumed past 100M
  (§9) before that copy exists, so the read can never play a later final as "the 100M final".
- **After a 50M STOP-HARM** fewer than twelve runs reach 100M and this spec cannot plan. No Kakuna read is
  registered for that case (§8).
- Its resolution is §11.4: at three runs a cell it finds, four times in five, a width gain on U of ≈ 3–4 pp and a
  breadth gain on forum72 of ≈ 4–5 pp. It can contradict a large in-house effect; it cannot find a +2 pp one
  alone, and it cannot confirm a breadth gain much below the +4 pp bar.

## 7. Reported, never a verdict

- **Curve dominance** on each run's own snapshot ladder (`<run>/snapshot_ladder/ladder.json`, ± 10 Elo), compared
  across runs only at matched snapshot COUNT (`python -m main.elo compare`), the newest node excluded. A factor
  DOMINATES iff its ON arms' mean is above its OFF arms' at every matched count from 30M to 100M and at least five
  of its six ON runs end above the OFF runs' median (three of four at registration). Ladders of different runs meet
  different opponent pools: this is pace evidence.
- **The 75M look**: the same cross at 75M, played with the 100M look. No class.
- **The equal-compute look (EQ):** the wide arms at an earlier step against the narrow arms at 100M, the same 54
  cells. The step is fixed BEFORE launch from §13 item 4's measured step-time ratio ρ: **eq step = 1,000,000 ×
  ⌊100 / ρ⌋** (at most 99M), written into §18 when the pilot lands. **Measured 2026-11-12: ρ = 1.392, so the EQ
  step is 71,000,000** (§18, slots S1 and S3; the look takes `--eq-step 71000000`). No class: the entity toy led
  at equal compute and lost at its plateau (`UNDERSTANDING.md` §4.9), so this is a cost statement. The realised GPU-hours of each run
  are reported beside it.
- **Every factor on every view** (N, home72, U, H, F, P72, HF), with intervals; breadth's on H, P72, U and N carry
  the M4 overlap and are labelled so.
- **The control against X26's snapshot at the matched step**, for continuity with the previous lineage (different
  observation: E13 and the mechanics rounds landed since; a difference is not attributable).

## 8. The looks

| look | when | what is read | consequence |
|---|---|---|---|
| **50M** | after phase 1 (all twelve at 50M) | each factor's HARM view | **STOP-HARM** iff hi < −2.0; else CONTINUE (REPORTED only under §4's fallback) |
| 75M | with the 100M look | everything, REPORTED | none |
| **100M** | after phase 2 | §5.3's classes, §5.4, then §6's labels when the Kakuna read lands | §9 |
| EQ | with the 100M look | §7, REPORTED | none |

- **The 50M harm stop is the only early rule, and it only stops.** No class at 50M can adopt, retire or extend.
  At a true effect of zero it fires in 0.0–0.3 % of simulations (§11.2), so it costs the 100M look nothing there.
  It is not free for a factor that is behind at 50M and would be ahead at 100M: that factor is stopped and never
  read (§16).
- **Harm views.** Width: N. Breadth: **HF** = ½ home72 + ½ forum72, the same sum the net rule reads (Δ̂_HF = half
  the net). So breadth is stopped only when its net at 50M is below −4 pp with 95 % one-sided confidence, never by
  home72 alone. (A registrar's choice, flagged in §17.)
- **After a STOP-HARM of a factor, NO class is read at 100M, for either factor.**
  - Its two ON arms get no phase 2 (width: W and WB; breadth: Br and WB). The stopped factor is recorded as
    STOPPED-FOR-HARM at 50M: its from-scratch arms are not continued. That is not a LOSS at 100M and claims
    nothing about 100M (§16).
  - The other factor's surviving comparison (W against C, or Br against C) is played at 100M and 75M with `look
    plan --survivor <that factor>`: 9 cells a panel, three runs against three. Its simple effect is REPORTED on
    every view with a 4-df interval (half-width at least 3.06 pp; for breadth, with the net's point estimate), under
    the outcome `NO-CLASS`, and goes to the owner.
  - Why still no class: at registration two runs against two found a +4 pp effect one time in six, and a class
    there would have been decided by noise. Three against three find it about six times in ten at a seed SD of 1.76
    and four in ten at 2.5 (§11.5), so a class is now arguable; Amendment 1 moves no class rule, and registering
    one is the orchestrator's decision, by an amendment before the 50M look. No Kakuna read is registered for this
    case.
  - If both factors stop, only the control continues.
- **Where the 50M look is played.** It gates phase 2, so it is played on the card between the phases (≈ 2 GPU-h,
  §12). On the three agent cores it takes ≈ 27 h with the card idle; that is the fallback, not the plan.
- **A look is planned and read by code** (`python -m main.b1.look`, driven by [`looks/drive_look.sh`](looks/drive_look.sh);
  both from the kit commit, the games from the pin: Amendment 4).
  It refuses to plan unless every run passes §10's guards. It reads INCONCLUSIVE, never a class, if:
  - any cell on any panel is missing, duplicated, or not OK by the ledger's rule;
  - any cell holds a number of completed pairs other than the registered one (fewer OR more: no games are added
    to a look);
  - more than a quarter of any cell's games are drawn (§10);
  - any row was played at another commit than the pin, or on another team set than its panel's, or the look's
    rows name more than one device or backend (a near-tie greedy decision can differ between CPU and CUDA, so a
    look is played on ONE device).
- **Ledger.** One family a panel, `b1_scratch_h2h_<panel>`; one request a look and panel, `b1_<look>_<panel>`
  (for example `b1_50m_forum72`); the EQ look's request carries its step (`b1_eq61m_<panel>`) and a survivor
  look's its factor (`b1_100m_breadthonly_<panel>`); protocol `gen3_eval_protocol_v1_h2h`.

## 9. What each outcome does

A WIN is not an adoption. It selects which runs continue to the plateau; the plateau candidate against P\* is its
own registered read (each at its own pin, Kakuna on U, clustered by team).

**The order of reading, fixed here:**

1. **Nothing acts before the confirming meter.** No row below acts, and no run is resumed past 100M, until the
   Kakuna read's record exists and neither factor's label is UNREAD (§6).
2. **Owner rows first.** If ANY row marked *owner* applies to either factor, the whole outcome goes to the owner
   and no other row acts.
3. **Otherwise each factor's own row applies**, and the cell that continues is set by which factors read WIN:
   width alone → W; breadth alone → Br; both → WB; neither → C.

| 100M outcome | action |
|---|---|
| **WIN** (for breadth: with the net rule passed) | The factor is in the continuing cell (rule 3). The cell's three runs continue to the plateau (`python -m main.plateau tick`). |
| **LOSS** | The factor is retired from scratch as specified (this width under this recipe; this corpus at this share). |
| **NO-USEFUL-GAIN** | **Not shown at this size**: no gain of +4 pp at 100M with three runs a cell. The factor is not carried forward by this experiment. It is never recorded or quoted as retired, nor as "no effect" (§16). |
| neither factor WIN, no owner row | The control is BOUNDARY-1's lineage; its three runs are the candidates. |
| *owner:* **UNDECIDED**, either factor | With the intervals and no pre-priced option (below). |
| *owner:* **WIN-NET-NEGATIVE** (breadth) | Not adopted, not retired: reported with its cost and gain. |
| *owner:* both WIN and the interaction reads **ANTAGONISM on N** | The choice of cell. |
| *owner:* IN-HOUSE-ONLY or KAKUNA-ONLY (§6), or an UNREAD that cannot be finished | Before any action. |
| *owner:* a **STOP-HARM** at 50M (§8) | No class for either factor; the survivor's reported simple effect. |
| *owner:* INCONCLUSIVE, or any §10 hazard | No adoption. |

**No extension is registered (Amendment 1).** The registration priced ONE option after an UNDECIDED: a third seed in
every arm (≈ 104 GPU-h), read as a second stage. That seed is now in the design from the start, so the option is
spent and the three-run class is read ONCE. An UNDECIDED at three runs goes to the owner with both factors'
intervals. A fourth seed an arm would cost ≈ 104 GPU-h more; that figure is quoted for scale, not registered, and
such a step would need its own amendment (seeds, a fresh scan, the kit's seed table, and its error rates as a
two-stage rule) before any of it is played.

## 10. Validity, exclusions, replacement, holds

- **A run counts at a look iff** (`main.b1.look.check_run`, all checked from the run's own records):
  its look checkpoint exists and its sidecar's step equals the file's; the run and the checkpoint carry ONE pin
  entry, the registered pin, from step 0; its lineage role is `fresh` with no fork parent; its immutable
  `original_command` is token for token the registered argv of its arm and seed (at either phase target);
  the last process still ran ten epochs at `ent_coef` 0.0125 (the run's metadata, its recorded arguments and the
  checkpoint's sidecar all say so) with the arm's `d_model`, `trunk_ffn_dim`, `team_mix` and `team_mix_sides`; and
  its K9(b) read through the look checkpoint's step is PASS (the next bullet). Across the look: one torch version and one init thread count (every fresh build records
  a thread count of 1, so that half cannot fire: §13 item 9), and no two runs with the same
  bytes; and every run's last process records the same public `cli_args` as the first control run's, except its
  own keys (`model`, `run_dir`, `run_name`, `seed`, `steps`) and its arm's lever keys, so any other knob that moved
  in one run and not in the control refuses the plan. Ten recorded keys are rewritten by ANY resume to the parser's
  default or an "inherited" label (`arch`, `batch_size`, `n_steps`, `unified_moves`, `unified_damage`,
  `allow_nonproduction_arch` and four `*_source` labels; measured on the kit's smoke, §4): those are compared only
  between two last processes of the same kind (fresh, six-hour restart child, typed resume), and the rollout shape
  they stand for is compared where a resume cannot rewrite it, `metadata.json`'s top-level `batch_size` and
  `n_steps`, which must equal the control's in every run. REPORTED in the driver's log beside these guards, never a gate: `python -m main.sidecar_audit
  --strict` and `python -m main.checkargs models/<run>` on every run of the look.
- **K9(b) is judged by the look, not by the trainer (Amendment 2).** Every run trains under `--behaviour-check
  warn`: at a behaviour-policy violation the trainer logs, runs its bounded scan and continues, where `fatal`
  would stop a 100M fresh run on one threshold crossing (X26 first crossed 1e-4 near 137M; the X35 forks train
  under `warn`). The judged read is `main.b1.look.k9b_read`, by X35 AM-6's rule (`main.x35.batch.judge_k9b`): on
  the rows it reads, the judged maximum below 1e-3 AND the 99th percentile below 1e-4, with X35's boundary
  band; the required tag family present; no non-finite row and no non-finite FATAL line in any generation of the
  child's full log; a full child log present. Verdicts PASS / VOID / INCONCLUSIVE; only PASS lets a run count. The
  ceilings are X35's and were NOT re-derived for fresh runs or for width 256.
  - **Where it is read:** at the early check, on the rows through 1M; at every look, for each run, on the rows
    through the look checkpoint's own step (not PASS ⇒ `look plan` exits 3 and no game is played); and at each
    run's END, on every row it wrote (§14's arm-end command, all twelve runs, after phase 2 and BEFORE the Kakuna
    `plan`). Every record goes into §18.
  - **VOID** ⇒ the run is "lost to its lever" (below): not a replacement cause, seed 5109 is not spent. The look
    that refuses is not played: it is INCONCLUSIVE and goes to the owner with every run's K9(b) record, the
    simple effects that remain REPORTED and unclassed.
  - **A look already read stands.** An arm-end VOID whose fires all lie on rows after the 100M look checkpoint's
    step leaves the 100M look standing and holds the Kakuna read (UNREAD, to the owner): the Kakuna read plays
    `final_model.zip`, which those rows trained.
  - **INCONCLUSIVE** (a missing record: no rows of the required family, no full child log) is repaired from the
    run's own records and re-read; it is never waived. If it cannot be recovered it is treated as VOID.
  - An early-check exit 3 on K9(b) stops that run before the next launch, like any other early-check refusal.
- **The early check, one run at a time (Amendment 1).** As soon as a run has written its first checkpoint
  (1,000,000 steps), from a clean checkout of the KIT commit (Amendment 4; `--pin` is still the launch pin):
  `python -m main.b1.look check-run --arm <C|W|Br|WB> --seed
  <seed> --pin <PIN>`. It applies the per-run guards above to that checkpoint (`check_run` with `expected_levers`
  resolved by the pin's own parser, which the kit commit carries byte for byte) and the K9(b) read on the rows
  through that checkpoint, and prints a JSON
  record. Exit 0 is required. Exit 3 stops the run and goes to
  the orchestrator before the next launch: a wrong argv then costs an hour, not a run (the replacement rule below
  applies). Exit 2 is a seed that is not the arm's. The JSON goes into §18 with `git rev-parse HEAD` of the main
  checkout at that launch: main's launcher starts every run and main may move during the campaign, while the pin
  fixes only the child's code. The check cannot see the across-look guards (one torch version, recipe drift against
  the control); those first run when the 50M look is planned.
- **A parity-gate FATAL, in any arm (Amendment 1).** If a run's child stops on a FATAL of a compile-parity or
  opponent-parity gate (`src/agents/model/opponent_parity.py`, `compile_control.py`; X60-T's `s3501` took one on a
  healthy run, 2026-11-06): ONE same-pin resume of that run is allowed, in its own folder, from its last
  checkpoint, to the phase's target, recorded in §18 with the FATAL's text. A second FATAL of the same run holds
  the campaign. A re-pin to fix a gate is the orchestrator's decision, recorded as an amendment, and it REPLACES
  the affected run or runs from scratch at the new pin; a look refuses a set of runs on more than one pin, so that
  amendment also says what becomes of the runs already trained and names the seeds and folders. A parity FATAL is
  not one of the two replacement causes below and does not spend seed 5109.
- **A startup fit refusal, in any arm, at any start (Amendment 3).** The memory fit check runs at every start of
  a run on the card (the first launch, each six-hour restart, the phase-2 resume) BEFORE that process trains a
  step. A refusal there (`UpdateWontFit`, the child's exit 3) is a START FAILURE: not a FATAL of the bullet above,
  not a VOID, not "lost to its lever" and not a replacement cause; it spends neither that bullet's one resume nor
  seed 5109. The operator records it and starts the SAME segment again from the same checkpoint on the same pin;
  a repeat on an otherwise empty card HOLDs that run for the orchestrator. §18 Amendment 3 holds the rule, what
  the launcher does and the three retry forms. The list below keeps its meaning for an out-of-memory failure
  AFTER a segment's first training step: that one is "lost to its lever".
- **Replacement, by a closed list.** A run is replaced by seed 5109 in ITS OWN arm, from step 0, only when it
  cannot reach a look for one of these two causes:
  - its files are lost or unreadable (a disk or filesystem fault, a corrupt checkpoint it cannot resume from);
  - it was launched with an argv other than its registered one (an operator error, which the guard above
    catches).
  **Anything else is "lost to its lever"**, including a NaN or divergence, an out-of-memory failure at the arm's own
  width, a failure to load the corpus, and a K9(b) VOID (Amendment 2). The cause is written into §18 by the orchestrator before any look that
  would contain the run is read. The look then takes `look plan --replace <the registered seed that 5109 replaced>`
  (for example `--replace 5103`). One replacement exists, for twelve runs now; a second loss goes to the owner.
- **A run lost to its lever is the result**: not replaced, reported. The factor statistics need equal runs an arm,
  so that look is INCONCLUSIVE and goes to the owner with the simple effects that remain, REPORTED and unclassed.
- **A timeout is never an outcome.** `main.h2h` ends a game at the declared turn limit and records it as a DRAW,
  half a win to each side; it records no aborted games. A cell with more than 25 % of its games drawn therefore
  makes the look INCONCLUSIVE, and `verdict.json` carries each panel's drawn share and its worst cell's. The
  Kakuna read applies its own quarter per panel to timed-out games. **If the rule trips** (the pre-launch review
  found no remedy registered; its worst cell on record is 2.62 % drawn): the look stays INCONCLUSIVE and goes to
  the owner with the drawn shares. No cell is replayed at another turn limit and no drawn game is dropped or
  rescored, except by an amendment written before the replay (after the first look, the owner's: header).
- **No game is excluded for its outcome.** No panel, cell, seed or look is added after a look is read.
- **HOLDs before launch** (the registration is re-read, not launched, if either holds): X60-T reads
  ADOPT-CANDIDATE (human teams would then enter the base of every run); the owner changes `ent_coef` after X67.
  **Both are closed (Amendment 2).** X60-T read NO-TRANSFER, final (`75abf79d8`, 2026-11-11): the HOLD fired only
  on ADOPT-CANDIDATE, and the breadth arm's corpus and share are unchanged. The owner decided `ent_coef` (0.0125
  for this experiment, 2026-11-11) and this file was re-read for it: Amendment 2 is that re-reading.

## 11. Power, registered honestly

Simulation (Amendment 1, three runs an arm): [`power_matrix_n3.py`](power_matrix_n3.py) →
[`power_matrix_n3.log`](power_matrix_n3.log), 50,000 draws a row, with game noise by view (cell SD 0.68 pp on N,
1.05 on home72, 1.48 on a 500-pair panel) and the registered rule. It imports the registration's own helpers, so
the score model is the same. The registration's two-run tables are [`power_matrix.py`](power_matrix.py) →
[`power_matrix.log`](power_matrix.log), kept as the record and quoted below only where a row says so.
Every row assumes what §5.2 assumes (independent runs, one seed variance in all four arms) unless it says otherwise.

### 11.1 The seed SD is the unknown that sets everything

- Measured: **1.76 pp**, pooled over 14 arm groups of FRESH 15M production-family runs (range 0.60–3.09; 90 nominal
  df that overstate it, the groups share seeds). [`seed_spread.log`](seed_spread.log).
- **Measured at `ent_coef` 0.05** (every group above trained at it); the runs here train at 0.0125 (Amendment
  2). UNVERIFIED that the spread is the same there. The floor does not move: Amendment 2 moves no bar.
- **UNVERIFIED at 100M.** Every row is at 15M; the one deeper hint is three toy seeds, 0.67 pp at ≈ 47M and 1.82 pp
  at 100M. Hence the floor at 1.76 and the 2.5 and 3.1 columns.
- **UNMEASURED on forum72 alone**, the view that decides breadth: every measured cross is on home teams. A
  per-panel seed SD above the N-view's is plausible (72 unfamiliar teams, half the pairs).

### 11.2 One factor at 100M (three runs a cell)

| view, true seed SD | median half-width | P(WIN) at +2 / +3 / +4 / +5 / +6 pp | at 0: NO-USEFUL-GAIN / UNDECIDED | false WIN / false LOSS at 0 |
|---|---|---|---|---|
| **N (width), 1.76** | 2.14 | 0.42 / 0.77 / **0.95** / 1.00 / 1.00 | 0.93 / 0.03 | 0.02 / 0.018 |
| N, 2.5 | 2.71 | 0.30 / 0.56 / **0.78** / 0.92 / 0.98 | 0.75 / 0.18 | 0.04 / 0.035 |
| N, 3.1 | 3.28 | 0.25 / 0.44 / **0.63** / 0.80 / 0.91 | 0.60 / 0.32 | 0.04 / 0.042 |
| **F (breadth), 1.76** | 2.19 | 0.40 / — / **0.94** / — / 1.00 | 0.92 / 0.04 | 0.02 / — |
| **F, 2.5** | 2.76 | 0.30 / — / **0.77** / — / 0.98 | 0.73 / 0.20 | 0.03 / — |
| F, 3.1 | 3.31 | 0.24 / — / **0.62** / — / 0.91 | 0.58 / 0.33 | 0.04 / — |
| N, 1.76, two runs a cell (the registration, `power_matrix.log`) | 3.04 | 0.19 / 0.44 / 0.72 / 0.90 / 0.97 | 0.72 / 0.27 | 0.01 / 0.009 |

- **In words.** The design finds a +4 pp effect about nineteen times in twenty if the seed SD at 100M is what it
  was at 15M, about four times in five if it is 2.5 pp and five times in eight at 3.1 (at two runs a cell: seven in
  ten, one in two, two in five). It finds +3 pp three times in four in the first case. It still cannot find +2 pp
  reliably (0.42, with UNDECIDED 0.17; at a seed SD of 3.1 UNDECIDED is the likeliest reading of a true +2, 0.51).
  A true zero (N, 1.76) reads NO-USEFUL-GAIN 93 % of the time, UNDECIDED 3 %, WIN 1.8 % and LOSS 1.8 %. At two runs
  it read UNDECIDED 27 % of the time: that is what the third seed buys.
- **Breadth is the weaker read**: one 500-pair panel, an unmeasured seed SD, and fork evidence of a small effect
  (X61P: +0.55 pp against Kakuna on F, and −3.28 pp [−4.14, −2.43] in-house on home teams). If the fork evidence
  holds from scratch, the likeliest breadth outcome is NO-USEFUL-GAIN. This is registered knowingly: the
  alternative meter could not fire.
- **The 50M harm stop** (N): P(STOP-HARM) at a seed SD of 1.76 is 0.000 at 0, 0.019 at −2, 0.42 at −4, 0.78 at −5,
  0.95 at −6 and 1.00 at −8; at 2.5 it is 0.001, 0.036, 0.31, 0.56, 0.79 and 0.98. It catches a clear harm, not a
  mild one. The same numbers are its cost: a factor that is 4–5 pp BEHIND at 50M and would have led at 100M (a wide
  arm that starts slowly) is stopped 42–78 % of the time and never read (§16).
- **Breadth's 50M stop reads HF, half its net.** With seed effects shared across panels (seed SD 1.76),
  P(STOP-HARM) is 0.000 at a true (home72, F) of (0, 0) and of (−4, +4), 0.019 at (−4, 0), 0.42 at (−8, 0), 0.95 at
  (−12, 0) and at (−6, −6), 1.00 at (−16, 0); a home cost of −12 with an F gain of +4 stops 42 % of the time. With
  independent seed effects the same rows read 0.000, 0.000, 0.004, 0.51, 1.00, 1.00, 1.00 and 0.51. So breadth is
  stopped at 50M about half the time by a net of −8 pp and almost always by −12; X61P's fork cost (−3.28 on home
  teams) would stop it less than 2 % of the time.
- **An interaction does not bias a main effect's interval**: it shifts the main effect by half its size, which is
  what a main effect is (at registration: width +4 with interaction +6 reads +6.99, P(WIN) 1.00).

**Breadth as registered: the WIN on F and the net rule together** (seed SD 1.76; each cell is P(WIN, adopted) /
P(WIN-NET-NEGATIVE)):

| true (home72, F), pp | true net | seed effects shared across panels | independent across panels |
|---|---|---|---|
| (0, +4) | +4 | 0.94 / 0.00 | 0.94 / 0.00 |
| (−2, +4) | +2 | 0.83 / 0.11 | 0.88 / 0.06 |
| (−4, +4) | 0 | 0.50 / 0.44 | 0.50 / 0.45 |
| (−6, +4) | −2 | 0.17 / 0.77 | 0.09 / 0.86 |
| (−3, +6) | +3 | 0.93 / 0.07 | 0.98 / 0.02 |
| (−6, +6) | 0 | 0.50 / 0.50 | 0.50 / 0.50 |
| (−9, +6) | −3 | 0.07 / 0.93 | 0.02 / 0.98 |
| (−3.28, +2), X61P's fork cost with a +2 gain | −1.28 | 0.26 / 0.15 | 0.16 / 0.25 |
| (0, 0) | 0 | 0.02 / 0.00 | 0.02 / 0.00 |

- **Near a true net of zero the net rule is a coin flip**: a WIN is adopted half the time. At a true net of −2 a
  WIN is still adopted 9–17 % of the time, and at +2 it is refused 6–11 % of the time. The net rule compares point
  estimates; it is the owner's exchange rate applied to this sample, not a test that the net is positive.
- At a seed SD of 2.5 the same rows are in the log: (0, +4) 0.77 / 0.00, (−4, +4) 0.50 / 0.27 shared and 0.47 /
  0.31 independent.

**If the arms do NOT share one seed variance** (no true effect; width on N; seed SDs of C, W, Br, WB):

| seed SDs (C, W, Br, WB) | false WIN | false LOSS | NO-USEFUL-GAIN | median half-width |
|---|---|---|---|---|
| 1.76 in all four | 0.018 | 0.018 | 0.93 | 2.13 |
| 1, 6, 1, 1 | 0.055 | 0.054 | 0.56 | 3.15 |
| 0.5, 8, 0.5, 0.5 | 0.068 | 0.068 | 0.41 | 3.94 |
| 1, 6, 1, 6 (both wide arms) | 0.056 | 0.055 | 0.38 | 4.39 |
| 6, 1, 6, 1 (both narrow arms) | 0.054 | 0.056 | 0.38 | 4.40 |
| 1, 1, 6, 6 (both breadth arms) | 0.056 | 0.055 | 0.37 | 4.41 |

The floor makes the rule conservative when the arms are alike (false WIN 1.8 %, not 5 %); one arm several times
noisier than the rest brings each tail back to 5–7 %.

**No two-stage rule (Amendment 1).** The registration's table for its extension (read at two runs, extend an
UNDECIDED, read again at three) is part (10) of `power_matrix.log`. It no longer applies: the class is read once,
at three runs, and the first table's error columns are the whole rate.

**Multiplicity.** Two factors are each read two-sided at 90 % (5 % a tail); nothing is corrected for there being
two. With no true effect in either, the chance that a given factor reads a false WIN is 1.8 % at a seed SD of 1.76
and about 4 % at 2.5 to 3.1; the chance that at least one of the two does is at most twice that. The interaction,
the simple effects, the 75M and EQ looks, the team bootstrap and every non-deciding view are REPORTED and decide
nothing, so they add no decision error.

### 11.3 The interaction

Median half-width 4.26 pp at seed SD 1.76 (5.43 at 2.5). P(SYNERGY) 0.43 / 0.77 / 0.95 / 1.00 at a true +4 / +6 /
+8 / +10 pp (0.31 / 0.56 / 0.78 / 0.92 at 2.5); false SYNERGY or ANTAGONISM ≈ 0.02 each at zero (0.04 at 2.5). An
interaction of +6 pp or more would usually show; one the size of the +4 pp bar would not.

### 11.4 Kakuna, the confirming meter

Simulated at three runs a cell (Amendment 1: [`kakuna_oc_n3.py`](kakuna_oc_n3.py) →
[`kakuna_oc_n3.log`](kakuna_oc_n3.log) section 4, which runs the registration's `kakuna_oc.py` with twelve runs;
4,000 experiments a cell, a rate's Monte-Carlo error at most 0.008), on the registered design (360 deals × 2 games
= 720 games a run a panel) and the registered analytic interval at its 90 % level, under an ASSUMED team model:
our-team SD 10 pp, their-team SD 9 pp, team-by-arm interaction SD 5 pp. The seed SD on Kakuna is unmeasured, so
three values are shown.

| contrast | seed SD | false POSITIVE / NEGATIVE at 0 | P(POSITIVE) at +2 / +3 / +4 / +5 / +6 pp | median half-width | found 80 % of the time from |
|---|---|---|---|---|---|
| `width_U` | 1.0 | 0.034 / 0.035 | 0.62 / 0.92 / 0.99 / 1.00 / 1.00 | 1.60 | 2.6 pp |
| `width_U` | 1.5 | 0.042 / 0.045 | 0.51 / 0.82 / 0.95 / 0.99 / 1.00 | 1.87 | 2.9 pp |
| `width_U` | 2.5 | 0.043 / 0.056 | 0.32 / 0.53 / 0.74 / 0.90 / 0.97 | 2.78 | 4.4 pp |
| `breadth_F` | 1.0 | 0.021 / 0.023 | 0.28 / 0.56 / 0.81 / 0.95 / 0.99 | 2.73 | 4.0 pp |
| `breadth_F` | 1.5 | 0.025 / 0.030 | 0.29 / 0.55 / 0.77 / 0.92 / 0.98 | 2.78 | 4.2 pp |
| `breadth_F` | 2.5 | 0.039 / 0.047 | 0.24 / 0.43 / 0.63 / 0.80 / 0.91 | 3.20 | 5.0 pp |
| `width_U`, two runs a cell (the registration, `kakuna_oc.log`) | 1.5 | 0.030 / 0.033 | 0.33 / 0.57 / 0.79 / 0.93 / 0.98 | 2.52 | 4.1 pp |
| `breadth_F`, two runs a cell (the registration) | 1.5 | 0.014 / 0.017 | 0.14 / 0.28 / 0.48 / 0.70 / 0.85 | 3.92 | 5.7 pp |

- **What it means for §6's labels.** A true +4 pp width gain on U is CONFIRMED nineteen times in twenty at a seed
  SD of 1.5 pp and three times in four at 2.5 (four in five and one in two at two runs). A true +4 pp breadth gain
  on forum72 is CONFIRMED about four times in five at 1.0 to 1.5 and five times in eight at 2.5 (one in two at
  best at two runs). So at three runs CONFIRMED is the expected label for a gain of the bar's size, and an
  UNCONFIRMED WIN is weak evidence that the gain is smaller; it still changes no decision (§6).
- **Below the bar the meter cannot see breadth.** It confirms a true +2 pp on forum72 about one time in four, and
  the fork evidence there is +0.55 pp. An UNCONFIRMED breadth WIN therefore means the in-house meter decided alone
  (§16).
- **A contradiction is rare when there is none**: with no true effect the meter reads NEGATIVE at most 5.6 % of the
  time, and POSITIVE (the KAKUNA-ONLY owner row) at most 4.3 %.
- **At three runs the interval is no longer conservative.** Its coverage at a true zero is 0.93 / 0.91 / 0.90 on U
  and 0.96 / 0.94 / 0.91 on forum72 at a seed SD of 1.0 / 1.5 / 2.5, against a nominal 0.90 (0.97 / 0.94 / 0.91 on
  U at two runs). Across the log's five sensitivity models (section 2: smaller team effects, a larger or absent
  interaction; seed SD 1.5) it is 0.889 to 0.925 on U and 0.908 to 0.961 on forum72: on U about one point BELOW
  nominal in four of the five (Monte-Carlo error 0.005; 0.906 or better at two runs). The rule is not changed:
  this amendment moves no bar and the meter changes no class. Flagged in §17 item 28.
- **Why the bootstraps are REPORTED and decide nothing.** The literal team × seed bootstrap still under-covers
  when the seed spread is large: on U 0.93 at a seed SD of 1.5 pp and 0.86 at 2.5, against 0.90 (section 3; 1,000
  experiments; 0.88 and 0.78 at two runs).
- Every number here is conditional on the assumed team model; the read's own record reports the measured variance
  components, which replace the assumption after the fact and change no class.

### 11.5 After a STOP-HARM of the other factor

No class is registered there (§8). What the surviving three-against-three comparison would show, had the +4 pp
bar been applied to it (4 df):

| view, seed SD | median half-width | P(lo > 0) at +4 / +6 / +8 / +10 pp | P(hi < +4) at 0 |
|---|---|---|---|
| N (width's survivor), 1.76 | 3.28 | 0.61 / 0.93 / 1.00 / 1.00 | 0.60 |
| N, 2.5 | 4.20 | 0.43 / 0.75 / 0.93 / 0.99 | 0.41 |
| F (breadth's survivor), 1.76 | 3.46 | 0.57 / 0.90 / 0.99 / 1.00 | 0.56 |
| F, 2.5 | 4.37 | 0.41 / 0.72 / 0.91 / 0.98 | 0.39 |

It finds +4 pp four to six times in ten and +6 pp seven to nine times in ten (at two runs an arm: one time in six
and one in two to three). It still carries no class: Amendment 1 moves no class rule (§8).

## 12. Cost and timeline

Rewritten 2026-11-12 from the measured ρ = 1.392 (§18, slots S1 and S2, which show the arithmetic). The figures
this replaces were built on a guessed ρ of 1.65: ≈ 311 GPU-h (270–340) and ≈ 16 days.

| item | GPU-h | CPU | calendar |
|---|---|---|---|
| phase 1, twelve runs to 50M | ≈ 136 (6 × 9.48 + 6 × 13.20) | — | ≈ 5.7 days |
| phase 2, 50M → 100M | ≈ 145 (6 × 10.07 + 6 × 14.02) | — | ≈ 6.0 days |
| starts: ≈ 60 over the twelve runs | ≈ 20 if every one is cold (0.33 h each) | — | ≈ 0.8 days |
| four looks, 270,000 games each | ≈ 2.0 each under the GPU lock | or ≈ 27 h each on the three agent cores | between runs |
| Kakuna, 25,920 games | — | ≈ 180 lane-h | ≈ 60 h on 3 lanes |
| **to the decision** | **≈ 281 of training (210–281) + ≈ 20 of starts + ≈ 8 of looks ≈ 309** | | **≈ 15.4 days** |
| after a WIN: three candidates to the plateau | ≈ 154 for a wide cell, ≈ 111 for a narrow one (GUESS) | ≈ 45 lane-h | ≈ 6.4 / 4.6 days |
| not registered, for scale: a fourth seed an arm | ≈ 94 | ≈ 60 lane-h | ≈ 3.9 days |

- **The narrow figure is X26's measured hours, and it is the central one**: `rb_x26_s2601` took 4.40 / 9.48 /
  19.55 GPU-h to 25M / 50M / 100M (704 s per 1M steps over its first 100M). X26 carried five ride-along heads,
  predates E13 and trained at `ent_coef` 0.05; whether its hours include its own starts is UNMEASURED. They are
  kept because they hold what a pilot cannot: evaluations, a growing self-play pool and 100M steps of it.
- **The wide figure is the narrow one × ρ, with ρ = 1.392 MEASURED** (§13 item 4, at the launch pin, 2026-11-12:
  734.6 s per 1M steps at width 256 against 527.7 at 128, medians over 37 steady updates each between 0.3M and
  3.9M steps, `ent_coef` 0.0125, the self-play pool still empty; the update alone is 1.450×, the collection
  1.126×). It replaces the guess of 1.65 (1.3–1.9). 13.20 / 27.21 GPU-h to 50M / 100M a wide run. **UNVERIFIED
  that the ratio holds after the pool is seeded and late in training**: no steady row was read after the seed
  (it came at 4M, where the pilots stopped).
- **The interval's two ends are two different measurements.** Its LOW end, ≈ 210 GPU-h, prices all twelve runs at
  the pilots' own steady rates (14.66 GPU-h per 100M at 128, 20.41 at 256): early training, pool empty,
  evaluation and checkpoint time excluded, so it is a floor and not an estimate. Its HIGH end is the central
  figure, ≈ 281 (X26's hours and × ρ). The control pilot's rate is 25 % below X26's average over 100M and 17 %
  below X26's over its first 25M (634 s per 1M); which of the differences above accounts for that is UNMEASURED.
  Nothing prices a run slower than X26: the first control run's realised hours to 50M are the check (S8).
- **Starts are priced separately; ρ does not hold them.** A cold start took 1,195 s (W) and 1,192 s (C) from the
  launch to the last startup line before the first rollout on the pilots, of which the inference service's start
  (its CUDA-graph captures) was 886.1 / 891.0 s, the trainer's compile prewarm ≈ 245 s and the fit check's dry
  update 21 to 31 s. A child lives six hours from its own start, so a narrow run takes 2 starts a phase (9.48 and
  10.07 h) and a wide one 3 (13.20 and 14.02 h): 6 × 4 + 6 × 6 = 60 starts, ≈ 20 GPU-h if every one cost the cold
  figure (≈ 15 for the service's part alone). Every restart and resume at the pin finds the shared compile cache
  the first start filled; what such a WARM start costs is UNMEASURED and is read off the first run's first
  restart (§18, slot S8). The de-risk's four cold starts at other commits read 938 to 1,011 s for the service
  alone.
- **The probes' own forwards** (`e112cbcbf`: about 1 % of the learner's wall on a diagnostics update, an
  estimate) are inside both pilots' rates and not in X26's.
- **After a WIN** the cell's three runs go on to the plateau: X26 took 36.95 GPU-h from 100M to its plateau at
  280M (56.5 − 19.55), so three runs cost ≈ 111 GPU-h in a narrow cell and ≈ 154 (× ρ) in a wide one. A GUESS
  either way: nothing says this lineage plateaus near 280M.
- **Figures elsewhere that were priced at the guessed ρ and are not edited:** §9's and Amendment 1's "≈ 104 GPU-h"
  for a seed an arm (≈ 94 at the measured ρ: 2 × 19.55 + 2 × 27.21), and Amendment 1's bill (twelve runs cost
  ≈ 94 GPU-h more than eight, ≈ 281 against ≈ 187; it was written as ≈ 104 more than ≈ 207). A look holds 2.25
  times the registration's cells (54 for 24), so the four looks cost ≈ 8 GPU-h instead of ≈ 4, and the Kakuna read
  half as much again.
- **Timeline.** Registered Mon 2026-11-09, amended Tue 11-10 and Thu 11-12 (twice). The launch day L is the
  orchestrator's GO after CE-3d's three forks. One card, the runs serial, the card doing nothing else. From L:
  phase 1 ends ≈ L + 6.1 days (136 GPU-h and ≈ 10 of starts), the 50M look that day on the card, phase 2 ends
  ≈ L + 12.6 days, the 100M, 75M and EQ looks ≈ 6 GPU-h after it, Kakuna ≈ 2.5 days on three lanes, the decision
  ≈ L + 15.4 days. At the interval's low end (≈ 210 GPU-h, 48 starts) the decision is ≈ L + 12.3 days.

## 13. The GPU pre-launch checklist

**Status, 2026-11-12: items 1 to 7 are GREEN at the launch pin.** Items 1, 6 and 7 and the CPU half of item 2 are
in §18's re-pin record; items 3, 4 and 5 and the card-visible half of item 2 ran on the card on 2026-11-12
(04:54 to 08:51 PST, ≈ 3.95 h of the lease) and every one passed: §18's slots S1 to S7 hold the records. Items 8
and 9 are checked at the launch (slot S8). The text below is the checklist as it was registered.

Nothing here could be verified on the card by the registrar, by Amendment 1 or by the re-pin: none had a GPU
lease. Every item is done at the LAUNCH pin. Items 1, 6 and 7 and the CPU form of item 2 are in §18's re-pin
record. Items 3, 4 and 5 and the card-visible form of item 2 are done by the agent the orchestrator leases the card
to (runbook: `~/.cache/gen3ai/handoffs/b1_pin_runbook_2026-11-12.md`), and each result is recorded in §18 before
the first launch. **No record made at another commit satisfies an item (Amendment 2)**: the de-risk of 2026-11-11
ran at `aee363bc9` and `e112cbcbf`, trainer code changed after both, and its pilots typed neither base knob.

1. **The kit's check at the pin**, from a clean checkout of it: `python -m main.b1.arms check --tree <checkout>
   --pin <PIN> --params` (seeds and names still free; corpus identity; levers; parameter counts 2,609,183 /
   6,258,079).
2. **Each first-seed dry run with the card visible, from a shell at nice ≤ 10**: `python -m main.launcher
   --dry-run <argv>` for the four arms passes EVERY preflight (on CPU from a nice-19 shell its two refusals are
   the CUDA preflight and the inherited-nice gate, §2).
3. **The CUDA-only slow tests at the pin**, banked in a leased `slow-gpu` window (`python -m
   main.ops.gpu_fill_window`, `../../../ops/testing.md` § The GPU slow tier). They have not run on the code this
   experiment trains: E13's ship skipped 9 (the compile-region gates on trained weights among them) and the width
   ship skipped 11 in `src/agents/model`. A FAIL in a test the wide or the production path touches holds the launch.
   History, which satisfies nothing: at `aee363bc9` the window read 29 of 30 PASS, the one FAIL being the
   production forward + backward compile at B = 2 on Triton 3.4.0 (fixed at `1d2aa1695`; its card run was this
   item's, and it passed: §18, slot S4); at `1fe7283f7` a shakedown read 30 of 32, the FAILs being that compile and the offload cell's
   ledger assertion (fixed at `ef89ca99f`).
4. **The wide-arm pilot, to trained weights** (Amendment 1; the two pilots take 1.2–2.6 GPU-h, an estimate that
   predates the measured ≈ 16-minute cold start of each: the first evaluation is at 2M and the pool is seeded at
   bots ≥ 0.55): W's argv into a scratch models dir, and C's the same, **each in the launch shape at the launch
   pin (the base typed; only the seed and the run name differ from the kit's argv, Amendment 2)**, each run until
   it has written its first 1,000,000-step checkpoint, finished its
   first evaluation cycle AND made its first promotion into the self-play pool, whichever comes last. The
   registration's pilot ran "a few updates", where every parity gate compares an initialisation; this one makes
   them compare TRAINED 256-wide weights and a promoted 256-wide snapshot. It gives (a) step time at 256 against
   128, over the steady updates after the compile → **ρ**, which REWRITES §12's budget and fixes §7's EQ step, both
   recorded in §18 before the first launch; (b) the bf16 compile at 256 completes and its compile-parity check
   passes; (c) the inference CUDA-graph capture at 256 succeeds; (d) peak GPU memory at 256 fits with the
   production batch: the startup fit check passes and `update_fit.json`'s headroom is at or above its 1,024 MiB
   bar, recorded with the demand (§18 Amendment 2 item 6; at `e112cbcbf` it read 1,318.5 MiB at 256 and 3,472.5 at
   128); (e) no parity gate FATALs through the first promotion. The width commit records (b) to (d) as
   UNVERIFIED on GPU. Any failure holds the launch and comes back as a build. The pilot's 1M checkpoints serve
   item 5. A FATAL after the launch is §10's rule.
5. **A 128-against-256 head-to-head cell on the card**: one short `main.h2h` cell between the W and the C pilot's
   1M checkpoint on the look's device and settings, to prove the engine pairs two widths in one process on the card.
   On CPU it is proven: `src/main/b1/width_pairing_integration_test.py` plays cross-width cells in both directions
   and a same-width cell of each width on ONE engine, from checkpoints built at the control's and the width arm's
   own resolved configs (3 tests, 520 s beside a live run, banked 2026-11-09). The GPU inference path (graph
   capture, bf16) is not covered by it.
6. **The stop-and-resume check at the pin**: `python -m main.b1.resume_check static --tree <checkout> --pin <PIN>`
   prints `PASS` (CPU, seconds; no lease). Its two-segment smoke (`python -m main.b1.resume_check smoke --models-dir
   <an empty scratch dir> --arm wb --half 4000`, CPU, ≈ 35 min, 8 GB peak) is banked in
   [`resume_check.log`](resume_check.log); it is repeated only if the trainer, launcher or model code at the launch
   pin differs from the tree it ran on. It does (the base knobs, the memory-fit unit and more: §18's re-pin
   record), so the re-pin repeats it at the launch pin and `resume_check.log` is that run's transcript. Not green
   ⇒ §4's fixed fallback block.
7. **The spec**: `python -m main.reads check-spec --spec <this folder>/read_spec_kakuna.json` at the pin reports
   `play_path_differs_at_head` 0, or names the files.
8. **The HOLDs of §10** are checked against the day's STATUS: X60-T's verdict, the owner's word on `ent_coef`.
   Both are closed by Amendment 2; the check at launch is that STATUS still says so.
9. **The core split stays frozen from the first launch to the last run's end** (an operator rule; Amendment 1).
   `python -m utils.core_split show` is recorded in §18 at every launch, and a launch whose split differs from the
   first launch's is not made. No code enforces it. The registration relied on the look refusing runs whose
   recorded `init_num_threads` differ, but every fresh build records 1 whatever the split
   (`utils.torch_state_guard.single_thread_build`), so that guard cannot fire. The core-split A/B that
   `designs/ops/core_split.md` proposes runs at X67, never on a BOUNDARY-1 run.

## 14. Commands

Run every training command from the MAIN checkout with the launcher's interpreter (`$GEN3AI_PYTHON`), one at a time,
**from a shell at nice ≤ 10** (Amendment 2: the launcher's inherited-nice gate, `src/main/launcher/nice_gate.py`,
refuses a GPU launch typed from a shell above nice 10). **`<PIN>` in every command below is the launch pin,
`15136b61634b667bb9da5ccc6647ff9265238829`** (§3; §18's re-pin record).

- **Launch block** (the Training session's): `launch_block.txt`, which is `python -m
  main.b1.arms describe --pin <PIN>` (twelve fresh launches, then twelve resumes). **It is written by §18's re-pin
  record and names the launch pin on its second line**; no other block launches anything. The fallback is `python -m main.b1.arms describe --block fallback --pin <PIN>`.
  The block is typed ONE COMMAND AT A TIME by the Training session: the arm chain (`main.x35`'s kit) knows only the
  X35 fork arms and refuses a fresh arm (`kind 'fresh' has no implementation`), and no chain backend was built for
  this unit. Each command is followed, commented, by its own `--dry-run` form; run it first, and expect "would
  launch" (a phase-2 dry run reads "RESTART of models/<run>"). Re-issuing a phase-2 command on a run that already
  reached 100M exits 0 and trains nothing. On a phase-2 launch the launcher's banner reads "`--model` launch
  (fork)"; that wording is the launcher's for every `--model` command. The dry run's `role : RESTART`, the child's
  `[Recipe] --model launch (restart)` lines and the lineage sidecar are what say it is the same run.
- **After every launch** (§10's early check; no lease, seconds), once the run has its first checkpoint, from a
  clean checkout of the KIT commit (Amendment 4), with `git -C <main checkout> rev-parse HEAD` and `python -m utils.core_split show`
  noted at the launch itself; all three go into §18:

  ```bash
  python -m main.b1.look check-run --arm W --seed 5103 --pin <PIN>     # exit 0 required; 3 = stop the run
  ```
- **At each run's end, before the Kakuna `plan`** (Amendment 2; §10's K9(b) rule; no lease, seconds): for all
  twelve runs, after phase 2, from a clean checkout of the KIT commit (Amendment 4). Each JSON goes into §18. No code orders it before
  `plan`: it is an operator rule.

  ```bash
  python -m main.b1.look check-run --arm W --seed 5103 --pin <PIN> --step 100000000 --arm-end   # exit 0 required
  ```
- **A look** (any agent holding the GPU lease for the `cuda` form; the `cpu` form needs no lease), from TWO clean
  detached checkouts (Amendment 4): the kit commit plans and reads, the pin plays every game:

  ```bash
  git worktree add --detach <PIN_DIR> <PIN>          # the launch pin of §18, never the registered 507d0c11…
  git worktree add --detach <KIT_DIR> <KIT>          # the kit commit of Amendment 4
  <KIT_DIR>/designs/research_state/measurements/boundary1_scratch_2026-11-09/looks/drive_look.sh \
      <KIT_DIR> <KIT> <PIN_DIR> <PIN> <WORK_DIR> 50M cuda
  # or 75M / 100M; EQ takes --eq-step <N>; after a replacement, --replace <the registered seed 5109 replaced>;
  # after a STOP-HARM of the other factor, --survivor width|breadth
  ```

  It plans (exit 3 = a §10 guard refused), registers each panel's family (idempotent), plays the 54 cells a panel
  with `python -m main.h2h play-many` under `gpu_lock.sh` + `mem_cap.sh`, writes `<WORK_DIR>/result/verdict.json`
  with `python -m main.b1.look read`, and then the REPORTED `<WORK_DIR>/result/team_bootstrap.json` (§5.5; `python
  -m main.b1.team_bootstrap --work <WORK_DIR>`). One look is played on ONE device. After the Kakuna read: `python -m
  main.b1.look read --work <WORK_DIR> --kakuna width=<class> --kakuna breadth=<class>` adds §6's labels (the full
  100M look only).
- **The Kakuna read** (CPU lanes, rule 5b), from the same checkout, state outside the repo:

  ```bash
  python -m main.reads check-spec --spec <spec>
  python -m main.reads plan   --spec <spec> --state ~/gen3ai_archive/boundary1_kakuna
  setsid nohup python -m main.reads run --state ~/gen3ai_archive/boundary1_kakuna --lane anchor \
      --concurrency 3 --hours 44 > ~/gen3ai_archive/boundary1_kakuna/anchor.log 2>&1 &
  python -m main.reads record --state ~/gen3ai_archive/boundary1_kakuna
  ```

  `plan` runs as soon as phase 2 ends and the twelve arm-end K9(b) reads are PASS, before any run continues (§6). `run` is resumable: it is re-issued until no
  unit is unplayed (≈ 180 lane-hours; one lane alone would play a ninth of the read in 20 h). `record` on an
  unfinished read gives UNREAD (§6), never a label to act on.

## 15. What was built for this registration

| tool | where | pinned by |
|---|---|---|
| the fresh-run kit: arms, the base every run types (Amendment 2), argvs, seeds, the collision scan, the launch block | `src/main/b1/arms.py` | `arms_test.py` |
| the stop-and-resume equivalence check | `src/main/b1/resume_check.py` | `resume_check_test.py` |
| the multi-panel cross: views, the floored statistic, classes, the net rule, the confirming labels, the interaction | `src/main/b1/cross.py` | `cross_test.py` |
| the look: run guards, checkpoint rule, cells plan, ledger names, the read; the per-run early check (`check-run`, Amendment 1); the K9(b) read at the early check, at each look and at arm end (`k9b_read`, `check-run --arm-end`, Amendment 2) | `src/main/b1/look.py`, `looks/drive_look.sh` | `look_test.py` |
| the REPORTED team-cluster bootstrap (Amendment 1) | `src/main/b1/team_bootstrap.py` | `team_bootstrap_test.py` |
| the Kakuna rule, clustered by team and seed | `src/main/b1/kakuna_rule.py` | `kakuna_rule_test.py` |
| the two widths on one head-to-head engine (CPU) | (no new code: `main.h2h`'s two-architecture engine) | `width_pairing_integration_test.py` (slow) |
| the operating characteristics | `power_matrix.py`, `kakuna_oc.py`; at three runs an arm (Amendment 1) `power_matrix_n3.py`, `kakuna_oc_n3.py`, which import the first two | each checks the rule code against its own formula |

Each test file holds cases that fail when the rule it pins is reverted (the floor, the df, the boundary band, the
net rule's sum, each guard's refusal, each estimator against a hand-computed case). For `main.b1.cross` and `main.b1.look` that was
measured once, on a hand-listed set (not an exhaustive one): 32 single-site changes (every registered constant, each
degrees-of-freedom and variance term, the deciding and harm views, the net rule's comparison, each guard), run
against the two test files in a shadow copy; 32 of 32 fail a test. (Measured at registration, on the two-seed kit;
not repeated for Amendment 1, whose kit changes are the seed table, `check-run` and the bootstrap, each with its
own tests.)

## 16. What is not claimed, whatever the outcome

- Nothing about the plateau from a 100M look.
- Nothing about widths other than 256, about depth, the critic, the policy tower or entropy.
- **Nothing about `ent_coef`** (Amendment 2). Every run trains at 0.0125 and none at 0.05, so no number here says
  what the coefficient did. It is a base choice resting on X67 (late forks, each LR-MOVED, no Kakuna game), not a
  finding of this experiment; a bad base cannot confound a contrast between arms that share it, but it can lower
  all four.
- **Width is read under the 128-wide recipe alone**: the learning rate, batch and epochs are production's, tuned
  at 128, and `ent_coef` 0.0125 was screened at 128 only. A LOSS or a NO-USEFUL-GAIN is about 256 under that recipe, not about width with a
  recipe of its own.
- Nothing about other corpora or shares than `n1_proc_v1` at 0.5 on both sides.
- No pp against Kakuna from a pp in-house: the mapping is unmeasured.
- **Kakuna cannot confirm a breadth gain much below +4 pp** (it finds a true +2 pp on forum72 one time in four,
  §11.4). An UNCONFIRMED breadth WIN is the in-house meter deciding alone: not a contradiction, not a
  confirmation.
- A NOT-SHOWN interaction is not additivity; a NO-USEFUL-GAIN is not "no effect"; UNDECIDED is not a null.
- A STOP-HARM at 50M says the factor was behind at 50M. It is not a LOSS at 100M: a factor that starts slowly and
  would have led at 100M is stopped and never read.
- A breadth WIN that passes the net rule is not evidence that the net is positive: near a true net of zero the
  rule passes about half the time (§11.2).
- Nothing from the survivor's three-against-three comparison after a STOP-HARM, beyond its printed interval.
- Nothing from the team bootstrap (§5.5) beyond "the class would, or would not, survive another 72 teams", and
  that only as an upper bound.
- Breadth's read on H, P72, U and N is contaminated by its own training teams and is never quoted as generalisation.

## 17. Deviations from the draft, and the registrar's choices (each flagged for the orchestrator)

1. **Breadth decided in-house on F** with the net rule, Kakuna confirming (orchestrator's ruling; §1, §5.3).
2. **Breadth's 50M harm view is HF** (§8): the draft had one harm rule on N; under the ruling the only recognised
   home-team cost is the net, so the stop reads the net's view. Registrar's choice.
3. **The Kakuna interval is analytic**, the bootstrap reported (§6; accepted by the orchestrator).
4. **Seeds:** one per run, 5101–5108, spare 5109 (the draft reused seed numbers across arms; accepted).
   Amendment 1 adds 5110–5113 (item 19).
5. **Checkpoints every 1M, typed** (accepted); looks read checkpoints, never `final_model.zip`.
6. **One pin** for training, the cross and Kakuna; registered in two commits (the tooling, then this file naming
   the tooling's commit as the pin).
7. **After a STOP-HARM no class is read**, for either factor: the survivor's simple effect (4 df at three runs an
   arm) is REPORTED and goes to the owner, and no Kakuna read is registered (§8; the draft's single-factor options
   had no such case). At registration the alternative, a class on two runs against two, would have found +4 pp one
   time in six; at three against three it is four to six in ten (item 22).
8. **A measured ρ above 1.9** goes to the orchestrator, not the owner (§12).
9. **The EQ step is fixed from the pilot's ρ** before launch (§7), not from the runs' own realised time; its
   ledger request carries the step.
10. **An unfinished Kakuna read is UNREAD**, never UNCONFIRMED: it is finished or replayed, and §9 waits (§6, §9).
11. **§9's order of reading**: the Kakuna record first, owner rows before any action, then the continuing cell
    from the set of WINs. No run continues before the read's `plan` has copied the twelve finals.
12. **A look's cells must hold exactly the registered pairs, at most a quarter of a cell's games drawn, rows at
    the pin on the panel's team set, and one device and backend** (§8, §10): the draft had none of the four.
    `main.h2h` scores a turn-limit game as a draw, so the draft's "aborted" clause could never have fired.
13. **Under the fallback block the 50M look stops nothing** (§4).
14. **Replacement is a closed list of two causes**; everything else is lost to its lever (§10).
15. **The extension** (re-read both factors at three runs an arm after an UNDECIDED) **is withdrawn by Amendment
    1**: three runs an arm are registered from the start and read once (§9).
16. **No amendment to §5, §6, §8 or §9 after the first look** except by the owner (header).
17. **The pin is an argument**, never a constant in the code (§3).
18. **A look refuses recipe drift** (§10): any recorded knob of a run's last process that differs from the first
    control run's, beyond the run's own keys and its arm's lever, refuses the plan. It compares records with
    records, so it catches a flag typed on one run's resume; it cannot catch a knob every run got wrong alike
    (that is `checkargs` at launch, §13 item 2). Its blind spot, by construction: one of the ten keys a resume
    rewrites, moved in a run whose last process is of another KIND than the control's (a replacement run still on
    its restart children beside eleven phase-2 resumes), is not compared in `cli_args`; of those ten, the rollout
    shape is still caught at `metadata.json`'s top level, the width keys by the lever check, and `arch` by the
    token-for-token `original_command`; `unified_moves` / `unified_damage` and the four `*_source` labels are not.

**Amendment 1's choices** (2026-11-10; the decisions are the orchestrator's, these are the registrar's readings of
them):

19. **Third seeds 5110–5113; the spare stays 5109 and stays ONE** for twelve runs. No registered seed or run name
    moves. The third seeds are the last four launches of each phase.
20. **Curve dominance asks five of six ON runs** (§7; three of four at registration): the nearest count not below
    three quarters. REPORTED either way.
21. **After a WIN the cell's THREE runs continue to the plateau** (§9, §12: ≈ 154 GPU-h in a wide cell, a guess). Continuing fewer
    is the orchestrator's to decide, by an amendment before the 100M look, with the rule that picks them.
22. **The survivor look stays unclassed** although three against three now finds +4 pp four to six times in ten
    (§8, §11.5): this amendment moves no class rule.
23. **A parity-gate FATAL: the second FATAL of the same run holds the campaign** (§10): the registrar's reading of
    "one same-pin resume allowed".
24. **The early check is a kit command** (`python -m main.b1.look check-run`), not a hand-written call: the same
    `check_run` and `expected_levers`, with an exit code and a JSON record (§10).
25. **The team bootstrap resamples per-batch team tallies**, one-way by the winning pilot's team, because the
    ledger banks no per-game row; its combined interval is an upper bound (§5.5).
26. **If the draw rule trips, nothing is replayed without an amendment** (§10).
27. **No re-pin in this amendment.** Three seeds need a kit change, hence a new pin, whatever `ent_coef` does; it
    is taken once, after that decision (§18).
28. **The Kakuna rule is unchanged although at three runs its interval is no longer conservative** (§11.4): its
    coverage on U is about one point below the nominal 0.90 in four of five sensitivity models, and a false
    NEGATIVE reaches 5.6 %. It labels and never decides; widening it would be a rule change after a simulation,
    which is the orchestrator's to make, by an amendment before the read.

**Amendment 2's choices** (2026-11-12; the `ent_coef` decision is the owner's, the rest the orchestrator's;
these are the registrar's readings of them):

29. **`ent_coef` is TYPED by the kit, and `designs/production_config.json` is not moved.** That overrides §18's
    re-pin step 1 as Amendment 1 wrote it and X67 §10's sentence that the config and the recipe design move
    together in this registration: the owner's yes is for this experiment only.
30. **`--behaviour-check warn` with X35's ceilings**, not re-derived for fresh runs or width 256 (§10).
31. **K9(b) is read through the look checkpoint's step at a look, and on every row at arm end.** That the arm-end
    reads come before the Kakuna `plan` is an operator rule; no code enforces it (§14).
32. **A K9(b) VOID is "lost to its lever"**, and an INCONCLUSIVE that cannot be repaired is treated as VOID (§10).
33. **This reader is stricter than X35's own**: a run with no full child log is INCONCLUSIVE, and a fresh run has
    no inherited rows to exclude, so every row is judged.
34. **No skip rule for §13.** The pre-launch review's F19 / F20 proposed letting a record made at another commit
    satisfy a GPU item under a code-equality rule. None is registered: trainer code changed after the de-risk, so
    every GPU item runs at the launch pin in full, and nothing here needs the rule.
35. **The de-risk's pilots fill no slot.** They typed neither base knob, reached no checkpoint and gave no ρ; they
    are read only as the memory-fit history and the cold-start readings (§12, §18).
36. **The pin holds this amendment and the kit; the records that name it are a later commit** (a commit cannot
    name itself, §3). Between the two the spec has no `pin` and the record-bound tests skip.
37. **No fit lever is typed.** The memory fit at width 256 is a trainer-code unit applying to every run (§18
    Amendment 2 item 6). What a fit refusal at a RESTART of a running campaign means was not decided by
    Amendment 2; **Amendment 3 decides it** (§18, §10: a start failure, retried from the same checkpoint).
38. **The two-segment smoke is repeated at the launch pin on CPU** (§13 item 6): trainer code moved.
39. **The seed floor and the narrow arm's cost were measured at 0.05** and are not re-measured at 0.0125; no bar
    moves for it (§11.1, §12).

**Amendment 3's choices** (2026-11-12; the classification is the orchestrator's, these are the registrar's
readings of it and of the code):

40. **A fit refusal is classed by WHEN it happens, not by the arm.** Before a segment's first training step it is
    a start failure; an out-of-memory failure after it stays "lost to its lever" (§10). The wide arms are the ones
    near the bar (374.5 MiB above it), but the rule is the same for all four.
41. **A start-failure retry is not the parity bullet's one resume** and does not count against it; no number of
    retries is registered. The limit is item 6's: one repeat on an empty card HOLDs the run.
42. **The move-aside at a fresh first start is registered although the launcher would accept the leftover dir**
    (it holds no checkpoint and no `model_config.json`). Read from the code, not executed. The one precedent on
    disk, CE-3d's refused FORK dir at another pin (`models/rb_x35_ce3dswap_s7321.FITCHECK_NO_MAGNET_COLUMN`),
    agrees: it holds the child logs, `crashes/`, `cuda_ledger.json`, `compile_cache/` and the fork's copied
    `snapshots/`, and no checkpoint, `metadata.json` or run-root `model_config.json`.
43. **The mid-phase retry is typed by hand.** The kit's `resume_argv` builds the phase-2 form alone and its
    resume check covers that form alone; form (c) differs in `--model` and `--steps` only and must pass
    `--dry-run` first. UNVERIFIED by execution: no run has been resumed from a periodic checkpoint with this kit.
44. **That a run counts after a retry is read from `main.b1.look` (now the kit commit's, Amendment 4), not executed**: the look reads the
    pin history, the lineage, `original_command`, the last process's recorded arguments and the K9(b) rows; it
    does not read `crashes/`.
45. **The seed's timing (item 7) changes nothing.** §13 item 4's stop rule read "its first promotion into the
    self-play pool" as the pool's SEED, which is what a fresh run's first promotion is; neither pilot ran on to a
    later promotion (slot S5).

## 18. Amendments and pre-launch records

### Amendment 1 (2026-11-10): three runs a cell, and the pre-launch review's fixes

Written before any launch, any game and any read, on the orchestrator's decisions after an independent pre-launch
review. **Blind: no BOUNDARY-1 run, checkpoint, game or Kakuna row exists. It moves no bar**: the +4 pp bar, the
−2 pp harm stop, the 1.76 pp floor, the 90 % level, the 0.005 pp band, the views, the classes and their order, the
net rule, the panels, the pairs a cell and the Kakuna rule are the registration's. `ent_coef` is unchanged.

1. **Three runs a cell, twelve runs, registered up front** (§2 to §5, §11, §12). Third seeds C 5110, W 5111, Br
   5112, WB 5113. Why: at two runs a cell a true zero read UNDECIDED 27 % of the time and a true +4 pp read WIN
   72 % (51 % at a seed SD of 2.5), and the registered answer to an UNDECIDED was a third seed (≈ 104 GPU-h) read
   as a second look. Three up front cost the same ≈ 104 GPU-h, are read once on 8 df (half-width floor 1.89 pp, was
   2.65), and find +4 pp 95 % / 78 % / 63 % of the time at a seed SD of 1.76 / 2.5 / 3.1.
2. **The extension is withdrawn** (§9, §10, §11.2, §12): an UNDECIDED at three runs goes to the owner with no
   pre-priced option.
3. **NO-USEFUL-GAIN reads "not shown at this size"**, never a retirement; a LOSS is scoped to this width under this
   recipe (§9, §16).
4. **The wide pilot runs to trained weights**: the first 1M checkpoint, the first eval cycle and the first pool
   promotion; its ρ rewrites §12 before the first launch (§13 item 4).
5. **A parity-gate FATAL**: one same-pin resume; a re-pin is the orchestrator's, by amendment, and replaces the
   affected runs from scratch (§10).
6. **The core split is frozen by an operator rule**, because the thread-count guard cannot fire (§13 item 9, §10).
7. **A per-run early check** after each run's first checkpoint, recorded with main's HEAD (§10, §14).
8. **A REPORTED team-cluster bootstrap** of each factor's deciding statistic (§5.5).
9. **Text from the review's small notes**: Kakuna's resolution for breadth (§1, §6, §11.4, §16); width under the
   128 recipe (§16); the draw rule's remedy (§10); where the 50M look is played and what it costs (§8, §12); the
   harm stop's cost to a slow starter (§11.2); the resume check's coverage (§4).
10. **Scaled with the runs, no rule moved**: 54 cells and 270,000 games a look; the interaction on 8 df; the
    survivor look three against three on 4 df, still unclassed; curve dominance five of six; the Kakuna read
    25,920 games (§5 to §8, §17 items 19 to 28).

**`ent_coef` stays an OPEN HOLD.** (Closed by Amendment 2, below.) It is 0.05 in every arm until X67 is read AND the owner says yes. This amendment
does not touch it. The value is a constant of the kit (`main.b1.look`) and of `designs/production_config.json`, so
changing it is a code change and therefore a new pin (the review's finding 4).

**Code, and what is deliberately NOT done here.** The amended kit (the seed table, `check-run`, the team bootstrap,
the look's driver) is on main in the commit immediately before this amendment's (`feat(b1): …`). The registered pin
`507d0c11…` holds two seeds an arm and cannot launch this design, so a re-pin is needed whatever is decided on
`ent_coef`. It is not taken now: a pin taken today would be taken again after that decision, with §13 repeated
twice. So the committed `argv_`, `checkargs_` and `dryrun_` transcripts, `read_spec_kakuna.json` and the renamed
launch block still describe eight runs at the registered pin, and the kit's record-bound tests skip until
`launch_block.txt` exists again.

**The single re-pin** (once; after the `ent_coef` decision, with §10's HOLDs clear), by the agent the orchestrator
names, recorded here as the next amendment before the first launch:

1. **Choose the pin**: a main commit that holds this amendment's kit and, if the owner changed `ent_coef`, that
   change (in `designs/production_config.json`, in `main.b1.look`'s `ENT_COEF`, in §1, §2 and §10 of this file,
   with their tests). If X60-T read ADOPT-CANDIDATE the registration is re-read first (§10). *(Amendment 2: the
   change is typed by the kit's base, `main.b1.arms`, and `designs/production_config.json` is not moved; the pin
   also holds Amendment 2 and the memory-fit unit with its follow-ups.)*
2. **Say what the pin adds**: the commits main gained since `507d0c119`, as §3 does, naming any that touch a
   trainer, launcher, model, observation or head-to-head file; `checkargs` at the pin states the production
   surface it resolves.
3. **Regenerate the records from a clean checkout of the pin**: `python -m main.b1.arms describe --pin <PIN> >
   launch_block.txt` (twelve launches, twelve resumes) and delete the superseded block; the `argv_`, `checkargs_`
   and `dryrun_` transcripts of each arm's first seed; `read_spec_kakuna.json` with the twelve models at the pin,
   and the model list `src/main/b1/kakuna_rule_test.py` pins.
4. **Check at the pin**: `python -m main.b1.arms check --tree <checkout> --pin <PIN> --params` (a fresh
   disjointness scan for 5101–5113; the corpus; the levers; the parameter counts), `python -m main.b1.resume_check
   static --tree <checkout> --pin <PIN>`, and `pytest src/main/b1`, where no record-bound test may skip.
5. **Repeat §13 items 1, 2, 6 and 7 at the pin**; item 6's two-segment smoke again only if trainer, launcher or
   model code moved *(it did: the smoke is repeated)*. Items 3, 4 and 5 are done at the pin on the card as
   written; items 8 and 9 at the launch.
6. **Write the pin** into §3 and §14, `designs/STATUS.md` and the backlog's MF-5 row.

Until that amendment exists nothing launches, and no command of §14 is run with the registered pin.

### Amendment 2 (2026-11-12): the base every run types, K9(b) judged by the look, the memory fit at width 256

Written before any launch, any game and any read, on the owner's decision of 2026-11-11 and the orchestrator's
decisions after the pre-launch review and the de-risk of 2026-11-11. **Blind: no `rb_b1*` run, checkpoint, game or
Kakuna row exists.** The de-risk's pilots (scratch models dirs, pilot seeds, the registered design's argvs without
this amendment's base) reached no checkpoint and played no game; what they gave is items 6 and 9. **It moves no
bar**: the +4 pp bar, the −2 pp harm stop, the 1.76 pp floor, the 90 % level, the 0.005 pp band, the views, the
classes and their order, the net rule, the panels, the pairs a cell, the seeds and the Kakuna rule are Amendment
1's. It changes the base all twelve runs share, and adds one condition for a run to count.

1. **`ent_coef` is 0.0125 in every run** (§1, §2, §10, §16; closes Amendment 1's OPEN HOLD).
   - **The decision is the owner's**: yes to 0.0125 for BOUNDARY-1, in writing to the orchestrator on 2026-11-11.
     It lifts the 2026-09-29 ruling for this experiment only.
   - **The basis is X67's PASS** (verdict at `aee363bc9`): D +1.97 pp [+1.35, +2.59] for 0.0125 against 0.05.
   - **Its caveats, which the decision was made with:** X67 read LATE FORKS (P* + 20M), not fresh 100M runs ("a
     late fork is not a fresh run", its §10); every fork was LR-MOVED (8.372e-05 against 1.0047e-04, a dose rate
     of 0.83×); it played no Kakuna game; its REPORTED fresh-run-floor interval, [−0.65, +4.58], spans zero.
     0.0125 is a base choice here, not a result of this experiment, which cannot test it (§16).
   - **Not re-measured at 0.0125:** §11's seed floor (1.76 pp) and §12's narrow-arm cost, both measured at 0.05.
     UNVERIFIED there; no bar moves for it.
   - **How it is carried:** every fresh argv types `--ent-coef 0.0125` (the kit's `BASE`); a resume inherits it;
     the look and the early check refuse a run whose metadata, recorded arguments or checkpoint sidecar say
     otherwise. `designs/production_config.json` stays at 0.05 (a kit test pins that).
2. **`--behaviour-check warn` in every run, and K9(b) judged by the look** (the review's F1; §2, §10, §14). The
   trainer warns and continues; `main.b1.look.k9b_read` judges by X35 AM-6's rule at the early check, at every
   look through its checkpoint's step, and at each run's end on every row. A VOID is "lost to its lever"; the look
   that refuses is not played and goes to the owner; a look already read stands; an INCONCLUSIVE is repaired or
   treated as VOID. §10 holds the full rule.
3. **Launch shells at nice ≤ 10** (F6; §14, §13 item 2, §2): the launcher's inherited-nice gate refuses a GPU
   launch typed from a shell above nice 10, so an agent's nice-19 shell cannot launch a run and its CPU dry runs
   show two refusals, not one.
4. **§13 item 4's cost** (F4): 1.2–2.6 GPU-h for the two pilots, plus each one's cold start, not "≈ 1 h".
5. **§10's HOLDs are closed**: X60-T read NO-TRANSFER (final, `75abf79d8`); `ent_coef` is decided (item 1).
6. **The memory fit at width 256** (§13 item 4(d), §17 item 37). All figures are MiB on the training card.
   - **The finding** (`aee363bc9`, the de-risk's wide pilot): the startup fit check refused the run
     (`UpdateWontFit`): demand 10,768.0 of a 10,770.5 card, headroom 2.5 against the 1,024 bar. The binding term
     was the first update's gradient probes (peak allocated 9,053 at the grad-balance probes and 9,106 at the
     per-term noise probe, against 7,139 at the forward), then the mid-run canary's fp32 reference (8,703). No
     argv lever that leaves the numbers identical cleared the bar.
   - **The fix** (`e112cbcbf`, trainer code, every run): the probes take their gradients from their own re-run of
     R1 on the same rows and parameters instead of holding the update's graph, and the canary's fp32 reference
     keeps 2,048 MiB of saved activations on the card and holds the rest in host memory. Training numbers are
     bit-identical in every arm; the probe values equal the old ones to the bit on one CPU thread.
   - **Read on the card at `e112cbcbf`** (the de-risk's validation starts, pre-amendment argvs): width 256 demand
     9,452.0, peak allocated 7,210.6, headroom **1,318.5** (294.5 above the bar); width 128 demand 7,298.0,
     headroom 3,472.5. Evidence: `~/.cache/gen3ai/b1_derisk_2026-11-11/results/` (`update_fit_w.json`,
     `probeval/`).
   - **Follow-ups the pin also holds:** `1fe7283f7` (the per-term noise estimator's state is always written, item
     7); `ef89ca99f` (the reference offload's ledger; its CUDA cell passed on the card); `13c50a5e6` (the dry
     update of the fit check is given the collector-only columns a run trains with, such as the magnet's);
     `1d2aa1695` (the production forward + backward compiles at B = 2 again on Triton 3.4.0; its card run is §13
     item 3's); `90699348f` (test-only: two CPU slow tests that pinned the update's compiled-dispatch and
     host-read counts were red on main since `e112cbcbf`, because the probes' re-runs of R1 are counted
     dispatches; they now count the training micro-batches and the re-runs separately, and no production code
     moved).
   - **`13c50a5e6` and this experiment:** no B1 run types `--magnet-kl-coef`, so the fix supplies nothing to a B1
     dry update. It is still on every B1 launch's path: the fit check's fixture buffer no longer inherits optional
     columns and its inputs go through the new table. Its logic passed on the card in a magnet run at the CE-3d
     pin (`9e97cad57`, a different base); main's commit has NOT run on the card. §13 item 4(d) at the launch pin
     is that run.
   - **What is NOT settled:** the margin at 256 is 294.5 above the bar on one reading; the card's free memory at
     a start varies with what else holds it (10,770.5 at the de-risk, 10,742.5 at the CE-3d start of 11-12). A fit
     refusal at a restart of a running wide run is not classified by this amendment (§17 item 37); Amendment 3
     classifies it, and at the launch pin the margin read 374.5 (slot S5). Nothing in the
     probe path is caught any more: an out-of-memory error in the dry update is `UpdateWontFit`, and one mid-run
     stops the run.
7. **The per-term noise keys may be absent, at either width, by their own gate.** The estimator's state
   (`train/noise_scale_g2_<g>`, `train/noise_scale_trsigma_<g>`, `train/noise_per_term_groups`) is always
   written; the derived value and ratio keys of a term are written only where that term's squared gradient is at
   least 1e-12. A fresh run's policy is uniform, so the entropy term's keys are absent at first at 128 and at 256
   alike. Nothing BOUNDARY-1 reads uses any of them, at any look, so no run types the switch that turns the probe
   off and no measurement exists in one arm only.
8. **No skip rule: every GPU item of §13 runs at the launch pin, in full** (F19 / F20; §13, §17 item 34). Trainer
   code changed after `aee363bc9` and after `e112cbcbf`, and the de-risk typed pre-amendment argvs, so none of its
   records satisfies an item. The code-equality rule the review proposed is not registered and not invoked.
9. **Costs the budget does not yet hold** (§12): a start cost 938 to 1,011 s on the de-risk's four starts, at both
   widths, each on a cold shared compile cache; what a warm start costs is UNMEASURED. The probes' own forwards
   add about 1 % of the learner's wall on a diagnostics update, an estimate, UNMEASURED.
10. **What only the card can fill** is listed under "Pre-launch and launch records" below (slots S1 to S8).
    Nothing launches until S1 to S7 are written there.

**Code.** The kit's change (the base, `k9b_read`, `check-run --arm-end`, the twelve-model spec without a pin) is
on main at `35362d348`, the commit before this amendment's. The memory-fit unit and its follow-ups are the commits
item 6 names. **The launch pin is not in this commit**: it is a main commit that holds this amendment, the kit and every
fix above, and the re-pin record that names it is a later commit (§17 item 36).

### Amendment 3 (2026-11-12): a startup fit refusal is a start failure

Written before any launch, any game and any read, on the orchestrator's decision after §13's GPU items ran at the
launch pin (slots S1 to S7 below). **Blind: no `rb_b1*` run, checkpoint, game or Kakuna row exists.** The two
pilots it draws on ran in a scratch models dir on pilot seeds, stopped just past 4M steps, and are read for S1,
S5 and S6 alone. **It moves no bar**: the +4 pp bar, the −2 pp harm stop, the 1.76 pp floor, the 90 % level, the
0.005 pp band, the views, the classes and their order, the net rule, the panels, the pairs a cell, the seeds, the
Kakuna rule, the K9(b) ceilings and §10's two replacement causes are as they were. It classifies one operational
failure that Amendment 2 left open (its item 6, §17 item 37) and reports one fact (item 7). The launch pin does
not move: this amendment is a `designs/` commit after it, like the re-pin record.

1. **Why now.** At the pin the fit check read, on two clean starts (no other compute app on the card, no display
   process; this process could reach 10,770.5 MiB both times): width 256 demand 9,372.0, headroom **1,398.5**,
   which is **374.5 above the 1,024 bar**; width 128 headroom 3,710.5 (slot S5). So anything that holds more than
   374.5 MiB of the card when a wide run starts gets that start refused. The registration did not say what such a
   refusal means once a campaign is running.
2. **The class.** `UpdateWontFit` at ANY start of a BOUNDARY-1 run (the first launch, a six-hour restart, the
   phase-2 resume; an out-of-memory error inside the dry update is the same refusal) is a **START FAILURE**. The
   check runs before that process's first rollout and before its first save, so the process trains no step and
   writes no checkpoint and no `metadata.json` or `model_config.json` (the run's recorded arguments, lineage and
   pin history are written at a save). It is therefore not a FATAL in §10's
   sense, not a VOID, not "lost to its lever" and not one of §10's two replacement causes: it spends neither seed
   5109 nor the parity bullet's one resume. An out-of-memory failure AFTER a segment's first training step keeps
   §10's class ("lost to its lever").
3. **What the launcher does (read from the code at the pin; main's launcher differs from the pin's by nothing
   under `src/` today).** The child exits 3 (`FATAL_CONFIG`). The launcher saves the child's output under
   `<run>/crashes/`, logs "Fatal config error — will NOT restart" and itself exits 3. **It does not retry**, at a
   first launch or at a restart: after a refusal the run has no launcher and no child, and it waits for the
   operator. The refusing process also overwrites `<run>/update_fit.json` and `<run>/cuda_ledger.json` with its
   own readings and appends to the run's child logs.
4. **What the operator records, before anything else**, into §18's launch records: `nvidia-smi` (the card's
   total and free memory and every compute app on it), whether a display process holds the card, a COPY of the
   refusing `update_fit.json` (the retry overwrites it), the crash log's name, and the checkpoint the refused
   start was given. The operator then removes only card holders it started itself, never another's process
   (one lease on the card at a time means there should be none).
5. **The retry: the SAME segment, from the same checkpoint, on the same pin and recipe.** Which command that is
   depends on where the run stood:
   - **(a) The first start of a fresh run** (the run dir holds no checkpoint and no training step). The failed
     run dir is MOVED ASIDE to `models/<run>.FITCHECK_REFUSED_<n>` (n = 1, 2, …), never deleted (CE-3d's
     precedent: `models/rb_x35_ce3dswap_s7321.FITCHECK_NO_MAGNET_COLUMN`), and the run's identical registered
     argv is launched again. The counted run's dir, child logs and `crashes/` then hold only its own processes,
     which is what the K9(b) read's "every generation of the child's full log" reads. By the code the launcher
     refuses a fresh launch only into a dir that holds a checkpoint or `model_config.json`, which this one does
     not; the move is the rule all the same.
   - **(b) The phase-2 resume's own first start** (phase 2 has trained nothing). The identical registered resume
     command again (`python -m main.b1.arms`'s, from `models/<run>/final_model.zip` to 100M). Nothing is moved.
   - **(c) A six-hour restart, or any other start after the segment has trained.** The launcher's own restart
     resumes from the checkpoint `<run>/latest.txt` names, in the same run dir (after a restart's graceful stop
     that is the stopped child's own save, `final_model_interrupted.zip`). It has
     exited, and the fresh argv cannot be typed again (the launcher refuses a fresh launch into a dir with
     progress), so the operator types the same-run resume from that same checkpoint: `python -m main.launcher
     --restart-interval-hours 6 --pin-commit <PIN> --model models/<run>/<the path latest.txt names> --steps
     <50000000 in phase 1, 100000000 in phase 2> --device cuda --log-level periodic`, with no `--run-name` (a run
     name beside `--model` is a fork) and no lever or recipe flag (a resume inherits them). **Never
     `final_model.zip` in phase 2 once phase 2 has trained**: until phase 2 ends that file is the 50M final, and
     resuming from it rewinds the run. The kit builds and checks only form (b); form (c) is the same command with
     another `--model` (and, in phase 1, another `--steps`), so its `--dry-run` must pass before it is typed, and
     the command goes into §18. Nothing is moved.
   - **What the look then sees.** The run still carries one pin entry from step 0 and its immutable registered
     `original_command`, so it counts (§10) if everything else holds. After form (c) in PHASE 1 the run's last
     process is a typed resume where the other runs' is a restart child, so §10's ten resume-rewritten keys are
     not compared for it at the 50M look; the rollout shape still is (`metadata.json`'s `batch_size`, `n_steps`).
6. **A repeat on an empty card is a build gap.** If the retry is refused again with no other compute app on the
   card and no display process holding it, that run HOLDs and goes to the orchestrator. A code fix would put the
   run on a second pin, which the look refuses (one pin entry from step 0, §10), so the orchestrator rules before
   any change and records the ruling as a further amendment, before the look it touches.
7. **Reported, no rule moves: when a fresh run seeds its pool at this base.** Both pilots seeded the self-play
   pool at step 4,000,000, their SECOND evaluation (bots 0.7125 at 256, 0.725 at 128, against the 0.55
   threshold); the first, at 2M, read 0.4437 and 0.4962. That is the first reading of the seed's timing for a
   fresh run at `ent_coef` 0.0125, one run a width.

### Amendment 4 (2026-11-12): the early check and every look are read from the kit at `1d592e6cd`; every run still trains at the launch pin

Written after [1]'s first launch attempt and before any look, on the orchestrator's decision. **Blind: no look has
been planned or played, no head-to-head game and no Kakuna row exists.** What has been read is what the early
check reads by design, from that one attempt: its first checkpoint's sidecar, its `metadata.json`, and its K9(b)
rows through step 1,000,192. **It moves no bar and no class**: the +4 pp bar, the −2 pp harm stop, the 1.76 pp
floor, the 90 % level, the 0.005 pp band, the views, the classes and their order, the net rule, the panels, the
pairs a cell, the seeds, the run names, the argvs, the Kakuna rule, the K9(b) ceilings, §10's guards as worded and
its two replacement causes are as they were. The launch pin does not move.

1. **What happened.** [1] `rb_b1c_s5101` (C, seed 5101) launched 2026-11-12 17:18:18 PST at the launch pin with
   its registered argv (main at `60efef08c`, the GO at `63745f628`). At its first checkpoint
   (`checkpoint_1000192_steps.zip`) §10's early check, run from a clean checkout of the pin, exited 3 with one
   problem stated twice, for the run and for the checkpoint: "its pin entry ends at 0, before the checkpoint's
   1,000,192". Every other guard passed and K9(b) read PASS (9 rows, maximum 4.8e-07). As §10 orders, the launch
   driver stopped the run (SIGTERM to its launcher, 17:32:34; its last logged step 1,086,720) and held for the
   orchestrator.
2. **Why it is the kit and not the run.** A pin entry's `last_step` is the step of the run's last
   `metadata.json` save at that commit (`agents.model.snapshot._update_pin_history`): a process's save at its
   START, and the save that CLOSES a segment (the final save, or the quit-save of a restart or a stop). A periodic
   checkpoint advances nothing; its sidecar copies the history as it stood. So in a healthy run `last_step` is the
   step the open segment STARTED at: 0 throughout a first segment, and below the checkpoint's own step in the
   sidecar of every periodic checkpoint a run ever writes. The pin's kit (`look._one_pin`) required `last_step`
   to reach the checkpoint in both records. No look checkpoint's sidecar can satisfy that, so the pin's kit would
   have refused all twelve runs at the early check and again at every look. Its tests passed on a fixture whose
   sidecars carried a `last_step` no run writes. The attempt's own records were what §10 asks for: one entry, the
   launch pin, from step 0, in the run and in the checkpoint.
3. **The rule is §10's, unchanged; the code that reads it is fixed** (`main.b1.look.pin_problems`, commit
   `1d592e6cd5ff2b6b9f634e67573a98e3cdaede44`). The run's `metadata.json` and the look checkpoint's sidecar must each
   carry exactly ONE pin entry, at the launch pin, with `first_step` 0; the sidecar's own `git_hash` (the commit
   of the process that wrote those bytes) must be the launch pin; and the run's entry must either reach the
   checkpoint (`last_step` ≥ its step) or be the very entry the checkpoint was written under (the same
   `last_step` as the sidecar's: the run has saved nothing since, so the checkpoint lies in its open segment).
   Still refused: a second entry, another commit, a `first_step` other than 0, and a run whose entry was closed
   below the checkpoint after the checkpoint was written. The regression tests carry the shapes real runs write
   (`rb_b1c_s5101`'s first checkpoint; `rb_x26_s2601`'s checkpoints before and after a restart) and fail on the
   pin's code. Run on the stopped attempt's records from a clean checkout of the kit commit, the check exits 0
   with no problem and K9(b) PASS.
4. **What is read from where, from now on.** The early check, the arm-end check (`check-run`, `--arm-end`) and
   each look's plan, read and team bootstrap (`main.b1.look`, `main.b1.team_bootstrap`, `looks/drive_look.sh`)
   are run from a clean detached checkout of the KIT commit `1d592e6cd`. Every run TRAINS at the launch pin
   (`--pin-commit 15136b616…`, each argv token for token as registered), every look's games are played by
   `main.h2h` from the pin checkout, the Kakuna read plays at the pin, and `--pin <PIN>` in every kit command is
   still the launch pin. The kit commit differs from the pin under `src/` and `data/` in two files,
   `src/main/b1/look.py` and `src/main/b1/look_test.py`; so the parser that resolves each arm's levers, the K9(b)
   reader's rule (`main.x35.batch`), the ledger and everything a game runs are the pin's own bytes.
   `drive_look.sh` now takes both checkouts and exits 5 unless the kit checkout is clean at the kit commit and
   differs from the pin under `src/` and `data/` only inside `src/main/b1/`. The text this replaces: §3's first
   bullet, §9's "planned and read by code", §10's early-check bullet, §14's three operator blocks, §17 item 44.
   Executed 2026-11-12 on CPU as far as the plan: both checkouts verified, `main.b1.look` imported from the kit
   and `main.h2h` from the pin, and the 50M plan REFUSED as it must (no run has a 50M checkpoint), with no family
   registered and no game played. **UNVERIFIED by execution: the play and read steps of the two-checkout form.**
5. **[1]'s first attempt is a kit-caused stop.** It is not a VOID, not a FATAL of §10, not a start failure
   (Amendment 3) and not a replacement cause: it spends neither Amendment 1's one same-pin resume nor seed 5109.
   The attempt is discarded whole, and no look, early check or Kakuna read reads anything it wrote. Its
   directory is moved aside to `~/.cache/gen3ai/b1_aside/rb_b1c_s5101.EARLYCHECK_KIT_1`, and [1] is relaunched
   from step 0 under its registered seed 5101, name `rb_b1c_s5101` and argv (the Training session's actions on
   the orchestrator's GO, after the kit commit lands), then early-checked from the kit commit. The folder goes
   OUTSIDE `models/` and `~/gen3ai_archive` because it still records seed 5101: under either root `python -m
   main.b1.arms check` would list it as two collision hits (its sidecar's seed, its name), and there
   the scan does not see it.
6. **Recorded, classified by nothing here: the stopped child's FATAL.** On the driver's SIGTERM the child logged
   `[CompileSentinel] FATAL at update end: dynamo RECOMPILED after the compile lock` and exited without a
   quit-save (no `final_model_interrupted.zip`; `latest.txt` still names `checkpoint_1000192`). The stop signal
   came first and the FATAL followed it within the same second; the run had been stopped by the early check
   already. What a FATAL of that kind means for a COUNTED run that is signalled is the orchestrator's ruling, in
   a further amendment if it changes anything. The registered flow sends a child no signal: the 6 h restart is
   the child's own, at a rollout boundary, and the 50M and 100M stops are completions.

### Pre-launch and launch records

Written here, each before the launch or look it gates: the re-pin; §13's
results; ρ, §12 rewritten from it, and the EQ step; which block launched (phased or fallback); for EVERY launch its
date, run, `git rev-parse HEAD` of the main checkout, `python -m utils.core_split show`, and the early check's
JSON once the run has its first checkpoint; every FATAL, resume and replacement with its cause; every K9(b) record
(Amendment 2); every start failure and its retry (Amendment 3).

**The re-pin record (2026-11-12; §18's single re-pin, steps 1 to 6).** Written by the agent the orchestrator
named, on CPU (no GPU lease), in the commit after the launch pin. It launches nothing and fills none of the slots
below.

1. **The launch pin is `15136b61634b667bb9da5ccc6647ff9265238829`**: main's commit of Amendment 2 (`prereg(b1):
   Amendment 2 …`). Its parent is the
   kit's commit `35362d348`; it also holds the memory-fit unit and its follow-ups (`e112cbcbf`, `1fe7283f7`,
   `ef89ca99f`, `13c50a5e6`, `1d2aa1695`) and the test-only `90699348f`. This record's commit differs from the pin
   only under `designs/` (the records, the spec's `pin`, this file, STATUS and the backlog).
2. **What the pin adds since the registered `507d0c119`**: 42 commits, 363 files, 153 of them under `src/`.
   - **Trainer, model or launcher code, on every run's path:** `e112cbcbf`, `1fe7283f7`, `ef89ca99f`, `13c50a5e6`
     (the memory fit, Amendment 2 item 6); `1d2aa1695` (the damage op reads OUR spread off the real context, so
     the CUDA forward + backward compiles at B = 2 again; every number bit-identical; no `ARCH_SIGNATURE`, config
     or floor change); `6e4a0300e` and `d4b0576ea` (the launcher's inherited-nice gate, Amendment 2 item 3).
   - **Trainer or model code behind a switch no BOUNDARY-1 run types** (each off by default, and each dry run
     says so): `251bbfb87` (the trailing magnet, `--magnet-kl-coef`, default 0.0; it adds the recipe row
     `magnet_kl_coef: 0.0` to `designs/production_config.json` and moves `MODEL_CONFIG_VERSION` 169 → 170);
     `3c7f3bedd` (`--belief-grad-mode move_label_only`); `d0e2715d8` (the human-replay clone's two uses behind the
     human-data lever; the dry runs print `human data : off`).
   - **Tests and goldens only:** `55334e88a` (two flag-off numerics goldens re-recorded behind config v165 to
     v168; its one head-to-head file is a test), `90699348f` (Amendment 2 item 6).
   - **The kit:** `c92a5e290` (three seeds a cell), `35362d348` (the base, the K9(b) reads).
   - **Not touched by any of the 42:** the observation encoder, the inference service, the Rust core, the
     head-to-head play code (`src/agents/observation`, `src/agents/inference`, `src/rust_sim`, `src/main/h2h`).
     `ARCH_SIGNATURE` (`gen3_mon_tied_gain_v1`) and the migration floor (145) are unchanged.
   - **The rest** are ops and gate fixes (`4038f8915`, `7c25230ba`, `b27c78383`, `5eabc4a79`, `218d093c0`,
     `f85f7f3dd`), a tools meter (`811c08ddc`), the X35 kit's CE-3d arms (`08b998d12`, `71c20b521`, `e9f55e723`)
     and registrations, verdicts and docs (X66, X67, X69, X60-T, CE-3d, R9's registry row, this file).
   - **`checkargs` and the dry runs at the pin resolve `production_config@8a9be0753ba9`** (the registered pin
     resolved `@51496e4209fc`; the whole difference is the `magnet_kl_coef: 0.0` row). Per argv: 1 of 32 recipe
     knobs differs from `recipe.fresh`, `ent_coef` 0.0125, TYPED, in all four arms; 2 of 42 ARCH-surface keys
     differ in W and WB (`d_model` 256, `trunk_ffn_dim` 512) and none in C and Br; `K9(b) check : warn (typed)`;
     0 unrecognised flags.
3. **The records, regenerated from a clean detached checkout of the pin** (`git worktree add --detach`,
   bootstrapped, `git status` empty): [`launch_block.txt`](launch_block.txt) (`python -m main.b1.arms describe
   --pin 15136b61634b667bb9da5ccc6647ff9265238829`: twelve fresh launches, then twelve resumes, each with its
   commented dry-run form; its second line
   names the pin), and the eight-run block of the registered pin is deleted; `argv_`, `checkargs_` and `dryrun_`
   for `rb_b1c_s5101`, `rb_b1w_s5103`, `rb_b1br_s5105`, `rb_b1wb_s5107`; `read_spec_kakuna.json` now carries
   `"pin"` (spec sha256 `00269676…613b6a`), its twelve models and the model list `kakuna_rule_test.py` pins
   unchanged.
4. **The checks at the pin** (the same checkout, CPU):
   - `python -m main.b1.arms check --tree <checkout> --pin 15136b61634b667bb9da5ccc6647ff9265238829 --params`: `CHECK
     PASSED`. (a) 664
     `metadata.json` files and 4,017 directory names under the main checkout's `models/` and `~/gen3ai_archive`:
     no seed of 5101–5113 and none of the sixteen run names is taken; (b) the corpus `n1_proc_v1@ALL` verifies
     (data sha256 `99e3425a…0bad6`, 12,000 teams); (c) the control resolves the base and each arm differs from it
     by its lever alone, at 50M and at 100M; (d) parameters 2,609,183 (C, Br) and 6,258,079 (W, WB).
   - `python -m main.b1.resume_check static --tree <checkout> --pin 15136b61634b667bb9da5ccc6647ff9265238829`: `PASS:
     no difference`.
   - `pytest src/main/b1` on this record's tree: 266 passed, 0 skipped (no record-bound test skips). The kit's slow
     tier ran
     at `35362d348`, whose `src/` is the pin's: 5 passed (the cross-width pairing on CPU, the parameter counts).
5. **§13 items 1, 2, 6 and 7 at the pin, on CPU:**
   - **Item 1**: the kit's check, above. It is run again immediately before the first launch (§3).
   - **Item 2, the CPU half**: the four first-seed dry runs, from a nice-19 shell with the card hidden, exit 3
     with exactly two refusals each, `cuda devices` and `inherited nice`; every other preflight passes (the pin
     resolves, the run dir would be created, disk 165.9 GiB free against 14.45 GiB required, no display process
     on the card, the Showdown validator present). The card-visible half is slot S7.
   - **Item 6**: the static check above, and the two-segment smoke REPEATED at the pin
     ([`resume_check.log`](resume_check.log) is its transcript): arm WB, seed 5107, half = 4,000 steps, CPU
     `--debug`, the base typed. `PASS: no difference`; 68 checks ok and 7 notes, every segment exited 0, on the first
     run at this pin (1,299 s, peak 9.49 GB of a 16 GB cap). Phase 1 stopped at 4,346 and phase 2 resumed at that
     count, ending at 8,510 against the run-through's 8,203 (an update boundary is a count of completed games' rows);
     the optimizer (280 entries) and 809 policy tensors are tensor-equal across the stop; both runs record `ent_coef`
     0.0125, one matchup hash and one code version, `15136b61634b`. Its coverage is §4's: one arm, the CPU, a
     debug rollout shape. The phased block stays the registered one.
   - **Item 7**: `python -m main.reads check-spec --spec <this folder>/read_spec_kakuna.json`, with the spec's
     `pin` filled in and read on a checkout whose HEAD is the pin, reports the pin `15136b616…`, `is_head` true and
     `play_path_differs_at_head` 0. (The spec committed AT the pin carries no pin, so there is nothing to read
     in the pin's own tree.) This record's commit changes files under `designs/` only, none of them on the play
     path, and the read is repeated on it before it is pushed.
   - **Also read at the pin, though no item asks it**: the three CPU slow tests of
     `src/agents/model/update_performance_shape_test.py` that `90699348f` repaired pass (127 s; their CUDA cell is
     one of §13 item 3's).
6. **The pin is written** into this file's header, §3 and §14, `designs/STATUS.md` and the backlog's MF-5 row.

**Not done by the re-pin, and not claimable from it:** anything on the card. §13 items 3, 4 and 5 and the
card-visible half of item 2 are slots S1 to S7 below; the runbook is
`~/.cache/gen3ai/handoffs/b1_pin_runbook_2026-11-12.md` (≈ 3.6 GPU-h, 3.0 to 4.4). `13c50a5e6` and `1d2aa1695` as
main's commits had not run on the card when the re-pin was written (Amendment 2 item 6); the pilots' startup fit
check and item 3's compile cell were their first card runs, and both passed (slots S4 and S5 below).

**Slots only the card can fill (all at the launch pin). S1 to S7 were filled on 2026-11-12, every one green; their
records follow the table. S8 is open until the launch.**

| slot | what | from |
|---|---|---|
| S1 | ρ (`rho_pre.rho_total`: wall-clock per step at 256 over 128, steady updates after the compile), with its provenance and regime | §13 item 4's two pilots |
| S2 | §12's table rewritten from ρ, with starts priced separately (cold, and warm once read) | S1 |
| S3 | the EQ step, 1,000,000 × ⌊100 / ρ⌋ (§7) | S1 |
| S4 | §13 item 3's record: the CUDA slow tests, each PASS / FAIL | the slow-gpu window |
| S5 | §13 item 4 (b) to (e)'s records, with the fit check's demand and headroom at 256 and at 128 | the pilots |
| S6 | §13 item 5's record: the cross-width cell | one played cell |
| S7 | §13 item 2 with the card visible, from a shell at nice ≤ 10: four dry runs, every preflight green | the leased agent |
| S8 | §13 items 8 and 9; the per-launch records; the warm start's cost; each run's realised GPU-hours; every start failure and its retry (Amendment 3); the twelve arm-end K9(b) records | at and after launch |

**The slot records (2026-11-12; S1 to S7 at the launch pin, on the card).** Run by the Training session under one
GPU lease, 04:54 to 08:51 PST (≈ 3.95 h; the runbook planned ≈ 3.6), from a clean detached checkout of
`15136b61634b667bb9da5ccc6647ff9265238829`, with CE-3d's chain paused between its first and second fork. Every
item passed and nothing was skipped. Evidence: `~/.cache/gen3ai/b1_pin_2026-11-12/` (`results/`, `logs/`,
`slow_gpu_out/`; a scratch root, not in the repository). The pilots are `rb_b1pinpilot_w_s9203` (W's argv) and
`rb_b1pinpilot_c_s9201` (C's), each the kit's launch argv with only the seed and the run name changed, into a
scratch models dir; neither is a BOUNDARY-1 run and neither is read at any look.

- **S1, ρ = 1.392** (`rho.json`: `rho_pre.rho_total` 1.39214). Seconds per 1M steps over the steady updates:
  **734.59 at width 256** (37 rows, steps 300,288 to 3,936,256; median update 60.92 s, median collection 10.09 s,
  the update 86.0 % of the wall) against **527.67 at width 128** (37 rows, steps 300,800 to 3,937,792; 42.01 s
  and 8.93 s, 82.7 %). By part: `rho_update` 1.450, `rho_collect` 1.126. Regime: `ent_coef` 0.0125, the pool
  EMPTY (every row is before the seed at 4M), the launcher at its own nice 10, CE-3d not training. `rho_post`
  (after the seed) is null: the pilots stopped within five rows of it. Over all rows, evaluations and checkpoints
  included, the walls are 757.62 and 547.58 s per 1M and their ratio 1.384. **What ρ does not say:** the ratio
  late in training or with a full pool (UNVERIFIED).
- **S2, §12 rewritten** (above). The arithmetic:
  - Narrow, from `rb_x26_s2601`: 9.48 GPU-h to 50M, 19.55 to 100M, so 10.07 for phase 2. Wide = narrow × 1.39214:
    13.20, 27.21, 14.02.
  - Phase 1: 6 × 9.48 + 6 × 13.20 = 56.9 + 79.2 = **136.1**. Phase 2: 6 × 10.07 + 6 × 14.02 = 60.4 + 84.1 =
    **144.5**. Training: **280.6 GPU-h** (it was ≈ 311 at the guessed 1.65).
  - The interval's low end, at the pilots' own rates: 527.67 s × 100 = 14.66 GPU-h per narrow run, 734.59 s × 100
    = 20.41 per wide run, 6 × 14.66 + 6 × 20.41 = **210.4**. It is a floor (pool empty, the first 4M steps,
    evaluation and checkpoint time excluded); the central figure is X26's. The registered interval was ρ's guessed
    range (270–340); with ρ measured, what remains uncertain is the narrow hours, and these two readings bound it
    from below only.
  - Starts: a child lives six hours from its start. Narrow: 2 starts in phase 1 (9.48 h) and 2 in phase 2
    (10.07 h); wide: 3 and 3 (13.20 h, 14.02 h; two six-hour children train ≈ 11.3 h between them once each has
    spent ≈ 0.33 h starting). 6 × 4 + 6 × 6 = **60 starts**. Cold, launch to the last startup line before the
    first rollout: 1,195 s (W, 06:11:52 to 06:31:47) and 1,192 s (C, 07:27:02 to 07:46:54), 0.33 h, of which
    `T2 up in 886.1s` and `891.0s`, the prewarm 244.7 and 244.4 s, the fit check's dry update 30.8 and 21.5 s.
    60 × 0.33 = **≈ 20 GPU-h** if every start is cold (60 × 0.247 = 14.8 for the service's part alone). Both
    pilots' starts were cold (a new shared compile-cache entry each). **The warm figure is OPEN: read off the
    first run's first restart (S8).**
  - To the decision: 280.6 + ≈ 20 + ≈ 8 (four looks) = **≈ 309 GPU-h**; (280.6 + 20) / 24 = 12.5 days of
    training on the one card, the looks ≈ 0.3 day, Kakuna ≈ 2.5 days: **≈ L + 15.4 days**. At the low end (48
    starts, since a wide run then takes 2 a phase): 210.4 + 15.9 + 8 = ≈ 234 GPU-h, ≈ L + 12.3 days.
  - After a WIN: 56.5 − 19.55 = 36.95 GPU-h a narrow run from 100M to X26's plateau step; × 3 = 110.9, and
    × 1.39214 = 154.3 in a wide cell (it was 183 at 1.65). A fourth seed an arm: 2 × 19.55 + 2 × 27.21 = 93.5
    (it was ≈ 104).
- **S3, the EQ step is 71,000,000**: 1,000,000 × ⌊100 / 1.39214⌋ = 1,000,000 × ⌊71.83⌋, below §7's 99M cap. The
  look takes `--eq-step 71000000` and its requests are named `b1_eq71m_<panel>` (§8's printed example,
  `b1_eq61m_<panel>`, was the name at the guessed ρ; the code builds the name from the step). The floor's next
  step is close: 72M needs ρ ≤ 1.3889, and the all-rows ratio (1.384) would give it. The registered statistic is
  `rho_pre.rho_total`, so 71M stands; the EQ look is REPORTED and carries no class.
- **S4, the CUDA-only slow tests: 32 of 32 PASS at the pin** (`python -m main.ops.gpu_fill_window`, the
  `slow-gpu` window; one commit on every row, the pin; no NOT-PASS row, no skip; the tests' own time 71.8 min, the
  window 76 min, its supervisor exit 0 at 06:11:20 PST). Among them the two cells that had failed before:
  `extractor_compiles_test.py::test_production_arch_compiles_forward_and_backward_on_cuda` (the production
  forward + backward compile at B = 2 on Triton 3.4.0, 163 s) and the reference-offload cell
  (`saved_offload_test.py::test_on_the_card_the_bounded_reference_arm_matches_the_unbounded_one_and_peaks_lower`).
  **So `1d2aa1695`'s fix is now VERIFIED on the card at main's code**; a separate confirmation the same morning
  (`~/.cache/gen3ai/triton_fwdbwd_2026-11-11/summary.txt`, stamped 04:44 PST; torch 2.8.0+cu126, Triton 3.4.0)
  read the commit before the fix failing on the same kernel and the fix passing (forward + backward at B = 2,
  inference at B = 8).
- **S5, the pilots, §13 item 4 (b) to (e): all PASS at both widths.**
  - **Events, the same at 256 and at 128:** first checkpoint at 1,000,192; checkpoints at 1,000,192, 2,000,128,
    3,000,064 and 4,000,000; first evaluation at 2,000,128 (bots 0.4437 W, 0.4962 C, no seed); second at
    4,000,000 (0.7125 W, 0.725 C) and **the pool seeded there** (`snapshot_000004000000.zip`; the self-play
    share then 65 % W, 71 % C). §13 item 4's "first promotion into the self-play pool" was read as this seed;
    neither pilot ran to a later promotion. Rows logged after the seed: 4 (W), 5 (C). Last logged step 4,330,926
    (W, 44 rows) and 4,429,200 (C, 45 rows). W ran 06:11:52 to 07:26:34, C 07:27:02 to 08:27:26; each launcher
    made no restart and attached one child; no crash log, no bad line.
  - **(b)** the bf16 compile at 256 completed and the startup parity check passed (both widths; the prewarm took
    244.7 s at 256, 244.4 s at 128). **(c)** the inference CUDA-graph capture succeeded (`T2 up … backend graph
    on cuda`). **(e)** no parity gate FATAL through the seed and after it; the mid-run canary passed at update 10
    (W at step 988,416, C at 988,672) and the memory trend read OK at update 44 in both.
  - **(d) the memory fit** (`update_fit.json`, schema `gen3_update_fit_v2`, 98,304 rows, MiB; before each launch
    `nvidia-smi` read 33 MiB used of 12,288 with no compute app and no display process):

    | width | demand (peak reserved) | peak allocated | headroom | reachable card | above the 1,024 bar |
    |---|---|---|---|---|---|
    | 256 (W) | 9,372.0 | 7,210.6 | **1,398.5** | 10,770.5 | **374.5** |
    | 128 (C) | 7,060.0 | 4,930.5 | 3,710.5 | 10,770.5 | 2,686.5 |

    At `e112cbcbf` width 256 read 9,452.0 and 1,318.5. **The margin at 256 is thin**: more than 374.5 MiB held
    by anything else at a wide run's start refuses that start. Amendment 3 says what that is and what follows.
    `13c50a5e6`'s fit-check path ran here on the card at main's code for the first time, in both pilots.
  - **K9(b)** (`main.b1.look.k9b_read`, ceilings 1e-3 on the maximum and 1e-4 on the 99th percentile): W PASS
    through 4,330,926 (44 rows, maximum 1.10e-05, 99th percentile 4.5e-06), C PASS through 4,429,200 (45 rows,
    1.20e-05, 5.5e-06); no behaviour-policy line in either child log. **A PASS inside 4M steps says nothing
    about 100M at `ent_coef` 0.0125** (X26 first crossed 1e-4 near 137M, at 0.05): the early check, each look
    and the arm-end read are the guard (§10).
- **S6, the cross-width cell on the card: PASS.** `main.h2h`, W's against C's `checkpoint_1000192_steps.zip`, one
  cell: "1 cell(s), 2 checkpoint(s) — 2 architecture(s)"; one engine, "64 envs, T2 graph on cuda; 2 slot
  group(s)" (up in 1,359.6 s); 500 mirrored pairs, 1,000 games, exit 0, the row valid under
  `gen3_eval_count_row_v2`. W scored 40.15 % [37.37, 42.93] (W/L/D 399/596/5); at 1M steps that is a pairing
  proof and no evidence about width.
- **S7, the four first-seed dry runs with the card visible: every preflight green.** `rb_b1c_s5101`,
  `rb_b1w_s5103`, `rb_b1br_s5105`, `rb_b1wb_s5107`, from a shell at nice 0 (`inherited nice : ✓ 0 (<= 10)`),
  main at `96ff6448f`: 0 refusals each and "DRY RUN — this command would launch"; the child's code by
  `--pin-commit`; `K9(b) check : warn (typed)`; `ent_coef 0.0125 (TYPED)` against production's 0.05; no display
  process on the card; disk 170.21 GiB free against 17.88 GiB required. **The disk figure is the same for all
  four arms and wrong for each**: the preflight sized a checkpoint at 54.7 MiB from another lineage's archive run
  with the same `ARCH_SIGNATURE`, which does not carry the width; a B1 checkpoint is 35.5 MiB at 128 and
  77.3 MiB at 256 (the pilots' files). No launch is at risk at 170 GiB free; the defect is filed in
  `designs/ops/TECH_DEBT_BACKLOG.md` (§2 (g)), and **what the twelve runs need in total is not computed by any
  gate** (UNVERIFIED).
- **S8: OPEN**, written at and after the launch: §13 items 8 and 9; each launch's record; the warm start's cost;
  each run's realised GPU-hours (§7; the first control run's hours to 50M check §12's central figure); every
  start failure and its retry (Amendment 3); the twelve arm-end K9(b) records.
