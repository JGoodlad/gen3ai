# BOUNDARY-1 — the from-scratch 2 × 2, width × team breadth: PRE-REGISTRATION (2026-11-09)

**Status: REGISTERED (tier FULL) before any launch, any game and any read. NOT LAUNCHED** (no `rb_b1*` run exists).
The launch is the orchestrator's GO through the Training session, after X66 and X67 and after §13's GPU checklist is
green. Nothing below changes after the first run launches except by a dated amendment (§18), written before the look
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
- **Breadth is decided on the in-house cross, not on Kakuna** (orchestrator ruling, 2026-11-09, "Option A"). The
  draft read breadth on Kakuna on U. At two seeds a cell Kakuna finds, 80 % of the time, only a gain of ≈ 4–6 pp
  on U and ≈ 5–7 pp on forum72 (§11.4), and the fork evidence for breadth there is +0.55 pp on F, so that rule
  could not have fired. Kakuna stays as the CONFIRMING meter for both
  factors. Twelve runs (three a cell) were declined: +104 GPU-h for a Kakuna rule that would still be unpowered.
- **The base is today's production**, which since the draft gained: per-run trunk width (`0220f8408`, config v169;
  production stays 128 and byte-identical), E13, their max HP as a belief (`3b1733f50`, v168), and the mechanics
  round 2 (`566d878b2`).
- **No critic or policy-tower change.** The architecture diagnosis (`a8265f36b`) found the tanh tower 0.897
  saturated but not dead, and the critic head's hidden layer unused; it recommends no change for this base. The
  draft's "critic is linear on 128" was wrong: the head has a hidden layer.
- **`ent_coef` stays 0.05 in every arm** (the owner's 2026-09-29 ruling). An X67 PASS changes it only with the
  owner's word, by an amendment that moves ALL eight runs, before launch.
- **Not in this experiment:** the MF-3 off-pool belief work, human teams (X60-T, §10's HOLD), E9, any search.
- **The team is the unit against Kakuna** (`96adb152c`, `UNDERSTANDING.md` §8 rule 10): §6's interval is clustered
  by team.

## 2. The arms

Every run is a FRESH `--arch production` launch: the architecture and the recipe both come from
`designs/production_config.json` (ten PPO epochs, `ent_coef` 0.05). An arm types its lever and nothing else.

| arm | kit key | what differs from the control | lever typed | learner parameters | runs (seed) |
|---|---|---|---|---|---|
| **C** control | `c` | nothing | — | 2,609,183 | `rb_b1c_s5101`, `rb_b1c_s5102` |
| **W** wide | `w` | the trunk at 256 with its feed-forward at 512 (production: 128 / 256) | `--d-model 256 --trunk-ffn-dim 512 --allow-nonproduction-arch` | 6,258,079 | `rb_b1w_s5103`, `rb_b1w_s5104` |
| **Br** breadth | `br` | on EACH side, every fresh team draw is an `n1_proc_v1` team with probability 0.5, else the usual pool draw | `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both` | 2,609,183 | `rb_b1br_s5105`, `rb_b1br_s5106` |
| **WB** wide + breadth | `wb` | both | both levers | 6,258,079 | `rb_b1wb_s5107`, `rb_b1wb_s5108` |

- **The two factors.** Width ON = {W, WB}, OFF = {C, Br}. Breadth ON = {Br, WB}, OFF = {C, W}.
- **The shared tail of every fresh argv:** `--restart-interval-hours 6 --pin-commit <PIN> --arch production [lever]
  --checkpoint-every-steps 1000000 --steps <target> --seed <seed> --device cuda --log-level periodic --run-name
  <run>`. The 1M checkpoint series is typed because the looks and the equal-compute read need it.
- **`--allow-nonproduction-arch`** is the consent the launcher demands for a width other than production's; it is
  part of the wide lever, not a second lever.
- **The corpus** is `n1_proc_v1`, tier `ALL`, 12,000 teams, data sha256 `99e3425a…0bad6`, as it verifies at the pin.
  Evaluation never draws from it: every look and the Kakuna read play the registered panels.
- **The argvs are code**, `src/main/b1/arms.py` (`python -m main.b1.arms describe`); the committed transcripts are
  what it prints at the pin: [`launch_block.txt`](launch_block.txt), `argv_<run>.txt`, `checkargs_<run>.txt`,
  `dryrun_<run>.txt` for the first seed of each arm. `python -m main.checkargs --argv` accepts each with 0
  unrecognised flags; each resolved config differs from the control's in its lever's keys alone (`d_model`,
  `trunk_ffn_dim`, `allow_nonproduction_arch`; `team_mix`, `team_mix_sides`), at both step targets. `python -m
  main.launcher --dry-run` on CPU refuses each for ONE reason, the CUDA preflight (`CUDA_VISIBLE_DEVICES=""` with
  `--device cuda`); every other preflight passes.
- **Parameter counts** are a CPU build of the four learners at the pin (`python -m main.b1.arms check --params`).

## 3. The pin, the seeds, and the checks before launch

- **ONE pin for everything: `507d0c11961da8d4243a4c8877bcc04ef220058c`** (main at registration, the commit that carries the kit, the cross
  statistic and the Kakuna rule). All eight runs train at it, every look plays at it, and the Kakuna read plays at
  it. It is `0220f8408` plus the nine commits main gained while this was written (`c4a842d12` to `c8c212214`: the light-screen
  tier's docs, the heavy slots admitted by memory, the core split and its owner, the off-pool belief meter with
  X68's records, three gate-reliability fixes; none touches a trainer, launcher, model or head-to-head file) and this unit's tooling;
  `designs/production_config.json` and `src/agents/model/model_version/` are unchanged between `0220f8408` and the
  pin, and `checkargs` at the pin resolves the same production surface (`production_config@51496e4209fc`).
- **The pin is an ARGUMENT of every command (`--pin`), never a constant in the code.** A commit cannot name
  itself, so the kit, the look and its driver carry no default pin and refuse without one. This file and the
  committed transcripts are the only places the pin is written.
- **Seeds 5101–5108, one per run; 5109 is the spare.** C 5101 / 5102, W 5103 / 5104, Br 5105 / 5106, WB 5107 /
  5108. No two runs share a seed, so no cell of the cross is a matched pair and the statistic treats every run as
  independent.
- **Disjointness, checked 2026-11-09:** `python -m main.b1.arms check --tree <a checkout of the pin>` scanned 640
  `metadata.json` files and 3,888 directory names under the main checkout's `models/` and `~/gen3ai_archive`: no
  recorded seed, no run name and no `_s51NN` directory name collides with 5101–5109 or the twelve run names. The
  same command verifies the corpus identity and that each arm differs from the control by its lever alone. **It is
  re-run immediately before the first launch** (§13 item 1): a hit refuses the launch. Run again later, it lists
  the experiment's own folders as `own` (a folder whose immutable `original_command` and seed are the registered
  run's at the pin); a foreign run under one of these names or seeds is still a hit.

## 4. The schedule

- **One run at a time on the one card, in this order within each phase:** C 5101, W 5103, Br 5105, WB 5107, C 5102,
  W 5104, Br 5106, WB 5108 (one seed of every arm before any arm's second, so an early stop leaves a balanced set).
- **Phase 1:** each run fresh to 50,000,000 steps. **The 50M look** (§8) is read. **Phase 2:** each surviving run is
  RESUMED in its own folder to 100,000,000 (`--model models/<run>/final_model.zip --steps 100000000`, no
  `--run-name`, no `--arch`: a same-run resume, never a fork).
- **Why stopping at 50M is the same run.** The claim is NOT bit equality with a run that never
  stopped, and none is needed: a stop at the 50M target is ONE MORE RESTART of the kind every run takes every six
  hours, at a chosen step, applied to all eight runs alike. Measured by the kit's two-segment smoke on CPU
  (`python -m main.b1.resume_check smoke`, arm WB, a phased run against a run-through of the same seed, 2026-11-09;
  §13 item 6 repeats it at the pin):
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
- **The fixed fallback.** If §13 item 6 (the stop-and-resume check at the pin) is not green at launch, the eight
  runs launch straight to 100M with `python -m main.b1.arms describe --block fallback --pin <PIN>`. **Under the
  fallback the 50M look is REPORTED and stops nothing**: it needs all eight runs at 50M, which first holds midway
  through the eighth run, when a stop could save at most half a run; and it can then be played only on CPU (≈ 12
  h), because the card is training. Which block launched is recorded in §18. The looks' checkpoint rule (§5.1) is
  the same under both blocks.
- **Restarts.** The launcher restarts each child every 6 h, as for every production run.
- **No run of the eight is thinned before §9's reading is closed.** The looks read the 1M checkpoints (the
  equal-compute look at a step fixed only by the pilot, §13), and phase 2 resumes with the self-play pool the stop
  left; the standard thin plan would cut both. Disk: ≈ 100 checkpoints a run, ≈ 55 MiB each at width 128
  (`rb_x60t_s3501`'s) and 73 MiB at width 256 (the smoke's WB zip), so ≈ 50 GiB for the eight runs; the launcher's
  dry run sizes the wide arms from a 128-wide checkpoint and understates them.

## 5. Meter 1, which decides: the head-to-head cross

### 5.1 The cells

- **24 cells a look:** every run against every run of another arm, once (4 arms, 6 arm pairs, 2 × 2 runs each).
  The arm later in (C, W, Br, WB) is the player; the other side's score is 100 − score.
- **Each cell:** 1,000 mirrored pairs on home72 and 500 on each of H, F and P72; both sides greedy (the deployed
  policy); one engine, the pin's; `--seed 0`; oracle reveal off. 120,000 games a look.
- **Which file a run plays at a look:** the first `checkpoints/checkpoint_<N>_steps.zip` with N at or after the
  look's step and below the look's step + 1,000,000. Never `final_model.zip`, so the phased and the fallback block
  read the same bytes.
- **Views.** A cell's score on a view is a fixed weighted mean of its panel rates: **N** = ½ home72 + ⅙ H + ⅙ F +
  ⅙ P72 (the owner's exchange rate r = 1 between home and unfamiliar teams, `../../../STATUS.md` §6.4); **F** =
  forum72 alone; **HF** = ½ home72 + ½ forum72; home72, U, H, P72 alone are reported.

### 5.2 A factor's statistic

For a factor and a view, rows = the factor's four ON runs, columns = its four OFF runs, 16 cells.

- **Δ̂** = the mean of the 16 cell scores − 50.
- **Variance.** A run's row mean is its mean score against all four OFF runs. Within one ON arm the other factor is
  constant, so the sample variance of that arm's two row means estimates the seed variance free of the other
  factor's effect. s²_rows = the mean over the two ON arms of that within-arm variance, **each floored at 1.76²**
  (the measured seed SD, §11.1: a lucky small variance from two seeds must not pass). s²_cols likewise over the two
  OFF arms. **V̂ = s²_rows / 4 + s²_cols / 4**, on **4 df** (one per arm), Student's t = 2.132.
- **The interval** is Δ̂ ± 2.132 · √V̂, two-sided 90 %. Its half-width is never below 2.653 pp (all four variances at
  the floor); its simulated median is 3.04 pp on N and 3.14 pp on F (§11.2).
- A bound within 0.005 pp of the threshold it is compared with FAILS that comparison and is named `at_boundary`
  (root rule 8: the conservative class, deterministically).
- **The assumption.** The 4-df t is exact when the four arms share one seed variance. If one arm's seed SD is
  several times the others', each 5 % tail grows (to ≈ 5–8 % in §11.2's cases); nothing here tests for it, and the
  four within-arm variances are printed with every statistic.
- Code: `main.b1.cross.factor_stat`. `power_matrix.py` classes every simulated draw twice, with the rule code and
  with its own transcription of the formula above (0 mismatches in 624,800). That checks the transcription; it
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

- With m(A, B) the mean of the four cells of arm A against arm B: **Î = ½ [(m(WB, Br) − m(W, C)) + (m(WB, W) −
  m(Br, C))]**: how much more width is worth with breadth on than off, estimated from both ways of pairing the four
  arms. Each run enters through ONE per-run quantity u: its mean score against the two arms that differ from its
  own in exactly one factor (for a WB or a C run: against W and against Br; for a W or a Br run: against C and
  against WB), which a run's own strength moves one for one. V̂ = the sum over the four arms of the within-arm
  variance of u (not of §5.2's row means), each floored at 1.76², divided by 2; on 4 df.
- Classes: SYNERGY (lo > 0), ANTAGONISM (hi < 0), NOT-SHOWN, **read on N**; the estimate and interval are also
  printed on F, without a class.
- **It cannot be resolved at this size:** its half-width is never below 5.31 pp and its median is 6.09 pp (§11.3).
  It is a point estimate with an honest interval; a NOT-SHOWN is not evidence of additivity.
- The six simple effects (W, Br, WB each against C; WB against Br and against W; Br against W) are reported with
  2-df intervals.

## 6. Meter 2, which confirms: Kakuna, clustered by team

- **Games.** Each run's 100M final plays Kakuna on H, F and P72, 720 games a panel (unit seeds 0–35), 17,280
  games. No Kakuna games on home72. Spec: [`read_spec_kakuna.json`](read_spec_kakuna.json) (read
  `boundary1_kakuna_2026-11-09`, rule `b1_kakuna`, `src/main/b1/kakuna_rule.py`).
- **Contrasts.** CONFIRMING: **`width_U`** = ½(W + WB) − ½(C + Br) on U, and **`breadth_F`** = ½(Br + WB) − ½(C +
  W) on F. REPORTED: `breadth_U`, `width_F`, and `interaction_U` = WB − W − Br + C on U (the full difference of
  differences, the same scale as §5.4's Î).
- **The interval** is analytic and two-way, computed once for each of the two team clusterings of record
  (`96adb152c`'s addendum): by OUR team, and dyadic (a team is one cluster in either role). With V_S the pooled
  within-arm seed variance of the contrast, V_T the team-cluster sandwich variance across the eight runs and V_TS
  the same sandwich with every cluster split by run: **V = max(V_S, V_TS) + max(0, V_T − V_TS)**. Satterthwaite
  df; two-sided 90 %; **the WIDER of the two schemes' intervals decides.** A tied game counts as a non-win. The
  lane deals teams with replacement, so a panel is not balanced by team (the record reports each panel's design).
  Classes: POSITIVE (lo > 0), NEGATIVE (hi < 0), SPANS-ZERO, with the same 0.005 pp boundary rule; INCONCLUSIVE if
  any run has fewer than 95 % of its games on a panel or more than a quarter of a panel's games ended by timeout.
- **REPORTED beside it:** two team × seed bootstraps on one shared index set (10,000 draws, seed 20261109). The
  draft named the bootstrap as the statistic; at two seeds an arm a seed bootstrap has three distinct resamples an
  arm and understates the spread, so the analytic interval is the registered one (deviation, §17).
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
- **The finals are copied before anything continues.** `python -m main.reads plan` copies the eight finals into
  the read's state and every unit re-hashes its copy. It is run when phase 2 ends; no run is resumed past 100M
  (§9) before that copy exists, so the read can never play a later final as "the 100M final".
- **After a 50M STOP-HARM** fewer than eight runs reach 100M and this spec cannot plan. No Kakuna read is
  registered for that case (§8).
- Its resolution is §11.4: it can contradict a large in-house effect; it cannot find a +2 pp one alone.

## 7. Reported, never a verdict

- **Curve dominance** on each run's own snapshot ladder (`<run>/snapshot_ladder/ladder.json`, ± 10 Elo), compared
  across runs only at matched snapshot COUNT (`python -m main.elo compare`), the newest node excluded. A factor
  DOMINATES iff its ON arms' mean is above its OFF arms' at every matched count from 30M to 100M and at least three
  of its four ON runs end above the OFF runs' median. Ladders of different runs meet different opponent pools: this
  is pace evidence.
- **The 75M look**: the same cross at 75M, played with the 100M look. No class.
- **The equal-compute look (EQ):** the wide arms at an earlier step against the narrow arms at 100M, the same 24
  cells. The step is fixed BEFORE launch from §13 item 4's measured step-time ratio ρ: **eq step = 1,000,000 ×
  ⌊100 / ρ⌋** (at most 99M), written into §18 when the pilot lands. No class: the entity toy led at equal compute
  and lost at its plateau (`UNDERSTANDING.md` §4.9), so this is a cost statement. The realised GPU-hours of each run
  are reported beside it.
- **Every factor on every view** (N, home72, U, H, F, P72, HF), with intervals; breadth's on H, P72, U and N carry
  the M4 overlap and are labelled so.
- **The control against X26's snapshot at the matched step**, for continuity with the previous lineage (different
  observation: E13 and the mechanics rounds landed since; a difference is not attributable).

## 8. The looks

| look | when | what is read | consequence |
|---|---|---|---|
| **50M** | after phase 1 (all eight at 50M) | each factor's HARM view | **STOP-HARM** iff hi < −2.0; else CONTINUE (REPORTED only under §4's fallback) |
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
    plan --survivor <that factor>`: 4 cells a panel, two runs against two. Its simple effect is REPORTED on every
    view with a 2-df interval (half-width at least 5.14 pp; for breadth, with the net's point estimate), under the
    outcome `NO-CLASS`, and goes to the owner.
  - Why no class: two runs against two find a +4 pp effect about one time in six (§11.5); a class there would be
    decided by noise. No Kakuna read is registered for this case.
  - If both factors stop, only the control continues.
- **A look is planned and read by code** (`python -m main.b1.look`, driven by [`looks/drive_look.sh`](looks/drive_look.sh)).
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
| **WIN** (for breadth: with the net rule passed) | The factor is in the continuing cell (rule 3). The cell's two runs continue to the plateau (`python -m main.plateau tick`). |
| **LOSS** | The factor is retired from scratch as specified (this width; this corpus at this share). |
| **NO-USEFUL-GAIN** | The factor is retired from scratch at this size and a 100M budget. |
| neither factor WIN, no owner row | The control is BOUNDARY-1's lineage; its two runs are the candidates. |
| *owner:* **UNDECIDED**, either factor | With ONE priced option, the extension below. No other extension. |
| *owner:* **WIN-NET-NEGATIVE** (breadth) | Not adopted, not retired: reported with its cost and gain. |
| *owner:* both WIN and the interaction reads **ANTAGONISM on N** | The choice of cell. |
| *owner:* IN-HOUSE-ONLY or KAKUNA-ONLY (§6), or an UNREAD that cannot be finished | Before any action. |
| *owner:* a **STOP-HARM** at 50M (§8) | No class for either factor; the survivor's reported simple effect. |
| *owner:* INCONCLUSIVE, or any §10 hazard | No adoption. |

**The extension (the one priced option after an UNDECIDED).** One more seed in every arm: four runs, ≈ 104 GPU-h.

- Its seeds are registered by an amendment after a fresh disjointness scan. The spare 5109 is not an extension
  seed.
- **Both factors are re-read at three runs an arm** with the same statistic on 8 df, and the three-run classes
  REPLACE the two-run ones for both factors, whichever was UNDECIDED. One time.
- Its error rates as a two-stage rule (stop at two runs unless UNDECIDED) are §11.2's last table.
- **Not built at registration:** `main.b1.cross` takes any equal number of runs an arm (the three-run rows of §11
  run the rule code); `main.b1.look`'s seed table holds two an arm. The amendment that registers the seeds also
  extends that table, with its tests, before the extension's runs launch.

## 10. Validity, exclusions, replacement, holds

- **A run counts at a look iff** (`main.b1.look.check_run`, all checked from the run's own records):
  its look checkpoint exists and its sidecar's step equals the file's; the run and the checkpoint carry ONE pin
  entry, the registered pin, from step 0; its lineage role is `fresh` with no fork parent; its immutable
  `original_command` is token for token the registered argv of its arm and seed (at either phase target);
  the last process still ran ten epochs at `ent_coef` 0.05 with the arm's `d_model`, `trunk_ffn_dim`, `team_mix`
  and `team_mix_sides`. Across the look: one torch version and one init thread count, and no two runs with the same
  bytes; and every run's last process records the same public `cli_args` as the first control run's, except its
  own keys (`model`, `run_dir`, `run_name`, `seed`, `steps`) and its arm's lever keys, so any other knob that moved
  in one run and not in the control refuses the plan. Ten recorded keys are rewritten by ANY resume to the parser's
  default or an "inherited" label (`arch`, `batch_size`, `n_steps`, `unified_moves`, `unified_damage`,
  `allow_nonproduction_arch` and four `*_source` labels; measured on the kit's smoke, §4): those are compared only
  between two last processes of the same kind (fresh, six-hour restart child, typed resume), and the rollout shape
  they stand for is compared where a resume cannot rewrite it, `metadata.json`'s top-level `batch_size` and
  `n_steps`, which must equal the control's in every run. REPORTED in the driver's log beside these guards, never a gate: `python -m main.sidecar_audit
  --strict` and `python -m main.checkargs models/<run>` on every run of the look.
- **Replacement, by a closed list.** A run is replaced by seed 5109 in ITS OWN arm, from step 0, only when it
  cannot reach a look for one of these two causes:
  - its files are lost or unreadable (a disk or filesystem fault, a corrupt checkpoint it cannot resume from);
  - it was launched with an argv other than its registered one (an operator error, which the guard above
    catches).
  **Anything else is "lost to its lever"**, including a NaN or divergence, an out-of-memory failure at the arm's
  own width, and a failure to load the corpus. The cause is written into §18 by the orchestrator before any look
  that would contain the run is read. The look then takes `look plan --replace <the registered seed that 5109
  replaced>` (for example `--replace 5103`). One replacement exists; a second loss goes to the owner.
- **A run lost to its lever is the result**: not replaced, reported. The factor statistics need equal runs an arm,
  so that look is INCONCLUSIVE and goes to the owner with the simple effects that remain, REPORTED and unclassed.
- **A timeout is never an outcome.** `main.h2h` ends a game at the declared turn limit and records it as a DRAW,
  half a win to each side; it records no aborted games. A cell with more than 25 % of its games drawn therefore
  makes the look INCONCLUSIVE, and `verdict.json` carries each panel's drawn share and its worst cell's. The
  Kakuna read applies its own quarter per panel to timed-out games.
- **No game is excluded for its outcome.** No panel, cell, seed or look is added after a look is read except by
  §9's one extension.
- **HOLDs before launch** (the registration is re-read, not launched, if either holds): X60-T reads
  ADOPT-CANDIDATE (human teams would then enter the base of every run); the owner changes `ent_coef` after X67.

## 11. Power, registered honestly

Simulation: [`power_matrix.py`](power_matrix.py) → [`power_matrix.log`](power_matrix.log), 50,000 draws a row,
with game noise by view (cell SD 0.68 pp on N, 1.05 on home72, 1.48 on a 500-pair panel) and the registered rule.
Every row assumes what §5.2 assumes (independent runs, one seed variance in all four arms) unless it says otherwise.

### 11.1 The seed SD is the unknown that sets everything

- Measured: **1.76 pp**, pooled over 14 arm groups of FRESH 15M production-family runs (range 0.60–3.09; 90 nominal
  df that overstate it, the groups share seeds). [`seed_spread.log`](seed_spread.log).
- **UNVERIFIED at 100M.** Every row is at 15M; the one deeper hint is three toy seeds, 0.67 pp at ≈ 47M and 1.82 pp
  at 100M. Hence the floor at 1.76 and the 2.5 and 3.1 columns.
- **UNMEASURED on forum72 alone**, the view that decides breadth: every measured cross is on home teams. A
  per-panel seed SD above the N-view's is plausible (72 unfamiliar teams, half the pairs).

### 11.2 One factor at 100M (two runs a cell)

| view, true seed SD | median half-width | P(WIN) at +2 / +3 / +4 / +5 / +6 pp | P(NO-USEFUL-GAIN) at 0 | false WIN / false LOSS at 0 |
|---|---|---|---|---|
| **N (width), 1.76** | 3.04 | 0.19 / 0.44 / 0.72 / 0.90 / 0.97 | 0.72 | 0.01 / 0.009 |
| N, 2.5 | 3.80 | 0.17 / 0.32 / 0.51 / 0.69 / 0.84 | 0.49 | 0.02 / 0.024 |
| N, 3.1 | 4.53 | 0.15 / 0.26 / 0.40 / 0.55 / 0.69 | 0.37 | 0.03 / 0.033 |
| **F (breadth), 1.76** | 3.14 | 0.18 / — / **0.69** / — / 0.96 | 0.67 | 0.01 / — |
| **F, 2.5** | 3.90 | 0.16 / — / **0.49** / — / 0.82 | 0.47 | 0.02 / — |
| F, 3.1 | 4.60 | 0.15 / — / 0.39 / — / 0.68 | 0.36 | 0.03 / — |
| N, 1.76, three runs a cell (§9's extension) | 2.13 | 0.42 / 0.77 / 0.95 / 1.00 / 1.00 | 0.93 | 0.02 / 0.018 |

- **In words.** The design finds a +4 pp effect about seven times in ten if the seed SD at 100M is what it was at
  15M, and about one time in two if it is 2.5 pp. It finds +6 pp almost always in the first case. It cannot find
  +2 pp. A true zero (N, 1.76) reads NO-USEFUL-GAIN 72 % of the time, UNDECIDED 27 %, WIN 0.9 % and LOSS 0.9 %.
- **Breadth is the weaker read**: one 500-pair panel, an unmeasured seed SD, and fork evidence of a small effect
  (X61P: +0.55 pp against Kakuna on F, and −3.28 pp [−4.14, −2.43] in-house on home teams). The likeliest breadth
  outcomes are NO-USEFUL-GAIN and UNDECIDED. This is registered knowingly: the alternative meter could not fire.
- **The 50M harm stop** (N, seed SD 1.76): P(STOP-HARM) 0.000 at 0, 0.009 at −2, 0.19 at −4, 0.45 at −5, 0.72 at
  −6, 0.98 at −8. It catches a disaster, not a mild harm.
- **Breadth's 50M stop reads HF, half its net.** With seed effects shared across panels, P(STOP-HARM) is 0.000 at a
  true (home72, F) of (0, 0) and of (−4, +4), 0.009 at (−4, 0), 0.19 at (−8, 0), 0.71 at (−12, 0) and at (−6, −6),
  0.97 at (−16, 0); a home cost of −12 with an F gain of +4 stops 19 % of the time. With independent seed effects
  the same rows read 0.000, 0.000, 0.001, 0.19, 0.89, 0.88, 1.00 and 0.18. So breadth is stopped at 50M only by a
  net near −12 pp or worse; X61P's fork cost (−3.28 on home teams) would not stop it.
- **An interaction does not bias a main effect's interval**: it shifts the main effect by half its size, which is
  what a main effect is (width +4 with interaction +6 reads +6.99, P(WIN) 1.00).

**Breadth as registered: the WIN on F and the net rule together** (seed SD 1.76; each cell is P(WIN, adopted) /
P(WIN-NET-NEGATIVE)):

| true (home72, F), pp | true net | seed effects shared across panels | independent across panels |
|---|---|---|---|
| (0, +4) | +4 | 0.68 / 0.00 | 0.68 / 0.00 |
| (−2, +4) | +2 | 0.66 / 0.02 | 0.65 / 0.03 |
| (−4, +4) | 0 | 0.47 / 0.21 | 0.43 / 0.25 |
| (−6, +4) | −2 | 0.21 / 0.48 | 0.13 / 0.56 |
| (−6, +6) | 0 | 0.50 / 0.46 | 0.50 / 0.47 |
| (−9, +6) | −3 | 0.12 / 0.84 | 0.05 / 0.92 |
| (−3.28, +2), X61P's fork cost with a +2 gain | −1.28 | 0.18 / 0.01 | 0.11 / 0.07 |
| (0, 0) | 0 | 0.01 / 0.00 | 0.01 / 0.00 |

- **Near a true net of zero the net rule is a coin flip**, and a WIN is adopted 43–50 % of the time. At a true net
  of −2 a WIN is still adopted 13–21 % of the time. The net rule compares point estimates; it is the owner's
  exchange rate applied to this sample, not a test that the net is positive.
- At a seed SD of 2.5 the same rows are in the log: (0, +4) 0.49 / 0.00, (−4, +4) 0.41 / 0.09 shared and 0.35 /
  0.14 independent.

**If the arms do NOT share one seed variance** (no true effect; width on N; seed SDs of C, W, Br, WB):

| seed SDs (C, W, Br, WB) | false WIN | false LOSS | NO-USEFUL-GAIN | median half-width |
|---|---|---|---|---|
| 1.76 in all four | 0.008 | 0.009 | 0.71 | 3.03 |
| 1, 6, 1, 1 | 0.051 | 0.050 | 0.40 | 3.87 |
| 0.5, 8, 0.5, 0.5 | 0.076 | 0.072 | 0.30 | 4.68 |
| 1, 6, 1, 6 (both wide arms) | 0.055 | 0.053 | 0.25 | 5.71 |
| 1, 1, 6, 6 (both breadth arms) | 0.055 | 0.056 | 0.25 | 5.73 |

The floor makes the rule conservative when the arms are alike (false WIN 0.8 %, not 5 %); one arm several times
noisier than the rest brings each tail back to 5–8 %.

**The extension as a two-stage rule** (read at two runs; only an UNDECIDED extends; then read at three):

| true effect on N, seed SD | WIN at two runs | + UNDECIDED, then WIN at three | total P(WIN) | total P(LOSS) |
|---|---|---|---|---|
| 0, 1.76 | 0.009 | 0.013 | **0.022** | 0.009 |
| +2, 1.76 | 0.191 | 0.248 | 0.439 | 0.000 |
| +4, 1.76 | 0.719 | 0.233 | 0.952 | 0.000 |
| 0, 2.5 | 0.023 | 0.024 | **0.047** | 0.023 |
| +2, 2.5 | 0.169 | 0.171 | 0.340 | 0.001 |
| +4, 2.5 | 0.507 | 0.293 | 0.799 | 0.000 |

**Multiplicity.** Two factors are each read two-sided at 90 % (5 % a tail); nothing is corrected for there being
two. With no true effect in either, the chance that a given factor reads a false WIN is 0.9–2.3 % at two runs
and 2.2–4.7 % through the extension (seed SD 1.76–2.5); the chance that at least one of the two does is at most
twice that. The interaction, the simple effects, the 75M and EQ looks and every non-deciding view are REPORTED
and decide nothing, so they add no decision error.

### 11.3 The interaction

Median half-width 6.09 pp at seed SD 1.76 (7.60 at 2.5). P(SYNERGY) 0.19 / 0.45 / 0.72 / 0.90 at a true +4 / +6 /
+8 / +10 pp; false SYNERGY or ANTAGONISM ≈ 0.01 each at zero. Only an interaction the size of X61P's whole in-pool
cost and more would show.

### 11.4 Kakuna, the confirming meter

Simulated (`kakuna_oc.py` → `kakuna_oc.log` section 4; 4,000 experiments a cell, a rate's Monte-Carlo error at most
0.008), on the registered design (360 deals × 2 games = 720 games a run a panel) and the registered analytic interval
at its 90 % level, under an ASSUMED team model: our-team SD 10 pp, their-team SD 9 pp, team-by-arm interaction SD 5
pp. The seed SD on Kakuna is unmeasured, so three values are shown.

| contrast | seed SD | false POSITIVE / NEGATIVE at 0 | P(POSITIVE) at +2 / +3 / +4 / +5 / +6 pp | median half-width | found 80 % of the time from |
|---|---|---|---|---|---|
| `width_U` | 1.0 | 0.013 / 0.018 | 0.37 / 0.68 / 0.91 / 0.99 / 1.00 | 2.26 | 3.5 pp |
| `width_U` | 1.5 | 0.030 / 0.033 | 0.33 / 0.57 / 0.79 / 0.93 / 0.98 | 2.52 | 4.1 pp |
| `width_U` | 2.5 | 0.041 / 0.044 | 0.21 / 0.37 / 0.52 / 0.69 / 0.82 | 3.70 | 5.8 pp |
| `breadth_F` | 1.0 | 0.007 / 0.009 | 0.12 / 0.28 / 0.52 / 0.75 / 0.91 | 3.85 | 5.3 pp |
| `breadth_F` | 1.5 | 0.014 / 0.017 | 0.14 / 0.28 / 0.48 / 0.70 / 0.85 | 3.92 | 5.7 pp |
| `breadth_F` | 2.5 | 0.023 / 0.027 | 0.14 / 0.25 / 0.39 / 0.56 / 0.69 | 4.30 | 7.1 pp |

- **What it means for §6's labels.** A true +4 pp width gain on U is CONFIRMED four times in five at a seed SD of
  1.5 pp and one time in two at 2.5; a true +4 pp breadth gain on forum72 is CONFIRMED one time in two at best. So
  **UNCONFIRMED (SPANS-ZERO) is the expected label for a real gain of the size the bar asks for**, on breadth most
  of all, and it changes no decision (§6). A contradiction is rare when there is none: with no true effect the meter
  reads NEGATIVE at most 4.4 % of the time.
- **The interval is conservative, by construction.** Its coverage at a true zero is 0.94 / 0.91 on U (seed SD 1.5 /
  2.5) against a nominal 0.90, and stays at or above 0.906 across the log's five sensitivity models (section 2:
  smaller team effects, a larger or absent interaction).
- **Why the bootstraps are REPORTED and decide nothing.** The literal team × seed bootstrap under-covers with two
  seeds a cell: 0.88 on U at a seed SD of 1.5 pp and 0.78 at 2.5, against 0.90 (section 3; 1,000 experiments).
- Every number here is conditional on the assumed team model; the read's own record reports the measured variance
  components, which replace the assumption after the fact and change no class.

### 11.5 After a STOP-HARM of the other factor

No class is registered there (§8). What the surviving two-against-two comparison could have shown, had the +4 pp
bar been applied to it (2 df):

| view, seed SD | median half-width | P(lo > 0) at +4 / +6 / +8 / +10 pp | P(hi < +4) at 0 |
|---|---|---|---|
| N (width's survivor), 1.76 | 5.39 | 0.17 / 0.50 / 0.80 / 0.94 | 0.17 |
| N, 2.5 | 6.74 | 0.16 / 0.36 / 0.59 / 0.76 | 0.15 |
| F (breadth's survivor), 1.76 | 5.77 | 0.16 / 0.44 / 0.73 / 0.89 | — |
| F, 2.5 | 7.08 | 0.15 / 0.33 / 0.55 / 0.72 | — |

It finds +4 pp one time in six and can almost never say NO-USEFUL-GAIN. That is why it carries no class.

## 12. Cost and timeline

| item | GPU-h | CPU | calendar |
|---|---|---|---|
| phase 1, eight runs to 50M | ≈ 100 (4 × 9.5 + 4 × 15.6) | — | ≈ 4.4 days |
| phase 2, 50M → 100M | ≈ 107 (4 × 10.1 + 4 × 16.6) | — | ≈ 4.6 days |
| four looks, 120,000 games each | ≈ 0.9 each under the GPU lock | or ≈ 12 h each on the three agent cores | between runs |
| Kakuna, 17,280 games | — | ≈ 120 lane-h | ≈ 40 h on 3 lanes |
| **to the decision** | **≈ 207 (180–227) + ≈ 4** | | **≈ 10 days** |
| after a WIN: two candidates to the plateau | ≈ 122 (GUESS) | ≈ 30 lane-h | ≈ 5 days |
| §9's extension (one more seed an arm) | ≈ 104 | ≈ 60 lane-h | ≈ 4.5 days |

- Narrow runs: MEASURED on `rb_x26_s2601` (4.40 / 9.48 / 19.55 GPU-h to 25M / 50M / 100M). X26 carried five
  ride-along heads and predates E13; **UNVERIFIED** that today's production costs the same.
- **The wide arm's step-time ratio ρ = 1.65 is a GUESS (1.3–1.9)** until §13 item 4 measures it. On CPU the learner's
  forward + backward is 1.94× and the policy forward 1.27–1.48×. The draft sent a measured ρ above 1.9 back to the
  owner for the budget; the owner's standing rule since is no cap on GPU commitments, so **a measured ρ above 1.9 is
  reported to the orchestrator, who decides**; it is not an automatic hold.
- **Timeline (if the card frees as planned).** Registered Mon 2026-11-09. Launch Fri 11-13 after X66 and X67.
  Phase 1 ends ≈ Tue 11-17, the 50M look the same day, phase 2 ends ≈ Sun 11-22, the 100M, 75M and EQ looks that
  day, Kakuna ≈ Tue 11-24, the decision ≈ Wed 11-25.

## 13. The GPU pre-launch checklist

Nothing here could be verified by the registrar: this unit had no GPU lease. Every item is done at the pin, by the
agent the orchestrator leases the card to, and its result is recorded in §18 before the first launch.

1. **The kit's check at the pin**, from a clean checkout of it: `python -m main.b1.arms check --tree <checkout>
   --pin <PIN> --params` (seeds and names still free; corpus identity; levers; parameter counts 2,609,183 /
   6,258,079).
2. **Each first-seed dry run with the card visible**: `python -m main.launcher --dry-run <argv>` for the four arms
   passes EVERY preflight (on CPU its one refusal was the CUDA preflight).
3. **The CUDA-only slow tests at the pin**, banked in a leased `slow-gpu` window (`python -m
   main.ops.gpu_fill_window`, `../../../ops/testing.md` § The GPU slow tier). They have not run on the code this
   experiment trains: E13's ship skipped 9 (the compile-region gates on trained weights among them) and the width
   ship skipped 11 in `src/agents/model`. A FAIL in a test the wide or the production path touches holds the launch.
4. **The wide-arm pilot** (≈ 15 min of the card): W's argv into a scratch models dir for a few updates, and C's the
   same, for (a) step time at 256 against 128 → **ρ**, which fixes §7's EQ step and §12's budget; (b) the bf16
   compile at 256 completes and its compile-parity check passes; (c) the inference CUDA-graph capture at 256
   succeeds; (d) peak GPU memory at 256 fits with the production batch. The width commit records all four as
   UNVERIFIED on GPU. Any failure holds the launch and comes back as a build.
5. **A 128-against-256 head-to-head cell on the card**: one short `main.h2h` cell between a W and a C pilot
   checkpoint on the look's device and settings, to prove the engine pairs two widths in one process on the card.
   On CPU it is proven: `src/main/b1/width_pairing_integration_test.py` plays cross-width cells in both directions
   and a same-width cell of each width on ONE engine, from checkpoints built at the control's and the width arm's
   own resolved configs (3 tests, 520 s beside a live run, banked 2026-11-09). The GPU inference path (graph
   capture, bf16) is not covered by it.
6. **The stop-and-resume check at the pin**: `python -m main.b1.resume_check static --tree <checkout> --pin <PIN>`
   prints `PASS` (CPU, seconds; no lease). Its two-segment smoke (`python -m main.b1.resume_check smoke --models-dir
   <an empty scratch dir> --arm wb --half 4000`, CPU, ≈ 35 min, 8 GB peak) is banked in
   [`resume_check.log`](resume_check.log) from this unit's tree, PASS; it is repeated only if the trainer, launcher
   or model code at the launch pin differs from the tree it ran on (§3 lists what the pin adds). Not green ⇒ §4's
   fixed fallback block.
7. **The spec**: `python -m main.reads check-spec --spec <this folder>/read_spec_kakuna.json` at the pin reports
   `play_path_differs_at_head` 0, or names the files.
8. **The HOLDs of §10** are checked against the day's STATUS: X60-T's verdict, the owner's word on `ent_coef`.
9. **The core split is settled before the first launch and not changed until the eighth run ends**
   (`python -m utils.core_split show`; `designs/ops/core_split.md` proposes an A/B at the next launch). A look
   refuses a set of runs whose recorded `init_num_threads` differ (§10), so a split changed between two runs of the
   eight costs the look, by design: a different thread count is a different trainer.

## 14. Commands

Run every training command from the MAIN checkout with the launcher's interpreter (`$GEN3AI_PYTHON`), one at a time.

- **Launch block** (the Training session's): [`launch_block.txt`](launch_block.txt), which is `python -m
  main.b1.arms describe --pin <PIN>`; the fallback is `python -m main.b1.arms describe --block fallback --pin
  <PIN>`.
  The block is typed ONE COMMAND AT A TIME by the Training session: the arm chain (`main.x35`'s kit) knows only the
  X35 fork arms and refuses a fresh arm (`kind 'fresh' has no implementation`), and no chain backend was built for
  this unit. Each command is followed, commented, by its own `--dry-run` form; run it first, and expect "would
  launch" (a phase-2 dry run reads "RESTART of models/<run>"). Re-issuing a phase-2 command on a run that already
  reached 100M exits 0 and trains nothing. On a phase-2 launch the launcher's banner reads "`--model` launch
  (fork)"; that wording is the launcher's for every `--model` command. The dry run's `role : RESTART`, the child's
  `[Recipe] --model launch (restart)` lines and the lineage sidecar are what say it is the same run.
- **A look** (any agent holding the GPU lease for the `cuda` form; the `cpu` form needs no lease), from a clean
  detached checkout of the pin:

  ```bash
  git worktree add --detach <PIN_DIR> 507d0c11961da8d4243a4c8877bcc04ef220058c
  <PIN_DIR>/designs/research_state/measurements/boundary1_scratch_2026-11-09/looks/drive_look.sh \
      <PIN_DIR> 507d0c11961da8d4243a4c8877bcc04ef220058c <WORK_DIR> 50M cuda
  # or 75M / 100M; EQ takes --eq-step <N>; after a replacement, --replace <the registered seed 5109 replaced>;
  # after a STOP-HARM of the other factor, --survivor width|breadth
  ```

  It plans (exit 3 = a §10 guard refused), registers each panel's family (idempotent), plays the 24 cells a panel
  with `python -m main.h2h play-many` under `gpu_lock.sh` + `mem_cap.sh`, and writes `<WORK_DIR>/result/verdict.json`
  with `python -m main.b1.look read`. One look is played on ONE device. After the Kakuna read: `python -m
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

  `plan` runs as soon as phase 2 ends, before any run continues (§6). `run` is resumable: it is re-issued until no
  unit is unplayed (≈ 120 lane-hours; one lane alone would play a sixth of the read in 20 h). `record` on an
  unfinished read gives UNREAD (§6), never a label to act on.

## 15. What was built for this registration

| tool | where | pinned by |
|---|---|---|
| the fresh-run kit: arms, argvs, seeds, the collision scan, the launch block | `src/main/b1/arms.py` | `arms_test.py` |
| the stop-and-resume equivalence check | `src/main/b1/resume_check.py` | `resume_check_test.py` |
| the multi-panel cross: views, the floored statistic, classes, the net rule, the confirming labels, the interaction | `src/main/b1/cross.py` | `cross_test.py` |
| the look: run guards, checkpoint rule, cells plan, ledger names, the read | `src/main/b1/look.py`, `looks/drive_look.sh` | `look_test.py` |
| the Kakuna rule, clustered by team and seed | `src/main/b1/kakuna_rule.py` | `kakuna_rule_test.py` |
| the two widths on one head-to-head engine (CPU) | (no new code: `main.h2h`'s two-architecture engine) | `width_pairing_integration_test.py` (slow) |
| the operating characteristics | `power_matrix.py`, `kakuna_oc.py` | each checks the rule code against its own formula |

Each test file holds cases that fail when the rule it pins is reverted (the floor, the df, the boundary band, the
net rule's sum, each guard's refusal, each estimator against a hand-computed case). For `main.b1.cross` and `main.b1.look` that was
measured once, on a hand-listed set (not an exhaustive one): 32 single-site changes (every registered constant, each
degrees-of-freedom and variance term, the deciding and harm views, the net rule's comparison, each guard), run
against the two test files in a shadow copy; 32 of 32 fail a test.

## 16. What is not claimed, whatever the outcome

- Nothing about the plateau from a 100M look.
- Nothing about widths other than 256, about depth, the critic, the policy tower or entropy.
- Nothing about other corpora or shares than `n1_proc_v1` at 0.5 on both sides.
- No pp against Kakuna from a pp in-house: the mapping is unmeasured.
- A NOT-SHOWN interaction is not additivity; a NO-USEFUL-GAIN is not "no effect"; UNDECIDED is not a null.
- A STOP-HARM at 50M says the factor was behind at 50M. It is not a LOSS at 100M: a factor that starts slowly and
  would have led at 100M is stopped and never read.
- A breadth WIN that passes the net rule is not evidence that the net is positive: near a true net of zero the
  rule passes about half the time (§11.2).
- Nothing from the survivor's two-against-two comparison after a STOP-HARM, beyond its printed interval.
- Breadth's read on H, P72, U and N is contaminated by its own training teams and is never quoted as generalisation.

## 17. Deviations from the draft, and the registrar's choices (each flagged for the orchestrator)

1. **Breadth decided in-house on F** with the net rule, Kakuna confirming (orchestrator's ruling; §1, §5.3).
2. **Breadth's 50M harm view is HF** (§8): the draft had one harm rule on N; under the ruling the only recognised
   home-team cost is the net, so the stop reads the net's view. Registrar's choice.
3. **The Kakuna interval is analytic**, the bootstrap reported (§6; accepted by the orchestrator).
4. **Seeds:** one per run, 5101–5108, spare 5109 (the draft reused seed numbers across arms; accepted).
5. **Checkpoints every 1M, typed** (accepted); looks read checkpoints, never `final_model.zip`.
6. **One pin** for training, the cross and Kakuna; registered in two commits (the tooling, then this file naming
   the tooling's commit as the pin).
7. **After a STOP-HARM no class is read**, for either factor: the survivor's 2-df simple effect is REPORTED and
   goes to the owner, and no Kakuna read is registered (§8; the draft's single-factor options had no such case).
   The alternative, a registered class on two runs against two, would find +4 pp one time in six.
8. **A measured ρ above 1.9** goes to the orchestrator, not the owner (§12).
9. **The EQ step is fixed from the pilot's ρ** before launch (§7), not from the runs' own realised time; its
   ledger request carries the step.
10. **An unfinished Kakuna read is UNREAD**, never UNCONFIRMED: it is finished or replayed, and §9 waits (§6, §9).
11. **§9's order of reading**: the Kakuna record first, owner rows before any action, then the continuing cell
    from the set of WINs. No run continues before the read's `plan` has copied the eight finals.
12. **A look's cells must hold exactly the registered pairs, at most a quarter of a cell's games drawn, rows at
    the pin on the panel's team set, and one device and backend** (§8, §10): the draft had none of the four.
    `main.h2h` scores a turn-limit game as a draw, so the draft's "aborted" clause could never have fired.
13. **Under the fallback block the 50M look stops nothing** (§4).
14. **Replacement is a closed list of two causes**; everything else is lost to its lever (§10).
15. **The extension re-reads BOTH factors at three runs an arm** and replaces both classes; its look path is built
    by its amendment, not now (§9).
16. **No amendment to §5, §6, §8 or §9 after the first look** except by the owner (header).
17. **The pin is an argument**, never a constant in the code (§3).
18. **A look refuses recipe drift** (§10): any recorded knob of a run's last process that differs from the first
    control run's, beyond the run's own keys and its arm's lever, refuses the plan. It compares records with
    records, so it catches a flag typed on one run's resume; it cannot catch a knob every run got wrong alike
    (that is `checkargs` at launch, §13 item 2). Its blind spot, by construction: one of the ten keys a resume
    rewrites, moved in a run whose last process is of another KIND than the control's (a replacement run still on
    its restart children beside seven phase-2 resumes), is not compared in `cli_args`; of those ten, the rollout
    shape is still caught at `metadata.json`'s top level, the width keys by the lever check, and `arch` by the
    token-for-token `original_command`; `unified_moves` / `unified_damage` and the four `*_source` labels are not.

## 18. Amendments and pre-launch records

None yet. The GPU checklist's results (§13), ρ and the EQ step are recorded here before the first launch.
