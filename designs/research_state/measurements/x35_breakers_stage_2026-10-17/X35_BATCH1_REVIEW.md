# X35 batch 1 + X47: ADVERSARIAL REVIEW of the launch sheet (2026-10-19)

**Verdict: GO-WITH-AMENDMENTS.** Two amendments BLOCK the first launch: **AM-1**, one pin for the whole batch, enforced;
and **AM-2**, a registered variance gate after the first six forks. AM-3 to AM-6 must be ruled before the arms they
touch. Nothing in the kit, the argvs or the launch sheet was edited here. The orchestrator rules on each amendment.

**What was reviewed:**
- [`BATCH1_LAUNCH_SHEET.md`](BATCH1_LAUNCH_SHEET.md) and [`STAGED_PLAN.md`](STAGED_PLAN.md);
- the kit `src/main/x35/` (`kit.py`, `readout.py`, `d1.py`, `launch_x35.sh`);
- [`design_compute_efficiency.md`](../../../endstate/design_compute_efficiency.md) §5 and
  [`design_bot_ideas_screen.md`](../../../endstate/design_bot_ideas_screen.md) §1–§2.1;
- the measurement-discipline memories: equivalence needs the delta's interval, matched-noise control, a replicate is
  pin + seed, strong claims need a plateau, and one lever per A/B.

The review ran on CPU only (`CUDA_VISIBLE_DEVICES=""`, `nice -n 19 taskset -c 5-7,13-15`). It only READ `models/`:
X26's TensorBoard (TB) events, its checkpoint list and its snapshot pool. Every simulation calls the kit's OWN
decision functions, or is cross-checked against them (0 mismatches in 2,000 draws). The scripts are in
[`review_2026-10-19/`](review_2026-10-19/).

---

## 0. The short version

The plumbing is careful: every argv is built from the frozen reference, dry-run checked, and gated by pre-checks.
The weak point is the STATISTICS. At 3 seeds per side, the registered rule R1 (the population score's Welch interval
at 97.5 %, read against the ±2.0 pp bar) almost never reaches a class:
- at a realistic seed spread, look 1 decides 1–15 % of the time when the true effect is ≤ 3 pp;
- NULL is out of reach at 3 seeds unless fork-to-fork noise is ≈ 0.

So batch 1, as written, buys mostly CONTINUE and UNDECIDED:
- the toy transfer calibration **cannot** record agreement, because UNDECIDED never counts as agreement;
- X47 has no registered look 2, so even a +5 pp depth effect reads GAIN only about 1 time in 3;
- D1's Kakuna interval ignores seed-to-seed variance (8–21 % false GAIN), and its in-house co-primary passes only
  15 % of the time at a true zero.

The fixes are not "more games". The interval's width is set by the number of SEEDS (the cluster count), which no
number of games can buy after the fact. The fixes are:
- measure the seed spread first, on the forks already at the front of the queue;
- move C0's look-2 seeds forward;
- make the toy calibration conditional on production reaching a class;
- give X47 a look 2.

The revised order (§3) costs about 91 GPU-h before its conditional arms, against 108, and decides more.

---

## 1. The power arithmetic (what R1 can decide)

**Inputs, with their sources:**
- **Measurement noise of one final's S (the population score).** R has 6 members at 500 mirrored pairs each, with a
  pair SD of 0.34 (`toy_crosses_2026-10-16/PREREG.md` §3). That gives 0.34/√500/√6 = **0.62 pp**. It is consistent
  with the closing test's σ_e of 1.1 pp per 1,000-pair cell.
- **Run-to-run SD.** Nothing measured it for a 20M fork of a plateau checkpoint: **UNVERIFIED**, and the key unknown.
  The brackets that exist are all FROM-SCRATCH runs:
  - production at 15M: 1.35 / 1.63 pp (the closing test's s²_C 1.82 / s²_R 2.67, §9), and ≈ 2.5 pp in R-T5's EB
    columns (s²_C 7.74);
  - the toy at its 100M cap: ≈ 1.65 pp (R-T6, per-seed +8.10 / +11.24 / +11.27, after removing cell noise);
  - the toy at 46–48M: ≈ 0 (R-T5 rows, s²_R 0.45 ≈ cell noise alone).

  Forks share their start, so their spread is plausibly SMALLER. The total per-final σ is therefore bracketed at
  **0.8–1.6 pp**, and 2.0–2.5 pp is not excluded.
- **The zero-training pre-read (F-RV-2), interim 2026-10-20:** X26's own 290–380M nodes on the registered S give an SD
  of 0.92 pp across 4 nodes (measurement se 0.55; mostly a +2 pp/100M climb), the low end of this bracket; descriptive
  only, see [`x26_drift_preread/`](x26_drift_preread/README.md).

**The arithmetic.** At equal variances, a 3 v 3 Welch interval at 97.5 % (df ≤ 4, t ≥ 3.50) has a half-width of
≥ 2.85 σ; the simulated median is 3.1 σ, because the sampled df falls below 4. NULL needs a half-width below 2.0, so
it needs σ < 0.70. The measurement noise alone is 0.62.

**Simulated R1** ([`sim_r1.py`](review_2026-10-19/sim_r1.py), 40,000 draws per cell, the registered two-look rule;
"final" = after look 2 where look 1 said CONTINUE):

| σ per final | true Δ | P(GAIN) final | P(NULL) final | look 1 decides | sent to look 2 | median half-width L1 / L2 |
|---|---|---|---|---|---|---|
| 0.8 | 0 | 0.000 | 0.416 | 0.15 | 0.42 | 2.48 / 1.39 |
| 0.8 | 3 | 0.283 | 0 | 0.09 | 0.92 | |
| 0.8 | 4 | 0.836 | 0 | 0.33 | 0.67 | |
| 1.2 | 0 | 0.000 | 0.098 | 0.03 | 0.49 | 3.73 / 2.09 |
| 1.2 | 3 | 0.134 | 0.001 | 0.04 | 0.96 | |
| 1.2 | 4 | 0.480 | 0 | 0.14 | 0.86 | |
| 1.2 | 5 | 0.840 | 0 | 0.33 | 0.67 | |
| 1.6 | 0 | 0.000 | 0.022 | 0.01 | 0.50 | 4.95 / 2.79 |
| 1.6 | 5 | 0.578 | 0 | 0.19 | 0.81 | |
| 2.0 | 6 | 0.644 | 0 | 0.21 | 0.79 | 6.20 / 3.49 |

**How to read it:**
- The smallest effect R1 detects with 80 % power, even WITH look 2, is ≈ **4 pp at σ 0.8, 5 pp at σ 1.2 and 6 pp at
  σ 1.6**.
- Compare the base rate: X26's plateau rule calls FLAT at ≤ ≈ +7 Elo (≈ +1 pp) per 50M. A breaker must therefore
  add ≥ 4–6 pp in 20M, at least 10× the base climb, to be read as GAIN.
- A truly inert arm reads NULL 2–10 % of the time at σ 1.2–1.6, and otherwise UNDECIDED. That is correct behaviour
  (`feedback_equivalence_needs_delta_ci`): the rule refuses to call equivalence it cannot see. But it means batch 1
  cannot say "this breaker does nothing".
- **The registered cost of a decided batch is not 108 GPU-h.** Look 2 is where R1 gets its power, and it adds
  2 seeds × (A1, D1, A2, C0) ≈ 35 GPU-h on production and ≈ 38 GPU-h on the toy. The guards add the G-E readers
  (≈ 1.1 GPU-h per arm final, F-RV-7). The full decided picture is ≈ **190 GPU-h**.

---

## 2. Findings, ranked by severity

### CRITICAL

**F-RV-1. Nothing enforces ONE pin across the batch, and the sheet reads as one pin PER ARM.**
- **The problem.**
  - The sheet sets `<PIN>` = "main HEAD at launch" (§1, §3, F-LS-6). Over 4.5 days of landings (dozens of commits
    landed on main on 2026-10-19 alone), each arm launched at "HEAD at its launch" lands on a different commit.
  - `launch_x35.sh` checks that the pin CARRIES the required commits (pre-check 2). It never checks that the pin
    EQUALS the pin of the X35 arms already created.
  - §5.6.1 registers "one pin per BATCH" and "a control is valid only at its own pin"
    (`feedback_replicate_means_pin_and_seed`). An arm at a later pin than its C0 therefore has NO valid control, and
    nothing would stop it.
- **AM-1 (amendment text):** "X35 batch 1's pin `X35_PIN_1` is ONE commit, chosen before the first arm and written
  once to `x35_pin_batch1.txt` in the stage directory (sha + date + the T62 verdict that chose it). Every batch-1 arm,
  look 2 included, launches with `--pin X35_PIN_1`. `launch_x35.sh` gains pre-check 2b: the typed `--pin` equals the
  file's sha, and every existing `models/rb_x35*` run's recorded pin (`metadata.json` `pin_history`, read through
  `main.lineage`) equals it; any difference is a HOLD." The kit change is small: `kit pin-check`, with a test that
  fails on revert. Until it lands, the Training session checks the pin by hand before every GO.

### HIGH

**F-RV-2. R1 cannot decide at look 1 (§1), and the batch has no rule for what to do when the seed spread is large.**
- **The problem.** The queue spends ≈ 108 GPU-h on twelve 3-seed reads before anyone learns σ. If σ ≥ 1.2, every
  3-seed arm after the first is near-certain UNDECIDED.
- **AM-2 (a registered internal-pilot gate).** It reads VARIANCE only, never an arm's Δ̂, so it does not touch the
  error rate of any test (a Wittes–Brittain-style internal pilot). Amendment text:
  > "**G1, after row 6** (A1 and C0 × s3501–s3503). Read S for the six finals in the next GPU window (36k games, ≈ 5
  > min at the 118 games/s figure, **UNVERIFIED** for the end state; or ≈ 9 h of CPU). Compute σ̂, the pooled
  > within-arm SD of S over the six finals (4 df). Then:
  > - **σ̂ ≤ 1.2 pp:** continue the registered order.
  > - **σ̂ > 1.2 pp** (a σ̂ within 1e-9 of 1.2 counts as above, rule 8): run C0 s3504–s3505 next (they serve every
  >   arm's look 2), re-estimate σ̂ on 5 C0 + 3 A1 (6 df), and HOLD D1, A2 and the toy calibration for the
  >   orchestrator's re-plan. At that σ, a 3 + 2-seed arm cannot reach a class on an effect under ≈ 4 + 1.74 σ̂ pp.
  >   The re-plan's options: fewer arms at 8 seeds (half-width 1.26 σ at 8 v 8); or S taken as the mean over each
  >   fork's last three 2M nodes, registered before any S is read.
  >
  > A1's own look-1 R1 is read as registered either way."
- **The zero-training pre-read.** Before any GPU is spent, `models/rb_x26_s2601` already holds every 1M node from
  280M to 380M: one seed of X26's OWN plain continuation past P\*.
  - Score its 290M, 300M, 320M, 340M and 380M nodes against R (5 × 6,000 games: ≈ 4 GPU-min, or ≈ 7 h of CPU).
    That shows how far S wanders in 10–100M of plain training at the plateau.
  - It is DESCRIPTIVE only. Its pin is `9c033489`, so it has no T62, its pool is not reseeded, and it resumed after
    the A3 stop. It never enters R1, but it tells the orchestrator before the first GO whether σ is likely near 0.6
    or near 2.

**F-RV-3. The toy transfer calibration (toy A1 + toy A2, ≈ 26 GPU-h, plus the toy C0) cannot record agreement unless
production ALSO reaches a class.**
- **Why.** §5.2's rule is: "TRANSFERS when the class is the same on both architectures; UNDECIDED on either side is
  'not shown', never agreement." At σ 1.2, a production arm reaches any class after both looks only:
  - ≈ 10 % of the time at a true 0;
  - ≈ 14 % at a true +3 pp.

  A2's role is "the plausible null", and NULL is the class the design reaches least. At σ 1.2 the joint P(NULL on
  both sides) is ≈ 0.01.
- **AM-3:** "Toy A1 (resp. toy A2) launches only after production A1 (resp. A2) has a FINAL class (GAIN, NULL or
  LOSS). An UNDECIDED production read ends that arm's calibration as NOT SHOWN, with no toy fork, because no toy
  result could change it. The toy C0 still runs (X47 needs it)."
  - What this loses: §5.1's descriptive Δ_b(P) − Δ_b(T), whose interval would span both sides' widths anyway.
  - What it saves: ≈ 26 GPU-h in expectation, with no information lost under the registered rule.
  - If the orchestrator wants the toy pair as RANKING data anyway, record it as such. It is not a calibration run.

**F-RV-4. X47 has no look 2 and no implemented readout.**
- **(a) Seeds.** `kit.X47_SEEDS = TOY_SEEDS[:3]` ("no look 2 registered"), while the bot screen §1.3 says "Looks come
  at 3 and 5 seeds". At look 1 alone, P(GAIN) at a true +5 pp is 0.33 (σ 1.2) or 0.65 (σ 0.8); at +4 pp it is
  0.14 / 0.33.
- **(b) Readout.** `readout.ARMS` is A1–A6. The s_eq node selection (the PRIMARY at equal effective learn time), the
  matched-steps SECONDARY and the PER-STEP ONLY verdict class are in no module. The toy-crosses `time_to_step.py` is
  a measurement script, not a registered estimator.
- **AM-4:** "X47 registers look 2: X47 s3604–s3605 against toy C0 s3604–s3605, at X35_PIN_1, launched only if X47's
  look 1 is CONTINUE. **B-X47-READ** (s_eq from Σ(`collect_ms` + `update_wall_s`) restart-aware, the rule-8 node rule
  of bot screen §1.2, and the verdict classes of §1.4, each self-checked on constructed cases) lands before X47's
  first READ, not its launch." If the orchestrator declines look 2, the sheet should say that X47 at 3 seeds is a
  RANKING read: a GAIN is reportable, and an UNDECIDED says nothing.

**F-RV-5. D1's registered rule is anticonservative on H and near-unpassable on P at 3 seeds.**
Source: [`sim_d1.py`](review_2026-10-19/sim_d1.py) and [`sim_d1_od.py`](review_2026-10-19/sim_d1_od.py), which use
`main.x35.d1`'s own functions. The panel levels 0.395 / 0.40 / 0.42 are ASSUMED; heldout's is from the X26 read.
- **H** pools Kakuna games over the seeds as if they were binomial, and the 2-of-3 sign rule is weak (P = 0.5 under
  the null). False GAIN at a true Δ of 0 after both looks:

  | seed SD | false GAIN |
  |---|---|
  | 0 pp | 3.6–4.5 % |
  | 2 pp | 8.5 % |
  | 4 pp | 20.4 % |

  The seed SD of a Kakuna win rate across fork seeds is **UNVERIFIED**. The ledger's own caveat on
  `best_response_gap`'s pooled cell is the same: "treats seed-to-seed spread as binomial noise, so its CI is too
  narrow".
- **P** is non-inferiority: the Welch lower bound must be > −2.0 pp. At a true Δ_P of 0 it passes with probability:

  | σ | 3 v 3 | 5 v 5 |
  |---|---|---|
  | 0.8 | 0.32 | 0.83 |
  | 1.2 | 0.15 | 0.44 |
  | 1.6 | 0.08 | 0.26 |

  So a D1 that truly helps on unseen teams and costs nothing in-pool reads **TRADE**, not ADOPT-CANDIDATE, most of
  the time at look 1.
- **AM-5 (before any D1 read; no D1 data exists, so this is a pre-data change):** "H's Newcombe half-widths are
  multiplied by √max(1, φ̂). φ̂ is the overdispersion: the between-seed variance of the per-seed equal-weight panel
  means, over its binomial expectation, averaged over the two arms (df 2(n−1)). It is self-checked on constructed
  cases." Simulated effect:
  - false GAIN held at 3.0 / 5.8 / 5.8 % (seed SD 0 / 2 / 4 pp);
  - power at +5 pp: 0.90 / 0.75 / 0.50. That loss of power is honest: it is the seed variance the current interval
    hides.

  Separately, the sheet should say that D1's ADOPT-CANDIDATE realistically needs look 2, because of P.

**F-RV-6. K9(b), the behaviour-policy check, is read "at every restart and daily"; an X35 arm is ≈ 4.3 h with no
restart.**
- **The problem.** With the 24 h restart interval, an arm can start and end between two daily reads. Under `warn`, a
  crossing is only LOGGED (F-LS-9), so a contaminated arm would enter R1 unnoticed. A crossing in b-ON but not in C0
  is exactly the asymmetric confound a matched control cannot absorb.
- **AM-6:** "Before each next arm's GO (and as the last act of the batch), the K9(b) ceilings of the arm that just
  ended are read over **steps > its fork step only** (the fork INHERITS the parent's TB at steps ≤ the fork step,
  `gen3_tb_inherit_v1`). An arm with any judged max ≥ 1e-3, p99 ≥ 1e-4 or a non-finite value is VOID for every read
  and is re-run at the same pin and seed after the cause is fixed. The read is a deterministic tool call
  (`kit k9b-check --run-dir`, exit 0 / 1, rule 8 at 1e-9 relative), not a hand reading of `tb_read`."
- **Checked here (inheritance cannot false-fire).** X26's inherited rows ≤ 280,000,170 hold 2,852 updates, with
  judged max 2.61e-4 and p99 max 2.66e-5: none at either ceiling ([`k9b_x26.py`](review_2026-10-19/k9b_x26.py)).
  A whole-TB max read today would therefore not fire falsely on a production fork. It would still MASK nothing only
  because the post-fork rows are included.

### MEDIUM

**F-RV-7. The G-E readers have no argv (F27), and G-E is a required guard.**
- **The problem.** `readout.decide` needs every guard for ADOPT-CANDIDATE, and an absent guard is INCONCLUSIVE. An A1
  GAIN therefore cannot reach a verdict until a best responder has been trained on A1's s3501 final and on C0's
  (5M steps, ≈ 1.1 GPU-h each). Ruling C4 deferred only R0. The readers are a different gap and are also unbudgeted.
- **Fix:** "Register and dry-run the R2 / G-E reader argv (the pinned OFFENSE archetype, `--rounds`) before the first
  READ. Budget ≈ 1.1 GPU-h per (arm, seed-3501 final), C0 included: ≈ 4.4 GPU-h production, the same on the toy if
  the toy reads G-E."

**F-RV-8. The seeded self-play pool is X26's pool AS OF 380M, filtered, not its pool at P\*; and no check makes it
identical across arms.**
- **What is in the pool.** `models/rb_x26_s2601/snapshots/` holds 20 snapshots today: 14 at ≤ 280M (164, 166, 176,
  194, 200, 220, 224, 228, 236, 244, 246, 252, 254, 268M) and 6 after P\* (286–376M). T54 drops the 6, so every
  production fork seeds **14** snapshots. The newest pre-P\* selves (270–280M), which the pool held at P\*, were
  evicted after P\* and are gone.
- **This is matched across arms**, because X26 has ended, so it biases no Δ. But:
  - (i) C0 is NOT "X26 trained on, unchanged": its pool is older and sparser, which is itself a little like A1's
    archive. That dilutes A1's contrast and should be declared.
  - (ii) The toy parent X34 is still LIVE (212M+). If it is ever resumed between toy arms, its pool moves and the
    toy arms seed different sets.
  - `kit.check_pool_seed` checks the step filter only, not the seeded SET.
- **Fix (kit):** "At the call, `refpop freeze` (or a new `kit pool-freeze`) records the parent's seeded set (names +
  sha256) after the step filter. `pool-seed-check` FAILs if a fork's `pool_seed.json` set differs. X34 is not
  resumed until the last toy arm ends." Also add to the sheet's C0 row: "C0 = P\* continued with the 14-snapshot
  filtered pool (164–268M), under T62", not "unchanged".

**F-RV-9. X47's equal-time PRIMARY has two biases nobody has measured.**
- **(a) A bias against X47.** The seeded pool is GROWN to depth 4. The opponents' identity rounds compute nothing new,
  yet X47's collect pays for their forwards, so s_eq lands earlier than the learner's own cost justifies. It is
  conservative for a GAIN, but it inflates PER-STEP ONLY.
- **(b) Load sensitivity.** `collect_ms` is CPU-bound for ≈ 40 % of a toy iteration (§1.1), and other agents' load
  moves it, so s_eq depends on box load during each fork. Ruling C2's interleaving halves the exposure but does not
  remove it.
- **Fix:**
  - record load1 and `gpu_wait_ms_per_host_step` per window (§5.4.3 already asks for it; the kit does not do it);
  - report s_eq on BOTH the registered Σ basis and an update-only basis (Σ `update_wall_s`); a class that differs
    between the two is UNDECIDED (§5.4.3's flip rule);
  - split ρ (X47's wall time per step over C0's) on the first 1M into collect and update, so the opponent-growth
    share is quantified.
- **CLOSED 2026-10-20 by the orchestrator (AM-X47-9; [`X47_READOUT_REVIEW.md`](X47_READOUT_REVIEW.md) §1.2,
  [launch sheet](BATCH1_LAUNCH_SHEET.md) §0).** Both biases are measured and small: (a) the grown pool costs ≤ ≈ 2.3M
  steps of s_eq, and (b) load costs ≤ ≈ 1.2M steps per seed. The update-only s_eq stays a REPORTED node, never read for
  S. **No flip rule.**

**F-RV-10. P\* is a Tier-1 plateau CANDIDATE.**
- **The problem.** The owner's two-tier rule (`design_league_decisions.md`, 2026-10-03) declares a plateau only when
  Tier 1 is flat AND the cycle monitor (Tier 2) shows no cycling. P\* was called on Tier 1 alone; Tier 2 landed on
  main only afterwards (`6730430b9`, offline: `python -m main.cycle_monitor`, `main.plateau status`).
  X35's verdicts are Δ against C0 and survive this. But A1's stated mechanism (cycling) and the "plateau-breaker"
  framing depend on it.
- **Fix:** run Tier 2 (`main.plateau status` / `main.cycle_monitor`) on X26's 230–380M nodes before the first GO if
  the CPU allows, else beside the first read, and record the verdict beside every X35 read. It
  is descriptive and does not gate the launch.

**F-RV-11. R1 leans toward A1 by construction (F20), and the X-block has one member (C5 open).**
- **The problem.** Five of R's six members are X26 nodes adjacent to A1's archive. An A1 GAIN could be
  "trained against the test set's neighbours".
- **Fix:**
  - rule C5 = (b), adding T\* to the X-block, if its band cell fits before the window;
  - for A1, report S_A (A-block) and S_X (X-block) separately with R1;
  - an A1 GAIN with Δ̂_X ≤ 0 is reported to the owner as "ARCHIVE-ADJACENT GAIN", not as a plain ADOPT-CANDIDATE.

**F-RV-12. C1 (the T62 pin) must be ruled BEFORE the first launch; this review agrees with (a).**
- The arguments for (a):
  - T62 is paid by b-ON and C0 alike;
  - FP32_ALT does not exist, so (b) idles the GPU for days;
  - the toy is τ-free on the forward that was checked.
- What to declare: T62 changes the τ inputs relative to the code that TRAINED P\* (pin `9c033489`), so every fork,
  C0 included, starts with a small shared input shift. A comparison of C0 with X26's own 290–380M nodes (F-RV-2's
  pre-read) carries that shift.

### LOW

- **F-RV-13. The cost header.** "≈ 108 GPU-h" is look 1 only. A decided batch is ≈ 190 GPU-h (§1). The sheet should
  say so, so that the GO is given against the real commitment.
- **F-RV-14. Document inconsistencies:**
  - §5.6.3 says A2's PFSP "starts cold: every factor is 1", but F31 / T54 say the kept snapshots' `pfsp_win_rates`
    carry the parent's LATEST values.
  - The bot screen says X47 looks at 3 and 5 seeds; the kit registers 3.
  - The sheet's disk figures (289 GiB, a 26 GiB margin) predate the 2026-10-19 18:38 cache reclaim (286.8 → 464.0
    GiB, `339cf18dc`). F-LS-5 is now loose rather than tight: re-read `df` at the call.
- **F-RV-15. Inherited TB rows.** K9(b) is safe (F-RV-6). Every OTHER in-run descriptor the read cites must still be
  restricted to steps > the fork step until tonight's meter fixes land (TECH_DEBT P1 "every INHERITED TensorBoard
  row"). That covers B6's Tier 1 / Hodge on each arm's ladder, `eval/win_rate_vs_bots`, and the lag curve.
- **F-RV-16. Ordering and drift.**
  - Every production arm and toy A1 / A2 stops at a FIXED STEP. For them, the GPU order matters only through
    wall-clock-dependent training logic, and none was found: no restart falls inside a 4.3 h arm, `data/` is
    frozen, and the code is pinned (**UNVERIFIED** beyond that reading).
  - So the ABBA order buys little for them, and the order can be chosen for INFORMATION (§3).
  - X47 is different: its primary is time-based (F-RV-9), so ruling C2's interleaving stays.
- **F-RV-17. The data freeze.** `data/` stays frozen from the first arm to the last look-2 fork: about 1.5 weeks with
  look 2 and the toy block. That queue cost should be named in the GO.

---

## 3. A cheaper, more decisive order (AM-7, the orchestrator's call)

| stage | forks | GPU-h | what it decides |
|---|---|---|---|
| P1 | A1 s3501, C0 s3501, C0 s3502, A1 s3502, A1 s3503, C0 s3503 (as now) | ≈ 26 | A1 look 1; the first σ̂ |
| **G1** | the S read of the 6 finals (GPU window) | ≈ 0.1 | AM-2's variance gate |
| P2 | C0 s3504, C0 s3505 (moved forward; registered look-2 seeds) | ≈ 8.7 | σ̂ on 5 C0 seeds; every arm's look-2 control is ready |
| P3 | A1 s3504, s3505, only if A1's look 1 is CONTINUE | ≈ 8.8 | A1's FINAL class: the top-ranked arm decided first |
| P4 | D1 s3501–3503, A2 s3501–3503 (if G1 passes) | ≈ 26 | look 1 for both |
| T1 | toy C0 s3601, X47 s3601, X47 s3602, toy C0 s3602, toy C0 s3603, X47 s3603 (C2's X47 / C0 order, without toy A1) | ≈ 30.5 | X47 look 1 |
| T2 | toy A1 / toy A2 × 3 each, only if production A1 / A2 has a FINAL class (AM-3) | 0–26 | the calibration, when it CAN be read |

- **The totals.** Before the conditionals: ≈ 91 GPU-h. With every conditional stage: ≈ 126. The current queue is 108
  for look 1 alone.
- **What the reorder buys.** The top-ranked arm gets a FINAL class within ≈ 44 GPU-h, and the batch learns its noise
  floor before committing the remaining ≈ 65.
- **What it gives up.** ABBA for D1 and A2 against C0, which F-RV-16 argues is cheap to give up for fixed-step arms.
  An early verdict on A2's "plausible null", which the rule cannot give anyway (F-RV-3).
- Every stage runs at X35_PIN_1 (AM-1). C6 (X44 between the blocks) is unaffected.

---

## 4. What was NOT verified (standing rule 7)

- **The run-to-run SD of a 20M fork's S** is unmeasured. Every power figure is conditional on the 0.8–1.6 pp bracket
  of §1, which is taken from from-scratch runs. The first real number is G1's.
- **The Kakuna seed SD** (F-RV-5) and the unfamiliar panels' levels are ASSUMED.
- **The 118 games/s GPU read rate** is for the pre-end-state net (`eval_and_rating.md`, already **UNVERIFIED** in
  §5.6.4).
- **"No wall-clock-dependent training logic"** (F-RV-16) is a reading of the queue and the launch path, not a code
  audit.
- **That X26's 270–280M snapshots were in its pool at P\*** is inferred from the cap of 20 and the promotion cadence.
  Today's directory shows only that they are absent now.
- **Tonight's builds were not read**: the Tier-2 cycle monitor landed (`6730430b9`) after this review's analysis and
  was not run here; the inherited-TB meter fixes are not on main at `6730430b9`.
- **Rule 8 in the simulations.** Each simulation uses continuous draws, so no draw lands within 1e-9 of a boundary;
  the kit's own functions apply rule 8 on the cross-check.
