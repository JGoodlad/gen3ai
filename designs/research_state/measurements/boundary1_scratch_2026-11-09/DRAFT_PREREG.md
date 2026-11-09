# BOUNDARY-1's from-scratch run: DRAFT PRE-REGISTRATION (2026-11-09)

**Status: DRAFT. NOT REGISTERED.** No run named here exists and no game has been played. The reasoning is
[`DRAFT_DESIGN.md`](DRAFT_DESIGN.md) (cited as "design §n"); this file is only what a registration would freeze.
It becomes a PREREG when the five steps of §10 are done, and not before.

**The owner picks one option** (`../../../STATUS.md` §6.5): **B** (the bundle), **S** (width alone) or **F** (the
same eight runs crossed 2 × 2). Everything below is common to the three unless a line names an option.

Abbreviations: P\* = `rb_x26_s2601` @ 280,000,170. home72, H (`heldout72_hpfix`), F (`forum72`), P72 (`p72`) are the
registered panels of `../../heldout_panels.json`; U = H, F, P72 at equal weight. Kakuna = Metamon's best public
model. pp = percentage points. A **look** is a read at one matched step.

## 1. The question and the hypothesis

Does the new arm, trained from scratch at BOUNDARY-1's code, beat the production recipe trained from scratch at
the same code, head to head, at 100M steps?

- **Option S:** the new arm is 256 wide (6,258,079 parameters against 2,609,183).
- **Option B:** the new arm is 256 wide AND trains on `n1_proc_v1` at share 0.5 from step 0.
- **Option F:** width and breadth are two factors, each read as a main effect; their interaction is REPORTED.

Direction: one-sided for a gain, with a symmetric LOSS class. The size the run is powered for is **+4 pp** in N
(§4); the backlog's own forecast for width is +1.5 pp, which this run reads as a WIN with probability ≈ 0.2
(design §0, "Honest expectation").

## 2. The arms

Every run: `python -m main.launcher --arch production --device cuda --log-level periodic --seed <s>`, on ONE pin
(§3), with ten PPO epochs and `ent_coef` 0.05 from `--arch production` unless design §9's screens moved the base
before registration (then the moved value is in the base of EVERY arm and is named in §10's fill).

| arm | typed beyond the line above | B | S | F |
|---|---|---|---|---|
| **C** control | nothing | 4 seeds | 4 seeds | 2 seeds |
| **W** wide | `--d-model 256 --trunk-ffn-dim 512 --allow-nonproduction-arch` | — | 4 seeds | 2 seeds |
| **Br** breadth | `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both` | — | — | 2 seeds |
| **WB** wide + breadth | both lines | 4 seeds | — | 2 seeds |

- **Seeds:** 5101, 5102, 5103, 5104, the same numbers in both arms of B and S. In F: C and WB take 5101, 5102; W and
  Br take 5103, 5104. Spare seed, used only by §7's replacement rule: 5105.
- **Names** follow `src/utils/era.py`; the registrar fills them (proposal: `rb_b1c_`, `rb_b1w_`, `rb_b1br_`,
  `rb_b1wb_` + `s<seed>`).
- **`--d-model` is the flag of branch `mf5-width-lever`** (`536aed01b`), not yet on main; §3 fixes the pin after it
  lands. The breadth line and the control line pass `python -m main.checkargs` at `0ecd1b4b4`.
- **If X60-T reads ADOPT-CANDIDATE** before registration, `--human-data teams:hr_spec_v1.r3@0.5` joins the base of
  every arm (the backlog's routing), the breadth line cannot be typed with it, and options B and F are VOID: the
  registration is option S.

## 3. The pin, and the checks before the first launch

- **One pin for all eight runs:** a commit on main that carries the width flag and E13. E9 is in iff it is on main
  when the pin is cut. A change to main after the pin is cut does not move it.
- Before the first launch, in order, each a refusal if it fails:
  1. `python -m main.checkargs --argv "<argv>"` for each arm: launches, and every recipe knob matches the fresh
     recipe except the arm's typed lever.
  2. `python -m main.launcher --dry-run <argv>` for each arm at the pin.
  3. **The wide arm's GPU pilot** (design §8): step time, bf16 compile, peak memory. Its measured cost a step over
     the control's is the ratio ρ. **If ρ > 1.9 the registration stops and the budget returns to the owner.**
  4. The head-to-head engine plays one cell of a 128-wide model against a 256-wide model and its self-check passes.
  5. The stop-and-resume check of §4.

## 4. The schedule and the looks

- **Phase 1:** every run to 50,000,000 steps, in the order C, new, C, new, … by ascending seed (F: C, W, Br, WB by
  ascending seed within an arm). **Phase 2:** every run resumed to 100,000,000 in the same order.
- **The stop-and-resume check.** The kit shows on a two-segment CPU smoke that a run stopped at a target and resumed
  has the same resolved config, schedules and optimizer state as one that ran through. If it cannot, the fallback is
  fixed here: each run goes straight to 100M in the same order, and the 50M look is taken once on the first two
  completed seeds an arm with the same rule on its own degrees of freedom.
- **Looks:** 50M (harm stop only), 75M (REPORTED), **100M (the one decision)**. A look's checkpoint is the first one
  a run saved at or after the look's step. A look is played only when every run it needs has reached it.
- A run that faults is resumed from its last checkpoint under the launcher's own rules; it is never restarted from
  step 0 and never replaced except by §7.

## 5. Meter 1, decisive: the head-to-head cross

- **Cells.** B and S: every new-arm run against every control run, 16 cells. F: every pair of runs from different
  arms, 24 cells.
- **Games a cell:** 1,000 mirrored pairs on home72 and 500 on each of H, F, P72 (`python -m main.h2h play-many
  --cells <plan> --pairs <n> --panel <panel>`), both sides greedy, one engine at the runs' pin, one request per
  look and panel so a re-run resumes.
- **A cell's score** is N = ½ · (its rate on home72) + ½ · (the mean of its rates on H, F and P72), in percent, from
  the row run's side.
- **The statistic (B, S):** `main.h2h.cross.cross_stat` on the 4 × 4 matrix of N. Δ̂ = mean − 50. V̂ = s²_rows / 4 +
  s²_columns / 4, where s² is the sample variance of the four row (column) means and **each s² is floored at
  (1.76 pp)²**, the measured seed SD (design §3). A 90 % interval, t on 6 df (1.943).
- **The statistic (F), one per factor:** rows are the factor's four ON runs, columns its four OFF runs. Δ̂ as
  above. s²_rows is the mean of the two within-arm sample variances of the row means (the rows of one arm are a
  group), each floored at (1.76 pp)²; s²_columns the same. V̂ = s²_rows / 4 + s²_columns / 4; t on 4 df (2.132).
  **Width's verdict is this statistic. Breadth's verdict is §6's** (its gain is claimed on unfamiliar teams
  against Kakuna), with this statistic as its in-house cost.

| class at 100M | condition |
|---|---|
| **WIN** | the interval's lower end > 0 |
| **LOSS** | its upper end < 0 |
| **NO-USEFUL-GAIN** | neither, and its upper end < +4.0 pp |
| **UNDECIDED** | none of the above |

- **The 50M look:** STOP-HARM iff the interval's upper end < −2.0 pp; otherwise CONTINUE. No other 50M outcome
  exists. The 75M value is reported and decides nothing.
- **Deterministic edges.** The floor makes V̂ > 0 always. A bound within 0.005 pp of its threshold is reported as
  AT-BOUNDARY and takes the conservative class (not WIN, not LOSS, not NO-USEFUL-GAIN, not STOP-HARM).
- **The breadth guard (B, and F's WB cell).** Δ̂ is also reported on F alone. A WIN whose F-only Δ̂ ≤ 0 is
  **WIN-HAZARDED** (hazard M4: 62 % of H's and 56.5 % of P72's full sets are in `n1_proc_v1`) and is not adopted
  without the owner.
- **REPORTED, never deciding:** Δ̂ per panel; the equal-compute read (the new arm at the first checkpoint where its
  cumulative GPU-seconds reach the control's at 100M, against the control at 100M, on cells already defined); the
  control arm against X26's snapshot at the matched step (one seed on X26's side).

## 6. Meter 2, confirming: Kakuna on U, clustered by team

- Every 100M final plays Kakuna on H, F and P72: 720 games a panel a final (36 units of 20), each of a panel's 72
  teams the same number of times, at the anchor's registered regime (`../../../ops/EXTERNAL_ANCHORS_SOP.md`;
  `python -m main.anchors`).
- **Statistic:** the new arm's mean rate minus the control's, on U. Interval: a two-way bootstrap, 10,000 draws,
  seed `20261109`: TEAMS resampled within each panel on ONE index set shared by every arm and contrast, and SEEDS
  resampled within an arm. 95 %, percentile.
- **Use, B and S:** it confirms or contradicts §5's class.
  - WIN and the Kakuna interval's upper end < 0 → **IN-HOUSE-ONLY**, to the owner.
  - Not WIN and the Kakuna interval's lower end > 0 → **KAKUNA-ONLY**, REPORTED, never adopted on this run.
- **Use, F, breadth's verdict:** GAIN iff the interval's lower end > 0 on F alone (the clean panel); the U value is
  reported with hazard M4. Breadth is adopted only if GAIN and net = Δ̂_home72 + Δ̂_F ≥ 0 on this meter (the owner's
  r = 1, `../../../STATUS.md` §6.4).
- Resolution ([`power.log`](power.log) part 2): it detects ≈ 2.7–3.3 pp with power 0.80 at a Kakuna seed SD of
  1.0–1.5 pp. It is not sized to find +2 pp.

## 7. Validity, exclusions, replacement

- A run counts iff it reached the look's step at the pin, its sidecar audit is clean (`python -m main.sidecar_audit`)
  and its resolved config equals its arm's (`python -m main.checkargs models/<run>`).
- **Replacement.** A seed whose run cannot reach the look for a reason that is not the arm's lever (a hardware
  fault past resume) is replaced by seed 5105 in EVERY arm that used that seed number, from step 0. A run that
  diverges or cannot train BECAUSE of its lever is not replaced: it is the result, reported, and the arm is read
  with the cells it has only if at least three seeds an arm remain; otherwise the look is INCONCLUSIVE.
- More than 25 % of a cell's games ending by timeout ⇒ that look is INCONCLUSIVE, never a class.
- No game is excluded for its outcome. No panel, cell or seed is added after a look is read except by §8's rule.

## 8. What each outcome does

| outcome at 100M | action |
|---|---|
| **WIN** (not hazarded, not IN-HOUSE-ONLY) | The two new-arm seeds nearest their arm's median Δ continue to the plateau (`python -m main.plateau tick`). Adoption is a separate read against P\*: each at its own pin, Kakuna on U, §6's statistic with teams paired. Option B then owes the ablation (design §5.5) before anything is called the cause. |
| **LOSS** | The new arm is retired as specified. The control is BOUNDARY-1's lineage. |
| **NO-USEFUL-GAIN** | S and F: the factor is retired from scratch at this size and a 100M budget. B: nothing is retired; reported as a bundle null. |
| **UNDECIDED** | To the owner with one priced option (≈ 104 GPU-h): seeds 5106 and 5107 in both arms, the same rule on 10 df, one time. No other extension. |
| **STOP-HARM** at 50M | Phase 2 runs for the control arm only. |
| any class with a hazard or INCONCLUSIVE | To the owner; no adoption. |

## 9. Power and budget

[`power.py`](power.py) → [`power.log`](power.log), seed `20261109`, 50,000 draws a cell.

| | P(WIN) at +2 / +3 / +4 / +5 / +6 pp | P(NO-USEFUL-GAIN) at 0 | false WIN at 0 |
|---|---|---|---|
| B, S: 4 v 4, seed SD 1.76 pp | 0.28 / 0.56 / 0.81 / 0.95 / 0.99 | 0.80 | 0.02 |
| the same at seed SD 2.5 pp | 0.22 / 0.40 / 0.60 / 0.77 / 0.90 | 0.57 | 0.03 |
| F, one factor: 4 v 4 on 4 df, seed SD 1.76 pp | 0.19 / 0.44 / 0.70 / 0.89 / 0.97 | 0.70 | 0.01 |
| the same at seed SD 2.5 pp | 0.17 / 0.32 / 0.50 / 0.68 / 0.83 | 0.48 | 0.02 |

**Budget to the decision, any option:** ≈ 207 GPU-h (180–227) at ρ = 1.65 (a GUESS until §3's pilot), ≈ 9 days of
the card; three looks of 80,000 games (F: 120,000) at ≈ 0.6 GPU-h each (F: ≈ 0.9) under the GPU lock between runs;
≈ 120 CPU lane-h for §6. A registration restates these with the measured ρ.

## 10. What turns this draft into a registration

1. **The owner's answer** picks B, S or F, and confirms +4 pp as the bar and r = 1 on N.
2. **The fills, each by a mechanical rule and none by a choice the run's data could inform:** the pin (§3); ρ and the
   budget (§3 item 3); E9 in or out (§3); X60-T's routing (§2); the base recipe after design §9's screens, if run;
   the run names; the stop-and-resume result (§4).
3. **The rule as code, executed:** §5's floored statistic with the panel average, §5's F statistic, §6's bootstrap
   as a `python -m main.reads` rule, each with a self-check on synthetic matrices whose class is known, and a
   0-mismatch comparison against `power.py`'s classes.
4. **One adversarial review** of the filled text (causal validity; the meter; the M4 guard).
5. **Registration** by the orchestrator: this file renamed without DRAFT, a ledger entry, and nothing changed after
   the first launch except by a dated amendment written before the read it touches.

## 11. What is not claimed, whatever the outcome

- Nothing about the plateau from the 100M look: the toy led at equal compute and lost at its plateau (design §4.2).
- Nothing about widths other than 256, or about depth, the critic, the policy tower or entropy.
- No pp against Kakuna from a pp in N: the mapping is unmeasured (design §12 item 7).
- In option B, nothing about which element caused a WIN, and nothing retired by a null.
