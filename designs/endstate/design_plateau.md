# Plateau detection end state — when plain training has stopped paying

**Status: ALWAYS-CURRENT SKELETON (owner, 2026-10-02).** The rule the era runs on is "plateau first"
([`era_plan_post_m5.md`](era_plan_post_m5.md)): plateau-breakers run only from a plateaued parent. This doc owns how
"plateaued" is DECIDED. A build or decision that differs updates it and its Decision record in the same commit.
Evaluation machinery: [`design_evaluation.md`](design_evaluation.md).

## 0. The definition (owner)
**Plateau = plain training gains less than 2 Elo per training GPU-hour.** Above that rate, the GPU's best use is
more plain training. Economic reading: on one GPU, an experiment displaces baseline training, so its price is the
baseline's current improvement rate × its hours, plus the information lost by testing on a still-climbing control.

## 1. The first-draft test (era plan) — and its known GAPS
Draft: once per 10M steps, the newest snapshot plays the snapshot ~7 GPU-h back (≈ 50M steps at N = 256) as a GSPRT
on mirrored pairs (H0 p ≤ 0.50, H1 p ≥ 0.52), AND the same against a fixed outside panel; plateau = both accept H0.

| # | gap (orchestrator self-review, 2026-10-02) | why it matters | TODO |
|---|---|---|---|
| G1 | **Repeated testing.** One test per 10M steps, each with α = β = 0.05, run dozens of times per run | the chance of a false "plateau" at SOME check grows with the number of checks: optional stopping again, across time | design for repeated monitoring: CUSUM with a declared average run length (ARL) to false alarm, OR alpha-spending across checks, OR a fresh independent confirmation test (two-stage, Fishtest-style) |
| G2 | **The window averages a decelerating curve.** Learning curves are concave, so a 7 h window's mean rate overstates the CURRENT rate | plateau is declared late. Conservative (we over-train), but it is a bias | add a curve-model estimate of the current slope (§3) |
| G3 | **Head-to-head vs past self is self-referential.** Co-evolving policies can cycle | "beats its past self" ≠ "stronger"; the outside panel is the guard | specify the panel read: a DIFFERENCE of two win rates (new vs panel, old vs panel), needing its own game count; derive it |
| G4 | **Elo-per-hour assumes the logistic model**, and an Elo scale anchored on one population | 2 pp ≈ 14 Elo holds near 50 % under BT; away from 50 % the mapping shifts | state the conversion used and its range of validity |
| G5 | **The noise floor AT the plateau is unknown** (4.9 pp was fresh 8M runs) | sets how small a lever effect the fork arms can detect | the paired control continuations measure it on the first plateau; record it here |
| G6 | **The AND rule's error rates combine.** Two tests must both accept H0 | lowers false-plateau risk and raises the chance of training past a real plateau | compute the combined operating characteristics by simulation (as `measurements/sprt_promotion/` did) |

## 2. Literature to review before the build (owner: a literature review in the proposal)
- **Learning-curve extrapolation and continue/stop decisions:** Domhan, Springenberg & Hutter 2015
  (extrapolating learning curves); Swersky, Snoek & Adams 2014 (freeze-thaw Bayesian optimization: model each run's
  curve and spend compute where the expected improvement is highest — our "price of an experiment" framing); Li et
  al. 2017 (Hyperband / successive halving).
- **Sequential change detection under repeated monitoring:** Page 1954 (CUSUM), Lorden 1971 / Moustakides 1986
  (CUSUM's optimality); group-sequential alpha spending (Lan & DeMets 1983).
- **Head-to-head testing practice:** Fishtest's GSPRT (Van den Bergh), already built for promotion.
- **Non-transitivity:** Czarnecki et al. 2020 (spinning tops), for why the outside panel is required.

## 3. Candidate design (TO VALIDATE, not adopted)
- **Primary:** a sequential test designed for REPEATED monitoring (G1): CUSUM on the per-checkpoint head-to-head
  increments, with a declared ARL to false alarm, or the two-stage confirmation. Mirrored pairs, background eval.
- **Secondary, the forecast:** fit a saturating curve to the strength series with uncertainty, and report the
  current slope (Elo per GPU-hour) and the expected remaining gain. It answers the owner's practical question
  ("should I check back later?") and corrects G2.
- **Guard:** the outside panel (G3).
- **Validate** on a finished run's snapshots (N0 has a full series): does the design declare a plateau where the
  ladder curve visibly flattens, and how often does it false-alarm on the climbing segments?

## 4. TODO
- [ ] Literature review (§2).
- [ ] Choose G1's repeated-monitoring design; simulate its operating characteristics (G6).
- [ ] Specify the panel read and its game count (G3); the Elo conversion (G4).
- [ ] Backtest on N0's snapshot series (§3).
- [ ] Build: the plateau test on the eval core, mirrored pairs, background eval (TASK_BACKLOG T19).
- [ ] Record the plateau noise floor from the first paired control continuations (G5).

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-02 | The plateau rate **(owner)** | < 2 Elo per training GPU-hour | a statistical band chosen without the economics | owner, 2026-10-02 |
| 2026-10-02 | First-draft test (orchestrator) | head-to-head GSPRT vs ~7 GPU-h back + panel, once per 10M steps | 0.51 per 10M steps (too fine for the window) | era plan; OPEN GAPS G1–G6 (§1) before any build |
