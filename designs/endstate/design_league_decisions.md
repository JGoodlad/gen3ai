# League decisions end state — promotion, eviction, plateau

**Status: ALWAYS-CURRENT SKELETON (owner, 2026-10-02).** This doc covers the three DECISIONS made about the
population: who joins the pool (promotion), who stays (eviction), and whether plain training has stopped paying
(plateau). They share one problem, measuring strength in a game that is not fully transitive, and they use one
machinery: the game ledger, the scheduler and the estimators in [`design_evaluation.md`](design_evaluation.md). A
build or decision that differs updates this doc and its Decision record in the same commit. Era context:
[`era_plan_post_m5.md`](era_plan_post_m5.md). This doc absorbs the earlier `design_plateau.md`.

## 0. Strength in a non-transitive game
- **Strength and transitivity are two axes.** Spinning tops (Czarnecki et al. 2020): real games are transitive overall,
  with cyclic width SMALL at low skill, WIDEST at middle skill (many viable styles beating each other) and narrowing
  toward the top. Self-play can stall in the wide middle.
- **A head-to-head against ONE past self measures transitive progress only.** In a cyclic regime it misleads both
  ways: it can beat its recent past forever while getting no better (a treadmill), or lose to it while drifting
  between holes.
- **So strength is measured against the POPULATION:**
  - (a) the win rate vs the **Nash mixture of the archive** (our own frozen past snapshots; max-entropy Nash
    averaging, Balduzzi et al. 2018). Real progress raises it; treadmill motion nets out;
  - (b) **exploitability**, the best-response gap (`main.best_response_gap`): a treadmill keeps it high, real
    robustness shrinks it (the PSRO line, Lanctot et al. 2017);
  - (c) a cheap SHAPE signal: the win rate vs the snapshot k back, as a function of k. Monotone rising = transitive
    progress; non-monotone (beats k = 1, loses to k = 5) = cycling.

## A. Promotion — who joins the pool
- **Built:** GSPRT on mirrored pairs (Fishtest's pentanomial method + a minimum pair count), H0 0.50 / H1 0.55,
  α = β = 0.05, cap 1,680 pairs, against a pool frozen at the test's start, on its own seeds; no re-tests
  (`sprt_promotion.py`, `e9c5ab2d`). Default OFF; flips ON at X26.
- TODO: promotion vs the pool's NASH MIXTURE rather than the pool sampled uniformly. That is consistent with §0, and
  the matrix (§B) makes it possible.

## B. Eviction — who stays, and the cycling defence
- **Today:** a sliding window of the 20 most recent (oldest out), or opt-in spread retention. It has no cycling
  defence.
- **Design (TASK_BACKLOG T20):**
  - the tiered pool matrix (`design_evaluation.md` §3), validated against a dense audit before it is trusted;
  - **max-entropy Nash averaging** decides who stays: evict a member only when it confidently has zero weight;
  - the **HodgeRank cyclic width** vs its noise floor is the cycling meter;
  - **when cycles are wide, GROW the pool** (CPU-resident, with a PFSP-chosen GPU active set, as in the AlphaStar
    league) rather than evicting harder.
- TODO: literature review (owner) before the build; the pool cap from the measured cyclic width (spinning tops).

## C. Plateau — has plain training stopped paying?
- **Definition (owner):** plateau = plain training gains < 2 Elo per training GPU-hour, measured as the slope of
  §0(a), strength vs the archive's Nash mixture, with §0(b) the gap NOT falling. Economic reading: on one GPU the
  price of an experiment is the baseline gain it displaces, plus the information lost on a still-climbing control.
- **Three kinds of plateau.** The diagnosis picks the lever. This mapping is a HYPOTHESIS to validate:

| kind | signature | lever family |
|---|---|---|
| TRUE plateau | §0(a) flat; cyclic width narrow; drift slow; gap small and stable | capacity / algorithm: discrimination, model size, later search |
| TREADMILL ("plug one hole, another opens") | beats the recent past but not the older; cyclic width WIDE; drift fast; gap stays high | population: train vs the archive's Nash mixture, grow the pool, exploiters |
| STAGNATION | §0(a) flat; drift ≈ 0 | exploration: forks (built OFF), later entropy |

- **The trap:** calling a treadmill a plateau and reaching for forks. On a treadmill, exploration only finds more
  holes faster.
- **Open gaps in any test design:**

| # | gap | TODO |
|---|---|---|
| G1 | repeated monitoring: one test per 10M steps, dozens of times a run; false "plateau" risk accumulates | CUSUM with a declared average run length to false alarm, OR alpha-spending (Lan & DeMets 1983), OR a fresh two-stage confirmation |
| G2 | a window averages a decelerating curve, so the plateau is called late | add a fitted-curve CURRENT slope + the expected remaining gain (Domhan et al. 2015; freeze-thaw BO, Swersky et al. 2014) |
| G3 | the archive's Nash mixture changes as snapshots are added | freeze a reference archive per check, or re-solve with care; specify |
| G4 | Elo-per-hour assumes the logistic model | state the conversion and its range |
| G5 | the noise floor AT the plateau is unknown (4.9 pp was fresh 8M runs) | the paired control continuations measure it on the first plateau |
| G6 | combined error rates of the joint (a) + (b) rule | simulate the operating characteristics |

- **Validate before the build:** backtest on N0's full snapshot series. Does the design call the plateau where the
  ladder curve flattens? How often does it false-alarm on climbing segments? Which plateau kind does it diagnose?

## TODO
- [ ] Literature review (owner): spinning tops; Nash averaging; PSRO / exploitability; HodgeRank; α-Rank +
  ResponseGraphUCB; learning-curve extrapolation; freeze-thaw BO; CUSUM; alpha-spending.
- [ ] Promotion vs the Nash mixture (§A).
- [ ] T20 build with its validation gate (§B).
- [ ] Plateau: choose G1's design, specify G3, the N0 backtest, then build.
- [ ] Validate the plateau-kind → lever mapping on the first real plateau.

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-02 | Promotion **(owner)** | GSPRT on mirrored pairs, Wald bounds | overshoot-corrected bounds | `measurements/sprt_promotion/` |
| 2026-10-02 | Pool defence against cycling **(owner)** | tiered matrix + Nash averaging + Hodge meter; grow rather than evict when cycles are wide | hope; recency-only eviction | TASK_BACKLOG T20 |
| 2026-10-02 | The plateau rate **(owner)** | < 2 Elo per training GPU-hour | a band chosen without the economics | owner |
| 2026-10-02 | The plateau MEASURE (orchestrator, owner discussion) | strength vs the archive's Nash mixture + the gap not falling, with the plateau KIND diagnosed | a head-to-head vs one past self (misses treadmills) | §0, §C |
| 2026-10-02 | Doc structure (orchestrator; the owner left it to the orchestrator) | ONE doc for the three population decisions (shared problem and machinery); the SYSTEM in `design_evaluation.md` | three separate docs | they share §0 and one machinery |
