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
    progress; non-monotone (beats k = 1, loses to k = 5) = cycling. **Since 2026-10-03 this is the CYCLE MONITOR's lag
    curve** (`design_evaluation.md` §2.5): a standing instrument that grows a thinned-archive matrix one row per 10M
    check from step 0, with four signals (Hodge cyclic share, lag-curve non-monotonicity, intransitive triangles, Nash
    support size), each against a parametric-bootstrap null (owner, 2026-10-03: "knowing whether we're cycling is
    important outside of just the plateau").

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
  - the tiered pool matrix (`design_evaluation.md` §2), validated against a dense audit before it is trusted;
  - **max-entropy Nash averaging** decides who stays: evict a member only when it confidently has zero weight;
  - the **HodgeRank cyclic width** vs its noise floor is the cycling meter;
  - **when cycles are wide, GROW the pool** (CPU-resident, with a PFSP-chosen GPU active set, as in the AlphaStar
    league) rather than evicting harder.
- **The system design (2026-10-03, `design_evaluation.md` §2.4, PROPOSED) finds that zero Nash weight is the COMMON case in a near-transitive pool** (N0: the point Nash is pure on one node; F-ED-3), so "evict only when confidently zero weight" is a necessary condition that rarely binds and does not choose WHICH member leaves.
- **Proposed rule (revised after the independent review, M5; owner Q6 there): eviction is a DECLARED LEDGER READ.**
  - It decides only when the active pool's matrix is DONE for every member involved, reading `as_of` the decision
    time, with the consumed rows' digest recorded.
  - Among members with posterior P(w > 0.01) < 0.05, evict the one with the HIGHEST posterior P(the newest pool member
    beats it). Ties go to the oldest, then the lowest sha256. Grow if none qualifies.
  - The first draft's PFSP-weight tie-break is REJECTED: `--pfsp-scale` defaults to 0, so it degenerated to "oldest",
    and PFSP's p is a stateful EMA, not a ledger read.
  - The eval CYCLE's rows are NOT an input (a moving, unmirrored player whose games select promotions).
  - Evicting from the ACTIVE pool is not deleting from the ARCHIVE. Nothing a reference, an open request or the cycle
    monitor names is ever deleted.
- The literature review is DONE (`design_evaluation.md` §0a). TODO: the pool cap from the measured cyclic width (spinning tops); no banked within-run pool shows cycles above noise (F-ED-1), the multi-lineage pool is unmeasured.

## C. Plateau — has plain training stopped paying?
- **PROPOSED TWO-TIER test (owner input 2026-10-03; `design_evaluation.md` §8; owner Q8 there).** It restores the
  owner-registered head-to-head as the PRIMARY, because the reference slope needs twice the pairs (a difference of two
  estimates) and inherits the Nash mixture's instability.
  - **Tier 1:** the GSPRT newest vs the W-back snapshot (W ≈ 7 GPU-h), H0 0.50 / H1 0.52, mirrored, α = β = 0.05.
    It costs E 3.2–5.5k games per check.
  - **Tier 2:** a READ of the cycle monitor (§0(c)) plus the outside panel.
  - **Plateau ONLY when Tier 1 is FLAT AND the monitor shows no cycling AND the panel is flat, at two consecutive
    checks.** A confirmed cycle is reported as a TREADMILL, with its own levers.
  - Strength vs a frozen reference mixture (§0(a)) is the REPORTED secondary and the kind diagnostic.

  - **AS BUILT (U9a, 2026-10-06): Tier 1 only, offline.** `python -m main.plateau tick <run>` plays each due
    check's GSPRT on `main.h2h`'s engine and writes one decision row per check (GAIN / FLAT / UNDECIDED /
    INCONCLUSIVE) to the eval ledger; `python -m main.plateau status <run>` prints the run's Tier-1 status
    (`TIER1_PLATEAU` = FLAT at two consecutive checks). Tier 2 (the cycle monitor) and the panel are NOT built, so
    a Tier-1 plateau is a candidate, never a declared plateau. Detail: `design_evaluation.md` §8.1 "As built".

  The definition below is the 2026-10-02 version that this proposal would amend.
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
| G1 | repeated monitoring: one test per 10M steps, dozens of times a run; false "plateau" risk accumulates | PROPOSED: a fresh GSPRT per check + plateau only at two consecutive checks (`design_evaluation.md` §8.3); CUSUM or a confidence sequence can replace the 2-check rule later |
| G2 | a window averages a decelerating curve, so the plateau is called late | add a fitted-curve CURRENT slope + the expected remaining gain (Domhan et al. 2015; freeze-thaw BO, Swersky et al. 2014) |
| G3 | the archive's Nash mixture changes as snapshots are added | moot for the proposed primary (a head-to-head); for the reported secondary, frozen references chosen by an N0 back-test (`design_evaluation.md` §2.3) |
| G4 | Elo-per-hour assumes the logistic model | state the conversion and its range |
| G5 | the noise floor AT the plateau is unknown (4.9 pp was fresh 8M runs) | the paired control continuations measure it on the first plateau |
| G6 | combined error rates of the joint (a) + (b) rule | simulate the operating characteristics of the two-tier rule (unit U9 in `design_evaluation.md` §10) |

- **Validate before the build:** backtest on N0's full snapshot series. Does the design call the plateau where the
  ladder curve flattens? How often does it false-alarm on climbing segments? Which plateau kind does it diagnose?

## TODO
- [x] Literature review (owner): spinning tops; Nash averaging; PSRO / exploitability; HodgeRank; α-Rank +
  ResponseGraphUCB; learning-curve extrapolation; freeze-thaw BO; CUSUM; alpha-spending — DONE 2026-10-03 in
  `design_evaluation.md` §0a (with confidence sequences added as G1's first candidate, §8).
- [ ] Promotion vs the Nash mixture (§A).
- [ ] T20 build with its validation gate (§B).
- [x] Plateau Tier 1, offline (U9a, 2026-10-06): `main.plateau`, one decision row per check.
- [ ] Plateau: the owner answers Q8 (two-tier, `design_evaluation.md` §11.2); the N0 back-test; then build the rest
  (U9: Tier 2 + the panel + the 2-check rule's joint OC).
- [ ] The cycle monitor (`design_evaluation.md` §2.5, U5): a standing instrument, from step 0.
- [ ] Validate the plateau-kind → lever mapping on the first real plateau.

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-02 | Promotion **(owner)** | GSPRT on mirrored pairs, Wald bounds | overshoot-corrected bounds | `measurements/sprt_promotion/` |
| 2026-10-02 | Pool defence against cycling **(owner)** | tiered matrix + Nash averaging + Hodge meter; grow rather than evict when cycles are wide | hope; recency-only eviction | TASK_BACKLOG T20 |
| 2026-10-02 | The plateau rate **(owner)** | < 2 Elo per training GPU-hour | a band chosen without the economics | owner |
| 2026-10-02 | The plateau MEASURE (orchestrator, owner discussion) | strength vs the archive's Nash mixture + the gap not falling, with the plateau KIND diagnosed | a head-to-head vs one past self (misses treadmills) | §0, §C |
| 2026-10-03 | The plateau test: TWO TIERS **(owner input; PROPOSED, owner Q8 in `design_evaluation.md`)** | Tier 1 = the registered head-to-head GSPRT vs the W-back snapshot (primary); Tier 2 = a read of the cycle monitor + the panel; plateau only when both say so, twice; the reference slope REPORTED | the reference slope as the primary (2× the pairs, mixture instability; independent review M2) | `design_evaluation.md` §8, F-ED-16, F-ED-17 |
| 2026-10-03 | The CYCLE MONITOR as a standing instrument **(owner input)** | a thinned-archive matrix grown one row per check from step 0, four signals with parametric-bootstrap nulls | cycling read only inside the plateau test | `design_evaluation.md` §2.5 |
| 2026-10-03 | Eviction rule **(orchestrator, after review M5; PROPOSED, owner Q6)** | a declared ledger read at matrix DONE: among confidently zero-weight members, evict the one the newest member most surely beats; ties oldest, then id; grow otherwise; archive deletion separate | the PFSP-weight tie-break | `design_evaluation.md` §2.4 |
| 2026-10-02 | Doc structure (orchestrator; the owner left it to the orchestrator) | ONE doc for the three population decisions (shared problem and machinery); the SYSTEM in `design_evaluation.md` | three separate docs | they share §0 and one machinery |
| 2026-10-06 | The plateau meter's FIRST slice (plateau-meter agent, for the owner's plateau-first plan) | Tier 1 BUILT offline (U9a): the registered GSPRT newest vs W-back as a pure rule over ledger rows, `main.plateau tick / status`, a Tier-1 status beside an explicit "Tier 2 not built" caveat | waiting for U4's scheduler and U5's monitor before any plateau read (a deep run would start with no "done" signal at all) | `design_evaluation.md` §8.1 "As built", Decision record |
