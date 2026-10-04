# The post-M5 era plan

**Status: ALWAYS-CURRENT (owner, 2026-10-02: "write it, but lightweight").** It says what the first era on the
Rust stack is FOR and in what order. Detail lives in the docs each step points to. A decision that changes the
order or the shape updates this doc and its Decision record in the same commit.

## The era label (owner, 2026-10-03)
Eras are named after Hoenn towns in journey order. **The Rustboro era (`rb`) starts with the X5 A/B and X26**;
every earlier run is pre-era history. The table (`src/utils/era.py`, the one source): Rustboro `rb` · Dewford
`dw` · Slateport `sp` · Mauville `mv` · Verdanturf `vt` · Fallarbor `fb` · Lavaridge `lv` · Fortree `ft` ·
Lilycove `lc` · Mossdeep `md` · Sootopolis `st` · Pacifidog `pd` · Ever Grande `eg`. A run's name carries its code
(`rb_x26_s1001`) and its `metadata.json` records an immutable `era` block; a boundary is where ratings or
anchors stop being comparable (Rustboro's is the fixed bots + the Rustboro anchor base), and the readers warn
across it. Starting the next era is one edit (`CURRENT_ERA`) plus a Decision row here.

## The principle
**One lever at a time, on top of one baseline, sized to the noise.** Two short fresh runs that differ only in
seed land about 4.9 pp apart on untaught (sizing F-SZ-10). So every lever arm gets a pre-registered meter and
bar, and either runs long enough (the X5 A/B runs 10–15M steps per arm) or is replicated. Levers stack one at a
time, never in bundles.

## Plateau first (owner, 2026-10-02)
**Plateau = plain training gains less than 2 Elo per training GPU-hour.** Above that rate, the GPU's best use is
more plain training. The owner checks back in; nothing else needs to happen.

- **Why:** on one GPU, an experiment hour displaces an hour of baseline training. So the PRICE of an experiment is
  the plain-training gain it displaces (the baseline's current rate × its hours), PLUS the information lost:
  - before a plateau, the control is still climbing fast;
  - a lever's effect is small next to that climb and next to the run-to-run floor (~4.9 pp ≈ 35 Elo on short fresh runs);
  - so plateau-breaker arms read NOT DETECTED, or the wrong sign, and the decisions persist.

  The v8 "gift" hump, both population-loop rounds and the closed 10M ladder were all plateau questions asked
  before a plateau.
- **Two kinds of lever:**
  - SPEED levers (epochs, batch, lr, architecture efficiency) are tested on fresh runs, per GPU-hour.
  - PLATEAU-BREAKERS (forks, exploration, exploiters, discrimination, team curriculum) are tested ONLY from a plateaued parent.
- **The plateau test** (DECIDED in [`design_league_decisions.md`](design_league_decisions.md) §C: the measure is strength vs the archive's NASH MIXTURE + the gap not falling, with the plateau KIND diagnosed; the head-to-head draft below is SUPERSEDED, see the gaps there):
  - once per 10M steps, the newest snapshot plays the snapshot from W ≈ 7 GPU-hours back (≈ 50M steps at N = 256);
  - it plays on mirrored pairs, as a GSPRT with H0 p ≤ 0.50 vs H1 p ≥ 0.52. 2 Elo/h × 7 h ≈ 14 Elo ≈ 2 pp;
  - that is ≈ 3–7k games, a few minutes on the eval core;
  - **AND** the same against a fixed OUTSIDE panel (frozen pool snapshots, the bots, SmallRL), because self-play can cycle;
  - plateau = both accept H0.
  - CUSUM over the checkpoint comparisons is the refinement if the onset needs to be caught sooner.
- **Plateau-breakers FORK from the plateau checkpoint, with PAIRED controls:**
  - two control continuations (seed only) give the floor AT the plateau;
  - one or two lever arms;
  - `--fork-lr` pinned and dose matched.

## The order

| # | step | what it is | meter / stop rule | detail |
|---|---|---|---|---|
| 0 | **Generalist baseline (X26)** | fresh, on the FINAL architecture (X5 landed), `recipe.fresh` at N = 256, KL early stop ON, detached ride-along heads (V ensemble, RND, A/B) | the comparator for everything below; the ride-alongs map uncertainty (ensemble), novelty (RND) and the value/advantage split (A/B) for free | `EXPERIMENT_BACKLOG.md` X26, X5 |
| 1 | **Pool + PFSP + exploiters (the population loop)** | the generalist trains against its pool, weighted by `--pfsp-scale`; exploiters train against a frozen generalist; a snapshot joins the pool only by SPRT on mirrored pairs (T6/T17) | **the best-response gap must FALL round over round** (`main.best_response_gap`). EXPLOITER RECIPE: choose teams by the generalist's RESPONSE weakness (the archetypes it loses AGAINST), at most 1–2 per archetype, 5–7 per round; FORK each from the current generalist (fork > scratch, measured); train vs a frozen snapshot at a matched budget and dose; RESET every round; promote by SPRT. PILOTING weakness is NOT an exploiter's job: see the team curriculum (backlog X30). ANTI-CYCLING: the pool matrix + Hodge cycling meter + equilibrium-weight eviction (TASK_BACKLOG T20), fed by best-effort eval (T19); both ready BEFORE this step | `designs/training/exploiter_and_distillation.md`, `eval_and_rating.md` |
| 2 | **Exploiters are OPPONENTS, never teachers** | structural: distillation is deleted (L3, `cbd20111`) | — | `deleted_flags.md` |
| 3 | **A DISCRIMINATION meter first, then its lever** | register one meter for the value's WITHIN-GAME discrimination, by phase and opponent class (exploiters especially), before choosing a lever. Candidate levers: X5's belief tokens (already in), the V ensemble, X21 calibration, exploiter-dense data | the registered meter | `design_q_head.md` |
| 4 | **Exploration that does not change the objective** | (a) opponent and team diversity (free, from step 1); (b) the FORK arm (built OFF, `gen3_fork_rust_v1`): counterfactual one-ply branches played to the end under common random numbers, subsample-eligible, enabled where the RND map shows regions the agent avoids; (c) X23 entropy anneal stays DEFERRED, because it changes the objective | a novelty or coverage read plus the step 3 meter. Exploration comes after discrimination, because the value must tell new states apart before exploring them pays | `designs/training/forks.md` §14 |
| 5 | **Main agent on the ladder teams** | the ladder campaign: about 5 iconic Smogon teams, staged | one meter plus a plateau test per stage; near goal: beat Kakuna at its best temperature on our teams | `design_ladder_campaign.md` |
| 6 | **A basin probe (X29, LATER)** | a NEW fresh generalist trained against the incumbent pool, with the pool's temperature annealed from high to 1, gated so the newcomer keeps ≥ 25 % wins (owner, 2026-10-02) | does a fresh learner converge to the incumbent's play, or find a different basin that beats it? | `EXPERIMENT_BACKLOG.md` X29 |

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-02 | The era's shape **(owner)** | generalist → PFSP + exploiters → no exploiter-as-teacher → a discrimination meter then its lever → exploration that doesn't change the objective → main-agent ladder loop → see how it goes | bundling levers; exploration bonuses in the reward | owner, 2026-10-02 |
| 2026-10-02 | Discrimination before exploration (orchestrator) | define and move the discrimination meter before enabling forks | forks first | exploring more pays only once the value tells the new states apart |
| 2026-10-02 | A basin probe **(owner)** | X29 queued after the loop is running | — | tests for a local optimum of PPO + exploiters |
| 2026-10-02 | Plateau first **(owner: "2 elo per hour feels right")** | plateau = < 2 Elo per training GPU-hour, by a registered head-to-head GSPRT (W ≈ 7 GPU-h, H1 0.52) plus an outside panel, checked once per 10M steps; plateau-breakers fork from the plateau with paired controls | plateau-breaker arms on fresh or still-climbing runs; a 0.51 band per 10M steps (too fine for the window) | the price of an experiment = the baseline gain it displaces |
| 2026-10-03 | Eras are named for Hoenn towns; the Rustboro era (`rb`) starts with the X5 A/B and X26 **(owner)** | one table in `src/utils/era.py` (name, two-letter code, order) + `CURRENT_ERA`; default run names minted `rb_…`; an explicit `--run-name` ACCEPTED AS TYPED with a warning when it lacks the prefix; an immutable `metadata.json` `era` block written at the creation save and never changed by a resume; a run without it is pre-era; `main.lineage`, the ELO headline and the critic gate's ladder read it and warn across eras | silently prefixing an explicit name (the directory would differ from the typed name quoted in scripts and the ledger); refusing an unprefixed name (breaks every convention); deriving the era from the name or a date (the record is the truth); stamping a resumed pre-era run | owner, 2026-10-03; `src/utils/era_test.py` |
