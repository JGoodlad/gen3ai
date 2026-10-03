# The post-M5 era plan

**Status: ALWAYS-CURRENT (owner, 2026-10-02: "write it, but lightweight").** It says what the first era on the
Rust stack is FOR and in what order. Detail lives in the docs each step points to. A decision that changes the
order or the shape updates this doc and its Decision record in the same commit.

## The principle
**One lever at a time, on top of one baseline, sized to the noise.** Two short fresh runs that differ only in
seed land about 4.9 pp apart on untaught (sizing F-SZ-10). So every lever arm gets a pre-registered meter and
bar, and either runs long enough (the X5 A/B runs 10–15M steps per arm) or is replicated. Levers stack one at a
time, never in bundles.

## The order

| # | step | what it is | meter / stop rule | detail |
|---|---|---|---|---|
| 0 | **Generalist baseline (X26)** | fresh, on the FINAL architecture (X5 landed), `recipe.fresh` at N = 256, KL early stop ON, detached ride-along heads (V ensemble, RND, A/B) | the comparator for everything below; the ride-alongs map uncertainty (ensemble), novelty (RND) and the value/advantage split (A/B) for free | `EXPERIMENT_BACKLOG.md` X26, X5 |
| 1 | **Pool + PFSP + exploiters (the population loop)** | the generalist trains against its pool, weighted by `--pfsp-scale`; exploiters train against a frozen generalist; a snapshot joins the pool only by SPRT on mirrored pairs (T6/T17) | **the best-response gap must FALL round over round** (`main.best_response_gap`) | `designs/training/exploiter_and_distillation.md`, `eval_and_rating.md` |
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
