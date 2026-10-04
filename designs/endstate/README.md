# End-state docs — where the system is going, and why

🚨 **ALWAYS-CURRENT (owner, 2026-09-27).** Every doc in this directory is kept current with what is
decided and built: a change that differs from one updates it in the same commit, saying what changed and
why. Each doc ends with a **Decision record** — dated decisions, what was chosen, what was rejected, and
the evidence — so a new person learns the history of the core architecture here, not by archaeology.
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) states what the model IS today;
[`../CHANGELOG.md`](../CHANGELOG.md) is the append-only history; these docs state where we are going
and why each fork was taken.

## Reading order for someone new

| # | doc | what it answers |
|---|---|---|
| 1 | [`design_three_tier_environment.md`](design_three_tier_environment.md) | Why a Rust battle core, one inference tier, and the retirement of poke-env — the environment end state |
| 2 | [`program_rust_core.md`](program_rust_core.md) | How we get there: milestones M1–M7, the parity gates, the one cutover, the deletion manifest, and where we are now |
| 3 | [`design_model_management.md`](design_model_management.md) | The model side: versions, checkpoints, baselines, lineage |
| 4 | [`design_q_head.md`](design_q_head.md) | The value side's end state: the Q decomposition (V + A + B + I), the opponent pointer, fixed-mass belief tokens, the offline omniscient twin |
| 5 | [`obs_enrichment_backlog.md`](obs_enrichment_backlog.md) | What the observation carries, what it should carry next, and on what terms |
| 6 | [`design_ladder_campaign.md`](design_ladder_campaign.md) | The research stages toward the ladder: artifacts, meters, plateau tests, the training ecology |
| 7 | [`design_learner_recipe.md`](design_learner_recipe.md) | The TRAINING RECIPE re-grounded knob by knob (rollout, batch, epochs, step size, entropy, λ, critic, opponents): live value, provenance, literature, recommendation, and the migration order for the first M5 era |
| 8 | [`design_own_ppo_loop.md`](design_own_ppo_loop.md) | Owning the PPO loop: every SB3 / sb3-contrib touchpoint on the production path, the staged plan to take SB3's loop off it (identity first, then declared hooks), the equivalence plan, the estimate, and the eval-dump KL-skip finding |
| 9 | [`era_plan_post_m5.md`](era_plan_post_m5.md) | What the first era on the Rust stack is FOR, and in what order: baseline → population loop → discrimination → exploration → ladder stages (lightweight) |
| 10 | [`design_evaluation.md`](design_evaluation.md) | The eval SYSTEM (DESIGN 2026-10-03, REVISED the same day after the independent review — SOUND WITH FIXES, M1–M6 resolved — and owner input; PROPOSED pending owner Q1–Q9): the literature review; the append-only count ledger v2 (request ids, batches and families, the no-duplicate-batch invariant, outcome digests, declared readers, the backfill); one scheduler; background eval in the trainer's GPU WINDOW only (CPU worker optional, with a trigger); the budget from precision (5.3–7.9 % of wall); the standing CYCLE MONITOR; the TWO-TIER plateau test; T20 with an eviction ledger read and a derived validation; the build plan (~22.5 agent-days) |
| 11 | [`design_league_decisions.md`](design_league_decisions.md) | The three POPULATION decisions — promotion, eviction (a declared ledger read, PROPOSED) and plateau (TWO-TIER, PROPOSED) — and how to measure strength in a non-transitive game, with the cycle monitor as the standing cycling read |
| 12 | [`design_x5_belief_tokens.md`](design_x5_belief_tokens.md) | X5's build spec and registered A/B (REVISED after independent review; M2 / M3 open for the owner): fixed-mass hypothesis tokens + OTHER, presence semantics for every opponent reduction, the flat opponent pointer, the K9 golden re-bake, role calibration, the KL early stop's analysis (OFF, kept for X28), and the group-sequential head-to-head design with its simulated power |
| 12b | [`design_x5_tradeoffs.md`](design_x5_tradeoffs.md) | What X5's chosen semantics (M2 = C, M3 = (c)) give up, how each limitation will be detected, and the richer options to revisit once X5 has run |

Ranked work lives outside this directory: experiments in
[`../research_state/EXPERIMENT_BACKLOG.md`](../research_state/EXPERIMENT_BACKLOG.md), build tasks in
[`../ops/TASK_BACKLOG.md`](../ops/TASK_BACKLOG.md), debt in [`../ops/TECH_DEBT_BACKLOG.md`](../ops/TECH_DEBT_BACKLOG.md);
what we believe about the research is [`../research_state/UNDERSTANDING.md`](../research_state/UNDERSTANDING.md).
