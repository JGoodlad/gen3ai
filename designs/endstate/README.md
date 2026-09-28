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

Ranked work lives outside this directory: experiments in
[`../research_state/EXPERIMENT_BACKLOG.md`](../research_state/EXPERIMENT_BACKLOG.md), build tasks in
[`../ops/TASK_BACKLOG.md`](../ops/TASK_BACKLOG.md), debt in [`../ops/TECH_DEBT_BACKLOG.md`](../ops/TECH_DEBT_BACKLOG.md);
what we believe about the research is [`../research_state/UNDERSTANDING.md`](../research_state/UNDERSTANDING.md).
