# Implementation: PPO Backbone Embedding Improvements

Improve the frozen embedding tables that the team completion model (Step 3/4) loads from
the PPO checkpoint. Two options, sequenced by cost.

See full design: `designs/ai_v6/design_ppo_embedding_improvements.md`

---

## Option A — Larger Species Embedding (Do Before Next Long Run)

`species_embedding_dim: 32 → 64` in `src/agents/observation/state_encoder.py` `get_layout()`.
One additional update: `slot_input_dim = 96` in `src/agents/model/team_completion_model.py`.

Requires a fresh PPO training run. Batch with any other architecture changes.

## Option B — Auxiliary Team Completion Loss (RL Fine-Tuning Stage)

Add `AuxTeamCompletionCallback` to inject team co-occurrence supervision during RL.
Wire via `--aux-team-loss-weight 0.05` (default off).

Do this after the policy has converged on basic strategy.

---

## Status

- [ ] Option A implemented
- [ ] Option B implemented
