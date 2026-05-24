# Design: PPO Backbone — Improving Embeddings for Team Completion

The team completion model (Step 3) loads and freezes embedding tables from a PPO
checkpoint — species (32-dim), move (16-dim), item (16-dim). Its quality is bounded by
how well those frozen embeddings encode **team co-occurrence signal**: does the species
embedding for Skarmory "know" it pairs with Blissey?

Currently there is no explicit training signal for this. PPO embeds the opponent team as
part of the observation but is only supervised by win/loss reward — embeddings learn
battle-action value, not team composition structure.

---

## Option A — Larger Species Embedding (Do Now)

### Motivation

32 dims for 387 species is tight. Each species needs to simultaneously encode identity,
base-stat profile, common sets, and team role. When the team completion model's slot
encoder concatenates `[species(32), move_pool(16), item(16)]`, the species component is
the primary signal and is width-bottlenecked.

There is a meaningful qualitative difference between Vaporeon and Suicune (similar stats,
very different team roles), between Gengar and Misdreavus, between Skarmory builds — 32
dims makes it harder to separate these while also encoding team-composition context.

### Change

`species_embedding_dim: 32 → 64` in `src/agents/observation/state_encoder.py` `get_layout()`.

Everything downstream flows automatically:
- `features_extractor.py` reads layout dims dynamically — no hardcoded dims to update
- `model_version.py` catches the shape change via `_WEIGHT_FIELDS` — old checkpoints
  are correctly rejected
- Team completion model's slot encoder input changes from `32+16+16=64` → `64+16+16=96`

### Files to Modify

| File | Change |
|------|--------|
| `src/agents/observation/state_encoder.py` | `species_embedding_dim: 32 → 64` in `get_layout()` |
| `src/agents/model/team_completion_model.py` | `slot_input_dim = 96` (64+16+16); update `nn.Linear(96, ROLE_TOKEN_DIM)` in `__init__` |

**Note:** requires a fresh PPO training run. Worth batching with any other architecture
changes before the next long run. Team completion also needs retraining from the new
checkpoint — expected regardless since Step 4 enrichment changes require a retrain.

### Verification

1. `python -m main.train_rl_agent --debug --steps 10000` — confirm `[ModelVersion] Round-trip smoke test PASSED`, no shape errors
2. Retrain team completion from the new checkpoint — `slot_input_dim=96` builds without errors; `species_top1` should meet or exceed the 32-dim baseline

---

## Option B — Auxiliary Team Completion Loss During RL Fine-Tuning (Later)

### Motivation

Even with a larger embedding, PPO's reward signal never directly teaches team
co-occurrence. During each rollout the agent already observes the opponent's partial team
(revealed slots filled, unrevealed zeroed). A lightweight auxiliary loss that predicts
which species occupy the unrevealed slots from the revealed context would inject team
composition supervision directly into the embedding training — free, without any new data
collection.

### Why Fine-Tuning Stage, Not From the Start

Early in RL training the policy is still learning basic move selection; opponent team
observations don't strongly reflect real Gen 3 OU team compositions. During fine-tuning
(policy already converged on basic strategy), battles are more representative of real play
and the co-occurrence signal in opponent observations is much cleaner.

The auxiliary loss weight (0.05 range) needs careful tuning against win-rate stability.
Doing this when RL is already stable and the baseline is well-characterized makes it far
easier to detect regressions and tune the weight correctly.

### Implementation

New file: `src/agents/training/aux_team_completion_loss.py`

```python
class AuxTeamCompletionCallback(BaseCallback):
    """on_rollout_end: predict masked opponent slots → CE loss → backprop through shared embeddings."""
```

Key design points:
- Accesses `self.model.policy.features_extractor.species_embedding` directly — same
  `nn.Parameter` object, gradients flow through to the shared weights
- Lightweight 2-layer transformer + species CE head sharing these embedding weights (no
  separate copy — the whole point is to train the embedding)
- Extracts opponent team slots from the rollout buffer, randomly masks 1–3 revealed slots,
  computes masked-prediction CE loss, scales by `aux_weight`, runs a separate backward
  pass after the PPO update step

Wire via `--aux-team-loss-weight 0.05` in `train_rl_agent.py` (default 0.0 = off).

### Files to Modify

| File | Change |
|------|--------|
| `src/agents/training/aux_team_completion_loss.py` | New: `AuxTeamCompletionCallback` |
| `src/main/train_rl_agent.py` | Add `--aux-team-loss-weight` flag; wire callback |

### Key Risk

Too high a weight pulls the embedding toward team-composition representation at the cost
of action-value encoding, degrading win rate. Start at 0.05 and monitor TensorBoard
`win_rate` vs baseline. If win rate drops >2%, halve the weight.

### Verification

1. Enable at weight 0.05; run ~500K RL steps; TensorBoard `win_rate` should not drop >2%
2. Run team completion eval before/after; `species_top1_by_ctx` with 1-context slot should
   show the clearest improvement (least context = most reliance on co-occurrence priors in
   the frozen embedding)

---

## Relationship to BC Pre-Training (Step 2)

BC pre-training (Step 2 of ai_v5) is the most principled long-term solution: training
on human replay data forces embeddings to encode team composition through action imitation,
not just reward. When Step 2 is complete, retrain team completion from the BC+RL
checkpoint. The improvement from a BC-warm-started embedding is expected to substantially
exceed either option above.

Option A and Option B are interim improvements for the current RL-only regime.
