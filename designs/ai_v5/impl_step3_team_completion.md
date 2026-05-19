# Implementation: Step 3 — Team Completion Model

This step builds a masked-slot prediction model that, given the opponent's revealed
Pokémon mid-game, samples plausible complete teams for the unrevealed slots. It is the
world-sampling step for MCTS: at the start of each search trajectory, sample one complete
team hypothesis from this model and run the trajectory in that fully-observed world.

## Motivation

At most points in a Gen 3 OU game, three to five of the opponent's six Pokémon are
unrevealed. Naive MCTS cannot simulate rollouts without knowing the opponent's full team.
The reference paper (for Gen 4 Random Battles) addressed this by calling the server's
team-generation procedure. For Gen 3 OU, teams are human-built sets, not procedurally
generated — there is no server procedure to call. Instead, we train a model that learns
the joint distribution over Gen 3 OU team compositions from the replay corpus and self-play
data, then sample from it at MCTS time.

Raw usage statistics give only marginal distributions and miss the joint structure that
defines real teambuilding: a Skarmory nearly always comes with Blissey; a Spikes-and-Roar
Skarmory implies a physical wall elsewhere; weather-clear teams don't run Tyranitar. A
learned model captures these correlations without hand-coding them.

---

## Data Sources

Three sources, in increasing richness:

| Source | Size | Notes |
|--------|------|-------|
| 770 curated teams (`data/teams/`) | ~48K examples | Available immediately; no training needed |
| Self-play JSONL (`--team-log`) | Scales with training time | Logged by `Gen3Env.reset()` |
| Scraped ladder replays (`replays/showdown/`) | Depends on collection time | Most realistic distribution |

Each complete team of 6 yields 63 training examples (all bitmask patterns for 1–5 masked
slots). Mask pattern is re-sampled each epoch so the model sees different masking on each
pass. Start training with the 770-team pool (immediate); add self-play and ladder data as
they accumulate.

---

## Architecture

### Frozen Backbone

The PPO checkpoint (BC-then-RL from Step 2) already encodes team-level structure through
its embedding tables, role encoder, and synergy attention. These weights are loaded and
frozen — `requires_grad=False` throughout. This gives the completion model a strong
representation of what each Pokémon is and how it relates to teammates, at zero
additional training cost.

Frozen components:

| Component | Output dims | Source in features_extractor.py |
|-----------|-------------|----------------------------------|
| Species embedding | 32 | `self.species_emb` |
| Move embedding | 16 | `self.move_emb` |
| Item embedding | 16 | `self.item_emb` |
| Ability embedding | 16 | `self.ability_emb` |
| Type embedding | 16 | `self.type_emb` |
| Move processor | 32 per slot | `self.move_net` |
| Role encoder | 128 per Pokémon | `self.role_encoder` |

Loading: extract these modules by name from `torch.load(checkpoint.zip)`. Copy weights
into the completion model. No PPO training state (optimizer, rollout buffers) is loaded.

### Trainable Head

Three new components, all randomly initialized:

**1. Mask token** — a single learned 128-dim embedding substituted in place of unknown
slot role tokens (analogous to `[MASK]` in BERT). Initialized to small random values.

**2. Completion transformer** — 2-layer `nn.TransformerEncoder` (4 heads, 128-dim model,
256-dim feedforward) over the 6 role token slots. Both observed and masked slots attend
to each other, so revealed Pokémon condition the predictions for hidden slots:
"given Skarmory in slot 0 and Blissey in slot 2, what belongs in slot 4?"

**3. Output heads** (applied only to masked slots):
- Species: `Linear(128 → num_species)` → softmax
- Item: `Linear(128 → num_items)` → softmax
- Ability: `Linear(128 → num_abilities)` → softmax
- Moves ×4: `Linear(128 → num_moves)` → softmax (independent, not autoregressive)

### Training Objective

BERT-style masked slot prediction. Loss is cross-entropy summed only over masked slots —
unmasked slots contribute zero loss.

```
L = Σ_{i ∈ masked} [ CE(species_i, label_i)
                    + CE(item_i, label_i)
                    + CE(ability_i, label_i)
                    + Σ_{j=0}^{3} CE(move_i_j, label_i_j) ]
```

---

## Training Stages

**Stage 1 — Bootstrap on 770 curated teams:**
- Frozen backbone, only transformer head + output heads train
- LR: 1e-3, batch: 64, epochs: 200
- Expected outcome: learns core Gen 3 OU co-occurrence patterns (Skarmory→Blissey,
  Tyranitar→Sand team, etc.)

**Stage 2 — Scale on self-play and ladder data:**
- Optionally unfreeze role encoder at LR: 1e-5; new head at LR: 3e-4
- Batch: 256, continuous training as new data arrives
- Expected outcome: learns human-specific biases and real ladder distribution

---

## MCTS Integration

At the start of each MCTS trajectory:

1. Encode revealed opponent slots through the frozen backbone → role tokens.
2. Fill unrevealed slots with the learned mask token.
3. Run completion transformer → per-slot softmax distributions.
4. Sample one complete team hypothesis: for each unrevealed slot, sample species/item/ability/moves
   from the per-slot distributions independently.
5. This sampled hypothesis becomes the "true" opponent team for the duration of this trajectory.

Trajectories see different hypotheses, so the MCTS visit counts aggregate over the
uncertainty about the opponent's team. The action selected at the root is optimal in
expectation over plausible worlds.

**Stochastic vs. argmax**: always sample stochastically at trajectory time. Argmax
produces the same hypothesis every trajectory, eliminating the uncertainty averaging that
makes PIMC work.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/agents/team_model/team_serializer.py` | `Pokemon → Showdown text` for `--team-log` |
| `src/agents/team_model/dataset.py` | `TeamCompletionDataset` — loads all three sources, random masking |
| `src/agents/team_model/completion_model.py` | Architecture, frozen backbone loader, forward pass |
| `src/agents/team_model/train.py` | Staged training script |
| `src/agents/team_model/sampler.py` | `sample_completion(revealed_slots, n_samples) -> list[Team]` — MCTS interface |

## Files to Modify

| File | Change |
|------|--------|
| `src/agents/training/gen3_env.py` | Add `team_log_path` arg, log both teams in `reset()` before `super().reset()` |
| `src/main/train_rl_agent.py` | Wire `--team-log` CLI arg |

---

## Verification

1. **Stage 1 qualitative checks:**
   - Given only Skarmory → top-3 species predictions include Blissey
   - Given Tyranitar → weather-setter species score low in remaining slots
   - Given Spikes Skarmory + Blissey → Roar Pokémon scores higher than Taunt Pokémon

2. **Perplexity baseline**: held-out perplexity on 10% of the 770-team pool should be
   lower than a flat marginal-usage-stats baseline.

3. **Sampling sanity**: 100 sampled completions given "Skarmory revealed" should show
   Blissey appearing in > 50% of samples (reflecting its real ladder co-occurrence).

4. **MCTS smoke test**: in the Step 4 debug run, confirm that `sampler.sample_completion()`
   is called at the start of each trajectory, returns a valid 6-mon team, and that the
   sampled teams show meaningful diversity across trajectories (not all identical).

---

## Final State

Step 3 is complete when:
- Stage 1 training has converged on the 770-team pool
- Qualitative checks pass (Skarmory→Blissey, Tyranitar→sand team)
- `sampler.sample_completion()` returns valid teams in < 5ms (fast enough for MCTS)

**Ready for Step 4: MCTS**

The completion model is the world-sampling oracle MCTS depends on. Once it is trained and
fast, Step 4 can wire it into the trajectory loop.
