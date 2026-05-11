# Gen3 RL Pipeline: Design Review & Roadmap (v2.3)

This document summarizes the current state of the Gen3 reinforcement learning pipeline, including an updated deep analysis of the 8.3M step training run, and outlines a progressive roadmap for architectural upgrades.

## 📊 Training Performance Analysis: The 8.3M Step Baseline

Based on the expanded 8.3M step TensorBoard logs, we have a much clearer picture of the agent's current capabilities:

1. **The Breakthrough:** The plateau we observed at 2.8M steps (-25 reward) was successfully broken! By 4M steps, the agent learned new strategies and pushed the reward to **-15**.
   - *What -15 means:* A consistent -15 reward indicates the agent is no longer being crushed. It is consistently knocking out 4-5 opponent Pokémon and likely securing a respectable number of outright victories against the `SimpleHeuristicsPlayer`.
2. **Value Network Confidence:** The `explained_variance` has climbed to ~0.69. This is an excellent score for a stochastic environment like Pokémon. It means the model accurately predicts whether it is currently in a winning or losing position 69% of the time.
3. **Policy Convergence:** The `entropy_loss` climbing from -1.4 to -0.8 shows the agent is becoming highly confident in its decisions. It is exploring less and exploiting its learned strategies more.
4. **The New Plateau:** For the last ~3 million steps (from 5M to 8.3M), the reward has oscillated between -15 and -20 without sustained upward momentum. The agent is highly confident (low entropy) but cannot seem to optimize further against this opponent using its current "Flat MLP" brain.

## 🛑 Why a Progressive Architectural Upgrade?

The Flat MLP architecture (~3,500 concatenated features into a single Linear layer) has reached its hard limit. It has memorized specific patterns to reach -15, but it lacks the structural capacity for generalized relational reasoning (e.g., dynamic threat assessment).

Instead of replacing the entire brain at once, we will upgrade the architecture **progressively**. This gives us a "story" for each upgrade: we can train, measure the impact on the reward curve, and verify the model learns faster or reaches a higher peak before adding the next piece of complexity.

---

## 🚀 Progressive Roadmap: The Brain Upgrades

We will execute these upgrades one by one, using the current 8.3M step / -15 reward as our benchmark.

### Step 1: The Shared Move Processor (The Generalizer)
**The Problem:** Currently, the agent has 4 separate networks for its 4 move slots. It doesn't inherently know that "Move Slot 1: Earthquake" is the exact same attack as "Move Slot 3: Earthquake" on a different Pokémon.
**The Upgrade:** Route all moves through a single, shared embedding network before stitching them into the Pokémon vector. 
**The Goal:** Drastically reduce parameter count and force the model to learn a generalized understanding of moves. We expect this to increase sample efficiency (steeper learning curve) but maybe only a minor bump in peak reward.
**High-Level Implementation:** 
- In `features_extractor.py`, extract all move features (ID, Type, Power, Accuracy) into a tensor of shape `[Batch, 12_pokemon * 4_moves, move_features]`.
- Pass this entire tensor through a new `MoveNetwork` (a small MLP).
- Reshape the output back to `[Batch, 12, 4 * hidden_dim]` and stitch it back into the Pokémon vector. This forces the model to use the exact same weights to evaluate a move regardless of which slot or which Pokémon it belongs to.

### Step 2: The Pokémon Role Encoder (The Synergy Builder)
**The Problem:** The Flat MLP receives a massive concatenated vector of Species, Stats, Items, and 4 Moves all at once. It has to figure out the synergy from scratch every time (e.g., struggling to learn that high Special Attack + Fire Blast = Special Wallbreaker).
**The Upgrade:** Implement a small, shared neural network (a Pointwise Feed-Forward layer) that processes each Pokémon vector individually *before* any team-level comparisons happen.
**The Goal:** Force the network to compress `[Species + Stats + Moves]` into a dense "Role Token". This teaches the agent the concept of Pokémon "Sets" (e.g., Physical TTar vs Special TTar) universally, regardless of which slot the Pokémon occupies.
**High-Level Implementation:**
- After Step 1, take the stitched vector (e.g., ~284 dimensions) for each Pokémon.
- Pass each Pokémon through a shared `PokemonRoleEncoder` MLP (e.g., `nn.Linear(284, 128)`).
- This creates 12 distinct "Role Tokens" of shape `[Batch, 12, 128]`, which perfectly prepares the data for the Attention layers in the next steps.

### Step 3: Active Matchup Attention (The Duelist)
**The Problem:** The Flat MLP treats the active Pokémon and the benched Pokémon with the same structural weight. It has to figure out on its own that indices 0-132 (Our Active) and 798-930 (Their Active) are the most important.
**The Upgrade:** Implement a localized Attention layer that explicitly compares our Active Role Token against their Active Role Token.
**The Goal:** Allow the agent to immediately calculate "Who wins the 1v1?" without the noise of the benches. We expect this to break the -15 plateau as the agent stops making unforced errors in the active matchup.
**High-Level Implementation & The Static Ordering Nuance:**
- *Crucial Rule:* We **never** sort the team arrays. Slot 1 remains Slot 1, avoiding state-flickering and ordering bugs.
- Instead of hardcoding index 0, we dynamically isolate the active tokens using PyTorch operations on the `is_active` flag (e.g., `active_idx = torch.argmax(active_flags, dim=1)`).
- We use `active_idx` to pluck `our_active_token` and `their_active_token` from the 12-Token sequence.
- Use a `nn.MultiheadAttention` layer where `our_active` acts as the Query, and `their_active` acts as the Key and Value.
- This creates a specific "Matchup Context" vector that tells the model exactly how our active Pokémon fares against the opponent's active threat. We concatenate this Matchup Context with the rest of the state before the final linear layer.

### Step 4: Full Team Cross-Attention (The Grandmaster)
**The Problem:** The agent cannot plan ahead. It doesn't look at the opponent's active Salamence and scan its *own* bench to find a counter.
**The Upgrade:** Implement a full Cross-Attention mechanism where the Active matchup queries both the allied bench (for safe switch-ins) and the opponent's bench (for switch predictions).
**The Goal:** This is the holy grail. The agent should begin making prediction-based switches and preserving key counters for late-game sweeps.
**High-Level Implementation:**
- Treat the 12 Pokémon Role Tokens as a Sequence `[Batch, 12, 128]`.
- Add learned Positional Embeddings to each token so the model knows which token is "Our Active", "Our Bench 1", "Their Active", etc. (Wait, with static ordering, the positional embeddings just denote Slot 1 through 6 for each team, and the active flag denotes the active status).
- Pass the entire sequence through a `nn.TransformerEncoder`. The Self-Attention mechanism inside the Transformer allows every Pokémon to mathematically "look" at every other Pokémon on the field to assess threats and counters before outputting the final feature vector.

---

## 🛠 Next Steps

Once we complete the progressive architecture upgrades (Steps 1-4), we will circle back to **Closing Blind Spots** (Adding Screens/Volatiles to the observation) and **Self-Play** to push the agent to expert-level play.
