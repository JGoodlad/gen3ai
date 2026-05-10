# Pokémon AI Observation Space: Entity-Based Architecture

This document defines the structured observation space for a Reinforcement Learning agent designed to play competitive Pokémon. The architecture prioritizes **generalization** through a mix of learned embeddings and explicit physical constants.

---

## 1. The Pokémon Module (Per-Mon)
Each Pokémon (Active and Benched) is represented by this block.

### A. Intrinsic Traits (Static)
*   **Species ID:** (Embedding) A learned vector representing the "meta-profile" (common moves, typical roles).
*   **Base Stats:** (6x Floats) Normalized HP, Atk, Def, SpA, SpD, Spe. Allows the model to understand the power level of unseen Pokémon.
*   **Typing:** (2x Embeddings/One-Hot) Represents the elemental identity and defensive profile.
*   **Possible Abilities:** (3x Embeddings) The legal abilities for that species. This allows the model to play against the "expected value" of an opponent's potential traits.

### B. Dynamic State (Variable)
*   **Current HP:** (Float) Current health as a percentage (0.0 to 1.0).
*   **Status Condition:** (One-Hot) [None, BRN, PAR, SLP, FRZ, PSN, TOX].
*   **Status Counters:** (Float) Turn counters for Sleep or Toxic progression.
*   **Stat Boosts:** (7x Integers/Floats) Current stages for Atk, Def, SpA, SpD, Spe, Accuracy, and Evasion (ranging from -6 to +6).
*   **Volatile Flags:** (Binary) Boolean flags for Taunt, Confusion, Encore, Substitute, etc.

---

## 2. The Move Module (The "Action" Context)
Used for your 4 current moves and the "Last Move Used" slots.

*   **Move ID:** (Embedding) Captures secondary effects and complex mechanics (e.g., pivoting, priority).
*   **Base Power:** (Float) Scaled value of the move's raw damage.
*   **Accuracy:** (Float) Probability of hitting (0.0 to 1.0).
*   **Type:** (Embedding) Used for calculating STAB and effectiveness.
*   **Category:** (One-Hot) [Physical, Special, Status].
*   **Last Move Used (Self):** (Embedding) Crucial for Choice-item detection and mechanics like Torment.
*   **Last Move Used (Opponent):** (Embedding) Used for prediction and revealing the opponent's strategy.

---

## 3. The Global Field State
Context that affects every entity on the battlefield.

*   **Weather:** (One-Hot) [None, Sun, Rain, Sand, Snow] + (Float) Turns remaining.
*   **Terrain:** (One-Hot) [None, Electric, Grassy, Misty, Psychic] + (Float) Turns remaining.
*   **Side Hazards:** (2x Sets)
    *   **Stealth Rock:** (Binary)
    *   **Spikes:** (Integer 0-3)
    *   **Toxic Spikes:** (Integer 0-2)
    *   **Sticky Web:** (Binary)

---

## 4. The Action Mask
The "Legal Guardrail" applied to the network output.

*   **Mask Vector:** (Binary Array) A 1D array matching the total action space. `1` for legal moves/switches, `0` for illegal actions (out of PP, fainted, trapped, etc.).

---

## Summary of Representation Strategy

| Data Type | Used For | Reasoning |
| :--- | :--- | :--- |
| **Embeddings** | Species, Moves, Abilities, Items | Handles high-cardinality data and learns relational "meta" nuances. |
| **One-Hot** | Weather, Status, Type, Category | Efficient for low-cardinality, mutually exclusive categories. |
| **Floats** | HP, Stats, Turn Counters | Direct mathematical relationships that the network can multiply/add naturally. |
| **Binary** | Hazards, Volatiles, Action Mask | Simple on/off switches for specific game rules. |

---

## 5. Future Work: Architectural Roadmap

To evolve this agent from a standard reinforcement learning bot to a Grandmaster-level AI, we must systematically implement advanced architectures. Each improvement should be added and tested in isolation to measure its specific impact on win rate and computational overhead.

*Note: **Training Cost** refers to the time/compute required to update the weights during the learning phase. **Inference Cost** refers to the compute required to generate a single move during a live battle.*

### Phase 2.0: Baseline Architecture (Current State)
*   **Architecture:** Multi-Layer Perceptron (MLP) with Learned Embeddings.
*   **Reward:** Shaped Win/Loss + Small Intermediate Crumbs.
*   **Training Cost:** **Low**. Easily trains on a single consumer GPU in hours.
*   **Inference Cost:** **Negligible**. Executes in microseconds. 

---

### Phase 2.1: Forward Dynamics Model (Standard Curiosity)
Instead of hand-coding rewards for damage, we add a secondary neural network that attempts to predict the next `Observation State` based on the current state and chosen action. The agent is rewarded based on its prediction error (Curiosity).

*   **How it Helps:** Eliminates human bias. The agent naturally explores every game mechanic to "surprise" itself, solving the sparse reward problem automatically.
*   **How it Hurts:** Susceptible to the "Noisy TV Problem." Pokémon has inherent randomness (damage rolls, critical hits, status chances). The model cannot perfectly predict random math, so it generates constant errors. The agent might learn to spam *Hydro Pump* just to farm the unpredictable variance.
*   **Training Cost:** **Low-Medium**. Requires calculating the loss for a second, parallel MLP during the backpropagation step.
*   **Inference Cost:** **Zero**. The dynamics model is only used to generate rewards during training. It is completely disabled during live battles.

---

### Phase 2.2: Random Network Distillation (RND)
To solve the "Noisy TV" variance problem from Phase 2.1, we replace the Forward Dynamics Model with RND. We create a completely random, frozen neural network (Target), and a second network (Predictor). The Predictor tries to guess what the random Target network will output for a given state. 

*   **How it Helps:** Because the Target network is completely deterministic, unpredictable damage rolls don't change its output. The agent is only rewarded for finding genuinely *new* game states (like a rare weather condition), completely ignoring uncontrollable RNG.
*   **How it Hurts:** It can be difficult to tune the scaling of the RND rewards. If the curiosity reward is too high, the agent might forget to actually try and win the battle.
*   **Training Cost:** **Low**. Passing data through two simple MLPs is computationally cheap.
*   **Inference Cost:** **Zero**. Like standard curiosity, RND is purely a training mechanism.

---

### Phase 2.3: Latent State Prediction (MuZero / Dreamer Style)
Instead of predicting the raw 150-number Observation Array, we pass the Observation through an Encoder to create a highly compressed **Latent State**. The Dynamics Model now predicts the *next* Latent State, not the raw numbers.

*   **How it Helps:** Forces the AI to understand the "concepts" of the game rather than memorizing exact math. It learns that 42% HP and 46% HP are functionally the exact same concept ("Healthy enough to survive a hit"). 
*   **How it Hurts:** Adds a layer of "blurriness." If the encoder over-compresses the data, the AI might miss a critical 1% HP threshold needed to secure a KO.
*   **Training Cost:** **Medium**. Requires training an Encoder, a Dynamics Model, and a Reward Predictor simultaneously.
*   **Inference Cost:** **Low**. During live play, the Observation just passes through the Encoder before hitting the main policy brain.

---

### Phase 2.4: Transformer-Based World Model (The End State)
We abandon the MLP entirely. We treat the battle as a sequence of tokens (`[Turn 1] -> [Action] -> [Turn 2]`). The Transformer World Model uses Self-Attention to look at the entire history of the battle to predict the future.

*   **How it Helps:** Ultimate generalization and perfect memory. The agent can remember that the opponent's Tyranitar revealed a Choice Band on Turn 3, and use that information on Turn 65. Furthermore, the agent can use the World Model to "dream" (simulate) future turns inside its own latent space without needing the actual Pokémon Showdown simulator, allowing for Monte Carlo Tree Search (MCTS) planning.
*   **How it Hurts:** Incredibly complex to build and tune. Requires massive amounts of self-play data to prevent overfitting.
*   **Training Cost:** **High**. Attention mechanisms scale quadratically $O(N^2)$ with the sequence length. While Pokémon battles are relatively short (<100 turns), training Transformers from scratch requires significant GPU VRAM and time.
*   **Inference Cost:** **Medium-High**. During a live battle, every single turn requires passing the *entire history of the match* through the attention blocks. While easily handled by a GPU, it is noticeably heavier than an MLP.