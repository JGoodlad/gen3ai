# TODO: Future Architectural Refinements (v2.1+)

This document tracks high-level architectural ideas to revisit after completing the core Step 3 (Attention) and Step 4 (Switching) milestones.

## 1. Hierarchical Move Latent (The "Refining Fusion" Model)

**Concept**: Instead of a flat concatenation of move data and battle context, move the architecture toward a two-stage hierarchical evaluation.

### Stage A: Modified Move Potential (Potency)
- **Input**: Move Data + Pokémon Status/Stats.
- **Output**: A latent representing "What can this move do right now?"
- **Reasoning**: This captures the effects of Burn (power reduction), Paralysis (execution risk), and Stat Boosts (damage potential) *before* considering the opponent or the game clock.

### Stage B: Strategic Move Utility (Utility)
- **Input**: Potency Latent + Global Context (HP, Turn, Weather, Phase).
- **Output**: Final Contextual Move Latent.
- **Reasoning**: This applies "Battle Logic" to the "Potential." It identifies if a weakened move is still the correct play given the endgame phase or if a setup move has lost its investment value due to the turn count.

---

## 2. Turn-on-Field Encoding

**Concept**: Fix the `ActiveContextEncoder` to actually encode the `active_turns` feature.
- **Motivation**: Necessary for perfect implementation of *Fake Out* and understanding the "faded" value of temporary boosts or effects.

---

## 3. Scalar Context Expansion ("Voice of the Scalars")

**Concept**: Instead of raw 1D scalars for HP, Turn, and Spikes, project them into a larger latent space (e.g., 32 or 64 dims) using a small projection network.
- **Problem**: In a 128-dim or 256-dim model, single scalars like `HP=0.85` have very low "influence" compared to multi-dimensional embeddings. They can easily be "drowned out" by the high-variance norms of larger vectors.
- **Solution**: Use an `nn.Linear` or `nn.Sequential` to expand the 12 global context scalars into a larger feature block before concatenation.
- **Extension**: Use Sinusoidal (Positional) Encoding for the **Turn Count** to help the model perceive the phase of the battle (Early/Mid/Late) with much higher resolution than a single float.
