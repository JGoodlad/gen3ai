# Technical Blueprint: Gen 3 (ADV) OU Latent Representations

This document defines the optimized embedding sizes and architectural logic for a Gen 3 (ADV) Pokémon AI. These dimensions are tuned for the 386-Pokémon meta to maximize learning speed and inference throughput on consumer hardware (i7 CPU).

---

## 1. Per-Pokémon Identity (Static & Semi-Static)
Each Pokémon on the field and in the party is represented by this 110-dimensional vector.

| Feature | Count ($N$) | Dim | Rationale |
| :--- | :--- | :--- | :--- |
| **Species ID** | 386 | **32** | Captures base identity and regional/form niches. |
| **Item ID** | ~300 | **16** | Clusters by function (Recovery, Power, Berry). |
| **Types (Shared)** | 18 | **8** | **Order-Invariant Summation:** $f(T1, T2) = E(T1) + E(T2)$. |
| **Abilities (2x)** | 78 | **16** | 8 dims per slot. Covers primary and possible secondary traits. |
| **Moves (4x)** | 354 | **32** | 8 dims per slot. Captures power, type, and Gen 3 split logic. |
| **Base Stats** | 6 | **6** | Normalized floats ($Stat / 255$). |
| **Total Identity** | -- | **110** | **Total input for 6 Pokémon: 660 Dims** |

---

## 2. Order-Invariant Type Logic
To prevent the model from treating "Water/Ground" and "Ground/Water" as different states, we use a **Summation Layer**.

*   **Implementation:** A single `nn.Embedding(18, 8)` table is shared.
*   **The Math:** The vectors for Type 1 and Type 2 are added together. 
*   **Result:** The network sees the same "Defensive Profile" regardless of move order or internal ID sorting. "None" types (Index 17) act as a zero-identity in the summation.

---

## 3.1 Dynamic Game State (The "Live" Board)
These features represent the current battlefield context and are concatenated with the Pokémon identities.

| Feature | Categories | Dim | Rationale |
| :--- | :--- | :--- | :--- |
| **Status Condition** | 7 | **4** | (None, SLP, PSN, TOX, PAR, BRN, FRZ). |
| **Weather** | 5 | **4** | (None, Rain, Sun, Sand, Hail). |
| **Stat Stages (7x)** | 13 | **14** | 2 dims per stage (Atk, Def, SpA, SpD, Spe, Acc, Eva). |
| **Gender** | 3 | **2** | (M, F, None). Relevant for ADV *Attract/Female-Magneton* meta. |
| **Current HP/PP** | -- | **5** | Normalized percentage floats (Current / Max). |

---

### 3. Temporal Feature Breakdown (The "Clock")
To handle the 1,000-turn Showdown limit without breaking the neural network's distribution, we use **Logarithmic Scaling**.

| Feature | Formula | Rationale |
| :--- | :--- | :--- |
| **Global Turn** | $\ln(1 + T) / \ln(1001)$ | High resolution for early game; squashes 1,000 turns into $[0, 1]$. |
| **Stalemate Clock** | $\min(\Delta T_{\text{KO}} / 50, 1.0)$ | Signals when a game has stalled; triggers "Aggressive Breakout" logic. |
| **Screen Turns** | $T_{\text{rem}} / 5$ | Reflect/Light Screen (linear is fine here as it's capped at 5). |
| **Status Counter** | $T_{\text{sleep}} / 7$ | Critical for Gen 3 "Sleep Clause" and Sleep Talk tracking. |

---

---

## 4. Technical Note: The ADV Physical/Special Split
**Critical Implementation Detail:** In Gen 3, the Physical/Special split is **Type-Based**, not Move-Based. 
*   **The Learning:** By feeding the 8-dim Type vector into the same layer as the Move embedding, the model is forced to learn that a *Shadow Ball* (Ghost) uses the **Attack** stat, while *Flamethrower* (Fire) uses the **Special Attack** stat.

---

## 5. Future Work: The Recurrent & Belief Roadmap

Current Model: **Reactive** $f(\text{game state})$.
Future Model: **Proactive** $f(\text{sequence})$.

### Phase A: The "Suitcase" (GRU Latent History)
*   **Goal:** Add a **Gated Recurrent Unit (GRU)** layer to the brain.
*   **Function:** Carry a 128-dim "Hidden State" vector across turns.
*   **Benefit:** Enables memory of "Choice Band" locks and revealed items even after switches.

### Phase B: Belief State Modeling
*   **Goal:** A "Prediction Head" to guess hidden opponent data.
*   **Mechanism:** When an opponent outspeeds or hits harder than expected, the internal "Belief Vector" snaps to a new coordinate (e.g., "Choice Band" or "Speed EVs").

### Phase C: Mamba/SSM Integration
*   **Goal:** Replace the GRU with a **Selective State Space Model**.
*   **Why:** Prevents "Memory Blur" in 100-turn ADV stall wars. Mamba can "pin" critical info (like a revealed HP Ice) indefinitely while ignoring "chip damage" turns.

---

## 6. Execution Strategy (Vectorization)
*   **Environment:** 10–50 parallel games on i7 via `SubprocVecEnv`.
*   **Inference:** Batch all observations into a single `[Batch_Size, ~700]` tensor.
*   **Optimization:** Target **PokeJAX** or **Poke-Env** for the rollout engine to maximize Steps-Per-Second.