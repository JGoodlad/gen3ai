# Embedding Implementation Specification (embedding_impl.md)

This document defines the exact bit-level and tensor-level layout for the Pokémon Showdown state extraction.

## 1. The Pokémon Vector (Per-Mon)
Each Pokémon is represented by a vector. We represent 6 Pokémon per side, with **Slot 0 always being the Active Pokémon**.

Total Dimensions per Pokémon: **132**

| Field | Sub-field | Type | Dim | Range / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Identity** | Species ID | Embedding | 32 | 1-386 (Gen 3 Dex) |
| | Item ID | Embedding | 16 | Known items in Gen 3 |
| | Item Known | Binary | 1 | 1 if revealed, 0 if unknown |
| | Combined Types| Embedding | 8 | $E(T1) + E(T2)$ |
| | Ability 1 | Embedding | 8 | Active/Primary guess |
| | Ability 1 Known| Binary | 1 | 1 if confirmed |
| | Ability 2 / 3 | Embedding | 16 | Potential legal abilities |
| **Condition** | Status | One-Hot | 7 | None, BRN, PAR, SLP, FRZ, PSN, TOX |
| | Status Turn | Float | 1 | Normalized counter (Toxic/Sleep) |
| **Moves** | Move 1-4 | Embedding | 32 | ID + Learned Metadata |
| | Move 1-4 Known | Binary | 4 | 1 if revealed |
| **Stats** | HP (%) | Float | 1 | 0.0 - 1.0 |
| | Stats (5x) | Float | 5 | Atk, Def, SpA, SpD, Spe (Base / 255) |

## 2. The Active Context (Slot 0 Only)
These fields are only populated for the Pokémon in **Slot 0** (the one on the field). For benched Pokémon (Slots 1-5), these are set to `0`.

| Field | Sub-field | Type | Dim | Range / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Boosts** | Stat Boosts (7x)| Float | 14 | 2 dims per stage (Atk, Def, SpA, SpD, Spe, Acc, Eva) |
| **Volatiles** | Volatile Flags | Binary | 8 | Confusion, Substitute, Taunt, Encore, etc. |
| **Temporal** | Turns on Field | Float | 1 | $\ln(1+T)$ |
| | Last Move Used | Embedding | 8 | Revealed ID of the previous action |

## 3. Global Environment
Context shared by both players.

| Field | Sub-field | Type | Dim | Range / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Weather** | Type | One-Hot | 5 | None, Sun, Rain, Sand, Hail |
| | Turns | Float | 1 | Normalized remaining turns |
| **Hazards (P1)**| Spikes | Integer | 1 | 0, 1, 2, 3 |
| **Hazards (P2)**| Spikes | Integer | 1 | 0, 1, 2, 3 |
| **Clock** | Turn Count | Float | 1 | $\ln(1+T) / \ln(1001)$ |
| | Screen R | Float | 1 | Reflect remaining |
| | Screen LS | Float | 1 | Light Screen remaining |

## 4. Handling Uncertainty & Unknowns

This system follows a **strict no-guessing policy** for opponent information. We only represent what has been explicitly revealed during the match or is fixed by the game rules.

### A. The "Unknown" Token (Index 0)
Every embedding table (Species, Item, Move, Ability) reserves **Index 0** as the `UNKNOWN` or `NONE` token. 
-   **Moves:** If an opponent's move has not been seen, it is set to Index 0.
-   **Items:** If an item has not been triggered (e.g., Leftovers, Choice Band reveal), it is set to Index 0.
-   **Abilities:** If the opponent's ability has not been revealed (e.g. via prompt or game effect), the `Ability 1` slot is set to Index 0.

### B. Confirmed vs. Possible Information
-   **Known Flags:** A flag of `1` (Confirmed) is only used when the information is 100% verified (your own team, or revealed opponent data).
-   **The "Possible" Slots:** For Abilities, we provide the full set of legal abilities for that species in the `Ability 2` and `Ability 3` slots. This allows the model to understand the *potential* traits of the opponent without us "guessing" which one they actually have.
-   **Moves:** We do not guess unrevealed moves. Unrevealed slots are always Index 0 until the move is used.

---

## Total Observation Space Calculation
- **Pokémon vectors:** $12 \times 132 = 1584$
- **Active Context:** $2 \times 31 = 62$
- **Global Environment:** $\approx 10$
- **Total:** **~1656 Dimensions**
