# Action System Improvements & Hardening

## 1. Principles of Integrity
- **Crash over Corruption:** If the system detects an ambiguous or "funky" state, it must terminate immediately with a detailed error message.
- **Single Source of Truth:** Every action index (0-10) must map to exactly one source of truth, with NO fallbacks or alternative ordering logic.
- **Fail-Fast Validation:** Action masks must be strictly validated against the current turn to prevent stale decision-making.

## 2. Hardened Ordering Strategy
To ensure "junk in junk out" is eliminated, the system will move to a rigid, single-path mapping:

### Switches (0-5)
- **Source:** The `side.pokemon` list from the most recent server request.
- **Integrity Check:** If the team contains duplicate species, the system will crash. In Gen 3 OU, this is a violation of Species Clause or an indicator of state corruption.
- **Mapping:** Index `i` always corresponds to Team Slot `i+1`.

### Moves (6-9)
- **Source:** The `active[0].moves` list from the server request.
- **Integrity Check:** If `last_request` or the `active` move list is missing when a move is required, the system will crash.
- **Mapping:** Index `i+6` always corresponds to Move Slot `i+1` as defined by the server.

### Struggle (10)
- **Source:** Dedicated struggle action.
- **Integrity Check:** Only valid if explicitly allowed by the server request (e.g., all moves disabled).

## 3. Implementation Plan: "The Great Simplification"
1. **Remove Fallbacks:** Delete the alphabetical sorting logic in both `mapper.py` and `mask_generator.py`.
2. **Enforce Request-Only Logic:** Make the `last_request` a mandatory dependency for action mapping.
3. **Add Corruption Checks:** Implement explicit checks for duplicate species and stale turns.

## 4. Verification: Real-World Fuzz Testing
To ensure the mapper is bulletproof, we will implement an integration "fuzz" test:
- **Environment:** Run against a local Pokémon Showdown server.
- **Method:** Use a `RandomPlayer` vs. `RLPlayer` (with the hardened mapper).
- **Checks:**
    - On every turn, assert that `mask_generator.get_mask(battle)` matches the legality of the `battle.available_moves` and `battle.available_switches`.
    - Verify that `action_to_order` never produces a server-rejected choice.
    - Run 1,000+ battles with random teams to catch edge cases (forced switches, trapping, choice-locks).

## 5. Completed Hardening Checklist
- [x] **Integrity-First Architecture**: Crash-over-corruption logic implemented.
- [x] **Single Source of Truth**: Removed all alphabetical fallbacks (`stable_ids`).
- [x] **Double Latch Protection**: 
    - **Moves**: Pins the server's slot IDs at observation.
    - **Pokémon**: Pins the team list ordering at observation.
- [x] **Forensic Reporting**: Hard crash on mid-decision state shifts with diagnostic reports.
- [x] **Verified**: Zero temporal desyncs detected in real-game telemetry.
