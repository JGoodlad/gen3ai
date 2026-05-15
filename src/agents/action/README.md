# Gen3 Action System

Translates between the RL model's 11-action discrete space and poke_env `BattleOrder`.

## Action Space

| Index | Meaning |
|---|---|
| 0–5 | Switch to team slot 0–5 |
| 6–9 | Use move in slot 0–3 |
| 10 | Struggle |

## Design: Crash Over Corruption

The system enforces a strict "crash over corruption" contract — ambiguous or stale state raises immediately rather than silently sending a wrong action.

### Decision Context Latch

`Gen3ActionMasker.get_mask()` pins the current server slot ordering onto `battle._gen3_decision_context` at observation time. `Gen3ActionMapper.action_to_order()` consumes that snapshot. If the server state has moved between observation and action (move IDs changed within the same turn), a `RuntimeError` is raised with a diagnostic message.

This prevents the subtle race condition where poke-env processes a background server message and mutates `last_request` while the model is executing.

### Invariants

- **No fallbacks**: every action index maps to exactly one source of truth (the latched context). There is no "try last_request if context is missing" path.
- **Duplicate species check**: if the team contains duplicate species (Species Clause violation or state corruption), the system crashes.
- **Struggle is explicit**: action 10 is only valid when the server returns `struggle` in `available_moves`.

## Tests

| File | Type | What it covers |
|---|---|---|
| `fuzz_test_unit.py` | Unit (standalone script) | Simulates 2000 random battle states including forced switches, disabled moves, and mid-turn request changes; verifies the latch provides 100% protection |
| `fuzz_e2e_test.py` | E2E (requires server) | Real battles vs RandomPlayer; asserts mask legality and that no server-rejected actions occur |
| `telemetry_e2e_test.py` | E2E (requires server) | Monitors for silent mid-decision state updates in live battles |
