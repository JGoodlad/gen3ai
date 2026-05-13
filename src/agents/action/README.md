# Gen3 Action System: Integrity & Mapping

This directory contains the core logic for translating between the RL model's discrete action space and the Pokémon Showdown `BattleOrder` system. It is designed with a **"Crash over Corruption"** philosophy to eliminate state-action desynchronization.

## Architecture: Decision Context Latching

To prevent race conditions (where the server state updates while the model is executing), we use a **Decision Context Latch**:

1.  **Observation Time**: `Gen3ActionMasker.get_mask()` captures a "pinned" snapshot of the server's move slots and team order.
2.  **Action Time**: `Gen3ActionMapper.action_to_order()` strictly consumes that snapshot. 
3.  **Integrity Check**: If the current server state has moved from the pinned snapshot, the system will raise a `RuntimeError` with a full forensic report rather than sending an invalid action.

## Validation Suite

### 1. Serverless Unit-Fuzz (`fuzz_test_unit.py`)
This is a high-speed validation tool that simulates thousands of random battle states (forced switches, disabled moves, status effects) and ensures the mapper never produces an invalid order.

**Run**:
```bash
PYTHONPATH=src python3 src/agents/action/fuzz_test_unit.py
```

### 2. Real-Game Telemetry (`telemetry_test.py`)
This tool plays actual battles against a random opponent and monitors for "Silent Updates"—instances where the server state changes during the decision window.

**Run**:
```bash
# Note: Requires a local Showdown server running on port 8001
PYTHONPATH=src python3 src/agents/action/telemetry_test.py
```

## Troubleshooting: Forensic Reports

When a desync is detected, the `mapper` will throw a `RuntimeError` with a diagnostic block:
- **Pinned (Observation)**: The moves the model actually "saw."
- **Current (Server)**: The moves currently in the server's request.
- **Delta**: Explains exactly what changed (e.g., "Moves changed under the model").

If you see this error frequently, check your `poke-env` queue-draining logic or reduce model inference latency.
