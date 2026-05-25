# AI v7 — Todo

**"Rustifying" the battle simulator.** v7 replaces the Node.js MCTS bridge (ai_v5/v6)
with a Rust battle simulator called via PyO3. The Rust sim runs in-process with near-zero
IPC overhead, enabling 50 000+ rollouts per turn rather than ~1 000 with the JS bridge.
This throughput increase unlocks two capabilities that were impractical before:

1. **Deep MCTS at inference time** — with 50× more rollouts in the same time budget,
   the agent can search far deeper before committing to a move.

2. **MCTS during training data generation** — generating PPO training samples using
   full MCTS on both sides. With the JS bridge, this would reduce throughput by ~1 000×;
   with the Rust sim, the overhead drops to ~5–10×, making it feasible at moderate scale.

The end goal: a model that can train against **either** fixed-depth search **or** full
MCTS, with the Rust sim as the common engine for both. Hidden state (opponent sleep
duration, unknown items, unrevealed team slots) is explicitly tagged via `SampledValues`
at state construction time so every MCTS rollout knows exactly what it made up.

---

## Version Context

| Version | Sim | Rollouts/turn (inference) | Training data |
|---------|-----|--------------------------|---------------|
| ai_v5 | Node.js bridge | ~1 000 | Raw policy |
| ai_v6 | Node.js bridge | ~1 000 | K=3 action sampling (shallow) |
| ai_v7 | Rust sim (PyO3) | 50 000+ | Full MCTS both sides |

---

## Step 1 — Showdown Bridge

A persistent Node.js subprocess that drives the Showdown sim library directly —
no server, no WebSocket, no `npm run showdown`. Just `require()` the already-compiled
`dist/` and expose a newline-delimited JSON protocol over stdin/stdout. This is the
oracle for all subsequent fuzz testing.

See `designs/ai_v7/impl_step1_showdown_bridge.md`.

**Deliverables:**
- `src/utils/bridge/sim_battle.js` — persistent Node.js bridge
- `src/utils/bridge/sim_battle_client.py` — Python wrapper
- `src/utils/bridge/sim_battle_client_test.py` — unit tests (mock subprocess)
- `src/utils/bridge/sim_battle_integration_test.py` — integration tests (real Node process)

---

## Step 2 — Rust Scaffold + First Fuzz

Stand up `src/sim/`, port Gen5RNG, deserialize Showdown battle state, implement
damage calc and basic turn resolution, expose via PyO3, and run the first fuzz
comparison against the bridge. The fuzz harness is Python — it holds both handles
(bridge subprocess + Rust via PyO3) and diffs their outputs.

See `designs/ai_v7/impl_step2_rust_scaffold.md`.

**Deliverables:**
- `src/sim/` — Rust crate (`state.rs`, `prng.rs`, `damage.rs`, `turn.rs`, `python.rs`)
- `src/sim/sim_binding_test.py` — Python pytest suite for the PyO3 API
- `src/agents/mcts/rust_sim_fuzz_e2e_test.py` — fuzz harness vs bridge

---

## Step 3 — Full Gen3 Mechanics

Extend the Rust sim to cover all Gen 3 mechanics until the fuzz harness passes 10k
battles without divergence. Work is mechanic-by-mechanic guided entirely by fuzz
failures — no speculative implementation. Mechanic groups are largely independent
after the damage formula is confirmed — parallel agents can tackle status, items,
hazards, volatiles, weather, complex moves, and abilities concurrently.

See `designs/ai_v7/impl_step3_gen3_mechanics.md`.

**Deliverables:**
- `src/sim/src/turn.rs` — all mechanic implementations
- `src/sim/src/damage.rs` — formula refinements
- `src/sim/src/data.rs` — physical/special split, item tables

**Stopping criterion:** 10k random battles, zero state divergences vs Showdown.

---

## Step 4 — Encoder Dual Path

Introduce `EncoderState` as the single encoder input type. Implement adapters from
both poke-env and Rust. Oracle fuzz test: at each real battle decision point, assert
both adapters produce identical observation vectors.

---

## Step 5 — MCTS with Rust Sim (Inference)

Replace the ai_v5/v6 Node.js bridge with the Rust sim for inference-time MCTS.
`fork()` becomes `rust_state.clone()` (a memcpy — no serialization, no IPC).
Leaf evaluation is batched (accumulate K leaves → one network call, amortizing GPU
round-trip latency). `SampledValues` makes all hidden-state guesses explicit.

**Key changes vs ai_v5 MCTS:**
- `SimClient` → in-process `rust_state.clone()` (fork is free)
- `step()` → `rust_sim.step_turn(state, p1, p2)` (no subprocess, no JSON)
- Leaf batch size K replaces the per-turn neural-net call; tune to GPU latency

---

## Step 6 — MCTS in Training + Evaluation

### 6a — Training with Rust MCTS

Use the Rust sim to generate PPO training samples via full MCTS on both sides. Both
the learning agent and its opponents select training-time actions by running N rollouts
from the current state, replacing the raw policy or the shallow action sampling from v6.

With the Rust sim, the overhead is ~5–10× vs. raw policy training (compared to ~1 000×
with the JS bridge), making this feasible at large scale.

**Training modes to benchmark:**
- **Raw policy** (v5 baseline): single forward pass per decision
- **Action sampling K=3** (v6): 3 rollouts per legal action, depth 1
- **MCTS N=1 000** (v7): full tree search, depth 5, one-side (learning agent only)
- **MCTS N=1 000 both sides** (v7): full tree search for both players in self-play

Each mode produces different training data quality at different throughput costs.
Measure win rate vs. v6 league after 15M steps per mode.

### 6b — Evaluation + Tuning

Benchmark rollout throughput vs. the ai_v5/v6 Node bridge. Ablate `max_depth` and leaf
batch size K. Measure win rate vs. v6 league.

**Targets:**
- ≥ 50k rollouts/turn at inference (vs ~1k in ai_v5/v6)
- Win rate ≥ v6 MCTS baseline with equal compute
- Training throughput ≥ 50k steps/hour with MCTS-quality samples

---

## Design Questions

- **Rust sim scope**: implement only mechanics that appear in the 32 sample teams, or
  aim for full Gen3OU coverage? Start narrow (sample teams cover ~80% of mechanics),
  expand as fuzz failures demand.
- **Leaf batch size K**: profile on dev hardware. Likely 64–256.
- **MCTS parallelism**: multiple Python workers each with their own in-process Rust
  sim, or single-process Rust MCTS calling back to Python only for leaf eval?
  Single-process Rust is simpler; try it first.
- **Training MCTS mode**: one-side (our agent only) vs. both sides? One-side is a
  cleaner experimental control; both sides is closer to AlphaZero-style self-play.
- **Curriculum**: start training with raw policy, then switch to MCTS after N steps?
  Or use MCTS from the start? MCTS from the start is more principled; raw-policy warmup
  may stabilise early training.
