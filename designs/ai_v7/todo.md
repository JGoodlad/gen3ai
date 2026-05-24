# AI v7 — Todo

Replace the Node.js MCTS battle sim bridge (ai_v5) with a Rust battle simulator called
via PyO3. The Rust sim runs in-process with near-zero IPC overhead, enabling millions of
rollouts per turn rather than thousands. Hidden state (opponent sleep duration, unknown
items, unrevealed team slots) is explicitly tagged at state construction time so every
MCTS rollout knows exactly what it made up.

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
failures — no speculative implementation.

**Stopping criterion:** 10k random battles, zero state divergences vs Showdown.

---

## Step 4 — Encoder Dual Path

Introduce `EncoderState` as the single encoder input type. Implement adapters from
both poke-env and Rust. Oracle fuzz test: at each real battle decision point, assert
both adapters produce identical observation vectors.

---

## Step 5 — MCTS with Rust Sim

Replace the ai_v5 Node.js sim bridge with the Rust sim. `fork()` becomes
`rust_state.clone()` (a memcpy). Leaf evaluation is batched (accumulate K leaves →
one network call). `SampledValues` makes all hidden-state guesses explicit.

---

## Step 6 — Evaluation + Tuning

Benchmark rollout throughput vs ai_v5 Node bridge. Ablate `max_depth` and leaf
batch size K. Measure win rate vs v6 league.

**Target:** ≥ 50k rollouts/turn (vs ~1k in ai_v5), win rate ≥ v6 MCTS baseline.

---

## Design Questions

- **Rust sim scope**: implement only mechanics that appear in the 32 sample teams, or
  aim for full Gen3OU coverage? Start narrow (sample teams cover ~80% of mechanics),
  expand as fuzz failures demand.
- **Leaf batch size K**: profile on dev hardware. Likely 64–256.
- **MCTS parallelism**: multiple Python workers each with their own in-process Rust
  sim, or single-process Rust MCTS calling back to Python only for leaf eval?
  Single-process Rust is simpler; try it first.
