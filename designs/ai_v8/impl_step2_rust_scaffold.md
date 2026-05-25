# Implementation: Step 2 — Rust Scaffold + First Fuzz

Stand up `src/sim/`, port Gen5RNG, deserialize Showdown battle state from JSON,
implement damage calc and basic turn resolution, expose via PyO3, and run the
first fuzz comparison against the Step 1 bridge.

The fuzz harness is Python — it holds both handles (Step 1 bridge subprocess +
Rust via PyO3) and diffs their outputs field-by-field. Rust's job is mechanics.
Python's job is driving both and comparing.

---

## Crate Layout

```
src/sim/
  Cargo.toml
  src/
    lib.rs           — pub mod declarations + crate-level docs
    state.rs         — BattleState, SideState, MonState, SampledValues, all enums
    prng.rs          — Gen5Rng
    data.rs          — static lookup tables loaded from data/pokemon/*.json
    damage.rs        — damage formula, type effectiveness, crit
    turn.rs          — action ordering, move execution, residuals, win check
    python.rs        — PyO3 bindings
    error.rs         — SimError enum
  tests/
    prng_test.rs     — RNG correctness vs known Showdown seeds
    damage_test.rs   — damage formula spot checks vs known values
  sim_binding_test.py   — Python pytest suite for the PyO3 API
```

`Cargo.toml`:
```toml
[package]
name    = "gen3-sim"
version = "0.1.0"
edition = "2021"

[dependencies]
pyo3       = { version = "0.21", features = ["extension-module"] }
serde      = { version = "1",    features = ["derive"] }
serde_json = "1"

[lib]
name       = "gen3_sim"
crate-type = ["cdylib"]
```

Build into the conda env:
```bash
cd src/sim
maturin develop --release
```

After this, `import gen3_sim` works from anywhere in the project.

---

## The `SampledValues` Struct

Every `BattleState` carries a `SampledValues` struct that explicitly records
everything that was guessed — not directly observed — at construction time.

During fuzz testing this struct is always empty: both teams are fully known, nothing
is guessed. During MCTS (Step 5), `from_poke_env()` populates it with the opponent's
sampled sleep durations, which team slots came from the completion model, and which
items were guessed rather than observed.

This means every MCTS rollout has a complete audit trail of what it made up. There
is no silent hidden-variable assumption.

```rust
/// Values not directly observable at construction time.
/// Always empty during fuzz testing — both teams fully known.
/// Populated by BattleState::from_poke_env() during MCTS.
#[derive(Debug, Clone, Default)]
pub struct SampledValues {
    /// Sleep turns remaining for each opponent slot.
    ///
    /// Gen 3 sleep duration is rolled ONCE on application (random(2,6) → 1–4
    /// turns). We observe how many turns the opponent has been asleep from
    /// the protocol stream, but not the total duration that was rolled, so
    /// remaining = total_rolled - turns_slept is unknown.
    ///
    /// None    = slot is not asleep, or this is our own Pokemon (known exactly).
    /// Some(n) = we sampled that n turns remain, drawn from the conditional
    ///           P(remaining=r | turns_slept=k) ∝ Uniform(1,4) truncated below k.
    pub opp_sleep_remaining: [Option<u8>; 6],

    /// Opponent team slots filled by the completion model (indices into opp_side.team).
    /// Empty when both teams are fully known.
    pub opp_completed_slots: Vec<usize>,

    /// Whether each opponent's held item was sampled (true) or directly
    /// revealed (false) via a protocol event (Leftovers residual, Berry
    /// activation, Knock Off message, etc.).
    pub opp_item_sampled: [bool; 6],
}
```

---

## State Structures

Mirrors the fields emitted by `serializeBattle()`. Only mutable battle state —
static data (base stats, move power, type chart) is loaded once at startup from
`data/pokemon/*.json`.

The field names in serde attributes map directly to Showdown's camelCase JSON output,
so `from_showdown_json()` is a direct deserialisation with no manual field mapping.

```rust
// state.rs (abbreviated — full implementation includes all enums and impls)

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BattleState {
    pub turn: u32,
    pub weather: Weather,
    #[serde(rename = "weatherMinTimeLeft")]
    pub weather_turns_remaining: u8,
    pub p1: SideState,
    pub p2: SideState,
    pub prng: PrngState,       // serialised seed for round-trip fidelity
    #[serde(skip)]
    pub sampled: SampledValues,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideState {
    #[serde(rename = "activeRequest")]
    pub active_slot: u8,
    pub spikes: u8,
    #[serde(rename = "sideConditions")]
    pub side_conditions: SideConditions,
    pub pokemon: Vec<MonState>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SideConditions {
    pub reflect: Option<ScreenState>,
    #[serde(rename = "lightscreen")]
    pub light_screen: Option<ScreenState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenState {
    pub duration: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonState {
    pub species: String,       // Showdown ID, e.g. "gengar"
    pub hp: u16,
    #[serde(rename = "maxhp")]
    pub max_hp: u16,
    pub status: Option<String>,  // None | "brn" | "slp" | "par" | "psn" | "tox" | "frz"
    #[serde(rename = "statusState")]
    pub status_state: StatusState,
    pub item: Option<String>,
    #[serde(rename = "lastItem")]
    pub last_item: Option<String>,
    pub boosts: Boosts,
    #[serde(rename = "moveSlots")]
    pub moves: Vec<MoveSlot>,
    pub volatiles: HashMap<String, VolatileState>,
    pub fainted: bool,
    pub transformed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatusState {
    /// sleep: turns remaining before wake. toxic: turns poisoned (for damage scaling).
    pub duration: u8,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Boosts {
    pub atk: i8, pub def: i8, pub spa: i8,
    pub spd: i8, pub spe: i8, pub accuracy: i8, pub evasion: i8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveSlot {
    pub id: String,   // Showdown ID, e.g. "surf"
    pub pp: u8,
    #[serde(rename = "maxpp")]
    pub max_pp: u8,
    pub disabled: bool,
}
```

---

## RNG

Port of Showdown's `Gen5RNG` — the 64-bit LCG used for seeded battles.

```rust
// prng.rs

/// seed_{n+1} = seed_n * 0x5D588B656C078965 + 0x00269EC3  (mod 2^64)
/// Matches Showdown's Gen5RNG exactly — same output for same seed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gen5Rng {
    seed: u64,
}

impl Gen5Rng {
    /// Construct from Showdown's serialised seed format: [a, b, c, d] u16 values.
    /// Showdown seed string "gen5,AAAABBBBCCCCDDDD" → parse each 4 hex chars.
    pub fn from_parts(parts: [u16; 4]) -> Self {
        let seed = ((parts[0] as u64) << 48)
                 | ((parts[1] as u64) << 32)
                 | ((parts[2] as u64) << 16)
                 |  (parts[3] as u64);
        Self { seed }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.seed = self.seed
            .wrapping_mul(0x5D588B656C078965)
            .wrapping_add(0x00269EC3);
        (self.seed >> 32) as u32
    }

    /// Matches Showdown: Math.floor(next_u32 * n / 2^32)
    pub fn random(&mut self, n: u32) -> u32 {
        ((self.next_u32() as u64 * n as u64) >> 32) as u32
    }

    pub fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        self.random(denominator) < numerator
    }
}
```

---

## Damage Formula

Gen 3 formula. The implementation order of operations must exactly match
Showdown's `getDamage()` in `battle-actions.ts` — rounding order is observable.

```
base  = floor(floor(floor(2 * level / 5 + 2) * power * atk / def) / 50) + 2
      × weather_modifier     (1.5 rain+Water / 0.5 rain+Fire / 1.5 sun+Fire / 0.5 sun+Water)
      × crit_modifier        (2.0; in Gen 3 crits IGNORE ALL stat stage modifiers,
                              both attacker boosts and defender drops)
      × random_modifier      (roll = random(39) + 217; damage = floor(base * roll / 255))
      × stab                 (1.5 if move type matches user type, else 1.0)
      × type_effectiveness   (product of both defending types: 0 / 0.25 / 0.5 / 1 / 2 / 4)
      × burn                 (0.5 if burned and using a physical move)
      × screen               (0.5 if Reflect for physical / Light Screen for special;
                              screens are IGNORED on crits)
```

Gen 3 specifics to get right:
- **Crits ignore all stat stages** — not just the asymmetric rule from later gens.
  When a crit lands, recalculate atk/def using base stats + EVs/IVs only, no boosts.
- **Random roll is 16 values** — `random(39) + 217` gives values in
  `{217, 220, 223, ..., 255}` (steps of ~3, not a continuous range).
- **Special split** — SpA for special moves, SpD for special defence (not gen 1/2 unified
  Special stat).
- **1 HP minimum** — damage is at least 1 unless the move is blocked entirely.

---

## Turn Resolution

```
1. Both players submit actions (move or switch)
2. Action order:
   a. Switches before moves (both sides)
   b. Two switches: faster Pokemon's switch first
   c. Two moves: higher Priority bracket first; within bracket, higher Speed first;
      speed tie → random(2) == 0 selects p1
3. Execute each action:
   a. Switch: apply switch-in (entry hazard damage, ability triggers)
   b. Move:
      i.   Pre-move checks: asleep → decrement counter, maybe wake; frozen → maybe thaw;
           paralysis → 25% chance to skip; confusion → 50% chance self-hit
      ii.  Accuracy check: random(100) < floor(move_accuracy * acc_stage * eva_stage)
      iii. Damage: compute via formula above
      iv.  Secondary effects at their proc rates (flinch, status, stat drops)
      v.   Recoil / drain
4. End-of-turn residuals (Showdown's strict ordering):
   a. Weather damage (Sandstorm: 1/16 to non-Rock/Steel/Ground)
   b. Leftovers recovery (+1/16 max HP)
   c. Burn damage (−1/8 max HP)
   d. Poison damage (−1/8 max HP)
   e. Toxic damage (−counter/16 max HP; counter increments after damage)
   f. Leech Seed drain
   g. Partial trap damage (Bind, Wrap, etc.)
   h. Perish Song counter decrement (faint at 0)
   i. Screen turn decrements
   j. Weather turn decrement (clear if expired)
5. Check win condition
```

Return type:

```rust
pub enum TurnResult {
    Ongoing,
    P1Wins,
    P2Wins,
    Draw,
}
```

---

## PyO3 Bindings

```rust
// python.rs
use pyo3::prelude::*;

/// Deserialise a Showdown serializeBattle() JSON string into a BattleState.
#[pyfunction]
pub fn from_showdown_json(json_str: &str) -> PyResult<BattleState> {
    serde_json::from_str(json_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

/// Step one turn. Returns ("ongoing"|"p1wins"|"p2wins"|"draw").
#[pyfunction]
pub fn step_turn(state: &mut BattleState, p1: &str, p2: &str) -> PyResult<String> {
    let result = state.step(p1, p2)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(match result {
        TurnResult::Ongoing => "ongoing",
        TurnResult::P1Wins  => "p1wins",
        TurnResult::P2Wins  => "p2wins",
        TurnResult::Draw    => "draw",
    }.to_owned())
}

/// Serialise back to JSON for diffing against Showdown output.
#[pyfunction]
pub fn to_showdown_json(state: &BattleState) -> PyResult<String> {
    serde_json::to_string(state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pymodule]
fn gen3_sim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_showdown_json, m)?)?;
    m.add_function(wrap_pyfunction!(step_turn, m)?)?;
    m.add_function(wrap_pyfunction!(to_showdown_json, m)?)?;
    Ok(())
}
```

---

## Fuzz Harness

`src/agents/mcts/rust_sim_fuzz_e2e_test.py` — Python holds both handles and diffs.

```python
# src/agents/mcts/rust_sim_fuzz_e2e_test.py
"""
Fuzz test: run random battles via the Showdown bridge (oracle) and the Rust sim
simultaneously, asserting identical state after every turn.

Requires: Node.js + deps/pokemon-showdown/dist/ (no server needed).
Run:  python src/agents/mcts/rust_sim_fuzz_e2e_test.py --battles 50
"""
import argparse
import json
import random
import sys

import gen3_sim                                           # Rust, via PyO3
from utils.bridge.sim_battle_client import SimBattleClient, legal_actions
from utils.team_loader import load_packed_teams


IGNORE_FIELDS = {'prng', 'log', 'hints', 'inputLog', 'id'}


def flatten(obj, prefix='') -> dict:
    """Flatten a nested dict to dotted-path keys."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in IGNORE_FIELDS:
                continue
            out.update(flatten(v, f'{prefix}.{k}' if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f'{prefix}[{i}]'))
    else:
        out[prefix] = obj
    return out


def diff_states(showdown: dict, rust_json: str, turn: int) -> list[str]:
    rust = json.loads(rust_json)
    sd = flatten(showdown)
    rs = flatten(rust)
    mismatches = []
    for key in sd:
        if key not in rs:
            mismatches.append(f'  turn {turn} | {key}: showdown={sd[key]}  rust=MISSING')
        elif sd[key] != rs[key]:
            mismatches.append(f'  turn {turn} | {key}: showdown={sd[key]}  rust={rs[key]}')
    return mismatches


def run_battle(bridge: SimBattleClient, p1_team: str, p2_team: str,
               seed: str) -> int:
    """Run one battle. Returns number of turns. Raises on divergence."""
    showdown_state = bridge.new_battle(p1_team, p2_team, seed=seed)
    rust_state     = gen3_sim.from_showdown_json(json.dumps(showdown_state))

    turn = 0
    winner = None
    while winner is None:
        p1_req, p2_req = bridge.request()
        p1_action = random.choice(legal_actions(p1_req))
        p2_action = random.choice(legal_actions(p2_req))

        # Step Rust first (same inputs)
        gen3_sim.step_turn(rust_state, p1_action, p2_action)

        # Step Showdown (oracle)
        showdown_state, winner = bridge.step(p1_action, p2_action)
        turn += 1

        # Compare
        rust_json = gen3_sim.to_showdown_json(rust_state)
        mismatches = diff_states(showdown_state, rust_json, turn)
        if mismatches:
            print(f'\nDIVERGENCE after turn {turn}:')
            print(f'  actions: p1={p1_action!r}  p2={p2_action!r}')
            for m in mismatches[:20]:
                print(m)
            if len(mismatches) > 20:
                print(f'  ... and {len(mismatches) - 20} more')
            sys.exit(1)

    bridge.free()
    return turn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--battles', type=int, default=5)
    args = parser.parse_args()

    teams = load_packed_teams('data/teams/')
    bridge = SimBattleClient()

    for i in range(args.battles):
        p1, p2 = random.sample(teams, 2)
        seed = f'gen5,{random.randint(0, 2**64 - 1):016x}'
        turns = run_battle(bridge, p1, p2, seed)
        print(f'Battle {i+1}/{args.battles}: {turns} turns  OK')

    bridge.close()
    print(f'\nPASS  {args.battles} battles')


if __name__ == '__main__':
    main()
```

---

## Milestones

### M1 — Crate builds and imports
`maturin develop --release` completes without errors.
`from gen3_sim import from_showdown_json, step_turn, to_showdown_json` works in the
conda env. No other functionality required yet.

### M2 — RNG matches Showdown
`cargo test` passes `prng_test.rs`. Given seed `[0x1234, 0x5678, 0x9abc, 0xdef0]`,
the first 20 values from `Gen5Rng::random(256)` must exactly match the output of
Showdown's `Gen5RNG` for the same seed. Compute the reference values once by running
a small Node.js snippet and hard-code them as the expected values in the test.

### M3 — State deserialises without panic
`from_showdown_json(state_json)` does not panic for 10 different real battle states
captured from the bridge. `state.turn` matches the JSON `turn` field in all cases.
No full turn resolution yet — just parsing.

### M4 — Damage matches for 5 known matchups
`cargo test` passes `damage_test.rs`. Five hard-coded matchups (species, move, level,
EVs, weather, crit, roll value) produce the exact same damage value as recorded from
Showdown's battle log. Cover: physical, special, STAB, super effective, burn, crit.

### M5 — Python binding tests pass
`pytest src/sim/sim_binding_test.py` passes. Covers: `from_showdown_json` round-trips
turn field, `step_turn` returns `"ongoing"` for a mid-game state, `to_showdown_json`
produces valid JSON. Uses fixture JSON files — no bridge or Showdown needed.

### M6 — Fuzz: 5 battles, switching only
Run the fuzz harness with both sides always choosing `"switch"` (or `"move 1"` if no
switch is available). Zero divergences across 5 complete battles. This validates
switching logic, HP tracking, faint detection, and win condition before moves complicate
things.

### M7 — Fuzz: 5 battles with random moves, first divergence fixed
Run the fuzz harness with random legal actions. The first divergence will appear —
likely a damage rounding issue or residual ordering bug. Fix it, re-run, repeat until
5 random-action battles pass.

### M8 — Fuzz: 50 battles pass
50 random battles, zero divergences. Surfaces rarer states: double switch, move
failure (accuracy miss), simultaneous faint, status applied on the same turn as KO.

**M8 is the Step 2 exit gate.** Step 3 (full mechanics) begins here — it is just
running `--battles 10000` and fixing failures until it passes.

---

## Python Binding Tests

```python
# src/sim/sim_binding_test.py
import json
import pytest
import gen3_sim


@pytest.fixture
def initial_state_json(tmp_path):
    """Load a fixture state captured from the bridge at turn 0."""
    # In practice: load from tests/fixtures/turn0_state.json
    # For now, minimal valid structure
    return json.dumps({
        "turn": 3,
        "weather": "",
        "weatherMinTimeLeft": 0,
        "p1": {"active_slot": 0, "spikes": 0,
               "sideConditions": {}, "pokemon": []},
        "p2": {"active_slot": 0, "spikes": 0,
               "sideConditions": {}, "pokemon": []},
        "prng": {"seed": [0x1234, 0x5678, 0x9abc, 0xdef0]},
    })


def test_from_showdown_json_parses_turn(initial_state_json):
    state = gen3_sim.from_showdown_json(initial_state_json)
    assert state is not None  # didn't panic


def test_to_showdown_json_roundtrips(initial_state_json):
    state = gen3_sim.from_showdown_json(initial_state_json)
    out   = gen3_sim.to_showdown_json(state)
    parsed = json.loads(out)
    assert parsed['turn'] == 3


def test_from_showdown_json_invalid_raises():
    with pytest.raises(Exception):
        gen3_sim.from_showdown_json("not json")
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/sim/Cargo.toml` | Crate manifest |
| `src/sim/src/lib.rs` | Module declarations |
| `src/sim/src/state.rs` | All state structs + `SampledValues` + serde |
| `src/sim/src/prng.rs` | `Gen5Rng` |
| `src/sim/src/data.rs` | Static data loader |
| `src/sim/src/damage.rs` | Damage formula, type chart, crit |
| `src/sim/src/turn.rs` | Turn resolution, residuals, win check |
| `src/sim/src/python.rs` | PyO3 bindings |
| `src/sim/src/error.rs` | `SimError` |
| `src/sim/tests/prng_test.rs` | RNG correctness (cargo test) |
| `src/sim/tests/damage_test.rs` | Damage formula spot checks (cargo test) |
| `src/sim/sim_binding_test.py` | Python API tests (pytest, no Showdown) |
| `src/agents/mcts/rust_sim_fuzz_e2e_test.py` | Fuzz harness (pytest e2e) |

---

## Final State

Step 2 is complete when M1–M8 all pass:
- `maturin develop` builds cleanly
- `cargo test` passes (RNG + damage unit tests)
- `pytest src/sim/sim_binding_test.py` passes
- Fuzz harness: 50 random battles, zero divergences

**Ready for Step 3:** Step 3 is just running `--battles 10000` and fixing fuzz
failures one mechanic at a time until it passes.
