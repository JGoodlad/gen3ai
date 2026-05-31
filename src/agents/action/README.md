# Gen3 Action System

Translates between the RL model's 11-action discrete space and poke_env `BattleOrder`,
routed entirely through our server-authoritative `LegalActions` snapshot.

## Action Space

| Index | Meaning |
|---|---|
| 0–5 | Switch to team slot 0–5 |
| 6–9 | Use move in request slot 0–3 |
| 10 | Struggle |

## Three stages over one immutable snapshot

A decision flows through three stages. The first two are **poke-env-free**; only the last
touches poke-env order types:

1. **mask** — `Gen3ActionMasker.mask_from_legal(legal)` builds the 11-dim binary mask
   purely from a `LegalActions` snapshot (no battle). `get_mask(battle, legal=…, live=…)` is a
   thin wrapper that snapshots the legality surface and runs the team-ordering integrity
   guards. The own-team roster those guards check is read through a `LiveView`
   (`live.ours.mons`) — built from the battle by default, or passed in via `live=` to reuse
   one already built this decision — so `action/` no longer reaches into the raw battle for
   state (`mask_generator.py` / `ordering_integrity.py` read only `LegalActions` + `LiveView`;
   `serialize.py` stays the lone Choice→`BattleOrder` poke-env seam).
2. **decode** (pure) — `Gen3ActionMapper.action_to_choice(action_idx, legal) -> Choice`
   resolves an action index against the captured snapshot into a tagged, poke-env-free
   `Choice` (`choice.py`). Fully testable with a `LegalActions` stub — no battle object.
3. **serialize** (the one poke-env touch) — `serialize.choice_to_order(choice, battle)`
   turns a `Choice` into the `BattleOrder` the client sends. This is the **only** module in
   `action/` that imports poke-env order/move types (`serialize.order_to_action` is the
   reverse boundary, for diagnostics).

The env / player compose them: snapshot `legal` once at observation time, build the mask
from it, store it on the `BattleContext`, and at action time decode + serialize against the
**same** snapshot (`Gen3ActionMapper.action_to_order(action, battle, legal=ctx.legal)`).

## Design: Crash Over Corruption

The system enforces a strict "crash over corruption" contract — ambiguous or stale state
raises immediately rather than silently sending a wrong action.

### The captured snapshot (replaces the old decision-context latch)

There is **no** `battle._gen3_decision_context` stash anymore. The immutable `LegalActions`
captured at observation time **is** the per-decision snapshot. The masker builds the mask
from it and the mapper decodes against it, so the two share one source by construction. If
poke-env processes a background message and `last_request` shifts while the model "thinks",
the decode is unaffected (it never re-reads the battle).
`Gen3ActionMapper.assert_decision_current(ctx, battle)` is the fail-loud guard run before
acting: it raises if the context is missing, from the wrong turn, or its snapshot's move
ordering no longer matches the server (a genuine mid-decision change).

### `LegalActions` is a hybrid source

`LegalActions.from_battle` (in `battle/live_view.py`) is deliberately hybrid:

- **`move_slots`** (which of the 4 move slots are legal, + pp / disabled / target) is
  **wire-truth**, read straight from the parsed server request
  (`last_request['active'][0]['moves']`).
- **`switches` / `force_switch` / `trapped` / `maybe_trapped` / `wait` / `struggle`** are
  poke-env's **derived** interpretation of that request (`available_switches`,
  `force_switch`, `available_moves`, …). We keep them byte-identical rather than
  re-deriving — they are the second poke-env-interpreted seam (alongside Choice→BattleOrder
  serialization) that a future fully-owned `Player` would re-derive.

### Struggle is single-sourced

`legal.struggle` is the ONE source of truth for "the active mon must Struggle." When all PP
is gone the server sends a lone `struggle` entry in the request moves; `from_battle` filters
it OUT of `move_slots` and surfaces it only as the flag. So:

- the masker sets bit 10 and never a move-slot bit for struggle;
- `action_to_choice(10, legal)` → `Choice.struggle()`, gated on `legal.struggle` (pressing
  10 when not legal raises);
- move slots 6–9 map only to real `move_slots[idx-6]`.

This removes the two-representations footgun behind the historical "struggle
double-enabling" mask bug. Action index `10 == STRUGGLE` is fixed (the reward manager's
struggle-loop tax depends on it).

### End-to-end send conformance

`Gen3ActionMapper.action_to_order` fail-loud-checks the whole chain — **offered** (mask) →
**picked** (action) → **sent** (order) — so we never silently send Showdown a different
move/switch than the model selected:

1. `mask[action] != 0` — the model only picks what the mask OFFERED;
2. `action_to_choice` raises if the action doesn't conform to `legal`;
3. `choice_to_order` raises if the choice can't resolve to a real move/switch;
4. the serialized order is **round-tripped** back through `order_to_action` and must equal
   the picked action — otherwise `RuntimeError` (catches a serialization drift, e.g. a
   duplicate-id or Hidden-Power mis-resolution). The round-trip is Hidden-Power-aware (bare
   `hiddenpower` ↔ typed `hiddenpowerice` count as one move, no false positive).

### Other invariants

- **Duplicate species check**: if the team contains duplicate species (Species Clause
  violation or state corruption), `get_mask` crashes.
- **Switch-ordering alignment** (`ordering_integrity.py`): switch action index *i*,
  switch-validity bit *i*, and per-Pokémon obs slot *i* must all refer to the same mon —
  verified against the snapshot's slot-indexed switches.

## Files

| File | Purpose |
|---|---|
| `choice.py` | `Choice` / `ChoiceKind` — the poke-env-free tagged action |
| `mapper.py` | `Gen3ActionMapper` — pure `action_to_choice`, the `action_to_order` convenience, `assert_decision_current`, reverse `order_to_action` |
| `serialize.py` | The single poke-env touch: `choice_to_order` + `order_to_action` |
| `mask_generator.py` | `Gen3ActionMasker` — `mask_from_legal` (pure) + `get_mask(battle)` |
| `ordering_integrity.py` | Move/team ordering alignment guards (now `legal`-driven) |
| `constants.py` | The 11-action layout constants |

## Tests

| File | Type | What it covers |
|---|---|---|
| `mapper_test.py` | Unit | Masker + pure `action_to_choice` (LegalActions STUB, no battle) + serialization + staleness guard + reverse map; struggle single-source regression |
| `ordering_integrity_test.py` | Unit | Move/team ordering alignment (snapshot-driven) |
| `fuzz_test_unit.py` | Standalone script | Snapshot-immutability simulation: corrupt the request mid-decision, prove the captured snapshot decodes identically (replaces the old latch race sim) |
| `fuzz_test.py` | Fuzz (local bridge, no server) | Real battles vs RandomPlayer; exhaustively decodes + serializes every legal action each turn |
| `telemetry_e2e_test.py` | E2E (requires server) | Monitors for silent mid-decision state updates in live battles |
