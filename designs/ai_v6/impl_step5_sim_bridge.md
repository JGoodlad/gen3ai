# AI v6 — Step 5 (Baby Step): Showdown Sim Bridge

The full MCTS design is in `designs/ai_v6/impl_step5_mcts.md`. This document
describes the minimal first implementation: getting `sim_bridge.js` working and
validated before touching any tree or rollout logic. Nothing here is throwaway —
every later piece of Step 5 sits on top of this bridge.

---

## What the Baby Step Proves

Before building `action_sampler.py`, `tree.py`, or `rollout.py`, three things
must be confirmed to work:

1. We can start a Showdown battle in-process from packed team strings
2. We can snapshot and restore state via `toJSON()` / `fromJSON()`
3. We can inject modified state (changed HP, injected Pokémon) and have the sim
   accept it — this is the core bet the PIMC approach rests on

**If step 3 fails, stop.** The state-extraction approach needs rethinking before
any further MCTS code is written.

---

## Why This Approach: State Extraction + PIMC

Pokémon is a partially-observable game. We never know the opponent's full team
until it is revealed. The standard technique for imperfect-information MCTS is
**PIMC (Perfect Information Monte Carlo)**:

1. **Sample** a complete-world hypothesis: fill the opponent's unrevealed slots
   from a distribution (uniform Gen3OU usage stats initially; team completion
   model later).
2. **Run MCTS** in that fully-observed world as if it were real.
3. **Repeat** for K different samples.
4. **Aggregate**: pick the action with the best average Q across all worlds.

The operation this requires is: *take what we know + a hypothesis about what we
don't know → produce a complete, searchable sim state*. That is what
`battle.toJSON()` + JSON modification + `Battle.fromJSON()` gives us. No
background shadow sim is needed; we build the state fresh at decision time.

---

## API Split: BattleStream + Direct Battle API

Two Showdown interfaces exist:

- **BattleStream** — text-protocol stream wrapper designed for network I/O
  (`>start`, `>player`, `>p1 move 1`). Handles team parsing and format
  validation. After setup, exposes `stream.battle` as the raw `Battle` object.
- **Direct Battle API** — synchronous, object-based. `battle.makeChoices()`,
  `battle.toJSON()`, `Battle.fromJSON()`. No async overhead. Right for search.

We use a **hybrid**:

| Operation | API | Reason |
|-----------|-----|--------|
| `new` (session init) | BattleStream | Handles `>start` / `>player` parsing correctly |
| `advance` (root sync) | BattleStream | Replays real-game moves via text protocol |
| `fork` | Direct Battle API | `Battle.fromJSON(battle.toJSON())` — in-process |
| `step` | Direct Battle API | `battle.makeChoices()` — synchronous |
| `inject` | Direct Battle API | `toJSON()` → modify → `fromJSON()` |

The baby step implements `new`, `fork`, `step`, `inject`, `free`. `advance`
(root sync with the live game) is deferred — the baby step hardcodes both-side
choices; live-game sync comes when PIMC is wired into `choose_move`.

---

## What This Step Builds

### `src/utils/bridge/sim_bridge.js`

Node.js subprocess. JSON-lines on stdin/stdout. Maintains a `Map<id, Battle>`
for live and forked sessions.

**Command protocol:**

```
→ {"cmd":"new", "id":"root", "p1_team":"<packed>", "p2_team":"<packed>",
               "format":"gen3ou", "seed":[s0,s1,s2,s3]}
← {"ok":true}

→ {"cmd":"fork", "src":"root", "id":"r0"}
← {"ok":true}

→ {"cmd":"step", "id":"r0", "p1":"move 1", "p2":"switch 3"}
← {"ok":true, "done":false, "winner":null, "log":["|move|...", "|damage|..."]}
← {"ok":true, "done":true,  "winner":"p1", "log":["|faint|...", "|win|..."]}

→ {"cmd":"inject", "src":"root", "id":"i0", "side":1, "slot":2, "pokemon":{...}}
← {"ok":true}

→ {"cmd":"free", "id":"r0"}
← {"ok":true}
```

**Internals:**

```js
// new: BattleStream for setup, then grab stream.battle
const stream = new BattleStream();
stream.write(`>start {"formatid":"${msg.format}","seed":${JSON.stringify(msg.seed)}}`);
stream.write(`>player p1 {"name":"p1","team":"${msg.p1_team}"}`);
stream.write(`>player p2 {"name":"p2","team":"${msg.p2_team}"}`);
// drain output until |turn|1
sessions.set(msg.id, stream.battle);

// fork: Direct Battle API — in-process, no IPC
const snap = sessions.get(msg.src).toJSON();
const fork = Battle.fromJSON(snap);
fork.prng = new PRNG();   // fresh seed so rollouts diverge stochastically
const log = [];
fork.send = (type, data) => log.push(...data.split('\n').filter(Boolean));
sessions.set(msg.id, { battle: fork, log });
fork.restart();

// step: Direct Battle API
const { battle, log } = sessions.get(msg.id);
battle.makeChoices(msg.p1, msg.p2);

// inject: toJSON → modify → fromJSON
const snap = sessions.get(msg.src).toJSON();
snap.sides[msg.side].pokemon[msg.slot] = msg.pokemon;
const injected = Battle.fromJSON(snap);
const log = [];
injected.send = (type, data) => log.push(...data.split('\n').filter(Boolean));
sessions.set(msg.id, { battle: injected, log });
injected.restart();
```

### `src/agents/mcts/sim_client.py`

Python subprocess wrapper. Starts `sim_bridge.js` on first use (persistent
process, restarted on crash).

```python
class SimClient:
    def new(self, id: str, p1_team: str, p2_team: str,
            format: str = "gen3ou", seed: list | None = None) -> None: ...
    def fork(self, src: str, id: str) -> None: ...
    def step(self, id: str, p1: str, p2: str) -> StepResult: ...
    def inject(self, src: str, id: str, side: int, slot: int,
               pokemon: dict) -> None: ...
    def free(self, id: str) -> None: ...
```

`StepResult` is a dataclass: `done: bool`, `winner: str | None`, `log: list[str]`.

### `src/agents/mcts/__init__.py`

Empty package marker.

---

## Build Order

**Step 0 — Clean battle** (~100 lines JS + ~80 lines Python)

`new` → 3× `step` with hardcoded choices → `free`.
Confirm: no crash, `log` contains `|turn|` lines, `done` is false after 3 turns.

**Step 1 — Round-trip fork** (~20 more lines JS)

`new` → `fork` → `step` on the fork.
Confirm: fork advances from the correct turn; root is unaffected.

**Step 2 — Injection test** (~20 more lines JS + ~30 lines Python test)

`new` → extract `toJSON()` from one of the sessions → set a benched Pokémon's
`hp` to `1` in Python → `inject` → `step` with a move targeting that slot →
confirm `done:true` and `winner` is set.

**This is the go/no-go gate.** If `Battle.fromJSON()` rejects the modified blob
or the HP change is not reflected in battle, the PIMC state-extraction approach
needs a rethink before any further search code is written.

---

## Showdown JSON Schema for `inject`

A Pokémon slot in `toJSON()` output (abridged — all fields required for
`fromJSON()` to work):

```json
{
  "set": {
    "species": "Salamence", "item": "Choice Band", "ability": "Intimidate",
    "moves": ["Dragon Claw", "Earthquake", "Rock Slide", "Brick Break"],
    "nature": "Adamant", "gender": "M", "level": 100,
    "evs": {"hp":0,"atk":252,"def":0,"spa":0,"spd":4,"spe":252},
    "ivs": {"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31}
  },
  "hp": 361, "maxhp": 361, "baseMaxhp": 361,
  "status": "", "statusState": {"id": ""},
  "volatiles": {},
  "boosts": {"atk":0,"def":0,"spa":0,"spd":0,"spe":0,"accuracy":0,"evasion":0},
  "moveSlots": [
    {"move":"Dragon Claw","id":"dragonclaw","pp":24,"maxpp":24,"disabled":false,"used":false},
    {"move":"Earthquake","id":"earthquake","pp":16,"maxpp":16,"disabled":false,"used":false},
    {"move":"Rock Slide","id":"rockslide","pp":16,"maxpp":16,"disabled":false,"used":false},
    {"move":"Brick Break","id":"brickbreak","pp":24,"maxpp":24,"disabled":false,"used":false}
  ],
  "item": "choiceband", "itemState": {"id":"choiceband"}, "lastItem": "",
  "ability": "intimidate", "abilityState": {"id":"intimidate"}, "baseAbility": "intimidate",
  "species": "[Species:salamence]", "baseSpecies": "[Species:salamence]",
  "speciesState": {"id":"salamence"},
  "details": "Salamence, L100, M",
  "types": ["Dragon","Flying"], "baseTypes": ["Dragon","Flying"],
  "apparentType": "Dragon/Flying",
  "baseStoredStats": {"hp":361,"atk":304,"def":198,"spa":214,"spd":198,"spe":259},
  "storedStats": {"atk":304,"def":198,"spa":214,"spd":198,"spe":259},
  "position": 1, "isActive": false, "fainted": false, "activeTurns": 0
}
```

Key notes:
- Species/move/item references use the `"[Species:id]"` format for the Dex lookup
- `baseStoredStats` and `storedStats` are computed from the Gen 3 stat formula
  using `set.evs`, `set.ivs`, and `set.nature` — must be pre-computed before
  injecting; the deserializer does NOT recompute from `set`
- `position` is the slot index in `side.pokemon[]`; active mon is `position: 0`
- `fainted: true` + `hp: 0` is the canonical fainted representation

---

## Files Created

| File | Purpose |
|------|---------|
| `src/utils/bridge/sim_bridge.js` | Node.js bridge subprocess |
| `src/agents/mcts/sim_client.py` | Python subprocess wrapper |
| `src/agents/mcts/__init__.py` | Package marker |

---

## Verification

1. `node src/utils/bridge/sim_bridge.js` — process starts, waits for input, no
   immediate crash
2. Step 0: Python test sends `new` + 3 `step` calls, all return `done:false`,
   `log` contains `|turn|2|`, `|turn|3|`, `|turn|4|`
3. Step 1: `fork` then `step` on fork returns `done:false`; direct `step` on
   root also returns `done:false` (fork was isolated)
4. Step 2: inject `hp:1` onto a bench slot, `step` with a move that hits it,
   response has `done:true` — **go/no-go gate for the whole PIMC approach**
