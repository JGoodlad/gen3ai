# Implementation: Step 1 — Showdown Bridge

A persistent Node.js subprocess that exposes the Showdown sim library over
newline-delimited JSON on stdin/stdout. This is the oracle for all fuzz testing in
subsequent steps.

## Does this need `npm run showdown`?

**No.** The bridge uses `BattleStream` — Showdown's designed in-process battle driver:

```js
const { BattleStream } = require('./deps/pokemon-showdown/dist/sim/battle-stream');
```

No WebSocket server, no open ports, no `npm run showdown`. The compiled `dist/`
already exists and is symlinked in every worktree (per CLAUDE.md setup). You need:
- Node.js runtime (already installed — showdown uses it)
- `deps/pokemon-showdown/dist/` (already compiled/symlinked)

`npm run showdown` is still needed for live training battles through poke-env. It is
not needed for the bridge, the fuzz harness, or MCTS rollouts.

`BattleStream` is the designed, stable API for programmatic battle control. It exposes
`stream.battle` — a direct reference to the live `Battle` object — which we use for
`State.serializeBattle()` and `activeRequest` access. The stream output (protocol text
starting with `|`) is deliberately ignored by the bridge; we read state from the object
directly rather than parsing text.

---

## Protocol

One JSON line in → one JSON line out. The bridge is stateful — it holds a
`Map<id, BattleStream>` in memory across commands.

### `new` — start a battle with known teams

```
→ {"cmd": "new", "id": "b1", "p1_team": "<packed>", "p2_team": "<packed>", "seed": [s0,s1,s2,s3]}
← {"ok": true, "state": { ...serializeBattle() output... },
   "p1_request": { ...activeRequest... }, "p2_request": { ...activeRequest... }}
```

`seed` is an optional array of four 16-bit integers (Gen 5 PRNG format). If omitted,
Showdown generates a random seed. For reproducible fuzz runs, pass a fixed seed.
Packed team format is the same as `Teams.pack()` output.

`state` is the initial battle state at turn 1, both active Pokémon on the field.
`p1_request` and `p2_request` are each side's legal action objects (see below) — so
the fuzz harness can immediately pick actions without a separate round-trip.

### `step` — advance one turn

```
→ {"cmd": "step", "id": "b1", "p1": "move 1", "p2": "switch 2"}
← {"ok": true, "state": { ...serializeBattle()... }, "winner": null,
   "p1_request": { ...activeRequest... }, "p2_request": { ...activeRequest... }}
   OR
← {"ok": true, "state": { ...serializeBattle()... }, "winner": "p1",
   "p1_request": null, "p2_request": null}
```

`state` is the state after both actions resolve (end of turn, residuals applied).
`winner` is `"p1"`, `"p2"`, `"draw"`, or `null` if the battle continues.
`p1_request` / `p2_request` are `null` when `winner` is set (battle over).

Action strings match Showdown's choice format: `"move 1"` through `"move 4"`,
`"switch 2"` through `"switch 6"`. Move indices are 1-based; switch indices refer
to team slot (1-based, skipping active and fainted).

**No separate `request` command.** Legal actions are returned with every `new` and
`step` response, eliminating one round-trip per turn in the fuzz harness.

Key fields in `activeRequest`:
```json
{
  "active": [{ "moves": [{"id": "surf", "pp": 24, "disabled": false}, ...] }],
  "side": { "pokemon": [{"active": true, "condition": "301/301", ...}, ...] }
}
```

### `free` — release a battle from memory

```
→ {"cmd": "free", "id": "b1"}
← {"ok": true}
```

---

## Implementation

```js
// src/utils/bridge/sim_battle.js
'use strict';
const path   = require('path');
const psPath = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream } = require(path.join(psPath, 'dist/sim/battle-stream'));
const { State }        = require(path.join(psPath, 'dist/sim/state'));

// BattleStream is the designed, stable Showdown API for programmatic battle control.
// stream.battle is a direct reference to the live Battle object — we read state from
// it directly and ignore the text protocol output (the | lines).
// BattleStream._writeLine processes commands synchronously, so stream.battle is
// immediately up to date after every stream.write() call.

const streams = new Map();  // Map<id, BattleStream>

function getRequests(battle) {
    return {
        p1_request: battle.sides[0].activeRequest,
        p2_request: battle.sides[1].activeRequest,
    };
}

function handleNew(msg) {
    const stream = new BattleStream();
    const seedStr = msg.seed ? `,"seed":${JSON.stringify(msg.seed)}` : '';
    stream.write(`>start {"formatid":"gen3anythinggoes"${seedStr}}`);
    stream.write(`>player p1 {"name":"p1","team":${JSON.stringify(msg.p1_team)}}`);
    stream.write(`>player p2 {"name":"p2","team":${JSON.stringify(msg.p2_team)}}`);
    streams.set(msg.id, stream);
    const b = stream.battle;
    return { ok: true, state: State.serializeBattle(b), ...getRequests(b) };
}

function handleStep(msg) {
    const stream = streams.get(msg.id);
    if (!stream) return { ok: false, error: `unknown id: ${msg.id}` };
    stream.write(`>p1 ${msg.p1}`);
    stream.write(`>p2 ${msg.p2}`);
    const b = stream.battle;
    const winner = b.ended ? (b.winner || 'draw') : null;
    return {
        ok: true,
        state: State.serializeBattle(b),
        winner,
        p1_request: winner ? null : b.sides[0].activeRequest,
        p2_request: winner ? null : b.sides[1].activeRequest,
    };
}

function handleFree(msg) {
    streams.delete(msg.id);
    return { ok: true };
}

// Newline-delimited JSON over stdin/stdout
let buf = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => {
    buf += chunk;
    let nl;
    while ((nl = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        let resp;
        try {
            const msg = JSON.parse(line);
            switch (msg.cmd) {
                case 'new':  resp = handleNew(msg);  break;
                case 'step': resp = handleStep(msg); break;
                case 'free': resp = handleFree(msg); break;
                default:     resp = { ok: false, error: `unknown cmd: ${msg.cmd}` };
            }
        } catch (e) {
            resp = { ok: false, error: e.message, stack: e.stack };
        }
        process.stdout.write(JSON.stringify(resp) + '\n');
    }
});

process.on('uncaughtException', e => {
    process.stdout.write(JSON.stringify({ ok: false, error: e.message }) + '\n');
});
```

---

## Python Client

```python
# src/utils/bridge/sim_battle_client.py
import json
import subprocess
import uuid
from pathlib import Path


class SimBattleClient:
    """Persistent bridge to the Showdown sim for programmatic battle control.

    Does not require a running Showdown server — the bridge drives BattleStream
    directly via Node.js require(). Legal actions are returned with every
    new_battle() and step() call, so no separate request() round-trip is needed.
    """

    def __init__(self):
        bridge = Path(__file__).parent / 'sim_battle.js'
        self._proc = subprocess.Popen(
            ['node', str(bridge)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._battle_id: str | None = None

    def _rpc(self, msg: dict) -> dict:
        self._proc.stdin.write(json.dumps(msg) + '\n')
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        resp = json.loads(line)
        if not resp.get('ok'):
            raise RuntimeError(f"sim_battle error: {resp.get('error')}")
        return resp

    def new_battle(
        self,
        p1_team: str,
        p2_team: str,
        seed: list[int] | None = None,
        battle_id: str | None = None,
    ) -> tuple[dict, dict, dict]:
        """Start a battle with packed team strings.

        Returns (state, p1_request, p2_request).
        state: initial serializeBattle() state (turn 1, both Pokémon on field).
        seed: optional [s0, s1, s2, s3] four 16-bit ints for reproducible battles.
        """
        self._battle_id = battle_id or str(uuid.uuid4())
        msg: dict = {
            'cmd': 'new',
            'id': self._battle_id,
            'p1_team': p1_team,
            'p2_team': p2_team,
        }
        if seed:
            msg['seed'] = seed
        resp = self._rpc(msg)
        return resp['state'], resp['p1_request'], resp['p2_request']

    def step(self, p1: str, p2: str) -> tuple[dict, str | None, dict | None, dict | None]:
        """Advance one turn.

        Returns (state, winner, p1_request, p2_request).
        winner: 'p1', 'p2', 'draw', or None if battle continues.
        p1_request, p2_request: None when winner is set.
        """
        resp = self._rpc({'cmd': 'step', 'id': self._battle_id, 'p1': p1, 'p2': p2})
        return resp['state'], resp['winner'], resp['p1_request'], resp['p2_request']

    def free(self) -> None:
        """Release the current battle from bridge memory."""
        if self._battle_id:
            self._rpc({'cmd': 'free', 'id': self._battle_id})
            self._battle_id = None

    def close(self) -> None:
        """Shut down the Node.js subprocess."""
        self._proc.stdin.close()
        self._proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.free()
        self.close()
```

---

## Parsing Legal Actions from `request`

The fuzz harness needs to pick random legal actions. Helper:

```python
def legal_actions(request: dict) -> list[str]:
    """Return all legal action strings for one side from an activeRequest."""
    actions = []

    # Moves available from the active Pokemon
    active = (request.get('active') or [{}])[0]
    for i, move in enumerate(active.get('moves', []), start=1):
        if not move.get('disabled'):
            actions.append(f'move {i}')

    # Switches (non-active, non-fainted team members)
    for i, mon in enumerate(request.get('side', {}).get('pokemon', []), start=1):
        if not mon.get('active') and mon.get('condition') != '0 fnt':
            actions.append(f'switch {i}')

    return actions or ['move 1']  # fallback: should never be empty mid-battle
```

---

## Milestones

### M1 — Bridge boots
`node src/utils/bridge/sim_battle.js` starts, accepts a `new` command with two
packed teams, returns a JSON response with `ok: true`, a `state` object containing
`turn: 1`, and non-null `p1_request`/`p2_request`. No crash on startup.

```bash
echo '{"cmd":"new","id":"t1","p1_team":"<packed>","p2_team":"<packed>"}' \
  | node src/utils/bridge/sim_battle.js
# → {"ok":true,"state":{"turn":1,...},"p1_request":{...},"p2_request":{...}}
```

### M2 — Turn steps
`step` with `"move 1"` for both sides advances turn to 2, returns updated state with
non-null requests. `winner` is null for a normal turn; requests are null when winner is set.

### M3 — Legal actions parse correctly
`legal_actions()` applied to the `p1_request`/`p2_request` in step responses returns
at least one valid string for both sides on every turn across 10 complete battles.

### M4 — Python client wraps cleanly
`SimBattleClient` starts the subprocess, runs a full battle via `new_battle → step* → free`,
cleans up without hanging. Context manager form works.

### M5 — 100 battles complete
100 random battles (random legal actions each turn) complete without any JS
exceptions, hangs, or malformed JSON responses. Measure mean turns per battle —
should be ~30–60 for Gen 3 OU random play.

---

## Tests

### Unit tests — `sim_battle_client_test.py`

Mock the subprocess; test that `SimBattleClient` constructs the right JSON messages
and parses responses correctly. No Node.js required.

```python
# src/utils/bridge/sim_battle_client_test.py
from unittest.mock import patch, MagicMock
import json
from utils.bridge.sim_battle_client import SimBattleClient, legal_actions


def test_legal_actions_moves_only():
    req = {'active': [{'moves': [
        {'id': 'surf', 'disabled': False},
        {'id': 'icebeam', 'disabled': True},
    ]}], 'side': {'pokemon': [{'active': True, 'condition': '200/200'}]}}
    assert legal_actions(req) == ['move 1']   # move 2 disabled


def test_legal_actions_with_switch():
    req = {'active': [{'moves': [{'id': 'surf', 'disabled': False}]}],
           'side': {'pokemon': [
               {'active': True,  'condition': '200/200'},
               {'active': False, 'condition': '150/150'},
               {'active': False, 'condition': '0 fnt'},
           ]}}
    assert legal_actions(req) == ['move 1', 'switch 2']  # slot 3 fainted


def test_rpc_constructs_correct_message():
    mock_resp = json.dumps({
        'ok': True, 'state': {'turn': 1},
        'p1_request': {}, 'p2_request': {},
    })
    with patch('subprocess.Popen') as mock_popen:
        proc = MagicMock()
        proc.stdout.readline.return_value = mock_resp + '\n'
        mock_popen.return_value = proc
        client = SimBattleClient()
        state, p1_req, p2_req = client.new_battle(
            'team1', 'team2', seed=[1, 2, 3, 4])
        written = proc.stdin.write.call_args[0][0]
        msg = json.loads(written.rstrip('\n'))
        assert msg['cmd'] == 'new'
        assert msg['p1_team'] == 'team1'
        assert msg['seed'] == [1, 2, 3, 4]


def test_step_returns_requests():
    mock_resp = json.dumps({
        'ok': True, 'state': {'turn': 2}, 'winner': None,
        'p1_request': {'active': [{'moves': [{'id': 'surf', 'disabled': False}]}],
                       'side': {'pokemon': [{'active': True, 'condition': '200/200'}]}},
        'p2_request': {'active': [{'moves': [{'id': 'tackle', 'disabled': False}]}],
                       'side': {'pokemon': [{'active': True, 'condition': '150/150'}]}},
    })
    with patch('subprocess.Popen') as mock_popen:
        proc = MagicMock()
        proc.stdout.readline.return_value = mock_resp + '\n'
        mock_popen.return_value = proc
        client = SimBattleClient()
        client._battle_id = 'test'
        state, winner, p1_req, p2_req = client.step('move 1', 'move 1')
        assert winner is None
        assert legal_actions(p1_req) == ['move 1']
```

### Integration tests — `sim_battle_integration_test.py`

Require a real Node.js process. Marked `@pytest.mark.integration`.

```python
# src/utils/bridge/sim_battle_integration_test.py
import pytest
from utils.bridge.sim_battle_client import SimBattleClient, legal_actions
from utils.team_loader import load_packed_teams

@pytest.mark.integration
def test_bridge_full_battle():
    teams = load_packed_teams('data/teams/')
    with SimBattleClient() as client:
        state, p1_req, p2_req = client.new_battle(teams[0], teams[1])
        assert state['turn'] == 1
        assert p1_req is not None and p2_req is not None

        winner = None
        turn = 0
        while winner is None and turn < 200:
            p1_action = legal_actions(p1_req)[0]
            p2_action = legal_actions(p2_req)[0]
            state, winner, p1_req, p2_req = client.step(p1_action, p2_action)
            turn += 1

        assert winner in ('p1', 'p2', 'draw')
        assert turn < 200, "battle did not finish in 200 turns"
        assert p1_req is None and p2_req is None  # requests null when battle ends

@pytest.mark.integration
def test_bridge_reproducible_with_seed():
    teams = load_packed_teams('data/teams/')
    seed = [0x1234, 0x5678, 0x9abc, 0xdef0]

    def run_battle(seed):
        turns_taken = []
        with SimBattleClient() as client:
            _, p1_req, p2_req = client.new_battle(teams[0], teams[1], seed=seed)
            winner = None
            while winner is None:
                state, winner, p1_req, p2_req = client.step(
                    legal_actions(p1_req)[0],
                    legal_actions(p2_req)[0],
                )
                turns_taken.append(state['turn'])
        return turns_taken, winner

    result1 = run_battle(seed)
    result2 = run_battle(seed)
    assert result1 == result2, "same seed should produce identical battle"
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/utils/bridge/sim_battle.js` | Persistent Node.js Showdown bridge |
| `src/utils/bridge/sim_battle_client.py` | Python wrapper |
| `src/utils/bridge/sim_battle_client_test.py` | Unit tests (mock subprocess) |
| `src/utils/bridge/sim_battle_integration_test.py` | Integration tests (real Node process) |

---

## Final State

Step 1 is complete when:
- M1–M5 all pass
- Integration tests pass: `pytest src/utils/bridge/ -m integration`
- 100 random battles complete cleanly
- `test_bridge_reproducible_with_seed` passes — same seed, same battle outcome

**Ready for Step 2:** the bridge is the oracle. Step 2 builds the Rust sim and
compares its output against this oracle turn-by-turn.
