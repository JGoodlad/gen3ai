# Implementation: Step 1 — Showdown Bridge

A persistent Node.js subprocess that exposes the Showdown sim library over
newline-delimited JSON on stdin/stdout. This is the oracle for all fuzz testing in
subsequent steps.

## Does this need `npm run showdown`?

**No.** The bridge `require()`s the Showdown sim library directly:

```js
const { Battle } = require('./deps/pokemon-showdown/dist/sim');
```

No WebSocket server, no open ports, no `npm run showdown`. The compiled `dist/`
already exists and is symlinked in every worktree (per CLAUDE.md setup). You need:
- Node.js runtime (already installed — showdown uses it)
- `deps/pokemon-showdown/dist/` (already compiled/symlinked)

`npm run showdown` is still needed for live training battles through poke-env. It is
not needed for the bridge, the fuzz harness, or MCTS rollouts.

---

## Protocol

One JSON line in → one JSON line out. The bridge is stateful — it holds a
`Map<id, Battle>` in memory across commands.

### `new` — start a battle with known teams

```
→ {"cmd": "new", "id": "b1", "p1_team": "<packed>", "p2_team": "<packed>", "seed": "gen5,xxxx"}
← {"ok": true, "state": { ...serializeBattle() output... }}
```

`seed` is optional. If omitted, Showdown generates a random seed. For reproducible
fuzz runs, pass a fixed seed. Packed team format is the same as `Teams.pack()` output
(what `validate_team.js` already uses).

`state` is the initial battle state before either player has acted — turn 0, both
active Pokémon on the field.

### `step` — advance one turn

```
→ {"cmd": "step", "id": "b1", "p1": "move 1", "p2": "switch 2"}
← {"ok": true, "state": { ...serializeBattle()... }, "winner": null}
   OR
← {"ok": true, "state": { ...serializeBattle()... }, "winner": "p1"}
```

`state` is the state after both actions resolve (end of turn, residuals applied).
`winner` is `"p1"`, `"p2"`, `"draw"`, or `null` if the battle continues.

Action strings match Showdown's choice format: `"move 1"` through `"move 4"`,
`"switch 2"` through `"switch 6"`. Move indices are 1-based; switch indices refer
to team slot (1-based, skipping active and fainted).

### `request` — get legal actions for both sides

```
→ {"cmd": "request", "id": "b1"}
← {"ok": true, "p1": { ...activeRequest... }, "p2": { ...activeRequest... }}
```

Returns each side's `activeRequest` object — the same structure Showdown sends
clients to tell them what choices are available. The fuzz harness uses this to
enumerate legal moves and switches before picking randomly.

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
const { Battle } = require(path.join(psPath, 'dist/sim'));
const { Teams }  = require(path.join(psPath, 'dist/sim'));
const { PRNG }   = require(path.join(psPath, 'dist/sim/prng'));
const { State }  = require(path.join(psPath, 'dist/sim/state'));

const battles = new Map();

function handleNew(msg) {
    const battle = new Battle({
        formatid: 'gen3anythinggoes',
        send: () => {},  // discard protocol output — we use serializeBattle()
    });
    if (msg.seed) battle.prng = new PRNG(msg.seed);
    battle.setPlayer('p1', { name: 'p1', team: msg.p1_team });
    battle.setPlayer('p2', { name: 'p2', team: msg.p2_team });
    battles.set(msg.id, battle);
    return { ok: true, state: State.serializeBattle(battle) };
}

function handleStep(msg) {
    const battle = battles.get(msg.id);
    if (!battle) return { ok: false, error: `unknown id: ${msg.id}` };
    battle.makeChoices(msg.p1, msg.p2);
    const winner = battle.ended
        ? (battle.winner || 'draw')
        : null;
    return { ok: true, state: State.serializeBattle(battle), winner };
}

function handleRequest(msg) {
    const battle = battles.get(msg.id);
    if (!battle) return { ok: false, error: `unknown id: ${msg.id}` };
    return {
        ok: true,
        p1: battle.sides[0].activeRequest,
        p2: battle.sides[1].activeRequest,
    };
}

function handleFree(msg) {
    battles.delete(msg.id);
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
                case 'new':     resp = handleNew(msg);     break;
                case 'step':    resp = handleStep(msg);    break;
                case 'request': resp = handleRequest(msg); break;
                case 'free':    resp = handleFree(msg);    break;
                default:        resp = { ok: false, error: `unknown cmd: ${msg.cmd}` };
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

    Does not require a running Showdown server — the bridge loads the sim
    library directly via Node.js require().
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
        seed: str | None = None,
        battle_id: str | None = None,
    ) -> dict:
        """Start a battle with packed team strings.

        Returns the initial serializeBattle() state (turn 0).
        seed: optional 'gen5,xxxx' string for reproducible battles.
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
        return self._rpc(msg)['state']

    def step(self, p1: str, p2: str) -> tuple[dict, str | None]:
        """Advance one turn. Returns (new_state, winner_or_None)."""
        resp = self._rpc({'cmd': 'step', 'id': self._battle_id, 'p1': p1, 'p2': p2})
        return resp['state'], resp['winner']

    def request(self) -> tuple[dict, dict]:
        """Returns (p1_active_request, p2_active_request)."""
        resp = self._rpc({'cmd': 'request', 'id': self._battle_id})
        return resp['p1'], resp['p2']

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
packed teams, returns a JSON response with `ok: true` and a `state` object containing
`turn: 0`. No crash on startup.

```bash
echo '{"cmd":"new","id":"t1","p1_team":"<packed>","p2_team":"<packed>"}' \
  | node src/utils/bridge/sim_battle.js
# → {"ok":true,"state":{"turn":0,...}}
```

### M2 — Turn steps
`step` with `"move 1"` for both sides advances turn to 1, returns updated state.
`winner` is null for a normal turn; correctly set when a side loses all Pokémon.

### M3 — Request parses legal actions
`request` returns non-empty `active.moves` and correct `side.pokemon` list.
`legal_actions()` returns at least one valid string for both sides on every turn
across 10 complete battles.

### M4 — Python client wraps cleanly
`SimBattleClient` starts the subprocess, runs a full battle via `new → step* → free`,
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
    with patch('subprocess.Popen') as mock_popen:
        proc = MagicMock()
        proc.stdout.readline.return_value = '{"ok":true,"state":{"turn":0}}\n'
        mock_popen.return_value = proc
        client = SimBattleClient()
        state = client.new_battle('team1', 'team2', seed='gen5,1234')
        written = proc.stdin.write.call_args[0][0]
        msg = json.loads(written.rstrip('\n'))
        assert msg['cmd'] == 'new'
        assert msg['p1_team'] == 'team1'
        assert msg['seed'] == 'gen5,1234'
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
        state = client.new_battle(teams[0], teams[1])
        assert state['turn'] == 0

        turn = 0
        winner = None
        while winner is None and turn < 200:
            p1_req, p2_req = client.request()
            p1_action = legal_actions(p1_req)[0]
            p2_action = legal_actions(p2_req)[0]
            state, winner = client.step(p1_action, p2_action)
            turn += 1

        assert winner in ('p1', 'p2', 'draw')
        assert turn < 200, "battle did not finish in 200 turns"

@pytest.mark.integration
def test_bridge_reproducible_with_seed():
    teams = load_packed_teams('data/teams/')
    seed = 'gen5,1234567890abcdef'

    def run_battle(seed):
        outcomes = []
        with SimBattleClient() as client:
            client.new_battle(teams[0], teams[1], seed=seed)
            winner = None
            while winner is None:
                p1_req, p2_req = client.request()
                state, winner = client.step(
                    legal_actions(p1_req)[0],
                    legal_actions(p2_req)[0],
                )
                outcomes.append(state['turn'])
        return outcomes, winner

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
