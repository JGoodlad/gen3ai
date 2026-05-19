# Implementation: Step 4 — MCTS

This step adds Monte Carlo Tree Search on top of the trained policy and value networks,
used purely at **inference time** as a policy improvement operator. The neural network is
trained entirely via PPO self-play (v3/v4); MCTS is never used to generate training data.
At the start of each search trajectory, the team completion model samples one complete
team hypothesis for the opponent's unrevealed slots; the trajectory runs in that
fully-observed world using Pokémon Showdown as the game simulator.

## Motivation

The PPO policy selects actions by a single forward pass — it does not look ahead. Against
strong opponents (the league agents from ai_v4, or humans on the ladder), short-horizon
planning matters: knowing that a predicted switch leads to a favourable matchup three turns
later changes the current move choice. MCTS provides this lookahead by simulating
trajectories through the game tree using the policy and value networks as heuristics,
without requiring the agent to store an explicit model of multi-turn dynamics.

**Why not use MCTS during training?** Pokémon Showdown environment stepping is the
bottleneck — each rollout takes ~10ms of env stepping vs. negligible GPU inference. Using
MCTS to generate training data would reduce sample throughput by ~1000× (one rollout per
training step instead of one step). With 150M training steps needed, this is not feasible.
The paper (Wang 2024) made this explicit: *"simulating the environment is very slow,
compared to a game like chess; generating gameplay using MCTS would not likely lead to
enough samples for a neural network to converge."*

The reference implementation achieved 1000–2000 rollouts within the 10-second per-turn
decision limit using 20 parallel workers. Gen 3 OU has a more constrained action space
than Gen 4 (fewer legal switches mid-game due to fixed teams) which may allow deeper
search at the same rollout budget.

---

## Algorithm

### Tree Policy

At each node, select the action maximising (Wang 2024, eq. 2.3):

```
at = argmax_a ( Q[s, a] + α · U(s, a) )

where U(s, a) = P[s, a]^β · sqrt(M[s]) / (N[s, a] + 1)
```

| Symbol | Meaning |
|--------|---------|
| `Q[s, a]` | Empirical mean return from taking action `a` in state `s` |
| `P[s, a]` | Neural network policy prior `π_θ(a \| s)` |
| `M[s]` | Total visit count for state `s` (incremented once per rollout visit) |
| `N[s, a]` | Visit count for `(s, a)` (incremented once per selection) |
| `α ∈ [0, 1]` | Exploration weight — how much to value the exploration bonus |
| `β ∈ [0, 1]` | Policy trust — how closely to follow the prior vs. treat actions uniformly |

`β = 1` recovers standard PUCT (full trust in prior); `β = 0` ignores the prior and
explores uniformly. Both are hyperparameters to tune.

`P[s, a]` is computed once per node on first visit and **never synced between workers**
— it is recomputed locally by each worker on demand. Recomputation is cheap (one forward
pass) relative to env stepping, and sending 11-element distributions across processes is
more expensive than recomputing them.

### Rollout Policy

All rollout moves (both our side and opponent) are selected by the trained neural network
policy from the respective player's observation. This produces realistic game trajectories
and avoids the high-variance returns of random rollouts.

A rollout terminates when:
- The game reaches a **terminal state** (`battle.finished`) — value is `+1`, `-1`, or `0`
- The rollout reaches a **leaf node** (a state not yet recorded in the tree) — value is
  `V_θ(sT)` from the critic head; the leaf is then added to the tree

Leaf termination is what makes MCTS tractable without running every rollout to completion.
A `max_depth` cap (default 20) is a practical guard against very long rollouts, but the
primary termination is hitting a leaf node.

### Opponent Modeling

During rollouts, the opponent's moves are selected by the same neural network policy,
from the opponent's perspective (mirrored observation). This is the simplest possible
opponent model and assumes the opponent plays similarly to our agent. Against exploiters
(from ai_v4 league) or human players who deviate significantly, this assumption weakens —
but it is a far better prior than random play, and improving it further is deferred to a
post-v5 research question.

### State Representation

The tree is stored as five dictionaries keyed by state hash:

| Dict | Type | Content |
|------|------|---------|
| `Q[s, a]` | `float` | Empirical mean return from `(s, a)` |
| `N[s, a]` | `int` | Number of times action `a` was taken from state `s` |
| `M[s]` | `int` | Number of times state `s` was visited (= `Σ_a N[s,a]`) |
| `P[s, a]` | `float` | Neural network prior (cached per node, not synced between workers) |
| `F[s]` | `int` | Total fainted Pokémon count in state `s` |

State hash: hash of `(battle_tag, turn, our_team_hp_vector, opp_team_hp_vector, active_species_pair)`.
Avoid hashing the full observation — it is slow and unnecessary.

### Backup Rules

When a rollout ends at state `sT` with value `v` (either `±1/0` for terminal, or
`V_θ(sT)` for a leaf), apply for each `(st, at)` along the rollout path:

```python
Q[st, at] = (N[st, at] * Q[st, at] + v) / (N[st, at] + 1)
N[st, at] += 1
M[st]     += 1
```

After all rollouts, select the action with the highest visit count from the root:

```python
a* = argmax_a N[s0, a]
```

Max visit count rather than max Q — less-visited actions have higher variance in their
Q estimates, so visit count is the more robust selection criterion at the root.

### Tree Pruning

`F[s]` — total fainted Pokémon count — is monotonically non-decreasing during a game.
Once the live game reaches `f1` total fainted Pokémon, all states `s'` where `F[s'] < f1`
can never be revisited. Prune them from the dictionaries before each sync cycle.

This bounds tree size to 2,000–15,000 nodes during a typical game (per the reference
implementation), keeping sync payloads small enough for efficient inter-process transfer.

---

## Handling Hidden Information (PIMC)

At the start of each MCTS trajectory:

1. Call `sampler.sample_completion(revealed_opponent_slots)` → one complete 6-mon team hypothesis.
2. Construct a fully-observed game state by filling the opponent's unrevealed slots with the
   sampled hypothesis.
3. Run the MCTS trajectory entirely in this fully-observed world.

Different trajectories see different hypotheses, so the visit counts in `Q` and `N` are
averaged over the distribution of plausible opponent teams. The action at the root with
the highest visit count `N[root, a]` is selected — it is the action that performs best
across the sampled worlds.

This is the same strategy the reference implementation uses for Random Battles (where
unknown sets are sampled via the server's generation procedure). The team completion
model fills the equivalent role for Gen 3 OU.

---

## Parallelisation

### Architecture

20 worker processes + 1 aggregator process (following the reference implementation).

Each worker:
1. Receives a copy of the master tree (`Q, N, M, P` dicts) from the aggregator.
2. Runs 10 rollouts, updating its local copy of the tree.
3. Sends the updated tree back to the aggregator.
4. Repeats until a time budget is exhausted.

The aggregator:
1. Waits for worker updates.
2. Merges received trees into the master copy: `Q_master[s,a] = weighted_average(Q_worker[s,a], N_worker[s,a])`.
3. Broadcasts the updated master tree to all waiting workers.

**P is not synced.** Neural network priors are recomputed locally by each worker on first
node visit. Sending a 11-action distribution per node per sync cycle is expensive; recomputing
it via a local GPU forward pass is faster given the small action space.

### Implementation

Workers and the aggregator run as Python `multiprocessing.Process` instances. The shared
tree state is passed via `multiprocessing.Queue` (or `Pipe` for lower latency). Each
worker holds a reference to the neural network (loaded in-process, not shared).

If 20 workers each hold the full model in GPU memory, VRAM becomes the bottleneck. Mitigations:
- Use CPU inference for rollout moves (the network is small enough that CPU throughput
  is acceptable for Gen 3's tiny action space)
- Or share a single GPU model via a dedicated inference server process that workers
  submit batches to

Start with CPU inference for simplicity; profile before committing to GPU batching.

### Time Budget

The Showdown server enforces a per-turn decision limit (~10 seconds for ladder play;
longer in local play). The MCTS loop runs until `time.monotonic() - turn_start > TIME_BUDGET`
(default 8 seconds, leaving 2 seconds for network round-trip).

Target: 1000–2000 rollouts per turn. Profile on the dev machine; adjust `max_workers`
and `max_depth` accordingly.

---

## Rollout Bridge

### State Serialization

Pokémon Showdown's `Battle` class has full round-trip serialization built in
(`deps/pokemon-showdown/dist/sim/state.js`):

```js
const json = battle.toJSON();           // Battle → plain object
const battle2 = Battle.fromJSON(json);  // plain object → new Battle
battle2.restart(send);                  // wire up output callback
```

`toJSON()` serializes the entire battle graph: sides, each Pokémon's set and in-battle
state, the action queue, PRNG seed, field conditions, and log. `fromJSON()` fully
reconstructs it. This means **state injection is a solved problem** — no custom
serialization needed.

The PRNG seed is captured in the snapshot, so rollouts from the same state are
deterministic across workers (each worker receives a copy with the same seed but runs
independently — stochastic divergence comes from different team hypotheses sampled by
the team completion model, not from PRNG variance).

### Bridge Architecture: One Persistent Process per Worker

Each MCTS worker owns one persistent `node sim_bridge.js` subprocess, launched when
the worker starts and kept alive for the duration of the MCTS search. 20 workers = 20
Node.js processes. Rationale:

- **No shared state**: each worker runs fully independent rollouts with its own in-memory
  `Battle` object — no locking, no message contention
- **Startup cost amortized**: the ~80ms Node.js startup happens once per turn's search,
  not once per rollout
- **Simple protocol**: line-delimited JSON over stdin/stdout, same as the existing bridge
  pattern in `src/utils/bridge/`
- **No socket overhead**: Unix domain sockets or TCP would add latency per round-trip;
  pipes are zero-copy on Linux

A single shared Node.js server with `worker_threads` is not worth the complexity —
Showdown's sim is synchronous CPU code and gains nothing from sharing a V8 heap across
simulations.

### Protocol

Each round-trip is one JSON line in, one JSON line out:

**Step a game turn:**
```json
→ {"cmd": "step", "state": <battle_json>, "p1": "move 1", "p2": "switch 3"}
← {"state": <new_battle_json>, "done": false, "winner": null}
← {"state": <final_battle_json>, "done": true,  "winner": "p1"}
```

**Query available choices:**
```json
→ {"cmd": "choices", "state": <battle_json>}
← {"p1": ["move 1", "move 2", "switch 3"], "p2": ["move 1", "switch 2", "switch 4"]}
```

`<battle_json>` is the output of `battle.toJSON()` — a plain JSON object, not a
string-escaped blob.

`state` in the step response is only returned when needed (leaf node reached or terminal)
— the Python rollout code tracks observations itself, so most responses are just
`{"done": false}`.

### Files to Create (Bridge)

| File | Purpose |
|------|---------|
| `src/utils/bridge/sim_bridge.js` | Persistent Node.js process: loads Battle from JSON, steps game, returns new state + done flag |
| `src/agents/mcts/sim_client.py` | Python wrapper: manages one `sim_bridge.js` subprocess, exposes `step(state_json, p1_choice, p2_choice) → (new_state, done, winner)` |

`sim_bridge.js` follows the same pattern as `validate_team.js` and `get_hp.js` — reads
newline-delimited JSON from stdin, writes newline-delimited JSON to stdout, imports from
`deps/pokemon-showdown/dist/sim/`.

---

## Integration with the Inference Player

`Gen3Player` (in `src/agents/inference/player.py`) currently selects actions via a single
policy forward pass. The MCTS variant wraps this:

```python
class MCTSPlayer(Gen3Player):
    def choose_move(self, battle):
        root_state = build_state(battle)
        action = mcts_search(
            root=root_state,
            policy_fn=self.model.policy,
            value_fn=self.model.value,
            completion_model=self.completion_model,
            time_budget=self.time_budget,
            n_workers=self.n_workers,
        )
        return self.action_to_order(action, battle)
```

`mcts_search()` is a blocking call that returns the best action index. The player's
existing `action_to_order()` and Showdown communication logic are unchanged.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/utils/bridge/sim_bridge.js` | Persistent Node.js bridge: load Battle from JSON, step one game turn, return new state + outcome |
| `src/agents/mcts/sim_client.py` | Python wrapper around `sim_bridge.js`: manages subprocess lifetime, exposes `step()` / `choices()` |
| `src/agents/mcts/tree.py` | `MCTSTree` — Q, N, M, P, F dicts; tree policy (α, β); backup; pruning |
| `src/agents/mcts/rollout.py` | Single trajectory: hidden-info sampling, sim_client stepping, value backup |
| `src/agents/mcts/worker.py` | Worker process: owns one `SimClient`, runs 10 rollouts, sends tree to aggregator |
| `src/agents/mcts/aggregator.py` | Aggregator process: merge worker trees, broadcast master |
| `src/agents/mcts/search.py` | `mcts_search()` — top-level entry point: spawn workers, collect result |
| `src/agents/inference/mcts_player.py` | `MCTSPlayer` — wraps `Gen3Player`, calls `mcts_search()` |
| `src/main/play_mcts.py` | Evaluation entry point with MCTS player |

## Files to Modify

| File | Change |
|------|--------|
| `src/agents/team_model/sampler.py` | Ensure `sample_completion()` is picklable (workers are forked) |

---

## Verification

1. **Single-worker smoke test**: disable parallelism (`n_workers=1`); run 10 rollouts on
   the first turn of a debug game. Confirm `Q` and `N` are populated, `F` is tracked, and
   the selected action is legal.

2. **Tree pruning**: after a faint event, confirm that all pre-faint states are removed
   from `Q`, `N`, `P`, `F` before the next sync cycle. Check tree size stays < 20K nodes
   across a full game.

3. **Rollout distribution**: with the team completion model active, log the sampled team
   hypotheses across 100 trajectories on a turn where 3 opponent slots are unrevealed.
   Confirm the hypotheses are diverse (not all identical) and all 6 slots are populated.

4. **Time budget**: on the dev machine, measure rollouts-per-second with 20 workers.
   Target ≥ 100 rollouts/second to reach 1000 rollouts in 10 seconds. If below target,
   profile first before reducing `max_depth` (depth reduction is a last resort).

5. **Win rate**: MCTSPlayer vs. the best league agent from ai_v4. MCTS should improve
   win rate by ≥ 5 percentage points over the raw PPO policy at the same checkpoint.
   A smaller or zero gain suggests the value head is noisy and may need fine-tuning.

---

## Final State

Step 4 is complete when:
- MCTSPlayer achieves ≥ 1000 rollouts per turn within the time budget
- Win rate vs. the league is measurably higher than the raw PPO policy
- Tree size stays bounded across a full game (pruning is working)
- No hangs or crashes in 100-game evaluation runs

**Post-v5 directions:**
- Opponent model: replace neural-net-policy opponent with an exploiter ensemble to handle
  diverse human play styles (section 5.2.3 of the reference paper)
- Value fine-tuning: if value head estimates are noisy, a short supervised fine-tuning
  pass on terminal-state values from the league replay log may help
- Distilled rollout policy: a lightweight policy clone for faster rollouts at greater depth
