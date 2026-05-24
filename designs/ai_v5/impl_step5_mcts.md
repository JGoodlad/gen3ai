# Implementation: Step 5 — MCTS

This step adds Monte Carlo Tree Search on top of the trained policy and value networks,
used at **inference time** as a policy improvement operator. The neural network is trained
entirely via PPO self-play (v3/v4); MCTS is not used to generate training data — that
comes later in v7 once the Rust sim makes it fast enough.

---

## Version Context

| Version | MCTS role | Simulator | Rollouts/turn |
|---------|-----------|-----------|---------------|
| ai_v5 | Inference only (ladder, eval) | Node.js bridge (JS) | ~1 000 |
| ai_v6 | Inference + very cheap training | Node.js bridge (JS) | ~20 per action in training |
| ai_v7 | Full training-time MCTS | Rust sim via PyO3 | 50 000+ |

v5 proves the approach and ships a working MCTS player. v6 integrates cheap MCTS into the
training loop. v7 replaces the JS bridge with the Rust sim for a ~50× throughput leap.

---

## Motivation

The PPO policy selects actions by a single forward pass — it does not look ahead. Against
strong opponents (the league agents from ai_v4, or humans on the ladder), short-horizon
planning matters: knowing that a predicted switch leads to a favourable matchup three turns
later changes the current move choice. MCTS provides this lookahead by simulating
trajectories through the game tree using the policy and value networks as heuristics,
without requiring the agent to store an explicit model of multi-turn dynamics.

**Why not use MCTS during training in v5?** Pokémon Showdown environment stepping is the
bottleneck — each rollout takes ~10ms of env stepping vs. negligible GPU inference. Using
MCTS to generate training data would reduce sample throughput by ~1000× (one rollout per
training step instead of one step). With 150M training steps needed, this is not feasible
with the JS bridge. The paper (Wang 2024) made this explicit: *"simulating the environment
is very slow... generating gameplay using MCTS would not likely lead to enough samples for
a neural network to converge."* The Rust sim in v7 changes this calculus.

---

## Phase 1 — Action Sampling

Before implementing the full UCB tree, deploy flat action sampling. It requires far less
machinery, is easy to debug, and delivers the "try each option K times before committing"
benefit the user asked for.

**Algorithm:**

1. At the start of a decision turn, enumerate all legal actions `a ∈ A` (4–6 in Gen 3 OU).
2. For each legal action, run `K` rollouts — fork the root, force that action for our side
   (with the opponent choosing via the neural-net policy), roll out `max_depth` turns.
3. Estimate `Q(root, a) = mean(returns across K rollouts)`.
4. Select `a* = argmax_a Q(root, a)`.

With `K=3` and `max_depth=3`, a 6-action turn costs 18 rollouts — completely within a
1-second budget. With 20 parallel workers, this drops to ~1 rollout per worker.

**Why K=3 specifically?** Three attempts per action expose early-turn randomness
(damage rolls, crits) without accumulating the compound variance of longer rollouts.
`V_θ` at the leaf handles the rest. Tune `K` and `max_depth` together:

| K | max_depth | Budget (6 actions, 1 worker) | Notes |
|---|-----------|------------------------------|-------|
| 3 | 3 | 18 rollouts | Phase 1 baseline; fast to ship |
| 10 | 3 | 60 rollouts | Better Q estimates, same depth |
| 20 | 5 | 120 rollouts | ~1 s with 20 workers |

Action sampling is the v5 baseline that goes to the ladder first. Full MCTS (Phase 2)
is added incrementally and compared head-to-head against it.

---

## Phase 2 — Full MCTS

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
| `M[s]` | Total visit count for state `s` |
| `N[s, a]` | Visit count for `(s, a)` |
| `α ∈ [0, 1]` | Exploration weight |
| `β ∈ [0, 1]` | Policy trust (`β=1` = standard PUCT; `β=0` = uniform exploration) |

`P[s, a]` is computed once per node on first visit and never synced between workers —
recomputed locally on demand. Recomputation (one forward pass) is cheaper than shipping
11-element distributions across processes.

### Rollout Policy

All rollout moves (both sides) are selected by the trained neural network policy from
the respective player's observation. A rollout terminates when:
- The game reaches a **terminal state** — value is `+1`, `-1`, or `0`
- The rollout reaches a **leaf node** — value is `V_θ(sT)`, leaf added to tree
- The rollout reaches `max_depth` turns — value is `V_θ(sT)`

### Stochasticity and max_depth

Gen 3 OU has meaningful per-turn randomness: damage rolls (±7.5%), critical hits
(~6.25%), miss chances, and variable sleep duration. A crit early in a rollout creates
a different game state that all subsequent turns evaluate from. With only ~50–100 rollouts
per action, rare-event branches can dominate Q rather than average out.

`V_θ(s)` is a better leaf estimator than a short noisy rollout — it was trained on
thousands of games where all random events occurred at their base rates. **Shallower depth
may outperform deeper depth:**

| `max_depth` | Variance source | Leaf estimator |
|-------------|-----------------|----------------|
| 20 | Accumulated crits compound | V_θ is a minor contributor |
| 5 | 1–2 crit opportunities | V_θ handles 95%+ of the evaluation |
| 3 | Negligible RNG accumulation | Nearly pure V_θ with action lookahead |

The primary benefit of MCTS in Gen 3 OU is "seeing that a switch now leads to a
favourable type matchup 3 turns out" — visible at depth 3–5 without accumulating
long-rollout variance.

**Recommended tuning order:** start at `max_depth=5`, compare `max_depth=3` and `max_depth=8`.
Do not target the Wang 2024 value of 20 (Random Battles has more diverse sets; longer
lookahead matters more there).

### State Representation

Five dicts keyed by state hash:

| Dict | Type | Content |
|------|------|---------|
| `Q[s, a]` | `float` | Empirical mean return from `(s, a)` |
| `N[s, a]` | `int` | Times action `a` was taken from state `s` |
| `M[s]` | `int` | Total visits to state `s` |
| `P[s, a]` | `float` | Neural network prior (cached per node) |
| `F[s]` | `int` | Total fainted Pokémon count in state `s` |

State hash: `(battle_tag, turn, our_team_hp_vector, opp_team_hp_vector, active_species_pair)`.

### Backup Rules

For each `(st, at)` along the rollout path, with leaf value `v`:

```python
Q[st, at] = (N[st, at] * Q[st, at] + v) / (N[st, at] + 1)
N[st, at] += 1
M[st]     += 1
```

Root action selection: `a* = argmax_a N[s0, a]` — visit count is more robust than Q at
the root (less-visited actions have higher Q variance).

### Tree Pruning

`F[s]` is monotonically non-decreasing. Once the live game reaches `f1` total fainted
Pokémon, prune all states `s'` where `F[s'] < f1` before each sync cycle.
Bounds tree size to 2,000–15,000 nodes during a typical game.

---

## Rollout Bridge

### Architecture

Each MCTS worker owns one persistent `node sim_bridge.js` subprocess, launched when
the worker starts and kept alive for the entire MCTS search. 20 workers = 20 Node.js
processes.

The bridge uses `BattleStream` for initial session setup (the designed, stable Showdown
API), then works directly with the `stream.battle` object. All fork operations are
**in-process `battle.toJSON()` / `Battle.fromJSON()` calls** — no serialized state ever
crosses IPC during rollouts.

```
Python                     Node.js bridge
  │  {"cmd":"new",...}         │
  │ ──────────────────────►    │  BattleStream.write(">start...")
  │                            │  BattleStream.write(">player p1 ...")
  │                            │  BattleStream.write(">player p2 ...")
  │  {"ok":true,"state":{}}    │  ◄── stream.battle for serializeBattle()
  │ ◄──────────────────────    │
  │                            │
  │  {"cmd":"fork",...}        │
  │ ──────────────────────►    │  Battle.fromJSON(root.toJSON())  ← in-process!
  │  {"ok":true}               │    fork.prng = new PRNG()
  │ ◄──────────────────────    │
  │                            │
  │  {"cmd":"step",...}        │
  │ ──────────────────────►    │  fork.makeChoices(p1, p2)
  │  {"ok":true,"state":{}}    │  ◄── State.serializeBattle(fork)
  │ ◄──────────────────────    │
```

### Root Synchronization

The bridge root session tracks the **real battle** turn-by-turn. After each real move
decision (once the live server confirms both players' choices), feed those moves to the
bridge so its root stays in sync:

```
→ {"cmd": "advance", "id": "root", "p1": "move 1", "p2": "switch 3"}
← {"ok": true}
```

This avoids any state transfer from poke-env — the bridge replays the same move sequence
and arrives at an identical state deterministically.

### Protocol

**Start a root session** (once per battle):
```
→ {"cmd": "new", "id": "root", "p1_team": "<packed>", "p2_team": "<packed>",
   "seed": [s0, s1, s2, s3]}
← {"ok": true, "state": { ...serializeBattle()... }}
```
Uses `BattleStream`:
```js
const stream = new BattleStream();
stream.write('>start {"formatid":"gen3anythinggoes","seed":[s0,s1,s2,s3]}');
stream.write(`>player p1 {"name":"p1","team":"${msg.p1_team}"}`);
stream.write(`>player p2 {"name":"p2","team":"${msg.p2_team}"}`);
// drain until |turn|1
sessions.set(msg.id, stream);
return { ok: true, state: State.serializeBattle(stream.battle) };
```

**Advance root** (after each real-game turn):
```
→ {"cmd": "advance", "id": "root", "p1": "move 1", "p2": "switch 3"}
← {"ok": true}
```
```js
stream.write(`>p1 ${msg.p1}`);
stream.write(`>p2 ${msg.p2}`);
// drain until |turn|N+1 or |win|...
```

**Fork to rollout session** (once per rollout):
```
→ {"cmd": "fork", "src": "root", "id": "r42"}
← {"ok": true}
```
```js
const fork = Battle.fromJSON(sessions.get(msg.src).battle.toJSON());
fork.prng = new PRNG();  // fresh seed — rollouts must diverge stochastically
battles.set(msg.id, fork);
```

**Step a rollout turn**:
```
→ {"cmd": "step", "id": "r42", "p1": "move 2", "p2": "move 1"}
← {"ok": true, "done": false, "state": { ...serializeBattle()... }}
← {"ok": true, "done": true, "winner": "p1", "state": { ... }}
```
Returns state so Python can encode the observation for neural-net leaf evaluation.

**Free a session**:
```
→ {"cmd": "free", "id": "r42"}
← {"ok": true}
```

---

## Handling Hidden Information (PIMC)

At the start of each MCTS trajectory:

1. Call `sampler.sample_completion(revealed_opponent_slots)` → one complete 6-mon team hypothesis.
2. Construct a fully-observed game state by filling the opponent's unrevealed slots with the
   sampled hypothesis.
3. Run the MCTS trajectory entirely in this fully-observed world.

Different trajectories see different hypotheses, so Q and N are averaged over the
distribution of plausible opponent teams. The team completion model (from ai_v4/v5 Step 4)
fills this role for Gen 3 OU.

---

## Parallelisation

### Architecture

20 worker processes + 1 aggregator process (following the reference implementation).

Each worker:
1. Receives the master tree (`Q, N, M, P` dicts) from the aggregator.
2. Runs 10 rollouts, updating its local copy.
3. Sends the updated tree back.
4. Repeats until the time budget is exhausted.

The aggregator merges received trees:
`Q_master[s,a] = weighted_average(Q_worker[s,a], N_worker[s,a])`.

**P is not synced.** Neural network priors are recomputed locally on first node visit —
cheaper than shipping 11-action distributions per node per sync cycle.

### VRAM

If 20 workers each hold the full model in GPU VRAM, memory becomes the bottleneck.
Start with CPU inference (the network is small enough for Gen 3's tiny action space);
profile before moving to GPU batching.

### Time Budget

The Showdown server enforces a per-turn limit (~10 s for ladder). The MCTS loop runs
until `time.monotonic() - turn_start > TIME_BUDGET` (default 8 s, leaving 2 s for the
round-trip).

---

## Integration with the Inference Player

```python
class MCTSPlayer(Gen3Player):
    def choose_move(self, battle):
        action = mcts_search(
            root_battle=battle,
            policy_fn=self.model.policy,
            value_fn=self.model.value,
            completion_model=self.completion_model,
            time_budget=self.time_budget,
            n_workers=self.n_workers,
        )
        return self.action_to_order(action, battle)
```

`mcts_search()` — blocking, returns the best action index. Phase 1 uses flat action
sampling; Phase 2 switches to the full UCB tree. The player's Showdown communication
logic is unchanged in both phases.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/utils/bridge/sim_bridge.js` | Persistent Node.js bridge: BattleStream setup, in-process forks, `new`/`advance`/`fork`/`step`/`free` |
| `src/agents/mcts/sim_client.py` | Python wrapper: manages one `sim_bridge.js` subprocess, exposes `new_battle()`, `advance()`, `fork()`, `step()`, `free()` |
| `src/agents/mcts/action_sampler.py` | Phase 1: K rollouts per legal action, returns Q estimates and best action |
| `src/agents/mcts/tree.py` | Phase 2: `MCTSTree` — Q, N, M, P, F; tree policy (α, β); backup; pruning |
| `src/agents/mcts/rollout.py` | Single trajectory: hidden-info sampling, sim_client stepping, value backup |
| `src/agents/mcts/worker.py` | Worker process: owns one `SimClient`, runs rollouts, sends tree to aggregator |
| `src/agents/mcts/aggregator.py` | Aggregator process: merge worker trees, broadcast master |
| `src/agents/mcts/search.py` | `mcts_search()` — top-level entry: spawn workers, return best action |
| `src/agents/inference/mcts_player.py` | `MCTSPlayer` — wraps `Gen3Player`, calls `mcts_search()` |
| `src/main/play_mcts.py` | Evaluation entry point for MCTS player |

## Files to Modify

| File | Change |
|------|--------|
| `src/agents/team_model/sampler.py` | Ensure `sample_completion()` is picklable (workers are forked processes) |

---

## Verification

1. **Bridge smoke test**: start bridge, send `new` with two packed teams, send `fork`,
   send `step` with `"move 1"` both sides, confirm `state.turn == 1`. Send `free`. No crash.

2. **Fork independence**: fork the same root 10 times, run 5 steps in each. Confirm
   step 3 damage rolls differ across forks (PRNG divergence working). A cluster of
   identical outcomes means `fork.prng = new PRNG()` is not being applied.

3. **Root sync**: run a 20-turn battle by alternating `advance` on the root, then forking
   and comparing `state.turn` to the expected value. Any desync means the root and the
   real battle are out of phase.

4. **Phase 1 smoke test** (`action_sampler.py`): run 10 rollouts on the first turn of a
   debug game with `K=3`, `max_depth=3`. Confirm Q estimates are non-trivial (not all 0)
   and the selected action is legal.

5. **Phase 2 tree sanity**: enable full MCTS (`n_workers=1`), run 100 rollouts. Confirm
   `Q`, `N`, `M` are populated, `F` is tracked, and the selected action matches the
   highest `N[root, a]`.

6. **Rollout distribution**: with PIMC active, log sampled team hypotheses across 100
   trajectories with 3 unrevealed opponent slots. Confirm diversity (not all identical)
   and all 6 slots populated.

7. **max_depth ablation**: run 50-game evaluations at `max_depth ∈ {3, 5, 8}` vs. the
   top league snapshot. Expect peak somewhere in 3–8.

8. **Time budget**: measure rollouts-per-second with 20 workers at the chosen `max_depth`.
   Target ≥ 100 rollouts/second (≥ 1000 rollouts in 10 s).

9. **Win rate**: MCTSPlayer vs. the best league agent from ai_v4. MCTS should improve
   win rate by ≥ 5 pp over the raw PPO policy at the same checkpoint.

---

## Final State

Step 5 is complete when:
- Phase 1 (action sampling, K=3) is on the ladder and collecting games
- Phase 2 (full MCTS) achieves ≥ 1000 rollouts/turn within the time budget
- Win rate vs. the league is measurably higher than the raw PPO policy
- Tree size stays bounded across a full game (pruning working)
- No hangs or crashes in 100-game evaluation runs

---

## Post-v5 Directions → v6 and v7

**ai_v6 — Cheap MCTS in Training**:
The Node.js bridge is too slow for full MCTS during PPO data collection, but very shallow
action sampling (`K=3`, `max_depth=1`) adds only ~30ms per decision. v6 integrates this
into the training loop so the model trains against better-quality action selections from
both sides. This closes the gap between "policy trained on random actions" and "policy
trained on MCTS actions" without waiting for the Rust sim.

**ai_v7 — Rust Sim → Training-time MCTS at Scale**:
The Rust sim (PyO3, in-process) targets 50 000+ rollouts/turn — enough for deep MCTS
during training data generation. With the Rust sim, PPO training can use full MCTS to
select every training-time action, dramatically improving data quality. The final model
can be evaluated against both fixed-depth search and full MCTS to measure the contribution
of each.
