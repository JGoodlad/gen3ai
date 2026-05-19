# Implementation: Step 4 — MCTS

This step adds Perfect Information Monte Carlo (PIMC) search on top of the trained
policy and value networks. At the start of each search trajectory, the team completion
model samples one complete team hypothesis for the opponent's unrevealed slots; the
trajectory runs in that fully-observed world. Visit counts across trajectories aggregate
over the uncertainty, producing an action selection that is optimal in expectation.

## Motivation

The PPO policy selects actions by a single forward pass — it does not look ahead. Against
strong opponents (the league agents from ai_v4, or humans on the ladder), short-horizon
planning matters: knowing that a predicted switch leads to a favourable matchup three turns
later changes the current move choice. MCTS provides this lookahead by simulating
trajectories through the game tree using the policy and value networks as heuristics,
without requiring the agent to store an explicit model of multi-turn dynamics.

The reference implementation (Huang et al., Gen 4 Random Battles) achieved 1000–2000
rollouts within the 10-second per-turn decision limit using 20 parallel workers. Gen 3 OU
has a more constrained action space than Gen 4 (fewer legal switches mid-game due to
fixed teams) which may allow deeper search at the same rollout budget.

---

## Algorithm

### PUCT Selection

At each node in the search tree, select the action maximising:

```
PUCT(s, a) = Q[s, a] + c_puct × P[s, a] × sqrt(N[s]) / (1 + N[s, a])
```

Where:
- `Q[s, a]` — empirical mean return from taking action `a` in state `s` (updated after each rollout)
- `P[s, a]` — neural network policy prior (probability of choosing `a` in state `s`)
- `N[s]` — total visit count for state `s`
- `N[s, a]` — visit count for the specific `(s, a)` pair
- `c_puct` — exploration constant (default 1.5; tune based on rollout depth)

`P[s, a]` is computed once per node on first visit, cached for the lifetime of the search.
It is **not** sent between workers (recomputed locally — cheap relative to env stepping).

### Rollout Policy

Rather than rolling out to terminal state with a random policy, use the trained neural
network policy for all rollout moves. This produces far more realistic game trajectories
and reduces the variance of the value estimate at the cost of a GPU forward pass per turn.
At the leaf node, the value head provides a scalar estimate without completing the game.

The rollout terminates when:
- The game reaches a terminal state (`battle.finished`), or
- Rollout depth exceeds `max_depth` (default 20 turns — beyond this, the value head is
  used as a terminal estimate regardless)

### Opponent Modeling

During rollouts, the opponent's moves are selected by the same neural network policy,
from the opponent's perspective (mirrored observation). This is the simplest possible
opponent model and assumes the opponent plays similarly to our agent. Against exploiters
(from ai_v4 league) or human players who deviate significantly, this assumption weakens —
but it is a far better prior than random play, and improving it further is deferred to a
post-v5 research question.

### State Representation

The tree is stored as four dictionaries keyed by state hash:

| Dict | Type | Content |
|------|------|---------|
| `Q[s, a]` | `float` | Mean return from `(s, a)` |
| `N[s, a]` | `int` | Visit count for `(s, a)` |
| `N[s]` | `int` | Total visit count for `s` |
| `P[s, a]` | `float` | Neural network prior for `(s, a)` (cached on first visit) |
| `F[s]` | `int` | Total fainted Pokémon count in state `s` |

State hash: hash of `(battle_tag, turn, our_team_hp_vector, opp_team_hp_vector, active_species_pair)`.
Avoid hashing the full observation — it is slow and unnecessary.

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
| `src/agents/mcts/tree.py` | `MCTSTree` — Q, N, P, F dicts; PUCT selection; tree pruning |
| `src/agents/mcts/rollout.py` | Single trajectory: hidden-info sampling, env stepping, value backup |
| `src/agents/mcts/worker.py` | Worker process: run 10 rollouts, send tree to aggregator |
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
