# Implementation: Step 2 — League Play

This step extends self-play into a structured league with dedicated exploiter agents and
prioritised opponent sampling (PFSP). Exploiters find systematic weaknesses in the Main
Agent; the Main Agent must generalise past those exploits. Together they break the
local-equilibrium traps that self-play alone eventually hits.

## Motivation

Self-play against a pool of past selves is powerful but fragile. The agent can converge
to a Nash equilibrium against its own historical distribution — a strategy that beats all
historical copies but is systematically exploitable by a fresh opponent trained to target
it. Concretely: if the Main Agent leans on Sand + Pursuit, a fresh exploiter quickly
learns to run Shed Shell Skarmory and Magneton, and the Main Agent has no pressure to
patch this because those counters never appear in its own snapshot pool.

AlphaStar's league training solved this with three agent roles. For Gen 3 OU, the full
three-tier structure is more complexity than the strategy space warrants. A two-tier
league (Main Agent + Exploiters) captures the essential dynamic.

---

## League Structure

### Main Agent

The generalist. Trains against a **weighted mixture** of all league members — its own
past snapshots plus frozen exploiter checkpoints — using PFSP sampling (below). The Main
Agent's snapshots are the primary signal for long-term improvement; exploiters apply
targeted pressure to prevent strategic laziness.

### Exploiters

Specialists that train only against the **current Main Agent checkpoint**, not the full
league. Their sole purpose is to find and amplify whatever the Main Agent currently does
poorly. They start from the current Main Agent checkpoint (not random init) so they begin
from a position of competence and diverge towards the exploit, rather than wasting steps
learning basics.

Each exploiter runs until it achieves > `exploit_threshold` win rate against the Main
Agent (default 70%), at which point:
1. Its final checkpoint is frozen and added to the league as a permanent member.
2. It resets to the current Main Agent checkpoint and begins finding a new exploit.

The frozen exploiter checkpoints are permanent — the Main Agent must keep beating them
even as it develops new strategies. This accumulates a diverse set of "known weaknesses"
that the Main Agent can never forget.

### League Size

Start with 2 exploiters. The pool of frozen exploiter checkpoints grows unboundedly;
PFSP sampling (below) handles the prioritisation so old exploits that the Main Agent has
clearly solved are sampled rarely but never dropped entirely.

---

## PFSP Sampling

The Main Agent's opponent is sampled from the full league each episode using Prioritised
Fictitious Self-Play:

```python
def pfsp_weight(win_rate: float, mode: str = "hard") -> float:
    if mode == "hard":
        # Concentrate effort on opponents near the agent's skill ceiling
        return (1.0 - win_rate) ** 2
    elif mode == "variance":
        # Maximise learning signal; peaks at 50% win rate
        return win_rate * (1.0 - win_rate)
```

Default: `mode="hard"`. Win rate estimates are maintained as an exponential moving
average over the last 1000 battles against each opponent (`α=0.05`), so estimates track
the current Main Agent state, not the full training history.

A minimum weight floor (default 0.05) ensures every opponent is sampled at least
occasionally — the Main Agent must not completely forget how to beat opponents it has
already mastered, as those opponents reappear as exploiter base checkpoints.

---

## Training Coordination

Two options, ordered by implementation complexity:

### Option A — Single-Process, Time-Multiplexed (recommended first)

One `train_rl_agent.py` process alternates between Main Agent training steps and
exploiter training steps within a single episode batch:

```
for each rollout collection:
    with probability main_fraction (default 0.7):
        collect rollout for Main Agent vs. PFSP-sampled opponent
        PPO update for Main Agent
    else:
        select active exploiter (round-robin)
        collect rollout for Exploiter vs. current Main Agent snapshot
        PPO update for Exploiter
        if exploiter.eval_win_rate > exploit_threshold:
            freeze exploiter → league
            reset exploiter to current Main Agent checkpoint
```

Both agents share the same `Gen3Env` / Showdown server. The opponent swap mechanism from
Step 1 (`_staged_opponent_path`) handles switching between Main and exploiter perspective.

**Pro:** Simpler to implement; single process; no shared-filesystem coordination.  
**Con:** Main Agent and exploiters cannot train in parallel; total throughput is shared.

### Option B — Multi-Process (parallel exploiters)

One process per agent (1 Main + N exploiters), all writing to a shared snapshot directory.
A `LeagueCoordinator` process reads `league_state.json` and orchestrates resets and ELO
updates.

`league_state.json` schema:

```json
{
  "main_agent": { "latest_snapshot": "snapshots/main_step_012000000.zip", "elo": 1847 },
  "exploiters": [
    { "id": 0, "base": "snapshots/main_step_010000000.zip",
      "current": "snapshots/exploit_0_step_001200000.zip",
      "win_rate_vs_main": 0.54, "resets": 2 },
    { "id": 1, "base": "snapshots/main_step_011000000.zip",
      "current": "snapshots/exploit_1_step_000600000.zip",
      "win_rate_vs_main": 0.33, "resets": 1 }
  ],
  "frozen_exploits": [
    "snapshots/exploit_0_frozen_r1.zip",
    "snapshots/exploit_0_frozen_r2.zip"
  ]
}
```

**Pro:** Full parallel throughput; Main Agent is never starved by exploiter updates.  
**Con:** File-locking, race conditions on `league_state.json`, harder to debug.

Implement Option A first. Migrate to Option B if the single-process throughput becomes
the bottleneck (typically > 3 exploiters running simultaneously).

---

## Exploiter Reset Policy

An exploiter resets when either:
- Its estimated win rate vs. Main Agent (EMA, last 1000 battles) exceeds `exploit_threshold` (70%), **or**
- It has trained for `max_exploit_steps` steps without reaching the threshold (default 3M steps) — the exploit attempt failed; reset with the Main Agent at a new base.

On reset:
1. Load current Main Agent checkpoint as the new exploiter base.
2. Reset exploiter EMA win rate to 0.5 (neutral prior).
3. Log reset event to TensorBoard: `league/exploiter_N_resets`.

Failed resets (hit `max_exploit_steps` without converging) are discarded — they provide
no useful frozen exploit. Only successful resets (win rate > threshold) produce a frozen
checkpoint that joins the permanent league.

---

## Metrics

| Metric | Source | Interpretation |
|--------|--------|----------------|
| `league/main_elo` | ELO tracker | Overall strength of Main Agent |
| `league/pfsp_weights` | Logged per opponent | Which opponents are hardest right now |
| `league/exploiter_N_win_rate` | EMA | Exploiter convergence speed |
| `league/exploiter_N_resets` | Counter | How many exploits found so far |
| `league/frozen_exploit_count` | Pool size | Diversity of permanent league |
| `eval/win_rate_vs_heuristic` | Spot eval | Regression guard vs. baseline |

---

## Diversity Monitoring

The PFSP weight distribution is the primary signal for league health. If it collapses —
all weight concentrating on 1–2 opponents while others sit at the floor — the Main Agent
has effectively solved the pool except for those opponents and is no longer receiving
broad strategic pressure.

### Effective Number of Opponents

Shannon entropy of the PFSP weight distribution gives a scalar diversity index:

```python
def pfsp_effective_opponents(weights: dict[str, float]) -> float:
    w = np.array(list(weights.values()))
    w = w / w.sum()
    return float(np.exp(-np.sum(w * np.log(w + 1e-8))))
```

Log as `league/pfsp_effective_opponents`. The value ranges from 1.0 (all weight on one
opponent) to N (uniform across all N league members). A healthy league should have
`pfsp_effective_opponents / league_size > 0.3`. Below 0.1, the Main Agent has collapsed
strategically and needs more exploiter resets or a reduced PFSP floor.

### Non-Transitivity Score

Periodically (every 5M steps), compute the N×N win-rate matrix across a sample of league
members and score what fraction of triples are non-transitive:

```python
def nontransitivity_score(W: np.ndarray) -> float:
    """Fraction of (A, B, C) triples where A>B, B>C, C>A (or reverse).
    0.0 = fully transitive (low diversity), higher = rock-paper-scissors dynamics."""
    n = len(W)
    count, total = 0, 0
    for i, j, k in itertools.combinations(range(n), 3):
        a_b, b_c, c_a = W[i,j] > 0.5, W[j,k] > 0.5, W[k,i] > 0.5
        if (a_b and b_c and c_a) or (not a_b and not b_c and not c_a):
            count += 1
        total += 1
    return count / total
```

Log as `league/nontransitivity_score`. With 5+ frozen exploits, a healthy score is > 0.15.
Below 0.05 (near-transitive ordering), the exploiter pool is not producing genuinely
diverse coverage and the pool needs more varied exploit strategies.

### Nash Effective Diversity (rigorous, computed infrequently)

The win-rate matrix is an empirical zero-sum game. Its Nash equilibrium σ* assigns weight
to strategies the Main Agent cannot dominate. The effective diversity is:

```
N_eff = exp(H(σ*))   where H(σ*) = -Σ σ*_i log σ*_i
```

Compute via linear programming every 10M steps. N_eff = 1 means one strategy dominates
Nash; N_eff = K means K strategies are all necessary to cover the opponent pool. Log as
`league/nash_effective_diversity`.

### Alarm Thresholds

| Metric | Healthy | Warning | Action |
|--------|---------|---------|--------|
| `pfsp_effective_opponents / league_size` | > 0.3 | < 0.1 | Reduce PFSP floor; force exploiter reset |
| `nontransitivity_score` | > 0.15 (5+ exploits) | < 0.05 | Increase `n_exploiters`; check exploit threshold |
| `nash_effective_diversity / league_size` | > 0.3 | < 0.1 | Same as above |
| Exploiter timeout rate | < 20% | > 50% | Main Agent has converged; reduce `exploit_threshold` |

An exploiter **timeout** (hits `max_exploit_steps` without reaching the 70% threshold) is
distinct from a reset failure — it means the Main Agent has already covered that strategic
region. A timeout rate above 50% is a strong signal the Main Agent is strategically
mature and the league is ready for v5 (MCTS).

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/agents/training/league_state.py` | `LeagueState` dataclass + JSON serialisation |
| `src/agents/training/pfsp_sampler.py` | PFSP weight computation + opponent sampling |
| `src/agents/training/exploiter_manager.py` | Exploiter lifecycle: base load, reset, freeze |
| `src/agents/training/league_callback.py` | SB3 callback: Main Agent + exploiter alternation, reset triggers |
| `src/main/train_league.py` | Entry point — wraps `train_rl_agent.py` logic with league setup |

## Files to Modify

| File | Change |
|------|--------|
| `src/agents/training/selfplay_callback.py` | Expose `win_rate_ema` per opponent for PFSP |
| `src/agents/training/snapshot_pool.py` | Add `opponent_filter` arg to `sample()` for role-specific pools |

---

## CLI Example

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_league.py \
  --model models/v4_selfplay_best.zip \
  --steps 75000000 \
  --n-envs 64 \
  --n-exploiters 2 \
  --exploit-threshold 0.70 \
  --max-exploit-steps 5000000 \
  --main-fraction 0.7 \
  --pfsp-mode hard \
  --league-dir models/v4_league \
  --reward-anneal-start 50000000 \
  --reward-anneal-end 70000000 \
  --device cuda
```

The anneal range matches the self-play run. The checkpoint from the end of self-play
already has `num_timesteps ≈ 75M`, which is past the `anneal_end` of 70M — shaping is
0.0 from the first league step. Passing the same flags is still correct; the step counter
is global and preserved across checkpoint loads.

---

## Verification

1. **Exploiter convergence**: In a short debug run (`--debug --steps 100000`), confirm
   exploiter win rate EMA moves away from 0.5 within the first 50K steps — exploiters
   should be learning to exploit the Main Agent.

2. **PFSP weights**: Log `league/pfsp_weights` per opponent; after 5M Main Agent steps,
   recently-frozen exploits should have higher weight than old Main Agent snapshots (the
   Main Agent beats its own past more easily than fresh exploits).

3. **No regression on heuristics**: Spot-eval `eval/win_rate_vs_heuristic` every 5M
   steps. Should stay ≥ 80% — the league should never cause the Main Agent to forget
   basic competency.

4. **Reset cycle**: Run until at least 2 exploiter resets. Confirm each frozen exploit
   checkpoint is correctly added to the PFSP pool and that subsequent Main Agent training
   reduces its win rate against that frozen checkpoint over time.

---

## Final State

Step 2 is complete when:
- At least 5 distinct frozen exploits are in the league (evidence of diverse weakness coverage)
- Main Agent ELO has increased > 200 points beyond its Step 1 self-play plateau
- PFSP sampling is stabilised — no single opponent dominates the weight distribution permanently (evidence that the Main Agent patches exploits rather than letting them fester)

**Ready for v5: MCTS**

League play produces a policy strong enough to serve as the rollout policy and value
estimator for PIMC search. The team completion model (see `designs/ai_v5/`) provides the
world-sampling step. League snapshots also become the opponents that MCTS must plan against.
