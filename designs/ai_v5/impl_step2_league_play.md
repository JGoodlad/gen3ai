# Implementation: Step 2 — League Play

This step extends self-play into a structured league with dedicated exploiter agents and
prioritised opponent sampling (PFSP). Exploiters find systematic weaknesses in the Main
Agent; the Main Agent must generalise past those exploits. Together they break the
local-equilibrium traps that self-play alone eventually hits.

> **Status: forward design (not yet built).** Step 1 self-play has landed; nothing in this
> document exists in code yet. It depends on reward annealing (`design_reward_annealing.md`)
> and the league tooling (`design_league_tooling.md`).

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

The two ideas this document adds beyond the original Step 2 sketch:
1. **The stable is a two-pool structure** (bounded recency + unbounded permanent), so the
   league never forgets a strategy it once needed to counter.
2. **Progress and diversity are measured off a payoff matrix** (Nash / RPP), not off a
   win rate against a moving pool — resolving the ELO ambiguity the original draft left open.

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
1. Its final checkpoint is frozen and added to the **permanent pool** of the league.
2. It resets to the current Main Agent checkpoint and begins finding a new exploit.

The frozen exploiter checkpoints are permanent — the Main Agent must keep beating them
even as it develops new strategies. This accumulates a diverse set of "known weaknesses"
that the Main Agent can never forget.

### League Size

Start with 2 exploiters. The permanent pool grows unboundedly; PFSP sampling (below)
handles the prioritisation so old exploits the Main Agent has clearly solved are sampled
rarely but never dropped.

---

## The Stable — two-pool structure

Step 1 ships a **single bounded sliding window** (`SnapshotPool`, `max_snapshots=20`, evict
oldest non-pinned, only the step-0 seed pinned). That is correct for a curriculum but is a
diversity hazard for a league: if the Main Agent cycles back into an old exploitable habit,
the snapshot that countered it may already have been evicted. AlphaStar's league never evicts.

Split the stable into two cooperating pools:

| Pool | Bound | Contents | Role |
|------|-------|----------|------|
| **Recency** | sliding window (~20) | recent Main-Agent snapshots | the curriculum; churns naturally |
| **Permanent** | unbounded | seed, **milestone** Main snapshots, **all frozen exploiters** | the diversity guarantee; never evicted |

A snapshot enters the **permanent** pool when it is a frozen exploiter, or when it is a
*milestone* Main snapshot (promoted on a genuine improvement — see Hardened Promotion). All
other promotions land in the recency pool and are eventually evicted. PFSP samples across the
**union** of both pools, so solved members become rare-but-present rather than disappearing.

Implementation: extend `SnapshotPool` with a `permanent: bool` flag per entry (a generalisation
of the existing `pinned`), or run two `SnapshotPool` instances behind a `Stable` facade. The
two-instance facade is cleaner and keeps the well-tested Step 1 pool unchanged; prefer it.

---

## Hardened Promotion

Step 1 promotes on `win_rate_vs_pool > promote_threshold` alone. In a league this can promote
a snapshot that is strong-vs-pool but has **regressed on the fixed bots** (overfit to the
pool) — poisoning the stable with a narrower agent. Gate promotion on **both** conditions:

```python
def should_promote(win_rate_vs_pool, win_rate_vs_bots, recent_bot_max, eps=0.03):
    return (win_rate_vs_pool > promote_threshold
            and win_rate_vs_bots >= recent_bot_max - eps)
```

A snapshot that clears **both** gates is a *milestone* and enters the **permanent** pool.
A snapshot that clears only the pool gate (but has regressed on bots) may still enter the
recency pool (it is a valid recent opponent) but is **not** marked permanent. This keeps the
permanent diversity guarantee honest.

---

## PFSP Sampling

The Main Agent's opponent is sampled from the full stable each episode using Prioritised
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

Win-rate estimates are an exponential moving average over the last ~1000 battles against each
opponent (`α=0.05`), so estimates track the current Main Agent state, not the full training
history.

**Mode mixing (AlphaStar lesson).** A single static `mode` is brittle. AlphaStar's main agents
mixed sampling: mostly `hard` (drive toward the agents you lose to) with a `variance` fraction
(competitive 50/50 matches give the cleanest gradient) plus the floor below. Default:
`0.75·hard + 0.20·variance + 0.05·uniform`, exposed as `--pfsp-mix`. The uniform component is
the minimum-weight floor (default 0.05) ensuring every opponent — including mastered ones that
reappear as exploiter base checkpoints — is sampled occasionally.

---

## Progress & diversity measurement (resolves the ELO question)

**The key reframing:** in a league, `win_rate_vs_pool` is **not** a progress metric. The pool
is a moving target — the Main Agent can sit at ~55% against it indefinitely while genuinely
strengthening or weakening. Progress must be measured against something stationary.

The original Step 2 draft referenced a `league/main_elo` from an undefined "ELO tracker". That
is **removed.** Plain Elo is actively misleading in the strongly non-transitive
(rock-paper-scissors) regime league play creates — which is precisely why AlphaStar reported
Nash-based metrics, not Elo, as its headline. The resolution:

| Metric | Role | Source |
|--------|------|--------|
| `eval/win_rate_vs_bots` | **primary progress anchor** | fixed bots never change → stationary; already built (Step 1) |
| `league/relative_population_performance` | **robust league progress** | latest agent vs the Nash mixture of the league (`league_metrics.py`) |
| `league/nash_effective_diversity` | league health / diversity | Nash σ* entropy over the payoff matrix |
| `league/main_glicko2` | *optional* human-readable number | Glicko-2 over pairwise results — **not load-bearing**, never gates anything |

`win_rate_vs_bots` is the day-to-day "are we improving" axis (free, stationary). RPP is the
non-transitivity-robust confirmation that the latest agent beats the equilibrium of its own
league. Glicko-2 (chosen over plain Elo because it models rating uncertainty under sparse
pairwise samples) is optional TUI sugar only. All of these consume the payoff matrix produced
by the tournament runner in `design_league_tooling.md`.

---

## Diversity Monitoring

The payoff matrix (`design_league_tooling.md` §1) feeds the league-health metrics. The PFSP
weight distribution is the cheap real-time signal; the Nash metrics are the rigorous periodic
ones; behavioural descriptors catch what win rates miss.

### Effective Number of Opponents (real-time)

Shannon entropy of the PFSP weight distribution gives a scalar diversity index:

```python
def pfsp_effective_opponents(weights: dict[str, float]) -> float:
    w = np.array(list(weights.values()))
    w = w / w.sum()
    return float(np.exp(-np.sum(w * np.log(w + 1e-8))))
```

Log as `league/pfsp_effective_opponents`. Ranges 1.0 (all weight on one opponent) to N
(uniform). Healthy: `pfsp_effective_opponents / league_size > 0.3`. Below 0.1 the Main Agent
has collapsed strategically — force exploiter resets or reduce the PFSP floor.

### Non-Transitivity Score (periodic, from the payoff matrix)

```python
def nontransitivity_score(W: np.ndarray) -> float:
    """Fraction of (A,B,C) triples that cycle (A>B, B>C, C>A or the reverse).
    0.0 = fully transitive (low diversity); higher = rock-paper-scissors dynamics."""
    n = len(W); count = total = 0
    for i, j, k in itertools.combinations(range(n), 3):
        a, b, c = W[i,j] > 0.5, W[j,k] > 0.5, W[k,i] > 0.5
        if (a and b and c) or (not a and not b and not c):
            count += 1
        total += 1
    return count / total
```

Log as `league/nontransitivity_score`. With 5+ frozen exploits, healthy > 0.15; below 0.05 the
exploiter pool is not producing genuinely diverse coverage.

### Nash Effective Diversity (rigorous, infrequent)

The win-rate matrix (centred to `W − 0.5`) is an antisymmetric zero-sum game; its symmetric
Nash equilibrium σ* solves a small LP (`scipy.optimize.linprog`). `N_eff = exp(H(σ*))`. N_eff
= 1 ⇒ one strategy dominates; N_eff = K ⇒ K strategies are all necessary. Robust to padding the
league with redundant agents. Log as `league/nash_effective_diversity`.

### Behavioural diversity (outcome-blind)

Win-rate diversity can miss two snapshots that play near-identically. Track the pool-wide spread
of behavioural descriptors (lead distribution, switch rate, hazard rate, status rate, turn
length, sweep-vs-attrition share) from the event log — see `design_league_tooling.md` §3. A
falling descriptor spread while Nash diversity looks fine is the early warning of behavioural
convergence.

### Alarm Thresholds

| Metric | Healthy | Warning | Action |
|--------|---------|---------|--------|
| `pfsp_effective_opponents / league_size` | > 0.3 | < 0.1 | Reduce PFSP floor; force exploiter reset |
| `nontransitivity_score` | > 0.15 (5+ exploits) | < 0.05 | Increase `n_exploiters`; check exploit threshold |
| `nash_effective_diversity / league_size` | > 0.3 | < 0.1 | Same as above |
| behavioural descriptor spread | stable/rising | falling | Force exploiter reset before outcome metrics catch up |
| Exploiter timeout rate | < 20% | > 50% | Main Agent has converged; ready for v6 (MCTS) |

An exploiter **timeout** (hits `max_exploit_steps` without reaching the 70% threshold) means
the Main Agent has already covered that strategic region. A timeout rate above 50% is a strong
signal the Main Agent is strategically mature and the league is ready for v6 MCTS.

---

## File-backed league state

Step 1's pool is reconstructable from its directory (crash-safe — no manifest). Extend that
discipline to the **whole** league so the state survives crashes and so the single-process →
multi-process migration (below) is "add more writers", not a rewrite. `league_state.json` is a
**cache/inspection view** rebuilt from the snapshot directory + `metadata.json` provenance, not
the source of truth:

```json
{
  "main_agent": { "latest_snapshot": "snapshots/main_step_012000000.zip" },
  "exploiters": [
    { "id": 0, "base": "snapshots/main_step_010000000.zip",
      "current": "snapshots/exploit_0_step_001200000.zip",
      "win_rate_vs_main": 0.54, "resets": 2 },
    { "id": 1, "base": "snapshots/main_step_011000000.zip",
      "current": "snapshots/exploit_1_step_000600000.zip",
      "win_rate_vs_main": 0.33, "resets": 1 }
  ],
  "frozen_exploits": ["snapshots/exploit_0_frozen_r1.zip", "snapshots/exploit_0_frozen_r2.zip"],
  "pfsp_win_rate_ema": { "...member id...": 0.41 }
}
```

Per-member provenance (`role`, `base_snapshot`, `resets`) lives in each snapshot's
`metadata.json` (`design_league_tooling.md` §6), so the directory alone reconstructs the league.

---

## Training Coordination

Two options, ordered by implementation complexity. **Build Option A first, but write the
league state as a file-backed artifact from day one so the A → B migration adds writers
rather than rewriting state.**

### Option A — Single-Process, Time-Multiplexed (recommended first)

One `train_league.py` process alternates between Main Agent and exploiter rollout collection:

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
            freeze exploiter → permanent pool; reset exploiter to current Main Agent
```

This needs a **hot-swap opponent path** in `Gen3Env` (the `_staged_opponent_path` mechanism
deferred in Step 1's todo) so the per-episode opponent can change without a launcher restart.
That mechanism does **not** exist yet and is a prerequisite for Option A.

**Pro:** single process; no shared-filesystem coordination. **Con:** Main and exploiters share
throughput.

### Option B — Multi-Process (parallel exploiters)

One process per agent (1 Main + N exploiters), all writing to the shared snapshot directory; a
`LeagueCoordinator` reads/writes `league_state.json` and orchestrates resets. **Pro:** full
parallel throughput; a crashing exploiter does not kill the Main Agent. **Con:** file-locking
and race conditions on shared state.

Migrate to B only if single-process throughput becomes the bottleneck (typically > 3 concurrent
exploiters).

---

## Exploiter Reset Policy

An exploiter resets when either:
- Its EMA win rate vs. the Main Agent (last ~1000 battles) exceeds `exploit_threshold` (70%), **or**
- It has trained `max_exploit_steps` without reaching the threshold (default 3–5M) — the
  exploit attempt failed.

On reset:
1. Load the current Main Agent checkpoint as the new exploiter base.
2. Reset exploiter EMA win rate to 0.5 (neutral prior).
3. Log `league/exploiter_N_resets`.

Only **successful** resets (win rate > threshold) produce a frozen checkpoint that joins the
permanent pool. Failed timeouts are discarded (no useful exploit) but the timeout itself is a
diversity signal (see Alarm Thresholds).

---

## Metrics

| Metric | Source | Interpretation |
|--------|--------|----------------|
| `eval/win_rate_vs_bots` | fixed-bot eval | **primary progress anchor** (stationary) |
| `league/relative_population_performance` | `league_metrics.py` | robust league progress vs Nash mixture |
| `league/nash_effective_diversity` | payoff matrix + LP | diversity / health |
| `league/nontransitivity_score` | payoff matrix | rock-paper-scissors index |
| `league/pfsp_effective_opponents` | PFSP weights | real-time concentration |
| `league/pfsp_weights` | per opponent | which opponents are hardest now |
| `league/exploiter_N_win_rate` | EMA | exploiter convergence speed |
| `league/exploiter_N_resets` | counter | exploits found so far |
| `league/frozen_exploit_count` | permanent pool size | diversity of permanent league |
| `league/behavioural_spread` | descriptors | outcome-blind diversity |
| `league/main_glicko2` | pairwise results | *optional* human-readable number (non-gating) |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/agents/training/stable.py` | Two-pool `Stable` facade (recency + permanent) over `SnapshotPool` |
| `src/agents/training/pfsp_sampler.py` | PFSP weight computation, mode mixing, opponent sampling |
| `src/agents/training/exploiter_manager.py` | Exploiter lifecycle: base load, reset, freeze → permanent |
| `src/agents/training/league_callback.py` | SB3 callback: Main + exploiter alternation, reset triggers, hardened promotion |
| `src/agents/training/league_state.py` | `LeagueState` view + JSON (re)serialisation from the directory |
| `src/main/train_league.py` | Entry point — wraps `train_rl_agent.py` logic with league setup |

(The payoff matrix, league metrics, behavioural descriptors, and inspector are in
`design_league_tooling.md`; reward annealing is in `design_reward_annealing.md`.)

## Files to Modify

| File | Change |
|------|--------|
| `src/agents/training/selfplay_callback.py` | Expose `win_rate_ema` per opponent for PFSP; hardened-promotion gate |
| `src/agents/training/snapshot_pool.py` | `permanent` flag; `opponent_filter` arg to `sample()`; set `role`/`base_snapshot` |
| `src/agents/training/gen3_env.py` | Hot-swap `_staged_opponent_path` (per-episode opponent swap, no restart) |

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
  --pfsp-mix 0.75,0.20,0.05 \
  --league-dir models/v4_league \
  --reward-anneal-start 50000000 \
  --reward-anneal-end 70000000 \
  --device cuda
```

The checkpoint from the end of self-play already has `num_timesteps ≈ 75M`, past the
`anneal_end` of 70M, so shaping is at the Tier-A floor (0) from the first league step — the
value head is win-probability-aligned for the MCTS handoff. The step counter is global and
preserved across checkpoint loads. See `design_reward_annealing.md` for the tier semantics.

---

## Verification

1. **Exploiter convergence**: `--debug --steps 100000`; exploiter win-rate EMA moves away from
   0.5 within the first 50K steps.
2. **Hardened promotion**: a snapshot strong-vs-pool but regressed-on-bots enters the recency
   pool but is **not** marked permanent; a snapshot clearing both gates is permanent.
3. **PFSP weights**: after 5M Main steps, recently-frozen exploits have higher weight than old
   Main snapshots (the Main Agent beats its own past more easily than fresh exploits).
4. **Payoff metrics**: run the tournament runner on a 3-member league; confirm
   `nontransitivity_score`, `nash_effective_diversity`, and RPP are produced and that RPP is
   positive when the latest member beats the others.
5. **No regression on heuristics**: `eval/win_rate_vs_bots` stays ≥ 0.80 throughout.
6. **Reset cycle**: run until ≥ 2 exploiter resets; confirm each frozen exploit joins the
   permanent pool and the Main Agent's win rate against it drops over subsequent training.

---

## Final State

Step 2 is complete when:
- At least 5 distinct frozen exploits are in the permanent pool (diverse weakness coverage).
- `league/relative_population_performance` has risen materially beyond its Step 1 self-play
  plateau (the non-transitivity-robust "the Main Agent got stronger" signal).
- PFSP sampling is stabilised — no single opponent permanently dominates the weight
  distribution (the Main Agent patches exploits rather than letting them fester).
- Exploiter timeout rate climbs above ~50% (the Main Agent is strategically mature).

**Ready for v6: MCTS**

League play produces a policy strong enough to serve as the rollout policy and value
estimator for PIMC search. The team completion model (see `designs/ai_v6/`) provides the
world-sampling step. League snapshots also become the opponents that MCTS must plan against.
Reward annealing (`design_reward_annealing.md`) has by this point left the value head
estimating win probability — the quantity MCTS leaf evaluation needs.
