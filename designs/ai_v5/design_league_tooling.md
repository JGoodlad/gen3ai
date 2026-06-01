# Design: League Tooling

The self-play pool (Step 1) and the league (Step 2) are only as good as our ability to
**measure their health and inspect their behaviour**. This document specifies the off-policy
tooling that the diversity metrics, the progress metrics, and the human-in-the-loop debugging
all depend on. None of it exists yet, and the **payoff-matrix tournament runner is the linchpin**
— every league health metric consumes its output.

## Why tooling is the bottleneck, not the algorithm

The Step 2 design specifies `nash_effective_diversity`, `nontransitivity_score`,
`pfsp_effective_opponents`, and relative population performance. All four are functions of an
**N×N pairwise win-rate matrix** across league members. Today nothing produces that matrix.
The PFSP win-rate EMAs collected *during training* are too noisy and too main-agent-centric to
serve as the matrix (they only measure main-vs-X, never X-vs-Y). So the metrics are stranded
until the tournament runner exists. Build it first.

---

## 1. Payoff-matrix tournament runner (build first)

### Purpose

Given a set of league members (snapshot `.zip` paths), play every ordered pair K games and
emit the win-rate matrix plus sampled replays. This is an **offline** tool — run on demand or
every K-million steps from a callback — and it must not perturb the live training server.

### Bridge-backed, no live server

Reuse `utils/bridge/local_battle_runner.py` (the same in-process `BattleStream` path that
powers every fuzz test — see root `CLAUDE.md`). No `npm run showdown`, no port, no websockets.
Two frozen `RLPlayer`s are driven head-to-head in-process. This means the runner can execute
while a training server occupies `:8001` without touching it, and parallelises across CPU
cores with no port contention.

### Output: `payoff_matrix.json`

```json
{
  "generated_at_step": 84000000,
  "games_per_pair": 200,
  "members": [
    {"id": "self_step_010000000", "role": "self",      "step": 10000000, "git_hash": "abc1234"},
    {"id": "exploit_0_frozen_r1", "role": "exploiter",  "step":  3200000, "git_hash": "def5678"},
    {"id": "seed",                "role": "seed",        "step":        0, "git_hash": "0000000"}
  ],
  "win_rate": [[0.50, 0.71, 0.93], [0.29, 0.50, 0.66], [0.07, 0.34, 0.50]],
  "n_games":  [[200,  200,  200 ], [200,  200,  200 ], [200,  200,  200 ]]
}
```

`win_rate[i][j]` = P(member i beats member j). The diagonal is 0.5 by convention (not played).
Teams are randomised per game exactly as in training, so the matrix reflects the team
distribution the agents actually face, not a fixed matchup.

### Cost control

A full N×N is `O(N²)` pairs. With the permanent pool growing unboundedly, cap it:
- Always include: the latest main snapshot, all frozen exploiters, the seed.
- Subsample the recency pool to `max_matrix_members` (default 12) evenly spaced.
- `log()` exactly which members were dropped — silent truncation reads as "we measured
  everything" when we didn't.

### File: `src/agents/training/payoff_matrix.py`

```python
def run_tournament(
    members: list[SnapshotEntry],
    games_per_pair: int = 200,
    sample_replays: int = 2,        # replays to archive per cell
    out_dir: Path = ...,
) -> PayoffMatrix: ...
```

---

## 2. Diversity & progress metrics module

### File: `src/agents/training/league_metrics.py`

Pure functions over a `PayoffMatrix`. These are the Step 2 metrics, now with a real matrix to
consume:

| Function | Returns | Notes |
|----------|---------|-------|
| `pfsp_effective_opponents(weights)` | exp(H) of the PFSP weight dist | 1 → collapsed, N → uniform |
| `nontransitivity_score(W)` | fraction of non-transitive triples | rock-paper-scissors index |
| `nash_meta(W)` | σ* (Nash mixture of the zero-sum payoff game) | via LP — see below |
| `nash_effective_diversity(W)` | exp(H(σ*)) | how many strategies the Nash needs |
| `relative_population_performance(W, latest_idx)` | scalar | **the robust progress metric** |

**Nash via LP.** The win-rate matrix (centred to `W − 0.5`) is an antisymmetric zero-sum game;
its symmetric Nash equilibrium σ* solves a small linear program (`scipy.optimize.linprog`, N ≤
~30 so it is instant). σ* puts weight only on the strategies that cannot be dominated — padding
the league with redundant agents does not inflate `nash_effective_diversity`, which is exactly
why AlphaStar reported Nash-based metrics rather than Elo.

**Relative population performance (RPP).** Given σ*, RPP measures how the **latest** agent
scores against the Nash mixture of the rest:
`RPP = Σ_j σ*_j · (win_rate[latest][j] − 0.5)`. Positive and rising ⇒ the latest agent
genuinely beats the equilibrium of its own league ⇒ real progress, robust to non-transitivity.
This is the league-era replacement for the dangling `league/main_elo`.

**Optional Glicko-2 (human-readable sugar only).** If a single "ladder-like" number is wanted
for the TUI, run Glicko-2 over the same pairwise results (`league_metrics.glicko2(matrix)`).
Glicko-2 (not plain Elo) because it models rating *uncertainty*, which matters with K=200
samples per pair. **Not load-bearing** — never gate promotion, reset, or curriculum on it; it
is misleading under the rock-paper-scissors dynamics the league deliberately creates. RPP and
`win_rate_vs_bots` are the metrics that gate decisions.

---

## 3. Behavioural descriptors (outcome-blind diversity)

Nash / non-transitivity measure **outcome** diversity. They are blind to **behavioural**
collapse — two snapshots that play near-identically but happen to trade wins look "diverse" to
a win-rate matrix. To catch mode collapse that PFSP weights cannot see, log cheap behavioural
descriptors per eval battle, computed **from the event log** (`battle/` `TurnView` / event
stream — no extra battles):

| Descriptor | Definition |
|------------|------------|
| `lead_dist` | distribution over which team slot is led |
| `switch_rate` | voluntary switches per turn |
| `hazard_rate` | fraction of games where hazards are set |
| `status_rate` | status-move uses per turn |
| `mean_turn_len` | mean game length |
| `sweep_share` | fraction of wins decided by a single sweeper (≥3 KOs by one mon) vs attrition |

Per-member descriptor vectors → a pool-wide **descriptor spread** (mean pairwise L2 distance).
A falling spread while `nash_effective_diversity` looks fine is the early warning that the
stable is converging behaviourally — the signal to force an exploiter reset or lower the PFSP
floor before the outcome metrics catch up.

---

## 4. League inspector / report

### File: `src/agents/training/league_report.py`

Reads the snapshot directory + each member's `metadata.json` + the latest `payoff_matrix.json`
+ descriptor logs, and renders a single static **HTML report** (and a compact text summary for
the launcher TUI). Per member:

- lineage: `role` (self / exploiter / seed), `step`, `git_hash`, `base_snapshot`
- current PFSP weight and Nash support σ*_i
- behavioural descriptor vector
- row of the win-rate matrix (who it beats / loses to)

Plus league-level: `nash_effective_diversity`, `nontransitivity_score`, RPP trajectory,
descriptor spread, and the alarm-threshold table (green/amber) from `impl_step2_league_play.md`.

This is the artifact you open when "is the league healthy?" needs an answer in 30 seconds.

---

## 5. Per-matchup replay sampler

For any cell (A, B) of the matrix, dump a handful of full battle replays so you can **watch**
what an exploiter is doing — the single most valuable feedback loop for obs/reward design,
because a confirmed exploit usually points at a missing observation feature or a gameable
shaping term. The tournament runner already archives `sample_replays` per cell; this is the
viewer/exporter (Showdown-replay-format `.html` or the project's existing replay format) keyed
by `(member_a, member_b)`.

---

## 6. Provenance schema extension

Every snapshot already writes `metadata.json` with `git_hash` (model versioning). Extend it so
lineage is reconstructable without a side database:

```json
{ "git_hash": "...", "step": 10000000, "role": "exploiter",
  "base_snapshot": "self_step_009000000.zip", "resets": 2 }
```

`SnapshotPool._write()` and the exploiter manager set `role` and `base_snapshot`. The inspector
and the payoff runner read them. This keeps the **whole league reconstructable from the
directory** — the same crash-safety property the Step 1 pool already has.

---

## Files

| File | Purpose |
|------|---------|
| `src/agents/training/payoff_matrix.py` | **New.** Tournament runner (bridge-backed) + `payoff_matrix.json` |
| `src/agents/training/league_metrics.py` | **New.** Nash / non-transitivity / RPP / effective-opponents / Glicko-2 |
| `src/agents/training/league_report.py` | **New.** HTML + TUI league inspector |
| `src/agents/training/behavioral_descriptors.py` | **New.** Descriptor extraction from the event log |
| `src/agents/training/snapshot_pool.py` | Set `role` / `base_snapshot` in `metadata.json` |
| `src/agents/training/payoff_matrix_test.py` | **New.** Matrix shape, diagonal, truncation logging |
| `src/agents/training/league_metrics_test.py` | **New.** Nash on hand-built RPS matrices; RPP sign |

---

## Build order

1. **`payoff_matrix.py`** — unblocks everything. Validate against a tiny 3-member pool.
2. **`league_metrics.py`** — Nash/RPP/non-transitivity over the matrix; unit-test on a known
   rock-paper-scissors matrix (`nontransitivity_score → 1.0`, `nash` uniform).
3. **`behavioral_descriptors.py`** — wire into the self-play eval loop.
4. **`league_report.py`** — assemble the above into the inspector.
5. Glicko-2 last (optional sugar).

The first two are prerequisites for Step 2's diversity-alarm thresholds to mean anything; the
rest can land alongside league play.
