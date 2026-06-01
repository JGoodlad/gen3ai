# Implementation: Step 1 — Team Evaluation

Run the v6 MCTS agent through all 32 sample teams, measure win rate per team against the
v5 league pool, and select the top 3 for specialisation.

## Motivation

The v6 generalist trains with a random team each episode, so it learns to play well
across the full distribution of 32 sample teams. But "well on average" hides real
variance — some teams play to the agent's strengths (proactive switch decisions, Spikes
pressure) and others expose its weaknesses (passive mons that benefit from more precise
predictions). Running a structured evaluation surfaces this variance and lets us invest
specialisation effort where it will pay off most.

This step requires no training — it is pure evaluation and takes a few hours.

---

## Evaluation Protocol

### Matchup Structure

For each of the 32 sample teams, run `N_GAMES` (default 200) battles:

- **Our agent**: v6 MCTS player, fixed to the team under evaluation
- **Opponent**: drawn from the v5 league pool using PFSP sampling (same distribution
  used during league training), so the evaluation reflects the range of opponents the
  specialised model will face

Run the full 32 × 200 = 6400 games. With parallelism across CPUs this takes ~2 hours at
the speed of local Showdown battles.

### Metrics Per Team

| Metric | Description |
|--------|-------------|
| `win_rate` | Fraction of games won (primary ranking key) |
| `avg_turns` | Mean game length — shorter games suggest dominant wins or quick losses |
| `faint_delta` | Mean (our fainted − opp fainted) at game end — positive = attrition advantage |
| `confidence_interval` | ±1.96 × sqrt(p(1-p)/N) — flag teams where CI > 5% |

Teams within the CI of each other should be considered tied. If the top-3 rankings are
ambiguous (e.g., teams 3 and 4 are within CI), increase `N_GAMES` to 500 for those
specific teams before making the final cut.

### Output

Results written to `data/team_eval/results.json`:

```json
{
  "model": "models/v5_best.zip",
  "n_games": 200,
  "teams": [
    {
      "team_file": "data/teams/skarmbliss_tar.txt",
      "win_rate": 0.74,
      "avg_turns": 28.3,
      "faint_delta": 1.2,
      "ci_95": 0.061
    },
    ...
  ]
}
```

Sorted by `win_rate` descending. The top 3 entries are the specialisation targets.

---

## Implementation

### `src/main/eval_teams.py`

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/eval_teams.py \
  --model models/v5_best.zip \
  --team-dir data/teams/ \
  --league-dir models/v4_league/ \
  --n-games 200 \
  --n-workers 8 \
  --output data/team_eval/results.json
```

Internally:
1. Load all `.txt` files from `--team-dir` (the 32 sample teams).
2. For each team, spin up `n_workers` parallel battle environments, each with the fixed
   team for player 1 and a PFSP-sampled league opponent for player 2.
3. Collect results and compute metrics.
4. Write `results.json` and print a ranked table to stdout.

The MCTS player is used for our side (`MCTSPlayer` from v6). League opponents use their
own `RLPlayer` (inference only, no MCTS — they are the training-time opponents, not
another MCTS agent).

### Parallelism

Each worker is an independent `Gen3Env` + Showdown server connection. The 32 teams are
distributed across workers round-robin; each worker runs one team at a time to completion
before moving to the next. No synchronisation needed between workers — results are
collected into a shared list via `multiprocessing.Queue`.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/main/eval_teams.py` | Evaluation script: fixed-team battles, metrics, results.json |

## Files to Modify

| File | Change |
|------|--------|
| `src/agents/inference/mcts_player.py` | Accept `fixed_team: str | None` — when set, always use this team instead of sampling |

---

## Verification

1. **Sanity check**: run 20 games with a known-strong team (e.g., the team the v3/v4
   agent was already informally tested with). Win rate should be ≥ 65% — if lower, the
   v6 MCTS player may not be calibrated yet.

2. **Variance check**: run the same team twice with different random seeds. Win rates
   should be within 5% of each other. Larger variance indicates MCTS stochasticity is
   dominating — increase `N_GAMES`.

3. **Ranking stability**: if teams 3 and 4 are within CI, run 500 games for those two
   specifically before making the final top-3 cut.

---

## Final State

Step 1 is complete when `results.json` is written, the top 3 teams are identified with
overlapping CIs resolved, and the team files are copied to `data/team_eval/top3/` for
use in Step 2.

**Ready for Step 2: Per-Team Specialisation**

Each of the 3 team files in `data/team_eval/top3/` becomes the fixed team for one
fine-tuning run.
