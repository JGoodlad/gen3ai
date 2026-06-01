# Implementation: Step 2 — Per-Team Specialisation

Fine-tune one model per top-3 team, starting from the v6 generalist checkpoint, with the
team fixed for the duration of training. The three runs are independent and can run in
parallel.

## Motivation

The v6 generalist spreads probability mass over all 32 teams. With a fixed team, the
model can sharpen its understanding of the specific win conditions that team enables:
a Spikes + Roar team needs Skarmory kept healthy for much longer than usual; a
Trick Room team has a completely different speed-tier intuition than an offensive team.

Because the architecture already encodes the full own-team observation in every forward
pass, team conditioning is implicit — fine-tuning with a fixed team doesn't require
any architectural change. It simply trains the weights to be optimal for that specific
starting configuration rather than the average across all 32.

---

## Training Setup

### Fixed Team

Pass the team file directly to `Gen3Teambuilder` as a single-entry pool. The env samples
from this pool each episode — with only one entry, it always uses the same team.

The opponent continues to draw from the full v5 league pool (PFSP-weighted), so the
model learns to play the fixed team against the full range of opponents it will face on
the ladder.

### Hyperparameters

Start from the v6 MCTS checkpoint. Fine-tuning is a shorter run than generalist training:

| Parameter | Value |
|-----------|-------|
| Starting checkpoint | `models/v5_best.zip` |
| Steps | 10M (extend to 15M if win rate still rising at 10M) |
| `n_envs` | 32 (half of generalist — fixed team reduces variance so fewer envs needed) |
| `lr` | 1e-4 (lower than generalist — we are sharpening, not relearning) |
| `ent_coef` | 0.01 (lower entropy target — specialisation should reduce breadth) |
| `n_steps`, `batch_size`, `n_epochs` | Unchanged from v6 |

### Opponent Sampling

Use the v5 league pool with PFSP sampling (same as league training). The specialised
model should remain capable against all league members, not just the easiest ones.

Optionally, after 5M steps, add the other two specialised team models (once they have
partially converged) as additional opponents at 20% weight. This trains each model to
handle the stable's internal matchups — useful if the three teams are eventually deployed
on the same account in rotation.

### Checkpointing

Save a checkpoint every 500K steps. The best checkpoint is determined by win rate against
a fixed evaluation set (50 games vs. the top league snapshot + 50 games vs.
`SimpleHeuristicsPlayer`), not by training reward.

---

## Run Command

Three independent runs, one per team:

```bash
export PYTHONPATH=$PYTHONPATH:src

# Team A
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py \
  --model models/v5_best.zip \
  --fixed-team data/team_eval/top3/team_a.txt \
  --steps 10000000 \
  --n-envs 32 \
  --lr 1e-4 \
  --ent-coef 0.01 \
  --league-dir models/v4_league/ \
  --save-dir models/v6_team_a/ \
  --device cuda

# Team B and Team C: same command, different --fixed-team and --save-dir
```

If three GPUs are available, launch all three simultaneously. If only one GPU, run
sequentially (~10 hours per run at 1M steps/hour).

---

## Files to Create

None — the only new artefact is a `--fixed-team` CLI flag.

## Files to Modify

| File | Change |
|------|--------|
| `src/main/train_rl_agent.py` | `--fixed-team PATH` — when set, pass a single-entry team pool to `Gen3Teambuilder` for player 1; opponent continues using the full pool |
| `src/agents/training/gen3_env.py` | Accept `fixed_team: str | None` in constructor; replace team sampler with a constant if set |

---

## Verification

1. **Learning curve**: `train/ep_rew_mean` should trend upward within the first 500K
   steps. If it is flat or declining, the `lr=1e-4` starting point may be too aggressive
   — reduce to 5e-5.

2. **Win rate progression**: spot-eval every 500K steps. Win rate against the top league
   snapshot should exceed the generalist's win rate with the same team (from Step 1) by
   the end of training. If not, the fine-tuning is not helping — diagnose entropy collapse
   first (`train/entropy_loss`).

3. **No regression on basics**: win rate vs. `SimpleHeuristicsPlayer` should stay ≥ 85%.
   A specialised model that forgets how to beat easy opponents has collapsed, not specialised.

---

## Final State

Step 2 is complete when all three models have converged (win rate stable for 500K+ steps)
and each outperforms the generalist on its target team. Checkpoints saved to:

```
models/v6_team_a/best.zip
models/v6_team_b/best.zip
models/v6_team_c/best.zip
```

**Ready for Step 3: Ladder Run**

Each checkpoint is paired with its team file. `play_ladder.py` loads the checkpoint and
passes the fixed team so the ladder agent always uses the right team/model pair.
