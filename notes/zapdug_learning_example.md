# ZapDug RL Training Results (1,000 Steps)

Comparison of learning performance using the "ZapDug" sample team.

```text
Starting RL training with team: ZapDug
Using cpu device
Training for 1000 steps...
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 25.2     |
|    ep_rew_mean     | -42.7    |
| time/              |          |
|    fps             | 492      |
|    iterations      | 1        |
|    time_elapsed    | 4        |
|    total_timesteps | 2048     |
---------------------------------
Training complete. Model saved to models/gen3ou_ppo_20260509_221804/final_model

Starting Evaluation...
Evaluating against RandomPlayer (100 battles)...
Win rate vs Random: 78%
Evaluating against SimpleHeuristicsPlayer (100 battles)...
Win rate vs Heuristic: 1%
```

### Analysis vs Gengar Superman TSS:
| Metric | Gengar TSS (1k steps) | ZapDug (1k steps) |
|--------|-----------------------|-------------------|
| Win Rate vs Random | 91% | 78% |
| Win Rate vs Heuristic | 22% | 1% |

**Observations:**
- ZapDug is significantly harder for a raw agent to pilot.
- The 1% win rate against the heuristic bot suggests that without specific trap-logic (Dugtrio usage), the team's individual power is lower than a standard TSS core.
