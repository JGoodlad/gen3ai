---
name: project_reward_shaping_verification
description: How to verify reward_manager.py shaping changes — retrain needed; offline pathology-replay is the proxy
metadata: 
  node_type: memory
  type: project
  originSessionId: 61b007eb-c2e3-42cd-884f-871d3e37dd3d
---

> **Archived 2026-09-08** — May-2026 method note; superseded by the reward golden (ledger 2026-09-08 gen3_reward_golden_v1). Preserved verbatim; nothing below is current.

Reward-shaping changes in `src/agents/training/reward_manager.py` cannot be empirically
validated by re-running a frozen checkpoint's inference games: the reward function does
not alter an already-trained policy's action distribution, so the old pathology rates
reproduce identically regardless of the change. A real "after" measurement requires a
full **retrain** (hours/days).

**Why:** the eval/loss-dump tooling (`probe_replay.py`, `/tmp/run1000/losses/`) operates on
a fixed policy. It measures the policy, not the reward.

**How to apply:** verify shaping changes with (1) unit tests in `reward_manager_test.py`,
(2) the `--debug --steps 10000` smoke test (needs `npm run showdown`), and (3) an offline
**pathology-replay proxy** — feed the documented loss sequences (immune-attack spam,
capped-setup spam, switch oscillation) through `Gen3RewardManager` directly and confirm
cumulative shaping makes spam catastrophic and a pivot/alternative strictly dominant.
Reuse the `_ctx`/`_delta`/`_battle`/`_make_mon` helpers from `reward_manager_test.py`.

Baseline pathology rates (1000-game eval, May 2026): ~29% of losses had a ≥10-turn
same-move run (max 66); median total switch-probability on no-progress turns was ~1.3%.
The May 2026 reward rewrite (uncapped escalating repetition/bounce taxes, dead-matchup
tax, raised immune penalty -0.5, ramped stall tax from turn 60) targets these. Related:
[[project_training_versions]].
