"""Battle-level (item-granularity) work-stealing for the eval pipeline.

Opponent-level work-stealing (one worker claims a whole opponent's N games) leaves a long tail:
when fewer opponents remain than workers, one worker grinds the last opponent's full N games
alone while the rest sit idle. This package splits each opponent into **shard units** (chunks of
games) so any idle worker drains a straggler's remaining games — the tail collapses to a single
shard instead of a whole opponent.

The public surface is intentionally tiny:

  - ``EvalItem``         — one opponent to measure (the parent declares these).
  - ``ShardedEvalPool``  — the deep coordinator: ``write_plan`` / ``from_plan`` /
                           ``claim_next`` / ``publish`` / ``collect``. Hides all the
                           lock/shard-file/partition mechanics.
  - ``ShardResult``      — the raw, additive per-shard record a worker emits.

Aggregation is **exact** for win_rate (Σwon/Σfinished), reward (Σreward/Σn) and ep_len
(Σturns/Σn): pooling the shards reproduces the unsharded value. The TD-residual tail pools raw δ
samples and computes the CVaR once (a CVaR can't be averaged); its aggregation is exact, but the
*captured sample set* shifts slightly with ``shard_games`` because the forensic capture quota
interacts with the shard count — so ``td_resid_tail`` is a shard-dependent sampled diagnostic, not
a per-game exact aggregate (win_rate etc. are exact regardless). Full notes:
``src/agents/training/CLAUDE.md`` → item-level work-stealing.

Extensibility (Glicko/TrueSkill): aggregation preserves exact per-opponent W/L counts (the
``counts`` map), and ``agents.training.rating`` defines the ``MatchRecord`` / ``RatingModel`` seam
those models will plug into. The seam is documented but the live ELO path is deliberately
unchanged — see ``rating.py``.
"""
from agents.training.eval_sharding.units import BOT, SENTINEL, FIXED, EvalItem, ShardUnit
from agents.training.eval_sharding.results import (
    ShardResult, OpponentResult, td_tail, TD_TAIL_FRAC, TD_TAIL_MIN_SAMPLES,
)
from agents.training.eval_sharding.pool import ShardedEvalPool, PLAN_NAME

__all__ = [
    "BOT", "SENTINEL", "FIXED",
    "EvalItem", "ShardUnit", "ShardResult", "OpponentResult",
    "ShardedEvalPool", "PLAN_NAME",
    "td_tail", "TD_TAIL_FRAC", "TD_TAIL_MIN_SAMPLES",
]
