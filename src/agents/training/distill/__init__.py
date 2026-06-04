"""Self-play opponent distillation — a cheaper drop-in opponent network.

Design: ``designs/ai_v5/{design_opponent_distillation,distill_integration}.md``.
Recipe (validated): a small per-slot ``CheapEncoder`` distilled (MSE) onto the teacher's frozen
12x128 role tokens, frozen, then a matchup-aware head distilled by soft-KL — ~4.7x cheaper, h2h
~0.44 vs the teacher. The opponent loads as a drop-in via ``DistilledOpponentModel`` so ``RLPlayer``
is unchanged.

KEY CONSTRAINT (the per-step barrier): distillation is **all-or-nothing** — a single full-opponent
worker straggles and gates the batch, so the pool is only ever 100% distilled or 100% full. See
``distill_integration.md`` §8.
"""
from agents.training.distill.student import (
    DistilledStudent,
    DistilledOpponentModel,
    save_distilled,
    load_distilled,
)

__all__ = ["DistilledStudent", "DistilledOpponentModel", "save_distilled", "load_distilled"]
