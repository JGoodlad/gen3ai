"""``ShardedEvalPool`` — the one deep interface the worker and parent talk to.

It hides every filesystem mechanic of battle-level work-stealing behind four calls:

    parent:  pool = ShardedEvalPool(items, shard_games);  pool.write_plan(run_dir)
             merged, missing = pool.collect(result_dir)        # after workers finish
    worker:  pool = ShardedEvalPool.from_plan(run_dir)
             while (unit := pool.claim_next(claim_dir)): ... ; pool.publish(result_dir, res)

The plan (items + shard_games) is written ONCE to ``plan.json`` by the parent and read by every
worker and by ``collect`` — so the partition is a single shared source of truth, not something the
two sides reconstruct independently. ``plan_units`` is pure, so everyone derives the identical
claim-ordered unit list from that one manifest. The worker never touches a lock file or a result
path; the parent never touches a shard file. That is the whole encapsulation: deep module, narrow
interface.
"""
from __future__ import annotations

import json
import os

from agents.training.eval_sharding.units import (
    EvalItem, ShardUnit, plan_units, shards_per_item,
)
from agents.training.eval_sharding.results import (
    ShardResult, aggregate, to_merged, write_shard_result,
)

PLAN_NAME = "plan.json"
_PLAN_VERSION = 1


class ShardedEvalPool:
    """Coordinates one eval cycle's battle-level work-stealing. Construct from the in-memory items
    (parent) or from the on-disk plan (worker/collect); both yield the identical unit list."""

    def __init__(self, items: list[EvalItem], shard_games: int, step: int | None = None):
        self.items = list(items)
        self.shard_games = int(shard_games)
        self.step = step
        self.units: list[ShardUnit] = plan_units(self.items, self.shard_games)
        self._shards_per_item = shards_per_item(self.items, self.shard_games)

    # ------------------------------------------------------------------ plan manifest

    def write_plan(self, run_dir: str) -> str:
        """Parent: serialize the plan to ``<run_dir>/plan.json`` (atomic) before spawning workers."""
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, PLAN_NAME)
        payload = {
            "version": _PLAN_VERSION,
            "step": self.step,
            "shard_games": self.shard_games,
            "items": [it.to_dict() for it in self.items],
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
        return path

    @classmethod
    def from_plan(cls, run_dir: str) -> "ShardedEvalPool":
        """Worker / collect: rebuild the pool from the parent's ``plan.json``."""
        with open(os.path.join(run_dir, PLAN_NAME)) as f:
            d = json.load(f)
        items = [EvalItem.from_dict(x) for x in d["items"]]
        return cls(items, int(d["shard_games"]), step=d.get("step"))

    # ------------------------------------------------------------------ sizing

    @property
    def n_units(self) -> int:
        return len(self.units)

    def shard_count(self, item_key: str) -> int:
        """How many shards ``item_key`` split into — for scaling a per-unit forensic quota."""
        return self._shards_per_item.get(item_key, 1)

    # ------------------------------------------------------------------ worker side

    def claim_next(self, claim_dir: str) -> ShardUnit | None:
        """Atomically claim the next unclaimed unit, or None when the pool is drained.

        Each unit is owned by exactly one worker via an ``O_EXCL`` lock — the first to create
        ``<claim_dir>/<unit_id>.lock`` wins; everyone else gets ``FileExistsError`` and moves on.
        Robust across independent processes with no shared memory (identical mechanism to the old
        opponent-level claim, now at unit granularity). Scans in the plan's LPT order.
        """
        for unit in self.units:
            lock = os.path.join(claim_dir, f"{unit.unit_id}.lock")
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return unit
            except FileExistsError:
                continue
        return None

    def publish(self, result_dir: str, res: ShardResult) -> None:
        """Persist one finished shard's raw result (atomic)."""
        write_shard_result(result_dir, res)

    # ------------------------------------------------------------------ parent side

    def collect(self, result_dir: str) -> tuple[dict, list[str]]:
        """Pool all present shards into the legacy ``merged`` dict + the fully-missing item keys.

        ``merged`` is byte-shape-compatible with the old ``merge_eval_results`` output (so every
        downstream consumer is untouched), plus the additive ``counts`` / ``coverage`` siblings.
        ``missing`` = item keys with zero shards present (a whole opponent lost to crashes), which
        the caller logs exactly as it always logged a missing opponent. Partial coverage (some
        shards present) is surfaced via ``merged["coverage"]`` for a softer warning.
        """
        per_opponent = aggregate(self.units, result_dir)
        merged = to_merged(per_opponent)
        present = set(per_opponent)
        missing = [it.key for it in self.items if it.key not in present]
        return merged, missing
