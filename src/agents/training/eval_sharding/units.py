"""What the eval work IS — items, shards, and the (pure) partition policy.

An ``EvalItem`` is one opponent the parent wants measured over ``n_games`` games. A
``ShardUnit`` is a contiguous chunk of those games — the atom of work-stealing. Splitting
items into shards is what lets an idle worker drain a straggler's remaining games instead of
one worker grinding a whole opponent alone (the long-tail fix).

``plan_units`` is a **pure, deterministic** function of ``(items, shard_games)`` — the parent and
every worker call it on the *identical* manifest inputs (``eval_sharding.pool`` reads them from one
shared ``plan.json``), so they all agree on the unit set and its claim order with no central queue.

The order is LPT-ish: cost-descending items, shards round-robined across items — so every
opponent starts early (per-opponent results flow as soon as its first shard lands) and the
expensive opponents (sentinel/fixed model-vs-model, stallers) lead, shortening the tail.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Opponent kinds. A bot is a scripted roster player; a sentinel is a self-play pool snapshot; a
# fixed opponent is a frozen model from another run (the stable cross-run yardstick, ext_<label>).
BOT = "bot"
SENTINEL = "sentinel"
FIXED = "fixed"
_KINDS = (BOT, SENTINEL, FIXED)

# A cheap a-priori cost proxy for ordering only (never correctness): model-vs-model matchups
# infer for BOTH players (~2x), staller archetypes drag long battles, random is the cheapest.
# Used solely to lead the claim order with the expensive work so it isn't the last thing left.
_KIND_COST = {SENTINEL: 2.0, FIXED: 2.0, BOT: 1.0}
_NAME_COST = {"staller": 1.5, "staller_v2": 1.5, "random": 0.5}


@dataclass(frozen=True)
class EvalItem:
    """One opponent to evaluate over ``n_games`` games.

    ``kind`` selects how a worker builds the opponent; the kind-specific fields carry what it
    needs. Bots need only ``key`` (the roster name). Sentinels/fixed carry the snapshot ``path``
    (+ ``step`` for a sentinel, ``config_path`` for a fixed opponent's arch gate). ``key`` is the
    label everything downstream uses (TensorBoard tags, result grouping, the work-steal id).
    Plain dataclass so it round-trips through ``plan.json`` via ``asdict``/``from_dict``.
    """
    key: str
    kind: str
    n_games: int
    path: str | None = None
    step: int | None = None
    config_path: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown EvalItem kind {self.kind!r} (expected one of {_KINDS})")
        if self.n_games < 1:
            raise ValueError(f"EvalItem {self.key!r}: n_games must be >= 1, got {self.n_games}")
        if self.kind in (SENTINEL, FIXED) and not self.path:
            raise ValueError(f"EvalItem {self.key!r} (kind={self.kind}) requires a path")

    @property
    def cost(self) -> float:
        """Ordering-only cost proxy (see module docstring)."""
        return _KIND_COST[self.kind] * _NAME_COST.get(self.key, 1.0)

    def to_dict(self) -> dict:
        d = {"key": self.key, "kind": self.kind, "n_games": self.n_games}
        if self.path is not None:
            d["path"] = self.path
        if self.step is not None:
            d["step"] = self.step
        if self.config_path is not None:
            d["config_path"] = self.config_path
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EvalItem":
        return cls(
            key=d["key"], kind=d["kind"], n_games=int(d["n_games"]),
            path=d.get("path"), step=d.get("step"), config_path=d.get("config_path"),
        )


@dataclass(frozen=True)
class ShardUnit:
    """One claimable chunk of an item's games — the atom of work-stealing.

    ``item`` is carried inline (denormalized) so a worker that claims a unit has everything it
    needs to build the opponent and play, with no second lookup. ``unit_id`` is the claim token
    (``f"{item.key}#{shard_index}"``); it names both the O_EXCL lock and the shard result file.
    """
    item: EvalItem
    shard_index: int
    n_games: int

    @property
    def unit_id(self) -> str:
        return f"{self.item.key}#{self.shard_index}"

    @property
    def item_key(self) -> str:
        return self.item.key

    @property
    def kind(self) -> str:
        return self.item.kind


def _split_games(n_games: int, shard_games: int) -> list[int]:
    """Split ``n_games`` into as-even-as-possible chunks of at most ``shard_games`` each.

    ``shard_games >= n_games`` (or <= 0) yields a single chunk == ``n_games`` (the opponent-level
    behaviour: exactly today's one-claim-per-opponent path). Otherwise ``ceil(n_games/shard_games)``
    chunks whose sizes differ by at most 1 and sum to ``n_games`` exactly (so Σshards == n_games,
    the invariant that makes aggregation exact).
    """
    if shard_games <= 0 or shard_games >= n_games:
        return [n_games]
    k = math.ceil(n_games / shard_games)
    base, rem = divmod(n_games, k)
    return [base + 1 if i < rem else base for i in range(k)]


def plan_units(items: list[EvalItem], shard_games: int) -> list[ShardUnit]:
    """Partition every item into shards and return the claim-ordered unit list (pure).

    Order = cost-descending items, shards round-robined across items: shard 0 of every item (in
    cost order), then shard 1 of every item, and so on. Deterministic given identical inputs —
    the parent and all workers compute the same list from the shared ``plan.json``.
    """
    by_item: dict[str, list[ShardUnit]] = {}
    for it in items:
        sizes = _split_games(it.n_games, shard_games)
        by_item[it.key] = [ShardUnit(it, i, sz) for i, sz in enumerate(sizes)]
    ordered = sorted(items, key=lambda it: (-it.cost, it.key))
    max_shards = max((len(by_item[it.key]) for it in items), default=0)
    units: list[ShardUnit] = []
    for s in range(max_shards):
        for it in ordered:
            shards = by_item[it.key]
            if s < len(shards):
                units.append(shards[s])
    return units


def shards_per_item(items: list[EvalItem], shard_games: int) -> dict[str, int]:
    """How many shards each item splits into (for forensic-quota scaling + coverage)."""
    return {it.key: len(_split_games(it.n_games, shard_games)) for it in items}
