"""The eval METRICS — raw per-shard records, their atomic IO, and exact aggregation.

A worker emits a ``ShardResult`` of **raw additive** quantities (counts + sums + the raw δ list),
never reduced ratios. The parent pools all of an item's shards back into one ``OpponentResult``.
The point is exactness: each metric pools with the SAME denominator the unsharded path uses, so the
pooled value equals the unsharded value bit-for-bit — win_rate = Σwon / Σ``n_finished`` (==
poke-env ``win_rate``), reward = Σreward / Σ``n_episodes`` (== ``mean_episode_reward``, whose denom
is the *reward-tracked* battle count), ep_len = Σturns / Σ``n_finished`` (the finished-battle mean).
Reward and ep_len intentionally carry DIFFERENT denominators (``n_episodes`` vs ``n_finished``) —
mirroring the two unsharded helpers — and that is still exact under pooling because each numerator is
summed against its own matching denominator. ``n_episodes`` == ``n_finished`` in the common case but
can be smaller (a battle that finishes without a finalized reward — e.g. a no-legal-move default), and
the pooling is exact either way (verified by ``test_sharded_aggregation_diverging_denominators``). The
TD-residual tail is the
subtle one — a CVaR cannot be averaged across shards, so each shard ships its **raw** δ samples
and the parent concatenates them and computes ``td_tail`` exactly once (``aggregate`` below).

(Caveat, by design: ``td_resid_tail`` is computed over the *captured* battles, and the forensic
quota that bounds that capture interacts with the shard count — so which battles land in the δ
pool shifts slightly with ``shard_games``. The *aggregation* is exact given a pool; the *sampled
pool* is shard-dependent. win_rate / reward / ep_len are exact regardless. See the package
``__init__`` and ``src/agents/training/CLAUDE.md``.)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

# CVaR tuning for the per-decision critic-surprise tail (eval/td_resid_tail_*). Lives here — the
# single source of truth — because aggregation owns the pooled computation; eval_callback re-exports
# these for the recorder + prober parity. Moved out of eval_callback to keep eval_sharding free of
# any import back into the callback (one-way dependency: eval_callback -> eval_sharding).
TD_TAIL_FRAC = 0.05
TD_TAIL_MIN_SAMPLES = 20

_SHARD_PREFIX = "shard__"
_SHARD_SUFFIX = ".json"


def td_tail(residuals, frac: float = TD_TAIL_FRAC, min_samples: int = TD_TAIL_MIN_SAMPLES):
    """Lower-tail mean (CVaR@frac) of ``residuals``; the single min if too few; None if empty."""
    ds = sorted(float(r) for r in residuals)
    if not ds:
        return None
    if len(ds) < min_samples:
        return ds[0]
    k = max(1, int(len(ds) * frac))
    return sum(ds[:k]) / k


@dataclass
class ShardResult:
    """Raw, additive metrics for ONE played shard. Everything here pools by summation; nothing is
    a pre-reduced ratio (that is what makes the parent aggregation exact)."""
    unit_id: str
    item_key: str
    worker_id: int
    n_won: int
    n_finished: int
    sum_reward: float
    n_episodes: int          # reward-tracked battles (reward denom; usually == n_finished, may be <)
    sum_ep_len: float        # Σ turns over finished battles
    duration_sec: float
    td_residuals: list[float] = field(default_factory=list)  # raw per-decision δ from captured battles


@dataclass
class OpponentResult:
    """One opponent's pooled metrics — the reduced shape the existing ``merged`` dict carries."""
    win_rate: float
    reward_mean: float
    ep_len: float
    td_resid_tail: float | None
    duration_sec: float
    n_won: int
    n_finished: int
    n_shards_done: int
    n_shards_total: int

    @property
    def coverage(self) -> float:
        """Fraction of this opponent's shards that reported (1.0 == full coverage)."""
        return self.n_shards_done / self.n_shards_total if self.n_shards_total else 0.0


def _shard_path(result_dir: str, unit_id: str) -> str:
    return os.path.join(result_dir, f"{_SHARD_PREFIX}{unit_id}{_SHARD_SUFFIX}")


def write_shard_result(result_dir: str, res: ShardResult) -> None:
    """Atomically persist one shard result (tmp + rename) so a reader never sees a half file."""
    out = _shard_path(result_dir, res.unit_id)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(asdict(res), f)
    os.replace(tmp, out)


def read_shard_result(result_dir: str, unit_id: str) -> ShardResult | None:
    """Read one shard result, or None if its file is absent (worker still running / crashed)."""
    p = _shard_path(result_dir, unit_id)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    return ShardResult(**d)


def aggregate(units, result_dir: str) -> dict[str, OpponentResult]:
    """Pool every present shard back into one ``OpponentResult`` per item key (exact).

    ``units`` is the full planned unit list (``plan_units`` output). Shards with no file on disk
    (worker crash / still running) are simply absent from the sums — the opponent's metrics are
    then over the games that *did* run, and ``coverage`` reflects the shortfall. Items whose every
    shard is missing get no entry (they are the "fully missing" set the caller treats as before).
    """
    # group planned units by item key, preserving total shard counts for coverage
    by_item: dict[str, list] = {}
    for u in units:
        by_item.setdefault(u.item_key, []).append(u)

    out: dict[str, OpponentResult] = {}
    for key, item_units in by_item.items():
        n_won = n_finished = n_episodes = 0
        sum_reward = sum_ep_len = duration = 0.0
        pooled_resid: list[float] = []
        done = 0
        for u in item_units:
            r = read_shard_result(result_dir, u.unit_id)
            if r is None:
                continue
            done += 1
            n_won += r.n_won
            n_finished += r.n_finished
            n_episodes += r.n_episodes
            sum_reward += r.sum_reward
            sum_ep_len += r.sum_ep_len
            duration += r.duration_sec
            pooled_resid.extend(r.td_residuals)
        if done == 0:
            continue  # fully missing opponent — caller reports it as missing, exactly as before
        out[key] = OpponentResult(
            win_rate=(n_won / n_finished) if n_finished else 0.0,
            reward_mean=(sum_reward / n_episodes) if n_episodes else 0.0,
            ep_len=(sum_ep_len / n_finished) if n_finished else 0.0,
            td_resid_tail=td_tail(pooled_resid) if pooled_resid else None,
            duration_sec=duration,
            n_won=n_won,
            n_finished=n_finished,
            n_shards_done=done,
            n_shards_total=len(item_units),
        )
    return out


def to_merged(per_opponent: dict[str, OpponentResult]) -> dict:
    """Project pooled results into the legacy ``merged`` dict shape every downstream consumer reads.

    Keeps the exact keys ``merge_eval_results`` has always returned (so record_per_opponent /
    build_bot_eval_block / record_elo / the pool & externals blocks are untouched), and adds a
    sibling ``counts`` map ({key: (n_won, n_finished)}) — additive, ignored by old readers, and the
    exact-record source a future Glicko/TrueSkill backfill needs (see eval_sharding rating seam).
    ``coverage`` ({key: fraction}) rides alongside for partial-coverage warnings.
    """
    merged = {"win_rates": {}, "reward_means": {}, "ep_lens": {},
              "td_resid_tails": {}, "durations_sec": {}, "counts": {}, "coverage": {}}
    for key, r in per_opponent.items():
        merged["win_rates"][key] = r.win_rate
        merged["reward_means"][key] = r.reward_mean
        merged["ep_lens"][key] = r.ep_len
        if r.td_resid_tail is not None:
            merged["td_resid_tails"][key] = r.td_resid_tail
        merged["durations_sec"][key] = r.duration_sec
        merged["counts"][key] = (r.n_won, r.n_finished)
        merged["coverage"][key] = r.coverage
    return merged
