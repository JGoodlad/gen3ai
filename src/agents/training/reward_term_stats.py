"""The `reward/` scalar group — WHAT THE REWARD IS MADE OF, live, per rollout.

`gen3_reward_term_export_v1`.

The reward is a registry of ~35 class-tagged terms (`reward_manager.RewardBreakdown`), and a run
STATES its composition once at startup (`reward_class_composition` -> the `metadata.json`
`reward_composition` block). What no run has ever recorded is the composition's MAGNITUDES: which
potential is actually carrying the signal, how big it is against the terminal, and whether a term
the census calls "active" ever emits anything at all. A term can be structurally present, listed in
the startup banner, and identically zero for a whole generation -- and until this group existed,
nothing said so.

**THE UNIT IS RAW REWARD**, the same units `--victory-value` is in, before PopArt and before the
discount. That is deliberate and it is the only frame in which the shares are meaningful: PopArt's
sigma moves over training, so a term expressed in normalized units would change while the reward
did not. Read `reward/*` beside `train/return_mean` / `popart/sigma` when you want the learned
frame.

**THE SHARE IS |.|-WEIGHTED, and it must be.** PBRS terms telescope to zero over an episode by
construction (`Phi(terminal)=0`), so a SIGNED share of a potential is ~0 for every healthy
potential and would report every one of them as inert. `<term>_abs_share` is
`sum|term| / sum_terms sum|term|` over the window: how much of the reward stream's total MOVEMENT
this term accounts for. The signed mean ships beside it as `<term>_mean`, which is where the
telescoping IS the reading (a potential whose signed mean drifts far from 0 over an
episode-complete window is not telescoping).

**THE RESIDUAL IS A GIGO GUARD, not a rounding term.** `reward/untracked_abs_mean` is
`mean |bd.total - sum(tracked terms)|`. The tracked set comes from `reward_class_composition`, i.e.
from the same `_pbrs_term_active` / `_bias_term_active` predicates the folds are gated on -- so a
non-zero residual means the census and the folds disagree about what this config emits, which is
exactly the class of defect the v9 drift (`--all-shaping-pbrs` silently ceasing to be passed) was.
It reads 0.0 on every correct config, and it is PUBLISHED rather than asserted because a reward
manager must never take down a run.

Everything here is PURE (floats in, floats out) -- no torch, no SB3, no I/O. The accumulator lives
in the reward manager (env-worker side) and is drained through `env_method`; see
`agents.training.reward_term_callback` for why the pull is an `env_method` and not an info dict.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

# Reward-class rollup names, matching `RewardClass`'s own values. Declared here rather than
# imported so this module stays free of the reward manager (which pulls in numpy, the data facade
# and the whole battle layer); `reward_term_stats_test` pins the two against each other.
CLASS_NAMES: tuple = ("terminal", "pbrs", "bias")

# The BIAS accumulate-and-refund MECHANISM. It is a `RewardBreakdown` float field and part of
# `total`, but it is deliberately not in the registry (it is not a term), so it gets its own
# rollup rather than being folded into `bias` -- at `--bias-additivity 1.0` it is identically 0.
REFUND_FIELD: str = "bias_refund"


class RewardTermAccumulator:
    """Per-decision running sums of the tracked reward terms, drained by an `env_method` pull.

    One instance per `Gen3RewardManager`, i.e. one per env worker. `observe(bd, total)` is called
    once per decision from `process_turn_reward`; `drain()` returns plain primitives and ZEROES the
    window, so two consecutive drains can never double-count.

    ``terms`` is the tracked set -- the ACTIVE terms of this run's composition plus
    ``bias_refund`` -- and it is fixed for the manager's lifetime, which is legal for exactly the
    reason the suppressed-term fast path is: every flag `_pbrs_term_active` / `_bias_term_active`
    reads is resume-immutable and value-checked by `check_reward_config`.
    """

    __slots__ = ("terms", "_sum", "_abs", "_n", "_total_sum", "_total_abs_sum", "_resid_abs_sum")

    def __init__(self, terms: Sequence[str]) -> None:
        self.terms: tuple = tuple(terms)
        self._sum: Dict[str, float] = {n: 0.0 for n in self.terms}
        self._abs: Dict[str, float] = {n: 0.0 for n in self.terms}
        self._n: int = 0
        self._total_sum: float = 0.0
        self._total_abs_sum: float = 0.0
        self._resid_abs_sum: float = 0.0

    def observe(self, breakdown, total: float) -> None:
        """Fold one decision's `RewardBreakdown`. ``total`` is the reward actually RETURNED, passed
        in rather than re-derived, so the residual compares against the number training saw."""
        tracked = 0.0
        for name in self.terms:
            v = float(getattr(breakdown, name, 0.0))
            if v:                                   # the overwhelmingly common case is 0.0
                self._sum[name] += v
                self._abs[name] += v if v > 0.0 else -v
                tracked += v
        self._n += 1
        t = float(total)
        self._total_sum += t
        self._total_abs_sum += t if t > 0.0 else -t
        r = t - tracked
        self._resid_abs_sum += r if r > 0.0 else -r

    def drain(self) -> dict:
        """Return this window's raw sums and ZERO them. Plain primitives only -- the payload
        crosses a `SubprocVecEnv` pipe."""
        out = {"n": self._n,
               "total_sum": self._total_sum,
               "total_abs_sum": self._total_abs_sum,
               "residual_abs_sum": self._resid_abs_sum,
               "sum": dict(self._sum),
               "abs": dict(self._abs)}
        self._sum = {n: 0.0 for n in self.terms}
        self._abs = {n: 0.0 for n in self.terms}
        self._n = 0
        self._total_sum = 0.0
        self._total_abs_sum = 0.0
        self._resid_abs_sum = 0.0
        return out


def merge_drained(payloads: Iterable[dict]) -> dict:
    """Sum any number of workers' `drain()` payloads into one. `None` entries (a worker with no
    accumulator) are skipped; an all-`None` input returns an empty window (``n == 0``)."""
    n = 0
    total_sum = 0.0
    total_abs_sum = 0.0
    resid = 0.0
    s: Dict[str, float] = {}
    a: Dict[str, float] = {}
    for p in payloads:
        if not p:
            continue
        n += int(p.get("n", 0))
        total_sum += float(p.get("total_sum", 0.0))
        total_abs_sum += float(p.get("total_abs_sum", 0.0))
        resid += float(p.get("residual_abs_sum", 0.0))
        for name, v in (p.get("sum") or {}).items():
            s[name] = s.get(name, 0.0) + float(v)
        for name, v in (p.get("abs") or {}).items():
            a[name] = a.get(name, 0.0) + float(v)
    return {"n": n, "total_sum": total_sum, "total_abs_sum": total_abs_sum,
            "residual_abs_sum": resid, "sum": s, "abs": a}


def reward_term_metrics(merged: Mapping, term_class: Mapping[str, str]) -> Dict[str, float]:
    """`reward/*` scalars (keys WITHOUT the `reward/` prefix) from a merged window.

    ``term_class`` maps each tracked term name to its class rollup -- one of `CLASS_NAMES`, or
    ``"refund"`` for `bias_refund`. A term absent from the map is rolled up as ``"other"`` rather
    than dropped, so a new registry entry is visible before anyone updates a table.

    Returns ``{}`` for an empty window (no decisions), so a rollout in which nothing was scored
    leaves a GAP in the curves rather than publishing a confident zero.
    """
    n = int(merged.get("n", 0))
    if n <= 0:
        return {}
    inv = 1.0 / n
    sums = merged.get("sum") or {}
    absv = merged.get("abs") or {}
    denom = 0.0
    for v in absv.values():
        denom += float(v)

    out: Dict[str, float] = {
        "n_decisions": float(n),
        "total_mean": float(merged.get("total_sum", 0.0)) * inv,
        "total_abs_mean": float(merged.get("total_abs_sum", 0.0)) * inv,
        "untracked_abs_mean": float(merged.get("residual_abs_sum", 0.0)) * inv,
    }
    class_abs: Dict[str, float] = {}
    class_sum: Dict[str, float] = {}
    for name in sorted(set(sums) | set(absv)):
        sv = float(sums.get(name, 0.0))
        av = float(absv.get(name, 0.0))
        out[f"{name}_mean"] = sv * inv
        out[f"{name}_abs_mean"] = av * inv
        # An all-zero window has no movement to apportion, and a share of 0/0 is not 0.0 -- it is
        # undefined. NaN renders as a gap rather than as "this term carries nothing".
        out[f"{name}_abs_share"] = (av / denom) if denom > 0.0 else float("nan")
        cls = term_class.get(name, "other")
        class_abs[cls] = class_abs.get(cls, 0.0) + av
        class_sum[cls] = class_sum.get(cls, 0.0) + sv
    for cls in sorted(class_abs):
        out[f"class_{cls}_mean"] = class_sum[cls] * inv
        out[f"class_{cls}_abs_mean"] = class_abs[cls] * inv
        out[f"class_{cls}_abs_share"] = (
            class_abs[cls] / denom) if denom > 0.0 else float("nan")
    return out


def tracked_terms(composition: Mapping) -> List[str]:
    """The tracked set from a `reward_class_composition(config)` census: every ACTIVE terminal,
    PBRS and BIAS term, plus the refund mechanism. Ordered terminal -> pbrs -> bias -> refund so
    the TensorBoard tag order reads the way the fold does."""
    terms: List[str] = list(composition.get("terminal_terms") or ())
    terms += list(composition.get("pbrs_terms") or ())
    terms += list(composition.get("bias_terms") or ())
    terms.append(REFUND_FIELD)
    seen: set = set()
    return [t for t in terms if not (t in seen or seen.add(t))]


def term_class_map(composition: Mapping) -> Dict[str, str]:
    """`{term name -> rollup name}` for `reward_term_metrics`, derived from the same census."""
    out: Dict[str, str] = {}
    for name in composition.get("terminal_terms") or ():
        out[name] = "terminal"
    for name in composition.get("pbrs_terms") or ():
        out[name] = "pbrs"
    for name in composition.get("bias_terms") or ():
        out[name] = "bias"
    out[REFUND_FIELD] = "refund"
    return out
