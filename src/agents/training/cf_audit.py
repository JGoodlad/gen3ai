"""cf_audit — the COUNTERFACTUAL AUDIT instrument (G2 of the counterfactual-value-grounding design).

Given a run's bridge-eval traces and a loadable checkpoint, this manufactures **ground-truth
value labels the on-policy stream structurally cannot produce**: for a sampled decision it
plays the RECORDED action and then rolls the rest of the battle out live ``R`` times with
fresh post-divergence dice, against the RELOADED real opponent. The win rate over those
rollouts is a tight Monte-Carlo estimate of that state's value — many samples of one state,
where training sees exactly one.

It emits two things, and they are useful independently:

1. **A BIAS MAP** — predicted win-prob vs tight-MC, per stratum, with battle-clustered CIs
   and the program's primary meter, ``sd_true_excess`` (§ *The meter*).
2. **LABEL ROWS** in the shared v1 schema (§ *The label schema*), which a training-side
   consumer reads without knowing anything about this module.

    python -m agents.training.cf_audit <run_dir> [--rollouts 8] [--states 400] \\
        [--step N] [--checkpoint PATH] [--impl rust] [--out DIR] [--seed S]

The meter — why ``sd_true_excess`` and not the mean gap
------------------------------------------------------
G0 (2026-08-22, 2,204 labels / 216 battles) measured the population-mean predicted−MC gap at
|0.05|–|0.07| **with a sign that flips with the population you weight to**, while the TRUE
within-decile spread of P(win) is 0.11–0.36 — a per-state error 2–6× the aggregate offset.
The head's defect is therefore RESOLUTION (it calls states the same that are not the same),
not an optimism offset. A lever that merely re-centres the head would move the mean gap and
leave the spread untouched, and would be scored a success by the wrong meter. So the headline
number here is the spread the head fails to resolve:

    Var(MC | decile)  =  Var(true P(win) | decile)  +  E[binomial sampling var]
    E[p̂(1−p̂)]/(R−1)   is EXACTLY unbiased for  p(1−p)/R  = the sampling var of an R-rollout mean
    sd_true_excess    =  sqrt(max(0, Var(MC) − E[p̂(1−p̂)]/(R−1)))

Subtracting the floor is what makes it a claim about the WORLD rather than about R. At zero
true effect (every state in the cell sharing one p) the estimator returns ~0 by construction,
and ``cf_audit_test.py`` pins that on synthetic data.

Selection-awareness
-------------------
Eval traces over-capture losses (an explicit win/loss quota), so a pooled gap convicts the
critic of the sampler's sins. Every aggregate here is computed **within an outcome stratum**
and only then recombined at the frame's own population shares; every CI is a bootstrap over
**battles**, never over states, because decisions inside one battle are not independent.

The label schema (v1) — a CONTRACT, do not change in place
----------------------------------------------------------
One JSON object per line in ``<out>/cf_labels/labels_<producer>_<seq>.jsonl``::

    {"schema": 1, "kind": "mc_winprob", "battle": "<record path>", "decision_idx": <int>,
     "obs_sha1": "<sha1 of the obs float32 bytes>", "obs_npz": "<states.npz path>::obs" | null,
     "obs_inline": "<base64 raw float32>" | null, "label": <float 0..1>, "n_rollouts": <int>,
     "wilson_lo": <float>, "wilson_hi": <float>, "policy_step": <int>, "opponent": "<str>",
     "created_unix": <float>}

``obs_npz`` points at the array; ``decision_idx`` selects the ROW of it. ``obs_sha1`` is
always present, so a consumer can verify the row it loaded is the row that was labelled.

Label trust before map trust
----------------------------
The ANCHOR arm runs FIRST and gates everything: recorded action + recorded dice must
reproduce the recorded battle outcome. If it fails on more than ``--anchor-tolerance`` of its
states the audit REFUSES to emit labels at all — a factory whose replay is not exact is GIGO,
and a bias map computed from it would be a measurement of the bug.

Known coverage gaps, printed and never silent: TURN-1 decisions (the offline replay driver
cannot open them — 3.35% of move decisions, exactly one per battle) and forced-switch rounds
(the re-roll layer anchors at start-of-turn move rounds).
"""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

SCHEMA_VERSION = 1
LABEL_KIND = "mc_winprob"

#: The 9 fixed eval bots. Everything else in an eval trace tree is a pool sentinel, which
#: matters because a bot is REBUILT EXACTLY (so its label is exact) while a sentinel is
#: reloaded from a pinned snapshot.
BOTS = ("random", "heuristic", "heuristic2", "aggressive", "aggressive_v2",
        "staller", "staller_v2", "setup_sweep", "setup_sweep_v2")

#: Declared, versioned sampler weights (design decision of record 3: a silent priority change
#: is a distribution-shift confound for every downstream readout). Bump the version when they
#: change; it is written into every bias map.
SAMPLER_VERSION = "cf_audit_strata_v1"

#: How much to over-sample the HIGH-CONFIDENCE-FROM-LOST-BATTLES region — the "0.827 class",
#: the population R1 is meant to supervise. Oversampling is corrected for at aggregation time
#: (cells are recombined at their FRAME shares), so this buys resolution, not bias.
CONVICTION_BOOST = 4.0
CONVICTION_DECILE = 7          # predicted win-prob >= 0.7
MAX_PER_BATTLE = 12            # a few long games must not carry a stratum


# ---------------------------------------------------------------------------
# The sampling frame — model-free, reads summary.json + states.npz only
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """One reconstructable move decision, with what the model recorded about it."""
    battle: str                 # the trace prefix (…/<opp>/<name>), the record's join key
    short: str                  # "<opponent>/<battle name>" — for humans
    opponent: str
    opp_class: str              # "bot" | "sentinel"
    outcome: str                # "win" | "loss"
    inv: int
    turn: int
    win_prob: float
    value: Optional[float]
    action: int
    move_rank: int              # position among this battle's move decisions
    n_moves: int

    @property
    def near_terminal(self) -> bool:
        return self.n_moves - 1 - self.move_rank <= 1


def build_frame(traces_dir: str) -> "tuple[List[Decision], Counter]":
    """Every reconstructable ``move_selection`` decision under ``traces_dir``, plus a census of
    what was skipped and why (a skip that is not counted is a skip that is not known)."""
    rows: List[Decision] = []
    skipped: Counter = Counter()
    for sp in sorted(glob.glob(os.path.join(traces_dir, "*", "*_summary.json"))):
        base = sp[: -len("_summary.json")]
        if not os.path.exists(base + "_reconstruction.json"):
            skipped["no_reconstruction_sibling"] += 1
            continue
        try:
            with open(sp) as f:
                summ = json.load(f)
            with np.load(base + "_states.npz") as z:
                npz = {k: z[k] for k in z.files}
        except Exception as exc:                                        # noqa: BLE001
            skipped[f"load:{type(exc).__name__}"] += 1
            continue
        outcome = ((summ.get("meta") or {}).get("result") or "").lower()
        if outcome not in ("win", "loss"):
            skipped[f"outcome:{outcome or 'none'}"] += 1
            continue
        if "win_probs" not in npz or "actions" not in npz:
            skipped["no_win_prob_head"] += 1
            continue
        opp = os.path.basename(os.path.dirname(sp))
        invs = summ.get("invocations", [])
        wps = np.asarray(npz["win_probs"], dtype=float)
        vals = np.asarray(npz.get("values", []), dtype=float)
        has = np.asarray(npz["has_state"], dtype=int)
        acts = np.asarray(npz["actions"], dtype=int)
        moves = [i for i, iv in enumerate(invs) if iv.get("phase") == "move_selection"]
        for rank, i in enumerate(moves):
            iv = invs[i]
            if i >= len(wps) or not has[i]:
                skipped["no_recorded_state"] += 1
                continue
            wp = float(wps[i])
            if not math.isfinite(wp):
                skipped["nan_win_prob"] += 1
                continue
            turn = int(iv.get("turn", -1))
            if turn <= 1:
                # Counted, never silent. The KEY is now a misnomer, kept so the reported
                # accounting stays comparable across runs: turn 1 became openable on BOTH impls
                # on 2026-08-23 (`gen3_search_turn1_open_v1`), so this is a sampler bound
                # matching `cf_producer.MIN_LABELABLE_TURN` — not the driver limitation it was
                # named for. Lowering the two together changes the audited population, so it is
                # its own change rather than a rider on the driver fix.
                skipped["turn_1_unopenable"] += 1
                continue
            rows.append(Decision(
                battle=base, short=f"{opp}/{os.path.basename(base)}", opponent=opp,
                opp_class="bot" if opp in BOTS else "sentinel", outcome=outcome, inv=i,
                turn=turn, win_prob=wp, value=float(vals[i]) if i < len(vals) else None,
                action=int(acts[i]) if i < len(acts) else -1,
                move_rank=rank, n_moves=len(moves)))
        skipped["forced_switch_rounds"] += sum(
            1 for iv in invs if iv.get("phase") != "move_selection")
    return rows, skipped


def sentinel_snapshots(run_dir: str) -> Dict[str, str]:
    """``sentinel_<i>`` → the snapshot path the eval manifest says it was. Pinning the real
    weights is strictly better than the ``self_model_approx`` fallback; the REGIME is handled
    separately (``prober.replay.build_opponent`` now plays a ckpt opponent stochastic, the
    regime ``eval_worker`` recorded)."""
    try:
        with open(os.path.join(run_dir, "metadata.json")) as f:
            md = json.load(f)
    except (OSError, ValueError):
        return {}
    sents = ((md.get("latest_eval") or {}).get("pool") or {}).get("sentinels") or []
    out = {}
    for i, s in enumerate(sents):
        snap = s.get("snapshot")
        if not snap:
            continue
        p = os.path.join(run_dir, "snapshots", snap)
        if os.path.exists(p):
            out[f"sentinel_{i}"] = p
    return out


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def turn_tercile_edges(frame: Sequence[Decision]) -> "tuple[float, float]":
    """The 1/3 and 2/3 quantiles of the frame's turn distribution — read off the DATA rather
    than hard-coded, because a run's game length is a property of its ecology."""
    if not frame:
        return 10.0, 23.0
    ts = np.asarray([d.turn for d in frame], dtype=float)
    return float(np.quantile(ts, 1 / 3)), float(np.quantile(ts, 2 / 3))


def stratum_of(d: Decision, edges: "tuple[float, float]") -> "tuple[int, str, int]":
    """``(confidence decile, battle outcome, turn tercile)`` — the audit's sampling cell."""
    decile = min(9, max(0, int(d.win_prob * 10)))
    lo, hi = edges
    tercile = 0 if d.turn <= lo else (1 if d.turn <= hi else 2)
    return decile, d.outcome, tercile


def stratified_sample(frame: Sequence[Decision], n_states: int, *, seed: int,
                      conviction_boost: float = CONVICTION_BOOST,
                      max_per_battle: int = MAX_PER_BATTLE,
                      ) -> "tuple[List[Decision], dict]":
    """Draw ``n_states`` decisions over (decile × outcome × turn-tercile), over-sampling the
    high-confidence-from-lost-battles region by ``conviction_boost``.

    Returns the sample and a DESIGN dict — the strata, the target and realized counts, the
    boost, the seed and the sampler version. That dict is written into the bias map, because
    an aggregate whose sampler is not on the record cannot be compared to another run's.
    """
    rng = random.Random(seed)
    edges = turn_tercile_edges(frame)
    by_cell: "Dict[tuple, List[Decision]]" = defaultdict(list)
    for d in frame:
        by_cell[stratum_of(d, edges)].append(d)

    # Weight each populated cell by its frame mass, boosted on the conviction region, then
    # allocate proportionally. A cell can never be asked for more than it has.
    weights = {}
    for cell, ds in by_cell.items():
        decile, outcome, _ = cell
        w = float(len(ds))
        if outcome == "loss" and decile >= CONVICTION_DECILE:
            w *= conviction_boost
        weights[cell] = w
    total_w = sum(weights.values()) or 1.0

    # LARGEST-REMAINDER allocation, not per-cell rounding. With many cells and a small
    # budget, `round(n * w/W)` is 0 everywhere and the sample comes out EMPTY — a silent
    # failure that looks like "no data" rather than "bad arithmetic".
    cells = sorted(by_cell, key=lambda c: (-weights[c], c))
    quotas = {c: n_states * weights[c] / total_w for c in cells}
    targets = {c: min(len(by_cell[c]), int(quotas[c])) for c in cells}
    left = n_states - sum(targets.values())
    for c in sorted(cells, key=lambda c: -(quotas[c] - int(quotas[c]))):
        if left <= 0:
            break
        if targets[c] < len(by_cell[c]):
            targets[c] += 1
            left -= 1

    per_battle: Counter = Counter()
    sample: List[Decision] = []
    pools = {}
    for cell in cells:
        pool = list(by_cell[cell])
        rng.shuffle(pool)
        pools[cell] = pool
        taken = 0
        for d in pool:
            if taken >= targets[cell]:
                break
            if per_battle[d.battle] >= max_per_battle:
                continue
            per_battle[d.battle] += 1
            sample.append(d)
            taken += 1
    # Top up any shortfall (a cell that hit the per-battle cap) from whatever is still
    # eligible, so `--states N` means N wherever N decisions actually exist.
    if len(sample) < n_states:
        chosen = {(d.battle, d.inv) for d in sample}
        spare = [d for cell in cells for d in pools[cell] if (d.battle, d.inv) not in chosen]
        rng.shuffle(spare)
        for d in spare:
            if len(sample) >= n_states:
                break
            if per_battle[d.battle] >= max_per_battle:
                continue
            per_battle[d.battle] += 1
            sample.append(d)
    rng.shuffle(sample)

    design = {
        "sampler_version": SAMPLER_VERSION,
        "seed": seed,
        "n_requested": n_states,
        "n_drawn": len(sample),
        "conviction_boost": conviction_boost,
        "conviction_decile_floor": CONVICTION_DECILE,
        "max_per_battle": max_per_battle,
        "turn_tercile_edges": list(edges),
        "cells": [
            {"decile": c[0], "outcome": c[1], "turn_tercile": c[2],
             "frame_n": len(by_cell[c]), "target": targets.get(c, 0),
             "drawn": sum(1 for d in sample if stratum_of(d, edges) == c)}
            for c in sorted(by_cell)
        ],
    }
    return sample, design


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_ci(wins: float, n: int, z: float = 1.96) -> "tuple[float, float]":
    """Wilson score interval — the right small-N binomial CI (a normal approximation gives
    the degenerate [0, 0] at 0 wins). ``(0.0, 1.0)`` for n == 0.

    ``wins`` may be FRACTIONAL: `cf_producer` scores a draw-at-the-turn-cap 0.5, so its success
    total is a sum over ``{0, 0.5, 1}``. The arithmetic is unchanged and well-defined, but such a
    sample is not Bernoulli, so the interval becomes an approximation that errs NARROW — which is
    why the label row carries `n_capped` beside it.
    """
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared) — the transform that turns Pearson into Spearman."""
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    # average tied ranks, so a head that emits one constant width scores 0 rather than an artifact
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def _is_flat(a: np.ndarray, rtol: float = 1e-9) -> bool:
    return bool(np.ptp(a) <= rtol * (float(np.abs(a).max()) + 1e-30))


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation, or ``None`` when it is undefined (n < 3, or either side flat).

    Hand-rolled rather than `scipy.stats.spearmanr` so the audit's headline does not acquire a
    dependency the training package does not otherwise have. A CONSTANT input returns None, not 0:
    "the head claims the same width everywhere" is a different finding from "the widths it claims
    are unrelated to the blur", and collapsing them would hide the more damning one.
    """
    a, b = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    if len(a) < 3 or len(a) != len(b):
        return None
    # FLAT to a relative tolerance, not `std() == 0`. A constant that arrives through a weighted
    # average is constant to ~1 ulp, not exactly — and an exact test there lets a genuinely flat
    # width fall through to `corrcoef`, which divides by ~1e-17 and reports a confident correlation
    # of pure float noise.
    if _is_flat(a) or _is_flat(b):
        return None
    ra, rb = _ranks(a), _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence[str], *,
                         draws: int = 2000, seed: int = 7,
                         ) -> "tuple[Optional[float], Optional[float]]":
    """95% CI by resampling CLUSTERS (battles) with replacement, not states.

    Decisions inside one battle share a board, a team matchup and a dice stream, so a
    state-level CI understates the width by however much that correlation is worth. This is
    the same discipline the pooled-correlation Simpson trap taught: resample the unit of
    independence, which here is the battle."""
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    by_c: "Dict[str, List[float]]" = defaultdict(list)
    for v, c in zip(values, clusters):
        by_c[c].append(float(v))
    keys = list(by_c)
    if len(keys) < 2:
        return None, None
    pools = [np.asarray(by_c[k]) for k in keys]
    means = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, len(pools), size=len(pools))
        means[i] = float(np.concatenate([pools[j] for j in idx]).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sd_true_excess(mc: Sequence[float], n_rollouts: int,
                   weights: "Optional[Sequence[float]]" = None) -> dict:
    """THE PRIMARY METER — the within-cell spread of the TRUE win probability, net of the
    R-rollout binomial noise floor.

    ``Var(MC) = Var(true p) + E[sampling var]``, and for an R-rollout mean the sampling
    variance is ``p(1−p)/R``. The plug-in ``E[p̂(1−p̂)]/(R−1)`` is EXACTLY unbiased for it
    (``E[p̂(1−p̂)] = (1−1/R)·p(1−p)``), so the subtraction leaves the real heterogeneity and
    nothing else. Clamped at 0 — a negative estimate means the cell's spread is at or below
    the floor, i.e. no resolvable structure, which is what 0 says.

    ``weights`` recombines sub-cells (e.g. win/loss) at their POPULATION shares rather than
    at this probe's deliberately balanced sampling shares:
    ``Var = Σ w_o (Var_o + (m_o − m)²)``. Pass one weight per element.
    """
    a = np.asarray(list(mc), dtype=float)
    n = len(a)
    if n < 3 or n_rollouts < 2:
        return {"n": int(n), "mean": float(a.mean()) if n else None, "sd_observed": None,
                "sd_binomial_floor": None, "sd_true_excess": None, "frac_variance_real": None}
    w = (np.ones(n) if weights is None else np.asarray(list(weights), dtype=float))
    w = w / w.sum()
    mean = float((w * a).sum())
    # Unbiased weighted variance (reliability weights): divide by (1 - Σw²).
    denom = 1.0 - float((w ** 2).sum())
    var = float((w * (a - mean) ** 2).sum() / denom) if denom > 0 else float(a.var(ddof=1))
    binom = float((w * (a * (1 - a) / (n_rollouts - 1))).sum())
    excess = max(0.0, var - binom)
    return {
        "n": int(n),
        "mean": mean,
        "sd_observed": math.sqrt(max(0.0, var)),
        "sd_binomial_floor": math.sqrt(max(0.0, binom)),
        "sd_true_excess": math.sqrt(excess),
        "frac_variance_real": (excess / var) if var > 0 else None,
    }


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

@dataclass
class LabelRow:
    """One row of the shared v1 schema. Field names and types are a CONTRACT with the
    training-side consumer — see the module header."""
    battle: str
    decision_idx: int
    obs_sha1: str
    label: float
    n_rollouts: int
    wilson_lo: float
    wilson_hi: float
    policy_step: int
    opponent: str
    obs_npz: Optional[str] = None
    obs_inline: Optional[str] = None
    created_unix: float = field(default_factory=time.time)
    schema: int = SCHEMA_VERSION
    kind: str = LABEL_KIND

    def to_json(self) -> dict:
        d = asdict(self)
        return {"schema": d.pop("schema"), "kind": d.pop("kind"), **d}


def obs_digest(obs: np.ndarray) -> str:
    """sha1 over the float32 bytes — the identity a consumer checks the row it loaded against."""
    return hashlib.sha1(np.ascontiguousarray(obs, dtype=np.float32).tobytes()).hexdigest()


def obs_b64(obs: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(obs, dtype=np.float32).tobytes()).decode()


def write_labels(rows: Sequence[LabelRow], out_dir: str, *, producer: str, seq: int) -> str:
    d = os.path.join(out_dir, "cf_labels")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"labels_{producer}_{seq}.jsonl")
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_json()) + "\n")
    return path


def _label_one(session, d: Decision, *, n_rollouts: int, opp_ckpt: Optional[str]) -> dict:
    """One tight-MC label: play the RECORDED action, roll to the end ``n_rollouts`` times."""
    out = session.replay_counterfactual(
        d.battle + "_summary.json", d.inv, d.action,
        n_rollouts=n_rollouts, opponent_ckpt=opp_ckpt)
    return out


# ---------------------------------------------------------------------------
# The bias map
# ---------------------------------------------------------------------------

def bias_map(labels: Sequence[dict], frame: Sequence[Decision], *, n_rollouts: int,
             design: dict, accounting: dict) -> dict:
    """Predicted vs tight-MC by stratum, selection-aware, battle-clustered.

    ``labels`` are the clean rows (dicts carrying ``win_prob``, ``mc``, ``outcome``,
    ``battle``, ``turn``, ``opponent``, ``opp_class``). Aggregates are computed WITHIN an
    outcome stratum and recombined at the frame's own (decile, outcome) shares — the eval
    capture quota over-samples losses, so a pooled number would convict the critic of the
    sampler's sins."""
    edges = tuple(design.get("turn_tercile_edges") or turn_tercile_edges(frame))
    frame_mass: Counter = Counter()
    for d in frame:
        frame_mass[(min(9, int(d.win_prob * 10)), d.outcome)] += 1

    def _cell(rows, name):
        if not rows:
            return None
        gap = [r["win_prob"] - r["mc"] for r in rows]
        lo, hi = cluster_bootstrap_ci(gap, [r["battle"] for r in rows])
        return {
            "stratum": name, "n": len(rows), "n_battles": len({r["battle"] for r in rows}),
            "mean_predicted": float(np.mean([r["win_prob"] for r in rows])),
            "mean_mc": float(np.mean([r["mc"] for r in rows])),
            "mean_gap": float(np.mean(gap)),
            "gap_ci": [lo, hi],
        }

    out: dict = {
        "n_rollouts": n_rollouts,
        "sampler": design,
        "accounting": accounting,
        "by_outcome": [], "by_decile_outcome": [], "by_turn_tercile": [],
        "by_opponent": [], "conviction_class": None,
        "resolution": [], "headline": {}, "evidential": None,
    }
    for oc in ("win", "loss"):
        c = _cell([r for r in labels if r["outcome"] == oc], oc)
        if c:
            out["by_outcome"].append(c)
    for dec in range(10):
        for oc in ("win", "loss"):
            c = _cell([r for r in labels
                       if min(9, int(r["win_prob"] * 10)) == dec and r["outcome"] == oc],
                      f"decile{dec}/{oc}")
            if c:
                c["decile"], c["outcome"] = dec, oc
                out["by_decile_outcome"].append(c)
    for t, name in ((0, f"turn<={edges[0]:.0f}"), (1, f"turn {edges[0]:.0f}-{edges[1]:.0f}"),
                    (2, f"turn>{edges[1]:.0f}")):
        for oc in ("win", "loss"):
            rows = [r for r in labels if r["outcome"] == oc
                    and (0 if r["turn"] <= edges[0] else (1 if r["turn"] <= edges[1] else 2)) == t]
            c = _cell(rows, f"{name}/{oc}")
            if c:
                out["by_turn_tercile"].append(c)
    for opp in sorted({r["opponent"] for r in labels}):
        c = _cell([r for r in labels if r["opponent"] == opp], opp)
        if c:
            out["by_opponent"].append(c)

    # THE CONVICTION CLASS — high predicted confidence in a battle that was LOST, and its
    # confidence-matched WON control. The difference (one clustered CI) is the readout; the
    # split below it is the thing a single realized outcome structurally cannot tell you.
    conv = [r for r in labels if r["win_prob"] >= 0.75 and r["outcome"] == "loss"]
    ctrl = [r for r in labels if r["win_prob"] >= 0.75 and r["outcome"] == "win"]
    if conv:
        c = _cell(conv, "wp>=0.75 & LOST")
        c["control"] = _cell(ctrl, "wp>=0.75 & WON")
        c["share_mc_ge_0.75"] = float(np.mean([r["mc"] >= 0.75 for r in conv]))
        c["share_mc_lt_0.50"] = float(np.mean([r["mc"] < 0.50 for r in conv]))
        c["share_mc_lt_0.25"] = float(np.mean([r["mc"] < 0.25 for r in conv]))
        c["median_predicted"] = float(np.median([r["win_prob"] for r in conv]))
        c["median_mc"] = float(np.median([r["mc"] for r in conv]))
        if ctrl:
            diff = [r["win_prob"] - r["mc"] for r in conv]
            dctl = [r["win_prob"] - r["mc"] for r in ctrl]
            lo, hi = cluster_bootstrap_ci(
                diff + [-x for x in dctl],
                [r["battle"] for r in conv] + [r["battle"] for r in ctrl])
            c["loss_minus_win_gap"] = float(np.mean(diff) - np.mean(dctl))
            c["loss_minus_win_ci"] = [lo, hi]
        out["conviction_class"] = c

    # RESOLUTION — the primary meter, per decile, population-recombined over the outcome
    # sub-cells so this probe's deliberate loss over-sampling cannot inflate the spread.
    out["resolution"] = resolution_cells(labels, frame_mass, n_rollouts)

    # THE EVIDENTIAL READ (the pre-registered A/B meter for `--cf-evidential`). Present only when
    # the audited checkpoint actually carries the head; absent — never zero — otherwise, because a
    # zero width is a claim the head made and a missing column is a fact about the checkpoint.
    if any(r.get("evid_width") is not None for r in labels):
        out["evidential"] = evidential_read(labels, frame_mass, n_rollouts)

    # THE TWIN-HEAD PAIRED READ (gen3_cf_twin_heads_v1) — the amended R1 primary. Present only when
    # the audited checkpoint actually carries heads B and C; absent, never zero, otherwise. Both
    # blocks are emitted together on purpose: `twin_paired` is the meter, `twin_resolution` is the
    # G0 continuity link, and reading the second without the first invites exactly the
    # absolute-level comparison its own `weighting` field warns against.
    if any(r.get("twin_b_pred") is not None for r in labels):
        out["twin_paired"] = paired_head_read(labels)
        out["twin_resolution"] = twin_resolution_read(labels, n_rollouts)
    _shadow = shadow_read(labels)
    if _shadow is not None:
        out["shadow"] = _shadow

    if out["resolution"]:
        mass = [frame_mass.get((r["decile"], "win"), 0) + frame_mass.get((r["decile"], "loss"), 0)
                for r in out["resolution"]]
        tot = sum(mass) or 1
        out["headline"] = {
            "population_weighted_gap": float(sum(
                m * (r["mean_predicted"] - r["mean"]) for m, r in zip(mass, out["resolution"])) / tot),
            "population_weighted_sd_true_excess": float(sum(
                m * (r["sd_true_excess"] or 0.0) for m, r in zip(mass, out["resolution"])) / tot),
            "n_labels": len(labels),
            "n_battles": len({r["battle"] for r in labels}),
        }
    return out


#: A decile cell needs this many labels (and this many per outcome sub-cell) before its spread is
#: reported. Below it `sd_true_excess` is mostly the estimator's own noise.
MIN_CELL_N, MIN_SUBCELL_N = 12, 3


def resolution_cells(labels: Sequence[dict], frame_mass: Counter, n_rollouts: int) -> List[dict]:
    """Per-decile `sd_true_excess` (+ the evidential columns when the rows carry them).

    Factored out of `bias_map` because the bootstrap re-runs it per draw — a CI whose point
    estimate came from different arithmetic than its resamples is not a CI of anything.
    """
    cells: List[dict] = []
    for dec in range(10):
        rows, w = [], []
        for oc in ("win", "loss"):
            sub = [r for r in labels
                   if min(9, int(r["win_prob"] * 10)) == dec and r["outcome"] == oc]
            mass = frame_mass.get((dec, oc), 0)
            if len(sub) < MIN_SUBCELL_N or mass <= 0:
                continue
            rows += sub
            w += [mass / len(sub)] * len(sub)
        if len(rows) < MIN_CELL_N:
            continue
        st = sd_true_excess([r["mc"] for r in rows], n_rollouts, weights=w)
        st["decile"] = dec
        wa = np.asarray(w)
        st["mean_predicted"] = float(np.average([r["win_prob"] for r in rows], weights=wa))
        widths = [r.get("evid_width") for r in rows]
        if all(x is not None for x in widths):
            # Population-weighted like everything else in this cell, so the confessed width and the
            # measured spread it is supposed to track are read off the SAME population.
            st["evid_width_mean"] = float(np.average(np.asarray(widths, dtype=float), weights=wa))
            st["evid_precision_mean"] = float(np.average(
                np.asarray([r["evid_precision"] for r in rows], dtype=float), weights=wa))
        cells.append(st)
    return cells


def evidential_read(labels: Sequence[dict], frame_mass: Counter, n_rollouts: int, *,
                    draws: int = 1000, seed: int = 11) -> dict:
    """THE PRE-REGISTERED METER for `--cf-evidential`: does the confessed width track the blur?

    G0 convicted the win-prob head of RESOLUTION — within a confidence decile the true P(win)
    spread is 0.11–0.36, which a point estimate cannot represent. The evidential head reads the
    same `value_pooled`, so it cannot REMOVE that blur; the only success it can have is
    **confessing** it, i.e. emitting a wide Beta exactly in the deciles where `sd_true_excess` is
    large. So the headline is the rank correlation ACROSS STRATA between the head's mean epistemic
    width and the measured spread — `width_vs_blur_spearman`.

    Rank, not Pearson: the claim is ordering ("wider where blurrier"), and the two quantities are
    not on a common scale (a Beta's std and a within-cell sd of an R-rollout mean). Rank
    correlation is also robust to the one decile with a long tail dragging a Pearson r.

    The CI is a bootstrap over BATTLES, not over strata: decisions inside one battle share a board,
    a matchup and a dice stream, and the per-decile cells are built FROM those decisions, so
    resampling strata would understate the width by however much that correlation is worth. Each
    draw rebuilds the cells from scratch through `resolution_cells`, which means a draw can lose a
    thin decile to the minimum-n floor — honest, and reported as `draws_usable`.

    ⚠️ **Wide everywhere and wide nowhere are the same null.** A flat width scores `None` here (see
    `spearman`), not 0, and a falling `cf/evid_nll` beside a null correlation is the standing
    learns≠helps kill rather than a result.
    """
    cells = resolution_cells(labels, frame_mass, n_rollouts)
    usable = [c for c in cells
              if c.get("evid_width_mean") is not None and c.get("sd_true_excess") is not None]
    point = spearman([c["evid_width_mean"] for c in usable],
                     [c["sd_true_excess"] for c in usable])

    by_battle: "Dict[str, List[dict]]" = defaultdict(list)
    for r in labels:
        by_battle[r["battle"]].append(r)
    keys = list(by_battle)
    rho: List[float] = []
    if point is not None and len(keys) >= 2:
        rng = np.random.default_rng(seed)
        for _ in range(draws):
            idx = rng.integers(0, len(keys), size=len(keys))
            resampled = [r for j in idx for r in by_battle[keys[int(j)]]]
            cs = resolution_cells(resampled, frame_mass, n_rollouts)
            us = [c for c in cs if c.get("evid_width_mean") is not None]
            v = spearman([c["evid_width_mean"] for c in us],
                         [c["sd_true_excess"] for c in us])
            if v is not None:
                rho.append(v)
    ci = ([float(np.percentile(rho, 2.5)), float(np.percentile(rho, 97.5))]
          if len(rho) >= 100 else [None, None])
    all_w = [r["evid_width"] for r in labels if r.get("evid_width") is not None]
    all_p = [r["evid_precision"] for r in labels if r.get("evid_precision") is not None]
    return {
        "width_vs_blur_spearman": point,
        "width_vs_blur_ci": ci,
        "n_strata": len(usable),
        "draws_usable": len(rho),
        "evid_width_mean": float(np.mean(all_w)) if all_w else None,
        "evid_precision_mean": float(np.mean(all_p)) if all_p else None,
        "n_labels_scored": len(all_w),
    }


# ---------------------------------------------------------------------------
# The TWIN-HEAD paired read + the SHADOW critic (gen3_cf_twin_heads_v1)
# ---------------------------------------------------------------------------

#: The three win-prob heads, in the order the factorial reads them. The KEY is the label-dict field
#: `attach_twin_heads` writes; head A's is the one the bias map already carries.
TWIN_HEADS = (("A", "win_prob"), ("B", "twin_b_pred"), ("C", "twin_c_pred"))

#: The pre-registered paired CONTRASTS and what each isolates. Declared here rather than assembled
#: at the call site so a reader (and a test) sees the whole factorial in one place, and so the
#: interpretation travels with the arithmetic.
TWIN_CONTRASTS = (
    ("B_minus_A", "B", "A", "COVERAGE — the same loss form on the cf-labelled states"),
    ("C_minus_B", "C", "B", "PRECISION — the same states, a tight-MC target instead of one draw"),
    ("C_minus_A", "C", "A", "the TOTAL effect (coverage + precision), i.e. the original R1 claim"),
)


def paired_head_read(labels: Sequence[dict], *, draws: int = 2000, seed: int = 13) -> dict:
    """THE PRIMARY meter of the twin-heads amendment: paired proper-score differences.

    For every labelled state the audit holds three predictions of the SAME quantity — heads A, B
    and C — and one tight-MC measurement of it. So the comparison is a paired proper-scoring read:
    per row, ``brier_X = (pred_X − mc)²`` and ``abs_err_X = |pred_X − mc|``, differenced ACROSS
    HEADS ON THE SAME ROW, with a battle-clustered bootstrap CI on the difference.

    WHY THIS IS THE PRIMARY AND `sd_true_excess` IS NOT, in this arm. G0's amended meter exists
    because there was only ever ONE head, so the only available question was "how much true spread
    does this head fail to resolve". With three heads on identical rows there is a strictly sharper
    question — "which label stream produced the head that is CLOSER to ground truth" — and it has
    two properties the spread meter cannot match here:

    * **The hidden-information floor cancels EXACTLY**, not merely in expectation. The floor is a
      property of the STATE (`tmp/hidden_info_floor_report.md`: 39% of the meter's variance, and
      concentrated — 49% of states carry almost none, the top 10% carry half). Every head scores
      the same states, so it is the same additive constant in every arm and it vanishes in a
      per-row difference. The amended §2 argued the floor cancels in an arm-vs-control difference
      at matched step; within-run twins strengthen that to matched STATE.
    * **No stratification, so no selection correction is owed.** The eval capture quota
      over-samples losses, which is why the bias map recombines its cells at population shares.
      A paired difference over identical rows carries that bias identically in both terms.

    ⚠️ **A NEAR-ZERO DIFFERENCE WITH A NEAR-ZERO |B−C| IS NOT A NULL RESULT — it is a coverage or
    dosage reading.** If the twins never diverged (check `mean_abs_pred_diff` and the live
    `cf/twin_b_vs_c_abs`), the label streams did not separate the heads and there is nothing to
    decompose yet. The kill in the runbook applies to a divergence that HAPPENED and bought nothing.
    """
    # FILTER the rows, never test `all` over the unfiltered list. `attach_twin_heads` skips any
    # label whose `states.npz` is missing or whose `inv` is out of range (routine after
    # `prober.groom`), so ONE unscored row out of a thousand would otherwise delete heads B and C
    # from `present` and report "nothing to pair" on a run that scored 99.9% — a null that is
    # entirely an accounting artifact, on the arm's PRIMARY meter.
    keys = [k for _n, k in TWIN_HEADS]
    labels = [r for r in labels if all(r.get(k) is not None for k in keys[1:])
              or all(r.get(k) is None for k in keys[1:])]
    scored = [r for r in labels if r.get("twin_b_pred") is not None]
    if scored:
        labels = scored
    present = [(name, key) for name, key in TWIN_HEADS
               if labels and all(r.get(key) is not None for r in labels)]
    if len(present) < 2 or not labels:
        return {"heads_present": [n for n, _k in present], "n_labels": len(labels),
                "contrasts": [], "note": "fewer than two heads scored — nothing to pair"}
    battles = [r["battle"] for r in labels]
    mc = np.asarray([r["mc"] for r in labels], dtype=float)
    per_head = {}
    for name, key in present:
        pred = np.asarray([r[key] for r in labels], dtype=float)
        per_head[name] = {"pred": pred,
                          "brier": (pred - mc) ** 2,
                          "abs_err": np.abs(pred - mc)}
    out: dict = {
        "heads_present": [n for n, _k in present],
        "n_labels": len(labels),
        "n_battles": len(set(battles)),
        "by_head": {n: {"mean_pred": float(v["pred"].mean()),
                        "brier": float(v["brier"].mean()),
                        "abs_err": float(v["abs_err"].mean())}
                    for n, v in per_head.items()},
        "contrasts": [],
    }
    for label, hi, lo, meaning in TWIN_CONTRASTS:
        if hi not in per_head or lo not in per_head:
            continue
        row = {"contrast": label, "isolates": meaning}
        for metric in ("brier", "abs_err"):
            # SIGN CONVENTION, stated because it is the thing a reader gets wrong: these are ERROR
            # scores, so a NEGATIVE difference means the first-named head is BETTER.
            diff = per_head[hi][metric] - per_head[lo][metric]
            ci_lo, ci_hi = cluster_bootstrap_ci(diff.tolist(), battles, draws=draws, seed=seed)
            row[metric] = float(diff.mean())
            row[f"{metric}_ci"] = [ci_lo, ci_hi]
        row["mean_abs_pred_diff"] = float(
            np.abs(per_head[hi]["pred"] - per_head[lo]["pred"]).mean())
        out["contrasts"].append(row)
    return out


def twin_resolution_read(labels: Sequence[dict], n_rollouts: int) -> dict:
    """Each head's own `sd_true_excess`, binned by ITS OWN prediction — the G0 continuity link.

    Reported BESIDE `paired_head_read`, never instead of it, and with one caveat stated in the
    output itself: these cells are **UNWEIGHTED**. The bias map recombines its decile cells at the
    eval frame's population (decile, outcome) shares to undo the capture quota's loss
    over-sampling — and that correction is unavailable for heads B and C, because the frame carries
    only head A's predictions, so B's and C's decile membership over the whole frame is unknown.
    Re-using A's masses for B's bins would be a number that looks population-weighted and is not.

    So: absolute levels here are NOT comparable with the bias map's
    `population_weighted_sd_true_excess`, and the field is named to make that impossible to miss.
    What they are good for is the SHAPE — whether a head's blur moved in the deciles the arm
    predicted it would.
    """
    out: dict = {"weighting": "UNWEIGHTED — not comparable with the bias map's "
                              "population_weighted_sd_true_excess; see the docstring",
                 "by_head": {}}
    for name, key in TWIN_HEADS:
        # Per-head row filtering, for `paired_head_read`'s reason: a single unscored row must not
        # delete a whole head's block.
        rows = [r for r in labels if r.get(key) is not None]
        if not rows:
            continue
        cells = []
        for dec in range(10):
            sub = [r for r in rows if min(9, int(float(r[key]) * 10)) == dec]
            if len(sub) < MIN_CELL_N:
                continue
            st = sd_true_excess([r["mc"] for r in sub], n_rollouts)
            st["decile"] = dec
            st["mean_predicted"] = float(np.mean([float(r[key]) for r in sub]))
            cells.append(st)
        tot = sum(c["n"] for c in cells) or 1
        out["by_head"][name] = {
            "cells": cells,
            "sample_weighted_sd_true_excess": float(
                sum(c["n"] * (c["sd_true_excess"] or 0.0) for c in cells) / tot),
        }
    return out


def shadow_read(labels: Sequence[dict], *, draws: int = 2000, seed: int = 17) -> Optional[dict]:
    """The SHADOW critic's audit block, or None when the checkpoint carries no shadow head.

    Two readings, and they answer different questions:

    * **`shadow_vs_live_v`** — the signed mean of (shadow V − live V) in real shaped-return units
      on the same states, battle-clustered. This is the staged-promotion meter: a shadow sitting
      systematically BELOW the live critic says the live critic is optimistic about the states the
      factory samples, measured against a ground-truth target rather than argued from a curve.
    * **`shadow_vs_live_v_abs`** — the same gap unsigned, so a large signed value can be told from
      a large scatter that happens to average near zero.

    ⚠️ **THE STANDING CAVEAT, and there is no field that removes it here.** This block compares two
    FITTED heads. A divergence is evidence about the PAIR; it does not say which of them moved, and
    this function computes no external anchor that would. Reading it as "the live critic is
    optimistic" needs the trainer-side `cf/shadow_live_v_vs_label` (live V against the MC label
    itself) beside it — that one has a ground-truth arm. An anchor column here was considered and
    NOT built: the audit's MC is a WIN PROBABILITY and the shadow's units are shaped return, so any
    mapping between them would be a crude direction check dressed as a calibration.

    Reported with `n` and `n_battles` on every line. The shadow is trained off ``mc_return`` rows,
    and a run whose producer never shipped one carries a randomly-initialised head whose divergence
    from the live critic is pure noise — the trainer-side `cf/shadow_coverage` is the scalar that
    says so, since a checkpoint cannot report what it was trained on.
    """
    rows = [r for r in labels if r.get("shadow_value") is not None]
    if not rows:
        return None
    battles = [r["battle"] for r in rows]
    shadow = np.asarray([r["shadow_value"] for r in rows], dtype=float)
    out: dict = {"n": len(rows), "n_battles": len(set(battles)),
                 "shadow_mean": float(shadow.mean())}
    live = [r.get("live_v") for r in rows]
    if all(v is not None for v in live):
        lv = np.asarray(live, dtype=float)
        diff = shadow - lv
        lo, hi = cluster_bootstrap_ci(diff.tolist(), battles, draws=draws, seed=seed)
        out.update(live_v_mean=float(lv.mean()),
                   shadow_vs_live_v=float(diff.mean()),
                   shadow_vs_live_v_ci=[lo, hi],
                   shadow_vs_live_v_abs=float(np.abs(diff).mean()))
    return out


def attach_twin_heads(session, labels: List[dict], npz_cache: "Dict[str, np.ndarray]") -> int:
    """Score every labelled state through the twin heads and the shadow critic, in place.

    Adds ``twin_b_pred`` / ``twin_c_pred`` (P(win) from heads B and C) and ``shadow_value`` /
    ``live_v`` (real-unit values) where the checkpoint carries them; returns how many rows were
    scored. **Zero is a first-class outcome** — a checkpoint without the heads leaves the fields
    ABSENT and every reader downstream omits its columns rather than filling them with a number
    that would read as a measurement.

    BEST-EFFORT, exactly like `attach_evidential`: the audit's products are the labels and the bias
    map, and a model that will not load (architecture drift — 79 of 79 archived runs cannot be
    re-loaded at any given HEAD) must cost the run these columns and nothing else.
    """
    if not labels or not hasattr(session, "probe_model"):
        return 0
    try:
        model, _choice = session.probe_model(labels[0]["battle"] + "_summary.json")
    except Exception as exc:                                            # noqa: BLE001
        print(f"  twin heads: no model ({type(exc).__name__}: {str(exc)[:120]}) — "
              f"columns omitted", flush=True)
        return 0
    if not hasattr(model, "cf_twin_batch"):
        return 0
    obs_rows, targets = [], []
    for r in labels:
        arr = npz_cache.get(r["battle"] + "_states.npz")
        if arr is None or not (0 <= r["inv"] < len(arr)):
            continue
        obs_rows.append(arr[r["inv"]])
        targets.append(r)
    if not obs_rows:
        return 0
    try:
        out = model.cf_twin_batch(np.stack(obs_rows))
    except Exception as exc:                                            # noqa: BLE001
        print(f"  twin heads: forward failed ({type(exc).__name__}: {str(exc)[:120]}) — "
              f"columns omitted", flush=True)
        return 0
    if not out:
        print("  twin heads: this checkpoint carries neither cf_twin_head_b nor cf_shadow_head — "
              "columns omitted", flush=True)
        return 0
    for col, values in out.items():        # `col`, not `field`: `field` is the dataclasses import
        for r, v in zip(targets, values):
            r[col] = float(v)
    print(f"  twin heads: scored {len(targets)} states ({', '.join(sorted(out))})", flush=True)
    return len(targets)


def render_markdown(bm: dict, *, run_dir: str, step: Optional[int], ckpt: Optional[str]) -> str:
    L = []
    ap = L.append
    h = bm.get("headline") or {}
    ap("# cf_audit — the counterfactual bias map\n")
    ap(f"**Run:** `{run_dir}`  ·  **step:** {step}  ·  **checkpoint:** `{ckpt}`  ·  "
       f"**R:** {bm['n_rollouts']}  ·  **sampler:** `{bm['sampler']['sampler_version']}` "
       f"(seed {bm['sampler']['seed']})\n")
    ap("## Headline\n")
    ap("```")
    ap(f"labels                          {h.get('n_labels')}   over {h.get('n_battles')} battles")
    ap(f"population-weighted gap         {h.get('population_weighted_gap'):+.4f}"
       if h.get("population_weighted_gap") is not None else "population-weighted gap  n/a")
    ap(f"population-weighted sd_true_excess  {h.get('population_weighted_sd_true_excess'):.4f}"
       if h.get("population_weighted_sd_true_excess") is not None else "sd_true_excess  n/a")
    ap("```")
    ap("\nThe **gap** is the offset a re-centring would fix; the **sd_true_excess** is the "
       "per-state spread the head does not resolve, and it is the primary meter. A lever that "
       "moves the first and not the second has not done the thing this program is for.\n")

    acc = bm.get("accounting") or {}
    ap("## Accounting\n")
    ap("| | |\n|---|---|")
    for k in ("frame_decisions", "frame_battles", "tasks_issued", "labelled", "errors",
              "anchors_issued", "anchors_reproduced", "rollouts"):
        if k in acc:
            ap(f"| {k.replace('_', ' ')} | {acc[k]} |")
    if acc.get("skipped"):
        ap(f"| skipped (frame) | {json.dumps(acc['skipped'])} |")
    ap("")

    def _tab(title, rows, cols=("stratum", "n", "n_battles", "mean_predicted", "mean_mc", "mean_gap")):
        if not rows:
            return
        ap(f"## {title}\n")
        ap("| " + " | ".join(cols) + " | 95% CI (battle-clustered) |")
        ap("|" + "---|" * (len(cols) + 1))
        for r in rows:
            ci = r.get("gap_ci") or [None, None]
            cells = []
            for c in cols:
                v = r.get(c)
                cells.append(f"{v:+.4f}" if isinstance(v, float) else str(v))
            ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                    if ci[0] is not None else "—")
            ap("| " + " | ".join(cells) + f" | {ci_s} |")
        ap("")

    _tab("By battle outcome (a description of two state POPULATIONS, not a calibration verdict)",
         bm.get("by_outcome"))
    _tab("By predicted decile × outcome", bm.get("by_decile_outcome"))
    _tab("By turn tercile", bm.get("by_turn_tercile"))
    _tab("By opponent", bm.get("by_opponent"))

    cc = bm.get("conviction_class")
    if cc:
        ap("## The conviction class — high confidence, lost battle\n")
        ap("```")
        ap(f"n={cc['n']} over {cc['n_battles']} battles")
        ap(f"predicted {cc['mean_predicted']:.3f} (median {cc['median_predicted']:.3f})  "
           f"vs tight-MC {cc['mean_mc']:.3f} (median {cc['median_mc']:.3f})")
        def _ci(v):
            return ("—" if not v or v[0] is None else f"[{v[0]:+.4f}, {v[1]:+.4f}]")
        ap(f"gap {cc['mean_gap']:+.4f}  CI {_ci(cc.get('gap_ci'))}")
        if cc.get("loss_minus_win_ci"):
            ap(f"LOSS - WIN difference {cc['loss_minus_win_gap']:+.4f}  "
               f"CI {_ci(cc['loss_minus_win_ci'])}")
        ap(f"MC >= 0.75 (the critic was RIGHT; the dice lost it)  {cc['share_mc_ge_0.75'] * 100:.1f}%")
        ap(f"MC <  0.50 (the critic was genuinely wrong)          {cc['share_mc_lt_0.50'] * 100:.1f}%")
        ap(f"MC <  0.25 (badly wrong)                             {cc['share_mc_lt_0.25'] * 100:.1f}%")
        ap("```")
        ap("\nA single realized outcome cannot separate those two populations. That separation "
           "is the whole case for a tight-MC label as an instrument.\n")

    res = bm.get("resolution")
    has_evid = bool(res) and any(r.get("evid_width_mean") is not None for r in res)
    if res:
        ap("## RESOLUTION — within-decile true spread vs the binomial floor\n")
        cols = ("| decile | n | predicted | MC | sd(MC) | binomial floor | "
                "**sd_true_excess** | % variance real |")
        rule = "|---|---|---|---|---|---|---|---|"
        if has_evid:
            cols += " Beta width | Beta precision |"
            rule += "---|---|"
        ap(cols)
        ap(rule)
        for r in res:
            fv = f"{r['frac_variance_real'] * 100:.1f}%" if r.get("frac_variance_real") else "—"
            line = (f"| {r['decile']} | {r['n']} | {r['mean_predicted']:.3f} | {r['mean']:.3f} | "
                    f"{r['sd_observed']:.3f} | {r['sd_binomial_floor']:.3f} | "
                    f"**{r['sd_true_excess']:.3f}** | {fv} |")
            if has_evid:
                w, p = r.get("evid_width_mean"), r.get("evid_precision_mean")
                line += (f" {w:.3f} |" if w is not None else " — |")
                line += (f" {p:.2f} |" if p is not None else " — |")
            ap(line)
        ap("")

    ev = bm.get("evidential")
    ap("## EVIDENTIAL — does the confessed width track the blur?\n")
    if not ev:
        # A one-line NOTE, never a row of zeros: "this checkpoint has no head" and "this head
        # claims no uncertainty" are opposite findings and must not render the same.
        ap("_The audited checkpoint carries no `cf_evid_head` (`--cf-evidential` off, or pre-v98) —"
           " the evidential columns are ABSENT, not zero._\n")
    else:
        rho, ci = ev.get("width_vs_blur_spearman"), ev.get("width_vs_blur_ci") or [None, None]
        ap("```")
        ap(f"labels scored                   {ev.get('n_labels_scored')}   over "
           f"{ev.get('n_strata')} strata")
        ap(f"mean Beta width (epistemic sd)  "
           f"{ev['evid_width_mean']:.4f}" if ev.get("evid_width_mean") is not None else
           "mean Beta width  n/a")
        ap(f"mean Beta precision (alpha+beta)    "
           f"{ev['evid_precision_mean']:.3f}" if ev.get("evid_precision_mean") is not None else
           "mean Beta precision  n/a")
        ap(f"width_vs_blur_spearman          "
           f"{rho:+.3f}" if rho is not None else
           "width_vs_blur_spearman          n/a (flat width, or <3 strata)")
        if ci[0] is not None:
            ap(f"  95% CI (battle-clustered)     [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
               f"over {ev.get('draws_usable')} usable draws")
        ap("```")
        ap("\nThe head reads the same `value_pooled` as the scalar one, so it cannot REMOVE the "
           "blur — only confess it. Success is this correlation, not a falling `nll`: **wide "
           "everywhere and wide nowhere are the same null**, and a flat width reports `n/a` rather "
           "than 0 so the two cannot be confused.\n")
    ap("## Caveats\n")
    ap("- Turn-1 decisions are excluded by construction (the offline replay driver cannot open "
       "them) and forced-switch rounds are structurally uncovered by the re-roll anchor.")
    ap("- The MC label is measured on the EVAL distribution, played greedy; the head was "
       "trained on a mostly-self-play mixture with a stochastic actor. Never quote a gap "
       "without naming the population — its SIGN depends on the weighting.")
    ap(f"- R = {bm['n_rollouts']}: a single label's own sd is at most "
       f"{0.5 / math.sqrt(bm['n_rollouts']):.3f} (95% half-width "
       f"±{1.96 * 0.5 / math.sqrt(bm['n_rollouts']):.2f}). Cell aggregates are honest; a single "
       "state's label is not a point value.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_traces(run_dir: str, step: Optional[int]) -> str:
    root = os.path.join(run_dir, "eval_traces")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"no eval_traces under {run_dir}")
    steps = sorted(glob.glob(os.path.join(root, "step_*")))
    if not steps:
        raise FileNotFoundError(f"no step_* trace dirs under {root}")
    if step is None:
        return steps[-1] if len(steps) == 1 else max(
            steps, key=lambda p: int(os.path.basename(p).split("_")[1]))
    want = os.path.join(root, f"step_{step}")
    if not os.path.isdir(want):
        raise FileNotFoundError(f"{want} does not exist (have: "
                                f"{[os.path.basename(s) for s in steps]})")
    return want


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m agents.training.cf_audit",
        description="Counterfactual audit of a checkpoint: tight-MC value labels + the bias map.")
    p.add_argument("run_dir", help="a models/<run> directory (needs eval_traces + metadata.json)")
    p.add_argument("--checkpoint", default=None,
                   help="trainee checkpoint override (default: the prober's exact->nearest->recent ladder)")
    p.add_argument("--rollouts", type=int, default=8,
                   help="R, rollouts per label (default 8; the per-state SNR is ~2-4:1 there)")
    p.add_argument("--states", type=int, default=200, help="how many decisions to label")
    p.add_argument("--step", type=int, default=None, help="which eval_traces/step_N (default: latest)")
    p.add_argument("--impl", default="rust", choices=["node", "rust"],
                   help="offline replay/re-roll driver (default rust)")
    p.add_argument("--out", default=None, help="output dir (default <run_dir>/cf_audit)")
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--anchors", type=int, default=20,
                   help="label-trust arm: R=1 recorded-dice replays that must reproduce the record")
    p.add_argument("--anchor-tolerance", type=float, default=0.9,
                   help="minimum anchor reproduction rate; below it the audit REFUSES to emit labels")
    p.add_argument("--producer", default="cf_audit", help="label-file producer tag")
    p.add_argument("--inline-obs", action="store_true",
                   help="embed the obs in each label row (base64 float32) instead of pointing at "
                        "states.npz — bigger files, but self-contained")
    p.add_argument("--deadline-min", type=float, default=0.0,
                   help="stop taking new tasks after this many minutes (0 = no bound)")
    return p


def attach_evidential(session, labels: List[dict], npz_cache: "Dict[str, np.ndarray]") -> int:
    """Score every labelled state through the checkpoint's evidential Beta head, in place.

    Adds ``evid_width`` (the Beta's epistemic std) and ``evid_precision`` (α+β) to each label dict;
    returns how many rows were scored. **Zero is a first-class outcome** — a checkpoint without
    ``cf_evid_head`` (pre-v98, or `--cf-evidential` off) leaves the fields absent, and every reader
    downstream omits its evidential columns rather than filling them with zeros.

    BEST-EFFORT, deliberately. The audit's products are the labels and the bias map; the evidential
    read is a rider on them. A model that will not load (architecture drift — 79 of 79 archived runs
    cannot be re-loaded at any given HEAD) must cost the run its evidential columns and nothing else,
    so every failure here is a printed line and a return of 0.
    """
    if not labels or not hasattr(session, "probe_model"):
        return 0
    try:
        model, _choice = session.probe_model(labels[0]["battle"] + "_summary.json")
    except Exception as exc:                                            # noqa: BLE001
        print(f"  evidential: no model ({type(exc).__name__}: {str(exc)[:120]}) — "
              f"columns omitted", flush=True)
        return 0
    if not hasattr(model, "cf_evidential_batch"):
        return 0
    obs_rows, targets = [], []
    for r in labels:
        arr = npz_cache.get(r["battle"] + "_states.npz")
        if arr is None or not (0 <= r["inv"] < len(arr)):
            continue
        obs_rows.append(arr[r["inv"]])
        targets.append(r)
    if not obs_rows:
        return 0
    try:
        out = model.cf_evidential_batch(np.stack(obs_rows))
    except Exception as exc:                                            # noqa: BLE001
        print(f"  evidential: forward failed ({type(exc).__name__}: {str(exc)[:120]}) — "
              f"columns omitted", flush=True)
        return 0
    if out is None:
        print("  evidential: this checkpoint carries no cf_evid_head — columns omitted", flush=True)
        return 0
    alpha, beta = out
    prec = alpha + beta
    # The Beta's std, the head's own `epistemic_std` in closed form. Computed here rather than
    # called through the module so the audit stays importable without torch on the hot path.
    width = np.sqrt(alpha * beta / (prec * prec * (prec + 1.0)))
    for r, a, b, w in zip(targets, alpha, beta, width):
        r["evid_alpha"], r["evid_precision"] = float(a), float(a + b)
        r["evid_width"] = float(w)
    print(f"  evidential: scored {len(targets)} states (mean width {float(width.mean()):.4f}, "
          f"mean precision {float(prec.mean()):.2f})", flush=True)
    return len(targets)


def _default_session(traces: str, *, impl: str, ckpt_override: Optional[str]):
    from main.prober.session import ProbeSession
    return ProbeSession(traces, impl=impl, ckpt_override=ckpt_override, compile_extractor=True)


def main(argv: "Optional[Sequence[str]]" = None, *, session_factory=None) -> int:
    """``session_factory(traces, impl=…, ckpt_override=…)`` returns the object the labeler
    calls ``replay_counterfactual`` on. It exists so the end-to-end test can run the REAL
    bridge rollouts without a current-architecture checkpoint on disk; production always
    takes the default (a ``ProbeSession``)."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    args = build_parser().parse_args(argv)
    import torch
    torch.set_num_threads(1)

    traces = _resolve_traces(args.run_dir, args.step)
    step = int(os.path.basename(traces).split("_")[1])
    out_dir = args.out or os.path.join(args.run_dir, "cf_audit")
    os.makedirs(out_dir, exist_ok=True)

    frame, skipped = build_frame(traces)
    if not frame:
        print(f"cf_audit: no labelable decisions under {traces} — "
              f"skipped {dict(skipped)}", file=sys.stderr)
        return 2
    sample, design = stratified_sample(frame, args.states, seed=args.seed)
    snaps = sentinel_snapshots(args.run_dir)
    print(f"cf_audit: frame {len(frame)} decisions / "
          f"{len({d.battle for d in frame})} battles → {len(sample)} sampled "
          f"(R={args.rollouts}, load={os.getloadavg()[0]:.1f})", flush=True)
    print(f"  skipped: {dict(skipped)}", flush=True)

    session = (session_factory or _default_session)(
        traces, impl=args.impl, ckpt_override=args.checkpoint)

    # ── LABEL TRUST FIRST. A bias map from an inexact replay measures the bug. ──
    rng = random.Random(args.seed ^ 0x5EED)
    anchor_pool = [d for d in frame if d.opp_class == "bot"]
    by_battle: "Dict[str, List[Decision]]" = defaultdict(list)
    for d in anchor_pool:
        by_battle[d.battle].append(d)
    bkeys = sorted(by_battle)
    rng.shuffle(bkeys)
    anchors, anchor_ok, anchor_err = [], 0, 0
    for b in bkeys[: args.anchors]:
        ds = sorted(by_battle[b], key=lambda d: d.move_rank)
        anchors.append(ds[len(ds) // 2])
    for d in anchors:
        try:
            out = _label_one(session, d, n_rollouts=1, opp_ckpt=None)
        except Exception as exc:                                        # noqa: BLE001
            anchor_err += 1
            print(f"  anchor {d.short} inv {d.inv}: {type(exc).__name__}: {str(exc)[:160]}",
                  flush=True)
            continue
        realized = "win" if out["wins"] else ("loss" if out["losses"] else "tie")
        anchor_ok += int(realized == d.outcome)
    n_anchor = len(anchors)
    rate = anchor_ok / n_anchor if n_anchor else 0.0
    print(f"  ANCHOR: {anchor_ok}/{n_anchor} reproduced the recorded outcome "
          f"({rate * 100:.1f}%), {anchor_err} errors", flush=True)
    trust_ok = n_anchor > 0 and rate >= args.anchor_tolerance

    # ── the tight-MC pass ──
    t0 = time.perf_counter()
    deadline = args.deadline_min * 60.0
    labels: List[dict] = []
    label_rows: List[LabelRow] = []
    n_err = 0
    npz_cache: "Dict[str, np.ndarray]" = {}
    for j, d in enumerate(sample):
        if deadline and (time.perf_counter() - t0) > deadline:
            print(f"  deadline hit at {j}/{len(sample)}", flush=True)
            break
        ckpt = snaps.get(d.opponent) if d.opp_class == "sentinel" else None
        try:
            out = _label_one(session, d, n_rollouts=args.rollouts, opp_ckpt=ckpt)
        except Exception as exc:                                        # noqa: BLE001
            n_err += 1
            if n_err <= 10:
                print(f"  {d.short} inv {d.inv}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            continue
        mc = out["win_rate"]
        if mc is None:
            n_err += 1
            continue
        labels.append({"battle": d.battle, "short": d.short, "inv": d.inv, "turn": d.turn,
                       "opponent": d.opponent, "opp_class": d.opp_class, "outcome": d.outcome,
                       "win_prob": d.win_prob, "value": d.value, "mc": float(mc),
                       "wins": out["wins"], "n": out["n_rollouts"],
                       "opponent_source": out["opponent_source"]})
        npz_path = d.battle + "_states.npz"
        arr = npz_cache.get(npz_path)
        if arr is None:
            with np.load(npz_path) as z:
                arr = np.asarray(z["obs"], dtype=np.float32)
            npz_cache[npz_path] = arr
        obs = arr[d.inv]
        lo, hi = wilson_ci(int(out["wins"]), int(out["n_rollouts"]))
        label_rows.append(LabelRow(
            battle=d.battle + "_reconstruction.json", decision_idx=d.inv,
            obs_sha1=obs_digest(obs),
            obs_npz=None if args.inline_obs else f"{npz_path}::obs",
            obs_inline=obs_b64(obs) if args.inline_obs else None,
            label=float(mc), n_rollouts=int(out["n_rollouts"]),
            wilson_lo=round(lo, 6), wilson_hi=round(hi, 6),
            policy_step=step, opponent=d.opponent))
        if j % 20 == 0:
            print(f"  {j + 1}/{len(sample)} ok={len(labels)} err={n_err} "
                  f"{(time.perf_counter() - t0) / 60:.1f}m load={os.getloadavg()[0]:.1f}", flush=True)

    # ── the EVIDENTIAL read: one batched forward of the audited checkpoint over the labelled
    #    states, attaching (width, precision) to each label row. Best-effort by design — the
    #    labels and the bias map are the audit's product, and a checkpoint that cannot be loaded
    #    (or carries no head) must cost the run its evidential columns, never its labels.
    n_evid = attach_evidential(session, labels, npz_cache)
    # ── the TWIN-HEAD + SHADOW read (gen3_cf_twin_heads_v1): the same shape, one more batched
    #    forward, attaching heads B/C's P(win) and the shadow critic's value to each row. Same
    #    best-effort contract for the same reason.
    n_twin = attach_twin_heads(session, labels, npz_cache)

    accounting = {
        "frame_decisions": len(frame), "frame_battles": len({d.battle for d in frame}),
        "evidential_scored": n_evid,
        "twin_scored": n_twin,
        "tasks_issued": len(sample), "labelled": len(labels), "errors": n_err,
        "anchors_issued": n_anchor, "anchors_reproduced": anchor_ok,
        "anchor_errors": anchor_err, "anchor_rate": round(rate, 4),
        "rollouts": sum(r["n"] for r in labels),
        "core_minutes": round((time.perf_counter() - t0) / 60, 2),
        "loadavg": os.getloadavg()[0],
        "skipped": dict(skipped),
        "label_trust_passed": trust_ok,
    }

    bm = bias_map(labels, frame, n_rollouts=args.rollouts, design=design, accounting=accounting)
    bm["run_dir"] = os.path.abspath(args.run_dir)
    bm["traces"] = traces
    bm["policy_step"] = step
    bm["impl"] = args.impl
    with open(os.path.join(out_dir, "bias_map.json"), "w") as fh:
        json.dump(bm, fh, indent=1)
    md = render_markdown(bm, run_dir=args.run_dir, step=step, ckpt=args.checkpoint)
    with open(os.path.join(out_dir, "bias_map.md"), "w") as fh:
        fh.write(md)

    if not trust_ok:
        print(f"\ncf_audit: LABEL TRUST FAILED — {anchor_ok}/{n_anchor} anchors reproduced the "
              f"recorded outcome ({rate * 100:.1f}% < {args.anchor_tolerance * 100:.0f}%). "
              f"REFUSING to emit labels; the bias map is written for diagnosis ONLY and must "
              f"not be quoted.", file=sys.stderr)
        return 3
    path = write_labels(label_rows, out_dir, producer=args.producer, seq=step)
    print(f"\ncf_audit: {len(labels)} labels / {accounting['rollouts']} rollouts / "
          f"{accounting['core_minutes']} core-min")
    print(f"  bias map: {os.path.join(out_dir, 'bias_map.md')}")
    print(f"  labels:   {path}")
    h = bm.get("headline") or {}
    if h:
        print(f"  gap {h['population_weighted_gap']:+.4f}   "
              f"sd_true_excess {h['population_weighted_sd_true_excess']:.4f}")
    ev = bm.get("evidential")
    if ev and ev.get("width_vs_blur_spearman") is not None:
        ci = ev.get("width_vs_blur_ci") or [None, None]
        ci_s = f"  CI [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci[0] is not None else ""
        print(f"  width_vs_blur_spearman {ev['width_vs_blur_spearman']:+.3f}{ci_s}   "
              f"over {ev['n_strata']} strata")
    return 0


if __name__ == "__main__":
    sys.exit(main())
