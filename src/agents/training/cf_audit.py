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
                # The offline replay driver cannot open turn 1. Counted, never silent.
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

def wilson_ci(wins: int, n: int, z: float = 1.96) -> "tuple[float, float]":
    """Wilson score interval — the right small-N binomial CI (a normal approximation gives
    the degenerate [0, 0] at 0 wins). ``(0.0, 1.0)`` for n == 0."""
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


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
        "resolution": [], "headline": {},
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
    for dec in range(10):
        rows, w = [], []
        for oc in ("win", "loss"):
            sub = [r for r in labels
                   if min(9, int(r["win_prob"] * 10)) == dec and r["outcome"] == oc]
            mass = frame_mass.get((dec, oc), 0)
            if len(sub) < 3 or mass <= 0:
                continue
            rows += sub
            w += [mass / len(sub)] * len(sub)
        if len(rows) < 12:
            continue
        st = sd_true_excess([r["mc"] for r in rows], n_rollouts, weights=w)
        st["decile"] = dec
        st["mean_predicted"] = float(np.average([r["win_prob"] for r in rows],
                                                weights=np.asarray(w)))
        out["resolution"].append(st)

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
    if res:
        ap("## RESOLUTION — within-decile true spread vs the binomial floor\n")
        ap("| decile | n | predicted | MC | sd(MC) | binomial floor | **sd_true_excess** | % variance real |")
        ap("|---|---|---|---|---|---|---|---|")
        for r in res:
            fv = f"{r['frac_variance_real'] * 100:.1f}%" if r.get("frac_variance_real") else "—"
            ap(f"| {r['decile']} | {r['n']} | {r['mean_predicted']:.3f} | {r['mean']:.3f} | "
               f"{r['sd_observed']:.3f} | {r['sd_binomial_floor']:.3f} | "
               f"**{r['sd_true_excess']:.3f}** | {fv} |")
        ap("")
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

    accounting = {
        "frame_decisions": len(frame), "frame_battles": len({d.battle for d in frame}),
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
