"""X72, the swap forks carried toward their plateau: THE STOP RULE of the rounds and THE ENTROPY FLOOR (PREREG §6, §7).

A frozen record of one registration, never a tool. It reads no game and no run: ``decide-round`` takes a matrix a
cross reader hands it, ``floor`` takes entropies a diagnostics reader hands it.

THE ROUND CROSS (PREREG §6). After round k (k = 1..4) the three round-k finals (rows, lineages s7341 / s7342 / s7343)
play the three round-(k-1) finals (columns; round 0 = the banked swap finals ``rb_x35_ce3dswap_s7321..s7323``), every
row against every column, 9 cells x 4,000 mirrored pairs, both sides greedy. ``h[i][j]`` is row i's score against
column j in pp.

THE DECIDING STATISTIC is about THESE SIX FILES, as the plateau meter's Tier 1 is about two nodes of one run
(``main.plateau``): did the three round-k finals, as a group, beat the three round-(k-1) finals? Game noise is the
only noise in that question, so the interval is the fixed-finals one:

    D = mean(h) - 50;  SE = sqrt(cell_var / 9), cell_var = 2500 / (2 x 4,000) = 0.3125 pp^2 (the binomial bound at
    50 %; banked 1,000-pair cells measured 0.80-1.10 pp against the bound's 1.118, X70's ``simulate``);
    90 % interval D +- z(0.95) SE = D +- 0.3065 pp.

    GAIN  iff the lower end > 0    (round k's finals beat round k-1's)
    LOSS  iff the upper end < 0
    FLAT  otherwise
    Rule 8: an end within 1e-9 pp of 0 fires nothing and is recorded.

REPORTED beside it, never deciding: X70's ``rect_stat`` (``../x70_ent0_control_2026-11-13/screen_rule.py``,
unchanged: V = s2R / 3 + s2C / 3 with binomial floors, Welch df, t), the interval that treats the three lineages as
a SAMPLE of forks. It is the one a claim about the RECIPE ("more dose under the swap helps a fork") would need, and
at three lineages it is 2 to 4 times wider (``simulate``). The stop rule makes no such claim (PREREG §6.2).

THE NEXT ACTION (PREREG §6.3), a pure function of the round, the cross's class and the floor:

    floor fired in round k           -> STOP,  ENT-FLOOR,              read round k-1  (the cross of round k is not played)
    cross INCONCLUSIVE               -> HOLD   (the orchestrator's; nothing is launched)
    LOSS                             -> STOP,  DOSE-HARM,              read round k-1
    FLAT                             -> STOP,  PLATEAU-AT-REACH,       read round k
    GAIN and k < 4                   -> CONTINUE to round k+1
    GAIN and k = 4                   -> STOP,  NOT-AT-PLATEAU-AT-CAP,  read round 4

"read round 0" means the banked swap finals: no new Kakuna read is played under this registration (PREREG §8.6).

THE ENTROPY FLOOR (PREREG §7): half the LOWEST banked swap end entropy, read from CE-3d's banked
``record/diag9_ce3d.json`` and never retyped (0.5 x 0.48377 = 0.24188 nats; X70's registered ENT-COLLAPSED bar). The
statistic is the mean of a run's last 20 ``-train/entropy_loss`` points at or before a registered mark. The floor
FIRES iff any lineage's statistic is below the floor by more than 1e-9 nats; a value within 1e-9 fires nothing and
is recorded.

    python x72_rule.py self-check                   # X70's banked verdict reproduced, hand cases, the action table
    python x72_rule.py simulate                     # one look's and the chain's operating characteristics (PREREG §6.4)
    python x72_rule.py decide-round <k> <h.json>    # h.json = {"h_pp": [[...] x 3] x 3} -> one JSON verdict + action
    python x72_rule.py floor <e1> <e2> <e3>         # three lineages' statistics (nats) -> fired or not

Run with ``PYTHONPATH=<a checkout>/src`` (``main.h2h.cross`` supplies ``t_ppf`` and ``RULE8_EPS``).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from main.h2h.cross import RULE8_EPS, t_ppf

HERE = Path(__file__).resolve().parent
CE3D_RECORD = HERE.parent / "ce3d_magnet_screen_2026-11-11" / "record"
X70_ANCHOR = HERE.parent / "x70_ent0_control_2026-11-13" / "record" / "anchor"

GAIN, LOSS, FLAT, INCONCLUSIVE = "GAIN", "LOSS", "FLAT", "INCONCLUSIVE"
Z_095 = 1.6448536269514722                   # the standard normal's 0.95 quantile (statistics.NormalDist().inv_cdf(0.95))
CONTINUE, STOP, HOLD = "CONTINUE", "STOP", "HOLD"
ENT_FLOOR, DOSE_HARM, PLATEAU, CAP = "ENT-FLOOR", "DOSE-HARM", "PLATEAU-AT-REACH", "NOT-AT-PLATEAU-AT-CAP"
N = 3                                        # three lineages: rows and columns
PAIRS = 4000
MAX_ROUNDS = 4
CI_LEVEL = 0.90
FLOOR_FRACTION = 0.5                         # of the lowest banked swap end entropy
ENT_EPS_NATS = 1e-9
GPU_H_A_ROUND = 12.75                        # three swap forks x +20M, measured (X70 PREREG §11); the price only


def cell_var(pairs: int) -> float:
    """One cell's binomial variance at 50 %, pp^2: 2500 / games."""
    return 2500.0 / (2 * pairs)


CELL_VAR_PP2 = cell_var(PAIRS)               # 0.3125


def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    m = _mean(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def rect_stat(h: Sequence[Sequence[float]], *, cv: float = CELL_VAR_PP2) -> Dict[str, Any]:
    """X70's statistic on an r x c matrix (pp), with its registered binomial floors."""
    r, c = len(h), len(h[0])
    rows = [_mean(line) for line in h]
    cols = [_mean([h[i][j] for i in range(r)]) for j in range(c)]
    raw_r, raw_c = _var(rows), _var(cols)
    s2r, s2c = max(raw_r, cv / c), max(raw_c, cv / r)
    tr, tc = s2r / r, s2c / c
    v = tr + tc
    df = v * v / (tr * tr / (r - 1) + tc * tc / (c - 1))
    d = _mean([x for line in h for x in line]) - 50.0
    q = t_ppf(0.5 + CI_LEVEL / 2.0, df)
    se = math.sqrt(v)
    return {"delta_hat_pp": d, "se_pp": se, "df": df, "t_0.95": q, "interval90_pp": [d - q * se, d + q * se],
            "s2_rows_raw": raw_r, "s2_cols_raw": raw_c, "s2_rows": s2r, "s2_cols": s2c,
            "floor_rows_bound": raw_r < cv / c, "floor_cols_bound": raw_c < cv / r,
            "row_means_pp": rows, "col_means_pp": cols, "shape": [r, c], "cell_var_pp2": cv}


def fixed_stat(h: Sequence[Sequence[float]], *, cv: float = CELL_VAR_PP2) -> Dict[str, Any]:
    """The deciding statistic: the mean of the cells against 50, game noise only (the finals are fixed)."""
    n = sum(len(line) for line in h)
    d = _mean([x for line in h for x in line]) - 50.0
    se = math.sqrt(cv / n)
    return {"delta_hat_pp": d, "se_pp": se, "z_0.95": Z_095, "interval90_pp": [d - Z_095 * se, d + Z_095 * se],
            "cells": n, "cell_var_pp2": cv}


def _shape_problems(h: Sequence[Sequence[float]]) -> List[str]:
    if len(h) != N or any(len(line) != N for line in h):
        return [f"the round cross is {N} x {N}; got {[len(line) for line in h]}"]
    if not all(math.isfinite(x) and 0.0 <= x <= 100.0 for line in h for x in line):
        return ["a cell is not a finite score in [0, 100] pp"]
    return []


def decide_round(h: Sequence[Sequence[float]], inconclusive: Sequence[str] = ()) -> Dict[str, Any]:
    """The class of one round cross (rows = round k's finals, columns = round k-1's; pp; 4,000 pairs a cell)."""
    reasons: List[str] = list(inconclusive) or _shape_problems(h)
    if reasons:
        return {"outcome": INCONCLUSIVE, "reasons": reasons}
    st = fixed_stat(h)
    lo, hi = st["interval90_pp"]
    near = [name for name, end in (("lower end vs 0", lo), ("upper end vs 0", hi)) if abs(end) <= RULE8_EPS]
    above = lo > 0.0 and "lower end vs 0" not in near
    below = hi < 0.0 and "upper end vs 0" not in near
    seed = rect_stat(h)
    return {"outcome": GAIN if above else LOSS if below else FLAT, "near_boundary": near, **st,
            "reported_lineages_as_a_sample": {k: seed[k] for k in (
                "interval90_pp", "se_pp", "df", "row_means_pp", "col_means_pp", "floor_rows_bound", "floor_cols_bound")}}


def next_action(k: int, outcome: Optional[str], floor_fired: bool) -> Dict[str, Any]:
    """What follows round ``k`` (1..4). ``outcome`` is the round cross's class, or None when the floor fired (the
    cross of a floored round is not played)."""
    if k not in range(1, MAX_ROUNDS + 1):
        raise ValueError(f"round {k!r} is not 1..{MAX_ROUNDS}")
    if floor_fired:
        return {"action": STOP, "arm_label": ENT_FLOOR, "read_round": k - 1}
    if outcome == LOSS:
        return {"action": STOP, "arm_label": DOSE_HARM, "read_round": k - 1}
    if outcome == FLAT:
        return {"action": STOP, "arm_label": PLATEAU, "read_round": k}
    if outcome == GAIN:
        if k < MAX_ROUNDS:
            return {"action": CONTINUE, "arm_label": None, "read_round": None, "next_round": k + 1}
        return {"action": STOP, "arm_label": CAP, "read_round": k}
    return {"action": HOLD, "arm_label": None, "read_round": None,
            "reason": f"the cross of round {k} is {outcome!r}: nothing is launched, the orchestrator rules"}


def swap_end_entropies() -> List[float]:
    """The three banked swap forks' end entropies (nats), from CE-3d's banked diagnostics; never retyped."""
    rec = json.loads((CE3D_RECORD / "diag9_ce3d.json").read_text())["summary"]["swap_end_entropy"]
    return [float(rec[f"rb_x35_ce3dswap_s{s}"]) for s in (7321, 7322, 7323)]


def floor_nats(swap_ends: Optional[Sequence[float]] = None) -> float:
    return FLOOR_FRACTION * min(swap_end_entropies() if swap_ends is None else swap_ends)


def floor(stats: Sequence[float], swap_ends: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Does the entropy floor fire on these lineages' statistics (nats; 1 to 3 of them, in lineage order)?"""
    xs = [float(x) for x in stats]
    if not 1 <= len(xs) <= N or not all(math.isfinite(x) and x >= 0.0 for x in xs):
        return {"fired": None, "reasons": [f"needs 1 to {N} finite statistics >= 0; got {xs}"]}
    bar = floor_nats(swap_ends)
    near = [i + 1 for i, x in enumerate(xs) if abs(x - bar) <= ENT_EPS_NATS]
    below = [i + 1 for i, x in enumerate(xs) if x < bar - ENT_EPS_NATS]
    return {"fired": bool(below), "below_floor": below, "within_1e-9_of_the_floor": near, "floor_nats": bar,
            "stats_nats": xs}


def self_check() -> Dict[str, Any]:
    probs: List[str] = []
    # 1. this file's rect_stat is X70's: its banked deciding matrix gives its banked verdict
    h70 = json.loads((X70_ANCHOR / "h.json").read_text())["h_pp"]
    v70 = json.loads((X70_ANCHOR / "verdict.json").read_text())
    ours = rect_stat(h70)
    for key in ("delta_hat_pp", "se_pp", "df"):
        if abs(ours[key] - v70[key]) > 1e-9:
            probs.append(f"X70's banked {key}: {ours[key]} != {v70[key]}")
    if any(abs(a - b) > 1e-9 for a, b in zip(ours["interval90_pp"], v70["interval90_pp"])):
        probs.append(f"X70's banked interval: {ours['interval90_pp']} != {v70['interval90_pp']}")
    # 2. hand cases on 3 x 3. Deciding: SE = sqrt(0.3125 / 9) = 0.18634, half-width 1.64485 x SE = 0.30650 pp.
    #    Reported: floors 0.3125 / 3 each side when the means agree, V = 2 * 0.3125 / 9, df 4, t 2.1318, 0.5618 pp.
    from statistics import NormalDist

    flat = [[50.0] * 3 for _ in range(3)]
    half = fixed_stat(flat)["interval90_pp"][1]
    if abs(Z_095 - NormalDist().inv_cdf(0.95)) > 1e-15 or abs(half - 0.30650) > 5e-6:
        probs.append(f"the deciding half-width is {half}, not 0.30650 pp")
    half_seed = rect_stat(flat)["interval90_pp"][1]
    if abs(half_seed - t_ppf(0.95, 4.0) * math.sqrt(2 * CELL_VAR_PP2 / 9)) > 1e-12 or abs(half_seed - 0.5618) > 5e-4:
        probs.append(f"the reported floored half-width is {half_seed}, not 0.5618 pp")
    apart = decide_round([[52.0] * 3, [50.5] * 3, [49.6] * 3])
    if apart["outcome"] != GAIN or apart["reported_lineages_as_a_sample"]["interval90_pp"][0] >= 0.0:
        probs.append(f"the fixed-finals class and the lineages-as-a-sample interval should differ here: {apart}")
    cases = {
        "all 50: FLAT": (flat, FLAT),
        "all 51: GAIN": ([[51.0] * 3 for _ in range(3)], GAIN),
        "all 49: LOSS": ([[49.0] * 3 for _ in range(3)], LOSS),
        "all 50.3 (inside the half-width): FLAT": ([[50.3] * 3 for _ in range(3)], FLAT),
        "all 50.31: GAIN": ([[50.31] * 3 for _ in range(3)], GAIN),
        "one lineage carries it (rows 52 / 50.5 / 49.6): GAIN": ([[52.0] * 3, [50.5] * 3, [49.6] * 3], GAIN),
        "rule 8, lower end exactly 0: FLAT": ([[50.0 + half] * 3 for _ in range(3)], FLAT),
        "rule 8, upper end exactly 0: FLAT": ([[50.0 - half] * 3 for _ in range(3)], FLAT),
        "a spread row: FLAT": ([[53.0, 53.0, 53.0], [50.0, 50.0, 50.0], [47.5, 47.5, 47.5]], FLAT),
        "2 x 3: INCONCLUSIVE": ([[51.0] * 3 for _ in range(2)], INCONCLUSIVE),
        "nan: INCONCLUSIVE": ([[float("nan")] * 3 for _ in range(3)], INCONCLUSIVE),
    }
    got = {}
    for name, (h, want) in cases.items():
        d = decide_round(h)
        got[name] = d["outcome"]
        if d["outcome"] != want or (name.startswith("rule 8") and not d["near_boundary"]):
            probs.append(f"{name}: {d}")
    if decide_round(flat, ["a cell is short"])["outcome"] != INCONCLUSIVE:
        probs.append("a caller's inconclusive reason was ignored")
    # 3. the action table, every (round, input)
    table = {}
    for k in range(1, MAX_ROUNDS + 1):
        for name, (oc, fl) in {"floor": (None, True), "floor+GAIN": (GAIN, True), LOSS: (LOSS, False),
                               FLAT: (FLAT, False), GAIN: (GAIN, False), INCONCLUSIVE: (INCONCLUSIVE, False)}.items():
            a = next_action(k, oc, fl)
            table[f"round {k}, {name}"] = [a["action"], a["arm_label"], a["read_round"]]
    want_table = {}
    for k in range(1, MAX_ROUNDS + 1):
        want_table[f"round {k}, floor"] = [STOP, ENT_FLOOR, k - 1]
        want_table[f"round {k}, floor+GAIN"] = [STOP, ENT_FLOOR, k - 1]
        want_table[f"round {k}, {LOSS}"] = [STOP, DOSE_HARM, k - 1]
        want_table[f"round {k}, {FLAT}"] = [STOP, PLATEAU, k]
        want_table[f"round {k}, {GAIN}"] = [CONTINUE, None, None] if k < MAX_ROUNDS else [STOP, CAP, k]
        want_table[f"round {k}, {INCONCLUSIVE}"] = [HOLD, None, None]
    if table != want_table:
        probs.append(f"the action table: {table}")
    # 4. the floor, from the banked record
    sw = swap_end_entropies()
    bar = floor_nats()
    if abs(bar - 0.5 * 0.4837687760591507) > 1e-15 or min(sw) != sw[0]:
        probs.append(f"the floor is {bar} from {sw}")
    fl_cases = {"the banked swap ends": (sw, False, []), "one lineage at 0.2": ([0.45, 0.2, 0.45], True, [2]),
                "one run alone at 0.24": ([0.24], True, [1]), "rule 8, exactly the floor": ([bar, 0.4, 0.4], False, []),
                "rule 8, 5e-10 under": ([bar - 5e-10, 0.4, 0.4], False, []), "2e-9 under": ([bar - 2e-9], True, [1])}
    got_fl = {}
    for name, (xs, want, who) in fl_cases.items():
        f = floor(xs)
        got_fl[name] = [f["fired"], f["below_floor"]]
        if f["fired"] is not want or f["below_floor"] != who or (name.startswith("rule 8") and not f["within_1e-9_of_the_floor"]):
            probs.append(f"floor {name}: {f}")
    if floor([])["fired"] is not None or floor([float("nan")])["fired"] is not None or floor([0.4] * 4)["fired"] is not None:
        probs.append("the floor read no value, a nan or four")
    return {"self_check": "OK" if not probs else "MISMATCH", "problems": probs,
            "x70_banked": {"delta_hat_pp": ours["delta_hat_pp"], "interval90_pp": ours["interval90_pp"], "df": ours["df"],
                           "outcome_there": v70["outcome"]},
            "deciding_half_width_pp": half, "reported_floored_half_width_pp": half_seed, "cases": got, "action_table": table, "floor_cases": got_fl,
            "swap_end_entropy_banked": sw, "floor_nats": bar}


# ----------------------------------------------------------------------------------------------- operating characteristics
#: one look: (sigma_L, sigma_F, tau) in pp. sigma_L = a lineage's persistent strength (shared by a final and its
#: ancestor, so it CANCELS between row i and column i); sigma_F = one round's fresh fork-to-fork spread; tau = a
#: final-by-final interaction that does not shrink with pairs.
LOOK_SCENARIOS = ((0.0, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.4, 0.0), (0.0, 0.5, 0.0), (0.0, 0.5, 0.3), (0.5, 0.3, 0.0),
                  (0.5, 0.5, 0.0), (1.0, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 1.76, 0.0))
DELTAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
#: the chain: per-round true gains (pp), round 1..4
GAIN_PATHS = {"no gain": (0.0, 0.0, 0.0, 0.0), "harm from round 2": (0.0, -1.0, -1.0, -1.0),
              "0.5 then nothing": (0.5, 0.0, 0.0, 0.0), "1 then nothing": (1.0, 0.0, 0.0, 0.0),
              "1, 0.5, 0.25, 0": (1.0, 0.5, 0.25, 0.0), "2, 1, 0.5, 0.25": (2.0, 1.0, 0.5, 0.25),
              "0.5 every round": (0.5, 0.5, 0.5, 0.5), "1 every round": (1.0, 1.0, 1.0, 1.0),
              "gain, then harm": (1.0, -1.0, -1.0, -1.0)}
CHAIN_SCENARIOS = ((0.0, 0.3, 0.0), (0.5, 0.5, 0.0), (0.0, 0.5, 0.3))
SIG_E_1000 = 1.05                            # the cell noise banked 1,000-pair crosses measured (X70's figure)


HALF_FIXED = Z_095 * math.sqrt(CELL_VAR_PP2 / (N * N))   # 0.3065 pp: the deciding half-width


def _classes(base: Any) -> Tuple[Any, Any, Any]:
    """(mean, the REPORTED lineages-as-a-sample half-width, its df) per replicate of ``base`` (reps x 3 x 3)."""
    import numpy as np
    from scipy.stats import t as student_t

    s2r = np.maximum(base.mean(axis=2).var(axis=1, ddof=1), CELL_VAR_PP2 / N)
    s2c = np.maximum(base.mean(axis=1).var(axis=1, ddof=1), CELL_VAR_PP2 / N)
    tr, tc = s2r / N, s2c / N
    v = tr + tc
    df = v * v / (tr * tr / (N - 1) + tc * tc / (N - 1))
    return base.mean(axis=(1, 2)), student_t.ppf(0.95, df) * np.sqrt(v), df


def simulate(reps: int = 50000, seed: int = 20261125) -> Dict[str, Any]:
    """Strength of lineage i's round-k final: G_k + a_i + u_ik, with G_k the cumulative true gain, a ~ N(0, sig_L^2)
    persistent, u ~ N(0, sig_F^2) fresh each round (u_i0 is the banked final's). Cell: h_ij = 50 + (strength of row i
    at k) - (strength of column j at k-1) + e, e ~ N(0, 1.05^2 * 1000 / pairs + tau^2). Score is taken linear in
    strength (X70's model; within a few pp of 50 the approximation is its own).

    The seed SDs: X70's simulation holds the provenance (the swap forks' row means varied by less than their
    binomial floor in three banked crosses, so 0.3-0.4 pp is the central case; 0.5 the fork evidence's upper figure;
    1.76 pp the FRESH-run figure no fork has shown). A LONGER fork's spread is unmeasured; sigma_F runs to 1.0."""
    import numpy as np

    rng = np.random.default_rng(seed)
    sig_cell = lambda tau: math.sqrt(SIG_E_1000 ** 2 * 1000.0 / PAIRS + tau ** 2)  # noqa: E731
    looks = []
    for sig_l, sig_f, tau in LOOK_SCENARIOS:
        a = rng.normal(0, sig_l, (reps, N))
        row = a + rng.normal(0, sig_f, (reps, N))
        col = a + rng.normal(0, sig_f, (reps, N))
        base = row[:, :, None] - col[:, None, :] + rng.normal(0, sig_cell(tau), (reps, N, N))
        m, hw, _df = _classes(base)
        looks.append({"sigma_L": sig_l, "sigma_F": sig_f, "tau": tau,
                      "p_gain": [round(float(((m + d - HALF_FIXED) > 0).mean()), 3) for d in DELTAS],
                      "p_loss_at": {f"{d:+g}": round(float(((m + d + HALF_FIXED) < 0).mean()), 3)
                                    for d in (0.0, -0.5, -1.0, -2.0)},
                      "reported_median_half_width_pp": round(float(np.median(hw)), 2),
                      "reported_p_lower_above_0": [round(float(((m + d - hw) > 0).mean()), 3) for d in DELTAS]})
    chains = []
    for sig_l, sig_f, tau in CHAIN_SCENARIOS:
        for name, gains in GAIN_PATHS.items():
            a = rng.normal(0, sig_l, (reps, N))
            prev = a + rng.normal(0, sig_f, (reps, N))
            alive = np.ones(reps, dtype=bool)
            stop_round = np.zeros(reps, dtype=int)
            label = np.zeros(reps, dtype=int)                 # 1 PLATEAU, 2 DOSE-HARM, 3 CAP
            for k, g in enumerate(gains, start=1):
                cur = a + rng.normal(0, sig_f, (reps, N))
                base = cur[:, :, None] - prev[:, None, :] + rng.normal(0, sig_cell(tau), (reps, N, N))
                m = base.mean(axis=(1, 2))
                gain, loss = (m + g - HALF_FIXED) > 0, (m + g + HALF_FIXED) < 0
                ends = alive & (~gain | (k == MAX_ROUNDS))
                stop_round[ends] = k
                label[ends & ~gain & ~loss] = 1
                label[ends & loss] = 2
                label[ends & gain] = 3
                alive &= gain & (k < MAX_ROUNDS)
                prev = cur
            chains.append({"sigma_L": sig_l, "sigma_F": sig_f, "tau": tau, "path": name, "gains_pp": list(gains),
                           "p_stop_after_round": [round(float((stop_round == k).mean()), 3) for k in range(1, MAX_ROUNDS + 1)],
                           "p_plateau": round(float((label == 1).mean()), 3), "p_dose_harm": round(float((label == 2).mean()), 3),
                           "p_cap": round(float((label == 3).mean()), 3),
                           "mean_rounds": round(float(stop_round.mean()), 2),
                           "mean_training_gpu_h": round(float(stop_round.mean()) * GPU_H_A_ROUND, 1)})
    return {"reps": reps, "seed": seed, "pairs": PAIRS, "sigma_e_at_1000_pairs": SIG_E_1000, "deltas_pp": list(DELTAS),
            "looks": looks, "chains": chains}


def _print_sim(res: Dict[str, Any]) -> None:
    print(f"ONE ROUND CROSS: 3 round-k finals x 3 round-(k-1) finals, {res['pairs']} pairs a cell; reps {res['reps']}, rng "
          f"{res['seed']}; cell noise SD {res['sigma_e_at_1000_pairs']} pp at 1,000 pairs (x sqrt(1000 / pairs)), plus tau")
    print(f"THE DECIDING CLASS (fixed finals, half-width {HALF_FIXED:.4f} pp). The 'true round gain' is the RECIPE's mean "
          "gain a round; a fork's own luck (sig_F) is on top of it and is real strength of that file.")
    print("sig_L  sig_F   tau  P(GAIN) at a true round gain of " + " / ".join(f"{d:+g}" for d in res["deltas_pp"])
          + "   P(LOSS) at 0 / -0.5 / -1 / -2")
    for s in res["looks"]:
        print(f"{s['sigma_L']:>5.2f}  {s['sigma_F']:>5.2f}  {s['tau']:>4.2f}  "
              + " / ".join(f"{p:.3f}" for p in s["p_gain"]) + "   " + " / ".join(f"{p:.3f}" for p in s["p_loss_at"].values()))
    print("\nTHE REPORTED INTERVAL (the lineages as a sample of forks; X70's rect_stat), same crosses")
    print("sig_L  sig_F   tau  median half-width  P(lower end > 0) at " + " / ".join(f"{d:+g}" for d in res["deltas_pp"]))
    for s in res["looks"]:
        print(f"{s['sigma_L']:>5.2f}  {s['sigma_F']:>5.2f}  {s['tau']:>4.2f}  {s['reported_median_half_width_pp']:>17.2f}  "
              + " / ".join(f"{p:.3f}" for p in s["reported_p_lower_above_0"]))
    print(f"\nTHE CHAIN of at most {MAX_ROUNDS} rounds (CONTINUE iff GAIN), {GPU_H_A_ROUND} GPU-h of training a round")
    print("sig_L  sig_F   tau  path (true gain a round, pp)   P(stop after round 1 / 2 / 3 / 4)   P(PLATEAU-AT-REACH)  "
          "P(DOSE-HARM)  P(CAP)  mean rounds  mean GPU-h")
    for c in res["chains"]:
        print(f"{c['sigma_L']:>5.2f}  {c['sigma_F']:>5.2f}  {c['tau']:>4.2f}  {c['path']:<28} "
              + " / ".join(f"{p:.3f}" for p in c["p_stop_after_round"])
              + f"   {c['p_plateau']:.3f}  {c['p_dose_harm']:.3f}  {c['p_cap']:.3f}  {c['mean_rounds']:.2f}  {c['mean_training_gpu_h']:.1f}")


def main(argv: Sequence[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "self-check"
    if cmd == "self-check":
        res = self_check()
        print(json.dumps(res, indent=1))
        return 0 if res["self_check"] == "OK" else 1
    if cmd == "simulate":
        _print_sim(simulate())
        return 0
    if cmd == "decide-round" and len(argv) == 4:
        k = int(argv[2])
        d = decide_round(json.loads(Path(argv[3]).read_text())["h_pp"])
        print(json.dumps({"round": k, **d, "next": next_action(k, d["outcome"], False)}, indent=1))
        return 0
    if cmd == "floor" and 3 <= len(argv) <= 2 + N:
        res = floor([float(x) for x in argv[2:]])
        print(json.dumps(res, indent=1))
        return 0 if res["fired"] is not None else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
