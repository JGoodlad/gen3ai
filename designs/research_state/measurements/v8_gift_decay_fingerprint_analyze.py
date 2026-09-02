"""M9b analysis — the THREE delta vectors, the four registered bars, and the axis tables.

Input : the ``*.jsonl.gz`` decision rows + ``*_cells.jsonl`` written by
        ``v8_gift_decay_fingerprint_probe.py``.
Output: ``v8_gift_decay_fingerprint_2026-09-01.json`` (+ the tables markdown).

THE UNIT OF MEASUREMENT is a decision on an IDENTICAL BOARD. Every row carries all three arms'
argmax on the same ``(obs, mask)``, so every delta below is a PAIRED difference over the same
rows — never a comparison of three different games. Rows come from all three arms' trajectories,
so a pooled delta is measured over a mixture of the three state distributions; the per-acting-arm
split is reported beside it, because an axis whose sign flips between them is measuring the state
shift rather than the policy change.

CIs are a cluster bootstrap over TEAMS — the unit probe P's, M4's and the timing probe's claims
generalise over, and the unit this one must match to be comparable to them. A cosine's CI
resamples teams ONCE and recomputes BOTH vectors from that resample, so the pairing between the
two vectors is preserved wherever they live on the same team set.

Run:
  python designs/research_state/measurements/v8_gift_decay_fingerprint_analyze.py \
      --rows '/tmp/m9b/rows_untaught_*.jsonl.gz' --cells '/tmp/m9b/rows_untaught_*_cells.jsonl'
  (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The AXES are IMPORTED from M4's analyzer, not re-typed: a drifted basis would make every
# cosine here incomparable to M4's published vectors, which is the point of reusing the
# instrument.
from v8_fold_behavioral_fingerprint_analyze import AXES, AXIS_NAMES, CLASSES  # noqa: E402

ARMS = ("parent", "peak", "final")
PAIRS = (("parent", "peak"), ("peak", "final"), ("parent", "final"))


def pair_name(a: str, b: str) -> str:
    return f"{a}->{b}"


def load_rows(patterns: list[str]) -> list[dict]:
    """Read the decision rows, TOLERATING a truncated tail (the producer flushes with
    ``Z_SYNC_FLUSH`` per cell, so a live file is readable up to the last flush)."""
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            n0 = len(rows)
            try:
                with gzip.open(path, "rt") as f:
                    for ln in f:
                        ln = ln.strip()
                        if ln:
                            try:
                                rows.append(json.loads(ln))
                            except json.JSONDecodeError:
                                pass  # a half-written final line
            except (EOFError, OSError) as e:
                print(f"[m9b] {path}: truncated stream ({type(e).__name__}) — kept "
                      f"{len(rows) - n0} rows", flush=True)
    return rows


def battle_key(r) -> tuple:
    """The identity of the battle a row belongs to. NOT ``r['tag']`` alone — the bridge numbers
    battles from 1 PER PROCESS, so two shards emit identical tags for different battles (M4
    caveat 10). ``(team, opp, arm)`` is unique to one shard because shards partition the team
    list, so prefixing it makes the tag unique again."""
    return (r["team"], r["opp"], r["arm"], r["tag"])


# ---------------------------------------------------------------------------------------------
# The per-team, per-axis, per-arm accumulator — ONE pass over the rows, everything downstream
# reads it. (A per-axis re-scan would be 25 passes, and the reliability control calls this
# hundreds of times.)
# ---------------------------------------------------------------------------------------------
def team_axis_arm(rows: list[dict], names: list[str] | None = None):
    """Returns ``(teams, S[team, axis, arm], N[team, axis])``.

    ``S`` sums the axis indicator evaluated at each arm's argmax over the rows that pass the
    axis filter; ``N`` counts those rows. The filter never depends on which arm chose — that is
    what keeps every difference paired.
    """
    names = names or AXIS_NAMES
    idx = {a[0]: a for a in AXES}
    specs = [idx[nm] for nm in names]
    acc: dict[str, np.ndarray] = {}
    cnt: dict[str, np.ndarray] = {}
    for r in rows:
        t = r["team"]
        a = acc.get(t)
        if a is None:
            a = acc[t] = np.zeros((len(specs), len(ARMS)))
            cnt[t] = np.zeros(len(specs))
        ia = r["idx"]
        for j, (_nm, filt, ind, _g) in enumerate(specs):
            if not filt(r):
                continue
            cnt[t][j] += 1
            for k, arm in enumerate(ARMS):
                a[j, k] += ind(r, ia[arm])
    teams = sorted(acc)
    S = np.array([acc[t] for t in teams]) if teams else np.zeros((0, len(specs), len(ARMS)))
    N = np.array([cnt[t] for t in teams]) if teams else np.zeros((0, len(specs)))
    return teams, S, N


def delta_matrix(S: np.ndarray, N: np.ndarray, a: str, b: str) -> np.ndarray:
    """Per-team delta ``rate(b) − rate(a)`` for every axis; NaN where a team has no rows."""
    ia, ib = ARMS.index(a), ARMS.index(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (S[:, :, ib] - S[:, :, ia]) / np.where(N > 0, N, np.nan)


def vec(S: np.ndarray, N: np.ndarray, a: str, b: str, rowsel=None) -> np.ndarray:
    """The equal-weight-over-teams delta vector (M4's / probe P's convention)."""
    d = delta_matrix(S, N, a, b)
    if rowsel is not None:
        d = d[rowsel]
    if d.size == 0:
        return np.zeros(N.shape[1])
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.nanmean(d, axis=0))


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else float("nan")


def split_half_reliability(S, N, a, b, draws=300, seed=5) -> float:
    """How well does this delta vector correlate WITH ITSELF across a random team split?

    The NOISE CEILING any cross-vector cosine can reach: two noisy vectors agree less than one,
    so a cosine only means 'a different shape' when the vectors are individually reliable. This
    programme has already been bitten once by omitting the control (a −0.020 that became +0.206).
    """
    d = delta_matrix(S, N, a, b)
    n = d.shape[0]
    if n < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    cos = []
    for _ in range(draws):
        perm = rng.permutation(n)
        h = n // 2
        with np.errstate(invalid="ignore"):
            va = np.nan_to_num(np.nanmean(d[perm[:h]], axis=0))
            vb = np.nan_to_num(np.nanmean(d[perm[h:]], axis=0))
        c = cosine(va, vb)
        if c == c:
            cos.append(c)
    return float(np.mean(cos)) if cos else float("nan")


def cosine_block(SA, NA, pa, SB, NB, pb, same_teams: bool, reps=4000, seed=907,
                 label: str = "") -> dict:
    """One cosine between delta vector ``pa`` (on slice A) and ``pb`` (on slice B), with:

      - a cluster-bootstrap-over-TEAMS 95% interval (the registered verdict instrument),
      - a permutation null on the axis labels, BOTH tails reported,
      - each vector's split-half reliability and the ceiling ``sqrt(r_a·r_b)`` they imply.

    ``same_teams`` preserves the pairing: when both vectors live on the same team set the
    bootstrap resamples that set ONCE and recomputes both from it. When they live on different
    slices (taught vs untaught) the two sets are resampled independently, which is the honest
    thing to do and gives a wider interval.
    """
    u = vec(SA, NA, *pa)
    v = vec(SB, NB, *pb)
    obs = cosine(u, v)
    rng = np.random.default_rng(seed)

    nA, nB = NA.shape[0], NB.shape[0]
    draws = np.empty(reps)
    for k in range(reps):
        ia = rng.integers(0, nA, nA)
        ib = ia if (same_teams and nA == nB) else rng.integers(0, nB, nB)
        draws[k] = cosine(vec(SA, NA, *pa, rowsel=ia), vec(SB, NB, *pb, rowsel=ib))
    draws = draws[~np.isnan(draws)]
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) \
        if draws.size else (float("nan"), float("nan"))

    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    prng = np.random.default_rng(seed + 1)
    null = np.array([cosine(prng.permutation(u), v) for _ in range(reps)])
    ra = split_half_reliability(SA, NA, *pa)
    rb = split_half_reliability(SB, NB, *pb)
    ceil = math.sqrt(ra * rb) if (ra == ra and rb == rb and ra > 0 and rb > 0) else float("nan")
    return {
        "label": label,
        "a": pair_name(*pa), "b": pair_name(*pb),
        "cosine": round(obs, 4),
        "boot_ci95": [round(lo, 4), round(hi, 4)],
        "boot_paired_teams": bool(same_teams),
        "perm_p_right": round(float((null >= obs).mean()), 4),
        "perm_p_left": round(float((null <= obs).mean()), 4),
        "perm_null_mean": round(float(null.mean()), 4),
        "sign_agreement": f"{int(sum(1 for x, y in zip(u, v) if x * y > 0))}/{len(u)}",
        "norm_a": round(float(nu), 5), "norm_b": round(float(nv), 5),
        "norm_ratio_b_over_a": round(float(nv / nu), 4) if nu > 0 else None,
        "reliability_a": round(ra, 4) if ra == ra else None,
        "reliability_b": round(rb, 4) if rb == rb else None,
        "noise_ceiling": round(ceil, 4) if ceil == ceil else None,
        "cosine_over_ceiling": round(obs / ceil, 4) if (ceil == ceil and ceil > 0) else None,
        "disattenuated_cosine": round(obs / ceil, 4) if (ceil == ceil and ceil > 0) else None,
    }


def shared_endpoint_null(S, N, reps=4000, seed=515) -> dict:
    """THE NULL THE REGISTERED B1 BAR DOES NOT CONTAIN — and it matters.

    ``peak->final`` is not an independent vector: by construction
    ``D = (parent->final) - (parent->peak) = B - A``. So ``cos(A, D)`` carries an ARITHMETIC
    component: if B were unrelated to A with a similar norm, ``E[cos(A, B-A)] ~ -|A|/|B-A| ~
    -0.71``. A negative cosine is therefore the DEFAULT, not evidence, and B1's "CI below zero"
    rule would fire under H3 just as readily as under H1.

    The honest null keeps the shared endpoint and destroys only the axis-level correspondence:
    permute B's axis labels, recompute ``cos(A, perm(B) - A)``. The observed value is evidence
    for a REVERSAL only if it sits below that distribution.

    The non-vacuous restatement of H1 is reported beside it: pure unlearning means the ENDPOINT
    lies back along the ascent direction, i.e. ``cos(A, B)`` high with ``|B| < |A|``.
    """
    A = vec(S, N, "parent", "peak")
    B = vec(S, N, "parent", "final")
    D = vec(S, N, "peak", "final")
    obs = cosine(A, D)
    rng = np.random.default_rng(seed)
    null = np.array([cosine(A, rng.permutation(B) - A) for _ in range(reps)])
    # THE CRISP DECOMPOSITION. Project the decay onto the ascent: D = k·A + resid.
    #   pure unlearning  => resid ~ 0 and k in (-1, 0)  (the endpoint is a SHRUNK ascent)
    #   a different move => most of D's energy is in resid
    aa = float(A @ A)
    k = float(A @ D) / aa if aa > 0 else float("nan")
    resid = D - k * A
    nd = float(np.linalg.norm(D))
    along = (abs(k) * float(np.linalg.norm(A)) / nd) ** 2 if nd > 0 else float("nan")
    return {
        "decay_projected_on_ascent_k": round(k, 4) if k == k else None,
        "decay_energy_ALONG_ascent": round(along, 4) if along == along else None,
        "decay_energy_ORTHOGONAL_to_ascent": round(1 - along, 4) if along == along else None,
        "residual_norm": round(float(np.linalg.norm(resid)), 5),
        "observed_cos_A_D": round(obs, 4),
        "arith_null_mean": round(float(null.mean()), 4),
        "arith_null_ci95": [round(float(np.percentile(null, 2.5)), 4),
                            round(float(np.percentile(null, 97.5)), 4)],
        "p_left_vs_arith_null": round(float((null <= obs).mean()), 4),
        "p_right_vs_arith_null": round(float((null >= obs).mean()), 4),
        "cos_A_B_endpoint_along_ascent": round(cosine(A, B), 4),
        "norm_A_parent_to_peak": round(float(np.linalg.norm(A)), 5),
        "norm_B_parent_to_final": round(float(np.linalg.norm(B)), 5),
        "norm_D_peak_to_final": round(float(np.linalg.norm(D)), 5),
        "norm_ratio_B_over_A": round(float(np.linalg.norm(B) / np.linalg.norm(A)), 4)
        if np.linalg.norm(A) > 0 else None,
        "note": "cos(A,D) below the arith null = a REVERSAL beyond arithmetic; inside it = the "
                "decay is simply a DIFFERENT direction (H3-shaped), not an un-doing of the "
                "ascent.",
    }


def axis_table(rows: list[dict], reps: int = 4000, seed: int = 20260901) -> dict:
    """Per-axis paired delta for all three arm pairs, cluster-bootstrapped over TEAMS."""
    teams, S, N = team_axis_arm(rows)
    if not teams:
        return {}
    rng = np.random.default_rng(seed)
    boot = [rng.integers(0, len(teams), len(teams)) for _ in range(reps)]

    # per-acting-arm split (pooled over rows): does the sign survive in every arm's own
    # state distribution, or is it the state shift talking?
    split_S: dict[str, np.ndarray] = {a: np.zeros((len(AXIS_NAMES), len(ARMS))) for a in ARMS}
    split_N: dict[str, np.ndarray] = {a: np.zeros(len(AXIS_NAMES)) for a in ARMS}
    realized = {a: np.zeros(len(AXIS_NAMES)) for a in ARMS}
    realized_n = {a: np.zeros(len(AXIS_NAMES)) for a in ARMS}
    for r in rows:
        acting = r["arm"]
        ia = r["idx"]
        for j, (_nm, filt, ind, _g) in enumerate(AXES):
            if not filt(r):
                continue
            split_N[acting][j] += 1
            realized_n[acting][j] += 1
            realized[acting][j] += ind(r, r["act_idx"])
            for k, arm in enumerate(ARMS):
                split_S[acting][j, k] += ind(r, ia[arm])

    out: dict[str, list] = {}
    for a, b in PAIRS:
        d = delta_matrix(S, N, a, b)
        ia_, ib_ = ARMS.index(a), ARMS.index(b)
        rows_out = []
        for j, name in enumerate(AXIS_NAMES):
            n_tot = float(N[:, j].sum())
            if n_tot < 30:
                rows_out.append({"axis": name, "n": int(n_tot), "skipped": "n<30"})
                continue
            live = N[:, j] > 0
            point = float(np.nanmean(d[live, j]))
            base = float(np.nanmean(S[live, j, ia_] / N[live, j]))
            bd = np.array([float(np.nanmean(d[bi, j][N[bi, j] > 0])) if (N[bi, j] > 0).any()
                           else np.nan for bi in boot])
            bd = bd[~np.isnan(bd)]
            lo, hi = float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))
            sd = float(bd.std(ddof=1)) if bd.size > 2 else float("nan")
            asplit = {}
            for acting in ARMS:
                na = split_N[acting][j]
                if na >= 20:
                    asplit[acting] = round(float((split_S[acting][j, ib_]
                                                  - split_S[acting][j, ia_]) / na), 5)
            rlz = None
            if realized_n[a][j] > 20 and realized_n[b][j] > 20:
                rlz = round(float(realized[b][j] / realized_n[b][j]
                                  - realized[a][j] / realized_n[a][j]), 5)
            rows_out.append({
                "axis": name, "group": dict((x[0], x[3]) for x in AXES)[name],
                "n": int(n_tot), "n_teams": int(live.sum()),
                "base_rate_a": round(base, 5), "delta": round(point, 5),
                "ci": [round(lo, 5), round(hi, 5)],
                "z": round(point / sd, 3) if sd and sd > 0 else None,
                "acting_arm_split": asplit,
                "realized_delta": rlz,
            })
        out[pair_name(a, b)] = rows_out
    return out


def divergence(rows: list[dict]) -> dict:
    """How much each arm pair differs at all, and where the flip mass sits."""
    out = {}
    for a, b in PAIRS:
        dis = 0
        n = 0
        kl = []
        dv = []
        flips: Counter = Counter()
        both_sw = same_sw = both_mv = same_mv = 0
        for r in rows:
            ai, bi = r["idx"][a], r["idx"][b]
            n += 1
            if ai != bi:
                dis += 1
                flips[(r["cls"][ai] or "?", r["cls"][bi] or "?")] += 1
            if "p" in r:
                p = np.asarray(r["p"][b])
                q = np.asarray(r["p"][a])
                m = (p > 0) & (q > 0)
                if m.any():
                    kl.append(float((p[m] * np.log(p[m] / q[m])).sum()))
            if "v" in r:
                dv.append(abs(r["v"][b] - r["v"][a]))
            if ai < 6 and bi < 6:
                both_sw += 1
                same_sw += int(ai == bi)
            elif 6 <= ai < 10 and 6 <= bi < 10:
                both_mv += 1
                same_mv += int(ai == bi)
        tot = sum(flips.values()) or 1
        sw = sum(v for (x, y), v in flips.items() if "SWITCH" in (x, y))
        per_class = {}
        for c in CLASSES:
            na = sum(1 for r in rows if r["cls"][r["idx"][a]] == c)
            nb = sum(1 for r in rows if r["cls"][r["idx"][b]] == c)
            if na == 0 and nb == 0:
                continue
            of = sum(v for (x, _y), v in flips.items() if x == c)
            per_class[c] = {"a_n": na, "b_n": nb, "net": nb - na,
                            "flip_out_rate": round(of / na, 4) if na else None}
        out[pair_name(a, b)] = {
            "n_decisions": n,
            "argmax_disagreement": round(dis / max(n, 1), 5),
            "mean_KL_b_from_a": round(float(np.mean(kl)), 5) if kl else None,
            "mean_abs_dV": round(float(np.mean(dv)), 5) if dv else None,
            "flip_mass_involving_SWITCH": round(sw / tot, 5),
            "both_switch_n": both_sw,
            "same_switch_target_rate": round(same_sw / both_sw, 4) if both_sw else None,
            "both_move_n": both_mv,
            "same_move_slot_rate": round(same_mv / both_mv, 4) if both_mv else None,
            "per_class_flow": per_class,
            "top_flip_transitions": [{"transition": f"{x}->{y}", "n": v,
                                      "share": round(v / tot, 4)}
                                     for (x, y), v in flips.most_common(10)],
        }
    # realized game length per acting arm
    per_tag: Counter = Counter()
    tag_arm = {}
    for r in rows:
        k = battle_key(r)
        per_tag[k] += 1
        tag_arm[k] = r["arm"]
    length = {}
    for arm in ARMS:
        v = [c for t, c in per_tag.items() if tag_arm[t] == arm]
        if v:
            length[arm] = {"battles": len(v), "mean_decisions": round(float(np.mean(v)), 2)}
    out["battle_length_by_acting_arm"] = length
    return out


def read_cells(patterns: list[str]) -> list[dict]:
    raw = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            for ln in open(path):
                try:
                    raw.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    best = {}
    for r in raw:
        best[(r["kind"], r["team"], r["opp"], r["arm"])] = r
    return list(best.values())


def win_rates(cells: list[dict], reps: int = 4000, seed: int = 31) -> dict:
    """Per-arm win rate and the paired per-team gains, cluster-bootstrapped over teams.

    This is the win-rate reproduction of the timing probe's curve on M4's 4-game CRN prefix; it
    is NOT an independent re-measurement (the battles are a subsample of the same ones).
    """
    by = defaultdict(lambda: [0, 0])
    for c in cells:
        k = (c["team"], c["arm"])
        by[k][0] += c["wins"]
        by[k][1] += c["finished"]
    teams = sorted({t for t, _a in by})
    per_team = {}
    for t in teams:
        per_team[t] = {a: (by[(t, a)][0] / by[(t, a)][1] if by[(t, a)][1] else float("nan"))
                       for a in ARMS if (t, a) in by}
    rng = np.random.default_rng(seed)
    out = {"n_teams": len(teams), "per_team": per_team,
           "pooled": {a: round(sum(by[(t, a)][0] for t in teams if (t, a) in by)
                               / max(sum(by[(t, a)][1] for t in teams if (t, a) in by), 1), 5)
                      for a in ARMS},
           "gains": {}}
    for a, b in PAIRS:
        g = np.array([per_team[t][b] - per_team[t][a] for t in teams
                      if a in per_team[t] and b in per_team[t]])
        if g.size < 3:
            continue
        pt = float(np.mean(g))
        bd = np.array([float(np.mean(g[rng.integers(0, g.size, g.size)])) for _ in range(reps)])
        out["gains"][pair_name(a, b)] = {
            "mean_pp": round(pt * 100, 3),
            "ci95_pp": [round(float(np.percentile(bd, 2.5)) * 100, 3),
                        round(float(np.percentile(bd, 97.5)) * 100, 3)],
            "z": round(pt / bd.std(ddof=1), 3) if bd.std(ddof=1) > 0 else None,
            "teams_positive": f"{int((g > 0).sum())}/{g.size}",
        }
    shortfall = [c for c in cells if c["finished"] != c["requested"]]
    out["cells"] = len(cells)
    out["short_cells"] = len(shortfall)
    out["short_cell_detail"] = [{"team": c["team"], "opp": c["opp"], "arm": c["arm"],
                                 "finished": c["finished"], "requested": c["requested"]}
                                for c in shortfall[:20]]
    return out


def top_z(table: list[dict], k: int = 5) -> list[dict]:
    live = [r for r in table if "delta" in r and r.get("z") is not None]
    live.sort(key=lambda r: -abs(r["z"]))
    return [{"axis": r["axis"], "delta": r["delta"], "ci": r["ci"], "z": r["z"]}
            for r in live[:k]]


def m4_reproduction(table: dict, m4_json: str, slice_key: str) -> dict:
    """`parent` and `final` are M4's own two arms, so `parent->final` here must reproduce M4's
    published vector. A low cosine means the instrument drifted and invalidates everything."""
    try:
        m4 = json.load(open(m4_json))
    except Exception as e:
        return {"unavailable": f"{type(e).__name__}: {e}"}
    fam = m4.get("v8", m4)
    ax = fam.get(slice_key, {}).get("axes") if isinstance(fam.get(slice_key), dict) \
        else fam.get(slice_key)
    if not ax:
        return {"unavailable": f"no {slice_key} axis table in {m4_json}"}
    md = {r["axis"]: r["delta"] for r in ax if "delta" in r}
    mine = {r["axis"]: r["delta"] for r in table.get("parent->final", []) if "delta" in r}
    shared = [a for a in AXIS_NAMES if a in md and a in mine]
    if len(shared) < 10:
        return {"unavailable": f"only {len(shared)} shared axes"}
    u = np.array([mine[a] for a in shared])
    v = np.array([md[a] for a in shared])
    return {"n_axes": len(shared), "cosine_vs_M4": round(cosine(u, v), 4),
            "norm_here": round(float(np.linalg.norm(u)), 5),
            "norm_M4": round(float(np.linalg.norm(v)), 5),
            "max_abs_axis_diff": round(float(np.max(np.abs(u - v))), 5)}


def render_tables(res: dict) -> str:
    """Regenerate every table in the record FROM THE JSON, so the markdown can never drift from
    the numbers it reports."""
    L = ["# M9b tables — regenerated from `v8_gift_decay_fingerprint_2026-09-01.json`", ""]
    for slc in ("untaught", "taught"):
        s = res.get(slc, {})
        if s.get("status") != "ok":
            L += [f"## {slc.upper()} — {s.get('status', 'MISSING')}", ""]
            continue
        L += [f"## {slc.upper()} — {s['n_teams']} teams · {s['n_battles']} battles · "
              f"{s['n_rows']} dual-scored decisions", ""]
        wr = s.get("win_rates", {})
        if "pooled" in wr:
            L += ["| arm | pooled WR |", "|---|---:|"]
            for arm in ARMS:
                L.append(f"| `{arm}` | {wr['pooled'][arm]:.4f} |")
            L += ["", "| contrast | mean pp | 95% cluster CI | z | teams + |", "|---|---:|---|---:|---:|"]
            for k, g in wr.get("gains", {}).items():
                L.append(f"| `{k}` | {g['mean_pp']:+.2f} | [{g['ci95_pp'][0]:+.2f}, "
                         f"{g['ci95_pp'][1]:+.2f}] | {g['z']} | {g['teams_positive']} |")
            L += ["", f"cells {wr.get('cells')} · short cells {wr.get('short_cells')}", ""]
        for pk in [pair_name(*p) for p in PAIRS]:
            tbl = s["axes"].get(pk, [])
            live = [r for r in tbl if "delta" in r]
            live.sort(key=lambda r: -abs(r.get("z") or 0))
            L += [f"### {slc} · `{pk}`", "",
                  "| axis | n | rate(a) | Δ(b−a) | 95% CI | z | realized Δ |",
                  "|---|---:|---:|---:|---|---:|---:|"]
            for r in live:
                L.append(f"| `{r['axis']}` | {r['n']} | {r['base_rate_a']:.3f} | "
                         f"**{r['delta']:+.4f}** | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | "
                         f"{r['z']} | {r['realized_delta']} |")
            d = s["divergence"].get(pk, {})
            L += ["", f"divergence: argmax disagreement **{d.get('argmax_disagreement')}** · "
                      f"mean KL(b‖a) {d.get('mean_KL_b_from_a')} · mean |ΔV| "
                      f"{d.get('mean_abs_dV')} · flip mass involving SWITCH "
                      f"{d.get('flip_mass_involving_SWITCH')} · same switch target "
                      f"{d.get('same_switch_target_rate')} (n={d.get('both_switch_n')})", ""]
    L += ["## The registered bars", "",
          "| bar | cosine | 95% cluster CI | perm p (L/R) | reliabilities | ceiling | verdict |",
          "|---|---:|---|---|---|---:|---|"]
    for k, b in res.get("bars", {}).items():
        if k.startswith("_") or "cosine" not in b:
            continue
        L.append(f"| `{k}` | {b['cosine']:+.4f} | [{b['boot_ci95'][0]:+.4f}, "
                 f"{b['boot_ci95'][1]:+.4f}] | {b['perm_p_left']} / {b['perm_p_right']} | "
                 f"{b['reliability_a']} / {b['reliability_b']} | {b['noise_ceiling']} | "
                 f"**{b['verdict']}** |")
    inf = res.get("bars", {}).get("_informational", {})
    if inf:
        L += ["", "### Informational cosines (never a registered verdict)", "",
              "| what | cosine | 95% CI | ceiling |", "|---|---:|---|---:|"]
        for k, b in inf.items():
            L.append(f"| `{k}` | {b['cosine']:+.4f} | [{b['boot_ci95'][0]:+.4f}, "
                     f"{b['boot_ci95'][1]:+.4f}] | {b['noise_ceiling']} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--untaught-rows", nargs="*", default=[])
    ap.add_argument("--untaught-cells", nargs="*", default=[])
    ap.add_argument("--taught-rows", nargs="*", default=[])
    ap.add_argument("--taught-cells", nargs="*", default=[])
    ap.add_argument("--m4-json", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "v8_fold_behavioral_fingerprint_2026-08-31.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=4000)
    a = ap.parse_args(argv)

    res: dict = {"meta": {"arms": list(ARMS), "pairs": [pair_name(*p) for p in PAIRS],
                          "n_axes": len(AXIS_NAMES), "bootstrap_reps": a.reps}}
    S = {}
    N = {}
    for slc, rp, cp in (("untaught", a.untaught_rows, a.untaught_cells),
                        ("taught", a.taught_rows, a.taught_cells)):
        if not rp:
            res[slc] = {"status": "NOT RUN"}
            continue
        rows = load_rows(rp)
        if not rows:
            res[slc] = {"status": "NO ROWS"}
            continue
        teams, s, n = team_axis_arm(rows)
        S[slc], N[slc] = s, n
        tbl = axis_table(rows, reps=a.reps)
        res[slc] = {
            "status": "ok",
            "n_rows": len(rows),
            "n_teams": len(teams),
            "n_battles": len({battle_key(r) for r in rows}),
            "axes": tbl,
            "top_z": {k: top_z(v) for k, v in tbl.items()},
            "divergence": divergence(rows),
            "win_rates": win_rates(read_cells(cp)) if cp else {"status": "no cells"},
            "m4_reproduction": m4_reproduction(tbl, a.m4_json, slc),
            "vectors": {pair_name(x, y): [round(float(z), 6)
                                          for z in vec(s, n, x, y)] for x, y in PAIRS},
            "axis_order": AXIS_NAMES,
        }

    # ---- the four registered bars -------------------------------------------------------
    bars = {}
    if "untaught" in S:
        b1 = cosine_block(S["untaught"], N["untaught"], ("parent", "peak"),
                          S["untaught"], N["untaught"], ("peak", "final"),
                          same_teams=True, reps=a.reps,
                          label="H1: is peak->final a REVERSAL of parent->peak (untaught)?")
        lo, hi = b1["boot_ci95"]
        b1["verdict"] = "PASS" if hi < 0 else ("FAIL" if lo > 0 else "UNDECIDED")
        # The registered rule is scored EXACTLY as written; the arithmetic null it does not
        # contain is reported beside it and never in place of it.
        b1["shared_endpoint_arithmetic_null"] = shared_endpoint_null(
            S["untaught"], N["untaught"], reps=a.reps)
        bars["B1_H1_overshoot"] = b1
    if "untaught" in S and "taught" in S:
        b2 = cosine_block(S["taught"], N["taught"], ("parent", "peak"),
                          S["untaught"], N["untaught"], ("peak", "final"),
                          same_teams=False, reps=a.reps,
                          label="H2: does untaught peak->final look like the TAUGHT parent->peak?")
        lo, hi = b2["boot_ci95"]
        b2["verdict"] = "PASS" if lo > 0 else ("FAIL" if hi < 0 else "UNDECIDED")
        bars["B2_H2_teacher_pull"] = b2
        b4 = cosine_block(S["taught"], N["taught"], ("parent", "peak"),
                          S["taught"], N["taught"], ("peak", "final"),
                          same_teams=True, reps=a.reps,
                          label="TS: does the TAUGHT side keep going in the teachers' direction?")
        lo, hi = b4["boot_ci95"]
        b4["verdict"] = "CONTINUES" if lo > 0 else ("REVERSES" if hi < 0 else "UNDECIDED")
        b4["shared_endpoint_arithmetic_null"] = shared_endpoint_null(
            S["taught"], N["taught"], reps=a.reps)
        bars["B4_taught_side"] = b4
    if "untaught" in S:
        t1 = {r["axis"] for r in res["untaught"]["top_z"]["parent->peak"][:3]}
        t2 = {r["axis"] for r in res["untaught"]["top_z"]["peak->final"][:3]}
        named_pass = any(bars.get(k, {}).get("verdict") == "PASS"
                         for k in ("B1_H1_overshoot", "B2_H2_teacher_pull"))
        disjoint = not (t1 & t2)
        bars["B3_H3_something_else"] = {
            "label": "H3: neither named account fires AND a distinct axis set moves",
            "top3_parent_to_peak": sorted(t1), "top3_peak_to_final": sorted(t2),
            "intersection": sorted(t1 & t2),
            "a_named_account_passed": bool(named_pass),
            "verdict": "PASS" if (not named_pass and disjoint) else "FAIL",
        }
        # informational companions, never a registered verdict
        bars["_informational"] = {
            "cos_untaught_parent_peak_vs_untaught_parent_final": cosine_block(
                S["untaught"], N["untaught"], ("parent", "peak"),
                S["untaught"], N["untaught"], ("parent", "final"),
                same_teams=True, reps=a.reps, label="how much of the peak survives to the end"),
        }
        if "taught" in S:
            bars["_informational"]["cos_taught_pf_vs_untaught_pf"] = cosine_block(
                S["taught"], N["taught"], ("parent", "final"),
                S["untaught"], N["untaught"], ("parent", "final"),
                same_teams=False, reps=a.reps,
                label="M4's headline contrast, recomputed here")
            bars["_informational"]["cos_taught_peak_final_vs_untaught_peak_final"] = \
                cosine_block(S["taught"], N["taught"], ("peak", "final"),
                             S["untaught"], N["untaught"], ("peak", "final"),
                             same_teams=False, reps=a.reps,
                             label="is the DECAY the same change on both slices?")
            # B2's companion: M4's PUBLISHED taught fingerprint is `parent->final`, not
            # `parent->peak`. Reported beside B2 so a reader comparing to M4's tables sees the
            # same endpoints M4 used; B2's registered form is unchanged.
            bars["_informational"]["cos_taught_parent_final_vs_untaught_peak_final"] = \
                cosine_block(S["taught"], N["taught"], ("parent", "final"),
                             S["untaught"], N["untaught"], ("peak", "final"),
                             same_teams=False, reps=a.reps,
                             label="H2 against M4's PUBLISHED taught vector (parent->final)")
    res["bars"] = bars

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    tpath = a.out.replace(".json", "_tables.md")
    with open(tpath, "w") as f:
        f.write(render_tables(res))
    print(json.dumps({k: v for k, v in bars.items() if not k.startswith("_")}, indent=1))
    print(f"[m9b] wrote {a.out} and {tpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
