r"""X63's k-v-k decision on forum72 against Kakuna as a ``main.reads`` rule ``x63_kvk`` (X63 PREREG §6–§7,
``measurements/x63_2026-10-29/PREREG.md``). Pure halves; the arithmetic is the PREREG's, nothing re-derived.

THE STATISTIC (§6), on ONE anchor group (``primary``: forum72; ``home``: home72), each model's score a PLAIN
proportion p̂ = wins / G of its counted games (a tie a non-win, the ``d1`` convention):

* Δ = mean_on p̂ − mean_off p̂; B = Σ_on p̂q̂/G / k_on² + Σ_off p̂q̂/G / k_off²;
* σ̂² = the mean, over the sides holding ≥ 2 models, of (that side's sample variance of p̂, ddof 1 − that side's mean
  p̂q̂/G); σ̂ = √max(0, σ̂²); df = Σ (k_side − 1). At Stage A (1 v 2) σ̂ IS X63-C's σ̂_CC = √(max(0, d² − b₁ − b₂) / 2)
  (X63-C PREREG §3): the OFF side's two-point variance is d² / 2;
* σ_eff = max(``sigma_floor``, σ̂); SE² = B + σ_eff² (1/k_on + 1/k_off);
* q = :data:`Z` if σ̂ ≤ ``sigma_floor``, else the Student-t 0.975 quantile on df (``scipy.stats.t``); LB / UB = Δ ∓ q·SE.

THE CLASSES (§7). Every comparison of a statistic with its bar is by EXACT fractions, and within ``rule8_eps`` takes the
conservative side (root rule 8; §6: not CONTINUE, not GAIN, not RETIRE, not WITHIN). Where §6 names no side, this
module's choice is the one that licenses less: σ̂ near ``sigma_floor`` takes the t quantile, σ̂ near
``inconclusive_sigma`` is INCONCLUSIVE, Stage C does not fire on a near Δ̂, a near EXCEEDED bound is not EXCEEDED, a
near floor LB does not PASS.

* Stage A (``on`` 1 model, ``off`` 2): CONTINUE iff Δ̂ ≥ ``screen``, else NOT PURSUED; σ̂_CC printed, and, when the spec
  registers ``stage_a_power`` (rows of ``power/power_x63_v2.log``'s REGISTERED table), P(NOT PURSUED) at a true +2 / +3
  pp at the row nearest σ̂_CC (a tie to the larger σ).
* Stages B (2 v 2) and C (3 v 3): INCONCLUSIVE iff σ̂ > ``inconclusive_sigma``; else GAIN iff LB > 0; else RETIRE iff
  UB < ``kill_bar``; else UNDECIDED. At B the decision says whether Stage C fires (UNDECIDED and Δ̂ ≥ ``stage_c_bar``).
* On a GAIN: FLOOR PASS iff the ``on`` models' POOLED home72 Wilson 95 % LB > 0.50; net = Δ_home + r·Δ_F, SE_net² =
  SE_home² + r²·SE_F² (SE_home by the same arithmetic on ``home``); WITHIN iff net ≥ 0; EXCEEDED iff net + Z·SE_net <
  0. The verdict, in the re-review's order: FLOOR fails ⇒ OVER-RATE (whatever r); else r UNSET (null) ⇒ UNPRICED (r* =
  −Δ_home / Δ_F printed); else WITHIN ⇒ ADOPT-CANDIDATE; else EXCEEDED ⇒ OVER-RATE; else TRADE.

A deciding group that is not conclusive (its assembled reasons, unplayed units, no ONE known engine, a model with no
counted game) is INCONCLUSIVE with those reasons, never decided (§8: two engines or two peer packs in a group).

REPORTED, never deciding: each group's per-model table; Δ_home and its interval at every stage; the 72-team cluster
bootstrap of Δ_F (our team per game, :data:`BOOT_DRAWS` draws at :data:`BOOT_SEED`, one draw of teams shared by every
model; "not computed" with the reason when a counted unit's rows carry no team); every ``reported`` anchor group's
per-model proportions (INCOMPLETE with its reasons when short). At Stage A the ``home`` group is reported-only too: no
Stage A verdict reads it, so a short home72 never blocks Stage A's record.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "gen3_read_x63_kvk_v1"
KEYS = ("on", "off", "primary", "home", "stage", "screen", "kill_bar", "stage_c_bar", "sigma_floor",
        "inconclusive_sigma", "r", "rule8_eps")
OPTIONAL = ("reported", "stage_a_power", "void")
#: (k_on, k_off) of each registered stage (PREREG §7: A is 1 v 2, B 2 v 2, C 3 v 3).
STAGE_K = {"A": (1, 2), "B": (2, 2), "C": (3, 3)}
#: The PREREG's normal quantile (§6, §7.4), as written there.
Z = 1.959964
#: FLOOR PASS's bar on the pooled home72 Wilson LB (§7.4).
FLOOR_BAR = 0.5
BOOT_DRAWS, BOOT_SEED = 10_000, 0

CONTINUE, NOT_PURSUED = "CONTINUE", "NOT PURSUED"
GAIN, RETIRE, UNDECIDED, INCONCLUSIVE = "GAIN", "RETIRE", "UNDECIDED", "INCONCLUSIVE"
ADOPT, TRADE, OVER_RATE, UNPRICED = "ADOPT-CANDIDATE", "TRADE", "OVER-RATE", "UNPRICED"
RULE_DOCS = {"x63_kvk": "X63's k-v-k decision on forum72 (PREREG §6–§7): Δ of per-model plain proportions, a binomial "
                        "part plus a floored seed term, z below the σ floor else t on df; Stage A screen CONTINUE / NOT "
                        "PURSUED; B / C INCONCLUSIVE > GAIN > RETIRE > UNDECIDED; a GAIN priced on home72 (FLOOR, r)"}


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def cmp(stat: float, bar: Any, eps: Any, *, above: bool) -> Tuple[bool, bool]:
    """``(crosses, near)`` by EXACT fractions (``rules_search_headroom.cmp``): within ``eps`` is never a crossing."""
    from main.reads.rules_search_headroom import cmp as _cmp

    return _cmp(stat, bar, eps, above=above)


# ------------------------------------------------------------------------------------------------ check (spec load)
def _side(name: str, labs: Any, models: Any, k: Optional[int]) -> Tuple[List[str], List[str]]:
    if not (isinstance(labs, list) and labs and all(isinstance(v, str) and v for v in labs)
            and len(set(labs)) == len(labs)):
        return [f"{name}: a list of >= 1 distinct model labels"], []
    missing = [lab for lab in labs if lab not in models]
    if missing:
        return [f"{name}: {missing} are not model labels"], []
    probs: List[str] = []
    if k is not None and len(labs) != k:
        probs.append(f"{name}: {len(labs)} model(s), the registered stage holds {k}")
    seeds = [int(models[lab].seed) for lab in labs]
    if len(set(seeds)) != len(seeds):
        probs.append(f"{name}: two models share a seed")
    loaded = [lab for lab in labs if models[lab].load]
    if loaded:
        probs.append(f"{name}: {loaded} carry load options (the read plays greedy finals)")
    return probs, sorted({models[lab].arm for lab in labs})


def _group_check(what: str, name: Any, anchor: Any, labs: Sequence[str]) -> List[str]:
    g = anchor.get(name) if isinstance(name, str) else None
    if g is None:
        return [f"{what}: {name!r} is not an anchor group"]
    probs = [f"{what} group {name!r} does not play {lab!r}" for lab in labs if lab not in g.models]
    if len(g.teamsets) != 1:
        probs.append(f"{what} group {name!r}: {len(g.teamsets)} team sets (the statistic reads ONE panel)")
    if g.per_game:
        probs.append(f"{what} group {name!r} is per_game (the read plays 20-game units)")
    if g.imports is not None:
        probs.append(f"{what} group {name!r} imports cells (every X63 game is played at the read's pin, §5)")
    return probs


def _power_check(sp: Any) -> List[str]:
    if sp is None:
        return []
    ok = (isinstance(sp, dict) and set(sp) == {"source", "rows"} and isinstance(sp["source"], str) and sp["source"]
          and isinstance(sp["rows"], list) and sp["rows"]
          and all(isinstance(r, dict) and set(r) == {"sigma_pp", "p_np_at_2pp", "p_np_at_3pp"}
                  and all(_num(v) for v in r.values()) and r["sigma_pp"] >= 0
                  and 0 <= r["p_np_at_2pp"] <= 1 and 0 <= r["p_np_at_3pp"] <= 1 for r in sp["rows"]))
    if not ok:
        return ["stage_a_power: {source, rows: [{sigma_pp, p_np_at_2pp, p_np_at_3pp}, ...]} (the REGISTERED rows of "
                "the power log, σ_arm = σ_ctl)"]
    sig = [r["sigma_pp"] for r in sp["rows"]]
    return ["stage_a_power: two rows share a sigma_pp"] if len(set(sig)) != len(sig) else []


def check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    from main.reads.rules import _check_keys, _void_check

    probs = _check_keys(p, KEYS, KEYS + OPTIONAL)
    stage = p.get("stage")
    if stage not in STAGE_K:
        probs.append(f"stage: one of {sorted(STAGE_K)}")
    k_on, k_off = STAGE_K[stage] if stage in STAGE_K else (None, None)
    pr_on, on_arms = _side("on", p.get("on"), models, k_on)
    pr_off, off_arms = _side("off", p.get("off"), models, k_off)
    probs += pr_on + pr_off
    both = list(p.get("on") or []) + list(p.get("off") or []) if not (pr_on or pr_off) else []
    if both and len(set(both)) != len(both):
        probs.append("on / off: a label on both sides")
    if len(on_arms) > 1:
        probs.append(f"on: its models span arms {on_arms} (the arm side is ONE arm)")
    if set(on_arms) & set(off_arms):
        probs.append(f"on / off: arm(s) {sorted(set(on_arms) & set(off_arms))} on both sides")
    probs += _group_check("primary", p.get("primary"), anchor, both)
    probs += _group_check("home", p.get("home"), anchor, both)
    if p.get("primary") is not None and p.get("primary") == p.get("home"):
        probs.append("primary / home: one group")
    rep = p.get("reported", [])
    if not (isinstance(rep, list) and all(isinstance(g, str) and g in anchor for g in rep) and len(set(rep)) == len(rep)
            and not set(rep) & {p.get("primary"), p.get("home")}):
        probs.append("reported: a list of distinct anchor groups other than primary and home")
    for key in ("screen", "kill_bar", "stage_c_bar", "sigma_floor", "inconclusive_sigma"):
        if not (_num(p.get(key)) and 0 < p[key] < 1):
            probs.append(f"{key}: a proportion in (0, 1)")
    if _num(p.get("sigma_floor")) and _num(p.get("inconclusive_sigma")) and not p["sigma_floor"] < p["inconclusive_sigma"]:
        probs.append("sigma_floor must be below inconclusive_sigma")
    r = p.get("r", "absent")
    if not (r is None or (_num(r) and r >= 0)):
        probs.append("r: a number >= 0, or null (UNSET: a GAIN with FLOOR PASS is UNPRICED)")
    if not (_num(p.get("rule8_eps")) and 0 <= p["rule8_eps"] < 0.001):
        probs.append("rule8_eps: a number in [0, 0.001) (a statistic this close to a bar is never past it)")
    probs += _power_check(p.get("stage_a_power"))
    if p.get("void") is not None and len(on_arms) == 1:
        probs += _void_check(dict(p, on=on_arms[0]), models, required=False)
    return probs


def reported(p: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """The REPORTED-only anchor groups: the ``reported`` list, and ``home`` at Stage A (no Stage A verdict reads it)."""
    out = [("anchor", str(g)) for g in p.get("reported") or ()]
    return out + ([("anchor", str(p["home"]))] if p.get("stage") == "A" else [])


# ------------------------------------------------------------------------------------------------ inputs
def team_counts(units: Sequence[Mapping[str, Any]], labels: Sequence[str]
                ) -> Tuple[Optional[Dict[str, Dict[str, List[int]]]], str]:
    """``({label: {our_team: [wins, games]}}, "")`` over the COUNTED units' per-game rows (``assemble.GAME_KEYS``:
    half, index, our_team, result, turns), or ``(None, why)`` when a counted unit carries no team per game."""
    out: Dict[str, Dict[str, List[int]]] = {lab: {} for lab in labels}
    for u in units:
        if u.get("label") not in out or not u.get("counted"):
            continue
        games = u.get("games")
        if not games:
            return None, f"counted unit {u.get('unit')} carries no per-game rows"
        if len(games) != int(u.get("n") or 0):
            return None, f"counted unit {u.get('unit')}: {len(games)} game rows, n {u.get('n')}"
        for g in games:
            team, result = g[2], g[3]
            if not (isinstance(team, str) and team):
                return None, f"counted unit {u.get('unit')}: a game with no our_team"
            c = out[u["label"]].setdefault(team, [0, 0])
            c[0] += int(result == "win")
            c[1] += 1
    return out, ""


def _block(cells: Mapping[str, Any], name: str, labels: Sequence[str], *, teams: bool) -> Dict[str, Any]:
    g = (cells.get("anchor") or {}).get(name)
    if g is None:
        return {"group": name, "missing": True, "inconclusive": [f"no assembled anchor group {name!r}"],
                "unplayed": 0, "engine": None, "regime_id": None, "teamsets": [], "models": {}}
    out: Dict[str, Any] = {
        "group": name, "teamsets": list(g.get("teamsets") or []), "engine": g.get("engine"),
        "regime_id": g.get("regime_id"), "inconclusive": list(g.get("inconclusive") or []),
        "unplayed": int(g.get("unplayed") or 0),
        "models": {lab: {ts: {k: (c.get(k) if k != "skipped" else len(c.get(k) or ()))
                              for k in ("wins", "ties", "n", "units", "planned", "skipped")}
                         for ts, c in ((g.get("cells") or {}).get(lab) or {}).items()} for lab in labels}}
    if teams:
        out["teams"], out["teams_why"] = team_counts(g.get("units") or (), labels)
    return out


def inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    labels = list(p["on"]) + list(p["off"])
    rep_labels = {g: list(spec.anchor[g].models) for g in p.get("reported") or ()}
    return {"schema": SCHEMA, "stage": p["stage"],
            "sides": {"on": list(p["on"]), "off": list(p["off"])},
            "seeds": {lab: int(spec.models[lab].seed) for lab in labels},
            "groups": {"primary": _block(cells, p["primary"], labels, teams=True),
                       "home": _block(cells, p["home"], labels, teams=False)},
            "reported": {g: _block(cells, g, labs, teams=False) for g, labs in rep_labels.items()}}


# ------------------------------------------------------------------------------------------------ the arithmetic
def t975(df: int) -> float:
    """The Student-t 0.975 quantile on ``df`` (exact, ``scipy.stats.t``)."""
    from scipy.stats import t as student_t

    return float(student_t.ppf(0.975, df))


def quantile(sigma_hat: float, floor: float, df: int, eps: float) -> Tuple[float, str]:
    """``(q, "z" | "t")``: :data:`Z` iff σ̂ is CLEARLY ≤ the floor; a σ̂ within ``eps`` of it takes t (rule 8: the
    larger quantile licenses less)."""
    above, near = cmp(sigma_hat, floor, eps, above=True)
    if above or near:
        return t975(df), "t"
    return Z, "z"


def kvk(on: Sequence[Tuple[int, int]], off: Sequence[Tuple[int, int]], floor: float, eps: float) -> Dict[str, Any]:
    """§6 on ``(wins, games)`` per model of each side."""
    sides = {"on": [w / n for w, n in on], "off": [w / n for w, n in off]}
    bin_ = {"on": [(w / n) * (1 - w / n) / n for w, n in on], "off": [(w / n) * (1 - w / n) / n for w, n in off]}
    k = {s: len(v) for s, v in sides.items()}
    mean = {s: math.fsum(v) / k[s] for s, v in sides.items()}
    delta = mean["on"] - mean["off"]
    b = math.fsum(bin_["on"]) / k["on"] ** 2 + math.fsum(bin_["off"]) / k["off"] ** 2
    terms: Dict[str, Dict[str, float]] = {}
    for s in ("on", "off"):
        if k[s] >= 2:
            var = math.fsum((x - mean[s]) ** 2 for x in sides[s]) / (k[s] - 1)
            terms[s] = {"sample_var": var, "mean_pq_over_g": math.fsum(bin_[s]) / k[s]}
            terms[s]["term"] = terms[s]["sample_var"] - terms[s]["mean_pq_over_g"]
    s2 = math.fsum(t["term"] for t in terms.values()) / len(terms) if terms else 0.0
    sigma_hat = math.sqrt(max(0.0, s2))
    df = sum(k[s] - 1 for s in ("on", "off"))
    sigma_eff = max(floor, sigma_hat)
    se = math.sqrt(b + sigma_eff ** 2 * (1 / k["on"] + 1 / k["off"]))
    q, kind = quantile(sigma_hat, floor, df, eps) if terms else (Z, "z")
    return {"delta": delta, "mean_on": mean["on"], "mean_off": mean["off"], "p_on": sides["on"],
            "p_off": sides["off"], "k_on": k["on"], "k_off": k["off"], "binomial_part": b, "seed_terms": terms,
            "sigma_hat_sq": s2, "sigma_hat": sigma_hat, "df": df, "sigma_eff": sigma_eff, "se": se, "q": q,
            "q_kind": kind, "lb": delta - q * se, "ub": delta + q * se}


def sigma_cc(d: float, b1: float, b2: float) -> float:
    """X63-C PREREG §3: σ̂_CC = √(max(0, d² − b₁ − b₂) / 2), d the two controls' difference, b their p̂q̂/G."""
    return math.sqrt(max(0.0, d * d - b1 - b2) / 2)


def _group_reasons(blk: Mapping[str, Any], labels: Sequence[str], what: str) -> List[str]:
    why = [f"{what}: {w}" for w in blk.get("inconclusive") or ()]
    if blk.get("unplayed"):
        why.append(f"{what}: {blk['unplayed']} unit(s) unplayed")
    if not (isinstance(blk.get("engine"), str) and blk["engine"]):
        why.append(f"{what}: its counted units are not on ONE known engine")
    if len(blk.get("teamsets") or ()) != 1:
        why.append(f"{what}: {len(blk.get('teamsets') or ())} team sets, not one")
    for lab in labels:
        cells = (blk.get("models") or {}).get(lab) or {}
        if sum(int(c.get("n") or 0) for c in cells.values()) <= 0:
            why.append(f"{what} {lab}: no counted game")
    return why


def _counts(blk: Mapping[str, Any], labs: Sequence[str]) -> List[Tuple[int, int]]:
    ts = blk["teamsets"][0]
    return [(int(blk["models"][lab][ts]["wins"]), int(blk["models"][lab][ts]["n"])) for lab in labs]


def _stat(blk: Mapping[str, Any], sides: Mapping[str, Sequence[str]], p: Mapping[str, Any]) -> Dict[str, Any]:
    return kvk(_counts(blk, sides["on"]), _counts(blk, sides["off"]), float(p["sigma_floor"]), float(p["rule8_eps"]))


def floor_check(blk: Mapping[str, Any], on: Sequence[str], eps: float) -> Dict[str, Any]:
    """FLOOR PASS iff the ``on`` models' POOLED Wilson 95 % LB on ``blk`` > :data:`FLOOR_BAR` (within eps: no)."""
    from main.anchors.results import wilson

    w, n = (sum(x) for x in zip(*_counts(blk, on)))
    phat, lo, hi = wilson(w, n)
    passes, near = cmp(lo, FLOOR_BAR, eps, above=True)
    return {"wins": w, "n": n, "p": phat, "wilson_lb": lo, "wilson_ub": hi, "bar": FLOOR_BAR, "pass": passes,
            "near": near}


def price(f: Mapping[str, Any], h: Mapping[str, Any], r: Optional[float], eps: float) -> Dict[str, Any]:
    """§7.4's price: net = Δ_home + r·Δ_F, SE_net² = SE_home² + r²·SE_F²; WITHIN iff net ≥ 0 (within eps: no);
    EXCEEDED iff net + Z·SE_net < 0 (within eps: no). ``r`` None (UNSET): r* = −Δ_home / Δ_F only."""
    out: Dict[str, Any] = {"r": r, "r_star": (-h["delta"] / f["delta"]) if f["delta"] else None}
    if r is None:
        return dict(out, net=None, se_net=None, within=None, exceeded=None)
    net = h["delta"] + r * f["delta"]
    se_net = math.sqrt(h["se"] ** 2 + r * r * f["se"] ** 2)
    up = net + Z * se_net
    w_cross, w_near = cmp(net, 0, eps, above=True)
    e_cross, e_near = cmp(up, 0, eps, above=False)
    return dict(out, net=net, se_net=se_net, net_upper=up, within=w_cross, within_near=w_near, exceeded=e_cross,
                exceeded_near=e_near)


def gain_verdict(floor: Mapping[str, Any], pr: Mapping[str, Any]) -> str:
    """The re-review's order (PREREG §7): FLOOR fails ⇒ OVER-RATE; r UNSET ⇒ UNPRICED; WITHIN ⇒ ADOPT-CANDIDATE;
    EXCEEDED ⇒ OVER-RATE; else TRADE."""
    if not floor["pass"]:
        return OVER_RATE
    if pr["r"] is None:
        return UNPRICED
    if pr["within"]:
        return ADOPT
    if pr["exceeded"]:
        return OVER_RATE
    return TRADE


def classify(f: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    """Stage A's screen, or Stage B / C's class, from the primary statistic ``f``."""
    eps = p["rule8_eps"]
    if p["stage"] == "A":
        go, near = cmp(f["delta"], p["screen"], eps, above=True)
        return {"class": CONTINUE if go else NOT_PURSUED, "tests": {"delta >= screen": {"crosses": go, "near": near}}}
    t = {"sigma_hat > inconclusive_sigma": cmp(f["sigma_hat"], p["inconclusive_sigma"], eps, above=True),
         "LB > 0": cmp(f["lb"], 0, eps, above=True), "UB < kill_bar": cmp(f["ub"], p["kill_bar"], eps, above=False)}
    inc = t["sigma_hat > inconclusive_sigma"][0] or t["sigma_hat > inconclusive_sigma"][1]
    cls = (INCONCLUSIVE if inc else GAIN if t["LB > 0"][0] else RETIRE if t["UB < kill_bar"][0] else UNDECIDED)
    out: Dict[str, Any] = {"class": cls, "tests": {k: {"crosses": v[0], "near": v[1]} for k, v in t.items()}}
    if p["stage"] == "B":
        c_go, c_near = cmp(f["delta"], p["stage_c_bar"], eps, above=True)
        out["stage_c_fires"] = cls == UNDECIDED and c_go
        out["tests"]["delta >= stage_c_bar"] = {"crosses": c_go, "near": c_near}
    return out


def stage_a_power(sp: Optional[Mapping[str, Any]], sigma_hat: float) -> Optional[Dict[str, Any]]:
    """The registered row nearest σ̂_CC (in pp; a tie to the larger σ), or None when the spec registers none."""
    if not sp:
        return None
    pp = 100 * sigma_hat
    row = min(sp["rows"], key=lambda r: (abs(Fraction(str(r["sigma_pp"])) - Fraction(pp)), -r["sigma_pp"]))
    return {"source": sp["source"], "sigma_cc_pp": pp, "row": dict(row)}


def bootstrap(blk: Mapping[str, Any], sides: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    """REPORTED: Δ_F's 72-team cluster bootstrap (our team per game), ONE draw of teams shared by every model."""
    import numpy as np

    teams = blk.get("teams")
    if teams is None:
        return {"computed": False, "why": blk.get("teams_why") or "no per-game team rows"}
    ids = sorted({t for by in teams.values() for t in by})
    labs = list(sides["on"]) + list(sides["off"])
    w = np.array([[teams[lab].get(t, [0, 0])[0] for t in ids] for lab in labs], dtype=float)
    n = np.array([[teams[lab].get(t, [0, 0])[1] for t in ids] for lab in labs], dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    pick = rng.integers(0, len(ids), size=(BOOT_DRAWS, len(ids)))
    counts = np.stack([np.bincount(row, minlength=len(ids)) for row in pick])           # draws x teams
    pw, pn = counts @ w.T, counts @ n.T                                                 # draws x models
    with np.errstate(invalid="ignore", divide="ignore"):
        ph = pw / pn
    k_on = len(sides["on"])
    d = ph[:, :k_on].mean(axis=1) - ph[:, k_on:].mean(axis=1)
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"computed": True, "clusters": len(ids), "draws": BOOT_DRAWS, "seed": BOOT_SEED, "used_draws": int(d.size),
            "ci95": [float(lo), float(hi)], "sd": float(d.std(ddof=1))}


def _reported_block(blk: Mapping[str, Any]) -> Dict[str, Any]:
    why = [f"{blk['group']}: {w}" for w in blk.get("inconclusive") or ()]
    if blk.get("unplayed"):
        why.append(f"{blk['group']}: {blk['unplayed']} unit(s) unplayed")
    rows = {lab: {ts: {"wins": c["wins"], "n": c["n"], "p": (c["wins"] / c["n"]) if c.get("n") else None}
                  for ts, c in by.items()} for lab, by in (blk.get("models") or {}).items()}
    return {"status": "INCOMPLETE" if why else "OK", "reasons": why, "engine": blk.get("engine"), "per_model": rows}


# ------------------------------------------------------------------------------------------------ decide
def decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    sides = {s: list(doc["sides"][s]) for s in ("on", "off")}
    labels = sides["on"] + sides["off"]
    prim, home = doc["groups"]["primary"], doc["groups"]["home"]
    base: Dict[str, Any] = {"schema": SCHEMA, "stage": doc["stage"], "sides": sides}
    rep: Dict[str, Any] = {g: _reported_block(b) for g, b in (doc.get("reported") or {}).items()}
    why = _group_reasons(prim, labels, "primary")
    if why:
        return dict(base, primary=None, reported=rep, decision={
            "verdict": INCONCLUSIVE, "class": INCONCLUSIVE, "stage": doc["stage"], "reasons": why,
            "inconclusive_kind": "inputs"})
    f = _stat(prim, sides, p)
    c = classify(f, p)
    home_why = _group_reasons(home, labels, "home")
    h = None if home_why else _stat(home, sides, p)
    rep["home"] = {"status": "OK", "stat": h} if h is not None else {"status": "INCOMPLETE", "reasons": home_why}
    rep["cluster_bootstrap_delta_f"] = bootstrap(prim, sides)
    dec: Dict[str, Any] = {"stage": doc["stage"], "class": c["class"], "tests": c["tests"], "reasons": []}
    if doc["stage"] == "A":
        dec.update(verdict=c["class"], sigma_cc=f["sigma_hat"],
                   stage_a_power=stage_a_power(p.get("stage_a_power"), f["sigma_hat"]))
    elif c["class"] == GAIN:
        if h is None:
            dec.update(verdict=INCONCLUSIVE, inconclusive_kind="inputs",
                       reasons=["a GAIN is priced on home72, which is not conclusive"] + home_why)
        else:
            fl = floor_check(home, sides["on"], float(p["rule8_eps"]))
            pr = price(f, h, None if p["r"] is None else float(p["r"]), float(p["rule8_eps"]))
            dec.update(verdict=gain_verdict(fl, pr), floor=fl, price=pr)
    else:
        dec["verdict"] = c["class"]
        if c["class"] == INCONCLUSIVE:
            dec["inconclusive_kind"] = "sigma"
    if "stage_c_fires" in c:
        dec["stage_c_fires"] = c["stage_c_fires"]
    return dict(base, primary=f, reported=rep, decision=dec)
