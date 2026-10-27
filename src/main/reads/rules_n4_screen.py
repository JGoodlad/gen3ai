r"""The one-look capacity SCREEN as a ``main.reads`` rule ``n4_screen`` (N4 PREREG §5, with the N4 REVIEW's M2 / M4 /
M5; X48 PREREG §5, §8, §10 registers it as built here). Pure halves; ``inputs`` reads the runs' dose records once.

THE METERS (N4 §5.2, verbatim in X48 §5.2):

* **S**: each final's ``readout.population_score`` against R (the S group ``s_group``); Δ_S = mean(ON) − mean(OFF) with
  ``readout.welch_interval(on, off, confidence)`` (two-sided 90 %: each bound one-sided 95 %). Every (final, member)
  cell must hold at least the S group's registered ``pairs`` (X48: 2,000, M3; N4: 500): the spec's ``pairs`` IS the
  cell floor, so one rule serves both reads.
* **K**: the anchor group ``anchor_group`` (Kakuna on forum72 alone) by the ``d1`` estimators at ``z``: ``strat_score``
  per arm (wins / games pooled over its finals; a tie a non-win), ``newcombe_strat(on, off)``, φ̂ =
  ``overdispersion(on_cells, off_cells)`` from the per-final cells and ``widen(Δ̂, lo, hi, √max(1, φ̂))``. φ̂ is
  printed with the class (review m2).

THE CLASSES, in order (X48 §5.3). Every bound is compared as an EXACT fraction against its bar (S's in pp, K's a
proportion), and a bound within ``rule8_eps`` of a bar is ``near``: never past it (root rule 8). ``near_scope``
"bound" (the default, X48's wording) leaves the other bound of that meter deciding; "meter" (N4 §5.3's wording, "that
meter is UNDECIDED") makes a near GO / NO-GO bound read the whole meter UNDECIDED.

1. INCONCLUSIVE: a registered input is missing (a node without S, an S cell below ``pairs``, a K arm short of
   ``d1.min_games`` counted games = more than one uncounted 20-game unit, an unplayed unit, a seed final absent), or a
   meter's cells span two engines or sit on an unknown one (the ``main.reads`` engine rule). A VOID seed never reaches
   the rule: the registered VOID gate (``params.void``, ``rules.void_gate``) refuses the decision first.
2. MIXED (M2, ``mixed_veto``), classed UNDECIDED and never GO: GO-S with UB_K < 0, or GO-K with UB_S < 0.
3. GO iff GO-S (LB_S > 0) or GO-K (LB_K > 0): ``GO (both)`` / ``GO (in-house)`` / ``GO (unfamiliar)`` with Δ̂_S and
   Δ̂_K; its text reads "growth at the plateau helped" (review m4).
4. NO-GO iff NO-GO-S (UB_S < ``s_bar``) and not GO-K. Its text is M5's: "no ≥ <s_bar> pp in-house gain from <lever>
   within <dose> of <start>" (``nogo_scope``), never "capacity does not bind", plus NO-GO-K (UB_K < ``k_bar``) or
   UNDECIDED-K and :data:`K_ONLY_LINE`. Labelled ``LR-confounded`` (M4) when the ON arm's mean LR is < 0.9× the OFF
   arm's (each arm's mean over its finals of the run's mean ``snapshot_history`` lr, ``metadata.json``, the dose
   record); a ratio within ``rule8_eps`` of 0.9 is not below it. An unreadable LR prints the label UNAVAILABLE.
   Its mirror (X48 NOTE 1 (d)): a GO is labelled ``LR-elevated`` when the ratio is > 1.1×. ``nogo_scope.qualifier``
   appends the PREREG's clause to the NO-GO text (X48 NOTE 1 (c)). Labels needing inputs the cells do not hold (X48
   NOTE 1 (b) "experts not engaged", NOTE 2 "read at the controller's learning rate") are post-rule lines, named in
   ``not_reported``.
5. UNDECIDED otherwise, with both intervals and the S screen's power at the observed σ.

The family false GO is ≈ 0.09, not 0.05 (M2): stated in every decision (:data:`FAMILY_FALSE_GO`).

REPORTED, never deciding (X48 §8; no adjectives): R1 at look 1 (``readout.decide_r1``, 97.5 %, ±2 pp) on the same S;
each arm's mean LR and the ratio; per-member Δ_S (Welch at ``confidence``); the slope (``slope``: S at the +10M
nodes, per arm S(final) − S(node), Δ_slope by Welch, a REPORTED-only S group that never blocks the record); the pin-
drift line (``pin_drift``: the OFF arm's re-played mean S against a banked value, descriptive); the observed σ of S
and of K's per-final score; the S screen's power at the observed σ (a seeded Monte Carlo, :func:`s_power`); each
meter's engine. Listed in ``not_reported`` (the cells do not carry them): ρ, the approx KL, G-F, the experts'
divergence, home72, K's power at the observed φ̂.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "gen3_read_n4_screen_v1"
KEYS = ("on", "off", "s_group", "anchor_group", "s_bar", "k_bar", "confidence", "z", "rule8_eps", "mixed_veto", "void",
        "nogo_scope")
OPTIONAL = ("near_scope", "slope", "pin_drift")
NEAR_SCOPES = ("bound", "meter")
SCOPE_KEYS = ("lever", "dose", "start")
SCOPE_OPTIONAL = ("qualifier",)
#: M4: a NO-GO whose ON arm's mean LR is below this fraction of the OFF arm's is labelled LR-confounded.
LR_CONFOUND_RATIO = Fraction(9, 10)
LR_CONFOUNDED = "LR-confounded"
#: X48 NOTE 1 (d) (its review M3), M4's mirror: a GO whose ON arm's mean LR is above this fraction is LR-elevated.
LR_ELEVATED_RATIO = Fraction(11, 10)
LR_ELEVATED = "LR-elevated"
#: The registered look: one look, no look 2 (N4 §5.4; X48 §2).
LOOK = 1

INCONCLUSIVE, MIXED, GO, NO_GO, UNDECIDED = "INCONCLUSIVE", "MIXED", "GO", "NO-GO", "UNDECIDED"
GO_S, NO_GO_S, UNDECIDED_S = "GO-S", "NO-GO-S", "UNDECIDED-S"
GO_K, NO_GO_K, UNDECIDED_K = "GO-K", "NO-GO-K", "UNDECIDED-K"
GO_TEXT = "growth at the plateau helped"
K_ONLY_LINE = "no unfamiliar-only effect ≥ ≈ 4 pp detected; smaller ones are not excluded"
FAMILY_FALSE_GO = ("≈ 0.09, not 0.05: α = 0.05 one-sided per meter, two meters, GO on either (N4 REVIEW M2); a screen "
                   "GO buys a production re-read, not an adoption")
#: The S screen's power at the observed σ (reported): seeded, so a re-decision prints the same numbers.
POWER_DRAWS, POWER_SEED = 20_000, 0
NOT_REPORTED = {
    "rho": "the wall per step needs the dose records' wall clock (X48 §8.2), outside the cells",
    "approx_kl": "the approx KL is in the runs' TensorBoard (main.ops.tb_read), outside the cells",
    "g_f": "the 3 x 3 head to head is its own h2h read (cross.matrix_from_cells)",
    "expert_divergence": "a weights-only script at read time (X48 §8.7, FINDING 8)",
    "home72": "a separate anchor group, played only if the lane has room (X48 §8.8)",
    "k_power": "K's power at the observed φ̂ is the PREREG's power.py, not re-simulated here",
    "experts_not_engaged": "X48 NOTE 1 (b)'s NO-GO label needs §8.7's D_k / null at the final (a weights script): a "
                           "post-rule line",
    "read_at_the_controllers_lr": "X48 NOTE 2's label needs LR-F look 1's class (another read): a post-rule line",
}
RULE_DOCS = {"n4_screen": "the one-look capacity screen (N4 §5 + M2 MIXED, M4 LR label, M5 NO-GO wording; X48 §5): "
                          "S by Welch 90 %, K by the d1 estimators on forum72; INCONCLUSIVE > MIXED > GO > NO-GO > "
                          "UNDECIDED"}


def _err(msg: str) -> Exception:
    from main.reads.rules import RuleInputError

    return RuleInputError(msg)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _labels(x: Any) -> bool:
    return isinstance(x, list) and len(x) >= 2 and all(isinstance(v, str) and v for v in x) and len(set(x)) == len(x)


# ------------------------------------------------------------------------------------------------ check (spec load)
def _side_check(name: str, labs: Any, models: Any) -> Tuple[List[str], Optional[str], List[int]]:
    if not _labels(labs):
        return [f"{name}: a list of >= 2 distinct model labels"], None, []
    missing = [lab for lab in labs if lab not in models]
    if missing:
        return [f"{name}: {missing} are not model labels"], None, []
    probs: List[str] = []
    arms = sorted({models[lab].arm for lab in labs})
    if len(arms) != 1:
        probs.append(f"{name}: its labels span arms {arms} (one arm a side)")
    seeds = [int(models[lab].seed) for lab in labs]
    if len(set(seeds)) != len(seeds):
        probs.append(f"{name}: two finals share a seed")
    return probs, arms[0] if len(arms) == 1 else None, seeds


def check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    from main.reads.rules import _check_keys, _void_check
    from main.x35 import readout as R

    probs = _check_keys(p, KEYS, KEYS + OPTIONAL)
    pr_on, on_arm, on_seeds = _side_check("on", p.get("on"), models)
    pr_off, off_arm, off_seeds = _side_check("off", p.get("off"), models)
    probs += pr_on + pr_off
    both = list(p.get("on") or []) + list(p.get("off") or []) if not (pr_on or pr_off) else []
    if both and len(set(both)) != len(both):
        probs.append("on / off: a label on both sides")
    if on_arm is not None and on_arm == off_arm:
        probs.append(f"on / off: both sides are arm {on_arm!r}")
    sg = s.get(p.get("s_group"))
    if sg is None:
        probs.append(f"s_group {p.get('s_group')!r} is not an S group")
    else:
        if sg.pairs < R.MIN_PAIRS:
            probs.append(f"S group pairs {sg.pairs} < {R.MIN_PAIRS} per cell (§5.6.4's floor)")
        miss = [lab for lab in both if lab not in sg.models]
        if miss:
            probs.append(f"s_group {sg.name!r} does not play {miss}")
    ag = anchor.get(p.get("anchor_group"))
    if ag is None:
        probs.append(f"anchor_group {p.get('anchor_group')!r} is not an anchor group")
    else:
        miss = [lab for lab in both if lab not in ag.models]
        if miss:
            probs.append(f"anchor_group {ag.name!r} does not play {miss}")
        if ag.per_game:
            probs.append(f"anchor_group {ag.name!r} is per_game (K reads 20-game units)")
        loaded = [lab for lab in both if lab in models and models[lab].load]
        if loaded:
            probs.append(f"{loaded} carry load options (the screen reads greedy finals)")
    for k, lo, hi in (("s_bar", 0.0, 100.0), ("k_bar", 0.0, 1.0), ("confidence", 0.0, 1.0)):
        if not (_num(p.get(k)) and lo < p[k] < hi):
            probs.append(f"{k}: a number in ({lo:g}, {hi:g})")
    if not (_num(p.get("z")) and p["z"] > 0):
        probs.append("z: a number > 0 (K's one-sided quantile, 1.6449)")
    if not (_num(p.get("rule8_eps")) and 0 <= p["rule8_eps"] < 0.01):
        probs.append("rule8_eps: a number in [0, 0.01) (a bound this close to a bar is never past it)")
    if not isinstance(p.get("mixed_veto"), bool):
        probs.append("mixed_veto: true / false (M2's MIXED class)")
    if p.get("near_scope", "bound") not in NEAR_SCOPES:
        probs.append(f"near_scope: one of {list(NEAR_SCOPES)}")
    sc = p.get("nogo_scope")
    if not (isinstance(sc, dict) and set(SCOPE_KEYS) <= set(sc) <= set(SCOPE_KEYS + SCOPE_OPTIONAL)
            and all(isinstance(v, str) and v for v in sc.values())):
        probs.append(f"nogo_scope: {{{', '.join(SCOPE_KEYS)}, [qualifier]}} (M5's NO-GO wording: the lever, the dose, "
                     "the start, and a clause the PREREG appends)")
    if on_arm is not None:
        probs += _void_check(dict(p, on=on_arm), models, required=True)
    probs += _slope_check(p.get("slope"), p, models, s, on_seeds, off_seeds)
    pd = p.get("pin_drift")
    if pd is not None and not (isinstance(pd, dict) and set(pd) == {"s_pp", "source"} and _num(pd.get("s_pp"))
                               and isinstance(pd.get("source"), str) and pd["source"]):
        probs.append("pin_drift: {s_pp: the banked OFF mean S, source: where it is recorded}")
    return probs


def _slope_check(sl: Any, p: Mapping[str, Any], models: Any, s: Any, on_seeds: List[int], off_seeds: List[int]
                 ) -> List[str]:
    if sl is None:
        return []
    if not (isinstance(sl, dict) and set(sl) == {"group", "on", "off"}):
        return ["slope: {group, on, off}: the +10M nodes' S group and their labels per side"]
    g = s.get(sl["group"])
    if g is None or sl["group"] == p.get("s_group"):
        return [f"slope.group {sl['group']!r}: an S group other than s_group"]
    probs: List[str] = []
    for side, want in (("on", on_seeds), ("off", off_seeds)):
        pr, _arm, seeds = _side_check(f"slope.{side}", sl[side], models)
        probs += pr
        if not pr:
            probs += [f"slope group {g.name!r} does not play {lab!r}" for lab in sl[side] if lab not in g.models]
            if sorted(seeds) != sorted(want):
                probs.append(f"slope.{side}: seeds {sorted(seeds)} are not {side}'s {sorted(want)}")
    return probs


def reported(p: Mapping[str, Any]) -> List[Tuple[str, str]]:
    return [("s", str(p["slope"]["group"]))] if p.get("slope") else []


# ------------------------------------------------------------------------------------------------ inputs
def run_dir(model: Any) -> Optional[str]:
    """The run directory whose ``metadata.json`` holds a model's dose record: a ``final`` ref's ``run``; a ``pinned``
    ref's ``final_model.zip`` parent or ``checkpoints/`` grandparent; else None."""
    ref = model.ref
    if ref.kind == "final":
        return None if ref.run is None else str(ref.run)
    path = Path(str(ref.path or ""))
    if path.name == "final_model.zip":
        return str(path.parent)
    if path.parent.name == "checkpoints":
        return str(path.parent.parent)
    return None


def mean_lr(run: Optional[str]) -> Dict[str, Any]:
    """The run's mean ``lr`` over its ``metadata.json`` ``snapshot_history`` nodes (X48 §8.3), or ``mean_lr`` None
    and why."""
    from main.reads.models import spec_path

    out: Dict[str, Any] = {"run": run, "mean_lr": None, "n_nodes": 0,
                           "source": "metadata.json snapshot_history (the dose record)"}
    if run is None:
        return dict(out, why="no run dir for this ref")
    try:
        meta = json.loads((spec_path(run) / "metadata.json").read_text())
        sh = meta["snapshot_history"]
        rows = list(sh.values()) if isinstance(sh, dict) else list(sh)
        lrs = [float(r["lr"]) for r in rows if isinstance(r, dict) and _num(r.get("lr"))]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(out, why=f"{type(exc).__name__}: {exc}")
    if not lrs:
        return dict(out, why="snapshot_history holds no lr")
    return dict(out, mean_lr=math.fsum(lrs) / len(lrs), n_nodes=len(lrs))


def _s_nodes(g: Mapping[str, Any], labels: Mapping[str, str], spec: Any) -> Dict[str, Dict[str, Any]]:
    out = {}
    for lab, side in labels.items():
        n = (g.get("nodes") or {}).get(lab)
        if n is None:
            out[lab] = {"side": side, "seed": int(spec.models[lab].seed), "s_pp": None, "n_pairs": {},
                        "per_member_pp": {}, "inconclusive": ["no node in the assembled S group"]}
            continue
        out[lab] = {"side": side, "seed": int(spec.models[lab].seed), "s_pp": n.get("s_pp"), "se_pp": n.get("se_pp"),
                    "n_pairs": dict(n.get("n_pairs") or {}), "per_member_pp": dict(n.get("per_member_pp") or {}),
                    "inconclusive": list(n.get("inconclusive") or [])}
    return out


def _s_block(cells: Mapping[str, Any], name: str, labels: Mapping[str, str], spec: Any) -> Dict[str, Any]:
    g = (cells.get("s") or {}).get(name)
    if g is None:
        return {"group": name, "missing": True, "inconclusive": [f"no assembled S group {name!r}"], "engine": None,
                "pairs_registered": int(spec.s[name].pairs), "nodes": {}}
    return {"group": name, "engine": g.get("engine"), "regime_id": g.get("regime_id"),
            "inconclusive": list(g.get("inconclusive") or []), "pairs_registered": int(spec.s[name].pairs),
            "nodes": _s_nodes(g, labels, spec)}


def inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    from main.reads.rules import _group
    from main.x35.batch import READ_LABEL

    sides = {**{lab: "on" for lab in p["on"]}, **{lab: "off" for lab in p["off"]}}
    on_arm = spec.models[p["on"][0]].arm
    a = _group(cells, "anchor", p["anchor_group"])
    _group(cells, "s", p["s_group"])                                        # the decided S group must exist
    counted = [u for u in a.get("units") or () if u.get("label") in sides and u.get("counted")]
    doc: Dict[str, Any] = {
        "schema": SCHEMA, "arm": READ_LABEL.get(on_arm, on_arm), "look": LOOK,
        "sides": {"on": {"arm": on_arm, "labels": list(p["on"])},
                  "off": {"arm": spec.models[p["off"][0]].arm, "labels": list(p["off"])}},
        "s": _s_block(cells, p["s_group"], sides, spec),
        "k": {"group": p["anchor_group"], "panels": list(a.get("teamsets") or []),
              "units_per_final": len(a.get("unit_seeds") or []), "regime_id": a.get("regime_id"),
              "inconclusive": list(a.get("inconclusive") or []), "unplayed": int(a.get("unplayed") or 0),
              "engines": sorted({u.get("engine") for u in counted}, key=lambda e: (e is None, str(e))),
              "seeds": {lab: int(spec.models[lab].seed) for lab in sides},
              "cells": {lab: (a.get("cells") or {}).get(lab) or {} for lab in sides}},
        "lr": {lab: dict(mean_lr(run_dir(spec.models[lab])), side=side) for lab, side in sides.items()}}
    sl = p.get("slope")
    if sl:
        lab_side = {**{lab: "on" for lab in sl["on"]}, **{lab: "off" for lab in sl["off"]}}
        doc["slope"] = _s_block(cells, sl["group"], lab_side, spec)
    if p.get("pin_drift"):
        doc["pin_drift"] = dict(p["pin_drift"])
    return doc


# ------------------------------------------------------------------------------------------------ decide
def cmp(stat: float, bar: Any, eps: Any, *, above: bool) -> Tuple[bool, bool]:
    """``(crosses, near)`` by EXACT fractions (``rules_search_headroom.cmp``): within ``eps`` is never a crossing."""
    from main.reads.rules_search_headroom import cmp as _cmp

    return _cmp(stat, bar, eps, above=above)


def _s_reasons(blk: Mapping[str, Any], what: str) -> List[str]:
    why = [f"{what}: {w}" for w in blk.get("inconclusive") or ()]
    if not (isinstance(blk.get("engine"), str) and blk["engine"]):
        why.append(f"{what}: its cells are not on ONE known engine (a meter on two engines is never pooled)")
    floor = int(blk["pairs_registered"])
    for lab, n in sorted((blk.get("nodes") or {}).items()):
        if not _num(n.get("s_pp")):
            why.append(f"{what} {lab}: no S ({'; '.join(n.get('inconclusive') or ['incomplete'])})")
        if not n.get("n_pairs"):
            why.append(f"{what} {lab}: no cell")
        short = {m: k for m, k in sorted((n.get("n_pairs") or {}).items()) if int(k) < floor}
        if short:
            why.append(f"{what} {lab}: cells below the registered {floor} pairs {short}")
    for side in ("on", "off"):
        if sum(1 for n in (blk.get("nodes") or {}).values() if n["side"] == side) < 2:
            why.append(f"{what}: fewer than 2 {side.upper()} finals")
    return why


def _k_reasons(k: Mapping[str, Any], labels: Mapping[str, Sequence[str]]) -> List[str]:
    from main.x35.d1 import min_games

    why = [f"K: {w}" for w in k.get("inconclusive") or ()]
    eng = list(k.get("engines") or [])
    if len(eng) != 1 or not eng[0]:
        why.append(f"K: counted units on {len(eng)} engine(s) {eng} (one known engine, or the meter is not read)")
    if k.get("unplayed"):
        why.append(f"K: {k['unplayed']} unit(s) unplayed")
    if not k.get("panels"):
        why.append("K: no panel")
    for side, labs in labels.items():
        floor = min_games(len(labs), int(k["units_per_final"]))
        for ts in k.get("panels") or ():
            n = sum(int(((k["cells"].get(lab) or {}).get(ts) or {}).get("n", 0)) for lab in labs)
            if n < floor:
                why.append(f"K {side} {ts}: {n} counted games < {floor} (at most one uncounted unit, d1's floor)")
            why += [f"K {side} {lab} {ts}: no counted game" for lab in labs
                    if int(((k["cells"].get(lab) or {}).get(ts) or {}).get("n", 0)) <= 0]
    return why


def welch(on: Sequence[float], off: Sequence[float], confidence: float) -> Dict[str, Any]:
    from main.x35.readout import welch_interval

    w = welch_interval(list(on), list(off), confidence=confidence)
    return {"delta": w.delta, "ci": list(w.ci), "mean_on": w.mean_on, "mean_off": w.mean_off, "df": w.df,
            "se": w.se, "confidence": confidence}


def _sd(xs: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = math.fsum(xs) / len(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def s_power(sd_on: float, sd_off: float, n_on: int, n_off: int, s_bar: float, confidence: float, eps: float
            ) -> Dict[str, Any]:
    """The S meter's P(GO-S) and P(NO-GO-S) at true Δ_S 0 and +``s_bar`` pp, at the observed per-final σ of each arm
    (a seeded Monte Carlo of the registered Welch rule, :data:`POWER_DRAWS` draws, seed :data:`POWER_SEED`)."""
    import numpy as np
    from scipy.stats import t as student_t

    rng = np.random.default_rng(POWER_SEED)
    z_on, z_off = rng.standard_normal((POWER_DRAWS, n_on)), rng.standard_normal((POWER_DRAWS, n_off))
    out: Dict[str, Any] = {"sigma_on_pp": sd_on, "sigma_off_pp": sd_off, "draws": POWER_DRAWS, "seed": POWER_SEED}
    for d in (0.0, float(s_bar)):
        a, b = d + sd_on * z_on, sd_off * z_off
        va, vb = a.var(axis=1, ddof=1) / n_on, b.var(axis=1, ddof=1) / n_off
        v = va + vb
        df = v * v / (va * va / (n_on - 1) + vb * vb / (n_off - 1))
        q = student_t.ppf(1 - (1 - confidence) / 2, df)
        delta = a.mean(axis=1) - b.mean(axis=1)
        lo, hi = delta - q * np.sqrt(v), delta + q * np.sqrt(v)
        go = lo > eps
        out[f"at_{d:+g}pp"] = {"go_s": float(go.mean()), "no_go_s": float((~go & (hi < s_bar - eps)).mean())}
    return out


def _lr_block(doc: Mapping[str, Any], eps: float) -> Dict[str, Any]:
    by = {side: [v.get("mean_lr") for v in doc["lr"].values() if v["side"] == side] for side in ("on", "off")}
    why = [f"{lab}: {v.get('why')}" for lab, v in sorted(doc["lr"].items()) if not _num(v.get("mean_lr"))]
    out: Dict[str, Any] = {"per_final": {lab: dict(v) for lab, v in sorted(doc["lr"].items())},
                           "confounded_below": float(LR_CONFOUND_RATIO), "elevated_above": float(LR_ELEVATED_RATIO)}
    if why:
        return dict(out, status="UNAVAILABLE", reasons=why, confounded=None, elevated=None)
    m_on, m_off = (math.fsum(by[s]) / len(by[s]) for s in ("on", "off"))
    if not m_off > 0:
        return dict(out, status="UNAVAILABLE", reasons=["the OFF arm's mean LR is not > 0"], confounded=None,
                    elevated=None)
    ratio = Fraction(m_on) / Fraction(m_off)
    tol = Fraction(str(eps))
    lo, hi = ratio - LR_CONFOUND_RATIO, ratio - LR_ELEVATED_RATIO
    near = abs(lo) <= tol or abs(hi) <= tol
    return dict(out, status="OK", mean_on=m_on, mean_off=m_off, ratio=float(ratio), near=near,
                confounded=bool(lo < 0 and abs(lo) > tol), elevated=bool(hi > 0 and abs(hi) > tol))


def _meters(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    """Both meters' statistics (pp for S, proportions for K) and their tests."""
    from main.x35.d1 import newcombe_strat, overdispersion, strat_score, widen

    nodes = doc["s"]["nodes"]
    on = [float(n["s_pp"]) for _l, n in sorted(nodes.items()) if n["side"] == "on"]
    off = [float(n["s_pp"]) for _l, n in sorted(nodes.items()) if n["side"] == "off"]
    s = welch(on, off, float(p["confidence"]))
    k = doc["k"]
    panels = tuple(k["panels"])
    labs = {side: list(doc["sides"][side]["labels"]) for side in ("on", "off")}
    cells = {side: {str(k["seeds"][lab]): {ts: (int(k["cells"][lab][ts]["wins"]), int(k["cells"][lab][ts]["n"]))
                                           for ts in panels} for lab in labs[side]} for side in ("on", "off")}
    pooled = {side: {ts: (sum(c[ts][0] for c in cells[side].values()), sum(c[ts][1] for c in cells[side].values()))
                     for ts in panels} for side in ("on", "off")}
    z = float(p["z"])
    sc = {side: strat_score(pooled[side], panels, z) for side in ("on", "off")}
    d, lo0, hi0 = newcombe_strat(sc["on"], sc["off"])
    od = overdispersion(cells["on"], cells["off"], panels)
    lo, hi = widen(d, lo0, hi0, od.factor)
    per_final = {side: [math.fsum(w / n for w, n in cells[side][seed].values()) / len(panels)
                        for seed in sorted(cells[side])] for side in ("on", "off")}
    return {"s": dict(s, on=on, off=off, sigma_on_pp=_sd(on), sigma_off_pp=_sd(off)),
            "k": {"delta": d, "ci": [lo, hi], "ci_unwidened": [lo0, hi0], "phi_hat": od.phi, "od": od.as_dict(),
                  "z": z, "panels": list(panels), "score_on": sc["on"].as_dict(), "score_off": sc["off"].as_dict(),
                  "pooled": {side: {ts: list(v) for ts, v in pooled[side].items()} for side in ("on", "off")},
                  "per_final_on": per_final["on"], "per_final_off": per_final["off"],
                  "sigma_on": _sd(per_final["on"]), "sigma_off": _sd(per_final["off"])}}


def _tests(m: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Tuple[bool, bool]]:
    eps = p["rule8_eps"]
    (s_lo, s_hi), (k_lo, k_hi) = m["s"]["ci"], m["k"]["ci"]
    return {"LB_S > 0": cmp(s_lo, 0, eps, above=True), "UB_S < s_bar": cmp(s_hi, p["s_bar"], eps, above=False),
            "LB_K > 0": cmp(k_lo, 0, eps, above=True), "UB_K < k_bar": cmp(k_hi, p["k_bar"], eps, above=False),
            "UB_S < 0": cmp(s_hi, 0, eps, above=False), "UB_K < 0": cmp(k_hi, 0, eps, above=False)}


def meter_classes(t: Mapping[str, Tuple[bool, bool]], near_scope: str) -> Dict[str, Any]:
    """Each meter's flags and class from the tests; ``near_scope`` "meter" reads a meter with a near GO / NO-GO bound
    UNDECIDED (N4 §5.3), "bound" leaves its other bound deciding (X48 §5.3)."""
    out: Dict[str, Any] = {}
    for m, go_t, nogo_t, cls in (("s", "LB_S > 0", "UB_S < s_bar", (GO_S, NO_GO_S, UNDECIDED_S)),
                                 ("k", "LB_K > 0", "UB_K < k_bar", (GO_K, NO_GO_K, UNDECIDED_K))):
        near = [n for n in (go_t, nogo_t) if t[n][1]]
        go, nogo = t[go_t][0], t[nogo_t][0]
        if near and near_scope == "meter":
            go = nogo = False
        out[m] = {"go": go, "no_go": nogo, "near": near,
                  "class": cls[0] if go else cls[1] if nogo else cls[2]}
    return out


def _pp(x: float) -> str:
    return f"{x:+.2f}"


def _intervals(m: Mapping[str, Any]) -> str:
    s, k = m["s"], m["k"]
    return (f"Δ_S {_pp(s['delta'])} pp [{_pp(s['ci'][0])}, {_pp(s['ci'][1])}] (Welch {s['confidence']:.0%}); "
            f"Δ_K {_pp(100 * k['delta'])} pp [{_pp(100 * k['ci'][0])}, {_pp(100 * k['ci'][1])}] "
            f"(φ̂ {k['phi_hat']:.3f})")


def classify(m: Mapping[str, Any], p: Mapping[str, Any], lr: Mapping[str, Any]) -> Dict[str, Any]:
    """The screen's class in the registered order (MIXED > GO > NO-GO > UNDECIDED), its label and text."""
    t = _tests(m, p)
    mc = meter_classes(t, str(p.get("near_scope", "bound")))
    go_s, go_k = mc["s"]["go"], mc["k"]["go"]
    mixed = bool(p["mixed_veto"]) and ((go_s and t["UB_K < 0"][0]) or (go_k and t["UB_S < 0"][0]))
    text: List[str] = []
    if mixed:
        cls, classed, verdict = MIXED, UNDECIDED, "MIXED (classed UNDECIDED)"
        text.append("GO on one meter while the other's upper bound is below 0 (M2): never GO; to the orchestrator "
                    "and the owner with both intervals")
    elif go_s or go_k:
        cls = classed = GO
        which = "both" if go_s and go_k else "in-house" if go_s else "unfamiliar"
        text.append(f"{GO_TEXT} (Δ̂_S {_pp(m['s']['delta'])} pp, Δ̂_K {_pp(100 * m['k']['delta'])} pp)")
        if lr["status"] != "OK":
            verdict = f"GO ({which}; LR label UNAVAILABLE)"
            text.append(f"the {LR_ELEVATED} label could not be read: {'; '.join(lr['reasons'])}")
        elif lr["elevated"]:
            verdict = f"GO ({which}; {LR_ELEVATED})"
            text.append(f"{LR_ELEVATED} (a weak GO): the ON arm's mean LR is {lr['ratio']:.4f}x the OFF arm's, "
                        f"> {float(LR_ELEVATED_RATIO):g}x")
        else:
            verdict = f"GO ({which})"
    elif mc["s"]["no_go"]:
        cls = classed = NO_GO
        sc = p["nogo_scope"]
        q = f", {sc['qualifier']}" if sc.get("qualifier") else ""
        text += [f"no ≥ {float(p['s_bar']):g} pp in-house gain from {sc['lever']} within {sc['dose']} of "
                 f"{sc['start']}{q}", f"{mc['k']['class']}: {K_ONLY_LINE}"]
        if lr["status"] != "OK":
            verdict = "NO-GO (LR label UNAVAILABLE)"
            text.append(f"the {LR_CONFOUNDED} label could not be read: {'; '.join(lr['reasons'])}")
        elif lr["confounded"]:
            verdict = f"NO-GO ({LR_CONFOUNDED})"
            text.append(f"{LR_CONFOUNDED} (a weak NO-GO): the ON arm's mean LR is {lr['ratio']:.4f}x the OFF "
                        f"arm's, < {float(LR_CONFOUND_RATIO):g}x")
        else:
            verdict = NO_GO
    else:
        cls = classed = verdict = UNDECIDED
    return {"class": cls, "classed_as": classed, "verdict": verdict, "meters": mc, "mixed": mixed,
            "tests": {k: {"crosses": v[0], "near": v[1]} for k, v in t.items()}, "text": text,
            "summary": f"{verdict}: {_intervals(m)}"}


def _slope(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    from main.x35.readout import ReadoutInputError

    blk = doc["slope"]
    why = _s_reasons(blk, "slope")
    if why:
        return {"status": "INCOMPLETE", "reasons": why}
    out: Dict[str, Any] = {"status": "OK", "engine": blk["engine"], "per_final": {}}
    rise: Dict[str, List[float]] = {"on": [], "off": []}
    for lab, n in sorted(blk["nodes"].items()):
        side_fin = {x["seed"]: float(x["s_pp"]) for x in doc["s"]["nodes"].values() if x["side"] == n["side"]}
        r = side_fin[n["seed"]] - float(n["s_pp"])
        rise[n["side"]].append(r)
        out["per_final"][lab] = {"side": n["side"], "seed": n["seed"], "s_node_pp": n["s_pp"],
                                 "s_final_pp": side_fin[n["seed"]], "rise_pp": r}
    try:
        out["delta_slope"] = welch(rise["on"], rise["off"], float(p["confidence"]))
    except ReadoutInputError as exc:
        out["delta_slope"] = {"why": str(exc)}
    return out


def _reported(doc: Mapping[str, Any], p: Mapping[str, Any], m: Mapping[str, Any]) -> Dict[str, Any]:
    from main.x35 import readout as R

    nodes = doc["s"]["nodes"]
    by_seed = {side: {str(n["seed"]): n["s_pp"] for n in nodes.values() if n["side"] == side} for side in ("on", "off")}
    try:
        r1: Any = json.loads(json.dumps(R.decide_r1(LOOK, by_seed["on"], by_seed["off"]).as_dict(), default=str))
    except (R.ReadoutInputError, ValueError) as exc:
        r1 = {"why": str(exc)}
    members = sorted({mb for n in nodes.values() for mb in n["per_member_pp"]})
    per_member: Dict[str, Any] = {}
    for mb in members:
        xs = {side: [float(n["per_member_pp"][mb]) for _l, n in sorted(nodes.items())
                     if n["side"] == side and _num(n["per_member_pp"].get(mb))] for side in ("on", "off")}
        try:
            per_member[mb] = welch(xs["on"], xs["off"], float(p["confidence"]))
        except R.ReadoutInputError as exc:
            per_member[mb] = {"why": str(exc)}
    out: Dict[str, Any] = {
        "r1_look1": r1, "per_member_delta_s": per_member,
        "sigma": {"s_on_pp": m["s"]["sigma_on_pp"], "s_off_pp": m["s"]["sigma_off_pp"],
                  "k_per_final_on": m["k"]["sigma_on"], "k_per_final_off": m["k"]["sigma_off"]},
        "engines": {"s": doc["s"]["engine"], "k": (doc["k"]["engines"] or [None])[0]}}
    s = m["s"]
    if s["sigma_on_pp"] is not None and s["sigma_off_pp"] is not None:
        out["s_power_at_observed_sigma"] = s_power(s["sigma_on_pp"], s["sigma_off_pp"], len(s["on"]), len(s["off"]),
                                                   float(p["s_bar"]), float(p["confidence"]), float(p["rule8_eps"]))
    if "slope" in doc:
        out["slope"] = _slope(doc, p)
    if "pin_drift" in doc:
        pd = doc["pin_drift"]
        out["pin_drift"] = {"off_mean_s_pp": s["mean_off"], "banked_s_pp": pd["s_pp"], "source": pd["source"],
                            "difference_pp": s["mean_off"] - float(pd["s_pp"])}
    return out


def decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    from main.x35.readout import ReadoutInputError

    labels = {side: list(doc["sides"][side]["labels"]) for side in ("on", "off")}
    lr = _lr_block(doc, float(p["rule8_eps"]))
    base: Dict[str, Any] = {"arm": doc["arm"], "look": doc["look"], "final": True, "family_false_go": FAMILY_FALSE_GO,
                            "not_reported": dict(NOT_REPORTED)}
    why = _s_reasons(doc["s"], "S") + _k_reasons(doc["k"], labels)
    m: Optional[Dict[str, Any]] = None
    if not why:
        try:
            m = _meters(doc, p)
        except ReadoutInputError as exc:
            why.append(f"a statistic is undefined: {exc}")
    if why or m is None:
        return dict(base, verdict=INCONCLUSIVE, **{"class": INCONCLUSIVE}, classed_as=INCONCLUSIVE, reasons=why,
                    screen=None, reported={"lr": lr})
    c = classify(m, p, lr)
    rep = _reported(doc, p, m)
    rep["lr"] = lr
    return dict(base, verdict=c["verdict"], **{"class": c["class"]}, classed_as=c["classed_as"], reasons=[],
                screen=dict(c, s=m["s"], k=m["k"]), reported=rep)
