r"""X60-T's registered read as a ``main.reads`` rule ``x60`` (``measurements/x60t_h_teams_2026-11-06/PREREG.md`` §5-§11,
§15; ``design_human_replay_lever.md`` §6.3-§6.6 with REVIEW F's fixes). Pure halves; ``inputs`` reads only the
assembled cells and, when the spec registers it, one sha-pinned guard artefact.

THE PRIMARY Δ_F (§5). ``primary``'s anchor groups hold every ON and OFF final on ``forum72`` at 60 units. Each final's
F rate is wins / games over its COUNTED units (a tie a non-win). :func:`x60_interval` is
``x60_redesign_2026-10-26/power.py``'s ``x60_interval`` PORTED: on equal per-final games it is that function's
arithmetic (``rules_x60_test`` holds the two equal on fixed and simulated inputs), with each arm's pooled rate
Σwins / Σgames where power.py takes the mean of equal-n rates, and each cell's own games in the binomial terms (the
reading of the PREREG's "ON's pooled F rate" when a short final is inside the floor). :func:`classes` and
:func:`rule_x60` are power.py's ``classes`` / ``rule_x60`` for one draw, with its constants (:data:`SIGMA_F`,
:data:`BAND`, :data:`TIE`, :data:`Z`):

* Look 1 (3 v 3, ``look: 1``) is futility only: Δ̂ ≤ :data:`TIE` ENDS the read with look 1's class (never GAIN), else
  CONTINUE. Look 2 (8 v 8) recomputes look 1 on s3501-s3503 (power.py's ``a[:, :3]``), so a look-2 decision IS
  ``rule_x60`` on the same counts.
* THE SEEDS ARE FIXED HERE (REVIEW I M3; :data:`LOOK_SEEDS`, :data:`CONTRAST_SEEDS`): a spec reading other seeds is
  refused at load, a label's run dir must end ``_s<its seed>``, and a run whose ``metadata.json`` records another seed
  makes the decision INCONCLUSIVE.
* GAIN iff LB > TIE; LOSS iff UB < −TIE; NO-USEFUL-GAIN iff UB < BAND − TIE (so an UB in [−TIE, 0) is NO-USEFUL-GAIN,
  power.py's); else NOT-RESOLVED; "equivalent" (inside ±BAND by TIE) reported, deciding nothing.
* The Welch guard's tie band (REVIEW I m3): σ̂²'s one-sided 95 % LB within :data:`GUARD_TIE` of (1 pp)² takes the
  WIDER of the two intervals (power.py's strict ``>`` elsewhere).
* The floor (X13): each arm counts ≥ 20 · units · finals − 20 games on F, every final > 0; else INCONCLUSIVE.

THE OTHER METERS (look 2; at look 1 the S meters given are REPORTED only, X5):

* Δ_P (§6, D1's rule verbatim): ``readout.welch_interval(on, off, confidence=d1.P_CONFIDENCE)`` on S home72,
  NON-INFERIOR iff LB > ``d1.P_BAR_PP`` (``readout._clears``): ``d1.d1_coprimary``'s arithmetic, whose seed-count check
  (3 or 5) does not hold 8 v 8.
* I_F (§7): Welch across finals on S forum72; GAIN iff the two-sided 95 % LB > 0; the guard passes iff the one-sided
  90 % LB (two-sided 80 %) > 0, else KAKUNA-SPECIFIC.
* R72 (§9): Welch 95 % on S R72; DELIVERED iff LB > 0. Decides only at look 2.
* goal (a) (§8): the ON finals' pooled Kakuna ``home72`` (``goal_a``), Wilson 95 % LB > 0.50, at the same floor form.
* G-F / G-E / G-C (§11): read only on a Δ_F GAIN, from the ``guards`` artefact (``{path, sha256}``, schema
  :data:`GUARDS_SCHEMA`, built by :mod:`main.reads.x60_guards`; AM-X60T-3, PREREG §24), each by ``main.x35.readout``'s
  own guard arithmetic, and each INCONCLUSIVE unless its inputs are PROVABLY this read's finals (``finals``: every ON /
  OFF label's sha256 from the state's resolved models):

  - G-F (X35 §5.6.4, launch sheet §4.5): the look's n × n matrix (8 × 8 at look 2), rows the ON finals and columns the
    OFF finals in seed order; every cell names its two finals by sha256, reads OK and holds ≥ ``readout.MIN_PAIRS``
    (500) completed mirrored pairs; ``h_pp`` must equal 100 × each cell's pentanomial pair score (re-derived here).
    ``cross_stat``; FAIL iff its two-sided 95 % upper bound < −2.0 (``guard_f``'s 3 / 5 size table does not hold 8 v 8,
    so the size is checked here against :data:`LOOK_FINALS`).
  - G-E (§23.5): ``r2`` = ``python -m main.x35.ge r2-input --arm x60t`` (pooled stat; readers ``rb_x35_ge_x60t_s3501``
    and ``rb_x35_ge_c0_s3501``), whose ``targets`` name the ON / OFF seed-3501 finals by sha256. FAIL iff the Newcombe
    95 % LOWER bound of gap(ON) − gap(OFF) > 0.
  - G-C (§21.5): the ``main.x35.gc`` record over the look's seed pairs (``gc.check_record``), and every pair's ON / OFF
    ``checkpoint_sha256`` is the read's final of that seed. FAIL iff B1's upper < −0.05 nats or B2's lower > 0.

  The not-played stub (status ``NOT PLAYED``, each block null) reads each INCONCLUSIVE ("not read"); not a GAIN:
  ``guards_not_read`` says why.
* T − D1 (§10, ``contrast``): ON's first five finals against D1's five on H + F at equal weight, units
  0 .. ``units`` − 1 of each, :func:`x60_interval`, one look, always 5 v 5, only at look 2 (X9); LADDER-BETTER iff
  LB > TIE, FORUM-BETTER iff UB < −TIE, NO-DIFFERENCE-AT-3PP iff inside ±BAND by TIE, else NOT-RESOLVED (power.py's
  two-sided ``classes``). Its groups are REPORTED-only: a short contrast is NOT READ, never a blocked record.
* REPORTED Δ_H / Δ_P72 / Δ_home72 (§12; AM-X60T-3, FINDING 28): per team set of the ``reported`` groups, every
  ON and OFF final's counted units with unit seed < :data:`REPORTED_UNITS` (0-6), pooled across the groups that hold
  that team set (:func:`pool_anchor`'s one-cell checks; ``h_contrast``'s units 7-29 are not read here), as Σwins / Σgames
  per arm and its difference. ``reported.pooled``; no verdict reads it, and a short or split one is INCOMPLETE.

THE VERDICT (§11) is :func:`verdict`, a pure function of the meters' classes (``rules_x60_test`` walks every
combination). ADOPT-CANDIDATE / TRADE / KAKUNA-SPECIFIC / LOSS / NO-TRANSFER / IN-HOUSE-ONLY / NOT-DELIVERED /
NO-USEFUL-GAIN (look 1) / NOT-RESOLVED, plus CONTINUE (look 1 sends Δ_F to look 2) and INCONCLUSIVE (a meter the
verdict needs is missing). TRADE holds as soon as one of its conditions is known (P not NON-INFERIOR, goal (a) fails, a
guard FAILs); ADOPT-CANDIDATE needs every one of them known and passing.

SEVERAL GROUPS, ONE CELL (the PREREG §4 import gap). A group takes ONE ``import`` (``spec._import``) and an import reads
only the earlier state's own units (``imports.anchor_rows``), so C0's X61P cells, played in two states, come in as TWO
groups, each with its one import and every check :mod:`main.reads.imports` makes. This rule pools the groups a meter
names (:func:`pool_anchor`, :func:`pool_s`) only when they are ONE cell: one engine, one regime id per team set, one
peer code and policy mode, one Kakuna team pack (anchor); one engine, regime, panel and R (S); each label in exactly one
group. Otherwise the meter is INCONCLUSIVE with the reason.

THE DETERMINISM GUARD (§4, optional ``determinism_guard: {group, against}``): every unit of ``group`` (re-played here) is
compared game for game, choices included (``assemble.compare_choices``), with the same (label, panel, unit seed) row
imported into ``against``; a difference is INCONCLUSIVE ("re-play the imported cells", X7).

THE VOID GATE (§15) is ``rules.void_gate`` (``params.void``): K9(b), ``pin`` (every recorded git hash, so
``pin_history``, is its arm's pin: ``{arm: sha}``, the ON arm at the side pin, C0 at ``f4b25238d``, D1 at
``440b61c32``) and ``dose`` (``main.dose`` against a C0 run, relative 1e-6, §13.3). At look 2 ``void`` and ``guards``
are required, ``void.pin`` is per arm and every final the read reads (ON, C0, D1) is gated (REVIEW I m4). Operator
checks (not in any run dir this module reads): the ``[HUMAN-DATA]`` stamp in every launch log (the run dir holds only
its record), the realised human share, the panel-contamination check, a FATAL naming a corpus team, the per-resume
note.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

SCHEMA = "gen3_read_x60_v1"
#: power.py's constants, unchanged (``rules_x60_test`` reads them from power.py).
Z = 1.959963985
SIGMA_F = 0.010
BAND = 0.03
TIE = 1e-9
#: The registered SEEDS (X1, X9; REVIEW I M3: fixed here, so no later spec can re-pick them): look 1 is ON and C0
#: s3501-s3503; look 2 ON and C0 s3501-s3508, recomputing look 1 on s3501-s3503; the contrast ON and D1 s3501-s3505.
LOOK1_SEEDS = (3501, 3502, 3503)
LOOK_SEEDS = {1: LOOK1_SEEDS, 2: tuple(range(3501, 3509))}
CONTRAST_SEEDS = tuple(range(3501, 3506))
LOOK_FINALS = {k: len(v) for k, v in LOOK_SEEDS.items()}
CONTRAST_FINALS = len(CONTRAST_SEEDS)
#: The Welch guard's tie band (REVIEW I m3, rule 8): σ̂²'s one-sided 95 % LB within this of the floor² takes the WIDER
#: of the registered and the Welch interval.
GUARD_TIE = 1e-12
#: The registered cells: (team set(s), units per final) of each Kakuna meter (§5, §8, §10) and each S meter's panel
#: (§6, §7; R72's panel name is the registration's ``<<TBD>>``, so any).
PRIMARY_CELL = ("forum72", 60)
GOAL_A_CELL = ("home72", 7)
CONTRAST_CELL = (("heldout72_hpfix", "forum72"), 30)
S_PANELS = {"p": "home72", "i_f": "forum72", "r72": None}
UNIT_GAMES = 20
GOAL_A_BAR = 0.5
#: REVIEW F m1's false rates of the contrast on the "continued" path, at σ 1 pp (§10): reported beside the read.
CONTRAST_FALSE_RATES = {"FORUM-BETTER": 0.009, "LADDER-BETTER": 0.027, "sigma_pp": 1.0}

GAIN, LOSS, NUG, NOT_RESOLVED = "GAIN", "LOSS", "NO-USEFUL-GAIN", "NOT-RESOLVED"
ADOPT, TRADE, KSPEC = "ADOPT-CANDIDATE", "TRADE", "KAKUNA-SPECIFIC"
NO_TRANSFER, IN_HOUSE_ONLY, NOT_DELIVERED = "NO-TRANSFER", "IN-HOUSE-ONLY", "NOT-DELIVERED"
NUG_LOOK1, CONTINUE, INCONCLUSIVE = "NO-USEFUL-GAIN (look 1)", "CONTINUE", "INCONCLUSIVE"
LADDER, FORUM, NO_DIFF = "LADDER-BETTER", "FORUM-BETTER", "NO-DIFFERENCE-AT-3PP"
PASS, FAIL = "PASS", "FAIL"
GUARDS = ("G-F", "G-E", "G-C")
#: The guards artefact (AM-X60T-2's stub, AM-X60T-3's filled one: :mod:`main.reads.x60_guards`).
GUARDS_SCHEMA = "gen3_x60t_guards_v1"
NOT_PLAYED, PLAYED = "NOT PLAYED", "PLAYED"
GUARD_BLOCKS = ("g_f", "r2", "g_c")
#: §23.5: G-E's two readers, each of its side's seed-3501 final (``main.x35.ge``).
GE_READERS = {"on": "ge_x60t", "off": "ge_c0"}
#: §12: the REPORTED Δ_H / Δ_P72 / Δ_home72 read units 0-6 of every final, both arms.
REPORTED_UNITS = 7
REPORTED_NAMES = {"heldout72_hpfix": "Δ_H", "p72": "Δ_P72", "home72": "Δ_home72"}
VERDICTS = (ADOPT, TRADE, KSPEC, LOSS, NO_TRANSFER, IN_HOUSE_ONLY, NOT_DELIVERED, NUG_LOOK1, NOT_RESOLVED, CONTINUE,
            INCONCLUSIVE)

KEYS = ("look", "on", "off", "primary")
OPTIONAL = ("p", "i_f", "r72", "goal_a", "contrast", "reported", "guards", "determinism_guard", "void")
#: Required at look 2 (the meters the verdict reads, the guards' artefact and the VOID gate; REVIEW I m4); at look 1
#: only i_f / r72 may be given (REPORTED, X5).
LOOK2_REQUIRED = ("p", "i_f", "r72", "goal_a", "guards", "void")
RULE_DOCS = {"x60": "X60-T (PREREG §5-§11): Δ_F by power.py's x60 interval (futility-only look 1 at 3 v 3, look 2 at "
                    "8 v 8), I_F (LB90 guard), Δ_P (D1's), goal (a), R72, T − D1 (5 v 5, reported) and the §11 verdict"}


# ------------------------------------------------------------------------------------------------ the interval
def x60_interval(on: Sequence[Sequence[Tuple[int, int]]], off: Sequence[Sequence[Tuple[int, int]]]
                 ) -> Dict[str, Any]:
    """power.py's ``x60_interval`` for ONE draw: ``on`` / ``off`` = per final, per panel ``(wins, games)``. Returns
    ``{delta, halfwidth, welch_used, ...}`` (proportions). Equal finals per arm (power.py's ``ns``)."""
    import numpy as np
    from scipy.stats import chi2, t as tdist

    if len(on) != len(off) or len(on) < 2:
        raise ValueError(f"x60_interval needs equal finals per arm, >= 2 (got {len(on)} v {len(off)})")
    aw, an = np.array([[w for w, _ in f] for f in on], float), np.array([[n for _, n in f] for f in on], float)
    cw, cn = np.array([[w for w, _ in f] for f in off], float), np.array([[n for _, n in f] for f in off], float)
    if aw.shape != cw.shape or (an <= 0).any() or (cn <= 0).any():
        raise ValueError("every final needs games on every panel, the same panels on both arms")
    ns, k = aw.shape
    a, c = aw / an, cw / cn                                   # per (final, panel) rates
    pa, pc = aw.sum(0) / an.sum(0), cw.sum(0) / cn.sum(0)     # pooled per panel (power.py: a.mean(1) at equal n)
    d = pa.mean() - pc.mean()
    vbin = ((pa * (1 - pa) / an.sum(0)).sum() + (pc * (1 - pc) / cn.sum(0)).sum()) / k ** 2
    ma, mc = a.mean(1), c.mean(1)
    w = (ma.var(ddof=1) + mc.var(ddof=1)) / 2
    b = (((a * (1 - a)) / an).sum(1).mean() + ((c * (1 - c)) / cn).sum(1).mean()) / 2 / k ** 2
    df = 2 * (ns - 1)
    s2 = max(SIGMA_F ** 2, w - b)
    hw = Z * math.sqrt(vbin + 2 * s2 / ns)
    guard_lb = w * df / chi2.ppf(0.95, df) - b
    guard_near = bool(abs(guard_lb - SIGMA_F ** 2) <= GUARD_TIE)
    guard = bool(guard_lb > SIGMA_F ** 2) and not guard_near
    va, vc = ma.var(ddof=1) / ns, mc.var(ddof=1) / ns
    dfw = (va + vc) ** 2 / max(va ** 2 / (ns - 1) + vc ** 2 / (ns - 1), 1e-30)
    hw_w = float(tdist.ppf(0.975, dfw) * math.sqrt(va + vc))
    if guard_near:                                            # REVIEW I m3: the tie band takes the WIDER interval
        guard = bool(math.isfinite(hw_w) and hw_w > hw)
    half = hw_w if guard else hw
    return {"delta": float(d), "halfwidth": float(half), "lo": float(d - half), "hi": float(d + half),
            "welch_used": guard, "guard_near": guard_near, "halfwidth_registered": float(hw), "halfwidth_welch": hw_w,
            "welch_df": float(dfw),
            "sigma_hat_sq": float(w - b), "sigma_hat": math.sqrt(max(0.0, float(w - b))), "W": float(w),
            "b": float(b), "df": df, "sigma2_lb95": float(guard_lb), "vbin": float(vbin), "finals": ns, "panels": k,
            "pooled_on": [float(x) for x in pa], "pooled_off": [float(x) for x in pc],
            "per_final_on": [float(x) for x in ma], "per_final_off": [float(x) for x in mc],
            "sd_on": float(ma.std(ddof=1)), "sd_off": float(mc.std(ddof=1))}


def classes(lo: float, hi: float, *, one_sided: bool = True) -> str:
    """power.py's ``classes`` for one interval: GAIN / LOSS / NO-USEFUL-GAIN (``one_sided``) or NO-DIFF / NOT-RESOLVED."""
    if lo > TIE:
        return GAIN
    if hi < -TIE:
        return LOSS
    small = (hi < BAND - TIE) if one_sided else (lo > -BAND + TIE) and (hi < BAND - TIE)
    return (NUG if one_sided else "NO-DIFF") if small else NOT_RESOLVED


def equivalent(lo: float, hi: float) -> bool:
    return lo > -BAND + TIE and hi < BAND - TIE


def rule_x60(on: Sequence[Sequence[Tuple[int, int]]], off: Sequence[Sequence[Tuple[int, int]]],
             look1: Sequence[int]) -> Dict[str, Any]:
    """power.py's ``rule_x60`` for one draw: look 1 on the finals at indices ``look1`` (power.py: the first three),
    look 2 on all. ``{class, stop, look1, look2, equivalent}``; ``look2`` is None when ``on`` holds only look 1, and
    ``class`` is then CONTINUE unless look 1 stopped."""
    s1 = x60_interval([on[i] for i in look1], [off[i] for i in look1])
    c1 = classes(s1["lo"], s1["hi"])
    c1 = NOT_RESOLVED if c1 == GAIN else c1                   # never GAIN at look 1 (cannot occur with Δ̂ <= 0)
    stop = s1["delta"] <= TIE
    s2 = x60_interval(on, off) if len(on) > len(look1) else None
    if stop or s2 is None:
        final, src = (c1 if stop else CONTINUE), s1
    else:
        final, src = classes(s2["lo"], s2["hi"]), s2
    return {"class": final, "stop": stop, "look1": dict(s1, look1_class=c1), "look2": s2,
            "equivalent": equivalent(src["lo"], src["hi"])}


# ------------------------------------------------------------------------------------------------ the verdict
def verdict(look: int, primary: Optional[Mapping[str, Any]], *, i_f: Optional[Mapping[str, bool]] = None,
            r72: Optional[bool] = None, p: Optional[str] = None, goal_a: Optional[bool] = None,
            guards: Optional[Mapping[str, str]] = None) -> Tuple[str, List[str]]:
    """§11's table. ``primary`` = ``{class, stop}`` (None: INCONCLUSIVE); ``i_f`` = ``{gain, lb90_pos}``; ``r72`` =
    DELIVERED; ``p`` = ``d1.NON_INFERIOR`` / ``d1.NOT_EXCLUDED``; ``goal_a`` = passes; ``guards`` = ``{G-F: PASS |
    FAIL | INCONCLUSIVE, ...}``. None = not known. ``(verdict, reasons)``."""
    from main.x35 import d1 as D

    if primary is None:
        return INCONCLUSIVE, ["Δ_F is INCONCLUSIVE"]
    cls = primary["class"]
    if primary["stop"]:
        word = {LOSS: LOSS, NUG: NUG_LOOK1}.get(cls, NOT_RESOLVED)
        return word, [f"look-1 futility stop (Δ̂ <= 0): Δ_F {cls}" + (" (undivided, X5)" if word == NUG_LOOK1 else "")]
    if look == 1:
        return CONTINUE, ["look 1: Δ̂ > 0, Δ_F goes to look 2"]
    if cls == LOSS:
        return LOSS, ["Δ_F LOSS at look 2"]
    if cls == NOT_RESOLVED:
        return NOT_RESOLVED, ["Δ_F NOT-RESOLVED at look 2"]
    if cls == NUG:
        if r72 is None:
            return INCONCLUSIVE, ["Δ_F NO-USEFUL-GAIN, but R72 is not read"]
        if not r72:
            return NOT_DELIVERED, ["Δ_F NO-USEFUL-GAIN, R72 not DELIVERED"]
        if i_f is None:
            return INCONCLUSIVE, ["Δ_F NO-USEFUL-GAIN, R72 DELIVERED, but I_F is not read"]
        if i_f["gain"]:
            return IN_HOUSE_ONLY, ["Δ_F NO-USEFUL-GAIN, R72 DELIVERED, I_F GAIN"]
        return NO_TRANSFER, ["Δ_F NO-USEFUL-GAIN, R72 DELIVERED, I_F not GAIN"]
    if cls != GAIN:
        raise ValueError(f"unknown Δ_F class {cls!r}")
    if i_f is None:
        return INCONCLUSIVE, ["Δ_F GAIN, but I_F is not read"]
    if not i_f["lb90_pos"]:
        return KSPEC, ["Δ_F GAIN, I_F's one-sided 90 % LB <= 0"]
    g = {n: (guards or {}).get(n) for n in GUARDS}
    bad, open_ = [], []
    if p is None:
        open_.append("Δ_P not read")
    elif p != D.NON_INFERIOR:
        bad.append("Δ_P: inferiority not excluded")
    if goal_a is None:
        open_.append("goal (a) not read")
    elif not goal_a:
        bad.append("goal (a) fails")
    bad += [f"{n} FAIL" for n in GUARDS if g[n] == FAIL]
    open_ += [f"{n} {g[n] or 'not read'}" for n in GUARDS if g[n] not in (PASS, FAIL)]
    if bad:
        return TRADE, ["Δ_F GAIN, I_F LB90 > 0, but " + "; ".join(bad)]
    if open_:
        return INCONCLUSIVE, ["Δ_F GAIN, I_F LB90 > 0, but " + "; ".join(open_)]
    return ADOPT, ["Δ_F GAIN, I_F LB90 > 0, Δ_P NON-INFERIOR, goal (a), G-F / G-E / G-C PASS"]


# ------------------------------------------------------------------------------------------------ the stage record
#: The key of Δ_F's ``{final, class}`` at a record's top level (``main.x35.batch.PRIMARY_KEY["x60t"]``).
STAGE_PRIMARY_KEY = "delta_f"


def stage_keys(models: Mapping[str, Any], decision: Mapping[str, Any]) -> Dict[str, Any]:
    """The X35 stage identity of an x60 record (``Rule.stage_keys``; AM-X60T-2): ``arm`` = the ON arm's registered
    read label (``main.x35.batch.READ_LABEL``: X60T), ``look``, and Δ_F's ``{final, class}``: FINAL at a look-1 futility
    stop or at look 2 whenever Δ_F has a class; NOT final at a look-1 CONTINUE or when Δ_F is INCONCLUSIVE (no class).
    Read from the record's ``models`` and ``decision`` only, so the decision is never reshaped."""
    from main.x35.batch import READ_LABEL

    on = list((decision.get("sides") or {}).get("on") or ())
    arms = sorted({str((models.get(lab) or {}).get("arm")) for lab in on})
    if len(arms) != 1 or arms == ["None"]:
        raise ValueError(f"the ON side {on} names arms {arms}, not one")
    look = int(decision["look"])
    dec = decision.get("decision") or {}
    cls = dec.get("delta_f_class")
    final = cls is not None and (bool(dec.get("stop_at_look1")) or (look == 2 and cls != CONTINUE))
    return {"arm": READ_LABEL.get(arms[0], arms[0]), "look": look,
            STAGE_PRIMARY_KEY: {"final": final, "class": cls}}


# ------------------------------------------------------------------------------------------------ check (spec load)
def _labels(name: str, labs: Any, models: Any) -> Tuple[List[str], List[str]]:
    if not (isinstance(labs, list) and labs and all(isinstance(v, str) for v in labs) and len(set(labs)) == len(labs)):
        return [f"{name}: a list of distinct model labels"], []
    probs = [f"{name}: {lab!r} is not a model label" for lab in labs if lab not in models]
    if probs:
        return probs, []
    probs += [f"{name}: {lab!r} carries load options (the read plays greedy finals)" for lab in labs
              if models[lab].load]
    arms = sorted({models[lab].arm for lab in labs})
    if len(arms) != 1:
        probs.append(f"{name}: its models span arms {arms} (one arm a side)")
    if len({models[lab].seed for lab in labs}) != len(labs):
        probs.append(f"{name}: two models share a seed")
    return probs, arms


def _groups(what: str, spec_groups: Any, d: Any) -> List[str]:
    gs = d.get("groups") if isinstance(d, dict) else None
    if not (isinstance(gs, list) and gs and all(isinstance(g, str) for g in gs) and len(set(gs)) == len(gs)):
        return [f"{what}.groups: a list of distinct group names"]
    return [f"{what}.groups: {g!r} is not a group of its lane" for g in gs if g not in spec_groups]


def _anchor_meter(what: str, d: Any, anchor: Any, labs: Sequence[str], keys: Sequence[str]) -> List[str]:
    if not isinstance(d, dict) or set(d) != set(keys):
        return [f"{what}: exactly {{{', '.join(keys)}}}"]
    probs = _groups(what, anchor, d)
    if probs:
        return probs
    if not (isinstance(d.get("units"), int) and not isinstance(d["units"], bool) and d["units"] > 0):
        probs.append(f"{what}.units: a positive integer")
        return probs
    sets = [d["teamset"]] if "teamset" in d else list(d.get("panels") or [])
    if not sets or not all(isinstance(t, str) for t in sets):
        return probs + [f"{what}: its team set(s)"]
    for t in sets:
        holders = [g for g in d["groups"] if t in anchor[g].teamsets]
        if not holders:
            probs.append(f"{what}: no group plays {t}")
        for lab in labs:
            n = sum(lab in anchor[g].models for g in holders)
            if n != 1:
                probs.append(f"{what}: {lab} plays {t} in {n} of its groups (exactly one)")
        for g in holders:
            if not set(range(d["units"])) <= set(anchor[g].unit_seeds):
                probs.append(f"{what}: group {g} does not plan unit seeds 0-{d['units'] - 1}")
            if getattr(anchor[g], "per_game", False):     # absent at the play pin de15dfc7b (no per_game groups)
                probs.append(f"{what}: group {g} is per_game (the read plays 20-game units)")
    return probs


def _s_meter(what: str, d: Any, s: Any, labs: Sequence[str]) -> List[str]:
    if not isinstance(d, dict) or set(d) != {"groups", "panel"}:
        return [f"{what}: exactly {{groups, panel}}"]
    probs = _groups(what, s, d)
    if probs:
        return probs
    for g in d["groups"]:
        if s[g].panel != d["panel"]:
            probs.append(f"{what}: S group {g} plays panel {s[g].panel!r}, not {d['panel']!r}")
    for lab in labs:
        n = sum(lab in s[g].models for g in d["groups"])
        if n != 1:
            probs.append(f"{what}: {lab} is in {n} of its groups (exactly one)")
    return probs


def _registered(what: str, d: Any, teamset: str, units: int) -> List[str]:
    if isinstance(d, dict) and (d.get("teamset"), d.get("units")) != (teamset, units):
        return [f"{what}: teamset / units are not the registered {teamset} / {units}"]
    return []


def check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    from main.reads.rules import _check_keys, _void_check

    probs = _check_keys(p, KEYS, KEYS + OPTIONAL)
    look = p.get("look")
    if look not in LOOK_FINALS:
        return probs + [f"look: one of {sorted(LOOK_FINALS)}"]
    pr_on, on_arms = _labels("on", p.get("on"), models)
    pr_off, off_arms = _labels("off", p.get("off"), models)
    probs += pr_on + pr_off
    if pr_on or pr_off:
        return probs
    on, off = list(p["on"]), list(p["off"])
    if set(on_arms) & set(off_arms) or set(on) & set(off):
        probs.append("on / off: one arm or label on both sides")
    want = LOOK_FINALS[look]
    if len(on) != want or len(off) != want:
        probs.append(f"look {look} registers {want} v {want} finals (got {len(on)} v {len(off)})")
    probs += _seed_set("on", on, models, LOOK_SEEDS[look]) + _seed_set("off", off, models, LOOK_SEEDS[look])
    probs += _seed_ties(on + off, models)
    probs += _anchor_meter("primary", p.get("primary"), anchor, on + off, ("groups", "teamset", "units"))
    probs += _registered("primary", p.get("primary"), *PRIMARY_CELL)
    for key in LOOK2_REQUIRED:
        if look == 2 and p.get(key) is None:
            probs.append(f"{key}: required at look 2")
        if look == 1 and key in ("p", "goal_a") and p.get(key) is not None:
            probs.append(f"{key}: look 1 is futility only on Δ_F (§5)")
    for key in ("p", "i_f", "r72"):
        if p.get(key) is not None:
            probs += _s_meter(key, p[key], s, on + off)
            if S_PANELS[key] is not None and isinstance(p[key], dict) and p[key].get("panel") != S_PANELS[key]:
                probs.append(f"{key}.panel: the registered {S_PANELS[key]} (§6-§7)")
    if p.get("goal_a") is not None:
        probs += _anchor_meter("goal_a", p["goal_a"], anchor, on, ("groups", "teamset", "units"))
        probs += _registered("goal_a", p["goal_a"], *GOAL_A_CELL)
    c = p.get("contrast")
    if c is not None:
        if look == 1:
            probs.append("contrast: read only after look 2, 5 v 5 (X9)")
        elif not isinstance(c, dict) or set(c) != {"on", "off", "groups", "panels", "units"}:
            probs.append("contrast: exactly {on, off, groups, panels, units}")
        else:
            pc_on, _ = _labels("contrast.on", c["on"], models)
            pc_off, c_off_arms = _labels("contrast.off", c["off"], models)
            probs += pc_on + pc_off
            if not (pc_on or pc_off):
                if not set(c["on"]) <= set(on):
                    probs.append("contrast.on: ON finals of this read")
                if len(c["on"]) != CONTRAST_FINALS or len(c["off"]) != CONTRAST_FINALS:
                    probs.append(f"contrast: always {CONTRAST_FINALS} v {CONTRAST_FINALS} (X9)")
                probs += _seed_set("contrast.on", c["on"], models, CONTRAST_SEEDS)
                probs += _seed_set("contrast.off", c["off"], models, CONTRAST_SEEDS)
                probs += _seed_ties(list(c["off"]), models)
                if set(c_off_arms) & (set(on_arms) | set(off_arms)):
                    probs.append("contrast.off: D1's finals, an arm of neither side of the primary")
                probs += _anchor_meter("contrast", {"groups": c["groups"], "panels": c["panels"], "units": c["units"]},
                                       anchor, list(c["on"]) + list(c["off"]), ("groups", "panels", "units"))
                if (tuple(c["panels"]) if isinstance(c["panels"], list) else None, c["units"]) != CONTRAST_CELL:
                    probs.append(f"contrast: panels / units are not the registered {CONTRAST_CELL} (§10)")
    rep = p.get("reported", [])
    if not (isinstance(rep, list) and all(isinstance(g, str) and g in anchor for g in rep) and len(set(rep)) == len(rep)):
        probs.append("reported: a list of distinct anchor groups")
    gd = p.get("guards")
    if gd is not None and not (isinstance(gd, dict) and set(gd) == {"path", "sha256"}
                               and isinstance(gd["path"], str) and isinstance(gd["sha256"], str)):
        probs.append("guards: {path, sha256} of the G-F / G-E / G-C inputs (read only on a Δ_F GAIN)")
    dg = p.get("determinism_guard")
    if dg is not None and not (isinstance(dg, dict) and set(dg) == {"group", "against"}
                               and dg.get("group") in anchor and dg.get("against") in anchor
                               and dg["group"] != dg["against"] and anchor[dg["against"]].imports is not None
                               and anchor[dg["group"]].imports is None):
        probs.append("determinism_guard: {group (re-played here), against (the group importing those units)}")
    if p.get("void") is not None and len(on_arms) == 1:
        probs += _void_check(dict(p, on=on_arms[0]), models, required=False)
        v = p["void"]
        if look == 2 and isinstance(v, dict) and isinstance(v.get("runs"), dict):
            gated = on + off + (list(c["off"]) if isinstance(c, dict) and isinstance(c.get("off"), list) else [])
            probs += [f"void.runs: {lab!r} has no run (at look 2 every final the read reads is gated, REVIEW I m4)"
                      for lab in gated if lab not in v["runs"]]
            if not isinstance(v.get("pin"), dict):
                probs.append("void.pin: at look 2 a pin per arm {arm: sha} (the side pin, f4b25238d, 440b61c32)")
    return probs


def _seed_set(name: str, labs: Sequence[str], models: Any, want: Sequence[int]) -> List[str]:
    got = sorted(int(models[lab].seed) for lab in labs if lab in models)
    return [] if got == sorted(want) else [f"{name}: seeds {got} are not the registered {list(want)} (REVIEW I M3)"]


def run_dir_of(ref: Any) -> Optional[str]:
    """The run dir a model ref names: a ``final`` ref's ``run``, a ``pinned`` ref's file's directory (``<run>/x.zip``)
    or its run (``<run>/checkpoints/x.zip``)."""
    import posixpath

    if getattr(ref, "kind", None) == "final":
        return str(ref.run)
    path = str(getattr(ref, "path", "") or "")
    d = posixpath.dirname(path)
    return posixpath.dirname(d) if posixpath.basename(d) == "checkpoints" else d


def path_seed(ref: Any) -> Optional[int]:
    """The ``_s<seed>`` the ref's run dir name ends with (``rb_x35_c0_s3501``), or None."""
    import posixpath
    import re

    m = re.search(r"_s(\d+)$", posixpath.basename(str(run_dir_of(ref) or "").rstrip("/")))
    return int(m.group(1)) if m else None


def _seed_ties(labs: Sequence[str], models: Any) -> List[str]:
    """REVIEW I M3: each label's seed is its run's: the ``_s<seed>`` of the ref's run dir equals the spec's seed (the
    run's RECORDED seed is checked at ``inputs``, where the run dir may be read)."""
    out = []
    for lab in labs:
        if lab not in models:
            continue
        got = path_seed(models[lab].ref)
        if got != int(models[lab].seed):
            out.append(f"{lab}: its run {run_dir_of(models[lab].ref)!r} names seed {got}, the spec {models[lab].seed}")
    return out


def recorded_seeds(spec: Any, labs: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """``{label: {run, recorded, ok}}``: each run's recorded seed (``metadata.json`` ``cli_args.seed``) against the
    spec's; ``recorded`` None (and ``ok`` True) where the record is not readable here (REVIEW I M3: "where readable")."""
    from main.reads.models import spec_path

    out: Dict[str, Dict[str, Any]] = {}
    for lab in labs:
        m = spec.models[lab]
        run = run_dir_of(getattr(m, "ref", None)) if getattr(m, "ref", None) is not None else None
        rec = None
        if run:
            try:
                rec = json.loads((spec_path(run) / "metadata.json").read_text()).get("cli_args", {}).get("seed")
            except (OSError, ValueError, AttributeError):
                rec = None
        ok = rec is None or (isinstance(rec, int) and not isinstance(rec, bool) and rec == int(m.seed))
        out[lab] = {"run": run, "spec": int(m.seed), "recorded": rec, "ok": ok}
    return out


def reported(p: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """The REPORTED-only groups: ``reported``; at look 1 the S meters given (X5); the contrast's groups that decide
    nothing else (T4: a trimmed read still decides)."""
    deciding = set((p.get("primary") or {}).get("groups") or ()) | set((p.get("goal_a") or {}).get("groups") or ())
    dg = p.get("determinism_guard") or {}
    deciding |= {g for g in (dg.get("group"), dg.get("against")) if g}
    out = [("anchor", str(g)) for g in p.get("reported") or () if g not in deciding]
    if p.get("look") == 1:
        out += [("s", str(g)) for key in ("i_f", "r72") for g in ((p.get(key) or {}).get("groups") or ())]
    out += [("anchor", str(g)) for g in ((p.get("contrast") or {}).get("groups") or ())
            if g not in deciding and ("anchor", str(g)) not in out]
    return out


# ------------------------------------------------------------------------------------------------ pooling groups
def pool_anchor(cells: Mapping[str, Any], groups: Sequence[str], teamsets: Sequence[str], labels: Sequence[str]
                ) -> Dict[str, Any]:
    """Each label's counted ``(wins, ties, games, units)`` per team set across ``groups``, and every reason they are not
    ONE cell (module docstring)."""
    from main.reads.side_identity import SIDE_KEYS

    why: List[str] = []
    got: Dict[str, Dict[str, Dict[str, int]]] = {lab: {} for lab in labels}
    engines: Set[Any] = set()
    sides: Dict[str, Set[Any]] = {k: set() for k in SIDE_KEYS}
    regimes: Dict[str, Set[Any]] = {t: set() for t in teamsets}
    packs: Dict[str, Set[str]] = {t: set() for t in teamsets}
    rows: List[Dict[str, Any]] = []
    for g in groups:
        grp = (cells.get("anchor") or {}).get(g)
        if grp is None:
            why.append(f"anchor.{g}: not assembled")
            continue
        why += [f"anchor.{g}: {w}" for w in grp.get("inconclusive") or ()]
        if grp.get("unplayed"):
            why.append(f"anchor.{g}: {grp['unplayed']} unit(s) unplayed")
        mine = [t for t in teamsets if t in (grp.get("teamsets") or ())]
        if not mine:
            continue
        engines.add(grp.get("engine"))
        for k in SIDE_KEYS:
            sides[k].add(grp.get(k))
        for t in mine:
            regimes[t].add((grp.get("regime_ids") or {}).get(t))
        for r in grp.get("units") or ():
            if r.get("counted") and r.get("panel") in mine:
                packs[r["panel"]].add(str(r.get("peer_pack", "legacy")))
            if r.get("label") in got and r.get("panel") in mine:
                rows.append(dict(r, group=g))
        for lab in labels:
            for t in mine:
                c = ((grp.get("cells") or {}).get(lab) or {}).get(t)
                if c is None:
                    continue
                if t in got[lab]:
                    why.append(f"{lab} {t}: in two of the pooled groups")
                    continue
                got[lab][t] = {"wins": int(c["wins"]), "ties": int(c.get("ties", 0)), "n": int(c["n"]),
                               "units": int(c["units"]), "skipped": len(c.get("skipped") or ())}
    if None in engines or len(engines) != 1:
        why.append(f"the pooled groups are not on ONE known engine: {sorted(str(e) for e in engines)}")
    for k, v in sides.items():
        if None in v or len(v) != 1:
            why.append(f"the pooled groups' {k.replace('_', ' ')} is not one known value: {sorted(map(str, v))}")
    for t in teamsets:
        if None in regimes[t] or len(regimes[t]) != 1:
            why.append(f"{t}: the pooled groups' regime ids are not one: {sorted(map(str, regimes[t]))}")
        if len(packs[t]) > 1:
            why.append(f"{t}: the pooled groups span Kakuna team packs {sorted(packs[t])}")
    why += [f"{lab} {t}: in none of the pooled groups" for lab in labels for t in teamsets if t not in got[lab]]
    return {"groups": list(groups), "counts": got, "engine": next(iter(engines)) if len(engines) == 1 else None,
            "regime_ids": {t: sorted(map(str, regimes[t])) for t in teamsets}, "rows": rows,
            "inconclusive": sorted(set(why))}


def pool_s(cells: Mapping[str, Any], groups: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
    """Each label's S node across ``groups``, and every reason they are not ONE S cell."""
    why: List[str] = []
    nodes: Dict[str, Dict[str, Any]] = {}
    seen: Dict[str, Set[Any]] = {k: set() for k in ("engine", "regime_id", "panel", "refpop_content_sha256")}
    for g in groups:
        grp = (cells.get("s") or {}).get(g)
        if grp is None:
            why.append(f"s.{g}: not assembled")
            continue
        why += [f"s.{g}: {w}" for w in grp.get("inconclusive") or ()]
        for k in seen:
            seen[k].add(grp.get(k))
        for lab, node in (grp.get("nodes") or {}).items():
            if lab not in labels:
                continue
            if lab in nodes:
                why.append(f"{lab}: an S node in two of the pooled groups")
            nodes[lab] = {"s_pp": node.get("s_pp"), "se_pp": node.get("se_pp"), "group": g}
    for k, v in seen.items():
        if None in v or len(v) != 1:
            why.append(f"the pooled S groups' {k} is not one known value: {sorted(map(str, v))}")
    for lab in labels:
        n = nodes.get(lab)
        if n is None:
            why.append(f"{lab}: no S node in the pooled groups")
        elif not (isinstance(n["s_pp"], (int, float)) and math.isfinite(n["s_pp"])):
            why.append(f"{lab}: no S")
    return {"groups": list(groups), "nodes": nodes, "engine": next(iter(seen["engine"])) if len(seen["engine"]) == 1
            else None, "regime_id": next(iter(seen["regime_id"])) if len(seen["regime_id"]) == 1 else None,
            "inconclusive": sorted(set(why))}


def unit_counts(rows: Sequence[Mapping[str, Any]], labels: Sequence[str], panels: Sequence[str], units: int
                ) -> Dict[str, Dict[str, Dict[str, int]]]:
    """``{label: {panel: {wins, n, units}}}`` over the COUNTED unit rows with unit seed < ``units`` (the contrast's
    0-29 nested in the primary's 0-59)."""
    out = {lab: {t: {"wins": 0, "n": 0, "units": 0} for t in panels} for lab in labels}
    for r in rows:
        if r.get("counted") and r.get("label") in out and r.get("panel") in panels and int(r["unit_seed"]) < units:
            c = out[r["label"]][r["panel"]]
            c["wins"] += int(r["wins"])
            c["n"] += int(r["n"])
            c["units"] += 1
    return out


# ------------------------------------------------------------------------------------------------ inputs
def _seeds(spec: Any, labs: Sequence[str]) -> Dict[str, int]:
    return {lab: int(spec.models[lab].seed) for lab in labs}


def _guard_doc(gd: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if gd is None:
        return {"registered": False}
    from main.reads.models import spec_path
    from utils.hashing import sha256_file

    path = spec_path(str(gd["path"]))
    try:
        got = sha256_file(path)
    except OSError as exc:
        return {"registered": True, "error": f"guards artefact {path}: {exc}"}
    if got != gd["sha256"]:
        return {"registered": True, "error": f"guards artefact {path}: sha256 {got[:12]} != the spec's "
                                             f"{str(gd['sha256'])[:12]}"}
    return {"registered": True, "path": str(gd["path"]), "sha256": got, "doc": json.loads(path.read_text())}


def _determinism(cells: Mapping[str, Any], dg: Mapping[str, Any]) -> Dict[str, Any]:
    """§4's guard: each unit of ``dg.group`` against the same (label, panel, unit seed) row imported into ``dg.against``
    (5-tuple and both players' choices; ``assemble.compare_choices``)."""
    from pathlib import Path

    from main.reads.assemble import compare_choices, five_tuple_ok

    g = (cells.get("anchor") or {}).get(dg["group"]) or {}
    a = (cells.get("anchor") or {}).get(dg["against"]) or {}
    bank = {(r["label"], r["panel"], int(r["unit_seed"])): r for r in a.get("units") or () if r.get("imported_from")}
    checks, why = [], []
    for u in g.get("units") or ():
        want = bank.get((u["label"], u["panel"], int(u["unit_seed"])))
        row: Dict[str, Any] = {"unit": u["unit"], "counted": bool(u.get("counted")), "banked": None if want is None
                               else {"unit": want["unit"], "imported_from": want["imported_from"]}}
        if want is None:
            why.append(f"{u['unit']}: no imported row of the same label, panel and unit seed in {dg['against']}")
        elif not u.get("counted") or not want.get("counted"):
            why.append(f"{u['unit']}: not counted (here or imported)")
        else:
            teq = [list(x) for x in u.get("games") or ()] == [list(x) for x in want.get("games") or ()] \
                and bool(u.get("games"))
            ft = five_tuple_ok(cells)
            if not teq:
                ch = {"choices": "not compared", "differing": [], "why": "the 5-tuples differ"}
            elif not want.get("cell_dir"):            # an imported unit names its capture: missing fails closed (H2)
                ch = {"choices": "unavailable" if ft else "inconclusive", "differing": [],
                      "why": "the imported row names no cell dir"}
            else:
                ch = compare_choices(Path(u["cell_dir"]) if u.get("cell_dir") else None, Path(want["cell_dir"]),
                                     want["games"], five_tuple_ok=ft)
            row.update(tuple_equal=teq, choices=ch["choices"], choices_why=ch["why"])
            if not (teq and ch["choices"] in ("equal", "unavailable")):
                why.append(f"{u['unit']}: games differ from the imported {want['unit']} ({ch['choices']}: "
                           f"{ch['why']})")
        checks.append(row)
    if not checks:
        why.append(f"no unit of {dg['group']} played")
    return {"group": dg["group"], "against": dg["against"], "units": checks, "passed": not why, "reasons": why}


def inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    on, off = list(p["on"]), list(p["off"])
    pr = p["primary"]
    doc: Dict[str, Any] = {"schema": SCHEMA, "look": p["look"], "sides": {"on": on, "off": off},
                           "seeds": _seeds(spec, on + off), "look1_seeds": list(LOOK1_SEEDS),
                           "primary": dict(pool_anchor(cells, pr["groups"], [pr["teamset"]], on + off),
                                           teamset=pr["teamset"], units=pr["units"])}
    doc["primary"].pop("rows")
    gated = on + off + list((p.get("contrast") or {}).get("off") or ())
    doc["recorded_seeds"] = recorded_seeds(spec, gated)
    for key in ("p", "i_f", "r72"):
        if p.get(key) is not None:
            doc[key] = dict(pool_s(cells, p[key]["groups"], on + off), panel=p[key]["panel"])
    if p.get("goal_a") is not None:
        ga = p["goal_a"]
        doc["goal_a"] = dict(pool_anchor(cells, ga["groups"], [ga["teamset"]], on), teamset=ga["teamset"],
                             units=ga["units"])
        doc["goal_a"].pop("rows")
    c = p.get("contrast")
    if c is not None:
        labs = list(c["on"]) + list(c["off"])
        pooled = pool_anchor(cells, c["groups"], c["panels"], labs)
        doc["contrast"] = {"on": list(c["on"]), "off": list(c["off"]), "panels": list(c["panels"]),
                           "units": c["units"], "groups": list(c["groups"]),
                           "counts": unit_counts(pooled["rows"], labs, c["panels"], c["units"]),
                           "engine": pooled["engine"], "regime_ids": pooled["regime_ids"],
                           "inconclusive": pooled["inconclusive"]}
    if p.get("reported"):
        doc["reported"] = {}
        for g in p["reported"]:
            grp = (cells.get("anchor") or {}).get(g) or {}
            doc["reported"][g] = {"teamsets": list(grp.get("teamsets") or ()), "engine": grp.get("engine"),
                                  "inconclusive": list(grp.get("inconclusive") or ()),
                                  "unplayed": int(grp.get("unplayed") or 0),
                                  "cells": {lab: {t: {"wins": x["wins"], "n": x["n"]} for t, x in by.items()}
                                            for lab, by in (grp.get("cells") or {}).items() if lab in on + off}}
        doc["reported_pooled"] = _pool_reported(cells, p["reported"], on, off)
    if p.get("determinism_guard") is not None:
        doc["determinism_guard"] = _determinism(cells, p["determinism_guard"])
    doc["guards"] = _guard_doc(p.get("guards"))
    if p.get("guards") is not None:
        # the guards' provenance: each final as the state resolved it (``models.json``), never a path or a label
        models = cells.get("models") or {}
        doc["finals"] = {lab: (models.get(lab) or {}).get("sha256") for lab in on + off}
    return doc


def _pool_reported(cells: Mapping[str, Any], groups: Sequence[str], on: Sequence[str], off: Sequence[str]
                   ) -> Dict[str, Any]:
    """§12's pooled REPORTED Δs (AM-X60T-3): per team set of ``groups``, every final's counted units with unit seed <
    :data:`REPORTED_UNITS` across the groups holding that team set, and every reason they are not one cell."""
    anchor = cells.get("anchor") or {}
    labs = list(on) + list(off)
    teamsets = sorted({str(t) for g in groups for t in (anchor.get(g) or {}).get("teamsets") or ()})
    out: Dict[str, Any] = {}
    for t in teamsets:
        mine = [g for g in groups if t in ((anchor.get(g) or {}).get("teamsets") or ())]
        pooled = pool_anchor(cells, mine, [t], labs)
        counts = unit_counts(pooled["rows"], labs, [t], REPORTED_UNITS)
        short = [f"{lab} {t}: {counts[lab][t]['units']} of {REPORTED_UNITS} units counted" for lab in labs
                 if counts[lab][t]["units"] != REPORTED_UNITS]
        out[t] = {"groups": mine, "units": REPORTED_UNITS, "counts": {lab: counts[lab][t] for lab in labs},
                  "engine": pooled["engine"], "regime_ids": pooled["regime_ids"][t],
                  "inconclusive": sorted(set(pooled["inconclusive"]) | set(short))}
    return out


# ------------------------------------------------------------------------------------------------ decide
def _floor(units: int, finals: int) -> int:
    return UNIT_GAMES * int(units) * int(finals) - UNIT_GAMES


def _arm_floor(blk: Mapping[str, Any], labs: Sequence[str], teamsets: Sequence[str], units: int, what: str
               ) -> List[str]:
    why = []
    for t in teamsets:
        tot = sum(int(blk["counts"][lab][t]["n"]) for lab in labs)
        if tot < _floor(units, len(labs)):
            why.append(f"{what} {t}: {tot} counted games < the floor {_floor(units, len(labs))} (X13)")
        why += [f"{what} {lab} {t}: no counted game" for lab in labs if int(blk["counts"][lab][t]["n"]) <= 0]
    return why


def _counts(blk: Mapping[str, Any], labs: Sequence[str], teamsets: Sequence[str]) -> List[List[Tuple[int, int]]]:
    return [[(int(blk["counts"][lab][t]["wins"]), int(blk["counts"][lab][t]["n"])) for t in teamsets] for lab in labs]


def _primary(doc: Mapping[str, Any]) -> Dict[str, Any]:
    on, off = doc["sides"]["on"], doc["sides"]["off"]
    blk, ts = doc["primary"], [doc["primary"]["teamset"]]
    why = list(blk["inconclusive"])
    if not why:
        why += _arm_floor(blk, on, ts, blk["units"], "ON") + _arm_floor(blk, off, ts, blk["units"], "OFF")
    if why:
        return {"class": None, "reasons": why}
    seeds = doc["seeds"]
    on = sorted(on, key=lambda lab: seeds[lab])
    off = sorted(off, key=lambda lab: seeds[lab])
    l1 = set(doc["look1_seeds"])
    i_on = [i for i, lab in enumerate(on) if seeds[lab] in l1]
    i_off = [i for i, lab in enumerate(off) if seeds[lab] in l1]
    # look 1's finals first, as power.py's a[:, :3]
    on = [on[i] for i in i_on] + [lab for i, lab in enumerate(on) if i not in i_on]
    off = [off[i] for i in i_off] + [lab for i, lab in enumerate(off) if i not in i_off]
    r = rule_x60(_counts(blk, on, ts), _counts(blk, off, ts), list(range(len(l1))))
    return dict(r, reasons=[], order={"on": on, "off": off})


def _welch(on: Sequence[float], off: Sequence[float], confidence: float) -> Dict[str, Any]:
    from main.x35 import readout as R

    return R.welch_interval(list(on), list(off), confidence=confidence).as_dict()


def _s_sides(doc: Mapping[str, Any], key: str) -> Tuple[Optional[Tuple[List[float], List[float]]], List[str]]:
    blk = doc.get(key)
    if blk is None:
        return None, [f"{key}: not registered"]
    if blk["inconclusive"]:
        return None, [f"{key}: {w}" for w in blk["inconclusive"]]
    get = [[float(blk["nodes"][lab]["s_pp"]) for lab in doc["sides"][s]] for s in ("on", "off")]
    return (get[0], get[1]), []


def _p_meter(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """Δ_P by D1's rule (``d1.d1_coprimary``'s arithmetic): Welch two-sided ``P_CONFIDENCE``, NON-INFERIOR iff LB >
    ``P_BAR_PP`` (within RULE8_EPS: not)."""
    from main.x35 import d1 as D
    from main.x35 import readout as R

    sides, why = _s_sides(doc, "p")
    if sides is None:
        return {"class": None, "reasons": why}
    try:
        st = R.welch_interval(sides[0], sides[1], confidence=D.P_CONFIDENCE)
    except R.ReadoutInputError as exc:
        return {"class": None, "reasons": [str(exc)]}
    ok, near = R._clears(st.ci[0], D.P_BAR_PP, above=True)
    pd = D.PDecision(2, D.NON_INFERIOR if ok else D.NOT_EXCLUDED, st,
                     (f"lower {st.ci[0]!r} within {R.RULE8_EPS} of {D.P_BAR_PP}",) if near else (), ())
    return dict(pd.as_dict(), sd_on=_sd(sides[0]), sd_off=_sd(sides[1]))


def _sd(xs: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = math.fsum(xs) / len(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _inhouse(doc: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """I_F / R72: Welch across finals, the 95 % LB > 0 (GAIN / DELIVERED); I_F's one-sided 90 % LB (two-sided 80 %)."""
    from main.x35 import readout as R

    sides, why = _s_sides(doc, key)
    if sides is None:
        return {"read": False, "reasons": why}
    try:
        w95, w80 = _welch(*sides, 0.95), _welch(*sides, 0.80)
    except R.ReadoutInputError as exc:
        return {"read": False, "reasons": [str(exc)]}
    pos95, near95 = R._clears(w95["ci_pp"][0], 0.0, above=True)
    pos90, near90 = R._clears(w80["ci_pp"][0], 0.0, above=True)
    return {"read": True, "welch95": w95, "lb90_pp": w80["ci_pp"][0], "gain": pos95, "lb90_pos": pos90,
            "near": [n for n, x in (("LB95", near95), ("LB90", near90)) if x], "sd_on": _sd(sides[0]),
            "sd_off": _sd(sides[1]), "reasons": []}


def _goal_a(doc: Mapping[str, Any]) -> Dict[str, Any]:
    from main.anchors.results import wilson
    from main.x35 import readout as R

    blk = doc.get("goal_a")
    if blk is None:
        return {"pass": None, "reasons": ["goal_a: not registered"]}
    on, ts = doc["sides"]["on"], [blk["teamset"]]
    why = list(blk["inconclusive"]) or _arm_floor(blk, on, ts, blk["units"], "goal (a) ON")
    if why:
        return {"pass": None, "reasons": why}
    w = sum(int(blk["counts"][lab][ts[0]]["wins"]) for lab in on)
    n = sum(int(blk["counts"][lab][ts[0]]["n"]) for lab in on)
    ph, lo, hi = wilson(w, n)
    ok, near = R._clears(lo, GOAL_A_BAR, above=True)
    return {"pass": ok, "wins": w, "n": n, "p": ph, "wilson_lb": lo, "wilson_ub": hi, "bar": GOAL_A_BAR,
            "near": near, "reasons": []}


def _by_seed(doc: Mapping[str, Any], side: str) -> List[str]:
    seeds = doc["seeds"]
    return sorted(doc["sides"][side], key=lambda lab: seeds[lab])


def _final_sha(doc: Mapping[str, Any], lab: str, why: List[str]) -> Optional[str]:
    sha = (doc.get("finals") or {}).get(lab)
    if not sha:
        why.append(f"{lab}: the state records no sha256 for its final")
    return sha


def _guard_f(doc: Mapping[str, Any], blk: Any) -> Any:
    """G-F from the artefact's ``g_f`` block (module docstring): every reason it is not this look's matrix of this
    read's finals is INCONCLUSIVE; else ``cross_stat`` on the matrix RE-DERIVED from the cells' pentanomials."""
    from agents.training.mirrored_pairs import pair_score
    from main.h2h.cross import CrossInputError, cross_stat
    from main.x35 import readout as R

    if blk is None:
        return R.GuardResult("G-F", R.INCONCLUSIVE, reasons=("no head-to-head matrix",))
    if not isinstance(blk, Mapping):
        return R.GuardResult("G-F", R.INCONCLUSIVE, reasons=("g_f: an object",))
    why: List[str] = [f"g_f: {w}" for w in blk.get("inconclusive") or ()]
    rows, cols = _by_seed(doc, "on"), _by_seed(doc, "off")
    n = LOOK_FINALS[int(doc.get("look", 2))]
    if len(rows) != n or len(cols) != n:
        why.append(f"g_f: look {doc.get('look')} registers {n} x {n}, the read has {len(rows)} x {len(cols)}")
    if list(blk.get("rows") or ()) != rows:
        why.append(f"g_f: rows {blk.get('rows')} are not the ON finals in seed order {rows}")
    if list(blk.get("cols") or ()) != cols:
        why.append(f"g_f: columns {blk.get('cols')} are not the OFF finals in seed order {cols}")
    if blk.get("min_pairs") != R.MIN_PAIRS:
        why.append(f"g_f: built at min_pairs {blk.get('min_pairs')!r}, the registered floor is {R.MIN_PAIRS}")
    sha = {lab: _final_sha(doc, lab, why) for lab in rows + cols}
    by: Dict[Tuple[Any, Any], Mapping[str, Any]] = {}
    for c in blk.get("cells") or ():
        k = (c.get("row"), c.get("col"))
        if k in by:
            why.append(f"g_f: cell {k} listed twice")
        by[k] = c
    extra = sorted(map(str, set(by) - {(r, c) for r in rows for c in cols}))
    if extra:
        why.append(f"g_f: {len(extra)} cell(s) outside the ON x OFF finals: {extra[:3]}")
    h: List[List[float]] = []
    for r in rows:
        line: List[float] = []
        for col in cols:
            c = by.get((r, col))
            if c is None:
                why.append(f"g_f: no cell {r} vs {col}")
                line.append(float("nan"))
                continue
            if c.get("player_sha256") != sha[r] or c.get("opponent_sha256") != sha[col]:
                why.append(f"g_f: cell {r} vs {col} was played by other finals ({str(c.get('player_sha256'))[:12]} "
                           f"vs {str(c.get('opponent_sha256'))[:12]})")
            if c.get("verdict") != "OK":
                why.append(f"g_f: cell {r} vs {col} reads {c.get('verdict')!r} {list(c.get('reasons') or ())}")
            pc = c.get("pairs")
            if not (isinstance(pc, list) and len(pc) == 5 and all(isinstance(x, int) and x >= 0 for x in pc)):
                why.append(f"g_f: cell {r} vs {col}: pairs is not a pentanomial of 5 counts")
                line.append(float("nan"))
                continue
            if sum(pc) < R.MIN_PAIRS:
                why.append(f"g_f: cell {r} vs {col}: {sum(pc)} completed pairs < {R.MIN_PAIRS}")
            sc = pair_score(pc)
            line.append(100.0 * sc if sc is not None else float("nan"))
        h.append(line)
    if blk.get("h_pp") != h and not why:
        why.append("g_f: h_pp is not its cells' pair scores (100 x the pentanomial score)")
    if why:
        return R.GuardResult("G-F", R.INCONCLUSIVE, reasons=tuple(why))
    try:
        st = cross_stat(h)
    except CrossInputError as exc:
        return R.GuardResult("G-F", R.INCONCLUSIVE, reasons=(str(exc),))
    fail, near = R._clears(st.ci95[1], -R.BAR_PP, above=False)
    return R.GuardResult("G-F", R.FAIL if fail else R.PASS, detail={"delta_hat_pp": st.delta_hat,
                         "ci95_pp": list(st.ci95), "df": st.df, "size": len(h), "rows": rows, "cols": cols},
                         near=(f"upper {st.ci95[1]!r} within {R.RULE8_EPS} of {-R.BAR_PP}",) if near else ())


def _guard_e(doc: Mapping[str, Any], r2: Any) -> Any:
    """G-E from the artefact's ``r2`` block (module docstring): INCONCLUSIVE unless its readers are §23.5's and their
    targets are this read's seed-3501 finals by sha256."""
    from main.x35 import ge as GE
    from main.x35 import readout as R

    if not r2:
        return R.guard_e(None)
    if not isinstance(r2, Mapping):
        return R.GuardResult("G-E", R.INCONCLUSIVE, reasons=("r2: an object",))
    why: List[str] = []
    if r2.get("stat") != GE.STAT:
        why.append(f"r2: stat {r2.get('stat')!r}, the registered {GE.STAT!r}")
    seed = GE.TARGET_SEEDS[0]
    for side, arm in GE_READERS.items():
        labs = [lab for lab in doc["sides"][side] if doc["seeds"][lab] == seed]
        if len(labs) != 1:
            why.append(f"r2: the {side} side has no single seed-{seed} final")
            continue
        want = GE.reader_run(arm, seed)
        got = ((r2.get("readers") or {}).get(side) or {}).get("run")
        if got != want:
            why.append(f"r2: the {side} reader is {got!r}, not {want}")
        tsha = ((r2.get("targets") or {}).get(side) or {}).get("sha256")
        fsha = _final_sha(doc, labs[0], why)
        if fsha and tsha != fsha:
            why.append(f"r2: the {side} reader's target {str(tsha)[:12]} is not {labs[0]}'s final {fsha[:12]}")
    if why:
        return R.GuardResult("G-E", R.INCONCLUSIVE, reasons=tuple(why))
    try:
        return R.guard_e(R.gap_diff(r2["on"], r2["off"]))
    except (R.ReadoutInputError, KeyError, TypeError, IndexError) as exc:
        return R.GuardResult("G-E", R.INCONCLUSIVE, reasons=(f"G-E input: {exc}",))


def _guard_c(doc: Mapping[str, Any], rec: Any) -> Any:
    """G-C from the artefact's ``g_c`` record (module docstring): ``readout.guard_c_record``, then INCONCLUSIVE unless
    every pair's ON / OFF checkpoint is this read's final of that seed by sha256."""
    from main.x35 import readout as R

    seeds = doc["seeds"]
    gc = R.guard_c_record(rec, [str(seeds[lab]) for lab in doc["sides"]["on"]],
                          [str(seeds[lab]) for lab in doc["sides"]["off"]])
    if gc.status == R.INCONCLUSIVE:
        return gc
    why: List[str] = []
    shas = rec.get("checkpoint_sha256") or {}
    pp = rec.get("per_pair") or {}
    for side in ("on", "off"):
        for lab in doc["sides"][side]:
            pair = pp.get(str(seeds[lab])) or {}
            got = shas.get(str(pair.get(side)))
            fsha = _final_sha(doc, lab, why)
            if fsha and got != fsha:
                why.append(f"g_c: seed {seeds[lab]}'s {side} read {pair.get(side)!r} is checkpoint "
                           f"{str(got)[:12]}, not {lab}'s final {fsha[:12]}")
    if why:
        return R.GuardResult("G-C", R.INCONCLUSIVE, detail=gc.detail, reasons=tuple(why))
    return gc


def _guards(doc: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """G-F / G-E / G-C by ``main.x35.readout``'s guard arithmetic, from the registered artefact (module docstring)."""
    from main.x35 import readout as R

    gd = doc["guards"]

    def every(why: str) -> Dict[str, Dict[str, Any]]:
        return {n: R.GuardResult(n, R.INCONCLUSIVE, reasons=(why,)).as_dict() for n in GUARDS}

    if not gd.get("registered") or gd.get("error"):
        return every(gd.get("error") or "not read (no guards artefact registered)")
    g = gd["doc"]
    if not isinstance(g, Mapping) or g.get("schema") != GUARDS_SCHEMA:
        return every(f"the guards artefact is not schema {GUARDS_SCHEMA}")
    status = g.get("status")
    if status not in (NOT_PLAYED, PLAYED):
        return every(f"the guards artefact's status {status!r} is neither {NOT_PLAYED!r} nor {PLAYED!r}")
    if status == NOT_PLAYED and any(g.get(k) is not None for k in GUARD_BLOCKS):
        return every(f"a {NOT_PLAYED} guards artefact carries a guard input")
    return {x.name: x.as_dict() for x in (_guard_f(doc, g.get("g_f")), _guard_e(doc, g.get("r2")),
                                          _guard_c(doc, g.get("g_c")))}


def _contrast(doc: Mapping[str, Any]) -> Dict[str, Any]:
    c = doc.get("contrast")
    if c is None:
        return {"status": "NOT READ", "why": "no contrast registered", "false_rates": CONTRAST_FALSE_RATES}
    why = list(c["inconclusive"])
    blk = {"counts": c["counts"]}
    if not why:
        why += _arm_floor(blk, c["on"], c["panels"], c["units"], "contrast ON")
        why += _arm_floor(blk, c["off"], c["panels"], c["units"], "contrast D1")
    if why:
        return {"status": "NOT READ", "why": "; ".join(why), "false_rates": CONTRAST_FALSE_RATES}
    s = x60_interval(_counts(blk, c["on"], c["panels"]), _counts(blk, c["off"], c["panels"]))
    raw = classes(s["lo"], s["hi"], one_sided=False)
    cls = {GAIN: LADDER, LOSS: FORUM, "NO-DIFF": NO_DIFF}.get(raw, NOT_RESOLVED)
    per_panel = {}
    for i, t in enumerate(c["panels"]):
        per_panel[t] = {"on": s["pooled_on"][i], "off": s["pooled_off"][i], "delta": s["pooled_on"][i] - s["pooled_off"][i]}
    return {"status": "READ", "class": cls, "stat": s, "per_panel_reported": per_panel,
            "attribution": "LADDER-BETTER: H-TEAMS's gain on its own population (H) plus its transfer to F exceeds "
                           "D1's gain on its own population (F) plus its transfer to H (m3); not 'ladder data beats "
                           "forum-team breadth'", "false_rates": CONTRAST_FALSE_RATES}


def _reported(doc: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for g, blk in (doc.get("reported") or {}).items():
        why = list(blk["inconclusive"]) + ([f"{blk['unplayed']} unit(s) unplayed"] if blk["unplayed"] else [])
        rows: Dict[str, Any] = {}
        for t in blk["teamsets"]:
            sides = {}
            for s in ("on", "off"):
                labs = [lab for lab in doc["sides"][s] if t in (blk["cells"].get(lab) or {})]
                w = sum(blk["cells"][lab][t]["wins"] for lab in labs)
                n = sum(blk["cells"][lab][t]["n"] for lab in labs)
                sides[s] = {"wins": w, "n": n, "p": w / n if n else None, "finals": len(labs)}
            d = (sides["on"]["p"] - sides["off"]["p"]) if sides["on"]["p"] is not None and sides["off"]["p"] \
                is not None else None
            rows[t] = dict(sides, delta=d)
        out[g] = {"status": "INCOMPLETE" if why else "OK", "reasons": why, "engine": blk["engine"], "panels": rows}
    return out


def _reported_pooled(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """§12's pooled Δ per team set (AM-X60T-3; module docstring): Σwins / Σgames per arm over units 0-6, REPORTED."""
    out: Dict[str, Any] = {}
    for t, blk in (doc.get("reported_pooled") or {}).items():
        sides: Dict[str, Any] = {}
        for s in ("on", "off"):
            labs = list(doc["sides"][s])
            w = sum(int(blk["counts"][lab]["wins"]) for lab in labs)
            n = sum(int(blk["counts"][lab]["n"]) for lab in labs)
            sides[s] = {"wins": w, "n": n, "p": w / n if n else None, "finals": len(labs),
                        "per_final": {lab: (blk["counts"][lab]["wins"] / blk["counts"][lab]["n"]
                                            if blk["counts"][lab]["n"] else None) for lab in labs}}
        d = sides["on"]["p"] - sides["off"]["p"] if sides["on"]["p"] is not None and sides["off"]["p"] is not None \
            else None
        out[t] = dict(sides, name=REPORTED_NAMES.get(t, f"Δ_{t}"), delta=d, units=blk["units"], groups=blk["groups"],
                      engine=blk["engine"], regime_ids=blk["regime_ids"],
                      status="INCOMPLETE" if blk["inconclusive"] else "OK", reasons=list(blk["inconclusive"]))
    return out


def decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    look = int(doc["look"])
    base: Dict[str, Any] = {"schema": SCHEMA, "look": look, "sides": dict(doc["sides"])}
    prim = _primary(doc)
    det = doc.get("determinism_guard")
    rep: Dict[str, Any] = {"panels": _reported(doc)}
    if doc.get("reported_pooled") is not None:
        rep["pooled"] = _reported_pooled(doc)
    i_f: Dict[str, Any] = _inhouse(doc, "i_f") if doc.get("i_f") else {"read": False, "reasons": ["i_f: absent"]}
    r72: Dict[str, Any] = _inhouse(doc, "r72") if doc.get("r72") else {"read": False, "reasons": ["r72: absent"]}
    out = dict(base, primary=prim, determinism_guard=det, reported=rep)
    bad_seeds = [f"{lab}: run {r['run']} recorded seed {r['recorded']}, the spec {r['spec']}"
                 for lab, r in (doc.get("recorded_seeds") or {}).items() if not r["ok"]]
    if bad_seeds:
        return dict(out, decision={"verdict": INCONCLUSIVE, "reasons": [
            "a label's seed is not its run's recorded seed (REVIEW I M3)"] + bad_seeds})
    if det is not None and not det["passed"]:
        return dict(out, decision={"verdict": INCONCLUSIVE, "reasons": [
            "the determinism guard failed: HOLD, re-play the imported cells at the play pin (X7)"] + det["reasons"]})
    if prim["class"] is None:
        return dict(out, decision={"verdict": INCONCLUSIVE, "reasons": ["Δ_F: " + w for w in prim["reasons"]]})
    if look == 1 or prim["stop"]:
        v, why = verdict(look, prim)
        rep.update(i_f=i_f, r72=r72)               # X5: REPORTED, never dividing a look-1 stop
        stop_dec: Dict[str, Any] = {"verdict": v, "reasons": why, "delta_f_class": prim["class"],
                                    "stop_at_look1": prim["stop"], "equivalent_reported": prim["equivalent"],
                                    "guards_not_read": "Δ_F is not a GAIN"}
        if look == 2:
            stop_dec["reasons"] = stop_dec["reasons"] + [
                "look 2 recomputes look 1's futility stop on s3501-s3503 (power.py's rule_x60): this read should have "
                "stopped at look 1"]
        return dict(out, decision=stop_dec)
    p_m = _p_meter(doc)
    ga = _goal_a(doc)
    gain = prim["class"] == GAIN
    gu = _guards(doc) if gain else None
    v, why = verdict(look, prim, i_f=({"gain": i_f["gain"], "lb90_pos": i_f["lb90_pos"]} if i_f["read"] else None),
                     r72=(r72["gain"] if r72["read"] else None), p=p_m["class"], goal_a=ga["pass"],
                     guards=None if gu is None else {n: x["status"] for n, x in gu.items()})
    dec: Dict[str, Any] = {"verdict": v, "reasons": why, "delta_f_class": prim["class"], "stop_at_look1": False,
                           "equivalent_reported": prim["equivalent"]}
    if not gain:
        dec["guards_not_read"] = f"Δ_F is {prim['class']}, not a GAIN: G-F / G-E / G-C are read only on a GAIN (§11)"
    if v == TRADE and p_m.get("stat"):
        dp, df_pp = p_m["stat"]["delta_pp"], 100.0 * prim["look2"]["delta"]
        dec["exchange_rate"] = ({"rate": df_pp / -dp, "delta_f_pp": df_pp, "delta_p_pp": dp}
                                if dp < 0.0 and abs(dp) > TIE else
                                {"rate": None, "delta_f_pp": df_pp, "delta_p_pp": dp, "why": "Δ̂_P >= 0"})
    rep["contrast_t_minus_d1"] = _contrast(doc)
    dec["contrast_class"] = rep["contrast_t_minus_d1"].get("class") or rep["contrast_t_minus_d1"]["status"]
    return dict(out, p=p_m, i_f=i_f, r72=r72, goal_a=ga, guards=gu, decision=dec)
