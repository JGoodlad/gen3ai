"""The REGISTERED interval and decision rules, by name (``designs/tools/reads.md`` §5). Pure functions; no game.

A read names ONE rule in its spec (``decision.rule``). Each :class:`Rule` has two halves, both pure:

* ``inputs(cells, params, spec)`` -- the rule's INPUT DOCUMENT from the assembled cells (``main.reads.assemble``):
  exactly the JSON the rule reads, written beside the record, so a decision is re-checkable from that file alone;
* ``decide(doc, params)`` -- the verdict document, as recorded.

THE ESTIMATORS are not re-derived here. Each is the one function a registered read already uses, re-exported under
one name so a new read imports this module and nothing else: Welch / R1's classes (:mod:`main.x35.readout`), the
equal-weight stratified Wilson, Newcombe-of-strata and AM-5's overdispersion φ̂ (:mod:`main.x35.d1`), Wilson and
Newcombe method 10 at z = 1.959963985 (:mod:`main.anchors.results`) and at z = 1.96 (``best_response_gap``). Rule 8 is
:func:`clears` (``readout``'s: within :data:`RULE8_EPS` of a boundary is never a crossing). Migrated here from the
per-read scripts (they had no home in ``src/``): :func:`cell_se` and :func:`s_se` (the X26 drift pre-read's declared
per-final interval of S), :func:`paired_unit_diff` (x26c's descriptive unit-paired difference) and
:func:`guards_not_read` (the orchestrator's A1 ruling (a): the guards gate only a QUALIFYING primary).

THE RULES (:data:`RULES`):

* ``x35_r1`` -- X35's R1 / R2 / guards / verdict (``readout.read``), on S per seed final (SPRT, LRF, A1 ...);
* ``d1`` -- D1's (X38 / X41 / X61P) H on the arm's primary Kakuna panels (``d1.arm_panels``: X61P's is forum72
  alone) widened by φ̂, and P (``d1.read_d1``); an optional ``reported_group`` adds per-panel rows, U and
  Δ_H − Δ_F (``d1.reported_panels``), which no rule reads;
* ``anchor_trajectory`` -- x26c PREREG §5: the goal-(b) bar on an anchor score, CLIMBING / FLAT on its Newcombe
  difference against a comparator, and the in-house S "rising" read against a banked node;
THE VOID GATE (:func:`void_gate`): an X35-stage rule (``x35_r1``, ``d1``) may register ``params.void = {k9b_dir, runs:
{label: run dir}}``; ``x35_r1`` on a batch-2 arm (SPRT, LRF) MUST. ``decide`` runs it BEFORE the rule: each named run's
AM-6 K9(b) record (``<k9b_dir>/<run>.json``) must read PASS, and a batch-2 arm's V-S1 / V-S2 (SPRT) or LR-F KL guard
(``main.x35.batch2_void.read_void``) too; anything else refuses the decision (a VOID seed enters no read; SPRT's
``assemble.py`` STOP, be60f24f FINDING 3).

REPORTED-ONLY GROUPS (``Rule.reported``): the groups a rule carries for description and no verdict reads (``d1``'s
``reported_group``). An incomplete one never blocks the record of the decided primary: the record names it INCOMPLETE
with its reasons (N1-P FINDING 7).

* ``prior_mix_ab`` -- the belief-prior-mix read (``belief_prior_mix_2026-10-26`` PREREG): each λ arm against a
  BANKED λ = 0 control (another read's committed per-game rows, sha-checked, never re-played), unit-paired on the
  same seeds, with a game-for-game determinism guard that re-plays a few banked units at λ = 0.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from main.anchors.results import Z95, newcombe, wilson
from main.x35 import d1 as D
from main.x35 import readout as R
from main.x35.d1 import newcombe_strat, overdispersion, strat_score, widen, wilson_pn
from main.x35.readout import RULE8_EPS, classify, welch_interval

__all__ = ["RULES", "RULE8_EPS", "Rule", "RuleInputError", "Z95", "cell_se", "classify", "clears", "decide",
           "guards_not_read", "inputs", "newcombe", "newcombe_diff_ci", "newcombe_strat", "overdispersion",
           "paired_unit_diff", "s_se", "strat_score", "void_gate", "welch_interval", "widen", "wilson", "wilson_pn"]


class RuleInputError(ValueError):
    """The cells cannot form this rule's input (a missing group, side or seed): named, never a default."""


def newcombe_diff_ci(w1: int, n1: int, w2: int, n2: int, z: float = 1.96) -> Tuple[float, float, float]:
    """``best_response_gap.newcombe_diff_ci`` (R2 / G-E's meter, z = 1.96)."""
    from agents.training.best_response_gap import newcombe_diff_ci as _n

    return _n(w1, n1, w2, n2, z)


def clears(stat: float, boundary: float, *, above: bool = True) -> Tuple[bool, bool]:
    """``(crosses, near)``: ``stat`` > ``boundary`` (``above``) or < it; within :data:`RULE8_EPS` is ``near`` and never
    a crossing (standing rule 8; ``readout``'s convention, which x26c's ``lb > bar + EPS`` equals)."""
    return R._clears(stat, boundary, above=above)


def cell_se(pairs: Sequence[int]) -> float:
    """The per-pair SE of one mirrored cell's mean score from its pentanomial (pair score i/4, i = 0..4)."""
    n = sum(pairs)
    if n <= 0:
        raise RuleInputError("a cell with no completed pair has no SE")
    mu = sum(c * i for i, c in enumerate(pairs)) / (4.0 * n)
    var = sum(c * (i / 4.0 - mu) ** 2 for i, c in enumerate(pairs)) / n
    return math.sqrt(var / n)


def s_se(per_member_se_pp: Sequence[float]) -> float:
    """se(S) = sqrt(Σ se_m²) / K over R's K members (pp): the pre-read's declared per-final interval."""
    if not per_member_se_pp:
        raise RuleInputError("no member SE")
    return math.sqrt(math.fsum(x * x for x in per_member_se_pp)) / len(per_member_se_pp)


def paired_unit_diff(per_unit: Mapping[Tuple[str, str, int], float], on: str, off: str, panels: Sequence[str],
                     unit_seeds: Sequence[int], z: float = Z95) -> Optional[Dict[str, Any]]:
    """DESCRIPTIVE: the panel-equal-weight mean of unit-paired win-rate differences (same panel, same unit seed),
    with a normal interval on the per-panel SEs (x26c ``verdict.py``); ``None`` when a panel has < 2 pairs."""
    by: Dict[str, List[float]] = {k: [] for k in panels}
    for k in panels:
        for seed in unit_seeds:
            a, b = per_unit.get((on, k, seed)), per_unit.get((off, k, seed))
            if a is not None and b is not None:
                by[k].append(a - b)
    if not all(len(v) > 1 for v in by.values()):
        return None
    K = len(panels)
    m = sum(sum(v) / len(v) for v in by.values()) / K
    var = sum((sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) / len(v) for v in by.values()) / (K * K)
    return {"d": m, "lo": m - z * math.sqrt(var), "hi": m + z * math.sqrt(var), "pairs": {k: len(v) for k, v in by.items()}}


def guards_not_read(v: Dict[str, Any], ruling: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The A1 ruling (a): on a NON-qualifying primary the guards were not played, so each named guard is recorded
    INCONCLUSIVE with the ruling's reason (``ruling = {"guards": [...], "why": str, "absent": {guard: why}}``);
    ``absent`` replaces a guard's reasons and keeps its status. A qualifying primary is left untouched (its
    guards decide, and an unplayed one is INCONCLUSIVE by the rule itself)."""
    if not ruling:
        return v
    primary = v.get("h") if "h" in v else v.get("r1")
    qualifies = (primary or {}).get("class") == R.GAIN or v.get("verdict") in (R.ADOPT, R.TRADE)
    if qualifies:
        return v
    for n in ruling.get("guards") or ():
        v["guards"][n].update(reasons=[str(ruling["why"])], status=R.INCONCLUSIVE)
    for n, why in (ruling.get("absent") or {}).items():
        v["guards"][n].update(reasons=[str(why)])
    return v


def _jsonable(d: Any) -> Any:
    """The record's own form (what ``json.dump(..., default=str)`` writes), so a decision compares to a file."""
    return json.loads(json.dumps(d, default=str))


# ------------------------------------------------------------------------------------------------ helpers
def _seed_side(spec: Any, labels: Sequence[str], on: str, off: str) -> Dict[str, Dict[str, str]]:
    """``{"on": {seed: label}, "off": {...}}`` by the spec's arm / seed; two finals of a side on one seed refuse."""
    out: Dict[str, Dict[str, str]] = {"on": {}, "off": {}}
    for lab in labels:
        m = spec.models[lab]
        side = "on" if m.arm == on else "off" if m.arm == off else None
        if side is None:
            continue
        if str(m.seed) in out[side]:
            raise RuleInputError(f"two {m.arm} finals share seed {m.seed}")
        out[side][str(m.seed)] = lab
    return out


def _group(cells: Mapping[str, Any], lane: str, name: Any) -> Dict[str, Any]:
    g = (cells.get(lane) or {}).get(name)
    if g is None:
        raise RuleInputError(f"no assembled {lane} group {name!r}")
    return dict(g)


def imported_s(entry: Mapping[str, Any], *, want_refpop: Optional[str] = None) -> Dict[str, Optional[float]]:
    """``{seed: S_pp}`` of ``entry["arm"]`` from ANOTHER read's committed S results (``finals`` rows of a g1 / d1s /
    sprt ``results.json``), its sha256 checked (``entry = {path, sha256, arm}``): a banked control, never re-played."""
    from main.reads.models import sha256_file, spec_path

    p = spec_path(str(entry["path"]))
    got = sha256_file(p)
    if got != entry["sha256"]:
        raise RuleInputError(f"imported S {p}: sha256 {got[:12]} != the spec's {str(entry['sha256'])[:12]}")
    doc = json.loads(p.read_text())
    if want_refpop is not None and doc.get("refpop_content_sha256") != want_refpop:
        # sprt_read/assemble.py's STOP: "the two reads used different R"
        raise RuleInputError(f"imported S {p}: R content {str(doc.get('refpop_content_sha256'))[:12]} is not this "
                             f"read's {want_refpop[:12]}")
    rows = doc["finals"]
    return {str(r["seed"]): (r["s_pp"] if not r.get("inconclusive") else None) for r in rows
            if r["arm"] == entry["arm"]}


def _check_keys(params: Mapping[str, Any], need: Sequence[str], allowed: Sequence[str]) -> List[str]:
    probs = [f"missing {k!r}" for k in need if k not in params]
    probs += [f"unknown {k!r}" for k in sorted(set(params) - set(allowed))]
    return probs


# ------------------------------------------------------------------------------------------------ the VOID gate
#: The batch-2 arms whose registered VOID guard reads the run dir (``main.x35.batch2_void.read_void``'s arm names).
BATCH2_VOID_ARM = {"SPRT": "sprt", "LRF": "lrfloor"}


def _void_check(p: Mapping[str, Any], models: Any, *, required: bool) -> List[str]:
    v = p.get("void")
    if v is None:
        return [f"void: required for arm {p.get('arm')!r} (its V-S1 / V-S2 / LR-F and K9(b) gate the read)"] if required \
            else []
    if not (isinstance(v, dict) and set(v) == {"k9b_dir", "runs"} and isinstance(v["k9b_dir"], str)
            and isinstance(v["runs"], dict) and v["runs"]):
        return ["void: {k9b_dir: <dir of AM-6 K9(b) records>, runs: {label: run dir}}"]
    probs = [f"void.runs: {lab!r} is not a model label" for lab in v["runs"] if lab not in models]
    probs += [f"void.runs.{lab}: a run dir path" for lab, r in v["runs"].items() if not (isinstance(r, str) and r)]
    on = [lab for lab, m in models.items() if m.arm == p.get("on")]
    probs += [f"void.runs: the ON model {lab!r} has no run (every ON seed is gated)" for lab in on
              if lab not in v["runs"]]
    return probs


def void_gate(spec: Any) -> Optional[Dict[str, Dict[str, Any]]]:
    """The registered VOID gate of ``spec`` (module docstring), or ``None`` when it registers none: ``{label: {run,
    k9b, k9b_file, batch2, pass}}``. Reads the K9(b) records and the run dirs (read-only)."""
    from pathlib import Path

    from main.reads.models import spec_path
    from main.x35 import batch2_void as V

    v = (spec.params or {}).get("void")
    if not v:
        return None
    b2 = BATCH2_VOID_ARM.get(str(spec.params.get("arm")))
    out: Dict[str, Dict[str, Any]] = {}
    for lab, run in sorted(v["runs"].items()):
        rd = spec_path(run)
        kf = spec_path(v["k9b_dir"]) / f"{Path(run).name}.json"
        try:
            k9b = str(json.loads(kf.read_text())["verdict"])
        except (OSError, ValueError, KeyError):
            k9b = "MISSING"
        b = V.read_void(b2, str(rd)).as_dict() if b2 else None
        out[lab] = {"run": str(run), "k9b": k9b, "k9b_file": str(kf), "batch2": b,
                    "pass": k9b == V.PASS and (b is None or b["verdict"] == V.PASS)}
    return out


def void_failures(gate: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """``{label: why}`` of every gated run that is not PASS (empty = the gate passes or none is registered)."""
    return {lab: f"K9(b) {g['k9b']}" + (f", {g['batch2']['guard']} {g['batch2']['verdict']}" if g.get("batch2") else "")
            for lab, g in (gate or {}).items() if not g.get("pass")}


# ------------------------------------------------------------------------------------------------ x35_r1
_R1_KEYS = ("arm", "look", "on", "off", "s_group", "import_off", "dose", "guards_not_read", "void")


def _r1_check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    probs = _check_keys(p, ("arm", "look", "on", "off", "s_group"), _R1_KEYS)
    if p.get("s_group") in s and s[p["s_group"]].pairs < R.MIN_PAIRS:
        probs.append(f"S group pairs {s[p['s_group']].pairs} < the registered {R.MIN_PAIRS} per cell")
    if p.get("arm") not in R.ARMS:
        probs.append(f"arm {p.get('arm')!r} is none of {sorted(R.ARMS)}")
    if p.get("look") not in R.LOOK_SEEDS:
        probs.append(f"look {p.get('look')!r} is none of {sorted(R.LOOK_SEEDS)}")
    if p.get("s_group") not in s:
        probs.append(f"s_group {p.get('s_group')!r} is not an S group")
    return probs + _void_check(p, models, required=p.get("arm") in BATCH2_VOID_ARM)


def _r1_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    g = _group(cells, "s", p["s_group"])
    sides = _seed_side(spec, list(g["nodes"]), p["on"], p["off"])
    r1: Dict[str, Any] = {side: {seed: g["nodes"][lab]["s_pp"] for seed, lab in sorted(sides[side].items())}
                          for side in ("on", "off")}
    if p.get("import_off"):
        if r1["off"]:
            raise RuleInputError("import_off with OFF finals also played in the S group")
        r1["off"] = imported_s(p["import_off"], want_refpop=spec.s[p["s_group"]].refpop_content_sha256)
    if g.get("inconclusive"):
        r1["inconclusive"] = list(g["inconclusive"])          # readout: an INCONCLUSIVE reason, never a score
    doc: Dict[str, Any] = {"arm": p["arm"], "look": p["look"], "r1": r1}
    if p.get("dose"):
        doc["dose"] = dict(p["dose"])
    return doc


def _r1_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    return guards_not_read(_jsonable(R.read(doc).as_dict()), p.get("guards_not_read"))


# ------------------------------------------------------------------------------------------------ d1
_D1_KEYS = ("arm", "look", "on", "off", "anchor_group", "s_group", "reported_group", "guards_not_read", "void",
            "import_off", "determinism_guard", "dose", "reported_without")
#: ``import_off``'s keys: a banked OFF side from ANOTHER read's committed d1 input (D3's PREREG §3 / §4)
_IMPORT_KEYS = ("path", "sha256", "labels", "regimes")


def _d1_check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    probs = _check_keys(p, ("arm", "look", "on", "off", "anchor_group", "s_group"), _D1_KEYS)
    if p.get("s_group") in s and s[p["s_group"]].pairs < R.MIN_PAIRS:
        probs.append(f"S group pairs {s[p['s_group']].pairs} < the registered {R.MIN_PAIRS} per cell")
    if p.get("arm") not in D.PANEL_ROLES:
        probs.append(f"arm {p.get('arm')!r} is none of {sorted(D.PANEL_ROLES)}")
    if p.get("look") not in R.LOOK_SEEDS:
        probs.append(f"look {p.get('look')!r} is none of {sorted(R.LOOK_SEEDS)}")
    a = anchor.get(p.get("anchor_group"))
    if a is None:
        probs.append(f"anchor_group {p.get('anchor_group')!r} is not an anchor group")
    elif tuple(a.teamsets) != D.arm_panels(str(p.get("arm"))):
        probs.append(f"anchor group teamsets {list(a.teamsets)} are not {p.get('arm')}'s primary panels "
                     f"{list(D.arm_panels(str(p.get('arm'))))}")
    elif len(a.unit_seeds) != D.arm_units(str(p.get("arm"))):
        probs.append(f"anchor group plays {len(a.unit_seeds)} unit seeds; {p.get('arm')} registers "
                     f"{D.arm_units(str(p.get('arm')))} per (seed final, panel)")
    rg = p.get("reported_group")
    if rg is not None:
        r = anchor.get(rg)
        if r is None:
            probs.append(f"reported_group {rg!r} is not an anchor group")
        elif rg == p.get("anchor_group"):
            probs.append(f"reported_group {rg!r} is the primary anchor_group")
        elif a is not None and set(r.teamsets) & set(a.teamsets):
            probs.append(f"reported_group {rg!r} repeats a primary panel {sorted(set(r.teamsets) & set(a.teamsets))}")
        elif a is not None and set(r.models) != set(a.models):
            probs.append(f"reported_group {rg!r} plays models {sorted(r.models)}, the primary {sorted(a.models)}")
    if p.get("s_group") not in s:
        probs.append(f"s_group {p.get('s_group')!r} is not an S group")
    probs += _d1_import_check(p, models=models, anchor=anchor)
    rw = p.get("reported_without")
    if rw is not None and (not isinstance(rw, list) or a is None or not set(rw) < set(a.teamsets)):
        probs.append("reported_without: a list of primary panels to drop (at least one panel stays)")
    return probs + _void_check(p, models, required=False)


def _d1_import_check(p: Mapping[str, Any], *, models: Any, anchor: Any) -> List[str]:
    """``import_off`` (``{path, sha256, labels: {seed: label}, regimes: {anchor, s}}``) needs a
    ``determinism_guard`` (``{group, label, banked: {path, sha256, cells}}``): banked Kakuna rows are pooled only
    when a guard group re-plays some of them game for game (D3's PREREG §3); the OFF arm then plays in no primary
    group."""
    probs: List[str] = []
    imp, dg = p.get("import_off"), p.get("determinism_guard")
    if imp is None and dg is None:
        return probs
    if not isinstance(imp, dict) or sorted(imp) != sorted(_IMPORT_KEYS) or not isinstance(imp.get("labels"), dict) \
            or not isinstance(imp.get("regimes"), dict) or sorted(imp["regimes"]) != ["anchor", "s"]:
        probs.append(f"import_off: {{{', '.join(_IMPORT_KEYS)}}} with labels {{seed: label}} and regimes {{anchor, s}}")
    if not isinstance(dg, dict) or dg.get("group") not in anchor or dg.get("group") == p.get("anchor_group") \
            or dg.get("label") not in models or models[dg["label"]].load:
        probs.append("determinism_guard: {group, label, banked}: an anchor group other than the primary playing the "
                     "UNCHANGED model")
    else:
        b = dg.get("banked") or {}
        if not (isinstance(b, dict) and isinstance(b.get("path"), str) and isinstance(b.get("sha256"), str)
                and isinstance(b.get("cells"), dict)):
            probs.append("determinism_guard.banked: {path, sha256, cells: {teamset: read_cell}}")
        if models[dg["label"]].arm != p.get("off"):
            probs.append(f"determinism_guard: label {dg['label']!r} is not an {p.get('off')} final")
    a = anchor.get(p.get("anchor_group"))
    if a is not None and any(models[lab].arm == p.get("off") for lab in a.models if lab in models):
        probs.append(f"import_off: the primary anchor group also plays {p.get('off')} finals")
    return probs


def imported_d1(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """The OFF side ``{h: {off, ties, units}, p: {off, per_final, panel, team_set}}`` of ANOTHER read's committed
    d1 input (``d1_read/read_look2_input.json``), the file's sha256 checked, restricted to ``labels``' seeds: a
    banked control, never re-played (D3's PREREG §3 / §4)."""
    from main.reads.models import sha256_file, spec_path

    path = spec_path(str(entry["path"]))
    got = sha256_file(path)
    if got != entry["sha256"]:
        raise RuleInputError(f"imported d1 input {path}: sha256 {got[:12]} != the spec's {str(entry['sha256'])[:12]}")
    doc = json.loads(path.read_text())
    labels = {str(k): str(v) for k, v in entry["labels"].items()}
    h, pp = doc["h"], doc["p"]
    missing = [s for s in labels if s not in h["off"] or s not in pp["off"] or labels[s] not in pp["per_final"]]
    if missing:
        raise RuleInputError(f"imported d1 input {path}: no OFF rows for seed(s) {missing}")
    return {"h": {k: {s: h[k]["off"][s] if k != "off" else h["off"][s] for s in labels}
                  for k in ("off", "ties", "units")},
            "p": {"off": {s: pp["off"][s] for s in labels}, "per_final": {labels[s]: pp["per_final"][labels[s]]
                                                                          for s in labels},
                  "panel": pp.get("panel"), "team_set": pp.get("team_set")}}


def _d1_guard(cells: Mapping[str, Any], dg: Mapping[str, Any]) -> Dict[str, Any]:
    """The game-for-game determinism guard: each re-played unit of ``dg``'s group against its banked unit."""
    bank = banked_units(dg["banked"])
    g = _group(cells, "anchor", dg["group"])
    checks = []
    for u in g["units"]:
        want = bank.get(u["panel"], {}).get(int(u["unit_seed"]))
        checks.append({"unit": u["unit"], "panel": u["panel"], "unit_seed": int(u["unit_seed"]),
                       "counted": bool(u["counted"]), "n": u.get("n"), "wins": u.get("wins"),
                       "banked": None if want is None else {"n": want["n"], "wins": want["wins"]},
                       "games_equal": (want is not None and u.get("games") is not None
                                       and [list(x) for x in u["games"]] == want["games"])})
    planned = sum(len(by) for by in bank.values())
    why = [] if checks else ["no guard unit played"]
    why += [f"{c['unit']}: not counted" for c in checks if not c["counted"]]
    why += [f"{c['unit']}: its games differ from the banked unit's" for c in checks if not c["games_equal"]]
    if len(checks) != planned:
        why.append(f"{len(checks)} guard unit(s) played, {planned} banked")
    return {"label": dg["label"], "group": dg["group"], "units": checks, "regime_id": g.get("regime_id"),
            "banked_source": {"path": dg["banked"]["path"], "sha256": dg["banked"]["sha256"]},
            "passed": not why, "reasons": why}


def h_input(group: Mapping[str, Any], spec: Any, on: str, off: str) -> Dict[str, Any]:
    """D1's H (``kakuna.d1h_input``'s shape) from an assembled anchor group: each seed final's own counts per
    panel over COUNTED units, ties and units beside."""
    out: Dict[str, Any] = {"on": {}, "off": {}, "ties": {"on": {}, "off": {}}, "units": {"on": {}, "off": {}}}
    sides = _seed_side(spec, list(group["cells"]), on, off)
    for side in ("on", "off"):
        for seed, lab in sorted(sides[side].items()):
            if not any(c["units"] or c["skipped"] for c in group["cells"][lab].values()):
                continue                                      # nothing played (``d1h_input`` has no such final)
            for ts, c in sorted(group["cells"][lab].items()):
                out[side].setdefault(seed, {})[ts] = [int(c["wins"]), int(c["n"])]
                out["ties"][side].setdefault(seed, {})[ts] = int(c["ties"])
                out["units"][side].setdefault(seed, {})[ts] = {"counted": int(c["units"]),
                                                               "skipped": list(c["skipped"])}
    return out


def p_input(group: Mapping[str, Any], spec: Any, on: str, off: str) -> Dict[str, Any]:
    """D1's P (``d1.p_input``'s shape) from an assembled S group on a registered panel."""
    sides = _seed_side(spec, list(group["nodes"]), on, off)
    out: Dict[str, Any] = {"on": {}, "off": {}, "inconclusive": list(group["inconclusive"]), "per_final": {},
                           "panel": group["panel"], "team_set": group["team_set"]}
    for side in ("on", "off"):
        for seed, lab in sorted(sides[side].items()):
            node = group["nodes"][lab]
            out[side][seed] = node["s_pp"]
            out["per_final"][lab] = node["population_score"]
    return out


def _d1_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    doc = {"arm": p["arm"], "look": p["look"],
           "h": h_input(_group(cells, "anchor", p["anchor_group"]), spec, p["on"], p["off"]),
           "p": p_input(_group(cells, "s", p["s_group"]), spec, p["on"], p["off"])}
    if p.get("reported_group") is not None:                  # REPORTED panels (``d1.reported_panels``), no rule
        from main.reads.assemble import group_reasons

        rg = _group(cells, "anchor", p["reported_group"])
        r = h_input(rg, spec, p["on"], p["off"])
        doc["reported_panels"] = {"on": r["on"], "off": r["off"]}
        why = group_reasons("anchor", p["reported_group"], rg)
        if why:                                              # INCOMPLETE: reported as such, never blocking (F7)
            doc["reported_panels"]["incomplete"] = why
    if p.get("import_off") is not None:
        _d1_import(doc, cells, p)
    if p.get("dose"):
        doc["dose"] = dict(p["dose"])
    return doc


def _d1_import(doc: Dict[str, Any], cells: Mapping[str, Any], p: Mapping[str, Any]) -> None:
    """Pool the banked OFF side into ``doc`` (H only when the determinism guard passes; P only on its regime)."""
    imp = p["import_off"]
    if doc["h"]["off"] or doc["p"]["off"]:
        raise RuleInputError("import_off with OFF finals also played in this read's primary groups")
    got = {"anchor": _group(cells, "anchor", p["anchor_group"]).get("regime_id"),
           "s": _group(cells, "s", p["s_group"]).get("regime_id")}
    for lane in ("anchor", "s"):
        if got[lane] != imp["regimes"][lane]:
            raise RuleInputError(f"import_off: this read's {lane} regime {got[lane]} is not the banked rows' "
                                 f"{imp['regimes'][lane]} (re-play the control)")
    bank = imported_d1(imp)
    if (bank["p"]["panel"], bank["p"]["team_set"]) != (doc["p"]["panel"], doc["p"]["team_set"]):
        raise RuleInputError(f"import_off: the banked P panel / team set {bank['p']['panel']} / "
                             f"{bank['p']['team_set']} is not this read's {doc['p']['panel']} / {doc['p']['team_set']}")
    guard = _d1_guard(cells, p["determinism_guard"])
    doc["import_off"] = {"path": imp["path"], "sha256": imp["sha256"], "labels": dict(imp["labels"]),
                         "regimes": dict(imp["regimes"]), "h_pooled": guard["passed"], "guard": guard}
    doc["p"]["off"] = bank["p"]["off"]
    doc["p"]["per_final"].update(bank["p"]["per_final"])
    if guard["passed"]:
        for k in ("off", "ties", "units"):
            tgt = doc["h"][k] if k == "off" else doc["h"][k]["off"]
            tgt.update(bank["h"][k])


def _d1_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    v = guards_not_read(_jsonable(D.read_d1(doc).as_dict()), p.get("guards_not_read"))
    why = (doc.get("reported_panels") or {}).get("incomplete")
    if why:
        v["reported"]["reported_panels"].update(status="INCOMPLETE", incomplete=list(why))
    imp = doc.get("import_off")
    if imp is not None and not imp["h_pooled"]:
        v["reasons"] = list(v["reasons"]) + [f"the determinism guard failed ({'; '.join(imp['guard']['reasons'])}): "
                                             "the banked OFF Kakuna rows are not pooled; re-play every OFF unit"]
    if p.get("reported_without"):
        v["reported"]["without"] = _d1_without(doc, list(p["reported_without"]))
    return v


def _d1_reported(p: Mapping[str, Any]) -> List[Tuple[str, str]]:
    return [("anchor", str(p["reported_group"]))] if p.get("reported_group") is not None else []


def _d1_without(doc: Mapping[str, Any], drop: Sequence[str]) -> Dict[str, Any]:
    """DESCRIPTIVE, no rule reads it: H's own statistic (``d1_primary``, AM-5 widening, sign rule) recomputed with
    the ``drop`` panels left out (the goal-(b) panel-validity ruling's "without H", 4f555e57)."""
    arm, look = str(doc.get("arm")), int(doc["look"])
    keep = tuple(k for k in D.arm_panels(arm) if k not in drop)
    sub = {side: {seed: {k: c for k, c in by.items() if k in keep} for seed, by in (doc["h"].get(side) or {}).items()}
           for side in ("on", "off")}
    h = D.d1_primary(look, sub["on"], sub["off"], panels=keep, roles=D.PANEL_ROLES.get(arm, D.PANEL_ROLE),
                     units=D.arm_units(arm))
    return dict(_jsonable(h.as_dict()), panels=list(keep), dropped=list(drop))


# --------------------------------------------------------------------------------------- anchor_trajectory
_AT_KEYS = ("on", "off", "anchor_group", "panels", "unit_seeds", "min_unit_frac", "goal_bar", "subset_below",
            "s1", "s_baseline", "k9b", "verdict_text")
#: the verdict's words (a read may register its own, e.g. x26c's "goal (b)")
AT_VERDICT_TEXT = {"met": "GOAL MET", "not_met": "goal not met"}
VOID = "VOID"


def _at_check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    probs = _check_keys(p, ("on", "off", "anchor_group", "goal_bar", "min_unit_frac"), _AT_KEYS)
    a = anchor.get(p.get("anchor_group"))
    if a is None:
        probs.append(f"anchor_group {p.get('anchor_group')!r} is not an anchor group")
    for k in ("on", "off"):
        if p.get(k) not in models:
            probs.append(f"{k} {p.get(k)!r} is not a model label")
    s1 = p.get("s1")
    if s1 is not None and (not isinstance(s1, dict) or s1.get("group") not in s or s1.get("label") not in models):
        probs.append("s1: {group: <S group>, label: <model>}")
    b = p.get("s_baseline")
    if s1 is not None and not (isinstance(b, dict) and isinstance(b.get("s_pp"), (int, float))
                               and isinstance(b.get("se_pp"), (int, float))):
        probs.append("s_baseline: {s_pp, se_pp, source} (the banked node S1 is read against)")
    return probs


def _at_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    g = _group(cells, "anchor", p["anchor_group"])
    a = spec.anchor[p["anchor_group"]]
    doc: Dict[str, Any] = {"on": p["on"], "off": p["off"], "panels": list(p.get("panels") or a.teamsets),
                           "unit_seeds": list(p.get("unit_seeds") or a.unit_seeds),
                           "units": list(g["units"]),
                           "s1": None, "k9b_verdict": None}
    doc["units_planned"] = len(doc["panels"]) * len(doc["unit_seeds"])
    if p.get("s1"):
        sg = _group(cells, "s", p["s1"]["group"])
        node = sg["nodes"][p["s1"]["label"]]
        bad = node["inconclusive"] or sg.get("inconclusive")
        doc["s1"] = None if bad else {"s_pp": node["s_pp"], "se_pp": node["se_pp"]}
    if p.get("k9b"):
        from main.reads.models import spec_path

        doc["k9b_verdict"] = json.loads(spec_path(str(p["k9b"])).read_text())["verdict"]
    return doc


def anchor_block(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    """x26c ``verdict.kakuna_block``: per label the stratified score over COUNTED units, Δ = Newcombe of strata,
    the unit-paired difference and the low-seed subset (both descriptive)."""
    on, off, panels = doc["on"], doc["off"], list(doc["panels"])
    pooled: Dict[str, Dict[str, Dict[str, Any]]] = {}
    per: Dict[Tuple[str, str, int], float] = {}
    for u in doc["units"]:
        c = pooled.setdefault(u["label"], {}).setdefault(u["panel"], {"wins": 0, "ties": 0, "n": 0, "units": 0,
                                                                          "skipped": []})
        if not u["counted"]:
            c["skipped"].append(u["unit"])
            continue
        c["units"] += 1
        c["wins"] += int(u["wins"])
        c["ties"] += int(u.get("ties", 0))
        c["n"] += int(u["n"])
        per[(u["label"], u["panel"], int(u["unit_seed"]))] = int(u["wins"]) / int(u["n"])
    out: Dict[str, Any] = {}
    for lab in (on, off):
        cells = pooled.get(lab) or {}
        missing = [k for k in panels if k not in cells]
        if missing:
            raise RuleInputError(f"{lab}: no units on {missing}")
        ss = strat_score({k: (cells[k]["wins"], cells[k]["n"]) for k in panels}, panels)
        out[lab] = dict(ss.as_dict(), units={k: cells[k]["units"] for k in panels},
                        skipped={k: cells[k]["skipped"] for k in panels}, ties={k: cells[k]["ties"] for k in panels},
                        units_counted=sum(cells[k]["units"] for k in panels))
    # the panels are passed: x26c's verdict.py omitted them, correct only because its panels were d1's default
    d, lo, hi = newcombe_strat(strat_score({k: (pooled[on][k]["wins"], pooled[on][k]["n"]) for k in panels}, panels),
                               strat_score({k: (pooled[off][k]["wins"], pooled[off][k]["n"]) for k in panels}, panels))
    out["delta"] = {"d": d, "lo": lo, "hi": hi}
    pu = paired_unit_diff(per, on, off, panels, doc["unit_seeds"])
    if pu is not None:
        out["paired_units"] = pu
    below = p.get("subset_below")
    if below is not None:
        sub = {}
        for lab in (on, off):
            w = {k: [0, 0] for k in panels}
            for u in doc["units"]:
                if u["label"] == lab and int(u["unit_seed"]) < int(below) and u["counted"]:
                    w[u["panel"]][0] += int(u["wins"])
                    w[u["panel"]][1] += int(u["n"])
            if all(n for _x, n in w.values()):
                sub[lab] = strat_score({k: (v[0], v[1]) for k, v in w.items()}, panels).as_dict()
        out[f"seeds_0_{int(below) - 1}"] = sub
    return out


def _at_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    """x26c PREREG §5 (``verdict.decide``): VOID on a non-PASS K9(b) or < ``min_unit_frac`` units counted; goal
    met iff the ON score's Wilson LB clears ``goal_bar``; CLIMBING iff Δ's LB clears 0, else FLAT; S1 RISING iff
    (S1 − baseline)'s normal LB clears 0 (se's added in quadrature), read only when FLAT (``d3_revisit``)."""
    k = anchor_block(doc, p)
    on, off = doc["on"], doc["off"]
    reasons = []
    if p.get("k9b") and doc.get("k9b_verdict") != "PASS":
        reasons.append(f"K9(b) {doc.get('k9b_verdict')}")
    for lab in (on, off):
        if k[lab]["units_counted"] < float(p["min_unit_frac"]) * int(doc["units_planned"]) - RULE8_EPS:
            reasons.append(f"{lab}: {k[lab]['units_counted']} / {doc['units_planned']} units counted "
                           f"(< {float(p['min_unit_frac']):.0%})")
    if reasons:
        return {"kakuna": k, "decision": {"verdict": VOID, "reasons": reasons}}
    goal = clears(k[on]["wilson95"][0], float(p["goal_bar"]))[0]
    climbing = clears(k["delta"]["lo"], 0.0)[0]
    dec: Dict[str, Any] = {"goal_met": goal, "trajectory": "CLIMBING" if climbing else "FLAT"}
    s1, base = doc.get("s1"), p.get("s_baseline")
    if p.get("s1") is not None:
        if s1 is not None and s1.get("se_pp") is not None:
            d = s1["s_pp"] - base["s_pp"]
            se = math.sqrt(s1["se_pp"] ** 2 + base["se_pp"] ** 2)
            rising = clears(d - Z95 * se, 0.0)[0]
            dec["s1"] = {"s_pp": s1["s_pp"], "se_pp": s1["se_pp"], "baseline": dict(base), "d_vs_baseline": d,
                         "lo": d - Z95 * se, "hi": d + Z95 * se, "rising": rising}
            if not climbing:
                dec["d3_revisit"] = rising
        else:
            dec["s1"] = None
            if not climbing:
                dec["d3_revisit"] = "UNREAD (S1 missing)"
    words = dict(AT_VERDICT_TEXT, **(p.get("verdict_text") or {}))
    dec["verdict"] = f"{words['met'] if goal else words['not_met']}; {dec['trajectory']}"
    return {"kakuna": k, "decision": dec}


# ------------------------------------------------------------------------------------------------ prior_mix_ab
_PM_KEYS = ("arms", "primary", "banked", "guard", "unfamiliar", "familiar", "heldout",
            "min_unit_frac", "home_margin_pp", "did_ub_pp")
PM_HELPS, PM_HELPS_COST, PM_HURTS, PM_ND = "PRIOR-HELPS", "PRIOR-HELPS-AT-A-COST", "PRIOR-HURTS", "NOT DETECTED"


def _pm_check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    probs = _check_keys(p, _PM_KEYS, _PM_KEYS)
    arms = p.get("arms") or {}
    if not isinstance(arms, dict) or not arms:
        return probs + ["arms: {name: {group, label}}"]
    for name, a in arms.items():
        if not isinstance(a, dict) or a.get("group") not in anchor or a.get("label") not in models:
            probs.append(f"arms.{name}: {{group: <anchor group>, label: <model>}}")
            continue
        if a["label"] not in anchor[a["group"]].models or len(anchor[a["group"]].models) != 1:
            probs.append(f"arms.{name}: group {a['group']!r} must play exactly its label {a['label']!r} (one "
                         "regime per group)")
        if "belief_prior_mix" not in models[a["label"]].load:
            probs.append(f"arms.{name}: model {a['label']!r} carries no load.belief_prior_mix")
    if p.get("primary") not in arms:
        probs.append(f"primary {p.get('primary')!r} is not an arm")
    g = p.get("guard") or {}
    if g.get("group") not in anchor or g.get("label") not in models or models[g["label"]].load:
        probs.append("guard: {group, label}: an anchor group playing the UNCHANGED model (no load options)")
    b = p.get("banked") or {}
    if not (isinstance(b, dict) and isinstance(b.get("path"), str) and isinstance(b.get("sha256"), str)
            and isinstance(b.get("cells"), dict)):
        probs.append("banked: {path, sha256, cells: {teamset: read_cell}}")
    panels = list(p.get("unfamiliar") or []) + [p.get("familiar")]
    for name, a in arms.items():
        if isinstance(a, dict) and a.get("group") in anchor:
            ts = set(anchor[a["group"]].teamsets)
            if not set(panels) <= ts:
                probs.append(f"arms.{name}: group {a['group']!r} lacks team set(s) {sorted(set(panels) - ts)}")
    if p.get("heldout") not in (p.get("unfamiliar") or []):
        probs.append("heldout: one of the unfamiliar panels (D0f's 'prior-only − full on heldout72')")
    return probs


def banked_units(entry: Mapping[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """``{teamset: {unit_seed: {wins, ties, n, games}}}`` from ANOTHER read's committed per-game rows
    (``entry = {path, sha256, cells: {teamset: read_cell}}``), the file's sha256 checked: the banked control."""
    from main.reads.assemble import GAME_KEYS
    from main.reads.models import sha256_file, spec_path

    p = spec_path(str(entry["path"]))
    got = sha256_file(p)
    if got != entry["sha256"]:
        raise RuleInputError(f"banked rows {p}: sha256 {got[:12]} != the spec's {str(entry['sha256'])[:12]}")
    cell_of = {v: k for k, v in dict(entry["cells"]).items()}
    out: Dict[str, Dict[int, Dict[str, Any]]] = {k: {} for k in entry["cells"]}
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        ts = cell_of.get(r.get("read_cell"))
        if ts is None:
            continue
        u = out[ts].setdefault(int(r["unit_seed"]), {"wins": 0, "ties": 0, "n": 0, "games": []})
        u["n"] += 1
        u["wins"] += int(r["result"] == "win")
        u["ties"] += int(r["result"] == "tie")
        u["games"].append([r.get(k) for k in GAME_KEYS])
    for ts in out:
        for u in out[ts].values():
            u["games"].sort(key=lambda g: (str(g[0]), int(g[1])))
    return out


def _pm_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    bank = banked_units(p["banked"])
    doc: Dict[str, Any] = {"banked": {ts: {str(s): {k: v for k, v in u.items() if k != "games"}
                                           for s, u in sorted(by.items())} for ts, by in bank.items()},
                           "banked_source": {"path": p["banked"]["path"], "sha256": p["banked"]["sha256"]},
                           "arms": {}, "guard": None}
    for name, a in p["arms"].items():
        g = _group(cells, "anchor", a["group"])
        lam = spec.models[a["label"]].load["belief_prior_mix"]
        doc["arms"][name] = {"label": a["label"], "lambda": lam, "regime_id": g.get("regime_id"),
                             "units_planned": {ts: len(g["unit_seeds"]) for ts in g["teamsets"]},
                             "units": [{k: u.get(k) for k in ("unit", "panel", "unit_seed", "counted", "wins", "ties",
                                                              "n", "our_regime", "belief_prior_mix")}
                                       for u in g["units"] if u["label"] == a["label"]]}
    gg = _group(cells, "anchor", p["guard"]["group"])
    checks = []
    for u in gg["units"]:
        want = bank.get(u["panel"], {}).get(int(u["unit_seed"]))
        checks.append({"unit": u["unit"], "panel": u["panel"], "unit_seed": int(u["unit_seed"]),
                       "counted": bool(u["counted"]), "n": u.get("n"), "wins": u.get("wins"),
                       "banked": None if want is None else {"n": want["n"], "wins": want["wins"]},
                       "games_equal": (want is not None and u.get("games") is not None
                                       and [list(x) for x in u["games"]] == want["games"]),
                       "our_regime": u.get("our_regime")})
    doc["guard"] = {"label": p["guard"]["label"], "units": checks, "regime_ids": gg.get("regime_ids")}
    return doc


def _pm_paired(arm: Mapping[str, Any], bank: Mapping[str, Mapping[str, Any]], panels: Sequence[str]
               ) -> Optional[Dict[str, Any]]:
    """The unit-paired Δ (arm − banked λ = 0) over ``panels``, equal weight (:func:`paired_unit_diff`)."""
    per: Dict[Tuple[str, str, int], float] = {}
    seeds = set()
    for u in arm["units"]:
        if u["counted"] and u["panel"] in panels:
            per[("on", u["panel"], int(u["unit_seed"]))] = int(u["wins"]) / int(u["n"])
            seeds.add(int(u["unit_seed"]))
    for ts in panels:
        for s, b in bank.get(ts, {}).items():
            per[("off", ts, int(s))] = int(b["wins"]) / int(b["n"])
            seeds.add(int(s))
    return paired_unit_diff(per, "on", "off", list(panels), sorted(seeds))


def _pm_class(d: Optional[Mapping[str, Any]]) -> str:
    if d is None:
        return "UNREAD"
    if clears(d["lo"], 0.0)[0]:
        return "GAIN"
    if clears(d["hi"], 0.0, above=False)[0]:
        return "LOSS"
    return PM_ND


def _pm_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    """The PREREG's rule (``belief_prior_mix_2026-10-26/PREREG.md`` §5): VOID on a failed determinism guard or an
    under-counted arm; per arm Δ_U (unfamiliar, unit-paired, equal weight) and Δ_home against the banked λ = 0 and
    its class; for an arm at λ = 1 (prior only) D0f's read too (the DiD and H3's kill rule). The verdict line names
    the primary arm first."""
    bank = {ts: {int(s): u for s, u in by.items()} for ts, by in doc["banked"].items()}
    unf, fam, held = list(p["unfamiliar"]), str(p["familiar"]), str(p["heldout"])
    reasons: List[str] = []
    g = doc["guard"]["units"]
    if not g:
        reasons.append("guard: no unit played")
    for c in g:
        if not c["counted"]:
            reasons.append(f"guard {c['unit']}: not counted")
        elif not c["games_equal"]:
            reasons.append(f"guard {c['unit']}: its games differ from the banked unit's (determinism broken)")
        if c.get("our_regime") != "greedy":
            reasons.append(f"guard {c['unit']}: our_regime {c.get('our_regime')!r} is not the banked 'greedy'")
    out: Dict[str, Any] = {"arms": {}}
    for name, a in doc["arms"].items():
        want = f"greedy+belief_prior_mix={float(a['lambda']):g}"
        bad = sorted({str(u["our_regime"]) for u in a["units"] if u["counted"] and u["our_regime"] != want})
        if bad:
            reasons.append(f"{name}: counted units stamped {bad}, not {want!r}")
        counted = {ts: sum(1 for u in a["units"] if u["counted"] and u["panel"] == ts) for ts in unf + [fam]}
        for ts, n in counted.items():
            planned = int(a["units_planned"][ts])
            if n < float(p["min_unit_frac"]) * planned - RULE8_EPS:
                reasons.append(f"{name} {ts}: {n} / {planned} units counted (< {float(p['min_unit_frac']):.0%})")
        cells = {}
        for ts in unf + [fam]:
            w = sum(int(u["wins"]) for u in a["units"] if u["counted"] and u["panel"] == ts)
            n = sum(int(u["n"]) for u in a["units"] if u["counted"] and u["panel"] == ts)
            bw = sum(int(b["wins"]) for b in bank[ts].values())
            bn = sum(int(b["n"]) for b in bank[ts].values())
            _p, lo, hi = wilson(w, n) if n else (None, None, None)
            cells[ts] = {"wins": w, "n": n, "score": w / n if n else None, "wilson95": [lo, hi],
                         "banked_wins": bw, "banked_n": bn, "paired": _pm_paired(a, bank, [ts])}
        ss_on = strat_score({k: (cells[k]["wins"], cells[k]["n"]) for k in unf}, unf) if all(
            cells[k]["n"] for k in unf) else None
        ss_off = strat_score({k: (cells[k]["banked_wins"], cells[k]["banked_n"]) for k in unf}, unf)
        nc = newcombe_strat(ss_on, ss_off) if ss_on is not None else None
        out["arms"][name] = {"lambda": a["lambda"], "cells": cells,
                             "u_score": None if ss_on is None else ss_on.as_dict(), "u_banked": ss_off.as_dict(),
                             "delta_u": _pm_paired(a, bank, unf), "delta_home": cells[fam]["paired"],
                             "newcombe_u": None if nc is None else {"d": nc[0], "lo": nc[1], "hi": nc[2]}}
    if reasons:
        return {"arms": out["arms"], "decision": {"verdict": VOID, "reasons": reasons}}
    margin = float(p["home_margin_pp"]) / 100.0
    dec: Dict[str, Any] = {"primary": p["primary"], "arms": {}}
    for name, r in out["arms"].items():
        du, dh = r["delta_u"], r["delta_home"]
        cu, ch = _pm_class(du), _pm_class(dh)
        harmed = dh is not None and (ch == "LOSS" or clears(dh["d"], -margin, above=False)[0])
        verdict = (PM_HELPS if cu == "GAIN" and not harmed else PM_HELPS_COST if cu == "GAIN"
                   else PM_HURTS if cu == "LOSS" else PM_ND)
        a: Dict[str, Any] = {"lambda": r["lambda"], "u_class": cu, "home_class": ch, "home_harmed": harmed,
                             "verdict": verdict, "d0f": None}
        if float(r["lambda"]) == 1.0 and du is not None and dh is not None:
            # D0f (prior only): the DiD (U − home72) and H3's kill rule (design_diversity_teams_opponents §3.3)
            d = du["d"] - dh["d"]
            se = math.sqrt(((du["hi"] - du["lo"]) / (2 * Z95)) ** 2 + ((dh["hi"] - dh["lo"]) / (2 * Z95)) ** 2)
            held_d = r["cells"][held]["paired"]
            killed = (clears(d + Z95 * se, float(p["did_ub_pp"]) / 100.0, above=False)[0]
                      and held_d is not None and not clears(held_d["lo"], 0.0)[0])
            a["d0f"] = {"did": d, "lo": d - Z95 * se, "hi": d + Z95 * se, "heldout": held_d, "h3_killed": killed}
        dec["arms"][name] = a
    order = [p["primary"]] + sorted(n for n in dec["arms"] if n != p["primary"])
    dec["verdict"] = " | ".join(
        f"lambda={dec['arms'][n]['lambda']:g}: {dec['arms'][n]['verdict']}"
        + (f", D0f H3 {'KILLED' if dec['arms'][n]['d0f']['h3_killed'] else 'NOT KILLED'}"
           if dec["arms"][n]["d0f"] else "") for n in order)
    return {"arms": out["arms"], "decision": dec}


# ------------------------------------------------------------------------------------------------ registry
@dataclass(frozen=True)
class Rule:
    name: str
    doc: str
    check_params: Callable[..., List[str]]
    inputs: Callable[[Mapping[str, Any], Mapping[str, Any], Any], Dict[str, Any]]
    decide: Callable[[Mapping[str, Any], Mapping[str, Any]], Dict[str, Any]]
    #: an X35 stage record (top-level arm / look / verdict / primary; ``batch.read_record`` loads it)
    x35_record: bool
    #: the REPORTED-only ``(lane, group)``s of these params: no verdict reads them, so an incomplete one never blocks
    #: the record (it is recorded INCOMPLETE with its reasons)
    reported: Callable[[Mapping[str, Any]], List[Tuple[str, str]]] = lambda p: []

    def verdict_of(self, decision: Mapping[str, Any]) -> str:
        v = decision.get("verdict") if self.x35_record else (decision.get("decision") or {}).get("verdict")
        if not isinstance(v, str) or not v:
            raise RuleInputError(f"{self.name}: the decision carries no verdict")
        return v


RULES: Dict[str, Rule] = {
    "x35_r1": Rule("x35_r1", "X35 R1 / R2 / guards (main.x35.readout.read)", _r1_check, _r1_inputs, _r1_decide, True),
    "d1": Rule("d1", "D1 H (unfamiliar Kakuna panels, φ̂-widened) + P (main.x35.d1.read_d1)", _d1_check, _d1_inputs,
               _d1_decide, True, _d1_reported),
    "anchor_trajectory": Rule("anchor_trajectory", "x26c PREREG §5: goal bar, CLIMBING / FLAT, S1 rising",
                              _at_check, _at_inputs, _at_decide, False),
    "prior_mix_ab": Rule("prior_mix_ab", "belief prior mix: λ arms vs a banked λ = 0, unit-paired; a determinism guard",
                         _pm_check, _pm_inputs, _pm_decide, False),
}


def inputs(rule: str, cells: Mapping[str, Any], params: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    return RULES[rule].inputs(cells, params, spec)


def decide(rule: str, doc: Mapping[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
    return RULES[rule].decide(doc, params)
