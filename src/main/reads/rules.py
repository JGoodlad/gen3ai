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
THE VOID GATE (:func:`void_gate`): an X35-stage rule (``x35_r1``, ``d1``, ``n4_screen``) may register ``params.void =
{k9b_dir, runs: {label: run dir}}``; ``x35_r1`` on a batch-2 arm (SPRT, LRF) and ``n4_screen`` always MUST. ``decide`` runs it BEFORE the rule: each named run's
AM-6 K9(b) record (``<k9b_dir>/<run>.json``) must read PASS, and a batch-2 arm's V-S1 / V-S2 (SPRT) or LR-F KL guard
(``main.x35.batch2_void.read_void``) too; anything else refuses the decision (a VOID seed enters no read; SPRT's
``assemble.py`` STOP, be60f24f FINDING 3). Two further guards read the RUN DIR's own artefacts and are OPT-IN, each a
key of ``params.void`` (a spec that names neither is gated exactly as before): ``pin`` (a commit sha or prefix: every
git hash the run recorded, ``metadata.json``'s ``git_hash`` and each ``pin_history`` span, must be that commit; N1-P's
PREREG §9 guard 3) and ``pool_seed`` (``{fork_step}``: ``<run>/snapshots/pool_seed.json`` is the T54 schema-2 record, its
``fork_step`` the registered one, no seeded snapshot later than it and every dropped one later; guard 4's record half).
What stays an operator check: guard 2's startup banner and held-out hit lines (the launcher log is outside the run dir)
and the pool-seed check's startup-line half.

REPORTED-ONLY GROUPS (``Rule.reported``): the groups a rule carries for description and no verdict reads (``d1``'s
``reported_group``). An incomplete one never blocks the record of the decided primary: the record names it INCOMPLETE
with its reasons (N1-P FINDING 7).

* ``prior_mix_ab`` -- the belief-prior-mix read (``belief_prior_mix_2026-10-26`` PREREG): each λ arm against a
  BANKED λ = 0 control (another read's committed per-game rows, sha-checked, never re-played), unit-paired on the
  same seeds, with a game-for-game determinism guard that re-plays a few banked units at λ = 0.
* ``score_bands`` -- the H-CTX read (``hctx_rc72_2026-10-26`` PREREG): ONE model's pooled score on ONE panel placed in
  registered bands (``>= kill_at`` KILL, ``<= support_at`` SUPPORT, else UNDECIDED; exact fractions), VOID unless a
  guard group replays banked units game for game (``prior_mix_ab``'s guard); its comparisons with the banked familiar
  and unfamiliar cells are descriptive. SUPERSEDED for that read, before any rc72 game, by ``premium_recovery``.
* ``premium_recovery`` -- the H-CTX read as amended (engine-identity rule): rc72, home72 and an unfamiliar panel, all
  played on ONE engine in one read; the share of the home premium (familiar − unfamiliar) rc72 recovers, by exact
  fractions against ``kill_frac`` / ``support_frac``; INCONCLUSIVE unless the premium's unit-bootstrap lower bound
  clears 0; the absolute bands against X51 survive only as a secondary, cross-engine line.
* ``x59s_floor`` / ``x59s_gate`` / ``x59s`` -- X59-S's floor pilot (superseded by the per-contrast gate, Amendment 5),
  the gate and the full read (``x59s_matched_students_2026-10-25`` PREREG
  Amendment 4; :mod:`main.reads.rules_x59s`): ``main.human_data.x59s_read``'s registered statistics on the
  students' Kakuna cells and the read's sha-recorded artefacts.
* ``search_headroom`` -- S-U (``design_search_u_headroom.md`` §3; :mod:`main.reads.rules_search_headroom`): a searched
  label against its raw twin, paired by deal in ONE ``per_game`` group; I = G_U − G_home72 and G_U by a within-panel
  unit bootstrap against exact-fraction bars, I futility-only at phase 1; VOID on G-PREFIX, engines, bytes or the
  search config.
* ``n4_screen`` -- the one-look capacity screen (N4 PREREG §5 with its review's M2 / M4 / M5; X48 PREREG §5;
  :mod:`main.reads.rules_n4_screen`): S by Welch 90 % on the finals, K by the ``d1`` estimators on forum72;
  INCONCLUSIVE > MIXED > GO > NO-GO > UNDECIDED, the LR-confounded label and M5's NO-GO wording; an X35-stage record.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from main.anchors.results import Z95, newcombe, wilson
from main.reads import engine as EN
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
    from main.reads.models import spec_path
    from utils.hashing import sha256_file

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
    if not (isinstance(v, dict) and {"k9b_dir", "runs"} <= set(v) <= {"k9b_dir", "runs", *VOID_RUN_DIR_GUARDS}
            and isinstance(v["k9b_dir"], str) and isinstance(v["runs"], dict) and v["runs"]):
        return ["void: {k9b_dir: <dir of AM-6 K9(b) records>, runs: {label: run dir}, [pin: <sha>], "
                "[pool_seed: {fork_step: <int>}]}"]
    probs = [f"void.runs: {lab!r} is not a model label" for lab in v["runs"] if lab not in models]
    if "pin" in v and not (isinstance(v["pin"], str) and re.fullmatch(r"[0-9a-f]{7,40}", v["pin"])):
        probs.append("void.pin: a commit sha (7 to 40 lowercase hex digits)")
    if "pool_seed" in v and not (isinstance(v["pool_seed"], dict) and set(v["pool_seed"]) == {"fork_step"}
                                 and isinstance(v["pool_seed"]["fork_step"], int)
                                 and not isinstance(v["pool_seed"]["fork_step"], bool) and v["pool_seed"]["fork_step"] > 0):
        probs.append("void.pool_seed: {fork_step: <the fork parent's step, a positive integer>}")
    probs += [f"void.runs.{lab}: a run dir path" for lab, r in v["runs"].items() if not (isinstance(r, str) and r)]
    on = [lab for lab, m in models.items() if m.arm == p.get("on")]
    probs += [f"void.runs: the ON model {lab!r} has no run (every ON seed is gated)" for lab in on
              if lab not in v["runs"]]
    return probs


#: The ``params.void`` keys that read the run dir itself (opt-in; module docstring).
VOID_RUN_DIR_GUARDS = ("pin", "pool_seed")


def void_pin_guard(run_dir: Any, want: str) -> Dict[str, Any]:
    """PREREG §9 guard 3's record half: every git hash the run recorded (``main.x35.batch.recorded_pins``: ``metadata.json``'s
    ``git_hash`` and each ``pin_history`` span) is the registered commit ``want`` (a sha or prefix of at least 7 digits).
    ``{want, recorded, pass, why}``; a run with no metadata, or no ``pin_history`` span, fails (it cannot show its pin)."""
    from main.x35 import batch as B

    try:
        recorded = B.recorded_pins(str(run_dir))
    except B.BatchError as exc:
        return {"want": want, "recorded": [], "pass": False, "why": str(exc)}
    meta = json.loads((Path(run_dir) / "metadata.json").read_text()) if recorded else {}
    spans = [e for e in meta.get("pin_history") or [] if isinstance(e, dict) and e.get("git_hash")]

    def same(a: str, b: str) -> bool:
        n = min(len(a), len(b))
        return n >= 7 and a[:n] == b[:n]

    if not recorded:
        why = "no metadata.json (or no recorded git hash): the run cannot show its pin"
    elif not spans:
        why = "metadata.json holds no pin_history span: the run cannot show its pin"
    else:
        bad = sorted({h for h in recorded if not same(h, want)})
        why = f"recorded commit(s) {[h[:12] for h in bad]} are not the registered {want[:12]}" if bad else ""
    return {"want": want, "recorded": sorted(set(recorded)), "pass": not why, "why": why}


def void_pool_seed_guard(run_dir: Any, fork_step: int) -> Dict[str, Any]:
    """PREREG §9 guard 4's record half (T54, ``kit.check_pool_seed``'s checks that need no call record or log): the run's
    ``snapshots/pool_seed.json`` is schema 2, its ``fork_step`` is ``fork_step`` (the fork parent's step, registered), every
    seeded snapshot is at or before it and every dropped step is after it. ``{fork_step, got, snapshots, dropped, pass,
    why}`` (``why`` lists every failure)."""
    from agents.training.pool_seed import SEED_RECORD_SCHEMA, pool_dir_for, read_seed_record, snapshot_step

    rec = read_seed_record(pool_dir_for(str(run_dir)))
    out: Dict[str, Any] = {"fork_step": fork_step, "got": None, "snapshots": 0, "dropped": 0, "pass": False, "why": ""}
    if rec is None:
        out["why"] = "no readable snapshots/pool_seed.json"
        return out
    probs: List[str] = []
    if rec.get("schema") != SEED_RECORD_SCHEMA:
        probs.append(f"schema {rec.get('schema')!r} != {SEED_RECORD_SCHEMA} (a pin without T54?)")
    fs = rec.get("fork_step")
    out["got"] = fs
    snaps, dropped = list(rec.get("snapshots") or []), list(rec.get("dropped_steps") or [])
    out["snapshots"], out["dropped"] = len(snaps), len(dropped)
    if fs is None:
        probs.append("fork_step is null (T54 could not read the step and seeded UNFILTERED)")
    elif int(fs) != int(fork_step):
        probs.append(f"fork_step {fs} != the registered {fork_step}")
    late = [n for n in snaps if (st := snapshot_step(str(n))) is None or st > int(fork_step)]
    if late:
        probs.append(f"seeded snapshot(s) later than fork_step {fork_step} (or unnamed): {late[:5]}")
    early = [d for d in dropped if not isinstance(d, int) or d <= int(fork_step)]
    if early:
        probs.append(f"dropped step(s) not later than fork_step {fork_step}: {early[:5]}")
    out["pass"], out["why"] = not probs, "; ".join(probs)
    return out


def void_gate(spec: Any) -> Optional[Dict[str, Dict[str, Any]]]:
    """The registered VOID gate of ``spec`` (module docstring), or ``None`` when it registers none: ``{label: {run,
    k9b, k9b_file, batch2, pass}}`` plus ``pin`` / ``pool_seed`` (:func:`void_pin_guard`, :func:`void_pool_seed_guard`) for
    the run-dir guards the spec registered (absent otherwise); ``pass`` is the AND of every guard registered. Reads the
    K9(b) records and the run dirs (read-only)."""
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
        row: Dict[str, Any] = {"run": str(run), "k9b": k9b, "k9b_file": str(kf), "batch2": b,
                               "pass": k9b == V.PASS and (b is None or b["verdict"] == V.PASS)}
        if "pin" in v:
            row["pin"] = void_pin_guard(rd, v["pin"])
            row["pass"] = row["pass"] and row["pin"]["pass"]
        if "pool_seed" in v:
            row["pool_seed"] = void_pool_seed_guard(rd, v["pool_seed"]["fork_step"])
            row["pass"] = row["pass"] and row["pool_seed"]["pass"]
        out[lab] = row
    return out


def void_failures(gate: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """``{label: why}`` of every gated run that is not PASS (empty = the gate passes or none is registered)."""
    out: Dict[str, str] = {}
    for lab, g in (gate or {}).items():
        if g.get("pass"):
            continue
        why = f"K9(b) {g['k9b']}" + (f", {g['batch2']['guard']} {g['batch2']['verdict']}" if g.get("batch2") else "")
        for key, name in (("pin", "pin"), ("pool_seed", "pool seed")):
            if g.get(key) is not None and not g[key].get("pass"):
                why += f"; {name}: {g[key].get('why')}"
        out[lab] = why
    return out


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
        _same_engine("x35_r1 import_off (S)", g.get("engine"), EN.bank_engine(p["import_off"], "s"))
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
    if not isinstance(imp, dict) or sorted(set(imp) - {"engines", "sides"}) != sorted(_IMPORT_KEYS) \
            or not isinstance(imp.get("engines", {}), dict) or not isinstance(imp.get("sides", {}), dict) \
            or not isinstance(imp.get("labels"), dict) \
            or not isinstance(imp.get("regimes"), dict) or sorted(imp["regimes"]) != ["anchor", "s"]:
        probs.append(f"import_off: {{{', '.join(_IMPORT_KEYS)}}} with labels {{seed: label}} and regimes {{anchor, s}} "
                     "(and optionally engines {anchor, s}: the bank's engines, main.reads.engine; sides {peer_code, "
                     "policy_mode}: the bank's, main.reads.side_identity)")
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
    from main.reads.models import spec_path
    from utils.hashing import sha256_file

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
    _same_engine("determinism_guard", g.get("engine"), EN.bank_engine(dg["banked"], "anchor", _bank_rows(dg["banked"])))
    _same_sides("determinism_guard", g, dg["banked"], _bank_rows(dg["banked"]))
    checks = []
    for u in g["units"]:
        want = bank.get(u["panel"], {}).get(int(u["unit_seed"]))
        checks.append({"unit": u["unit"], "panel": u["panel"], "unit_seed": int(u["unit_seed"]),
                       "counted": bool(u["counted"]), "n": u.get("n"), "wins": u.get("wins"),
                       "banked": None if want is None else {"n": want["n"], "wins": want["wins"]},
                       **guard_games(u, want)})
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
    _same_engine("import_off (anchor)", _group(cells, "anchor", p["anchor_group"]).get("engine"),
                 EN.bank_engine(imp, "anchor"))
    _same_sides("import_off (anchor)", _group(cells, "anchor", p["anchor_group"]), imp)
    _same_engine("import_off (S)", _group(cells, "s", p["s_group"]).get("engine"), EN.bank_engine(imp, "s"))
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
    s1: Any = doc.get("s1")
    base: Any = p.get("s_baseline")
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


def _same_engine(what: str, here: Optional[str], there: Optional[str]) -> None:
    """A banked control pools only on THIS read's engine (REVIEW F F-M2; :func:`main.reads.engine.require_same`)."""
    try:
        EN.require_same(what, here, there)
    except EN.EngineError as exc:
        raise RuleInputError(str(exc)) from None


def _same_sides(what: str, group: Mapping[str, Any], entry: Mapping[str, Any],
                rows: Sequence[Mapping[str, Any]] = ()) -> None:
    """A banked anchor control pools only with THIS read's Metamon peer code and policy load mode (TECH_DEBT P1 (b);
    :func:`main.reads.side_identity.require_same`; the bank's from :func:`main.reads.side_identity.bank_sides`)."""
    from main.reads import side_identity as SI

    try:
        SI.require_same(what, {k: group.get(k) for k in SI.SIDE_KEYS}, SI.bank_sides(entry, rows))
    except SI.SideError as exc:
        raise RuleInputError(str(exc)) from None


def _bank_rows(entry: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The banked per-game rows of ``entry``'s registered cells (what :func:`main.reads.engine.bank_engine` reads)."""
    from main.reads.models import spec_path

    cells = set(dict(entry["cells"]).values())
    rows = [json.loads(ln) for ln in spec_path(str(entry["path"])).read_text().splitlines() if ln.strip()]
    return [r for r in rows if r.get("read_cell") in cells]


def guard_games(u: Mapping[str, Any], want: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """THE game-for-game comparison of a re-played guard unit ``u`` (an assembled unit row) with its banked unit
    ``want`` (:func:`banked_units`): the 5-tuple (:data:`main.reads.assemble.GAME_KEYS`) AND each game's choices from
    both captures (:func:`main.reads.assemble.compare_choices`; ``measurements/kakuna_determinism_2026-10-27/`` §6).

    ``games_equal`` is True iff the 5-tuples are equal and the choices are not seen to differ. When either side has no
    capture the choices are ``unavailable`` (``choices_why`` says why) and the 5-tuple decides alone, as before."""
    from main.reads.assemble import compare_choices

    tuple_eq = bool(want is not None and u.get("games") is not None
                    and [list(x) for x in u["games"]] == want["games"])
    if not tuple_eq:
        ch: Dict[str, Any] = {"choices": "not compared", "differing": [], "why": "the 5-tuples differ"}
    else:
        assert want is not None
        here = Path(u["cell_dir"]) if u.get("cell_dir") else None
        ch = compare_choices(here, bank_cell(want), want["games"],
                             unavailable="the banked rows name no single source unit")
    return {"games_equal": tuple_eq and ch["choices"] != "differ", "tuple_equal": tuple_eq,
            "choices": ch["choices"], "choices_differing": ch["differing"], "choices_why": ch["why"]}


def bank_cell(want: Mapping[str, Any]) -> Optional[Path]:
    """The banked unit's archived cell dir: its rows' one ``source_unit`` (``<read state>/<unit>``, relative to
    :func:`utils.paths.gen3ai_archive_dir`) as ``<archive>/<read state>/cells/<unit>``; None when the rows name none or
    several."""
    from utils.paths import gen3ai_archive_dir

    src = want.get("sources") or []
    if len(src) != 1 or not src[0]:
        return None
    rel = Path(str(src[0]))
    return gen3ai_archive_dir() / rel.parent / "cells" / rel.name


def banked_units(entry: Mapping[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """``{teamset: {unit_seed: {wins, ties, n, games}}}`` from ANOTHER read's committed per-game rows
    (``entry = {path, sha256, cells: {teamset: read_cell}}``), the file's sha256 checked: the banked control."""
    from main.reads.assemble import GAME_KEYS
    from main.reads.models import spec_path
    from utils.hashing import sha256_file

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
        u = out[ts].setdefault(int(r["unit_seed"]), {"wins": 0, "ties": 0, "n": 0, "games": [], "sources": []})
        u["n"] += 1
        u["wins"] += int(r["result"] == "win")
        u["ties"] += int(r["result"] == "tie")
        u["games"].append([r.get(k) for k in GAME_KEYS])
        if r.get("source_unit") not in u["sources"]:
            u["sources"].append(r.get("source_unit"))
    for ts in out:
        for u in out[ts].values():
            u["games"].sort(key=lambda g: (str(g[0]), int(g[1])))
    return out


def _pm_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    bank = banked_units(p["banked"])
    there = EN.bank_engine(p["banked"], "anchor", _bank_rows(p["banked"]))
    for name, a in p["arms"].items():
        _same_engine(f"prior_mix_ab arm {name}", _group(cells, "anchor", a["group"]).get("engine"), there)
    _same_engine("prior_mix_ab guard", _group(cells, "anchor", p["guard"]["group"]).get("engine"), there)
    bank_rows = _bank_rows(p["banked"])
    for name, gname in [*((f"arm {n}", a["group"]) for n, a in p["arms"].items()), ("guard", p["guard"]["group"])]:
        _same_sides(f"prior_mix_ab {name}", _group(cells, "anchor", gname), p["banked"], bank_rows)
    doc: Dict[str, Any] = {"banked": {ts: {str(s): {k: v for k, v in u.items() if k not in ("games", "sources")}
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
                       **guard_games(u, want), "our_regime": u.get("our_regime")})
    doc["guard"] = {"label": p["guard"]["label"], "units": checks, "regime_ids": gg.get("regime_ids")}
    return doc


def _pm_paired(arm: Mapping[str, Any], bank: Mapping[str, Mapping[Any, Any]], panels: Sequence[str]
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
        cells: Dict[str, Dict[str, Any]] = {}
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
        arm_dec: Dict[str, Any] = {"lambda": r["lambda"], "u_class": cu, "home_class": ch, "home_harmed": harmed,
                                   "verdict": verdict, "d0f": None}
        if float(r["lambda"]) == 1.0 and du is not None and dh is not None:
            # D0f (prior only): the DiD (U − home72) and H3's kill rule (design_diversity_teams_opponents §3.3)
            d = du["d"] - dh["d"]
            se = math.sqrt(((du["hi"] - du["lo"]) / (2 * Z95)) ** 2 + ((dh["hi"] - dh["lo"]) / (2 * Z95)) ** 2)
            held_d = r["cells"][held]["paired"]
            killed = (clears(d + Z95 * se, float(p["did_ub_pp"]) / 100.0, above=False)[0]
                      and held_d is not None and not clears(held_d["lo"], 0.0)[0])
            arm_dec["d0f"] = {"did": d, "lo": d - Z95 * se, "hi": d + Z95 * se, "heldout": held_d,
                              "h3_killed": killed}
        dec["arms"][name] = arm_dec
    order = [p["primary"]] + sorted(n for n in dec["arms"] if n != p["primary"])
    dec["verdict"] = " | ".join(
        f"lambda={dec['arms'][n]['lambda']:g}: {dec['arms'][n]['verdict']}"
        + (f", D0f H3 {'KILLED' if dec['arms'][n]['d0f']['h3_killed'] else 'NOT KILLED'}"
           if dec["arms"][n]["d0f"] else "") for n in order)
    return {"arms": out["arms"], "decision": dec}


# ------------------------------------------------------------------------------------------------ score_bands
_SB_KEYS = ("group", "label", "panel", "kill_at", "support_at", "min_unit_frac", "banked", "guard", "familiar",
            "unfamiliar", "verdict_text")
SB_KILL, SB_SUPPORT, SB_UNDECIDED = "KILL", "SUPPORT", "UNDECIDED"


def _sb_check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    probs = _check_keys(p, ("group", "label", "panel", "kill_at", "support_at", "min_unit_frac", "banked", "guard",
                            "familiar", "unfamiliar"), _SB_KEYS)
    a = anchor.get(p.get("group"))
    if a is None or p.get("label") not in models:
        probs.append("group / label: an anchor group and the model it plays")
    elif list(a.models) != [p["label"]] or list(a.teamsets) != [p.get("panel")]:
        probs.append(f"group {p.get('group')!r} must play exactly {p.get('label')!r} on exactly {p.get('panel')!r}")
    try:
        lo, hi = float(p["support_at"]), float(p["kill_at"])
        if not 0.0 < lo < hi < 1.0:
            probs.append("0 < support_at < kill_at < 1")
    except (KeyError, TypeError, ValueError):
        probs.append("support_at / kill_at: numbers")
    g = p.get("guard") or {}
    if g.get("group") not in anchor or g.get("label") not in models or models[g["label"]].load:
        probs.append("guard: {group, label}: an anchor group playing the UNCHANGED model (no load options)")
    b = p.get("banked") or {}
    if not (isinstance(b, dict) and isinstance(b.get("path"), str) and isinstance(b.get("sha256"), str)
            and isinstance(b.get("cells"), dict)):
        probs.append("banked: {path, sha256, cells: {teamset: read_cell}}")
    else:
        need: Set[Any] = set(p.get("unfamiliar") or []) | {p.get("familiar")}
        if not need <= set(b["cells"]):
            probs.append(f"banked.cells lacks {sorted(need - set(b['cells']))}")
        if g.get("group") in anchor and not set(anchor[g["group"]].teamsets) <= set(b["cells"]):
            probs.append("guard: every team set it plays must be a banked cell (its games are compared with them)")
    return probs


def _sb_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    bank = banked_units(p["banked"])
    g = _group(cells, "anchor", p["group"])
    doc: Dict[str, Any] = {
        "label": p["label"], "panel": p["panel"], "regime_id": g.get("regime_id"),
        "units_planned": len(g["unit_seeds"]),
        "units": [{k: u.get(k) for k in ("unit", "panel", "unit_seed", "counted", "wins", "ties", "n", "our_regime")}
                  for u in g["units"] if u["label"] == p["label"]],
        "banked": {ts: {str(s): {k: v for k, v in u.items() if k not in ("games", "sources")}
                        for s, u in sorted(by.items())}
                   for ts, by in bank.items()},
        "banked_source": {"path": p["banked"]["path"], "sha256": p["banked"]["sha256"]}}
    gg = _group(cells, "anchor", p["guard"]["group"])
    checks = []
    for u in gg["units"]:
        want = bank.get(u["panel"], {}).get(int(u["unit_seed"]))
        checks.append({"unit": u["unit"], "panel": u["panel"], "unit_seed": int(u["unit_seed"]),
                       "counted": bool(u["counted"]), "n": u.get("n"), "wins": u.get("wins"),
                       "banked": None if want is None else {"n": want["n"], "wins": want["wins"]},
                       **guard_games(u, want), "our_regime": u.get("our_regime")})
    doc["guard"] = {"label": p["guard"]["label"], "units": checks, "regime_ids": gg.get("regime_ids")}
    return doc


def _sb_band(w: int, n: int, p: Mapping[str, Any]) -> str:
    """The registered band of the score ``w / n``, compared EXACTLY (fractions: no rounding error can sit on a
    boundary, rule 8): ``>= kill_at`` KILL, ``<= support_at`` SUPPORT, else UNDECIDED."""
    from fractions import Fraction

    x = Fraction(int(w), int(n))
    if x >= Fraction(str(p["kill_at"])):
        return SB_KILL
    if x <= Fraction(str(p["support_at"])):
        return SB_SUPPORT
    return SB_UNDECIDED


def _sb_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    """A score against two registered bands (``hctx_rc72_2026-10-26`` PREREG §5): VOID on a failed determinism guard
    (a guard unit not counted, not greedy, or whose games differ from the banked unit's) or fewer than
    ``min_unit_frac`` units counted; otherwise the band of the pooled score (:func:`_sb_band`). Descriptive, read by no
    verdict: Wilson 95 %, the unit-level dispersion ratio and the interval widened by it, Newcombe against the banked
    familiar cell and against the banked unfamiliar panels' equal-weight mean, and where the score sits between them."""
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
    units = [u for u in doc["units"] if u["counted"]]
    bad = sorted({str(u["our_regime"]) for u in units if u["our_regime"] != "greedy"})
    if bad:
        reasons.append(f"counted units stamped {bad}, not 'greedy'")
    if len(units) < float(p["min_unit_frac"]) * int(doc["units_planned"]) - RULE8_EPS:
        reasons.append(f"{len(units)} / {doc['units_planned']} units counted (< {float(p['min_unit_frac']):.0%})")
    w, n = sum(int(u["wins"]) for u in units), sum(int(u["n"]) for u in units)
    t = sum(int(u.get("ties") or 0) for u in units)
    if not n:
        reasons.append("no game counted")
        return {"score": None, "decision": {"verdict": VOID, "reasons": reasons}}
    sc, lo, hi = wilson(w, n)
    fr = [int(u["wins"]) / int(u["n"]) for u in units]
    disp = None
    if len(fr) >= 2:
        m = sum(fr) / len(fr)
        obs = sum((x - m) ** 2 for x in fr) / (len(fr) - 1)
        exp = sc * (1.0 - sc) / (n / len(fr))
        phi = obs / exp if exp > 0 else None
        disp = {"phi": phi, "units": len(fr),
                "widened95": list(widen(sc, lo, hi, math.sqrt(max(1.0, phi)))) if phi is not None else None}
    bank = doc["banked"]

    def _cell(ts: str) -> Tuple[int, int]:
        return (sum(int(u["wins"]) for u in bank[ts].values()), sum(int(u["n"]) for u in bank[ts].values()))

    fw, fn = _cell(str(p["familiar"]))
    unf = list(p["unfamiliar"])
    ss_u = strat_score({k: _cell(k) for k in unf}, unf)
    on = strat_score({"x": (w, n)}, ["x"])
    d_u = newcombe_strat(on, ss_u)
    d_f = newcombe(w, n, fw, fn)
    fam = fw / fn
    score = {"wins": w, "ties": t, "n": n, "score": sc, "wilson95": [lo, hi], "dispersion": disp,
             "units_counted": len(units), "units_planned": doc["units_planned"],
             "banked_familiar": {"panel": p["familiar"], "wins": fw, "n": fn, "score": fam},
             "banked_unfamiliar": ss_u.as_dict(),
             "minus_familiar": {"d": d_f[0], "lo": d_f[1], "hi": d_f[2]},
             "minus_unfamiliar": {"d": d_u[0], "lo": d_u[1], "hi": d_u[2]},
             "premium_kept": (sc - ss_u.mean) / (fam - ss_u.mean) if fam != ss_u.mean else None}
    if reasons:
        return {"score": score, "decision": {"verdict": VOID, "reasons": reasons}}
    band = _sb_band(w, n, p)
    words = dict({SB_KILL: SB_KILL, SB_SUPPORT: SB_SUPPORT, SB_UNDECIDED: SB_UNDECIDED}, **(p.get("verdict_text") or {}))
    return {"score": score, "decision": {
        "band": band, "kill_at": p["kill_at"], "support_at": p["support_at"],
        "verdict": f"{words[band]}: {p['panel']} {w}/{n} = {sc:.3f} [{lo:.3f}, {hi:.3f}]"}}


# ------------------------------------------------------------------------------------------------ premium_recovery
_PR_KEYS = ("label", "arms", "kill_frac", "support_frac", "min_unit_frac", "bootstrap", "secondary", "verdict_text")
_PR_ARMS = ("rc", "familiar", "unfamiliar")
PR_KILL, PR_SUPPORT, PR_UNDECIDED, PR_INCONCLUSIVE = "KILL", "SUPPORT", "UNDECIDED", "INCONCLUSIVE"


def _pr_check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    probs = _check_keys(p, _PR_KEYS[:-1], _PR_KEYS)
    if p.get("label") not in models:
        probs.append(f"label {p.get('label')!r} is not a model label")
    arms = p.get("arms") or {}
    if not isinstance(arms, dict) or set(arms) != set(_PR_ARMS):
        return probs + [f"arms: {{{', '.join(_PR_ARMS)}: {{panel, groups: [anchor group, ...]}}}}"]
    seen: List[str] = []
    for name, a in arms.items():
        if not (isinstance(a, dict) and isinstance(a.get("panel"), str) and isinstance(a.get("groups"), list)
                and a["groups"]):
            probs.append(f"arms.{name}: {{panel, groups: [...]}}")
            continue
        for g in a["groups"]:
            grp = anchor.get(g)
            if grp is None:
                probs.append(f"arms.{name}: {g!r} is not an anchor group")
            elif list(grp.models) != [p.get("label")] or list(grp.teamsets) != [a["panel"]]:
                probs.append(f"arms.{name}: group {g!r} must play exactly {p.get('label')!r} on {a['panel']!r}")
            seen.append(g)
    if len(seen) != len(set(seen)):
        probs.append("arms: a group belongs to one arm")
    try:
        lo, hi = float(p["support_frac"]), float(p["kill_frac"])
        if not 0.0 < lo < hi < 1.0:
            probs.append("0 < support_frac < kill_frac < 1")
    except (KeyError, TypeError, ValueError):
        probs.append("support_frac / kill_frac: numbers")
    b = p.get("bootstrap") or {}
    if not (isinstance(b, dict) and isinstance(b.get("B"), int) and b["B"] >= 1000 and isinstance(b.get("seed"), int)):
        probs.append("bootstrap: {B: int >= 1000, seed: int}")
    sec = p.get("secondary") or {}
    bk = sec.get("banked") or {}
    if not (isinstance(bk, dict) and isinstance(bk.get("path"), str) and isinstance(bk.get("sha256"), str)
            and isinstance(bk.get("cells"), dict) and sec.get("familiar") in bk.get("cells", {})
            and set(sec.get("unfamiliar") or []) <= set(bk.get("cells", {})) and sec.get("unfamiliar")
            and isinstance(sec.get("kill_at"), (int, float)) and isinstance(sec.get("support_at"), (int, float))):
        probs.append("secondary: {kill_at, support_at, familiar, unfamiliar: [...], banked: {path, sha256, cells}} "
                     "(the cross-engine line)")
    return probs


def _pr_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    sec = p["secondary"]
    bank = banked_units(sec["banked"])
    doc: Dict[str, Any] = {"label": p["label"], "arms": {}, "secondary_bank": {
        ts: {str(s): {k: v for k, v in u.items() if k not in ("games", "sources")} for s, u in sorted(by.items())}
        for ts, by in bank.items()}, "secondary_source": {"path": sec["banked"]["path"],
                                                         "sha256": sec["banked"]["sha256"]}}
    for name, a in p["arms"].items():
        arm: Dict[str, Any] = {"panel": a["panel"], "groups": {}, "units": []}
        for gname in a["groups"]:
            g = _group(cells, "anchor", gname)
            arm["groups"][gname] = {"engine": g.get("engine"), "regime_id": g.get("regime_id"),
                                    "units_planned": len(g["unit_seeds"])}
            for u in g["units"]:
                if u["label"] != p["label"]:
                    continue
                want = bank.get(u["panel"], {}).get(int(u["unit_seed"]))
                row = {k: u.get(k) for k in ("unit", "panel", "unit_seed", "counted", "wins", "ties", "n",
                                             "our_regime")}
                row["group"] = gname
                row["x51_games_equal"] = (None if want is None or u.get("games") is None
                                          else [list(x) for x in u["games"]] == want["games"])
                arm["units"].append(row)
        doc["arms"][name] = arm
    return doc


def _pr_boot(units: Mapping[str, Sequence[Tuple[int, int]]], B: int, seed: int) -> Dict[str, List[float]]:
    """Percentile 95 % intervals of the premium (familiar − unfamiliar), the recovery (rc − unfamiliar), rc −
    familiar and the recovered FRACTION, resampling each arm's 20-game units with replacement, independently
    (``random.Random(seed)``: deterministic)."""
    import random

    rng = random.Random(seed)
    out: Dict[str, List[float]] = {"premium": [], "recovery": [], "rc_minus_familiar": [], "fraction": []}
    for _ in range(B):
        p = {}
        for name, us in units.items():
            pick = [us[rng.randrange(len(us))] for _u in us]
            p[name] = sum(w for w, _ in pick) / sum(n for _, n in pick)
        prem, rec = p["familiar"] - p["unfamiliar"], p["rc"] - p["unfamiliar"]
        out["premium"].append(prem)
        out["recovery"].append(rec)
        out["rc_minus_familiar"].append(p["rc"] - p["familiar"])
        if prem != 0:
            out["fraction"].append(rec / prem)

    def pct(xs: List[float]) -> List[float]:
        xs = sorted(xs)
        return [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]] if xs else [float("nan")] * 2

    return {k: pct(v) for k, v in out.items()}


def _pr_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    """The H-CTX read's AMENDED rule (``hctx_rc72_2026-10-26`` PREREG, Amendment 1): on ONE engine, the share of the
    home premium (familiar − unfamiliar) that rc72 recovers (rc − unfamiliar). VOID if a counted unit is not greedy,
    an arm has fewer than ``min_unit_frac`` of its planned units counted, or the groups' engines are not ONE known
    engine. INCONCLUSIVE if the premium's unit-bootstrap 95 % lower bound is not above 0. Otherwise, by EXACT fractions
    of the pooled scores: recovery ≥ ``kill_frac`` × premium KILL, ≤ ``support_frac`` × premium SUPPORT, else
    UNDECIDED. The SECONDARY line (cross-engine, read by no verdict): rc's absolute band against X51's banked bars."""
    from fractions import Fraction

    reasons: List[str] = []
    engines = sorted({str(g["engine"]) for a in doc["arms"].values() for g in a["groups"].values()
                      if g["engine"] is not None})
    if any(g["engine"] is None for a in doc["arms"].values() for g in a["groups"].values()) or len(engines) != 1:
        reasons.append(f"engine: the groups must share ONE known engine, got {engines or 'none'}")
    pooled: Dict[str, Dict[str, Any]] = {}
    per_unit: Dict[str, List[Tuple[int, int]]] = {}
    for name, a in doc["arms"].items():
        us = [u for u in a["units"] if u["counted"]]
        bad = sorted({str(u["our_regime"]) for u in us if u["our_regime"] != "greedy"})
        if bad:
            reasons.append(f"{name}: counted units stamped {bad}, not 'greedy'")
        planned = sum(int(g["units_planned"]) for g in a["groups"].values())
        if len(us) < float(p["min_unit_frac"]) * planned - RULE8_EPS:
            reasons.append(f"{name}: {len(us)} / {planned} units counted (< {float(p['min_unit_frac']):.0%})")
        w, n = sum(int(u["wins"]) for u in us), sum(int(u["n"]) for u in us)
        sc, lo, hi = wilson(w, n) if n else (None, None, None)
        match = [u["x51_games_equal"] for u in us if u["x51_games_equal"] is not None]
        pooled[name] = {"panel": a["panel"], "wins": w, "ties": sum(int(u.get("ties") or 0) for u in us), "n": n,
                        "units_counted": len(us), "units_planned": planned, "score": sc, "wilson95": [lo, hi],
                        "x51_units_replayed": {"compared": len(match), "equal": sum(bool(x) for x in match)}}
        per_unit[name] = [(int(u["wins"]), int(u["n"])) for u in us]
    out: Dict[str, Any] = {"arms": pooled, "engine": engines[0] if len(engines) == 1 else None}
    if any(not pooled[k]["n"] for k in _PR_ARMS):
        return dict(out, decision={"verdict": VOID, "reasons": reasons + ["an arm has no counted game"]})
    fr = {k: Fraction(pooled[k]["wins"], pooled[k]["n"]) for k in _PR_ARMS}
    prem, rec = fr["familiar"] - fr["unfamiliar"], fr["rc"] - fr["unfamiliar"]
    b = p["bootstrap"]
    ci = _pr_boot(per_unit, int(b["B"]), int(b["seed"]))
    est = {"premium": float(prem), "recovery": float(rec), "rc_minus_familiar": float(fr["rc"] - fr["familiar"]),
           "fraction": float(rec / prem) if prem else None, "boot95": ci, "bootstrap": dict(b)}
    out["estimates"] = est
    # the SECONDARY line: cross-engine, against X51's banked cells
    sec = p["secondary"]
    bank = doc["secondary_bank"]

    def _cell(ts: str) -> Tuple[int, int]:
        return (sum(int(u["wins"]) for u in bank[ts].values()), sum(int(u["n"]) for u in bank[ts].values()))

    x_rc = fr["rc"]
    band = (PR_KILL if x_rc >= Fraction(str(sec["kill_at"])) else PR_SUPPORT
            if x_rc <= Fraction(str(sec["support_at"])) else PR_UNDECIDED)
    fw, fn = _cell(str(sec["familiar"]))
    unf = list(sec["unfamiliar"])
    out["secondary_cross_engine"] = {
        "band": band, "kill_at": sec["kill_at"], "support_at": sec["support_at"],
        "x51_familiar": fw / fn, "x51_unfamiliar_mean": strat_score({k: _cell(k) for k in unf}, unf).mean,
        "note": "cross-engine: X51 played on engine E0; read by no verdict"}
    words = dict({k: k for k in (PR_KILL, PR_SUPPORT, PR_UNDECIDED, PR_INCONCLUSIVE)}, **(p.get("verdict_text") or {}))
    tail = (f"{pooled['rc']['panel']} {float(fr['rc']):.3f}, {pooled['familiar']['panel']} "
            f"{float(fr['familiar']):.3f}, {pooled['unfamiliar']['panel']} {float(fr['unfamiliar']):.3f}; premium "
            f"{float(prem):+.3f} [{ci['premium'][0]:+.3f}, {ci['premium'][1]:+.3f}]")
    if reasons:
        return dict(out, decision={"verdict": VOID, "reasons": reasons})
    if not clears(ci["premium"][0], 0.0)[0]:
        return dict(out, decision={"band": PR_INCONCLUSIVE, "verdict": f"{words[PR_INCONCLUSIVE]}: the premium is "
                                   f"not resolved on this engine ({tail})"})
    kill, sup = Fraction(str(p["kill_frac"])), Fraction(str(p["support_frac"]))
    verdict = PR_KILL if rec >= kill * prem else PR_SUPPORT if rec <= sup * prem else PR_UNDECIDED
    frac = f"{float(rec / prem):.2f} [{ci['fraction'][0]:.2f}, {ci['fraction'][1]:.2f}]"
    return dict(out, decision={"band": verdict, "kill_frac": p["kill_frac"], "support_frac": p["support_frac"],
                               "verdict": f"{words[verdict]}: rc72 recovers {frac} of the home premium ({tail}); "
                                          f"secondary, cross-engine: {words[band]}"})


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
    "score_bands": Rule("score_bands", "one model's score on one panel against two registered bands; a determinism "
                        "guard against banked rows (the H-CTX rc72 read)", _sb_check, _sb_inputs, _sb_decide, False),
    "premium_recovery": Rule("premium_recovery", "the share of the home premium a panel recovers, on ONE engine, "
                             "unit-bootstrap (the H-CTX read, amended)", _pr_check, _pr_inputs, _pr_decide, False),
}

from main.reads import rules_x59s as _X59S  # noqa: E402  (it imports this module lazily: no cycle at import)

RULES.update({
    "x59s_floor": Rule("x59s_floor", _X59S.RULE_DOCS["x59s_floor"], _X59S._floor_check, _X59S._floor_inputs,
                       _X59S._floor_decide, False),
    "x59s_gate": Rule("x59s_gate", _X59S.RULE_DOCS["x59s_gate"], _X59S._gate_check, _X59S._gate_inputs,
                      _X59S._gate_decide, False),
    "x59s": Rule("x59s", _X59S.RULE_DOCS["x59s"], _X59S._full_check, _X59S._full_inputs, _X59S._full_decide, False),
    "x59s_restriction": Rule("x59s_restriction", _X59S.RULE_DOCS["x59s_restriction"], _X59S._restr_check,
                             _X59S._restr_inputs, _X59S._restr_decide, False),
})

from main.reads import rules_search_headroom as _SH  # noqa: E402  (it imports this module lazily: no cycle at import)

RULES["search_headroom"] = Rule("search_headroom", _SH.RULE_DOCS["search_headroom"], _SH.check, _SH.inputs, _SH.decide,
                                False)

from main.reads import rules_n4_screen as _N4  # noqa: E402  (it imports this module lazily: no cycle at import)

RULES["n4_screen"] = Rule("n4_screen", _N4.RULE_DOCS["n4_screen"], _N4.check, _N4.inputs, _N4.decide, True,
                          _N4.reported)

from main.reads import rules_deviation_sign as _DS  # noqa: E402  (it imports this module lazily: no cycle at import)

RULES["deviation_sign"] = Rule("deviation_sign", _DS.RULE_DOCS["deviation_sign"], _DS.check, _DS.inputs, _DS.decide,
                               False)


def inputs(rule: str, cells: Mapping[str, Any], params: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    return RULES[rule].inputs(cells, params, spec)


def decide(rule: str, doc: Mapping[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
    return RULES[rule].decide(doc, params)
