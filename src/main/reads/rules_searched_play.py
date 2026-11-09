r"""The registered rule ``searched_play``: a SEARCHED label plays whole games live against its RAW TWIN, deal by deal, on
the U panels only (the S′ live-play read, ``designs/endstate/design_sprime_live_play.md``).

ONE ``per_game`` anchor group plays the searched label (``load.search``, e.g. ``r0s4_rolls@10``) and its raw twin
(``main.reads.spec.search_twins``: the same pinned bytes, no search) on exactly the ``u_panels``. A DEAL (panel, unit
seed, game) is a PAIR iff both arms' one-game units COUNTED (``main.anchors.results.unit_counts``) and it is not a peer
drop (below). A game scores y = win 1, tie ½, loss 0; a pair's D = y_searched − y_raw.

THE ESTIMAND. A UNIT is one (panel, unit seed); its statistic is the mean of its deals' D, exact fractions. G_panel =
the mean of the panel's unit statistics; G_U = the equal-weight mean of the ``u_panels``' G.

UNPAIRED DEALS. A deal is FINISHED when each arm's unit counted or spent its attempts (``main.reads.attempts``); a
finished deal that is not a pair is UNPAIRED, of two kinds:

* PRE-TREATMENT: a PEER DROP (below) only. Excluded, never imputed.
* POST-TREATMENT: every other unpaired deal: the searched side missing (the raw unit counted), the raw side missing,
  or neither counted. It may depend on the treatment (a searched game that runs long), so it is IMPUTED in the robust
  classes below.

THE ROBUST CLASSES. Three estimates of G_U, each with its own bootstrap: PAIRED (the pairs only, REPORTED); GO-IMPUTED
(every post-treatment unpaired deal at its worst case for GO: searched missing D = −y_raw, raw missing D = y_s − 1,
neither D = −1); NULL-IMPUTED (the opposite: searched missing D = 1 − y_raw, raw missing D = y_s, neither D = +1). An
imputed deal enters its unit's statistic like a pair. Every bar is compared as an EXACT fraction
(``search_headroom.cmp``); a bound within ``rule8_eps`` of its bar is ``near``, never a crossing (root rule 8), and the
decision is flagged ``boundary``:

* GO iff LB(G_U, GO-imputed) > 0, spelled by the registered preset's dice: **GO-HONEST** for a leak-honest preset
  (:data:`HONEST_DICE`: ``r0s4_honest``, ``r0s4_honest_inc``: the search reads the searched side's information set
  alone), **GO-LEAKY** for every other (its world builder replays the true seed);
* NULL iff not GO and UB(G_U, NULL-imputed) < ``null_margin_pp`` / 100;
* STOP (undecided) otherwise. ``go_below_margin`` flags a GO whose NULL-imputed UB is also below the margin.

GOAL B (decided as a flag, ``goal_b``): the searched arm's pooled-U score (``strat_score``, a tie a non-win) with
every POST-treatment unpaired deal's searched game at its worst case (a missing searched game a LOSS; a counted one at
its result), peer drops excluded. ``reached`` iff the class is GO and that score's LB > 0.50 (exact fractions, rule 8:
within ``rule8_eps`` is ``near``, never reached); its label is GOAL-B-REACHED for GO-HONEST and GOAL-B-REACHED (leaky)
for GO-LEAKY.

THE EXCLUSION FLAG (``max_excluded_frac``, optional, ``{panel: fraction}`` over exactly the ``u_panels``): per panel,
the share of the searched arm's search-path decisions (searched + excluded, every COUNTED searched unit's search
record) that were EXCLUSIONS. ``excess_exclusions`` is set iff any panel's share exceeds its bound or sits within
``rule8_eps`` of it (a narrowing flag: near counts as set). It never VOIDs and never changes the class; the PREREG
registers what it narrows.

THE TIMER FLAG (``timer``, optional: ``{reserve_s, overhead_s, max_cut_frac}``; the cost read's DRAFT §1 / §7 R3): every
COUNTED searched unit's decision log replayed through Showdown's turn timer (:mod:`main.reads.timer_model`, every
decision of ours in order, our response the decision's wall + ``overhead_s``) with the BANK-AWARE deadline
min(bank, 150) − ``reserve_s`` − ``overhead_s``, the cut step running on (the design's safety net; for a preset that
carries the live net the block must be that net, so the replay reproduces the cuts it played). ``timer_safe`` is set
iff, pooled over U, the replay forfeits 0 games AND cuts ≤ ``max_cut_frac`` of the searched decisions (every decision
whose search produced a pick; exact fractions; within ``rule8_eps`` of the bound is ``near`` and NOT safe). A decided
flag: it never VOIDs and never changes the class; its label TIMER-SAFE (or NOT TIMER-SAFE) is in the verdict. The
safety net's RECORDED cuts (``why: "deadline"``) are counted beside the exclusions (``exclusions.by_panel.net_cut``)
and in the timer report; the prototype's policy table is REPORTED.

THE BOOTSTRAP (``search_headroom``'s scheme on the U panels), run once per estimate with a FRESH
``numpy.random.default_rng(seed)``: for each panel in ``u_panels`` order, its unit statistics in unit-seed order as a
float vector ``x`` of length n, ``idx = rng.integers(0, n, size=(B, n))`` and the panel's replicates
``x[idx].mean(axis=1)``; replicate b of G_U is the mean of the panels' replicate b; each interval
``numpy.percentile(reps, [2.5, 97.5])``.

PEER DROPS (the dlive read's FINDING DL-7, registered before any game here): each G-PREFIX mismatch (``assemble``'s
``g_prefix.mismatches``: the pair's protocol differs before the search's first turn) is classified by
:func:`main.reads.peer_divergence.classify` on the pair's two captures (raw first) with branch turn = the search's
first turn. ``peer_first`` (Kakuna chose differently first on an identical view, our commands on identical views all
equal, before that turn) makes the deal a PEER DROP. Any other class, or an unreadable capture, is a VOID reason. This
rule OWNS its G-PREFIX mismatches (``Rule.owns_g_prefix``): ``assemble`` does not mark the group inconclusive for them,
so a peer drop does not block the record and every other mismatch VOIDs it. (The lane still HOLDS on a mismatch for a
look; the operator releases it.)

THE LIVE IDENTITY GUARD (``identity``): ``python -m main.value_search.live_identity rescore`` re-runs S′ offline on the
hashed sample of the searched label's live decisions (``cells``) and writes ``rescore_json``. The rule reads that JSON
(its sha256 recorded), checks it was run on these cells, these panels, the registered engine and the raw label's
bytes, re-derives the registered-order first N of the sampled records from the cells and requires the JSON's re-scored
states to be exactly those (N = the registered ``max_rescored``; the JSON's own ``max`` must equal it), and needs its
PASS with >= ``min_rescored`` states. For a leak-honest preset the JSON must have compared ``search_key`` and the
``honest`` build record too; for the INCREMENTAL one (``r0s4_honest_inc``) the spec registers ``identity.carry``
(``rebuild``: each state's carry rebuilt from I; ``seeded``: the carry the live record kept) and the JSON must have
re-scored with that mode (``live_identity``'s ``--carry``). A PINNED leak-honest read's JSON must record a clean
re-scoring checkout whose play path is the pin's (``inputs.code``).

VOID (named reasons, before any estimate): G-PREFIX not run on the group; a G-PREFIX mismatch not classed a peer drop;
G-PREFIX checked != pairs + peer drops; counted units on two engines, an unknown engine, or another than ``engine``;
any counted unit, or a resolved model, on bytes other than the raw label's pinned sha256; a counted searched unit not
stamped / played as ``expected_search``'s preset; a counted raw unit carrying a search stamp or a regime other than
greedy; a counted unit whose ``games.jsonl`` disagrees with its summary; a panel with < 2 units holding a pair; any
UNFINISHED deal (an arm's unit neither counted nor out of attempts: the read is incomplete, never silently short); in any U
panel, post-treatment unpaired deals above ``max_post_unpaired_frac`` of the panel's planned deals, or pre-treatment
above ``max_pre_unpaired_frac``; the live identity guard missing, inconsistent, short of ``min_rescored`` or FAIL.

REPORTED, never decided: the three estimates with their intervals, G per panel (each estimate's own replicates); each
arm's pooled U score over the paired games (``strat_score``, a tie a non-win) and the searched arm's LB against 0.50;
the discordance share (b + c only, never b − c); the counts per panel (pairs, pre- and post-treatment unpaired by
cause, unfinished); the deviation rate, exclusions and search CPU-s per game (each searched unit's search record); the
identity guard's summary; the DISTILLATION FIELDS (``distill``, X65-F2): per panel, the counted searched units'
searched-path records and how many lack ``q``, ``log_pi``, ``root_v``, ``frame_sha256`` or ``log_sha256`` (and, for a
leak-honest preset, ``search_key``), and the counted searched units with no capture file.
"""
from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "gen3_read_searched_play_v1"
KEYS = ("group", "searched", "raw", "u_panels", "expected_search", "engine", "null_margin_pp", "bootstrap",
        "rule8_eps", "max_post_unpaired_frac", "max_pre_unpaired_frac", "identity", "state")
#: optional params (a spec registered without them reads as before)
OPTIONAL = ("max_excluded_frac", "timer")
#: the TIMER block's keys (optional; the cost read's DRAFT §7 R3): the bank-aware replay's reserve and overhead and the
#: decided flag's bound on the cut share
TIMER_KEYS = ("reserve_s", "overhead_s", "max_cut_frac")
TIMER_SAFE = "TIMER-SAFE"
IDENTITY_KEYS = ("cells", "rescore_json", "min_rescored", "max_rescored")
#: the identity block's key an INCREMENTAL preset (``r0s4_honest_inc``) must register, and only it: how its re-score
#: gets each state's carry (``live_identity.CARRY_MODES``: ``rebuild`` or ``seeded``)
IDENTITY_CARRY = "carry"
RAW_REGIME = "greedy"
GO_LEAKY, GO_HONEST, NULL, STOP, VOID = "GO-LEAKY", "GO-HONEST", "NULL", "STOP", "VOID"
#: the preset dice of a LEAK-HONEST search (``main.anchors.search_presets``: ``fresh`` = ``r0s4_honest``, ``fresh_inc`` =
#: ``r0s4_honest_inc``; the live searcher's own ``main.anchors.search_side.HONEST_DICE``, pinned equal by a test)
HONEST_DICE: Tuple[str, ...] = ("fresh", "fresh_inc")
#: the dice of the INCREMENTAL leak-honest builder (its re-score carries a carry mode)
INC_DICE = "fresh_inc"
#: the fields every searched-path record keeps for distillation (X65-F2; ``search_key`` too for a leak-honest preset)
DISTILL_SEARCHED = ("q", "log_pi", "root_v", "frame_sha256", "log_sha256")
DISTILL_EXCLUDED = ("frame_sha256", "log_sha256")
PEER_DROP_CLASS = "peer_first"
COUNTED, EXHAUSTED, PENDING = "counted", "exhausted", "pending"
#: an unpaired deal's cause: PRE-treatment (a peer drop) or POST-treatment (the rest)
PRE_CAUSES, POST_CAUSES = ("peer_drop",), ("searched_missing", "raw_missing", "neither")
ESTIMATES = ("paired", "go_imputed", "null_imputed")
RULE_DOCS = {"searched_play": "a searched label plays whole games live against its raw twin, paired by deal on U in "
                              "ONE per_game group; G_U by a within-panel unit bootstrap; GO (GO-HONEST for a "
                              "leak-honest preset, else GO-LEAKY) iff the GO-imputed LB > 0, NULL iff the NULL-imputed "
                              "UB < the margin, else STOP; goal b and excess exclusions as flags; peer drops by "
                              "AM-DL-PEER's classifier; VOID on G-PREFIX, engines, bytes, the search config, unpaired "
                              "deals or the live identity guard"}
_UNIT_KEYS = ("unit", "label", "panel", "unit_seed", "game", "counted", "wins", "ties", "n", "our_regime", "search",
              "search_played", "model_sha256", "engine", "games")


def _err(msg: str) -> Exception:
    from main.reads.rules import RuleInputError

    return RuleInputError(msg)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _abs(x: Any) -> bool:
    return isinstance(x, str) and bool(x) and os.path.isabs(x)


def _incremental(expected: Any) -> Optional[bool]:
    """Whether a registered search spelling names the incremental builder (None when it does not parse)."""
    from main.anchors.search_presets import SearchSpecError, parse

    try:
        return parse(expected)[0].dice == INC_DICE
    except SearchSpecError:
        return None


def go_class(preset: Mapping[str, Any]) -> str:
    """The GO class's name for a registered preset (its ``as_dict()``): GO-HONEST for leak-honest dice."""
    return GO_HONEST if (preset or {}).get("dice") in HONEST_DICE else GO_LEAKY


def cells_dir(state: str, group: str, label: str) -> str:
    """The searched label's cells dir of a read state (``main.reads.layout``)."""
    from main.reads import layout as LY

    return os.path.normpath(str(LY.anchor_state(Path(state), group, label) / "cells"))


# ------------------------------------------------------------------------------------------------ check (spec load)
def check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    from main.anchors.search_presets import SearchSpecError, canonical
    from main.reads.rules import _check_keys
    from main.reads.spec import SpecError, search_twins

    probs = _check_keys(p, KEYS, KEYS + OPTIONAL)
    g = anchor.get(p.get("group"))
    if g is None:
        return probs + [f"group {p.get('group')!r} is not an anchor group"]
    if not g.per_game:
        probs.append(f"group {g.name!r} is not per_game (a deal is one game of each arm)")
    srch: Any = p.get("searched")
    raw: Any = p.get("raw")
    if srch == raw or set(g.models) != {srch, raw}:
        probs.append(f"group {g.name!r} must play exactly the searched label and its raw twin, plays {list(g.models)}")
    else:
        try:
            twin = search_twins(models, g.models, f"anchor.{g.name}").get(srch)
        except SpecError as exc:
            twin = None
            probs.append(str(exc))
        if twin != raw:
            probs.append(f"raw {raw!r} is not searched {srch!r}'s raw twin (same ref, no search)")
        try:
            want = canonical(p.get("expected_search"))
            if want != p.get("expected_search"):
                probs.append(f"expected_search: spell it {want!r}")
            if models[srch].load.get("search") != want:
                probs.append(f"searched {srch!r} loads search {models[srch].load.get('search')!r}, not "
                             f"expected_search {p.get('expected_search')!r}")
        except SearchSpecError as exc:
            probs.append(f"expected_search: {exc}")
        ref = models[raw].ref
        if ref.kind != "pinned" or not ref.sha256:
            probs.append(f"raw {raw!r} must be a pinned ref with its sha256 (a raw arm on other bytes is VOID)")
    u = p.get("u_panels")
    if not (isinstance(u, list) and u and all(isinstance(x, str) for x in u) and len(set(u)) == len(u)):
        probs.append("u_panels: a list of distinct team sets")
    elif set(u) != set(g.teamsets) or len(u) != len(g.teamsets):
        probs.append(f"u_panels {u} must be exactly group {g.name!r}'s team sets {list(g.teamsets)} (no home panel)")
    if "max_excluded_frac" in p:
        mx = p["max_excluded_frac"]
        if not (isinstance(mx, dict) and isinstance(u, list) and set(mx) == set(u)
                and all(_num(v) and 0 < v < 1 for v in mx.values())):
            probs.append("max_excluded_frac: {panel: a fraction in (0, 1)} over exactly the u_panels")
    if "timer" in p:
        probs += _check_timer(p["timer"], p.get("expected_search"))
    if not (isinstance(p.get("engine"), str) and p["engine"]):
        probs.append("engine: the registered engine stamp (port=..;env=..)")
    if not (_num(p.get("null_margin_pp")) and p["null_margin_pp"] > 0):
        probs.append("null_margin_pp: a number > 0 (pp)")
    bs = p.get("bootstrap")
    if not (isinstance(bs, dict) and set(bs) == {"B", "seed"} and _int(bs.get("B")) and bs["B"] >= 1000
            and _int(bs.get("seed"))):
        probs.append("bootstrap: {B: int >= 1000, seed: int}")
    eps: Any = p.get("rule8_eps")
    if not (_num(eps) and 0 <= eps < 0.01):
        probs.append("rule8_eps: a number in [0, 0.01) (a bound this close to a bar is never a crossing)")
    for k in ("max_post_unpaired_frac", "max_pre_unpaired_frac"):
        if not (_num(p.get(k)) and 0 <= p[k] <= 0.2):
            probs.append(f"{k}: a number in [0, 0.2]")
    if not _abs(p.get("state")):
        probs.append("state: the read's state dir, an absolute path")
    idn = p.get("identity")
    inc = _incremental(p.get("expected_search"))
    want_keys = set(IDENTITY_KEYS) | ({IDENTITY_CARRY} if inc else set())
    if not (isinstance(idn, dict) and set(idn) == want_keys):
        probs.append(f"identity: {{{', '.join(sorted(want_keys))}}}"
                     + (f" ({IDENTITY_CARRY!r} for an incremental preset only)" if inc is False else
                        f" ({IDENTITY_CARRY!r}: how the re-score gets an incremental record's carry)" if inc else ""))
    else:
        if inc:
            from main.value_search.live_identity import CARRY_MODES

            if idn[IDENTITY_CARRY] not in CARRY_MODES:
                probs.append(f"identity.{IDENTITY_CARRY}: one of {list(CARRY_MODES)}")
        probs += [f"identity.{k}: an absolute path" for k in ("cells", "rescore_json") if not _abs(idn[k])]
        if not (_int(idn["min_rescored"]) and idn["min_rescored"] >= 1):
            probs.append("identity.min_rescored: an int >= 1")
        if not (_int(idn["max_rescored"]) and _int(idn["min_rescored"]) and idn["max_rescored"] >= idn["min_rescored"]):
            probs.append("identity.max_rescored: an int >= min_rescored (the registered first N of the sample)")
        if _abs(p.get("state")) and _abs(idn["cells"]) and isinstance(srch, str) and \
                os.path.normpath(idn["cells"]) != cells_dir(p["state"], g.name, srch):
            probs.append(f"identity.cells {idn['cells']} is not the searched label's cells dir "
                         f"{cells_dir(p['state'], g.name, srch)}")
    return probs


def _check_timer(t: Any, expected: Any) -> List[str]:
    """The ``timer`` block: exactly :data:`TIMER_KEYS`, reserve and overhead >= 0, the bound in (0, 1); for a preset
    carrying the live safety net, the net the units PLAYED (its reserve, ``timer_model.OVERHEAD_S``), so the replay
    reproduces the net's own cuts."""
    from main.anchors.search_presets import SearchSpecError, parse
    from main.reads import timer_model as TM

    if not (isinstance(t, dict) and set(t) == set(TIMER_KEYS)):
        return [f"timer: {{{', '.join(TIMER_KEYS)}}}"]
    probs = [f"timer.{k}: a number >= 0 (s)" for k in ("reserve_s", "overhead_s") if not (_num(t[k]) and t[k] >= 0)]
    if not (_num(t["max_cut_frac"]) and 0 < t["max_cut_frac"] < 1):
        probs.append("timer.max_cut_frac: a fraction in (0, 1)")
    try:
        net = parse(expected)[0].net_reserve_s
    except SearchSpecError:
        net = None
    if net is not None and not probs and (Fraction(str(t["reserve_s"])) != Fraction(str(net))
                                          or Fraction(str(t["overhead_s"])) != Fraction(str(TM.OVERHEAD_S))):
        probs.append(f"timer: the preset's safety net played reserve {net} s, overhead {TM.OVERHEAD_S} s; the replay "
                     f"must be that net (reserve_s {t['reserve_s']}, overhead_s {t['overhead_s']})")
    return probs


# ------------------------------------------------------------------------------------------------ inputs (I/O here)
def planned_deals(state: Path, group: str, searched: str, raw: str, units: Sequence[Mapping[str, Any]],
                  exhausted: Sequence[str]) -> List[Dict[str, Any]]:
    """Every PLANNED deal of the group (the two labels' ``plan.json``), each arm's unit and its status: ``counted``,
    ``exhausted`` (its retries spent: ``main.reads.attempts``) or ``pending``."""
    from main.reads import layout as LY
    from main.x35 import kakuna as KA

    counted = {str(u["unit"]) for u in units if u["counted"]}
    spent = set(exhausted)
    by: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for side, lab in (("searched", searched), ("raw", raw)):
        st = LY.anchor_state(state, group, lab)
        if not (st / "plan.json").is_file():
            raise _err(f"{st}: no plan.json (the read is planned before it is decided)")
        for u in KA.load_plan(st):
            if u.game is None:
                raise _err(f"unit {u.name}: not a per-game unit")
            d = by.setdefault((u.teamset, int(u.unit_seed), int(u.game)), {})
            if side in d:
                raise _err(f"deal {(u.teamset, u.unit_seed, u.game)} planned twice for {lab!r}")
            d[side] = {"unit": u.name, "status": COUNTED if u.name in counted else EXHAUSTED if u.name in spent
                       else PENDING}
    out = []
    for (ts, seed, game), d in sorted(by.items()):
        if set(d) != {"searched", "raw"}:
            raise _err(f"deal {ts}:{seed}:{game} is planned for one arm only")
        out.append({"deal": f"{ts}:{seed}:{game}", "panel": ts, "unit_seed": seed, "game": game, **d})
    return out


def classify_prefix(state: Path, group: str, searched: str, raw: str, mismatches: Sequence[Mapping[str, Any]],
                    from_turn: int) -> List[Dict[str, Any]]:
    """Each G-PREFIX mismatch classified on its pair's captures (raw first), branch turn ``from_turn``."""
    from main.reads import layout as LY
    from main.reads.peer_divergence import classify_pair

    out = []
    for m in mismatches:
        row: Dict[str, Any] = {k: m.get(k) for k in ("deal", "searched", "raw", "why")}
        row.update(classify_pair(LY.anchor_state(state, group, raw) / "cells" / str(m["raw"]),
                                 LY.anchor_state(state, group, searched) / "cells" / str(m["searched"]), from_turn))
        out.append(row)
    return out


def read_identity(idn: Mapping[str, Any], *, u_panels: Sequence[str], engine: str, model_sha256: str,
                  honest: bool = False, pin: Optional[str] = None) -> Dict[str, Any]:
    """The live identity guard's summary from its ``rescore`` JSON: its own verdict, the rule's consistency checks
    (``problems``: these cells, panels, engine and bytes; exactly the registered-order first N of the cells' sampled
    records; for a leak-honest preset the honest fields compared; for an incremental one the registered carry mode;
    for a PINNED leak-honest read, a clean re-scoring checkout whose play path is the pin's), the JSON's sha256 and
    counts."""
    from main.value_search import live_identity as LI
    from utils.hashing import sha256_file

    p = Path(idn["rescore_json"])
    if not p.is_file():
        return {"verdict": None, "problems": [f"{p}: missing (live_identity rescore writes it before the record)"]}
    try:
        doc = json.loads(p.read_text())
    except ValueError as exc:
        return {"verdict": None, "problems": [f"{p}: unreadable ({exc})"]}
    probs: List[str] = []
    inp = doc.get("inputs") or {}
    if doc.get("schema") != LI.RESCORE_SCHEMA:
        probs.append(f"schema {doc.get('schema')!r}, not {LI.RESCORE_SCHEMA!r}")
    want = {"cells": os.path.normpath(str(idn["cells"])), "panels": list(u_panels), "engine": engine,
            "model_sha256": model_sha256}
    have = {"cells": os.path.normpath(str(inp.get("cells"))), "panels": inp.get("panels"),
            "engine": inp.get("engine"), "model_sha256": inp.get("model_sha256")}
    probs += [f"it re-scored with {k} {have[k]!r}, not {want[k]!r}" for k in want if have[k] != want[k]]
    if inp.get("max") != idn["max_rescored"]:
        probs.append(f"it re-scored the first {inp.get('max')!r} sampled records, not the registered max_rescored "
                     f"{idn['max_rescored']}")
    if honest:
        miss = [f for f in LI.RESCORE_FIELDS + LI.HONEST_FIELDS if f not in (inp.get("fields") or ())]
        if miss:
            probs.append(f"a leak-honest preset's re-score must compare {list(LI.HONEST_FIELDS)} too; it did not "
                         f"compare {miss}")
    if inp.get("carry") != idn.get(IDENTITY_CARRY):
        probs.append(f"it re-scored with carry {inp.get('carry')!r}, not the registered {idn.get(IDENTITY_CARRY)!r}")
    if honest and pin:
        probs += _rescore_at_pin(inp.get("code"), pin)
    try:
        order = LI.registered_order(Path(idn["cells"]), list(u_panels), int(inp.get("max") or 0))
        if order != doc.get("rescored"):
            probs.append(f"its {len(doc.get('rescored') or [])} re-scored state(s) are not the cells' registered-order "
                         f"first {len(order)} sampled record(s)")
    except (LI.IdentityInputError, ValueError, TypeError) as exc:
        probs.append(f"the cells' sampled records: {exc}")
    n = int(doc.get("n_rescored") or 0)
    if n < int(idn["min_rescored"]):
        probs.append(f"{n} state(s) re-scored, fewer than min_rescored {idn['min_rescored']}")
    return {"verdict": doc.get("verdict"), "reason": doc.get("reason"), "problems": probs,
            "rescore_json": {"path": str(p), "sha256": sha256_file(p)}, "n_sampled": doc.get("n_sampled"),
            "n_rescored": n, "by_panel": doc.get("by_panel"), "mismatches": len(doc.get("mismatches") or ()),
            "cpu_s": doc.get("cpu_s")}


#: the re-score's own code, OFF the play path: the comparator and the root guards it runs (byte-equal to the pin's)
RESCORE_CODE = ("src/main/value_search/live_identity.py", "src/main/value_search/sign_gate.py")


def _rescore_at_pin(code: Any, pin: str) -> List[str]:
    """The re-score of a pinned leak-honest read ran the pin's search AND the pin's re-score: its recorded checkout is
    clean, its play path (``main.reads.playpath``) is the pin's, and :data:`RESCORE_CODE` (off the play path) is
    byte-equal to the pin's; the pin itself always passes."""
    from main.reads import playpath as PP

    if not (isinstance(code, Mapping) and code.get("commit")):
        return ["a pinned leak-honest read's re-score must record its checkout (inputs.code); it did not"]
    c = str(code["commit"])
    if code.get("dirty"):
        return [f"it re-scored from a DIRTY checkout of {c[:12]}"]
    try:
        diff = PP.compare(str(pin), c)[0]
        repo = PP.default_repo()
        blobs = [f for f in RESCORE_CODE if PP._git(repo, "rev-parse", f"{pin}:{f}") != PP._git(repo, "rev-parse", f"{c}:{f}")]
    except PP.PlayPathError as exc:
        return [f"its checkout {c[:12]} against the pin {pin}: {exc}"]
    if diff:
        return [f"it re-scored at {c[:12]}, whose play path differs from the pin {pin}'s: {diff[:4]}"]
    return [f"it re-scored at {c[:12]}, whose re-score code differs from the pin {pin}'s: {blobs}"] if blobs else []


def inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    from main.anchors.search_presets import parse
    from main.reads.rules import _group

    g = _group(cells, "anchor", p["group"])
    preset, from_turn = parse(p["expected_search"])
    labels = (p["searched"], p["raw"])
    models = cells.get("models") or {}
    units = [{k: u.get(k) for k in _UNIT_KEYS} for u in g["units"] if u["label"] in labels]
    state = Path(p["state"])
    gp = g.get("g_prefix")
    raw_sha = spec.models[p["raw"]].ref.sha256
    return {
        "schema": SCHEMA, "searched": p["searched"], "raw": p["raw"], "u_panels": list(p["u_panels"]),
        "expected_search": p["expected_search"], "expected_regime": f"{RAW_REGIME}+search={p['expected_search']}",
        "expected_preset": preset.as_dict(), "from_turn": from_turn, "raw_sha256": raw_sha,
        "resolved_sha256": {lab: (models.get(lab) or {}).get("sha256") for lab in labels},
        "engine_registered": p["engine"],
        "group": {"name": p["group"], "engine": g.get("engine"), "regime_id": g.get("regime_id"),
                  "teamsets": g.get("teamsets"), "unit_seeds": g.get("unit_seeds"), "unplayed": g.get("unplayed"),
                  "exhausted_units": list(g.get("exhausted_units") or []),
                  "inconclusive": list(g.get("inconclusive") or []), "g_prefix": gp},
        "units": units,
        "deals": planned_deals(state, p["group"], p["searched"], p["raw"], units, g.get("exhausted_units") or ()),
        "prefix_classes": (classify_prefix(state, p["group"], p["searched"], p["raw"], gp.get("mismatches") or (),
                                           from_turn) if gp else None),
        "identity": read_identity(p["identity"], u_panels=p["u_panels"], engine=p["engine"], model_sha256=raw_sha,
                                  honest=preset.dice in HONEST_DICE, pin=getattr(spec, "pin", None)),
        "distill": distill_report(state, p["group"], p["searched"], units, honest=preset.dice in HONEST_DICE),
        **({"timer": timer_report(state, p["group"], p["searched"], units, p["timer"])} if "timer" in p else {})}


def timer_report(state: Path, group: str, searched: str, units: Sequence[Mapping[str, Any]], t: Mapping[str, Any]
                 ) -> Dict[str, Any]:
    """The TIMER REPLAY over every COUNTED searched unit's decision log (outcome-blind: never a result): per panel and
    pooled U, the registered bank-aware replay's exact counts (``timer_model.tally``: the reserve and overhead of
    ``t``, the cut step running on, the design §3's safety net), the safety-net cuts the units RECORDED, and the
    REPORTED policy table (``timer_model.replay``)."""
    from main.reads import layout as LY
    from main.reads import timer_model as TM

    cells = LY.anchor_state(state, group, searched) / "cells"
    games: List[Tuple[str, List[Dict[str, Any]]]] = []
    rec: Dict[str, Dict[str, int]] = {}
    for u in units:
        if u["label"] != searched or not u["counted"]:
            continue
        log = cells / str(u["unit"]) / "search_decisions.jsonl"
        if not log.is_file():
            raise _err(f"unit {u['unit']}: counted, but no decision log {log}")
        recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        games.append((str(u["panel"]), TM.rows_of(recs)))
        for k in (str(u["panel"]), "U"):
            a = rec.setdefault(k, {"net_cut": 0, "net_cut_path": 0})
            a["net_cut"] += sum(1 for r in recs if r.get("why") == "deadline")
            a["net_cut_path"] += sum(1 for r in recs if (r.get("net") or {}).get("cut"))
    kw = {"reserve": float(t["reserve_s"]), "overhead": float(t["overhead_s"]), "background": True}
    acc: Dict[str, Dict[str, int]] = {}
    for panel, rows in games:
        sim = TM.simulate(rows, **kw)
        n_s = sum(1 for r in rows if r["searched"])
        for k in (panel, "U"):
            a = acc.setdefault(k, {"games": 0, "searched": 0, "forfeit": 0, "cut": 0, "cut_searched": 0, "path": 0,
                                   "battle_cut": 0})
            a["games"] += 1
            a["searched"] += n_s
            for kk, v in sim.items():
                a[kk] += v
    return {"policy": {"reserve_s": t["reserve_s"], "overhead_s": t["overhead_s"], "background": True},
            "by_panel": acc, "recorded": rec, "table": TM.replay(games)}


def distill_cells(cells: Path, *, honest: bool, names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """X65-F2's distillation fields over a searched label's cells dir (REPORTED; outcome-blind): per panel (the cell
    name's ``<label>__<panel>__…``), the searched-path records, those lacking a field (by field), and the cells with
    no capture file. ``names``: only these cells (the counted units); None = every complete attempt dir (never a
    ``.partial.<epoch>`` one), the checkpoint's form. Reads the decision logs and lists the capture dirs; never a
    result, a summary or ``games.jsonl``."""
    cells = Path(cells)
    need_s = DISTILL_SEARCHED + (("search_key",) if honest else ())
    need_x = DISTILL_EXCLUDED + (("search_key",) if honest else ())
    if names is None:
        names = sorted(d.name for d in cells.iterdir() if d.is_dir() and ".partial." not in d.name) \
            if cells.is_dir() else []
    out: Dict[str, Dict[str, Any]] = {}
    for name in names:
        panel = str(name).split("__")[1] if str(name).count("__") >= 3 else "?"
        a = out.setdefault(panel, {"units": 0, "searched": 0, "excluded": 0, "lacking": 0, "by_field": {},
                                   "units_without_capture": 0})
        a["units"] += 1
        d = cells / str(name)
        cap = d / "capture"
        if not (cap.is_dir() and any(cap.iterdir())):
            a["units_without_capture"] += 1
        log = d / "search_decisions.jsonl"
        for x in (log.read_text().splitlines() if log.is_file() else ()):
            if not x.strip():
                continue
            r = json.loads(x)
            if r.get("searched") is True:
                a["searched"] += 1
                need = need_s
            elif r.get("why") == "excluded":
                a["excluded"] += 1
                need = need_x
            else:
                continue
            gone = [f for f in need if r.get(f) in (None, "", {})]
            a["lacking"] += int(bool(gone))
            for f in gone:
                a["by_field"][f] = a["by_field"].get(f, 0) + 1
    return {"fields_searched": list(need_s), "fields_excluded": list(need_x), "by_panel": out}


def distill_report(state: Path, group: str, searched: str, units: Sequence[Mapping[str, Any]], *, honest: bool
                   ) -> Dict[str, Any]:
    """:func:`distill_cells` over every COUNTED searched unit of the group (the record's form)."""
    from main.reads import layout as LY

    names = [str(u["unit"]) for u in units if u["label"] == searched and u["counted"]]
    return distill_cells(LY.anchor_state(state, group, searched) / "cells", honest=honest, names=names)


# ------------------------------------------------------------------------------------------------ pairing
def _score(u: Mapping[str, Any]) -> Fraction:
    n, w, t = int(u["n"]), int(u["wins"]), int(u.get("ties") or 0)
    if n != 1:
        raise _err(f"unit {u['unit']}: n = {n}, not a one-game (per_game) unit")
    return Fraction(2 * w + t, 2)


def peer_drops(doc: Mapping[str, Any]) -> set:
    return {str(c["deal"]) for c in doc.get("prefix_classes") or () if c.get("peer_drop")}


def pair_deals(doc: Mapping[str, Any]) -> Tuple[Dict[str, Dict[int, List[Tuple[Fraction, Fraction]]]],
                                               Dict[str, Dict[str, Any]]]:
    """``(pairs, tally)``: ``pairs[panel][unit seed]`` = ``[(raw, searched), ...]`` in game order over the deals both
    arms COUNTED that are not peer drops; ``tally[panel]``: its planned deals, pairs, unfinished deals, unpaired deals
    by cause (pre- and post-treatment) and the post-treatment ones (``{unit_seed, game, cause, y}``: ``y`` the counted
    side's score, None when neither counted)."""
    side = {doc["searched"]: "s", doc["raw"]: "r"}
    idx: Dict[Tuple[str, str, int, int], Mapping[str, Any]] = {}
    for u in doc["units"]:
        if not u["counted"]:
            continue
        k = (side[u["label"]], str(u["panel"]), int(u["unit_seed"]), int(u["game"]))
        if k in idx:
            raise _err(f"deal {k[1:]} counted twice for {u['label']!r}")
        idx[k] = u
    drops = peer_drops(doc)
    u_panels = list(doc["u_panels"])
    pairs: Dict[str, Dict[int, List[Tuple[Fraction, Fraction]]]] = {ts: {} for ts in u_panels}
    tally: Dict[str, Dict[str, Any]] = {
        ts: {"planned": 0, "pairs": 0, "unfinished": 0, **{c: 0 for c in PRE_CAUSES + POST_CAUSES}, "pre": 0, "post": 0,
             "post_deals": []} for ts in u_panels}
    for d in doc["deals"]:
        ts, seed, game = str(d["panel"]), int(d["unit_seed"]), int(d["game"])
        if ts not in tally:
            raise _err(f"deal {d['deal']}: panel {ts!r} is not a U panel")
        t = tally[ts]
        t["planned"] += 1
        st, rt = d["searched"]["status"], d["raw"]["status"]
        s_u, r_u = idx.get(("s", ts, seed, game)), idx.get(("r", ts, seed, game))
        if (st == COUNTED) != (s_u is not None) or (rt == COUNTED) != (r_u is not None):
            raise _err(f"deal {d['deal']}: its planned status disagrees with the counted units")
        if PENDING in (st, rt):
            t["unfinished"] += 1
            continue
        if s_u is not None and r_u is not None:
            if d["deal"] in drops:
                t["peer_drop"] += 1
                t["pre"] += 1
            else:
                pairs[ts].setdefault(seed, []).append((_score(r_u), _score(s_u)))
                t["pairs"] += 1
            continue
        cause = "raw_missing" if s_u is not None else "searched_missing" if r_u is not None else "neither"
        y = _score(s_u) if s_u is not None else _score(r_u) if r_u is not None else None
        t[cause] += 1
        t["post"] += 1
        t["post_deals"].append({"unit_seed": seed, "game": game, "cause": cause, "y": None if y is None else str(y)})
    return pairs, tally


def imputed_d(cause: str, y: Optional[Fraction], case: str) -> Fraction:
    """A post-treatment unpaired deal's D at its worst case for ``case`` (``go`` or ``null``; module docstring)."""
    if cause == "neither":
        return Fraction(-1 if case == "go" else 1)
    assert y is not None
    if cause == "searched_missing":                 # the raw game is known: y = y_raw
        return -y if case == "go" else 1 - y
    if cause == "raw_missing":                      # the searched game is known: y = y_searched
        return y - 1 if case == "go" else y
    raise _err(f"cause {cause!r}")


def unit_stats(pairs: Mapping[str, Mapping[int, Sequence[Tuple[Fraction, Fraction]]]],
               tally: Optional[Mapping[str, Mapping[str, Any]]] = None, case: Optional[str] = None
               ) -> Dict[str, Dict[int, Fraction]]:
    """``{panel: {unit seed: mean D}}`` (exact): the pairs, plus with ``case`` every post-treatment unpaired deal
    imputed for it (an imputed deal enters its unit like a pair)."""
    out: Dict[str, Dict[int, Fraction]] = {}
    for ts, by in pairs.items():
        ds: Dict[int, List[Fraction]] = {seed: [s - r for r, s in v] for seed, v in by.items()}
        if case is not None and tally is not None:
            for e in tally[ts]["post_deals"]:
                y = None if e["y"] is None else Fraction(e["y"])
                ds.setdefault(int(e["unit_seed"]), []).append(imputed_d(e["cause"], y, case))
        out[ts] = {seed: sum(v, Fraction(0)) / len(v) for seed, v in sorted(ds.items())}
    return out


def _games_disagree(u: Mapping[str, Any]) -> bool:
    gs = u.get("games")
    if not gs:
        return False
    want = "win" if int(u["wins"]) else "tie" if int(u.get("ties") or 0) else "loss"
    return len(gs) != 1 or str(gs[0][3]) != want


def void_reasons(doc: Mapping[str, Any], pairs: Mapping[str, Mapping[int, Sequence[Any]]],
                 tally: Mapping[str, Mapping[str, Any]], p: Mapping[str, Any]) -> List[str]:
    """Every named VOID reason of the document (empty = the read may be decided)."""
    why: List[str] = []
    gp = doc["group"].get("g_prefix")
    if not gp:
        why.append("G-PREFIX: not run on the group")
    else:
        why += [f"G-PREFIX: deal {c['deal']} diverged before the search's first turn and is not a peer drop "
                f"(class {c['class']}: {c.get('why')})" for c in doc.get("prefix_classes") or () if not c["peer_drop"]]
        n_pairs = sum(len(v) for by in pairs.values() for v in by.values())
        n_drops = sum(int(t["peer_drop"]) for t in tally.values())
        if int(gp.get("checked", -1)) != n_pairs + n_drops:
            why.append(f"G-PREFIX checked {gp.get('checked')} pair(s), the rule pairs {n_pairs} with {n_drops} peer "
                       "drop(s): every pair must be checked")
    counted = [u for u in doc["units"] if u["counted"]]
    engines = {u.get("engine") for u in counted}
    known = sorted(str(e) for e in engines if e)
    if None in engines or "" in engines:
        why.append(f"engine: {sum(1 for u in counted if not u.get('engine'))} counted unit(s) on an unknown engine")
    if len(known) > 1:
        why.append(f"engine: counted units span {len(known)} engines {known}")
    reg = doc.get("engine_registered")
    if known and known != [reg]:
        why.append(f"engine: counted units played on {known}, not the registered {reg}")
    want_sha = doc.get("raw_sha256")
    for lab, sha in sorted(doc["resolved_sha256"].items()):
        if sha != want_sha:
            why.append(f"checkpoint: {lab!r} resolved to {str(sha)[:12]}, not the raw arm's {str(want_sha)[:12]}")
    other = [u["unit"] for u in counted if u.get("model_sha256") != want_sha]
    if other:
        why.append(f"checkpoint: {len(other)} counted unit(s) played other bytes than {str(want_sha)[:12]} "
                   f"(first {other[0]})")
    bad_s, bad_r = [], []
    for u in counted:
        sp = u.get("search_played")
        if u["label"] == doc["searched"]:
            ok = (u.get("our_regime") == doc["expected_regime"] and u.get("search") == doc["expected_search"]
                  and isinstance(sp, dict) and sp.get("search") == doc["expected_search"]
                  and sp.get("our_search") == doc["expected_search"] and sp.get("preset") == doc["expected_preset"])
            if not ok:
                bad_s.append(u["unit"])
        elif u.get("our_regime") != RAW_REGIME or u.get("search") is not None or sp is not None:
            bad_r.append(u["unit"])
    if bad_s:
        why.append(f"search config: {len(bad_s)} counted searched unit(s) not stamped / played as the registered "
                   f"{doc['expected_search']} {doc['expected_preset']} (first {bad_s[0]})")
    if bad_r:
        why.append(f"raw arm: {len(bad_r)} counted raw unit(s) carry a search stamp or a regime other than "
                   f"{RAW_REGIME!r} (first {bad_r[0]})")
    dis = [u["unit"] for u in counted if _games_disagree(u)]
    if dis:
        why.append(f"results: {len(dis)} counted unit(s) whose games.jsonl disagrees with the summary (first {dis[0]})")
    for ts, by in pairs.items():
        if len(by) < 2:
            why.append(f"{ts}: {len(by)} unit(s) hold a pair (the unit bootstrap needs >= 2)")
    unfinished = {ts: int(t["unfinished"]) for ts, t in tally.items() if t["unfinished"]}
    if unfinished:
        # a unit neither counted nor out of attempts (e.g. a FAILED attempt whose retry never ran): never dropped
        why.append(f"{sum(unfinished.values())} deal(s) unfinished: the read is incomplete ({unfinished})")
    for ts, t in tally.items():
        for kind, key, causes in (("post", "max_post_unpaired_frac", POST_CAUSES),
                                  ("pre", "max_pre_unpaired_frac", PRE_CAUSES)):
            if t["planned"] and Fraction(int(t[kind]), int(t["planned"])) > Fraction(str(p[key])):
                txt = ", ".join(f"{c} {t[c]}" for c in causes)
                why.append(f"{ts}: {t[kind]} of {t['planned']} planned deals {kind}-treatment unpaired ({txt}), above "
                           f"{key} {p[key]}")
    idn = doc.get("identity") or {}
    if idn.get("problems"):
        why += [f"identity: {x}" for x in idn["problems"]]
    if idn.get("verdict") != "PASS":
        why.append(f"identity: the live identity guard reads {idn.get('verdict')!r} ({idn.get('reason')})")
    return why


# ------------------------------------------------------------------------------------------------ estimates
def bootstrap(stats: Mapping[str, Mapping[int, Fraction]], u_panels: Sequence[str], B: int, seed: int
              ) -> Dict[str, Any]:
    """``{"G": {panel: [lo, hi]}, "G_U": [lo, hi]}``: the module docstring's exact scheme, a fresh rng."""
    import numpy as np

    rng = np.random.default_rng(seed)
    reps: Dict[str, Any] = {}
    for ts in u_panels:
        x = np.array([float(v) for _s, v in sorted(stats[ts].items())])
        idx = rng.integers(0, len(x), size=(B, len(x)))
        reps[ts] = x[idx].mean(axis=1)
    gu = np.stack([reps[ts] for ts in u_panels]).mean(axis=0)

    def ci(r: Any) -> List[float]:
        lo, hi = np.percentile(r, [2.5, 97.5])
        return [float(lo), float(hi)]

    return {"G": {ts: ci(reps[ts]) for ts in u_panels}, "G_U": ci(gu)}


def classify(go_ci: Sequence[float], null_ci: Sequence[float], margin_pp: Any, eps: Any, go: str = GO_LEAKY
             ) -> Dict[str, Any]:
    """GO (``go``: GO-LEAKY or GO-HONEST) / NULL / STOP from the GO-imputed LB and the NULL-imputed UB (proportions;
    the margin's pp / 100), exact fractions, rule 8."""
    from main.reads.rules_search_headroom import cmp

    margin = Fraction(str(margin_pp)) / 100
    t = {"LB(G_U go_imputed) > 0": cmp(go_ci[0], 0, eps, above=True),
         "UB(G_U null_imputed) < null_margin": cmp(null_ci[1], margin, eps, above=False)}
    up, below = t["LB(G_U go_imputed) > 0"][0], t["UB(G_U null_imputed) < null_margin"][0]
    cls = go if up else NULL if below else STOP
    return {"class": cls, "boundary": any(near for _c, near in t.values()), "go_below_margin": up and below,
            "tests": {k: {"crosses": v[0], "near": v[1]} for k, v in t.items()}}


def _search_report(doc: Mapping[str, Any], pairs_deals: set) -> Dict[str, Any]:
    """The deviation rate, exclusions and search CPU-s per game over the PAIRED searched games, per U panel and pooled."""
    acc: Dict[str, Dict[str, Any]] = {k: {"games": 0, "searched": 0, "deviations": 0, "excluded": 0,
                                          "search_cpu_s": 0.0} for k in [*doc["u_panels"], "U"]}
    for u in doc["units"]:
        if u["label"] != doc["searched"] or not u["counted"]:
            continue
        if (str(u["panel"]), int(u["unit_seed"]), int(u["game"])) not in pairs_deals:
            continue
        sp = u.get("search_played") or {}
        for k in (str(u["panel"]), "U"):
            a = acc[k]
            a["games"] += 1
            a["searched"] += int(sp.get("searched") or 0)
            a["deviations"] += int(sp.get("deviations") or 0)
            a["excluded"] += int(sp.get("excluded") or 0)
            a["search_cpu_s"] += float(sp.get("search_cpu_s") or 0.0)
    for a in acc.values():
        n, gm = a["searched"], a["games"]
        a.update(deviation_rate=a["deviations"] / n if n else None,
                 excluded_per_game=a["excluded"] / gm if gm else None,
                 search_cpu_s_per_game=a["search_cpu_s"] / gm if gm else None)
    return acc


def exclusions(doc: Mapping[str, Any], bounds: Optional[Mapping[str, Any]], eps: Any) -> Dict[str, Any]:
    """Per U panel and pooled: the searched arm's search-path decisions over EVERY counted searched unit, its
    exclusions and their share (exact); with ``bounds``, each panel's test against its bound (a bound's ``near``
    counts as exceeding) and ``excess`` = any panel over it."""
    from main.reads.rules_search_headroom import cmp

    acc: Dict[str, Dict[str, Any]] = {k: {"units": 0, "searched": 0, "excluded": 0, "net_cut": 0}
                                      for k in [*doc["u_panels"], "U"]}
    for u in doc["units"]:
        if u["label"] != doc["searched"] or not u["counted"]:
            continue
        sp = u.get("search_played") or {}
        for k in (str(u["panel"]), "U"):
            acc[k]["units"] += 1
            acc[k]["searched"] += int(sp.get("searched") or 0)
            acc[k]["excluded"] += int(sp.get("excluded") or 0)
            # the SAFETY NET's cuts (an envelope preset's summary ``net_cut``: searched decisions π played on the
            # deadline), counted beside the exclusions; REPORTED, never part of the exclusion share
            acc[k]["net_cut"] += int(sp.get("net_cut") or 0)
    excess = False
    for k, a in acc.items():
        n = a["searched"] + a["excluded"]
        share = Fraction(a["excluded"], n) if n else None
        a["share"] = float(share) if share is not None else None
        a["exact"] = str(share) if share is not None else None
        a["net_cut_share_of_searched"] = (a["net_cut"] / a["searched"]) if a["searched"] else None
        if bounds is not None and k in bounds:
            if share is None:
                a["test"] = {"bound": bounds[k], "exceeds": None, "near": None}
                continue
            over, near = cmp(share, bounds[k], eps, above=True)        # exact: a Fraction of counts
            a["test"] = {"bound": bounds[k], "exceeds": over or near, "near": near}
            excess = excess or over or near
    return {"by_panel": acc, "excess": excess if bounds is not None else None}


def timer_flag(rep: Optional[Mapping[str, Any]], t: Optional[Mapping[str, Any]], eps: Any) -> Optional[Dict[str, Any]]:
    """The decided ``timer_safe`` flag (module docstring) from :func:`timer_report`'s pooled-U counts: 0 games
    forfeited AND the cut share of searched decisions ≤ ``max_cut_frac`` (exact; within ``rule8_eps`` of the bound is
    ``near`` and NOT safe, root rule 8). None when no timer block is registered. Never VOIDs, never moves the class."""
    from main.reads.rules_search_headroom import cmp

    if t is None or rep is None:
        return None
    u = (rep.get("by_panel") or {}).get("U") or {}
    games, searched = int(u.get("games") or 0), int(u.get("searched") or 0)
    forfeits, cut = int(u.get("forfeit") or 0), int(u.get("cut_searched") or 0)
    out: Dict[str, Any] = {"bound": t["max_cut_frac"], "games": games, "forfeits": forfeits, "searched": searched,
                           "cut_searched": cut, "safe": False}
    if not games or not searched:
        return dict(out, share=None, exact=None, why="no counted searched decision")
    share = Fraction(cut, searched)
    over, near = cmp(share, t["max_cut_frac"], eps, above=True)
    return dict(out, share=float(share), exact=str(share), cut_test={"exceeds": over, "near": near},
                safe=forfeits == 0 and not over and not near)


def goal_b(pairs: Mapping[str, Mapping[int, Sequence[Tuple[Fraction, Fraction]]]], tally: Mapping[str, Mapping[str, Any]],
           u_panels: Sequence[str], cls: str, eps: Any) -> Dict[str, Any]:
    """The decided goal-b flag (module docstring): the searched arm's pooled-U score with every post-treatment unpaired
    deal's searched game at its worst case, its LB against 0.50."""
    from main.reads.rules import strat_score
    from main.reads.rules_search_headroom import cmp
    from main.x35.readout import ReadoutInputError

    counts = {}
    for ts in u_panels:
        wins = sum(1 for v in pairs[ts].values() for pr in v if pr[1] == 1)
        n = sum(len(v) for v in pairs[ts].values())
        for e in tally[ts]["post_deals"]:
            n += 1
            wins += int(e["cause"] == "raw_missing" and Fraction(e["y"]) == 1)      # the searched game counted; else a loss
        counts[ts] = (wins, n)
    out: Dict[str, Any] = {"counts": {k: list(v) for k, v in counts.items()}, "imputation": "a missing searched game "
                           "is a loss; a tie is a non-win", "reached": False, "label": None}
    try:
        sc = strat_score(counts, list(u_panels))
    except ReadoutInputError as exc:
        return dict(out, score=None, ci95=None, why=str(exc))
    c, near = cmp(sc.ci[0], "0.5", eps, above=True)
    reached = cls in (GO_HONEST, GO_LEAKY) and c
    return dict(out, score=sc.mean, ci95=list(sc.ci), lb_above_0_50={"crosses": c, "near": near}, reached=reached,
                label=("GOAL-B-REACHED" if cls == GO_HONEST else "GOAL-B-REACHED (leaky)") if reached else None)


def decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    from main.reads.rules_search_headroom import _goal_b, cmp

    pairs, tally = pair_deals(doc)
    u = list(doc["u_panels"])
    flat = [pr for by in pairs.values() for v in by.values() for pr in v]
    disc = sum(1 for r, s in flat if r != s)
    counts = {"by_panel": {ts: {"planned": t["planned"], "pairs": t["pairs"], "unfinished": t["unfinished"],
                                "pre_treatment": {"total": t["pre"], **{c: t[c] for c in PRE_CAUSES}},
                                "post_treatment": {"total": t["post"], **{c: t[c] for c in POST_CAUSES}}}
                           for ts, t in tally.items()},
              "units_with_pairs": {ts: len(pairs[ts]) for ts in pairs},
              "uncounted_units": {lab: sum(1 for x in doc["units"] if x["label"] == lab and not x["counted"])
                                  for lab in (doc["searched"], doc["raw"])},
              "unplayed_units": doc["group"].get("unplayed")}
    out: Dict[str, Any] = {"schema": SCHEMA, "counts": counts, "engine": doc["group"].get("engine"),
                           "discordance": {"pairs": len(flat), "discordant": disc,
                                           "share": disc / len(flat) if flat else None},
                           "peer_drops": {"classes": doc.get("prefix_classes"),
                                          "by_panel": {ts: t["peer_drop"] for ts, t in tally.items()}},
                           "identity": dict(doc.get("identity") or {}),
                           "distill": doc.get("distill"),
                           "exclusions": exclusions(doc, p.get("max_excluded_frac"), p["rule8_eps"]),
                           **({"timer": dict(doc.get("timer") or {},
                                             flag=timer_flag(doc.get("timer"), p["timer"], p["rule8_eps"]))}
                              if "timer" in p else {})}
    reasons = void_reasons(doc, pairs, tally, p)
    if reasons:
        return dict(out, decision={"class": VOID, "verdict": VOID, "reasons": reasons})
    b = p["bootstrap"]
    est: Dict[str, Any] = {}
    for name, case in zip(ESTIMATES, (None, "go", "null")):
        st = unit_stats(pairs, tally, case)
        g = {ts: sum(st[ts].values(), Fraction(0)) / len(st[ts]) for ts in u}
        gu = sum((g[ts] for ts in u), Fraction(0)) / len(u)
        ci = bootstrap(st, u, int(b["B"]), int(b["seed"]))            # a fresh default_rng(seed) per estimate
        est[name] = {"G": {ts: {"point": float(g[ts]), "ci95": ci["G"][ts]} for ts in u},
                     "G_U": {"point": float(gu), "ci95": ci["G_U"]},
                     "exact": {"G": {ts: str(g[ts]) for ts in g}, "G_U": str(gu)}}
    est["bootstrap"] = {"B": int(b["B"]), "seed": int(b["seed"]), "scheme": "numpy default_rng(seed), fresh per "
                        "estimate; u_panels in order; within-panel unit resampling; percentile 2.5 / 97.5"}
    go = go_class(doc["expected_preset"])
    cls = classify(est["go_imputed"]["G_U"]["ci95"], est["null_imputed"]["G_U"]["ci95"], p["null_margin_pp"],
                   p["rule8_eps"], go)
    cls["goal_b"] = goal_b(pairs, tally, u, cls["class"], p["rule8_eps"])
    cls["excess_exclusions"] = out["exclusions"]["excess"]
    if "timer" in p:
        cls["timer_safe"] = bool(out["timer"]["flag"]["safe"])
    goal = _goal_b({"u_panels": u}, pairs)
    lb = (goal["searched"].get("ci95") or [None])[0]
    if lb is not None:
        c, near = cmp(lb, "0.5", p["rule8_eps"], above=True)
        goal["searched_lb_above_0_50"] = {"lb": lb, "crosses": c, "near": near}
    drops = peer_drops(doc)
    pdeals = {(str(d["panel"]), int(d["unit_seed"]), int(d["game"])) for d in doc["deals"]
              if d["searched"]["status"] == COUNTED and d["raw"]["status"] == COUNTED and d["deal"] not in drops}
    out["estimates"] = est
    out["score_u"] = goal
    out["search"] = _search_report(doc, pdeals)
    pp = lambda x: f"{100 * x:+.2f}"                                                  # noqa: E731
    flags = [f for f in ("boundary", "go_below_margin", "excess_exclusions") if cls[f]]
    gi, ni, pa = est["go_imputed"]["G_U"], est["null_imputed"]["G_U"], est["paired"]["G_U"]
    verdict = (f"{cls['class']}: G_U (searched − raw, by deal, U) paired {pp(pa['point'])} pp "
               f"[{pp(pa['ci95'][0])}, {pp(pa['ci95'][1])}]; GO-imputed LB {pp(gi['ci95'][0])} ({go} iff > 0); "
               f"NULL-imputed UB {pp(ni['ci95'][1])} (NULL iff < {p['null_margin_pp']} pp)"
               + (f"; {cls['goal_b']['label']}" if cls["goal_b"]["reached"] else "")
               + ((f"; {TIMER_SAFE}" if cls["timer_safe"] else f"; NOT {TIMER_SAFE}") if "timer" in p else "")
               + (f"; flagged {', '.join(flags)}" if flags else ""))
    return dict(out, decision=dict(cls, verdict=verdict))
