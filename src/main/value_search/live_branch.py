"""THE DEVIATION-ONLY LIVE READ (``designs/research_state/measurements/dlive_read_2026-10-28/``; the S-U gate
review's M3 step 3, ``su_headroom_2026-10-27/REVIEW_gate.md``): against REAL Kakuna, does playing the search S′'s
deviation at a state, then P* thereafter, win more often than the raw game, in which P* played π's top action there?

The self-referee T of the S-U gate and its re-gate shares S′'s model of the opponent (P* on both sides; REVIEW_gate
M2). Here the truth is the real opponent: a deviation's two arms are two REAL games against Kakuna from the same state.
π's arm is the raw game, already played; S′'s arm is a FORCED BRANCH of it (``main.anchors --our-force``,
:mod:`main.anchors.forced_branch`): the deal replayed to the state (deterministic on one engine), S′'s pick forced
there, P* greedy after it. Both arms run on the deal's own battle seed, so they share every random number until their
play diverges.

Five verbs; the raw games are a ``main.reads`` anchor read (``load.input_log``, whole 20-game units), the branches a
``main.x35.kakuna`` state of per-game units:

* ``select``: per raw game, the :data:`sign_gate.PER_GAME` kept decisions (turns 10-30, >= 2 legal) with the smallest
  ``sign_gate.select_hash`` (the gate's and V-RANK's draw), from COUNTED raw units of every named (group, label) state;
  written once, frozen.
* ``score``: S′ (the registered preset, run offline through the live searcher's ``_search`` exactly as the gate did,
  :class:`sign_gate.GateScorer`) and, on a deviation, T (REPORTED), in the REGISTERED ORDER: per panel by (unit seed,
  game, draw hash). A panel stops at its QUOTA of S′ deviations: the read's size is a count of S′'s deviations, which
  reads no outcome. Resumable.
* ``plan``: one forced-branch unit per scored deviation (S′'s pick forced; kind ``branch``) and one NULL control per
  (panel, raw unit) (π's own top forced at that unit's scored state with the smallest draw hash; kind ``null``): a
  ``main.x35.kakuna`` plan whose units are the raw unit's own deal (``--deal-game G``) plus ``--our-force``. Written
  once; the branches play with ``python -m main.x35.kakuna run --state <branch state>``.
* ``read``: :func:`collect` then :func:`read` (below): the registered decision and the reported rows. ``main.reads``
  records it through the rule ``deviation_live`` (:mod:`main.reads.rules_deviation_live`).

THE ESTIMAND. Per deviation i, D_i = y(branch_i) - y(raw game of i), y = 1 / ½ / 0 for a win / tie / loss of ours.
Per panel the mean of D over its countable deviations (both games counted); pooled U = the mean of the U panels'
(equal weight). Interval: a game-clustered percentile bootstrap (``sign_gate.cluster_ratio_boot``: one
``default_rng(seed)``, B draws, panels resampled in the registered order; a raw game is the cluster, since its two
states share y(raw)). **GO iff LB(pooled U) > bar (0)**, a bound within ``rule8_eps`` of the bar reads STOP flagged
``boundary``; else STOP.

THE TEETH (each a VOID, never a skip): G-BRANCH (live, at the branch decision: :mod:`main.anchors.forced_branch`) failed
in any attempt of any unit; the branch's capture is not byte-identical to the raw game's (``main.reads.prefix.compare``:
the START document and both sides' protocol through ``|turn|<the branch turn>``); a NULL control whose whole protocol,
outcome or length differs from its raw game's; a branch that forced another action than the scored pick; counted units
on another engine than the registered one; a scored row on another preset; more planned branches than scored
deviations or a deviation without its branch; more than ``max_drop_frac`` of a panel's branches dropped (never counted
after ``kakuna.MAX_ATTEMPTS``).

    python -m main.value_search.live_branch select --raw-state DIR [--raw-state DIR] --out selected.jsonl
    python -m main.value_search.live_branch score  --selected selected.jsonl --model ZIP --search r0s4_rolls@10
        --quota heldout72_hpfix=600,forum72=600,p72=600,home72=300 --engine STAMP --out scored.jsonl [--budget-s S]
    python -m main.value_search.live_branch plan   --selected selected.jsonl --scored scored.jsonl
        --raw-state DIR [--raw-state DIR] --quota ... --out BRANCH_STATE
    python -m main.value_search.live_branch read   --branch-state BRANCH_STATE --scored scored.jsonl --params P.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PLAN_SCHEMA = "gen3_live_branch_plan_v1"
ROW_SCHEMA = "gen3_live_branch_row_v1"
VERDICT_SCHEMA = "gen3_live_branch_verdict_v1"
SELECT_SCHEMA = "gen3_live_branch_selected_v1"
RESULT = {"win": 1.0, "tie": 0.5, "loss": 0.0}
#: the null control's capture comparison runs to the end of the game (no ``|turn|`` this late exists)
WHOLE_GAME_TURN = 10 ** 6


class LiveBranchError(RuntimeError):
    """A state, a plan or a branch that cannot be used honestly: named, never skipped."""


def _jsonl(p: Path) -> List[Dict[str, Any]]:
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def parse_quota(text: str) -> Dict[str, int]:
    """``"heldout72_hpfix=600,home72=300"`` -> ``{panel: quota}`` (each an int >= 1)."""
    out: Dict[str, int] = {}
    for part in (x.strip() for x in str(text).split(",") if x.strip()):
        name, sep, val = part.partition("=")
        if not sep or not name or name in out:
            raise LiveBranchError(f"quota {text!r}: '<panel>=<n>' pairs, each panel once")
        try:
            n = int(val)
        except ValueError:
            raise LiveBranchError(f"quota {text!r}: {val!r} is not an integer") from None
        if n < 1:
            raise LiveBranchError(f"quota {text!r}: {name} needs >= 1")
        out[name] = n
    if not out:
        raise LiveBranchError("no quota named")
    return out


# --------------------------------------------------------------------------------------------------- the order
def order_key(r: Mapping[str, Any]) -> Tuple[int, int, str, str]:
    """The REGISTERED order of the states: unit seed, game (the kept decision's battle index), panel, draw hash. A
    panel's quota takes the shortest prefix of ITS states in this order."""
    return int(r["unit_seed"]), int(r["index"]), str(r["panel"]), str(r["select_hash"])


def is_deviation(row: Mapping[str, Any]) -> bool:
    s = row["search"]
    return bool(s.get("searched")) and bool(s.get("deviation"))


def to_score(selected: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]],
             quota: Mapping[str, int]) -> List[Dict[str, Any]]:
    """The selected states still to score, in the registered order, given what ``scored`` holds: a panel takes no
    state once its scored deviations reach its quota (a state of a panel with no quota is never scored)."""
    done = {r["key"] for r in scored}
    devs: Dict[str, int] = {}
    for r in scored:
        devs[r["panel"]] = devs.get(r["panel"], 0) + int(is_deviation(r))
    over = sorted(p for p, n in devs.items() if n > quota.get(p, 0))
    if over:
        raise LiveBranchError(f"scored deviations exceed the quota on {over}")
    out = []
    for r in sorted(selected, key=order_key):
        if r["key"] in done or r["panel"] not in quota:
            continue
        out.append(dict(r))
    return out


def quota_prefix(selected: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]],
                 quota: Mapping[str, int]) -> Dict[str, Any]:
    """Check that ``scored`` is exactly each panel's registered-order prefix up to its quota of deviations:
    ``{panel: {"states", "deviations", "met"}}``; a gap, an extra state or a state past the quota raises."""
    by_key = {r["key"]: r for r in scored}
    out: Dict[str, Any] = {}
    for panel, q in quota.items():
        states = [r for r in sorted(selected, key=order_key) if r["panel"] == panel]
        n_dev = n_states = 0
        stopped = False
        for r in states:
            row = by_key.get(r["key"])
            if stopped:
                if row is not None:
                    raise LiveBranchError(f"{r['key']}: scored past {panel}'s quota of {q} deviations")
                continue
            if row is None:
                raise LiveBranchError(f"{r['key']}: a gap in {panel}'s registered order (unscored before the quota)")
            n_states += 1
            n_dev += int(is_deviation(row))
            stopped = n_dev >= q
        out[panel] = {"states": n_states, "deviations": n_dev, "met": n_dev >= q, "selected": len(states)}
    stray = sorted({r["panel"] for r in scored} - set(quota))
    if stray:
        raise LiveBranchError(f"scored states on panels with no quota: {stray}")
    return out


# --------------------------------------------------------------------------------------------------- select
def select_states(raw_states: Sequence[Path]) -> List[Dict[str, Any]]:
    """The gate's draw (``sign_gate.select``) over the COUNTED units of every raw kakuna state (``<state>/anchor/<group>/
    <label>``), tagged with the state they came from."""
    from main.value_search import sign_gate as SG

    out: List[Dict[str, Any]] = []
    for st in raw_states:
        st = Path(st)
        label, group = st.name, st.parent.name
        cells = SG.counted_cells(st.parent.parent.parent, group, label)
        for r in SG.select(SG.kept_records(cells)):
            out.append({**r, "schema": SELECT_SCHEMA, "raw_state": str(st)})
    keys = [r["key"] for r in out]
    if len(set(keys)) != len(keys):
        raise LiveBranchError("a state is drawn from two raw states")
    return out


# ---------------------------------------------------------------------------------------------------- plan
def _raw_units(raw_states: Sequence[Path]) -> Dict[str, Tuple[Path, Any]]:
    from main.x35 import kakuna as KA

    out: Dict[str, Tuple[Path, Any]] = {}
    for st in raw_states:
        for u in KA.load_plan(Path(st)):
            if u.name in out:
                raise LiveBranchError(f"raw unit {u.name} planned in two states")
            out[u.name] = (Path(st), u)
    return out


def plan_branches(selected: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]],
                  quota: Mapping[str, int], raw_states: Sequence[Path], out: Path) -> List[Dict[str, Any]]:
    """Write the branch state ``out`` (``plan.json``, ``branch_plan.jsonl``, ``specs/``): one forced branch per scored
    deviation (S′'s pick) and one null control per (panel, raw unit) (module docstring). Written once."""
    import dataclasses

    from main.anchors.forced_branch import spec_from_kept

    out = Path(out)
    if (out / "plan.json").exists():
        raise LiveBranchError(f"{out / 'plan.json'} exists: a branch plan is written once")
    prefix = quota_prefix(selected, scored, quota)
    sel = {r["key"]: r for r in selected}
    raw = _raw_units(raw_states)
    rows: List[Dict[str, Any]] = []
    nulls: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for row in sorted(scored, key=lambda r: order_key(sel[r["key"]])):
        rec = sel[row["key"]]
        cur = nulls.get((rec["panel"], rec["cell"]))
        if cur is None or rec["select_hash"] < cur["select_hash"]:
            nulls[(rec["panel"], rec["cell"])] = rec
        if is_deviation(row):
            rows.append({"kind": "branch", "rec": rec, "action": int(row["search"]["played"])})
    for (_p, _c), rec in sorted(nulls.items()):
        rows.append({"kind": "null", "rec": rec, "action": int(rec["pi_argmax"])})
    (out / "specs").mkdir(parents=True, exist_ok=True)
    units, meta = [], []
    for x in rows:
        rec = x["rec"]
        if rec["cell"] not in raw:
            raise LiveBranchError(f"{rec['key']}: raw cell {rec['cell']} is in no raw state's plan")
        st, ru = raw[rec["cell"]]
        if ru.game is not None or ru.search is not None:
            raise LiveBranchError(f"raw unit {ru.name}: a whole raw 20-game unit is the branches' source")
        game = int(rec["index"])
        spec = spec_from_kept(rec, x["action"], game=game)
        name = f"{ru.name}__g{game:02d}__{spec['kind']}__t{spec['turn']}n{spec['n']}"
        sp = out / "specs" / f"{name}.json"
        sp.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
        u = dataclasses.replace(ru, name=name, games=1, game=game, input_log=None, force=str(sp.resolve()))
        units.append(u)
        meta.append({"schema": PLAN_SCHEMA, "unit": name, "kind": spec["kind"], "key": spec["key"],
                     "panel": rec["panel"], "cell": rec["cell"], "raw_state": str(st), "unit_seed": int(ru.unit_seed),
                     "game": game, "deal": rec["deal"], "battle": spec["battle"], "turn": spec["turn"], "n": spec["n"],
                     "action": spec["action"], "pi_argmax": spec["pi_argmax"], "spec": str(sp)})
    names = [u.name for u in units]
    if len(set(names)) != len(names):
        raise LiveBranchError("two branch units share a name")
    # the queue: the registered order (unit seed, game), branches before the unit's null control
    order = sorted(range(len(units)), key=lambda i: (meta[i]["unit_seed"], meta[i]["game"], meta[i]["panel"],
                                                     meta[i]["kind"] == "null", meta[i]["key"]))
    units, meta = [units[i] for i in order], [meta[i] for i in order]
    (out / "plan.json").write_text(json.dumps({"units": [u.__dict__ for u in units]}, indent=1) + "\n")
    with open(out / "branch_plan.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m, sort_keys=True) + "\n")
    (out / "quota.json").write_text(json.dumps({"quota": dict(quota), "prefix": prefix}, indent=1, sort_keys=True)
                                    + "\n")
    return meta


# --------------------------------------------------------------------------------------------------- collect
def _game_row(cell: Path, battle: str) -> Optional[Dict[str, Any]]:
    for r in _jsonl(cell / "games.jsonl"):
        if r.get("battle_tag") == battle:
            return r
    return None


def _attempts(state: Path) -> Dict[str, int]:
    n: Dict[str, int] = {}
    for e in _jsonl(state / "units.jsonl"):
        if e.get("event") == "end":
            n[e["unit"]] = n.get(e["unit"], 0) + 1
    return n


def _unit_cpu(state: Path) -> Dict[str, float]:
    cpu: Dict[str, float] = {}
    for e in _jsonl(state / "units.jsonl"):
        if e.get("event") == "end" and e.get("cpu_s") is not None:
            cpu[e["unit"]] = cpu.get(e["unit"], 0.0) + float(e["cpu_s"])
    return cpu


def _g_branch_fails(state: Path, unit: str) -> List[str]:
    """Every G-BRANCH failure any attempt of ``unit`` recorded (the live cell and the moved-aside partials)."""
    from main.anchors.runner import FORCE_RECORD_NAME

    fails = []
    for d in [state / "cells" / unit, *sorted((state / "cells").glob(f"{unit}.partial.*"))]:
        for r in _jsonl(d / FORCE_RECORD_NAME):
            if r.get("g_branch") != "pass":
                fails.append(f"{d.name}: {r.get('mismatch')}")
    return fails


def _capture(cell: Path, battle: str) -> Dict[str, Any]:
    p = cell / "capture" / f"{battle}.json"
    try:
        return dict(json.loads(p.read_text()))
    except (OSError, ValueError) as exc:
        raise LiveBranchError(f"{p}: unreadable capture ({exc})") from None


def collect(branch_state: Path) -> List[Dict[str, Any]]:
    """One row per planned branch unit: its status (``counted`` / ``dropped`` / ``pending``), the two games' outcomes,
    the teeth (G-BRANCH in every attempt, the capture comparison, the null control's identity), the engines, the
    forced action and the unit's CPU."""
    from main.anchors.forced_branch import load_spec
    from main.reads.prefix import compare
    from main.x35 import kakuna as KA

    st = Path(branch_state)
    units = {u.name: u for u in KA.load_plan(st)}
    tries, cpu = _attempts(st), _unit_cpu(st)
    out = []
    for m in _jsonl(st / "branch_plan.jsonl"):
        u = units.get(m["unit"])
        if u is None:
            raise LiveBranchError(f"{m['unit']}: in branch_plan.jsonl, not in plan.json")
        spec = load_spec(m["spec"])
        row: Dict[str, Any] = {"schema": ROW_SCHEMA, **{k: m[k] for k in ("unit", "kind", "key", "panel", "cell",
                                                                          "unit_seed", "game", "deal", "battle",
                                                                          "turn", "n", "action", "pi_argmax")},
                               "attempts": tries.get(u.name, 0), "cpu_s": cpu.get(u.name),
                               "g_branch_fails": _g_branch_fails(st, u.name)}
        raw_cell = Path(m["raw_state"]) / "cells" / m["cell"]
        rr = _game_row(raw_cell, m["battle"])
        if rr is None:
            raise LiveBranchError(f"{m['unit']}: the raw cell {m['cell']} has no game {m['battle']}")
        row.update(y_raw=RESULT[rr["result"]], raw_turns=int(rr["turns"]), raw_engine=rr.get("engine_stamp"))
        if spec["action"] != m["action"] or spec["key"] != m["key"]:
            row["spec_mismatch"] = True
        if KA.is_done(st, u):
            cell = st / "cells" / u.name
            br = _jsonl(cell / "games.jsonl")
            if len(br) != 1:
                raise LiveBranchError(f"{u.name}: a counted branch unit with {len(br)} game rows")
            b = br[0]
            recs = _jsonl(cell / "branch.jsonl")
            passed = [r for r in recs if r.get("g_branch") == "pass"]
            row.update(status="counted", y_branch=RESULT[b["result"]], branch_turns=int(b["turns"]),
                       branch_engine=b.get("engine_stamp"), branch_battle=b.get("battle_tag"),
                       forced=int(passed[0]["forced"]) if len(passed) == 1 else None,
                       branch_index=passed[0].get("index") if len(passed) == 1 else None,
                       our_regime=b.get("our_regime"))
            try:
                cmp = compare(_capture(raw_cell, m["battle"]), _capture(cell, str(b.get("battle_tag"))),
                              WHOLE_GAME_TURN if m["kind"] == "null" else int(m["turn"]))
                row["capture"] = {"ok": bool(cmp["ok"]), "why": cmp["why"]}
            except (LiveBranchError, Exception) as exc:      # noqa: BLE001 - an unreadable capture is a failed tooth
                row["capture"] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
            row["d"] = row["y_branch"] - row["y_raw"]
        elif row["attempts"] >= KA.MAX_ATTEMPTS:
            row["status"] = "dropped"
        else:
            row["status"] = "pending"
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------------------ read
def void_reasons(rows: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]],
                 params: Mapping[str, Any]) -> List[str]:
    """The registered teeth (module docstring), each a named VOID."""
    why: List[str] = []
    gb = [r["unit"] for r in rows if r["g_branch_fails"]]
    if gb:
        why.append(f"G-BRANCH: {len(gb)} unit(s) failed the live prefix check (first {gb[0]})")
    counted = [r for r in rows if r["status"] == "counted"]
    cap = [r["unit"] for r in counted if not r["capture"]["ok"]]
    if cap:
        why.append(f"capture: {len(cap)} counted branch(es) whose protocol is not the raw game's through the branch "
                   f"turn (first {cap[0]})")
    nul = [r["unit"] for r in counted if r["kind"] == "null"
           and (r["y_branch"] != r["y_raw"] or r["branch_turns"] != r["raw_turns"])]
    if nul:
        why.append(f"null control: {len(nul)} replay(s) of π's own action changed the outcome or length (first "
                   f"{nul[0]})")
    forced = [r["unit"] for r in counted if r.get("forced") != r["action"] or r.get("spec_mismatch")
              or r.get("branch_index") != r["game"]]
    if forced:
        why.append(f"forced action: {len(forced)} branch(es) did not play their planned action in their raw game's "
                   f"series index (first {forced[0]})")
    reg = params.get("engine")
    eng = {r.get("raw_engine") for r in counted} | {r.get("branch_engine") for r in counted}
    if None in eng or "" in eng:
        why.append("engine: a counted game on an unknown engine")
    known = sorted(str(e) for e in eng if e)
    if len(known) > 1 or (reg and known and known != [reg]):
        why.append(f"engine: the counted games played on {known}, the registered one is {reg}")
    preset = params.get("preset")
    if preset is not None:
        off = [r["key"] for r in scored if r["search"].get("searched") and (r["search"].get("config") or {})
               .get("name") != preset]
        if off:
            why.append(f"search: {len(off)} scored state(s) not on the registered preset {preset!r} (first {off[0]})")
    devs = {r["key"] for r in scored if is_deviation(r)}
    planned = {r["key"] for r in rows if r["kind"] == "branch"}
    if devs != planned:
        why.append(f"plan: {len(devs - planned)} scored deviation(s) without a branch, {len(planned - devs)} branch(es) "
                   "without a scored deviation")
    by_pick = {r["key"]: int(r["search"]["played"]) for r in scored if is_deviation(r)}
    wrong = [r["unit"] for r in rows if r["kind"] == "branch" and by_pick.get(r["key"]) != r["action"]]
    if wrong:
        why.append(f"plan: {len(wrong)} branch(es) force another action than S′'s scored pick (first {wrong[0]})")
    cap_frac = float(params["max_drop_frac"])
    for panel in sorted({r["panel"] for r in rows}):
        pr = [r for r in rows if r["panel"] == panel and r["kind"] == "branch"]
        drop = sum(1 for r in pr if r["status"] == "dropped")
        if pr and drop / len(pr) > cap_frac:
            why.append(f"drops: {drop} of {len(pr)} {panel} branches never counted (> {cap_frac:g})")
    return why


def _groups(rows: Sequence[Mapping[str, Any]], value: Any) -> Dict[str, Dict[str, List[float]]]:
    g: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        g.setdefault(r["panel"], {}).setdefault(r["deal"], []).append(float(value(r)))
    return g


def _diff_boot(ga: Mapping[str, Mapping[str, Sequence[float]]], u: Sequence[str], home: str, *, B: int,
               seed: int) -> Dict[str, Any]:
    """U - home (pooled U equal-weighted minus home's ratio), one stream: each replicate draws the U panels' clusters
    in order, then home's (``sign_gate.read``'s U - home72)."""
    rng = np.random.default_rng(seed)

    def ratio(p: str, idx: Optional[np.ndarray]) -> float:
        ks = sorted(ga.get(p, {}))
        arrs = [np.asarray([v for v in ga[p][k] if not np.isnan(v)]) for k in ks]
        sel = range(len(ks)) if idx is None else idx
        n = sum(len(arrs[j]) for j in sel)
        return sum(float(arrs[j].sum()) for j in sel) / max(n, 1)

    point = float(np.mean([ratio(p, None) for p in u])) - ratio(home, None)
    bs = []
    for _ in range(int(B)):
        vals = {p: ratio(p, rng.integers(0, len(ga.get(p, {})), len(ga.get(p, {})))) for p in [*u, home]}
        bs.append(float(np.mean([vals[p] for p in u])) - vals[home])
    return {"est": point, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def read(rows: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]],
         params: Mapping[str, Any]) -> Dict[str, Any]:
    """The registered decision (module docstring) and the REPORTED rows. ``params``: ``u_panels``, ``home_panel``,
    ``bar``, ``rule8_eps``, ``bootstrap`` {B, seed}, ``max_drop_frac``, ``engine``, ``preset``. A pending branch
    refuses; a failing tooth returns VOID."""
    from main.value_search.sign_gate import cluster_ratio_boot

    pend = [r["unit"] for r in rows if r["status"] == "pending"]
    if pend:
        raise LiveBranchError(f"{len(pend)} branch unit(s) not yet played to a verdict (first {pend[0]})")
    u, home = list(params["u_panels"]), str(params["home_panel"])
    B, seed = int(params["bootstrap"]["B"]), int(params["bootstrap"]["seed"])
    why = void_reasons(rows, scored, params)
    if why:
        return {"schema": VERDICT_SCHEMA, "verdict": "VOID", "void": why, "boundary": False}
    sc = {r["key"]: r for r in scored}
    dev = [r for r in rows if r["kind"] == "branch" and r["status"] == "counted"]
    primary = cluster_ratio_boot(_groups(dev, lambda r: r["d"]), u, B=B, seed=seed)
    bar, eps = float(params["bar"]), float(params["rule8_eps"])
    boundary = abs(primary["lo"] - bar) <= eps
    verdict = "GO" if primary["lo"] > bar + eps else "STOP"

    def sign(r: Mapping[str, Any]) -> float:
        return float("nan") if r["d"] == 0 else float(r["d"] > 0)

    def t_imp(r: Mapping[str, Any]) -> float:
        ref = sc[r["key"]].get("referee")
        return float("nan") if ref is None else float(ref["T_pick"] - ref["T_top"])

    def gap(r: Mapping[str, Any]) -> float:
        return r["d"] - t_imp(r)

    def agree(r: Mapping[str, Any]) -> float:
        ref = sc[r["key"]].get("referee")
        if ref is None or r["d"] == 0 or ref["T_pick"] == ref["T_top"]:
            return float("nan")
        return float((r["d"] > 0) == (ref["T_pick"] > ref["T_top"]))

    def stratum(pred: Any) -> Dict[str, Any]:
        rs = [r for r in dev if pred(sc[r["key"]])]
        return cluster_ratio_boot(_groups(rs, lambda r: r["d"]), u, B=B, seed=seed) if rs else {"n": 0}

    def exposed(s: Mapping[str, Any]) -> bool:
        leak = s["search"].get("leak") or {}
        fl = s.get("speed_flip") or {}
        return bool(leak.get("rewritten_moves") or leak.get("rewritten_switches") or fl.get("speed_pinned"))

    rep: Dict[str, Any] = {
        "home_d": cluster_ratio_boot(_groups(dev, lambda r: r["d"]), [home], B=B, seed=seed),
        "u_minus_home_d": _diff_boot(_groups(dev, lambda r: r["d"]), u, home, B=B, seed=seed),
        "sign_live_u": cluster_ratio_boot(_groups(dev, sign), u, B=B, seed=seed),
        "sign_live_home": cluster_ratio_boot(_groups(dev, sign), [home], B=B, seed=seed),
        "t_imp_u": cluster_ratio_boot(_groups(dev, t_imp), u, B=B, seed=seed),
        "t_imp_home": cluster_ratio_boot(_groups(dev, t_imp), [home], B=B, seed=seed),
        "live_minus_t_u": cluster_ratio_boot(_groups(dev, gap), u, B=B, seed=seed),
        "live_minus_t_home": cluster_ratio_boot(_groups(dev, gap), [home], B=B, seed=seed),
        "t_live_sign_agreement_u": cluster_ratio_boot(_groups(dev, agree), u, B=B, seed=seed),
        "speed_pinned_u": stratum(lambda s: bool((s.get("speed_flip") or {}).get("speed_pinned"))),
        "speed_not_pinned_u": stratum(lambda s: not (s.get("speed_flip") or {}).get("speed_pinned")),
        "leak_exposed_u": stratum(exposed),
        "leak_clean_u": stratum(lambda s: not exposed(s)),
    }
    per: Dict[str, Any] = {}
    for p in [*u, home]:
        pr = [r for r in rows if r["panel"] == p]
        br = [r for r in pr if r["kind"] == "branch"]
        nl = [r for r in pr if r["kind"] == "null"]
        cpu = [float(r["cpu_s"]) for r in pr if r.get("cpu_s") is not None]
        per[p] = {"branches": len(br), "counted": sum(1 for r in br if r["status"] == "counted"),
                  "dropped": sum(1 for r in br if r["status"] == "dropped"),
                  "split": sum(1 for r in br if r["status"] == "counted" and r["d"] != 0),
                  "nulls": len(nl), "nulls_counted": sum(1 for r in nl if r["status"] == "counted"),
                  "games": len({r["deal"] for r in br}),
                  "branch_cpu_s": {"mean": float(np.mean(cpu)) if cpu else None, "total": float(np.sum(cpu))}}
    rep["per_set"] = per
    return {"schema": VERDICT_SCHEMA, "verdict": verdict, "boundary": boundary,
            "rule": (f"GO iff LB(mean D = y(S′ pick branch) - y(raw), pooled U {u}, panels equal-weighted, "
                     f"game-clustered) > {bar} (rule 8: within {eps} = STOP)"),
            "primary": primary, "reported": rep, "n_branches": len(rows), "params": dict(params)}


# ------------------------------------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.value_search.live_branch", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    a = sub.add_parser("select")
    a.add_argument("--raw-state", action="append", required=True, help="a raw kakuna state <state>/anchor/<g>/<label>")
    a.add_argument("--out", required=True)
    b = sub.add_parser("score")
    b.add_argument("--selected", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--search", required=True)
    b.add_argument("--quota", required=True)
    b.add_argument("--engine", required=True, help="the registered engine stamp: the scorer refuses another")
    b.add_argument("--out", required=True)
    b.add_argument("--budget-s", type=float, default=None,
                   help="start no new state after this many seconds (a bounded chunk ends cleanly; resumable)")
    c = sub.add_parser("plan")
    c.add_argument("--selected", required=True)
    c.add_argument("--scored", required=True)
    c.add_argument("--raw-state", action="append", required=True)
    c.add_argument("--quota", required=True)
    c.add_argument("--out", required=True)
    d = sub.add_parser("read")
    d.add_argument("--branch-state", required=True)
    d.add_argument("--scored", required=True)
    d.add_argument("--params", required=True)
    d.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    try:
        return _run(args)
    except LiveBranchError as exc:
        print(f"[live_branch] REFUSED: {exc}", file=sys.stderr)
        return 2


def _run(args: Any) -> int:
    if args.verb == "select":
        out = Path(args.out)
        if out.exists():
            raise LiveBranchError(f"{out}: a selection is written once (frozen)")
        sel = select_states([Path(x) for x in args.raw_state])
        with open(out, "w") as f:
            for r in sel:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        by: Dict[str, int] = {}
        for r in sel:
            by[r["panel"]] = by.get(r["panel"], 0) + 1
        print(json.dumps({"states": len(sel), "per_panel": by}))
        return 0
    if args.verb == "score":
        return _score(args)
    if args.verb == "plan":
        meta = plan_branches(_jsonl(Path(args.selected)), _jsonl(Path(args.scored)), parse_quota(args.quota),
                             [Path(x) for x in args.raw_state], Path(args.out))
        by: Dict[str, Dict[str, int]] = {}
        for m in meta:
            k = by.setdefault(m["panel"], {"branch": 0, "null": 0})
            k[m["kind"]] += 1
        print(json.dumps({"units": len(meta), "per_panel": by}))
        return 0
    rows = collect(Path(args.branch_state))
    res = read(rows, _jsonl(Path(args.scored)), json.loads(Path(args.params).read_text()))
    txt = json.dumps(res, indent=1, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(txt + "\n")
    print(txt)
    return 0


def _score(args: Any) -> int:
    from main.anchors.search_presets import canonical
    from main.value_search.sign_gate import GateScorer

    from main.anchors.server import engine_stamp

    quota = parse_quota(args.quota)
    search = canonical(args.search)
    here = engine_stamp()
    if here != args.engine:
        raise LiveBranchError(f"this checkout's engine is {here}, the registered one {args.engine}: no state is scored")
    stamp = Path(args.out).with_name("engine.txt")
    if stamp.exists() and stamp.read_text().strip() != here:
        raise LiveBranchError(f"{stamp} records {stamp.read_text().strip()}, this checkout's engine is {here}")
    stamp.write_text(here + "\n")
    sel = _jsonl(Path(args.selected))
    t0 = time.monotonic()
    scored = _jsonl(Path(args.out))
    todo = to_score(sel, scored, quota)
    devs: Dict[str, int] = {}
    for r in scored:
        devs[r["panel"]] = devs.get(r["panel"], 0) + int(is_deviation(r))
    sc: Any = None
    n = 0
    try:
        for r in todo:
            if devs.get(r["panel"], 0) >= quota[r["panel"]]:
                continue                                  # the panel met its quota: its later states are never scored
            if args.budget_s is not None and time.monotonic() - t0 > args.budget_s:
                print(f"[live_branch] budget {args.budget_s:g} s spent: {n} scored this chunk", flush=True)
                return 0
            if sc is None:
                sc = GateScorer(args.model, search)
            rec = sc.score(r)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
                f.flush()
            n += 1
            devs[r["panel"]] = devs.get(r["panel"], 0) + int(is_deviation(rec))
            # progress only: no outcome is printed while scoring
            print(f"[live_branch] {len(scored) + n} {r['panel']} cpu {rec['search'].get('cpu_s')}", flush=True)
        print(json.dumps({"scored": len(scored) + n, "deviations": devs, "left": 0}))
        return 0
    finally:
        if sc is not None:
            sc.close()

if __name__ == "__main__":
    sys.exit(main())
