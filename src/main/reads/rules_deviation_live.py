r"""The deviation-only live read's registered rule ``deviation_live`` (``measurements/dlive_read_2026-10-28/``; the S-U
gate review's M3 step 3).

The raw games are ONE raw label (a pinned P*, ``load.input_log`` = the kept turn window) in one or two anchor groups
(``u_group`` plays the ``u_panels``, ``home_group`` the ``home_panel``; whole 20-game units) against Kakuna. Three
ARTEFACTS are made before the read is decided by :mod:`main.value_search.live_branch`, and read by path with their
sha256 recorded: ``selected`` (the hashed draw), ``scored`` (S′ and T in the registered order, each panel to its
``quota`` of S′ deviations) and the ``branch_state`` (the forced-branch units' plan and cells: S′'s pick forced at each
deviation, and the null controls).

THE ESTIMAND AND THE CLASS: :func:`main.value_search.live_branch.read`: the mean of D = y(S′'s branch) - y(raw game)
over the countable deviations, pooled U equal-weighted, game-clustered bootstrap. GO iff its LB > ``bar`` (a bound
within ``rule8_eps`` of the bar reads STOP, flagged); else STOP.

VOID (named, before any estimate): the raw arm's counted units on two engines, an unknown one or another than the
registered ``engine``, on other bytes than the pinned sha256, carrying a search stamp, or a regime other than greedy; a
selected state from a cell that is not a counted cell, whose hash is not its draw's, or more than ``per_game`` of one
game; a scored state never selected; the scored rows not each panel's registered-order prefix up to its quota; and
every tooth of :func:`main.value_search.live_branch.void_reasons` (G-BRANCH, the capture prefix, the null controls,
the forced action, the engine, the preset, the plan, the drop cap).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Tuple

SCHEMA = "gen3_read_deviation_live_v1"
KEYS = ("label", "u_group", "home_group", "u_panels", "home_panel", "quota", "bar", "bootstrap", "rule8_eps",
        "input_log", "per_game", "search", "max_drop_frac", "selected", "scored", "branch_state")
OPTIONAL = ("engine",)
RAW_REGIME = "greedy"
VOID = "VOID"
RULE_DOCS = {"deviation_live": "the deviation-only live read: mean D = y(S′'s forced branch) - y(raw game) at S′'s "
                               "deviations from pi on raw P* vs Kakuna states, pooled U; GO iff its LB > bar, else STOP"}
_UNIT_KEYS = ("unit", "label", "panel", "unit_seed", "counted", "our_regime", "search", "model_sha256", "engine")
_SEL_KEYS = ("cell", "panel", "deal", "turn", "n", "key", "select_hash", "index", "unit_seed")


def _err(msg: str) -> Exception:
    from main.reads.rules import RuleInputError

    return RuleInputError(msg)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    from main.anchors.input_log import InputLogError, canonical
    from main.anchors.search_presets import SearchSpecError, canonical as search_canonical
    from main.reads.rules import _check_keys

    probs = _check_keys(p, KEYS, KEYS + OPTIONAL)
    lab: Any = p.get("label")
    u, home = p.get("u_panels"), p.get("home_panel")
    if not (isinstance(home, str) and isinstance(u, list) and u and all(isinstance(x, str) for x in u)
            and len(set(u)) == len(u) and home not in u):
        return probs + ["home_panel: a team set; u_panels: distinct team sets, home_panel not among them"]
    groups = {p.get("u_group"), p.get("home_group")}
    for gname in sorted(str(x) for x in groups):
        g = anchor.get(gname)
        if g is None:
            probs.append(f"group {gname!r} is not an anchor group")
            continue
        if list(g.models) != [lab]:
            probs.append(f"group {g.name!r} must play exactly {lab!r}, plays {list(g.models)}")
        if g.per_game:
            probs.append(f"group {g.name!r}: the raw games are whole 20-game units (not per_game)")
    gu, gh = anchor.get(p.get("u_group")), anchor.get(p.get("home_group"))
    if gu is not None and gh is not None:
        want = ({*u, home} if gu is gh else None)
        if want is not None and set(gu.teamsets) != want:
            probs.append(f"group {gu.name!r} must play exactly {sorted(want)}, plays {list(gu.teamsets)}")
        if want is None and (set(gu.teamsets) != set(u) or set(gh.teamsets) != {home}):
            probs.append(f"group {gu.name!r} must play exactly the u_panels {u} and {gh.name!r} exactly {home!r}")
    if lab in models:
        m = models[lab]
        if m.ref.kind != "pinned" or not m.ref.sha256:
            probs.append(f"{lab!r} must be a pinned ref with its sha256")
        if "search" in m.load:
            probs.append(f"{lab!r} must be RAW (no load.search)")
        try:
            if m.load.get("input_log") != canonical(str(p.get("input_log"))):
                probs.append(f"{lab!r} keeps input logs {m.load.get('input_log')!r}, not input_log {p.get('input_log')!r}")
        except InputLogError as exc:
            probs.append(f"input_log: {exc}")
    else:
        probs.append(f"label {lab!r} is not a model of the spec")
    q = p.get("quota")
    if not (isinstance(q, dict) and set(q) == {*u, home} and all(_int(v) and v >= 1 for v in q.values())):
        probs.append(f"quota: {{panel: int >= 1}} for exactly {[*u, home]}")
    if not (_num(p.get("bar")) and -1 < p["bar"] < 1):
        probs.append("bar: a number in (-1, 1)")
    bs = p.get("bootstrap")
    if not (isinstance(bs, dict) and set(bs) == {"B", "seed"} and _int(bs.get("B")) and bs["B"] >= 1000
            and _int(bs.get("seed"))):
        probs.append("bootstrap: {B: int >= 1000, seed: int}")
    eps: Any = p.get("rule8_eps")
    if not (_num(eps) and 0 <= eps < 0.01):
        probs.append("rule8_eps: a number in [0, 0.01)")
    if not (_int(p.get("per_game")) and p["per_game"] >= 1):
        probs.append("per_game: an int >= 1")
    try:
        if search_canonical(p.get("search")) != p.get("search"):
            probs.append(f"search: spell it canonically ({search_canonical(p.get('search'))!r})")
    except SearchSpecError as exc:
        probs.append(f"search: {exc}")
    if not (_num(p.get("max_drop_frac")) and 0 <= p["max_drop_frac"] <= 0.5):
        probs.append("max_drop_frac: a number in [0, 0.5]")
    for k in ("selected", "scored", "branch_state"):
        if not (isinstance(p.get(k), str) and p[k]):
            probs.append(f"{k}: the artefact's path")
    if "engine" in p and not (isinstance(p["engine"], str) and p["engine"]):
        probs.append("engine: the registered engine stamp (port=..;env=..)")
    return probs


def _jsonl(path: str) -> Tuple[List[Dict[str, Any]], str]:
    from main.reads.models import spec_path
    from utils.hashing import sha256_file

    pp = spec_path(path)
    if not pp.is_file():
        raise _err(f"artefact {path}: missing (it is made before the read is decided)")
    return [json.loads(x) for x in pp.read_text().splitlines() if x.strip()], sha256_file(pp)


def inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    from main.reads.models import spec_path
    from main.reads.rules import _group
    from main.value_search.live_branch import LiveBranchError, collect
    from utils.hashing import sha256_file

    units: List[Dict[str, Any]] = []
    groups = []
    for gname in dict.fromkeys([p["u_group"], p["home_group"]]):
        g = _group(cells, "anchor", gname)
        groups.append({"name": gname, "engine": g.get("engine"), "regime_id": g.get("regime_id"),
                       "teamsets": g.get("teamsets"), "unit_seeds": g.get("unit_seeds"), "unplayed": g.get("unplayed")})
        units += [{k: u.get(k) for k in _UNIT_KEYS} for u in g["units"] if u["label"] == p["label"]]
    sel, sel_sha = _jsonl(p["selected"])
    sc, sc_sha = _jsonl(p["scored"])
    bs = spec_path(p["branch_state"])
    try:
        rows = collect(bs)
    except (LiveBranchError, OSError, ValueError) as exc:
        raise _err(f"branch_state {p['branch_state']}: {exc}") from None
    arts = {"selected": {"path": p["selected"], "sha256": sel_sha}, "scored": {"path": p["scored"], "sha256": sc_sha}}
    for key, name in (("branch_units", "plan.json"), ("branch_plan", "branch_plan.jsonl")):
        arts[key] = {"path": str(bs / name), "sha256": sha256_file(bs / name)}
    return {"schema": SCHEMA, "label": p["label"], "raw_sha256": spec.models[p["label"]].ref.sha256,
            "engine_registered": p.get("engine"), "groups": groups, "units": units, "artefacts": arts,
            "selected": [{k: r.get(k) for k in _SEL_KEYS} for r in sel], "scored": sc, "branches": rows}


def void_reasons(doc: Mapping[str, Any], p: Mapping[str, Any]) -> List[str]:
    from main.value_search.live_branch import LiveBranchError, quota_prefix
    from main.value_search.sign_gate import select_hash

    why: List[str] = []
    counted = [u for u in doc["units"] if u["counted"]]
    engines = {u.get("engine") for u in counted}
    known = sorted(str(e) for e in engines if e)
    if None in engines or "" in engines:
        why.append(f"engine: {sum(1 for u in counted if not u.get('engine'))} counted raw unit(s) on an unknown engine")
    if len(known) > 1:
        why.append(f"engine: counted raw units span {len(known)} engines {known}")
    reg = doc.get("engine_registered")
    if reg and known and known != [reg]:
        why.append(f"engine: counted raw units played on {known}, not the registered {reg}")
    bad = [u["unit"] for u in counted if u.get("model_sha256") != doc["raw_sha256"]
           or u.get("our_regime") != RAW_REGIME or u.get("search") is not None]
    if bad:
        why.append(f"raw arm: {len(bad)} counted unit(s) on other bytes, a search stamp or a regime other than "
                   f"{RAW_REGIME!r} (first {bad[0]})")
    ok_cells = {u["unit"] for u in counted}
    stray = [r["key"] for r in doc["selected"] if r["cell"] not in ok_cells]
    if stray:
        why.append(f"selection: {len(stray)} state(s) from a cell that is not a counted cell (first {stray[0]})")
    wrong = [r["key"] for r in doc["selected"] if r["select_hash"] != select_hash(r["deal"], r["turn"], r["n"])]
    if wrong:
        why.append(f"selection: {len(wrong)} state(s) whose hash is not their draw's (first {wrong[0]})")
    per: Dict[str, int] = {}
    for r in doc["selected"]:
        per[r["deal"]] = per.get(r["deal"], 0) + 1
    if any(k > int(p["per_game"]) for k in per.values()):
        why.append(f"selection: a game with more than {p['per_game']} states")
    ck = [r["key"] for r in doc["scored"]]
    if len(set(ck)) != len(ck):
        why.append("scored: a state scored twice")
    if set(ck) - {r["key"] for r in doc["selected"]}:
        why.append("scored: a scored state never selected")
    try:
        quota_prefix(doc["selected"], doc["scored"], p["quota"])
    except LiveBranchError as exc:
        why.append(f"quota: {exc}")
    return why


def _params(p: Mapping[str, Any], doc: Mapping[str, Any]) -> Dict[str, Any]:
    from main.anchors.search_presets import parse

    return {"u_panels": list(p["u_panels"]), "home_panel": p["home_panel"], "bar": p["bar"],
            "rule8_eps": p["rule8_eps"], "bootstrap": dict(p["bootstrap"]), "max_drop_frac": p["max_drop_frac"],
            "engine": doc.get("engine_registered"), "preset": parse(p["search"])[0].name}


def decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    from main.value_search.live_branch import LiveBranchError, quota_prefix, read

    why = void_reasons(doc, p)
    if why:
        return {"schema": SCHEMA, "verdict": VOID, "void": why,
                "decision": {"class": VOID, "verdict": f"{VOID}: " + "; ".join(why)}}
    try:
        res = read(doc["branches"], doc["scored"], _params(p, doc))
    except LiveBranchError as exc:
        raise _err(str(exc)) from None
    if res["verdict"] == VOID:
        return {"schema": SCHEMA, "verdict": VOID, "void": res["void"],
                "decision": {"class": VOID, "verdict": f"{VOID}: " + "; ".join(res["void"])}}
    pr = res["primary"]
    line = (f"{res['verdict']}: pooled-U mean D (S′'s branch - raw) {pr['est']:+.4f} [{pr['lo']:+.4f}, "
            f"{pr['hi']:+.4f}] (n {pr['n']}; GO iff LB > {p['bar']}"
            + ("; LB within rule 8's eps: STOP" if res["boundary"] else "") + ")")
    return {"schema": SCHEMA, "verdict": res["verdict"], "boundary": res["boundary"], "rule": res["rule"],
            "primary": pr, "reported": res["reported"], "quota": quota_prefix(doc["selected"], doc["scored"],
                                                                                p["quota"]),
            "artefacts": doc["artefacts"], "groups": doc["groups"],
            "decision": {"class": res["verdict"], "verdict": line}}
