r"""V-RANK's registered rule ``v_rank`` (``designs/endstate/design_v_rank_on_u.md``; the driver and the teeth are
:mod:`main.value_search.v_rank`, the estimator :mod:`main.value_search.v_rank_stat`): does the search's value estimate
rank the moves of one position as well on unfamiliar teams (U) as on home72?

The raw games are ONE raw label (a pinned P*, ``load.input_log`` = the kept turn window, whole 20-game units against
Kakuna) in the ``u_groups`` (each plays exactly the ``u_panels``) and the ``home_groups`` (each exactly the
``home_panel``); a later look lists the earlier look's groups as imports beside its own. Three kinds of ARTEFACT are
made before the read is decided by ``python -m main.value_search.v_rank``, and read by path with their sha256
recorded: the ``selected`` files (the stratified draw), the ``scored`` DIRECTORIES (the shadow arms' part files: Q̂ of
every legal action with its per-world values, and the ``oracle_leaf8`` self-referee) and the ``branch_states`` (the
plan ROOTS; each holds its shards ``s<k>``: the forced one-game units, their cells and ``peer_drops.jsonl``).

THE ESTIMAND AND THE CLASSES: :func:`main.value_search.v_rank_stat.read`. β = Σ q_c·y_c / Σ q_c² within position (y
the REAL game's result after each legal action), the TEAMS we pilot bootstrapped within panel (the team-clustered
interval every Kakuna number carries); β_U the mean of the U panels', Δβ = β_U - β_home. ``RANK-NULL-ON-U`` iff
UB(β_U) < ``null_bar`` (it dominates); ``RANK-HOLDS-ON-U`` iff LB(Δβ) > -``margin`` and LB(β_U) > 0;
``RANK-WORSE-ON-U`` iff UB(Δβ) < 0 and not HOLDS; else ``UNDECIDED``. ``look`` 1 reads NULL only (anything else is
``CONTINUE``); a bound within ``rule8_eps`` of a bar that matters reads UNDECIDED.

VOID (named, before any estimate): the raw arm's counted units on two engines, an unknown one or another than the
registered ``engine``, on other bytes than the pinned sha256, carrying a search stamp, or a regime other than greedy; a
state SCORED on other bytes than the raw arm's (the scorer stamps its zip's sha256 on every row); a selected state
from a cell that is not a counted cell, or a counted raw unit with no selected state; and every tooth of
:func:`main.value_search.v_rank.void_reasons` (the draw, the preset, the shadow arms' records, the plan, G-BRANCH, the
capture prefix, the null controls, the forced action, the engine) and its drop cap.

PEER DROPS are registered from the start (the dlive read's AM-DL-PEER, :func:`rules_deviation_live.apply_peer_drops`):
every shard must hold its ``peer_drops.jsonl`` (``v_rank classify``), or the inputs refuse. A position with a branch
that never counted is left out WHOLE and counts against ``max_drop_frac`` of its panel's searched positions.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

SCHEMA = "gen3_read_v_rank_v1"
KEYS = ("label", "u_groups", "home_groups", "u_panels", "home_panel", "look", "null_bar", "margin", "bootstrap",
        "rule8_eps", "input_log", "search", "referee", "max_drop_frac", "selected", "scored", "branch_states")
OPTIONAL = ("engine",)
RAW_REGIME = "greedy"
VOID = "VOID"
RULE_DOCS = {"v_rank": "V-RANK: the within-position slope β of the real outcome (every legal action forced as a real "
                       "game vs Kakuna) on the search's Q̂, U against home72; NULL iff UB(β_U) < null_bar, HOLDS iff "
                       "LB(Δβ) > -margin and LB(β_U) > 0, WORSE iff UB(Δβ) < 0; look 1 reads NULL only"}
_UNIT_KEYS = ("unit", "label", "panel", "unit_seed", "counted", "our_regime", "search", "model_sha256", "engine")
_SEL_KEYS = ("cell", "panel", "deal", "team", "turn", "n", "key", "select_hash", "index", "unit_seed", "stratum",
             "n_legal", "pi_argmax")
#: what the record keeps of a scored state's search (the worlds' teams and chains stay in the scored files, by sha256)
_SEARCH_KEYS = ("searched", "why", "error", "played", "deviation", "q", "log_pi", "root_v", "spent", "config", "cpu_s",
                "species_hit")


def _err(msg: str) -> Exception:
    from main.reads.rules import RuleInputError

    return RuleInputError(msg)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _names(x: Any) -> bool:
    return isinstance(x, list) and bool(x) and all(isinstance(v, str) and v for v in x) and len(set(x)) == len(x)


def check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    from main.anchors.input_log import InputLogError, canonical
    from main.anchors.search_presets import SearchSpecError, canonical as search_canonical
    from main.reads.rules import _check_keys
    from main.value_search.v_rank import ORACLE, ORACLE_SEEDS

    probs = _check_keys(p, KEYS, KEYS + OPTIONAL)
    lab: Any = p.get("label")
    u: Any = p.get("u_panels")
    home: Any = p.get("home_panel")
    if not (isinstance(home, str) and _names(u) and home not in u):
        return probs + ["home_panel: a team set; u_panels: distinct team sets, home_panel not among them"]
    gu: Any = p.get("u_groups")
    gh: Any = p.get("home_groups")
    if not (_names(gu) and _names(gh) and not set(gu) & set(gh)):
        return probs + ["u_groups, home_groups: distinct anchor group names, no group in both"]
    for gname, want in [(g, set(u)) for g in gu] + [(g, {home}) for g in gh]:
        g = anchor.get(gname)
        if g is None:
            probs.append(f"group {gname!r} is not an anchor group")
            continue
        if list(g.models) != [lab]:
            probs.append(f"group {g.name!r} must play exactly {lab!r}, plays {list(g.models)}")
        if g.per_game:
            probs.append(f"group {g.name!r}: the raw games are whole 20-game units (not per_game)")
        if set(g.teamsets) != want:
            probs.append(f"group {g.name!r} must play exactly {sorted(want)}, plays {list(g.teamsets)}")
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
    if p.get("look") not in (1, 2) or isinstance(p.get("look"), bool):
        probs.append("look: 1 or 2")
    for k in ("null_bar", "margin"):
        if not (_num(p.get(k)) and 0 < p[k] < 1):
            probs.append(f"{k}: a number in (0, 1)")
    bs = p.get("bootstrap")
    if not (isinstance(bs, dict) and set(bs) == {"B", "seed"} and _int(bs.get("B")) and bs["B"] >= 1000
            and _int(bs.get("seed"))):
        probs.append("bootstrap: {B: int >= 1000, seed: int}")
    eps: Any = p.get("rule8_eps")
    if not (_num(eps) and 0 <= eps < 0.01):
        probs.append("rule8_eps: a number in [0, 0.01)")
    try:
        if search_canonical(p.get("search")) != p.get("search"):
            probs.append(f"search: spell it canonically ({search_canonical(p.get('search'))!r})")
    except SearchSpecError as exc:
        probs.append(f"search: {exc}")
    if p.get("referee") != {"preset": ORACLE, "seeds": ORACLE_SEEDS}:
        probs.append(f"referee: {{'preset': {ORACLE!r}, 'seeds': {ORACLE_SEEDS}}} (the self-referee the scorer runs)")
    if not (_num(p.get("max_drop_frac")) and 0 <= p["max_drop_frac"] <= 0.5):
        probs.append("max_drop_frac: a number in [0, 0.5]")
    for k in ("selected", "scored", "branch_states"):
        if not _names(p.get(k)):
            probs.append(f"{k}: the artefacts' paths, a non-empty list of distinct strings")
    if "engine" in p and not (isinstance(p["engine"], str) and p["engine"]):
        probs.append("engine: the registered engine stamp (port=..;env=..)")
    return probs


def inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    from main.reads.models import spec_path
    from main.reads.rules import _group
    from main.value_search import v_rank as V
    from utils.hashing import sha256_file

    units: List[Dict[str, Any]] = []
    groups = []
    for gname in [*p["u_groups"], *p["home_groups"]]:
        g = _group(cells, "anchor", gname)
        groups.append({"name": gname, "engine": g.get("engine"), "regime_id": g.get("regime_id"),
                       "teamsets": g.get("teamsets"), "unit_seeds": g.get("unit_seeds"), "unplayed": g.get("unplayed")})
        units += [{k: u.get(k) for k in _UNIT_KEYS} for u in g["units"] if u["label"] == p["label"]]
    arts: Dict[str, Any] = {"selected": {}, "scored": {}, "branch_states": {}}
    sel: List[Dict[str, Any]] = []
    for path in p["selected"]:
        pp = spec_path(path)
        if not pp.is_file():
            raise _err(f"artefact {path}: missing (it is made before the read is decided)")
        sel += [json.loads(x) for x in pp.read_text().splitlines() if x.strip()]
        arts["selected"][path] = sha256_file(pp)
    files = []
    for path in p["scored"]:
        got = V.scored_files(spec_path(path))
        if not got:
            raise _err(f"artefact {path}: no scored part file (it is made before the read is decided)")
        files += got
        arts["scored"][path] = {f.name: sha256_file(f) for f in got}
    try:
        scored = V.load_scored(files)
        shards = [s for path in p["branch_states"] for s in V.shard_states(spec_path(path))]
        rows, peer, sha = V.load_branches(shards, need_peer=True)
    except V.VRankError as exc:
        raise _err(str(exc)) from None
    arts["branch_states"] = sha
    slim = [{**{k: v for k, v in r.items() if k != "search"},
             "search": {k: r["search"][k] for k in _SEARCH_KEYS if k in r["search"]}} for r in scored]
    return {"schema": SCHEMA, "label": p["label"], "raw_sha256": spec.models[p["label"]].ref.sha256,
            "engine_registered": p.get("engine"), "groups": groups, "units": units, "artefacts": arts,
            "selected": [{k: r.get(k) for k in _SEL_KEYS} for r in sel], "scored": slim, "branches": rows,
            "peer_drops": peer}


def void_reasons(doc: Mapping[str, Any]) -> List[str]:
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
    other = [r["key"] for r in doc["scored"] if r.get("model_sha256") != doc["raw_sha256"]]
    if other:
        why.append(f"shadow arms: {len(other)} state(s) scored on other bytes than the raw arm's (first {other[0]})")
    ok_cells = {u["unit"] for u in counted}
    stray = [r["key"] for r in doc["selected"] if r["cell"] not in ok_cells]
    if stray:
        why.append(f"selection: {len(stray)} state(s) from a cell that is not a counted cell (first {stray[0]})")
    bare = sorted(ok_cells - {r["cell"] for r in doc["selected"]})
    if bare:
        why.append(f"selection: {len(bare)} counted raw unit(s) with no selected state (first {bare[0]})")
    return why


def _params(p: Mapping[str, Any], doc: Mapping[str, Any]) -> Dict[str, Any]:
    from main.anchors.search_presets import parse

    return {"u_panels": list(p["u_panels"]), "home_panel": p["home_panel"], "look": p["look"],
            "null_bar": p["null_bar"], "margin": p["margin"], "rule8_eps": p["rule8_eps"],
            "bootstrap": dict(p["bootstrap"]), "max_drop_frac": p["max_drop_frac"],
            "engine": doc.get("engine_registered"), "preset": parse(p["search"])[0].name}


def _void(why: List[str], doc: Mapping[str, Any], **more: Any) -> Dict[str, Any]:
    return {"schema": SCHEMA, "verdict": VOID, "void": why, "peer_drops": doc.get("peer_drops"), **more,
            "decision": {"class": VOID, "verdict": f"{VOID}: " + "; ".join(why)}}


def decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    from main.value_search import v_rank as V

    why = void_reasons(doc)
    if why:
        return _void(why, doc)
    try:
        res = V.read(doc["selected"], doc["scored"], doc["branches"], _params(p, doc))
    except V.VRankError as exc:
        raise _err(str(exc)) from None
    if res["verdict"] == VOID:
        return _void(res["void"], doc, **({"counts": res["counts"]} if "counts" in res else {}))
    pr = res["primary"]

    def tri(x: Mapping[str, float]) -> str:
        return f"{x['est']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"

    n = res["positions"]
    line = (f"{res['verdict']} (look {res['look']}): β_U {tri(pr['beta_u'])}, β_home {tri(pr['beta_home'])}, Δβ "
            f"{tri(pr['delta'])}; positions U {sum(n[k] for k in p['u_panels'])} / home {n[p['home_panel']]}"
            + (f"; Δβ's own class {res['delta_class']}" if res["verdict"] == "RANK-NULL-ON-U" else "")
            + (f"; {res['note']}" if res.get("note") else "")
            + ("; a bound within rule 8's eps of a bar" if res["boundary"] else ""))
    units = [u for u in doc["units"] if u["counted"]]
    cover = {k: {"raw_units_counted": sum(1 for u in units if u["panel"] == k)} for k in [*p["u_panels"],
                                                                                       p["home_panel"]]}
    return {"schema": SCHEMA, "verdict": res["verdict"], "look": res["look"], "boundary": res["boundary"],
            "delta_class": res["delta_class"], "note": res["note"], "rule": res["rule"], "primary": pr,
            "positions": n, "counts": res["counts"], "extras": res["extras"], "reported": res["reported"],
            "coverage": cover, "artefacts": doc["artefacts"], "groups": doc["groups"],
            "peer_drops": doc.get("peer_drops"), "decision": {"class": res["verdict"], "verdict": line}}
