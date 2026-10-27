r"""The S-U gate test's registered rule ``deviation_sign`` (``measurements/su_gate_2026-10-27/``; the S-U
review's "gate test", ``su_headroom_2026-10-27/REVIEW.md``).

ONE anchor group plays ONE raw label (a pinned P*, ``load.input_log`` = the kept turn window) against Kakuna on
``home_panel`` and the ``u_panels``; its counted cells kept the honest input logs of our decisions in the window. Two
ARTEFACTS, made before the read is decided by :mod:`main.value_search.sign_gate`, are read by path with their sha256
recorded: ``selected`` (the hashed draw of states, ``sign_gate select``) and ``scored`` (S and the self-referee on
each, ``sign_gate score``).

THE ESTIMAND AND THE CLASS: :func:`main.value_search.sign_gate.read` (sign accuracy of S's deviations under the
self-referee, pooled U equal-weighted, a game-clustered bootstrap): GO iff its LB > ``bar`` (a bound within
``rule8_eps`` of the bar reads STOP and is flagged), else STOP.

VOID (named, before any estimate): counted cells on two engines, an unknown one, or another than the registered
``engine``; a counted cell on other bytes than the label's pinned sha256, carrying a search stamp, or a regime other
than greedy; a selected state from a cell that is not a counted cell of the group, or whose hash is not its draw's;
more than ``per_game`` states of one game; a selected state not scored, scored twice, or a scored state never selected.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Tuple

SCHEMA = "gen3_read_deviation_sign_v1"
KEYS = ("group", "label", "home_panel", "u_panels", "bar", "bootstrap", "rule8_eps", "input_log", "per_game",
        "selected", "scored")
OPTIONAL = ("engine",)
RAW_REGIME = "greedy"
VOID = "VOID"
RULE_DOCS = {"deviation_sign": "the S-U gate test: sign accuracy of S's deviations from pi under the self-referee on "
                               "raw P* vs Kakuna states, pooled U; GO iff its LB > bar, else STOP"}


def _err(msg: str) -> Exception:
    from main.reads.rules import RuleInputError

    return RuleInputError(msg)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    from main.anchors.input_log import InputLogError, canonical
    from main.reads.rules import _check_keys

    probs = _check_keys(p, KEYS, KEYS + OPTIONAL)
    g = anchor.get(p.get("group"))
    if g is None:
        return probs + [f"group {p.get('group')!r} is not an anchor group"]
    lab: Any = p.get("label")
    if list(g.models) != [lab]:
        probs.append(f"group {g.name!r} must play exactly {lab!r}, plays {list(g.models)}")
    elif lab in models:
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
    if g.per_game:
        probs.append(f"group {g.name!r}: the gate plays whole 20-game units (not per_game)")
    home, u = p.get("home_panel"), p.get("u_panels")
    if not (isinstance(home, str) and isinstance(u, list) and u and all(isinstance(x, str) for x in u)
            and len(set(u)) == len(u) and home not in u):
        probs.append("home_panel: a team set; u_panels: distinct team sets, home_panel not among them")
    elif set(g.teamsets) != {home, *u}:
        probs.append(f"the panels {[home, *u]} must be exactly group {g.name!r}'s team sets {list(g.teamsets)}")
    if not (_num(p.get("bar")) and 0 < p["bar"] < 1):
        probs.append("bar: a number in (0, 1)")
    bs = p.get("bootstrap")
    if not (isinstance(bs, dict) and set(bs) == {"B", "seed"} and isinstance(bs.get("B"), int)
            and not isinstance(bs.get("B"), bool) and bs["B"] >= 1000 and isinstance(bs.get("seed"), int)
            and not isinstance(bs.get("seed"), bool)):
        probs.append("bootstrap: {B: int >= 1000, seed: int}")
    eps: Any = p.get("rule8_eps")
    if not (_num(eps) and 0 <= eps < 0.01):
        probs.append("rule8_eps: a number in [0, 0.01)")
    if not (isinstance(p.get("per_game"), int) and not isinstance(p.get("per_game"), bool) and p["per_game"] >= 1):
        probs.append("per_game: an int >= 1")
    for k in ("selected", "scored"):
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


_UNIT_KEYS = ("unit", "label", "panel", "unit_seed", "counted", "our_regime", "search", "model_sha256", "engine")


def inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    from main.reads.rules import _group

    g = _group(cells, "anchor", p["group"])
    sel, sel_sha = _jsonl(p["selected"])
    sc, sc_sha = _jsonl(p["scored"])
    return {"schema": SCHEMA, "label": p["label"], "raw_sha256": spec.models[p["label"]].ref.sha256,
            "engine_registered": p.get("engine"),
            "group": {"name": p["group"], "engine": g.get("engine"), "regime_id": g.get("regime_id"),
                      "teamsets": g.get("teamsets"), "unit_seeds": g.get("unit_seeds"), "unplayed": g.get("unplayed")},
            "units": [{k: u.get(k) for k in _UNIT_KEYS} for u in g["units"] if u["label"] == p["label"]],
            "artefacts": {"selected": {"path": p["selected"], "sha256": sel_sha},
                          "scored": {"path": p["scored"], "sha256": sc_sha}},
            "selected": [{k: r.get(k) for k in ("cell", "panel", "deal", "turn", "n", "key", "select_hash")}
                         for r in sel],
            "scored": sc}


def void_reasons(doc: Mapping[str, Any], p: Mapping[str, Any]) -> List[str]:
    from main.value_search.sign_gate import select_hash

    why: List[str] = []
    counted = [u for u in doc["units"] if u["counted"]]
    engines = {u.get("engine") for u in counted}
    known = sorted(str(e) for e in engines if e)
    if None in engines or "" in engines:
        why.append(f"engine: {sum(1 for u in counted if not u.get('engine'))} counted unit(s) on an unknown engine")
    if len(known) > 1:
        why.append(f"engine: counted units span {len(known)} engines {known}")
    reg = doc.get("engine_registered")
    if reg and known and known != [reg]:
        why.append(f"engine: counted units played on {known}, not the registered {reg}")
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
    over = [d for d, k in per.items() if k > int(p["per_game"])]
    if over:
        why.append(f"selection: {len(over)} game(s) with more than {p['per_game']} states")
    sk = [r["key"] for r in doc["selected"]]
    ck = [r["key"] for r in doc["scored"]]
    if len(set(ck)) != len(ck):
        why.append("scored: a state scored twice")
    if set(sk) != set(ck):
        why.append(f"scored: {len(set(sk) - set(ck))} selected state(s) not scored, {len(set(ck) - set(sk))} scored "
                   "state(s) never selected")
    return why


def decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    from main.value_search.sign_gate import read

    why = void_reasons(doc, p)
    if why:
        return {"schema": SCHEMA, "verdict": VOID, "void": why}
    res = read(doc["scored"], {"home_panel": p["home_panel"], "u_panels": list(p["u_panels"]), "bar": p["bar"],
                               "rule8_eps": p["rule8_eps"], "bootstrap": dict(p["bootstrap"]),
                               "expected_states": len(doc["selected"])})
    return {"schema": SCHEMA, "verdict": res["verdict"], "boundary": res["boundary"], "rule": res["rule"],
            "primary": res["primary"], "reported": res["reported"], "n_states": res["n_states"],
            "artefacts": doc["artefacts"], "engine": doc["group"]["engine"]}
