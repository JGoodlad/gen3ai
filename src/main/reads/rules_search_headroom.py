r"""S-U's registered rule ``search_headroom`` (``designs/endstate/design_search_u_headroom.md`` §3, §4.4; build B4).

ONE ``per_game`` anchor group plays a SEARCHED label (``load.search``, e.g. ``r0s4@10``) and its RAW TWIN (the same
pinned bytes, no search) on home72 and the U panels. Each DEAL (panel, unit seed, game index) is played by both arms;
it is a PAIR only when BOTH arms' one-game units COUNTED (``main.anchors.results.unit_counts``, so a game the
non-battle-termination guard voided is simply not a pair). A deal where only one side counted is EXCLUDED and counted
in the document, never imputed.

THE ESTIMANDS (§3). A game scores win 1, tie ½, loss 0. A UNIT is one (panel, unit seed); its statistic is the mean
paired difference (searched − raw) over that seed's pairs. G_panel = the mean of the panel's unit statistics;
G_home72 = G of ``home_panel``; G_U = the equal-weight mean of the ``u_panels``' G; I = G_U − G_home72. Point
estimates are exact fractions.

THE BOOTSTRAP (reproducible exactly): ``rng = numpy.random.default_rng(seed)``; for each panel in the order
``u_panels`` (as registered) then ``home_panel``, its unit statistics in unit-seed order as a float vector ``x`` of
length n, ``idx = rng.integers(0, n, size=(B, n))`` and the panel's B replicates are ``x[idx].mean(axis=1)``. So the
units are resampled WITHIN each panel, and the U panels' draws come first: G_U's replicates do not depend on how many
home72 units there are (phase 2 adds home72 seeds and leaves G_U's interval as phase 1 read it). Replicate b of G_U is
the mean of the U panels' replicate b; of I, G_U's minus home72's. Each interval is ``numpy.percentile(reps, [2.5,
97.5])`` (linear interpolation; H-MID's scheme).

THE CLASSES (§3), every bar compared as an EXACT fraction (the bound's float exactly, the bar's decimal exactly), and
a bound within ``rule8_eps`` of its bar is ``near``: never a crossing (root rule 8), so that class reads UNDECIDED.

* Primary A (I): U-HEADROOM iff LB(I) > 0 and LB(G_U) > 0; U-LESS iff UB(I) < 0; NO-U-PREMIUM iff UB(I) <
  ``no_u_premium_pp`` (it may co-occur with either); else UNDECIDED. At ``phase`` 1 I is read FUTILITY-ONLY (§4.4):
  only U-LESS / NO-U-PREMIUM can be read, and reaching one STOPS the read; otherwise (a would-be U-HEADROOM included)
  A reads UNDECIDED and the read proceeds to phase 2. At phase 2 every class of A is read.
* Primary B (G_U, LEAK-INCLUSIVE): DOES-NOT-CLOSE-B iff UB(G_U) < ``dnc_b_pp`` (creditable); U-GAIN (leaky) iff
  LB(G_U) > 0; GOAL-B-REACHED (leaky) iff the searched arm's pooled U score over its paired games (a tie a non-win,
  ``main.x35.d1.strat_score``'s stratified Wilson) has LB > ``goal_b``, reported, never "goal (b) met"; else
  UNDECIDED. G_U's classes are final at phase 1.

VOID (named reasons, before any estimate): G-PREFIX not run, or any deal it found diverging; fewer pairs than
G-PREFIX checked (or more); counted units on two engines, on an unknown engine, or on another than the registered
``engine``; any counted unit, or a resolved model, on other bytes than the raw label's pinned sha256; a counted
searched unit whose ``our_regime``, planned search, or played search record (name and preset config) is not the
registered ``expected_search``; a counted raw unit carrying any search stamp; a counted unit whose ``games.jsonl``
result disagrees with its summary; a panel with fewer than 2 units holding a pair.

REPORTED, never decided: G per panel, I per U panel (forum72's is the F secondary), the pooled discordance (the share
of paired deals whose outcomes differ, b + c only: its sign b − c is never shown), the counts, the deviation rate and
search CPU-s per searched decision by set (from each searched unit's search record). The design's level-at-10
subgroup, world-quality covariate, raw-arm-vs-X51 guard and H-MID Δbias re-read need data the cells do not carry
(the captures, ``search_decisions.jsonl``, X51's rows): they are listed in ``not_reported``.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "gen3_read_search_headroom_v1"
KEYS = ("group", "searched", "raw", "home_panel", "u_panels", "phase", "bars", "bootstrap", "rule8_eps",
        "expected_search")
OPTIONAL = ("engine",)
BAR_KEYS = ("no_u_premium_pp", "dnc_b_pp", "goal_b")
RAW_REGIME = "greedy"

U_HEADROOM, U_LESS, NO_U_PREMIUM, UNDECIDED = "U-HEADROOM", "U-LESS", "NO-U-PREMIUM", "UNDECIDED"
DNC_B, U_GAIN, GOAL_B = "DOES-NOT-CLOSE-B", "U-GAIN (leaky)", "GOAL-B-REACHED (leaky)"
VOID = "VOID"
NOT_REPORTED = {
    "level_at_10_subgroup": "needs each pair's turn-10 state (the captures), which the cells do not carry",
    "world_quality_by_set": "needs search_decisions.jsonl's per-decision world records",
    "raw_arm_vs_x51": "needs X51's banked rows (a cross-engine guard, REPORTED outside the rule)",
    "h_mid_delta_bias": "needs the raw arm's captures re-read by H-MID's script",
}
RULE_DOCS = {"search_headroom": "S-U: searched − raw P* paired by deal on home72 and U; I = G_U − G_home72 and G_U "
                                "by a within-panel unit bootstrap, exact-fraction bars; phase-1 futility on I; VOID "
                                "on G-PREFIX, engines, bytes or the search config"}


def _err(msg: str) -> Exception:
    from main.reads.rules import RuleInputError

    return RuleInputError(msg)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


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
    srch, raw = p.get("searched"), p.get("raw")
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
    home, u = p.get("home_panel"), p.get("u_panels")
    if not (isinstance(home, str) and isinstance(u, list) and u and all(isinstance(x, str) for x in u)
            and len(set(u)) == len(u) and home not in u):
        probs.append("home_panel: a team set; u_panels: distinct team sets, home_panel not among them")
    elif set(g.teamsets) != {home, *u}:
        probs.append(f"the panels {[home, *u]} must be exactly group {g.name!r}'s team sets {list(g.teamsets)}")
    if p.get("phase") not in (1, 2) or isinstance(p.get("phase"), bool):
        probs.append("phase: 1 (G_U final, I futility-only) or 2 (every class of I)")
    b = p.get("bars")
    if not (isinstance(b, dict) and set(b) == set(BAR_KEYS) and all(_num(b[k]) for k in BAR_KEYS)
            and b["no_u_premium_pp"] > 0 and b["dnc_b_pp"] > 0 and 0 < b["goal_b"] < 1):
        probs.append(f"bars: {{{', '.join(BAR_KEYS)}}}: pp > 0, pp > 0, 0 < goal_b < 1")
    bs = p.get("bootstrap")
    if not (isinstance(bs, dict) and set(bs) == {"B", "seed"} and isinstance(bs.get("B"), int)
            and not isinstance(bs.get("B"), bool) and bs["B"] >= 1000 and isinstance(bs.get("seed"), int)
            and not isinstance(bs.get("seed"), bool)):
        probs.append("bootstrap: {B: int >= 1000, seed: int}")
    eps = p.get("rule8_eps")
    if not (_num(eps) and 0 <= eps < 0.01):
        probs.append("rule8_eps: a number in [0, 0.01) (a bound this close to a bar is never a crossing)")
    if "engine" in p and not (isinstance(p["engine"], str) and p["engine"]):
        probs.append("engine: the registered engine stamp (port=..;env=..)")
    return probs


# ------------------------------------------------------------------------------------------------ inputs
_UNIT_KEYS = ("unit", "label", "panel", "unit_seed", "game", "counted", "wins", "ties", "n", "our_regime", "search",
              "search_played", "model_sha256", "engine", "games")


def inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    from main.anchors.search_presets import parse
    from main.reads.rules import _group

    g = _group(cells, "anchor", p["group"])
    preset, from_turn = parse(p["expected_search"])
    labels = (p["searched"], p["raw"])
    models = cells.get("models") or {}
    return {
        "schema": SCHEMA, "searched": p["searched"], "raw": p["raw"], "home_panel": p["home_panel"],
        "u_panels": list(p["u_panels"]), "phase": p["phase"], "expected_search": p["expected_search"],
        "expected_regime": f"{RAW_REGIME}+search={p['expected_search']}", "expected_preset": preset.as_dict(),
        "from_turn": from_turn, "raw_sha256": spec.models[p["raw"]].ref.sha256,
        "resolved_sha256": {lab: (models.get(lab) or {}).get("sha256") for lab in labels},
        "engine_registered": p.get("engine"),
        "group": {"name": p["group"], "engine": g.get("engine"), "regime_id": g.get("regime_id"),
                  "teamsets": g.get("teamsets"), "unit_seeds": g.get("unit_seeds"), "unplayed": g.get("unplayed"),
                  "inconclusive": list(g.get("inconclusive") or []), "g_prefix": g.get("g_prefix")},
        "units": [{k: u.get(k) for k in _UNIT_KEYS} for u in g["units"] if u["label"] in labels]}


# ------------------------------------------------------------------------------------------------ pairing
def _score(u: Mapping[str, Any]) -> Fraction:
    n, w, t = int(u["n"]), int(u["wins"]), int(u.get("ties") or 0)
    if n != 1:
        raise _err(f"unit {u['unit']}: n = {n}, not a one-game (per_game) unit")
    return Fraction(2 * w + t, 2)


def pair_deals(doc: Mapping[str, Any]) -> Tuple[Dict[str, Dict[int, List[Tuple[Fraction, Fraction]]]],
                                               Dict[str, Dict[str, int]]]:
    """``(pairs, excluded)``: ``pairs[panel][unit seed]`` = ``[(raw, searched), ...]`` in game order over the deals
    both arms COUNTED; ``excluded[panel]`` = ``{searched_only, raw_only}`` deals (one side counted: not a pair)."""
    side = {doc["searched"]: "s", doc["raw"]: "r"}
    idx: Dict[Tuple[str, str, int, int], Mapping[str, Any]] = {}
    for u in doc["units"]:
        if not u["counted"]:
            continue
        k = (side[u["label"]], str(u["panel"]), int(u["unit_seed"]), int(u["game"]))
        if k in idx:
            raise _err(f"deal {k[1:]} counted twice for {u['label']!r}")
        idx[k] = u
    panels = [doc["home_panel"], *doc["u_panels"]]
    pairs: Dict[str, Dict[int, List[Tuple[Fraction, Fraction]]]] = {ts: {} for ts in panels}
    excl = {ts: {"searched_only": 0, "raw_only": 0} for ts in panels}
    for ts, seed, game in sorted({k[1:] for k in idx}):
        s, r = idx.get(("s", ts, seed, game)), idx.get(("r", ts, seed, game))
        if s is not None and r is not None:
            pairs[ts].setdefault(seed, []).append((_score(r), _score(s)))
        else:
            excl[ts]["searched_only" if s is not None else "raw_only"] += 1
    return pairs, excl


def _games_disagree(u: Mapping[str, Any]) -> bool:
    gs = u.get("games")
    if not gs:
        return False
    want = "win" if int(u["wins"]) else "tie" if int(u.get("ties") or 0) else "loss"
    return len(gs) != 1 or str(gs[0][3]) != want


def void_reasons(doc: Mapping[str, Any], pairs: Mapping[str, Mapping[int, Sequence[Any]]]) -> List[str]:
    """Every named VOID reason of the document (empty = the read may be decided)."""
    why: List[str] = []
    gp = doc["group"].get("g_prefix")
    if not gp:
        why.append("G-PREFIX: not run on the group")
    else:
        why += [f"G-PREFIX: deal {m['deal']} diverged before the search's first turn ({m['why']})"
                for m in gp.get("mismatches") or ()]
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
    n_pairs = sum(len(v) for by in pairs.values() for v in by.values())
    if gp and int(gp.get("checked", -1)) != n_pairs:
        why.append(f"G-PREFIX checked {gp.get('checked')} pair(s), the rule pairs {n_pairs}: every pair must be checked")
    for ts, by in pairs.items():
        if len(by) < 2:
            why.append(f"{ts}: {len(by)} unit(s) hold a pair (the unit bootstrap needs >= 2)")
    return why


# ------------------------------------------------------------------------------------------------ estimates
def unit_stats(pairs: Mapping[str, Mapping[int, Sequence[Tuple[Fraction, Fraction]]]]) -> Dict[str, Dict[int, Fraction]]:
    """``{panel: {unit seed: mean (searched − raw) over its pairs}}`` (exact)."""
    return {ts: {seed: sum((s - r for r, s in v), Fraction(0)) / len(v) for seed, v in sorted(by.items())}
            for ts, by in pairs.items()}


def bootstrap(stats: Mapping[str, Mapping[int, Fraction]], home: str, u_panels: Sequence[str], B: int,
              seed: int) -> Dict[str, Any]:
    """Percentile 95 % intervals over units resampled within each panel (the module docstring's exact scheme):
    ``{"G": {panel: [lo, hi]}, "G_U": [..], "I": [..], "I_by_panel": {u panel: [..]}}``."""
    import numpy as np

    rng = np.random.default_rng(seed)
    reps: Dict[str, Any] = {}
    for ts in [*u_panels, home]:
        x = np.array([float(v) for _s, v in sorted(stats[ts].items())])
        idx = rng.integers(0, len(x), size=(B, len(x)))
        reps[ts] = x[idx].mean(axis=1)
    gu = np.stack([reps[ts] for ts in u_panels]).mean(axis=0)

    def ci(r: Any) -> List[float]:
        lo, hi = np.percentile(r, [2.5, 97.5])
        return [float(lo), float(hi)]

    return {"G": {ts: ci(reps[ts]) for ts in [home, *u_panels]}, "G_U": ci(gu), "I": ci(gu - reps[home]),
            "I_by_panel": {ts: ci(reps[ts] - reps[home]) for ts in u_panels}}


def cmp(stat: float, bar: Any, eps: Any, *, above: bool) -> Tuple[bool, bool]:
    """``(crosses, near)`` by EXACT fractions: ``stat`` (a float, taken exactly) against ``bar`` (its decimal);
    within ``eps`` of the bar is ``near`` and never a crossing (root rule 8)."""
    d = Fraction(stat) - Fraction(str(bar))
    if abs(d) <= Fraction(str(eps)):
        return False, True
    return (d > 0 if above else d < 0), False


def classify(ci: Mapping[str, Sequence[float]], goal_lb: Optional[float], bars: Mapping[str, Any], eps: Any,
             phase: int) -> Dict[str, Any]:
    """Primary A on I and Primary B on G_U from the bounds (proportions; the bars' pp / 100), with the phase rule."""
    prem, dnc = Fraction(str(bars["no_u_premium_pp"])) / 100, Fraction(str(bars["dnc_b_pp"])) / 100
    (i_lo, i_hi), (g_lo, g_hi) = ci["I"], ci["G_U"]
    t = {"LB(I) > 0": cmp(i_lo, 0, eps, above=True), "LB(G_U) > 0": cmp(g_lo, 0, eps, above=True),
         "UB(I) < 0": cmp(i_hi, 0, eps, above=False),
         "UB(I) < no_u_premium": cmp(i_hi, prem, eps, above=False),
         "UB(G_U) < dnc_b": cmp(g_hi, dnc, eps, above=False)}
    if goal_lb is not None:
        t["LB(score_U) > goal_b"] = cmp(goal_lb, bars["goal_b"], eps, above=True)
    held = [c for c, ok in ((U_HEADROOM, t["LB(I) > 0"][0] and t["LB(G_U) > 0"][0]), (U_LESS, t["UB(I) < 0"][0]),
                            (NO_U_PREMIUM, t["UB(I) < no_u_premium"][0])) if ok]
    a: Dict[str, Any] = {"phase": phase, "futility_only": phase == 1}
    if phase == 1:
        read = [c for c in held if c in (U_LESS, NO_U_PREMIUM)]
        a.update(classes=read or [UNDECIDED], not_read_at_phase_1=[c for c in held if c not in read],
                 stop=bool(read), proceed_to_phase_2=not read)
    else:
        a["classes"] = held or [UNDECIDED]
    b = [c for c, ok in ((DNC_B, t["UB(G_U) < dnc_b"][0]), (U_GAIN, t["LB(G_U) > 0"][0]),
                         (GOAL_B, t.get("LB(score_U) > goal_b", (False, False))[0])) if ok]
    return {"primary_a": a, "primary_b": {"classes": b or [UNDECIDED], "leak_inclusive": True},
            "tests": {k: {"crosses": v[0], "near": v[1]} for k, v in t.items()}}


def _goal_b(doc: Mapping[str, Any], pairs: Mapping[str, Mapping[int, Sequence[Tuple[Fraction, Fraction]]]]
            ) -> Dict[str, Any]:
    """Each arm's U score over the PAIRED games (a tie a non-win, X51's rule), ``strat_score``; Newcombe-of-strata of
    searched − raw, descriptive."""
    from main.reads.rules import newcombe_strat, strat_score
    from main.x35.readout import ReadoutInputError

    u = list(doc["u_panels"])
    out: Dict[str, Any] = {}
    scores = {}
    for arm, i in (("searched", 1), ("raw", 0)):
        counts = {ts: (sum(1 for v in pairs[ts].values() for p in v if p[i] == 1),
                       sum(len(v) for v in pairs[ts].values())) for ts in u}
        try:
            sc = strat_score(counts, u)
            scores[arm] = sc
            out[arm] = {"counts": {k: list(v) for k, v in counts.items()}, "score": sc.mean, "ci95": list(sc.ci)}
        except ReadoutInputError as exc:
            out[arm] = {"counts": {k: list(v) for k, v in counts.items()}, "score": None, "ci95": None,
                        "why": str(exc)}
    if len(scores) == 2:
        d, lo, hi = newcombe_strat(scores["searched"], scores["raw"])
        out["newcombe_strat_searched_minus_raw"] = [d, lo, hi]
    return out


def _search_report(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """The deviation rate and search CPU-s per searched decision, home72 and U, over the PAIRED searched games."""
    def deal(u: Mapping[str, Any]) -> Tuple[str, int, int]:
        return str(u["panel"]), int(u["unit_seed"]), int(u["game"])

    raw = {deal(u) for u in doc["units"] if u["label"] == doc["raw"] and u["counted"]}
    sets = {doc["home_panel"]: "home", **{ts: "U" for ts in doc["u_panels"]}}
    acc = {k: {"searched": 0, "deviations": 0, "search_cpu_s": 0.0, "games": 0} for k in ("home", "U")}
    for u in doc["units"]:
        sp = u.get("search_played") or {}
        if u["label"] != doc["searched"] or not u["counted"] or deal(u) not in raw:
            continue
        a = acc[sets[u["panel"]]]
        a["games"] += 1
        a["searched"] += int(sp.get("searched") or 0)
        a["deviations"] += int(sp.get("deviations") or 0)
        a["search_cpu_s"] += float(sp.get("search_cpu_s") or 0.0)
    for a in acc.values():
        n = a["searched"]
        a.update(deviation_rate=a["deviations"] / n if n else None,
                 cpu_s_per_searched_decision=a["search_cpu_s"] / n if n else None,
                 search_cpu_s_per_game=a["search_cpu_s"] / a["games"] if a["games"] else None)
    return acc


def decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    pairs, excl = pair_deals(doc)
    home, u = doc["home_panel"], list(doc["u_panels"])
    flat = [pr for by in pairs.values() for v in by.values() for pr in v]
    disc = sum(1 for r, s in flat if r != s)
    counts = {"pairs": {ts: sum(len(v) for v in pairs[ts].values()) for ts in pairs},
              "units_with_pairs": {ts: len(pairs[ts]) for ts in pairs}, "excluded_one_side": excl,
              "uncounted_units": {lab: sum(1 for x in doc["units"] if x["label"] == lab and not x["counted"])
                                  for lab in (doc["searched"], doc["raw"])},
              "unplayed_units": doc["group"].get("unplayed")}
    out: Dict[str, Any] = {"counts": counts, "engine": doc["group"].get("engine"), "phase": doc["phase"],
                           "discordance": {"pairs": len(flat), "discordant": disc,
                                           "share": disc / len(flat) if flat else None},
                           "not_reported": dict(NOT_REPORTED)}
    reasons = void_reasons(doc, pairs)
    if reasons:
        return dict(out, decision={"verdict": VOID, "reasons": reasons})
    st = unit_stats(pairs)
    g = {ts: sum(st[ts].values(), Fraction(0)) / len(st[ts]) for ts in st}
    gu = sum((g[ts] for ts in u), Fraction(0)) / len(u)
    i = gu - g[home]
    b = p["bootstrap"]
    ci = bootstrap(st, home, u, int(b["B"]), int(b["seed"]))
    goal = _goal_b(doc, pairs)
    goal_lb = (goal["searched"]["ci95"] or [None])[0]
    cls = classify(ci, goal_lb, p["bars"], p["rule8_eps"], int(doc["phase"]))
    out["estimates"] = {
        "G": {ts: {"point": float(g[ts]), "ci95": ci["G"][ts]} for ts in [home, *u]},
        "G_U": {"point": float(gu), "ci95": ci["G_U"]}, "I": {"point": float(i), "ci95": ci["I"]},
        "I_by_panel": {ts: {"point": float(g[ts] - g[home]), "ci95": ci["I_by_panel"][ts]} for ts in u},
        "exact": {"G": {ts: str(g[ts]) for ts in g}, "G_U": str(gu), "I": str(i)},
        "bootstrap": {"B": int(b["B"]), "seed": int(b["seed"]), "scheme": "numpy default_rng; u_panels then home; "
                      "within-panel unit resampling; percentile 2.5 / 97.5"}}
    out["goal_b_score"] = goal
    out["search"] = _search_report(doc)
    a, bb = cls["primary_a"], cls["primary_b"]
    pp = lambda x: f"{100 * x:+.2f}"                                                  # noqa: E731
    step = ("" if doc["phase"] != 1 else " -> STOP (a futility class of I)" if a["stop"] else " -> phase 2")
    verdict = (f"A (I, phase {doc['phase']}{', futility-only' if a['futility_only'] else ''}): "
               f"{' + '.join(a['classes'])}{step}; B (G_U, leak-inclusive): {' + '.join(bb['classes'])}; "
               f"I {pp(float(i))} pp [{pp(ci['I'][0])}, {pp(ci['I'][1])}], G_U {pp(float(gu))} pp "
               f"[{pp(ci['G_U'][0])}, {pp(ci['G_U'][1])}], G_{home} {pp(float(g[home]))} pp")
    return dict(out, decision=dict(cls, verdict=verdict))
