"""Aggregate the battery's probes per ARM and rank the catalogue (`gen3_probe_battery_v1`).

Per fact and slot: each arm's mean over its seeds with a t-interval (95 %), the control score and the selectivity
(real − control), the random-init arm's score, and the second arm minus the first with a Welch interval. Binary facts
are put on R²'s 0-to-1 scale as ``2·AUC − 1`` (the Gini coefficient) wherever facts are RANKED against each other; the
raw AUC is kept beside it.

The catalogue reads each fact at its DECISION SITES (``Fact.sites``): ``final`` is the best of them at the trunk's
LAST layer (tokens) or the heads (``PI`` the policy state, ``VF`` the critic pool); ``input`` the best token site at
the trunk INPUT; ``L1`` after the first layer. A fact is flagged (a HUNT candidate) when, in the reference arm:
``final < POOR`` · the trunk LOSES it (``final < input − LOSS`` with the per-seed interval of the loss below 0) ·
learning adds nothing over a random network (``final − random < LEARN``) · it is not selective
(``selectivity < SELECT``) — or when the arms differ (the gap's interval excludes 0 by more than ``GAP``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

POOR = 0.5
LOSS = 0.05
LEARN = 0.10
SELECT = 0.10
GAP = 0.03
EASY = 0.8
SPECIES_DETERMINED = 0.8
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 20: 2.086, 30: 2.042}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    keys = sorted(_T95)
    for k in keys:
        if df <= k:
            return _T95[k]
    return 1.96


def parse_arms(specs: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"[probe_battery] --arm/--random takes ARM=label,label (got {s!r})")
        k, v = s.split("=", 1)
        out[k] = [x for x in v.split(",") if x]
    return out


def gini(kind: str, x: Optional[float]) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return float(x) if kind == "c" else 2.0 * float(x) - 1.0


def mean_ci(xs: Sequence[float]) -> Dict[str, Any]:
    a = np.array([x for x in xs if x is not None and not math.isnan(x)], dtype=float)
    if len(a) == 0:
        return {"n": 0, "mean": None, "ci": None}
    if len(a) == 1:
        return {"n": 1, "mean": round(float(a[0]), 4), "ci": None}
    hw = t95(len(a) - 1) * float(a.std(ddof=1)) / math.sqrt(len(a))
    return {"n": int(len(a)), "mean": round(float(a.mean()), 4), "ci": [round(float(a.mean() - hw), 4),
                                                                       round(float(a.mean() + hw), 4)]}


def welch(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """``mean(b) − mean(a)`` with a Welch 95 % interval."""
    x = np.array([v for v in a if v is not None and not math.isnan(v)], dtype=float)
    y = np.array([v for v in b if v is not None and not math.isnan(v)], dtype=float)
    if len(x) < 2 or len(y) < 2:
        return {"diff": None, "ci": None, "t": None}
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    d = float(y.mean() - x.mean())
    df = (vx + vy) ** 2 / ((vx ** 2) / (len(x) - 1) + (vy ** 2) / (len(y) - 1)) if se > 0 else 1
    hw = t95(int(max(1, math.floor(df)))) * se
    return {"diff": round(d, 4), "ci": [round(d - hw, 4), round(d + hw, 4)],
            "t": round(d / se, 2) if se > 0 else None}


def load_probes(probes: Path, labels: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for lab in labels:
        p = Path(probes) / f"{lab}.json"
        if not p.exists():
            raise SystemExit(f"[probe_battery report] no probes for {lab} ({p})")
        out[lab] = json.loads(p.read_text())
    return out


def _get(pr: Dict[str, Any], slot: str, fact: str, key: str = "metric") -> Optional[float]:
    r = (pr["slots"].get(slot) or {}).get(fact)
    return None if r is None else r.get(key)


def last_depth(pr: Dict[str, Any]) -> str:
    from main.probe_battery.probes import DEPTH_NAMES

    return DEPTH_NAMES[int(pr["n_layers"])]


def site_slot(site: str, depth: str) -> str:
    return site if site in ("PI", "VF") else f"{site}@{depth}"


def summarise(facts: Sequence[Dict[str, Any]], arms: Dict[str, List[str]], randoms: Dict[str, List[str]],
              P: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per fact: every slot's per-arm scores; the catalogue read (final / input / L1 / random / selectivity)."""
    any_lab = next(iter(P.values()))
    slots = list(any_lab["slots"])
    out: Dict[str, Any] = {}
    for f in facts:
        name, kind = f["name"], f["kind"]
        row: Dict[str, Any] = {"family": f["family"], "kind": kind, "about": f["about"], "sites": f["sites"],
                               "desc": f["desc"], "n_valid": f["n_valid"], "base": f["base"],
                               "species_lookup": f.get("species_lookup"), "slots": {}, "arms": {}}
        for slot in slots:
            ent: Dict[str, Any] = {}
            for arm, labs in list(arms.items()) + [(f"{k}_random", v) for k, v in randoms.items()]:
                ms = [_get(P[lab], slot, name) for lab in labs]
                cs = [_get(P[lab], slot, name, "ctrl") for lab in labs]
                if all(m is None for m in ms):
                    continue
                g = [gini(kind, m) for m in ms]
                gc = [gini(kind, c) for c in cs]
                sel = [None if a is None or b is None else a - b for a, b in zip(g, gc)]
                ent[arm] = {"raw": mean_ci([m for m in ms if m is not None]), "score": mean_ci(g),
                            "ctrl": mean_ci(gc), "selectivity": mean_ci(sel), "per_seed": g}
            if ent:
                row["slots"][slot] = ent
        # the catalogue read, per arm
        for arm, labs in arms.items():
            dl = last_depth(P[labs[0]])
            per_seed = {"final": [], "input": [], "L1": [], "sel": [], "best_site": None}
            best_site, best_val = None, -1e9
            for site in f["sites"]:
                s = site_slot(site, dl)
                v = (row["slots"].get(s, {}).get(arm) or {}).get("score", {}).get("mean")
                if v is not None and v > best_val:
                    best_site, best_val = site, v
            if best_site is None:
                row["arms"][arm] = None
                continue
            tok_sites = [s for s in f["sites"] if s not in ("PI", "VF")]
            for lab in labs:
                fin = gini(kind, _get(P[lab], site_slot(best_site, dl), name))
                ctrl = gini(kind, _get(P[lab], site_slot(best_site, dl), name, "ctrl"))
                inp = [gini(kind, _get(P[lab], f"{s}@in", name)) for s in tok_sites]
                l1 = [gini(kind, _get(P[lab], f"{s}@L1", name)) for s in tok_sites]
                inp = [x for x in inp if x is not None]
                l1 = [x for x in l1 if x is not None]
                per_seed["final"].append(fin)
                per_seed["input"].append(max(inp) if inp else None)
                per_seed["L1"].append(max(l1) if l1 else None)
                per_seed["sel"].append(None if fin is None or ctrl is None else fin - ctrl)
            rnd = []
            rkey = f"{arm}_random"
            if arm in randoms:
                for lab in randoms[arm]:
                    rnd.append(gini(kind, _get(P[lab], site_slot(best_site, dl), name)))
            loss = [None if a is None or b is None else a - b for a, b in zip(per_seed["final"], per_seed["input"])]
            row["arms"][arm] = {"best_site": best_site, "final": mean_ci(per_seed["final"]),
                                "input": mean_ci([x for x in per_seed["input"] if x is not None]),
                                "L1": mean_ci([x for x in per_seed["L1"] if x is not None]),
                                "selectivity": mean_ci(per_seed["sel"]), "final_minus_input": mean_ci(loss),
                                "random": mean_ci([x for x in rnd if x is not None]), "per_seed_final": per_seed["final"],
                                "random_key": rkey}
        out[name] = row
    return out


def gaps(summary: Dict[str, Any], ref: str, alt: str) -> None:
    """``alt − ref`` (Welch) at the reference arm's best site, and at every slot, in place."""
    for name, row in summary.items():
        a = row["arms"].get(ref)
        b = row["arms"].get(alt)
        if a and b:
            row["gap"] = welch(a["per_seed_final"], b["per_seed_final"])
            row["gap"]["site"] = a["best_site"]
        for slot, ent in row["slots"].items():
            if ref in ent and alt in ent:
                ent["gap"] = welch(ent[ref]["per_seed"], ent[alt]["per_seed"])


def classify(row: Dict[str, Any], ref: str, alt: Optional[str]) -> Tuple[str, List[str]]:
    """``(tier, notes)``. The TIER is what the reference arm represents at the fact's decision sites: ``poor`` =
    ``final < POOR`` or the other arm significantly WORSE (a hunt candidate either way), ``easy`` = ``final >= EASY``
    with ``max(input, L1) >= EASY - 0.1`` (readable early), else ``ok``. The NOTES are diagnostics of CAUSE, never a
    tier by themselves: the trunk LOSES it (``final - input < -LOSS``, interval below 0), learning adds nothing over a
    random network (``final - random < LEARN``: the architecture alone delivers what is there), species-determined
    (lookup >= ``SPECIES_DETERMINED``: the read may be species memory), not selective over the control."""
    a = row["arms"].get(ref)
    if not a or a["final"]["mean"] is None:
        return "n/a", ["too rare to read"]
    fin = a["final"]["mean"]
    notes: List[str] = []
    worse = False
    g = row.get("gap")
    if alt and g and g.get("ci") and g["ci"][1] < -GAP:
        notes.append(f"{alt} WORSE than {ref}")
        worse = True
    if alt and g and g.get("ci") and g["ci"][0] > GAP:
        notes.append(f"{alt} better than {ref}")
    fmi = a["final_minus_input"]
    if fmi["mean"] is not None and fmi["mean"] < -LOSS and fmi["ci"] and fmi["ci"][1] < 0:
        notes.append(f"the trunk LOSES it ({fmi['mean']:+.2f} vs its input token)")
    rnd = a["random"]["mean"]
    if rnd is not None and fin - rnd < LEARN:
        notes.append("a random network reads it as well")
    sl = gini(row["kind"], row.get("species_lookup")) if row.get("species_lookup") is not None else None
    if sl is not None and sl >= SPECIES_DETERMINED:
        notes.append(f"species-determined (lookup {sl:.2f})")
    if a["selectivity"]["mean"] is not None and a["selectivity"]["mean"] < SELECT and sl is not None and sl >= 0.5:
        notes.append("not selective over a species-keyed control")
    early = max([x for x in (a["input"]["mean"], a["L1"]["mean"]) if x is not None] or [0.0])
    if fin < POOR or worse:
        tier = "poor"
    elif fin >= EASY and early >= EASY - 0.1:
        tier = "easy"
    else:
        tier = "ok"
    return tier, notes


def write_report(*, probes: Path, arms: Dict[str, List[str]], randoms: Dict[str, List[str]], out: Path) -> None:
    from main.probe_battery.bank import refuse_models_output

    refuse_models_output(out)
    facts = json.loads((Path(probes) / "facts.json").read_text())
    labels = sorted({lab for v in list(arms.values()) + list(randoms.values()) for lab in v})
    P = load_probes(probes, labels)
    S = summarise(facts, arms, randoms, P)
    names = list(arms)
    ref, alt = names[0], (names[1] if len(names) > 1 else None)
    if alt:
        gaps(S, ref, alt)
    for name, row in S.items():
        row["tier"], row["flags"] = classify(row, ref, alt)
        if alt:
            row["tier_" + alt], row["flags_" + alt] = classify(row, alt, None)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(S, indent=1, sort_keys=True))
    (out / "catalogue.md").write_text(render(S, ref, alt))


def _fmt(m: Dict[str, Any]) -> str:
    if not m or m.get("mean") is None:
        return "—"
    if m.get("ci"):
        return f"{m['mean']:.2f} [{m['ci'][0]:.2f}, {m['ci'][1]:.2f}]"
    return f"{m['mean']:.2f}"


def render(S: Dict[str, Any], ref: str, alt: Optional[str]) -> str:
    order = {"poor": 0, "ok": 1, "easy": 2, "n/a": 3}
    rows = sorted(S.items(), key=lambda kv: (order[kv[1]["tier"]],
                                             (kv[1]["arms"].get(ref) or {}).get("final", {}).get("mean") or 0.0))
    head = (f"| tier | fact | family | site | {ref} final | input | L1 | random | selectivity | species lookup |"
            + (f" {alt} final | {alt} − {ref} |" if alt else "") + " flags |")
    sep = "|" + "---|" * (head.count("|") - 1)
    lines = [head, sep]
    for name, r in rows:
        a = r["arms"].get(ref) or {}
        b = (r["arms"].get(alt) or {}) if alt else {}
        g = r.get("gap") or {}
        gs = "—" if not g.get("ci") else f"{g['diff']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}]"
        cells = [r["tier"], f"`{name}`", r["family"], a.get("best_site", "—"), _fmt(a.get("final", {})),
                 _fmt(a.get("input", {})), _fmt(a.get("L1", {})), _fmt(a.get("random", {})),
                 _fmt(a.get("selectivity", {})),
                 "—" if r.get("species_lookup") is None else f"{gini(r['kind'], r['species_lookup']):.2f}"]
        if alt:
            cells += [_fmt(b.get("final", {})), gs]
        cells.append("; ".join(r["flags"]))
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"
