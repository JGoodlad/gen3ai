"""THE S-U GATE'S LEAK-REDUCED RE-GATE: the registered read (``../PREREG.md``), written and committed BEFORE any state
of the re-gate is scored.

Inputs, all read-only: the gate's frozen draw (``~/gen3ai_archive/su_gate_2026-10-27/gate/selected.jsonl``), the
gate's scored rows (``…/gate/scored.jsonl``, S = ``r0s4@10``), and the re-gate's scored rows (S' = ``r0s4_rolls@10``,
``~/gen3ai_archive/su_regate_rolls_2026-10-28/scored.jsonl``, made by ``python -m main.value_search.sign_gate score
--search r0s4_rolls@10``).

1. VOID / DEFECT checks (named; any one stops the read, never a skip):
   - the scored keys are exactly the selected keys, each once (and ``sign_gate.read``'s own count and duplicate refusals);
   - every searched row of the re-gate carries the preset ``r0s4_rolls`` config (``dice == "rolls"``), every gate row
     ``r0s4`` (``dice == "seed"``) (``--control`` reads the gate's own rows as both);
   - THE SPEED-PIN TOOTH: the Speed +-1 marker is a property of the TRUE root only, so every re-gate row's
     ``speed_flip`` must equal the gate's row for the same key, and its unflipped control must pass;
   - THE REFEREE TOOTH: the referee is a function of (the true log, the key's 24 truth seeds, the two actions); where
     S' deviates to the SAME action as S did, its referee outcome strings and T must equal the gate's exactly.
2. The DECISION: ``main.value_search.sign_gate.read`` on the re-gate's rows with the gate's registered params (rule
   ``deviation_sign``: pooled-U sign accuracy, H / F / P72 equal-weighted, game-clustered percentile bootstrap,
   ``default_rng(20261027)``, B 2,000; GO iff LB > 0.50 + 0.0005, else STOP, a bound within 0.0005 flagged
   ``boundary``).
3. REPORTED, never decided: the SPEED-PIN stratum (pinned / not pinned) and the gate review's leak split (exposed =
   a rewritten past opponent choice OR a pinned Speed; clean = neither), each per pooled U and home72, with the same
   estimator and seed as the gate review's ``gate_review_checks.py`` (so ``--control`` reproduces its digits); how
   S' moved against S (per panel: agree / same deviation / other deviation / deviation gained / lost / exclusions);
   the paired IMP difference S' - S on states searched under both, game-clustered (rng 20261027, B 2,000).

    PYTHONPATH=src python designs/research_state/measurements/su_regate_rolls_2026-10-28/scripts/regate_read.py \\
        [--scored PATH] [--control] [--out PATH]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

import numpy as np

from main.value_search import sign_gate as G

GATE = Path.home() / "gen3ai_archive" / "su_gate_2026-10-27" / "gate"
SELECTED = GATE / "selected.jsonl"
GATE_SCORED = GATE / "scored.jsonl"
REGATE_SCORED = Path.home() / "gen3ai_archive" / "su_regate_rolls_2026-10-28" / "scored.jsonl"
OUT = Path(__file__).resolve().parent.parent / "regate_result.json"
U = ["heldout72_hpfix", "forum72", "p72"]
HOME = "home72"
B, SEED = 2000, 20261027
#: the gate's registered decision params (``su_gate_2026-10-27/read_spec.json``), with the draw's size
PARAMS = {"home_panel": HOME, "u_panels": U, "bar": 0.5, "rule8_eps": 0.0005, "bootstrap": {"B": B, "seed": SEED}}


class RegateDefect(RuntimeError):
    """A VOID or a failed tooth (module docstring §1): the read stops."""


def jl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def rewrite(r: Mapping[str, Any]) -> bool:
    lk = r["search"].get("leak") or {}
    return bool(lk.get("rewritten_moves") or lk.get("rewritten_switches"))


def pinned(r: Mapping[str, Any]) -> bool:
    return r["speed_flip"]["speed_pinned"] > 0


def exposed(r: Mapping[str, Any]) -> bool:
    return rewrite(r) or pinned(r)


def sign(r: Mapping[str, Any]) -> float:
    return G.state_values(r)["sign"]


def strat(rows: List[Mapping[str, Any]], keep: Callable[[Mapping[str, Any]], bool]) -> Dict[str, Any]:
    g: Dict[str, Dict[str, List[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g[r["panel"]][r["deal"]].append(sign(r) if keep(r) else float("nan"))
    out = {}
    for name, panels in (("U", U), ("home72", [HOME])):
        b = G.cluster_ratio_boot(g, panels, B=B, seed=SEED)
        out[name] = {"est": round(b["est"], 4), "lo": round(b["lo"], 4), "hi": round(b["hi"], 4), "n": b["n"],
                     "per_panel": {p: [round(v[0], 4), v[1]] for p, v in b["per_panel"].items()}}
    return out


def checks(sel: List[Mapping[str, Any]], gate: Dict[str, Mapping[str, Any]], rows: List[Mapping[str, Any]],
           dice: str) -> Dict[str, Any]:
    sk = [r["key"] for r in sel]
    ck = [r["key"] for r in rows]
    if len(set(ck)) != len(ck):
        raise RegateDefect("a state is scored twice")
    if set(sk) != set(ck) or len(sk) != len(ck):
        raise RegateDefect(f"{len(set(sk) - set(ck))} selected state(s) not scored, {len(set(ck) - set(sk))} scored "
                           "state(s) never selected")
    want = {"seed": "r0s4", "rolls": "r0s4_rolls"}[dice]
    bad = [r["key"] for r in rows if r["search"].get("searched")
           and (r["search"]["config"].get("dice") != dice or r["search"]["config"].get("name") != want)]
    if bad:
        raise RegateDefect(f"{len(bad)} searched row(s) not on preset {want} (dice {dice}); first {bad[0]}")
    # the marker's fields only: ``cpu_s`` is a timing (FINDING R-1: the registered tooth compared it too, and refused
    # 1,539 rows whose marker was equal)
    mk = lambda f: {k: f[k] for k in ("ok", "speed_mons", "speed_pinned")}  # noqa: E731
    flip_bad = [r["key"] for r in rows if mk(r["speed_flip"]) != mk(gate[r["key"]]["speed_flip"])
                or not r["speed_flip"]["ok"]]
    if flip_bad:
        raise RegateDefect(f"SPEED-PIN TOOTH: {len(flip_bad)} row(s) whose Speed +-1 marker differs from the gate's "
                           f"(a true-root property) or whose control failed; first {flip_bad[0]}")
    same, ref_bad = 0, []
    for r in rows:
        g = gate[r["key"]]
        if r.get("referee") and g.get("referee") and r["search"]["played"] == g["search"]["played"]:
            same += 1
            a, b = r["referee"], g["referee"]
            if (a["out"], a["T_pick"], a["T_top"]) != (b["out"], b["T_pick"], b["T_top"]):
                ref_bad.append(r["key"])
    if ref_bad:
        raise RegateDefect(f"REFEREE TOOTH: {len(ref_bad)} same-pick deviation(s) whose referee differs from the "
                           f"gate's; first {ref_bad[0]}")
    return {"states": len(rows), "speed_pin_tooth": "pass", "referee_tooth": {"same_pick_deviations": same,
                                                                              "pass": True}}


def moved(gate: Dict[str, Mapping[str, Any]], rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in U + [HOME]:
        c: Dict[str, int] = collections.Counter()
        for r in rows:
            if r["panel"] != p:
                continue
            s, s0 = r["search"], gate[r["key"]]["search"]
            if not s.get("searched") or not s0.get("searched"):
                c["excluded_" + ("both" if not s.get("searched") and not s0.get("searched")
                                 else "rolls_only" if not s.get("searched") else "seed_only")] += 1
            elif not s["deviation"] and not s0["deviation"]:
                c["both_agree_with_pi"] += 1
            elif s["deviation"] and s0["deviation"]:
                c["same_deviation" if s["played"] == s0["played"] else "other_deviation"] += 1
            else:
                c["deviation_gained" if s["deviation"] else "deviation_lost"] += 1
        out[p] = dict(sorted(c.items()))
    return out


def imp_diff(gate: Dict[str, Mapping[str, Any]], rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    g: Dict[str, Dict[str, List[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        a, b = G.state_values(r)["imp"], G.state_values(gate[r["key"]])["imp"]
        g[r["panel"]][r["deal"]].append(a - b if a == a and b == b else float("nan"))
    out = {}
    for name, panels in (("U", U), ("home72", [HOME])):
        bb = G.cluster_ratio_boot(g, panels, B=B, seed=SEED)
        out[name] = {"est": round(bb["est"], 4), "lo": round(bb["lo"], 4), "hi": round(bb["hi"], 4), "n": bb["n"]}
    return out


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scored", default=str(REGATE_SCORED))
    ap.add_argument("--control", action="store_true",
                    help="read the GATE's own rows as the re-gate (dice 'seed'): must reproduce the gate's verdict "
                         "and the gate review's leak split")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    sel = jl(SELECTED)
    gate = {r["key"]: r for r in jl(GATE_SCORED)}
    rows = jl(GATE_SCORED if a.control else Path(a.scored))
    ck = checks(sel, gate, rows, "seed" if a.control else "rolls")
    res = G.read(rows, {**PARAMS, "expected_states": len(sel)})
    pr = res["primary"]
    line = (f"{res['verdict']}: pooled-U sign accuracy {pr['est']:.3f} [{pr['lo']:.3f}, {pr['hi']:.3f}] (n {pr['n']}; "
            f"GO iff LB > 0.5" + ("; LB within rule 8's eps: STOP" if res["boundary"] else "") + ")")
    out = {"schema": "gen3_su_regate_rolls_v1", "control": bool(a.control), "verdict_line": line,
           "verdict": res["verdict"], "boundary": res["boundary"], "checks": ck,
           "primary": {k: pr[k] for k in ("est", "lo", "hi", "n", "per_panel")},
           "reported": {
               "decision_rows": res["reported"],
               "speed_pin": {"pinned": strat(rows, pinned), "not_pinned": strat(rows, lambda r: not pinned(r))},
               "leak_split": {"exposed": strat(rows, exposed), "clean": strat(rows, lambda r: not exposed(r)),
                              "share_exposed_of_searched": {
                                  p: round(float(np.mean([exposed(r) for r in rows if r["panel"] == p
                                                          and r["search"].get("searched")])), 3) for p in U + [HOME]}},
               "moved_vs_gate": moved(gate, rows),
               "imp_minus_gate": imp_diff(gate, rows)}}
    txt = json.dumps(out, indent=1, sort_keys=True, default=str)
    dest = Path(a.out) if a.out else (None if a.control else OUT)
    if dest is not None:
        dest.write_text(txt + "\n")
    print(line)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
