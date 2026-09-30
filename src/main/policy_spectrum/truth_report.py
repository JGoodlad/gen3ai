"""Gate ④ AT SCALE — the readout the owner asked for (M5 Lane S, 2026-09-29).

For a chain of policies (N0@75M → C_fix → K2 → K3), per move CATEGORY, on the turns branched to
the end under one or more CONTINUATIONS:

- HEALTHY SHARPENING = the policy's mass on that category's DOMINATED actions (separably worse than
  the best by > ε) falls — read on turns where the category has a dominated action;
- STARVATION = the share of turns where some NEAR-BEST action of that category (gap ≤ ε) gets < 1 %
  of the mass rises — read on DECISIVE turns (≥ 1 dominated action anywhere: on a turn where every
  action is near-best, concentrating the mass starves nothing that matters) where the category has a
  near-best action.

Each step is a PAIRED delta on the same turns (battle-clustered bootstrap). The noise check re-reads
the 64-seed truth through its first 16 seeds (the seeds are nested). The continuation check asks
whether each step's verdict (the sign of a delta whose interval excludes 0) agrees across
continuations.

⚠️ The subset OVER-SAMPLES turns where setup / recovery / hazard / status is legal; per-category
readouts condition on the category anyway, and the ALL-category rows are reported both on every turn
and on the unweighted remainder (turns drawn as v1 or ``random``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from main.policy_spectrum.bank import Bank
from main.policy_spectrum.categories import CATEGORIES
from main.policy_spectrum.spectrum import _boot_ci
from main.policy_spectrum.truth import EPS, STARVE, first_seeds, turn_truth

CATS = ("all",) + CATEGORIES


def turn_table(bank: Bank, rows: Sequence[dict], eps: float = EPS,
               keep: Optional[set] = None) -> List[dict]:
    """Per ok turn: its truth + the per-action categories (policy-free, computed once)."""
    idx = {d["id"]: i for i, d in enumerate(bank.decisions)}
    out = []
    for r in rows:
        if not r.get("ok") or (keep is not None and r["id"] not in keep):
            continue
        i = idx[r["id"]]
        d = bank.decisions[i]
        t = turn_truth(r, eps)
        out.append({"i": i, "battle": d["battle"], "t": t, "acts": np.array(t["actions"]),
                    "cats": np.array([d["cats"][str(a)] for a in t["actions"]]),
                    "decisive": bool(t["dominated"].any())})
    return out


def _arrays(table: Sequence[dict], probs: np.ndarray, cat: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """(values, battle ids) for: dominated mass (turns with a dominated `cat` action), starvation
    and near-best mass (decisive turns with a near-best `cat` action)."""
    dom_v, dom_b, st_v, st_b, nb_v, ss_v, ss_b = [], [], [], [], [], [], []
    for T in table:
        p = probs[T["i"], T["acts"]]
        sel = np.ones(len(p), bool) if cat == "all" else (T["cats"] == cat)
        t = T["t"]
        dsel = sel & t["dominated"]
        if dsel.any():
            dom_v.append(float(p[dsel].sum()))
            dom_b.append(T["battle"])
        nsel = sel & t["near"]
        if T["decisive"] and nsel.any():
            st_v.append(float((p[nsel] < STARVE).any()))
            st_b.append(T["battle"])
            nb_v.append(float(p[nsel].sum()))
        ssel = sel & t["strict_near"]
        if T["decisive"] and ssel.any():            # near-best even at its UPPER bound (noise-robust)
            ss_v.append(float((p[ssel] < STARVE).any()))
            ss_b.append(T["battle"])
    return {"dominated_mass": (np.array(dom_v), np.array(dom_b)),
            "starved": (np.array(st_v), np.array(st_b)),
            "near_mass": (np.array(nb_v), np.array(st_b)),
            "starved_strict": (np.array(ss_v), np.array(ss_b))}


def level(table, probs, cat) -> dict:
    out = {}
    for k, (v, b) in _arrays(table, probs, cat).items():
        out[k] = {"n": int(len(v)), "mean": round(float(v.mean()), 4) if len(v) else None,
                  "ci": _boot_ci(v, b) if len(v) else None}
    return out


def step(table, pa, pb, cat) -> dict:
    A, B = _arrays(table, pa, cat), _arrays(table, pb, cat)
    out = {}
    for k in A:
        d = B[k][0] - A[k][0]
        out[k] = {"n": int(len(d)), "delta": round(float(d.mean()), 4) if len(d) else None,
                  "ci": _boot_ci(d, A[k][1]) if len(d) else None}
    return out


def verdict(q: dict) -> str:
    """'+' / '−' when the interval excludes 0, '0' when it straddles, '?' when there is none."""
    if q.get("ci") is None:
        return "?"
    lo, hi = q["ci"]
    return "+" if lo > 0 else ("−" if hi < 0 else "0")


def analyse(bank: Bank, rows_by_cont: Dict[str, List[dict]], probs: Dict[str, np.ndarray],
            chain: Sequence[str], why: Dict[str, str], eps: float = EPS,
            nested: Sequence[int] = (16,)) -> dict:
    unweighted = {i for i, w in why.items() if w in ("v1", "random")}
    out: dict = {"eps": eps, "starve_mass": STARVE, "chain": list(chain), "continuations": {}}
    for cont, rows in rows_by_cont.items():
        S = len(rows[0]["seeds"])
        variants = {f"S{S}": rows}
        for k in nested:
            if k < S:
                variants[f"S{k}(nested)"] = [first_seeds(r, k) for r in rows]
        cv = {}
        for name, rs in variants.items():
            table = turn_table(bank, rs, eps)
            table_u = turn_table(bank, rs, eps, keep=unweighted)
            near_counts = {c: int(sum(1 for T in table if T["decisive"] and (T["cats"] == c)[T["t"]["near"]].any()))
                           for c in CATEGORIES}
            cv[name] = {
                "turns": len(table), "decisive": int(sum(T["decisive"] for T in table)),
                "mean_near_best": round(float(np.mean([T["t"]["near"].sum() for T in table])), 3),
                "decisive_turns_with_near_best": near_counts,
                "levels": {lab: {c: level(table, probs[lab], c) for c in CATS} for lab in chain},
                "levels_unweighted_all": {lab: level(table_u, probs[lab], "all") for lab in chain},
                "steps": {f"{a} → {b}": {c: step(table, probs[a], probs[b], c) for c in CATS}
                          for a, b in zip(chain, chain[1:])},
                "overall": {f"{chain[0]} → {chain[-1]}": {c: step(table, probs[chain[0]], probs[chain[-1]], c)
                                                          for c in CATS}},
            }
        out["continuations"][cont] = cv
    # agreement of verdicts across continuations (full S only)
    fulls = {c: v[max(v, key=lambda n: int(n[1:].split("(")[0]))] for c, v in out["continuations"].items()}
    agree = {}
    for key in list(next(iter(fulls.values()))["steps"]) + list(next(iter(fulls.values()))["overall"]):
        for c in CATS:
            for m in ("dominated_mass", "starved", "starved_strict"):
                vs = {}
                for cont, f in fulls.items():
                    blk = f["steps"].get(key) or f["overall"].get(key)
                    vs[cont] = verdict(blk[c][m])
                agree[f"{key} | {c} | {m}"] = vs
    out["verdict_agreement"] = agree
    return out


def render(res: dict) -> str:
    L = ["# Gate ④ at scale — healthy sharpening vs starvation, by category", ""]
    chain = res["chain"]
    for cont, cv in res["continuations"].items():
        for name, v in cv.items():
            L += [f"## Continuation: {cont} — {name}", "",
                  f"{v['turns']} turns branched, {v['decisive']} decisive; {v['mean_near_best']} near-best "
                  f"actions per turn. Decisive turns with a near-best action, by category: "
                  + ", ".join(f"{c} {n}" for c, n in v["decisive_turns_with_near_best"].items()) + ".", ""]
            L += ["**Levels** — dominated mass (turns with a dominated action of the category) / starved share "
                  "(decisive turns with a near-best action of it):", "",
                  "| policy | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
            for lab in chain:
                cells = []
                for c in CATS:
                    q = v["levels"][lab][c]
                    dm, st = q["dominated_mass"]["mean"], q["starved"]["mean"]
                    cells.append(f"{'—' if dm is None else f'{dm:.3f}'} / {'—' if st is None else f'{st:.3f}'}")
                L.append(f"| {lab} | " + " | ".join(cells) + " |")
            L += ["", "**Paired steps** (Δ dominated mass ; Δ starved share; (+) / (−) = the 95 % battle-clustered interval excludes 0, (0) = it does not):", "",
                  "| step | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
            for key, blk in list(v["steps"].items()) + list(v["overall"].items()):
                cells = []
                for c in CATS:
                    a, b = blk[c]["dominated_mass"], blk[c]["starved"]
                    fa = "—" if a["delta"] is None else f"{a['delta']:+.3f} ({verdict(a)})"
                    fb = "—" if b["delta"] is None else f"{b['delta']:+.3f} ({verdict(b)})"
                    cells.append(f"{fa} ; {fb}")
                L.append(f"| {key} | " + " | ".join(cells) + " |")
            L += ["", "**Strict starvation** (the starved action is near-best even at its upper bound, "
                  "gap + 1.96·SE ≤ ε) — Δ per step:", "",
                  "| step | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
            for key, blk in list(v["steps"].items()) + list(v["overall"].items()):
                cells = []
                for c in CATS:
                    q = blk[c]["starved_strict"]
                    cells.append("—" if q["delta"] is None else f"{q['delta']:+.3f} ({verdict(q)}) n={q['n']}")
                L.append(f"| {key} | " + " | ".join(cells) + " |")
            L.append("")
    L += ["## Verdict agreement across continuations (+ / − = interval excludes 0; 0 = straddles)", "",
          "| step \\| category \\| measure | " + " | ".join(res["continuations"]) + " |",
          "|---|" + "---|" * len(res["continuations"])]
    for k, vs in res["verdict_agreement"].items():
        L.append(f"| {k} | " + " | ".join(vs[c] for c in res["continuations"]) + " |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    import argparse

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import load_probs
    from main.policy_spectrum.truth import load_rows

    ap = argparse.ArgumentParser(prog="python -m main.policy_spectrum.truth_report")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--rows", action="append", required=True, help="<label>=<rows.jsonl[.gz]>")
    ap.add_argument("--reads", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args(argv)
    bank = load_bank(Path(a.bank))
    why = json.loads(Path(a.subset).read_text())["why"]
    chain = a.chain.split(",")
    probs = {lab: load_probs(Path(a.reads), lab, bank) for lab in chain}
    rows = {}
    for spec in a.rows:
        lab, _, path = spec.partition("=")
        rows[lab] = load_rows(Path(path))
    res = analyse(bank, rows, probs, chain, why)
    Path(a.out_json).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    Path(a.out_md).write_text(render(res))
    print(f"[truth_report] wrote {a.out_json} and {a.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
