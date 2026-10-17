"""The collapse pilot's STARVATION readout (type (a) of `designs/endstate/design_collapse_and_counterfactuals.md`).

Input: a policy-spectrum bank, ground-truth rows (`main.policy_spectrum.truth run`: every legal action x S shared dice
seeds played to the end in the TRUE world under a declared greedy continuation), and the reader's per-checkpoint
probabilities (`main.policy_spectrum read`). Forward-free: every number here is arithmetic on those files.

Definitions (the Lane S ones, `main.policy_spectrum.truth`, eps = 0.1 on the +-1 scale, starve mass 1 %):
- NEAR-BEST: paired gap to the best <= eps; DOMINATED: gap - 1.96 SE > eps; DECISIVE turn: >= 1 dominated action.
- STARVED turn: some near-best action has pi < 1 %.
The pilot's refinements (new here):
- TIE-COLLAPSE starvation: starved AND the policy's argmax is itself near-best (greedy play loses nothing at this
  turn; what is lost is the alternative's probability mass: its learning signal and the policy's unpredictability).
- MISRANK starvation: starved AND the argmax is NOT near-best (a near-best move sits under 1 % while the top choice is
  worse by more than eps on the means) -- the harmful case at greedy play.
- HELD-OUT GAIN: to avoid the winner's curse, the near-best / starved sets are chosen on the FIRST half of the seeds
  and the gain of the best starved move over the argmax, v(starved*) - v(argmax), is scored on the SECOND half.
  Turns with no starved move score 0. Reported per decisive turn (the value a greedy player would recover by
  playing the starved move) and per starved turn.
- IMP (held out): the rollout argmax chosen on the first half, v(best) - v(argmax) scored on the second half -- the
  search ledger's decision-improvement currency, here in the TRUE world under the declared continuation.
- Greedy regret V* - v(argmax), and the sampled regret V* - sum pi v (the Lane S `regret`).
Rule 8: a (turn, action) with |pi - 0.01| < 1e-6 is a boundary input; a turn holding one for ANY compared policy is
EXCLUDED for all of them (counted), so paired reads share one turn set.
Gaps are multiples of 1/S on +-1 / 0 outcomes and eps = 0.1 is not, for S in {16, 32, 64}: no gap boundary ties.

    designs/research_state/measurements/collapse_pilot_2026-10-17/run.sh \
        designs/research_state/measurements/collapse_pilot_2026-10-17/starvation_readout.py \
        --bank <bank> --rows <label>=<rows.jsonl[.gz]> [...] --reads <dir> --labels L1,L2,... --out <json>
"""
import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from main.policy_spectrum.bank import load_bank
from main.policy_spectrum.reader import load_probs
from main.policy_spectrum.spectrum import _boot_ci
from main.policy_spectrum.truth import EPS, STARVE, first_seeds, load_rows, turn_truth

BOUNDARY = 1e-6


def _split(row: dict, half: int) -> dict:
    return dict(row, seeds=row["seeds"][half:], outcomes={a: v[half:] for a, v in row["outcomes"].items()})


def boundary_ids(bank, rows: list[dict], probs: np.ndarray) -> set:
    """Rule 8: the turns where some legal action's pi is within BOUNDARY of the 1 % starve mass."""
    idx = {d["id"]: i for i, d in enumerate(bank.decisions)}
    out = set()
    for r in rows:
        if r.get("ok"):
            acts = [int(a) for a in r["outcomes"]]
            if np.any(np.abs(probs[idx[r["id"]], acts] - STARVE) < BOUNDARY):
                out.add(r["id"])
    return out


def per_turn(bank, rows: list[dict], probs: np.ndarray, eps: float = EPS,
             exclude: frozenset = frozenset()) -> dict[str, np.ndarray]:
    """``exclude``: the UNION of every compared policy's boundary turns, so paired reads share one turn set."""
    idx = {d["id"]: i for i, d in enumerate(bank.decisions)}
    keys = ("decisive", "starved", "tie", "misrank", "gain", "imp", "greedy_regret", "regret", "near_mass", "dom_mass",
            "n_near", "entropy", "rank1", "argmax_near")
    out: dict[str, list] = {k: [] for k in keys}
    out["battle"] = []
    excluded = 0
    for r in rows:
        if not r.get("ok"):
            continue
        if r["id"] in exclude:
            excluded += 1
            continue
        i = idx[r["id"]]
        t = turn_truth(r, eps)
        acts = t["actions"]
        p = probs[i, acts]
        s = len(r["seeds"])
        h = s // 2
        ta = turn_truth(first_seeds(r, h), eps)              # selection half
        tb = turn_truth(_split(r, h), eps)                    # scoring half
        am = int(np.argmax(p))
        starved = t["near"] & (p < STARVE)
        sa = ta["near"] & (p < STARVE)
        gain = 0.0
        if sa.any():
            j = int(np.flatnonzero(sa)[np.argmax(ta["value"][sa])])
            gain = float(tb["value"][j] - tb["value"][am])
        imp = float(tb["value"][int(np.argmax(ta["value"]))] - tb["value"][am])
        pl = p[p > 0]
        out["decisive"].append(float(t["dominated"].any()))
        out["starved"].append(float(starved.any()))
        out["tie"].append(float(starved.any() and t["near"][am]))
        out["misrank"].append(float(starved.any() and not t["near"][am]))
        out["gain"].append(gain)
        out["imp"].append(imp)
        out["greedy_regret"].append(float(t["vstar"] - t["value"][am]))
        out["regret"].append(float(t["vstar"] - (p * t["value"]).sum()))
        out["near_mass"].append(float(p[t["near"]].sum()))
        out["dom_mass"].append(float(p[t["dominated"]].sum()))
        out["n_near"].append(float(t["near"].sum()))
        out["entropy"].append(float(-(pl * np.log(pl)).sum()))
        out["rank1"].append(float(p.max()))
        out["argmax_near"].append(float(t["near"][am]))
        out["battle"].append(bank.decisions[i]["battle"])
    res = {k: np.array(v) for k, v in out.items()}
    res["excluded"] = np.array([excluded])
    return res


def summarize(T: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    out = {"turns": int(mask.sum()), "excluded_boundary": int(T["excluded"][0])}
    cl = T["battle"][mask]
    for k in ("starved", "tie", "misrank", "gain", "imp", "greedy_regret", "regret", "near_mass", "dom_mass", "n_near",
              "entropy", "rank1", "argmax_near"):
        v = T[k][mask]
        out[k] = {"mean": round(float(v.mean()), 4) if len(v) else None, "ci": _boot_ci(v, cl)}
    st = mask & (T["starved"] > 0)
    out["gain_per_starved_turn"] = {"n": int(st.sum()),
                                    "mean": round(float(T["gain"][st].mean()), 4) if st.any() else None,
                                    "ci": _boot_ci(T["gain"][st], T["battle"][st]) if st.any() else None}
    return out


def paired(A: dict[str, np.ndarray], B: dict[str, np.ndarray], decisive: bool) -> dict:
    if not np.array_equal(A["battle"], B["battle"]):
        raise SystemExit("paired reads differ in their turn sets (a boundary exclusion differs): report unpaired")
    m = A["decisive"] > 0 if decisive else np.ones(len(A["battle"]), bool)
    out = {"turns": int(m.sum())}
    for k in ("starved", "tie", "misrank", "gain", "imp", "greedy_regret", "regret", "dom_mass", "entropy", "rank1"):
        d = B[k][m] - A[k][m]
        out[k] = {"delta": round(float(d.mean()), 4), "ci": _boot_ci(d, A["battle"][m])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--rows", action="append", required=True, help="<continuation label>=<rows file>")
    ap.add_argument("--reads", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pair", action="append", default=[], help="<label a>:<label b>, beyond the chain's own")
    ap.add_argument("--seeds", default="", help="also read at these nested seed counts, e.g. 16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bank = load_bank(Path(a.bank))
    labels = [x for x in a.labels.split(",") if x]
    probs = {lab: load_probs(Path(a.reads), lab, bank) for lab in labels}
    res: dict = {"eps": EPS, "starve": STARVE, "labels": labels, "by_rows": {}}
    for spec in a.rows:
        cname, _, path = spec.partition("=")
        rows_full = load_rows(Path(path))
        s_full = sorted({len(r["seeds"]) for r in rows_full if r.get("ok")})
        variants = {f"S{s_full[-1]}": rows_full}
        for k in [int(x) for x in a.seeds.split(",") if x]:
            variants[f"S{k}"] = [first_seeds(r, k) for r in rows_full]
        for vname, rows in variants.items():
            key = f"{cname}|{vname}"
            excl = frozenset().union(*(boundary_ids(bank, rows, probs[lab]) for lab in labels))
            T = {lab: per_turn(bank, rows, probs[lab], exclude=excl) for lab in labels}
            block: dict = {"rows": len(rows), "refused": sum(1 for r in rows if not r.get("ok")), "levels": {},
                           "paired": {}}
            for lab in labels:
                n = len(T[lab]["battle"])
                block["levels"][lab] = {"all": summarize(T[lab], np.ones(n, bool)),
                                        "decisive": summarize(T[lab], T[lab]["decisive"] > 0)}
            extra = [tuple(p.split(":", 1)) for p in a.pair]
            for x, y in list(pairwise(labels)) + [(labels[0], labels[-1])] + extra:
                block["paired"][f"{x}->{y}"] = {"all": paired(T[x], T[y], False),
                                                "decisive": paired(T[x], T[y], True)}
            res["by_rows"][key] = block
            print(f"== {key}: {block['rows']} rows, {block['refused']} refused")
            for lab in labels:
                d = block["levels"][lab]["decisive"]
                print(f"  {lab:>10}: decisive {d['turns']:4d} starved {d['starved']['mean']:.3f} "
                      f"(tie {d['tie']['mean']:.3f} / misrank {d['misrank']['mean']:.3f}) "
                      f"gain {d['gain']['mean']:+.4f} imp {d['imp']['mean']:+.4f} greedy-regret {d['greedy_regret']['mean']:.4f} "
                      f"regret {d['regret']['mean']:.4f} dom {d['dom_mass']['mean']:.3f} H {d['entropy']['mean']:.3f}")
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
