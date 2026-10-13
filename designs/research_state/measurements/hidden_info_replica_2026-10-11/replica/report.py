"""Markdown tables from ``score.py``'s result.json (pasted into the README by hand).

    python -m replica.report <result.json>
"""
from __future__ import annotations

import json
import sys


def f(x, d=4):
    return "—" if x is None else f"{x:+.{d}f}"


def ci(v, d=4):
    if not v or v.get("point") is None:
        return "not computed"
    lo, hi = v["ci"]
    return f"{f(v['point'], d)} [{f(lo, d)}, {f(hi, d)}]"


def main() -> int:
    r = json.load(open(sys.argv[1]))
    print("| read | states (end2 / prod) | IMP_O | IMP_P | IMP_F | IMP_F+M | IMP_O-M |")
    print("|---|---|---|---|---|---|---|")
    for name in ("end2", "prod", "pooled"):
        c = r["common"].get(name)
        if not isinstance(c, dict):
            continue
        t = c["LEAF-K8"]
        n = " / ".join(str(v) for v in t["n_states"].values())
        print(f"| {name} | {n} | " + " | ".join(ci(t["IMP"][a]) for a in ("O", "P", "F", "F+M", "O-M")) + " |")
    print()
    print("| read | G = O - F | G_P = O - P | F+M - F | O - (O-M) | s_M | n_M | B2 | B3 | B4 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in ("end2", "prod", "pooled"):
        c = r["common"].get(name)
        if not isinstance(c, dict):
            continue
        t = c["LEAF-K8"]
        v = r["verdicts"][name]
        print(f"| {name} | {ci(t['G'])} | {ci(t['G_P'])} | {ci(t['dFM'])} | {ci(t['dOM'])} | {ci(t['s_M'], 2)} | "
              f"{ci(t['n_M'], 2)} | {v.get('B2_G_lower_gt_0.02')} | {v.get('B3_sufficiency')} | "
              f"{v.get('B4_necessity')} |")
    print()
    print("| scorer | read | IMP_O | IMP_P | IMP_F | IMP_F+M | IMP_O-M | G |")
    print("|---|---|---|---|---|---|---|---|")
    for sc in ("LEAF-K1", "LEAF-K4"):
        for name in ("end2", "prod", "pooled"):
            c = r["common"].get(name)
            if not isinstance(c, dict):
                continue
            t = c[sc]
            print(f"| {sc} | {name} | " + " | ".join(ci(t["IMP"][a]) for a in ("O", "P", "F", "F+M", "O-M"))
                  + f" | {ci(t['G'])} |")
    print()
    print("Exclusions (states without 8 accepted worlds, per arm):")
    for cp, arms in r["excluded"].items():
        print(f"- {cp}: " + "; ".join(f"{a} {v['n']} ({', '.join(f'{k} {n}' for k, n in v['reasons'].items())})"
                                      for a, v in sorted(arms.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
