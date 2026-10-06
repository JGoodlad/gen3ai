"""Compare the CPU and GPU identity plays row by row (declared ledger reads, one per protocol).

    PYTHONPATH=<tree>/src python compare.py <scratch dir>   # writes compare.json beside this file"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from agents.training import eval_ledger as L

PROTOS = {"off": "gen3_eval_protocol_v1_h2h", "one": "gen3_eval_protocol_v1_h2h_oracle_one_sided",
          "one2": "gen3_eval_protocol_v1_h2h_oracle_one_sided", "both": "gen3_eval_protocol_v1_h2h_oracle_both_sided"}
DECL = {p: L.ReaderDecl(name=f"x5ab_look1_gpu_identity_{p}", purposes=L.ALL_PURPOSES,
                        regime=L.RegimeFilter(protocol=p, play="greedy", opponent_play="greedy", mirrored=True),
                        requests="any", selection="include", flags_ok=frozenset(), inference="conditional")
        for p in set(PROTOS.values())}
FIELDS = ("counts", "pairs", "teams", "seed")


def rows(root: Path, proto: str):
    out = {}
    for got in L.read_by_regime(DECL[proto], root=root).values():
        for r in got.rows:
            out[(r["player"]["sha256"], r["opponent"]["sha256"], r["request"]["batch"])] = r
    return out


def main() -> int:
    s = Path(sys.argv[1])
    report = {}
    ok = True
    for mode, proto in PROTOS.items():
        a, b = s / f"ledger_cpu_{mode}", s / f"ledger_cuda_{mode}"
        if not (a.exists() and b.exists()):
            report[mode] = "not played"
            continue
        ra, rb = rows(a, proto), rows(b, proto)
        res = []
        for k in sorted(set(ra) | set(rb)):
            x, y = ra.get(k), rb.get(k)
            if x is None or y is None:
                res.append({"cell": [k[0][:12], k[1][:12], k[2]], "equal": False, "why": "missing on one device"})
                ok = False
                continue
            diff = [f for f in FIELDS if x[f] != y[f]]
            cx, cy = x["compute"], y["compute"]
            dig = {"outcome_digest": (cx.get("outcome_digest"), cy.get("outcome_digest")),
                   "outcome_digest_all": (cx.get("outcome_digest_all"), cy.get("outcome_digest_all"))}
            # the ALL-games digest (every game's W/L/D and length) must be equal; the clean-games digest hashes the
            # games OUTSIDE each device's own 2e-3 near-tie list, so it is compared only when the lists agree
            if dig["outcome_digest_all"][0] != dig["outcome_digest_all"][1] or dig["outcome_digest_all"][0] is None:
                diff.append("outcome_digest_all")
            same_list = cx.get("near_tie_games") == cy.get("near_tie_games")
            if same_list and dig["outcome_digest"][0] != dig["outcome_digest"][1]:
                diff.append("outcome_digest")
            nt = {k2: (cx.get(k2), cy.get(k2)) for k2 in cx if "near" in k2}
            res.append({"cell": [x["player"]["id"], x["opponent"]["id"], k[2]], "equal": not diff, "differs": diff,
                        "wld": [[x["counts"][c] for c in "wld"], [y["counts"][c] for c in "wld"]],
                        "near_tie": nt, "digests": dig,
                        "devices": [cx.get("device"), cy.get("device")], "backends": [cx.get("backend"),
                                                                                       cy.get("backend")]})
            ok &= not diff
        report[mode] = res
    (Path(__file__).resolve().parent / "compare.json").write_text(json.dumps({"all_equal": ok, "modes": report},
                                                                            indent=1, default=str) + "\n")
    for mode, res in report.items():
        if isinstance(res, str):
            print(mode, res)
            continue
        for r in res:
            print(mode, r["cell"], "EQUAL" if r["equal"] else f"DIFFERS {r.get('differs', r.get('why'))}")
    print("ALL EQUAL" if ok else "NOT ALL EQUAL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
