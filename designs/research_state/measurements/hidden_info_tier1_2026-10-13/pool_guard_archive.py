"""Run the world consistency guard (``world_guard.check_teams``, the call ``pool_worlds.pool_sample`` now makes) over the
ARCHIVED POOL (T01) worlds of the Tier 1 read, to confirm 0 false positives: every world of every state, both
checkpoints, the registered 600 and the supplement.

    PYTHONPATH=<this checkout>/src python -I pool_guard_archive.py [max_states_per_cp]     # writes pool_guard_archive.json

A raise is recorded with its message (a real defect or a false positive; the report names each) and the run goes on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import t1_common as T  # noqa: E402
import t1_data as D  # noqa: E402

from main.policy_spectrum.bank import load_bank  # noqa: E402
from main.policy_spectrum.truth import cmd_index, to_log  # noqa: E402
from main.value_search import world_guard as WG  # noqa: E402
from main.value_search import worlds as W  # noqa: E402


def main() -> None:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    res = {"states": 0, "worlds": 0, "flagged_states": 0, "flagged": [], "per_cp": {}}
    for cp in T.CPS:
        banks = {w: load_bank(T.bank_path(cp, w)) for w in ("registered", "supp")}
        dec = {}
        for w, b in banks.items():
            for d in b.decisions:
                dec[d["id"]] = (b, d)
        rows = D.code_worlds(cp, "T01")
        n = nw = 0
        for did in sorted(rows)[:cap]:
            r = rows[did]
            if not (r.get("ok") and r.get("teams")):
                continue
            b, d = dec[did]
            bat = b.battles[b.battle_index[d["battle"]]]
            log, at = to_log(bat), cmd_index(bat, d["side"], d["n"])
            root = W.prepare_root(log, at, d["side"], did)
            n += 1
            nw += len(r["teams"])
            try:
                WG.check_teams(root, r["teams"], did)
            except WG.WorldConsistencyError as e:
                res["flagged_states"] += 1
                res["flagged"].append({"cp": cp, "id": did, "error": str(e)[:400]})
        res["per_cp"][cp] = {"states": n, "worlds": nw}
        res["states"] += n
        res["worlds"] += nw
        print(cp, res["per_cp"][cp], "flagged so far", res["flagged_states"], flush=True)
    (HERE / "pool_guard_archive.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "flagged"}), res["flagged"][:5])


if __name__ == "__main__":
    main()
