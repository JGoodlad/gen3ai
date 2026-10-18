"""The PAIRED descriptor of X26 - K2 (beside the headline Newcombe interval in cells.json).

Every K2 game has an X26 twin with the same unit seed, half and index, and both sides' team
draws are identical in each twin (checked here: our side's team per game, and the PEER's `team_draws` from both
halves' peer reports, per unit pair). The heldout72 copy of the 10-17 `paired.py`. The paired difference is the mean of
(X26 won) - (K2 won) over the twins (a tie is not a win, as in the Wilson cells), with a normal
interval and a seed-cluster bootstrap (20,000 resamples, Random(0)). DESCRIPTOR only.

    python paired.py   (reads ../rows.jsonl, writes ../paired.json)
"""
import json
import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]


def main():
    rows = [json.loads(line) for line in (OUT / "rows.jsonl").read_text().splitlines()]

    def key(q):
        return (q["unit_seed"], q["half"], q["index"])

    res = {}
    for ts, (x, k) in {"heldout72": ("X26_greedy_hl72", "K2_greedy_hl72")}.items():
        X = {key(q): (bool(q["won"]), q["our_team"]) for q in rows if q["read_cell"] == x}
        K = {key(q): (bool(q["won"]), q["our_team"]) for q in rows if q["read_cell"] == k}
        ks = sorted(set(X) & set(K))
        if not ks:
            continue
        if not all(X[i][1] == K[i][1] for i in ks):
            raise SystemExit(f"{ts}: twins disagree on our team")
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import hq
        peer_equal = 0
        for s_ in sorted({i[0] for i in ks}):
            draws = []
            for cell in (x, k):
                d_ = hq.STATE / "cells" / f"{cell}_{s_:02d}"
                draws.append([json.loads(p_.read_text()).get("team_draws") for p_ in sorted(d_.glob("*/peer_*_report.json"))])
            if draws[0] != draws[1] or not all(draws[0]):
                raise SystemExit(f"{ts}: seed {s_}: the peer's team draws differ between twins (or are missing)")
            peer_equal += 1
        d = [int(X[i][0]) - int(K[i][0]) for i in ks]
        n = len(d)
        m = sum(d) / n
        se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
        seeds = sorted({i[0] for i in ks})
        by = {s: [int(X[i][0]) - int(K[i][0]) for i in ks if i[0] == s] for s in seeds}
        rng = random.Random(0)
        bs = []
        for _ in range(20000):
            smp = [v for s in (rng.choice(seeds) for _ in seeds) for v in by[s]]
            bs.append(sum(smp) / len(smp))
        bs.sort()
        res[ts] = {"pairs": n, "seeds": seeds, "x26_minus_k2": m,
                   "normal95": [m - 1.96 * se, m + 1.96 * se],
                   "seed_bootstrap95": [bs[500], bs[19499]],
                   "x26_only_wins": sum(v == 1 for v in d), "k2_only_wins": sum(v == -1 for v in d),
                   "units_with_equal_peer_draws": peer_equal}
        r = res[ts]
        print(f"{ts:7s} pairs {n} paired {m:+.3f} normal [{r['normal95'][0]:+.3f}, {r['normal95'][1]:+.3f}] "
              f"seed-boot [{r['seed_bootstrap95'][0]:+.3f}, {r['seed_bootstrap95'][1]:+.3f}] "
              f"discordant X26-only {r['x26_only_wins']} / K2-only {r['k2_only_wins']}")
    (OUT / "paired.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
