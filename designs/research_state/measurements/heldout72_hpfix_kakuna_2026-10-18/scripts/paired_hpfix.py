"""The PAIRED descriptors on heldout72_hpfix: X26 - K2, and per model heldout72_hpfix - heldout72 (HP Dark), beside the
headline Newcombe intervals in cells.json. `heldout_teams_kakuna_2026-10-18/scripts/paired.py` with the cells renamed
and the cross-set pairs added: a heldout72_hpfix game and its heldout72 twin (same model, unit seed, half and index)
draw the SAME team ids on both sides (checked), so the pair differs only in the Hidden Power repair. Its docstring:

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
OLD_STATE = Path("/home/goodlad/dev/gen3ai-reads/heldout72_kakuna_2026-10-18")
OLD_ROWS = OUT.parent / "heldout_teams_kakuna_2026-10-18" / "rows.jsonl"


def main():
    rows = [json.loads(line) for line in (OUT / "rows.jsonl").read_text().splitlines()]
    rows += [json.loads(line) for line in OLD_ROWS.read_text().splitlines()]

    def key(q):
        return (q["unit_seed"], q["half"], q["index"])

    res = {}
    for ts, (x, k) in {"heldout72_hpfix X26-K2": ("X26_greedy_hl72hp", "K2_greedy_hl72hp"),
                         "X26 hpfix-hl72": ("X26_greedy_hl72hp", "X26_greedy_hl72"),
                         "K2 hpfix-hl72": ("K2_greedy_hl72hp", "K2_greedy_hl72")}.items():
        X = {key(q): (bool(q["won"]), q["our_team"]) for q in rows if q["read_cell"] == x}
        K = {key(q): (bool(q["won"]), q["our_team"]) for q in rows if q["read_cell"] == k}
        ks = sorted(set(X) & set(K))
        if not ks:
            continue
        if not all(X[i][1] == K[i][1] for i in ks):
            raise SystemExit(f"{ts}: twins disagree on our team")
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import hq_hpfix as hq
        peer_equal = 0
        for s_ in sorted({i[0] for i in ks}):
            draws = []
            for cell in (x, k):
                d_ = (hq.STATE if cell.endswith("hp") else OLD_STATE) / "cells" / f"{cell}_{s_:02d}"
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
        res[ts] = {"pairs": n, "seeds": seeds, "a_minus_b": m, "a": x, "b": k,
                   "normal95": [m - 1.96 * se, m + 1.96 * se],
                   "seed_bootstrap95": [bs[500], bs[19499]],
                   "a_only_wins": sum(v == 1 for v in d), "b_only_wins": sum(v == -1 for v in d),
                   "units_with_equal_peer_draws": peer_equal}
        r = res[ts]
        print(f"{ts:22s} pairs {n} paired {m:+.3f} normal [{r['normal95'][0]:+.3f}, {r['normal95'][1]:+.3f}] "
              f"seed-boot [{r['seed_bootstrap95'][0]:+.3f}, {r['seed_bootstrap95'][1]:+.3f}] "
              f"discordant a-only {r['a_only_wins']} / b-only {r['b_only_wins']}")
    # The change of X26 - K2 between the sets: per 4-way twin (X26hp - K2hp) - (X26hl72 - K2hl72), same seed bootstrap.
    D = {}
    for q in rows:
        D.setdefault(q["read_cell"], {})[key(q)] = int(bool(q["won"]))
    cs = ["X26_greedy_hl72hp", "K2_greedy_hl72hp", "X26_greedy_hl72", "K2_greedy_hl72"]
    ks = sorted(set.intersection(*[set(D[c]) for c in cs]))
    d = [(D[cs[0]][i] - D[cs[1]][i]) - (D[cs[2]][i] - D[cs[3]][i]) for i in ks]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
    seeds = sorted({i[0] for i in ks})
    by = {s: [v for i, v in zip(ks, d) if i[0] == s] for s in seeds}
    rng = random.Random(0)
    bs = []
    for _ in range(20000):
        smp = [v for s in (rng.choice(seeds) for _ in seeds) for v in by[s]]
        bs.append(sum(smp) / len(smp))
    bs.sort()
    res["change of X26-K2 (hpfix - hl72)"] = {"quads": n, "mean": m, "normal95": [m - 1.96 * se, m + 1.96 * se],
                                               "seed_bootstrap95": [bs[500], bs[19499]]}
    print(f"change of X26-K2 (hpfix - hl72) quads {n} {m:+.3f} normal [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}] "
          f"seed-boot [{bs[500]:+.3f}, {bs[19499]:+.3f}]")
    (OUT / "paired.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
