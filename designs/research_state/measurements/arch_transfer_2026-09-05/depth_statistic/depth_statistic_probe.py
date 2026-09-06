"""THE DEPTH STATISTIC — how much of the search's improvement needs depth >= 2?

Two halves, both offline:

* ``committed`` — read the three committed defensive-search cells and report the depth columns
  they DO carry (``max_depth_realized``, ``n_deepened``, ``realized_mean.depth``) plus the gate /
  futility shares. This is the half that establishes whether the existing artifacts can answer the
  question at all.
* ``cell`` — read the fresh paired depth-1 / depth-3 cell and compute the pre-registered
  statistics: the depth-engagement fraction, the depth-attributable action change, and the paired
  mirror win-rate difference.

Run:
    python designs/research_state/measurements/arch_transfer_2026-09-05/depth_statistic/depth_statistic_probe.py committed
    python .../depth_statistic_probe.py cell <d1.jsonl> <d3.jsonl>

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

MEAS = Path(__file__).resolve().parents[2]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Newcombe's hybrid-score interval for a difference of two independent proportions."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def rows(path: Path) -> list[dict]:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sum(rs, key):
    return sum(r.get(key, 0) or 0 for r in rs)


def committed() -> None:
    print("=" * 78)
    print("COMMITTED DEFENSIVE CELLS — what depth do they actually record?")
    print("=" * 78)
    for name in ("defensive_search_first_cell", "defensive_search_iter2",
                 "defensive_search_iter3"):
        rs = rows(MEAS / f"{name}_2026-08-29_rows.jsonl.gz")
        dec = _sum(rs, "n_decisions")
        handled = _sum(rs, "n_defensive")
        forced = _sum(rs, "n_defensive_forced")
        raced = _sum(rs, "n_defensive_raced")
        sep = _sum(rs, "n_defensive_separated")
        over = _sum(rs, "n_defensive_overruled")
        fut = _sum(rs, "n_defensive_futility")
        trunc = _sum(rs, "deadline_truncated")
        deep = _sum(rs, "n_deepened")
        caps = sorted(str(r.get("max_depth")) for r in {json.dumps(x.get("max_depth")): x for x in rs}.values())
        realized = sorted({r.get("max_depth_realized") for r in rs}, key=lambda v: (v is None, v))
        pf, lf, uf = wilson(forced, handled)
        pu, lu, uu = wilson(fut, raced)
        print("")
        print("-- %s  (%d orientation-games)" % (name, len(rs)))
        print("   decisions %d  handled-by-strategy %d  raced %d" % (dec, handled, raced))
        print("   --max-depth cap %s   max_depth_realized observed %s   n_deepened TOTAL %d"
              % (caps, realized, deep))
        print("   GATE FORCED  %d/%d = %.4f [%.4f,%.4f]" % (forced, handled, pf, lf, uf))
        print("   futility     %d/%d of raced = %.4f [%.4f,%.4f]   (deadline_truncated %d)"
              % (fut, raced, pu, lu, uu, trunc))
        print("   separated %d/%d = %.4f   overruled %d/%d = %.4f"
              % (sep, raced, sep / raced, over, handled, over / handled))


def cell(p1: str, p3: str) -> None:
    a1, a3 = rows(Path(p1)), rows(Path(p3))
    print("=" * 78)
    print("FRESH PAIRED CELL — depth-1 control vs depth-3, matched seeded games")
    print("=" * 78)
    out: dict = {}
    for tag, rs in (("depth1", a1), ("depth3", a3)):
        searched = _sum(rs, "n_searched")
        changed = _sum(rs, "n_changed")
        deep = _sum(rs, "n_deepened")
        dec = _sum(rs, "n_decisions")
        md = [r.get("realized_mean", {}).get("depth", 0.0) for r in rs]
        bm = [r.get("realized_mean", {}).get("beam", 0.0) for r in rs]
        fb: dict = {}
        for r in rs:
            for k, v in (r.get("fallbacks") or {}).items():
                fb[k] = fb.get(k, 0) + v
        out[tag] = dict(games=len(rs), decisions=dec, searched=searched, changed=changed,
                        deepened=deep, fallbacks=fb,
                        mean_beam=sum(bm) / len(bm) if bm else 0.0,
                        mean_depth=sum(md) / len(md) if md else 0.0,
                        max_depth_realized=sorted({r.get("max_depth_realized") for r in rs},
                                                  key=lambda v: (v is None, v)))
        p, lo, hi = wilson(deep, searched)
        pc, lc, hc = wilson(changed, searched)
        out[tag]["deepen_rate"] = [p, lo, hi]
        out[tag]["change_rate"] = [pc, lc, hc]
        print("")
        print("-- %s: %d orientation-games, %d decisions, %d searched"
              % (tag, len(rs), dec, searched))
        print("   realized depth: mean %.3f   max observed %s   mean beam %.3f"
              % (out[tag]["mean_depth"], out[tag]["max_depth_realized"], out[tag]["mean_beam"]))
        print("   fallbacks: %s" % (dict(sorted(out[tag]["fallbacks"].items(),
                                                key=lambda kv: -kv[1])),))
        print("   DEEPENED (depth>=2)  %d/%d = %.4f [%.4f,%.4f]" % (deep, searched, p, lo, hi))
        print("   action CHANGED       %d/%d = %.4f [%.4f,%.4f]" % (changed, searched, pc, lc, hc))

    d, lo, hi = newcombe(out["depth3"]["changed"], out["depth3"]["searched"],
                         out["depth1"]["changed"], out["depth1"]["searched"])
    out["change_rate_diff"] = [d, lo, hi]
    print("")
    print("   DEPTH-ATTRIBUTABLE change rate (d3 - d1): %+.4f [%+.4f,%+.4f]" % (d, lo, hi))
    if out["depth3"]["changed"]:
        base = out["depth3"]["changed"] / out["depth3"]["searched"]
        out["depth_share_of_changes"] = d / base
        print("   as a SHARE of all depth-3 changes: %+.4f" % (d / base))

    def paired(rs):
        by: dict = {}
        for r in rs:
            if not r.get("finished") or r.get("error"):
                continue
            by.setdefault(r["game"], {})[r["orientation"]] = (
                0.5 if r.get("tied") else float(r.get("won", 0)))
        return {g: (v[0] + v[1]) / 2.0 for g, v in by.items() if 0 in v and 1 in v}

    q1, q3 = paired(a1), paired(a3)
    shared = sorted(set(q1) & set(q3))
    print("")
    print("   paired swap-pairs: depth1 %d, depth3 %d, shared %d" % (len(q1), len(q3), len(shared)))
    for tag, q in (("depth1", q1), ("depth3", q3)):
        v = list(q.values())
        m = sum(v) / len(v)
        se = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) * (len(v) - 1))) if len(v) > 1 else 0.0
        out[tag]["paired_win_rate"] = [m, m - 1.96 * se, m + 1.96 * se]
        print("   %s paired mirror win rate %.4f [%.4f,%.4f]  (null 0.50)"
              % (tag, m, m - 1.96 * se, m + 1.96 * se))
    if shared:
        dv = [q3[g] - q1[g] for g in shared]
        m = sum(dv) / len(dv)
        se = math.sqrt(sum((x - m) ** 2 for x in dv) / (len(dv) * (len(dv) - 1))) if len(dv) > 1 else 0.0
        out["depth_dividend_paired"] = [m, m - 1.96 * se, m + 1.96 * se, len(shared)]
        print("   DEPTH DIVIDEND (d3 - d1), paired on %d shared games: %+.4f [%+.4f,%+.4f]"
              % (len(shared), m, m - 1.96 * se, m + 1.96 * se))

    with open(Path(__file__).with_name("cell_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "committed":
        committed()
    else:
        cell(sys.argv[2], sys.argv[3])
