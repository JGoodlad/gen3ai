"""Print a starvation_readout JSON's levels and paired deltas as compact tables (decisive turns unless --all).

    python designs/research_state/measurements/collapse_pilot_2026-10-17/show_readout.py <readout.json> [--all]
"""
import json
import sys

KEYS = ("starved", "tie", "misrank", "gain", "imp", "greedy_regret", "dom_mass", "entropy")


def fmt(x: dict) -> str:
    ci = x.get("ci")
    v = x.get("mean", x.get("delta"))
    return f"{v:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else f"{v:+.3f}"


def main() -> int:
    with open(sys.argv[1]) as f:
        res = json.load(f)
    scope = "all" if "--all" in sys.argv else "decisive"
    for key, block in res["by_rows"].items():
        print(f"\n== {key} ({scope})")
        for lab, lv in block["levels"].items():
            d = lv[scope]
            print(f"  {lab:>10} n={d['turns']:4d} " + " ".join(f"{k}={d[k]['mean']:.3f}" for k in KEYS)
                  + f" gain/starved={d['gain_per_starved_turn']['mean']}")
        for pair, pv in block["paired"].items():
            d = pv[scope]
            print(f"  {pair:>22}: " + " | ".join(f"{k} {fmt(d[k])}" for k in ("starved", "misrank", "tie", "dom_mass",
                                                                             "greedy_regret", "gain")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
