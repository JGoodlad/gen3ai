"""Static-token screen look 3: the SECONDARY panel's BOT component + the sentinel MONOTONICITY, at zero GPU cost
(REPORTED, never gated).

The bots component is look 1's reader (``../static_screen_look1_2026-10-08/panel_bots.py``) UNCHANGED — each run's
LAST in-loop eval row in ``eval_results.jsonl`` (the 8 training bots, 100 games each, the run's recorded eval regime;
random excluded), X5's HARM flag at static − legacy < −2δ = −7 pp on the bots component alone — over all EIGHT seeds
per arm. Added for look 3 (DESCRIPTIVE): the same row's sentinel MONOTONICITY, Kendall's τ over its self-play
sentinel win rates newest → oldest (``agents.training.selfplay_callback._monotonicity_score``'s formula, restated
here so the reader needs no import: +1 = every older sentinel is at least as easy as every newer one; below ~0.6 is
that function's "potential cycling" note). The frozen-pool and SmallRL components are not read (new play).

    python panel_bots.py [--models M] [--out panel_bots.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "static_screen_look1_2026-10-08"))
import panel_bots as PB  # noqa: E402

SEEDS = (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008)


def monotonicity(win_rates: List[float]) -> float:
    n = len(win_rates)
    if n < 2:
        return 1.0
    conc = sum(win_rates[i] <= win_rates[j] for i in range(n) for j in range(i + 1, n))
    return 2.0 * conc / (n * (n - 1) // 2) - 1.0


def last_row(run: Path) -> Dict[str, Any]:
    rows = [json.loads(x) for x in (run / "eval_results.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if isinstance(r.get("bots"), dict) and r["bots"]]
    r = max(rows, key=lambda r: r["step"])
    sent = r.get("sentinels") or []
    steps = [s["step"] for s in sent]
    if steps != sorted(steps, reverse=True):
        raise SystemExit(f"{run.name}: sentinels not newest-first: {steps}")
    wr = [s["win_rate"] for s in sent]
    return {"sentinel_steps": steps, "sentinel_win_rates": wr, "monotonicity": monotonicity(wr)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(PB.ARCHIVE_MODELS))
    ap.add_argument("--out", default=str(HERE / "panel_bots.json"))
    a = ap.parse_args()
    out: Dict[str, Any] = {"arms": {}}
    for arm in ("static", "legacy"):
        runs = {}
        for s in SEEDS:
            rd = Path(a.models) / f"rb_st_{arm}_s{s}"
            runs[f"s{s}"] = {**PB.last_eval(rd), **last_row(rd)}
        out["arms"][arm] = {"runs": runs, "mean_pp": st.mean(r["bots_mean_pp"] for r in runs.values()),
                            "monotonicity_mean": st.mean(r["monotonicity"] for r in runs.values())}
    d = out["arms"]["static"]["mean_pp"] - out["arms"]["legacy"]["mean_pp"]
    out["static_minus_legacy_pp"] = d
    out["harm_flag"] = d < -PB.HARM_PP
    out["note"] = "bots component only; frozen pool + SmallRL not read; monotonicity DESCRIPTIVE"
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for arm in ("static", "legacy"):
        A = out["arms"][arm]
        print(arm, round(A["mean_pp"], 2), "mono mean", round(A["monotonicity_mean"], 3))
        for k, v in A["runs"].items():
            print(f"  {k} step {v['step']} bots {v['bots_mean_pp']:.2f} mono {v['monotonicity']:.2f} "
                  f"sentinels {v['sentinel_win_rates']}")
    print("static - legacy", round(d, 2), "pp; HARM flag", out["harm_flag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
