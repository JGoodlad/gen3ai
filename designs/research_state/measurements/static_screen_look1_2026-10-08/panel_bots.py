"""Static-token screen look 1: the SECONDARY panel's BOT component, at zero GPU cost (REPORTED, never gated).

§8.1's outside panel is (frozen pool, bots, SmallRL) with X5's HARM flag: static's panel point estimate worse than
legacy's by more than 2δ = 7 pp ⇒ the owner is told before adoption. This reads ONLY what the runs already played:
each run's LAST in-loop eval row in ``eval_results.jsonl`` (the 8 training bots + random, 100 games each, the run's
recorded eval regime) — no new game. The frozen-pool and SmallRL components are NOT read here (a common frozen pool
needs new h2h play and SmallRL a Metamon peer; both would cost more GPU than the brief's ~20 min allowance once the
engine compiles are counted), so the HARM flag here is on the bots component alone.

    python panel_bots.py [--models M] [--out panel_bots.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict

ARCHIVE_MODELS = Path("/home/goodlad/dev/gen3ai/models")
SEEDS = (1001, 1002, 1003)
HARM_PP = 7.0


def last_eval(run: Path) -> Dict[str, Any]:
    rows = [json.loads(x) for x in (run / "eval_results.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if isinstance(r.get("bots"), dict) and r["bots"]]
    r = max(rows, key=lambda r: r["step"])
    bots = {k: v for k, v in r["bots"].items() if k != "random"}
    return {"step": r["step"], "n_games_per_bot": r.get("n_games"), "evaluated_at": r.get("evaluated_at"),
            "bots_ex_random": bots, "bots_mean_pp": 100 * st.mean(bots.values()), "random": r["bots"].get("random")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ARCHIVE_MODELS))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "panel_bots.json"))
    a = ap.parse_args()
    out: Dict[str, Any] = {"arms": {}}
    for arm in ("static", "legacy"):
        runs = {f"s{s}": last_eval(Path(a.models) / f"rb_st_{arm}_s{s}") for s in SEEDS}
        out["arms"][arm] = {"runs": runs, "mean_pp": st.mean(r["bots_mean_pp"] for r in runs.values())}
    d = out["arms"]["static"]["mean_pp"] - out["arms"]["legacy"]["mean_pp"]
    out["static_minus_legacy_pp"] = d
    out["harm_flag"] = d < -HARM_PP
    out["note"] = "bots component only; frozen pool + SmallRL not read (see docstring)"
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for arm in ("static", "legacy"):
        A = out["arms"][arm]
        print(arm, round(A["mean_pp"], 2), {k: (v["step"], round(v["bots_mean_pp"], 2)) for k, v in A["runs"].items()})
    print("static - legacy", round(d, 2), "pp; HARM flag", out["harm_flag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
