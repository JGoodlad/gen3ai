"""Closing test §3.8: the SECONDARY panel's BOT component + sentinel MONOTONICITY (REPORTED, never gated).

The static screen's look-3 reader (``../../static_screen_look3_2026-10-09/panel_bots.py``) on the closing test's
sixteen runs: each run's LAST in-loop eval row in ``eval_results.jsonl`` (the 8 training bots, the run's recorded
eval regime; random excluded) and that row's sentinel monotonicity. §3.8's HARM note: the end state's bots mean below
production's by MORE than 4 pp (2 × the 2-pp floor) is told to the owner before adoption. No new game is played.

    python panel_bots.py [--models M] [--out panel_bots.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent.parent


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PB1 = _load("panel_bots_look1", MEAS / "static_screen_look1_2026-10-08" / "panel_bots.py")  # last_eval
PB3 = _load("panel_bots_look3", MEAS / "static_screen_look3_2026-10-09" / "panel_bots.py")  # last_row (monotonicity)

SEEDS = (2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008)
ARMS = {"endstate": "rb_es_end2_s{}", "production": "rb_es_prod_s{}"}
HARM_PP = 4.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(PB1.ARCHIVE_MODELS))
    ap.add_argument("--out", default=str(HERE / "panel_bots.json"))
    a = ap.parse_args()
    out: Dict[str, Any] = {"arms": {}}
    for arm, pat in ARMS.items():
        runs = {}
        for s in SEEDS:
            rd = Path(a.models) / pat.format(s)
            runs[f"s{s}"] = {**PB1.last_eval(rd), **PB3.last_row(rd)}
        out["arms"][arm] = {
            "runs": runs,
            "mean_pp": st.mean(r["bots_mean_pp"] for r in runs.values()),
            "monotonicity_mean": st.mean(r["monotonicity"] for r in runs.values()),
        }
    d = out["arms"]["endstate"]["mean_pp"] - out["arms"]["production"]["mean_pp"]
    out["endstate_minus_production_pp"] = d
    out["harm_flag"] = d < -HARM_PP
    out["note"] = "bots component only (8 training bots, random excluded); REPORTED, never gated (§3.8)"
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for arm in ARMS:
        A = out["arms"][arm]
        print(arm, round(A["mean_pp"], 2), "mono", round(A["monotonicity_mean"], 3),
              {k: (v["step"], round(v["bots_mean_pp"], 2)) for k, v in A["runs"].items()})
    print("endstate - production", round(d, 2), "pp; HARM flag", out["harm_flag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
