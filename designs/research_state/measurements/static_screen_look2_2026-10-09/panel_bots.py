"""Static-token screen look 2: the SECONDARY panel's BOT component at zero GPU cost (REPORTED, never gated).

Look 1's reader (``../static_screen_look1_2026-10-08/panel_bots.py``) UNCHANGED — each run's LAST in-loop eval row in
``eval_results.jsonl`` (the 8 training bots, 100 games each, the run's recorded eval regime; random excluded), X5's
HARM flag at static − legacy < −2δ = −7 pp on the bots component alone — over all FIVE seeds per arm. The frozen-pool
and SmallRL components are not read (new play; see look 1's docstring).

    python panel_bots.py [--models M] [--out panel_bots.json]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "static_screen_look1_2026-10-08"))
import panel_bots as PB  # noqa: E402

PB.SEEDS = (1001, 1002, 1003, 1004, 1005)

if __name__ == "__main__":
    if not any(a == "--out" or a.startswith("--out=") for a in sys.argv[1:]):
        sys.argv += ["--out", str(HERE / "panel_bots.json")]
    raise SystemExit(PB.main())
