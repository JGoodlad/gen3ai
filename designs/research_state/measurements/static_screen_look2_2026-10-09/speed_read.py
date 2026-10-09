"""Static-token screen look 2: the registered IN-ARM speed s (design_static_tokens.md §8.1 "Speed rule"), REPORTED.

Look 1's reader (``../static_screen_look1_2026-10-08/speed_read.py``) UNCHANGED — the same cycle definition, the same
exclusions in the same order (every child's first 11 updates, so every resume window; a non-advancing record; a
compile canary; an in-loop eval; the QUIET rule: no CPU-sampler row of ``<run>*.cpu.jsonl`` in the window, or a
contention factor ≥ 1.05) and the same rule (s ≤ 5 % / 5–15 % / > 15 %, rule 8) — over all FIVE seeds per arm.

    python speed_read.py [--models M] [--cpu-dir ~/.claude/jobs/st_screen] [--out speed.json]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "static_screen_look1_2026-10-08"))
import speed_read as SR  # noqa: E402

SR.SEEDS = (1001, 1002, 1003, 1004, 1005)

if __name__ == "__main__":
    if not any(a == "--out" or a.startswith("--out=") for a in sys.argv[1:]):
        sys.argv += ["--out", str(HERE / "speed.json")]
    raise SystemExit(SR.main())
