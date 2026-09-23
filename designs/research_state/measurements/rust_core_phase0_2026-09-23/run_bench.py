"""Phase-0 baseline launcher — runs a benchmark SCRIPT without importing a ``__main__`` module.

``search_decision_benchmark._pool`` (and through it ``profile_expand_many.py``) borrows the
honest arm's team pool from ``main.search_dividend.__main__``. The Phase-0 brief forbids importing
any ``__main__`` module, so this launcher registers a stub under that name carrying ONLY the one
function the benchmark reads — a verbatim copy of ``main.search_dividend.__main__._pool`` (same
facade, same deterministic ``Random(0)`` subsample) — and then runs the requested script as
``__main__`` through ``runpy``. Nothing else is changed; the benchmark code runs unmodified.

    export PYTHONPATH=$PYTHONPATH:src
    python3 designs/research_state/measurements/rust_core_phase0_2026-09-23/run_bench.py \
        src/main/search_dividend/search_decision_benchmark.py --traces … --roads view …
"""

from __future__ import annotations

import runpy
import sys
import types
from typing import List


def _pool(n: int) -> List[str]:
    # VERBATIM copy of main.search_dividend.__main__._pool (2026-09-23, da0ab0b0).
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder

    packed = list(Gen3Teambuilder(TeamLoader().get_all_teams()).packed_teams)
    if n and n < len(packed):
        import random as _r
        return _r.Random(0).sample(packed, n)
    return packed


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_bench.py <script.py> [script args…]")
    import main.search_dividend  # noqa: F401  — the parent package, so the stub has a home

    stub = types.ModuleType("main.search_dividend.__main__")
    stub._pool = _pool  # type: ignore[attr-defined]
    sys.modules["main.search_dividend.__main__"] = stub
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
