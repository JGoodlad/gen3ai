"""WARM the Rust binaries the routine tests resolve lazily — BEFORE the ship gate's pytest session starts.

A routine test that resolves a Rust binary (``resolve_*_bin``) builds it with ``cargo`` on first use IN ITS
OWN BODY: a fresh worktree or a moved ref pays ~20 s of incremental build inside whichever test is first on
each xdist worker, and the 8 s bar (``utils.ship_bar``) then FAILS a 2 s test for a cost that is the
environment's (2026-10-19: ``live_side_test::test_a_bot_client_never_forfeits_and_reports_no_argmax``, 22 s at
contention x2.42). Fixture scope cannot move a cost paid in a test body, so the gate builds first, before the
first test: the root conftest's ``pytest_sessionstart`` calls :func:`warm` when ``$GEN3AI_SHIP_WARM=1``
(``ship_gate.sh`` sets it), inside the gate's own lock and heavy-job slot; ``python -m utils.ship_gate_warm``
runs it by hand. It builds the SELF-CHECK profile — what every pytest session resolves
(``POKESIM_EMISSION_SELFCHECK=1``, the root conftest's default) — and is a no-op cargo build when the tree is
warm.

A build that fails is printed and counted, never raised past: the tests that need it report their own red.
Exit status: 0 = every binary resolved; 1 = at least one did not.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, List, Optional, Sequence, Tuple


def _targets() -> List[Tuple[str, Callable[[], object]]]:
    from utils.bridge import sim_bridge_bin as sb
    from utils.rust_env import build as env_build
    from utils.rust_env import bot_reader_bin

    return [("sim_bridge", sb.resolve_sim_bridge_bin),
            ("search_driver", sb.resolve_search_driver_bin),
            ("core_events", sb.resolve_core_events_bin),
            ("live_reader", sb.resolve_live_reader_bin),
            ("bot_reader", bot_reader_bin.resolve_bot_reader_bin),
            ("rust_env (selfcheck)", lambda: env_build.ensure_built("selfcheck"))]


def warm(targets: Optional[Sequence[Tuple[str, Callable[[], object]]]] = None, *,
         emit: Callable[[str], object] = print) -> int:
    """Resolve (build if needed) every target; returns the number that failed."""
    os.environ.setdefault("POKESIM_EMISSION_SELFCHECK", "1")
    failed = 0
    for name, fn in (_targets() if targets is None else targets):
        t0 = time.time()
        try:
            fn()
            emit(f"[ship_gate_warm] {name}: ready ({time.time() - t0:.1f} s)")
        except Exception as exc:                      # noqa: BLE001 — named, counted, never raised
            failed += 1
            emit(f"[ship_gate_warm] {name}: NOT BUILT ({time.time() - t0:.1f} s): {str(exc)[:300]}")
    return failed


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    return 1 if warm() else 0


if __name__ == "__main__":
    sys.exit(main())
