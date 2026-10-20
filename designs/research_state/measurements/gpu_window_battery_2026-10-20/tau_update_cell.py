"""FP32_ALT LEARNER cell: the frozen GPU suite's `update_bf16` phase (the learner's real compiled `train()` at the
production shape, X26's training surface) with the τ solve FORCED to one arm -- ``--arm fp32_naive`` (the pre-T62
naive fp32 sum, "old") or ``--arm fp32_alt`` (FP32_ALT, "new"). The method is the 2026-10-19 battery's ``t62_u_*``
cells (`t62_update_cell.py`) with only the forcing changed: `tau_force` installs its module-global ``fixed_size_tau``
wrapper BEFORE anything is built or compiled. One arm per process, so each compiles its own program in its own
hermetic cache.

    python3 tau_update_cell.py --arm fp32_alt <cfg.json>   # a cfg written by battery.py (the suite's phase_config)
    python3 tau_update_cell.py --check                     # CPU: the API, the arms, the wrapper reaches the learner

After the phase, the cell adds a ``tau_force`` block to the phase's result json. The wrapper replaced mid-cell, or
never called, makes the result's status ``fail`` and the exit 4 (the battery HOLDs).

The compile sentinel's startup parity gate reads the committed K9 golden: an arm whose solve differs from the golden's
may be REFUSED there, and the suite cell then CONTINUES with install + prewarm only (recorded as ``gate.verdict``) --
the timing is still the compiled update's.

Exit: the phase's code · 3 a precondition (the FP32_ALT API is absent) · 4 the forcing was violated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tau_force as TF

UPDATE_ARMS = ("fp32_naive", "fp32_alt")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=UPDATE_ARMS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("cfg", nargs="?")
    a = ap.parse_args(argv)

    from agents.model import hypothesis_set as HS
    bad = TF.check_api(HS)
    if bad:
        print("PRECONDITION FAILED: " + "; ".join(bad), file=sys.stderr)
        return 3
    if a.check:
        from agents.training import (
            gpu_bench_suite,  # noqa: F401 -- the phase worker imports
            gpu_bench_update,  # noqa: F401
        )
        print("ok: " + TF.cpu_reach_check(HS, UPDATE_ARMS, learner=True))
        return 0
    if not (a.arm and a.cfg):
        ap.error("--arm and cfg are required (or --check)")
    TF.install(HS, a.arm)                            # BEFORE anything is built or compiled
    from agents.training.gpu_bench_suite import phase_main
    rc = phase_main("update_bf16", a.cfg)
    rep = TF.report(HS, (a.arm,))
    out = Path(json.loads(Path(a.cfg).read_text())["out"])
    try:
        res = json.loads(out.read_text())
    except (OSError, ValueError) as e:
        print(f"no readable phase result {out}: {e!r}", file=sys.stderr)
        return rc or 4
    res["tau_force"] = rep
    if rep["violations"]:
        res["status"] = "fail"
        res["reason"] = "; ".join(rep["violations"])
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1, default=str))
    tmp.replace(out)
    if rep["violations"]:
        print(res["reason"], file=sys.stderr)
        return 4
    return rc


if __name__ == "__main__":
    sys.exit(main())
