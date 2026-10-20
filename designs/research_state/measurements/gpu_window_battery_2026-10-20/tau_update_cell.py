"""FP32_ALT LEARNER cell: the frozen GPU suite's `update_bf16` phase (the learner's real compiled `train()` at the
production shape, X26's training surface) with the τ solve FORCED to one arm -- ``--arm fp32_naive`` (the pre-T62
naive fp32 sum, "old") or ``--arm fp32_alt`` (FP32_ALT, "new"). The method is the 2026-10-19 battery's ``t62_u_*``
cells (`t62_update_cell.py`) with only the forcing changed: `tau_force` installs its module-global ``fixed_size_tau``
wrapper BEFORE anything is built or compiled. One arm per process, so each compiles its own program in its own
hermetic cache.

    python3 tau_update_cell.py --arm fp32_alt <cfg.json>   # a cfg written by battery.py (the suite's phase_config)
    python3 tau_update_cell.py --check                     # CPU: the API, the arms, the wrapper reaches the learner
    python3 tau_update_cell.py --check --prove             # + the CPU PROOF that both arms reach the same compiled path

After the phase, the cell adds a ``tau_force`` block to the phase's result json. The wrapper replaced mid-cell, or
never called, makes the result's status ``fail`` and the exit 4 (the battery HOLDs).

FAIRNESS (`learner_path.py`): both arms must run the IDENTICAL compiled learner path, differing only in the tau
solve. The suite's rule -- a REFUSED startup parity gate makes the cell "continue with install + prewarm only" -- would
give one arm a different path from the other's, so this cell records its path MARKERS (``path``) and FAILS (exit 8,
status ``fail``, the battery HOLDs) when the gate is not PASS (``--learner-gate on``, the default) or the compile trace
never reached the forced tau wrapper. ``--learner-gate off`` (``cfg["gate"]``) skips the gate by config in ALL FOUR
cells, so the paths are identical by construction. The gate compares no recorded golden output (it judges compiled vs
eager vs the fp32 reference live, on the golden's input rows); `--check` proves on the CPU that both arms reach the same
markers through the real `arm_compile_sentinel`.

Exit: the phase's code · 3 a precondition (the FP32_ALT API is absent) · 4 the forcing was violated · 8 the arm did
NOT run the production compiled path (the gate is not PASS / a path marker is missing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import learner_path as LP
import tau_force as TF

UPDATE_ARMS = ("fp32_naive", "fp32_alt")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=UPDATE_ARMS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--prove", action="store_true",
                    help="with --check: also run the real arm_compile_sentinel per arm on the CPU (~5 min) and assert "
                         "both arms reach the identical compiled-path markers (the battery's --dry-run does; the "
                         "window's own check does not, so the GPU is not idle for it)")
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
        reach = TF.cpu_reach_check(HS, UPDATE_ARMS, learner=True)
        rules = LP.selftest()
        proof = LP.cpu_path_proof(HS, TF, UPDATE_ARMS) if a.prove else "path proof not run (--prove)"
        print(f"ok: {reach}; {rules}; {proof}")
        return 0
    if not (a.arm and a.cfg):
        ap.error("--arm and cfg are required (or --check)")
    cfg = json.loads(Path(a.cfg).read_text())
    mode = str(cfg.get("gate", LP.GATE_ON))
    if mode not in LP.GATE_MODES:
        print(f"PRECONDITION FAILED: cfg gate {mode!r} is not one of {LP.GATE_MODES}", file=sys.stderr)
        return 3
    TF.install(HS, a.arm)                            # BEFORE anything is built or compiled
    from agents.training.gpu_bench_suite import phase_main
    rc = phase_main("update_bf16", a.cfg)
    rep = TF.report(HS, (a.arm,))
    out = Path(cfg["out"])
    try:
        res = json.loads(out.read_text())
    except (OSError, ValueError) as e:
        print(f"no readable phase result {out}: {e!r}", file=sys.stderr)
        return rc or 4
    res["tau_force"] = rep
    markers = LP.path_markers(res, mode, a.arm)
    unfair = LP.path_violations(markers) if res.get("status") == "ok" else []
    res["path"] = {"mode": mode, "arm": a.arm, "markers": markers, "violations": unfair}
    if rep["violations"]:
        res["status"] = "fail"
        res["reason"] = "; ".join(rep["violations"])
    elif unfair:
        res["status"] = "fail"
        res["reason"] = ("not the production compiled path (the other arm would differ): " + "; ".join(unfair)
                         + (" -- re-run the four learner cells with `--learner-gate off` for an identical path"
                            if mode == LP.GATE_ON else ""))
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1, default=str))
    tmp.replace(out)
    if rep["violations"]:
        print(res["reason"], file=sys.stderr)
        return 4
    if unfair:
        print(res["reason"], file=sys.stderr)
        return 8
    return rc


if __name__ == "__main__":
    sys.exit(main())
