"""The `tau_cuda_parity` cell's PASS bars and verdict, in a module with NO torch import: the cell
(`tau_cuda_parity_cell.py`) judges its own record with `judge`, and `read_battery.py` re-judges the stored record with
the same function and prints `lines`. The bars are preregistered (2026-10-20, before any CUDA datum; the cell's docstring
and the README say why each number): a deterministic comparison of seeded inputs against fixed numbers.
"""
from __future__ import annotations

from typing import Any

BAR_TAU_ULPS = 4
BAR_LOGP_VIOLATORS = 1e-4
BAR_LOGP_FIXTURE_P99 = 1e-4


def judge(res: dict[str, Any]) -> list[str]:
    """The failures of a cell result against the preregistered bars (empty = PASS). Pure: the cell's `--prove` plants
    misses and the reader re-judges the stored record."""
    f: list[str] = []
    p = res.get("tau_pytest") or {}
    if p.get("max_ulps") is None:
        f.append(f"read 1 (pytest CUDA case) has no record: {p.get('why', 'not run')}")
    elif not (p["max_ulps"] <= BAR_TAU_ULPS):
        f.append(f"read 1: synthetic-row compiled-vs-eager tau max {p['max_ulps']:.2f} ulp > {BAR_TAU_ULPS}")
    if p.get("returncode") not in (None, 0):
        f.append(f"read 1: the pytest CUDA case exited {p['returncode']}")
    t = res.get("tau_captured") or {}
    if t.get("max_ulps") is None:
        f.append("read 3 (tau on the captured rows) has no record")
    elif not (t["max_ulps"] <= BAR_TAU_ULPS):
        f.append(f"read 3: captured-row compiled-vs-eager tau max {t['max_ulps']:.2f} ulp > {BAR_TAU_ULPS} "
                 f"(worst set {t.get('worst_set')})")
    viol = res.get("violators") or {}
    if not viol.get("dlogp"):
        f.append("read 2: no violator rows were measured")
    else:
        worst = max(viol["dlogp"])
        if not (worst <= BAR_LOGP_VIOLATORS):
            f.append(f"read 2: a violator's taken-action |dlogp| {worst:.3g} > {BAR_LOGP_VIOLATORS:g}")
    fx = (res.get("sets") or {}).get("fixture")
    if not fx:
        f.append("read 2: the fixture rows were not measured")
    elif not (fx["dlogp_p99"] <= BAR_LOGP_FIXTURE_P99):
        f.append(f"read 2: fixture p99 |dlogp| {fx['dlogp_p99']:.3g} > {BAR_LOGP_FIXTURE_P99:g}")
    for name, s in (res.get("sets") or {}).items():
        if not s.get("finite", False):
            f.append(f"set {name}: a non-finite value")
    tf = res.get("tau_force") or {}
    f += [f"tau forcing: {v}" for v in tf.get("violations", [])]
    if not tf:
        f.append("no tau_force record")
    return f


def verdict(res: dict[str, Any] | None) -> dict[str, Any]:
    """``{"verdict": PASS | FAIL | INCOMPLETE, "failures": [...]}``: INCOMPLETE when the cell did not finish (absent, a
    skip, still running, or a record that is not the cell's); the stored record is re-judged, never trusted."""
    if not res or res.get("cell") != "tau_cuda_parity":
        return {"verdict": "INCOMPLETE", "failures": ["tau_cuda_parity has no result"]}
    if res.get("status") not in ("ok", "fail"):
        return {"verdict": "INCOMPLETE", "failures": [f"status {res.get('status')!r}: {res.get('reason', '')}".strip()]}
    fails = judge(res)
    return {"verdict": "FAIL" if fails else "PASS", "failures": fails}


def _g(x: Any) -> str:
    return "n/a" if x is None else f"{x:.3g}"


def lines(res: dict[str, Any] | None) -> list[str]:
    """The reader's section: the verdict, the headline numbers and what the verdict means for the cost read."""
    v = verdict(res)
    if v["verdict"] == "INCOMPLETE":
        return [f"TAU ON CUDA (REVIEW F2): INCOMPLETE -- {'; '.join(v['failures'])} -- FP32_ALT's 'two programs land on "
                "one tau' is UNMEASURED on the production pair, so the cost verdict below is a cost only"]
    assert res is not None
    p, t, viol, sets = res["tau_pytest"], res["tau_captured"], res["violators"], res["sets"]
    out = [f"TAU ON CUDA (REVIEW F2): {v['verdict']} -- compiled Triton vs eager tau, max |d| synthetic rows "
           f"{_g(p.get('max_abs_dtau'))} ({_g(p.get('max_ulps'))} ulp), captured rows {_g(t.get('max_abs'))} "
           f"({_g(t.get('max_ulps'))} ulp, p99 {_g(t.get('p99_abs'))}, {t.get('live_rows')} live rows; bar "
           f"{BAR_TAU_ULPS} ulp); taken-action |d log pi| compiled rollout vs learner: violators "
           f"{', '.join(f'{n} {x:.2g}' for n, x in zip(viol['set'], viol['dlogp']))} (bar {BAR_LOGP_VIOLATORS:g}), fixture "
           f"max {_g(sets['fixture']['dlogp_max'])} p99 {_g(sets['fixture']['dlogp_p99'])} (p99 bar "
           f"{BAR_LOGP_FIXTURE_P99:g}; max reported, not judged) on {sets['fixture']['rows']} rows"]
    if v["failures"]:
        out.append("  FAILED: " + "; ".join(v["failures"]) + " -- the battery HELD here (exit 9)")
    return out


def selftest() -> str:
    good = {"cell": "tau_cuda_parity", "status": "ok", "tau_pytest": {"max_ulps": 0.0, "max_abs_dtau": 0.0, "returncode": 0},
            "tau_captured": {"max_ulps": 1.0, "max_abs": 1e-6, "p99_abs": 0.0, "live_rows": 900, "worst_set": "fixture"},
            "violators": {"set": ["u1", "u2", "u3"], "dlogp": [1e-5, 2e-5, 3e-5]},
            "sets": {"fixture": {"dlogp_p99": 2e-5, "dlogp_max": 4e-5, "rows": 915, "finite": True}},
            "tau_force": {"violations": []}}
    assert verdict(good)["verdict"] == "PASS" and "PASS" in lines(good)[0]
    bad = {**good, "violators": {"set": ["u1"], "dlogp": [2e-4]}}
    assert verdict(bad)["verdict"] == "FAIL" and any("FAILED" in x for x in lines(bad))
    assert verdict(None)["verdict"] == "INCOMPLETE" and verdict({**good, "status": "skip"})["verdict"] == "INCOMPLETE"
    return "tau parity: PASS on a clean record, FAIL on a planted violator, INCOMPLETE when absent or skipped"
