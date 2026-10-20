"""THE FAIRNESS RULE OF THE LEARNER ABBA CELLS (staged 2026-10-20): both tau arms must run the IDENTICAL compiled
learner path, differing ONLY in the tau solve. Pure helpers (no torch import at module level) shared by
`tau_update_cell.py` (a cell judges ITS path), `read_battery.py` (the reader judges the four cells TOGETHER) and the
CPU proof (`cpu_path_proof`, run by `tau_update_cell.py --check`).

WHAT "THE PATH" IS. A learner cell runs the frozen suite's `update_bf16` phase (`gpu_bench_update._run_cell`), whose
compile stage is one of two branches:

* ``gate on`` (default; the launch's own path): `compile_trainer.arm_compile_sentinel` -- install R1 (fullgraph),
  the STARTUP PARITY GATE (compiled vs eager vs the fp32 reference, all computed LIVE in the process on the K9
  learner golden's real labelled ROWS), the prewarm, the lock, the canary. The gate compares NO recorded output of the
  golden: re-recording the golden under fp32_alt (d4af9f1e) changed no input to it. (The 2026-10-19 battery's
  fp32 arm, run against a golden recorded under fp64, PASSED its gate; `/home/goodlad/gen3ai_archive/
  gpu_battery_2026-10-19/cells/t62_u_old_*.json`.) But a verdict is not guaranteed for an arm whose compiled and eager
  programs disagree on a golden row, and the suite's rule is that a REFUSED gate makes the cell CONTINUE "with install
  + prewarm only" -- a DIFFERENT path from the other arm's. That is the unfairness this module closes: a cell whose
  gate is not PASS is a FAILED cell (the battery HOLDs), never a number.
* ``gate off`` (``battery.py --learner-gate off``): the gate is skipped by config for ALL FOUR cells and the cell
  installs + prewarms R1 the same way in each. Identical by construction; compile stays ON.

Either way the cell records its path MARKERS (`path_markers`), and `path_violations` / `fair_pair` turn a deviation
into a typed failure. A marker is a boolean read from the cell's own result, never a timing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: ``--learner-gate`` values.
GATE_ON, GATE_OFF = "on", "off"
GATE_MODES = (GATE_ON, GATE_OFF)

#: the gate's PASS-path lines (``compile_trainer.arm_compile_sentinel`` / ``compile_regions.gate_regions``) that must
#: appear in a ``gate on`` cell's recorded lines -- one per step of the production compiled path. Substrings, never
#: counts: a count such as "N graphs in process" is cumulative over a process and differs by cell for no reason.
LINE_MARKERS: dict[str, str] = {
    "region_installed": "installing ['R1_learner_micro_step']",
    "parity_pass": "parity PASS",
    "inventory_one_graph": "inventory == declaration: R1 1 graph",
    "prewarmed": "prewarmed 1 production signature",
    "canary_armed": "CompileCanary] armed",
}


def path_markers(res: Mapping[str, Any], mode: str, arm: str) -> dict[str, Any]:
    """The compiled-path markers of one learner cell's result (pure). ``res`` is the phase's result json with the
    cell's ``tau_force`` block; ``arm`` is the arm the cell forced."""
    gate = res.get("gate") or {}
    lines = "\n".join(str(x) for x in (gate.get("lines") or []))
    tf = res.get("tau_force") or {}
    traced = int((tf.get("trace_calls") or {}).get(arm, 0))
    m: dict[str, Any] = {"mode": mode, "gate_verdict": gate.get("verdict")}
    if mode == GATE_ON:
        for k, sub in LINE_MARKERS.items():
            m[k] = sub in lines
    m["tau_reached_by_the_compile_trace"] = traced > 0
    return m


def path_violations(markers: Mapping[str, Any]) -> list[str]:
    """Why a cell did NOT run the production compiled path (empty = it did). ``gate on`` demands a PASS verdict and
    every line marker; ``gate off`` demands the config skip; both demand the compile trace reached the tau wrapper."""
    mode = markers.get("mode")
    v: list[str] = []
    if mode == GATE_ON:
        if markers.get("gate_verdict") != "PASS":
            v.append(f"the startup parity gate is {markers.get('gate_verdict')!r}, not PASS: this arm would run "
                     "install + prewarm only while the other arm runs the gated path")
        v += [f"path marker {k} is False" for k in LINE_MARKERS if not markers.get(k)]
    elif mode == GATE_OFF:
        if markers.get("gate_verdict") != "SKIPPED":
            v.append(f"gate off but the verdict is {markers.get('gate_verdict')!r}, not SKIPPED")
    else:
        v.append(f"unknown gate mode {mode!r}")
    if not markers.get("tau_reached_by_the_compile_trace"):
        v.append("the compile trace never reached the tau wrapper (the compiled program is not the forced arm's)")
    return v


def fair_pair(markers: Sequence[Mapping[str, Any]]) -> list[str]:
    """Differences among several cells' markers (empty = every cell ran the identical path). The markers carry no
    arm-specific value, so any difference is a different path."""
    if not markers:
        return ["no cells"]
    first = dict(markers[0])
    out = []
    for i, m in enumerate(markers[1:], 1):
        diff = sorted(k for k in set(first) | set(m) if first.get(k) != m.get(k))
        if diff:
            out.append(f"cell {i} differs from cell 0 in {diff}: "
                       + ", ".join(f"{k}: {first.get(k)!r} vs {m.get(k)!r}" for k in diff))
    return out


def selftest() -> str:
    """Pure CPU check of the rules above on synthetic results (no torch): a PASS result is clean; a REFUSED gate, a
    missing line marker, a gate-mode mismatch and a never-traced tau each fail. Raises on any miss."""
    good_lines = [
        "[CompileRegions] installing ['R1_learner_micro_step'] (fullgraph=True, static shapes)",
        "[CompileRegions] parity PASS (bf16 trunk: the bf16 rule) -- R1 [trained weights]",
        "[CompileRegions] inventory == declaration: R1 1 graph (fullgraph=True)",
        "[CompileControl] prewarmed 1 production signature(s) [R1 learner_micro_step train/grad B=2048]: 3 graphs",
        "[CompileCanary] armed: R1 (the learner micro-step) compiled vs eager",
    ]

    def res(verdict: str, lines: list[str], traced: int, arm: str) -> dict[str, Any]:
        return {"gate": {"verdict": verdict, "lines": lines}, "tau_force": {"trace_calls": {arm: traced}}}

    a = path_markers(res("PASS", good_lines, 6, "fp32_naive"), GATE_ON, "fp32_naive")
    b = path_markers(res("PASS", good_lines[:3] + [good_lines[3].replace("3 graphs", "9 graphs")] + good_lines[4:],
                         2, "fp32_alt"), GATE_ON, "fp32_alt")
    assert not path_violations(a) and not path_violations(b), (a, b)
    assert not fair_pair([a, b, a, b]), fair_pair([a, b])
    refused = path_markers(res("REFUSED", good_lines[:1], 6, "fp32_naive"), GATE_ON, "fp32_naive")
    why = path_violations(refused)
    assert any("not PASS" in w for w in why) and any("parity_pass" in w for w in why), why
    assert fair_pair([a, refused]), "a REFUSED arm beside a PASS arm must differ"
    untraced = path_markers(res("PASS", good_lines, 0, "fp32_alt"), GATE_ON, "fp32_alt")
    assert any("compile trace" in w for w in path_violations(untraced)), path_violations(untraced)
    off = path_markers(res("SKIPPED", [], 3, "fp32_alt"), GATE_OFF, "fp32_alt")
    assert not path_violations(off), path_violations(off)
    assert fair_pair([a, off]), "gate on beside gate off must differ"
    bad_off = path_markers(res("PASS", good_lines, 3, "fp32_alt"), GATE_OFF, "fp32_alt")
    assert path_violations(bad_off), "gate off with a PASS verdict is a mode violation"
    return ("learner_path rules ok: PASS/PASS fair; REFUSED, missing marker, untraced tau, on-vs-off and "
            "off-with-PASS each fail")


def cpu_path_proof(HS: Any, tau_force: Any, arms: Sequence[str], batch: int = 16) -> str:
    """CPU PROOF that both arms reach the SAME compiled-path markers (the dry run's evidence; a CUDA verdict stays
    UNVERIFIED until the window). For each arm, on the cell's own learner (the seeded production-surface K9 build,
    bf16 trunk, the cell's argv) it runs the REAL `compile_trainer.arm_compile_sentinel` -- install, the gate on the
    golden's real rows, the prewarm, the lock, the canary -- with R1's backend ``eager`` (dynamo still TRACES the
    function through the forcing wrapper, `fullgraph=True`; no inductor build), records the sentinel's own lines, and
    asserts: the verdict is PASS, `path_markers` is violation-free, and the markers equal the other arm's. About 2.5
    min per arm at nice 19. ``tau_force`` is the forcing module (its wrapper is already installed by the caller)."""
    import shlex
    import time

    import torch

    from agents.model import compile_control as cc
    from agents.model import compile_regions as cr
    from agents.model import compile_trainer as ct
    from agents.training import learner_golden as LG
    from agents.training.gpu_bench_suite import PRODUCTION_ARGV
    from agents.training.gpu_bench_update import resolved_args

    orig_install = cr.install

    def eager_install(model: Any, **kw: Any) -> Any:
        kw.setdefault("backend", "eager")
        return orig_install(model, **kw)

    args = resolved_args(shlex.split(PRODUCTION_ARGV), "cpu")
    got: dict[str, dict[str, Any]] = {}
    walls: dict[str, float] = {}
    cr.install = eager_install                                   # the CPU stand-in for the inductor build
    try:
        for arm in arms:
            tau_force.set_arm(arm)
            cc._reset_control_for_tests()
            torch._dynamo.reset()
            model = LG.build_learner(args=args, trunk_precision="bf16")
            model.batch_size, model.grad_accum_steps = int(batch), 1
            lines: list[str] = []
            traced0 = int(tau_force.TRACED[arm])                 # the proof counts ITS OWN traces, not the check's
            t0 = time.perf_counter()
            try:
                ct.arm_compile_sentinel(model, batch_size=int(batch), emit=lines.append)
                verdict = "PASS"
            except ct.CompileTrainerError as exc:
                verdict, lines = "REFUSED", lines + [str(exc)[:500]]
            walls[arm] = time.perf_counter() - t0
            rep = tau_force.report(HS, (arm,))
            rep["trace_calls"] = {arm: int(tau_force.TRACED[arm]) - traced0}
            res = {"gate": {"verdict": verdict, "lines": [x for x in lines if "parity" in x.lower() or "R1" in x]},
                   "tau_force": rep}
            got[arm] = path_markers(res, GATE_ON, arm)
            cr.uninstall(model)
            cc._reset_control_for_tests()
            torch._dynamo.reset()
            torch._dynamo.config.error_on_recompile = False
    finally:
        cr.install = orig_install
    for arm, m in got.items():
        why = path_violations(m)
        if why:
            raise RuntimeError(f"arm {arm}: not the production compiled path on the CPU stand-in: {why}")
    diff = fair_pair([got[a] for a in arms])
    if diff:
        raise RuntimeError(f"the arms ran different compiled paths: {diff}")
    return (f"both arms reach the identical compiled path (arm_compile_sentinel, CPU eager-backend stand-in, B{batch}): "
            + "; ".join(f"{a} {', '.join(k for k in LINE_MARKERS)} all True, gate PASS in {walls[a]:.0f} s" for a in arms))
