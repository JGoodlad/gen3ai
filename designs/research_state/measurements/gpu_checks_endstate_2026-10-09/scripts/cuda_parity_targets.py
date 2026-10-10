"""The R1 startup gate on CUDA for one configuration, with the PER-PARAMETER verdict of the NEW paths named
(gpu_checks_endstate, 2026-10-09). GPU: run under the lease, `scripts/ops/gpu_lock.sh`, GEN3AI_TEST_ALLOW_GPU=1.

    python cuda_parity_targets.py --cfg {P,R,E} --out results/parity_targets_<cfg>.json

The real launch's `[CompileRegions] parity PASS` line names only the WORST parameter. This runs the SAME gate
(`compile_regions.gate_regions`: R1 = the learner micro-step, train / grad, B = 2,048, the K9 golden's real
labelled rows, Inductor, fp32 'highest'; the fresh rung and the seeded-perturbation rung) on the seeded golden
learner built for the configuration's resolved namespace (`learner_golden.build_learner(args=...)`), and records
every parameter's compiled-vs-eager gradient relative error as the gate computes it
(`compile_trainer.per_param_grad_errors`, wrapped, never changed), so the verdict on the parameters of the extra
pre-LN trunk round, the noisy-OR / principled-op projection, the end-of-turn residual, the switch-hazard and the
other new paths is READ, not inferred from the worst line. A parameter whose gradient is below the gate's floor in
a rung is listed as NOT JUDGED in that rung (the gate's own rule), never as a pass.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

CFG = {
    "P": ["--arch", "production"],
    "R": ["--arch", "static_recovery"],
    "E": ["--arch", "static_recovery", "--move-resolution", "on", "--speed-physics", "on",
          "--no-value-threat-inject", "--op-reduction", "principled", "--obs-facts", "v1",
          "--allow-nonproduction-arch"],
}
#: parameter-name fragments of the paths this check exists for (matched case-insensitively)
TARGETS = r"extra_rounds|eot_residual|hazard|op_worst|noisy|principled|op_reduction|resolution|speed|" \
          r"obs_facts|actor_state"
B = 2048
# THIS checkout's src first: a script run by path gets its own dir as sys.path[0], and the editable install
# resolves `agents` to the MAIN checkout (whose code moves under a running diagnosis — it did, 2026-10-09).
sys.path.insert(0, os.environ.get("DIAG_SRC") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "src")))


def resolved_args(argv):
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    parser = build_parser()
    a = parser.parse_args(list(argv) + ["--device", "cuda", "--compile-trainer"])
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(a, parser)
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, choices=sorted(CFG))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch

    from agents.model import compile_control as cc
    from agents.model import compile_regions as cr
    from agents.model import compile_trainer as ct
    from agents.training import learner_golden as LG

    assert torch.cuda.is_available(), "needs the GPU (lease + GEN3AI_TEST_ALLOW_GPU=1)"
    args = resolved_args(CFG[a.cfg])
    t0 = time.perf_counter()
    model = LG.build_learner(args=args)
    dev = torch.device("cuda")
    model.policy.to(dev)
    model.device = dev
    torch.set_float32_matmul_precision("highest")
    model.batch_size = B
    fe = model.policy.features_extractor
    names = [n for n, _ in ct.grad_parameters(model, fe)]
    calls = []
    real = ct.per_param_grad_errors

    def wrapped(cg, eg, sizes, **kw):
        out = real(cg, eg, sizes, **kw)
        calls.append({i: float(e) for i, e in out})
        return out

    ct.per_param_grad_errors = wrapped          # read at call time by `_param_verdict`; restored below
    lines = []
    cc.control().install()
    cr.install(model)
    try:
        model.policy.set_training_mode(True)
        rules = cr.gate_regions(model, batch_size=B, say=lines.append)
        verdict = "PASS"
    except Exception as exc:  # noqa: BLE001 — the verdict IS the exception here, recorded whole
        rules, verdict = [repr(exc)[:4000]], "FAIL"
    finally:
        ct.per_param_grad_errors = real
        cr.uninstall(model)
    wall = time.perf_counter() - t0
    tgt = [i for i, n in enumerate(names) if re.search(TARGETS, n, re.I)]
    per_target = {}
    for i in tgt:
        per_target[names[i]] = [("NOT JUDGED (below the floor)" if i not in c else
                                 ("NaN" if math.isnan(c[i]) else round(c[i], 9))) for c in calls]
    judged = [len(c) for c in calls]
    worst = [max(c.values()) if c else None for c in calls]
    res = {"cfg": a.cfg, "argv": CFG[a.cfg], "verdict": verdict, "rules": rules, "gate_line": lines,
           "n_params": len(names), "rungs_judged_params": judged, "rungs_worst_rel_err": worst,
           "target_params": per_target, "wall_s": round(wall, 1), "torch": torch.__version__,
           "device": torch.cuda.get_device_name(0)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("cfg", "verdict", "n_params", "rungs_judged_params",
                                          "rungs_worst_rel_err", "wall_s")}))
    for n, v in per_target.items():
        print(f"  {n}: {v}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
