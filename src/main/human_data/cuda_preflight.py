r"""X59-S's CUDA PRE-FLIGHT: the first GPU minutes, every check PASS or FAIL (PREREG Amendment 2, REVIEW E M7).

    GEN3AI_GPU_LEASE_TOKEN=<token> scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 16 <PY> \
        -m main.human_data.cuda_preflight --rows <r2>/rows --registration <results/registration.json> \
        --scratch <dir> --out <results/cuda_preflight.json> [--device cuda] [--arm human] [--students 20]

Run ONCE, before the first counted student, inside the GPU lease. Exit 0 iff every check passes; any FAIL stops the
window (nothing trains). The checks (``--arm``, S-H by default; the ≥ 1500 follow-up runs it on ``human_1500``,
Amendment 7; seed 5901, the registered recipe, scratch dirs only):

1. ``init``: seed 5901's ``init_sha256`` on the device equals the CPU-built digest.
2. ``parity``: :data:`STEPS` steps on the CPU and on the device from that init: the step-0 validation NLL agrees within
   :data:`TOL_VAL`, the losses at step 100 and :data:`STEPS` within :data:`TOL_LOSS`, the parameters' relative L2
   distance at :data:`STEPS` is below :data:`TOL_PARAM`.
3. ``precision``: inside the loop, matmul and cuDNN TF32 are off and the fp32 matmul precision is ``highest``.
4. ``resume``: 100 steps, then a resume to :data:`STEPS`: the step counter is :data:`STEPS`, the learning rate equals
   the uninterrupted run's, the loss at :data:`STEPS` agrees with it within :data:`TOL_LOSS`, and the resumed
   optimiser state is on the device.
5. ``export``: the device student (marked finished in a scratch copy) passes ``student.export``'s reload and
   anchors-bare checks, and its CPU logits agree with its device logits within :data:`TOL_LOGIT` on 256
   validation rows.
6. ``cost`` (reported, PASS iff measured): ms a step on the device, peak host RSS and peak device memory, and the
   window extrapolated to ``--students`` students (:data:`STUDENTS` by default; GPU-h) and to three concurrent students' host memory.
7. ``retry`` is the window runner's (``main.human_data.window``): it continues a student only on exit 3.
8. ``learns``: the device run continued to :data:`LONG` steps has a training loss below the uniform NLL.
"""
from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from main.human_data import rowset as RS
from main.human_data import student as ST

STEPS = 200
LONG = 1000
TOL_VAL = 1e-5
TOL_LOSS = 1e-3
TOL_PARAM = 1e-3
TOL_LOGIT = 1e-4
SEED = 5901
STUDENTS = 20         # 15 primary + 5 S-P0 (descriptive)


def _state(d: Path) -> Dict[str, Any]:
    import torch as th

    return th.load(d / "state.pt", weights_only=False, map_location="cpu")


def _rel_l2(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    import torch as th

    num = sum(float(((a[k].double() - b[k].double()) ** 2).sum()) for k in a if th.is_floating_point(a[k]))
    den = sum(float((a[k].double() ** 2).sum()) for k in a if th.is_floating_point(a[k]))
    return (num / max(den, 1e-300)) ** 0.5


def _run(rows: Path, out: Path, device: str, limit: int, reg: Optional[Dict[str, Any]], arm: str = "human") -> int:
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows_path.txt").write_text(str(rows.resolve()) + "\n")
    return ST.train(arm, rows, None, SEED, out, 3, 0, device=device, limit_steps=limit, registration=reg)


def _logits(sd: Dict[str, Any], rows: np.ndarray, mask: np.ndarray, device: str) -> np.ndarray:
    import torch as th

    from main.fresh_checkpoint import build_fresh_model
    from main.x50_r1.hom import policy_logits_t

    model, _, _ = build_fresh_model(SEED)
    pol = model.policy.to(device)
    pol.load_state_dict(sd)
    pol.eval()
    with th.no_grad():
        lg = policy_logits_t(pol, th.from_numpy(rows).to(device), th.from_numpy(mask.astype(np.float32)).to(device))
    return lg.float().cpu().numpy()


def preflight(rows: Path, scratch: Path, device: str, reg: Optional[Dict[str, Any]], steps: int = STEPS,
              long: int = LONG, arm: str = "human", students: int = STUDENTS) -> Dict[str, Any]:
    import torch as th

    from main.fresh_checkpoint import build_fresh_model
    from main.policy_spectrum.reader import inference_globals

    RS._guard_out(scratch)
    if scratch.exists():
        shutil.rmtree(scratch)
    checks: Dict[str, Dict[str, Any]] = {}
    with inference_globals(ST.THREADS):
        cpu_init = ST._state_digest(build_fresh_model(SEED)[0].policy.state_dict())
    rc = {"cpu": _run(rows, scratch / "cpu", "cpu", steps, reg, arm),
          "dev": _run(rows, scratch / "dev", device, steps, reg, arm)}
    c, d = _state(scratch / "cpu"), _state(scratch / "dev")
    checks["init"] = {"cpu": cpu_init, "device": d["init_sha256"], "pass": d["init_sha256"] == cpu_init}
    mid = str(min(100, steps))
    dl = {k: abs(c["loss_at"][k] - d["loss_at"][k]) for k in (mid, str(steps))}
    checks["parity"] = {"val0_abs": abs(c["hist"][0] - d["hist"][0]), "loss_abs": dl,
                        "param_rel_l2": _rel_l2(c["model"], d["model"]), "rc": rc}
    checks["parity"]["pass"] = (rc == {"cpu": 3, "dev": 3} and checks["parity"]["val0_abs"] <= TOL_VAL
                                and max(dl.values()) <= TOL_LOSS and checks["parity"]["param_rel_l2"] < TOL_PARAM)
    pf = d.get("precision", {})
    want = {"matmul_allow_tf32": False, "cudnn_allow_tf32": False, "float32_matmul_precision": "highest"}
    checks["precision"] = {**pf, "pass": pf == want if device.startswith("cuda") else bool(pf)}
    half = max(1, steps // 2)
    r1 = _run(rows, scratch / "resume", device, half, reg, arm)
    r2 = _run(rows, scratch / "resume", device, steps, reg, arm)
    r = _state(scratch / "resume")
    total = int(d["total"])
    on_dev = r.get("resumed_opt_devices", [])
    checks["resume"] = {"rc": [r1, r2], "step": r["step"], "lr": ST.lr_at(r["step"] - 1, total),
                        "lr_uninterrupted": ST.lr_at(d["step"] - 1, total),
                        "loss_abs": abs(r["loss_at"][str(steps)] - d["loss_at"][str(steps)])
                        if str(steps) in r.get("loss_at", {}) else None, "opt_devices": on_dev}
    checks["resume"]["pass"] = (r["step"] == d["step"] == steps and checks["resume"]["loss_abs"] is not None
                                and checks["resume"]["loss_abs"] <= TOL_LOSS and bool(on_dev)
                                and all(x.split(":")[0] == device.split(":")[0] for x in on_dev))
    ex = scratch / "export"
    ex.mkdir()
    shutil.copy(scratch / "dev" / "rows_path.txt", ex / "rows_path.txt")
    th.save({**d, "done": True}, ex / "state.pt")
    try:
        side = ST.export(ex)
        man, _ = RS.load_manifest(rows)
        probe = RS.load_shard(rows, man["val"], ("rows", "mask"))
        with inference_globals(ST.THREADS, device):
            a = _logits(d["model"], probe["rows"][:256], probe["mask"][:256], device)
        with inference_globals(ST.THREADS):
            b = _logits(d["model"], probe["rows"][:256], probe["mask"][:256], "cpu")
        fin = np.isfinite(a) & np.isfinite(b)
        gap = float(np.abs(a[fin] - b[fin]).max()) if fin.any() else float("inf")
        same_inf = bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
        checks["export"] = {"bare_load": side["anchors_bare_load_bit_equal"], "cpu_vs_device_logit_abs": gap,
                            "pass": bool(side["anchors_bare_load_bit_equal"] and same_inf and gap <= TOL_LOGIT)}
    except (ST.StudentError, RS.RowsetError) as exc:
        checks["export"] = {"error": str(exc), "pass": False}
    rl = _run(rows, scratch / "dev", device, long, reg, arm)
    dlong = _state(scratch / "dev")
    loss_long = dlong.get("loss_at", {}).get(str(long))
    checks["learns"] = {"rc": rl, "loss": loss_long, "uniform_val_nll": dlong.get("uniform_val_nll"),
                        "pass": loss_long is not None and loss_long < dlong.get("uniform_val_nll", -1)}
    ms = 1000 * dlong["train_s"] / max(dlong["step"], 1)
    gpu_peak = th.cuda.max_memory_allocated() / 2 ** 30 if device.startswith("cuda") else None
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    checks["cost"] = {"ms_per_step": round(ms, 2), "steps_per_student": total,
                      "students": students, "gpu_h_for_window_serial": round(students * total * ms / 3.6e6, 2),
                      "host_peak_gb": round(host_peak, 2), "device_peak_gb": gpu_peak,
                      "host_gb_three_concurrent_est": round(3 * host_peak, 1), "pass": ms > 0}
    return {"schema": "gen3_x59s_cuda_preflight_v1", "arm": arm, "device": device, "torch": th.__version__,
            "cuda_device": th.cuda.get_device_name(0) if device.startswith("cuda") else None,
            "rows": str(rows), "checks": checks, "pass": all(v["pass"] for v in checks.values())}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.human_data.cuda_preflight", description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", required=True, help="the registered human rowset's rows (S-H)")
    ap.add_argument("--registration", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", default="human", choices=("human",) + ST.FOLLOWUP_ARMS[:2],
                    help="the human-label arm whose registered rows it trains (human_1500: the follow-up, Amendment 7)")
    ap.add_argument("--students", type=int, default=STUDENTS, help="the window's student count (the cost extrapolation)")
    a = ap.parse_args(argv)
    reg = json.loads(Path(a.registration).read_text())
    try:
        res = preflight(Path(a.rows), Path(a.scratch), a.device, reg, arm=a.arm, students=a.students)
    except (ST.StudentError, RS.RowsetError) as exc:
        res = {"schema": "gen3_x59s_cuda_preflight_v1", "error": str(exc), "pass": False}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
