"""GPU REPLAY of the T2 trainee-load parity gate that stopped ``rb_x63u_s6302`` (2026-10-31 00:02:10, step 404,485,146).

The FATAL: ``ParityFailure: graph group=pool0 slot=24 bucket=256 rows=256: ... legal_logprob (max|delta| 5.19e-03 >
0.001)``, raised by ``InferenceService.load`` when ``RustCollector.after_update`` loaded the TRAINEE's post-update
weights into its T2 slot (24 of 31). The failing weights are the run's ``final_model_exception.zip`` (saved by the
exception handler 0.2 s before the crash log; this replay re-gates them, which is what confirms that).

CPU diagnosis (``cpu_tie_probe.py``, README): on fixture row 54 (and 55) the opponent's slot-5 mon has two candidate moves
(202, 280) whose composed move logits are EXACTLY equal (-1.1628687) and straddle the E4 cut K = 6 of the X5 move order
(`hypothesis_tokens.build_op_roster`'s `stable_order`). Which of the two is read inside the top 6 is decided by
rounding: a one-ulp relative weight perturbation swaps them and moves row 54's legal log-probs by 5.19e-03 / 5.20e-03 --
the gate's exact number. That is the TECH_DEBT P3 class (a discrete selection at a near-tie), which K9(b) excludes but
the T2 load gate does not.

THIS SCRIPT decides it ON THE GPU, at the pin's code (run it with ``PYTHONPATH=<a worktree at 39eaa326>/src``):

1. SERVICE: builds a T2 service of the run's shape (one group of ``--slots`` slots, the failing weights in every
   slot, ``--lanes`` lanes, buckets 8/64/256, the trainee slot uncapped and the others capped at 64, backend
   ``graph``), then ``load(trainee_slot, policy)`` -- the exact call that raised. ``ParityFailure`` at startup or at
   the load = REPRODUCED.
2. DIRECT: the same compiled ``decide`` (``compile_on``, ``dynamic=False``; no CUDA graph) vs the eager
   ``policy_reference`` at each bucket, per row: which fixture rows exceed the 1e-3 bar, by how much.
3. TIES: an eager forward that records every selection-boundary gap (``hypothesis_set.boundary_gap`` /
   ``order_gap``) over ALL opponent mons (the roster's own gap masks out non-addressable mons, which is how row 54
   slipped past ``near_tie_rows``), reduced per row.

VERDICT (deterministic, registered here before any GPU read): ``TIE CLASS`` iff the direct read has rows over the bar
and EVERY such row is one of the PREDICTED fixture rows 54 / 55; ``DIVERGENCE BEYOND THE TIE CLASS`` if any over-bar row
is another row; ``NOT REPRODUCED`` if no row is over the bar. Each over-bar row's selection gap is REPORTED, not judged:
the absolute gap is <= 2e-4 on about half the fixture rows (tiny presences), so it cannot attribute a row by itself. Run ``--ckpt`` on the failing weights and ``--control`` on
the last clean checkpoint (404,000,148), which is expected to PASS.

    PYTHONPATH=<pin>/src GEN3AI_GPU_LEASE_TOKEN=... scripts/ops/gpu_lock.sh \\
      <gen3ai_torch28 python> <this file> --run-dir ~/gen3ai_archive/x63u_s6302_parity_fatal_2026-10-31 \\
      --ckpt final_model_exception.zip --control checkpoint_404000148_steps.zip --out /tmp/x63u_replay.json

Not a registered read; a frozen record of one diagnosis. CPU-runnable for its test (``--device cpu --no-compile``).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

#: The gate's bar (``parity_probe.LOGPROB_BAR``), restated so the verdict is readable without the import.
BAR = 1e-3
#: The fixture holds 64 rows; a bucket of 256 cycles them four times.
N_FIXTURE = 64
#: THE PREDICTION (cpu_tie_probe.out.txt): the only fixture rows whose eager log-probs jump under one-ulp weight noise are
#: the two that carry the exactly tied move pair (202, 280) of the opponent's slot-5 mon at the cut K = 6.
PREDICTED_ROWS = frozenset({54, 55})

TIE_CLASS = "TIE CLASS"
BEYOND = "DIVERGENCE BEYOND THE TIE CLASS"
NOT_REPRODUCED = "NOT REPRODUCED"


def row_report(e_logp: torch.Tensor, s_logp: torch.Tensor, mask: torch.Tensor) -> List[Dict[str, Any]]:
    """Per row: max|served - eager| over LEGAL actions, the action, eager's log-prob there; worst first."""
    e = torch.where(mask, e_logp, torch.zeros_like(e_logp)).double()
    s = torch.where(mask, s_logp, torch.zeros_like(s_logp)).double()
    d = (s - e).abs()
    worst, act = d.max(-1)
    out = []
    for r in torch.argsort(worst, descending=True).tolist():
        a = int(act[r])
        out.append({"row": r, "fixture_row": r % N_FIXTURE, "max_dlogp": float(worst[r]), "action": a,
                    "eager_logp": float(e_logp[r, a])})
    return out


def attribute(rows: Sequence[Dict[str, Any]], gap_by_fixture_row: Sequence[float], bar: float = BAR,
              predicted: frozenset = PREDICTED_ROWS) -> Tuple[str, List[Dict[str, Any]]]:
    """The verdict and the over-bar rows (each with its selection gap, reported). Pure. "Over" is the gate's own
    test (``check_numerics``: ``not (err < bar)``), so a row AT the bar and a NaN are over."""
    over = [dict(r, gap=float(gap_by_fixture_row[r["fixture_row"]])) for r in rows if not (r["max_dlogp"] < bar)]
    if not over:
        return NOT_REPRODUCED, []
    if all(r["fixture_row"] in predicted for r in over):
        return TIE_CLASS, over
    return BEYOND, over


@contextlib.contextmanager
def recording_gaps(n_rows: int) -> Iterator[torch.Tensor]:
    """Patch ``hypothesis_set.boundary_gap`` / ``order_gap`` to fold every gap they return into a per-row minimum
    (a [B, ...] gap is reduced over its trailing dims; no addressable mask). Yields the [n_rows] float64 minimum."""
    import agents.model.hypothesis_set as hset

    acc = torch.full((n_rows,), math.inf, dtype=torch.float64)
    saved = {}

    def wrap(name: str) -> None:
        fn = getattr(hset, name)
        saved[name] = fn

        def rec(*a: Any, **k: Any) -> torch.Tensor:
            g = fn(*a, **k)
            if torch.is_tensor(g) and g.dim() >= 1 and g.shape[0] == n_rows:
                red = g.detach().reshape(n_rows, -1).double().cpu()
                red = torch.where(torch.isnan(red), torch.full_like(red, math.inf), red).amin(-1)
                torch.minimum(acc, red, out=acc)
            return g
        setattr(hset, name, rec)

    for name in ("boundary_gap", "order_gap"):
        if hasattr(hset, name):
            wrap(name)
    try:
        yield acc
    finally:
        for name, fn in saved.items():
            setattr(hset, name, fn)


def selection_gaps(policy: Any, obs: torch.Tensor, mask: torch.Tensor) -> List[float]:
    from agents.inference.service.decision import policy_reference

    with recording_gaps(obs.shape[0]) as acc:
        policy_reference(policy, obs, mask)
    return [float(x) for x in acc]


def direct_read(policy: Any, D: int, buckets: Sequence[int], device: torch.device,
                compile_it: bool) -> Dict[int, List[Dict[str, Any]]]:
    """``decide`` (compiled unless ``compile_it`` is False) vs ``policy_reference`` at each bucket."""
    from agents.inference.service.decision import DecisionModule, policy_reference
    from agents.inference.service.engine import decide
    from agents.inference.service.parity import fixture_rows

    fn = decide
    if compile_it:
        from agents.model.inductor_compile import compile_on
        fn = compile_on(decide, device=device, dynamic=False)
    mod = DecisionModule(policy)
    out = {}
    with torch.no_grad():
        for b in buckets:
            o, m = fixture_rows(D, b)
            obs, mask = torch.as_tensor(o, device=device), torch.as_tensor(m, device=device)
            for _ in range(2):
                s_logp = fn(mod, obs, mask)[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            e_logp, _ = policy_reference(policy, obs, mask)
            out[int(b)] = row_report(e_logp.cpu(), s_logp.cpu(), mask.cpu())
    return out


def service_read(policy: Any, *, device: str, backend: str, slots: int, lanes: int, trainee_slot: int,
                 buckets: Sequence[int], opp_cap: int) -> Dict[str, Any]:
    """T2 at the run's shape; then ``load(trainee_slot, policy)``. Reports PASS or the ParityFailure text."""
    from agents.inference.service import InferenceService, ParityFailure, ServiceSpec, SlotGroupSpec

    caps = tuple(max(buckets) if s == trainee_slot else opp_cap for s in range(slots))
    spec = ServiceSpec(groups=(SlotGroupSpec("pool0", slots, policy),), device=device, backend=backend,
                       buckets=tuple(buckets), lanes=lanes, slot_bucket_caps=caps,
                       max_rows_per_flush=max(1024, slots * max(buckets)))
    t0 = time.perf_counter()
    phase = "startup"
    try:
        svc = InferenceService(spec).startup()
        phase = "load"
        report = svc.load(trainee_slot, policy, "replay:final_model_exception")
        return {"status": "PASS", "phase": phase, "seconds": time.perf_counter() - t0, "report": report.line()}
    except ParityFailure as exc:
        return {"status": "ParityFailure", "phase": phase, "seconds": time.perf_counter() - t0, "message": str(exc)}


def replay_one(policy: Any, *, device: torch.device, args: argparse.Namespace) -> Dict[str, Any]:
    from agents.inference.service.parity import fixture_rows

    sp = policy.observation_space
    D = sp["observation"].shape[0] if hasattr(sp, "spaces") else sp.shape[0]
    o, m = fixture_rows(D, N_FIXTURE)
    gaps = selection_gaps(policy, torch.as_tensor(o, device=device), torch.as_tensor(m, device=device))
    direct = direct_read(policy, D, args.buckets, device, compile_it=not args.no_compile)
    verdicts = {}
    for b, rows in direct.items():
        v, over = attribute(rows, gaps)
        verdicts[b] = {"verdict": v, "over_bar": over[:12], "worst": rows[:3]}
    svc = None if args.no_service else service_read(
        policy, device=str(device), backend=args.backend, slots=args.slots, lanes=args.lanes,
        trainee_slot=args.trainee_slot, buckets=args.buckets, opp_cap=args.opp_cap)
    near = sorted(range(N_FIXTURE), key=lambda r: gaps[r])[:6]
    return {"direct": verdicts, "service": svc, "smallest_gaps": [(r, gaps[r]) for r in near]}


def overall(res: Dict[str, Any]) -> str:
    vs = {v["verdict"] for v in res["direct"].values()}
    if BEYOND in vs:
        return BEYOND
    if TIE_CLASS in vs:
        return TIE_CLASS
    return NOT_REPRODUCED


def _load(ckpt: str, run_dir: str, device: str) -> Any:
    from main.capacity import load_policy

    model, _, _ = load_policy(ckpt, run_dir, device)
    return model.policy.eval()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True, help="dir holding model_config.json (and the zips, if relative)")
    ap.add_argument("--ckpt", required=True, help="the failing weights (final_model_exception.zip)")
    ap.add_argument("--control", default=None, help="a clean checkpoint, expected to PASS")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="graph")
    ap.add_argument("--buckets", type=lambda s: tuple(int(x) for x in s.split(",")), default=(8, 64, 256))
    ap.add_argument("--slots", type=int, default=31)
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--trainee-slot", type=int, default=24)
    ap.add_argument("--opp-cap", type=int, default=64)
    ap.add_argument("--no-compile", action="store_true", help="direct read eager-vs-eager (the CPU test)")
    ap.add_argument("--no-service", action="store_true", help="skip the T2 service read")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    if device.type == "cuda" and not os.environ.get("GEN3AI_GPU_LEASE_TOKEN"):
        print("refusing: a CUDA replay runs only under the GPU lease (GEN3AI_GPU_LEASE_TOKEN, scripts/ops/gpu_lock.sh)")
        return 2
    results: Dict[str, Any] = {"bar": BAR, "predicted_rows": sorted(PREDICTED_ROWS)}
    for tag, path in (("ckpt", args.ckpt), ("control", args.control)):
        if path is None:
            continue
        full = path if os.path.isabs(path) else os.path.join(args.run_dir, path)
        res = replay_one(_load(full, args.run_dir, args.device), device=device, args=args)
        res["path"] = full
        res["overall"] = overall(res)
        results[tag] = res
        print(f"[{tag}] {os.path.basename(full)}: {res['overall']}; service: "
              f"{res['service'] and (res['service']['status'] + ' at ' + res['service']['phase'])}")
        for b, v in res["direct"].items():
            w = v["worst"][0]
            print(f"   bucket {b}: {v['verdict']}; worst row {w['row']} (fixture {w['fixture_row']}) "
                  f"max|dlogp| {w['max_dlogp']:.2e} action {w['action']} eager logp {w['eager_logp']:.3f}; "
                  f"over bar: {[(r['fixture_row'], round(r['max_dlogp'], 6), r['gap']) for r in v['over_bar']]}")
        if res["service"] and res["service"]["status"] != "PASS":
            print(f"   service message: {res['service']['message'][:300]}")
        print(f"   smallest selection gaps (fixture row, gap): {res['smallest_gaps']}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
