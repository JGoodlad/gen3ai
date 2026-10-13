"""THE CUDA-BUILD VALIDATION — torch 2.8.0+cu126 against +cu128 on the LOCAL card (owner, 2026-10-12: "PyTorch update").

WHY. The shared env `gen3ai_torch28` is torch 2.8.0+cu126, whose wheels carry no sm_120 (Blackwell: RTX 5090 / 5070 Ti)
kernels. The CUDA 12.8 twin `gen3ai_torch28_cu128` (`environment_torch28_cu128.yml`) has the SAME packages except the
CUDA build, and the planned multi-card cloud benchmark (`designs/ops/cloud_gpu_benchmark.md`) runs on +cu128. Only a
local run on BOTH builds separates "the card" from "the CUDA build" — this tool does that run on the RTX 3080 Ti.

WHAT IT MEASURES (all on the GPU, under the lease, through `scripts/ops/gpu_lock.sh` and `mem_cap.sh`):

* (a) UPDATE CELLS — per architecture (`E` the end state, `P` production; the perf phase's `configs.py` argvs, ride-along
  flags and all), per trunk precision (fp32, bf16) and per build: the perf phase's `update_bench.py`, i.e. the real
  `train()` of a learner built for the trainer argv at the production shape, compiled exactly as a launch compiles it
  (R1 `fullgraph`, the startup PARITY GATE, the prewarm). Recorded: the gate's verdict, `update_s_est` (= `train_ms`),
  the marginal epoch, peak reserved memory, compile + gate + prewarm seconds, the walls' spread, the load average and
  every other GPU process. Each cell is its own process, so a cu126 cell and a cu128 cell never share a compile cache
  (the hermetic cache is stamped with the torch build).
* (b) NUMERICS CELLS — per precision and build: ONE eager micro-batch (the K9 learner golden's real labelled rows, the
  seeded learner, `matmul precision highest`), loss and the flat gradient, repeated three times so the SAME-build
  run-to-run noise is on the page next to the CROSS-build difference: loss relative difference, gradient relative L2,
  cosine, the worst parameters. A report, never a verdict (a bar would need the derivation the R1 gate's bars got).
  The two builds start from byte-identical weights (the seeded CPU build) or the comparison is marked INVALID.
* (c) THE BASELINE the cloud ladder compares against = (a)'s table.

    scripts/ops/cuda_build_validate.sh plan                      # the cells, their commands and the time; touches nothing
    scripts/ops/cuda_build_validate.sh cpu-smoke --out /tmp/x    # the CODE PATH on CPU in both envs (no GPU, no lease);
    #   measured 2026-10-12 on 3 busy cores: a numerics cell ~270-290 s, an end-state update cell ~20 min PER train() call
    #   (use `--only numerics`, or `--archs P`, for a quick look)
    GEN3AI_GPU_LEASE_TOKEN=<token> scripts/ops/cuda_build_validate.sh run
    scripts/ops/cuda_build_validate.sh summarize --out <dir>     # re-render SUMMARY.md / summary.json from the JSONs

INCREMENTAL AND RESUMABLE: a finished cell (its JSON carries the result) is skipped on the next `run`, so a window cut
short loses only its tail; `--force` redoes. NUMERICS FIRST (~2 min each): a cu128 build that cannot run on this card at
all shows up before the first 7-minute compile. A BENCHMARK WARNS, IT NEVER STRETCHES: nothing is rescaled; the load and
the other GPU processes are recorded.

The GPU is LEASED (root CLAUDE.md, standing rule 5): `run` REFUSES without `GEN3AI_GPU_LEASE_TOKEN` (or its `_FILE`), and
every GPU cell goes through `scripts/ops/gpu_lock.sh`, which itself refuses (exit 5 / 6) when someone else holds the card.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.paths import repo_path

ENV_NAMES = {"cu126": "gen3ai_torch28", "cu128": "gen3ai_torch28_cu128"}
#: the torch BUILD each env must carry (a wrong env is a preflight refusal, never a silent comparison of one build with itself)
EXPECTED_BUILD = {"cu126": "+cu126", "cu128": "+cu128"}
BUILDS = ("cu126", "cu128")
PRECISIONS = ("fp32", "bf16")
#: the perf phase's configs (`configs.py`): E = the end state, P = production; ride-along flags included
ARCHS = ("E", "P")
BENCH_DIR = ("designs", "research_state", "measurements", "perf_phase_2026-10-10", "scripts")
#: minutes: a cold compile + the gate + prewarm (~5 min measured for E on this card) + the timed calls; numerics need no compile
EST_UPDATE_MIN, EST_NUMERICS_MIN = 8.0, 2.0
NOISE_REPS = 3
EXIT_USAGE = 2


@dataclasses.dataclass(frozen=True)
class Cell:
    name: str
    kind: str            # "numerics" | "update"
    build: str           # cu126 | cu128
    prec: str            # fp32 | bf16
    arch: str            # E | P
    out_json: str


# --------------------------------------------------------------------------------------- planning

def default_out_dir() -> str:
    return str(repo_path("designs", "research_state", "measurements", f"cuda_build_{time.strftime('%Y-%m-%d')}"))


def plan_cells(archs: Sequence[str], precs: Sequence[str], builds: Sequence[str], out_dir: str) -> List[Cell]:
    """NUMERICS first (cheap: a build that cannot run here shows up in minutes), then the update cells. Within a
    pair the two builds ALTERNATE which goes first, so a slow drift of the card (thermals, clocks) does not always
    land on the same build."""
    cells: List[Cell] = []
    flip = 0
    for prec in precs:
        order = list(builds) if flip % 2 == 0 else list(reversed(builds))
        flip += 1
        for b in order:
            cells.append(Cell(f"numerics_{prec}_{b}", "numerics", b, prec, "-",
                              os.path.join(out_dir, "results", f"numerics_{prec}_{b}.json")))
    for arch in archs:
        for prec in precs:
            order = list(builds) if flip % 2 == 0 else list(reversed(builds))
            flip += 1
            for b in order:
                cells.append(Cell(f"update_{arch}_{prec}_{b}", "update", b, prec, arch,
                                  os.path.join(out_dir, "results", f"update_{arch}_{prec}_{b}.json")))
    return cells


def load_configs() -> Dict[str, str]:
    """The perf phase's trainer argvs by tag (`configs.py`: stdlib-only at import, so no torch is loaded)."""
    path = repo_path(*BENCH_DIR, "configs.py")
    spec = importlib.util.spec_from_file_location("perf_phase_configs", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    configs: Dict[str, str] = dict(mod.CONFIGS)
    return configs


def sibling_python(env_name: str, prefix: Optional[str] = None) -> str:
    """`<conda envs dir>/<env_name>/bin/python3`, the envs dir found from the running interpreter's own prefix
    (`…/envs/<x>` -> `…/envs`; a base install -> `<base>/envs`) — names, never a machine path."""
    p = os.path.abspath(prefix or sys.prefix)
    parent = os.path.dirname(p)
    envs = parent if os.path.basename(parent) == "envs" else os.path.join(p, "envs")
    return os.path.join(envs, env_name, "bin", "python3")


def lease_token_present(env: Optional[Dict[str, str]] = None) -> bool:
    e = os.environ if env is None else env
    return bool((e.get("GEN3AI_GPU_LEASE_TOKEN") or "").strip() or (e.get("GEN3AI_GPU_LEASE_TOKEN_FILE") or "").strip())


def cell_argv(cell: Cell, *, pythons: Dict[str, str], configs: Dict[str, str], args: argparse.Namespace,
              cpu: bool) -> List[str]:
    """The command (argv list) of one cell, WITHOUT the gpu_lock / mem_cap / timeout wrappers."""
    py = pythons[cell.build]
    src = str(repo_path("src"))
    env = ["env", f"PYTHONPATH={src}", f"DIAG_SRC={src}"]
    if cell.kind == "numerics":
        return env + [py, "-m", "main.ops.cuda_build_validate", "numerics-worker", "--build", cell.build,
                      "--prec", cell.prec, "--argv", configs["E"], "--device", "cpu" if cpu else "cuda",
                      "--out", cell.out_json]
    cmd = env + [py, str(repo_path(*BENCH_DIR, "update_bench.py")), "--tag", cell.name, "--argv", configs[cell.arch],
                 "--trunk-precision", cell.prec, "--epochs", args.epochs, "--warm", str(args.warm),
                 "--reps", str(args.reps), "--out", cell.out_json]
    return cmd + (["--cpu-smoke"] if cpu else [])


def wrap(inner: List[str], *, cpu: bool, args: argparse.Namespace) -> List[str]:
    """`[taskset] [nice] [gpu_lock.sh] mem_cap.sh <GB> timeout <s> <inner>` — the lock innermost-but-one so the cell's
    wall timeout counts only the command (a timeout goes INSIDE, `gpu_lock.sh --help`)."""
    out: List[str] = []
    if args.cpus:
        out += ["taskset", "-c", args.cpus]
    if args.nice:
        out += ["nice", "-n", str(args.nice)]
    if not cpu:
        out += [str(repo_path("scripts", "ops", "gpu_lock.sh"))]
    out += [str(repo_path("scripts", "ops", "mem_cap.sh")), str(args.mem_gb), "timeout", str(int(args.cell_timeout_min * 60))]
    return out + inner


def is_done(cell: Cell) -> bool:
    """A cell is finished when its JSON carries the RESULT (update_bench saves progressively, so a cut-off run leaves a
    partial file that must be redone)."""
    try:
        with open(cell.out_json, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return False
    return ("train" in d) if cell.kind == "update" else bool(d.get("done"))


# --------------------------------------------------------------------------------------- the numerics worker

def _rel_l2(a: "Any", b: "Any") -> float:
    num = float((a.double() - b.double()).norm())
    den = float(a.double().norm())
    return num / den if den > 0 else (0.0 if num == 0 else math.inf)


def resolved_args(argv: Sequence[str], device: str) -> Any:
    """The trainer argv resolved like a launch (as `update_bench.resolved_args`)."""
    import contextlib
    import io

    from main.train.config import resolve_config
    from main.train.parser import build_parser
    p = build_parser()
    a = p.parse_args(list(argv) + ["--device", device] + (["--compile-trainer"] if device == "cuda" else []))
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(a, p)
    return a


def numerics_worker(a: argparse.Namespace) -> int:
    """ONE eager micro-batch, loss + flat gradient, ``NOISE_REPS`` times, saved for the cross-build compare."""
    import torch

    from agents.model import compile_regions as CR
    from agents.model import compile_trainer as CT
    from agents.training import learner_golden as LG
    from agents.training.instrumented_ppo.micro_step import micro_step

    cpu = a.device == "cpu"
    torch.set_float32_matmul_precision("highest")
    res: Dict[str, Any] = {"build": a.build, "prec": a.prec, "device": a.device, "torch": torch.__version__,
                           "cuda": torch.version.cuda, "argv": a.argv, "done": False, "load1_start": os.getloadavg()[0]}
    if not cpu:
        assert torch.cuda.is_available(), "needs the GPU (lease + gpu_lock)"
        torch.cuda.set_device(0)
        res["gpu"] = torch.cuda.get_device_name(0)
        res["capability"] = list(torch.cuda.get_device_capability(0))
        res["arch_list"] = torch.cuda.get_arch_list()
    t0 = time.perf_counter()
    args = resolved_args(shlex.split(a.argv), a.device)
    model = LG.build_learner(args=args)
    res["init_params_sha256"] = LG.params_sha256(model)        # CPU bytes, before the move: must match across builds
    dev = torch.device(a.device)
    model.policy.to(dev)
    model.device = dev
    model.batch_size = 256 if cpu else int(args.batch_size)
    if a.prec != "fp32":
        from agents.model.trunk_precision import set_trunk_precision
        res["trunk_rounds_set"] = set_trunk_precision(model.policy, a.prec)
    batch = CR.r1_batch(model, model.batch_size)
    res["rows"], res["batch_source"] = int(model.batch_size), batch.source
    call_args = CR._r1_args(model, batch)
    names = [n for n, _ in CT.grad_parameters(model, model.policy.features_extractor)]
    arms = []
    for _ in range(NOISE_REPS):
        arms.append(CR._r1_arm(model, micro_step, call_args))
        if not cpu:
            torch.cuda.synchronize()
    losses = [float(x["loss"][0]) for x in arms]
    res["losses"] = losses
    res["grad_norm"] = float(arms[0]["grad"].double().norm())
    res["finite"] = bool(all(bool(torch.isfinite(x["grad"]).all()) and math.isfinite(float(x["loss"][0])) for x in arms))
    # SAME-build run-to-run noise (atomics, algorithm choice): the floor the cross-build difference is read against
    res["noise_grad_rel_l2"] = max(_rel_l2(arms[0]["grad"], x["grad"]) for x in arms[1:])
    res["noise_loss_rel"] = max(abs(losses[0] - v) / max(abs(losses[0]), 1e-30) for v in losses[1:])
    res["seconds"] = time.perf_counter() - t0
    res["load1_end"] = os.getloadavg()[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"loss": arms[0]["loss"].cpu(), "grad": arms[0]["grad"].cpu(), "sizes": arms[0]["grad_sizes"].cpu(),
                "names": names}, a.out[:-5] + ".pt")
    res["done"] = True
    with open(a.out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    os.replace(a.out + ".tmp", a.out)
    print(f"[numerics {a.build}/{a.prec}] loss {losses[0]:.9g} grad_norm {res['grad_norm']:.6g} noise "
          f"{res['noise_grad_rel_l2']:.3g} ({res['seconds']:.0f}s)", flush=True)
    return 0


def compare_numerics(p126: str, p128: str) -> Dict[str, Any]:
    """The cross-build difference of two numerics cells (`.json` paths; the `.pt` tensors sit beside them)."""
    import torch
    with open(p126, encoding="utf-8") as fh:
        j6 = json.load(fh)
    with open(p128, encoding="utf-8") as fh:
        j8 = json.load(fh)
    out: Dict[str, Any] = {"prec": j6["prec"], "torch": [j6["torch"], j8["torch"]]}
    if j6["init_params_sha256"] != j8["init_params_sha256"]:
        out["valid"] = False
        out["why"] = "the two builds did NOT start from byte-identical weights (init_params_sha256 differs)"
        return out
    t6 = torch.load(p126[:-5] + ".pt")
    t8 = torch.load(p128[:-5] + ".pt")
    g6, g8 = t6["grad"].double(), t8["grad"].double()
    l6, l8 = float(t6["loss"][0]), float(t8["loss"][0])
    per: List[Tuple[float, str]] = []
    for name, c6, c8 in zip(t6["names"], g6.split([int(s) for s in t6["sizes"]]), g8.split([int(s) for s in t8["sizes"]])):
        if float(c6.norm()) > 0 or float(c8.norm()) > 0:
            per.append((_rel_l2(c6, c8), name))
    per.sort(reverse=True)
    cos = float((g6 * g8).sum() / (g6.norm() * g8.norm())) if float(g6.norm()) > 0 and float(g8.norm()) > 0 else None
    out.update({
        "valid": True, "loss": [l6, l8], "loss_rel": abs(l6 - l8) / max(abs(l6), 1e-30),
        "grad_rel_l2": _rel_l2(g6, g8), "grad_cosine": cos, "bitwise_identical": bool(torch.equal(t6["grad"], t8["grad"])),
        "param_rel_l2_median": per[len(per) // 2][0] if per else None,
        "param_rel_l2_worst": [{"name": n, "rel_l2": v} for v, n in per[:5]],
        "noise_grad_rel_l2": [j6["noise_grad_rel_l2"], j8["noise_grad_rel_l2"]],
        "noise_loss_rel": [j6["noise_loss_rel"], j8["noise_loss_rel"]],
        "finite": bool(j6["finite"] and j8["finite"])})
    return out


# --------------------------------------------------------------------------------------- the summary

def _spread(walls: Dict[str, List[float]]) -> Optional[float]:
    """(max - min) / median of the highest epoch count's timed walls: the cell's own repeat noise."""
    if not walls:
        return None
    key = max(walls, key=lambda k: int(k))
    v = sorted(walls[key])
    if len(v) < 2 or v[len(v) // 2] <= 0:
        return None
    return (v[-1] - v[0]) / v[len(v) // 2]


def _read(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            d: Dict[str, Any] = json.load(fh)
        return d
    except (OSError, ValueError):
        return None


def summarize(out_dir: str, archs: Sequence[str], precs: Sequence[str]) -> Dict[str, Any]:
    """Everything the cells recorded, as one dict: ``update`` rows per (arch, prec) with the cu128 / cu126 ratios, and
    ``numerics`` per precision with the cross-build compare."""
    res = os.path.join(out_dir, "results")
    summary: Dict[str, Any] = {"out_dir": out_dir, "update": {}, "numerics": {}}
    for arch in archs:
        for prec in precs:
            row: Dict[str, Any] = {}
            for b in BUILDS:
                d = _read(os.path.join(res, f"update_{arch}_{prec}_{b}.json"))
                if d is None or "train" not in d:
                    row[b] = {"status": "MISSING" if d is None else "INCOMPLETE"}
                    continue
                tr = d["train"]
                row[b] = {"status": "OK", "torch": d.get("torch"), "gate": (d.get("gate") or {}).get("verdict"),
                          "update_s_est": tr["update_s_est"], "train_ms_est": 1e3 * tr["update_s_est"],
                          "marginal_epoch_s": tr["marginal_epoch_s"], "per_micro_ms": tr["per_micro_ms_marginal"],
                          "first_epoch_overhead_s": tr["first_epoch_overhead_s"], "peak_reserved_mib": d.get("peak_reserved_mib"),
                          "compile_gate_prewarm_s": (d.get("phases_s") or {}).get("compile_gate_prewarm"),
                          "spread": _spread(tr.get("walls_s") or {}), "load1": [d.get("load1_start"), d.get("load1_end")],
                          "gpu_others": [d.get("gpu_others_start"), d.get("gpu_others_end")], "params": (d.get("shape") or {}).get("params")}
            a6, a8 = row.get("cu126", {}), row.get("cu128", {})
            if a6.get("status") == "OK" and a8.get("status") == "OK":
                row["ratio_cu128_over_cu126"] = {k: a8[k] / a6[k] for k in ("update_s_est", "marginal_epoch_s", "peak_reserved_mib")
                                                 if a6.get(k) and a8.get(k)}
            summary["update"][f"{arch}_{prec}"] = row
    for prec in precs:
        p6 = os.path.join(res, f"numerics_{prec}_cu126.json")
        p8 = os.path.join(res, f"numerics_{prec}_cu128.json")
        if is_done_path(p6) and is_done_path(p8):
            try:
                summary["numerics"][prec] = compare_numerics(p6, p8)
            except Exception as exc:  # noqa: BLE001 — a missing .pt is a finding, not a crash
                summary["numerics"][prec] = {"valid": False, "why": f"{type(exc).__name__}: {exc}"}
        else:
            summary["numerics"][prec] = {"valid": None, "why": "MISSING (a numerics cell has not finished)"}
    return summary


def is_done_path(path: str) -> bool:
    d = _read(path)
    return bool(d and d.get("done"))


def _f(x: Any, fmt: str = "{:.4g}") -> str:
    return "-" if x is None else fmt.format(x)


def render_markdown(summary: Dict[str, Any]) -> str:
    L: List[str] = ["# CUDA build validation: torch 2.8.0+cu126 vs +cu128", "",
                    f"Results: `{summary['out_dir']}/results/`. A benchmark warns, never stretches: the load average and the "
                    "other GPU processes are in `summary.json`.", "",
                    "## (a)+(c) Update cells (the local baseline for the cloud ladder)", "",
                    "| config | build | gate | update s (= train_ms/1000) | marginal epoch s | per-micro ms | peak MiB | "
                    "compile+gate+prewarm s | spread | torch |",
                    "|---|---|---|---|---|---|---|---|---|---|"]
    for key, row in summary["update"].items():
        for b in BUILDS:
            c = row.get(b, {})
            if c.get("status") != "OK":
                L.append(f"| {key} | {b} | {c.get('status', 'MISSING')} | | | | | | | |")
                continue
            L.append(f"| {key} | {b} | {c['gate']} | {_f(c['update_s_est'])} | {_f(c['marginal_epoch_s'])} | {_f(c['per_micro_ms'])} | "
                     f"{_f(c['peak_reserved_mib'], '{:.0f}')} | {_f(c['compile_gate_prewarm_s'], '{:.0f}')} | "
                     f"{_f(c['spread'], '{:.1%}')} | {c['torch']} |")
        r = row.get("ratio_cu128_over_cu126")
        if r:
            L.append(f"| **{key}** | cu128 / cu126 | | **{_f(r.get('update_s_est'), '{:.3f}')}** | {_f(r.get('marginal_epoch_s'), '{:.3f}')} | | "
                     f"{_f(r.get('peak_reserved_mib'), '{:.3f}')} | | | |")
    L += ["", "## (b) Numerics: same weights, same batch, eager (cross-build difference read against the same-build noise)", "",
          "| prec | valid | loss cu126 | loss cu128 | loss rel diff | grad rel-L2 | cosine | bitwise equal | worst parameter | "
          "same-build noise (grad rel-L2) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for prec, n in summary["numerics"].items():
        if not n.get("valid"):
            L.append(f"| {prec} | {n.get('valid')} | {n.get('why', '')} | | | | | | | |")
            continue
        w = n["param_rel_l2_worst"][0] if n["param_rel_l2_worst"] else {"name": "-", "rel_l2": None}
        L.append(f"| {prec} | yes | {_f(n['loss'][0], '{:.9g}')} | {_f(n['loss'][1], '{:.9g}')} | {_f(n['loss_rel'], '{:.3g}')} | "
                 f"{_f(n['grad_rel_l2'], '{:.3g}')} | {_f(n['grad_cosine'], '{:.9f}')} | {n['bitwise_identical']} | "
                 f"{w['name']} ({_f(w['rel_l2'], '{:.3g}')}) | {_f(max(n['noise_grad_rel_l2']), '{:.3g}')} |")
    return "\n".join(L) + "\n"


def write_summary(out_dir: str, archs: Sequence[str], precs: Sequence[str]) -> Dict[str, Any]:
    s = summarize(out_dir, archs, precs)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=1, default=str)
    with open(os.path.join(out_dir, "SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(s))
    return s


# --------------------------------------------------------------------------------------- the driver

def preflight(args: argparse.Namespace, pythons: Dict[str, str], *, gpu: bool) -> List[str]:
    """The reasons NOT to start (empty = go). Nothing here touches the GPU."""
    from main.launcher.torch_runtime import interpreter_torch
    bad: List[str] = []
    for b in BUILDS:
        py = pythons[b]
        if not os.access(py, os.X_OK):
            bad.append(f"{b}: no interpreter at {py} (conda env create -f environment_torch28_cu128.yml for the twin)")
            continue
        ver = interpreter_torch(py)
        if ver is None or EXPECTED_BUILD[b] not in ver:
            bad.append(f"{b}: {py} has torch {ver}, wanted a {EXPECTED_BUILD[b]} build — comparing a build with itself proves nothing")
    if gpu and not lease_token_present():
        bad.append("no GPU lease token: this tool never touches the GPU unleased (root CLAUDE.md rule 5). "
                   "The orchestrator grants ONE lease; `scripts/ops/gpu_lease.sh acquire --owner <you>`, then "
                   "GEN3AI_GPU_LEASE_TOKEN=<token> in front of this call")
    unknown = [x for x in args.archs if x not in ARCHS] + [x for x in args.precs if x not in PRECISIONS]
    if unknown:
        bad.append(f"unknown arch/precision {unknown} (archs {ARCHS}, precisions {PRECISIONS})")
    return bad


def print_plan(cells: List[Cell], pythons: Dict[str, str], configs: Dict[str, str], args: argparse.Namespace, cpu: bool) -> None:
    n_up = sum(c.kind == "update" for c in cells)
    n_nu = len(cells) - n_up
    print(f"PLAN: {len(cells)} cells ({n_nu} numerics + {n_up} update), estimated "
          f"{n_nu * EST_NUMERICS_MIN + n_up * EST_UPDATE_MIN:.0f} min of GPU. Resumable: finished cells are skipped.")
    for c in cells:
        mark = "done" if is_done(c) else "todo"
        print(f"  [{mark}] {c.name}")
        print("      " + " ".join(shlex.quote(x) for x in wrap(cell_argv(c, pythons=pythons, configs=configs, args=args, cpu=cpu),
                                                                 cpu=cpu, args=args)))


def run_cells(cells: List[Cell], pythons: Dict[str, str], configs: Dict[str, str], args: argparse.Namespace, cpu: bool) -> int:
    logs = os.path.join(args.out, "logs")
    os.makedirs(logs, exist_ok=True)
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    failed: List[str] = []
    for i, c in enumerate(cells, 1):
        if args.only and not any(o in c.name for o in args.only):
            continue
        if is_done(c) and not args.force:
            print(f"[{i}/{len(cells)}] {c.name}: already done, skipped", flush=True)
            continue
        cmd = wrap(cell_argv(c, pythons=pythons, configs=configs, args=args, cpu=cpu), cpu=cpu, args=args)
        print(f"[{i}/{len(cells)}] {c.name} ...", flush=True)
        t0 = time.time()
        with open(os.path.join(logs, c.name + ".log"), "w", encoding="utf-8") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False).returncode
        print(f"[{i}/{len(cells)}] {c.name}: exit {rc} in {time.time() - t0:.0f}s", flush=True)
        if rc in (5, 6) and not cpu:
            print(f"STOPPED: gpu_lock.sh exited {rc} — the GPU is {'held by a one-off job' if rc == 5 else 'LEASED to someone else'}; "
                  "this tool waits for nobody. Nothing further was started.", file=sys.stderr)
            return rc
        if rc != 0:
            failed.append(f"{c.name} (exit {rc}; log {os.path.join(logs, c.name + '.log')})")
    write_summary(args.out, args.archs, args.precs)
    print(f"SUMMARY: {os.path.join(args.out, 'SUMMARY.md')}", flush=True)
    if failed:
        print("FAILED CELLS:\n  " + "\n  ".join(failed), file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cuda_build_validate.sh", description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=("plan", "run", "cpu-smoke", "summarize", "numerics-worker"))
    p.add_argument("--out", default=None, help="results directory (default: designs/research_state/measurements/cuda_build_<date>)")
    p.add_argument("--archs", default=",".join(ARCHS), type=lambda s: [x for x in s.split(",") if x],
                   help="perf-phase config tags: E (end state), P (production)")
    p.add_argument("--precs", default=",".join(PRECISIONS), type=lambda s: [x for x in s.split(",") if x])
    p.add_argument("--epochs", default="1,3", help="update_bench's timed epoch counts")
    p.add_argument("--warm", type=int, default=1)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--mem-gb", type=int, default=32, help="mem_cap per cell")
    p.add_argument("--cell-timeout-min", type=float, default=50.0, help="wall timeout per cell (inside the lock)")
    p.add_argument("--cpus", default="", help="taskset -c list for every cell (e.g. the agent cores while a run is live)")
    p.add_argument("--nice", type=int, default=0)
    p.add_argument("--only", default="", type=lambda s: [x for x in s.split(",") if x], help="run only cells whose name contains one of these")
    p.add_argument("--force", action="store_true", help="redo finished cells")
    p.add_argument("--py-cu126", default="", help="interpreter of the cu126 env (default: the sibling gen3ai_torch28)")
    p.add_argument("--py-cu128", default="", help="interpreter of the cu128 env (default: the sibling gen3ai_torch28_cu128)")
    # numerics-worker (internal)
    p.add_argument("--build", default="")
    p.add_argument("--prec", default="fp32")
    p.add_argument("--argv", default="")
    p.add_argument("--device", default="cuda")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.mode == "numerics-worker":
        if not a.out:
            print("numerics-worker needs --out", file=sys.stderr)
            return EXIT_USAGE
        return numerics_worker(a)
    a.out = a.out or default_out_dir()
    pythons = {"cu126": a.py_cu126 or sibling_python(ENV_NAMES["cu126"]), "cu128": a.py_cu128 or sibling_python(ENV_NAMES["cu128"])}
    if a.mode == "summarize":
        s = write_summary(a.out, a.archs, a.precs)
        print(render_markdown(s))
        return 0
    cpu = a.mode == "cpu-smoke"
    if cpu:
        a.epochs, a.warm, a.reps = "1", 0, 1
    cells = plan_cells(a.archs, a.precs, BUILDS, a.out)
    if cpu:
        cells = [c for c in cells if c.arch in ("-", a.archs[0]) and c.prec == a.precs[0]]
    configs = load_configs()
    if a.mode == "plan":
        print_plan(cells, pythons, configs, a, cpu=False)
        bad = preflight(a, pythons, gpu=False)
        for b in bad:
            print(f"PREFLIGHT WOULD REFUSE: {b}")
        return 0
    bad = preflight(a, pythons, gpu=not cpu)
    if bad:
        for b in bad:
            print(f"REFUSING: {b}", file=sys.stderr)
        return EXIT_USAGE
    print_plan(cells, pythons, configs, a, cpu=cpu)
    return run_cells(cells, pythons, configs, a, cpu=cpu)


if __name__ == "__main__":
    sys.exit(main())
