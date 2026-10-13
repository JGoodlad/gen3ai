"""`main.ops.cuda_build_validate` (+ `scripts/ops/cuda_build_validate.sh`) — the GPU validation of the CUDA 12.8 twin env.

Nothing here touches a GPU, a lease or a real second env. What is pinned, deterministically:

* the PLAN: numerics first, the two builds alternate within a pair, every GPU cell goes through `gpu_lock.sh` and
  `mem_cap.sh` with the wall timeout INSIDE the lock, the CPU smoke goes through neither lock;
* the REFUSALS: no lease token, a wrong-build interpreter (comparing a build with itself proves nothing), an unknown
  config tag; and `run` stops at once on a `gpu_lock.sh` exit 5 / 6 instead of waiting or going on;
* RESUMABILITY: a finished cell is skipped, a partial one is redone, a failed one is reported and the rest still run;
* the SUMMARY: the table and the cross-build compare are computed from the cell JSONs / tensors, an init mismatch is
  INVALID rather than a number.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import main.ops.cuda_build_validate as V
from utils.paths import repo_path


def _fake_python(path: Path, torch_version: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho {torch_version}\n")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def pythons(tmp_path):
    return {"cu126": _fake_python(tmp_path / "a" / "bin" / "python3", "2.8.0+cu126"),
            "cu128": _fake_python(tmp_path / "b" / "bin" / "python3", "2.8.0+cu128")}


def _args(tmp_path, *extra):
    return V.build_parser().parse_args(["plan", "--out", str(tmp_path / "out"), *extra])


# ---- the plan -------------------------------------------------------------------------------------------

def test_the_perf_phase_configs_and_bench_script_exist():
    configs = V.load_configs()
    assert {"E", "P"} <= set(configs) and "--arch endstate" in configs["E"] and "--arch production" in configs["P"]
    assert repo_path(*V.BENCH_DIR, "update_bench.py").is_file()


def test_the_plan_runs_numerics_first_and_alternates_the_builds(tmp_path):
    cells = V.plan_cells(("E", "P"), ("fp32", "bf16"), V.BUILDS, str(tmp_path))
    kinds = [c.kind for c in cells]
    assert kinds == ["numerics"] * 4 + ["update"] * 8, kinds
    assert len({c.name for c in cells}) == len(cells)
    first_builds = [cells[i].build for i in range(0, 12, 2)]
    assert set(first_builds) == {"cu126", "cu128"}, "the builds must alternate which goes first within a pair"
    for i in range(0, 12, 2):
        assert {cells[i].build, cells[i + 1].build} == {"cu126", "cu128"} and cells[i].prec == cells[i + 1].prec


def test_a_gpu_cell_goes_through_gpu_lock_then_mem_cap_with_the_timeout_INSIDE(tmp_path, pythons):
    args = _args(tmp_path, "--cpus", "5-7", "--nice", "19", "--mem-gb", "40", "--cell-timeout-min", "30")
    cell = V.plan_cells(("E",), ("fp32",), V.BUILDS, args.out)[-1]
    cmd = V.wrap(V.cell_argv(cell, pythons=pythons, configs=V.load_configs(), args=args, cpu=False), cpu=False, args=args)
    names = [os.path.basename(x) for x in cmd[:9]]
    assert names[:4] == ["taskset", "-c", "5-7", "nice"], names
    i_lock = next(i for i, x in enumerate(cmd) if x.endswith("gpu_lock.sh"))
    i_cap = next(i for i, x in enumerate(cmd) if x.endswith("mem_cap.sh"))
    assert i_lock + 1 == i_cap, "gpu_lock.sh wraps mem_cap.sh"
    assert cmd[i_cap + 1:i_cap + 4] == ["40", "timeout", "1800"], "the wall timeout goes INSIDE the lock"
    assert cmd[i_cap + 4] == "env" and any(x.startswith("PYTHONPATH=") for x in cmd)
    assert pythons[cell.build] in cmd and "update_bench.py" in " ".join(cmd)


def test_the_cpu_smoke_takes_no_gpu_lock_and_runs_the_cpu_path(tmp_path, pythons):
    args = _args(tmp_path)
    for cell in (V.plan_cells(("E",), ("fp32",), V.BUILDS, args.out)[0], V.plan_cells(("E",), ("fp32",), V.BUILDS, args.out)[-1]):
        cmd = V.wrap(V.cell_argv(cell, pythons=pythons, configs=V.load_configs(), args=args, cpu=True), cpu=True, args=args)
        assert not any(x.endswith("gpu_lock.sh") for x in cmd), "a CPU cell must not touch the GPU lock"
        assert "--cpu-smoke" in cmd or "cpu" in cmd


def test_the_default_envs_are_found_beside_the_running_interpreters_own(tmp_path):
    envs = tmp_path / "conda" / "envs"
    assert V.sibling_python("gen3ai_torch28_cu128", str(envs / "gen3ai_torch28")) == str(envs / "gen3ai_torch28_cu128" / "bin" / "python3")
    assert V.sibling_python("gen3ai_torch28", str(tmp_path / "conda")) == str(envs / "gen3ai_torch28" / "bin" / "python3")


# ---- the refusals ---------------------------------------------------------------------------------------

def test_preflight_refuses_without_a_lease_token(tmp_path, pythons, monkeypatch):
    monkeypatch.delenv("GEN3AI_GPU_LEASE_TOKEN", raising=False)
    monkeypatch.delenv("GEN3AI_GPU_LEASE_TOKEN_FILE", raising=False)
    bad = V.preflight(_args(tmp_path), pythons, gpu=True)
    assert any("no GPU lease token" in b for b in bad), bad
    assert V.preflight(_args(tmp_path), pythons, gpu=False) == [], "the CPU smoke and the plan need no lease"
    monkeypatch.setenv("GEN3AI_GPU_LEASE_TOKEN", "abc")
    assert V.preflight(_args(tmp_path), pythons, gpu=True) == []


def test_run_REFUSES_exit_2_without_a_token_and_starts_nothing(tmp_path, pythons, monkeypatch, capsys):
    monkeypatch.delenv("GEN3AI_GPU_LEASE_TOKEN", raising=False)
    monkeypatch.delenv("GEN3AI_GPU_LEASE_TOKEN_FILE", raising=False)
    started = []
    monkeypatch.setattr(V, "run_cells", lambda *a, **k: started.append(a))
    rc = V.main(["run", "--out", str(tmp_path / "o"), "--py-cu126", pythons["cu126"], "--py-cu128", pythons["cu128"]])
    assert rc == V.EXIT_USAGE and started == []
    assert "REFUSING" in capsys.readouterr().err


def test_preflight_refuses_two_envs_of_the_SAME_build(tmp_path, monkeypatch):
    monkeypatch.setenv("GEN3AI_GPU_LEASE_TOKEN", "abc")
    same = {"cu126": _fake_python(tmp_path / "x" / "bin" / "python3", "2.8.0+cu126"),
            "cu128": _fake_python(tmp_path / "y" / "bin" / "python3", "2.8.0+cu126")}
    bad = V.preflight(_args(tmp_path), same, gpu=True)
    assert len(bad) == 1 and "cu128" in bad[0] and "+cu128" in bad[0] and "proves nothing" in bad[0], bad
    missing = {"cu126": same["cu126"], "cu128": str(tmp_path / "nope" / "bin" / "python3")}
    assert any("no interpreter" in b for b in V.preflight(_args(tmp_path), missing, gpu=True))


def test_an_unknown_config_or_precision_is_refused(tmp_path, pythons):
    bad = V.preflight(_args(tmp_path, "--archs", "E,Z", "--precs", "fp16"), pythons, gpu=False)
    assert bad and "unknown" in bad[0] and "Z" in bad[0] and "fp16" in bad[0]


# ---- the run loop ---------------------------------------------------------------------------------------

class _Proc:
    def __init__(self, rc):
        self.returncode = rc


def _run_args(tmp_path, *extra):
    a = V.build_parser().parse_args(["run", "--out", str(tmp_path / "out"), "--archs", "E", "--precs", "fp32", *extra])
    return a


def test_run_cells_skips_done_redoes_partial_reports_failures_and_continues(tmp_path, pythons, monkeypatch):
    a = _run_args(tmp_path)
    cells = V.plan_cells(("E",), ("fp32",), V.BUILDS, a.out)
    os.makedirs(os.path.dirname(cells[0].out_json))
    Path(cells[0].out_json).write_text(json.dumps({"done": True}))                       # numerics: done
    upd = [c for c in cells if c.kind == "update"]
    Path(upd[0].out_json).write_text(json.dumps({"tag": "partial"}))                      # update: partial -> redone
    calls = []

    def fake_run(cmd, **kw):
        name = next(c.name for c in cells if c.name in " ".join(cmd) or c.out_json in cmd)
        calls.append(name)
        return _Proc(1 if name == upd[0].name else 0)

    monkeypatch.setattr(V.subprocess, "run", fake_run)
    rc = V.run_cells(cells, pythons, V.load_configs(), a, cpu=False)
    assert rc == 1
    assert cells[0].name not in calls, "a finished cell must be skipped"
    assert upd[0].name in calls and upd[1].name in calls, "a failed cell must not stop the rest"
    assert (Path(a.out) / "SUMMARY.md").is_file() and (Path(a.out) / "summary.json").is_file()


@pytest.mark.parametrize("rc", [5, 6])
def test_run_cells_STOPS_AT_ONCE_when_gpu_lock_says_the_card_is_not_ours(tmp_path, pythons, monkeypatch, rc, capsys):
    a = _run_args(tmp_path)
    cells = V.plan_cells(("E",), ("fp32",), V.BUILDS, a.out)
    calls = []
    monkeypatch.setattr(V.subprocess, "run", lambda cmd, **kw: (calls.append(cmd), _Proc(rc))[1])
    assert V.run_cells(cells, pythons, V.load_configs(), a, cpu=False) == rc
    assert len(calls) == 1, "nothing may start after a lock refusal"
    assert "waits for nobody" in capsys.readouterr().err


def test_a_lock_exit_code_is_just_a_failure_in_the_cpu_smoke(tmp_path, pythons, monkeypatch):
    a = _run_args(tmp_path)
    cells = V.plan_cells(("E",), ("fp32",), V.BUILDS, a.out)
    monkeypatch.setattr(V.subprocess, "run", lambda cmd, **kw: _Proc(6))
    assert V.run_cells(cells, pythons, V.load_configs(), a, cpu=True) == 1


def test_is_done_needs_the_result_not_just_a_file(tmp_path):
    u = V.Cell("u", "update", "cu126", "fp32", "E", str(tmp_path / "u.json"))
    n = V.Cell("n", "numerics", "cu126", "fp32", "-", str(tmp_path / "n.json"))
    assert not V.is_done(u) and not V.is_done(n)
    Path(u.out_json).write_text(json.dumps({"tag": "x"}))
    Path(n.out_json).write_text(json.dumps({"done": False}))
    assert not V.is_done(u) and not V.is_done(n)
    Path(u.out_json).write_text(json.dumps({"train": {}}))
    Path(n.out_json).write_text(json.dumps({"done": True}))
    assert V.is_done(u) and V.is_done(n)
    Path(u.out_json).write_text("{not json")
    assert not V.is_done(u)


# ---- the summary ----------------------------------------------------------------------------------------

def _update_json(path: Path, scale: float, gate="PASS"):
    path.write_text(json.dumps({
        "torch": "2.8.0+cu12x", "gate": {"verdict": gate}, "peak_reserved_mib": 1000.0 * scale,
        "phases_s": {"compile_gate_prewarm": 300.0}, "load1_start": 2.0, "load1_end": 2.5, "shape": {"params": 14_000_000},
        "gpu_others_start": [], "gpu_others_end": [],
        "train": {"update_s_est": 50.0 * scale, "marginal_epoch_s": 5.0 * scale, "per_micro_ms_marginal": 100.0 * scale,
                  "first_epoch_overhead_s": 1.0, "walls_s": {"1": [5.0, 5.2], "3": [15.0, 15.6]}}}))


def _numerics(dirp: Path, build: str, prec: str, grad, *, sha="same", noise=0.0):
    j = dirp / f"numerics_{prec}_{build}.json"
    j.write_text(json.dumps({"build": build, "prec": prec, "torch": f"2.8.0+{build}", "init_params_sha256": sha, "done": True,
                             "finite": True, "noise_grad_rel_l2": noise, "noise_loss_rel": 0.0}))
    torch.save({"loss": torch.tensor([1.5 if build == "cu126" else 1.5 + 1e-7]), "grad": torch.tensor(grad, dtype=torch.float32),
                "sizes": torch.tensor([2, 2]), "names": ["a.weight", "b.weight"]}, str(j)[:-5] + ".pt")


def test_the_summary_computes_the_cross_build_ratios_and_the_numerics_compare(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _update_json(res / "update_E_fp32_cu126.json", 1.0)
    _update_json(res / "update_E_fp32_cu128.json", 1.1)
    _numerics(res, "cu126", "fp32", [1.0, 2.0, 3.0, 4.0])
    _numerics(res, "cu128", "fp32", [1.0, 2.0, 3.0, 4.0 + 4e-6])
    s = V.write_summary(str(tmp_path), ("E",), ("fp32",))
    row = s["update"]["E_fp32"]
    assert row["cu126"]["status"] == "OK" and row["cu128"]["train_ms_est"] == pytest.approx(55000.0)
    assert row["ratio_cu128_over_cu126"]["update_s_est"] == pytest.approx(1.1)
    assert row["cu126"]["spread"] == pytest.approx((15.6 - 15.0) / 15.6, rel=0.05)
    n = s["numerics"]["fp32"]
    assert n["valid"] and not n["bitwise_identical"] and 0 < n["grad_rel_l2"] < 1e-5
    assert n["param_rel_l2_worst"][0]["name"] == "b.weight"
    md = (tmp_path / "SUMMARY.md").read_text()
    assert "E_fp32" in md and "cu128 / cu126" in md and "b.weight" in md


def test_identical_gradients_read_as_bitwise_equal_and_an_init_mismatch_is_INVALID(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _numerics(res, "cu126", "bf16", [1.0, 2.0, 3.0, 4.0])
    _numerics(res, "cu128", "bf16", [1.0, 2.0, 3.0, 4.0])
    n = V.summarize(str(tmp_path), ("E",), ("bf16",))["numerics"]["bf16"]
    assert n["valid"] and n["bitwise_identical"] and n["grad_rel_l2"] == 0.0
    _numerics(res, "cu128", "bf16", [1.0, 2.0, 3.0, 4.0], sha="DIFFERENT")
    n = V.summarize(str(tmp_path), ("E",), ("bf16",))["numerics"]["bf16"]
    assert n["valid"] is False and "byte-identical weights" in n["why"] and "grad_rel_l2" not in n


def test_missing_and_incomplete_cells_are_named_not_hidden(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    (res / "update_E_fp32_cu126.json").write_text(json.dumps({"tag": "partial"}))
    s = V.summarize(str(tmp_path), ("E",), ("fp32",))
    assert s["update"]["E_fp32"]["cu126"]["status"] == "INCOMPLETE" and s["update"]["E_fp32"]["cu128"]["status"] == "MISSING"
    assert "ratio_cu128_over_cu126" not in s["update"]["E_fp32"]
    assert s["numerics"]["fp32"]["valid"] is None
    assert "INCOMPLETE" in V.render_markdown(s) and "MISSING" in V.render_markdown(s)
