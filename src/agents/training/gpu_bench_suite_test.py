"""The frozen GPU benchmark suite (`gpu_bench_suite`, `gpu_bench_update`, `gpu_bench_collect`) — the pure parts: the
frozen geometry against the production record, the schedule and estimate arithmetic, the schema validator, the
collector read, the Amdahl estimate. CPU only, no GPU, no run data. The end-to-end CPU code path is
`gpu_bench_suite_integration_test.py`."""
from __future__ import annotations

import copy
import json
import math

import pytest

from agents.training import gpu_bench_collect as GC
from agents.training import gpu_bench_suite as S
from agents.training import gpu_bench_update as U


# ------------------------------------------------------------------------------------------- the frozen geometry

def test_full_geometry_is_the_production_sizing_and_recipe():
    """The suite is FROZEN on the production shape: if `production_config.json` moves, this fails and the suite is
    re-frozen with a `SUITE_VERSION` bump (never silently measuring an old shape)."""
    from utils.paths import repo_path
    cfg = json.loads(repo_path("designs", "production_config.json").read_text())
    sizing, fresh = cfg["recipe"]["sizing"], cfg["recipe"]["fresh"]
    assert S.FULL["n_envs"] == sizing["n_envs"] == 256
    assert S.FULL["n_steps"] == sizing["n_steps"] == 384
    assert S.FULL["n_steps"] * S.FULL["n_envs"] == sizing["rollout_target_samples"] == 98_304
    assert fresh["batch_size"] == 2048 and fresh["grad_accum_steps"] == 32 and fresh["n_epochs"] == 10
    assert fresh["trunk_precision"] == "bf16"


def test_every_update_point_keeps_the_production_effective_batch():
    prod = 2048 * 32
    for name in S.phase_names(quick=False):
        if name.startswith(("update_", "headroom_")):
            cfg = S.phase_config(name, False, "cuda", "/x")
            b = cfg["batch_size"] or 2048
            k = cfg["grad_accum"] or 32
            assert b * k == prod, name
            assert (cfg["n_steps"] * cfg["n_envs"]) % b == 0, name
    assert [p for p in S.phase_names(False) if p.startswith("headroom_")] == ["headroom_b4096", "headroom_b8192"]


def test_production_argv_is_the_live_runs_training_surface():
    assert S.PRODUCTION_ARGV.startswith("--arch production ")
    for flag in ("--ridealong-ensemble 5", "--ridealong-rnd ", "--ridealong-adv 5", "--ridealong-opp 5",
                 "--ridealong-rnd-variants all"):
        assert flag in S.PRODUCTION_ARGV + " "


def test_phase_configs_and_order():
    names = S.phase_names(False)
    assert names[0] == "host" and names[-1] == "collect"
    assert S.phase_config("update_fp32", False, "cuda", "/o")["trunk_precision"] == "fp32"
    assert S.phase_config("headroom_b8192", False, "cuda", "/o")["trunk_precision"] == "bf16"
    col = S.phase_config("collect", False, "cuda", "/o")
    assert col["steps"] == 98_304 * (col["warm"] + col["timed"] + 2)
    assert col["pool_snapshots"] == 20 and col["pool_win_rate"] == pytest.approx(0.94)
    assert S.phase_config("collect", True, "cpu", "/o")["extra"][:2] == ["--n-envs", "2"]
    with pytest.raises(ValueError):
        S.phase_config("nope", False, "cuda", "/o")


# ------------------------------------------------------------------------------------------- the update arithmetic

def test_schedule_warm_timed_then_one_diag():
    assert U.schedule(1, [1, 3], 2) == [(1, False), (1, True), (1, True), (3, True), (3, True), (1, "diag")]
    assert U.schedule(0, [2], 1) == [(2, True), (1, "diag")]


def test_phase_for_call_keeps_timed_calls_plain_and_diag_diagnostic():
    every = 10
    sched = U.schedule(1, [1, 3], 2)
    phases = [U.phase_for_call(i, t, every) for i, (_e, t) in enumerate(sched)]
    for (_, t), ph in zip(sched, phases):
        if t is True:
            assert ph % every != 0          # a PLAIN update (9 of 10 production updates)
    assert phases[-1] == every              # the diag call sits on the cadence
    assert U.phase_for_call(3, True, 1) == 3


def test_update_estimate_two_point_and_one_point():
    est = U.update_estimate({1: 6.0, 3: 14.0}, 10)
    assert est["marginal_epoch_s"] == pytest.approx(4.0)
    assert est["update_s_est"] == pytest.approx(6.0 + 9 * 4.0)
    assert est["first_epoch_overhead_s"] == pytest.approx(2.0)
    one = U.update_estimate({2: 8.0}, 10)
    assert one["update_s_est"] == pytest.approx(40.0)
    with pytest.raises(ValueError):
        U.update_estimate({}, 10)


def test_headroom_points_same_effective_batch():
    assert U.headroom_points(2048, 32, (2, 4, 3, 64)) == [(4096, 16), (8192, 8)]


def test_is_oom_by_type_and_message():
    import torch
    assert U.is_oom(torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate"))
    assert U.is_oom(RuntimeError("compile failed: CUDA out of memory"))
    assert not U.is_oom(RuntimeError("shape mismatch"))


# ------------------------------------------------------------------------------------------- the collector read

def _snap(**kw):
    base = {k: 0.0 for k in ("host_steps", "trainee_decisions", "p2_policy_decisions", "p2_external_decisions",
                             "games_ended", "games_cut", "quarantines", "near_boundary", "t")}
    base.update({f"s_{k}": 0.0 for k in GC.COLLECTOR_PHASES + ("fill",)})
    base.update(kw)
    return base


def test_stats_delta_per_host_step():
    a = _snap(host_steps=400.0, trainee_decisions=98_304.0, p2_policy_decisions=90_000.0, s_gpu_wait=15.0, s_core=6.0,
              s_fill=0.3)
    b = _snap()
    d = GC.stats_delta(a, b)
    assert d["gpu_wait_ms_per_host_step"] == pytest.approx(37.5)
    assert d["core_ms_per_host_step"] == pytest.approx(15.0)
    assert d["trainee_rows_per_host_step"] == pytest.approx(245.76)
    assert d["fill_ms"] == pytest.approx(300.0)


def _rollout(i, timed, wall, gw):
    r = {"index": i, "timed": timed, "wall_s": wall, "env_steps_per_s": 98_304 / wall, "trainee_decisions": 98_304,
         "host_steps": 400.0, "p2_policy_share": 0.9}
    r.update({f"{k}_ms_per_host_step": 1.0 for k in GC.COLLECTOR_PHASES})
    r["gpu_wait_ms_per_host_step"] = gw
    return r


def test_summarise_uses_timed_rollouts_only():
    rs = [_rollout(0, False, 99.0, 99.0), _rollout(1, True, 20.0, 30.0), _rollout(2, True, 22.0, 32.0),
          _rollout(3, True, 30.0, 40.0)]
    s = GC.summarise(rs)
    assert s["n_timed"] == 3
    assert s["wall_s_median"] == pytest.approx(22.0)
    assert s["gpu_wait_ms_per_host_step_median"] == pytest.approx(32.0)
    assert s["gpu_wait_s_per_rollout_median"] == pytest.approx(32.0 * 400 / 1000)
    assert s["wall_spread_rel"] == pytest.approx(10.0 / 22.0)
    assert GC.summarise([_rollout(0, False, 1.0, 1.0)]) == {}


def test_trainer_argv_is_fresh_uncompiled_and_never_names_the_trainer():
    argv = GC.trainer_argv(["--arch", "production"], run_dir="/s/r", steps=1000, device="cuda", seed=7,
                           extra=["--n-envs", "2"])
    assert argv[:4] == ["--arch", "production", "--n-envs", "2"]
    for flag in ("--no-compile-trainer", "--run-dir", "--eval-freq", "--checkpoint-every-steps"):
        assert flag in argv
    assert "--model" not in argv
    assert argv[argv.index("--eval-freq") + 1] == GC.NO_EVAL_FREQ
    with pytest.raises(ValueError):
        GC.trainer_argv(["--x", "train_rl_" + "agent"], run_dir="/s", steps=1, device="cpu", seed=0)


# ------------------------------------------------------------------------------------------- the sampler

def test_parse_sampler_skips_na_and_short_lines():
    text = "50, 6000, 200.5, 1800, 60\n[N/A], 6100, [N/A], 1900, 61\ngarbage\n70, 6200, 210.5, 2000, 62\n"
    s = S.parse_sampler(text)
    assert s["samples"] == 3
    assert s["utilization.gpu"] == {"mean": 60.0, "max": 70.0}
    assert s["power.draw"]["mean"] == pytest.approx(205.5)
    assert s["memory.used"]["max"] == 6200.0


# ------------------------------------------------------------------------------------------- the schema

def _ok_update(update_s=40.0, peak=6000.0, verdict="PASS"):
    return {"status": "ok", "shape": {"rows": 98_304, "batch_size": 2048, "grad_accum": 32},
            "gate": {"verdict": verdict}, "train": {"update_s_est": update_s, "marginal_epoch_s": 4.0,
                                                    "median_s": {"1": 6.0, "3": 14.0}},
            "phases_s": {"compile_gate_prewarm": 250.0}, "calls": [], "nvidia_smi_before": {},
            "nvidia_smi_after": {}, "gpu_sampler": {}, "wall_s": 400.0, "peak_reserved_mib": peak}


def _ok_suite(update_bf16=40.0, gw=37.0):
    host = {"status": "ok", "cpu_model": "AMD Ryzen 7 9800X3D", "cpu_logical": 16, "ram_gib": 62.0,
            "torch": "2.8.0+cu126", "commit": "abc", "gpu_name": "NVIDIA GeForce RTX 3090", "capability": "sm_86",
            "vram_gib": 24.0, "arch_list": ["sm_86"]}
    col = {"status": "ok", "pool": {"n_snapshots": 20}, "buffer_shape": {"rows": 98_304}, "rollouts": [],
           "summary": {"env_steps_per_s_median": 3800.0, "wall_s_median": 26.0, "gpu_wait_ms_per_host_step_median": gw,
                       "core_ms_per_host_step_median": 15.0}, "nvidia_smi_before": {}, "nvidia_smi_after": {},
           "wall_s": 600.0}
    return {"schema": S.SCHEMA, "suite_version": S.SUITE_VERSION, "quick": False, "device": "cuda",
            "argv": S.PRODUCTION_ARGV, "geometry": S.FULL, "live_cycle": S.LIVE_CYCLE, "tag": "t",
            "phases": {"host": host, "update_bf16": _ok_update(update_bf16), "update_fp32": _ok_update(52.0),
                       "headroom_b4096": _ok_update(38.0, 9000.0), "headroom_b8192": {"status": "oom",
                                                                                      "oom": {"where": "compile"}},
                       "collect": col}}


def test_validate_accepts_a_complete_suite_with_an_oom_headroom_point():
    assert S.validate(_ok_suite()) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s["phases"].pop("collect"), "phase collect: missing"),
    (lambda s: s["phases"]["update_fp32"].update(status="timeout"), "update_fp32: status 'timeout'"),
    (lambda s: s["phases"]["update_bf16"]["train"].pop("update_s_est"), "train.update_s_est"),
    (lambda s: s["phases"]["update_bf16"]["gate"].update(verdict="REFUSED"), "compile parity gate 'REFUSED'"),
    (lambda s: s["phases"]["update_bf16"].update(status="oom"), "update_bf16: status 'oom'"),
    (lambda s: s["phases"]["collect"]["buffer_shape"].update(rows=4096), "rollout rows 4096"),
    (lambda s: s["phases"]["host"].pop("gpu_name"), "phase host: missing gpu_name"),
    (lambda s: s.update(schema="other"), "schema"),
])
def test_validate_names_every_problem(mutate, needle):
    s = copy.deepcopy(_ok_suite())
    mutate(s)
    probs = S.validate(s)
    assert any(needle in p for p in probs), probs


# ------------------------------------------------------------------------------------------- the comparison

def test_card_row_reads_missing_phases_as_none():
    s = _ok_suite()
    s["phases"]["update_fp32"]["status"] = "error"
    r = S.card_row(s)
    assert r["update_bf16_s"] == 40.0 and r["update_fp32_s"] is None
    assert r["headroom_mib"] == pytest.approx(24 * 1024 - 6000.0)
    assert r["headroom_points"]["headroom_b8192"] == "OOM"
    assert r["headroom_points"]["headroom_b4096"]["update_s_est"] == 38.0


def test_amdahl_reference_is_one_and_the_arithmetic_is_shown():
    live = {"train_s": 42.0, "collect_s": 27.0, "other_s": 5.0, "gpu_wait_s": 15.0}
    ref = S.card_row(_ok_suite(40.0, 37.0))
    same = S.amdahl(ref, ref, live)
    assert same is not None and same["speedup"] == pytest.approx(1.0)
    fast = S.card_row(_ok_suite(20.0, 18.5))          # half the update, half the GPU wait
    a = S.amdahl(fast, ref, live)
    assert a is not None
    assert a["train_s"] == pytest.approx(21.0)
    assert a["collect_s"] == pytest.approx(12.0 + 7.5)
    assert a["cycle_s"] == pytest.approx(21.0 + 19.5 + 5.0)
    assert a["speedup"] == pytest.approx(74.0 / 45.5)
    assert a["speedup_train_only"] == pytest.approx(74.0 / (21.0 + 27.0 + 5.0))
    broken = dict(fast, update_bf16_s=None)
    assert S.amdahl(broken, ref, live) is None


def test_live_cycle_adds_up_to_the_live_fps():
    """The calibration's parts must reproduce the live cycle (98,304 steps at `time/fps` 1,314 = 74.8 s) to 1 %."""
    c = S.LIVE_CYCLE
    cycle = c["train_s"] + c["collect_s"] + c["other_s"]
    assert math.isclose(cycle, 98_304 / 1314.0, rel_tol=0.01)
    assert 0 < c["gpu_wait_s"] < c["collect_s"]


def test_render_markdown_one_row_per_card_and_reference_reads_one():
    ref_suite = _ok_suite(40.0, 37.0)
    fast = _ok_suite(20.0, 18.5)
    fast["tag"] = "fast"
    fast["meta"] = {"dph": 0.5}
    rows = [S.card_row(ref_suite), S.card_row(fast)]
    md = S.render_markdown(rows, rows[0]).splitlines()
    assert len(md) == 4 and md[1].startswith("|---|")
    assert md[0].count("|") == md[1].count("|") == md[2].count("|") == md[3].count("|")
    assert "| 1.00x |" in md[2] and "| 2.00x |" in md[3] and "| 0.500 |" in md[3]
    assert md[3].rstrip().endswith("x |")
    no_collect = S.card_row(fast)
    no_collect.update(gpu_wait_ms_per_host_step=None, tag="nocol")
    md2 = S.render_markdown([rows[0], no_collect], rows[0]).splitlines()
    assert "| 2.00x |" in md2[3] and md2[3].rstrip().endswith("| — |")     # the update ratio needs no collect read
    assert "OOM" in md[2]
