"""The THROUGHPUT A/B scaffold (``throughput`` + ``hooks``): the math, the ``/proc`` reading, the
NOT-BUILT hooks, the GPU-lock re-exec decision and the ``models/`` refusal are routine UNIT tests (no
battles). ``test_smoke_both_arms_tiny`` runs both arms end to end at N = 2 and is ``sim`` +
``integration``.
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from main.rust_core_m5 import hooks as H
from main.rust_core_m5 import throughput as T


# ------------------------------------------------------------------ the ratio + CI

def test_ratio_ci_point_is_the_geometric_mean_of_pair_ratios():
    r = T.ratio_ci([20.0, 40.0, 10.0, 80.0], [10.0, 10.0, 10.0, 10.0], seed=1)
    assert math.isclose(r["point"], (2 * 4 * 1 * 8) ** 0.25, rel_tol=1e-12)
    assert r["per_pair"] == pytest.approx([2.0, 4.0, 1.0, 8.0])
    assert r["lo"] <= r["point"] <= r["hi"]
    assert r["lo"] >= 1.0 - 1e-12 and r["hi"] <= 8.0 + 1e-12   # a bootstrap of a mean stays inside the data


def test_ratio_ci_constant_ratio_has_a_degenerate_ci():
    r = T.ratio_ci([3.0, 6.0, 9.0, 12.0, 15.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert r["point"] == pytest.approx(3.0) and r["lo"] == pytest.approx(3.0) and r["hi"] == pytest.approx(3.0)
    assert "note" not in r


def test_ratio_ci_covers_the_truth_on_noisy_synthetic_pairs():
    """Over many synthetic replicates the 95 % CI covers the true ratio most of the time (a
    percentile bootstrap on 12 pairs under-covers a little; the bar is loose on purpose)."""
    rng = np.random.default_rng(7)
    true = 5.0
    hits = 0
    for rep in range(200):
        den = rng.uniform(100, 200, 12)
        num = den * true * np.exp(rng.normal(0, 0.05, 12))
        r = T.ratio_ci(list(num), list(den), seed=rep, n_boot=2000)
        assert r["lo"] <= r["point"] <= r["hi"] and r["hi"] / r["lo"] < 1.2
        hits += r["lo"] < true < r["hi"]
    assert 0.85 <= hits / 200 <= 1.0, hits


def test_ratio_ci_one_pair_and_bad_input():
    r = T.ratio_ci([2.0], [1.0])
    assert r["point"] == 2.0 and r["lo"] is None and r["hi"] is None and "one pair" in r["note"]
    with pytest.raises(ValueError):
        T.ratio_ci([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        T.ratio_ci([0.0], [1.0])


def test_schedule_interleaves_ab_ba():
    assert T.schedule(3, ("python", "rust")) == [(0, "python"), (0, "rust"), (1, "rust"), (1, "python"),
                                                 (2, "python"), (2, "rust")]
    assert T.schedule(2, ("rust",)) == [(0, "rust"), (1, "rust")]


def test_summarize_pairs_blocks_by_pair():
    def blk(arm, pair, dps, cpd):
        return {"arm": arm, "pair": pair, "decisions": 100, "decisions_per_s": dps, "ms_per_vec_step": 1.0,
                "vec_steps_per_s": 1.0, "cpu_us_per_decision": cpd, "inference_share": 0.0}

    blocks = [blk("python", 0, 100.0, 1000.0), blk("rust", 0, 1000.0, 100.0),
              blk("rust", 1, 1200.0, 80.0), blk("python", 1, 100.0, 1000.0)]
    per, ratios = T.summarize(blocks, ("python", "rust"), seed=0)
    assert per["rust"]["decisions_per_s_mean"] == 1100.0 and per["python"]["blocks"] == 2
    assert ratios["decisions_per_s_rust_over_python"]["per_pair"] == pytest.approx([10.0, 12.0])
    assert ratios["cpu_s_per_decision_rust_over_python"]["per_pair"] == pytest.approx([0.1, 0.08])
    assert T.summarize(blocks[1:3], ("rust",), seed=0)[1] == {}


# ------------------------------------------------------------------ /proc over a tree

_BURN = "import time\nt = time.process_time()\nwhile time.process_time() - t < {s}:\n    pass\ntime.sleep(30)\n"


def test_tree_cpu_counts_a_burning_descendant():
    """A child that burns ~0.6 s of CPU inside the window shows up in the tree delta and in its own
    subtree, and the parent's own share stays small."""
    before = T.tree_snapshot(os.getpid())
    p = subprocess.Popen([sys.executable, "-c", _BURN.format(s=0.6)])
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            s = T.read_stat(p.pid)
            if s is not None and s[2] / T._TICK >= 0.55:   # tick granularity: 0.6 s may read 0.59
                break
            time.sleep(0.05)
        else:
            pytest.fail("the burning child never reached 0.55 CPU-s in 15 s")
        after = T.tree_snapshot(os.getpid())
        assert any(k[0] == p.pid for k in after), "the child is not in this process's tree"
        own = T._subtree(after, [p.pid])
        child = T.cpu_delta(before, after, only=own)
        assert 0.5 <= child <= 2.0, child
        assert T.cpu_delta(before, after) >= child
        host = T.cpu_delta(before, after, only={k for k in after if k[0] == os.getpid()})
        assert host < child
    finally:
        p.kill()   # the process this test started, by its PID
        p.wait()


def test_read_stat_of_a_gone_process_is_none():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert T.read_stat(p.pid) is None


def test_cpu_delta_counts_a_new_process_whole_and_ignores_the_vanished():
    before = {(1, 10): (0, 100), (2, 20): (1, 50)}
    after = {(1, 10): (0, 160), (3, 30): (1, 40)}
    assert T.cpu_delta(before, after) == pytest.approx((60 + 40) / T._TICK)


# ------------------------------------------------------------------ the hooks

def test_random_legal_only_picks_legal_actions_and_is_seeded():
    rng = np.random.default_rng(0)
    masks = rng.random((500, 11)) < 0.3
    masks[np.arange(500), rng.integers(0, 11, 500)] = True
    a = H.RandomLegal(5)(np.zeros((500, 4), np.float32), masks)
    assert a.dtype == np.int64 and masks[np.arange(500), a].all()
    assert (H.RandomLegal(5)(None, masks) == a).all()
    one = np.zeros((2000, 11), bool)
    one[:, [2, 7]] = True
    counts = np.bincount(H.RandomLegal(1)(None, one), minlength=11)
    assert counts[2] + counts[7] == 2000 and 850 < counts[2] < 1150
    with pytest.raises(ValueError):
        H.RandomLegal(0)(None, np.zeros((1, 11), bool))


def test_uniform_random_declares_the_in_core_random_bot_route():
    u = H.UniformRandom(seed=9)
    assert u.rust_routes() == [{"kind": "bot", "bot": "random", "seed": 9}] and u.ep_opp(3) == 0
    assert isinstance(u, H.OpponentMix) and isinstance(H.RandomLegal(0), H.TraineeInference)
    assert isinstance(H.StepCollector(4), H.Collector)


def test_the_lane_g_hooks_are_built_and_refuse_a_missing_input():
    """Lane G plugged in the three hooks J declared: each constructs from its inputs and refuses by name
    without them (never a silent fallback to a built one)."""
    assert H.OPPONENTS["production"] is H.ProductionMix and H.COLLECTORS["complete_game"] is H.CompleteGameCollector
    assert H.INFERENCE["learner"] is H.LearnerSampling
    with pytest.raises(ValueError, match="--pool"):
        H.ProductionMix()
    with pytest.raises(ValueError, match="--ckpt"):
        H.LearnerSampling(None)
    c = H.CompleteGameCollector(96)
    assert isinstance(c, H.Collector)
    c.observe_step(2)
    c.observe(np.array([True, False, True]))
    c.observe_fill()
    assert c.describe() == {"name": "complete_game", "target": 96, "host_steps": 2, "fills": 1, "episodes_ended": 4}
    assert not c.ready()


def test_the_production_arms_need_all_three_hooks(tmp_path):
    out = tmp_path / "t.json"
    for extra in (["--opponent", "production"], ["--collector", "complete_game"],
                  ["--opponent", "production", "--collector", "complete_game"]):
        assert T.main(["--out", str(out), "--n-envs", "2", *extra]) == 2
    # all three, but no pool: refused by the hook, by name
    assert T.main(["--out", str(out), "--n-envs", "2", "--opponent", "production", "--collector", "complete_game",
                   "--inference", "learner", "--ckpt", "/nonexistent.zip", "--device", "cpu"]) != 0
    assert not out.exists()


def test_step_collector_counts_windows():
    c = H.StepCollector(3)
    for i in range(7):
        c.observe(np.array([i == 4, False]))
    assert c.describe() == {"name": "fixed_window_steps", "n_steps": 3, "steps": 7, "windows": 2, "episodes_ended": 1}
    assert not c.ready()
    with pytest.raises(ValueError):
        H.StepCollector(0)


# ------------------------------------------------------------------ the GPU lock + --out

def test_gpu_lock_reexec_only_for_cuda_and_only_without_a_verified_holder():
    py = "/x/python3"
    cpu = ["--out", "o.json", "--inference", "t2", "--device", "cpu"]
    assert T.gpu_lock_reexec_argv(cpu, None, py) is None
    assert T.gpu_lock_reexec_argv(["--out", "o.json", "--device", "cuda"], None, py) is None   # random inference: no GPU
    assert T.gpu_lock_reexec_argv(["--out", "o.json", "--inference", "learner", "--device", "cuda"], None, py) is not None
    gpu = ["--out", "o.json", "--inference", "t2", "--device", "cuda:0", "--n-envs", "48"]
    assert T.gpu_lock_reexec_argv(gpu, None, py) == [py, "-m", "utils.gpu_lock", "--",
                                                     py, "-m", "main.rust_core_m5.throughput", *gpu]
    assert T.gpu_lock_reexec_argv(gpu, 1234, py) is None      # a VERIFIED holder ancestor: run here


def test_out_under_models_is_refused(tmp_path):
    models = tmp_path / "models"
    with pytest.raises(ValueError, match="run archive"):
        T.refuse_models_out(str(models / "run" / "t.json"), [None, models])
    with pytest.raises(ValueError):
        T.refuse_models_out(str(models), [models])
    assert T.refuse_models_out(str(tmp_path / "models_not" / "t.json"), [models]) == (tmp_path / "models_not" / "t.json")
    from utils.paths import main_models_dir

    md = main_models_dir()
    if md is not None:   # the default list holds the real archive
        with pytest.raises(ValueError):
            T.refuse_models_out(str(md / "x" / "throughput.json"))


def test_the_cli_refuses_models_before_building_anything(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(models))
    with pytest.raises(ValueError, match="run archive"):
        T.main(["--out", str(models / "run" / "t.json"), "--n-envs", "2"])
    assert not (models / "run").exists()


# ------------------------------------------------------------------ the smoke (both arms, real battles)

@pytest.mark.sim
@pytest.mark.integration
def test_smoke_both_arms_tiny(tmp_path):
    out = tmp_path / "throughput_smoke.json"
    t0 = time.monotonic()
    rc = T.main(["--out", str(out), "--n-envs", "2", "--threads", "2", "--pairs", "1", "--block-seconds", "0.5",
                 "--warmup-steps", "2", "--profile", "selfcheck", "--front", "proc"])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["schema"] == T.SCHEMA and res["regime"]["n_envs"] == 2 and res["regime"]["threads"] == 2
    for k in ("front", "profile", "inference", "opponent", "collector", "git", "load_start", "load_end", "cpu_count"):
        assert k in res["regime"], k
    assert res["regime"]["inference"]["name"] == "random_legal"
    assert res["regime"]["opponent"]["name"] == "uniform_random"
    assert res["regime"]["collector"]["name"] == "fixed_window_steps"
    # a busy box re-runs a block (the superseded row stays in `blocks`, uncounted)
    assert [(b["pair"], b["arm"]) for b in res["blocks"] if not b.get("superseded")] == [(0, "python"), (0, "rust")]
    for b in res["blocks"]:
        assert b["decisions"] > 0 and b["cpu_s_tree"] > 0 and b["cpu_us_per_decision"] > 0, b
    assert res["arm_info"]["rust"]["labels"] and res["arm_info"]["rust"]["after_freeze"]
    assert all(v == 0 for v in res["arm_info"]["rust"]["after_freeze"].values())
    assert res["arm_info"]["python"]["obs_source"] == "core"
    assert set(res["startup_s"]) == {"python", "rust"}
    r = res["ratios"]["decisions_per_s_rust_over_python"]
    assert r["point"] > 0 and r["n_pairs"] == 1
    print(f"smoke wall {time.monotonic() - t0:.1f} s")


def test_the_owner_breakdown_always_sums_to_the_step_and_bills_only_the_unhidden_core():
    from main.rust_core_m5.throughput import owner_breakdown

    serial = {"submit": 0.1, "flush": 1.0, "gpu_wait": 6.0, "draw": 0.1, "opp_draw": 0.2, "write": 0.3,
              "core": 1.5, "post": 0.1, "fill": 0.0}
    out = owner_breakdown(serial, 10.0)
    assert set(out) == {"core_step", "forward_launch_and_flush", "gpu_wait", "sampling", "host_glue"}
    assert abs(sum(out.values()) - 10.0) < 1e-9 and abs(out["host_glue"] - (0.4 + 0.7)) < 1e-9
    over = dict(serial, core=3.0, core_wait_unhidden=0.25)
    out = owner_breakdown(over, 9.0)
    assert out["core_step"] == 0.25 and out["core_step_total_both_halves"] == 3.0
    assert abs(sum(v for k, v in out.items() if k != "core_step_total_both_halves") - 9.0) < 1e-9
    py = owner_breakdown({"trainee_forward": 2.0, "vec_step_env_and_opponents": 50.0}, 54.5)
    assert abs(py["host_glue"] - 2.5) < 1e-9


def test_the_production_arm_names_parse_and_a_bad_one_is_refused():
    from main.rust_core_m5.throughput import parse_rust_arm

    assert parse_rust_arm("rust_serial_keyed") == ("serial", "keyed", None)
    assert parse_rust_arm("rust_overlap_generator_p4") == ("overlap", "generator", 4)
    for bad in ("rust", "rust_serial", "rust_serial_keyed_p0", "rust_serial_keyed_4", "rust_async_keyed"):
        assert parse_rust_arm(bad) is None, bad


def test_a_thread_count_suffix_parses_beside_the_snapshot_suffix():
    from main.rust_core_m5.throughput import parse_arm_threads, parse_rust_arm

    assert parse_rust_arm("rust_serial_keyed_t12") == ("serial", "keyed", None)
    assert parse_rust_arm("rust_serial_keyed_p8_t16") == ("serial", "keyed", 8)
    assert parse_arm_threads("rust_serial_keyed_t12") == 12
    assert parse_arm_threads("rust_serial_keyed_p8") is None and parse_arm_threads("rust_serial_keyed") is None
    assert parse_rust_arm("rust_serial_keyed_t0") is None


def test_a_busy_block_is_flagged_and_a_superseded_one_is_never_counted():
    assert T.block_busy(1.5) and not T.block_busy(0.4) and not T.block_busy(None)

    def blk(arm, pair, dps, **kw):
        return {"arm": arm, "pair": pair, "decisions": 100, "decisions_per_s": dps, "ms_per_vec_step": 1.0,
                "vec_steps_per_s": 1.0, "cpu_us_per_decision": 10.0, "inference_share": 0.0, **kw}

    blocks = [blk("rust_serial_keyed", 0, 1000.0), blk("rust_overlap_keyed", 0, 100.0, busy=True, superseded=True),
              blk("rust_overlap_keyed", 0, 1500.0, busy=False), blk("rust_serial_keyed", 1, 1000.0),
              blk("rust_overlap_keyed", 1, 1500.0, busy=True)]
    per, ratios = T.summarize(blocks, ("rust_serial_keyed", "rust_overlap_keyed"), seed=0)
    assert per["rust_overlap_keyed"]["blocks"] == 2 and per["rust_overlap_keyed"]["decisions_per_s_mean"] == 1500.0
    assert per["rust_overlap_keyed"]["busy_blocks_counted"] == 1                # still busy after re-runs: flagged
    r = ratios["decisions_per_s_rust_overlap_keyed_over_rust_serial_keyed"]
    assert r["per_pair"] == pytest.approx([1.5, 1.5])


def test_a_block_reads_the_box_and_the_per_phase_cpu():
    class Arm:
        name = "rust_serial_keyed"

        def step(self):
            t = time.process_time()
            while time.process_time() - t < 0.005:
                pass
            return 4, 0.0, 0.004, 0

        def roots(self):
            return []

    b = T.run_block(Arm(), seconds=0.3, min_steps=1)
    for k in ("load1_start", "system_busy_cpus", "self_cpus", "bystander_cpus", "busy", "core_s",
              "host_main_thread_cpu_share", "host_process_cpus"):
        assert k in b, k
    assert 0.5 < b["host_main_thread_cpu_share"] <= 1.2           # the main thread burned the block
    assert b["core_s"] > 0 and b["env_core_cpus_during_step"] == 0.0   # no core process in this arm
