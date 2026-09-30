"""THE M5 LANE E GATE — slice N with a POLICY opponent (``rust_env_opponents_parity`` is the harness;
its docstring states the method and the bars).

COMMIT tier (routine, CPU): T2's EAGER backend vs the per-env EAGER ``RLPlayer`` path, on a pool of
perturbed fresh snapshots — greedy at the production stall threshold, and SAMPLED at a low threshold
(every episode ends in the stall forfeit) with a mid-run POOL REFRESH that must be a LOAD (the
service's ``*_after_freeze`` counters 0) and whose new snapshot must actually be played. No
allowlist: zero divergences of any kind, zero phantom polls. Teeth: one corrupted recorded action
fails; the wrapper polling the opponent on a step whose order is never sent fails
(``gen3_no_phantom_opponent_poll_v1``).

``slow``: the COMPILED per-env path (``--compile-opponents``). MILESTONE (``slow`` + an idle GPU,
``GEN3AI_TEST_ALLOW_GPU=1``, under ``scripts/ops/gpu_lock.sh``): T2's ``graph``
backend on CUDA vs the compiled CPU path on a real production pool.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from unittest.mock import MagicMock

import numpy as np
import pytest

from utils.paths import main_models_dir, src_path

pytestmark = [pytest.mark.sim, pytest.mark.integration]

FEATURES = ("--profile", "selfcheck", "--features", "emission-selfcheck")


@pytest.fixture(scope="module")
def built():
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the Lane E gate cannot run (install rustup)")
    crate = src_path("rust_env")
    env = dict(os.environ, CARGO_TARGET_DIR=str(crate / "target"))
    r = subprocess.run([cargo, "build", "--lib", *FEATURES, "--manifest-path", str(crate / "Cargo.toml")],
                       env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"building the cdylib failed:\n{r.stderr[-4000:]}"
    return True


def _cfg(**kw):
    base = {"pool": "fresh:2", "n_envs": 4, "threads": 2, "episodes": 10, "mode": "greedy", "device": "cpu",
            "backend": "eager", "buckets": [2, 8], "lanes": 1, "turn_limit": 0, "compile": False,
            "refresh_at": 0, "key_base": 71_000, "teams": "pool"}
    base.update(kw)
    return base


#: The DECLARED |Δ legal log-prob| bar per tier, from the measured compiled-vs-eager / eager-vs-eager
#: maxima (2026-09-29/30): eager CPU vs eager CPU 9.5e-7 (COMMIT), the COMPILED CPU per-env path vs
#: eager T2 1.43e-6 over 1,881 decisions, T2's graph backend on CUDA vs compiled CPU 4.5e-5 over 3,777.
#: The CPU bars are ~7-10x the measured maximum; the GPU bar is T2's own legal log-prob gate bar.
#: The tie rule (`judge_flips`): an argmax flip with margin < NEAR_TIE_FACTOR x bar is a TIE — each of
#: the two compared log-probs moves by less than the bar, so a flip needs margin < 2 x bar; at or
#: above it no rounding can explain the flip and it is FATAL.
BAR_EAGER = 1e-5
BAR_COMPILED_CPU = 1e-5
BAR_GPU = 1e-3


def _assert_clean(summary, rep, *, dlogp_bar):
    from agents.training.rust_env_opponents_parity import NEAR_TIE_FACTOR, judge_flips, near_ties

    assert rep["div"] == {}, (rep["div"], {k: rep.get(k) for k in ("row_examples", "outcome_examples", "p2_count_examples")})
    assert rep["phantom_polls"] == 0, rep["phantom_polls"]
    assert rep["episodes"] == summary["n_episodes"] and rep["decisions"] == summary["p2_decisions"] > 0, (rep, summary)
    assert rep["max_dlogp"] < dlogp_bar, rep["max_dlogp"]
    ties, fatal = judge_flips(rep, dlogp_bar)
    assert not fatal, f"argmax flips at margin >= {NEAR_TIE_FACTOR} x {dlogp_bar}: {fatal}"
    near = near_ties(rep, dlogp_bar)
    print(f"[laneE gate] {rep['decisions']} decisions, max|dlogp| {rep['max_dlogp']:.3g} (bar {dlogp_bar}), "
          f"{ties} TIE flip(s) (margin < {NEAR_TIE_FACTOR * dlogp_bar:g}), decisions in the tie band {near}: "
          f"{[f for f in rep['flips']][:5]}")
    assert all(v == 0 for v in summary["svc_counters_delta"].values()), summary["svc_counters_delta"]
    assert all(v == 0 for v in summary["core_after_freeze"].values()), summary["core_after_freeze"]
    return ties


def test_commit_greedy_t2_equals_the_per_env_path(built, tmp_path):
    from agents.training.rust_env_opponents_parity import run

    summary, rep = run(_cfg(episodes=6), str(tmp_path))
    _assert_clean(summary, rep, dlogp_bar=BAR_EAGER)
    assert rep["decisions"] >= 120, rep["decisions"]


def test_commit_sampled_with_stall_forfeits_and_a_refresh(built, tmp_path):
    from agents.training.rust_env_opponents_parity import run

    summary, rep = run(_cfg(n_envs=8, episodes=40, mode="sampled", turn_limit=6, refresh_at=20, key_base=71_900),
                       str(tmp_path))
    _assert_clean(summary, rep, dlogp_bar=BAR_EAGER)
    assert rep["p1_forfeits"] == rep["episodes"], "every episode must end in the stall forfeit at this threshold"
    ref = summary["refresh"]
    assert ref is not None and len(ref["routes_loaded"]) == 1, ref
    assert summary["route_counts"][ref["routes_loaded"][0]] > 0, ("the refreshed snapshot was never played", summary["route_counts"])
    assert summary["svc_loads"] == 3, summary["svc_loads"]           # 2 at startup + 1 refresh: LOADS, never compiles


def test_commit_keyed_sampling_replays_from_its_key(built, tmp_path):
    """M5 Lane G (F-LE-8): p2's sample is the KEYED DRAW; the replay recomputes it from the per-env path's
    OWN log-probs and the decision's key (no generator state crosses) — equal on every sent decision,
    through stall forfeits and a mid-run refresh."""
    from agents.training.rust_env_opponents_parity import run

    summary, rep = run(_cfg(n_envs=8, episodes=40, mode="keyed", turn_limit=6, refresh_at=20, key_base=71_950),
                       str(tmp_path))
    _assert_clean(summary, rep, dlogp_bar=BAR_EAGER)
    assert rep["decisions"] >= 200, rep["decisions"]
    assert summary["refresh"] is not None and summary["svc_loads"] == 3


def test_the_keyed_gate_has_teeth(built, tmp_path):
    """The replay must draw from the RECORDED key: shift every decision's key (its dec_n) and the replay's
    own draws move off the recorded actions — FATAL flips (margins far above the tie band), while the
    recorded actions are still played on, so the battles stay aligned."""
    from agents.training.rust_env_opponents_parity import judge_flips, run

    def mutate(rec):
        for ep in rec["episodes"]:
            for d in ep.p2:
                d.dec_n += 1000

    _summary, rep = run(_cfg(n_envs=2, episodes=2, mode="keyed", key_base=71_560), str(tmp_path), mutate=mutate)
    _ties, fatal = judge_flips(rep, BAR_EAGER)
    assert any(f["kind"] == "keyed" for f in fatal), rep["flips"]


def test_the_gate_has_teeth(built, tmp_path):
    """One recorded opponent action moved to another legal action must be caught."""

    from agents.training.rust_env_opponents_parity import run

    from agents.training.rust_env_opponents_parity import judge_flips

    def mutate(rec):
        for e in rec["episodes"]:
            for d in e.p2:
                legal = np.flatnonzero(d.mask)
                srt = np.sort(d.logp[legal])[::-1]
                if len(legal) > 1 and srt[0] - srt[1] > 0.1:     # a DECISIVE row: the flip is not a tie
                    d.greedy = int(legal[legal != d.greedy][0])
                    return
        raise AssertionError("no decisive decision with two legal actions")

    _summary, rep = run(_cfg(n_envs=2, episodes=2, key_base=71_500), str(tmp_path), mutate=mutate)
    ties, fatal = judge_flips(rep, BAR_EAGER)
    assert len(fatal) == 1 and ties == 0 and fatal[0]["kind"] == "greedy", rep["flips"]


def test_the_tie_rule_is_declared_and_exact():
    """`judge_flips`: margin < 2 x bar is a TIE, >= is FATAL, a flip with no margin is FATAL."""
    from agents.training.rust_env_opponents_parity import judge_flips

    rep = {"flips": [{"margin": 0.0}, {"margin": 1.99e-5}, {"margin": 2e-5}, {"margin": None}, {"margin": 0.3}]}
    ties, fatal = judge_flips(rep, 1e-5)
    assert ties == 2 and [f["margin"] for f in fatal] == [2e-5, None, 0.3]


def test_the_declared_torch_state_is_pinned_restored_and_guarded():
    """F-LJ-6: the gate pins its thread count, RESTORES the one it found, and refuses an undeclared
    global (fp32 matmul precision) loudly. FAILS on revert (the leak: `record` set the count and left it)."""
    import torch

    from agents.training.rust_env_opponents_parity import GateStateError, declared_torch_state

    before = torch.get_num_threads()
    with declared_torch_state(3):
        assert torch.get_num_threads() == 3
    assert torch.get_num_threads() == before
    prec = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        with pytest.raises(GateStateError, match="float32_matmul_precision"):
            with declared_torch_state(2):
                pass
        assert torch.get_num_threads() == before
    finally:
        torch.set_float32_matmul_precision(prec)
    with pytest.raises(GateStateError, match="num_threads"):       # something inside moved it
        with declared_torch_state(2):
            torch.set_num_threads(5)
    assert torch.get_num_threads() == before


def test_the_gate_pool_does_not_depend_on_the_process_thread_count(tmp_path):
    """F-LJ-6's mechanism, pinned: a fresh snapshot built after the process was left at 8 threads and
    one built at 3 threads are BIT-IDENTICAL (built at BUILD_THREADS). FAILS on revert: unpinned, the
    orthogonal init differs in ~94 of 721 tensors."""
    import io
    import zipfile

    import torch

    from agents.training.rust_env_opponents_parity import build_gate_pool

    got = []
    before = torch.get_num_threads()
    try:
        for k, t in enumerate((8, 3)):
            torch.set_num_threads(t)
            build_gate_pool(tmp_path / f"p{k}" / "pool", "fresh:0")
            assert torch.get_num_threads() == t, "the build must restore the count it found"
            z = next((tmp_path / f"p{k}" / "pool_held").glob("snapshot_*.zip"))
            got.append(torch.load(io.BytesIO(zipfile.ZipFile(z).read("policy.pth")), map_location="cpu", weights_only=False))
    finally:
        torch.set_num_threads(before)
    a, b = got
    assert a.keys() == b.keys()
    assert [k for k in a if not torch.equal(a[k], b[k])] == []


def test_the_wrapper_never_polls_the_opponent_on_a_step_it_does_not_send():
    """gen3_no_phantom_opponent_poll_v1: with ``agent2_to_move`` False the order would be dropped, so
    ``choose_move`` (an embed that records a decision, a sample, a bot's RNG draw) must not run.
    FAILS on revert: the wrapper called it whenever ``battle2`` was not a ``wait``."""
    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

    env = MagicMock()
    env.agent1.username, env.agent2.username = "a1", "a2"
    env.observation_spaces = {"a1": MagicMock()}
    env.action_spaces = {"a1": MagicMock()}
    env.battle2.wait = False
    env.battle2.teampreview = False
    env.agent2_to_move = False
    env.step.return_value = ({"a1": 0}, {"a1": 0.0}, {"a1": False}, {"a1": False}, {"a1": {}})
    opp = MagicMock()
    w = SingleAgentWrapper(env, opp)
    w._settle_opponent_battle = lambda: None
    w.step(0)
    opp.choose_move.assert_not_called()
    env.agent2_to_move = True
    w.step(0)
    opp.choose_move.assert_called_once()


@pytest.mark.slow
def test_slow_compiled_per_env_path(built, tmp_path):
    from agents.training.rust_env_opponents_parity import run

    summary, rep = run(_cfg(n_envs=8, episodes=24, mode="sampled", compile=True, key_base=72_000), str(tmp_path))
    _assert_clean(summary, rep, dlogp_bar=BAR_COMPILED_CPU)
    assert rep["compiled"]


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["sampled", "greedy", "keyed"])
def test_milestone_gpu_graph_backend_vs_compiled_cpu_on_a_real_pool(built, tmp_path, mode):
    """Run under ``scripts/ops/gpu_lock.sh`` (the GPU lock, re-entrant) with ``GEN3AI_TEST_ALLOW_GPU=1``."""
    import torch

    if os.environ.get("GEN3AI_TEST_ALLOW_GPU") != "1" or not torch.cuda.is_available():
        pytest.skip("GPU tier: set GEN3AI_TEST_ALLOW_GPU=1 on an idle GPU (under the GPU lock)")
    models = main_models_dir()
    snaps = None if models is None else models / "ai_v14_06_lbat_ctrl_fix" / "snapshots"
    if snaps is None or not snaps.is_dir():
        pytest.skip("no run archive with ai_v14_06_lbat_ctrl_fix/snapshots")
    from agents.training.rust_env_opponents_parity import run

    summary, rep = run(_cfg(pool=f"run:{snaps}:6", device="cuda", backend="graph", buckets=[2, 8, 16], n_envs=16,
                            threads=4, episodes=64, mode=mode, compile=True, refresh_at=120,
                            key_base={"sampled": 73_000, "greedy": 74_000, "keyed": 75_000}[mode]), str(tmp_path))
    _assert_clean(summary, rep, dlogp_bar=BAR_GPU)


def test_the_near_tie_counts_use_the_tiers_bar_and_the_flip_rules_threshold():
    """One threshold: a decision is in the tie band iff its margin < NEAR_TIE_FACTOR x the TIER's bar —
    the same band `judge_flips` excuses — not a fixed 2e-3 (coordinator, 2026-09-30)."""
    from agents.training.rust_env_opponents_parity import near_ties

    rep = {"margins_greedy": [0.0, 1.5e-5, 2.5e-5, 1e-3], "margins_sample": [1e-6, 3e-3]}
    assert near_ties(rep, 1e-5) == {"greedy": 2, "sample": 1}
    assert near_ties(rep, 1e-3) == {"greedy": 4, "sample": 1}          # 1e-3 < 2 x 1e-3
    assert near_ties({}, 1e-5) == {"greedy": 0, "sample": 0}
