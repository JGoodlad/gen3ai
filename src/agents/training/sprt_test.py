"""T6 SPRT PROMOTION (`gen3_sprt_promotion_v1`) — the test itself, and the three discipline rules.

The statistics are Fishtest's GSPRT (Van den Bergh): a pentanomial over mirrored pairs, the maximum-
likelihood pentanomial under each hypothesis's mean constraint solved for a 1-D Lagrange multiplier by
bisection, the LLR from the two constrained fits, a minimum pair count, Wald bounds, a declared cap.
"""
from __future__ import annotations

import json
import math
import random
from types import SimpleNamespace

import pytest

from agents.training import sprt as S
from agents.training import sprt_promotion as SP


# ---------------------------------------------------------------- the GSPRT

def test_the_bounds_are_walds_for_alpha_beta_five_percent():
    lo, hi = S.bounds()
    assert lo == pytest.approx(math.log(0.05 / 0.95)) == pytest.approx(-2.944, abs=1e-3)
    assert hi == pytest.approx(-lo)


def test_lambda_bisection_converges_to_the_constraint_and_is_deterministic():
    """(d) the λ solve is pinned: the constrained MLE's defining residual is ~0 to 1e-12 after the FIXED
    iteration count, for random pentanomials and both hypothesis means, and repeated solves are
    bit-identical."""
    rng = random.Random(7)
    for _ in range(200):
        counts = [rng.randrange(0, 50) for _ in range(5)]
        if not any(counts):
            continue
        p = S._empirical(counts)
        for mu in (S.P0, S.P1, 0.3, 0.8):
            lam = S.mle_lambda(p, mu)
            assert abs(S.constraint_residual(p, mu, lam)) < 1e-12, (counts, mu)
            assert S.mle_lambda(p, mu) == lam                         # bit-identical
            q = [pi / (1.0 + lam * (x - mu)) for pi, x in zip(p, S.SCORES)]
            assert sum(q) == pytest.approx(1.0, abs=1e-9)             # a distribution...
            assert sum(qi * x for qi, x in zip(q, S.SCORES)) == pytest.approx(mu, abs=1e-9)  # ...with mean mu
        assert S.llr(counts) == S.llr(counts)


def test_the_llr_points_the_right_way_and_is_zero_with_no_pairs():
    assert S.llr([0] * 5) == 0.0
    assert S.llr([0, 0, 10, 0, 30]) > 2.944          # 30 sweeps, 10 splits: strongly H1
    assert S.llr([30, 0, 10, 0, 0]) < -2.944
    mid = S.llr([10, 20, 40, 20, 10])                # mean score 0.5: closer to H0
    assert mid < 0


def test_the_llr_matches_the_simulators_vectorized_solve():
    np = pytest.importorskip("numpy")
    import importlib.util
    from utils.paths import repo_path

    spec = importlib.util.spec_from_file_location(
        "sprt_sim", repo_path("designs", "research_state", "measurements", "sprt_promotion", "simulate.py"))
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    counts = np.array([[1, 2, 3, 4, 5], [0, 0, 9, 0, 1], [7, 0, 0, 0, 0], [3, 1, 4, 1, 5]])
    got, var = sim.llr_many(counts, S.P0, S.P1, S.REGULARIZE, with_var=True)
    for row, g, v in zip(counts.tolist(), got, var):
        assert g == pytest.approx(S.llr(row), abs=1e-9)
        assert v == pytest.approx(S.llr_increment_var(row), abs=1e-9)


# ---------------------------------------------------------------- the sequential state

def test_a_strong_candidate_is_accepted_and_a_weak_one_rejected():
    good = S.SprtState(S.SprtConfig(min_pairs=0))
    while good.verdict == S.CONTINUE:
        good.add_batch([0, 4, 10, 10, 16])
    assert good.verdict == S.ACCEPT and good.reason == S.BY_BOUND
    bad = S.SprtState(S.SprtConfig(min_pairs=0))
    while bad.verdict == S.CONTINUE:
        bad.add_batch([16, 10, 10, 4, 0])
    assert bad.verdict == S.REJECT and bad.reason == S.BY_BOUND
    with pytest.raises(RuntimeError):
        good.add_batch([1, 0, 0, 0, 0])              # a decided test takes no more games


def test_no_decision_before_the_minimum_and_the_cap_is_a_REJECT():
    st = S.SprtState(S.SprtConfig(min_pairs=80, batch_pairs=40, max_pairs=200))
    assert st.add_batch([0, 0, 0, 0, 40]) == S.CONTINUE          # 40 sweeps, but below the minimum
    assert st.add_batch([0, 0, 0, 0, 40]) == S.ACCEPT            # at the minimum the bound decides
    # a candidate sitting AT the indifference point (score 0.525) runs into the cap: that is a REJECT
    mid = S.SprtState(S.SprtConfig(min_pairs=0, batch_pairs=40, max_pairs=80))
    assert mid.add_batch([19, 0, 0, 0, 21]) == S.CONTINUE
    assert (mid.add_batch([19, 0, 0, 0, 21]), mid.reason) == (S.REJECT, S.BY_CAP)
    assert mid.next_batch_pairs() == 0


def test_the_verdict_record_names_the_test_and_flags_its_score_as_selected():
    st = S.SprtState(S.SprtConfig(min_pairs=0))
    st.add_batch([0, 0, 0, 0, 40])
    out = st.to_json()
    assert out["config"]["schema"] == S.SCHEMA and out["config"]["bounds_mode"] in ("wald", "siegmund")
    assert "test_score_selected" in out and "score" not in out
    with pytest.raises(ValueError):
        S.SprtConfig(p0=0.55, p1=0.5)


def test_siegmund_bounds_move_inward_and_wald_does_not():
    counts = [5, 5, 10, 5, 5]
    lo, hi = S.bounds()
    assert S.decision_bounds(S.SprtConfig(bounds_mode="wald"), counts) == (lo, hi)
    slo, shi = S.decision_bounds(S.SprtConfig(bounds_mode="siegmund"), counts)
    assert lo < slo <= 0.0 <= shi < hi


# ---------------------------------------------------------------- the three discipline rules

def test_the_test_seed_namespace_is_disjoint_from_every_cycles():
    from agents.training.rust_eval.launch import cycle_seed

    cycles = {cycle_seed(0, s) for s in range(0, 40_000_000, 2_000_000)}
    tests = {SP.sprt_seed(0, s, b) for s in range(0, 40_000_000, 2_000_000) for b in range(50)}
    assert not cycles & tests                        # rule 2: no selection game is replayed as a decision game
    assert SP.sprt_seed(1, 2, 3) == SP.sprt_seed(1, 2, 3) != SP.sprt_seed(1, 2, 4)


def test_a_candidate_with_any_record_is_never_tested_again_and_an_interrupted_one_is_abandoned(tmp_path):
    d = str(tmp_path)
    assert SP.already_tested(d, 4_000_000) is None
    SP.append_log(d, {"event": "start", "step": 4_000_000})
    SP.append_log(d, {"event": "start", "step": 6_000_000})
    SP.append_log(d, {"event": "verdict", "step": 6_000_000, "promoted": False})
    assert SP.abandon_unfinished(d, emit=lambda *_: None) == [4_000_000]   # rule 3: interrupted ⇒ not promoted
    assert SP.already_tested(d, 4_000_000)["event"] == "abandoned"
    assert SP.already_tested(d, 6_000_000)["event"] == "verdict"
    assert SP.abandon_unfinished(d, emit=lambda *_: None) == []             # idempotent
    rows = [json.loads(x) for x in (tmp_path / SP.LOG_NAME).read_text().splitlines()]
    assert [r["event"] for r in rows] == ["start", "start", "verdict", "abandoned"]


def test_the_pool_is_split_evenly_and_a_batch_that_played_nothing_abandons_the_test():
    assert SP.split_pairs(40, ["s0", "s1", "s2"]) == {"s0": 14, "s1": 13, "s2": 13}
    job = SP.SprtJob(step=1, snapshot="x", sentinels=[{"label": "sentinel_0", "path": "p", "step": 0}],
                     cfg=S.SprtConfig(), run_seed=0, sf=1.0)
    items = job.plan_items(40)
    assert [(it.key, it.n_games) for it in items] == [("sentinel_0", 80)]  # pairs → games, even
    assert job.fold({"pairs": {}}) == S.REJECT and job.state.reason == "abandoned"


class _Pool:
    def __init__(self):
        self.added, self.summary = [], {}

    def add_from_path(self, path, step):
        self.added.append(step)

    def persist_summary(self, **kw):
        self.summary.update(kw)


def _cb(tmp_path, *, env_core="rust"):
    from agents.training.sprt_promotion import SprtPromotionMixin

    rec = {}
    cb = SprtPromotionMixin()
    cb._init_sprt(True)
    cb._sprt_cfg = S.SprtConfig(min_pairs=0, batch_pairs=40, max_pairs=200)
    cb._model_dir = str(tmp_path)
    cb._eval_root = str(tmp_path / ".eval_runs")
    cb._env_core = env_core
    cb._pool = _Pool()
    cb._pool_generation = 0
    cb.model = SimpleNamespace(seed=3)
    cb.logger = SimpleNamespace(record=lambda k, v: rec.__setitem__(k, v), dump=lambda step: None)
    cb._spawn_snapshot_ladder_update = lambda step: rec.__setitem__("ladder", step)
    return cb, rec


def _pending(tmp_path):
    snap = tmp_path / "snap.zip"
    snap.write_text("weights")
    return {"snapshot": str(snap), "sentinels": [{"label": "sentinel_0", "path": "p0", "step": 2_000_000},
                                                 {"label": "sentinel_1", "path": "p1", "step": 4_000_000}]}


def test_the_rust_path_tests_ONLY_its_own_fresh_pairs_and_promotes_on_accept(tmp_path):
    cb, rec = _cb(tmp_path)
    seen = []

    def play(job):
        seen.append((job.batch, [s["step"] for s in job.sentinels]))
        n = job.state.next_batch_pairs()
        return {"pairs": {"sentinel_0": [0, 0, 0, 0, n // 2], "sentinel_1": [0, 0, 0, 0, n - n // 2]}}

    cb._sprt_play_rust = play
    assert cb._sprt_begin(6_000_000, _pending(tmp_path), sf=0.9) is False
    assert cb._pool.added == [6_000_000] and rec["ladder"] == 6_000_000 and cb._pool_generation == 1
    assert all(pool == [2_000_000, 4_000_000] for _b, pool in seen)      # rule 1: the pool frozen at start
    assert rec["eval/sprt_promoted"] == 1.0
    log = SP.read_log(str(tmp_path))
    assert [r["event"] for r in log] == ["start", "verdict"] and log[-1]["promoted"] is True
    # rule 3: the same candidate is never tested twice
    cb._sprt_play_rust = lambda job: pytest.fail("a decided candidate was re-tested")
    cb._sprt_begin(6_000_000, _pending(tmp_path), sf=0.9)


def test_a_rejected_candidate_is_not_promoted_and_an_empty_pool_tests_nothing(tmp_path):
    cb, rec = _cb(tmp_path)
    cb._sprt_play_rust = lambda job: {"pairs": {"sentinel_0": [20, 0, 0, 0, 0], "sentinel_1": [20, 0, 0, 0, 0]}}
    cb._sprt_begin(8_000_000, _pending(tmp_path), sf=0.9)
    assert cb._pool.added == [] and rec["eval/sprt_promoted"] == 0.0
    assert cb._sprt_begin(9_000_000, {"snapshot": "x", "sentinels": []}, sf=0.9) is False
    assert SP.already_tested(str(tmp_path), 9_000_000) is None


def test_the_shipped_schedule_is_the_one_the_committed_monte_carlo_chose():
    from utils.paths import repo_path

    res = json.loads(repo_path("designs", "research_state", "measurements", "sprt_promotion",
                               "results.json").read_text())
    cfg = S.SprtConfig()
    assert (cfg.batch_pairs, cfg.min_pairs, cfg.max_pairs, cfg.bounds_mode) == (
        res["batch_pairs"], res["chosen"]["min_pairs"], res["cap_pairs"], res["chosen"]["bounds_mode"])
    assert (res["p0"], res["p1"], res["alpha"], res["beta"]) == (S.P0, S.P1, S.ALPHA, S.BETA)


def test_the_python_path_chains_batches_and_promotes_with_its_own_pushes(tmp_path, monkeypatch):
    from agents.training import eval_collect

    cb, rec = _cb(tmp_path, env_core="python")
    launched = []
    cb._sprt_launch_python = lambda job: launched.append(job.batch)
    cb._push_self_play_target = lambda sf: rec.__setitem__("pushed", sf)
    cb._prune_and_push_pfsp = lambda: rec.__setitem__("pfsp", True)
    assert cb._sprt_begin(6_000_000, _pending(tmp_path), sf=0.7) is True       # a batch is in flight
    assert launched == [0] and SP.already_tested(str(tmp_path), 6_000_000)["event"] == "start"
    batches = iter([{"pairs": {"sentinel_0": [5, 5, 10, 0, 0], "sentinel_1": [0, 0, 10, 5, 5]}},
                    {"pairs": {"sentinel_0": [0, 0, 0, 0, 20], "sentinel_1": [0, 0, 0, 0, 20]}}])
    monkeypatch.setattr(eval_collect, "merge_eval_results", lambda run_dir, names: (next(batches), []))
    for _ in range(2):
        cb._pending = {"kind": "sprt", "procs": [], "run_dir": str(tmp_path / "b"), "names": ["sentinel_0"]}
        cb._sprt_collect_python()
    assert launched == [0, 1]                                                  # one more batch, then a verdict
    assert cb._pool.added == [6_000_000] and rec["pushed"] == 0.7 and rec["pfsp"]
    assert cb._pool.summary["pool_generation"] == 1 and cb._sprt_job is None and cb._pending is None


def test_a_test_cut_off_by_shutdown_is_recorded_ABANDONED(tmp_path):
    cb, rec = _cb(tmp_path, env_core="python")
    cb._sprt_launch_python = lambda job: None
    cb._sprt_begin(6_000_000, _pending(tmp_path), sf=0.7)
    cb._sprt_abandon_in_flight("the drain budget ran out")
    last = SP.read_log(str(tmp_path))[-1]
    assert last["event"] == "verdict" and last["promoted"] is False and "abandoned" in last["reason"]
    assert cb._pool.added == []
