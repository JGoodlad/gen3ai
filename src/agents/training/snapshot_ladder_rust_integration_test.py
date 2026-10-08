"""The snapshot ladder's games ON THE RUST EVAL ENGINE, end to end (CPU, tiny): P2 (2026-10-06).

A run dir with two seeded perturbed-fresh production snapshots; ``_measure_missing`` plays the pair through
``snapshot_ladder_play`` and appends ONE row. FAILS ON REVERT of the move: the poke-env path wrote a row with no
transport, no core stamp, no regime and no per-seat counts. The numbers are not the point; the plumbing is."""
from __future__ import annotations

import json

import pytest

from agents.training import snapshot_ladder as sl
from agents.training import snapshot_ladder_play as LP

pytestmark = [pytest.mark.sim, pytest.mark.integration]

ENGINE = dict(n_envs=8, threads=2, torch_threads=2, front="ffi", profile="selfcheck")


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    from agents.training.rust_eval import offline as PAR
    from agents.training.rust_rollout.testkit import build_selfcheck

    build_selfcheck()
    run = tmp_path_factory.mktemp("ladder_rust") / "run_ladder_test"
    with PAR.declared_torch_state(1):
        PAR.build_models(run / "snapshots", n_sentinels=1)       # snapshot_..1000.zip, snapshot_..2000.zip
    steps = sl.pool_snapshot_steps(str(run))
    a, b = steps[1], steps[0]
    n = sl._measure_missing(str(run), [(a, b)], n_games=8, engine_kw=ENGINE)
    again = sl._measure_missing(str(run), [(a, b)], n_games=8, engine_kw=ENGINE)   # measured: plays nothing
    rows = [json.loads(x) for x in open(sl.games_log_path(str(run)))]
    with LP.open_engine(str(run), [(a, b)], **ENGINE) as eng:
        r0, r0b, r1 = eng.play(a, b, 8, batch=0), eng.play(a, b, 8, batch=0), eng.play(a, b, 8, batch=1)
    return {"run": str(run), "a": a, "b": b, "n": n, "again": again, "rows": rows, "r0": r0, "r0b": r0b, "r1": r1}


def test_the_edge_is_played_on_the_rust_eval_engine_and_stamped(played):
    assert played["n"] == 1 and played["again"] == 0 and len(played["rows"]) == 1
    r = played["rows"][0]
    assert (r["a"], r["b"]) == (played["a"], played["b"])
    assert r["transport"] == "rust_eval" and r["encoder"] == "rust" and r["protocol"] == LP.PROTOCOL
    assert r["mirrored"] is True and r["n_pairs"] == 4 and r["seat_split"] == [2, 2]
    # `core_stamp` is the eval core's build stamp: the in-process `ffi` front of this test records "" (the process
    # front, the ladder's default, records the build), so only its PRESENCE is asserted here
    assert "core_stamp" in r and r["regime_id"] and len(r["sha_a"]) == 64 and r["sha_a"] != r["sha_b"]
    assert r["games_played"] == 8 and r["games"] + r["draws"] == 8 and r["wins_a"] <= r["games"]
    assert sum(r["by_seat"]["a_p1"]) == 4 and sum(r["by_seat"]["b_p1"]) == 4
    assert r["wins_a"] == r["by_seat"]["a_p1"][0] + r["by_seat"]["b_p1"][0]


def test_an_edge_is_a_pure_function_of_the_pair_and_its_batch(played):
    keep = ("wins_a", "draws", "by_seat", "pair_counts", "cycle_seed")
    assert {k: played["r0"][k] for k in keep} == {k: played["r0b"][k] for k in keep}
    assert {k: played["rows"][0][k] for k in keep} == {k: played["r0"][k] for k in keep}
    assert played["r1"]["cycle_seed"] != played["r0"]["cycle_seed"]   # a re-measurement takes NEW games


def test_the_fit_over_rust_rows_is_stamped_rust(played, monkeypatch):
    monkeypatch.setattr(sl.elo_mod, "load_bot_anchors", lambda: None)
    lad = sl.fit_ladder(played["run"], write=False)
    assert lad["recipe"]["transport"] == "rust_eval" and lad["n_frozen_pairs_measured"] == 1
