"""``main.h2h`` end to end on the real engine (CPU, tiny): an edge played in batches writes valid §0b v2 rows under
claims for one request, a re-run resumes instead of replaying, a longer plan plays only the new batches, ``read``
pools them, the ledger audits clean, every row's outcome digest is the digest of the games it played, and
``models/`` is refused (except the archive's own ledger). The numbers are not the point (perturbed-fresh
checkpoints): the plumbing is.

The plays are MODULE fixtures (the tier budget is per test call); the tests only read what they wrote."""
from __future__ import annotations

import json

import pytest

from agents.training import eval_ledger as L
from agents.training import mirrored_pairs as MP
from main.h2h import cli as CLI
from main.h2h import play as PL
from main.h2h import stats as ST

pytestmark = [pytest.mark.sim, pytest.mark.integration]

COMPUTE = PL.Compute(device="cpu", backend="eager", n_envs=8, threads=2, torch_threads=2, front="ffi",
                     profile="selfcheck")


ROWS = L.ReaderDecl(name="h2h_edge_test", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                    selection="include", flags_ok=frozenset(), inference="conditional")


def rows_of(out):
    return list(L.read(ROWS, root=out).rows)


def shards_of(out):
    return sorted(p.name for p in (out / "rows" / "h2h").glob("*"))


def edge(out, a, b, pairs, batch_pairs=3):
    return PL.play_edge(str(out), PL.resolve_player(a), PL.resolve_player(b), pairs=pairs, batch_pairs=batch_pairs,
                        schedule_seed=3, run_label="test-study", compute=COMPUTE, emit=lambda _m: None)


@pytest.fixture(scope="module")
def played(built, checkpoints, tmp_path_factory):
    """One edge played as 2 batches of 3 pairs, re-run with the same plan, then extended to 3 batches."""
    a, b = checkpoints
    out = tmp_path_factory.mktemp("h2h_edge") / "ledger"
    s1 = edge(out, a, b, 6)
    rows1, shards1 = rows_of(out), shards_of(out)
    s2 = edge(out, a, b, 6)
    shards2 = shards_of(out)
    s3 = edge(out, a, b, 9)
    return {"out": out, "a": a, "b": b, "s1": s1, "s2": s2, "s3": s3, "rows1": rows1, "shards1": shards1,
            "shards2": shards2, "rows3": rows_of(out)}


def test_each_batch_is_one_valid_row_and_the_edge_pools_them(played):
    rows, s1 = played["rows1"], played["s1"]
    assert [r["seed"]["batch"] for r in rows] == [0, 1] and all(L.validate_row(r) == [] for r in rows)
    assert all(r["run"] == "test-study" and r["purpose"] == "audit" and r["regime"]["mirrored"] for r in rows)
    assert all(r["schema"] == L.SCHEMA and r["regime"]["protocol"] == PL.PROTOCOL for r in rows)
    assert len({r["request"]["id"] for r in rows}) == 1 and [r["request"]["batch"] for r in rows] == [0, 1]
    assert s1["pairs"] == 6 and s1["games"] == 12 and s1["batches"] == 2
    assert s1["pair_counts"] == MP.add_counts(*[r["pairs"]["counts"] for r in rows])
    assert {r["player"]["sha256"] for r in rows} != {r["opponent"]["sha256"] for r in rows}


def test_a_rerun_of_the_same_plan_plays_and_writes_nothing(played):
    assert played["shards2"] == played["shards1"], "no new shard: nothing was played"
    assert played["s2"]["pair_counts"] == played["s1"]["pair_counts"]


def test_a_longer_plan_plays_only_the_new_batch_and_never_replays_a_banked_one(played):
    rows3 = played["rows3"]
    assert len(rows3) == 3 and played["s3"]["pairs"] == 9
    old = [r for r in rows3 if r["seed"]["batch"] < 2]
    assert [r["pairs"]["counts"] for r in old] == [r["pairs"]["counts"] for r in played["rows1"]]
    assert len({r["seed"]["cycle_seed"] for r in rows3}) == 3, "each batch is its own games"
    assert len(shards_of(played["out"])) == 2, "the new batch is a new writer's shard, the old one untouched"
    assert all(n.endswith(".jsonl.gz") for n in shards_of(played["out"])), "a clean exit closes its shard"


def test_the_ledger_audits_clean_after_resumes(played):
    from agents.training.eval_ledger import audit as AU

    rep = AU.audit(played["out"])
    assert rep.ok, rep.problems
    assert rep.counts["claims"] == 3 and rep.counts["live_claims"] == 0 and rep.counts["requests"] == 1


def test_a_different_batch_size_over_banked_rows_is_refused(played):
    with pytest.raises(PL.H2HError, match="different batch size"):
        edge(played["out"], played["a"], played["b"], 9, batch_pairs=4)


def test_the_cli_reads_the_rows_and_its_pooled_numbers_are_the_readers(played, capsys):
    assert CLI.main(["read", str(played["out"]), "--json"]) == 0
    summ = json.loads(capsys.readouterr().out)
    assert len(summ) == 1 and summ[0]["pairs"] == 9
    again = ST.edge_summary(played["rows3"])
    assert summ[0]["score"] == pytest.approx(again["score"]) and summ[0]["pair_counts"] == again["pair_counts"]
    assert CLI.main(["read", str(played["out"])]) == 0
    assert "all valid under gen3_eval_count_row_v2" in capsys.readouterr().out


def test_play_refuses_to_write_under_models(built, checkpoints, tmp_path, monkeypatch):
    a, b = checkpoints
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(archive))
    with pytest.raises(L.LedgerPathError, match="REFUSED"):
        CLI.main(["play", "--player", a, "--opponent", b, "--pairs", "2", "--out", str(archive / "ledger")])
    assert not (archive / "ledger").exists()


def test_each_rows_digest_is_the_digest_of_the_games_it_played(built, checkpoints, played):
    """Replay batch 0 of the banked edge on a fresh engine: its outcome vector hashes to the row's digest, and the
    game-for-game replay audit (§9.2) holds."""
    a, b = (PL.resolve_player(x) for x in checkpoints)
    row = next(r for r in played["rows1"] if r["request"]["batch"] == 0)
    eng = PL.H2HEngine(a, b, COMPUTE, emit=lambda _m: None)
    try:
        games, _st = eng.play_batch(row["pairs"]["n_pairs"], row["seed"]["cycle_seed"])
    finally:
        eng.close()
    sc = PL.score_games(games, eng.team_packed, row["pairs"]["n_pairs"])
    assert sc.outcome_digest == row["compute"]["outcome_digest"]
    assert sc.near_tie_idx == row["compute"]["near_tie_games"]
