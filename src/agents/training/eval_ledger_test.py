"""``eval_ledger`` — the §0b COUNT-row schema, its validator, the per-writer shard and the reader.

Every test builds a VALID row first (``make_row``) and breaks ONE thing: a schema that accepts a row it should
not is the failure that matters, so each rule has a row that must be refused for exactly that rule."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from agents.training import eval_ledger as L

SHA_A, SHA_B = "a" * 64, "b" * 64
TEAMS = {L.team_id("team one"): {"p": [2, 1], "o": [2, 1]}, L.team_id("team two"): {"p": [2, 1], "o": [2, 1]}}


def make_row(row_id: str = "w1:0", **over):
    regime = L.with_regime_id({"play": "greedy", "opponent_play": "greedy", "mirrored": True,
                               "mirror_rule": "gen3_mirrored_pairs_v1", "eval_core": "rust", "turn_limit": 250,
                               "seed_rule": "gen3_eval_game_seed_v1", "team_source": "pool"})
    p = {"id": "run_a@100", "sha256": SHA_A, "path": "/x/a.zip", "run": "run_a", "step": 100, "rung": "explicit_zip"}
    o = {"id": "run_b@200", "sha256": SHA_B, "path": "/x/b.zip", "run": "run_b", "step": 200, "rung": "explicit_zip"}
    # 2 pairs, 4 games: player W L / W L -> pairs (W,L) twice = 2 half-points each, W=2, L=2
    row = {"schema": L.SCHEMA, "row_id": row_id, "supersedes": None, "ts": "2026-10-03T12:00:05+00:00",
           "t_start": "2026-10-03T12:00:00+00:00", "t_end": "2026-10-03T12:00:04+00:00", "run": "study",
           "commit": "deadbeef", "player": p, "opponent": o, "regime": regime, "compute": {"device": "cpu"},
           "purpose": "audit", "counts": {"w": 2, "l": 2, "d": 0},
           "pairs": {"counts": [0, 0, 2, 0, 0], "n_pairs": 2, "voided": 0}, "teams": copy.deepcopy(TEAMS),
           "seed": {"rule": "gen3_eval_game_seed_v1", "schedule_seed": 0, "schedule_key": "h2h:k", "batch": 0,
                    "cycle_seed": 7, "item_key": "h2h", "game_lo": 0, "game_hi": 3}}
    row.update(over)
    return row


def test_a_valid_row_passes():
    assert L.validate_row(make_row()) == []
    L.check_row(make_row())


@pytest.mark.parametrize("name,mutate,needle", [
    ("schema", lambda r: r.update(schema="v0"), "schema"),
    ("missing key", lambda r: r.pop("teams"), "missing"),
    ("unknown key", lambda r: r.update(extra=1), "unknown keys"),
    ("purpose outside the closed list", lambda r: r.update(purpose="study"), "purpose"),
    ("bad timestamp", lambda r: r.update(t_start="yesterday"), "t_start"),
    ("span reversed", lambda r: r.update(t_start="2026-10-03T13:00:00+00:00"), "t_start is after t_end"),
    ("bad sha", lambda r: r["player"].update(sha256="xyz"), "sha256"),
    ("regime tampered", lambda r: r["regime"].update(turn_limit=100), "regime_id"),
    ("regime play vocabulary", lambda r: r["regime"].update(play="argmax"), "play"),
    ("counts negative", lambda r: r["counts"].update(w=-1), "non-negative"),
    ("pairs sum", lambda r: r["pairs"].update(n_pairs=3), "n_pairs"),
    ("games != 2 pairs", lambda r: r["counts"].update(w=3), "games"),
    ("half-points != 2W + D", lambda r: r["pairs"].update(counts=[0, 0, 0, 2, 0]), "half-points"),
    ("regime tampered (mirrored flag)", lambda r: r["regime"].update(mirrored=False), "regime_id"),
    ("team id", lambda r: r["teams"].update({"bad": {"p": [0, 0], "o": [0, 0]}}), "team id"),
    ("team wins exceed games", lambda r: r["teams"][next(iter(r["teams"]))].update(p=[1, 2]), "more wins than games"),
    ("team games do not sum", lambda r: r["teams"][next(iter(r["teams"]))].update(p=[3, 1]), "team games"),
    ("team wins do not match W", lambda r: r["teams"][next(iter(r["teams"]))].update(p=[2, 2]), "wins"),
    ("seed range", lambda r: r["seed"].update(game_hi=9), "game-index range"),
])
def test_each_rule_refuses_its_own_broken_row(name, mutate, needle):
    row = make_row()
    mutate(row)
    problems = L.validate_row(row)
    assert problems, f"{name}: a broken row passed"
    assert any(needle in p for p in problems), (name, problems)


def test_an_unmirrored_row_carries_null_pairs_and_the_arithmetic_still_ties():
    regime = L.with_regime_id({**make_row()["regime"], "mirrored": False})
    row = make_row(regime=regime, pairs=None, seed={**make_row()["seed"], "game_hi": 3})
    assert L.validate_row(row) == []
    bad = make_row(regime=regime)          # pairs still present on an unmirrored row
    assert any("pairs: must be null" in p for p in L.validate_row(bad))


def test_regime_id_is_a_function_of_the_identity_fields_only():
    base = make_row()["regime"]
    assert L.regime_id(base) == base["regime_id"]
    assert L.regime_id({**base, "regime_id": "ignored"}) == base["regime_id"]
    assert L.regime_id({**base, "play": "sampled"}) != base["regime_id"]


def test_team_id_names_a_team_not_a_position():
    assert L.team_id("abc") == L.team_id("abc") and L.team_id("abc") != L.team_id("abd")
    assert L.team_id("abc").startswith("t:") and len(L.team_id("abc")) == 18


def test_writer_appends_one_validated_row_per_line_to_its_own_shard(tmp_path):
    w = L.LedgerWriter(tmp_path, writer_id="w1")
    assert w.path.name == "ledger.w1.jsonl" and w.next_row_id() == "w1:0"
    w.append(make_row("w1:0"))
    w.append(make_row("w1:1"))
    lines = w.path.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["row_id"] == "w1:1"
    other = L.LedgerWriter(tmp_path, writer_id="w2")
    other.append(make_row("w2:0"))
    assert {Path(p).name for p in L.shard_paths(tmp_path)} == {"ledger.w1.jsonl", "ledger.w2.jsonl"}
    assert [r["row_id"] for r in L.read_rows(tmp_path)] == ["w1:0", "w1:1", "w2:0"]


def test_writer_refuses_a_row_that_fails_the_schema_or_skips_an_id(tmp_path):
    w = L.LedgerWriter(tmp_path, writer_id="w1")
    with pytest.raises(L.LedgerSchemaError, match="next id"):
        w.append(make_row("w1:5"))
    bad = make_row("w1:0")
    bad["counts"]["w"] = 7
    with pytest.raises(L.LedgerSchemaError):
        w.append(bad)
    assert not w.path.exists(), "a refused row must leave no trace"


def test_reader_validates_every_line_and_names_the_file_and_line(tmp_path):
    shard = tmp_path / "ledger.w9.jsonl"
    good, bad = make_row("w9:0"), make_row("w9:1")
    bad["purpose"] = "nonsense"
    shard.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(L.LedgerSchemaError, match=r"ledger\.w9\.jsonl:2"):
        L.read_rows(tmp_path)
    shard.write_text(json.dumps(good) + "\nnot json\n")
    with pytest.raises(L.LedgerSchemaError, match="not JSON"):
        L.read_rows(tmp_path)


def test_reader_drops_a_superseded_row_and_reads_a_closed_gz_shard(tmp_path):
    first, fix = make_row("w1:0"), make_row("w2:0", supersedes="w1:0")
    (tmp_path / "ledger.w1.jsonl").write_text(json.dumps(first) + "\n")
    with gzip.open(tmp_path / "ledger.w2.jsonl.gz", "wt") as f:
        f.write(json.dumps(fix) + "\n")
    assert [r["row_id"] for r in L.read_rows(tmp_path)] == ["w2:0"]


def test_a_directory_under_models_is_refused(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    (archive / "run_x").mkdir(parents=True)
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(archive))
    for bad in (archive, archive / "run_x", archive / "_ledger"):
        with pytest.raises(L.LedgerPathError, match="REFUSED"):
            L.refuse_under_models(bad)
        with pytest.raises(L.LedgerPathError):
            L.LedgerWriter(bad)
    L.refuse_under_models(tmp_path / "elsewhere")           # outside: fine
    from utils.paths import repo_root

    with pytest.raises(L.LedgerPathError):                   # this checkout's own models/ too (a worktree's dies with it)
        L.refuse_under_models(repo_root() / "models" / "anything")
