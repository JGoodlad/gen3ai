"""`python -m main.elo compare` - the matched-COUNT cross-run read (`main/elo_compare.py`).

Fixtures are tiny runs whose `games.jsonl` is written pair by pair (all unstamped rows, so one transport);
the committed `ladder.json` is a REAL stamped fit of the steps the test names, so a groomed pool is a committed
file that rates fewer steps than the log holds. Nothing here plays a game or touches `models/`.
"""
from __future__ import annotations

import json
import os

import pytest

from agents.training import elo as elo_mod
from agents.training import snapshot_ladder as sl
from main import elo as elo_cli
from main import elo_compare as ec

M = 1_000_000


@pytest.fixture(autouse=True)
def _fixed_anchors(monkeypatch):
    """The bot anchor file is CWD-relative; pin it so `anchored_to_bots` does not depend on where pytest ran."""
    monkeypatch.setattr(elo_mod, "load_bot_anchors",
                        lambda path=elo_mod.BOT_ANCHORS_PATH: {"ratings": {"random": 1000.0}, "base": 1500.0})


def _build(tmp_path, name, steps, *, committed=None, gain=6, log=True, stamp=True):
    """A run whose later snapshots beat earlier ones by `gain` points of win rate per step of distance."""
    run = str(tmp_path / name)
    os.makedirs(os.path.join(run, "snapshot_ladder"))
    for i, lo in enumerate(steps):
        for j in range(i + 1, len(steps)):
            wins = min(95, 50 + gain * (j - i))
            sl._append_game(run, steps[j], lo, wins, 100)
    doc = sl.fit_ladder(run, write=False, steps=committed or steps)
    if not stamp:
        doc.pop("recipe", None)
    with open(sl.ladder_json_path(run), "w") as fh:
        json.dump(doc, fh)
    if not log:
        os.remove(sl.games_log_path(run))
    return run


def _final(run, name="final_model.zip"):
    open(os.path.join(run, name), "w").close()


S5 = [i * M for i in range(1, 6)]


def test_both_sides_are_refit_on_their_first_n_not_read_off_the_final_fit(tmp_path):
    a = _build(tmp_path, "a", S5, gain=6)
    b = _build(tmp_path, "b", S5[:3], gain=9)
    rep = ec.compare([a, b])
    assert rep["at_snapshots"] == 3 and rep["available_matched"] == 3
    # A rates 5 nodes in its final fit; the read is its FIRST-3 refit, which differs from the final fit's 3rd node
    first3 = sl.fit_ladder(a, first_n=3, write=False, steps=S5)["ratings"][str(S5[2])]
    final = json.load(open(sl.ladder_json_path(a)))["ratings"][str(S5[2])]
    got = rep["sides"][0]["at_count"]
    assert got["step"] == S5[2] and got["elo"] == pytest.approx(first3)
    assert first3 != pytest.approx(final, abs=0.05), "fixture must separate the first-n refit from the final fit"
    d = rep["sides"][1]["delta_vs_first"]
    assert d["elo"] == pytest.approx(rep["sides"][1]["at_count"]["elo"] - got["elo"])
    assert d["se"] > 0 and d["ci95"] > 0


def test_a_groomed_pool_is_counted_over_games_jsonl_not_the_committed_survivors(tmp_path):
    groomed = _build(tmp_path, "g", S5, committed=S5[2:])          # ladder.json rates only 3M..5M
    other = _build(tmp_path, "o", S5)
    rep = ec.compare([groomed, other], at_snapshots=3)
    side = rep["sides"][0]
    assert side["n_own_rated_ever"] == 5 and side["n_groomed_out"] == 2
    assert [nd["step"] for nd in side["nodes"]] == S5[:3], "the run's FIRST three, not the survivors' 3M..5M"


def test_a_forks_inherited_parent_snapshots_are_not_counted(tmp_path):
    fork = _build(tmp_path, "f", S5)
    with open(os.path.join(fork, "metadata.json"), "w") as fh:
        json.dump({"lineage": {"fork_parent": {"path": "p"}, "fork_step": S5[1]}}, fh)
    other = _build(tmp_path, "o", S5[:3])
    rep = ec.compare([fork, other])
    side = rep["sides"][0]
    assert side["n_own_rated_ever"] == 3 and side["fork_step"] == S5[1]
    assert [nd["step"] for nd in side["nodes"]] == S5[2:]


def test_the_matched_count_cannot_exceed_what_every_side_has(tmp_path):
    a, b = _build(tmp_path, "a", S5), _build(tmp_path, "b", S5[:3])
    with pytest.raises(ec.CompareRefusal, match="at most 3"):
        ec.compare([a, b], at_snapshots=4)
    with pytest.raises(ec.CompareRefusal, match="at most 3"):
        ec.compare([a, b], at_snapshots=0)


def test_one_run_or_the_same_run_twice_is_not_a_comparison(tmp_path):
    a = _build(tmp_path, "a", S5)
    with pytest.raises(ec.CompareRefusal, match="at least two"):
        ec.compare([a])
    with pytest.raises(ec.CompareRefusal, match="twice"):
        ec.compare([a, a])


def test_a_side_with_no_games_log_and_a_groomed_pool_is_refused(tmp_path):
    a = _build(tmp_path, "a", S5)
    b = _build(tmp_path, "b", S5, committed=S5[2:], log=False)
    with open(os.path.join(b, "eval_results.jsonl"), "w") as fh:     # its eval cycles drew 1M, no longer rated
        fh.write(json.dumps({"sentinels": [{"step": S5[0]}, {"step": S5[3]}]}) + "\n")
    with pytest.raises(ec.CompareRefusal, match="GROOMED"):
        ec.compare([a, b])


def test_a_log_less_side_is_read_from_its_committed_file_only_when_it_is_exactly_n_and_current(tmp_path):
    a = _build(tmp_path, "a", S5[:3])
    b = _build(tmp_path, "b", S5[:3], log=False)
    rep = ec.compare([a, b])
    assert "committed ladder.json" in rep["sides"][1]["source"]
    assert rep["sides"][1]["at_count"]["elo"] == pytest.approx(json.load(open(sl.ladder_json_path(b)))["ratings"][str(S5[2])])
    # more own snapshots than n and no log: the first-n fit does not exist
    big = _build(tmp_path, "big", S5, log=False)
    with pytest.raises(ec.CompareRefusal, match="no games.jsonl to refit from"):
        ec.compare([a, big])
    # a stale recipe stamp on the log-less fallback is a refusal, not a label
    stale = _build(tmp_path, "stale", S5[:3], log=False, stamp=False)
    with pytest.raises(ec.CompareRefusal, match="STALE LADDER RECIPE"):
        ec.compare([a, stale])


def test_a_transport_mismatch_is_refused_unless_allowed_and_then_labelled(tmp_path):
    a = _build(tmp_path, "a", S5[:3])
    b = _build(tmp_path, "b", S5[:3])
    rows = [json.loads(line) for line in open(sl.games_log_path(b))]
    with open(sl.games_log_path(b), "w") as fh:        # b's edges all played on the Rust engine
        for r in rows:
            r["transport"] = sl.TRANSPORT_RUST
            fh.write(json.dumps(r) + "\n")
    with pytest.raises(ec.CompareRefusal, match="transport"):
        ec.compare([a, b])
    rep = ec.compare([a, b], allow_transport_mix=True)
    assert any("TRANSPORT MIX ACCEPTED" in w for w in rep["transport_warning"])
    assert "TRANSPORT MIX ACCEPTED" in ec.render(rep)


def test_a_prefix_with_no_edge_inside_itself_is_refused_not_quoted_unmatched(tmp_path):
    a = _build(tmp_path, "a", S5[:3])
    fork = str(tmp_path / "fk")
    os.makedirs(os.path.join(fork, "snapshot_ladder"))
    for lo in S5[:3]:                                   # every pair touches the late 5M only
        sl._append_game(fork, S5[4], lo, 70, 100)
    doc = sl.fit_ladder(fork, write=False, steps=S5[:3] + [S5[4]])
    json.dump(doc, open(sl.ladder_json_path(fork), "w"))
    with pytest.raises(ec.CompareRefusal, match="no measured edge inside itself"):
        ec.compare([a, fork], at_snapshots=3)


def test_unfinished_runs_and_cross_era_are_labelled_and_the_read_writes_nothing(tmp_path):
    a = _build(tmp_path, "a", S5[:3])
    b = _build(tmp_path, "b", S5[:3])
    _final(a)
    with open(os.path.join(b, "metadata.json"), "w") as fh:
        json.dump({"era": {"code": "rb", "name": "Rustboro", "order": 1}}, fh)
    before = sorted((root, f, os.path.getmtime(os.path.join(root, f))) for root, _, fs in os.walk(tmp_path) for f in fs)
    rep = ec.compare([a, b])
    text = ec.render(rep)
    assert rep["unfinished"] == ["b"] and "UNFINISHED: b" in text
    assert rep["era_warning"] and "CROSS-ERA" in text
    after = sorted((root, f, os.path.getmtime(os.path.join(root, f))) for root, _, fs in os.walk(tmp_path) for f in fs)
    assert before == after


def test_the_cli_dispatches_from_main_elo_and_exits_2_on_a_refusal(tmp_path, monkeypatch, capsys):
    a, b = _build(tmp_path, "a", S5), _build(tmp_path, "b", S5[:3])
    monkeypatch.setattr("sys.argv", ["main.elo", "compare", a, b, "--json"])
    assert elo_cli.main() == 0
    assert json.loads(capsys.readouterr().out)["at_snapshots"] == 3
    monkeypatch.setattr("sys.argv", ["main.elo", "compare", a, b, "--at-snapshots", "9"])
    assert elo_cli.main() == 2
    assert "REFUSED" in capsys.readouterr().err
    assert ec.compare_main([a, b]) == 0
    assert "DELTA vs a" in capsys.readouterr().out
