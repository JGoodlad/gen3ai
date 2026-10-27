"""The per-game ``our_team`` of an anchor-vs-anchor cell is the team the peer PLAYED (H-CTX review F1, 2026-10-27).

Metamon's per-battle CSV names the NEXT draw in most games (``main.anchors.peer_teams``), so ``read_peer_battles`` no
longer stores that column as the team. The fixture is ONE REAL banked pair cell (the team-strength lane's
``kak_away_000``: Kakuna against Kakuna, 2 halves x 10 games): the peer's CSV and draw log verbatim, and the START
command of its own side in every capture. Each test fails on the revert it names.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from main.anchors import runner
from main.anchors.peer_teams import (
    CAUSE,
    FROM_DRAW_LOG,
    attribute_pair_half,
    attribute_played_teams,
    captured_sixes,
    packed_species,
)
from main.anchors.peers import PeerPlan
from main.anchors.session import SeriesFailure, read_peer_battles

FIXTURE = Path(__file__).parent / "testdata" / "peer_pair_cell_tsr_kak_away_000.json"
HALVES = ("ours_challenge", "peer_challenge")


@pytest.fixture(scope="module")
def cell() -> dict:
    return json.loads(FIXTURE.read_text())


def _materialise(cell: dict, half: str, root: Path) -> SimpleNamespace:
    """The half on disk as the harness leaves it: ``results/battle_log_*.csv``, ``capture/*.json``, the team files."""
    h = cell["halves"][half]
    results, capture, teams = root / "results", root / "capture", root / "teams"
    for d in (results, capture, teams):
        d.mkdir(parents=True, exist_ok=True)
    (results / f"battle_log_{h['our_name']}_gen3ou.csv").write_text(h["csv"])
    for c in h["captures"]:
        (capture / f"battle-gen3ou-{c['n']}.json").write_text(
            json.dumps({"tag": f"battle-gen3ou-{c['n']}", "seed": 0, "chunks": [],
                        "commands": ["START " + json.dumps(c["start"])]}))
    for label, text in cell["team_files"].items():
        (teams / label).write_text(text)
    return SimpleNamespace(results=results, capture=capture, teams=teams, draws=h["team_draws"],
                           our_name=h["our_name"], peer_name=h["peer_name"])


def _played(cell: dict, half: str) -> list:
    return [packed_species(c["start"][c["our_slot"]]["team"]) for c in cell["halves"][half]["captures"]]


def _species_of(cell: dict, label: str) -> frozenset:
    from main.anchors.team_overlap import parse_team

    return parse_team(cell["team_files"][label], label).species6


@pytest.mark.parametrize("half", HALVES)
def test_the_csv_column_is_a_race_between_the_played_draw_and_the_next_one(cell: dict, half: str, tmp_path: Path) -> None:
    """The measured fact: every CSV row is draw ``i`` or draw ``i + 1``, and most are the NEXT one."""
    disk = _materialise(cell, half, tmp_path)
    csv_teams = [r.csv_team for r in read_peer_battles(disk.results)]
    assert len(csv_teams) == len(disk.draws) == 10
    assert all(c in (disk.draws[i], *disk.draws[i + 1:i + 2]) for i, c in enumerate(csv_teams))
    next_only = [i for i, c in enumerate(csv_teams) if i + 1 < len(disk.draws) and c == disk.draws[i + 1]
                 and c != disk.draws[i]]
    played_only = [i for i, c in enumerate(csv_teams) if c == disk.draws[i] and (
        i + 1 == len(disk.draws) or c != disk.draws[i + 1])]
    assert len(next_only) >= 5, "the column is NOT the team played: most rows name the next draw"
    assert played_only, "...and the race is not even a fixed offset: some rows do name the played team"


@pytest.mark.parametrize("half", HALVES)
def test_read_peer_battles_no_longer_stores_the_csv_column_as_the_team_played(cell: dict, half: str,
                                                                              tmp_path: Path) -> None:
    """Fails on revert: ``our_team=<csv column>`` made a per-team read of a pair cell's rows off by one."""
    recs = read_peer_battles(_materialise(cell, half, tmp_path).results)
    assert all(r.our_team is None for r in recs)
    assert all(r.csv_team for r in recs)


@pytest.mark.parametrize("half", HALVES)
def test_the_played_team_is_the_draw_log_and_matches_the_captured_six_on_a_real_cell(cell: dict, half: str,
                                                                                    tmp_path: Path) -> None:
    disk = _materialise(cell, half, tmp_path)
    out, info = attribute_pair_half(read_peer_battles(disk.results), draws=disk.draws, our_name=disk.our_name,
                                    peer_name=disk.peer_name, capture_dir=disk.capture, team_dir=disk.teams)
    assert [r.our_team for r in out] == disk.draws
    assert {r.team_from for r in out} == {FROM_DRAW_LOG}
    assert info == {"our_team_attribution": FROM_DRAW_LOG, "our_team_capture_check": "checked 10 of 10 games"}
    # the label is the six the capture shows, game for game (the CSV's label was NOT, in more than half the games)
    played = _played(cell, half)
    assert [_species_of(cell, r.our_team) for r in out] == played
    assert sum(_species_of(cell, r.csv_team) != s for r, s in zip(out, played)) >= 3


@pytest.mark.parametrize("half", HALVES)
def test_the_played_mons_guard_throws_when_the_recorded_team_is_not_what_was_played(cell: dict, half: str,
                                                                                   tmp_path: Path) -> None:
    """THE GUARD: a draw log shifted by one (the CSV's very error) names six mons the capture did not show."""
    disk = _materialise(cell, half, tmp_path)
    recs = [replace(r, csv_team=None) for r in read_peer_battles(disk.results)]      # so only the capture guard speaks
    shifted = disk.draws[1:] + disk.draws[:1]
    assert any(_species_of(cell, a) != _species_of(cell, b) for a, b in zip(shifted, disk.draws))
    with pytest.raises(SeriesFailure) as err:
        attribute_pair_half(recs, draws=shifted, our_name=disk.our_name, peer_name=disk.peer_name,
                            capture_dir=disk.capture, team_dir=disk.teams)
    assert err.value.cause == CAUSE and "capture's START holds" in err.value.detail


@pytest.mark.parametrize("half", HALVES)
def test_the_csv_cross_check_throws_when_the_csv_and_the_draw_log_are_not_one_series(cell: dict, half: str,
                                                                                    tmp_path: Path) -> None:
    """A previous attempt's rows appended to the file, or a draw that never became a battle: a CSV row that is
    neither draw ``i`` nor draw ``i + 1``."""
    disk = _materialise(cell, half, tmp_path)
    recs = read_peer_battles(disk.results)
    other = next(label for label in disk.draws if label not in (disk.draws[0], disk.draws[1]))
    recs[0] = replace(recs[0], csv_team=other)
    with pytest.raises(SeriesFailure) as err:
        attribute_played_teams(recs, disk.draws)
    assert err.value.cause == CAUSE and "game 1" in err.value.detail


def test_fewer_draws_than_games_throws_and_no_draw_log_is_unattributed_not_guessed(cell: dict, tmp_path: Path) -> None:
    disk = _materialise(cell, "ours_challenge", tmp_path)
    recs = read_peer_battles(disk.results)
    with pytest.raises(SeriesFailure) as err:
        attribute_played_teams(recs, disk.draws[:9])
    assert err.value.cause == CAUSE and "10 games recorded but the peer logged 9" in err.value.detail
    out, info = attribute_pair_half(recs, draws=[], our_name=disk.our_name, peer_name=disk.peer_name,
                                    capture_dir=disk.capture, team_dir=disk.teams)
    assert all(r.our_team is None and r.team_from == "" for r in out)
    assert info["our_team_attribution"].startswith("unattributed")


def test_an_incomplete_capture_set_is_skipped_never_a_disagreement(cell: dict, tmp_path: Path) -> None:
    """A capture missing (the front end writes it after the client sees `|win|`) shifts the alignment, so the played-mons
    guard does not run; that is a skip the report names, never a false throw."""
    disk = _materialise(cell, "ours_challenge", tmp_path)
    next(disk.capture.glob("battle-gen3ou-3.json")).unlink()
    out, info = attribute_pair_half(read_peer_battles(disk.results), draws=disk.draws, our_name=disk.our_name,
                                    peer_name=disk.peer_name, capture_dir=disk.capture, team_dir=disk.teams)
    assert [r.our_team for r in out] == disk.draws
    assert info["our_team_capture_check"] == "skipped: 9 captures for 10 games"


def test_captures_of_the_other_half_are_not_this_halfs(cell: dict, tmp_path: Path) -> None:
    """Both halves' captures share one directory; the half's two usernames pick its own."""
    a = _materialise(cell, "ours_challenge", tmp_path)
    b = _materialise(cell, "peer_challenge", tmp_path)
    assert len(captured_sixes(a.capture, a.our_name, a.peer_name)) == 10
    assert captured_sixes(a.capture, a.our_name, a.peer_name) == _played(cell, "ours_challenge")
    assert captured_sixes(b.capture, b.our_name, b.peer_name) == _played(cell, "peer_challenge")


# ------------------------------------------------------------------------------------- the call site (the wiring)
def _half_runner(monkeypatch: pytest.MonkeyPatch, cell: dict, disk: SimpleNamespace, draws: list, tmp_path: Path):
    """``runner.run_peer_pair_half`` with the processes and sockets stubbed (as ``guard_wiring_test`` does); the peer
    report carries ``draws`` and the CSV is the real cell's."""
    logs = {}
    for side in ("theirs", "ours"):
        log = tmp_path / side / "peer.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("clean\n")
        logs[side] = PeerPlan(label=f"metamon:{side}", argv=["x", "--results-dir", str(disk.results)], cwd=tmp_path,
                              env={}, ready_pattern="", log_path=log, team_dir=disk.teams)

    async def noop(*_a, **_k):
        return None

    ok = {"regime_verified_decisions": True, "peer_error_free": True}
    monkeypatch.setattr(runner, "our_peer_plan", lambda *a: logs["ours"])
    monkeypatch.setattr(runner, "peer_plan", lambda *a: logs["theirs"])
    monkeypatch.setattr(runner, "start_peer", lambda pp, nice=0: SimpleNamespace(returncode=0))
    for name in ("await_peer_ready", "watch_peer_pair", "await_peer_exit"):
        monkeypatch.setattr(runner, name, noop)
    monkeypatch.setattr(runner, "stop_peer", lambda _p: None)
    monkeypatch.setattr(runner, "pair_adapters", lambda _plan: (
        SimpleNamespace(read_report=lambda _pp: dict(ok)),
        SimpleNamespace(read_report=lambda _pp: {**ok, "team_draws": list(draws)})))
    monkeypatch.setattr(runner.peers_mod, "team_source_count", lambda _pp: 72)
    monkeypatch.setattr(runner, "team_set_digest_or_none", lambda _d: None)
    monkeypatch.setattr(runner, "pair_exit_note", lambda *a: None)
    plan = SimpleNamespace(our_username=disk.our_name[:-1], peer_username=disk.peer_name[:-1], nice=0,
                           peer_ready_timeout_s=1, first_game_timeout_s=1, progress_timeout_s=1,
                           capture_dir=disk.capture)
    return plan


@pytest.mark.asyncio
async def test_the_pair_half_attributes_the_played_team_and_stamps_how(cell: dict, tmp_path: Path, monkeypatch) -> None:
    """Fails if ``run_peer_pair_half`` stops calling the attribution: its records would carry no team at all."""
    disk = _materialise(cell, "ours_challenge", tmp_path)
    plan = _half_runner(monkeypatch, cell, disk, disk.draws, tmp_path)
    records, report, failure = await runner.run_peer_pair_half(plan, None, "ours_challenge", 10)
    assert failure is None
    assert [r.our_team for r in records] == disk.draws
    assert report["our_team_attribution"] == FROM_DRAW_LOG
    assert report["our_team_capture_check"] == "checked 10 of 10 games"
    row = runner.rows_from(records, "ours_challenge", SimpleNamespace(stamp=dict), report, 1)[0].to_json()
    assert row["our_team"] == disk.draws[0] and row["our_team_from"] == FROM_DRAW_LOG


@pytest.mark.asyncio
async def test_a_draw_log_that_is_not_the_series_played_fails_the_half_by_name(cell: dict, tmp_path: Path,
                                                                                  monkeypatch) -> None:
    disk = _materialise(cell, "peer_challenge", tmp_path)
    plan = _half_runner(monkeypatch, cell, disk, disk.draws[1:] + disk.draws[:1], tmp_path)
    records, report, failure = await runner.run_peer_pair_half(plan, None, "peer_challenge", 10)
    assert failure is not None and failure.cause == CAUSE
    assert all(r.our_team is None for r in records), "a half that fails the guard attributes NO team"
    assert report["our_team_attribution"].startswith("FAILED")
