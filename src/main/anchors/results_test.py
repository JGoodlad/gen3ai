"""The results schema, and the ONE rule it exists to make unrepresentable.

🚨 **A number never leaves this tool without its regime.** The 2026-09-14 matched-regime battery
showed why: "temperature 1.0" is a 35% perturbation of `SmallRL` and a 12% perturbation of
`SyntheticRLV2`, the two models' temperature effects come out with opposite signs because of it,
and the mixed-regime de-risk headline of 0.742 fell to 0.520 on the like-for-like matched cell
(−22.2 pp [−34.1, −9.4]). A win rate without its regime is not a weaker number; it is one nobody
can place. `test_every_written_row_carries_its_regime_and_provenance` is the gate on that, and it
reads the WRITTEN FILE rather than the dataclass — "it is a field" and "it is in the file" stop
being the same statement the moment anyone writes a second producer.

The interval arithmetic is pinned against published worked values, not against itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.anchors.results import (
    REQUIRED_ROW_FIELDS,
    CellSpec,
    GameRow,
    newcombe,
    render,
    summarize,
    wilson,
    write_games,
    write_summary,
)


def _cell(**over) -> CellSpec:
    base = dict(
        opponent="metamon:SmallRL", opponent_kind="metamon", opponent_agent="SmallRL",
        opponent_version="SmallRL@ckpt40", opponent_commit="0a00a759",
        our_regime="greedy", their_regime="greedy", regime_matched=True,
        teamset="away", our_team_source="/tmp/competitive", our_team_count=20,
        their_team_source="competitive", their_team_count=20,
        battle_format="gen3ou", model_spec="models/run", model_zip="models/run/final_model.zip",
        model_step=75_005_952, model_rung="explicit_zip",
        search_time_ms=None, search_parallelism=None,
        forfeit_turn_limit=250, server_uri="ws://localhost:9500/showdown/websocket",
        showdown_pin="e0551883f",
    )
    base.update(over)
    return CellSpec(**base)


def _row(cell: CellSpec, result: str, *, half: str = "ours_challenge", index: int = 1,
         turns: int = 40, **over) -> GameRow:
    kwargs = dict(
        cell=cell, half=half, index=index, battle_tag=f"battle-gen3ou-{index}",
        turns=turns, result=result,
        won={"win": True, "loss": False, "tie": None}[result],
        hit_forfeit_limit=False, our_team=f"team_{index}.gen3ou_team",
        n_decisions=38, n_defaults=0, n_redecides=0, t_finished=1.0 * index,
        our_stochastic_kwargs=[False], their_sample_kwargs=[False],
        their_argmax_match_rate=1.0,
    )
    kwargs.update(over)
    return GameRow(**kwargs)


# ----------------------------------------------------------------------------- interval arithmetic
@pytest.mark.parametrize(
    "k, n, lo, hi",
    [
        # Worked Wilson 95% values for the two headline rows of the 2026-09-14 battery.
        (65, 100, 0.553, 0.736),    # SmallRL, greedy away — the bar IS cleared
        (42, 100, 0.328, 0.518),    # SyntheticRLV2, greedy away — NOT DETECTED
        (31, 80, 0.288, 0.497),     # Foul Play @ 1000 ms
        (89, 120, 0.657, 0.812),    # the de-risk's mixed-regime SmallRL read
    ],
)
def test_wilson_matches_the_published_cells(k: int, n: int, lo: float, hi: float) -> None:
    p, got_lo, got_hi = wilson(k, n)
    assert p == pytest.approx(k / n)
    assert got_lo == pytest.approx(lo, abs=5e-4)
    assert got_hi == pytest.approx(hi, abs=5e-4)


def test_wilson_is_defined_at_the_boundaries_and_at_n_zero() -> None:
    """Wald is not, which is the whole reason this is Wilson: cells here sit near 0 and 1."""
    _, lo, hi = wilson(0, 50)
    assert lo == 0.0 and 0.0 < hi < 0.15
    _, lo, hi = wilson(50, 50)
    assert hi == 1.0 and 0.85 < lo < 1.0
    p, lo, hi = wilson(0, 0)
    assert p != p and lo != lo and hi != hi        # NaN, not a substituted zero


def test_newcombe_reproduces_the_published_differences() -> None:
    """The matched-vs-de-risk drop, with the CI CLEAR of zero — so it is DETECTED, not 'equal'."""
    d, lo, hi = newcombe(52, 100, 89, 120)         # 0.520 vs 0.742
    assert d == pytest.approx(-0.222, abs=1e-3)
    assert lo == pytest.approx(-0.341, abs=2e-3)
    assert hi == pytest.approx(-0.094, abs=2e-3)
    assert hi < 0, "this effect must read DETECTED"


def test_newcombe_covering_zero_is_what_not_detected_looks_like() -> None:
    d, lo, hi = newcombe(50, 100, 52, 100)
    assert lo < 0 < hi and abs(d) < 0.05


# -------------------------------------------------------------------------------- the row schema
def test_every_written_row_carries_its_regime_and_provenance(tmp_path: Path) -> None:
    """THE GATE. Read back from the FILE, because that is what a reader ever sees."""
    cell = _cell()
    rows = [_row(cell, "win", index=1), _row(cell, "loss", index=2)]
    path = tmp_path / "games.jsonl"
    assert write_games(path, rows) == 2

    written = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(written) == 2
    for row in written:
        missing = [f for f in REQUIRED_ROW_FIELDS if f not in row]
        assert not missing, (
            f"a written row is missing {missing} — a win rate without its regime, its team set, "
            "its opponent version or its search budget cannot be placed beside any other number "
            "(metamon_matched_regime_2026-09-14 §4.1)"
        )
        assert row["our_regime"] == "greedy"
        assert row["their_argmax_match_rate"] == 1.0
        assert row["model_step"] == 75_005_952


def test_a_foulplay_row_carries_its_realized_width_not_just_its_budget() -> None:
    """A wall-clock budget is a WIDTH METER, so the nominal number alone is not reproducible."""
    cell = _cell(opponent="foulplay", opponent_kind="foulplay", opponent_agent="",
                 their_regime="search:1000ms", search_time_ms=1000, search_parallelism=1)
    row = _row(cell, "loss", their_argmax_match_rate=None,
               their_visits_mean=1_401_233.0, their_visits_n=38).to_json()
    assert row["search_time_ms"] == 1000
    assert row["their_visits_mean"] == pytest.approx(1_401_233.0)
    assert row["their_regime"] == "search:1000ms"


# ----------------------------------------------------------------------------------- the summary
def test_a_tie_is_in_the_denominator_and_not_the_numerator() -> None:
    cell = _cell()
    rows = [_row(cell, "win", index=1), _row(cell, "loss", index=2), _row(cell, "tie", index=3)]
    s = summarize(cell, rows, status="OK")
    assert (s["n"], s["wins"], s["losses"], s["ties"]) == (3, 1, 1, 1)
    assert s["win_rate"] == pytest.approx(1 / 3)


def test_the_summary_splits_by_role_because_a_single_role_is_not_a_cell() -> None:
    cell = _cell()
    rows = ([_row(cell, "win", half="ours_challenge", index=i) for i in range(1, 4)]
            + [_row(cell, "loss", half="peer_challenge", index=i) for i in range(4, 7)])
    s = summarize(cell, rows, status="OK")
    assert s["by_half"]["ours_challenge"]["win_rate"] == 1.0
    assert s["by_half"]["peer_challenge"]["win_rate"] == 0.0
    assert s["win_rate"] == 0.5


def test_a_team_source_asymmetry_is_flagged_rather_than_averaged_away() -> None:
    """Both sides must draw from the same distribution, or the number mixes skill with matchup."""
    even = summarize(_cell(our_team_count=20, their_team_count=20), [], status="OK")
    assert even["team_source_asymmetry"] is False
    odd = summarize(_cell(our_team_count=719, their_team_count=72), [], status="OK")
    assert odd["team_source_asymmetry"] is True


def test_a_failed_summary_keeps_the_games_that_did_finish_and_says_so(tmp_path: Path) -> None:
    """A partial n is honest only when it is LABELLED partial — never a quiet short series."""
    cell = _cell()
    rows = [_row(cell, "win", index=1)]
    s = summarize(cell, rows, status="FAILED",
                  failure={"cause": "peer_exited", "detail": "rc=1 RecursionError"})
    assert s["status"] == "FAILED" and s["n"] == 1
    assert s["failure"]["cause"] == "peer_exited"
    path = tmp_path / "summary.json"
    write_summary(path, s)
    assert json.loads(path.read_text())["failure"]["cause"] == "peer_exited"


def test_render_states_the_regime_before_the_number() -> None:
    """The human-readable block obeys the same rule as the file. Order is deliberate."""
    cell = _cell()
    text = render(summarize(cell, [_row(cell, "win", index=1)], status="OK"))
    assert text.index("regime") < text.index("WIN RATE")
    assert "matched=True" in text
    assert "argmax_match_rate = 1.0000" in text
    assert "Wilson 95%" in text


def test_render_names_a_failure_rather_than_printing_a_bare_rate() -> None:
    cell = _cell()
    text = render(summarize(cell, [_row(cell, "win", index=1)], status="FAILED",
                            failure={"cause": "no_progress", "detail": "0 games in 900s"}))
    assert "FAILURE" in text and "no_progress" in text
