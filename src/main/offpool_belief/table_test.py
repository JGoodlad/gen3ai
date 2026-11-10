"""The table: arms of several checkpoints, pairing on teams, and what a Δ column's word means."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from main.offpool_belief import SCORE_SCHEMA, OffPoolBeliefError
from main.offpool_belief import metrics as M
from main.offpool_belief import score as SC
from main.offpool_belief import table as T
from main.offpool_belief.bank import Bank

N_TEAMS = 12


def _doc(label: str, shift: float, *, rows_sha: str = "a" * 64, recorded: int = 161) -> Dict[str, Any]:
    """A score document whose every team has 2 units a family; the posterior's team mean is ``prior + shift`` on
    every metric, the prior's is ``-1 - 0.1 * team``."""
    ts = SC.TeamSums(N_TEAMS, [M.POST, M.PRIOR, "mix0.25"])
    for f in M.FAMILIES:
        ts.units[f][:] = 2
    prior = -1.0 - 0.1 * np.arange(N_TEAMS)
    for metric in M.METRICS:
        ts.sums[metric][M.PRIOR][:] = 2 * prior
        ts.sums[metric][M.POST][:] = 2 * (prior + shift)
        ts.sums[metric]["mix0.25"][:] = 2 * (prior + 0.05)
    ts.excluded["moves_truth_illegal"] = np.ones(N_TEAMS, dtype=np.int64)
    man = {"panel": "p72", "panel_held_out": True, "panel_content_digest": "d", "rows_sha256": rows_sha,
           "n_rows": 24, "n_games": 12, "games_per_team": 1, "seed": 1, "pilot": {"sha256": "p"}, "code": {},
           "team_labels": [f"t{i}" for i in range(N_TEAMS)]}
    block = SC.panel_block(Bank(manifest=man, rows={}, path=Path("bank")), ts)
    block["obs_gap"] = SC.obs_gap(recorded, 168)
    return {"schema": SCORE_SCHEMA, "label": label, "checkpoint": {"path": label, "sha256": label},
            "panels": {"p72": block}}


def test_an_arm_is_the_team_mean_of_its_checkpoints_and_pairs_with_the_reference() -> None:
    arms = {"C0": [_doc("c1", -0.5), _doc("c2", -0.3)], "N1": [_doc("n1", +0.2)]}
    doc = T.build(arms, versus="C0")
    cell = doc["panels"]["p72"]
    assert cell["kind"] == "OFF-POOL" and cell["n_teams"] == N_TEAMS
    row = cell["metrics"]["species_logp"]
    assert row["units"] == 2 * N_TEAMS
    c0, n1 = row["arms"]["C0"], row["arms"]["N1"]
    # C0's two checkpoints sit 0.5 and 0.3 below the prior on every team: the arm sits 0.4 below, exactly, with a
    # zero-width paired interval (the same shift on every team)
    assert c0["n_checkpoints"] == 2
    for k in ("mean", "lo", "hi"):
        assert c0["post_minus_prior"][k] == pytest.approx(-0.4, abs=1e-9)
        assert n1["minus_C0"][k] == pytest.approx(0.6, abs=1e-9)
        assert c0["mix0.25_minus_prior"][k] == pytest.approx(0.05, abs=1e-9)
    assert "minus_C0" not in c0
    assert c0["checkpoint_range"] == pytest.approx([-1.55 - 0.5, -1.55 - 0.3])
    assert T.verdict(c0["post_minus_prior"]) == "WORSE" and T.verdict(n1["post_minus_prior"]) == "BETTER"
    assert cell["excluded"]["C0"] == {"moves_truth_illegal": N_TEAMS}
    md = T.markdown(doc)
    assert "### p72 — OFF-POOL (12 teams" in md and "| C0 | 2 |" in md and "belief − C0" in md
    assert "[[162, 163, 165, 166, 167, 168]]" in md                  # the semantics gap is on the page


def test_a_column_an_older_score_file_lacks_is_named_absent_never_a_partial_row() -> None:
    """A score file written before ``moves_ll`` existed has no sums for it. The table must still build (the banked
    baseline scores stay readable), with that column ABSENT for every arm on the panel and said so on the page."""
    old, new = _doc("old", -0.5), _doc("new", +0.2)
    del old["panels"]["p72"]["sums"]["moves_ll"]
    doc = T.build({"A": [old], "B": [new]}, versus="A")
    cell = doc["panels"]["p72"]
    assert cell["absent"] == {"moves_ll": ["old"]}
    assert "moves_ll" not in cell["metrics"] and "moves_logp" in cell["metrics"]
    md = T.markdown(doc)
    assert "**moves_ll** is ABSENT: old were scored before the column existed." in md
    full = T.build({"A": [_doc("a", -0.5)], "B": [new]}, versus="A")["panels"]["p72"]
    assert full["absent"] == {} and full["metrics"]["moves_ll"]["arms"]["B"]["minus_A"]["mean"] == pytest.approx(0.7)


def test_the_verdict_word_needs_the_whole_interval_on_one_side() -> None:
    assert T.verdict({"mean": 0.1, "lo": -0.01, "hi": 0.2}) == "NOT DETECTED"
    assert T.verdict({"mean": 0.1, "lo": 0.0, "hi": 0.2}) == "NOT DETECTED"      # touching zero is not a detection
    assert T.verdict({"mean": 0.1, "lo": 0.01, "hi": 0.2}) == "BETTER"
    assert T.verdict({"mean": -0.1, "lo": -0.2, "hi": -0.01}) == "WORSE"
    assert T.verdict({"mean": 0.1, "lo": None, "hi": None}) == "n/a"


def test_arms_scored_on_different_banks_are_refused() -> None:
    with pytest.raises(OffPoolBeliefError, match="different banks"):
        T.build({"A": [_doc("a", 0.0)], "B": [_doc("b", 0.0, rows_sha="b" * 64)]})
    with pytest.raises(OffPoolBeliefError, match="different banks"):
        T.build({"A": [_doc("a", 0.0), _doc("a2", 0.0, rows_sha="b" * 64)]})
    with pytest.raises(OffPoolBeliefError, match="not one of the arms"):
        T.build({"A": [_doc("a", 0.0)]}, versus="Z")
    with pytest.raises(OffPoolBeliefError, match="no score on panel"):
        T.build({"A": [_doc("a", 0.0)]}, panels=["forum72"])
