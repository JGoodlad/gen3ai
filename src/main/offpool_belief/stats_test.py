"""Team-clustered statistics: determinism, absent teams, pairing."""
from __future__ import annotations

import numpy as np
import pytest

from main.offpool_belief import stats as S


def test_team_means_leave_a_team_without_units_absent() -> None:
    got = S.team_means([3.0, 0.0, -4.0], [2, 0, 4])
    assert got[0] == 1.5 and np.isnan(got[1]) and got[2] == -1.0
    with pytest.raises(ValueError):
        S.team_means([1.0], [1, 2])


def test_the_mean_is_over_teams_not_rows() -> None:
    # team 0: 100 units of mean 0; team 1: 1 unit of value 1. The row mean is 1/101, the team mean is 0.5.
    tm = S.team_means([0.0, 1.0], [100, 1])
    assert S.mean_ci(tm)["mean"] == 0.5


def test_the_interval_is_a_pure_function_of_the_inputs() -> None:
    v = np.linspace(-1.0, 1.0, 12)
    a, b = S.mean_ci(v), S.mean_ci(v.copy())
    assert a == b and a["lo"] < a["mean"] < a["hi"] and a["n_teams"] == 12 and a["empty_draws"] == 0
    assert np.array_equal(S.boot_index(12), S.boot_index(12))
    assert not np.array_equal(S.boot_index(12, seed=1), S.boot_index(12, seed=2))


def test_absent_teams_and_tiny_columns() -> None:
    v = np.array([np.nan, 2.0, 4.0, np.nan])
    got = S.mean_ci(v)
    assert got["mean"] == 3.0 and got["n_teams"] == 2 and 2.0 <= got["lo"] <= got["hi"] <= 4.0
    assert got["empty_draws"] > 0                       # some resamples draw only the absent teams: dropped, counted
    one = S.mean_ci([np.nan, 5.0])
    assert one == {"mean": 5.0, "lo": None, "hi": None, "n_teams": 1, "empty_draws": 0}
    assert S.mean_ci([np.nan, np.nan])["mean"] is None
    with pytest.raises(ValueError):
        S.mean_ci(np.zeros(5), S.boot_index(4))


def test_a_paired_difference_keeps_the_pairing() -> None:
    # b = a - 0.1 on every team: the difference is exactly 0.1 with a zero-width interval, although each column's
    # own interval is wide. An unpaired comparison would not detect it.
    a = np.linspace(-5.0, 5.0, 20)
    d = S.paired_diff(a, a - 0.1)
    assert abs(d["mean"] - 0.1) < 1e-12 and abs(d["lo"] - 0.1) < 1e-9 and abs(d["hi"] - 0.1) < 1e-9
    col = S.mean_ci(a)
    assert col["hi"] - col["lo"] > 1.0
    # a team missing on either side is absent from the difference
    b = a - 0.1
    b[3] = np.nan
    assert S.paired_diff(a, b)["n_teams"] == 19
    with pytest.raises(ValueError):
        S.paired_diff(a, a[:5])
