"""THE ARCHIVE VALIDATION — `main.best_response_gap` against the six real exploiter runs.

`integration`, not `slow`: it reads JSON off the run archive and plays NOTHING, so it costs
milliseconds — but it needs `models/`, which is not committed and exists only in the MAIN
checkout, so every test here SKIPS when `utils.paths.main_models_dir()` is None.

What it pins is the thing a refactor would quietly break: that the meter still reproduces the
BANKED vs-target endpoints (ledger 2026-09-21 — era-1's 0.740 x3, era-2's 0.700 / 0.530 / 0.450
and the 0.657 / 0.525 / 0.455 pooled rates), still assigns the right archetype and round from
what each run wrote down, and still REFUSES the cross-era comparison on the 4.5x dose gap that
confounded that week's read.
"""
from __future__ import annotations

import os

import pytest

from agents.training import best_response_gap as brg
from agents.training.best_response_gap import UnmatchedDoseError
from utils.paths import main_models_dir

pytestmark = pytest.mark.integration

#: (run name, archetype, banked endpoint, banked pooled wins) — from the runs' own
#: `eval_results.jsonl`, and quoted in the ledger entries of 2026-09-20/21.
ERA1 = [("ai_v13_05_exploit_big5starmie", "balance", 0.74, 279),
        ("ai_v13_06_exploit_ddtar_spikes", "offense", 0.74, 295),
        ("ai_v13_10_exploit_stall", "stall", 0.74, 289)]
ERA2 = [("ai_v13_13_exploit5_offense", "offense", 0.70, 263),
        ("ai_v13_14_exploit5_balance", "balance", 0.53, 210),
        ("ai_v13_15_exploit5_stall", "stall", 0.45, 182)]

#: `main.dose`'s reading of each era, and the gap that refuses the comparison.
ERA1_DOSE, ERA2_DOSE = 3.815e-08, 8.392e-09
#: Both eras trained the same number of post-fork steps — the budget is MATCHED and must stay so,
#: which is what makes DOSE the cause the refusal names.
BUDGET = 8_060_928


def _archive():
    root = main_models_dir()
    if root is None:
        pytest.skip("no models/ archive (not the MAIN checkout)")
    return root


def _read(name):
    root = _archive()
    run_dir = root / name
    if not run_dir.is_dir():
        pytest.skip(f"{name} is not in this archive")
    return brg.read_exploiter(str(run_dir), brg.load_teamsets())


@pytest.fixture(scope="module")
def repo_cwd():
    """The recorded `--trainee-team(s)` paths are repo-relative, so the provenance reader needs
    the repo root as cwd — the same requirement `main.best_response_gap` states in its refusal."""
    from utils.paths import repo_root
    old = os.getcwd()
    os.chdir(str(repo_root()))
    yield
    os.chdir(old)


@pytest.mark.parametrize("name,archetype,endpoint,pooled_wins", ERA1 + ERA2)
def test_banked_endpoint_and_pooled_rate_reproduce(repo_cwd, name, archetype, endpoint,
                                                   pooled_wins):
    run = _read(name)
    assert run.endpoint().rate == pytest.approx(endpoint)
    assert run.pooled() == (pooled_wins, 400)
    assert run.archetype == archetype
    assert len(run.post_fork) == 4
    assert all(p.post_fork for p in run.series), "no run in this archive has a pre-fork row"


@pytest.mark.parametrize("name,expected", [(n, "1/5") for n, *_ in ERA1]
                         + [(n, "5/5") for n, *_ in ERA2])
def test_teamset_membership_separates_the_two_eras(repo_cwd, name, expected):
    """Era-1 pins ONE team — the anchor of a registered 5-team set; era-2 pins all five. Same
    archetype, different subgame restriction, and the report must keep that visible."""
    assert _read(name).membership == expected


@pytest.mark.parametrize("name,target,step", [(n, "ai_v13_02_flywheel_winprob", 75_005_952)
                                              for n, *_ in ERA1]
                         + [(n, "ai_v13_12_plateau", 95_158_272) for n, *_ in ERA2])
def test_target_budget_and_dose_are_read_from_the_run(repo_cwd, name, target, step):
    run = _read(name)
    assert run.target_run == target
    assert run.target_step == step
    assert run.target_file.endswith(f"{target}/final_model.zip")
    assert run.budget == BUDGET
    assert run.target_pins_own_teams is False, "both targets are unpinned generalists"
    assert not run.lineage_derived, "all six recorded their lineage at fork time"


def test_the_cross_era_comparison_refuses_on_dose(repo_cwd):
    runs = [_read(n) for n, *_ in ERA1 + ERA2]
    for r in runs[:3]:
        assert r.dose_rate == pytest.approx(ERA1_DOSE, rel=1e-3), r.run
    for r in runs[3:]:
        assert r.dose_rate == pytest.approx(ERA2_DOSE, rel=1e-3), r.run
    with pytest.raises(UnmatchedDoseError) as exc:
        brg.check_matched(runs)
    assert "4.55x apart" in str(exc.value)
    assert "budget" not in str(exc.value).lower().split("dose")[0]


def test_within_an_era_the_exploiters_are_matched(repo_cwd):
    assert brg.check_matched([_read(n) for n, *_ in ERA1]) == []
    assert brg.check_matched([_read(n) for n, *_ in ERA2]) == []


def test_allow_unmatched_prints_the_table_and_carries_the_confound(repo_cwd):
    runs = [_read(n) for n, *_ in ERA1 + ERA2]
    found = brg.check_matched(runs, allow_unmatched=True)
    assert found and all(m.kind == "dose" for m in found)
    doc = brg.build_report(runs, stat="pooled", mismatches=found, draws=4000)
    assert doc["unmatched"] is True
    assert [b["round"] for b in doc["rounds"]] == [1, 2]
    assert doc["rounds"][0]["target_run"] == "ai_v13_02_flywheel_winprob"
    assert doc["rounds"][1]["target_run"] == "ai_v13_12_plateau"
    assert len(doc["caveats"]) == 3, "one TEAMSET SIZE caveat per archetype (1/5 -> 5/5)"

    per = {p["archetype"]: p for p in doc["deltas"][0]["per_archetype"]}
    assert set(per) == {"offense", "balance", "stall"}
    assert per["offense"]["delta"] == pytest.approx(0.6575 - 0.7375)
    assert per["balance"]["delta"] == pytest.approx(0.5250 - 0.6975)
    assert per["stall"]["delta"] == pytest.approx(0.4550 - 0.7225)
    d = doc["deltas"][0]
    assert d["mean_delta"] == pytest.approx(-0.17333, abs=1e-4)
    assert d["hi"] < 0 and "GAP FELL" in d["verdict"]


def test_endpoint_stat_reproduces_the_quoted_gaps(repo_cwd):
    """The ledger quotes ENDPOINTS. On that statistic era-1 is flat at +24.00 pp on all three."""
    runs = [_read(n) for n, *_ in ERA1 + ERA2]
    doc = brg.build_report(runs, stat="endpoint", draws=2000)
    r1 = {row["archetype"]: row["gap"] for row in doc["rounds"][0]["rows"]}
    r2 = {row["archetype"]: row["gap"] for row in doc["rounds"][1]["rows"]}
    assert sorted(r1) == ["balance", "offense", "stall"]
    assert list(r1.values()) == pytest.approx([0.24, 0.24, 0.24])
    assert r2["offense"] == pytest.approx(0.20)
    assert r2["balance"] == pytest.approx(0.03)
    assert r2["stall"] == pytest.approx(-0.05)


def test_cli_refuses_then_prints(repo_cwd, tmp_path, capsys):
    from main import best_response_gap as cli
    names = [n for n, *_ in ERA1 + ERA2]
    assert cli.main([*names, "--no-json"]) == 2
    assert "unmatched_dose" in capsys.readouterr().err

    out = str(tmp_path / "brgap.json")
    assert cli.main([*names, "--allow-unmatched", "--json", out, "--draws", "4000"]) == 0
    printed = capsys.readouterr().out
    assert "THE GAP FELL" in printed and "UNMATCHED" in printed
    assert os.path.isfile(out)
