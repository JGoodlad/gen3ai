"""The retention hold of REGISTERED FRESH CELLS whose read is not yet recorded (``tiers.OPEN_FRESH_CELLS``), on a
SYNTHETIC tree: no planner plans a deletion in one of their runs, no ``--apply`` path executes one, and a registered
card cannot be left out of the hold. Each test fails when its part of the hold is removed.

    python -m pytest src/main/ops/archive_grooming/fresh_cell_hold_test.py -q

(in a linked worktree, first: export PYTHONPATH=$PWD/src)
"""

from __future__ import annotations

import json
import os

import pytest

from main.ops.archive_grooming import dryrun as ag
from main.ops.archive_grooming import tiers
from main.ops.archive_grooming.dryrun_test import make_run
from main.ops.archive_grooming.thin_run_test import check, make_fork, plan, zips
from main.ops.archive_grooming.tiered_test import _stamp_old, census
from utils.paths import repo_path

#: The registered runs of the OPEN card (Q5-N) and its spare-seed replacement (its PREREG's §2 and §4).
HELD = ["rb_q5n_s5201", "rb_q5n_s5202", "rb_q5n_s5203", "rb_q5n_s5209"]
#: Q6-N's runs and its spare: RELEASED by its STOPPED-AT-50M read (ledger 2026-11-24), back under the ordinary policies.
RELEASED = ["rb_q6n_s5211", "rb_q6n_s5212", "rb_q6n_s5213", "rb_q6n_s5219"]
#: Same-shaped runs outside the hold: the DROPPED wide arms and a pilot. Each test's control (a non-vacuous plan).
NEIGHBOURS = ["rb_q6w_s5214", "rb_q5w_s5204", "q6n_pilot_s5299"]
MEASUREMENTS = os.path.join("designs", "research_state", "measurements")


def registered_arm_files() -> "list[str]":
    """Every committed ``fresh_arm_*.json`` a PREREG registered (it carries ``prereg``; a frame draft does not)."""
    root = str(repo_path(MEASUREMENTS))
    found = []
    for d in sorted(os.listdir(root)):
        folder = os.path.join(root, d)
        if os.path.isdir(folder):
            found += [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                      if f.startswith("fresh_arm_") and f.endswith(".json")]
    docs = {p: json.load(open(p)) for p in found}
    return [p for p, doc in docs.items() if doc.get("prereg")]


# ------------------------------------------------------------------------------------- the registry gate
def test_every_run_a_registered_fresh_arm_file_names_is_held_or_released():
    """The gate: dropping a card from ``OPEN_FRESH_CELLS`` without recording its release FAILS here, and so does
    registering a new card without holding its runs."""
    files = registered_arm_files()
    assert len(files) >= 2, files                     # Q6-N and Q5-N: the scan is not vacuous
    unheld = []
    for path in files:
        doc = json.load(open(path))
        for row in [*doc["fresh"], *(doc.get("own_control") or [])]:
            run = row["run"]
            if not tiers.fresh_cell_hold(run) and not run.startswith(tuple(tiers.RELEASED_FRESH_CELLS)):
                unheld.append(f"{run} ({os.path.relpath(path, str(repo_path('.')))})")
    assert not unheld, f"registered fresh-cell runs no retention hold covers: {unheld}"


def test_each_open_cell_names_its_prereg_and_each_release_its_record():
    assert set(tiers.OPEN_FRESH_CELLS) == {"rb_q5n_"}                  # edited only with a recorded release
    assert set(tiers.RELEASED_FRESH_CELLS) == {"rb_q6n_"}
    for prefix, prereg in tiers.OPEN_FRESH_CELLS.items():
        assert prefix.endswith("_") and os.path.isfile(str(repo_path(prereg))), (prefix, prereg)
    assert not set(tiers.OPEN_FRESH_CELLS) & set(tiers.RELEASED_FRESH_CELLS)
    assert all(where.strip() for where in tiers.RELEASED_FRESH_CELLS.values())


@pytest.mark.parametrize("name", HELD)
def test_the_hold_covers_the_registered_runs_and_the_spares(name):
    assert "registered fresh cell" in (tiers.fresh_cell_hold(name) or "")
    assert tiers.retention_hold(name) == tiers.fresh_cell_hold(name)


@pytest.mark.parametrize("name", RELEASED)
def test_a_released_cell_is_no_longer_held(name):
    """Q6-N's STOPPED-AT-50M read released its prefix: fails if ``rb_q6n_`` returns to ``OPEN_FRESH_CELLS``."""
    assert tiers.fresh_cell_hold(name) is None and tiers.retention_hold(name) is None
    where = tiers.RELEASED_FRESH_CELLS["rb_q6n_"]
    assert "ledger.md" in where and "STOPPED-AT-50M" in where


@pytest.mark.parametrize("name", NEIGHBOURS + ["rb_q6n", "rb_q6nx_s1", "xrb_q6n_s5211"])
def test_the_hold_is_the_prefix_and_nothing_wider(name):
    assert tiers.fresh_cell_hold(name) is None and tiers.retention_hold(name) is None


def test_retention_hold_is_also_the_banked_b1_rule():
    assert tiers.retention_hold("rb_b1c_s5101") == tiers.BANKED_B1_REASON
    assert tiers.fresh_cell_hold("rb_b1c_s5101") is None


# ------------------------------------------------------------------------------------- the three planners
@pytest.mark.parametrize("name", HELD)
def test_thin_run_refuses_a_fresh_cell_run(tmp_path, name):
    with pytest.raises(ag.ThinRefused, match="registered fresh cell"):
        check(make_fork(str(tmp_path), name), tmp_path)


def test_thin_run_refuses_a_fresh_cell_run_reached_through_a_symlink(tmp_path):
    real = make_fork(str(tmp_path / "elsewhere"), "rb_q5n_s5202")
    link = tmp_path / "rb_x35_alias_s1"
    os.symlink(real, link)
    with pytest.raises(ag.ThinRefused, match="registered fresh cell"):
        check(str(link), tmp_path)


def test_thin_run_still_plans_a_neighbour(tmp_path):
    run = make_fork(str(tmp_path), "rb_q6w_s5214")
    check(run, tmp_path)
    assert plan(run)["delete"]


def test_the_thin_run_cli_exits_3_and_deletes_nothing(tmp_path, capsys):
    models = tmp_path / "models"
    run = make_fork(str(models), "rb_q5n_s5202")
    before = zips(run)
    x35 = tmp_path / "x35"
    x35.mkdir()
    (x35 / "reference_population_none.json").write_text(json.dumps({"a_block": []}))
    rc = ag.main(["--thin-run", "rb_q5n_s5202", "--models-dir", str(models), "--repo-root", str(tmp_path),
                  "--x35-dir", str(x35), "--log-dir", str(tmp_path / "log"), "--apply"])
    assert rc == 3 and "registered fresh cell" in capsys.readouterr().err
    assert zips(run) == before and not (tmp_path / "log").exists()


def _old_tree(tmp_path, runs):
    models = tmp_path / "models"
    models.mkdir()
    for run in runs:
        make_run(str(models), run, latest_step=25000)
    repo = tmp_path / "repo"
    repo.mkdir()
    _stamp_old(str(models))
    return str(models), str(repo)


def test_the_standing_policy_never_closes_a_fresh_cell_run(tmp_path):
    models, repo = _old_tree(tmp_path, HELD + NEIGHBOURS)
    c = ag.build_census(models, repo, recent_days=0)
    for run in HELD:
        r = c["runs"][run]
        assert r["status"] == "REFERENCED" and not r["plan"]["delete"], run
        assert any("registered fresh cell" in why for why in r["status_reasons"])
    for run in NEIGHBOURS:
        assert c["runs"][run]["status"] == "CLOSED" and c["runs"][run]["plan"]["delete"], run


def test_the_tiered_policy_gives_a_fresh_cell_run_no_plan(tmp_path):
    models, repo = _old_tree(tmp_path, HELD + NEIGHBOURS)
    c = census(models, repo)
    for run in HELD:
        r = c["runs"][run]
        assert r["plan"]["delete"] == [] and "REVIEW HOLD" in r["plan"]["skipped"] and r["tier"] == 1, run
        assert "registered fresh cell" in r["review_hold"]
    for run in NEIGHBOURS:
        assert c["runs"][run]["plan"]["delete"], run


# ------------------------------------------------------------------------------------- the two apply paths
def _files(root: str) -> "list[str]":
    return sorted(os.path.join(dp, f) for dp, _d, fs in os.walk(root) for f in fs)


@pytest.mark.parametrize("policy", ["standing", "tiered"])
def test_apply_plan_refuses_a_plan_that_names_a_held_run(tmp_path, policy):
    """The execution-time re-check: a plan for a held run (forged here from a neighbour's) deletes NOTHING."""
    models, repo = _old_tree(tmp_path, ["rb_q5n_s5203", "rb_q6w_s5214"])
    c = ag.build_census(models, repo, recent_days=0) if policy == "standing" else census(models, repo)
    forged = c["runs"]["rb_q6w_s5214"]["plan"]["delete"]
    assert forged
    c["runs"]["rb_q5n_s5203"]["plan"]["delete"] = list(forged)
    c["runs"] = {"rb_q5n_s5203": c["runs"]["rb_q5n_s5203"], "rb_q6w_s5214": c["runs"]["rb_q6w_s5214"]}
    before = _files(models)
    with pytest.raises(ag.UnsafeDeletion, match="HELD"):
        ag.apply_plan(c)
    assert _files(models) == before


@pytest.mark.parametrize("name", ["rb_q5n_s5201", "rb_b1w_s5103"])
def test_apply_thin_run_refuses_a_plan_for_a_held_run(tmp_path, name):
    run = make_fork(str(tmp_path), name)
    p = plan(run)                                        # the planner itself has no hold: the refusal is the check's
    assert p["delete"]
    before = zips(run)
    with pytest.raises(ag.UnsafeDeletion, match="HELD"):
        ag.apply_thin_run(p, str(tmp_path / "log"))
    assert zips(run) == before and not (tmp_path / "log").exists()
