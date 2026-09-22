"""`python -m main.elo refit [--apply]` — putting a committed ladder back on the CURRENT recipe.

Landing the recipe stamp (`0f230405`) made every `snapshot_ladder/ladder.json` already on disk
read `absent`, so every cross-run reader refuses it. That is the intended loud outcome, but it
leaves 93 runs' headline ratings unquotable until each is refit. This is the tool that closes it,
and these are the properties that make it safe to point at an archive of banked results:

* it REFITS, it never plays — `games.jsonl` is append-only and never stale;
* it fits the COMMITTED file's node set, so a delta means "the recipe moved this node" and not
  "the pool grew" (BT re-solves every node on every add; the newest node is inflated);
* `--apply` KEEPS the file it replaces as `snapshot_ladder/ladder.pre_recipe.json` — the only
  surviving evidence of what a banked number was quoted from — and refuses to overwrite one.

The fixture is a three-snapshot run whose committed ratings are deliberately 50 Elo above what
the raw pairs support, standing in for the pre-recipe inflation.
"""
from __future__ import annotations

import json
import os

import pytest

from agents.training import snapshot_ladder as sl
from main import elo as elo_cli


STEPS = [2_000_000, 4_000_000, 6_000_000]


def _build_run(tmp_path, *, stamped: bool = False, games: bool = True) -> str:
    run = str(tmp_path / "run_fixture")
    os.makedirs(os.path.join(run, "snapshot_ladder"), exist_ok=True)
    if games:
        with open(sl.games_log_path(run), "w") as fh:
            for a, b, wins_a in ((STEPS[0], STEPS[1], 30), (STEPS[0], STEPS[2], 20),
                                 (STEPS[1], STEPS[2], 40)):
                fh.write(json.dumps({"a": a, "b": b, "wins_a": wins_a, "games": 100}) + "\n")
    # What the raw pairs actually support, then shifted +50 to stand in for the stale recipe.
    honest = sl.fit_ladder(run, write=False, steps=STEPS) if games else None
    ratings = ({k: round(v + 50.0, 1) for k, v in honest["ratings"].items()} if honest
               else {str(s): 1900.0 + 50 * i for i, s in enumerate(STEPS)})
    doc = {"version": 1, "converged": True, "anchored_to_bots": True,
           "ratings": ratings, "se": {str(s): 9.0 for s in STEPS}}
    if stamped:
        doc["eval_sentinel_edges_dropped"] = 0
        doc["recipe"] = sl.ladder_recipe(0)
        doc["ratings"] = honest["ratings"]
        doc["se"] = honest["se"]
    with open(sl.ladder_json_path(run), "w") as fh:
        json.dump(doc, fh)
    return run


# ── the read ────────────────────────────────────────────────────────────────────────────────

def test_the_report_names_the_committed_file_as_stale_and_measures_every_node(tmp_path):
    run = _build_run(tmp_path)
    rep = elo_cli.ladder_refit_report(run)
    assert rep["recipe_status"] == "absent"
    assert sl.recipe_status(rep["refit"])[0] == "current"
    assert [n["step"] for n in rep["nodes"]] == STEPS
    # every node is 50 Elo below its committed value, by construction
    assert all(n["delta"] == pytest.approx(-50.0, abs=0.2) for n in rep["nodes"])
    assert rep["max_abs_delta"] == pytest.approx(50.0, abs=0.2)
    assert rep["newest"]["step"] == STEPS[-1]


def test_the_READ_writes_nothing_at_all(tmp_path):
    run = _build_run(tmp_path)
    before = {p: os.path.getmtime(os.path.join(run, "snapshot_ladder", p))
              for p in os.listdir(os.path.join(run, "snapshot_ladder"))}
    elo_cli.ladder_refit_report(run)
    assert elo_cli.refit_main([run]) == 0
    after = {p: os.path.getmtime(os.path.join(run, "snapshot_ladder", p))
             for p in os.listdir(os.path.join(run, "snapshot_ladder"))}
    assert before == after, "a read must never touch models/"


def test_the_node_set_is_the_COMMITTED_one_not_the_pool(tmp_path):
    """BT re-solves on every add, so a refit over a DIFFERENT node set is a different object and
    its deltas would conflate the recipe change with a node-set change."""
    run = _build_run(tmp_path)
    doc = json.load(open(sl.ladder_json_path(run)))
    del doc["ratings"][str(STEPS[-1])]                 # the committed file rates only two nodes
    with open(sl.ladder_json_path(run), "w") as fh:
        json.dump(doc, fh)
    rep = elo_cli.ladder_refit_report(run)
    assert [n["step"] for n in rep["nodes"]] == STEPS[:2]
    assert rep["refit"]["ratings"].keys() == {str(s) for s in STEPS[:2]}


def test_a_run_with_no_raw_pair_log_REFUSES_rather_than_guess(tmp_path):
    run = _build_run(tmp_path, games=False)
    with pytest.raises(FileNotFoundError, match="CANNOT be refit"):
        elo_cli.ladder_refit_report(run)
    assert elo_cli.refit_main([run]) == 2


# ── the write ───────────────────────────────────────────────────────────────────────────────

def test_apply_writes_the_stamped_fit_and_KEEPS_the_committed_file(tmp_path):
    run = _build_run(tmp_path)
    committed = json.load(open(sl.ladder_json_path(run)))
    assert elo_cli.refit_main(["--apply", run]) == 0

    new = json.load(open(sl.ladder_json_path(run)))
    assert sl.recipe_status(new)[0] == "current"
    assert new["ratings"] != committed["ratings"]

    backup = sl.pre_recipe_backup_path(run)
    assert os.path.exists(backup)
    assert json.load(open(backup)) == committed, "the pre-recipe file is kept VERBATIM"


def test_a_SECOND_apply_with_nothing_to_do_is_a_no_op(tmp_path):
    run = _build_run(tmp_path)
    assert elo_cli.refit_main(["--apply", run]) == 0
    backup_before = json.load(open(sl.pre_recipe_backup_path(run)))
    ladder_before = json.load(open(sl.ladder_json_path(run)))

    assert elo_cli.refit_main(["--apply", run]) == 0
    assert json.load(open(sl.pre_recipe_backup_path(run))) == backup_before
    assert json.load(open(sl.ladder_json_path(run))) == ladder_before


def test_a_second_apply_that_WOULD_change_the_file_refuses_to_clobber_the_pre_recipe_copy(
        tmp_path):
    """The pre-recipe file is the only surviving evidence of what a banked number was quoted
    from. A later refit (more games appended, a bumped fitter) must not overwrite it."""
    run = _build_run(tmp_path)
    assert elo_cli.refit_main(["--apply", run]) == 0
    backup_before = json.load(open(sl.pre_recipe_backup_path(run)))
    ladder_before = json.load(open(sl.ladder_json_path(run)))
    with open(sl.games_log_path(run), "a") as fh:      # a new measurement moves the fit
        fh.write(json.dumps({"a": STEPS[0], "b": STEPS[2], "wins_a": 80, "games": 100}) + "\n")

    assert elo_cli.refit_main(["--apply", run]) == 2
    assert json.load(open(sl.pre_recipe_backup_path(run))) == backup_before
    assert json.load(open(sl.ladder_json_path(run))) == ladder_before


def test_apply_on_an_ALREADY_CURRENT_file_is_a_no_op(tmp_path):
    run = _build_run(tmp_path, stamped=True)
    before = json.load(open(sl.ladder_json_path(run)))
    assert elo_cli.refit_main(["--apply", run]) == 0
    assert json.load(open(sl.ladder_json_path(run))) == before
    assert not os.path.exists(sl.pre_recipe_backup_path(run)), \
        "a no-op must not leave a spurious pre_recipe copy"


# ── the headline, and the message that names the fix ────────────────────────────────────────

def test_the_elo_CLI_WITHHOLDS_the_headline_on_an_unstamped_file_and_names_refit_apply(tmp_path):
    run = _build_run(tmp_path)
    line = elo_cli.ladder_headline(run)
    assert "HEADLINE WITHHELD" in line
    assert "python -m main.elo refit --apply" in line
    # the withheld number must not be printed anywhere in the refusal
    assert str(int(float(json.load(open(sl.ladder_json_path(run)))["ratings"][str(STEPS[-1])]))) \
        not in line


def test_the_headline_QUOTES_the_number_once_the_file_is_stamped(tmp_path):
    run = _build_run(tmp_path, stamped=True)
    line = elo_cli.ladder_headline(run)
    assert "HEADLINE (dense)" in line
    assert "WITHHELD" not in line


def test_EVERY_readers_refusal_names_the_apply_tool(tmp_path):
    """`recipe_refusal` words the fix once, so `main.critic_gate`, `--exploiter-ladder auto` and
    `main.ops.plateau_signal` all name the same two commands."""
    msg = sl.recipe_refusal("/x/snapshot_ladder/ladder.json", "absent", "no `recipe` block",
                            "/x")
    assert "python -m main.elo refit --apply /x" in msg
    assert "--fit-only" in msg          # the in-place refit is still named
    assert "+73.1" in msg               # and what the refusal is preventing
