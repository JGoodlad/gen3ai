"""THE LADDER RECIPE STAMP — a rating is only comparable to one fitted the same way.

The defect this file pins (tech-debt P1, 2026-09-14). A committed
`<run>/snapshot_ladder/ladder.json` fitted BEFORE `3e6875a5` folded the eval cycles'
greedy-vs-stochastic SENTINEL edges into the greedy-vs-greedy frozen matrix. Measured on
`ai_v12_02_winprob_critic` (`flywheel_armS_reads_2026-09-14/` §2.1): the committed file reads
**2057.3** at its newest node and the current recipe refits the SAME 20 nodes to **1984.2** — a
**+73.1 Elo** gap that FLIPPED THE SIGN of a cross-run delta. Nothing in the file said which
recipe produced it, so nothing could have caught it.

Now `fit_ladder` stamps a `recipe` block and every CROSS-RUN reader refuses or refits. The two
fixtures below are literally "before" and "after": the same ratings, differing only in the stamp.
"""
from __future__ import annotations

import json
import os

import pytest

from agents.training import snapshot_ladder as sl


# ── the two fixtures: a pre-recipe file and a current one ────────────────────────────────────

def _ratings():
    return {"ratings": {"2000000": 1900.0, "4000000": 1950.0, "6000000": 2000.0},
            "se": {"2000000": 9.0, "4000000": 9.0, "6000000": 9.0},
            "converged": True, "anchored_to_bots": True, "version": 1}


def pre_recipe_ladder() -> dict:
    """A file as written before 2026-09-22 — and before `3e6875a5`, so `eval_sentinel_edges_dropped`
    is not even present. This is the shape of the 2026-09-08 file that read +73.1 Elo high."""
    return _ratings()


def post_fix_but_unstamped_ladder() -> dict:
    """Written between `3e6875a5` and the stamp: the sentinel edges WERE dropped, but nothing else
    about the fit is pinned. Still refused — the count alone was only ever a proxy."""
    return {**_ratings(), "eval_sentinel_edges_dropped": 12}


def current_ladder() -> dict:
    return {**_ratings(), "eval_sentinel_edges_dropped": 12,
            "recipe": sl.ladder_recipe(12)}


def _write(run_dir, doc):
    os.makedirs(os.path.join(run_dir, "snapshot_ladder"), exist_ok=True)
    with open(sl.ladder_json_path(run_dir), "w") as fh:
        json.dump(doc, fh)
    return sl.ladder_json_path(run_dir)


# ── the stamp itself ─────────────────────────────────────────────────────────────────────────

def test_the_fitter_writes_a_recipe_block(tmp_path, monkeypatch):
    run = str(tmp_path)
    steps = [100, 200, 300]
    for i, a in enumerate(steps):
        for b in steps[i + 1:]:
            sl._append_game(run, b, a, 65, 100)
    monkeypatch.setattr(sl.elo_mod, "load_bot_anchors", lambda: None)
    monkeypatch.setattr(sl.elo_mod, "load_rows", lambda run_dir, source="log": [])
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda run_dir: steps)
    ladder = sl.fit_ladder(run)
    r = ladder["recipe"]
    assert r["name"] == sl.LADDER_RECIPE_NAME
    assert r["fitter_version"] == sl.LADDER_FITTER_VERSION
    assert r["eval_sentinel_edges_dropped"] is True          # the POLICY, always true here
    assert r["eval_sentinel_edges_dropped_count"] == 0       # the COUNT for THIS run
    assert len(r["commit"]) in (0, 40)
    # and it survives the round trip to disk
    on_disk = json.load(open(sl.ladder_json_path(run)))
    assert sl.recipe_status(on_disk)[0] == "current"


def test_the_POLICY_and_the_COUNT_are_different_facts(tmp_path, monkeypatch):
    """A run that never measured a sentinel pair drops ZERO edges and is still on the current
    recipe. That is exactly why the count can never be the stamp: `0` and "fitted by a tree that
    did not drop them" are indistinguishable.
    """
    assert sl.recipe_status({**_ratings(), "eval_sentinel_edges_dropped": 0})[0] == "absent"
    assert sl.recipe_status(current_ladder())[0] == "current"


def test_recipe_status_classifies_the_before_and_after_fixtures():
    assert sl.recipe_status(pre_recipe_ladder())[0] == "absent"
    assert sl.recipe_status(post_fix_but_unstamped_ladder())[0] == "absent"
    assert sl.recipe_status(current_ladder())[0] == "current"
    # a NULL block reads absent, not current — the 2026-09-08 file records exactly that
    assert sl.recipe_status({**_ratings(), "recipe": None})[0] == "absent"
    assert sl.recipe_status({**_ratings(), "recipe": {}})[0] == "absent"


def test_a_BUMPED_fitter_version_reads_as_differs():
    """The stamp must catch the NEXT recipe change, not only the one that prompted it."""
    doc = current_ladder()
    doc["recipe"] = {**doc["recipe"], "fitter_version": sl.LADDER_FITTER_VERSION + 1}
    status, detail = sl.recipe_status(doc)
    assert status == "differs" and str(sl.LADDER_FITTER_VERSION) in detail
    doc["recipe"] = {**doc["recipe"], "name": "gen3_ladder_recipe_v2",
                     "fitter_version": sl.LADDER_FITTER_VERSION}
    assert sl.recipe_status(doc)[0] == "differs"


def test_check_recipe_refuses_with_the_refit_command(tmp_path):
    path = _write(str(tmp_path), pre_recipe_ladder())
    with pytest.raises(sl.LadderRecipeError) as exc:
        sl.check_recipe(pre_recipe_ladder(), path, run_dir=str(tmp_path), what="arm")
    msg = str(exc.value)
    assert "STALE LADDER RECIPE" in msg
    assert "--fit-only" in msg and str(tmp_path) in msg
    assert "+73.1" in msg, "the refusal must carry the size of the error it is preventing"
    sl.check_recipe(current_ladder(), path, run_dir=str(tmp_path))   # does not raise


# ── the readers ──────────────────────────────────────────────────────────────────────────────

def test_critic_gate_load_ladder_records_the_stamp(tmp_path):
    from main import critic_gate
    _write(str(tmp_path), pre_recipe_ladder())
    doc = critic_gate.load_ladder(str(tmp_path), what="run")
    assert doc["recipe_status"] == "absent"
    _write(str(tmp_path), current_ladder())
    assert critic_gate.load_ladder(str(tmp_path), what="run")["recipe_status"] == "current"


def test_the_exploiter_auto_ladder_REFUSES_a_stale_file_it_cannot_refit(tmp_path):
    """`--exploiter-ladder auto:` picks rungs BY ELO, and the pre-recipe inflation is
    non-uniform (+21..+29 on the newest nodes only) — so a stale file builds a different
    curriculum. With no `games.jsonl` there is nothing to refit from, and it stops."""
    from agents.training import exploiter_ladder
    run = str(tmp_path)
    _write(run, pre_recipe_ladder())
    with pytest.raises(ValueError) as exc:
        exploiter_ladder._auto_rung_paths(run, 2)
    assert "STALE LADDER RECIPE" in str(exc.value)
    assert "explicit rung list" in str(exc.value)


def test_the_exploiter_auto_ladder_REFITS_a_stale_file_when_it_can(tmp_path, monkeypatch, capsys):
    from agents.training import exploiter_ladder
    run = str(tmp_path)
    _write(run, pre_recipe_ladder())
    for step in (2000000, 4000000, 6000000):
        os.makedirs(os.path.join(run, "snapshots"), exist_ok=True)
        open(os.path.join(run, "snapshots", f"snapshot_{step:012d}.zip"), "w").close()
    steps = [2000000, 4000000, 6000000]
    for i, a in enumerate(steps):
        for b in steps[i + 1:]:
            sl._append_game(run, b, a, 65, 100)
    monkeypatch.setattr(sl.elo_mod, "load_bot_anchors", lambda: None)
    monkeypatch.setattr(sl.elo_mod, "load_rows", lambda run_dir, source="log": [])
    paths = exploiter_ladder._auto_rung_paths(run, 2)
    assert len(paths) == 2
    out = capsys.readouterr().out
    assert "STALE fit recipe" in out and "REFITTING" in out
    # the committed file is NOT rewritten by a read
    assert sl.recipe_status(json.load(open(sl.ladder_json_path(run))))[0] == "absent"


def test_a_CURRENT_file_is_used_as_is_by_the_exploiter_ladder(tmp_path):
    from agents.training import exploiter_ladder
    run = str(tmp_path)
    _write(run, current_ladder())
    for step in (2000000, 4000000, 6000000):
        os.makedirs(os.path.join(run, "snapshots"), exist_ok=True)
        open(os.path.join(run, "snapshots", f"snapshot_{step:012d}.zip"), "w").close()
    assert len(exploiter_ladder._auto_rung_paths(run, 3)) == 3   # no games.jsonl needed


def test_latest_promoted_elo_does_NOT_check_the_stamp(tmp_path):
    """A WITHIN-RUN trend scalar written by the run's own pinned code. Refusing here would stop a
    live run logging its own curve, and no cross-run comparison is being made."""
    run = str(tmp_path)
    _write(run, pre_recipe_ladder())
    assert sl.latest_promoted_elo(run) == (6000000, 2000.0, 9.0)
