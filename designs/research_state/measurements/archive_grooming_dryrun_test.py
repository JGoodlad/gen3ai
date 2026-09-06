"""The retention policy, pinned on a SYNTHETIC tree under tmp_path.

Nothing here touches the real ``models/`` archive — every fixture is built from
scratch in a temp dir, which is also the only place ``--apply`` is ever exercised.

Unmarked and fast (no torch, no battles, no model load) — but ``pytest.ini`` sets
``testpaths = src tools``, so a test under ``designs/`` is NOT collected by the
routine gate and is run directly, the same way
``arch_transfer_2026-09-05/content_locality/kl_unit_test.py`` is:

Run:
    python -m pytest designs/research_state/measurements/archive_grooming_dryrun_test.py -q

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import archive_grooming_dryrun as ag  # noqa: E402


# --------------------------------------------------------------- fixtures ----


def _write(path: str, blob: bytes = b"x") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)


def make_run(root: str, name: str, *, n_ckpt: int = 25, n_trace_steps: int = 6,
             latest_step: "int | None" = None, meta: "dict | None" = None,
             ckpt_bytes: int = 1000) -> str:
    """A synthetic run dir with the layout the real archive uses."""
    run = os.path.join(root, name)
    os.makedirs(run, exist_ok=True)
    for i in range(n_ckpt):
        step = (i + 1) * 1000
        _write(os.path.join(run, "checkpoints", f"checkpoint_{step}_steps.zip"),
               b"z" * ckpt_bytes)
        _write(os.path.join(run, "checkpoints", f"checkpoint_{step}_steps.json"),
               b"{}")
    for i in range(n_trace_steps):
        step = (i + 1) * 5000
        d = os.path.join(run, "eval_traces", f"step_{step}")
        _write(os.path.join(d, "b0_summary.json"), b"{}")
        _write(os.path.join(d, "b0_states.npz"), b"n" * 500)
        _write(os.path.join(d, "snapshot.zip"), b"s" * 2000)
    _write(os.path.join(run, "tb", "events.out.tfevents.1"), b"t" * 100)
    _write(os.path.join(run, "best_model", "best_model.zip"), b"b" * 100)
    _write(os.path.join(run, "snapshots", "snapshot_1.zip"), b"p" * 100)
    _write(os.path.join(run, "snapshot_ladder", "ladder.json"), b"{}")
    _write(os.path.join(run, "eval_results.jsonl"), b"{}\n")
    _write(os.path.join(run, "model_config.json"),
           json.dumps({"config_version": 42, "arch_signature": "sig"}).encode())
    _write(os.path.join(run, "metadata.json"), json.dumps(meta or {}).encode())
    if latest_step is not None:
        _write(os.path.join(run, "latest.txt"),
               f"checkpoints/checkpoint_{latest_step}_steps.zip".encode())
    return run


@pytest.fixture
def archive(tmp_path):
    """models/ with three runs; the repo half is an empty non-git dir."""
    models = tmp_path / "models"
    models.mkdir()
    make_run(str(models), "ai_v9_01_alpha_0101", latest_step=7000)
    make_run(str(models), "ai_v9_02_beta_0102", latest_step=25000, meta={
        "lineage": {
            "role": "fork",
            "fork_parent": {
                "run_name": "ai_v9_01_alpha_0101",
                "resolved_file": "/x/ai_v9_01_alpha_0101/checkpoints/"
                                 "checkpoint_13000_steps.zip",
            },
            "teachers": [],
        }
    })
    make_run(str(models), "ai_v8_99_era_0103", latest_step=25000)
    repo = tmp_path / "repo"
    repo.mkdir()
    return str(models), str(repo)


# ------------------------------------------------------- checkpoint policy ----


def test_first_last_and_every_tenth_are_kept():
    names = []
    for i in range(25):
        names += [f"checkpoint_{(i + 1) * 1000}_steps.zip",
                  f"checkpoint_{(i + 1) * 1000}_steps.json"]
    keep, delete = ag.plan_checkpoints(names)
    kept_steps = sorted({ag.checkpoint_step(n) for n in keep})
    # indices 0, 10, 20 of the ascending step list, plus the last (index 24)
    assert kept_steps == [1000, 11000, 21000, 25000]
    assert keep["checkpoint_1000_steps.zip"].startswith("first")
    assert "last" in keep["checkpoint_25000_steps.zip"]
    assert "every-10th" in keep["checkpoint_11000_steps.zip"]
    assert len(delete) == 2 * (25 - 4)


def test_a_zip_and_its_sidecar_are_never_split():
    names = []
    for i in range(15):
        names += [f"checkpoint_{(i + 1) * 100}_steps.zip",
                  f"checkpoint_{(i + 1) * 100}_steps.json"]
    keep, delete = ag.plan_checkpoints(names)
    for group in (keep, delete):
        steps = [ag.checkpoint_step(n) for n in group]
        for s in set(steps):
            assert steps.count(s) == 2, f"step {s} was split across keep/delete"


def test_latest_txt_pin_survives_a_plain_every_nth_rule():
    names = [f"checkpoint_{(i + 1) * 1000}_steps.zip" for i in range(25)]
    _keep0, delete0 = ag.plan_checkpoints(names)
    assert "checkpoint_7000_steps.zip" in delete0        # not otherwise kept
    keep, delete = ag.plan_checkpoints(
        names, latest_pin="checkpoints/checkpoint_7000_steps.zip")
    assert "checkpoint_7000_steps.zip" not in delete
    assert "latest.txt pin" in keep["checkpoint_7000_steps.zip"]


def test_a_checkpoint_another_run_resolved_to_is_kept():
    names = [f"checkpoint_{(i + 1) * 1000}_steps.zip" for i in range(25)]
    keep, delete = ag.plan_checkpoints(
        names, pinned_files={"checkpoint_13000_steps.zip"})
    assert "checkpoint_13000_steps.zip" not in delete
    assert "lineage" in keep["checkpoint_13000_steps.zip"]


def test_an_unrecognised_name_is_kept_not_guessed_at():
    keep, delete = ag.plan_checkpoints(
        ["checkpoint_1000_steps.zip", "checkpoint_2000_steps.zip", "README"])
    assert "README" in keep
    assert "README" not in delete


def test_a_run_shorter_than_the_stride_keeps_only_its_ends():
    """`first + last + every 10th` is aggressive by design — under 11 checkpoints
    it degenerates to the two ends, and that is the intended behaviour, not a
    short-run exemption."""
    names = [f"checkpoint_{(i + 1) * 1000}_steps.zip" for i in range(3)]
    keep, delete = ag.plan_checkpoints(names)
    assert sorted(keep) == ["checkpoint_1000_steps.zip", "checkpoint_3000_steps.zip"]
    assert delete == ["checkpoint_2000_steps.zip"]


def test_a_single_checkpoint_run_loses_nothing():
    keep, delete = ag.plan_checkpoints(
        ["checkpoint_1000_steps.zip", "checkpoint_1000_steps.json"])
    assert delete == []
    assert len(keep) == 2


# ------------------------------------------------------------- safety net ----


@pytest.mark.parametrize("rel", [
    "tb/events.out.tfevents.1",
    "best_model/best_model.zip",
    "snapshots/snapshot_1.zip",
    "snapshot_ladder/ladder.json",
    "metadata.json",
    "model_config.json",
    "latest.txt",
    "eval_results.jsonl",
    "checkpoints/../metadata.json",
    "../other_run/checkpoints/checkpoint_1000_steps.zip",
])
def test_assert_safe_refuses_everything_outside_the_two_touchable_dirs(rel):
    with pytest.raises(ag.UnsafeDeletion):
        ag._assert_safe("/run", [os.path.join("/run", rel)])


def test_assert_safe_admits_the_two_touchable_dirs():
    ag._assert_safe("/run", ["/run/checkpoints/checkpoint_1000_steps.zip",
                             "/run/eval_traces/step_5000",
                             "/run/eval_traces/step_5000/snapshot.zip"])


# ----------------------------------------------------------- census + plan ----


def test_census_never_proposes_a_protected_path(archive):
    models, repo = archive
    c = ag.build_census(models, repo, recent_days=0)
    for name, r in c["runs"].items():
        for rel in r["plan"]["delete"]:
            top = rel.split(os.sep)[0]
            assert top in ag.TOUCHABLE_SUBDIRS, f"{name}: {rel}"
            assert os.path.basename(rel) not in ag.PROTECTED_FILES


def test_v8_era_runs_are_referenced_and_get_no_plan(archive):
    models, repo = archive
    c = ag.build_census(models, repo, recent_days=0)
    v8 = c["runs"]["ai_v8_99_era_0103"]
    assert v8["status"] == "REFERENCED"
    assert v8["plan"]["delete"] == []
    assert any("v8-era" in s for s in v8["status_reasons"])


def test_recent_activity_protects_a_run(archive):
    models, repo = archive
    c = ag.build_census(models, repo, recent_days=7)   # the tree was just built
    assert c["totals"]["n_closed"] == 0
    assert c["totals"]["freed_bytes"] == 0
    for r in c["runs"].values():
        assert r["mtime_source"] in ag._ACTIVITY_SUBDIRS


def test_a_metadata_backfill_does_not_read_as_activity(archive):
    """The 2026-09-01 defect, pinned.

    `main.lineage --backfill --apply` restamped 153 of 217 `metadata.json` in one
    pass, and a run-root mtime therefore read the WHOLE archive as touched-today:
    195 of 217 runs came back protected on recency alone.  Recency must be read
    from training output, so an old run whose bookkeeping files were rewritten
    an instant ago stays CLOSED.
    """
    models, repo = archive
    old = time.time() - 400 * 86400
    for name in os.listdir(models):
        run = os.path.join(models, name)
        for dp, _d, fs in os.walk(run):
            for f in fs:
                os.utime(os.path.join(dp, f), (old, old))
    # ...then a bookkeeping pass rewrites the run-root files, right now
    for name in os.listdir(models):
        for f in ("metadata.json", "model_config.json", "latest.txt",
                  "eval_results.jsonl"):
            p = os.path.join(models, name, f)
            if os.path.exists(p):
                os.utime(p, None)

    c = ag.build_census(models, repo, recent_days=7)
    for name, r in c["runs"].items():
        assert not any("within" in s for s in r["status_reasons"]), \
            f"{name} read as recent off a run-root restamp"
    assert c["runs"]["ai_v9_01_alpha_0101"]["status"] == "CLOSED"
    assert c["totals"]["freed_bytes"] > 0


def test_a_parents_pinned_checkpoint_is_kept(archive):
    models, repo = archive
    c = ag.build_census(models, repo, recent_days=0)
    parent = c["runs"]["ai_v9_01_alpha_0101"]
    assert parent["status"] == "CLOSED"          # a fork parent of a CLOSED child
    assert "checkpoint_13000_steps.zip" in parent["plan"]["keep"]
    assert "lineage" in parent["plan"]["keep"]["checkpoint_13000_steps.zip"]
    assert "checkpoints/checkpoint_13000_steps.zip" not in parent["plan"]["delete"]


def test_traces_keep_the_three_newest_steps(archive):
    models, repo = archive
    c = ag.build_census(models, repo, recent_days=0)
    r = c["runs"]["ai_v9_01_alpha_0101"]
    removed = [d for d in r["plan"]["delete"] if d.startswith("eval_traces")]
    # 6 step dirs, 3 kept -> 3 removed, and 2 of the 3 kept lose their snapshot
    assert sum(1 for d in removed if d.endswith("snapshot.zip")) == 2
    assert sum(1 for d in removed if not d.endswith("snapshot.zip")) == 3
    assert "eval_traces/step_30000" not in removed


def test_a_committed_file_naming_a_planned_path_vetoes_the_run(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    make_run(str(models), "ai_v9_03_gamma_0104", latest_step=25000)
    repo = tmp_path / "repo"
    (repo / "designs").mkdir(parents=True)

    base = ag.build_census(str(models), str(repo), recent_days=0)
    assert base["runs"]["ai_v9_03_gamma_0104"]["plan"]["delete"]
    assert base["totals"]["n_excluded_by_named_file"] == 0

    # PROSE (not a script — prose alone must not protect a whole run) that names
    # one of the exact checkpoints the plan would delete
    (repo / "designs" / "note.md").write_text(
        "we read models/ai_v9_03_gamma_0104/checkpoints/checkpoint_5000_steps.zip\n")
    monkey = ag.committed_files
    try:
        ag.committed_files = lambda root: ["designs/note.md"]
        c = ag.build_census(str(models), str(repo), recent_days=0)
    finally:
        ag.committed_files = monkey

    r = c["runs"]["ai_v9_03_gamma_0104"]
    assert r["plan"]["delete"] == []
    assert r["status"] == "REFERENCED"
    assert c["totals"]["n_excluded_by_named_file"] == 1
    assert c["excluded_by_named_file"][0]["run"] == "ai_v9_03_gamma_0104"


@pytest.mark.parametrize("origin,protects", [
    ("designs/research_state/measurements/probe.py", True),
    ("scripts/rerun.sh", True),
    ("designs/research_state/ledger_archive.md", False),
    ("designs/ai_v12/notes.json", False),
])
def test_a_script_naming_a_run_protects_it_but_prose_does_not(tmp_path, origin,
                                                              protects):
    models = tmp_path / "models"
    models.mkdir()
    make_run(str(models), "ai_v9_04_delta_0105", latest_step=25000)
    repo = tmp_path / "repo"
    (repo / os.path.dirname(origin)).mkdir(parents=True, exist_ok=True)
    (repo / origin).write_text("models/ai_v9_04_delta_0105\n")

    monkey = ag.committed_files
    try:
        ag.committed_files = lambda root: [origin]
        c = ag.build_census(str(models), str(repo), recent_days=0)
    finally:
        ag.committed_files = monkey

    r = c["runs"]["ai_v9_04_delta_0105"]
    if protects:
        assert r["status"] == "REFERENCED"
        assert r["plan"]["delete"] == []
        assert any("committed script" in s for s in r["status_reasons"])
    else:
        assert r["status"] == "CLOSED"
        assert r["plan"]["delete"], "prose alone must not close the plan down"
        assert r["n_named_by_committed_prose"] == 1


def test_a_symlinked_run_dir_is_held_out_unless_opted_in(tmp_path):
    """Eight run dirs in the live archive are symlinks into launcher worktrees, so
    a deletion "in models/" lands physically outside it and `du -sh models/` cannot
    even see the bytes. Held out by default; opt in explicitly."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    real = make_run(str(elsewhere), "real_run_0101", latest_step=25000)

    models = tmp_path / "models"
    models.mkdir()
    os.symlink(real, str(models / "ai_v9_50_linked_0101"))
    repo = tmp_path / "repo"
    repo.mkdir()

    c = ag.build_census(str(models), str(repo), recent_days=0)
    r = c["runs"]["ai_v9_50_linked_0101"]
    assert r["is_symlink"] is True
    assert r["status"] == "REFERENCED"
    assert r["plan"]["delete"] == []
    assert any("SYMLINK" in s for s in r["status_reasons"])
    assert c["totals"]["n_symlinked_runs"] == 1
    # the fixture is ~80 KB, so the GB column rounds to 0.0 — assert on the split,
    # which is the fact that matters: none of the archive's bytes are under models/
    assert c["totals"]["in_models_gb"] == 0.0
    assert r["sizes"]["total"] > 0
    assert c["symlinked_runs"][0]["realpath"] == os.path.realpath(real)

    opted = ag.build_census(str(models), str(repo), recent_days=0,
                            follow_symlinked_runs=True)
    assert opted["runs"]["ai_v9_50_linked_0101"]["plan"]["delete"]


def test_assert_safe_resolves_symlinks_before_judging_containment(tmp_path):
    """An abspath-only guard passes on a path that reads `<run>/checkpoints/x` while
    the file physically sits somewhere else entirely."""
    outside = tmp_path / "outside"
    (outside / "checkpoints").mkdir(parents=True)
    (outside / "checkpoints" / "x.zip").write_bytes(b"z")
    run = tmp_path / "run"
    run.mkdir()
    os.symlink(str(outside / "checkpoints"), str(run / "checkpoints"))

    # the realpath of run/checkpoints/x.zip is outside run/ -> refused
    with pytest.raises(ag.UnsafeDeletion):
        ag._assert_safe(str(run), [str(run / "checkpoints" / "x.zip")])
    # but when the RUN DIR itself is the symlink, both sides resolve together
    linked_run = tmp_path / "linked_run"
    os.symlink(str(outside), str(linked_run))
    ag._assert_safe(str(linked_run), [str(linked_run / "checkpoints" / "x.zip")])


def test_dry_run_writes_no_change_to_the_tree(archive):
    models, repo = archive
    before = {p: os.path.getsize(p)
              for p in (os.path.join(dp, f)
                        for dp, _d, fs in os.walk(models) for f in fs)}
    ag.build_census(models, repo, recent_days=0)
    after = {p: os.path.getsize(p)
             for p in (os.path.join(dp, f)
                       for dp, _d, fs in os.walk(models) for f in fs)}
    assert before == after


# ----------------------------------------------------------------- --apply ----


def test_apply_deletes_exactly_the_plan_and_nothing_else(archive):
    """The ONLY place --apply runs: a synthetic tree under tmp_path."""
    models, repo = archive
    c = ag.build_census(models, repo, recent_days=0)

    planned = {os.path.join(c["runs"][n]["run_dir"], rel)
               for n, r in c["runs"].items() for rel in r["plan"]["delete"]}
    assert planned, "the fixture must produce a non-empty plan"

    survivors_before = {os.path.join(dp, f)
                        for dp, _d, fs in os.walk(models) for f in fs}
    ag.apply_plan(c)
    survivors_after = {os.path.join(dp, f)
                       for dp, _d, fs in os.walk(models) for f in fs}

    gone = survivors_before - survivors_after
    for g in gone:
        assert any(g == p or g.startswith(p + os.sep) for p in planned), g
    for p in planned:
        assert not os.path.exists(p)

    assert c["applied"] is True
    assert c["apply_result"]["errors"] == []
    assert c["apply_result"]["bytes_freed"] > 0

    # every protected artefact is still there, in every run
    for name in c["runs"]:
        run = os.path.join(models, name)
        for rel in ("tb/events.out.tfevents.1", "best_model/best_model.zip",
                    "snapshots/snapshot_1.zip", "snapshot_ladder/ladder.json",
                    "metadata.json", "model_config.json", "latest.txt",
                    "eval_results.jsonl"):
            assert os.path.exists(os.path.join(run, rel)), f"{name}/{rel}"


def test_apply_is_off_by_default_in_the_cli(archive, tmp_path, capsys):
    models, repo = archive
    before = {os.path.join(dp, f) for dp, _d, fs in os.walk(models) for f in fs}
    rc = ag.main(["--models-dir", models, "--repo-root", repo,
                  "--recent-days", "0",
                  "--out-prefix", str(tmp_path / "report")])
    assert rc == 0
    after = {os.path.join(dp, f) for dp, _d, fs in os.walk(models) for f in fs}
    assert before == after
    out = capsys.readouterr().out
    assert "NOTHING WAS DELETED IN THIS PASS" in out
    assert os.path.exists(str(tmp_path / "report.md"))
    assert os.path.exists(str(tmp_path / "report.json"))
    md = open(str(tmp_path / "report.md")).read()
    assert "NOTHING WAS DELETED IN THIS PASS" in md
    assert "--apply" in md
