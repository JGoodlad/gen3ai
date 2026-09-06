"""The TIERED retention policy, pinned on a SYNTHETIC tree under ``tmp_path``.

Sibling of ``archive_grooming_dryrun_test.py`` (which pins the STANDING policy and
must keep passing unchanged — that is the whole point of shipping the tiered policy
beside the standing one rather than replacing it).

Nothing here touches the real ``models/`` archive: every fixture is built from
scratch in a temp dir, which is also the only place ``--apply`` is ever exercised.

Run:
    python -m pytest designs/research_state/measurements/archive_grooming_tiered_test.py -q

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import archive_grooming_dryrun as ag        # noqa: E402
import archive_grooming_tiers as tiers      # noqa: E402
from archive_grooming_dryrun_test import make_run   # noqa: E402


# --------------------------------------------------------------- fixtures ----


def _stamp_old(models: str, days: int = 400) -> None:
    """Push every file back, so recency (tier 0) never masks the era grading."""
    old = time.time() - days * 86400
    for dp, _d, fs in os.walk(models):
        for f in fs:
            os.utime(os.path.join(dp, f), (old, old))


def _add_root_checkpoints(run: str, steps: "list[int]", size: int = 4000) -> None:
    """The LEGACY layout: checkpoints at the run ROOT, not under checkpoints/.

    13 pre-v8 runs in the live archive are like this and hold 20.8 GB — MORE than
    the standing policy frees archive-wide, and completely invisible to it, because
    the standing planner only ever looks inside ``checkpoints/``.
    """
    for s in steps:
        with open(os.path.join(run, f"checkpoint_{s}_steps.zip"), "wb") as fh:
            fh.write(b"z" * size)
        with open(os.path.join(run, f"checkpoint_{s}_steps.json"), "w") as fh:
            json.dump({"num_timesteps": s}, fh)


@pytest.fixture
def tiered_archive(tmp_path):
    """One run per tier, plus the pieces each tier's rule turns on."""
    models = tmp_path / "models"
    models.mkdir()

    # tier 2 — a plain closed ai_v9 run
    make_run(str(models), "ai_v9_70_closed_0820", latest_step=25000)
    # tier 3 — a closed ai_v8 run nothing reaches for
    make_run(str(models), "ai_v8_70_closed_0725", latest_step=25000)
    # tier 4 — pre-v8, in the LEGACY root layout, with a final_model to resolve
    t4 = make_run(str(models), "ai_v7_70_closed_0703", n_ckpt=0, latest_step=None)
    _add_root_checkpoints(t4, [1000, 2000, 3000])
    with open(os.path.join(t4, "final_model.zip"), "wb") as fh:
        fh.write(b"f" * 9000)
    with open(os.path.join(t4, "launcher_child.log"), "wb") as fh:
        fh.write(b"L" * 5000)
    with open(os.path.join(t4, "command.txt"), "w") as fh:
        fh.write("--steps 1\n")

    repo = tmp_path / "repo"
    repo.mkdir()
    _stamp_old(str(models))
    return str(models), str(repo)


def census(models: str, repo: str, **kw):
    kw.setdefault("recent_days", 0)
    return ag.build_census_tiered(models, repo, **kw)


# ------------------------------------------------------------ era + tiers ----


@pytest.mark.parametrize("name,generation,expect_pre_v8", [
    ("ai_v5_9_attend_0610", "ai_v5", True),
    ("ai_v6_01_belief_0613", "ai_v6", True),
    ("ai_v7_22_hyper_0717", "ai_v7", True),
    ("ai_v8_04_distill_0722", "ai_v8", False),
    ("ai_v9_21_gen17_0820", "ai_v9", False),
    ("ai_v12_01_winprob_0906", "ai_v12", False),
    # the un-prefixed run dirs the owner named: graded by DATE against 07-17
    ("run_20260715_091837", "unknown", True),
    ("run_20260830_180409", "unknown", False),
    ("run_20260906_083317", "unknown", False),
    ("warmstart_generic_0715", "unknown", True),
])
def test_era_grading_reads_the_generation_then_the_date(name, generation,
                                                        expect_pre_v8):
    era, how = tiers.era_of(name, generation)
    assert tiers.is_pre_v8(era) is expect_pre_v8, f"{name} -> {era} ({how})"
    assert how, "every era decision must say where it came from"


def test_an_unreadable_era_is_graded_as_the_GENTLEST_closed_tier(tmp_path):
    """A name we cannot date must never buy an AGGRESSIVE plan.  The failure
    direction of an unknown is 'treat it gently', not 'treat it as ancient'."""
    models = tmp_path / "models"
    models.mkdir()
    make_run(str(models), "some_experiment", latest_step=25000)
    repo = tmp_path / "repo"
    repo.mkdir()
    _stamp_old(str(models))
    c = census(str(models), str(repo))
    r = c["runs"]["some_experiment"]
    assert r["era"] == "unknown"
    assert r["tier"] == 2, "an unreadable era must land in the gentlest closed tier"


def test_each_run_lands_in_the_tier_its_era_names(tiered_archive):
    models, repo = tiered_archive
    c = census(models, repo)
    assert c["runs"]["ai_v9_70_closed_0820"]["tier"] == 2
    assert c["runs"]["ai_v8_70_closed_0725"]["tier"] == 3
    assert c["runs"]["ai_v7_70_closed_0703"]["tier"] == 4


def test_a_live_process_puts_a_run_and_its_ancestors_in_tier_0(tiered_archive,
                                                               monkeypatch):
    models, repo = tiered_archive
    make_run(models, "ai_v9_71_child_0901", latest_step=25000, meta={
        "original_command": "train_rl_agent.py --model "
                            "models/ai_v8_70_closed_0725/final_model.zip"})
    _stamp_old(models)
    monkeypatch.setattr(ag, "live_run_dirs",
                        lambda d: {"ai_v9_71_child_0901": "python -m main.launcher …"})
    c = census(models, repo)
    assert c["runs"]["ai_v9_71_child_0901"]["tier"] == 0
    parent = c["runs"]["ai_v8_70_closed_0725"]
    assert parent["tier"] == 0, "a live run's ancestor must not be graded on its era"
    assert any("ancestor" in s for s in parent["tier_reasons"])
    assert parent["plan"]["delete"] == []


def test_recent_training_output_is_tier_0(tiered_archive):
    models, repo = tiered_archive
    c = census(models, repo, recent_days=7)     # the fixture was stamped 400d old…
    assert all(r["tier"] != 0 for r in c["runs"].values())
    for dp, _d, fs in os.walk(os.path.join(models, "ai_v9_70_closed_0820", "tb")):
        for f in fs:
            os.utime(os.path.join(dp, f), None)  # …now one run trains again
    c2 = census(models, repo, recent_days=7)
    r = c2["runs"]["ai_v9_70_closed_0820"]
    assert r["tier"] == 0 and r["plan"]["delete"] == []


# ------------------------------------------------- the ARGV reference graph ----


def test_the_model_graph_reads_original_command_not_just_lineage(tmp_path):
    """The defect this closes, measured on the live archive: the ``v8rep_*``
    replication arms carry ``lineage: null``, so a lineage-only graph cannot see
    that they fork ``ai_v8_04`` and distil from ``ai_v8_09``/``_06``/``_13`` — and
    the ledger tail does not name those teachers either."""
    models = tmp_path / "models"
    models.mkdir()
    make_run(str(models), "ai_v8_80_parent_0722", latest_step=25000)
    make_run(str(models), "ai_v8_81_teacher_0723", latest_step=25000)
    make_run(str(models), "v8rep_p1_A_0905", latest_step=25000, meta={
        "lineage": None,       # exactly what the live replication arms carry
        "original_command":
            "train_rl_agent.py --model models/ai_v8_80_parent_0722/final_model.zip "
            "--distill-teacher /x/models/ai_v8_81_teacher_0723:*",
    })
    repo = tmp_path / "repo"
    repo.mkdir()
    _stamp_old(str(models))

    runs = {n: ag.scan_run(str(models), n) for n in ag.discover_runs(str(models))}
    g = tiers.build_model_graph(runs)
    kinds = {k for k, p, _f in g["refs_out"]["v8rep_p1_A_0905"]}
    parents = {p for _k, p, _f in g["refs_out"]["v8rep_p1_A_0905"]}
    assert parents == {"ai_v8_80_parent_0722", "ai_v8_81_teacher_0723"}
    assert "argv fork_parent" in kinds and "argv teacher" in kinds
    assert "ai_v8_80_parent_0722" in g["fork_parents"]
    assert "ai_v8_81_teacher_0723" not in g["fork_parents"], \
        "a TEACHER is not a fork parent — it does not seed a pool"

    c = census(str(models), str(repo))
    assert c["runs"]["ai_v8_80_parent_0722"]["tier"] == 1
    assert c["runs"]["ai_v8_81_teacher_0723"]["tier"] == 1


def test_a_run_never_references_itself(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    make_run(str(models), "ai_v9_82_solo_0820", latest_step=25000, meta={
        "original_command": "train_rl_agent.py --run-name ai_v9_82_solo_0820"})
    repo = tmp_path / "repo"
    repo.mkdir()
    _stamp_old(str(models))
    c = census(str(models), str(repo))
    assert c["runs"]["ai_v9_82_solo_0820"]["tier"] == 2


def test_an_archive_enumerating_artifact_is_not_a_reference(tmp_path):
    """🚨 The defect that made the FIRST tiered run useless: the committed census
    names every run in ``models/`` by construction, so reading it as evidence put
    all 118 non-tier-0 runs in tier 1 and graded nothing."""
    models = tmp_path / "models"
    models.mkdir()
    make_run(str(models), "ai_v7_83_old_0703", latest_step=25000)
    repo = tmp_path / "repo"
    (repo / tiers.MEASUREMENT_PREFIX).mkdir(parents=True)
    _stamp_old(str(models))

    rel_book = tiers.MEASUREMENT_PREFIX + "archive_grooming_dryrun_2026-09-06.md"
    rel_real = tiers.MEASUREMENT_PREFIX + "fold_capacity_telemetry.md"
    (repo / rel_book).write_text("| `ai_v7_83_old_0703` | ai_v7 | 1.0 |\n")
    (repo / rel_real).write_text("arm ai_v7_83_old_0703 scored +0.04\n")

    monkey = ag.committed_files
    try:
        ag.committed_files = lambda root: [rel_book]
        only_book = census(str(models), str(repo))
        ag.committed_files = lambda root: [rel_book, rel_real]
        with_real = census(str(models), str(repo))
    finally:
        ag.committed_files = monkey

    assert only_book["runs"]["ai_v7_83_old_0703"]["tier"] == 4, \
        "the tool's own report must not protect the runs it enumerates"
    assert with_real["runs"]["ai_v7_83_old_0703"]["tier"] == 1, \
        "a real measurement artifact naming the run DOES protect it"


def test_a_committed_measurement_artifact_protects_but_plain_prose_does_not(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    make_run(str(models), "ai_v7_84_old_0703", latest_step=25000)
    repo = tmp_path / "repo"
    (repo / "designs" / "research_state" / "measurements").mkdir(parents=True)
    (repo / "designs" / "ai_v7").mkdir(parents=True)
    _stamp_old(str(models))

    prose = "designs/ai_v7/notes.md"
    (repo / prose).write_text("we once ran ai_v7_84_old_0703\n")
    monkey = ag.committed_files
    try:
        ag.committed_files = lambda root: [prose]
        c = census(str(models), str(repo))
    finally:
        ag.committed_files = monkey
    assert c["runs"]["ai_v7_84_old_0703"]["tier"] == 4, \
        "ordinary prose names nearly every run forever — it must not protect one"


# ------------------------------------------------------- the snapshots rule ----


def test_a_pool_is_kept_only_for_a_fork_parent(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    parent = make_run(str(models), "ai_v9_85_parent_0820", latest_step=25000)
    make_run(str(models), "ai_v9_86_orphan_0820", latest_step=25000)
    make_run(str(models), "ai_v9_87_child_0821", latest_step=25000, meta={
        "lineage": {"role": "fork", "fork_parent": {
            "run_name": "ai_v9_85_parent_0820",
            "resolved_file": "/x/checkpoints/checkpoint_13000_steps.zip"}}})
    # the pool metadata a fork's auto-seed needs, beside the zips
    for f in ("summary.json", "win_rate_vs_bots.txt"):
        with open(os.path.join(parent, "snapshots", f), "w") as fh:
            fh.write("{}")
    repo = tmp_path / "repo"
    repo.mkdir()
    _stamp_old(str(models))

    c = census(str(models), str(repo))
    kept = c["runs"]["ai_v9_85_parent_0820"]["plan"]["snapshots"]
    freed = c["runs"]["ai_v9_86_orphan_0820"]["plan"]["snapshots"]
    assert kept["action"] == "keep" and "fork parent" in kept["reason"]
    assert "snapshots" not in c["runs"]["ai_v9_85_parent_0820"]["plan"]["delete"]
    assert freed["action"] == "delete"
    assert "snapshots" in c["runs"]["ai_v9_86_orphan_0820"]["plan"]["delete"]
    assert os.path.isdir(os.path.join(models, "ai_v9_86_orphan_0820", "snapshots")), \
        "a dry run must not have removed it"


def test_a_kept_pool_is_never_thinned_only_a_PROPOSAL_is_reported(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    parent = make_run(str(models), "ai_v9_88_parent_0820", latest_step=25000)
    for i in range(1, 10):
        with open(os.path.join(parent, "snapshots",
                               f"snapshot_{i * 1000:012d}.zip"), "wb") as fh:
            fh.write(b"s" * 1000)
    make_run(str(models), "ai_v9_89_child_0821", latest_step=25000, meta={
        "lineage": {"fork_parent": {"run_name": "ai_v9_88_parent_0820"}}})
    repo = tmp_path / "repo"
    repo.mkdir()
    _stamp_old(str(models))

    c = census(str(models), str(repo))
    sp = c["runs"]["ai_v9_88_parent_0820"]["plan"]["snapshots"]
    assert sp["action"] == "keep"
    assert sp["delete"] == [], "a kept pool contributes nothing to the plan"
    # 10 snapshots (9 + the fixture's own): every 4th + the newest => 4 kept, 6 dropped
    assert sp["n_snapshots"] == 10
    assert sp["n_thin_would_drop"] == 6
    assert sp["thin_proposal_bytes"] > 0
    assert c["tiers"]["snapshots"]["gb_thin_proposal"] >= 0.0


def test_a_tier_0_run_is_exempt_from_the_snapshots_rule(tiered_archive, monkeypatch):
    models, repo = tiered_archive
    monkeypatch.setattr(ag, "live_run_dirs",
                        lambda d: {"ai_v9_70_closed_0820": "…main.launcher…"})
    c = census(models, repo)
    r = c["runs"]["ai_v9_70_closed_0820"]
    assert r["tier"] == 0
    assert r["plan"].get("snapshots") is None
    assert r["plan"]["delete"] == []


# ---------------------------------------------------------------- tier 3 ----


def test_tier3_keeps_the_ends_and_the_pin_but_takes_no_stride(tiered_archive):
    models, repo = tiered_archive
    c = census(models, repo)
    keep = c["runs"]["ai_v8_70_closed_0725"]["plan"]["keep"]
    steps = sorted({ag.checkpoint_step(n) for n in keep})
    assert steps == [1000, 25000], "tier 3 is first + last (+ pin) only"
    assert not any("every-" in v for v in keep.values())
    # …and the gentler tier on the SAME tree does take the stride
    v9 = c["runs"]["ai_v9_70_closed_0820"]["plan"]["keep"]
    assert any("every-10th" in v for v in v9.values())


def test_every_zero_means_no_stride_not_a_huge_one():
    names = [f"checkpoint_{(i + 1) * 1000}_steps.zip" for i in range(25)]
    keep, delete = ag.plan_checkpoints(names, every=tiers.TIER3_CHECKPOINT_EVERY)
    assert sorted(ag.checkpoint_step(n) for n in keep) == [1000, 25000]
    assert len(delete) == 23


# ---------------------------------------------------------------- tier 4 ----


def test_tier4_keeps_the_record_and_the_resolver_s_pick(tiered_archive):
    models, repo = tiered_archive
    c = census(models, repo)
    r = c["runs"]["ai_v7_70_closed_0703"]
    keep, delete = r["plan"]["keep"], r["plan"]["delete"]

    res = r["plan"]["resolved"]
    assert res["ok"] and res["rung"], "the RUNG must be recorded, not just the file"
    assert res["rel"] in keep

    for must in ("metadata.json", "model_config.json", "eval_results.jsonl",
                 "tb", "snapshot_ladder", "command.txt"):
        assert must in keep, must
        assert must not in delete
    for gone in ("best_model", "snapshots", "eval_traces", "launcher_child.log"):
        assert gone in delete, gone
    # the weights go, the per-checkpoint RECORD stays
    assert "checkpoint_1000_steps.zip" in delete
    assert keep["checkpoint_1000_steps.json"].startswith("checkpoint sidecar")


def test_tier4_reaches_the_LEGACY_root_checkpoints_the_standing_policy_cannot(
        tiered_archive):
    """20.8 GB across 13 pre-v8 runs lives at the run ROOT, outside
    ``checkpoints/`` — more than the standing policy frees archive-wide."""
    models, repo = tiered_archive
    standing = ag.build_census(models, repo, recent_days=0)
    tiered = census(models, repo)
    run = "ai_v7_70_closed_0703"
    assert standing["runs"][run]["plan"]["delete"] == [] or all(
        p.startswith(("checkpoints/", "eval_traces/"))
        for p in standing["runs"][run]["plan"]["delete"])
    assert any(p == "checkpoint_2000_steps.zip"
               for p in tiered["runs"][run]["plan"]["delete"])
    assert tiered["runs"][run]["plan"]["bytes_freed"] > \
        standing["runs"][run]["plan"]["bytes_freed"]


def test_tier4_refuses_a_run_whose_final_model_does_not_resolve(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    run = make_run(str(models), "ai_v7_90_broken_0703", n_ckpt=0, latest_step=None)
    for junk in ("best_model/best_model.zip",):
        os.remove(os.path.join(run, junk))
    repo = tmp_path / "repo"
    repo.mkdir()
    _stamp_old(str(models))

    c = census(str(models), str(repo))
    r = c["runs"]["ai_v7_90_broken_0703"]
    assert r["tier"] == 4
    assert r["plan"]["delete"] == [], "an unresolvable run is left ENTIRELY alone"
    assert "REFUSED" in r["plan"]["skipped"]
    assert c["tiers"]["refusals"][0]["run"] == "ai_v7_90_broken_0703"
    assert c["totals"]["n_tier4_refused"] == 1


def test_tier4_keeps_the_latest_txt_target_so_the_pin_still_resolves(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    run = make_run(str(models), "ai_v6_91_old_0613", n_ckpt=6, latest_step=3000)
    with open(os.path.join(run, "final_model.zip"), "wb") as fh:
        fh.write(b"f" * 9000)
    repo = tmp_path / "repo"
    repo.mkdir()
    _stamp_old(str(models))
    c = census(str(models), str(repo))
    keep = c["runs"]["ai_v6_91_old_0613"]["plan"]["keep"]
    assert "checkpoints/checkpoint_3000_steps.zip" in keep
    assert "checkpoints/checkpoint_3000_steps.zip" not in \
        c["runs"]["ai_v6_91_old_0613"]["plan"]["delete"]


@pytest.mark.parametrize("rel", [
    "metadata.json", "model_config.json", "latest.txt", "eval_results.jsonl",
    "tb", "tb/events.out.tfevents.1", "snapshot_ladder",
])
def test_assert_safe_tiered_refuses_a_plan_that_names_a_kept_path(rel):
    with pytest.raises(tiers.TieredRefusal):
        tiers.assert_safe_tiered("/run", [rel], {rel})


def test_assert_safe_tiered_refuses_a_plan_that_would_swallow_a_kept_path():
    with pytest.raises(tiers.TieredRefusal):
        tiers.assert_safe_tiered("/run", ["checkpoints"],
                                 {"checkpoints/checkpoint_1_steps.zip"})


def test_assert_safe_tiered_refuses_an_escape(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(tiers.TieredRefusal):
        tiers.assert_safe_tiered(str(run), ["../elsewhere"], set())


# ------------------------------------------------------------ review holds ----


@pytest.mark.parametrize("run", sorted(tiers.REVIEW_HOLDS))
def test_a_review_hold_suppresses_the_plan_entirely(tmp_path, run):
    """The three runs a human read and HELD.  A hold is not a softer tier — it is
    no plan at all, because what is uncertain is which files the banked claim
    rests on."""
    models = tmp_path / "models"
    models.mkdir()
    make_run(str(models), run, latest_step=25000)
    repo = tmp_path / "repo"
    repo.mkdir()
    _stamp_old(str(models))
    c = census(str(models), str(repo))
    r = c["runs"][run]
    assert r["tier"] == 1
    assert r["plan"]["delete"] == []
    assert "REVIEW HOLD" in r["plan"]["skipped"]
    assert r["review_hold"]


def test_the_two_RELEASED_review_runs_are_not_held(tmp_path):
    """``ai_v5_11`` and ``ai_v5_12`` were read and RELEASED — and tier 4 keeps the
    two things their ledger lines rest on anyway (`tb/` and the sidecars)."""
    for run in ("ai_v5_11_tail2_53m_0611", "ai_v5_12_bias_05_N_0612"):
        assert run not in tiers.REVIEW_HOLDS
    models = tmp_path / "models"
    models.mkdir()
    r5 = make_run(str(models), "ai_v5_11_tail2_53m_0611", n_ckpt=0, latest_step=None)
    _add_root_checkpoints(r5, [1000, 2000])
    with open(os.path.join(r5, "final_model.zip"), "wb") as fh:
        fh.write(b"f" * 9000)
    repo = tmp_path / "repo"
    repo.mkdir()
    _stamp_old(str(models))
    c = census(str(models), str(repo))
    r = c["runs"]["ai_v5_11_tail2_53m_0611"]
    assert r["tier"] == 4 and r["plan"]["delete"]
    assert "tb" in r["plan"]["keep"]
    assert "checkpoint_1000_steps.json" in r["plan"]["keep"]


# ------------------------------------------------ the shared safety devices ----


def test_a_symlinked_run_dir_is_still_held_out_under_the_tiered_policy(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    real = make_run(str(elsewhere), "real_0101", latest_step=25000)
    models = tmp_path / "models"
    models.mkdir()
    os.symlink(real, str(models / "ai_v7_92_linked_0703"))
    repo = tmp_path / "repo"
    repo.mkdir()
    _stamp_old(str(elsewhere))

    c = census(str(models), str(repo))
    r = c["runs"]["ai_v7_92_linked_0703"]
    assert r["tier"] == 0 and r["plan"]["delete"] == []
    assert any("SYMLINK" in s for s in r["tier_reasons"])
    assert c["totals"]["n_symlinked_runs"] == 1
    opted = census(str(models), str(repo), follow_symlinked_runs=True)
    assert opted["runs"]["ai_v7_92_linked_0703"]["tier"] == 4


def test_the_named_path_veto_still_fires_under_the_tiered_policy(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    run = make_run(str(models), "ai_v7_93_old_0703", n_ckpt=0, latest_step=None)
    _add_root_checkpoints(run, [1000, 2000])
    with open(os.path.join(run, "final_model.zip"), "wb") as fh:
        fh.write(b"f" * 9000)
    repo = tmp_path / "repo"
    (repo / "designs").mkdir(parents=True)
    _stamp_old(str(models))

    assert census(str(models), str(repo))["runs"]["ai_v7_93_old_0703"]["plan"]["delete"]

    (repo / "designs" / "note.md").write_text(
        "decoded models/ai_v7_93_old_0703/checkpoint_1000_steps.zip\n")
    monkey = ag.committed_files
    try:
        ag.committed_files = lambda root: ["designs/note.md"]
        c = census(str(models), str(repo))
    finally:
        ag.committed_files = monkey
    r = c["runs"]["ai_v7_93_old_0703"]
    assert r["plan"]["delete"] == []
    assert c["totals"]["n_excluded_by_named_file"] == 1


def test_a_tiered_dry_run_writes_no_change_to_the_tree(tiered_archive):
    models, repo = tiered_archive
    before = {p: os.path.getsize(p)
              for p in (os.path.join(dp, f)
                        for dp, _d, fs in os.walk(models) for f in fs)}
    census(models, repo)
    after = {p: os.path.getsize(p)
             for p in (os.path.join(dp, f)
                       for dp, _d, fs in os.walk(models) for f in fs)}
    assert before == after


def test_the_standing_policy_is_unchanged_by_the_tiered_one(tiered_archive):
    """The two policies must differ only where they are MEANT to."""
    models, repo = tiered_archive
    s = ag.build_census(models, repo, recent_days=0)
    assert s.get("policy_name") is None      # standing carries no policy label
    assert "tiers" not in s
    for r in s["runs"].values():
        for rel in r["plan"]["delete"]:
            assert rel.split(os.sep)[0] in ag.TOUCHABLE_SUBDIRS
        assert "snapshots" not in r["plan"]


# ----------------------------------------------------------------- --apply ----


def test_apply_deletes_exactly_the_tiered_plan_and_nothing_else(tiered_archive):
    """The ONLY place --apply runs: a synthetic tree under tmp_path."""
    models, repo = tiered_archive
    c = census(models, repo)
    planned = {os.path.join(c["runs"][n]["run_dir"], rel)
               for n, r in c["runs"].items() for rel in r["plan"]["delete"]}
    assert planned

    keeps = {n: set(c["runs"][n]["plan"].get("keep_rel") or ()) for n in c["runs"]}
    before = {os.path.join(dp, f) for dp, _d, fs in os.walk(models) for f in fs}
    ag.apply_plan(c)
    after = {os.path.join(dp, f) for dp, _d, fs in os.walk(models) for f in fs}

    for g in before - after:
        assert any(g == p or g.startswith(p + os.sep) for p in planned), g
    for p in planned:
        assert not os.path.exists(p)
    assert c["apply_result"]["errors"] == []

    # every keep-list entry, in every tier, survived
    for n, keep in keeps.items():
        for rel in keep:
            assert os.path.exists(os.path.join(models, n, rel)), f"{n}/{rel}"
    # and the tier-4 run is still resolvable afterwards
    t4 = os.path.join(models, "ai_v7_70_closed_0703")
    assert tiers.resolve_final_model(t4)["ok"]


def test_apply_is_off_by_default_in_the_tiered_cli(tiered_archive, tmp_path, capsys):
    models, repo = tiered_archive
    before = {os.path.join(dp, f) for dp, _d, fs in os.walk(models) for f in fs}
    rc = ag.main(["--policy", "tiered", "--models-dir", models, "--repo-root", repo,
                  "--recent-days", "0", "--out-prefix", str(tmp_path / "rep")])
    assert rc == 0
    assert before == {os.path.join(dp, f) for dp, _d, fs in os.walk(models) for f in fs}
    out = capsys.readouterr().out
    assert "policy                  tiered" in out
    assert "NOTHING WAS DELETED IN THIS PASS" in out
    assert "--policy tiered --apply" in out
    md = open(str(tmp_path / "rep.md")).read()
    assert "The TIERED policy" in md
    assert "The snapshots rule" in md
    assert "wasn't a 'novel' outcome" in md, \
        "the owner's reason must be quoted verbatim"
    assert "--policy tiered --apply" in md


def test_the_default_policy_is_still_standing(tiered_archive, tmp_path, capsys):
    models, repo = tiered_archive
    ag.main(["--models-dir", models, "--repo-root", repo, "--recent-days", "0",
             "--out-prefix", str(tmp_path / "rep")])
    out = capsys.readouterr().out
    assert "policy                  standing" in out
    assert "per tier:" not in out
