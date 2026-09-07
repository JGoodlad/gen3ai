"""Pure filesystem tests for trace discovery (no torch)."""

import json
import os

import pytest

from main.prober.discovery import (
    build_trace_tree,
    list_checkpoints,
    load_model_config,
    read_eval_manifest,
    resolve_checkpoint,
    resolve_model_for_step,
)


def _touch(path: str, content: str = "{}") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _make_run(tmp_path, with_npz=True):
    """A run dir with eval_traces across 2 steps / a few opponents."""
    run = tmp_path / "run_x"
    et = run / "eval_traces"
    # step 2000000 first written, but discovery must sort steps ascending
    layout = {
        2000000: {"heuristic": [("win", 1), ("loss", 2)], "random": [("win", 1)]},
        1000000: {"setup_sweep": [("loss", 3), ("win", 1)]},
    }
    for step, opps in layout.items():
        for opp, battles in opps.items():
            for outcome, idx in battles:
                prefix = et / f"step_{step}" / opp / f"{outcome}_{idx:03d}"
                _touch(str(prefix) + "_summary.json")
                if with_npz:
                    _touch(str(prefix) + "_states.npz", "")
    return run, et


def test_run_dir_groups_and_sorts(tmp_path):
    run, _ = _make_run(tmp_path)
    tree = build_trace_tree(str(run))
    assert not tree.is_empty
    assert tree.run_dir == os.path.abspath(str(run))
    # steps ascending
    assert [s.step for s in tree.steps] == [1000000, 2000000]
    # opponents in roster order: random before heuristic
    step2 = next(s for s in tree.steps if s.step == 2000000)
    assert [o.name for o in step2.opponents] == ["random", "heuristic"]
    # battles sorted by outcome (win first) then index
    heur = next(o for o in step2.opponents if o.name == "heuristic")
    assert [(b.outcome, b.index) for b in heur.battles] == [("win", 1), ("loss", 2)]
    assert heur.battles[0].npz_path is not None


def test_sharded_filenames_parse_outcome_and_stay_distinct(tmp_path):
    """The work-stealing eval names traces `<outcome>_s<shard>_<idx>`; discovery
    must parse the outcome (not fall to '?') and keep two shards' same-idx traces
    distinct (unique index → unique short_id / sort key)."""
    et = tmp_path / "run" / "eval_traces"
    for fn in ("loss_s1_005", "loss_s3_005", "win_s0_002", "loss_012"):
        _touch(str(et / "step_30000000" / "aggressive" / f"{fn}_summary.json"))
    battles = build_trace_tree(str(et)).all_battles()
    assert {b.outcome for b in battles} == {"loss", "win"}        # parsed, NOT "?"
    losses = sorted(b.index for b in battles if b.outcome == "loss")
    assert losses == [12, 1005, 3005]   # un-sharded idx unchanged (shard 0); shard folded into index
    assert len(set(losses)) == 3                                  # the two same-idx shards stay distinct


def test_missing_npz_tolerated(tmp_path):
    run, _ = _make_run(tmp_path, with_npz=False)
    tree = build_trace_tree(str(run))
    assert all(b.npz_path is None for b in tree.all_battles())


def test_eval_traces_dir_input(tmp_path):
    run, et = _make_run(tmp_path)
    tree = build_trace_tree(str(et))
    assert not tree.is_empty
    assert tree.run_dir == os.path.abspath(str(run))


def test_single_summary_input(tmp_path):
    run, et = _make_run(tmp_path)
    one = str(et / "step_1000000" / "setup_sweep" / "win_001_summary.json")
    tree = build_trace_tree(one)
    battles = tree.all_battles()
    assert len(battles) == 1
    assert battles[0].opponent == "setup_sweep" and battles[0].step == 1000000
    assert tree.run_dir == os.path.abspath(str(run))


def test_resolve_checkpoint_precedence(tmp_path):
    run = tmp_path / "run_y"
    run.mkdir()
    # nothing yet → error
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(run))
    # latest.txt path
    _touch(str(run / "checkpoint_4000000_steps.zip"), "")
    _touch(str(run / "latest.txt"), "checkpoint_4000000_steps.zip")
    assert resolve_checkpoint(str(run)).endswith("checkpoint_4000000_steps.zip")
    # 🚨 THE LAST SNAPSHOT WINS — best_model does NOT (`gen3_last_snapshot_resolution_v1`).
    # This assertion is INVERTED from what it said until 2026-09-06, and deliberately: the prober
    # kept a second opinion about what "this run's model" means while every training-side ref went
    # through `resolve_model_ref`, whose rungs put `best_model` LAST because it is the BOT-WIN-RATE
    # export, not the run's latest state. Measured over five archived runs, ALL FIVE disagreed —
    # the prober loading a bot-selected export while its own tier label read "most recent".
    _touch(str(run / "best_model" / "best_model.zip"), "")
    assert resolve_checkpoint(str(run)).endswith("checkpoint_4000000_steps.zip"), \
        "best_model must NOT outrank the run's last snapshot"
    # explicit override wins over everything
    override = str(run / "checkpoint_4000000_steps.zip")
    assert resolve_checkpoint(str(run), override=override) == override
    # bad override raises
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(run), override=str(run / "nope.zip"))


def test_load_model_config(tmp_path):
    run = tmp_path / "run_z"
    run.mkdir()
    assert load_model_config(str(run)) is None        # absent
    assert load_model_config(None) is None
    _touch(str(run / "model_config.json"), json.dumps({"arch": "gen3_x"}))
    assert load_model_config(str(run)) == {"arch": "gen3_x"}


# --- model-resolution ladder -------------------------------------------------

def _trace_at(run, step, opp="random"):
    p = run / "eval_traces" / f"step_{step}" / opp / "win_001_summary.json"
    _touch(str(p))


def _manifest(run, step, snapshot=None, git="abc123"):
    m = {"step": step, "git_hash": git, "arch_signature": "gen3_x", "snapshot": snapshot}
    _touch(str(run / "eval_traces" / f"step_{step}" / "eval_manifest.json"), json.dumps(m))


def test_manifest_and_checkpoint_listing(tmp_path):
    run = tmp_path / "r"
    _trace_at(run, 2000000)
    _manifest(run, 2000000, git="deadbeef")
    _touch(str(run / "checkpoint_3200000_steps.zip"), "")
    _touch(str(run / "checkpoint_6400000_steps.zip"), "")
    _touch(str(run / "final_model.zip"), "")  # no step → excluded
    assert read_eval_manifest(str(run), 2000000)["git_hash"] == "deadbeef"
    assert list_checkpoints(str(run)) == [
        (3200000, str(run / "checkpoint_3200000_steps.zip")),
        (6400000, str(run / "checkpoint_6400000_steps.zip")),
    ]


def test_list_checkpoints_finds_new_subdir_and_dedupes(tmp_path):
    run = tmp_path / "r"
    # Current layout: checkpoints/ subdir.
    _touch(str(run / "checkpoints" / "checkpoint_3200000_steps.zip"), "")
    # Legacy root checkpoint at a different step.
    _touch(str(run / "checkpoint_1000000_steps.zip"), "")
    # A copy-backported step present in BOTH locations → checkpoints/ wins, listed once.
    _touch(str(run / "checkpoint_6400000_steps.zip"), "")
    _touch(str(run / "checkpoints" / "checkpoint_6400000_steps.zip"), "")
    assert list_checkpoints(str(run)) == [
        (1000000, str(run / "checkpoint_1000000_steps.zip")),
        (3200000, str(run / "checkpoints" / "checkpoint_3200000_steps.zip")),
        (6400000, str(run / "checkpoints" / "checkpoint_6400000_steps.zip")),
    ]


def test_ladder_nearest_resolves_checkpoint_in_subdir(tmp_path):
    # The nearest-checkpoint ladder must work when checkpoints live under checkpoints/.
    run = tmp_path / "r"
    _trace_at(run, 2000000)
    _manifest(run, 2000000)  # no snapshot retained → falls to nearest
    _touch(str(run / "checkpoints" / "checkpoint_3200000_steps.zip"), "")
    tree = build_trace_tree(str(run))
    c = resolve_model_for_step(tree, 2000000)
    assert c.tier == "nearest" and c.path.endswith("checkpoints/checkpoint_3200000_steps.zip")


def test_ladder_exact_when_snapshot_retained(tmp_path):
    run = tmp_path / "r"
    _trace_at(run, 2000000)
    _manifest(run, 2000000, snapshot="snapshot.zip")
    _touch(str(run / "eval_traces" / "step_2000000" / "snapshot.zip"), "")
    _touch(str(run / "checkpoint_6400000_steps.zip"), "")
    tree = build_trace_tree(str(run))
    c = resolve_model_for_step(tree, 2000000)
    assert c.tier == "exact" and c.path.endswith("step_2000000/snapshot.zip")
    assert c.is_exact and c.manifest["snapshot"] == "snapshot.zip"


def test_ladder_nearest_when_no_snapshot(tmp_path):
    run = tmp_path / "r"
    for s in (1000000, 5000000):
        _trace_at(run, s)
        _manifest(run, s)  # no snapshot retained
    _touch(str(run / "checkpoint_3200000_steps.zip"), "")
    _touch(str(run / "checkpoint_6400000_steps.zip"), "")
    tree = build_trace_tree(str(run))
    # step 5M → nearest is 6.4M (Δ1.4M) over 3.2M (Δ1.8M)
    c = resolve_model_for_step(tree, 5000000)
    assert c.tier == "nearest" and c.path.endswith("checkpoint_6400000_steps.zip")
    assert "Δ1,400,000" in c.detail


def test_ladder_stale_snapshot_pointer_falls_through(tmp_path):
    run = tmp_path / "r"
    _trace_at(run, 2000000)
    _manifest(run, 2000000, snapshot="snapshot.zip")  # claims a snapshot…
    # …but the file was pruned — must NOT pick exact
    _touch(str(run / "checkpoint_3200000_steps.zip"), "")
    tree = build_trace_tree(str(run))
    c = resolve_model_for_step(tree, 2000000)
    assert c.tier == "nearest"


def test_ladder_recent_and_override_and_forced_tier(tmp_path):
    run = tmp_path / "r"
    _trace_at(run, 2000000)
    _manifest(run, 2000000, snapshot="snapshot.zip")
    _touch(str(run / "eval_traces" / "step_2000000" / "snapshot.zip"), "")
    _touch(str(run / "best_model" / "best_model.zip"), "")
    _touch(str(run / "checkpoint_3200000_steps.zip"), "")
    tree = build_trace_tree(str(run))
    # forced "recent" skips the exact snapshot
    assert resolve_model_for_step(tree, 2000000, tier="recent").tier == "recent"
    # forced "nearest" skips exact too
    assert resolve_model_for_step(tree, 2000000, tier="nearest").tier == "nearest"
    # override beats everything
    c = resolve_model_for_step(tree, 2000000, override="/x/y.zip")
    assert c.tier == "override" and c.path == "/x/y.zip"


def test_ladder_no_models_at_all(tmp_path):
    run = tmp_path / "r"
    _trace_at(run, 2000000)
    _manifest(run, 2000000)
    tree = build_trace_tree(str(run))
    c = resolve_model_for_step(tree, 2000000)
    assert c.tier == "none" and c.path is None


def test_the_resolution_rung_is_REPORTED_not_just_the_file(tmp_path):
    """A file without its rung cannot be read: `best_model` means "chosen by bot win rate" and a
    checkpoint means "the run's latest state", and those are different claims about what the probe
    is even showing. `resolve_model_ref` returns the provenance; the prober must carry it."""
    from main.prober.discovery import resolve_checkpoint_with_rung
    run = tmp_path / "run_rung"
    run.mkdir()
    _touch(str(run / "checkpoint_4000000_steps.zip"), "")
    _touch(str(run / "latest.txt"), "checkpoint_4000000_steps.zip")
    path, rung = resolve_checkpoint_with_rung(str(run))
    assert path.endswith("checkpoint_4000000_steps.zip")
    assert rung and rung != "override"
    # an override says so, rather than borrowing whatever rung the run would have used
    override = str(run / "checkpoint_4000000_steps.zip")
    assert resolve_checkpoint_with_rung(str(run), override=override) == (override, "override")


def test_a_run_with_ONLY_best_model_still_resolves_and_is_labelled(tmp_path):
    """The fallback must still WORK — demoting best_model is not removing it. A run whose only
    artifact is the bot-selected export is probeable; it just has to say that is what it is."""
    from main.prober.discovery import resolve_checkpoint_with_rung
    run = tmp_path / "run_bm"
    (run / "best_model").mkdir(parents=True)
    _touch(str(run / "best_model" / "best_model.zip"), "")
    path, rung = resolve_checkpoint_with_rung(str(run))
    assert path.endswith("best_model.zip")
    assert "best_model" in rung, f"the rung must name the fallback, got {rung!r}"
