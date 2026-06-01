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
        2000000: {"Heuristic": [("win", 1), ("loss", 2)], "Random": [("win", 1)]},
        1000000: {"SetupSweep": [("loss", 3), ("win", 1)]},
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
    # opponents in roster order: Random before Heuristic
    step2 = next(s for s in tree.steps if s.step == 2000000)
    assert [o.name for o in step2.opponents] == ["Random", "Heuristic"]
    # battles sorted by outcome (win first) then index
    heur = next(o for o in step2.opponents if o.name == "Heuristic")
    assert [(b.outcome, b.index) for b in heur.battles] == [("win", 1), ("loss", 2)]
    assert heur.battles[0].npz_path is not None


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
    one = str(et / "step_1000000" / "SetupSweep" / "win_001_summary.json")
    tree = build_trace_tree(one)
    battles = tree.all_battles()
    assert len(battles) == 1
    assert battles[0].opponent == "SetupSweep" and battles[0].step == 1000000
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
    # best_model wins over latest
    _touch(str(run / "best_model" / "best_model.zip"), "")
    assert resolve_checkpoint(str(run)).endswith("best_model.zip")
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

def _trace_at(run, step, opp="Random"):
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
