"""`best_model/best_model.zip` and the self-play pool's `snapshots/snapshot_<N>.zip` are HARD LINKS to the
checkpoint saved at the same step when byte-identical, not 62 MB duplicates (`agents.training.weights_link`).

THE WASTE. After the eval snapshot (`eval_snapshot_dedup_test.py`), two more copies of the same weights were
written at a coincident step: the best-model export (`shutil.copy2` of the cycle's frozen snapshot, in BOTH eval
callbacks) and the pool entry (`SnapshotPool.add_from_path`, a copy on every promotion or seeding). Pinned here:

* the link is made when the same-step checkpoint is byte-identical and on the same filesystem, a copy otherwise
  (no checkpoint, other bytes, another size, no link possible) - for the helper, both callbacks' best-save and
  the pool;
* 🚨 a re-save over an existing link NEVER writes through it (a plain `copy2` onto a linked `best_model.zip`
  would truncate the CHECKPOINT through the shared inode);
* 🚨 the writers of a weights name (`model.save`: the periodic checkpointer, the pool's live `add`) save through
  `utils.atomic_io.atomic_save_model` (a temp file, then a replace), so a re-reached step cannot rewrite - or, torn,
  corrupt - the other names of the inode;
* the link survives the checkpoint's deletion (retention) and the pool's eviction leaves the checkpoint alone;
* every reader keeps working: a hard link is an ordinary file path.
"""
import errno
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.training.eval_callback import PerOpponentEvalCallback
from agents.training.eval_collect import checkpoint_at_step, store_best_model
from agents.training.selfplay_callback import SelfPlayCallback
from agents.training.snapshot_pool import SnapshotPool
from agents.training.weights_link import link_or_copy, sha256_file

WEIGHTS = b"PK-weights-" + bytes(range(256)) * 40          # stands for a ~62 MB zip; content is what is compared
OTHER = b"PK-OTHER---" + bytes(range(256)) * 40            # same length, different bytes
STEP = 2_000_128


def _run(tmp_path, *, checkpoint=WEIGHTS, step=STEP, name="rb_run"):
    """A run dir with the cycle's transient snapshot and (optionally) the periodic checkpoint at `step`."""
    run = tmp_path / name
    transient = run / ".eval_runs" / f"step_{step}" / "snapshot.zip"
    transient.parent.mkdir(parents=True)
    transient.write_bytes(WEIGHTS)
    ckpt = run / "checkpoints" / f"checkpoint_{step}_steps.zip"
    if checkpoint is not None:
        ckpt.parent.mkdir()
        ckpt.write_bytes(checkpoint)
    return run, transient, ckpt


# ---------------------------------------------------------------------------------------
# the helper
# ---------------------------------------------------------------------------------------

def test_an_identical_twin_makes_dst_a_second_name_for_its_inode(tmp_path):
    _run_dir, transient, ckpt = _run(tmp_path)
    dst = tmp_path / "out" / "x.zip"
    dst.parent.mkdir()
    mode, digest, note = link_or_copy(str(transient), str(dst), str(ckpt))
    assert mode == "hardlink" and digest == sha256_file(str(transient)) and "hard link to checkpoints/" in note
    assert os.path.samefile(dst, ckpt) and os.stat(dst).st_nlink == 2
    assert dst.read_bytes() == WEIGHTS


@pytest.mark.parametrize("twin_bytes, why", [(None, "cadences"), (OTHER, "not byte-identical"),
                                             (WEIGHTS + b"x", "not the same size")])
def test_no_twin_or_a_different_twin_is_a_copy_that_says_why(tmp_path, twin_bytes, why):
    _run_dir, transient, ckpt = _run(tmp_path, checkpoint=twin_bytes)
    dst = tmp_path / "out" / "x.zip"
    dst.parent.mkdir()
    mode, _digest, note = link_or_copy(str(transient), str(dst), str(ckpt) if twin_bytes is not None else None)
    assert mode == "copy" and why in note
    assert dst.read_bytes() == WEIGHTS and os.stat(dst).st_nlink == 1
    if twin_bytes is not None:
        assert not os.path.samefile(dst, ckpt) and ckpt.read_bytes() == twin_bytes, "the twin is untouched"


def test_when_a_link_cannot_be_made_it_copies_and_says_why(tmp_path, monkeypatch):
    _run_dir, transient, ckpt = _run(tmp_path)
    dst = tmp_path / "out" / "x.zip"
    dst.parent.mkdir()

    def _no_link(src, dst_, *a, **k):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", _no_link)
    mode, _digest, note = link_or_copy(str(transient), str(dst), str(ckpt))
    assert mode == "copy" and "linking failed" in note and dst.read_bytes() == WEIGHTS
    assert sorted(os.listdir(dst.parent)) == ["x.zip"], "a failed link strands no temp"


def test_a_resave_over_an_existing_link_never_writes_through_it(tmp_path):
    """THE hazard. `dst` is already a second name for the checkpoint; the next save carries different bytes (a
    later best). A plain `copy2(src, dst)` would open the shared inode for writing and overwrite the CHECKPOINT."""
    _run_dir, transient, ckpt = _run(tmp_path)
    dst = tmp_path / "out" / "x.zip"
    dst.parent.mkdir()
    link_or_copy(str(transient), str(dst), str(ckpt))
    assert os.path.samefile(dst, ckpt)
    later = tmp_path / "later.zip"
    later.write_bytes(OTHER)
    mode, _digest, _note = link_or_copy(str(later), str(dst), None)
    assert mode == "copy" and dst.read_bytes() == OTHER
    assert ckpt.read_bytes() == WEIGHTS, "the checkpoint must not follow dst"
    assert not os.path.samefile(dst, ckpt)


def test_a_relink_over_the_same_inode_strands_no_tmp(tmp_path):
    _run_dir, transient, ckpt = _run(tmp_path)
    dst = tmp_path / "out" / "x.zip"
    dst.parent.mkdir()
    link_or_copy(str(transient), str(dst), str(ckpt))
    mode, _digest, _note = link_or_copy(str(transient), str(dst), str(ckpt))
    assert mode == "hardlink" and sorted(os.listdir(dst.parent)) == ["x.zip"], "a temp name was stranded"
    assert os.stat(ckpt).st_nlink == 2, "two names, not three"


# ---------------------------------------------------------------------------------------
# best_model/best_model.zip - both eval callbacks
# ---------------------------------------------------------------------------------------

def test_store_best_model_links_to_the_same_step_checkpoint(tmp_path, capsys):
    run, transient, ckpt = _run(tmp_path)
    dst = tmp_path / "best" / "best_model.zip"
    dst.parent.mkdir()
    how = store_best_model(str(run), STEP, str(transient), str(dst))
    assert os.path.samefile(dst, ckpt) and "hard link to checkpoints/" in how and "stored once" in how
    assert checkpoint_at_step(str(run), STEP) == str(ckpt)
    assert "COPIED" not in capsys.readouterr().out


def test_store_best_model_copies_without_a_warning_when_cadences_do_not_coincide(tmp_path, capsys):
    run, transient, _ckpt = _run(tmp_path, checkpoint=None)
    dst = tmp_path / "best" / "best_model.zip"
    dst.parent.mkdir()
    assert store_best_model(str(run), STEP, str(transient), str(dst)) == ""
    assert dst.read_bytes() == WEIGHTS and os.stat(dst).st_nlink == 1
    assert "COPIED" not in capsys.readouterr().out


def test_store_best_model_warns_about_a_copy_made_although_a_checkpoint_exists(tmp_path, capsys):
    run, transient, ckpt = _run(tmp_path, checkpoint=OTHER)
    dst = tmp_path / "best" / "best_model.zip"
    dst.parent.mkdir()
    assert store_best_model(str(run), STEP, str(transient), str(dst)) == ""
    assert dst.read_bytes() == WEIGHTS and ckpt.read_bytes() == OTHER
    out = capsys.readouterr().out
    assert "COPIED although a checkpoint exists" in out and "not byte-identical" in out


def test_the_per_opponent_eval_callback_links_its_best_model(tmp_path, capsys):
    run, transient, ckpt = _run(tmp_path)
    best = tmp_path / "best"
    best.mkdir()
    cb = PerOpponentEvalCallback(model_dir=str(run), best_model_save_path=str(best))
    cb.model = MagicMock()
    cb._maybe_save_best(STEP, {"snapshot": str(transient)}, {"heuristic": 0.9, "staller": 0.8})
    dst = best / "best_model.zip"
    assert os.path.samefile(dst, ckpt) and dst.read_bytes() == WEIGHTS
    assert "hard link to checkpoints/" in capsys.readouterr().out
    # the cycle's transient snapshot is dropped by `_cleanup`; the best model (a second name) must survive it
    transient.unlink()
    assert dst.read_bytes() == WEIGHTS and ckpt.read_bytes() == WEIGHTS


def test_the_selfplay_callback_links_its_best_model(tmp_path):
    run, transient, ckpt = _run(tmp_path)
    best = tmp_path / "best"
    best.mkdir()
    cb = SelfPlayCallback(pool=MagicMock(), model_dir=str(run), best_model_save_path=str(best))
    cb.model = MagicMock()
    cb._maybe_save_best(STEP, {"snapshot": str(transient)}, 0.85)
    dst = best / "best_model.zip"
    assert os.path.samefile(dst, ckpt) and dst.read_bytes() == WEIGHTS


def test_a_later_best_never_writes_through_the_earlier_link_into_the_checkpoint(tmp_path):
    """Two bests in one run: the first at step A (linked to checkpoint A), the second at step B. The second
    export must replace `best_model.zip` atomically - checkpoint A keeps its bytes."""
    run, transient_a, ckpt_a = _run(tmp_path, step=STEP)
    best = tmp_path / "best"
    best.mkdir()
    cb = PerOpponentEvalCallback(model_dir=str(run), best_model_save_path=str(best))
    cb.model = MagicMock()
    cb._maybe_save_best(STEP, {"snapshot": str(transient_a)}, {"heuristic": 0.7})
    assert os.path.samefile(best / "best_model.zip", ckpt_a)
    step_b = STEP + 1_000_000
    transient_b = run / ".eval_runs" / f"step_{step_b}" / "snapshot.zip"
    transient_b.parent.mkdir(parents=True)
    transient_b.write_bytes(OTHER)                                 # no checkpoint at step B
    cb._maybe_save_best(step_b, {"snapshot": str(transient_b)}, {"heuristic": 0.9})
    assert (best / "best_model.zip").read_bytes() == OTHER
    assert ckpt_a.read_bytes() == WEIGHTS, "checkpoint A followed best_model.zip"


# ---------------------------------------------------------------------------------------
# SnapshotPool.add_from_path
# ---------------------------------------------------------------------------------------

def _pool(tmp_path) -> SnapshotPool:
    version = MagicMock()
    version.to_json.return_value = "{}"
    with patch("agents.training.snapshot_pool.load_opponent_snapshot"):
        return SnapshotPool(pool_dir=tmp_path / "snapshots", current_version=version)


def test_add_from_path_links_to_the_same_step_checkpoint(tmp_path):
    run, transient, ckpt = _run(tmp_path)
    pool = _pool(run)
    entry = pool.add_from_path(transient, STEP, twin=ckpt)
    assert os.path.samefile(entry.path, ckpt) and entry.path.read_bytes() == WEIGHTS
    assert os.stat(ckpt).st_nlink == 2
    assert transient.exists(), "the source is not moved (the trainer cleans up the scratch itself)"


def test_add_from_path_without_a_twin_is_the_plain_copy_it_was(tmp_path):
    run, transient, _ckpt = _run(tmp_path, checkpoint=None)
    entry = _pool(run).add_from_path(transient, STEP)
    assert entry.path.read_bytes() == WEIGHTS and os.stat(entry.path).st_nlink == 1


def test_add_from_path_never_links_a_different_checkpoint(tmp_path):
    run, transient, ckpt = _run(tmp_path, checkpoint=OTHER)
    entry = _pool(run).add_from_path(transient, STEP, twin=ckpt)
    assert entry.path.read_bytes() == WEIGHTS and not os.path.samefile(entry.path, ckpt)
    assert ckpt.read_bytes() == OTHER


def test_a_repromotion_over_a_linked_entry_never_writes_through_into_the_checkpoint(tmp_path):
    run, transient, ckpt = _run(tmp_path)
    pool = _pool(run)
    pool.add_from_path(transient, STEP, twin=ckpt)
    transient.write_bytes(OTHER)                                   # the same step, re-run with other weights
    entry = pool.add_from_path(transient, STEP)
    assert entry.path.read_bytes() == OTHER and ckpt.read_bytes() == WEIGHTS


def test_eviction_unlinks_the_pool_name_only(tmp_path):
    run, transient, ckpt = _run(tmp_path)
    pool = _pool(run)
    pool.max_snapshots = 1
    entry = pool.add_from_path(transient, STEP, twin=ckpt)
    other = run / "other.zip"
    other.write_bytes(OTHER)
    pool.add_from_path(other, STEP + 1_000_000)                    # over the cap -> the linked entry is evicted
    assert not entry.path.exists() and ckpt.read_bytes() == WEIGHTS
    assert os.stat(ckpt).st_nlink == 1


def test_a_live_add_over_a_linked_name_does_not_rewrite_the_checkpoint(tmp_path):
    """`SnapshotPool.add` saves through the atomic writer; the entry's name may be a link to a checkpoint."""
    run, transient, ckpt = _run(tmp_path)
    pool = _pool(run)
    pool.add_from_path(transient, STEP, twin=ckpt)

    def _save(path):                                               # what `MaskablePPO.save` does: open + write
        with open(path, "wb") as f:
            f.write(OTHER)

    model = MagicMock()
    model.save.side_effect = _save
    pool.add(model, STEP)
    assert (run / "snapshots" / f"snapshot_{STEP:012d}.zip").read_bytes() == OTHER
    assert ckpt.read_bytes() == WEIGHTS, "the in-place save followed the link into the checkpoint"


def test_the_checkpointer_gives_a_re_reached_step_a_fresh_inode(tmp_path):
    """A restart that resumed from an EARLIER checkpoint re-reaches step N and `model.save`s over the old
    `checkpoint_N`, which a best-model / pool / eval-snapshot name may share. A save that wrote in place would rewrite
    all of them (and a torn one would corrupt all of them)."""
    from main.train.run_io import _TrackingCheckpointCallback

    ckpt_dir = tmp_path / "run" / "checkpoints"
    cb = _TrackingCheckpointCallback(interval_env_steps=100, save_path=str(ckpt_dir))
    model = MagicMock()
    model.num_timesteps = 0

    def _save(path):
        with open(path, "wb") as f:
            f.write(OTHER)

    model.save.side_effect = _save
    cb.model = model
    cb._init_callback()
    cb._on_training_start()
    old = ckpt_dir / "checkpoint_150_steps.zip"
    old.write_bytes(WEIGHTS)
    best = tmp_path / "best_model.zip"
    os.link(old, best)                                             # another name for the old step-150 weights
    model.num_timesteps = 150
    cb.disk_guard = None
    try:
        cb._on_step()
    except Exception:                                              # noqa: BLE001 - only the save matters here
        pass
    assert old.read_bytes() == OTHER, "the new checkpoint is written"
    assert best.read_bytes() == WEIGHTS, "the other name kept the old weights"
    assert not os.path.samefile(old, best)


def test_a_pool_name_survives_the_checkpoints_deletion(tmp_path):
    """Retention deleting the checkpoint frees nothing the pool still needs: the inode lives while one name does."""
    run, transient, ckpt = _run(tmp_path)
    entry = _pool(run).add_from_path(transient, STEP, twin=ckpt)
    ckpt.unlink()
    assert entry.path.read_bytes() == WEIGHTS and os.stat(entry.path).st_nlink == 1
    assert isinstance(entry.path, Path)
