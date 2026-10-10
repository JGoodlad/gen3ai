"""`eval_traces/step_<N>/snapshot.zip` is a HARD LINK to the checkpoint saved at the same step, not a duplicate.

THE WASTE. Each eval cycle persisted a 62 MB copy of the weights at that step, and the periodic checkpointer had just
written the SAME bytes (the checkpointer is first in the callback list, both name the file by `num_timesteps`, and
their cadences coincide): 416 MB of a run's 484 MB of eval traces, the battle records being 21 MB. Measured on a
finished screen run (`rb_st_legacy_s1001`, five coincident steps): the two files were sha256-identical at every one.

`persist_eval_snapshot` now makes `snapshot.zip` a second name for the checkpoint's inode when - and only when - the
checkpoint at that step exists on the same filesystem and is BYTE-IDENTICAL (sha256), else a copy. Pinned here:

* the link is made when identical, a copy when not (no checkpoint at the step, another content, another size, no
  link possible), and the manifest says which;
* 🚨 a re-persist over an existing link NEVER writes through it (a plain `copy2` would truncate the CHECKPOINT);
* the link survives the checkpoint's deletion (retention is safe) and the checkpoint survives the snapshot's removal
  (`prune_eval_snapshots`, `main.prober.groom`);
* every reader keeps working - a hard link is an ordinary file path - and the groomer no longer counts a linked file
  as reclaimed space.
"""
import errno
import json
import os

import pytest

from agents.training import eval_collect
from agents.training.eval_collect import (
    checkpoint_at_step, persist_eval_snapshot, prune_eval_snapshots, store_eval_snapshot,
)
from agents.training.eval_launch import EVAL_MANIFEST_NAME, EVAL_SNAPSHOT_NAME
from main.prober.discovery import build_trace_tree, resolve_model_for_step
from main.prober.groom import groom_run

WEIGHTS = b"PK-weights-" + bytes(range(256)) * 40          # stands for a ~62 MB zip; content is what is compared
OTHER = b"PK-OTHER---" + bytes(range(256)) * 40            # same length, different bytes


def _run(tmp_path, step=2_000_128, *, checkpoint=WEIGHTS, name="rb_run"):
    """A run dir with an eval step dir + manifest, the cycle's transient snapshot, and (optionally) the checkpoint."""
    run = tmp_path / name
    step_dir = run / "eval_traces" / f"step_{step}"
    step_dir.mkdir(parents=True)
    (step_dir / EVAL_MANIFEST_NAME).write_text(json.dumps({"step": step, "snapshot": None, "git_hash": "abc"}))
    transient = run / ".eval_runs" / f"step_{step}" / "snapshot.zip"
    transient.parent.mkdir(parents=True)
    transient.write_bytes(WEIGHTS)
    ckpt = run / "checkpoints" / f"checkpoint_{step}_steps.zip"
    if checkpoint is not None:
        ckpt.parent.mkdir()
        ckpt.write_bytes(checkpoint)
    return str(run), step, str(transient), str(ckpt)


def _manifest(run, step):
    return json.loads((open(os.path.join(run, "eval_traces", f"step_{step}", EVAL_MANIFEST_NAME)).read()))


def _snap(run, step):
    return os.path.join(run, "eval_traces", f"step_{step}", EVAL_SNAPSHOT_NAME)


# ---------------------------------------------------------------------------------------
# the link is made when identical, a copy otherwise
# ---------------------------------------------------------------------------------------

def test_an_identical_checkpoint_at_the_same_step_makes_snapshot_zip_a_hard_link(tmp_path, capsys):
    run, step, transient, ckpt = _run(tmp_path)
    persist_eval_snapshot(run, step, transient, keep_n=10)
    snap = _snap(run, step)
    assert os.path.samefile(snap, ckpt), "snapshot.zip must be a second name for the checkpoint's inode"
    assert os.stat(snap).st_nlink == 2 and os.stat(ckpt).st_nlink == 2
    assert open(snap, "rb").read() == WEIGHTS
    m = _manifest(run, step)
    assert m["snapshot"] == EVAL_SNAPSHOT_NAME, "the manifest pointer every reader follows is unchanged"
    assert m["snapshot_storage"]["mode"] == "hardlink"
    assert m["snapshot_storage"]["checkpoint"] == os.path.join("checkpoints", os.path.basename(ckpt))
    assert len(m["snapshot_storage"]["sha256"]) == 64
    assert "hard link" in capsys.readouterr().out


def test_no_checkpoint_at_the_step_falls_back_to_a_copy(tmp_path, capsys):
    """Eval and checkpoint cadences do not always coincide."""
    run, step, transient, _ckpt = _run(tmp_path, checkpoint=None)
    assert checkpoint_at_step(run, step) is None
    persist_eval_snapshot(run, step, transient, keep_n=10)
    snap = _snap(run, step)
    assert open(snap, "rb").read() == WEIGHTS and os.stat(snap).st_nlink == 1
    m = _manifest(run, step)
    assert m["snapshot"] == EVAL_SNAPSHOT_NAME and m["snapshot_storage"]["mode"] == "copy"
    assert m["snapshot_storage"]["checkpoint"] is None and "cadences" in m["snapshot_storage"]["note"]
    assert "COPIED" not in capsys.readouterr().out, "a non-coincident step is normal and must not warn"


def test_a_checkpoint_that_is_not_byte_identical_is_never_linked_and_is_warned_about(tmp_path, capsys):
    run, step, transient, ckpt = _run(tmp_path, checkpoint=OTHER)
    persist_eval_snapshot(run, step, transient, keep_n=10)
    snap = _snap(run, step)
    assert not os.path.samefile(snap, ckpt)
    assert open(snap, "rb").read() == WEIGHTS, "the snapshot is the bytes that played the cycle"
    assert open(ckpt, "rb").read() == OTHER, "the checkpoint is untouched"
    assert _manifest(run, step)["snapshot_storage"]["mode"] == "copy"
    out = capsys.readouterr().out
    assert "COPIED although a checkpoint exists" in out and "not byte-identical" in out


def test_a_checkpoint_of_a_different_size_is_never_linked(tmp_path):
    run, step, transient, ckpt = _run(tmp_path, checkpoint=WEIGHTS + b"x")
    persist_eval_snapshot(run, step, transient, keep_n=10)
    assert not os.path.samefile(_snap(run, step), ckpt)
    assert "size" in _manifest(run, step)["snapshot_storage"]["note"]


def test_when_a_link_cannot_be_made_it_copies_and_says_why(tmp_path, monkeypatch):
    """EXDEV / EPERM / a filesystem without hard links: the cycle must still get its snapshot."""
    run, step, transient, ckpt = _run(tmp_path)

    def _no_link(src, dst, *a, **k):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", _no_link)
    persist_eval_snapshot(run, step, transient, keep_n=10)
    snap = _snap(run, step)
    assert open(snap, "rb").read() == WEIGHTS and not os.path.samefile(snap, ckpt)
    m = _manifest(run, step)["snapshot_storage"]
    assert m["mode"] == "copy" and "linking failed" in m["note"]
    assert not os.path.exists(snap + ".tmp"), "a failed link must leave no .tmp behind"


def test_store_eval_snapshot_reports_mode_digest_and_note(tmp_path):
    run, step, transient, ckpt = _run(tmp_path)
    dst = _snap(run, step)
    mode, digest, note = store_eval_snapshot(transient, dst, ckpt)
    assert mode == "hardlink" and len(digest) == 64 and "checkpoints/" in note
    mode2, digest2, _ = store_eval_snapshot(transient, dst, None)
    assert mode2 == "copy" and digest2 == ""


# ---------------------------------------------------------------------------------------
# 🚨 never write THROUGH an existing link
# ---------------------------------------------------------------------------------------

def test_a_re_persist_over_an_existing_link_never_touches_the_checkpoint(tmp_path):
    """A restart can re-persist a step. If the first persist left a hard link and the second cannot link (different
    content), a plain `shutil.copy2(snapshot, dst)` would open `dst` for writing - truncating the CHECKPOINT through
    the shared inode. The replace is atomic and never opens the old inode."""
    run, step, transient, ckpt = _run(tmp_path)
    persist_eval_snapshot(run, step, transient, keep_n=10)
    assert os.path.samefile(_snap(run, step), ckpt)
    with open(transient, "wb") as f:                       # the re-evaluated cycle's snapshot now differs
        f.write(OTHER)
    persist_eval_snapshot(run, step, transient, keep_n=10)
    assert open(ckpt, "rb").read() == WEIGHTS, "THE CHECKPOINT WAS WRITTEN THROUGH THE LINK"
    assert open(_snap(run, step), "rb").read() == OTHER
    assert not os.path.samefile(_snap(run, step), ckpt) and os.stat(ckpt).st_nlink == 1


def test_persisting_the_same_step_twice_is_idempotent(tmp_path):
    run, step, transient, ckpt = _run(tmp_path)
    persist_eval_snapshot(run, step, transient, keep_n=10)
    persist_eval_snapshot(run, step, transient, keep_n=10)
    assert os.path.samefile(_snap(run, step), ckpt) and os.stat(ckpt).st_nlink == 2
    assert not os.path.lexists(_snap(run, step) + ".tmp"), (
        "rename() of a link onto another name of the same inode is a no-op - the .tmp must not be stranded")
    assert open(ckpt, "rb").read() == WEIGHTS


# ---------------------------------------------------------------------------------------
# retention is safe both ways
# ---------------------------------------------------------------------------------------

def test_the_snapshot_survives_the_checkpoint_being_deleted(tmp_path):
    """Hard links survive deletion of the other name - the checkpoint-retention tool can never orphan a trace."""
    run, step, transient, ckpt = _run(tmp_path)
    persist_eval_snapshot(run, step, transient, keep_n=10)
    os.remove(ckpt)
    assert open(_snap(run, step), "rb").read() == WEIGHTS
    assert os.stat(_snap(run, step)).st_nlink == 1


def test_pruning_an_old_snapshot_leaves_its_checkpoint_intact(tmp_path):
    run, step, transient, ckpt = _run(tmp_path)
    persist_eval_snapshot(run, step, transient, keep_n=10)
    prune_eval_snapshots(run, keep_n=1)                    # still within retention
    assert os.path.exists(_snap(run, step))
    # a newer step pushes the first one out of a keep-1 window
    run2_step = step + 1_000_000
    step_dir = os.path.join(run, "eval_traces", f"step_{run2_step}")
    os.makedirs(step_dir)
    open(os.path.join(step_dir, EVAL_SNAPSHOT_NAME), "wb").write(OTHER)
    prune_eval_snapshots(run, keep_n=1)
    assert not os.path.exists(_snap(run, step))
    assert open(ckpt, "rb").read() == WEIGHTS and os.stat(ckpt).st_nlink == 1


def test_the_groomer_does_not_count_a_linked_snapshot_as_reclaimed_space(tmp_path):
    """Removing a hard link whose checkpoint stands frees no blocks, and a report that said it did would be a lie."""
    run, step, transient, ckpt = _run(tmp_path)
    persist_eval_snapshot(run, step, transient, keep_n=10)
    rep = groom_run(run, keep_trace_steps=10, keep_snapshots=0, apply=False)
    assert rep["dropped_snapshots"] == [step]
    assert rep["bytes_reclaimed"] == 0, rep
    assert rep["bytes_hardlinked_not_freed"] == len(WEIGHTS)
    assert groom_run(run, keep_trace_steps=10, keep_snapshots=0, apply=True)["applied"]
    assert not os.path.exists(_snap(run, step)) and open(ckpt, "rb").read() == WEIGHTS


def test_the_groomer_still_counts_an_unlinked_snapshot(tmp_path):
    run, step, transient, _ckpt = _run(tmp_path, checkpoint=None)
    persist_eval_snapshot(run, step, transient, keep_n=10)
    rep = groom_run(run, keep_trace_steps=10, keep_snapshots=0, apply=False)
    assert rep["bytes_reclaimed"] == len(WEIGHTS) and rep["bytes_hardlinked_not_freed"] == 0


# ---------------------------------------------------------------------------------------
# readers keep working - a hard link is an ordinary file path
# ---------------------------------------------------------------------------------------

def test_the_prober_resolves_the_linked_snapshot_as_the_exact_model(tmp_path):
    run, step, transient, _ckpt = _run(tmp_path)
    os.makedirs(os.path.join(run, "eval_traces", f"step_{step}", "random"))
    with open(os.path.join(run, "eval_traces", f"step_{step}", "random", "win_001_summary.json"), "w") as f:
        f.write("{}")
    persist_eval_snapshot(run, step, transient, keep_n=10)
    tree = build_trace_tree(run)
    c = resolve_model_for_step(tree, step)
    assert c.tier == "exact" and c.path.endswith(f"step_{step}/snapshot.zip") and c.is_exact
    assert c.manifest["snapshot"] == EVAL_SNAPSHOT_NAME


def test_eval_trace_gen_prefers_the_linked_snapshot(tmp_path):
    from main.ops.eval_trace_gen import resolve_checkpoint
    run, step, transient, _ckpt = _run(tmp_path)
    persist_eval_snapshot(run, step, transient, keep_n=10)
    path, got = resolve_checkpoint(__import__("pathlib").Path(run), step)
    assert got == step and str(path).endswith(f"step_{step}/snapshot.zip")


def test_a_zip_read_through_the_link_is_the_checkpoints_zip(tmp_path):
    """The real thing, byte for byte: a genuine zip written once is readable by name through either path."""
    import zipfile
    run, step, transient, ckpt = _run(tmp_path, checkpoint=None)
    with zipfile.ZipFile(transient, "w") as z:
        z.writestr("data", "{}")
        z.writestr("policy.pth", b"\x00" * 64)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    os.link(transient, ckpt)                                # a checkpoint with the same bytes
    persist_eval_snapshot(run, step, transient, keep_n=10)
    with zipfile.ZipFile(_snap(run, step)) as z:
        assert z.namelist() == ["data", "policy.pth"] and z.testzip() is None
    assert os.path.samefile(_snap(run, step), ckpt)


def test_keep_n_zero_persists_nothing(tmp_path):
    run, step, transient, _ckpt = _run(tmp_path)
    persist_eval_snapshot(run, step, transient, keep_n=0)
    assert not os.path.exists(_snap(run, step))
    assert _manifest(run, step)["snapshot"] is None


def test_the_module_the_callbacks_import_from_is_the_one_tested():
    """Both eval callbacks import `persist_eval_snapshot` from `eval_callback`, which re-exports it."""
    from agents.training import eval_callback
    assert eval_callback.persist_eval_snapshot is eval_collect.persist_eval_snapshot
    pytest.importorskip("agents.training.selfplay_callback")
