"""`maturity_series`: the battery's trigger (anchors of a named run), the read-only checkpoint copy and the
cap-new-work rule. Pure: temporary files only."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from main.hidden_info_study import maturity_series as M


def _ckpts(tmp_path: Path, steps) -> Path:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    for s in steps:
        (run / "checkpoints" / f"checkpoint_{s}_steps.zip").write_bytes(f"ckpt {s}".encode())
        (run / "checkpoints" / f"checkpoint_{s}_steps.json").write_text("{}")
    (run / "checkpoints" / "not_a_checkpoint.zip").write_bytes(b"x")
    return run


def test_checkpoint_files_parse_and_sort(tmp_path):
    run = _ckpts(tmp_path, [25000027, 1000192, 100000037])
    assert [s for s, _ in M.checkpoint_files(run)] == [1000192, 25000027, 100000037]


def test_anchor_is_the_first_checkpoint_in_its_window(tmp_path):
    # X26's own steps around the four registered anchors, plus a gap at 50M (no file in [50M, 75M))
    steps = [24000222, 25000027, 26000100, 99000102, 100000037, 101000217, 174000112, 175000114, 225000152,
             226000061, 75000010]
    files = M.checkpoint_files(_ckpts(tmp_path, steps))
    a = M.anchors(files)
    assert {m: s for m, (s, _p) in a.items()} == {25_000_000: 25000027, 75_000_000: 75000010,
                                                  100_000_000: 100000037, 150_000_000: 174000112,
                                                  175_000_000: 175000114, 225_000_000: 225000152}
    assert 50_000_000 not in a          # never filled from a neighbour


def test_pending_anchors_skip_registered_and_label_sorts(tmp_path):
    files = M.checkpoint_files(_ckpts(tmp_path, [25000027, 50000001, 100000037]))
    s = M.SERIES["x26"]
    pend = M.pending_anchors(files, s, registered={"x26_025M"})
    assert list(pend) == ["x26_050M", "x26_100M"]
    assert M.anchor_label("x26", 225_000_000) == "x26_225M"
    with pytest.raises(M.SeriesError):
        M.anchor_label("x26", 25_500_000)


def test_series_x26_is_the_pinned_run():
    s = M.SERIES["x26"]
    assert s.run == "rb_x26_s2601" and s.pin.startswith("9c033489") and s.play_commit() == s.pin


def test_copy_checkpoint_read_only_with_sha(tmp_path):
    src = tmp_path / "models" / "run" / "c.zip"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"weights" * 1000)
    mode0 = src.stat().st_mode
    dst = tmp_path / "archive" / "c.zip"
    rec = M.copy_checkpoint(src, dst, forbid_under=[tmp_path / "models"])
    assert rec["copied"] and rec["sha256"] == M.sha256_file(src) == M.sha256_file(dst)
    assert not (dst.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    assert src.stat().st_mode == mode0                     # the source is untouched
    again = M.copy_checkpoint(src, dst, forbid_under=[tmp_path / "models"])
    assert not again["copied"] and again["sha256"] == rec["sha256"]
    assert not list(dst.parent.glob("*.part"))


def test_copy_checkpoint_refuses_the_run_archive_and_a_different_copy(tmp_path):
    src = tmp_path / "models" / "run" / "c.zip"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"a")
    with pytest.raises(M.SeriesError, match="may not be written under"):
        M.copy_checkpoint(src, tmp_path / "models" / "copy.zip", forbid_under=[tmp_path / "models"])
    dst = tmp_path / "out" / "c.zip"
    dst.parent.mkdir()
    dst.write_bytes(b"b")
    with pytest.raises(M.SeriesError, match="exists with sha"):
        M.copy_checkpoint(src, dst)
    with pytest.raises(M.SeriesError, match="no such checkpoint"):
        M.copy_checkpoint(tmp_path / "missing.zip", tmp_path / "x.zip")
    os.chmod(dst, 0o644)


def test_select_new_work_caps_new_work_not_the_id_list():
    ids = [f"s{i:02d}" for i in range(10)]
    done = set(ids[:4])
    assert M.select_new_work(ids, done, 3) == ["s04", "s05", "s06"]
    # the E2 stall: once the first `limit` ids are done, capping the list first returns only done ids forever
    done = set(ids[:3])
    assert M.select_new_work(ids, done, 3) == ["s03", "s04", "s05"]
    assert [x for x in ids[:3] if x not in done] == []      # what the defective order would have worked on
    assert M.select_new_work(ids, done, None) == ids[3:]
    assert M.select_new_work(ids, {x: 1 for x in ids}, 5) == []


def test_plateau_anchor_is_the_exact_declared_file_with_its_sha(tmp_path):
    import dataclasses

    run = _ckpts(tmp_path, [275000010, 280000170, 281000000])
    files = M.checkpoint_files(run)
    p = run / "checkpoints" / "checkpoint_280000170_steps.zip"
    s = dataclasses.replace(M.SERIES["x26"], plateau=280000170, plateau_sha256=M.sha256_file(p))
    lab, (step, path) = M.plateau_anchor(files, s)
    assert lab == "x26_P280M" and step == 280000170 and path == p
    assert lab not in {M.anchor_label("x26", m) for m in M.anchors(files)}      # never a 25M-grid label
    with pytest.raises(M.SeriesError, match="not the declared"):
        M.plateau_anchor(files, dataclasses.replace(s, plateau_sha256="0" * 64))
    with pytest.raises(M.SeriesError, match="no checkpoint file at the declared"):
        M.plateau_anchor(files, dataclasses.replace(s, plateau=280000171))       # never the nearest file
    with pytest.raises(M.SeriesError, match="no plateau declared"):
        M.plateau_label(dataclasses.replace(s, plateau=None))


def test_x26_declares_its_plateau_checkpoint():
    s = M.SERIES["x26"]
    assert s.plateau == 280000170 and M.plateau_label(s) == "x26_P280M"
    assert s.plateau_sha256 == "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"
