"""``main.reads.models``: the SETTLED rule (a final, never a checkpoint) and the sha-checked copy with its sidecars."""
from __future__ import annotations

import os

import pytest

from main.reads import models as M
from main.reads.spec import Model, ModelRef

NOW = 2_000_000_000.0


def _run(tmp_path, *, final_age=None, ckpt_age=None):
    run = tmp_path / "rb_demo"
    (run / "checkpoints").mkdir(parents=True)
    (run / "model_config.json").write_text("{}")
    if ckpt_age is not None:
        c = run / "checkpoints" / "checkpoint_480000085_steps.zip"
        c.write_bytes(b"ckpt")
        os.utime(c, (NOW - ckpt_age, NOW - ckpt_age))
    if final_age is not None:
        z = run / "final_model.zip"
        z.write_bytes(b"final")
        os.utime(z, (NOW - final_age, NOW - final_age))
    return run


def _final(run, **kw):
    return Model("x", "arm", 1, ModelRef(kind="final", run=str(run), min_step=480_000_000, **kw))


def _step(n):
    return lambda _p: n


def test_a_quiet_checkpoint_is_never_read_while_the_final_is_being_written(tmp_path):
    """x26c FINDING 2: the checkpoint >= the step was quiet, the final was not; the old rule played the checkpoint."""
    run = _run(tmp_path, final_age=30, ckpt_age=3600)
    r = M.resolve(_final(run), now=NOW, step_of=_step(480_081_209))
    assert isinstance(r, M.NotReady) and "quiet" in r.why


def test_no_final_is_not_ready_even_with_a_checkpoint_past_the_step(tmp_path):
    run = _run(tmp_path, ckpt_age=3600)
    r = M.resolve(_final(run), now=NOW, step_of=_step(480_081_209))
    assert isinstance(r, M.NotReady) and "no final" in r.why


def test_a_quiet_final_short_of_its_step_is_not_ready(tmp_path):
    run = _run(tmp_path, final_age=3600)
    r = M.resolve(_final(run), now=NOW, step_of=_step(479_999_999))
    assert isinstance(r, M.NotReady) and "< 480,000,000" in r.why


def test_the_done_marker_gates_a_quiet_final(tmp_path):
    run = _run(tmp_path, final_age=3600)
    marker = tmp_path / "status.txt"
    marker.write_text("X35END: STARTED rb_demo\n")
    m = _final(run, done_marker=(str(marker), "X35END: DONE rb_demo"))
    assert isinstance(M.resolve(m, now=NOW, step_of=_step(480_081_209)), M.NotReady)
    marker.write_text("X35END: DONE rb_demo: ok\n")
    r = M.resolve(m, now=NOW, step_of=_step(480_081_209))
    assert isinstance(r, M.Resolved) and r.source.endswith("final_model.zip") and r.step == 480_081_209


def test_a_pinned_file_must_hash_to_the_spec(tmp_path):
    f = tmp_path / "a.zip"
    f.write_bytes(b"bytes")
    good = Model("a", "arm", 1, ModelRef(kind="pinned", path=str(f), sha256=M.sha256_file(f)))
    assert isinstance(M.resolve(good, step_of=_step(7)), M.Resolved)
    bad = Model("a", "arm", 1, ModelRef(kind="pinned", path=str(f), sha256="0" * 64))
    with pytest.raises(M.ModelError, match="sha256"):
        M.resolve(bad, step_of=_step(7))


def test_the_copy_carries_its_provenance_and_refuses_without_model_config(tmp_path):
    run = _run(tmp_path, final_age=3600)
    z = run / "final_model.zip"
    (run / "final_model.json").write_text('{"sidecar": 1}')
    dst, sha = M.copy_checked(z, tmp_path / "copy")
    assert sha == M.sha256_file(z) and (tmp_path / "copy" / "model_config.json").is_file()
    assert (tmp_path / "copy" / "final_model.json").is_file()
    (run / "model_config.json").unlink()
    with pytest.raises(M.ModelError, match="model_config.json"):      # d1_read FINDING 4: the void h_d attempt
        M.copy_checked(z, tmp_path / "copy2")


def test_verify_refuses_a_copy_whose_bytes_changed(tmp_path):
    run = _run(tmp_path, final_age=3600)
    r = M.Resolved("x", str(run / "final_model.zip"), M.sha256_file(run / "final_model.zip"), 1)
    e = M.copy_resolved(r, tmp_path / "c")
    M.verify(e)
    with open(e["copy"], "ab") as f:
        f.write(b"!")
    with pytest.raises(M.ModelError, match="sha256"):
        M.verify(e)


def test_a_models_path_resolves_under_the_run_archive(monkeypatch, tmp_path):
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(tmp_path))
    assert M.spec_path("models/rb_x/final_model.zip") == tmp_path / "rb_x" / "final_model.zip"
    assert M.spec_path("/abs/p.zip").as_posix() == "/abs/p.zip"
