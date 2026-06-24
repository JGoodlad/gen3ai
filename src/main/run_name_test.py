"""Tests for `--run-name` run-dir resolution (`_resolve_fresh_model_dir`).

A memorable name → `models/<name>/` instead of a date-stamped `models/run_<ts>/`; the exploiter
mode gets a derived default; and a fresh NAMED run refuses to clobber an existing run's dir."""
import os

import pytest

from main.train_rl_agent import _resolve_fresh_model_dir


def test_run_name_maps_to_models_subdir():
    assert _resolve_fresh_model_dir("my_exploiter_v1", None, None) == os.path.join("models", "my_exploiter_v1")


def test_no_name_falls_back_to_timestamp():
    d = _resolve_fresh_model_dir(None, None, None)
    assert d.startswith("models/run_")          # the legacy date-stamped default, unchanged


def test_exploiter_label_derives_a_default_name():
    # ext_ prefix stripped → a readable models/exploiter_vs_<target> default when unnamed.
    d = _resolve_fresh_model_dir(None, "ext_ai_v6_13_outgoing_dmg_0620", None)
    assert d == os.path.join("models", "exploiter_vs_ai_v6_13_outgoing_dmg_0620")


def test_explicit_name_beats_exploiter_default():
    d = _resolve_fresh_model_dir("crush_v3", "ext_ai_v6_13", None)
    assert d == os.path.join("models", "crush_v3")


@pytest.mark.parametrize("bad", ["a/b", "../escape", "foo/../bar", ".", "-x"])
def test_invalid_run_name_exits(bad):
    with pytest.raises(SystemExit):
        _resolve_fresh_model_dir(bad, None, None)


def test_clobber_guard_blocks_writing_into_an_existing_run(tmp_path, monkeypatch):
    # Naming a fresh run after an EXISTING run (one with a metadata.json) must FATAL, not overwrite it.
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "models" / "ai_v6_13_outgoing_dmg_0620"
    existing.mkdir(parents=True)
    (existing / "metadata.json").write_text("{}")
    with pytest.raises(SystemExit):
        _resolve_fresh_model_dir("ai_v6_13_outgoing_dmg_0620", None, None)


def test_clobber_guard_allows_resuming_the_same_run(tmp_path, monkeypatch):
    # ...but if --model points at a checkpoint INSIDE that dir, it's a legit resume → allowed.
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "models" / "keep_going"
    (run / "checkpoints").mkdir(parents=True)
    (run / "metadata.json").write_text("{}")
    ckpt = run / "checkpoints" / "checkpoint_100_steps.zip"
    ckpt.write_text("")
    assert _resolve_fresh_model_dir("keep_going", None, str(ckpt)) == os.path.join("models", "keep_going")


def test_clobber_guard_allows_a_fresh_name(tmp_path, monkeypatch):
    # A name with no existing dir → fine (the common case).
    monkeypatch.chdir(tmp_path)
    assert _resolve_fresh_model_dir("brand_new_run", None, None) == os.path.join("models", "brand_new_run")
