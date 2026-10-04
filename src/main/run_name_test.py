"""Tests for `--run-name` run-dir resolution (`_resolve_fresh_model_dir`).

A memorable name → `models/<name>/` instead of a date-stamped `models/rb_run_<ts>/`; the exploiter
mode gets a derived default; and a fresh NAMED run refuses to clobber an existing run's dir."""

import pytest

from main.train_rl_agent import _resolve_fresh_model_dir


@pytest.fixture(autouse=True)
def _archive(run_archive):
    """Every test here resolves against its own run archive (`<tmp_path>/models`); the real one is
    sealed. The `models/<name>` the old tests asserted is now `<archive>/<name>`, absolute."""
    return run_archive


def test_run_name_maps_to_models_subdir(run_archive):
    assert _resolve_fresh_model_dir("my_exploiter_v1", None, None) == str(run_archive / "my_exploiter_v1")


def test_no_name_falls_back_to_timestamp(run_archive):
    d = _resolve_fresh_model_dir(None, None, None)
    assert d.startswith(str(run_archive / "rb_run_"))       # the date-stamped default, era-prefixed


def test_exploiter_label_derives_a_default_name(run_archive):
    # ext_ prefix stripped → a readable <archive>/rb_exploiter_vs_<target> default when unnamed.
    d = _resolve_fresh_model_dir(None, "ext_ai_v6_13_outgoing_dmg_0620", None)
    assert d == str(run_archive / "rb_exploiter_vs_ai_v6_13_outgoing_dmg_0620")


def test_explicit_name_beats_exploiter_default(run_archive):
    d = _resolve_fresh_model_dir("crush_v3", "ext_ai_v6_13", None)
    assert d == str(run_archive / "crush_v3")


@pytest.mark.parametrize("bad", ["a/b", "../escape", "foo/../bar", ".", "-x"])
def test_invalid_run_name_exits(bad):
    with pytest.raises(SystemExit):
        _resolve_fresh_model_dir(bad, None, None)


def test_clobber_guard_blocks_writing_into_an_existing_run(run_archive):
    # Naming a fresh run after an EXISTING run (one with a metadata.json) must FATAL, not overwrite it.
    existing = run_archive / "ai_v6_13_outgoing_dmg_0620"
    existing.mkdir(parents=True)
    (existing / "metadata.json").write_text("{}")
    with pytest.raises(SystemExit):
        _resolve_fresh_model_dir("ai_v6_13_outgoing_dmg_0620", None, None)


def test_clobber_guard_allows_resuming_the_same_run(run_archive):
    # ...but if --model points at a checkpoint INSIDE that dir, it's a legit resume → allowed.
    run = run_archive / "keep_going"
    (run / "checkpoints").mkdir(parents=True)
    (run / "metadata.json").write_text("{}")
    ckpt = run / "checkpoints" / "checkpoint_100_steps.zip"
    ckpt.write_text("")
    assert _resolve_fresh_model_dir("keep_going", None, str(ckpt)) == str(run)


def test_clobber_guard_allows_a_fresh_name(run_archive):
    # A name with no existing dir → fine (the common case).
    assert _resolve_fresh_model_dir("brand_new_run", None, None) == str(run_archive / "brand_new_run")
