"""Unit tests for the per-cycle eval manifest writer + trace grooming."""

import json
import os

from agents.training.eval_callback import PerOpponentEvalCallback, write_eval_manifest


def _seed_run(tmp_path, git="abc123", arch="gen3_test_v1", cfgver=2):
    run = tmp_path / "run"
    run.mkdir()
    with open(run / "model_config.json", "w") as f:
        json.dump({"arch_signature": arch, "config_version": cfgver}, f)
    with open(run / "metadata.json", "w") as f:
        json.dump({"git_hash": git}, f)
    return str(run)


def test_manifest_records_identity(tmp_path):
    run = _seed_run(tmp_path)
    m = write_eval_manifest(run, 8_000_000, opponents=["Random", "Heuristic"], n_games=100)
    assert m["step"] == 8_000_000 and m["num_timesteps"] == 8_000_000
    assert m["git_hash"] == "abc123"
    assert m["arch_signature"] == "gen3_test_v1" and m["config_version"] == 2
    assert m["opponents"] == ["Random", "Heuristic"] and m["n_games"] == 100
    assert m["snapshot"] is None and m["saved_at"]
    # written to the right place
    path = os.path.join(run, "eval_traces", "step_8000000", "eval_manifest.json")
    assert os.path.exists(path)
    assert json.load(open(path))["git_hash"] == "abc123"


def test_prune_eval_traces_keeps_n_most_recent(tmp_path):
    run = tmp_path / "run"
    for s in (1, 2, 3, 4, 5):
        d = run / "eval_traces" / f"step_{s}000000" / "Random"
        os.makedirs(d, exist_ok=True)
        (d / "win_001_summary.json").write_text("{}")
    cb = PerOpponentEvalCallback(model_dir=str(run), keep_eval_trace_steps=2)
    cb._prune_eval_traces()
    kept = sorted(os.listdir(run / "eval_traces"))
    assert kept == ["step_4000000", "step_5000000"]   # 2 most-recent kept


def test_prune_eval_traces_zero_keeps_all(tmp_path):
    run = tmp_path / "run"
    for s in (1, 2, 3):
        os.makedirs(run / "eval_traces" / f"step_{s}000000", exist_ok=True)
    cb = PerOpponentEvalCallback(model_dir=str(run), keep_eval_trace_steps=0)
    cb._prune_eval_traces()
    assert len(os.listdir(run / "eval_traces")) == 3   # 0 = keep all


def test_manifest_git_fallback_when_metadata_absent(tmp_path):
    run = tmp_path / "bare"
    run.mkdir()
    # no model_config.json / metadata.json → arch None, git falls back to repo hash
    m = write_eval_manifest(str(run), 1000, opponents=[], n_games=0)
    assert m["arch_signature"] is None and m["config_version"] is None
    # git_hash is either a real repo hash (string) or None — never crashes
    assert m["git_hash"] is None or isinstance(m["git_hash"], str)
