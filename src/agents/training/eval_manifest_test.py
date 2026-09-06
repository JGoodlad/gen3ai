"""Unit tests for the per-cycle eval manifest writer + trace grooming."""

import json
import os

import pytest

from agents.training.eval_callback import (
    _FORENSIC_LOSS_QUOTA, _FORENSIC_WIN_QUOTA, PerOpponentEvalCallback, record_eval_selection,
    write_eval_manifest,
)
from agents.training.eval_sharding.results import (
    ShardResult, aggregate, to_merged, write_shard_result,
)
from agents.training.eval_sharding.units import BOT, EvalItem, ShardUnit
from agents.training.trace_selection import (
    SELECTION_SCHEMA, UNKNOWN_LABEL, describe_selection, manifest_win_rates, read_selection,
)


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
    m = write_eval_manifest(run, 8_000_000, opponents=["random", "heuristic"], n_games=100)
    assert m["step"] == 8_000_000 and m["num_timesteps"] == 8_000_000
    assert m["git_hash"] == "abc123"
    assert m["arch_signature"] == "gen3_test_v1" and m["config_version"] == 2
    assert m["opponents"] == ["random", "heuristic"] and m["n_games"] == 100
    assert m["snapshot"] is None and m["saved_at"]
    # written to the right place
    path = os.path.join(run, "eval_traces", "step_8000000", "eval_manifest.json")
    assert os.path.exists(path)
    assert json.load(open(path))["git_hash"] == "abc123"


def test_prune_eval_traces_keeps_n_most_recent(tmp_path):
    run = tmp_path / "run"
    for s in (1, 2, 3, 4, 5):
        d = run / "eval_traces" / f"step_{s}000000" / "random"
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


# ── the TRACE SELECTION the manifest records (`gen3_trace_selection_manifest_v1`) ────────────
#
# Eval traces are written under a quota that PREFERS LOSSES (by design — the prober is a
# loss-forensics tool), and before this shipped NOTHING in the trace tree said so, so every
# consumer that averages over traces silently inherited a loss-enriched sample.

def test_the_manifest_states_the_selection_RULE_at_launch(tmp_path):
    """Written at LAUNCH, because the rule is known then. The COUNTS are not — no battle has been
    played — so `selection` is null until collect."""
    run = _seed_run(tmp_path)
    m = write_eval_manifest(run, 1000, opponents=["heuristic"], n_games=100)
    assert m["selection_schema"] == SELECTION_SCHEMA
    assert "LOSS-ENRICHED" in m["selection_rule"]
    assert str(_FORENSIC_LOSS_QUOTA) in m["selection_rule"]
    assert str(_FORENSIC_WIN_QUOTA) in m["selection_rule"]
    assert m["selection"] is None


def test_a_cycle_that_never_COLLECTS_reads_UNKNOWN_not_uniform(tmp_path):
    """A crashed cycle leaves `selection: null`. That must read exactly like a legacy manifest —
    UNKNOWN — and never like 'the quota captured nothing' or 'the sample is unbiased'."""
    run = _seed_run(tmp_path)
    m = write_eval_manifest(run, 1000, opponents=["heuristic"], n_games=100)
    assert read_selection(m) is None
    assert describe_selection(m) == UNKNOWN_LABEL


def test_record_eval_selection_writes_the_per_opponent_counts_and_rates(tmp_path):
    run = _seed_run(tmp_path)
    write_eval_manifest(run, 1000, opponents=["heuristic", "random"], n_games=100)
    merged = {"counts": {"heuristic": (90, 100), "random": (99, 100)},
              "traces": {"heuristic": (5, 15), "random": (5, 6)}}
    block = record_eval_selection(run, 1000, merged)

    assert block["schema"] == SELECTION_SCHEMA
    h = block["opponents"]["heuristic"]
    assert (h["battles_played"], h["battles_won"]) == (100, 90)
    assert (h["traces_written"], h["traces_won"]) == (15, 5)
    # THE FINDING, in one assertion: 5 of 90 wins traced against 10 of 10 losses.
    assert h["capture_rate_win"] == pytest.approx(5 / 90)
    assert h["capture_rate_loss"] == pytest.approx(1.0)
    assert h["capture_rate_loss"] > h["capture_rate_win"]

    on_disk = json.loads(
        (tmp_path / "run" / "eval_traces" / "step_1000" / "eval_manifest.json").read_text())
    assert read_selection(on_disk) is not None
    assert manifest_win_rates(on_disk)["heuristic"] == pytest.approx(0.90)
    assert manifest_win_rates(on_disk)["random"] == pytest.approx(0.99)
    # the identity fields the manifest already carried are untouched by the patch
    assert on_disk["git_hash"] == "abc123" and on_disk["n_games"] == 100


def test_the_recorded_sums_reconcile_on_every_opponent(tmp_path):
    run = _seed_run(tmp_path)
    write_eval_manifest(run, 1000, opponents=["a", "b", "c"], n_games=50)
    merged = {"counts": {"a": (50, 50), "b": (0, 50), "c": (17, 43)},
              "traces": {"a": (5, 5), "b": (0, 10), "c": (4, 13)}}
    block = record_eval_selection(run, 1000, merged)
    for name, e in block["opponents"].items():
        assert e["traces_won"] <= e["battles_won"], name
        assert e["traces_written"] <= e["battles_played"], name
        for k in ("capture_rate_win", "capture_rate_loss"):
            assert e[k] is None or 0.0 <= e[k] <= 1.0, (name, k)


def test_an_opponent_whose_shards_report_no_trace_counts_records_ZERO_not_missing(tmp_path):
    """A legacy shard file (written before this shipped) deserializes with traces_* == 0. That is
    an honest zero for THAT opponent; the cycle is still 'selection recorded'."""
    run = _seed_run(tmp_path)
    write_eval_manifest(run, 1000, opponents=["heuristic"], n_games=10)
    block = record_eval_selection(run, 1000, {"counts": {"heuristic": (7, 10)}, "traces": {}})
    e = block["opponents"]["heuristic"]
    assert e["traces_written"] == 0 and e["capture_rate_win"] == 0.0


def test_recording_a_selection_never_raises_when_the_manifest_is_gone(tmp_path):
    """Provenance for an offline reader must not be able to take a training run down at an eval
    boundary. It warns and leaves the block null, which reads as UNKNOWN."""
    run = _seed_run(tmp_path)
    assert record_eval_selection(run, 4242, {"counts": {"h": (1, 2)}, "traces": {}}) is None
    assert record_eval_selection(None, 1, {"counts": {"h": (1, 2)}}) is None
    assert record_eval_selection(run, 1, {"counts": {}}) is None


def test_the_shard_layer_pools_trace_counts_exactly_and_reads_legacy_shards(tmp_path):
    """`traces_*` are additive like every other ShardResult field, and DEFAULTED so a shard file
    written before this shipped still deserializes (its counts are then 0)."""
    item = EvalItem(key="h", kind=BOT, n_games=10)
    units = [ShardUnit(item=item, shard_index=0, n_games=5),
             ShardUnit(item=item, shard_index=1, n_games=5)]
    d = str(tmp_path)
    write_shard_result(d, ShardResult(unit_id=units[0].unit_id, item_key="h", worker_id=0, n_won=4,
                                      n_finished=5, sum_reward=1.0, n_episodes=5, sum_ep_len=50.0,
                                      duration_sec=1.0, traces_written=3, traces_won=1))
    # a LEGACY shard: no traces_* keys at all
    legacy = {"unit_id": units[1].unit_id, "item_key": "h", "worker_id": 1, "n_won": 3,
              "n_finished": 5, "sum_reward": 1.0, "n_episodes": 5, "sum_ep_len": 50.0,
              "duration_sec": 1.0, "td_residuals": []}
    with open(os.path.join(d, f"shard__{units[1].unit_id}.json"), "w") as f:
        json.dump(legacy, f)

    pooled = aggregate(units, d)["h"]
    assert (pooled.n_won, pooled.n_finished) == (7, 10)
    assert (pooled.traces_written, pooled.traces_won) == (3, 1)   # 3+0, 1+0
    assert to_merged({"h": pooled})["traces"]["h"] == (1, 3)      # (won, written)
