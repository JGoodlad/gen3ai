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
    # The tuple grew a third element (traces_drawn) with the DRAW bucket, 2026-09-07.
    assert to_merged({"h": pooled})["traces"]["h"] == (1, 3, 0)   # (won, written, drawn)


# ── The KEEP-ALL default (`gen3_keep_all_eval_traces_v1`, 2026-09-08) ──────────────────────────
# The eval-trace retention default is PINNED here, not left to whoever edits the parser next.
# It was 20, and that default is what deleted arm A's (`ai_v12_02_winprob_critic`) 10M-step
# traces off disk — the win-prob critic ladder's own registered A@10M comparator — before the
# read that needed them was ever run. A cycle is ~55 MB; a 75M run's full set ~3 GB. Disk is
# recoverable and a matched-step comparator is not, so the default keeps EVERYTHING and a run
# that wants a cap says so.


def test_the_eval_trace_retention_DEFAULT_keeps_every_cycle():
    """0 = keep all, at the constant, at both callbacks, and through argparse."""
    from agents.training.artifact_retention import KEEP_EVAL_TRACE_STEPS_DEFAULT

    assert KEEP_EVAL_TRACE_STEPS_DEFAULT == 0, (
        "the eval-trace retention default must KEEP ALL — a positive cap here silently deletes "
        "the matched-step comparator a later ladder read is scored against")

    # Both callbacks that groom traces take the constant, not a literal of their own: a default
    # that agrees only by coincidence is the shape that drifts.
    import inspect

    from agents.training.eval_callback import PerOpponentEvalCallback
    from agents.training.selfplay_callback import SelfPlayCallback

    for cls in (PerOpponentEvalCallback, SelfPlayCallback):
        param = inspect.signature(cls.__init__).parameters["keep_eval_trace_steps"]
        assert param.default == KEEP_EVAL_TRACE_STEPS_DEFAULT, (
            f"{cls.__name__} carries its own eval-trace retention default "
            f"({param.default!r}) instead of KEEP_EVAL_TRACE_STEPS_DEFAULT")

    # And the flag a launch actually types resolves to it.
    from main.train.parser import build_parser

    assert build_parser().parse_args([]).keep_eval_trace_steps == KEEP_EVAL_TRACE_STEPS_DEFAULT


def test_a_positive_retention_cap_still_prunes():
    """Keep-all is a DEFAULT, not a removal of the mechanism — an explicit cap must still work."""
    assert _FORENSIC_WIN_QUOTA > 0 and _FORENSIC_LOSS_QUOTA > 0  # sanity: module imported fine
    cb = PerOpponentEvalCallback(model_dir="/nonexistent", keep_eval_trace_steps=3)
    assert cb._keep_eval_trace_steps == 3
    assert PerOpponentEvalCallback(model_dir="/nonexistent")._keep_eval_trace_steps == 0


# ── The CAPTURE QUOTA is settable, and its default does not move ───────────────────────────────
# (`gen3_forensic_quota_flag_v1`, 2026-09-08.) The quota used to be three module constants with no
# way to set them, so every arm ever run captured the same ~190 traces per cycle and every
# trace-based read was power-limited at that n. Three flags now raise it per-arm. The DEFAULT is
# pinned because `ai_v12_11_ladder_ctrl10M` — the win-prob ladder's registered control — ran to
# completion at 5/10/5: a default change would put later arms on a different capture regime than
# the control they are differenced against.


def test_the_capture_quota_DEFAULT_is_the_shipped_controls():
    from agents.training.eval_callback import ForensicQuota, _FORENSIC_DRAW_QUOTA
    from main.train.parser import build_parser

    assert ForensicQuota() == (5, 10, 5), (
        "the forensic capture default moved — ai_v12_11_ladder_ctrl10M ran at 5/10/5 and every "
        "ladder arm is differenced against it, so this is a measurement-regime boundary")
    assert (_FORENSIC_WIN_QUOTA, _FORENSIC_LOSS_QUOTA, _FORENSIC_DRAW_QUOTA) == (5, 10, 5)

    a = build_parser().parse_args([])
    assert (a.forensic_win_quota, a.forensic_loss_quota,
            a.forensic_draw_quota) == (5, 10, 5)


def test_a_raised_quota_reaches_the_manifests_STATED_RULE(tmp_path):
    """The rule the manifest states must be the rule the recorder was given, not the default.

    A manifest that states 5/10/5 while the worker captured 40/40/10 is worse than no manifest:
    rule 17 reweights by exactly this record.
    """
    from agents.training.eval_callback import ForensicQuota

    run = _seed_run(tmp_path)
    raised = ForensicQuota(win=40, loss=40, draw=10)
    m = write_eval_manifest(run, 10_000_000, opponents=["random"], n_games=100, quota=raised)
    for n in ("40", "10"):
        assert n in m["selection_rule"]
    assert "the first 5 wins" not in m["selection_rule"]

    # …and again at COLLECT, which rewrites both the rule and the quota block.
    merged = {"counts": {"random": (60, 100)}, "traces": {"random": (40, 80, 0)},
              "draws": {"random": 0}}
    block = record_eval_selection(run, 10_000_000, merged, quota=raised)
    assert (block["win_quota"], block["loss_quota"], block["draw_quota"]) == (40, 40, 10)

    # A caller that says NOTHING gets the default — never zero, which would read as
    # "this cycle deliberately captured nothing".
    m2 = write_eval_manifest(run, 12_000_000, opponents=["random"], n_games=100)
    assert str(_FORENSIC_LOSS_QUOTA) in m2["selection_rule"]


def test_the_per_shard_split_never_silently_disables_capture():
    from agents.training.eval_callback import ForensicQuota

    # The floor of 1 is what keeps a heavily-sharded opponent from becoming a hole…
    assert ForensicQuota(5, 10, 5).per_shard(4) == (2, 3, 2)
    assert ForensicQuota(1, 1, 1).per_shard(8) == (1, 1, 1)
    # …but an explicit 0 means "capture none of these" and must SURVIVE the division, or the
    # floor would turn the search teacher's `win_quota=0` into one win per shard.
    assert ForensicQuota(0, 1000, 0).per_shard(4) == (0, 250, 0)
    # A negative is nonsense, not a smaller quota.
    assert ForensicQuota(-3, 10, 5).clamped() == (0, 10, 5)


def test_the_quota_survives_the_worker_process_boundary():
    """It crosses as a plain dict in the worker cfg; `coerce` is the only reader."""
    from agents.training.eval_callback import ForensicQuota

    raised = ForensicQuota(win=40, loss=40, draw=10)
    assert ForensicQuota.coerce(raised._asdict()) == raised
    # An OLDER parent sends no key at all. That is the DEFAULT, never zero.
    assert ForensicQuota.coerce(None) == ForensicQuota()
