"""Unit tests for the counterfactual label PRODUCER driver (`cf_producer.py`).

Pure and in-process: the priority arithmetic, the state file's crash-safety contract, the
throttle, the stale-trainer pause, the anchor refusal, and the ecology field every row must carry.

The REAL end-to-end composition — a bridge battle → the tap's ring → one producer cycle → the
REAL `CfLabelBuffer` — lives in `cf_producer_integration_test.py`, and it is the deliverable. These
tests exist to pin the arithmetic and the failure modes that are expensive to reach from there; a
producer-side unit test alone is exactly how the last two contract bugs shipped.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from agents.training import cf_producer as P


# ---------------------------------------------------------------------------
# The priority arithmetic
# ---------------------------------------------------------------------------

class TestPriorityScoring:
    def test_entropy_is_normalized_by_the_support_size(self):
        """A 2-way coin-flip must outrank a 9-way near-certainty, which raw entropy inverts."""
        coin = P.normalized_entropy([0.5, 0.5])
        wide = P.normalized_entropy([0.92] + [0.01] * 8)
        assert coin == pytest.approx(1.0)
        assert wide < coin, "un-normalized entropy would rank the 9-way state higher"

    def test_a_degenerate_decision_scores_zero_entropy(self):
        assert P.normalized_entropy([1.0]) == 0.0
        assert P.normalized_entropy([]) == 0.0
        assert P.normalized_entropy([1.0, 0.0, 0.0]) == 0.0

    def test_entropy_is_bounded_to_the_unit_interval(self):
        for k in (2, 3, 4, 11):
            assert P.normalized_entropy([1.0 / k] * k) == pytest.approx(1.0)

    def test_critic_surprise_is_the_conviction_region(self):
        # Sure of a win, and lost the battle: the "0.827 class" G0 measured at +0.23.
        assert P.critic_surprise(0.9, 0.0) == pytest.approx(0.9)
        # Sure of a win, and won: nothing to learn.
        assert P.critic_surprise(0.9, 1.0) == pytest.approx(0.1)

    def test_a_tie_is_half_not_a_loss(self):
        """A turn-cap draw is uninformative about conviction, not evidence the head was wrong."""
        assert P.critic_surprise(0.5, 0.5) == 0.0
        assert P.critic_surprise(1.0, 0.5) == pytest.approx(0.5)

    def test_no_win_prob_head_yields_no_surprise_term_rather_than_a_confident_zero(self):
        assert P.critic_surprise(None, 0.0) == 0.0
        assert P.critic_surprise(float("nan"), 0.0) == 0.0

    def test_surprise_dominates_the_declared_weighting(self):
        """The weights are a DECLARATION; this pins their ordering so a silent edit fails a test."""
        assert P.PRIORITY_WEIGHTS["critic_surprise"] > P.PRIORITY_WEIGHTS["policy_entropy"]
        # Max entropy cannot outrank a moderate surprise.
        assert P.priority_score(0.4, 0.0) > P.priority_score(0.0, 1.0)
        assert P.priority_score(0.5, 0.5) == pytest.approx(0.5 + 0.35 * 0.5)

    def test_sampler_version_is_stamped_and_stable(self):
        assert P.SAMPLER_VERSION == "cf_producer_priority_v1"


class TestMoveRoundFilter:
    def test_a_forced_switch_round_is_not_labelable(self):
        """The counterfactual divergence anchors at a start-of-turn MOVE round; a mask offering
        only switches is a mid-turn forced switch, which has no valid recorded answer to script."""
        switches_only = np.zeros(11, dtype=np.int8)
        switches_only[[1, 2, 3]] = 1
        assert not P.is_move_round(switches_only)

    def test_a_move_round_is_labelable(self):
        m = np.zeros(11, dtype=np.int8)
        m[[1, 6, 7]] = 1
        assert P.is_move_round(m)

    def test_struggle_counts_as_a_move_round(self):
        m = np.zeros(11, dtype=np.int8)
        m[10] = 1
        assert P.is_move_round(m)


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

class TestCheckpointResolution:
    def test_no_checkpoint_is_none_not_a_crash(self, tmp_path):
        assert P.resolve_latest_checkpoint(str(tmp_path)) is None

    def test_the_highest_step_wins_over_latest_txt(self, tmp_path):
        """`latest.txt` and the newest zip disagree exactly in the window between a checkpoint
        write and the pointer update; the higher step is the one whose weights are on disk."""
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        (ck / "checkpoint_100_steps.zip").write_text("a")
        (ck / "checkpoint_900_steps.zip").write_text("b")
        (tmp_path / "latest.txt").write_text("checkpoints/checkpoint_100_steps.zip")
        path, step = P.resolve_latest_checkpoint(str(tmp_path))
        assert step == 900 and path.endswith("checkpoint_900_steps.zip")

    def test_a_legacy_run_root_checkpoint_is_found(self, tmp_path):
        (tmp_path / "checkpoint_42_steps.zip").write_text("a")
        path, step = P.resolve_latest_checkpoint(str(tmp_path))
        assert step == 42

    def test_a_dangling_latest_txt_does_not_hide_the_glob(self, tmp_path):
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        (ck / "checkpoint_7_steps.zip").write_text("a")
        (tmp_path / "latest.txt").write_text("checkpoints/gone.zip")
        assert P.resolve_latest_checkpoint(str(tmp_path))[1] == 7


# ---------------------------------------------------------------------------
# The state file
# ---------------------------------------------------------------------------

class TestProducerState:
    def test_a_fresh_state_round_trips(self, tmp_path):
        st = P.ProducerState.load(str(tmp_path))
        st.claim("a_reconstruction.json")
        st.labels_total = 5
        st.save()
        again = P.ProducerState.load(str(tmp_path))
        assert again.is_processed("a_reconstruction.json")
        assert again.labels_total == 5
        assert again.sampler_version == P.SAMPLER_VERSION

    def test_the_state_file_is_human_readable(self, tmp_path):
        st = P.ProducerState.load(str(tmp_path))
        st.save()
        body = json.loads((tmp_path / P.STATE_FILENAME).read_text())
        # Indented, and it declares the sampler it was drawn under — the file IS the operator
        # surface (this process has no TensorBoard).
        assert "\n" in (tmp_path / P.STATE_FILENAME).read_text()
        assert body["sampler_version"] and body["sampler_weights"]

    def test_a_corrupt_state_file_starts_fresh_rather_than_crashing(self, tmp_path):
        (tmp_path / P.STATE_FILENAME).write_text("{not json")
        st = P.ProducerState.load(str(tmp_path))
        assert st.processed == [] and st.labels_total == 0

    def test_the_processed_set_is_bounded_oldest_first(self, tmp_path):
        st = P.ProducerState.load(str(tmp_path))
        for i in range(10):
            st.claim(f"r{i}", keep=4)
        assert len(st.processed) == 4
        assert st.is_processed("r9") and not st.is_processed("r0")

    def test_claim_is_idempotent(self, tmp_path):
        st = P.ProducerState.load(str(tmp_path))
        st.claim("x")
        st.claim("x")
        assert st.processed == ["x"]

    def test_an_unknown_key_in_a_newer_state_file_is_ignored(self, tmp_path):
        (tmp_path / P.STATE_FILENAME).write_text(json.dumps({"labels_total": 3, "future": 1}))
        st = P.ProducerState.load(str(tmp_path))
        assert st.labels_total == 3


# ---------------------------------------------------------------------------
# The cycle — with the bridge and the model both stubbed
# ---------------------------------------------------------------------------

def _args(run_dir, **over):
    base = dict(run_dir=str(run_dir), rollouts=2, top_n=2, records_per_cycle=4,
                max_labels_per_hour=2000, cycle_seconds=0.0, anchor_every=50,
                stale_checkpoint_minutes=90.0, lag_warn_steps=150_000,
                keep_processed=P.DEFAULT_KEEP_PROCESSED, impl="rust", device="cpu",
                compile_extractor=False, seed=1, cycles=1)
    base.update(over)
    return SimpleNamespace(**base)


class _StubSnapshot:
    """A Snapshot stand-in: scores deterministically, and its players are never asked to act
    because every bridge call is stubbed out in these tests."""

    def __init__(self, path, step, *, win_probs=None):
        self.path, self.step = path, step
        self.mappings = None
        self._wp = win_probs

    def score(self, obs, masks):
        n = len(obs)
        wp = None if self._wp is None else np.asarray(self._wp[:n], dtype=float)
        return wp, np.linspace(0.1, 0.9, n)

    def make_player(self, record, side, *, role):
        return SimpleNamespace(role=role, side=side)


def _mk_run(tmp_path, *, step=1000):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / f"checkpoints/checkpoint_{step}_steps.zip").write_text("x")
    (run / P.RECORDS_DIRNAME).mkdir()
    return run


def _mk_record(run, name="0000000000000000001_1_tag_reconstruction.json"):
    p = run / P.RECORDS_DIRNAME / name
    p.write_text(json.dumps({"v": 1, "format_id": "gen3ou", "prng_seed": "1,2,3,4",
                             "input_log": [], "commands": []}))
    return p


class TestCycle:
    def test_no_checkpoint_yet_waits_politely(self, tmp_path):
        run = tmp_path / "run"
        (run / P.RECORDS_DIRNAME).mkdir(parents=True)
        prod = P.CfProducer(_args(run), snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        assert prod.cycle() == 0
        assert "waiting for a checkpoint" in prod.heartbeat
        assert prod.snapshot is None

    def test_a_record_is_claimed_before_its_work_so_a_crash_cannot_double_label(self, tmp_path):
        """Crash safety, measured on the ORDER rather than argued: when `process_record` raises,
        the record must ALREADY be marked processed and persisted."""
        run = _mk_run(tmp_path)
        _mk_record(run)
        prod = P.CfProducer(_args(run), snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.run_anchor = lambda: True                              # type: ignore[assignment]
        seen = {}

        def _boom(path):
            seen["claimed_at_call_time"] = P.ProducerState.load(str(run)).is_processed(
                os.path.basename(path))
            raise RuntimeError("crash mid-record")

        prod.process_record = _boom                                 # type: ignore[assignment]
        assert prod.cycle() == 0
        assert seen["claimed_at_call_time"] is True, (
            "the state file must be durable BEFORE the rollouts run, or a crash re-labels")
        assert P.ProducerState.load(str(run)).skip_reasons.get("error:RuntimeError") == 1

    def test_a_second_cycle_over_the_same_ring_produces_no_duplicate_labels(self, tmp_path):
        """State-file idempotence: re-running over an unchanged ring is a no-op."""
        run = _mk_run(tmp_path)
        _mk_record(run)
        calls = []

        def _one(path):
            calls.append(path)
            return [{"schema": 1, "kind": "mc_winprob", "label": 0.5, "policy_step": 1000,
                     "obs_inline": "", "obs_sha1": "x", "n_rollouts": 2}]

        prod = P.CfProducer(_args(run), snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.run_anchor = lambda: True                              # type: ignore[assignment]
        prod.process_record = _one                                  # type: ignore[assignment]
        prod.cycle()
        prod.cycle()
        assert len(calls) == 1, "the second cycle re-processed a record it had already done"
        files = os.listdir(run / P.LABELS_DIRNAME)
        assert len(files) == 1

        # ...and a FRESH process (state read back off disk) agrees.
        prod2 = P.CfProducer(_args(run), snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod2.run_anchor = lambda: True                             # type: ignore[assignment]
        prod2.process_record = _one                                 # type: ignore[assignment]
        prod2.cycle()
        assert len(calls) == 1

    def test_the_throttle_stops_production_without_stopping_the_loop(self, tmp_path):
        run = _mk_run(tmp_path)
        _mk_record(run)
        prod = P.CfProducer(_args(run, max_labels_per_hour=3),
                            snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.run_anchor = lambda: True                              # type: ignore[assignment]
        prod.label_times.extend([time.time()] * 3)
        called = []
        prod.process_record = lambda path: called.append(path) or []  # type: ignore[assignment]
        assert prod.cycle() == 0
        assert called == [], "the throttle must stop the record loop, not just log"
        assert "PRODUCING" in prod.heartbeat

    def test_the_throttle_window_is_one_hour_and_slides(self, tmp_path):
        run = _mk_run(tmp_path)
        prod = P.CfProducer(_args(run, max_labels_per_hour=2),
                            snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.label_times.extend([time.time() - 4000, time.time() - 3800])
        assert prod._rate_per_hour() == 0.0, "labels older than an hour must fall out"
        assert not prod._throttled()

    def test_a_stale_trainer_pauses_production_and_says_so_once(self, tmp_path, capsys):
        run = _mk_run(tmp_path)
        _mk_record(run)
        prod = P.CfProducer(_args(run, stale_checkpoint_minutes=1.0),
                            snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.run_anchor = lambda: True                              # type: ignore[assignment]
        called = []
        prod.process_record = lambda path: called.append(path) or []  # type: ignore[assignment]
        prod.refresh_snapshot()
        prod._last_new_ckpt_unix = time.time() - 600                # 10 min of no new checkpoint
        prod.cycle()
        prod.cycle()
        out = capsys.readouterr().out
        assert called == [], "a paused producer must not fill the buffer with stale-policy labels"
        assert out.count("NO new checkpoint") == 1, "the pause must announce itself exactly once"
        assert "PAUSED" in prod.heartbeat

    def test_a_new_checkpoint_resumes_a_paused_producer(self, tmp_path):
        run = _mk_run(tmp_path)
        prod = P.CfProducer(_args(run, stale_checkpoint_minutes=1.0),
                            snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.run_anchor = lambda: True                              # type: ignore[assignment]
        prod.refresh_snapshot()
        prod._last_new_ckpt_unix = time.time() - 600
        prod.cycle()
        assert prod._producing is False
        (run / "checkpoints/checkpoint_2000_steps.zip").write_text("y")
        prod.cycle()
        assert prod._producing is True and prod.snapshot.step == 2000

    def test_stale_checking_can_be_disabled(self, tmp_path):
        run = _mk_run(tmp_path)
        prod = P.CfProducer(_args(run, stale_checkpoint_minutes=0.0),
                            snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.refresh_snapshot()
        prod._last_new_ckpt_unix = 0.0
        prod._check_trainer_alive()
        assert prod._producing is True


class TestAnchorRefusal:
    def test_a_failed_anchor_exits_three_and_writes_no_labels(self, tmp_path):
        run = _mk_run(tmp_path)
        _mk_record(run)
        prod = P.CfProducer(_args(run), snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.run_anchor = lambda: False                             # type: ignore[assignment]
        prod.process_record = lambda path: pytest.fail(             # type: ignore[assignment]
            "no record may be processed after a failed anchor")
        assert prod.cycle() == 3
        assert prod.anchor_failed is True
        assert not os.path.isdir(run / P.LABELS_DIRNAME)

    def test_the_anchor_runs_before_the_first_record_not_after(self, tmp_path):
        run = _mk_run(tmp_path)
        _mk_record(run)
        order = []
        prod = P.CfProducer(_args(run), snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.run_anchor = lambda: order.append("anchor") or True    # type: ignore[assignment]
        prod.process_record = lambda path: order.append("record") or []  # type: ignore[assignment]
        prod.cycle()
        assert order == ["anchor", "record"]

    def test_an_anchor_exception_counts_as_a_failure_not_a_pass(self, tmp_path):
        """An anchor that CRASHED proves nothing about the replay's exactness, so it must not be
        allowed to read as a pass — the same rule that makes a TIMEOUT its own bucket."""
        run = _mk_run(tmp_path)
        rec = _mk_record(run)
        rec.write_text("{ not a record")
        prod = P.CfProducer(_args(run), snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.refresh_snapshot()
        assert prod.run_anchor() is False
        assert prod.state.anchors_run == 1 and prod.state.anchors_reproduced == 0

    def test_an_empty_ring_cannot_anchor_and_says_none(self, tmp_path):
        run = _mk_run(tmp_path)
        prod = P.CfProducer(_args(run), snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.refresh_snapshot()
        assert prod.run_anchor() is None
        assert prod.cycle() == 0, "no record to anchor on is not an anchor FAILURE"

    def test_the_anchor_repeats_on_the_declared_cadence(self, tmp_path):
        run = _mk_run(tmp_path)
        for i in range(4):
            _mk_record(run, f"000000000000000000{i}_1_t_reconstruction.json")
        runs = []
        prod = P.CfProducer(_args(run, anchor_every=2, records_per_cycle=2),
                            snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.run_anchor = lambda: (runs.append(1),                  # type: ignore[assignment]
                                   setattr(prod.state, "records_since_anchor", 0),
                                   prod.state.__setattr__("anchors_run",
                                                          prod.state.anchors_run + 1), True)[-1]
        prod.process_record = lambda path: []                       # type: ignore[assignment]
        prod.cycle()      # startup anchor, then 2 records
        prod.cycle()      # 2 records since the anchor -> anchor again
        assert len(runs) == 2


# ---------------------------------------------------------------------------
# The label rows
# ---------------------------------------------------------------------------

class TestLabelRows:
    def _row(self, **over):
        from agents.training.obs_materializer import RecordDecision
        obs = np.arange(8, dtype=np.float32)
        d = RecordDecision(index=3, turn=11, action=7, choice="move icebeam",
                           mask=np.ones(11, dtype=np.int8), obs=obs)
        kw = dict(record_path="/r/x_reconstruction.json", decision=d, wins=5, n=8,
                  step=24_000_000, surprise=0.4, entropy=0.6, score=0.61, win_prob=0.8)
        kw.update(over)
        return P.label_row(**kw)

    def test_every_row_names_the_ecology_it_was_measured_in(self):
        """THE ECOLOGY DECISION, enforced: a training record names no opponent, so the row must
        say `self_current` and never a bot name it cannot verify."""
        r = self._row()
        assert r["opponent"] == P.OPPONENT_LABEL == "self_current"
        assert r["label_regime"] == "self_current_stochastic_both_sides"

    def test_the_row_is_the_shared_v1_schema(self):
        r = self._row()
        assert r["schema"] == 1 and r["kind"] == "mc_winprob"
        assert r["label"] == pytest.approx(5 / 8)
        assert r["n_rollouts"] == 8 and r["policy_step"] == 24_000_000
        assert r["wilson_lo"] <= r["label"] <= r["wilson_hi"]
        assert r["obs_npz"] is None and r["obs_inline"]

    def test_the_obs_digest_is_of_the_bytes_actually_shipped(self):
        import base64
        r = self._row()
        arr = np.frombuffer(base64.b64decode(r["obs_inline"]), dtype=np.float32)
        from agents.training.cf_audit import obs_digest
        assert obs_digest(arr) == r["obs_sha1"]

    def test_the_sampler_and_its_score_ride_on_every_row(self):
        r = self._row()
        assert r["sampler_version"] == P.SAMPLER_VERSION
        assert r["priority"]["critic_surprise"] == pytest.approx(0.4)
        assert r["priority"]["win_prob"] == pytest.approx(0.8)

    def test_a_headless_checkpoints_rows_say_win_prob_none_not_zero(self):
        assert self._row(win_prob=None)["priority"]["win_prob"] is None

    def test_a_batch_is_a_new_file_written_atomically(self, tmp_path):
        rows = [self._row(), self._row()]
        p1 = P.write_label_batch(str(tmp_path), rows, step=100, seq=1)
        p2 = P.write_label_batch(str(tmp_path), rows, step=100, seq=2)
        assert os.path.basename(p1) == "labels_cf_producer_100_1.jsonl"
        assert p1 != p2, "a batch must never reuse a name — the buffer keys offsets on (name,inode)"
        assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert len(open(p1).read().strip().split("\n")) == 2


class TestTraineeSide:
    def test_a_training_record_defaults_to_p1(self):
        """BridgeSession seats env.agent1 — the trainee — on p1, always. A training record names
        no trainee, so that transport invariant is the only thing that can answer."""
        from utils.bridge.reconstruction import ReconstructionRecord
        rec = ReconstructionRecord(format_id="gen3ou", prng_seed="1,2,3,4",
                                   input_log=(), commands=())
        assert P._trainee_side(rec) == "p1"

    def test_a_record_that_names_a_trainee_is_honoured(self):
        from utils.bridge.reconstruction import ReconstructionRecord
        rec = ReconstructionRecord(
            format_id="gen3ou", prng_seed="1,2,3,4",
            input_log=('>player p1 {"name": "A", "team": ""}',
                       '>player p2 {"name": "B", "team": ""}'),
            commands=(), trainee_username="B")
        assert P._trainee_side(rec) == "p2"


class TestCli:
    def test_a_run_without_the_tap_refuses_rather_than_idling(self, tmp_path, capsys):
        (tmp_path / "run").mkdir()
        rc = P.main([str(tmp_path / "run")])
        assert rc == 2
        assert "--cf-records" in capsys.readouterr().err

    def test_a_missing_run_dir_refuses(self, tmp_path):
        assert P.main([str(tmp_path / "nope")]) == 2

    def test_every_help_string_renders(self):
        """An unescaped `%` in a help string is a TypeError nobody sees until --help is typed —
        the exact bug `checkargs_test` was written for."""
        P.build_parser().format_help()

    def test_the_defaults_are_the_declared_ones(self):
        a = P.build_parser().parse_args(["/tmp/x"])
        assert a.rollouts == 8 and a.top_n == 3 and a.impl == "rust"
        assert a.max_labels_per_hour == 2000 and a.anchor_every == 50
        assert a.lag_warn_steps == 150_000, "must match the buffer's DEFAULT_LAG_BOUND"

    def test_the_lag_warning_default_tracks_the_buffers_bound(self):
        from agents.training.cf_label_buffer import DEFAULT_LAG_BOUND
        assert P.build_parser().parse_args(["/tmp/x"]).lag_warn_steps == DEFAULT_LAG_BOUND
