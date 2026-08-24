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

    def test_a_forced_checkpoints_step_parses(self):
        """SIGUSR1 writes `checkpoint_forced_<step:010d>_<HHMMSS>.zip` — a resumable checkpoint
        under a second name. Reading only the periodic form makes its step unparseable, and an
        unparseable step ranks BELOW every periodic zip in `_key`."""
        assert P.step_from_checkpoint_name("checkpoint_forced_0000060000_120000.zip") == 60000
        assert P.step_from_checkpoint_name("checkpoint_50000_steps.zip") == 50000
        assert P.step_from_checkpoint_name("final_model.zip") is None

    def test_a_NEWER_forced_checkpoint_beats_an_older_periodic_one(self, tmp_path):
        """The regression: an operator hits the launcher's `c` (force checkpoint) after the last
        periodic save. Before the fix the producer resolved the OLDER periodic zip and went on
        stamping its step — silently labelling against a snapshot it had already moved past."""
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        (ck / "checkpoint_50000_steps.zip").write_text("a")
        (ck / "checkpoint_forced_0000060000_120000.zip").write_text("b")
        (tmp_path / "latest.txt").write_text(
            "checkpoints/checkpoint_forced_0000060000_120000.zip")
        path, step = P.resolve_latest_checkpoint(str(tmp_path))
        assert step == 60000, "a newer FORCED checkpoint must outrank an older periodic one"
        assert path.endswith("checkpoint_forced_0000060000_120000.zip")

    def test_a_forced_checkpoint_is_found_without_latest_txt(self, tmp_path):
        """It must be reachable by the GLOB too, not only through the pointer file — `latest.txt`
        is written after the zip, so there is a window in which it names the previous save."""
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        (ck / "checkpoint_forced_0000012288_091921.zip").write_text("a")
        assert P.resolve_latest_checkpoint(str(tmp_path))[1] == 12288

    def test_an_older_forced_checkpoint_still_loses_to_a_newer_periodic_one(self, tmp_path):
        """The other direction, so the fix is a step comparison and not a name preference."""
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        (ck / "checkpoint_forced_0000040448_092309.zip").write_text("a")
        (ck / "checkpoint_50000_steps.zip").write_text("b")
        path, step = P.resolve_latest_checkpoint(str(tmp_path))
        assert step == 50000 and path.endswith("checkpoint_50000_steps.zip")


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


class TestProducerRetentionRace:
    """The trainer's `--cf-records-keep` ring deletes records this process is still working from.

    MEASURED on `ai_v9_29_rev1_0823`: 176 records lost to `FileNotFoundError` across 67 cycles,
    with "538 pending" against a ring of 512 — an excess that is a guaranteed loss by arithmetic,
    because the loop walked the OLD end of the ring, which is the end the ring deletes from.
    """

    def test_a_record_deleted_after_enumeration_is_a_counted_skip_not_an_error(self, tmp_path):
        """The ring deletes between the listdir and the read. That is an ordinary outcome of two
        processes sharing one directory — it must never look like a corrupt record or a crash."""
        run = _mk_run(tmp_path)
        _mk_record(run, name="0000000000000000001_1_a_reconstruction.json")
        _mk_record(run, name="0000000000000000002_1_b_reconstruction.json")
        prod = P.CfProducer(_args(run), snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.run_anchor = lambda: True                              # type: ignore[assignment]
        real_enum = prod._pending_records

        def _enumerate_then_the_ring_deletes():
            out = real_enum()
            os.remove(out[-1])           # the OLDEST — exactly what the ring prunes
            return out

        prod._pending_records = _enumerate_then_the_ring_deletes    # type: ignore[assignment]
        done = []
        prod.process_record = lambda path: done.append(path) or []  # type: ignore[assignment]

        assert prod.cycle() == 0, "a vanished record must never be an exception path"
        st = P.ProducerState.load(str(run))
        assert st.records_vanished == 1
        assert not any(k.startswith("error:") for k in st.skip_reasons), st.skip_reasons
        assert [os.path.basename(p) for p in done] == \
            ["0000000000000000002_1_b_reconstruction.json"], "the survivor must still be processed"
        assert "1 vanished" in prod.heartbeat, "ring pressure must be visible on the heartbeat"

    def test_a_vanished_record_is_claimed_so_it_is_never_retried(self, tmp_path):
        run = _mk_run(tmp_path)
        p = _mk_record(run)
        prod = P.CfProducer(_args(run), snapshot_loader=lambda p_, s: _StubSnapshot(p_, s))
        prod.run_anchor = lambda: True                              # type: ignore[assignment]
        real_enum = prod._pending_records
        prod._pending_records = lambda: (real_enum(), os.remove(p))[0]  # type: ignore[assignment]
        prod.cycle()
        assert P.ProducerState.load(str(run)).is_processed(os.path.basename(p))

    def test_records_are_taken_NEWEST_first(self, tmp_path):
        """The ring deletes from the OLD end, so the oldest pending record is the one already
        promised away. Working it first is how ~90% of the yield was lost inside the window."""
        run = _mk_run(tmp_path)
        for i in (1, 2, 3):
            _mk_record(run, name=f"000000000000000000{i}_1_t_reconstruction.json")
        prod = P.CfProducer(_args(run, records_per_cycle=1),
                            snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.run_anchor = lambda: True                              # type: ignore[assignment]
        done = []
        prod.process_record = lambda path: done.append(path) or []  # type: ignore[assignment]
        prod.cycle()
        assert [os.path.basename(p) for p in done] == \
            ["0000000000000000003_1_t_reconstruction.json"]

    def test_the_record_is_read_at_enumeration_time_not_at_process_time(self, tmp_path):
        """The gap the race wins is enumerate → anchor → claim+fsync → open. Deleting the file
        AFTER the batch is loaded must not disturb the record already in hand."""
        run = _mk_run(tmp_path)
        p = _mk_record(run)
        prod = P.CfProducer(_args(run), snapshot_loader=lambda p_, s: _StubSnapshot(p_, s))
        alive = prod._load_batch([str(p)])
        os.remove(p)                                     # the ring, mid-cycle
        assert alive == [str(p)]
        assert prod._preloaded[str(p)].format_id == "gen3ou", (
            "the record must survive the deletion of its file")

    def test_an_anchor_whose_record_vanishes_is_NOT_an_anchor_failure(self, tmp_path):
        """An anchor failure exits 3 and stops the factory. A ring deletion must not be able to
        do that — the anchor simply had no record to adjudicate this cycle."""
        run = _mk_run(tmp_path)
        ghost = str(run / P.RECORDS_DIRNAME / "0000000000000000009_1_g_reconstruction.json")
        prod = P.CfProducer(_args(run), snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.refresh_snapshot()
        prod._newest_record = lambda: ghost                         # type: ignore[assignment]
        assert prod.run_anchor() is None
        assert prod.state.anchors_errored == 0 and prod.state.anchors_run == 0
        assert prod.state.records_vanished == 1


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

    def test_an_anchor_ERROR_is_counted_and_reported_apart_from_a_MISMATCH(self, tmp_path):
        """The two refusals have OPPOSITE diagnoses, so they must not print the same sentence.

        A mismatch says the replay is inexact. An exception (a wedged bridge child, a transport
        error, a contention `ProgressTimeout`) says nothing whatever about exactness — it never
        returned a verdict. The producer printed the MISMATCH text for both until 2026-08-23, and
        that turned ONE flaky `cf_producer_integration_test` failure into an investigation of a
        replay-exactness gap that did not exist. `cf_audit` has always kept `anchor_errors` apart.
        """
        run = _mk_run(tmp_path)
        rec = _mk_record(run)
        rec.write_text("{ not a record")
        prod = P.CfProducer(_args(run), snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.refresh_snapshot()
        assert prod.run_anchor() is False
        assert prod.state.anchors_errored == 1, "an ERRORED anchor was not counted as one"
        assert prod.anchor_error and "JSONDecode" in prod.anchor_error
        assert prod.anchor_mismatch is None, "an exception is not a mismatch"
        # the state file (the whole operator surface — there is no TensorBoard here) carries it
        prod.state.save()
        assert json.loads(rec.parent.parent.joinpath(P.STATE_FILENAME).read_text())[
            "anchors_errored"] == 1

        msg = P.anchor_refusal_message(error=prod.anchor_error, mismatch=None, state_path="/s")
        assert "COULD NOT RUN" in msg
        assert "did not reproduce the recorded outcome" not in msg, (
            "an anchor that never ran must not be reported as a replay-exactness failure")

    def test_an_anchor_TIMEOUT_self_diagnoses_instead_of_accusing_the_replay(self):
        """A timeout is never a semantic outcome (root CLAUDE.md). If the anchor died of one on a
        starved box, the message must say so and print the load average — the same rule the
        per-battle bridge bound and `bridge_impl_parity_test` already follow."""
        msg = P.anchor_refusal_message(
            error="ProgressTimeout: no progress for 30.0s", mismatch=None, state_path="/s")
        assert "COULD NOT RUN" in msg and "contention: load average" in msg
        assert "the replay is not exact" not in msg
        # a NON-timeout error is still an error, but must not claim contention it did not measure
        other = P.anchor_refusal_message(
            error="RuntimeError: boom", mismatch=None, state_path="/s")
        assert "COULD NOT RUN" in other and "contention: load average" not in other

    def test_a_MISMATCH_still_names_the_defect_and_shows_both_outcomes(self):
        msg = P.anchor_refusal_message(
            error=None, mismatch=("win", "Bob", 1.0, 0.0, []), state_path="/s")
        assert "ANCHOR MISMATCH" in msg and "the replay is not exact" in msg
        assert "'win'" in msg and "'Bob'" in msg, (
            "a refusal that does not print WHAT disagreed cannot be diagnosed from the log alone")
        assert "RAN OUT" not in msg, "no side was exhausted; the message must not say one was"
        withx = P.anchor_refusal_message(
            error=None, mismatch=("win", "Bob", 1.0, 0.0, ["p2"]), state_path="/s")
        assert "RAN OUT" in withx and "'p2'" in withx

    def test_a_full_replay_that_RAN_OUT_of_script_fails_the_anchor_even_on_a_matching_winner(
            self, tmp_path, monkeypatch):
        """The SENSITIVE half of the oracle.

        A `divergence_turn=None` replay scripts EVERY decision of both sides, so a side that runs
        out of recorded commands and finishes on the live policy has already diverged from the
        recording — whatever the winner turned out to be. Comparing outcomes alone catches that
        only when the random fallback happens to flip the result, i.e. about half the time. The
        2026-08-23 hunt for an intermittent anchor refusal needed a detector that fires on the
        divergence rather than on its coin flip. Honest scope: it did NOT catch the class that hunt
        found (a forfeit race, which consumes no script), and it was empty on all 274 instrumented
        healthy replays — so the stricter check costs a correct run nothing and has never fired.
        """
        run = _mk_run(tmp_path)
        p = run / P.RECORDS_DIRNAME / "0000000000000000002_1_tag_reconstruction.json"
        p.write_text(json.dumps({
            "v": 1, "format_id": "gen3ou", "prng_seed": "1,2,3,4", "commands": [],
            "input_log": ['>start {"formatid":"gen3ou","seed":[1,2,3,4]}',
                          '>player p1 {"name":"Ann","team":""}',
                          '>player p2 {"name":"Bob","team":""}']}))
        prod = P.CfProducer(_args(run), snapshot_loader=lambda p_, s: _StubSnapshot(p_, s))
        prod.refresh_snapshot()
        res = {}
        monkeypatch.setattr(
            P, "replay_battle", lambda record, impl="node": SimpleNamespace(
                outcome={"winner": "Ann"}))            # the record says the TRAINEE (p1) won
        monkeypatch.setattr(P, "_run_one", lambda record, **kw: dict(res))

        # 1) the winner MATCHES and no side was exhausted → the anchor passes
        res.update(outcome="win", script_exhausted=[])
        assert prod.run_anchor() is True

        # 2) the SAME matching winner, but a side ran out of script → it must NOT pass
        res.update(outcome="win", script_exhausted=["p2"])
        assert prod.run_anchor() is False, (
            "a full replay that fell through to a live policy was accepted as exact")
        assert prod.anchor_mismatch is not None and prod.anchor_mismatch[4] == ["p2"]
        assert prod.state.anchors_errored == 0, "an exhausted script is not an ERROR"

    def test_a_FORFEIT_record_is_not_adjudicable_and_the_anchor_skips_it(self, tmp_path, capsys):
        """The class behind the 2026-08-23 intermittent `ANCHOR REFUSED`.

        A battle that hits the 250-turn cap ends with `['forcelose', <side>]` in
        `record.commands`, which `install_scripted_prefix` drops (it keys the script on
        `s == side`). Both scripted players then re-derive a forfeit from their own
        `_handle_stall` and the winner becomes a race — measured 7/12 and 8/12 refusals
        re-anchoring ONE such record, against 12/12 correct once the opponent's stall threshold was
        made unreachable (the mechanism proof). It is a faithfulness limit of live scripted replay,
        not a defect the anchor can report, so the class is EXCLUDED — visibly and by count, never
        by retrying the same record until it passes.
        """
        run = _mk_run(tmp_path)
        older = _mk_record(run, "0000000000000000001_1_a_reconstruction.json")
        newer = run / P.RECORDS_DIRNAME / "0000000000000000009_1_b_reconstruction.json"
        newer.write_text(json.dumps({
            "v": 1, "format_id": "gen3ou", "prng_seed": "1,2,3,4", "input_log": [],
            "commands": [["p1", "move 1"], ["p2", "move 1"], ["forcelose", "p1"]]}))

        from utils.bridge.reconstruction import ReconstructionRecord
        rec = ReconstructionRecord.load(str(newer))
        assert P.record_is_full_replay_anchorable(rec) is False
        assert P.record_is_full_replay_anchorable(ReconstructionRecord.load(str(older))) is True

        prod = P.CfProducer(_args(run), snapshot_loader=lambda p_, s: _StubSnapshot(p_, s))
        assert prod._newest_record() == str(older), (
            "the anchor took the forfeit-terminated record it cannot adjudicate")
        assert prod.state.anchors_skipped_unanchorable == 1
        assert "FORFEIT" in capsys.readouterr().out, (
            "the skip must ANNOUNCE itself — a silent coverage bound is an invisible one")

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

    def test_the_row_carries_head_Bs_SINGLE_OUTCOME_stream(self):
        """gen3_cf_twin_heads_v1. `outcome_label` is the RECORDED battle's realized outcome — the
        SAME quantity the on-policy BCE eats, on the states the sampler selected. That identity is
        what makes B−A a read of COVERAGE alone; a row that shipped anything else there would make
        the contrast measure two things at once."""
        assert self._row(outcome_label=0.0)["outcome_label"] == 0.0
        assert self._row(outcome_label=1.0)["outcome_label"] == 1.0
        assert self._row(outcome_label=0.5)["outcome_label"] == 0.5      # the turn cap / a tie
        # ABSENT is a first-class value: an older consumer ignores it, a newer one supervises
        # nothing extra rather than being handed a fabricated 0.
        assert self._row()["outcome_label"] is None

    def test_the_mc_return_stream_is_written_ONLY_when_it_was_measured(self):
        """A `null` and an absent key mean the same thing to the buffer — but writing the reward
        provenance unconditionally would imply a measurement that was not taken, and the digest is
        the one field a reader uses to decide whether the number is theirs."""
        bare = self._row()
        assert "mc_return" not in bare and "reward_sha1" not in bare
        got = self._row(mc_return=-2.5, mc_return_n=8, reward_sha1="deadbeef",
                        reward_composition="1 TERMINAL + 7 PBRS + 1 BIAS (x)")
        assert got["mc_return"] == pytest.approx(-2.5) and got["mc_return_n"] == 8
        assert got["reward_sha1"] == "deadbeef"
        # The human line rides beside the digest for the reason the launch banner exists: a hex
        # digest says two rewards DIFFER and nothing about how.
        assert "PBRS" in got["reward_composition"]

    def test_the_new_streams_do_not_disturb_the_v1_schema_version(self):
        """`schema` is a REFUSAL gate: a consumer skips every row whose version it does not know.
        Bumping it for additive-optional fields would make a new producer's output unreadable by an
        existing trainer — the opposite of backward compatible."""
        assert self._row(outcome_label=1.0, mc_return=1.0, mc_return_n=4,
                         reward_sha1="x")["schema"] == 1

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


# ---------------------------------------------------------------------------
# The throughput contract: ONE compiled signature, and independent rollout arms
# ---------------------------------------------------------------------------

class _RecordingPolicy:
    """A policy stand-in that REMEMBERS the shape and dtype of every forward it was handed.

    The whole point of the scoring changes is *which signature* reaches a compiled graph, so the
    test subject is the call record, not the numbers — and the numbers are checked too, because a
    chunked forward that changes the ranking would be a silent sampler change.
    """

    def __init__(self, *, win_head: bool = True) -> None:
        self.calls: list = []
        self.features_extractor = SimpleNamespace(last_win_prob_logits=None)
        self._win_head = win_head

    def get_distribution(self, obs):
        import torch as th
        o, m = obs["observation"], obs["action_mask"]
        self.calls.append({"n": int(o.shape[0]), "obs_dtype": o.dtype, "mask_dtype": m.dtype})
        # Deterministic per-row logits so a chunked pass and a batched one are comparable.
        logits = th.arange(o.shape[0], dtype=th.float32).unsqueeze(1) + th.arange(
            m.shape[1], dtype=th.float32).unsqueeze(0)
        if self._win_head:
            self.features_extractor.last_win_prob_logits = th.zeros(o.shape[0], 1)
        return SimpleNamespace(distribution=SimpleNamespace(logits=logits))


def _snapshot_with(policy, *, compiled: bool):
    return P.Snapshot("/ckpt.zip", 7, SimpleNamespace(policy=policy), None, compiled=compiled)


class TestScoreForwardSignature:
    def _inputs(self, n=5):
        obs = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
        masks = np.ones((n, 3), dtype=np.int8)
        masks[:, 2] = 0
        return obs, masks

    def test_a_compiled_snapshot_scores_ONE_ROW_AT_A_TIME(self):
        """B=1 is the shape every rollout forwards at; a batched score would force a SECOND trace.

        Measured 2026-08-23 on the live checkpoint: with a batched score in front of them, the
        first label's rollouts cost 79 s against 3 s for the second — pure recompilation."""
        pol = _RecordingPolicy()
        obs, masks = self._inputs(5)
        _snapshot_with(pol, compiled=True).score(obs, masks)
        assert [c["n"] for c in pol.calls] == [1, 1, 1, 1, 1]

    def test_an_eager_snapshot_still_takes_the_single_batched_forward(self):
        """There is no graph to keep one signature for, so the cheap path stays the cheap path."""
        pol = _RecordingPolicy()
        obs, masks = self._inputs(5)
        _snapshot_with(pol, compiled=False).score(obs, masks)
        assert [c["n"] for c in pol.calls] == [5]

    def test_the_mask_reaches_the_graph_as_float32_not_int8(self):
        """A materialized mask is int8 and a live one is float32 — and dynamo guards on DTYPE as
        hard as on shape. That mismatch measured a 19.5 s re-trace on the first scored row."""
        import torch as th
        for compiled in (True, False):
            pol = _RecordingPolicy()
            obs, masks = self._inputs(3)
            assert masks.dtype == np.int8
            _snapshot_with(pol, compiled=compiled).score(obs, masks)
            assert {c["mask_dtype"] for c in pol.calls} == {th.float32}
            assert {c["obs_dtype"] for c in pol.calls} == {th.float32}

    def test_chunking_does_not_change_a_single_number(self):
        """The sampler ranks on these values, so the two paths must agree exactly."""
        obs, masks = self._inputs(6)
        wp_c, ent_c = _snapshot_with(_RecordingPolicy(), compiled=True).score(obs, masks)
        wp_e, ent_e = _snapshot_with(_RecordingPolicy(), compiled=False).score(obs, masks)
        assert np.allclose(ent_c, ent_e)
        assert np.allclose(wp_c, wp_e)

    def test_a_headless_checkpoint_reports_no_win_probs_through_either_path(self):
        for compiled in (True, False):
            obs, masks = self._inputs(4)
            wp, ent = _snapshot_with(
                _RecordingPolicy(win_head=False), compiled=compiled).score(obs, masks)
            assert wp is None and len(ent) == 4


class TestCompiledGraphWarmUp:
    def test_a_warm_up_that_raises_is_survivable_and_says_so(self, capsys):
        """It is a perf warm-up, not a gate: a model whose spaces are not what we assumed must
        cost the warm-up and nothing else."""
        broken = SimpleNamespace(observation_space={}, policy=SimpleNamespace())
        assert P._warm_the_compiled_graph(broken) >= 0.0
        assert "warm-up skipped" in capsys.readouterr().out

    def test_it_forwards_the_LIVE_signature_both_keys(self):
        """`maybe_compile_extractor` warms with `observation` ALONE; every real call also carries
        `action_mask`, and a dict's KEY SET is part of the guard — so warming with one key leaves
        the first real decision to re-trace (19.5 s, measured)."""
        seen = {}

        class _Space:
            def __init__(self, n):
                self.shape = (n,)

        def _get_distribution(obs):
            seen["keys"] = sorted(obs)
            seen["shapes"] = {k: tuple(v.shape) for k, v in obs.items()}

        model = SimpleNamespace(
            observation_space={"observation": _Space(9), "action_mask": _Space(11)},
            policy=SimpleNamespace(get_distribution=_get_distribution))
        P._warm_the_compiled_graph(model)
        assert seen["keys"] == ["action_mask", "observation"]
        assert seen["shapes"] == {"observation": (1, 9), "action_mask": (1, 11)}


def _seed_is_even(seed) -> bool:
    """A stable win/loss rule over a `fresh_seeds` entry (a `"a,b,c,d"` gen-5 seed string)."""
    return sum(int(x) for x in str(seed).split(",")) % 2 == 0


def _rollout_record():
    from utils.bridge.reconstruction import ReconstructionRecord
    return ReconstructionRecord(
        format_id="gen3ou", prng_seed="1,2,3,4",
        input_log=('>start {"formatid":"gen3ou","seed":[1,2,3,4]}',
                   '>player p1 {"name":"Ann","team":"x"}',
                   '>player p2 {"name":"Bob","team":"y"}'),
        commands=())


def _decision():
    from agents.training.obs_materializer import RecordDecision
    return RecordDecision(index=4, turn=6, action=7, choice="move icebeam",
                          mask=np.ones(11, dtype=np.int8), obs=np.zeros(4, dtype=np.float32))


class TestRolloutArms:
    """`_rollout` is 93% of this process's wall (measured), so its arms are the thing to get right:
    independent, aggregated by a SUM, and individually survivable."""

    def _prod(self, tmp_path, **over):
        run = _mk_run(tmp_path)
        prod = P.CfProducer(_args(run, rollouts=6, mc_return=False, **over),
                            snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.refresh_snapshot()
        return prod

    @pytest.mark.parametrize("conc", [1, 4])
    def test_the_arms_aggregate_to_the_same_label_sequential_or_overlapped(
            self, tmp_path, monkeypatch, conc):
        """Overlapping the arms is a SCHEDULING change: wins/n are order-free counts."""
        monkeypatch.setattr(P, "_run_one", lambda record, **kw: {
            "outcome": "win" if _seed_is_even(kw["post_t_seed"]) else "loss"})
        prod = self._prod(tmp_path, rollout_concurrency=conc)
        wins, n, returns = prod._rollout(_rollout_record(), "p1", _decision(), tag="t")
        assert n == 6 and returns == []
        assert wins == self._expected_wins(prod)

    def _expected_wins(self, prod):
        from main.prober.falsifier import fresh_seeds
        return sum(_seed_is_even(s)
                   for s in fresh_seeds(prod.args.rollouts, salt=f"t:4:cfp{prod.args.seed}"))

    @pytest.mark.parametrize("conc", [1, 4])
    def test_one_arm_that_DIES_costs_that_arm_and_nothing_else(
            self, tmp_path, monkeypatch, conc, capsys):
        """The bridge child is spawned per arm, so one can die (a transport error, a wedged
        child, a `ProgressTimeout` on a saturated box). The label must then be measured over the
        arms that finished — never crash the record, never count the dead arm as a loss, which
        would bias every label a dead child touched DOWNWARD."""
        calls = {"n": 0}

        def _boom(record, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("local_sim_bridge error: child died")
            return {"outcome": "win"}

        monkeypatch.setattr(P, "_run_one", _boom)
        prod = self._prod(tmp_path, rollout_concurrency=conc)
        wins, n, _ = prod._rollout(_rollout_record(), "p1", _decision(), tag="t")
        assert calls["n"] == 6, "a dead arm must not stop the arms after it"
        assert (wins, n) == (5, 5), "the label is over the arms that FINISHED"
        assert "rollout failed" in capsys.readouterr().out

    def test_every_arm_gets_its_own_players_and_its_own_dice(self, tmp_path, monkeypatch):
        """Independence is what makes overlapping them legal at all."""
        seen = []
        monkeypatch.setattr(P, "_run_one", lambda record, **kw: (
            seen.append((id(kw["trainee"]), id(kw["opponent"]), tuple(kw["post_t_seed"]))),
            {"outcome": "loss"})[1])
        prod = self._prod(tmp_path, rollout_concurrency=3)
        prod._rollout(_rollout_record(), "p1", _decision(), tag="t")
        assert len({s[0] for s in seen}) == 6 and len({s[1] for s in seen}) == 6
        assert len({s[2] for s in seen}) == 6, "the arms must not share a dice stream"

    def test_all_arms_dead_is_a_skip_not_a_label(self, tmp_path, monkeypatch):
        """`process_record` reads n == 0 as `rollouts_all_failed`; it must not see a phantom 0.0."""
        monkeypatch.setattr(P, "_run_one", lambda record, **kw: (_ for _ in ()).throw(
            RuntimeError("dead")))
        prod = self._prod(tmp_path, rollout_concurrency=4)
        assert prod._rollout(_rollout_record(), "p1", _decision(), tag="t") == (0, 0, [])


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

    def test_the_compile_defaults_ON_and_can_be_turned_off(self):
        """It is a FALLBACK, not an opt-in: 93% of this process's wall is the rollout forward and
        compiling it is 6.4x, so a producer launched with no flags must get it."""
        assert P.build_parser().parse_args(["/tmp/x"]).compile_extractor is True
        assert P.build_parser().parse_args(
            ["/tmp/x", "--no-compile-extractor"]).compile_extractor is False

    def test_the_rollout_concurrency_default_is_declared(self):
        a = P.build_parser().parse_args(["/tmp/x"])
        assert a.rollout_concurrency == 1, (
            "measured a wash — the forwards serialize on poke-env's single loop thread either "
            "way — so the default stays the sequential path")
        assert P.build_parser().parse_args(
            ["/tmp/x", "--rollout-concurrency", "4"]).rollout_concurrency == 4

    def test_the_lag_warning_default_tracks_the_buffers_bound(self):
        from agents.training.cf_label_buffer import DEFAULT_LAG_BOUND
        assert P.build_parser().parse_args(["/tmp/x"]).lag_warn_steps == DEFAULT_LAG_BOUND
