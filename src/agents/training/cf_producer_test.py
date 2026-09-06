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
        wins, n, n_capped, returns, _seeds = prod._rollout(
            _rollout_record(), "p1", _decision(), tag="t")
        assert n == 6 and n_capped == 0 and returns == []
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
        wins, n, _capped, _, _ = prod._rollout(_rollout_record(), "p1", _decision(), tag="t")
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
        assert prod._rollout(
            _rollout_record(), "p1", _decision(), tag="t")[:4] == (0, 0, 0, [])


class TestDrawAtCap:
    """`gen3_cf_draw_at_cap_v1` — a rollout that reaches the 250-turn stall-forfeit cap is a DRAW.

    THE DEFECT (measured 2026-08-23, before the fix, over 16 capped lines on both `node` and
    `rust`). Both sides of a rollout are players that stall-forfeit at `MAX_TURNS`, so at the cap
    BOTH forfeit and the winner is whichever ``FORCELOSE`` the sim processes first. That is not a
    fact about the position — and it is not a coin flip either: **p1's forfeit is always processed
    first**, and `_trainee_side` seats a training record's trainee on p1 ALWAYS. So every capped
    rollout scored a hard 0, biasing tight-MC P(win) labels DOWNWARD on exactly the stall-shaped
    positions where the cap is reachable, with nothing anywhere recording that it had happened.

    These are the label-side half, driven off a stubbed `_run_one` so they are fast and
    deterministic; that the runner actually SETS ``capped`` on a real battle is
    `counterfactual_test`'s `_battle_outcome` pair plus the `sim` test in
    `cf_producer_integration_test`.
    """

    def _prod(self, tmp_path, **over):
        run = _mk_run(tmp_path)
        prod = P.CfProducer(_args(run, rollouts=8, mc_return=False, **over),
                            snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.refresh_snapshot()
        return prod

    def test_a_capped_rollout_scores_half_whichever_side_the_forfeit_hit(
            self, tmp_path, monkeypatch, capsys):
        """The whole point: BOTH orderings must map to the same number.

        REVERT-VERIFIED. With `rollout_outcome_score`'s ``capped`` branch removed (the pre-fix
        `res["outcome"] == "win"`), the ``win`` arm scores 8.0 → label 1.0 and the ``loss`` arm
        scores 0.0 → label 0.0 — the manufactured 1s and 0s this test exists to forbid.
        """
        prod = self._prod(tmp_path)
        for raced in ("win", "loss"):
            monkeypatch.setattr(P, "_run_one", lambda record, _o=raced, **kw: {
                "outcome": _o, "ended": True, "turns": 250, "capped": True})
            wins, n, n_capped, _, _ = prod._rollout(
                _rollout_record(), "p1", _decision(), tag="t")
            assert (n, n_capped) == (8, 8)
            assert wins == pytest.approx(4.0), (
                f"a capped rollout recorded as a {raced!r} must score 0.5, not by which side's "
                f"forfeit landed first")
            assert wins / n == pytest.approx(0.5)
        assert "stall-forfeit CAP" in capsys.readouterr().out, (
            "a capped rollout must SAY so once — the operator's only warning that a chunk of the "
            "corpus is draws-at-cap")

    def test_a_genuine_tie_is_a_draw_too_and_used_to_score_a_LOSS(self, tmp_path, monkeypatch):
        """Same family, same line of code: ``outcome == "win"`` is False for a tie, so a drawn
        rollout scored 0. A tie IS a draw and scores 0.5."""
        monkeypatch.setattr(P, "_run_one", lambda record, **kw: {
            "outcome": "tie", "ended": True, "turns": 90, "capped": False})
        prod = self._prod(tmp_path)
        wins, n, n_capped, _, _ = prod._rollout(_rollout_record(), "p1", _decision(), tag="t")
        assert (wins, n, n_capped) == (4.0, 8, 0), "a tie is 0.5 and is NOT a cap"

    def test_a_mixed_batch_counts_only_the_capped_arms(self, tmp_path, monkeypatch):
        """`n_capped` must be the count of CAPPED arms, not of draws — the two are different
        questions and only the first tells a reader the label sat on a stall-shaped position."""
        calls = {"n": 0}

        def _mixed(record, **kw):
            calls["n"] += 1
            if calls["n"] <= 3:
                return {"outcome": "loss", "ended": True, "turns": 250, "capped": True}
            return {"outcome": "win", "ended": True, "turns": 40, "capped": False}

        monkeypatch.setattr(P, "_run_one", _mixed)
        prod = self._prod(tmp_path)
        wins, n, n_capped, _, _ = prod._rollout(_rollout_record(), "p1", _decision(), tag="t")
        assert (n, n_capped) == (8, 3)
        assert wins == pytest.approx(3 * 0.5 + 5 * 1.0)

    def test_the_row_records_n_capped_beside_n_rollouts(self):
        """A 0.5 from 8 draws-at-cap and a 0.5 from 4 wins + 4 losses are the same number about
        different positions, and the reader cannot re-derive which afterwards. So the count is
        WRITTEN — a schema ADDITION, never a change to an existing field."""
        from agents.training.obs_materializer import RecordDecision
        d = RecordDecision(index=3, turn=210, action=7, choice="move icebeam",
                           mask=np.ones(11, dtype=np.int8), obs=np.arange(8, dtype=np.float32))
        row = P.label_row(record_path="/r/x_reconstruction.json", decision=d, wins=4.0, n=8,
                          n_capped=8, step=24_000_000, surprise=0.4, entropy=0.6, score=0.61,
                          win_prob=0.8)
        assert row["n_capped"] == 8 and row["n_rollouts"] == 8
        assert row["label"] == pytest.approx(0.5)
        assert row["schema"] == 1, "an additive field must NOT move the refusal gate"
        # A fractional success total must still produce a usable interval that brackets the label.
        assert row["wilson_lo"] <= row["label"] <= row["wilson_hi"]
        # The default is 0, so every pre-existing caller keeps writing a row that reads "no caps"
        # rather than a row that is silent about it.
        bare = P.label_row(record_path="/r/x_reconstruction.json", decision=d, wins=5.0, n=8,
                           step=1, surprise=0.0, entropy=0.0, score=0.0, win_prob=None)
        assert bare["n_capped"] == 0

    def test_the_producer_state_counts_capped_rollouts(self, tmp_path):
        """It must never go invisible: the state file is this process's whole operator surface,
        and the heartbeat prints the ratio as soon as it is non-zero."""
        st = P.ProducerState(path=str(tmp_path / "s.json"))
        assert st.rollouts_capped == 0
        st.rollouts_capped += 3
        st.save()
        assert json.loads((tmp_path / "s.json").read_text())["rollouts_capped"] == 3
        assert P.ProducerState.load(str(tmp_path)).rollouts_capped == 0, "a fresh dir starts at 0"


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


# ---------------------------------------------------------------------------
# The PER-ACTION sweep (`--q-labels`, gen3_cf_q_labels_v1)
# ---------------------------------------------------------------------------

def _q_decision(**over):
    """A labelable decision carrying the full legal-action → choice-string map the sweep needs."""
    from agents.training.obs_materializer import RecordDecision
    base = dict(index=4, turn=6, action=7, choice="move icebeam",
                mask=np.zeros(11, dtype=np.int8), obs=np.zeros(4, dtype=np.float32),
                choices={0: "switch zapdos", 1: "switch skarmory",
                         6: "move rockslide", 7: "move icebeam", 8: "move earthquake"})
    base.update(over)
    base["mask"][list(base["choices"] or {0, 1, 6, 7, 8})] = 1
    return RecordDecision(**base)


class TestQLabelsOffIsByteIdentical:
    """The flag defaults OFF, and OFF must be the file the producer has always written."""

    def test_the_default_is_off(self):
        assert P.build_parser().parse_args(["/tmp/x"]).q_labels is False

    def test_a_row_without_a_sweep_carries_no_q_KEYS_AT_ALL(self):
        """Not `null`, not `[]` — ABSENT. A consumer distinguishes "this producer does not do
        per-action labels" from "it swept and every arm failed", and only absence says the first."""
        row = P.label_row(record_path="b.json", decision=_decision(), wins=1.0, n=2, step=5,
                          surprise=0.1, entropy=0.2, score=0.3, win_prob=0.4)
        for k in ("q_labels", "taken_action", "q_sweep"):
            assert k not in row

    def test_the_off_row_key_set_is_unchanged(self):
        """A frozen census, so an additive field can never quietly become a mandatory one."""
        row = P.label_row(record_path="b.json", decision=_decision(), wins=1.0, n=2, step=5,
                          surprise=0.1, entropy=0.2, score=0.3, win_prob=0.4,
                          outcome_label=1.0, mc_return=0.5, mc_return_n=2, reward_sha1="abc")
        assert set(row) == {
            "schema", "kind", "battle", "decision_idx", "obs_sha1", "obs_npz", "obs_inline",
            "label", "n_rollouts", "wilson_lo", "wilson_hi", "policy_step", "opponent",
            "sampler_version", "label_regime", "turn", "recorded_action", "n_capped", "priority",
            "outcome_label", "mc_return", "mc_return_n", "reward_sha1", "reward_composition",
            "created_unix"}

    def test_the_DICE_derivation_is_byte_identical_to_the_pre_sweep_one(self, tmp_path,
                                                                       monkeypatch):
        """`_rollout` now routes its salt through `cf_q_labels`; a run with the flag off must
        still draw the seeds it always drew, or every existing label file becomes incomparable."""
        from main.prober.falsifier import fresh_seeds
        seen = []
        monkeypatch.setattr(P, "_run_one", lambda record, **kw: (
            seen.append(kw["post_t_seed"]) or {"outcome": "win"}))
        run = _mk_run(tmp_path)
        prod = P.CfProducer(_args(run, rollouts=5, seed=20260822, mc_return=False),
                            snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.refresh_snapshot()
        d = _decision()
        prod._rollout(_rollout_record(), "p1", d, tag="tag.json")
        assert seen == fresh_seeds(5, salt=f"tag.json:{d.index}:cfp20260822")


class _QProd:
    """Builds a producer with the sweep on and every bridge call stubbed."""

    @staticmethod
    def make(tmp_path, monkeypatch, *, outcome_of=None, **over):
        calls = []

        def _one(record, **kw):
            calls.append({"post_t_seed": kw["post_t_seed"],
                          "substitute_choice": kw["substitute_choice"]})
            if outcome_of is not None:
                return outcome_of(kw)
            return {"outcome": "win", "ended": True, "turns": 20, "capped": False}

        monkeypatch.setattr(P, "_run_one", _one)
        run = _mk_run(tmp_path)
        opts = dict(rollouts=4, mc_return=False, q_labels=True, q_top_n=1,
                    q_rollouts=0, q_max_actions=0)
        opts.update(over)
        args = _args(run, **opts)
        prod = P.CfProducer(args, snapshot_loader=lambda p, s: _StubSnapshot(p, s))
        prod.refresh_snapshot()
        return prod, calls


class TestQSweepPairing:
    """THE property: sibling actions must be rolled out on the SAME dice."""

    def test_every_sibling_action_actually_received_the_same_seeds(self, tmp_path, monkeypatch):
        prod, calls = _QProd.make(tmp_path, monkeypatch)
        d = _q_decision()
        prod._q_labels(_rollout_record(), "p1", d, tag="t.json",
                       base=(2.0, 4, 0, Q_SEEDS(d)))
        by_choice = {}
        for c in calls:
            by_choice.setdefault(c["substitute_choice"], []).append(c["post_t_seed"])
        assert len(by_choice) == 4, "four sibling arms (the recorded one was reused free)"
        assert len({tuple(v) for v in by_choice.values()}) == 1, \
            "sibling actions drew DIFFERENT dice — the sweep's ranking would be noise"

    def test_a_producer_that_DERIVES_SEEDS_PER_ACTION_fails_loudly(self, tmp_path, monkeypatch):
        """The regression guard, expressed as the bug: perturb the dice per action and the seam
        must refuse rather than ship a sweep whose ranking is noise."""
        prod, _calls = _QProd.make(tmp_path, monkeypatch)
        real = prod._rollout
        bad = {"n": 0}

        def _drifting(*a, **kw):
            bad["n"] += 1
            kw["seeds"] = [f"{s}-{bad['n']}" for s in kw["seeds"]]     # a per-action salt
            return real(*a, **kw)

        prod._rollout = _drifting
        d = _q_decision()
        with pytest.raises(RuntimeError, match="DICE ARE NOT PAIRED"):
            prod._q_labels(_rollout_record(), "p1", d, tag="t.json",
                           base=(2.0, 4, 0, Q_SEEDS(d)))

    def test_the_check_reads_the_BASE_arms_OBSERVED_seeds(self, tmp_path, monkeypatch):
        """The reused recorded arm is adjudicated on the seeds the per-state rollout REPORTED, not
        on seeds this method re-derives — a check that recomputes its own input proves nothing."""
        prod, _ = _QProd.make(tmp_path, monkeypatch)
        with pytest.raises(RuntimeError, match="DICE ARE NOT PAIRED"):
            prod._q_labels(_rollout_record(), "p1", _q_decision(), tag="t.json",
                           base=(2.0, 4, 0, ["not", "the", "sweeps", "dice"]))


def Q_SEEDS(d, *, tag="t.json", seed=1, n=4):
    from agents.training.cf_q_labels import q_arm_seeds
    return q_arm_seeds(tag=tag, decision_index=int(d.index), producer_seed=seed, n=n)


class TestQSweepContent:
    def test_the_recorded_actions_q_label_IS_the_rows_own_label(self, tmp_path, monkeypatch):
        """Free, and an identity rather than an approximation: same salt, same R, same substituted
        choice, so re-rolling it would buy a second sample of a number already in hand."""
        prod, calls = _QProd.make(tmp_path, monkeypatch)
        d = _q_decision()
        entries, prov = prod._q_labels(_rollout_record(), "p1", d, tag="t.json",
                                       base=(3.0, 4, 0, Q_SEEDS(d)))
        rec = [e for e in entries if e["action"] == d.action]
        assert rec == [{"action": 7, "label": 0.75, "n_rollouts": 4}]
        assert prov["recorded_arm_reused"] is True
        assert d.choice not in [c["substitute_choice"] for c in calls], \
            "the recorded action must not be rolled out a second time"

    def test_a_DIFFERENT_q_rollouts_re_rolls_the_recorded_arm(self, tmp_path, monkeypatch):
        """An anchor measured over more arms than the siblings it anchors would make
        `q_labels[recorded] - q_labels[other]` a comparison between two sample sizes."""
        prod, calls = _QProd.make(tmp_path, monkeypatch, q_rollouts=2)
        d = _q_decision()
        entries, prov = prod._q_labels(_rollout_record(), "p1", d, tag="t.json",
                                       base=(3.0, 4, 0, Q_SEEDS(d)))
        assert prov["recorded_arm_reused"] is False
        assert all(e["n_rollouts"] == 2 for e in entries)
        assert d.choice in [c["substitute_choice"] for c in calls]

    def test_one_entry_per_legal_action_each_naming_its_own_index(self, tmp_path, monkeypatch):
        prod, _ = _QProd.make(tmp_path, monkeypatch)
        d = _q_decision()
        entries, prov = prod._q_labels(_rollout_record(), "p1", d, tag="t.json",
                                       base=(2.0, 4, 0, Q_SEEDS(d)))
        assert sorted(e["action"] for e in entries) == [0, 1, 6, 7, 8]
        assert prov["arms"] == 5 and prov["version"] == "cf_q_sweep_v1"

    def test_an_arm_whose_rollouts_ALL_FAIL_is_omitted_and_counted(self, tmp_path, monkeypatch):
        def _die(kw):
            if kw["substitute_choice"] == "move earthquake":
                raise RuntimeError("bridge died")
            return {"outcome": "win"}
        prod, _ = _QProd.make(tmp_path, monkeypatch, outcome_of=_die)
        d = _q_decision()
        entries, _ = prod._q_labels(_rollout_record(), "p1", d, tag="t.json",
                                    base=(2.0, 4, 0, Q_SEEDS(d)))
        assert 8 not in [e["action"] for e in entries], "a zero-evidence arm must not be shipped"
        assert prod.state.q_skip_reasons.get("arm_all_failed") == 1
        assert prod.state.records_skipped == 0, "a lost ARM is not a lost RECORD"

    def test_a_record_scanned_without_the_choice_map_is_a_counted_skip(self, tmp_path,
                                                                      monkeypatch):
        prod, calls = _QProd.make(tmp_path, monkeypatch)
        d = _q_decision(choices=None)
        entries, _ = prod._q_labels(_rollout_record(), "p1", d, tag="t.json",
                                    base=(2.0, 4, 0, Q_SEEDS(d)))
        assert entries == [] and calls == []
        assert prod.state.q_skip_reasons.get("no_choice_map") == 1


class TestQBudgetKnobs:
    def test_q_max_actions_caps_the_arms(self, tmp_path, monkeypatch):
        prod, calls = _QProd.make(tmp_path, monkeypatch, q_max_actions=3)
        d = _q_decision()
        entries, prov = prod._q_labels(_rollout_record(), "p1", d, tag="t.json",
                                       base=(2.0, 4, 0, Q_SEEDS(d)))
        assert prov["arms"] == 3 and len(entries) == 3
        assert len(calls) == 2 * 4, "3 arms minus the free recorded one, at R=4"

    def test_q_max_actions_always_keeps_the_recorded_action(self, tmp_path, monkeypatch):
        prod, _ = _QProd.make(tmp_path, monkeypatch, q_max_actions=1)
        d = _q_decision()
        entries, _ = prod._q_labels(_rollout_record(), "p1", d, tag="t.json",
                                    base=(2.0, 4, 0, Q_SEEDS(d)))
        assert [e["action"] for e in entries] == [7]

    def test_q_rollouts_sets_the_arms_evidence(self, tmp_path, monkeypatch):
        prod, calls = _QProd.make(tmp_path, monkeypatch, q_rollouts=2)
        d = _q_decision()
        entries, prov = prod._q_labels(_rollout_record(), "p1", d, tag="t.json",
                                       base=(2.0, 4, 0, Q_SEEDS(d)))
        assert prov["rollouts_per_arm"] == 2
        assert len(calls) == 5 * 2, "5 arms (nothing reused at a different R) at R=2"

    def test_q_top_n_bounds_the_SWEPT_decisions_per_record(self, tmp_path, monkeypatch):
        """The multiplied budget rides the sampler's own ranking, so it lands on the decisions
        already judged most informative rather than on a random subset of them."""
        swept = []
        prod, _ = _QProd.make(tmp_path, monkeypatch, top_n=3, q_top_n=2)
        prod._q_labels = lambda *a, **kw: (swept.append(kw["tag"]) or ([], {}))
        decisions = [_q_decision(index=i, turn=i + 2) for i in range(3)]
        monkeypatch.setattr(P, "scan_record", lambda *a, **kw: decisions)
        monkeypatch.setattr(P, "replay_battle", lambda rec, **kw: SimpleNamespace(
            p1_chunks=[], p2_chunks=[], outcome={"winner": "Ann"}))
        prod._preloaded["rec.json"] = _rollout_record()
        prod.process_record("rec.json")
        assert len(swept) == 2, "--q-top-n 2 of --top-n 3"

    def test_q_top_n_zero_sweeps_nothing_while_the_flag_is_on(self, tmp_path, monkeypatch):
        prod, _ = _QProd.make(tmp_path, monkeypatch, q_top_n=0)
        assert prod._q_top_n == 0


class TestQCostMeter:
    def test_the_state_file_carries_the_measured_multiplier(self, tmp_path, monkeypatch):
        """`(arms rolled + free) / rows` IS the ~n_legal multiplier — reported, not assumed, since
        it depends on how many actions are legal where the sampler happens to look."""
        prod, _ = _QProd.make(tmp_path, monkeypatch)
        d = _q_decision()
        prod._q_labels(_rollout_record(), "p1", d, tag="t.json", base=(2.0, 4, 0, Q_SEEDS(d)))
        st = prod.state
        assert st.q_rows == 1 and st.q_entries_total == 5
        assert st.q_arms_rolled == 4 and st.q_arms_reused == 1
        assert (st.q_arms_rolled + st.q_arms_reused) / st.q_rows == 5.0
        assert st.q_rollouts_total == 16          # 4 arms x R=4
        assert st.q_wall_seconds >= 0.0

    def test_the_counters_round_trip_through_the_state_file(self, tmp_path, monkeypatch):
        prod, _ = _QProd.make(tmp_path, monkeypatch)
        d = _q_decision()
        prod._q_labels(_rollout_record(), "p1", d, tag="t.json", base=(2.0, 4, 0, Q_SEEDS(d)))
        prod.state.save()
        st = P.ProducerState.load(str(tmp_path / "run"))
        assert st.q_entries_total == 5 and st.q_arms_rolled == 4

    def test_each_per_action_label_counts_against_max_labels_per_hour(self, tmp_path,
                                                                     monkeypatch):
        """Otherwise `--q-labels` silently multiplies the box load by the arm count against a cap
        the operator set — and this producer's whole premise is that it is a sidecar."""
        prod, _ = _QProd.make(tmp_path, monkeypatch)
        d = _q_decision()
        before = len(prod.label_times)
        prod._q_labels(_rollout_record(), "p1", d, tag="t.json", base=(2.0, 4, 0, Q_SEEDS(d)))
        assert len(prod.label_times) - before == 4, "one per ROLLED arm"

    def test_the_HEARTBEAT_shows_the_multiplier_and_vanishes_when_the_flag_is_off(
            self, tmp_path, monkeypatch, capsys):
        """The producer has no TensorBoard, so the heartbeat and the state file ARE the operator
        surface — a cost that multiplies by ~9 must not be invisible on it."""
        prod, _ = _QProd.make(tmp_path, monkeypatch)
        prod.state.q_rows, prod.state.q_entries_total = 5, 40
        prod.state.q_arms_rolled, prod.state.q_arms_reused = 35, 5
        prod.state.q_wall_seconds = 50.0
        prod._emit_heartbeat(time.perf_counter(), new=1, produced=1)
        assert "q 40 entries / 5 rows (35 arms rolled, 5 free, 8.0x, 1.2s/entry)" in prod.heartbeat
        prod._q_on = False
        prod._emit_heartbeat(time.perf_counter(), new=1, produced=1)
        assert " q " not in prod.heartbeat, "the field must not appear on a run without the sweep"
        capsys.readouterr()

    def test_an_exhausted_throttle_ships_a_PARTIAL_sweep_not_a_broken_one(self, tmp_path,
                                                                         monkeypatch):
        prod, _ = _QProd.make(tmp_path, monkeypatch, max_labels_per_hour=1)
        d = _q_decision()
        entries, _ = prod._q_labels(_rollout_record(), "p1", d, tag="t.json",
                                    base=(2.0, 4, 0, Q_SEEDS(d)))
        # The free recorded arm always lands; the throttle then bites part-way through the
        # siblings, so the sweep is SHORT rather than absent or broken.
        assert 7 in [e["action"] for e in entries]
        assert 1 <= len(entries) < 5
        assert prod.state.q_skip_reasons.get("throttled") == 1


# ---------------------------------------------------------------------------
# The schema, round-tripped through BOTH sides
# ---------------------------------------------------------------------------

class TestQSchemaRoundTrip:
    """Producer writes → the REAL `CfLabelBuffer` reads. The end the mission calls mating."""

    def _write(self, tmp_path, rows, *, step=1000):
        return P.write_label_batch(str(tmp_path / "labels"), rows, step=step, seq=1)

    def _row(self, **over):
        kw = dict(record_path="b.json", decision=_q_decision(), wins=3.0, n=4, step=1000,
                  surprise=0.1, entropy=0.2, score=0.3, win_prob=0.4, outcome_label=1.0,
                  q_labels=[{"action": 7, "label": 0.75, "n_rollouts": 4},
                            {"action": 0, "label": 0.25, "n_rollouts": 4},
                            {"action": 6, "label": 0.5, "n_rollouts": 4}],
                  q_sweep={"version": "cf_q_sweep_v1", "arms": 3})
        kw.update(over)
        return P.label_row(**kw)

    def _buffer(self, tmp_path):
        from agents.training.cf_label_buffer import CfLabelBuffer
        return CfLabelBuffer(str(tmp_path / "labels"), obs_dim=4, lag_bound=0)

    def test_the_written_row_parses_into_the_consumers_per_action_stream(self, tmp_path):
        self._write(tmp_path, [self._row()])
        buf = self._buffer(tmp_path)
        assert buf.poll(1000) == 1
        row = buf.sample(1)[0]
        assert row.q_labels == ((7, 0.75, 4), (0, 0.25, 4), (6, 0.5, 4))
        assert row.taken_action == 7

    def test_the_liveness_counters_the_head_launches_on(self, tmp_path):
        self._write(tmp_path, [self._row()])
        buf = self._buffer(tmp_path)
        buf.poll(1000)
        stats = buf.stats(1000)
        assert stats["cf/q_label_coverage"] == 1.0
        assert stats["cf/q_labels_per_row"] == 3.0

    def test_a_SHUFFLED_list_reads_identically(self, tmp_path):
        """The object-not-arrays property, demonstrated rather than asserted: each entry names its
        own action index, so the wire ORDER carries no meaning and cannot be got wrong. Three
        parallel arrays written in the wrong order would read as valid and be silently wrong."""
        import random
        entries = [{"action": 7, "label": 0.75, "n_rollouts": 4},
                   {"action": 0, "label": 0.25, "n_rollouts": 2},
                   {"action": 6, "label": 0.5, "n_rollouts": 8}]
        rng = random.Random(0)
        seen = set()
        for i in range(6):
            shuffled = list(entries)
            rng.shuffle(shuffled)
            d = tmp_path / f"t{i}"
            (d / "labels").mkdir(parents=True)
            self._write(d, [self._row(q_labels=shuffled)])
            buf = self._buffer(d)
            buf.poll(1000)
            seen.add(tuple(sorted(buf.sample(1)[0].q_labels)))
        assert len(seen) == 1 and seen == {((0, 0.25, 2), (6, 0.5, 8), (7, 0.75, 4))}

    def test_the_consumer_scatters_each_entry_into_ITS_OWN_column(self, tmp_path):
        from agents.training.cf_label_buffer import batch_tensors
        self._write(tmp_path, [self._row()])
        buf = self._buffer(tmp_path)
        buf.poll(1000)
        b = batch_tensors(buf.sample(1), "cpu")
        assert b.q_mask[0].tolist() == [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        assert b.q_label[0, 7].item() == pytest.approx(0.75)
        assert b.q_label[0, 0].item() == pytest.approx(0.25)
        assert b.q_label[0, 6].item() == pytest.approx(0.5)
        assert b.q_label[0, 8].item() == 0.0, "an unswept action stays masked-off at zero"
        assert b.taken_mask[0].item() == 1.0 and int(b.taken_action[0]) == 7

    def test_a_MALFORMED_entry_is_a_field_skip_that_keeps_the_rows_other_streams(self, tmp_path):
        """The consumer's declared contract, exercised from the producer's own writer."""
        self._write(tmp_path, [self._row(q_labels=[
            {"action": 7, "label": 0.75, "n_rollouts": 4},
            {"action": 99, "label": 0.5, "n_rollouts": 4},          # out of range
            {"action": 6, "label": 3.0, "n_rollouts": 4},           # out of [0,1]
            ["not", "an", "object"]])]) 
        buf = self._buffer(tmp_path)
        assert buf.poll(1000) == 1, "the ROW survives a bad ENTRY"
        row = buf.sample(1)[0]
        assert row.q_labels == ((7, 0.75, 4),)
        assert row.label == pytest.approx(0.75) and row.outcome_label == 1.0
        assert buf.field_skipped_total == 3

    def test_an_OLD_row_still_reads_on_the_NEW_consumer(self, tmp_path):
        """Additive-optional in the other direction: `schema` is a REFUSAL gate, so the sweep may
        never bump it — a v2 row would be unreadable by every existing trainer."""
        row = P.label_row(record_path="b.json", decision=_q_decision(), wins=1.0, n=2, step=1000,
                          surprise=0.1, entropy=0.2, score=0.3, win_prob=0.4)
        assert row["schema"] == 1
        self._write(tmp_path, [row])
        buf = self._buffer(tmp_path)
        buf.poll(1000)
        r = buf.sample(1)[0]
        assert r.q_labels == () and r.taken_action is None
        assert buf.stats(1000)["cf/q_label_coverage"] == 0.0

    def test_a_RE_LABELLED_state_replaces_its_per_action_block_keep_NEWEST(self, tmp_path):
        """The #28 dedup rule, inherited rather than reimplemented. `q_labels` rides the SAME row as
        the per-state label precisely so it dedups on the same obs digest — a second row for one
        state would collide and one of them would vanish, and a per-action stream that appended
        instead would give that state N x the weight of every other."""
        old = self._row(q_labels=[{"action": 7, "label": 0.0, "n_rollouts": 2}])
        new = self._row(q_labels=[{"action": 7, "label": 1.0, "n_rollouts": 8},
                                  {"action": 0, "label": 0.5, "n_rollouts": 8}])
        assert old["obs_sha1"] == new["obs_sha1"], "same state, or this tests nothing"
        P.write_label_batch(str(tmp_path / "labels"), [old], step=1000, seq=1)
        P.write_label_batch(str(tmp_path / "labels"), [new], step=1000, seq=2)
        buf = self._buffer(tmp_path)
        buf.poll(1000)
        assert len(buf) == 1 and buf.replaced_total == 1
        assert buf.sample(1)[0].q_labels == ((7, 1.0, 8), (0, 0.5, 8))

    def test_an_EXPIRED_row_takes_its_per_action_block_with_it(self, tmp_path):
        """Expiry is a whole-row property too: a per-action label measured under a policy the
        trainer has moved past is exactly as stale as the per-state one beside it."""
        from agents.training.cf_label_buffer import CfLabelBuffer
        self._write(tmp_path, [self._row()], step=1000)
        buf = CfLabelBuffer(str(tmp_path / "labels"), obs_dim=4, lag_bound=10)
        assert buf.poll(5000) == 0 and len(buf) == 0
        assert buf.expired_total == 1
        assert buf.stats(5000)["cf/q_label_coverage"] == 0.0

    def test_a_swept_row_that_lost_every_arm_is_distinguishable_from_an_old_row(self, tmp_path):
        self._write(tmp_path, [self._row(q_labels=[])])
        buf = self._buffer(tmp_path)
        buf.poll(1000)
        r = buf.sample(1)[0]
        assert r.q_labels == () and r.taken_action == 7, \
            "`taken_action` present says the sweep RAN; its absence says the producer is older"


# ══════════════════════════════════════════════════ the EXTRACTION parity golden ══
#
# The declared sampler moved to `cf_producer_sampler.py`, the checkpoint + the loaded model to
# `cf_producer_snapshot.py` and the label schema + batch writer to `cf_producer_labels.py` on
# 2026-09-06 (the file-size ratchet's third cut of the 1,000-2,000 band). Every one of them is
# still CALLED HERE through `cf_producer`'s re-exports, which is the point: this golden is what
# makes each move an EXTRACTION rather than a rewrite, and it has not been regenerated for any of
# them. Each claimed to be pure refactoring, and this is the evidence for that claim rather than a
# promise of it: every public entry point of the module — the priority arithmetic, checkpoint
# resolution, `Snapshot.score` over a recording stub, the state file's whole round trip, the label
# row and its batch file, the outcome scalars, the anchor predicates and their refusal texts, and
# the CLI's declared defaults — run on one synthetic fixture, JSON-serialised canonically and
# pinned by digest. The blob was captured from the tree BEFORE the extraction and reproduces
# byte-for-byte after it.
#
# The named values beside the digest are not decoration. A bare hash cannot say WHICH entry point
# moved when it fails, and a test whose only assertion is an opaque digest is one refactor away
# from being regenerated on autopilot.
#
# ⚠️ Regenerating: only ever after establishing WHY it changed. A new CLI flag legitimately moves
# it (the parser's defaults are in the blob on purpose — a silently changed default is exactly the
# class this pins); a change in the arithmetic, the schema or a refusal text is not legitimate
# without a stated reason. Print the current digest with::
#
#     python -c "import hashlib; from agents.training import cf_producer_test as t; \
#                print(hashlib.sha256(t._parity_blob().encode()).hexdigest())"

_PARITY_SHA256 = "121fe20156da385aad915bb0fb2943c402057e553d4b47fb3c6a134e66b7ea67"

#: `created_unix` is `time.time()` at construction, so it is dropped rather than pinned — the row
#: is otherwise byte-stable and this is the only field of it that is not.
_VOLATILE_ROW_KEYS = ("created_unix",)


class _ParityPolicy:
    """A recording stand-in for `model.policy`: deterministic per-row logits, an optional win-prob
    stash, and it remembers the shape/dtype signature every call arrived with — which is the half
    of `Snapshot.score` that is a CONTRACT (B=1 under compile, float32 masks always) rather than
    arithmetic."""

    def __init__(self, *, win_head: bool = True) -> None:
        self.calls: "list[dict]" = []
        self.features_extractor = SimpleNamespace(last_win_prob_logits=None)
        self._win_head = win_head

    def get_distribution(self, obs):
        import torch as th
        o, m = obs["observation"], obs["action_mask"]
        self.calls.append({"n": int(o.shape[0]), "obs_dtype": str(o.dtype),
                           "mask_dtype": str(m.dtype)})
        logits = (th.arange(o.shape[0], dtype=th.float32).unsqueeze(1)
                  + th.arange(m.shape[1], dtype=th.float32).unsqueeze(0))
        if self._win_head:
            self.features_extractor.last_win_prob_logits = th.full((o.shape[0], 1), 0.25)
        return SimpleNamespace(distribution=SimpleNamespace(logits=logits))


def _parity_record():
    """A `ReconstructionRecord` stand-in with the two fields the outcome/anchor readers touch."""
    return SimpleNamespace(
        commands=[("p1", "move 1"), ("p2", "switch 3"), ("p1", "move 2")],
        username=lambda side: {"p1": "Trainee", "p2": "Opp"}[side])


def _parity_decision(**over):
    from agents.training.obs_materializer import RecordDecision
    kw = dict(index=3, turn=11, action=7, choice="move icebeam",
              mask=np.ones(11, dtype=np.int8), obs=np.arange(8, dtype=np.float32))
    kw.update(over)
    return RecordDecision(**kw)


def _strip_volatile(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _VOLATILE_ROW_KEYS}


def _parity_readouts() -> dict:
    """Every public entry point of `cf_producer`, on one synthetic fixture."""
    import shutil
    import tempfile
    from pathlib import Path

    from agents.action.constants import MOVE_START
    from agents.training.cf_producer import (DEFAULT_KEEP_PROCESSED, LABELS_DIRNAME,
                                             MIN_LABELABLE_TURN, OPPONENT_LABEL, PRIORITY_WEIGHTS,
                                             RECORDS_DIRNAME, SAMPLER_VERSION, STATE_FILENAME,
                                             ProducerState, Snapshot, _outcome_scalar,
                                             anchor_refusal_message, build_parser, critic_surprise,
                                             is_move_round, label_row, normalized_entropy,
                                             priority_score, record_is_full_replay_anchorable,
                                             resolve_latest_checkpoint, rollout_outcome_score,
                                             step_from_checkpoint_name, write_label_batch)

    out: dict = {
        "constants": {"SAMPLER_VERSION": SAMPLER_VERSION,
                      "PRIORITY_WEIGHTS": dict(PRIORITY_WEIGHTS),
                      "MIN_LABELABLE_TURN": MIN_LABELABLE_TURN,
                      "OPPONENT_LABEL": OPPONENT_LABEL,
                      "STATE_FILENAME": STATE_FILENAME,
                      "LABELS_DIRNAME": LABELS_DIRNAME,
                      "RECORDS_DIRNAME": RECORDS_DIRNAME,
                      "DEFAULT_KEEP_PROCESSED": DEFAULT_KEEP_PROCESSED},
    }

    # -- the declared sampler ------------------------------------------------------
    out["normalized_entropy"] = [
        normalized_entropy([]), normalized_entropy([1.0]),
        normalized_entropy([0.5, 0.5]), normalized_entropy([0.25] * 4),
        normalized_entropy([0.7, 0.2, 0.1]), normalized_entropy([0.9, 0.1, 0.0, 0.0]),
    ]
    out["critic_surprise"] = [critic_surprise(None, 1.0), critic_surprise(0.9, 0.0),
                              critic_surprise(0.5, 0.5), critic_surprise(0.2, 1.0),
                              critic_surprise(float("nan"), 1.0)]
    out["priority_score"] = [priority_score(0.9, 0.1), priority_score(0.0, 1.0),
                             priority_score(0.4, 0.6),
                             priority_score(0.4, 0.6, {"critic_surprise": 2.0,
                                                       "policy_entropy": 0.0})]
    switch_only = np.zeros(11, dtype=np.int8)
    switch_only[:MOVE_START] = 1
    move_only = np.zeros(11, dtype=np.int8)
    move_only[MOVE_START] = 1
    out["is_move_round"] = [is_move_round(np.ones(11, dtype=np.int8)),
                            is_move_round(switch_only), is_move_round(move_only)]

    # -- the checkpoint ------------------------------------------------------------
    out["step_from_checkpoint_name"] = [
        step_from_checkpoint_name("/m/r/checkpoints/checkpoint_1234_steps.zip"),
        step_from_checkpoint_name("/m/r/checkpoints/checkpoint_forced_0000009999_131415.zip"),
        step_from_checkpoint_name("/m/r/best_model/best_model.zip"),
    ]
    tmp = tempfile.mkdtemp(prefix="cfprod_parity_")
    try:
        run = os.path.join(tmp, "run")
        cks = os.path.join(run, "checkpoints")
        os.makedirs(cks)
        out["resolve_no_checkpoint"] = resolve_latest_checkpoint(run)
        for name in ("checkpoint_100_steps.zip", "checkpoint_900_steps.zip",
                     "checkpoint_forced_0000001500_101112.zip"):
            open(os.path.join(cks, name), "w").close()
        with open(os.path.join(run, "latest.txt"), "w") as fh:
            fh.write("checkpoints/checkpoint_100_steps.zip")
        got = resolve_latest_checkpoint(run)
        out["resolve_latest_checkpoint"] = [os.path.basename(got[0]), got[1]]

        # -- the state file --------------------------------------------------------
        st = ProducerState.load(run)
        for i in range(6):
            st.claim(f"rec_{i}.json", keep=4)
        st.claim("rec_2.json", keep=4)                     # idempotent
        st.note_skip("no_move_round")
        st.note_skip("no_move_round")
        st.note_q_skip("all_arms_dead")
        st.labels_total, st.rollouts_total, st.rollouts_capped = 12, 96, 3
        st.records_vanished, st.anchors_run, st.anchors_reproduced = 2, 1, 1
        st.save()
        body = json.loads(Path(st.path).read_text())
        body.pop("started_unix")
        body.pop("updated_unix")
        out["producer_state"] = body
        out["producer_state_reload"] = sorted(ProducerState.load(run).processed)
        out["producer_state_is_processed"] = [st.is_processed("rec_5.json"),
                                              st.is_processed("rec_0.json")]

        # -- the label batch file --------------------------------------------------
        rows = [label_row(record_path="/r/a_reconstruction.json", decision=_parity_decision(),
                          wins=5.0, n=8, step=24_000_000, surprise=0.4, entropy=0.6, score=0.61,
                          win_prob=0.8, n_capped=1)]
        p = write_label_batch(os.path.join(run, LABELS_DIRNAME), rows, step=24_000_000, seq=3)
        out["label_batch"] = {
            "name": os.path.basename(p),
            "lines": [_strip_volatile(json.loads(ln))
                      for ln in Path(p).read_text().splitlines()],
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # -- the label row, over its optional streams ----------------------------------
    out["label_row_minimal"] = _strip_volatile(label_row(
        record_path="/r/x_reconstruction.json", decision=_parity_decision(), wins=5.0, n=8,
        step=24_000_000, surprise=0.4, entropy=0.6, score=0.61, win_prob=0.8))
    out["label_row_headless"] = _strip_volatile(label_row(
        record_path="/r/x_reconstruction.json", decision=_parity_decision(), wins=0.0, n=8,
        step=1, surprise=0.0, entropy=0.6, score=0.21, win_prob=None))
    out["label_row_full"] = _strip_volatile(label_row(
        record_path="/r/x_reconstruction.json", decision=_parity_decision(index=4, turn=19),
        wins=4.5, n=8, step=24_000_000, surprise=0.4, entropy=0.6, score=0.61, win_prob=0.8,
        n_capped=2, outcome_label=1.0, mc_return=0.37, mc_return_n=8, reward_sha1="deadbeef",
        reward_composition="terminal+pbrs",
        q_labels=[{"action": 7, "label": 0.75, "n": 8}, {"action": 0, "label": 0.5, "n": 8}],
        q_sweep={"version": "cf_q_sweep_v1", "arms": 2}))

    # -- the loaded snapshot's scoring contract ------------------------------------
    obs = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    masks = np.ones((5, 3), dtype=np.int8)
    masks[:, 2] = 0
    for tag, compiled, win_head in (("compiled", True, True), ("eager", False, True),
                                    ("headless", False, False)):
        pol = _ParityPolicy(win_head=win_head)
        snap = Snapshot("/ckpt.zip", 7, SimpleNamespace(policy=pol), None, compiled=compiled)
        wp, ent = snap.score(obs, masks)
        out[f"score_{tag}"] = {
            "win_probs": None if wp is None else [round(float(v), 12) for v in wp],
            "entropies": [round(float(v), 12) for v in ent],
            "calls": pol.calls,
            "step": snap.step, "path": snap.path, "compiled": snap.compiled,
        }

    # -- the outcome scalars and the anchor ----------------------------------------
    rec = _parity_record()
    out["_outcome_scalar"] = [_outcome_scalar(rec, "p1", {"winner": "Trainee"}),
                              _outcome_scalar(rec, "p1", {"winner": "Opp"}),
                              _outcome_scalar(rec, "p1", {}),
                              _outcome_scalar(rec, "p2", {"winner": "Opp"})]
    out["rollout_outcome_score"] = [rollout_outcome_score({"outcome": "win"}),
                                    rollout_outcome_score({"outcome": "loss"}),
                                    rollout_outcome_score({"outcome": "tie"}),
                                    rollout_outcome_score({"outcome": "unfinished"}),
                                    rollout_outcome_score({"outcome": "win", "capped": True})]
    out["anchorable"] = [
        record_is_full_replay_anchorable(rec),
        record_is_full_replay_anchorable(SimpleNamespace(
            commands=rec.commands + [("forcelose", "p1")])),
    ]
    out["anchor_refusal"] = {
        "mismatch": anchor_refusal_message(
            error=None, mismatch=("loss", "Trainee", "Opp", "Trainee", ()),
            state_path="/run/cf_producer_state.json"),
        "mismatch_exhausted": anchor_refusal_message(
            error=None, mismatch=("loss", "Trainee", "Opp", "Trainee", ("p2",)),
            state_path="/run/cf_producer_state.json"),
        "error": anchor_refusal_message(error="ConnectionResetError: boom", mismatch=None,
                                        state_path="/run/cf_producer_state.json"),
    }
    # A TIMEOUT appends `describe_contention()`, which reads the live load average — so only the
    # self-diagnosing PREFIX is pinned, never the box's current numbers.
    timeout = anchor_refusal_message(error="ProgressTimeout: no progress", mismatch=None,
                                     state_path="/run/cf_producer_state.json")
    out["anchor_refusal_timeout_head"] = timeout.split("\n")[0]

    # -- the CLI's declared defaults -----------------------------------------------
    ns = build_parser().parse_args(["/models/run"])
    out["cli_defaults"] = {k: v for k, v in sorted(vars(ns).items())}
    return out


def _parity_blob() -> str:
    return json.dumps(_parity_readouts(), sort_keys=True, separators=(",", ":"), default=repr)


def test_every_public_entry_point_matches_the_pre_extraction_golden():
    """The sampler, the snapshot and the label schema live in their own modules now; nothing they
    compute, write or refuse may have moved."""
    import hashlib
    blob = _parity_blob()
    assert '"' in blob and len(blob) > 6_000, "the fixture produced nothing to compare"
    got = hashlib.sha256(blob.encode()).hexdigest()
    assert got == _PARITY_SHA256, (
        f"cf_producer's entry points changed: {got} != {_PARITY_SHA256}. Read the named-value "
        f"test below to find which one, and do not regenerate the digest without a stated reason.")


def test_the_named_golden_values_are_what_the_digest_stands_for():
    """The digest says 'nothing moved'; these say WHAT it is that did not move — so a failure
    names the entry point instead of a hash, and so the golden cannot be regenerated blind."""
    r = _parity_readouts()
    assert r["constants"]["SAMPLER_VERSION"] == "cf_producer_priority_v1"
    assert r["constants"]["PRIORITY_WEIGHTS"] == {"critic_surprise": 1.0, "policy_entropy": 0.35}
    # The last two are the point of the normalization: a 3-way skew, and a 4-slot mask with two
    # ZERO-probability actions — which is a 2-support decision, so it divides by log(2), not log(4).
    assert r["normalized_entropy"] == [0.0, 0.0, 1.0, 1.0,
                                       pytest.approx(0.7298466991620974, abs=1e-12),
                                       pytest.approx(0.46899559358928117, abs=1e-12)]
    assert r["critic_surprise"] == [0.0, 0.9, 0.0, 0.8, 0.0]
    assert r["priority_score"] == [pytest.approx(0.935, abs=1e-12), 0.35,
                                   pytest.approx(0.61, abs=1e-12), pytest.approx(0.8, abs=1e-12)]
    assert r["is_move_round"] == [True, False, True]
    assert r["step_from_checkpoint_name"] == [1234, 9999, None]
    assert r["resolve_no_checkpoint"] is None
    assert r["resolve_latest_checkpoint"] == ["checkpoint_forced_0000001500_101112.zip", 1500]
    # The processed ring is bounded oldest-first, and `claim` is idempotent.
    assert r["producer_state_reload"] == ["rec_2.json", "rec_3.json", "rec_4.json", "rec_5.json"]
    assert r["producer_state"]["skip_reasons"] == {"no_move_round": 2}
    assert r["producer_state"]["q_skip_reasons"] == {"all_arms_dead": 1}
    assert r["producer_state"]["records_skipped"] == 2
    assert r["label_batch"]["name"] == "labels_cf_producer_24000000_3.jsonl"
    row = r["label_row_minimal"]
    assert row["schema"] == 1 and row["kind"] == "mc_winprob"
    assert row["opponent"] == "self_current"
    assert row["label_regime"] == "self_current_stochastic_both_sides"
    assert row["label"] == 0.625 and row["n_rollouts"] == 8
    assert (row["wilson_lo"], row["wilson_hi"]) == (0.305738, 0.863158)
    assert row["priority"] == {"score": 0.61, "critic_surprise": 0.4, "policy_entropy": 0.6,
                               "win_prob": 0.8}
    assert r["label_row_headless"]["priority"]["win_prob"] is None
    assert "mc_return" not in r["label_row_minimal"] and "q_labels" not in r["label_row_minimal"]
    assert r["label_row_full"]["mc_return"] == 0.37 and r["label_row_full"]["taken_action"] == 7
    # B=1 under compile, one batched forward eager, float32 masks on BOTH paths.
    assert [c["n"] for c in r["score_compiled"]["calls"]] == [1, 1, 1, 1, 1]
    assert [c["n"] for c in r["score_eager"]["calls"]] == [5]
    assert {c["mask_dtype"] for c in r["score_compiled"]["calls"]} == {"torch.float32"}
    assert r["score_compiled"]["win_probs"] == r["score_eager"]["win_probs"]
    assert r["score_compiled"]["entropies"] == r["score_eager"]["entropies"]
    assert r["score_headless"]["win_probs"] is None
    assert r["_outcome_scalar"] == [1.0, 0.0, 0.5, 1.0]
    assert r["rollout_outcome_score"] == [1.0, 0.0, 0.5, 0.5, 0.5]
    assert r["anchorable"] == [True, False]
    assert r["anchor_refusal"]["mismatch"].startswith("cf_producer: ANCHOR MISMATCH")
    assert "RAN OUT of recorded commands" in r["anchor_refusal"]["mismatch_exhausted"]
    assert r["anchor_refusal"]["error"].startswith("cf_producer: ANCHOR COULD NOT RUN")
    assert r["anchor_refusal_timeout_head"].startswith("cf_producer: ANCHOR COULD NOT RUN")
    d = r["cli_defaults"]
    assert (d["rollouts"], d["top_n"], d["records_per_cycle"]) == (8, 3, 4)
    assert d["max_labels_per_hour"] == 2000 and d["anchor_every"] == 50
