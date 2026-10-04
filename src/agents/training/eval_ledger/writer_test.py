"""``eval_ledger.writer`` + ``store`` + ``queue`` — the archive root, one writer per file, the claim protocol, the
deterministic void rule, the uniqueness refusals, the pins, closing a shard.

DETERMINISTIC BY CONSTRUCTION: time is a :class:`testkit.Clock` the test moves by hand, and a writer's liveness is
an oracle the test owns (``alive=``). No sleep stands in for timing; a "dead writer" is a writer the oracle calls
dead, on the same host."""
from __future__ import annotations

import gzip
import json
import os

import pytest

from agents.training.eval_ledger import queue as Q
from agents.training.eval_ledger import schema as S
from agents.training.eval_ledger import store as ST
from agents.training.eval_ledger import testkit as K
from agents.training.eval_ledger import writer as W

HOST = "testhost"


class Alive:
    def __init__(self):
        self.dead = set()

    def __call__(self, pid):
        return pid not in self.dead


def writer(root, clock, alive, pid, producer="h2h", **kw):
    return W.LedgerWriter(root, producer, clock=clock, alive=alive, host=HOST, pid=pid, **kw)


@pytest.fixture
def env(tmp_path):
    clock, alive = K.Clock(), Alive()
    return tmp_path / "ledger", clock, alive


def opened(w, rid="r1", **kw):
    args = dict(kind="adhoc", purpose="audit", spec={"batch_pairs": 2})
    args.update(kw)
    return w.open_request(rid, **args)


def claim(w, req, batch=0, p=K.SHA_A, o=K.SHA_B, reg=None, wall=60.0):
    return w.claim(req["request_id"], batch=batch, player=p, opponent=o,
                   regime_id=(reg or K.regime())["regime_id"], expected_wall_s=wall)


def row_for(w, req, batch=0, **kw):
    return K.make_row(w.next_row_id(), request=req, batch=batch, ts=w.clock().isoformat(timespec="seconds"), **kw)


# ------------------------------------------------------------------------------------------------ root + layout
def test_the_default_root_is_the_run_archives_ledger(run_archive):
    w = W.LedgerWriter(None, "h2h")
    assert w.root == (run_archive / "_ledger").resolve()
    assert (w.root / "README.md").exists() and (w.root / "requests").is_dir()
    assert w.writer_id.endswith(f"-{os.getpid()}-h2h")


def test_a_writer_refuses_models_except_the_archive_ledger_itself(run_archive):
    for bad in (run_archive / "out", run_archive / "_ledger" / "rows", run_archive):
        with pytest.raises(ST.LedgerPathError, match="REFUSED"):
            W.LedgerWriter(bad, "h2h")
    W.LedgerWriter(run_archive / "_ledger", "h2h")


def test_the_archive_is_sealed_in_tests_without_the_fixture():
    from utils.paths import RunArchiveError

    with pytest.raises(RunArchiveError):
        ST.archive_ledger_root()


def test_one_writer_per_file(env):
    root, clock, alive = env
    w = writer(root, clock, alive, 11)
    req = opened(w)
    w.append_row(row_for(w, req), claim(w, req))
    with pytest.raises(ST.LedgerPathError, match="one writer per file"):
        W.LedgerWriter(root, "h2h", writer_id=w.writer_id)


def test_writer_ids_carry_host_and_pid_and_parse_back():
    wid = ST.make_writer_id("bot_rr", host="my-box.local", pid=4242)
    ident = ST.parse_writer_id(wid)
    assert ident == ST.WriterIdent(host="my_box.local", pid=4242, producer="bot_rr")
    assert ST.parse_writer_id("20261003T221524Z-3541960") is None, "a v1 writer id has no host"
    with pytest.raises(ValueError):
        ST.make_writer_id("Bad-Producer")


# ------------------------------------------------------------------------------------------------ claims
def test_a_claimed_batch_appends_and_records_a_row_event(env):
    root, clock, alive = env
    w = writer(root, clock, alive, 11)
    req = opened(w)
    c = claim(w, req)
    assert c.expires_at == clock() + __import__("datetime").timedelta(seconds=W.CLAIM_FLOOR_S), "10-min floor"
    rid = w.append_row(row_for(w, req), c)
    st = Q.fold(e for e, _ in ST.scan_events(root))
    assert st.rows[c.unit]["row_id"] == rid and not st.problems
    with pytest.raises(W.AlreadyRecordedError):
        claim(w, req)
    assert claim(w, req, batch=1, wall=1000.0).expires_at == clock() + __import__("datetime").timedelta(seconds=4000)


def test_a_live_claim_is_held_against_a_second_writer(env):
    root, clock, alive = env
    w1, w2 = writer(root, clock, alive, 11), writer(root, clock, alive, 22)
    req = opened(w1)
    claim(w1, req)
    with pytest.raises(W.ClaimHeldError, match="not known dead"):
        claim(w2, req)


def test_a_dead_writers_claim_is_voided_and_the_batch_replayed_and_the_dead_writer_cannot_append(env):
    root, clock, alive = env
    w1, w2 = writer(root, clock, alive, 11), writer(root, clock, alive, 22)
    req = opened(w1)
    c1 = claim(w1, req)
    row1 = row_for(w1, req)
    alive.dead.add(11)                           # the writer died mid-batch (same host: its pid is checkable)
    c2 = claim(w2, req)                          # voids c1 (pid_dead), claims the unit, replays the same seeds
    st = Q.fold(e for e, _ in ST.scan_events(root))
    assert st.claims[c1.seq].void_reason == "pid_dead" and st.claims[c2.seq].live
    w2.append_row(row_for(w2, req), c2)
    with pytest.raises(W.ClaimVoidedError, match="DROPPED"):
        w1.append_row(row1, c1)                  # a "zombie" that comes back drops its batch
    assert not w1.rows_path.exists(), "the dropped batch never reached disk"


def test_an_expired_claim_is_voided_but_a_live_unexpired_one_is_not(env):
    root, clock, alive = env
    w1, w2 = writer(root, clock, alive, 11), writer(root, clock, alive, 22)
    req = opened(w1)
    c1 = claim(w1, req, wall=60.0)               # expires at the 10-minute floor
    clock.advance(W.CLAIM_FLOOR_S)               # exactly AT expiry: not yet expired (now > expires_at is the rule)
    with pytest.raises(W.ClaimHeldError):
        claim(w2, req)
    clock.advance(1)
    claim(w2, req)
    st = Q.fold(e for e, _ in ST.scan_events(root))
    assert st.claims[c1.seq].void_reason == "expired"


def test_another_hosts_claim_is_voided_only_by_expiry(env):
    root, clock, alive = env
    w1 = W.LedgerWriter(root, "h2h", clock=clock, alive=alive, host="otherbox", pid=11)
    w2 = writer(root, clock, alive, 22)
    req = opened(w1)
    claim(w1, req)
    alive.dead.add(11)                           # pid 11 on THIS host says nothing about otherbox's pid 11
    with pytest.raises(W.ClaimHeldError):
        claim(w2, req)
    clock.advance(W.CLAIM_FLOOR_S + 1)
    claim(w2, req)


def test_a_writer_killed_between_its_row_and_its_row_event_is_repaired_not_replayed(env):
    root, clock, alive = env
    w1, w2 = writer(root, clock, alive, 11), writer(root, clock, alive, 22)
    req = opened(w1)
    c1 = claim(w1, req)
    row = row_for(w1, req)
    ST.append_line(w1.rows_path, row)            # the row reached disk; the process died before its `row` event
    alive.dead.add(11)
    with pytest.raises(W.AlreadyRecordedError, match="repaired"):
        claim(w2, req)
    st = Q.fold(e for e, _ in ST.scan_events(root))
    assert st.rows[c1.unit]["row_id"] == row["row_id"] and st.claims[c1.seq].void_seq is None


def test_void_dead_sweeps_every_voidable_claim(env):
    root, clock, alive = env
    w1, w2 = writer(root, clock, alive, 11), writer(root, clock, alive, 22)
    req = opened(w1)
    claim(w1, req, batch=0)
    claim(w2, req, batch=1)
    alive.dead.add(11)
    out = writer(root, clock, alive, 33).void_dead()
    assert [(e["batch"], e["reason"]) for e in out] == [(0, "pid_dead")]


# ------------------------------------------------------------------------------------------------ uniqueness + pins
def test_a_second_row_on_one_seed_block_is_refused_across_requests(env):
    root, clock, alive = env
    w = writer(root, clock, alive, 11)
    r1, r2 = opened(w, "r1"), opened(w, "r2")
    w.append_row(row_for(w, r1), claim(w, r1))
    with pytest.raises(W.DuplicateBatchError, match="seed block"):
        w.append_row(row_for(w, r2), claim(w, r2))      # same cycle seed, another request: the same games


def test_a_replay_request_may_replay_a_seed_block(env):
    root, clock, alive = env
    w = writer(root, clock, alive, 11)
    r1 = opened(w, "r1")
    rep = opened(w, "replay", kind="audit_replay")
    w.append_row(row_for(w, r1), claim(w, r1))
    w.append_row(row_for(w, rep), claim(w, rep))


def test_one_regime_per_request_and_the_request_terms_are_pinned(env):
    root, clock, alive = env
    w = writer(root, clock, alive, 11)
    req = opened(w)
    w.append_row(row_for(w, req), claim(w, req))
    other = K.regime(turn_limit=100)
    with pytest.raises(W.RequestSpecError, match="one regime per request"):
        w.append_row(row_for(w, req, batch=1, reg=other), claim(w, req, batch=1, reg=other))
    with pytest.raises(W.RequestSpecError, match="other terms"):
        opened(w, spec={"batch_pairs": 4})
    assert opened(w) == req, "a re-open with the same terms is the same request"
    pinned = opened(w, "pinned", regime_id=K.regime()["regime_id"])
    with pytest.raises(W.RequestSpecError, match="pins regime"):
        claim(w, pinned, reg=other)
    bad_purpose = row_for(w, req, batch=2, purpose="ab", cycle_seed=99)
    with pytest.raises(W.RequestSpecError, match="purpose"):
        w.append_row(bad_purpose, claim(w, req, batch=2))


def test_a_pinned_family_refuses_another_protocol(env):
    root, clock, alive = env
    w = writer(root, clock, alive, 11)
    w.register_family("x5_ab", decision_kind="ab_verdict", rule="x5 §7.4 v1", protocol="gen3_eval_protocol_v1_h2h")
    with pytest.raises(W.RequestSpecError, match="pins protocol"):
        opened(w, "look1", kind="ab_cell", purpose="ab", family="x5_ab", protocol="gen3_eval_protocol_v1_bot_rr")
    look = opened(w, "look1", kind="ab_cell", purpose="ab", family="x5_ab")
    assert look["protocol"] == "gen3_eval_protocol_v1_h2h", "a request in a family inherits its pinned protocol"
    other = K.regime(protocol="gen3_eval_protocol_v1_bot_rr")
    with pytest.raises(W.RequestSpecError, match="pins protocol"):
        w.append_row(row_for(w, look, reg=other, purpose="ab"), claim(w, look, reg=other))
    with pytest.raises(W.RequestSpecError, match="not registered"):
        opened(w, "look9", kind="ab_cell", purpose="ab", family="nope")
    with pytest.raises(W.RequestSpecError, match="other terms"):
        w.register_family("x5_ab", decision_kind="ab_verdict", rule="another rule", protocol="gen3_eval_protocol_v1_h2h")


def test_a_row_contradicting_its_request_or_its_claim_is_refused(env):
    root, clock, alive = env
    w = writer(root, clock, alive, 11)
    req = opened(w)
    c = claim(w, req)
    forged = row_for(w, req)
    forged["request"]["opened"] = "2026-10-03T11:59:59+00:00"
    with pytest.raises(W.RequestSpecError, match="contradicts"):
        w.append_row(forged, c)
    with pytest.raises(W.LedgerClaimError, match="unit"):
        w.append_row(row_for(w, req, batch=5), c)
    with pytest.raises(W.LedgerClaimError, match="under its claim"):
        w.append_row(row_for(w, req))
    with pytest.raises(W.LedgerClaimError, match="FOR a request"):
        w.append_row(K.make_row(w.next_row_id()))


def test_only_a_backfill_writer_writes_request_less_rows(env):
    root, clock, alive = env
    bf = writer(root, clock, alive, 11, producer="backfill", allow_unrequested=True)
    bf.append_row(K.make_row(bf.next_row_id()))
    assert bf.rows_path.exists()


def test_an_inconsistent_requests_stream_stops_the_writer(env):
    root, clock, alive = env
    w = writer(root, clock, alive, 11)
    req = opened(w)
    e = {"schema": S.EVENT_SCHEMA, "event": "row", "seq": 99, "ts": clock.iso(), "writer_id": "x",
         "request_id": req["request_id"], "batch": 0, "player": K.SHA_A, "opponent": K.SHA_B,
         "regime_id": K.regime()["regime_id"], "claim_seq": 42, "row_id": "x:0", "seed_key": None}
    ST.append_line(root / "requests" / "events.forged.jsonl", e)
    with pytest.raises(W.LedgerClaimError, match="inconsistent"):
        claim(w, req)


# ------------------------------------------------------------------------------------------------ lock + close
def test_the_lock_is_bounded_and_reentrant(env):
    root, clock, alive = env
    root.mkdir(parents=True)
    import fcntl

    (root / "requests").mkdir()
    fd = os.open(root / "requests" / ".lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)           # another holder (a separate open file description)
        with pytest.raises(ST.LedgerLockTimeout, match="stuck"):
            with ST.locked(root, timeout_s=0):
                pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    with ST.locked(root):
        with ST.locked(root):                    # re-entrant within the process
            pass


def test_close_gzips_the_shard_and_a_reader_sees_one_copy(env):
    root, clock, alive = env
    w = writer(root, clock, alive, 11)
    req = opened(w)
    w.append_row(row_for(w, req), claim(w, req))
    plain = str(w.rows_path)
    w.close()
    assert not os.path.exists(plain) and os.path.exists(plain + ".gz")
    with gzip.open(plain + ".gz", "rt") as f:
        assert json.loads(f.readline())["row_id"].endswith(":0")
    with open(plain, "w") as f:                  # a close "in flight": both files present
        f.write(gzip.open(plain + ".gz", "rt").read())
    assert ST.one_per_stem([plain, plain + ".gz"]) == [plain + ".gz"]
    assert len(ST.scan_rows(root)) == 1


def test_close_stale_gzips_only_a_dead_idle_writers_shard_on_this_host(env):
    root, clock, alive = env
    w_dead, w_live = writer(root, clock, alive, 11), writer(root, clock, alive, 22)
    req = opened(w_dead)
    w_dead.append_row(row_for(w_dead, req), claim(w_dead, req))
    w_live.append_row(row_for(w_live, req, batch=1), claim(w_live, req, batch=1))
    alive.dead.add(11)
    now = os.path.getmtime(w_dead.rows_path) + ST.STALE_IDLE_S
    young = ST.close_stale(root, now=now - 1, alive=alive, host=HOST)
    assert young.closed == [] and any("idle" in s for s in young.skipped)
    dry = ST.close_stale(root, now=now, alive=alive, host=HOST, apply=False)
    assert dry.closed == [str(w_dead.rows_path)] and w_dead.rows_path.exists()
    rep = ST.close_stale(root, now=now + 10 ** 6, alive=alive, host=HOST)
    assert rep.closed == [str(w_dead.rows_path)] and any("alive" in s for s in rep.skipped)
    assert ST.close_stale(root, now=now + 10 ** 6, alive=alive, host="elsewhere").closed == []
