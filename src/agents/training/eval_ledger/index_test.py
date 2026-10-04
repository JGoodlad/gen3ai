"""The persisted indexes (F-ED-22): ``event_index`` (the writer's incremental fold of ``requests/``) and ``row_index``
(the reader's per-request row index) — what they cost, what they must equal, and how they fail safe.

PERFORMANCE IS CHECKED BY STRUCTURE, never by a clock (``designs/ops/testing.md``, "Performance-shape tests"): the
counters in ``store.IO`` count every line and byte the ledger parses, so "a claim costs the same at any archive size"
is an EXACT equality of counts between a small and a large archive, and the teeth case shows the same counter growing
under the pre-index full fold (``GEN3AI_LEDGER_INDEX=0``). Everything else is a differential: the index must produce,
event for event and row for row, what the full fold / full scan produces.

DETERMINISTIC BY CONSTRUCTION: a hand-moved clock, an owned liveness oracle, explicit writer ids; the only processes
are the concurrency test's, and they are joined with a bound."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from agents.training import eval_ledger as L
from agents.training.eval_ledger import audit as AU
from agents.training.eval_ledger import event_index as EI
from agents.training.eval_ledger import incremental as INC
from agents.training.eval_ledger import queue as Q
from agents.training.eval_ledger import row_index as RI
from agents.training.eval_ledger import schema as S
from agents.training.eval_ledger import store as ST
from agents.training.eval_ledger import testkit as K
from agents.training.eval_ledger import writer as W

HOST = "testhost"
OWN = L.ReaderDecl(name="t.own", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="own",
                   selection="include", flags_ok=frozenset(), inference="conditional")
OWN_EXCL = L.ReaderDecl(name="t.own_excl", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="own",
                        selection="exclude", flags_ok=frozenset(), inference="conditional")
FAMILY = L.ReaderDecl(name="t.fam", purposes=frozenset({"ab"}), regime=L.RegimeFilter(), requests="family",
                      selection="include", flags_ok=frozenset(), inference="across_runs", decision_kind="ab_verdict")
ANY = L.ReaderDecl(name="t.any", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                   selection="include", flags_ok=frozenset(), inference="conditional")


SIDE = {K.SHA_A: 0, K.SHA_B: 1, K.SHA_C: 2, K.SHA_D: 3}


class Alive:
    def __init__(self) -> None:
        self.dead: set = set()

    def __call__(self, pid: int) -> bool:
        return pid not in self.dead


def wid(n: int, pid: int, producer: str = "h2h") -> str:
    return f"20261003T12{n:02d}00Z.0-{HOST}-{pid}-{producer}"


def mk(root: Path, clock: K.Clock, alive: Alive, pid: int, n: Optional[int] = None, producer: str = "h2h",
       **kw: Any) -> W.LedgerWriter:
    return W.LedgerWriter(root, producer, clock=clock, alive=alive, host=HOST, pid=pid,
                          writer_id=wid(n if n is not None else pid % 60, pid, producer), **kw)


def grow(w: W.LedgerWriter, cycles: int, tag: str, seed0: int = 1000) -> None:
    """``cycles`` in-loop-shaped cycles: an open, three claimed rows, a done."""
    reg = K.regime()
    for i in range(cycles):
        rid = f"{tag}:{i:03d}"
        req = w.open_request(rid, kind="cycle", purpose="cycle", regime_id=reg["regime_id"], protocol=K.PROTO,
                             spec={})
        for o, opp in enumerate((K.SHA_B, K.SHA_C, K.SHA_D)):
            c = w.claim(rid, batch=0, player=K.SHA_A, opponent=opp, regime_id=reg["regime_id"], expected_wall_s=1)
            w.append_row(K.make_row(w.next_row_id(), request=req, batch=0, p=K.SHA_A, o=opp, reg=reg,
                                    ts=w.clock().isoformat(timespec="seconds"), purpose="cycle",
                                    cycle_seed=seed0 + i * 3 + o), c)
        w.finish_request(rid)


def fold_of(root: Path) -> Q.QueueState:
    return Q.fold(e for e, _ in ST.scan_events(root))


def assert_index_equals_fold(root: Path) -> None:
    with ST.locked(root):
        got = EI.EventIndex(root).full_state()
    assert EI.state_diff(fold_of(root), got) == []
    assert EI.EventIndex(root).verify() == []


@pytest.fixture(autouse=True)
def fresh_counters():
    EI.COUNTERS.reset()
    RI.COUNTERS.reset()
    ST.IO.reset()
    yield


@pytest.fixture
def env(tmp_path):
    return tmp_path / "ledger", K.Clock(), Alive()


# ------------------------------------------------------------------------------------------------ the SHAPE of the cost
def _one_cycle_cost(root: Path, clock: K.Clock, alive: Alive, pid: int) -> Dict[str, int]:
    """What ONE fresh process's cycle (open, three claimed rows, done) parses, against a warm index."""
    w = mk(root, clock, alive, pid)
    ST.IO.reset()
    EI.COUNTERS.reset()
    grow(w, 1, "z", seed0=900_000)
    return {"lines": ST.IO.lines_parsed, "bytes": ST.IO.bytes_read, "full_builds": EI.COUNTERS.full_builds}


def test_a_cycle_parses_the_same_lines_at_any_archive_size(tmp_path):
    """F-ED-22: every claim and append used to fold the WHOLE archive's requests stream. Now a cycle (1 open, 3 claims,
    3 appends, 1 done) at 2 and at 12 earlier cycles parses EXACTLY the same number of lines — its own new events —
    and nothing like the archive's ~100."""
    costs = {}
    for n_cycles in (2, 12):
        root = tmp_path / f"ledger{n_cycles}"
        clock, alive = K.Clock(), Alive()
        grow(mk(root, clock, alive, 11), n_cycles, "old")
        costs[n_cycles] = _one_cycle_cost(root, clock, alive, 22)
    small, big = costs[2], costs[12]
    assert big["lines"] == small["lines"], costs
    assert big["full_builds"] == small["full_builds"] == 0, "a warm index is not rebuilt"
    assert big["lines"] <= 20, f"a cycle's own events, not the archive: {costs}"
    assert big["bytes"] - small["bytes"] <= 160 and big["bytes"] < 40_000, costs


def test_the_counters_see_the_pre_index_full_fold(tmp_path, monkeypatch):
    """TEETH: the same cycle under ``GEN3AI_LEDGER_INDEX=0`` (the pre-index fold) parses the whole archive, so the
    equality above is a real test — it fails when the writer's state stops being incremental."""
    costs = {}
    monkeypatch.setenv(EI.ENV_SWITCH, "0")
    for n_cycles in (2, 12):
        root = tmp_path / f"ledger{n_cycles}"
        clock, alive = K.Clock(), Alive()
        grow(mk(root, clock, alive, 11), n_cycles, "old")
        costs[n_cycles] = _one_cycle_cost(root, clock, alive, 22)["lines"]
    assert costs[12] > 3 * costs[2] and costs[12] > 400, costs


class Box:
    """A ledger root with a hand-moved clock and one writer (the reader tests' shape)."""

    def __init__(self, root: Path, pid: int = 11) -> None:
        self.root, self.clock, self.alive = root, K.Clock(), Alive()
        self.w = mk(root, self.clock, self.alive, pid)

    def open(self, rid: str = "r1", **kw: Any) -> Dict[str, Any]:
        args: Dict[str, Any] = dict(kind="adhoc", purpose="audit", spec={})
        args.update(kw)
        return self.w.open_request(rid, **args)

    def put(self, req: Dict[str, Any], batch: int = 0, p: str = K.SHA_A, o: str = K.SHA_B, advance: int = 60,
            reg: Optional[Dict[str, Any]] = None, **kw: Any) -> Dict[str, Any]:
        self.clock.advance(advance)
        reg = reg or K.regime()
        c = self.w.claim(req["request_id"], batch=batch, player=p, opponent=o, regime_id=reg["regime_id"],
                         expected_wall_s=60)
        row = K.make_row(self.w.next_row_id(), request=req, batch=batch, p=p, o=o, reg=reg, ts=self.clock.iso(),
                         purpose=req["purpose"], cycle_seed=kw.pop("cycle_seed", 7000 + 10 * batch + SIDE.get(o, 9)), **kw)
        self.w.append_row(row, c)
        return row


def _read_cost(root: Path, rid: str) -> Dict[str, int]:
    L.read(OWN, root=root, request_id=rid)                       # the first read builds the index
    ST.IO.reset()
    RI.COUNTERS.reset()
    got = L.read(OWN, root=root, request_id=rid)
    assert len(got) == 3
    return {"lines": ST.IO.lines_parsed, "fast": RI.COUNTERS.fast_reads, "fallbacks": RI.COUNTERS.fallbacks,
            "builds": RI.COUNTERS.full_builds}


def test_a_read_of_one_request_parses_only_its_rows_at_any_archive_size(tmp_path, monkeypatch):
    """The reader half: ``read(requests="own")`` costs the request's rows (3), at 2 or 40 earlier cycles; and it
    took the FAST path (a silent fallback to the scan would also return the right rows)."""
    costs = {}
    for n_cycles in (2, 12):
        root = tmp_path / f"ledger{n_cycles}"
        grow(mk(root, K.Clock(), Alive(), 11), n_cycles, "old")
        costs[n_cycles] = _read_cost(root, "old:001")
    assert costs[2] == costs[12] == {"lines": 3, "fast": 1, "fallbacks": 0, "builds": 0}, costs
    monkeypatch.setenv(EI.ENV_SWITCH, "0")                         # TEETH: the scan parses every row
    ST.IO.reset()
    L.read(OWN, root=tmp_path / "ledger12", request_id="old:001")
    assert ST.IO.lines_parsed >= 36


# ------------------------------------------------------------------------------------------------ the index IS the fold
def _life(root: Path, clock: K.Clock, alive: Alive, step: Optional[Any] = None) -> List[str]:
    """One scripted life of a ledger — two families of outcomes, a void by a dead pid, by expiry, a repair, a refused
    duplicate, a close — and what each step raised. ``step`` runs after every step."""
    out: List[str] = []

    def do(label: str, fn: Any) -> Any:
        try:
            r = fn()
            out.append(f"{label}: ok")
        except (W.LedgerClaimError, S.LedgerSchemaError, ST.LedgerLockTimeout) as e:
            r = None
            out.append(f"{label}: {type(e).__name__}: {str(e).replace(str(root), '<root>')[:90]}")
        if step is not None:
            step(label)
        return r

    reg = K.regime()
    w1, w2, w3, w4 = (mk(root, clock, alive, 11), mk(root, clock, alive, 22), mk(root, clock, alive, 33),
                      mk(root, clock, alive, 44))

    def cl(w, req, batch=0, o=K.SHA_B):
        return w.claim(req["request_id"], batch=batch, player=K.SHA_A, opponent=o, regime_id=reg["regime_id"],
                       expected_wall_s=60)

    def row(w, req, batch=0, o=K.SHA_B, seed=None, **kw):
        return K.make_row(w.next_row_id(), request=req, batch=batch, o=o, ts=clock.iso(), purpose=req["purpose"],
                          cycle_seed=100 + batch if seed is None else seed, **kw)

    do("family", lambda: w1.register_family("fam", decision_kind="ab_verdict", rule="r v1", protocol=K.PROTO))
    do("family again", lambda: w1.register_family("fam", decision_kind="ab_verdict", rule="r v1", protocol=K.PROTO))
    do("family other terms", lambda: w1.register_family("fam", decision_kind="ab_verdict", rule="r v2",
                                                        protocol=K.PROTO))
    r1 = do("open r1", lambda: w1.open_request("r1", kind="adhoc", purpose="audit", spec={"a": 1}))
    r2 = do("open r2", lambda: w1.open_request("r2", kind="ab_cell", purpose="ab", family="fam"))
    do("open r1 other spec", lambda: w1.open_request("r1", kind="adhoc", purpose="audit", spec={"a": 2}))
    c0 = do("claim r1/0", lambda: cl(w1, r1, 0))
    do("append r1/0", lambda: w1.append_row(row(w1, r1, 0), c0))
    do("claim r1/0 again", lambda: cl(w1, r1, 0))
    do("claim r1/1 (w1)", lambda: cl(w1, r1, 1))
    alive.dead.add(11)                                            # w1 dies holding r1/1
    c1 = do("claim r1/1 (w2) voids w1", lambda: cl(w2, r1, 1))
    do("append r1/1", lambda: w2.append_row(row(w2, r1, 1), c1))
    do("claim r1/2 (w2)", lambda: cl(w2, r1, 2))
    clock.advance(1)
    ST.append_line(w2.rows_path, row(w2, r1, 2))                  # w2's row reached disk; it died before the event
    alive.dead.add(22)
    do("claim r1/2 (w3) repairs", lambda: cl(w3, r1, 2))
    do("claim r2/0 (w3)", lambda: cl(w3, r2, 0))
    clock.advance(W.CLAIM_FLOOR_S + 1)
    c4 = do("claim r2/0 (w4) voids by expiry", lambda: cl(w4, r2, 0))
    do("dup seed block", lambda: w4.append_row(row(w4, r2, 0, seed=100), c4))
    do("append r2/0", lambda: w4.append_row(row(w4, r2, 0, seed=555), c4))
    do("void_dead", lambda: w4.void_dead())
    do("done r1", lambda: w4.finish_request("r1"))
    do("claim on closed r1", lambda: cl(w4, r1, 9))
    do("cancel r2", lambda: w4.finish_request("r2", cancel_reason="test"))
    do("finish unknown", lambda: w4.finish_request("nope"))
    return out


def test_the_index_equals_the_fold_after_every_step_of_a_full_life(env):
    root, clock, alive = env
    seen: List[str] = []

    def check(label: str) -> None:
        assert_index_equals_fold(root)
        seen.append(label)

    out = _life(root, clock, alive, step=check)
    assert len(seen) == len(out) > 20, "every step was checked"
    assert any("voids by expiry: ok" in o for o in out) and any("duplicate" in o.lower() or "dup seed" in o for o in out)
    assert EI.COUNTERS.full_builds == 1, "one build, then incremental catch-ups only"


def test_a_from_scratch_rebuild_equals_the_incremental_index(env):
    root, clock, alive = env
    _life(root, clock, alive)
    with ST.locked(root):
        before = EI.EventIndex(root).full_state()
        EI.EventIndex(root).rebuild()
        after = EI.EventIndex(root).full_state()
    assert EI.state_diff(before, after) == [] and EI.state_diff(fold_of(root), after) == []


def test_the_same_life_with_and_without_the_index_writes_identical_streams_and_refusals(tmp_path, monkeypatch):
    """The differential: every refusal and every byte of the streams are those of the full fold."""
    results = {}
    for mode in ("1", "0"):
        monkeypatch.setenv(EI.ENV_SWITCH, mode)
        root = tmp_path / f"ledger_{mode}"
        out = _life(root, K.Clock(), Alive())
        files = {p.name: p.read_text() for p in sorted((root / "requests").glob("events.*.jsonl"))}
        shards = {p.name: p.read_text() for p in sorted((root / "rows").rglob("ledger.*.jsonl*")) if p.suffix != ".gz"}
        results[mode] = (out, files, shards)
    assert results["1"][0] == results["0"][0]
    assert results["1"][1] == results["0"][1] and results["1"][2] == results["0"][2]
    assert (tmp_path / "ledger_1" / EI.INDEX_DIRNAME).is_dir() and not (tmp_path / "ledger_0" / EI.INDEX_DIRNAME).exists()


# ------------------------------------------------------------------------------------------------ stale / corrupt
def test_events_appended_behind_the_index_are_folded_not_rebuilt(env):
    root, clock, alive = env
    w1, w2 = mk(root, clock, alive, 11), mk(root, clock, alive, 22)
    grow(w1, 3, "a")
    EI.COUNTERS.reset()
    grow(w2, 2, "b", seed0=5000)                                  # another process's events: unseen by w1's last sync
    grow(w1, 1, "c", seed0=6000)
    assert EI.COUNTERS.full_builds == 0, EI.COUNTERS.rebuild_reasons
    assert_index_equals_fold(root)


def test_a_truncated_stream_rebuilds_the_index(env):
    root, clock, alive = env
    w = mk(root, clock, alive, 11)
    grow(w, 3, "a")
    lines = w.events_path.read_text().splitlines(keepends=True)
    w.events_path.write_text("".join(lines[:-10]))                # the file lost its last ten events
    EI.COUNTERS.reset()
    st = Q.fold(e for e, _ in ST.scan_events(root))
    with ST.locked(root):
        got = EI.EventIndex(root).full_state()
    assert any("truncated" in r for r in EI.COUNTERS.rebuild_reasons), EI.COUNTERS.rebuild_reasons
    assert EI.state_diff(st, got) == []


def test_a_stream_rewritten_at_its_tail_rebuilds_the_index(env):
    root, clock, alive = env
    w = mk(root, clock, alive, 11)
    grow(w, 3, "a")
    with ST.locked(root):
        EI.EventIndex(root).state()                               # the cursor is at the end of the file
    text = w.events_path.read_text()
    assert HOST in text[-60:]
    w.events_path.write_text(text[:-60] + text[-60:].replace(HOST, "testhosT"))     # same size, the tail differs
    with ST.locked(root):
        EI.EventIndex(root).state(request="a:002")
    assert any("rewritten" in r for r in EI.COUNTERS.rebuild_reasons), EI.COUNTERS.rebuild_reasons
    assert_index_equals_fold(root)


def test_a_vanished_stream_file_rebuilds_the_index(env):
    root, clock, alive = env
    w1, w2 = mk(root, clock, alive, 11), mk(root, clock, alive, 22)
    grow(w1, 2, "a")
    grow(w2, 2, "b", seed0=5000)
    w1.events_path.unlink()
    with ST.locked(root):
        got = EI.EventIndex(root).full_state()
    assert any("vanished" in r for r in EI.COUNTERS.rebuild_reasons)
    assert EI.state_diff(fold_of(root), got) == [] and not any(r.startswith("a:") for r in got.requests)


def test_a_gzipped_events_file_is_the_same_stream_not_a_rebuild(env):
    root, clock, alive = env
    w1, w2 = mk(root, clock, alive, 11), mk(root, clock, alive, 22)
    grow(w1, 2, "a")
    with ST.locked(root):
        EI.EventIndex(root).state()                               # everything of w1's is folded
    EI.COUNTERS.reset()
    ST.gzip_shard(w1.events_path)                                # the same bytes, closed into a .gz
    grow(w2, 1, "b", seed0=5000)
    assert EI.COUNTERS.full_builds == 0, EI.COUNTERS.rebuild_reasons
    assert_index_equals_fold(root)


def test_an_event_with_a_seq_below_the_indexed_maximum_rebuilds_the_index(env):
    """``seq`` is assigned under the lock, so a new event always folds after the old: one that does not (a gap filled
    by hand) must be placed by the from-scratch, SORTED fold — the index rebuilds."""
    root, clock, alive = env
    w = mk(root, clock, alive, 11)
    w.open_request("r1", kind="adhoc", purpose="audit", spec={})
    w.finish_request("r1")
    opened = json.loads(w.events_path.read_text().splitlines()[0])
    ST.append_line(root / "requests" / "events.f1.jsonl", dict(opened, seq=10, request_id="r2"))
    with ST.locked(root):
        EI.EventIndex(root).state(request="r2")                  # folds seq 10: the maximum is 10
    assert EI.COUNTERS.full_builds == 1
    ST.append_line(root / "requests" / "events.f2.jsonl", dict(opened, seq=5, request_id="r3"))
    with ST.locked(root):
        got = EI.EventIndex(root).full_state()
    assert any("out of order" in r for r in EI.COUNTERS.rebuild_reasons), EI.COUNTERS.rebuild_reasons
    assert EI.state_diff(fold_of(root), got) == [] and {"r1", "r2", "r3"} <= set(got.requests)


@pytest.mark.parametrize("damage", ["garbage", "truncated", "deleted", "other_format"])
def test_a_corrupt_or_foreign_index_is_discarded_or_rebuilt_never_trusted(env, damage):
    root, clock, alive = env
    w = mk(root, clock, alive, 11)
    grow(w, 3, "a")
    db = root / EI.INDEX_DIRNAME / "events.sqlite3"
    assert db.exists()
    INC.close_keepers(db)                      # a process restart: the WAL is checkpointed, the file holds everything
    if damage == "garbage":
        db.write_bytes(b"this is not a database" * 100)
    elif damage == "truncated":
        db.write_bytes(db.read_bytes()[:300])
    elif damage == "deleted":
        for p in db.parent.glob("events.sqlite3*"):
            p.unlink()
    else:
        c = sqlite3.connect(str(db))
        c.execute("UPDATE meta SET v = '\"gen3_eval_event_index_v0\"' WHERE k = 'format'")
        c.commit()
        c.close()
    EI.COUNTERS.reset()
    grow(w, 1, "b", seed0=5000)                                   # the next operation heals it
    assert EI.COUNTERS.full_builds == 1, (damage, EI.COUNTERS.rebuild_reasons)
    assert (EI.COUNTERS.corrupt_discards == 1) == (damage in ("garbage", "truncated")), EI.COUNTERS.rebuild_reasons
    assert_index_equals_fold(root)


def test_a_malformed_event_is_the_same_typed_error_with_or_without_the_index(env, monkeypatch):
    root, clock, alive = env
    w = mk(root, clock, alive, 11)
    req = w.open_request("r1", kind="adhoc", purpose="audit", spec={})
    reg = K.regime()
    msgs = {}
    for label, bad in (("not json", "{this is not json\n"), ("schema", json.dumps({"event": "claim", "seq": 9}) + "\n")):
        f = root / "requests" / f"events.bad-{label.replace(' ', '')}.jsonl"
        f.write_text(bad)
        for mode in ("1", "0"):
            monkeypatch.setenv(EI.ENV_SWITCH, mode)
            with pytest.raises(S.LedgerSchemaError) as ei:
                w.claim("r1", batch=0, player=K.SHA_A, opponent=K.SHA_B, regime_id=reg["regime_id"],
                        expected_wall_s=1)
            msgs[(label, mode)] = str(ei.value)
        f.unlink()
        assert msgs[(label, "1")] == msgs[(label, "0")], msgs
    assert req["request_id"] == "r1"


def test_audit_detects_a_tampered_event_index_and_rebuild_index_repairs_it(env, capsys):
    """The index is a cache the writer trusts between audits; ``audit`` is what re-folds the streams against it."""
    root, clock, alive = env
    w = mk(root, clock, alive, 11)
    grow(w, 3, "a")
    assert AU.audit(root).ok
    c = sqlite3.connect(str(root / EI.INDEX_DIRNAME / "events.sqlite3"))
    c.execute("DELETE FROM seed_keys WHERE rowid = (SELECT min(rowid) FROM seed_keys)")
    c.execute("UPDATE claims SET row_id = NULL, live = 1 WHERE seq = (SELECT min(seq) FROM claims)")
    c.commit()
    c.close()
    rep = AU.audit(root)
    assert not rep.ok and any(p.startswith("event index:") for p in rep.problems), rep.problems
    from main import eval_ledger as CLI

    assert CLI.main(["audit", "--root", str(root)]) == 1
    assert CLI.main(["audit", "--root", str(root), "--rebuild-index"]) == 0
    assert "OK" in capsys.readouterr().out
    assert AU.audit(root).ok and EI.COUNTERS.full_builds >= 1
    assert_index_equals_fold(root)


def test_audit_detects_a_tampered_row_index(env):
    root, clock, alive = env
    box = Box(root)
    r1 = box.open("r1")
    box.put(r1, 0)
    box.put(r1, 1)
    L.read(OWN, root=root, request_id="r1")
    assert AU.audit(root).ok
    c = sqlite3.connect(str(root / EI.INDEX_DIRNAME / "rows.sqlite3"))
    c.execute("UPDATE rows SET request_id = 'elsewhere' WHERE rowid = (SELECT min(rowid) FROM rows)")
    c.commit()
    c.close()
    rep = AU.audit(root)
    assert not rep.ok and any(p.startswith("row index:") for p in rep.problems), rep.problems
    assert AU.audit(root, rebuild_index=True).ok


# ------------------------------------------------------------------------------------------------ uniqueness across the index boundary
def test_a_seed_block_recorded_long_before_is_still_refused(env):
    """The seed-block uniqueness index covers every old row event: a fresh writer, 12 cycles later, is refused."""
    root, clock, alive = env
    w1 = mk(root, clock, alive, 11)
    grow(w1, 12, "old", seed0=1000)
    w2 = mk(root, clock, alive, 22)
    reg = K.regime()
    req = w2.open_request("new", kind="cycle", purpose="cycle", regime_id=reg["regime_id"], protocol=K.PROTO, spec={})
    c = w2.claim("new", batch=0, player=K.SHA_A, opponent=K.SHA_B, regime_id=reg["regime_id"], expected_wall_s=1)
    dup = K.make_row(w2.next_row_id(), request=req, batch=0, p=K.SHA_A, o=K.SHA_B, reg=reg, ts=clock.iso(),
                     purpose="cycle", cycle_seed=1000 + 5 * 3)   # old:005's seed block
    with pytest.raises(W.DuplicateBatchError, match="seed block"):
        w2.append_row(dup, c)
    assert not w2.rows_path.exists()


def test_a_recorded_unit_is_refused_at_the_claim_and_a_planted_duplicate_stops_the_writer(env):
    root, clock, alive = env
    w = mk(root, clock, alive, 11)
    grow(w, 10, "old")
    reg = K.regime()
    req = w.open_request("live1", kind="cycle", purpose="cycle", regime_id=reg["regime_id"], protocol=K.PROTO, spec={})
    c = w.claim("live1", batch=0, player=K.SHA_A, opponent=K.SHA_B, regime_id=reg["regime_id"], expected_wall_s=1)
    w.append_row(K.make_row(w.next_row_id(), request=req, batch=0, reg=reg, ts=clock.iso(), purpose="cycle",
                            cycle_seed=99_999), c)
    with pytest.raises(W.AlreadyRecordedError):
        w.claim("live1", batch=0, player=K.SHA_A, opponent=K.SHA_B, regime_id=reg["regime_id"], expected_wall_s=1)
    forged = json.loads(w.events_path.read_text().splitlines()[1])      # cycle 0's first claim, replayed by hand
    assert forged["event"] == "claim"
    ST.append_line(root / "requests" / "events.forged.jsonl", dict(forged, seq=10_000))
    with pytest.raises(W.LedgerClaimError, match="inconsistent"):
        w.open_request("after", kind="adhoc", purpose="audit", spec={})
    assert not AU.audit(root).ok


@pytest.mark.parametrize("kind", ["batch key", "seed block", "row_id"])
def test_a_duplicate_planted_in_an_old_shard_is_refused_by_the_indexed_read(env, kind):
    root, clock, alive = env
    box = Box(root)
    r1, r2 = box.open("r1"), box.open("r2")
    row = box.put(r1, 0)
    L.read(OWN, root=root, request_id="r1")
    assert RI.COUNTERS.fast_reads == 1 and RI.COUNTERS.fallbacks == 0
    if kind == "batch key":
        ST.append_line(root / "rows" / "h2h" / "ledger.planted-1-h2h.jsonl", dict(row, row_id="planted-1-h2h:0"))
    elif kind == "seed block":
        other = K.make_row("planted-2-h2h:0", request=r2, batch=0, cycle_seed=row["seed"]["cycle_seed"], ts=row["ts"])
        ST.append_line(root / "rows" / "h2h" / "ledger.planted-2-h2h.jsonl", other)
    else:
        (root / "rows" / "h2h" / "ledger.copy-1-h2h.jsonl").write_text(box.w.rows_path.read_text())
    with pytest.raises(L.DuplicateBatchError, match=kind):
        L.read(OWN, root=root, request_id="r1")
    assert RI.COUNTERS.fallbacks == 1, "the index refused to answer; the scan raised"
    with pytest.raises(L.DuplicateBatchError):
        L.read(OWN, root=root, request_id="r2")                         # the guarantee is ledger-wide


# ------------------------------------------------------------------------------------------------ the row index IS the scan
def _rows_view(got: Any) -> Any:
    return ([r["row_id"] for r in got.rows], got.digest, got.regime_id, got.excluded,
            None if got.as_of is None else got.as_of.isoformat())


def _both(monkeypatch: Any, decl: Any, root: Path, expect_fast: Optional[bool], **kw: Any) -> Any:
    f0 = RI.COUNTERS.fast_reads
    fast = L.read(decl, root=root, **kw)
    used = RI.COUNTERS.fast_reads - f0
    monkeypatch.setenv(EI.ENV_SWITCH, "0")
    try:
        slow = L.read(decl, root=root, **kw)
    finally:
        monkeypatch.delenv(EI.ENV_SWITCH)
    assert _rows_view(fast) == _rows_view(slow)
    assert [r for r in fast.rows] == [r for r in slow.rows]
    if expect_fast is not None:
        assert (used == 1) is expect_fast, (kw, RI.COUNTERS)
    return fast


def _history(box: Box) -> List[str]:
    """Several requests, a family, two regimes, a correction and a selecting decision; returns every row's ts."""
    box.w.register_family("fam", decision_kind="ab_verdict", rule="r v1", protocol=K.PROTO)
    r1, r2 = box.open("r1"), box.open("r2", purpose="promotion", kind="sprt")
    a = box.open("a1", kind="ab_cell", purpose="ab", family="fam")
    b = box.open("a2", kind="ab_cell", purpose="ab", family="fam")
    rows = [box.put(r1, 0), box.put(r1, 1), box.put(r2, 0, cycle_seed=77), box.put(a, 0, cycle_seed=91),
            box.put(a, 1, cycle_seed=92), box.put(b, 0, cycle_seed=93), box.put(r1, 2, o=K.SHA_C)]
    fix = dict(rows[1], row_id="fix-1-h2h:0", supersedes=rows[1]["row_id"], ts=box.clock.iso(30), run="corrected")
    ST.append_line(box.root / "rows" / "h2h" / "ledger.fix-1-h2h.jsonl", fix)
    box.clock.advance(120)
    consumed = L.read(OWN, root=box.root, request_id="r2")
    box.w.append_decision(kind="promotion", subject=K.SHA_A, consumed=consumed, verdict="PROMOTE",
                          rule="sprt_promotion", rule_version="1", request_id="r2")
    return sorted({r["ts"] for r in rows} | {fix["ts"]})


def test_indexed_reads_equal_the_scan_for_every_scope_and_every_as_of(env, monkeypatch):
    root, _clock, _alive = env
    box = Box(root)
    stamps = _history(box)
    t0 = S.parse_ts(stamps[0])
    assert t0 is not None
    import datetime as _dt

    as_ofs: List[Optional[str]] = [None, (t0 - _dt.timedelta(seconds=1)).isoformat(), "2099-01-01T00:00:00+00:00"]
    for ts in stamps:                                            # exactly AT each row's ts (both sides use <=) and after
        as_ofs += [ts, (S.parse_ts(ts) + _dt.timedelta(seconds=1)).isoformat()]  # type: ignore[operator]
    for as_of in as_ofs:
        for rid in ("r1", "r2", "a1", "nope"):
            _both(monkeypatch, OWN, root, True, request_id=rid, as_of=as_of)
        _both(monkeypatch, OWN_EXCL, root, True, request_id="r1", as_of=as_of)
        try:
            _both(monkeypatch, FAMILY, root, True, family="fam", as_of=as_of)
        except L.ReaderDeclError:                                # not registered as of an early time: both raise
            monkeypatch.setenv(EI.ENV_SWITCH, "0")
            with pytest.raises(L.ReaderDeclError):
                L.read(FAMILY, root=root, family="fam", as_of=as_of)
            monkeypatch.delenv(EI.ENV_SWITCH)
    assert RI.COUNTERS.fallbacks == 0, "a correction is not a duplicate: the fast path answers it"
    corrected = L.read(OWN, root=root, request_id="r1")
    assert [r["run"] for r in corrected].count("corrected") == 1


def test_family_registrations_as_of_come_from_the_event_index(env, monkeypatch):
    root, clock, alive = env
    box = Box(root)
    box.clock.advance(60)
    t_before = box.clock.iso(-1)
    box.w.register_family("fam", decision_kind="ab_verdict", rule="r v1", protocol=K.PROTO)
    t_at = box.clock.iso()
    a = box.open("a1", kind="ab_cell", purpose="ab", family="fam")
    box.put(a, 0, cycle_seed=91)
    ei = EI.EventIndex(root)
    assert ei.family_as_of("fam", S.parse_ts(t_before)) is None
    assert ei.family_as_of("fam", S.parse_ts(t_at))["rule"] == "r v1"
    assert ei.family_as_of("fam", None)["family_id"] == "fam" and ei.family_as_of("nope", None) is None
    assert len(L.read(FAMILY, root=root, family="fam")) == 1
    with pytest.raises(L.ReaderDeclError, match="not registered"):
        L.read(FAMILY, root=root, family="fam", as_of=t_before)


def test_closing_a_shard_does_not_rebuild_the_row_index_and_new_rows_are_seen(env, monkeypatch):
    root, clock, alive = env
    box = Box(root)
    r1 = box.open("r1")
    box.put(r1, 0)
    before = _both(monkeypatch, OWN, root, True, request_id="r1")
    builds = RI.COUNTERS.full_builds
    plain = str(box.w.rows_path)
    box.w.close()                                                # gzip: the same rows under another name
    assert os.path.exists(plain + ".gz") and not os.path.exists(plain)
    after = _both(monkeypatch, OWN, root, True, request_id="r1")
    assert _rows_view(before)[:2] == _rows_view(after)[:2] and RI.COUNTERS.full_builds == builds
    w2 = mk(root, clock, alive, 22)
    c = w2.claim("r1", batch=1, player=K.SHA_A, opponent=K.SHA_B, regime_id=K.regime()["regime_id"], expected_wall_s=1)
    w2.append_row(K.make_row(w2.next_row_id(), request=r1, batch=1, ts=clock.iso(), cycle_seed=4242), c)
    assert len(_both(monkeypatch, OWN, root, True, request_id="r1")) == 2
    assert RI.COUNTERS.full_builds == builds


def test_a_shard_edited_in_place_is_noticed_by_the_row_body_check(env, monkeypatch):
    root, clock, alive = env
    box = Box(root)
    r1 = box.open("r1")
    box.put(r1, 0)
    _both(monkeypatch, OWN, root, True, request_id="r1")
    builds = RI.COUNTERS.full_builds
    text = box.w.rows_path.read_text()
    assert '"run":"study"' in text
    box.w.rows_path.write_text(text.replace('"run":"study"', '"run":"stuby"'))     # same length: offsets unchanged
    got = _both(monkeypatch, OWN, root, True, request_id="r1")
    assert got.rows[0]["run"] == "stuby", "answered as the scan does"
    assert RI.COUNTERS.full_builds == builds + 1, "after rebuilding: the row's bytes were not what the index recorded"


def test_an_unterminated_row_line_falls_back_to_the_scan(env, monkeypatch):
    root, clock, alive = env
    box = Box(root)
    r1 = box.open("r1")
    box.put(r1, 0)
    with open(box.w.rows_path, "a") as f:
        f.write('{"row_id": "half')                              # a writer mid-append, or killed
    msgs = {}
    for mode in ("1", "0"):
        monkeypatch.setenv(EI.ENV_SWITCH, mode)
        with pytest.raises(S.LedgerSchemaError) as ei:
            L.read(OWN, root=root, request_id="r1")
        msgs[mode] = str(ei.value)
    assert "not JSON" in msgs["1"] and msgs["1"] == msgs["0"]
    assert RI.COUNTERS.fallbacks == 1


def test_a_directory_without_an_index_is_scanned_and_never_littered(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    rows = [K.make_v1_row("20261003T120000Z-1:0")]
    (legacy / "ledger.20261003T120000Z-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = L.read(L.ReaderDecl(name="t.legacy", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                              selection="include", flags_ok=frozenset(S.FLAGS), inference="conditional"), root=legacy)
    assert len(got) == 1
    assert len(L.read(L.ReaderDecl(name="t.legacy_own", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(),
                                   requests="own", selection="include", flags_ok=frozenset(S.FLAGS),
                                   inference="conditional"), root=legacy, request_id="r")) == 0
    with pytest.raises(L.ReaderDeclError, match="not registered"):
        L.read(FAMILY, root=legacy, family="f")
    assert sorted(p.name for p in legacy.iterdir()) == ["ledger.20261003T120000Z-1.jsonl"], "no .ledger_index/ here"
    assert RI.COUNTERS.syncs == 0


# ------------------------------------------------------------------------------------------------ concurrency
_WORKER = r"""
import sys
from agents.training import eval_ledger as L
from agents.training.eval_ledger import testkit as K
root, tag, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
w = L.LedgerWriter(root, producer=f"par{tag}")
reg = K.regime()
for i in range(n):
    rid = f"p{tag}:{i}"
    req = w.open_request(rid, kind="adhoc", purpose="audit", spec={})
    c = w.claim(rid, batch=0, player=K.SHA_A, opponent=K.SHA_B, regime_id=reg["regime_id"], expected_wall_s=1)
    row = K.make_row(w.next_row_id(), request=req, batch=0, reg=reg, ts=L.utc_now(), cycle_seed=tag * 1000 + i)
    w.append_row(row, c)
    w.finish_request(rid)
    if i % 4 == 0:
        L.read(L.ReaderDecl(name="t.par", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="own",
                            selection="include", flags_ok=frozenset(), inference="conditional"),
               root=root, request_id=rid)
w.close()
"""


def test_concurrent_writer_processes_leave_one_consistent_index(tmp_path):
    """Three processes, each its own file, one shared lock: no event lost or doubled, every ``seq`` once, and the
    index is exactly the fold of what they wrote (writers AND readers update it under contention)."""
    from utils.paths import src_root

    root = tmp_path / "ledger"
    ST.ensure_layout(root)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(src_root())] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    n, procs = 12, 3
    ps = [subprocess.Popen([sys.executable, "-c", _WORKER, str(root), str(t), str(n)], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for t in range(procs)]
    outs = []
    for p in ps:
        try:
            out, _ = p.communicate(timeout=240)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            pytest.fail(f"a writer process did not finish in 240 s:\n{out}")
        outs.append((p.returncode, out))
    assert all(rc == 0 for rc, _ in outs), outs
    seqs = sorted(int(e["seq"]) for e, _ in ST.scan_events(root))
    assert seqs == list(range(1, len(seqs) + 1)) and len(seqs) == procs * n * 4
    rep = AU.audit(root)
    assert rep.ok, rep.problems
    assert rep.counts["rows"] == procs * n
    assert_index_equals_fold(root)
